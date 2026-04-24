# 01. 개별 주식 데이터 수집 및 피처 생성

**목적**: Gu, Kelly, Xiu (2020) 방법론을 개별 주식 유니버스에 적용하기 위한 월별 패널 데이터를 생성한다.

**유니버스**: S&P 500 동적 구성
- Wikipedia 현재 구성 + 변경 히스토리로 역방향 재구성
- 연구 기간(2004~2024) 내 S&P 500에 한 번이라도 편입된 종목 전체 (~500~700개)
- 월별 패널에서 각 날짜의 실제 S&P 500 구성 종목만 포함 → **생존편향 제거**

**수집 기간**: 2004-01-01 ~ 2024-12-31 (**20년**)
- Walk-forward: IS = 누적 확장(expanding window), OOS = 1개월 → OOS 창 약 180회

**산출 피처 (~50개)**

| 그룹 | 피처 |
|---|---|
| 수익률 | log_ret_1d, simple_ret_1d, excess_ret_1d |
| 모멘텀 | mom_1w/1m/3m/6m/12m/12m_skip_1m, chmom, indmom |
| 변동성/리스크 | vol_20d/60d/252d, beta_252d, idiovol_21d, ivol_63d |
| 유동성 | dollar_vol_21d, amihud_21d, log_mcap, vol_surge |
| 가격 패턴 | high52w_ratio, low52w_ratio, maxret_21d, ma_gap_20_60 |
| 기술적 지표 | rsi_14, bb_pct, intraday_range, autocorr_21d |
| 리스크 조정 수익 | sharpe_21d, sharpe_63d, sortino_63d, ir_63d |
| 분포 | skew_63d, kurt_63d, mdd_252d |
| 거시 민감도 | rate_sensitivity |
| 횡단면(월별) | ret_rank_1m, ret_rank_3m, vol_rank, avg_corr |
| FF 팩터 (거시) | mkt_rf, smb, hml, rmw, cma, rf, mom_factor |

**산출물**: `data/monthly_panel.csv`

---

## 전체 흐름

```
Section 0  설정 및 경로 정의
Section 1  동적 유니버스 구성 (Wikipedia 변경 히스토리 → 역방향 재구성)
Section 2  개별 종목 가격 수집 (yfinance, 2004~2024)
Section 3  Fama-French 5 Factor + MOM 수집 (Ken French)
Section 4  섹터 ETF 수집 (indmom 계산용)
Section 4b SPY & ^TNX 수집 (ir_63d, rate_sensitivity 계산용)
Section 5  개별 패널 피처 생성 (build_panel)
Section 6  월별 패널 생성 + 동적 필터 + 횡단면 피처
Section 7  저장 및 요약
```

---

## Section 0. 설정

| 항목 | 값 |
|---|---|
| `PRICE_START` | 2004-01-01 |
| `PRICE_END` | 2024-12-31 |
| `CACHE_DIR` | `data/cache/{ticker}.pkl` |
| `PANELS_DIR` | `data/panels/{ticker}.csv` |
| `BENCH_DIR` | `data/benchmarks/` |
| `SPY_AC` | 전역 변수 — ir_63d 계산용 |
| `DGS10_DAILY` | 전역 변수 — rate_sensitivity 계산용 |

**GICS 섹터 ↔ 섹터 ETF 매핑**

| 섹터 | ETF |
|---|---|
| Energy | XLE |
| Materials | XLB |
| Industrials | XLI |
| Consumer Discretionary | XLY |
| Consumer Staples | XLP |
| Health Care | XLV |
| Financials | XLF |
| Information Technology | XLK |
| Communication Services | XLC |
| Utilities | XLU |
| Real Estate | XLRE |

---

## Section 1. 동적 유니버스 구성

### 핵심 아이디어

생존편향(survivorship bias)을 제거하기 위해 **현재 S&P 500에서 역방향**으로 변경 히스토리를 적용해 월말 기준 구성을 재구성한다.

```
현재 S&P 500 (~500종목)
  ↓ 역방향으로 변경 이벤트 적용
2024-12-31 구성 → 2024-11-30 구성 → ... → 2004-01-31 구성
```

### 주요 함수

| 함수 | 역할 |
|---|---|
| `fetch_sp500_tables()` | Wikipedia에서 현재 구성 + 변경 히스토리 테이블 수집 |
| `parse_current_sp500(table)` | 현재 S&P 500 구성 파싱 → ticker / gics_sector |
| `parse_changes_table(raw_table)` | 변경 히스토리 파싱 → date / added / removed |
| `build_monthly_membership(...)` | 역방향 재구성 → `{월말 Timestamp: frozenset of tickers}` |
| `get_sp500_members_at(date)` | 임의 날짜의 S&P 500 구성 반환 — O(log n) (bisect 활용) |

### 저장 파일

| 파일 | 내용 |
|---|---|
| `data/sp500_membership.pkl` | 월말 기준 멤버십 딕셔너리 (캐시) |
| `data/universe.csv` | 연구 기간 중 등장한 모든 종목 목록 (ticker, gics_sector) |

### 유니버스 규모

Wikipedia 변경 히스토리가 약 2000년대 초반까지 커버하므로, 연구 기간(2004~2024) 내 등장한 종목은 **500~700개** 수준이다. GICS 섹터 정보가 없는 종목(오래된 퇴출 종목 등)은 제외된다.

---

## Section 2. 개별 종목 가격 수집

- `yfinance.Ticker.history(auto_adjust=False, actions=True)` 사용
- OHLCV + Dividends + Stock Splits 전체 저장
- `data/cache/{ticker}.pkl`에 캐시 → 재실행 시 스킵

```python
def safe_download(ticker, start, end, max_retry=3):
    # 실패 시 지수 백오프 재시도 (최대 3회)
    # 성공 시 timezone 제거, index.name = 'date'
```

> **주의**: 유니버스가 ~500~700개이므로 최초 수집에 상당한 시간이 소요된다. 이후 실행은 캐시로 즉시 완료.

---

## Section 3. Fama-French 5 Factor + MOM 수집

- 출처: Ken French Data Library (ZIP 다운로드)
- FF5: `Mkt-RF, SMB, HML, RMW, CMA, RF` (일별)
- MOM: `mom_factor` (일별)
- 저장: `data/ff_factors.csv`

```
FF5_URL = .../F-F_Research_Data_5_Factors_2x3_daily_CSV.zip
MOM_URL = .../F-F_Momentum_Factor_daily_CSV.zip
```

모든 값은 소수 단위 (`/100` 처리).

---

## Section 4. 섹터 ETF 수집 (indmom 계산용)

GICS 11개 섹터별 대표 ETF(XLE, XLB, ...) 수집 후 252일 rolling return 계산.

```python
indmom_by_sector[sector] = (1 + s.pct_change()).rolling(252).apply(np.prod) - 1
```

각 종목의 `indmom` 피처 = 해당 종목 GICS 섹터 ETF의 12개월 수익률.

---

## Section 4b. SPY & ^TNX 수집

| 데이터 | 용도 | 저장 |
|---|---|---|
| SPY | ir_63d 기준 벤치마크 수익률 | `data/benchmarks/SPY.pkl` |
| ^TNX | 10년 국채 수익률 (금리 민감도) | `data/dgs10_daily.csv` |

두 시리즈는 전역 변수 `SPY_AC`, `DGS10_DAILY`에 저장되어 `build_panel()` 내에서 참조된다.

---

## Section 5. 개별 패널 피처 생성

### `build_panel(ticker, gics_sector, price_df, ff_df)` 함수

각 종목별로 일별 피처를 계산해 `data/panels/{ticker}.csv`에 저장한다.

#### 피처 계산 상세

**수익률**

| 피처 | 계산식 |
|---|---|
| `simple_ret_1d` | `adj_close.pct_change()` |
| `log_ret_1d` | `log(adj_close / adj_close.shift(1))` |
| `excess_ret_1d` | `simple_ret_1d - rf` |

**모멘텀**

| 피처 | 계산식 |
|---|---|
| `mom_1w` | 5일 누적 수익률 |
| `mom_1m` | 21일 누적 수익률 |
| `mom_3m` | 63일 누적 수익률 |
| `mom_6m` | 126일 누적 수익률 |
| `mom_12m` | 252일 누적 수익률 |
| `mom_12m_skip_1m` | 12개월 수익률 / 1개월 수익률 − 1 (단기 반전 제외) |
| `chmom` | `mom_6m(t) − mom_6m(t−126일)` (모멘텀 변화율) |

**변동성 / 리스크**

| 피처 | 계산식 |
|---|---|
| `vol_20d` | `log_ret.rolling(20).std() × √252` |
| `vol_60d` | `log_ret.rolling(60).std() × √252` |
| `vol_252d` | `log_ret.rolling(252).std() × √252` |
| `beta_252d` | `cov(excess_ret, mkt_rf, 252d) / var(mkt_rf, 252d)` |
| `idiovol_21d` | CAPM 잔차 `std × √252` (21일 윈도우) |
| `ivol_63d` | CAPM 잔차 `std × √252` (63일 윈도우) |

**유동성**

| 피처 | 계산식 |
|---|---|
| `dollar_vol_21d` | `(adj_close × volume).rolling(21).mean()` |
| `amihud_21d` | `mean(|ret| / dollar_vol) × 1e6` (21일) |
| `market_cap` | `adj_close × shares_outstanding` (현재 발행주식 수 근사) |
| `log_mcap` | `log(market_cap)` |

**가격 패턴**

| 피처 | 계산식 |
|---|---|
| `high52w_ratio` | `adj_close / rolling(252).max()` |
| `low52w_ratio` | `adj_close / rolling(252).min()` |
| `maxret_21d` | `simple_ret.rolling(21).max()` |
| `ma_gap_20_60` | `(MA20 − MA60) / MA60` |

**기술적 지표**

| 피처 | 계산식 |
|---|---|
| `rsi_14` | 14일 RSI = `100 − 100/(1 + avg_gain/avg_loss)` |
| `bb_pct` | Bollinger Band %B: `(price − lower) / (upper − lower)` (20일, 2σ) |
| `intraday_range` | `(High − Low) / 전일 adj_close` |
| `vol_surge` | `volume / volume.rolling(21).mean()` |
| `autocorr_21d` | 22일 윈도우 lag-1 수익률 자기상관 |

**리스크 조정 수익**

| 피처 | 계산식 |
|---|---|
| `sharpe_21d` | `mean(exc, 21d) / std(exc, 21d) × √252` |
| `sharpe_63d` | `mean(exc, 63d) / std(exc, 63d) × √252` |
| `sortino_63d` | `mean(exc, 63d) / downside_std(exc, 63d) × √252` |
| `ir_63d` | `mean(active_ret, 63d) / std(active_ret, 63d) × √252` (vs SPY) |

**분포 지표**

| 피처 | 계산식 |
|---|---|
| `skew_63d` | `ret.rolling(63).skew()` |
| `kurt_63d` | `ret.rolling(63).kurt()` |
| `mdd_252d` | `(adj_close / rolling(252).max()) − 1` |

**거시 민감도**

| 피처 | 계산식 |
|---|---|
| `rate_sensitivity` | `excess_ret.rolling(63).corr(Δ^TNX)` |

### 산출 컬럼 순서 (총 38개, indmom 별도 조인)

```python
['ticker', 'gics_sector',
 'adj_close', 'volume', 'dividends', 'split_ratio',
 'simple_ret_1d', 'log_ret_1d', 'excess_ret_1d',
 'mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf', 'mom_factor',
 'mom_1w', 'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12m_skip_1m', 'chmom',
 'vol_20d', 'vol_60d', 'vol_252d',
 'beta_252d', 'idiovol_21d', 'ivol_63d',
 'dollar_vol_21d', 'amihud_21d',
 'market_cap', 'log_mcap',
 'high52w_ratio', 'low52w_ratio', 'maxret_21d', 'ma_gap_20_60',
 'rsi_14', 'bb_pct', 'intraday_range', 'vol_surge', 'autocorr_21d',
 'sharpe_21d', 'sharpe_63d', 'sortino_63d', 'ir_63d',
 'skew_63d', 'kurt_63d', 'mdd_252d',
 'rate_sensitivity']
```

---

## Section 6. 월별 패널 생성

### 처리 순서

#### 1. 패널 로드 + indmom 조인 + 종속변수 계산

종속변수: **21일 선행 초과수익률**

$$\text{fwd\_excess\_ret\_1m} = \prod_{k=1}^{21}(1 + r^{simple}_{t+k}) - 1 - rf_{t \to t+21}$$

#### 2. 21거래일 단위 서브샘플링

레이블 중복(overlap) 제거: 각 종목별로 21거래일 간격으로 샘플링.

#### 3. 동적 유니버스 필터

```python
for date, group in monthly_df.groupby(level='date'):
    members = get_sp500_members_at(date)
    filtered.append(group[group['ticker'].isin(members)])
```

각 날짜에 실제로 S&P 500에 속했던 종목만 유지 → 생존편향 제거.

#### 4. avg_corr 계산

63거래일(≈ 90일) 롤링 윈도우 내 전체 종목 간 페어와이즈 상관계수의 평균.

```python
for date in sorted(monthly_df.index.unique()):
    window = daily_ret_df.loc[date - 90d: date]
    corr_m = window.corr()
    avg_corr[ticker] = corr_m[ticker].drop(ticker).mean()
```

#### 5. 횡단면 랭킹

날짜별 백분위 랭킹 (`rank(pct=True)`):

| 피처 | 기준 |
|---|---|
| `ret_rank_1m` | `mom_1m` |
| `ret_rank_3m` | `mom_3m` |
| `vol_rank` | `vol_20d` |

---

## Section 7. 저장

- **저장 경로**: `data/monthly_panel.csv`
- **출력 통계**: 피처별 NaN 비율, 섹터별 종목 수, 전체 컬럼 목록

---

## 완료 요약

| 항목 | 내용 |
|---|---|
| 유니버스 | S&P 500 동적 구성 (2004~2024 기간 중 편입 종목 전체, GICS 섹터 보유 종목) |
| 생존편향 | Wikipedia 변경 히스토리 역방향 재구성으로 제거 |
| 수집 기간 | 2004-01-01 ~ 2024-12-31 (20년) |
| 피처 수 | ~50개 (수익률 · 모멘텀 · 변동성 · 리스크 · 유동성 · 기술적 · 거시) |
| 횡단면 피처 | ret_rank_1m, ret_rank_3m, vol_rank, avg_corr (월별 패널 단계에서 계산) |
| 종속변수 | `fwd_excess_ret_1m`: 1개월 선행 초과수익률 |
| 산출물 | `data/monthly_panel.csv` |

---

## 다음 단계: `02_XGBoost_WalkForward.ipynb`

- **학습 방식**: 누적 확장 윈도우(expanding IS) + OOS = 1개월
- **모델**: XGBRegressor (Huber loss) + Optuna 하이퍼파라미터 탐색
- **정규화**:
  - 개별 주식 피처 → 횡단면 랭크 → [-1, 1] 정규화
  - 거시 피처(FF 팩터, rate_sensitivity 등) → rolling z-score
- **평가**: R²_OOS (benchmark=0, 논문 기준), IC (Information Coefficient) 분석
- **출력**: Q (예측 수익률), Ω (불확실성) → Black-Litterman 입력
