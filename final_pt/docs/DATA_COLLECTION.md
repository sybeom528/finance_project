# 데이터 수집 구조 (01_DataCollection.ipynb)

## 개요

저위험 아노말리(Low-Risk Anomaly) + LSTM 변동성 예측 + Black-Litterman 연구를 위한 S&P500 데이터 수집 파이프라인.
**한 번 실행 후 캐시로 재사용** — 각 섹션은 파일이 이미 존재하면 스킵한다.

---

## 파라미터

```python
PRICE_START = '2004-01-01'   # yfinance 수집 시작 (rolling 워밍업용)
PRICE_END   = '2026-03-31'   # yfinance 수집 종료 (fwd_ret_1m 계산 여유분, 2025-12 + 1Q)
PANEL_START = '2005-01-01'   # 월별 패널 시작 (START_PRED=2010 기준 60개월 훈련 확보)
PANEL_END   = '2025-12-31'   # 월별 패널 종료 (TEST + HOLD_OUT 모두 포함)
```

**분석 구간**: 2010-01 ~ 2025-12 (BL walk-forward)
- TEST: 2010-01 ~ 2023-12 (K_CUT)
- HOLD_OUT: 2024-01 ~ 2025-12 (Streamlit 대시보드 검증용)

**훈련 윈도우**: 60개월 → 2010-01 첫 예측 가능

---

## 섹션별 흐름

### 1. 유니버스 구성
- Wikipedia S&P500 현재 구성 + 변경 히스토리 → **역방향 재구성**
- 매월 그 시점에 실제 편입된 종목만 패널에 포함 (생존편향 완화)
- 중복 제거: `GOOG` → `GOOGL`만 사용, `BRK.A` → `BRK.B`만 사용
- **결과**: 833개 역사적 유니버스

저장 파일:
- `sp500_membership.pkl`: `{date: frozenset(tickers)}` — 월말별 편입 종목 (267개월)
- `universe.csv`: ticker + gics_sector (833개)

### 2. 가격 데이터 수집
- yfinance 배치 다운로드 (200개씩) — SPY + USMV + SPLV + ^IRX + 833개 유니버스
- `auto_adjust=True` (수정 주가 기준)

저장 파일:
- `prices_raw.pkl`: MultiIndex 컬럼 (Price × Ticker) — OHLCV

### 3. 오염 티커 탐지 및 제거
탐지 후 시각화(근거 차트) → 제거 순서로 진행.

**기준 A**: 하루 사이 가격 10배 이상 변화 (yfinance 데이터 오류)
**기준 B**: 60일 이상 연속 0 log_return (티커 혼재/데이터 오류)

제거 결과 13개 (1차 실행 시):
```
['AMCR', 'BMC', 'CBE', 'CFC', 'CPWR', 'GLK', 'HOT', 'MEE', 'POM', 'PTV', 'RSH', 'SW', 'TIE']
```

제거 후 `prices_raw.pkl` 덮어씀. 캐시 재실행 시 추가 제거 0건 (이미 정제됨).
close shape: **(5595, 824)**

### 4. 발행주식수 수집
- yfinance `get_shares_full()` — 시가총액 계산용
- 수집 실패 종목 (51개)은 Close 가격으로 log_mcap 대체 (순위 근사치)

저장 파일:
- `shares_outstanding.pkl`: `{ticker: pd.Series}` — 782개 종목

### 5. 피처 계산 (compute_features)
일별 계산 후 월말(ME)로 리샘플링.

| 변수 | 계산 방식 | 용도 |
|---|---|---|
| vol_21d | rolling(21).std(log_ret) × √252 (연환산) | 단기 변동성 EDA |
| vol_60d | rolling(60).std(log_ret) × √252 | 중기 변동성 EDA |
| vol_252d | rolling(252).std(log_ret) × √252 | **02b 저위험 분류 기준** |
| beta_252d | cov(excess_ret, mkt_ret) / var(mkt_ret), rolling 252일 | **02b 저베타 분류 기준** |
| close | 일별 → 월말 last | log_mcap 계산 |
| log_mcap | log(Close × shares) — shares 없으면 log(Close) | BL prior 가중치 |
| fwd_ret_1m | 21거래일 복리수익률, 다음 달 기준 | **02b portfolio sort 측정 (forward)** |
| ret_1m | 월말 Close pct_change | 02b 시각화·요약 |
| spy_ret, rf_1m | SPY 월수익률, ^IRX 환산 무위험률 | CAPM α 회귀 |
| usmv_ret, splv_ret | USMV, SPLV 저변동 ETF 월수익률 (2011~) | 저변동 ETF 카테고리 벤치마크 |
| gics_sector | GICS 섹터 | 섹터 분석 |

**Look-ahead bias 방지**: fwd_ret_1m 외 모든 변수는 과거 데이터만 사용. fwd_ret_1m 은 portfolio sort 의 측정 변수로만 사용 (forward portfolio sort 표준, Frazzini-Pedersen 2014).

### 6. 월별 패널 구성
- 일별 피처 → 월말 리샘플링(last)
- **생존편향 필터**: sp500_membership 기준, 그 달에 실제 편입된 종목만 포함
- ret_1m: 월말 Close 기준 pct_change (생존편향 필터 이전 전체 가격 기준 계산)
- rf_1m, spy_ret, usmv_ret, splv_ret 조인

저장 파일:
- `monthly_panel.csv`: **(103,878행 × 13컬럼), 617개 종목, 2005-01 ~ 2025-12**

### 7. 일별 수익률 저장
```python
daily_ret = np.log(close / close.shift(1))
```
오염 티커 제거 후 저장.

**용도**:
- BL walk-forward 백테스트 (bl_functions.py) 에서 일별 LW 공분산 추정
- 03b LSTM 입력의 원천 (rv_d/rv_w/rv_m 계산)

**멤버십 필터 없음**: prices_raw.pkl 전체(2004-01 ~ 2026-03)에서 직접 계산.
S&P500 편입 여부와 무관하게 yfinance가 수집한 모든 기간의 수익률이 포함됨.
→ 공분산 추정 시 편입 이전 데이터도 활용 가능.

저장 파일:
- `daily_returns.pkl`: **(5595거래일 × 824종목), NaN 31.5%**

**NaN 31.5% 발생 원인** (멤버십 필터와 무관):
| 원인 | 설명 |
|---|---|
| 상장폐지 종목 | 상폐일 이후 가격 없음 → NaN |
| IPO가 2004 이후인 종목 | IPO 이전 가격 없음 → NaN |
| yfinance 미수집 종목 | 데이터 전무 (상장폐지/미수집) |

### 8. 보조 데이터 수집
| 파일 | 내용 | 용도 |
|---|---|---|
| `sector_etf.pkl` | 11개 섹터 ETF 12개월 수익률 (indmom) | 05b_Analyze 섹터 분석 |
| `ff_factors_daily.csv` | FF5 + MOM 일별 | CAPM α 회귀 (02b §5, 05b_Analyze) |
| `ff5_monthly.csv` | FF5 + MOM 월별 | 05b_Analyze alpha decomposition (6-factor 회귀) |
| `macro_daily.csv` | VIX, t10y2y, DXY, 원자재, FRED | HMM 레짐 분류 + 03b LSTM vix 외생변수 입력 |

매크로 변수: `wti_crude, dxy, gold, copper, silver, skew_idx, vix, t10y2y, icsa, sahm, cpi, unrate`

---

## 저장 파일 목록

```
data/
├── sp500_membership.pkl    # 월별 S&P500 편입 종목 (267개월)
├── universe.csv            # 833개 역사적 유니버스
├── shares_outstanding.pkl  # 발행주식수 시계열 (782개)
├── prices_raw.pkl          # 일별 OHLCV (오염 13개 제거 후)
├── daily_returns.pkl       # 일별 로그수익률 (824종목) — LW 공분산 + 03b LSTM 입력
├── monthly_panel.csv       # 월별 패널 (617종목 × 13변수) — 02b 검증 + BL 백테스트
├── sector_etf.pkl          # 섹터 ETF 수익률 (11개 섹터)
├── ff_factors_daily.csv    # FF5+MOM 일별 — CAPM α 회귀
├── ff5_monthly.csv         # FF5+MOM 월별 — 05b_Analyze alpha decomposition
└── macro_daily.csv         # 매크로 12개 — HMM + 03b LSTM vix 입력
```

---

## 유니버스 → 패널 종목 수 흐름

```
유니버스 전체:      833개
오염 티커 제거:     -13개  (1차 실행 시, 캐시 후 -0개)
가격 데이터 전무:   -?개   (yfinance 미보유, 상장폐지 등)
데이터 252일 미만:  -?개   (상장 기간 짧음, 피처 계산 스킵)
시점별 멤버십 필터:        (각 월에 S&P500 편입 종목만)
────────────────────────────
최종 패널 종목:     617개  (2005-2025 중 한 번이라도 편입된 종목)
```

월별로는 332~498 종목 (평균 412). 시점에 따라 S&P500 구성이 변하므로 월별 종목 수가 다름.

---

## 주의사항

- **데이터 재수집 불필요**: 캐시 파일 존재 시 자동 스킵. 재수집 필요 시 해당 파일 삭제 후 실행.
- **오염 티커 13개 고정**: prices_raw.pkl에 이미 반영되어 있음. 01_DataCollection 재실행 시 0개 추가 제거 (이미 정제됨).
- **PANEL_START = 2005**: START_PRED=2010 기준 60개월 훈련 데이터 확보 목적. 2010 이전 데이터는 훈련 윈도우로만 사용.
- **fwd_ret_1m**: forward 21d 수익률. 02b portfolio sort 의 *측정 변수* 로 사용 (Frazzini-Pedersen 표준). BL 모델 입력으로는 사용하지 않음.
- **daily_returns.pkl NaN 31.5%**: 멤버십 필터 미적용. NaN은 상장폐지 이후 구간, 2004 이후 IPO 종목의 상장 전 구간, yfinance 미수집 종목에서 발생.
- **USMV/SPLV 출시일**: USMV 2011-10-18, SPLV 2011-05-05 — 그 이전은 NaN (정상).
- **^IRX NaN**: 공휴일 5일 → ffill().bfill() 처리, rf_daily NaN 0개.
