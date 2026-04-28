# Phase 2 — Black-Litterman 통합 단계 (Phase 1.5 → Low-Risk Anomaly BL)

## Context

### 왜 이 단계가 필요한가

[Phase 1.5](../Phase1_5_Volatility/) 의 v8 Performance-Weighted Ensemble 이 **변동성 예측의 학술 베이스라인 (HAR-RV) 을 통계적으로 유의 우위** 로 능가하면서 (DM 검정 6/7 종목 5% 유의), Phase 1.5 의 단일 질문 **"변동성 예측이 가능한가?"** 에 **YES** 라는 답이 도출되었습니다.

본 Phase 2 는 그 결과를 **포트폴리오 구축에 실제로 적용** 하는 단계입니다. 적용 프레임워크는 **Pyo & Lee (2018, Pacific-Basin Finance Journal 51) "Exploiting the Low-Risk Anomaly Using Machine Learning to Enhance the Black-Litterman Framework"** 입니다.

### 본 단계의 핵심 질문

> **"Phase 1.5 v8 ensemble 의 변동성 예측 정확도 향상이, Black-Litterman 포트폴리오의 위험조정 수익으로 이전되는가?"**

평가 지표:
- Sharpe Ratio (위험조정 수익)
- Annual Alpha (벤치마크 대비 초과수익)
- Max Drawdown (위기 방어력)
- Cumulative Return (누적 수익)

비교군 4개:
1. **Phase 2 BL** (본 프로젝트) — Phase 1.5 ensemble 예측 변동성으로 P 행렬 구성
2. **SPY** (S&P 500 ETF) — 시장 벤치마크
3. **EqualWeight (1/N)** — Top 50 동일 가중 (DeMiguel et al. 2009 강력 baseline)
4. **McapWeight** — Top 50 시가총액 가중

---

## 추천 접근법

### 1. 폴더 구조 — `Phase2_BL_Integration/` 신규 (Phase 1, 1.5 와 형제)

```
시계열_Test/
├── Phase1_LSTM/                       # 보존, 변경 금지
├── Phase1_5_Volatility/               # 보존, v8 ensemble 결과 활용
└── Phase2_BL_Integration/             # ⭐ 본 단계
    ├── README.md, PLAN.md, 재천_WORKLOG.md
    ├── 00_setup_and_utils.ipynb       # 환경 부트스트랩 (Phase 1.5 복사)
    ├── 01_universe_construction.ipynb # ⭐ 매년 시총 상위 50 산정
    ├── 02_data_collection.ipynb       # ⭐ 종목별 일별 데이터 수집 + 통합 패널
    ├── 03_phase15_ensemble_top50.ipynb # ⭐ 50 종목 ensemble 학습 (가장 무거움)
    ├── 04_BL_yearly_rebalance.ipynb   # ⭐ BL 백테스트 (매월 리밸런싱, 매년 universe 갱신)
    ├── 05_comparison.ipynb            # ⭐ 4 비교군 + 시각화 + 보고서
    ├── scripts/
    │   ├── universe.py                # 시총 상위 50 + fallback 로직
    │   ├── volatility_ensemble.py     # Performance ensemble (신규 종목 reset)
    │   ├── covariance.py              # 일별 ret 기반 Σ + LedoitWolf
    │   ├── black_litterman.py         # 서윤범 99_baseline 함수 import
    │   ├── backtest.py                # transaction_cost 인자화
    │   └── benchmarks.py              # SPY, 1/N, Mcap, BL
    ├── data/                          # 원천·중간 데이터
    │   ├── universe_top50_history.csv         # 매년 universe (50×N)
    │   ├── prices_daily/{ticker}.csv          # 종목별 일별 OHLCV + Adj Close
    │   ├── ensemble_predictions_top50.csv     # Phase 1.5 ensemble 50 종목 확장
    │   ├── market_data.csv                    # SPY, VIX, ^TNX 일별
    │   ├── ff3_monthly.csv                    # Fama-French (옵션)
    │   └── daily_panel.csv                    # 통합 일별 패널 (서윤범 형식 확장)
    └── outputs/
        ├── 03_ensemble_top50/
        ├── 04_BL_yearly/
        └── 05_comparison/
```

**scripts 전략**:
- 서윤범 [`99_baseline.ipynb`](../../서윤범/low_risk/99_baseline.ipynb) 의 BL 함수 (`compute_pi`, `build_P`, `compute_omega`, `black_litterman`, `optimize_portfolio`) 를 `scripts/black_litterman.py` 로 추출 (변경 0).
- Phase 1.5 v8 ensemble 의 `weights_performance` 로직을 `scripts/volatility_ensemble.py` 로 이전하면서 **신규 종목 reset 분기** 추가 (결정 4).
- 기타 신규 작성 모듈은 본 결정사항 (1~8) 대응.

---

### 2. 8 가지 핵심 결정사항 (사용자 확정, 2026-04-28)

| # | 영역 | 결정 | 구현 위치 |
|---|---|---|---|
| **1** | 거래비용 | **0 default, 인자화** (`transaction_cost: float = 0.0`) | `backtest.py` |
| **2** | 부족 종목 | **51위 이하로 자동 대체** (max 80 후보 검토) | `universe.py` |
| **3** | 시총 데이터 출처 | **서윤범 01_DataCollection 로직 재사용** (Wikipedia + yfinance) | `universe.py` |
| **4** | Performance ensemble warmup | **신규 편입 종목만 0.5/0.5 reset, 기존 종목은 history 유지** | `volatility_ensemble.py` |
| **5** | BL Σ (공분산) | **일별 ret 으로 추정 후 × 21 월별 환산** + LedoitWolf shrinkage | `covariance.py` |
| **6** | 벤치마크 | **SPY + 1/N + Mcap + BL_Phase15 (4종)** | `benchmarks.py` |
| **7** | OOS 구조 | **Phase 1.5 와 동일 — 21일 forward, 매월 walk-forward** | `04_BL_yearly_rebalance.ipynb` |
| **8** | 상장폐지 처리 | **남은 종목 시가총액 비중으로 비례 전이** | `backtest.py` |

#### 결정 1. 거래비용 인자화 (예시)

```python
def backtest_strategy(weights_history, returns, transaction_cost: float = 0.0):
    """transaction_cost: 매 거래 회전당 비율 (예: 0.001 = 0.1%)"""
    portfolio_returns = []
    prev_w = None
    for date in weights_history.index:
        cur_w = weights_history.loc[date]
        cost = (cur_w - prev_w).abs().sum() * transaction_cost if prev_w is not None else 0
        gross_ret = (cur_w * returns.loc[date]).sum()
        portfolio_returns.append(gross_ret - cost)
        prev_w = cur_w
    return portfolio_returns
```

→ 본 단계 1차 결과는 `transaction_cost=0` 으로 내고, 추후 0.0005 / 0.001 / 0.002 로 민감도 분석.

#### 결정 4. Performance ensemble 가중치 — 신규 종목만 reset

```python
def get_yearly_weights(cur_universe, prev_universe, weight_history):
    """매년 1월 universe 변경 시 가중치 처리.
    
    - 기존 종목: 이전 마지막 fold 가중치 유지 (history 보존)
    - 신규 편입 종목: 0.5/0.5 warmup
    """
    weights = {}
    for ticker in cur_universe:
        if ticker in prev_universe and ticker in weight_history:
            weights[ticker] = weight_history[ticker][-1]   # 이전 마지막 fold
        else:
            weights[ticker] = {'w_v4': 0.5, 'w_har': 0.5}  # 신규 reset ⭐
    return weights
```

#### 결정 5. BL Σ — **일별 ret 으로 추정 후 × 21 월별 환산** (사용자 결정 2026-04-28)

```python
from sklearn.covariance import LedoitWolf

# IS 5년 일별 수익률
returns_daily_is = daily_panel.pivot('date', 'ticker', 'log_ret').loc[is_start:is_end]

# Ledoit-Wolf shrinkage on daily (T/N=25.2 안정)
lw = LedoitWolf().fit(returns_daily_is.values)
Sigma_daily = pd.DataFrame(lw.covariance_,
                           index=returns_daily_is.columns,
                           columns=returns_daily_is.columns)

# 월별 환산 (i.i.d. 가정 하 단순 21 곱)
Sigma_monthly = Sigma_daily * 21    # ⭐ 핵심 한 줄
```

**환산 근거 (i.i.d. 근사)**:

```
가정:
  E[r_d] ≈ 0 (일별 평균 0)
  Cov(r_d_t, r_d_s) = 0 for t ≠ s (자기상관 0)

수학:
  Var(r_m) = Var(Σ_t r_d_t) = 21 × Var(r_d)
  Cov(r_m_X, r_m_Y) = 21 × Cov(r_d_X, r_d_Y)

→ Σ_m = 21 × Σ_d
```

**Phase 1.5 와의 시간 스케일 정합**:

| BL 입력 | 단위 | Phase 1.5 와 정합? |
|---|---|---|
| Σ_monthly | 월별 (= 21일) | ✅ Phase 1.5 OOS = 21일 |
| Q = 0.003 | **월 0.3%** (= 연 3.6%) | ✅ 월별 의미 |
| π = λ × Σ × w_mkt | 월별 | ✅ |
| Ω = τ · P · Σ · P^T | 월별 | ✅ |
| μ_BL | 월별 | ✅ |
| MVO 출력 (가중치) | **매월 1회 갱신** | ✅ Phase 1.5 OOS 빈도 |
| 백테스트 리밸런싱 | 매월 | ✅ |

→ **모든 BL 입력이 월별 단위로 통일**. Phase 1.5 의 21일 forward 예측 (= 월별 forward 변동성) 과 1:1 매칭.

**가정 위반 영향**:
- S&P 500 일별 ret lag 1 자기상관 ~0.02~0.05 → 21일 누적 분산 과소추정 ~5% (수용 가능)
- LedoitWolf shrinkage 가 추가로 완화 → 단순 × 21 환산이 충분히 정확

#### 결정 7. OOS = Phase 1.5 일관

```
Phase 1.5 와 동일:
  ├→ OOS 길이: 21 영업일 (1개월)
  ├→ Walk-Forward: 매월 슬라이딩
  ├→ IS 길이: 1,250 영업일 (5년)
  ├→ Embargo: 63 영업일
  ├→ Purge: 21 영업일
  └→ Sequence lookback: 63 영업일

Phase 2 적용:
  ├→ 매년 1월 1일 universe 갱신 (50 종목)
  ├→ 그 해 동안 매월 (12 fold) BL 리밸런싱
  └→ 각 fold: 직전 IS 끝 시점 변동성 예측 → P 빌드 → BL → 21일 보유
```

#### 결정 8. 상장폐지 종목 비중 전이

```python
def handle_delisting(weights, delisted, valid_tickers, mcaps):
    """폐지 종목 비중을 남은 종목 시총 비중으로 비례 배분."""
    for d in delisted:
        delisted_w = weights.pop(d, 0)
        if delisted_w == 0: continue
        remaining = [t for t in valid_tickers if t != d]
        total_mcap = sum(mcaps[t] for t in remaining)
        for t in remaining:
            weights[t] += delisted_w * (mcaps[t] / total_mcap)
    return weights
```

---

### 3. 데이터 요구량 — 일별 단위 정밀 산정

#### 3-1. 단일 OOS 연도 t 의 데이터 범위

| 영역 | 길이 (영업일) | 기간 (예: t=2024) |
|---|---|---|
| OOS | 252 | 2024-01-01 ~ 2024-12-31 |
| IS | 1,260 (5년) | 2019-01-01 ~ 2023-12-31 |
| Embargo | 63 | IS와 OOS 사이 |
| Purge | 21 | OOS 직전 |
| Sequence lookback | 63 | IS 시작 직전 (워밍업) |
| HAR monthly | 22 | seq_len 직전 |
| Forward target | 21 | OOS 끝 이후 (마지막 fold 검증) |
| 안전 마진 | 30 | 결측·휴장 |
| **합계** | **~1,732** | **2017-04 ~ 2024-12 (≈ 7.7년)** |

#### 3-2. 6 OOS 연도 (2020 ~ 2025) 종합

| OOS 연도 | 데이터 시작 | 데이터 끝 | universe 산정 시점 |
|---|---|---|---|
| 2020 | 2013-04-01 | 2020-12-31 | 2019-12-31 |
| 2021 | 2014-04-01 | 2021-12-31 | 2020-12-31 |
| 2022 | 2015-04-01 | 2022-12-31 | 2021-12-31 |
| 2023 | 2016-04-01 | 2023-12-31 | 2022-12-31 |
| 2024 | 2017-04-01 | 2024-12-31 | 2023-12-31 |
| 2025 | 2018-04-01 | 2025-12-31 | 2024-12-31 |

→ **종합 데이터 수집**: 2013-04 ~ 2025-12 (약 12.7년 일별).

#### 3-3. Universe 추정 (unique 종목 수)

```
6 연도 × 50 종목 = 300 entries
종목 중복 (대형주 안정성) 고려 → unique ~80 ~ 120 종목
부족 종목 fallback (51~80위) 포함 → 실제 수집 대상 ~150 종목
```

→ **데이터 용량 추정**: 150 종목 × 일별 12.7년 OHLCV ≈ **약 50~80MB** (CSV 압축 전).

---

### 4. 단계별 작업 흐름

#### Step 0 (현재 단계) — 폴더 + 문서 작성

| 산출물 | 상태 |
|---|---|
| `Phase2_BL_Integration/` 폴더 | ✅ 본 세션 |
| `PLAN.md` (본 문서) | ✅ 본 세션 |
| `README.md` | ✅ 본 세션 |
| `재천_WORKLOG.md` (시작점) | ✅ 본 세션 |

#### Step 1 — Universe Construction

| 작업 | 산출물 |
|---|---|
| 서윤범 01 로직 분석·import | — |
| 매년 시총 상위 50 (전년도 12월 31일 종가 기준) | `universe_top50_history.csv` |
| 부족 종목 51위 이하 fallback | (위 csv 의 추가 컬럼) |
| 종목별 데이터 가용성 검증 (≥1,732일) | (검증 통과 종목만 universe) |

**예상 시간**: 1~2 시간 (yfinance 다운로드 포함).

#### Step 2 — Data Collection (일별 패널 구성)

| 작업 | 산출물 |
|---|---|
| 150 종목 × 12.7년 일별 OHLCV 수집 | `prices_daily/{ticker}.csv` |
| 시장 데이터 (SPY, VIX, ^TNX) | `market_data.csv` |
| Fama-French 3팩터 (자동 다운로드) | `ff3_monthly.csv` |
| 무위험 수익률 (FRED DGS3MO) | `risk_free.csv` |
| 통합 일별 패널 (서윤범 형식 확장) | `daily_panel.csv` |

**예상 시간**: 30분 ~ 1시간 (yfinance API 의존).

#### Step 3 — Phase 1.5 Ensemble → 50 종목 확장 (가장 무거움)

| 작업 | 산출물 |
|---|---|
| Phase 1.5 v8 ensemble 코드 import | `scripts/volatility_ensemble.py` |
| 150 unique 종목 × 매년 universe = 학습 대상 | — |
| 종목별 walk-forward (IS=1250, OOS=21, Step=21) 학습 | `ensemble_predictions_top50.csv` |
| 신규 편입 종목 reset, 기존 종목 history 유지 | (위 csv 의 weight 컬럼) |

**예상 시간**: ⚠️ **수 시간** (병목). GPU 활용 시 단축 가능. 종목별 학습 = 150 × 평균 ~80 fold = ~12,000 fold 학습.

**가속 옵션**:
- 옵션 A: 종목별 학습 (Phase 1.5 일관, 정확도 ↑)
- 옵션 B: 풀 학습 (티커 임베딩 추가, 속도 ↑)

→ 1차는 **옵션 A** (Phase 1.5 일관성). 시간 부족 시 옵션 B 검토.

#### Step 4 — BL Yearly Rebalance Backtest

| 작업 | 산출물 |
|---|---|
| 매년 universe 갱신 + 매월 리밸런싱 | `bl_weights.csv` |
| Σ (일별 ret + LedoitWolf, 결정 5) | (계산 중간 산출) |
| P (Phase 1.5 ensemble 정렬, 하위 30% long / 상위 30% short) | (계산 중간 산출) |
| Q = 0.003 고정, Ω = τ·P·Σ·P^T (서윤범 baseline 동일) | (계산 중간 산출) |
| MVO 가중치 + 상장폐지 종목 비중 전이 | `bl_returns.csv` |

**예상 시간**: 1~2 시간.

#### Step 5 — Comparison + Report

| 작업 | 산출물 |
|---|---|
| 4 비교군 (SPY / 1/N / Mcap / BL) Sharpe·Alpha·MDD·CumRet | `comparison_metrics.json` |
| 시각화 (누적수익 / drawdown / rolling Sharpe / 연도별 분해) | `outputs/05_comparison/*.png` |
| 종합 보고서 자동 생성 | `comparison_report.md` |
| τ 민감도 분석 (옵션) | (5 종 τ × 4 비교군) |
| 거래비용 민감도 (0 / 0.05% / 0.1% / 0.2%) | (4 종 × 4 비교군) |

**예상 시간**: 1~2 시간.

---

### 5. 사용자 체크포인트

| Step | 사용자 확인 시점 | 합의 필요 사항 |
|---|---|---|
| 0 | **본 PLAN.md 검토** | 8 결정사항 + 폴더 구조 확정 |
| 1 | universe csv 검토 | 매년 50 종목 list 합리성 (대형주 일관성) |
| 2 | daily_panel.csv 샘플 검토 | 데이터 결측·이상치 수준 |
| 3 | ensemble 50 종목 결과 (RMSE 분포) | 7 종목 baseline 과 일관성 |
| 4 | 첫 OOS 연도 (2020) 백테스트 결과 | 미리 보기, 합리성 점검 |
| 5 | 최종 비교 보고서 | 4 비교군 vs Phase 2 BL 결론 |

---

### 6. Phase 1.5 → Phase 2 통합 매핑 (재정리)

| Phase 1.5 v8 결과 | Phase 2 적용 위치 |
|---|---|
| `Performance Ensemble` 변동성 예측치 | **P 행렬 정렬 기준** ⭐ |
| Walk-Forward 21일 OOS | 매월 BL 리밸런싱 시 직전 21일 예측 |
| 종목별 fold OOS RMSE | Performance ensemble 가중치 결정 |
| 7 종목 (SPY/QQQ/DIA/EEM/XLF/GOOGL/WMT) | 150 unique 종목으로 확장 (Top 50 매년) |

→ **변경 핵심**: P 행렬의 정렬 기준 `vol_252d` (현재 변동성) → `ensemble_predicted_vol` (예측 변동성) ⭐

---

### 7. 누수 방지 체크리스트 (Phase 2 특수)

| # | 함정 | 방어 |
|---|---|---|
| 1 | universe 산정 시 미래 시총 사용 | OOS 시작 직전 영업일 종가 기준 (cutoff = year-1년 12월 마지막 거래일) |
| 2 | Phase 1.5 ensemble 의 OOS 와 Phase 2 백테스트 시점 mismatch | walk-forward IS 끝 = 백테스트 시점 t 동기화 |
| 3 | Σ 추정에 OOS 데이터 포함 | IS 5년만 슬라이싱 (`returns_daily.loc[is_start:is_end]`) |
| 4 | 상장폐지 종목의 미래 가격 참조 | 폐지 시점 이후 NaN 처리 + 비중 전이 |
| 5 | Performance ensemble warmup 가중치가 OOS 정보 사용 | 첫 fold 0.5/0.5 고정 (신규 종목) |
| 6 | 거래비용 모델링 시 future turnover 가정 | 매 시점 직전 weights 와 차이로만 계산 |

---

### 8. 함정·리스크

- **Step 3 (50 종목 ensemble) 가 시간 병목**: GPU 부재 시 학습 시간 ↑↑. 최악 24시간+. **대응**: 옵션 B (풀 학습) 또는 학습 종목 축소 (대형주 30개).
- **공분산 행렬 안정성**: T/N=25.2 가 안정 영역이지만 일별 자기상관 영향 (~5% 분산 과소추정). LedoitWolf 적용으로 보강.
- **상장폐지 종목 비중 전이의 학술 정당성**: 단순 "비중 박탈 후 시총 비례 배분" 은 보수적 가정. 실무에서는 청산가 또는 합병가 적용 가능 — Phase 2 에서는 단순 가정 채택.
- **τ 의 임의성**: Pyo & Lee (2018) 도 명시한 한계. 본 단계는 baseline 의 단일 TAU 채택 후 민감도 분석 옵션.
- **Q = 0.003 의 보수성**: 월 0.3% = 연 3.6% 는 BAB 학술 추정 (연 7.4%) 의 절반. 본 baseline 수치 그대로 채택. 추후 옵션 (FF3 회귀, Phase 1 LSTM 추정) 으로 확장 가능.

---

## Critical Files (수정·신규 작성 대상)

**복사 (변경 없음)**:
- `시계열_Test/Phase2_BL_Integration/00_setup_and_utils.ipynb` — 출처: [Phase1_5_Volatility/00_setup_and_utils.ipynb](../Phase1_5_Volatility/00_setup_and_utils.ipynb)

**서윤범 코드 추출 + 모듈화**:
- `시계열_Test/Phase2_BL_Integration/scripts/black_litterman.py` — 출처: [서윤범/low_risk/99_baseline.ipynb](../../서윤범/low_risk/99_baseline.ipynb) §함수 정의 셀 (`compute_pi`, `build_P`, `compute_omega`, `black_litterman`, `optimize_portfolio`)

**신규 작성**:
- `시계열_Test/Phase2_BL_Integration/scripts/universe.py` — 매년 시총 상위 50 + fallback (결정 2, 3)
- `시계열_Test/Phase2_BL_Integration/scripts/volatility_ensemble.py` — Performance ensemble + 신규 종목 reset (결정 4)
- `시계열_Test/Phase2_BL_Integration/scripts/covariance.py` — 일별 ret + LedoitWolf (결정 5)
- `시계열_Test/Phase2_BL_Integration/scripts/backtest.py` — transaction_cost 인자화 + 상장폐지 처리 (결정 1, 8)
- `시계열_Test/Phase2_BL_Integration/scripts/benchmarks.py` — SPY/1N/Mcap/BL (결정 6)
- `시계열_Test/Phase2_BL_Integration/01_universe_construction.ipynb` — Step 1
- `시계열_Test/Phase2_BL_Integration/02_data_collection.ipynb` — Step 2
- `시계열_Test/Phase2_BL_Integration/03_phase15_ensemble_top50.ipynb` — Step 3
- `시계열_Test/Phase2_BL_Integration/04_BL_yearly_rebalance.ipynb` — Step 4
- `시계열_Test/Phase2_BL_Integration/05_comparison.ipynb` — Step 5

**재사용 (변경 없음)**:
- [Phase1_5_Volatility/results/lstm_ensemble/](../Phase1_5_Volatility/results/lstm_ensemble/) — Phase 1.5 v8 결과 (7 종목, 50 종목 확장의 검증 기준)
- [서윤범/low_risk/99_baseline.ipynb](../../서윤범/low_risk/99_baseline.ipynb) — BL baseline 함수 출처

---

## Verification (End-to-End)

1. **Step 0 산출물 확인** (본 세션)
   - 폴더 구조 + 3 문서 (README/PLAN/WORKLOG)
   - 8 결정사항 명시 + 단계별 계획 합의

2. **Step 1 universe 합리성**
   - 6 연도 × 50 = 300 entries 산출
   - unique 종목 수 80~120 범위 확인
   - 매년 변경 비율 5~15% 확인 (대형주 안정성)
   - look-ahead 차단 검증 (cutoff = year-1년 12월 마지막 거래일)

3. **Step 2 데이터 무결성**
   - 150 종목 × 12.7년 OHLCV 결측치 < 5%
   - 거래일 정렬 (모든 종목 공통)
   - Adj Close 단조증가성 (분할 보정 확인)

4. **Step 3 ensemble 결과 일관성**
   - 7 종목 baseline (Phase 1.5) 과 동일 종목 RMSE 비교
   - 50 종목 평균 RMSE 분포 (0.25 ~ 0.40 예상)
   - Performance ensemble 가중치 history 추적

5. **Step 4 백테스트 누수 검증**
   - Σ, π, Phase 1.5 예측 모두 IS 종료 시점 이전 정보만 사용
   - 매년 1월 1일 universe 변경 시점에 미래 정보 0
   - 상장폐지 종목 처리 정합성

6. **Step 5 4 비교군 결과**
   - Sharpe / Alpha / MDD / CumRet 4 메트릭 × 4 비교군
   - 연도별 분해 (2020 COVID, 2022 긴축 등 체제 변화 진단)
   - τ 민감도 (옵션) + 거래비용 민감도

7. **재현성**
   - seed=42 고정
   - 각 단계 산출물 CSV/JSON 직렬화
   - `Run All` 재실행 시 동일 결과 보장

---

## Appendix: 학술 배경 (참고용)

### A. Pyo & Lee (2018) 핵심 기여

> "ML 변동성 예측 → 자산 분류 → BL 뷰 주입" 을 KOSPI 200 (2005-2016) 에서 검증.
>
> 결과: Sharpe 0.70 (BL) vs 0.36 (CAPM 균형) vs 0.59 (KOSPI200). 알파 +2.46% 연.

### B. 본 Phase 2 의 차별성

| 항목 | Pyo & Lee (2018) | **Phase 2** |
|---|---|---|
| 시장 | KOSPI 200 | **S&P 500 Top 50** |
| 변동성 모델 | ANN (단일) | **LSTM/HAR Performance Ensemble** ⭐ (Phase 1.5 v8) |
| 외부지표 | — | **VIX** (3ch_vix 입력) ⭐ |
| Q (View 수익률) | FF3 회귀 | **0.003 고정** (서윤범 baseline) |
| Ω (View 불확실성) | 동적 분산 | **τ·P·Σ·P^T** (He-Litterman 표준) |
| 종목 변경 | 매월 | **매년 1월 1일** ⭐ |
| 비교군 | KOSPI200, CAPM | **SPY + 1/N + Mcap + BL (4종)** ⭐ |

### C. 본 Phase 2 의 학술 가치

1. **Pyo & Lee (2018) 의 미국 시장 재현·확장**
2. **단일 ANN → LSTM/HAR Ensemble 업그레이드** (DM 검정 통계 우위)
3. **외부지표 (VIX) 의 효과 통합**
4. **DeMiguel et al. (2009) 의 1/N 강력 baseline 직접 비교**
5. **Phase 1.5 의 변동성 예측 정확도가 포트폴리오 성과로 이전되는지 정량 검증**

### D. 미적용 옵션 (추후 단계)

- **공매도 비활성화** (long-only 만): 현재 baseline 은 long-short P 행렬 → long-only 실험 별도
- **τ 민감도 정밀 분석**: Pyo & Lee 의 6 점 (100, 10, 1, 0.1, 0.01, 0.001) 재현
- **Q 동적 추정**: FF3 회귀 또는 Phase 1 LSTM 수익률 추정
- **Ω 동적 추정**: Pyo & Lee 의 전월 백테스팅 잔차 분산 (논문 정합)
- **종목 수 확장**: 50 → 100 → S&P 500 전체 (HRP / factor model 적용)
