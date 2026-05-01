# low_risk 파이프라인 전체 구조

> 저위험 이상현상(Low-Risk Anomaly) 기반 액티브 ETF — GARCH + Black-Litterman 포트폴리오 구축

---

## 전체 흐름 요약

```
01_DataCollection
      ↓  monthly_panel.csv
02_LowRiskAnomaly        03_VolatilityEDA
(투자 테제 검증)           (ML 적용 타당성 검증)
      ↓                         ↓
04_Prior_Universe_Analysis (유니버스·Prior 설계 결정)
      ↓
05_VolatilityPrediction   (GARCH walk-forward 예측 + 평가)
      ↓  vol_predicted.csv
┌─────────────────────────────────────────────────────────┐
│  실험 단계 (Q 방식 탐색)                                    │
│  06a_Q_Sensitivity_Predicted   예측 vol Q 민감도 + Expanding-Q│
│  06b_Q_Sensitivity_Baseline    Baseline vol Q 민감도 + Exp-Q │
│  07_BL_Q_Methods      예측 vol × Q 추정 방식 8종 비교 (비-레짐) │
└─────────────────────────────────────────────────────────┘
      ↓  Q_OPTIMAL (06a expanding 마지막 Q* 자동 로드)
08_Regime_Q_Portfolio    ← 레짐 기반 Q 전략 (Regime3 + Regime+λ)
                           SPY 12M vol 분위수로 명시적 레짐 분류

── 참고용 ──────────────────────────────────────────────────
99_baseline              Baseline vol(vol_21d) 직접 사용 + 다중 Q 비교
```

---

## 공통 유틸리티: `bl_utils.py`

코어 BL 파이프라인(노트북 04, 06~08, 98, 99)에서 공유하는 함수 모음. 각 노트북은 imports 셀에서 `from bl_utils import *` 한 줄로 모든 공통 함수를 사용한다.

| 카테고리 | 함수 |
|---------|------|
| 핵심 BL | `compute_sigma`, `compute_pi`, `build_P`, `compute_omega`, `black_litterman`, `optimize_portfolio` |
| 성과 평가 | `performance(ret, rf, label, verbose=False)` — Sharpe / Calmar / MDD / 누적수익 dict 반환 |
| Q 추정 | `compute_Q_hist`, `compute_Q_momentum`, `compute_Q_lambda`, `compute_Q_raw_lambda`, `compute_Q_pi_ratio`, `compute_Q_vol_spread`, `compute_Q_ff3` |
| FF3 | `download_ff3()` — Tuck 서버에서 월별 FF3 다운로드 (소수 단위) |
| 시각화 | `drawdown(ret)`, `rolling_sharpe(ret, rf, window=12)`, `shade_high_vol(ax, dates, ...)` |

이 모듈 하나로 이전에 7~8개 노트북에 중복 정의되어 있던 함수들을 단일 진실 원천(single source of truth)으로 통합했다. 함수 시그니처/이름 통일:
- `q_lambda` → `compute_Q_lambda`, `q_pi_ratio` → `compute_Q_pi_ratio`, `_download_ff3` → `download_ff3`
- `performance(ret, label)` → `performance(ret, rf, label)` (rf를 명시적 인자로)
- `compute_Q_raw_lambda(lam_raw, q_base, lam_mean)` 신규 추가 — raw λ 부호 기반 자연 게이팅 (07에서 사용)

---

## 파일별 상세 설명

### 01_DataCollection.ipynb

**역할**: 전체 파이프라인의 원시 데이터 생성

**입력**
- Wikipedia S&P500 역사적 편출입 내역 (scraping)
- yfinance: S&P500 전종목 + SPY + ^IRX 일별 OHLCV (2004~2025)

**처리 단계**
| 단계 | 내용 |
|------|------|
| 1 | Wikipedia 편출입 히스토리 역방향 재구성 → 월별 멤버십 구축 (생존편향 완화) |
| 2 | GOOGL/GOOG 중복 제거 (`DUPLICATE_DROP`) |
| 3 | 일별 가격 이상값 탐지 (하루 10배↑ 변화 = yfinance 오류) |
| 4 | 일별 피처 계산: 로그수익률, vol_252d, beta_252d, log_mcap |
| 5 | 월말 리샘플링 + S&P500 편입 종목만 필터링 → 패널 구성 |

**출력**: `data/monthly_panel.csv` — `(date, ticker)` 멀티인덱스, 컬럼: `ret_1m, vol_21d, vol_252d, beta_252d, log_mcap, spy_ret, rf_1m`

**파라미터**
- `PANEL_START = '2004-01-01'` (수집 시작)
- 이상값 기준: 하루 10배 이상 가격 변화

---

### 02_LowRiskAnomaly.ipynb

**역할**: 투자 테제 검증 — "저위험 주식이 고위험보다 위험 대비 수익이 높은가?"

**입력**: `data/monthly_panel.csv`

**방법**: 포트폴리오 정렬(Portfolio Sort)
- `beta_252d` / `vol_252d` 각각 기준으로 5분위 분류 (Q1=저위험, Q5=고위험)
- 동일가중 포트폴리오, 다음 달 수익률 측정 (look-ahead bias 방지)
- 전기간(2004~2025) + 최근(2016~2025) 구간 분석

**확인 지표**
- 분위별 연환산수익률, 변동성, Sharpe Ratio
- 누적 수익률 시계열 (vs SPY)
- Q별 평균 시가총액 순위 (소형주 편향 여부 확인)

**결론**: 저변동성 Q1이 고변동성 Q5 대비 높은 누적수익률 + 높은 Sharpe → 이상현상 확인

---

### 03_VolatilityEDA.ipynb

**역할**: ML 변동성 예측 적용의 통계적 타당성 검증

**입력**: `data/monthly_panel.csv` (vol_21d 컬럼 사용)

**검정 항목**

| 검정 | 목적 | 기준 |
|------|------|------|
| ADF (Augmented Dickey-Fuller) | 정상성 확인 — ML 회귀 적용 전제조건 | p < 0.05 |
| Ljung-Box (lag-1) | 자기상관 유의성 — 예측 가능성 존재 여부 | p < 0.05 |
| ARCH LM | 변동성 군집 확인 — 단순 평균보다 ARCH/GARCH가 유리한가 | p < 0.05 |

**출력**: 전종목 검정 결과 테이블 + 비율 요약  
(정상성 ~90%↑ / Ljung-Box ~85%↑ / ARCH ~85%↑ 유의 → GARCH 적용 근거 확보)

---

### 04_Prior_Universe_Analysis.ipynb

**역할**: BL 설계 파라미터 확정 — 유니버스 크기, Prior 타입, PCT_GROUP 결정

**입력**: `data/monthly_panel.csv`

**분석 항목**

| Part | 내용 |
|------|------|
| 1 | 커버리지 기준별 유니버스 크기 (50%~100%) |
| 1-1 | p/T 비율 진단 (Ledoit-Wolf 공분산 추정 품질) |
| 2 | Prior 비교 — 시가총액 vs 1/N |
| 3 | 유니버스 규모별 저위험 이상현상 신호 강도 |
| 4 | Mini Backtest — Prior 선택에 따른 BL 성과 비교 |
| 5 | P 행렬 가중치 비교 |
| 6 | 종합 판단 |
| 7 | 동적 유니버스 실험 (N=50/100/150/200/full) |
| 8 | PCT_GROUP IC 단조성 분석 (|t|≥2.0 기준 최소 pct 선택) |

**출력**: `outputs/04_*.png` 시각화 파일들, `outputs/04_dynamic_universe_stats.csv`

---

### 05_VolatilityPrediction.ipynb

**역할**: GARCH(1,1) walk-forward 변동성 예측 + Baseline(`vol_21d`) 대비 성능 평가

**입력**: `data/monthly_panel.csv`

**모델**: GARCH(1,1) — 03 EDA에서 ARCH 효과 85%↑ 유의 확인

**Walk-forward 설계**
- 학습 윈도우(`TRAIN_WINDOW=60`): 직전 60개월 월별 수익률 → GARCH 파라미터 피팅에 사용
- 최소 관측치(`MIN_OBS=36`): 36개월 미만이면 GARCH 미실행, 역사적 변동성(fallback)으로 대체
- 타겟: 다음 달 `vol_21d` (연환산, 1-step ahead)
- 스케일: 수익률 × 100 (percent, 수치 안정성)
- 예측 기간: **2011-01 ~ 2025-12** (`START_PRED = '2011-01-01'`)

**GARCH vs Baseline 평가** (구 04_5_GARCH_Evaluation 통합)

| 분석 | 핵심 지표 |
|------|----------|
| Rank IC 비교 | GARCH IC vs Baseline IC — 절대 예측 정확도 |
| P 행렬 일치율 | 저위험/고위험 분류가 두 방법 간 겹치는 비율 |
| P 분류 정확도 | Precision — 실제 저위험 종목을 맞히는 비율 |
| 변동성 분리도 | 예측 저위험 그룹의 실제 vol이 고위험보다 낮은가 |
| P 팩터 수익률 | long 저위험 − short 고위험 실현 스프레드 |
| 레짐별 분석 | SPY 12M 롤링 변동성 기준 고/중/저변동성 구간별 IC |

**핵심 발견**
- 전기간 IC: GARCH ≈ Baseline (근소 우위)
- **저변동성 레짐**: GARCH IC +0.051 우위 (73.3% 승률)
- **고변동성 레짐**: GARCH IC -0.006 역전 (45.0% 승률) → 고변동 구간에서 GARCH 신뢰도 저하 → 09 레짐 적응형 Q의 동기

**출력**
- `data/vol_predicted.csv` — `(date, ticker, vol_pred)` long format
- `outputs/05_garch/` — IC 비교, factor returns, p_overlap, p_precision 등 평가 산출물

---

### 06a_Q_Sensitivity_Predicted.ipynb / 06b_Q_Sensitivity_Baseline.ipynb

**역할**: Q_FIXED 민감도 + **Expanding-Window 최적 Q 선택**으로 사후 선택 편향 제거. Vol 소스별로 분할.

> **2026-05-01 분할**: 옛 `06_Q_Sensitivity` 단일 노트북이 GARCH+Baseline 16개 백테스트로 30분 timeout 발생 → 06a(예측 vol)와 06b(Baseline vol)로 분할. 동시에 expanding-window Q 선택을 도입해 단일 시점 OOS 평가의 사후 선택 편향(전체 Sharpe 본 후 Q 결정)을 제거.

**Pass 1 — In-sample (per-Q)**: 8개 Q 후보로 walk-forward 실행. 전체 OOS Sharpe로 max Q를 선택 — **참고용** (편향 있음).

**Pass 2 — Expanding-Window (no bias)**: 매 시점 t에서 누적 Sharpe[0:t]로 Q* 선택해 t에 적용. 메인 결과.

**입력**:
- 06a: `data/monthly_panel.csv`, `data/vol_predicted.csv`
- 06b: `data/monthly_panel.csv`만 (vol_21d 직접 사용)

**파라미터**
- `TRAIN_WINDOW = 60` / `TAU = 0.1` / `PCT_GROUP = 0.30`
- Q 후보: `[0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]`
- `WARMUP = 24`개월 (초기는 `Q_DEFAULT=0.003` fallback)

**출력 (06a)**
- `outputs/06a_Q_Sensitivity_Predicted/q_sensitivity_pred_stats.csv` — Pass 1 per-Q 통계 (참고)
- `outputs/06a_Q_Sensitivity_Predicted/q_expanding_pred_returns.csv` — Pass 2 시계열 (메인)
- `outputs/06a_Q_Sensitivity_Predicted/q_expanding_pred_log.csv` — 매월 Q* + 후보별 expanding Sharpe → **07/08이 마지막 Q*를 자동 로드**

**출력 (06b)**: 동일 구조, 파일명만 `_baseline_*`로 변경. 99_baseline의 Q 결정 근거.

**시간**: 각 ~15분 (8 Q × 1 vol × 180개월 walk-forward + Pass 2 즉시)

---

### 07_BL_Q_Methods.ipynb

**역할**: 예측 vol(GARCH 또는 LSTM 등 활성 모델) × Q 추정 방식 7종 성과 비교 (비-레짐)

> **이전 구조 정리(2026-05-01)**: 옛 `07_BL_Q_Comparison`(Baseline×5Q), 옛 `08_BL_VolQ_Grid`(2×5 격자), 옛 `10_Q_Adaptive_Comparison`(6종 적응 전략)을 통합. 단일 신호로 Q를 결정하는 비-레짐 방식 7종을 한 노트북에 집중. 명시적 SPY 레짐 분류가 필요한 Regime3·Regime+λ는 08번에서 별도 처리.

**입력**: `data/monthly_panel.csv`, `data/vol_predicted.csv`, `outputs/06a_Q_Sensitivity_Predicted/q_expanding_pred_log.csv` (Q_FIXED = expanding 마지막 Q* 자동 로드)  
**분석 기간**: 2010-01 ~ 2024-12 (15년 OOS)

**비교 대상 Q 방식 8종** (Q_vol_spread 추가)

| 방식 | 공식 | 특징 |
|------|------|------|
| `Q_FIXED` | 06a expanding 마지막 Q* 자동 로드 | 편향 없는 OOS 기준 |
| `Q_hist` | mean(저위험 − 고위험) 60M window | 학습 기간 평균 실현수익 |
| `Q_momentum` | mean(저위험 − 고위험) 12M window | 단기 추세 |
| `Q_lambda` | Q_FIXED × clip(λ_t/λ_mean, 0.1, 3.0) | clipped λ 스케일 (항상 양수) |
| `Q_ff3` | FF3 회귀 → P·r̂ | 팩터모델 기반 |
| `Q_pi_ratio` | Q_FIXED × clip(\|P·π\|/spread_ref, 0.1, 3.0) | CAPM 스프레드 비율 |
| `Q_raw_lambda` | max(0, Q_FIXED × λ_raw/λ_mean) | raw λ 부호 자연 게이팅 |

**벤치마크**: CAPM (BL 없음), SPY

**파라미터**
- `TRAIN_WINDOW = 60` / `TAU = 0.1` / `PCT_GROUP = 0.30`
- `Q_FIXED` (06 자동 로드) / `MOMENTUM_WINDOW = 12` / `LAM_MEAN = 2.5`

**출력**: 9개 전략(Q 7종 + CAPM + SPY) 성과 비교표 + 3-panel 시각화, `data/q_methods_returns.csv`, `outputs/07_BL_Q_Methods/q_methods_stats.csv`, `q_methods_log.csv`(시점별 Q 값)  
**관련 문서**: `07_BL_Q_Methods_RESULTS.md`

> **vol 소스 교체**: 노트북은 `vol_predicted.csv`만 읽으므로, GARCH가 아닌 LSTM/Transformer 등 다른 예측 모델로도 동일 파일 스키마(date, ticker, vol_pred)만 맞추면 코드 수정 없이 그대로 적용된다. 자세한 가이드는 본 문서 하단 "## 예측 모델 교체 가이드" 참조.

---

### 08_Regime_Q_Portfolio.ipynb

**역할**: 최종 포트폴리오 — 레짐 기반 Q 전략(Regime3 + Regime+λ). SPY 변동성 레짐을 명시적으로 분류해 Q를 동적으로 조정.

**동기**: 05 GARCH 평가 결과에서 고변동성 레짐의 GARCH 예측력 저하 확인 (IC -0.006 역전). Q를 레짐별로 다르게 설정해 잘못된 뷰 신호를 억제.

**입력**: `data/monthly_panel.csv`, `data/vol_predicted.csv`, `outputs/06a_Q_Sensitivity_Predicted/q_expanding_pred_log.csv` (Q_OPTIMAL = expanding 마지막 Q* 자동 로드)  
**분석 기간**: 2010-01 ~ 2024-12 (15년 OOS)

**레짐 정의** (expanding quantile — look-ahead bias 없음)
```python
spy_roll_vol = spy_series.rolling(12).std() * sqrt(12)
q33 = spy_roll_vol.expanding(min_periods=24).quantile(0.33)
q67 = spy_roll_vol.expanding(min_periods=24).quantile(0.67)
```

**비교 전략 5종 (BL 3 + 벤치마크 2)**

| 전략 | Q 산출 방식 |
|------|-----------|
| **Regime3** | 저→Q_OPTIMAL / 중→×0.5 / 고→0 (이산 3분류) |
| **Regime+λ** | 고변동성→0 (하드스탑) / 그 외→Q_lambda(λ) (연속 스케일) |
| Fixed-Q | Q_OPTIMAL 항상 고정 (비교 기준선) |
| CAPM | π = λΣw_mkt (BL 없음) |
| SPY | S&P 500 매수보유 |

**파라미터**
- `TRAIN_WINDOW = 60` / `TAU = 0.1` / `PCT_GROUP = 0.30`
- `Q_MID_SCALE = 0.5` / `Q_HIGH_VOL = 0.0` / `LAM_MEAN = 2.5`

**출력**
- 4-panel 시각화 (누적수익률, 낙폭, Q 시계열, 롤링 Sharpe)
- 레짐별 성과 breakdown (Regime3 vs Regime+λ vs Fixed-Q)
- 레짐 confusion matrix (지속률·전이확률)
- Q_HIGH_VOL 민감도 분석 (Regime3)
- 포트폴리오 구성 EDA (Fixed-Q 기준)
- `outputs/08_Regime_Q_Portfolio/` 저장

---

### 99_baseline.ipynb (참고용)

**역할**: GARCH 없이 현재 vol_21d로 직접 분류하는 BL 베이스라인

- 논문 구현 참고: Pyo & Lee (2018)
- Baseline vol 직접 사용 + 다중 Q 방식 통합 비교
- 07/08의 Baseline-vol 대조군 역할 (예측 모델 없이 Baseline만 쓰는 시나리오)

---

## 데이터 파일 의존 관계

```
monthly_panel.csv              ← 01_DataCollection 출력
      ├── 02_LowRiskAnomaly
      ├── 03_VolatilityEDA
      ├── 04_Prior_Universe_Analysis
      ├── 05_VolatilityPrediction
      │         ↓
      │   vol_predicted.csv          ← GARCH(현재) 또는 LSTM(교체 가능)
      │         ├── 06a_Q_Sensitivity_Predicted (예측 vol Q 민감도 + Expanding-Q)
      │         │         ↓
      │         │   outputs/06a_Q_Sensitivity_Predicted/q_expanding_pred_log.csv (07/08이 자동 로드)
      │         ├── 06b_Q_Sensitivity_Baseline  (Baseline vol Q 민감도 + Expanding-Q)
      │         │         ↓
      │         │   outputs/06b_Q_Sensitivity_Baseline/q_expanding_baseline_log.csv (99 참고)
      │         │         ├── 07_BL_Q_Methods         (예측 vol × 비-레짐 7Q)
      │         │         │         ↓  data/q_methods_returns.csv
      │         │         └── 08_Regime_Q_Portfolio   (Regime3 + Regime+λ)
      │         └── 99_baseline                       (Baseline vol 직접 사용)
```

---

## 핵심 파라미터 일람 (코어 파이프라인 공통)

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `TRAIN_WINDOW` | 60 | BL 학습 윈도우 (월) — 일별 공분산 시 T = 60×21 = 1260일 |
| `TAU` | 0.1 | BL 불확실성 스케일 |
| `PCT_GROUP` | 0.30 | 저위험/고위험 분류 비율 (상하위 30%) |
| `MAX_WEIGHT` | 0.10 | 종목 최대 비중 (MVO 제약) |
| `START_PRED` | 2010-01-01 | OOS 백테스트 시작 (전 노트북 통일) |
| `END_PRED` | 2024-12-31 | OOS 백테스트 종료 (15년 = 180개월) |
| λ 범위 | [0.5, 10.0] | CAPM 위험회피계수 클리핑 (`compute_pi`) |

---

## 파일 번호 체계

| 번호 | 파일 | 역할 | 성격 |
|------|------|------|------|
| 01~03 | DataCollection / LowRiskAnomaly / VolatilityEDA | 데이터 수집 / 검증 | 전처리·탐색 |
| **04** | **Prior_Universe_Analysis** | **유니버스·Prior·PCT_GROUP 설계 결정** | **설계** |
| 05 | VolatilityPrediction | 변동성 예측 (GARCH; LSTM 등으로 교체 가능) | 핵심 모델 |
| **06a** | **Q_Sensitivity_Predicted** | **예측 vol Q 민감도 + Expanding-Q (편향 제거)** | **실험** |
| **06b** | **Q_Sensitivity_Baseline** | **Baseline vol Q 민감도 + Expanding-Q** | **실험** |
| **07** | **BL_Q_Methods** | **예측 vol × Q 방식 7종 비교 (비-레짐)** | **실험** |
| **08** | **Regime_Q_Portfolio** | **최종 포트폴리오 (Regime3 + Regime+λ + Q_vol_spread_hard)** | **결과** |
| 99 | baseline | Baseline vol 대조군 (다중 Q) | 참고 |

> **2026-05-01 구조 정리**: 옛 `07_BL_Q_Comparison`(Baseline×5Q), 옛 `08_BL_VolQ_Grid`(2×5 격자), 옛 `10_Q_Adaptive_Comparison`(6종 적응 전략), 옛 `98_2006_baseline`(2006~2024 장기) 모두 폐기. 비-레짐 Q 방식 8종은 새 `07_BL_Q_Methods`에, 레짐 기반 Q 전략 3종(Regime3 + Regime+λ + Q_vol_spread_hard)은 새 `08_Regime_Q_Portfolio`에 분담. OOS는 2010-01 ~ 2024-12로 통일.

---

## 예측 모델 교체 가이드 (GARCH ↔ LSTM ↔ ...)

### 핵심 인터페이스: `data/vol_predicted.csv`

다운스트림 노트북(06, 08, 09, 10, 99)은 **모두 `data/vol_predicted.csv` 한 파일**만 읽는다. 이 파일은 model-agnostic이며 활성 예측 모델의 출력으로 덮어써진다.

**스키마 (long format)**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `date` | datetime (월말) | 예측 대상 월 |
| `ticker` | str | 종목 코드 |
| `vol_pred` | float | 다음 달 연환산 변동성 예측값 |

LSTM(또는 다른 모델)이 동일 스키마로 `vol_predicted.csv`를 생성하기만 하면 06~10, 99의 코드 수정 없이 그대로 작동한다.

### GARCH/LSTM 동시 보존 (발표 시나리오)

05_VolatilityPrediction 실행 시 두 곳에 저장된다:
- `data/vol_predicted.csv` — **활성 예측 파일** (다운스트림이 읽음)
- `data/vol_predicted_garch.csv` — GARCH 영구 백업

LSTM 노트북(추후 추가 예정)도 동일 패턴을 따른다:
- `data/vol_predicted.csv` ← LSTM 결과로 덮어씀
- `data/vol_predicted_lstm.csv` ← LSTM 영구 백업

### 모델 전환 절차

```bash
# GARCH 결과로 다운스트림 분석
cp data/vol_predicted_garch.csv data/vol_predicted.csv
# → 06, 08, 09, 10, 99를 차례로 재실행

# LSTM 결과로 다운스트림 분석
cp data/vol_predicted_lstm.csv data/vol_predicted.csv
# → 06, 08, 09, 10, 99를 차례로 재실행
```

또는 LSTM 노트북 마지막 셀에서 `vol_predicted.csv`로 저장하면 자동으로 활성 모델이 LSTM이 된다.

### 변수명 규약

다운스트림 노트북에서 모델별 라벨이 박힌 변수명은 사용하지 않는다:

| 의미 | 사용할 변수명 |
|------|--------------|
| 예측 vol에 대한 최적 Q | `Q_PRED` (옛 `Q_GARCH` 폐기) |
| Baseline vol에 대한 최적 Q | `Q_BASELINE` (06에서만 사용) |
| 예측 모델 비의존 통합 Q | `Q_OPTIMAL` (09, 10) |

이 규약 덕분에 LSTM/Transformer 등 다른 모델로 교체해도 변수명 의미가 흐려지지 않는다.

### 99_baseline의 Q 방식 적용

99_baseline은 `vol_21d`(현재 실현 vol)을 P 행렬 분류에 직접 사용하는 baseline 노트북으로, FF3·Hist·Momentum·Lambda 등 다양한 Q 추정 방식을 통합 비교한다. 이 Q 방식들은 모두 [bl_utils.py](bl_utils.py)에 함수 형태로 추출되어 있어, **LSTM vol을 사용하는 새 노트북에서도 한 줄 import로 즉시 재사용 가능**하다:

```python
from bl_utils import (compute_Q_hist, compute_Q_momentum,
                     compute_Q_lambda, compute_Q_pi_ratio, compute_Q_ff3)
```
