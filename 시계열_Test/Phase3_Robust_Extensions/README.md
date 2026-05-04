# Phase 3 — Robust Extensions

> **프로젝트 목표**: ML 변동성 예측 (LSTM + HAR Ensemble) 을 Black-Litterman (BL) 포트폴리오 모델에 통합하여, 17년 (2009~2025) 백테스트로 서윤범 99_baseline (Sharpe 1.157) 대비 우위 여부를 학술적으로 검증합니다.

핵심 비교 가설: **ML 통합 BL** (BL_ml_sw, BL_ml_cs) 이 **Trailing 변동성 BL** (BL_trailing, 서윤범 baseline) 을 통계적으로 능가하는가?

---

## 폴더 구조

```
Phase3_Robust_Extensions/
├── README.md                       # 본 문서
├── *.ipynb                         # 분석 노트북 (16개)
├── 03_v2.py                        # 03_v2.ipynb 의 Python 스크립트 변환본 (백그라운드 실행용)
│
├── data/                           # 입력 데이터 + 학습 결과 캐시
│   ├── prices_daily/               # 종목별 raw OHLCV (yfinance)
│   ├── *.csv                       # universe·panel·예측 결과
│   └── *.pkl                       # 멤버십·sector·BL 캐시
│
├── outputs/                        # 노트북 산출물 (자동 생성)
│   ├── 01_universe_coverage.png
│   ├── 02a_stockwise/, 02a_v2_stockwise/, 02b_crosssec/
│   ├── 03_bl_backtest/, 03_v2_bl_backtest/
│   ├── 04_v2_compare/
│   ├── 05a_v2_eval_stockwise/, 05b_v2_eval_crosssec/, 05c_v2_eval_compare/
│   └── 06_pct_sensitivity/
│
└── scripts/                        # 재사용 가능한 Python 모듈
    ├── setup.py                    # 환경 부트스트랩 (한글 폰트·시드·경로)
    ├── universe.py                 # S&P 500 멤버십 추적
    ├── universe_extended.py        # 멤버십 → flat universe 변환
    ├── data_collection.py          # yfinance OHLCV·시장 데이터 수집
    ├── covariance.py               # LedoitWolf 공분산 추정 + 월별 환산
    ├── volatility_ensemble.py      # LSTM + HAR Ensemble (Phase 1.5 v8)
    ├── models_cs.py                # Cross-Sectional LSTM (Ticker Embedding)
    ├── black_litterman.py          # BL 5-함수 (서윤범 99 추출)
    ├── benchmarks.py               # Equal/Mcap Weight + SPY
    ├── backtest.py                 # 백테스트 엔진 + 메트릭
    └── diagnostics.py              # 4-Layer 평가 모듈 (RMSE/Sharpe/CAPM α/...)
```

---

## 환경 요구사항

| 항목 | 버전/요구 |
|---|---|
| Python | 3.10+ |
| OS | Windows 10/11, macOS, Linux |
| GPU | (선택) RTX 4090 24GB — 02a/02b 학습 시 필요 |
| RAM | 16GB+ 권장 (panel 3.3M 행 로드) |

### 필수 패키지

```
numpy, pandas, scipy, scikit-learn, matplotlib, seaborn,
yfinance, requests, tqdm, pickle, joblib,
torch (GPU 학습 시), pandas-datareader (FRED fallback)
```

한글 폰트 설정: `scripts/setup.py` 의 `bootstrap()` 이 OS별로 자동 분기 (Windows: Malgun Gothic, macOS: AppleGothic, Linux: koreanize-matplotlib).

---

## ⚠️ 외부 Phase 폴더 의존성

본 Phase 3 폴더는 **분석 단계에서는 완전 독립적**이지만, **모델 재학습 단계에서는 Phase 1.5 폴더가 필수**입니다.

### 의존 메커니즘

`scripts/volatility_ensemble.py` 가 import 되는 시점에 Phase 1.5 의 4개 모듈을 동적으로 로드합니다:

```python
# scripts/volatility_ensemble.py (line 73~76)
_dataset    = _load_phase15_module("dataset", "dataset.py")
_models     = _load_phase15_module("models", "models.py")
_train      = _load_phase15_module("train", "train.py")
_baselines  = _load_phase15_module("baselines_volatility", "baselines_volatility.py")
```

→ Phase 1.5 폴더가 없으면 `FileNotFoundError: Phase 1.5 모듈 없음` 에러 발생

### 노트북별 외부 의존성 매트릭스

| 노트북 | Phase 1.5 (`Phase1_5_Volatility/`) | Phase 2 (`Phase2_BL_Integration/`) | 비고 |
|---|---|---|---|
| **01** universe_extended | ❌ 불필요 | ❌ 불필요 | yfinance 만 필요 |
| **02a, 02a_v2, 02b** (학습) | ✅ **필수** | ⚠️ §6 sanity check 시만 | 4개 모듈 동적 로드 |
| **03_v2, 03 (legacy)** | ❌ 불필요 | ❌ 불필요 | PHASE2_DIR import 만 (실제 read 없음) |
| **04, 04_v2** | ❌ 불필요 | ❌ 불필요 | 결과 CSV 만 read |
| **05a, 05a_v2** | ❌ 불필요 | ❌ 불필요 | yfinance (sector mapping, cache 있음) |
| **05b, 05b_v2** | ⚠️ 조건부 | ❌ 불필요 | `data/models_cs/*.pt` 존재 시만 import (현재 SKIP) |
| **05c, 05c_v2** | ❌ 불필요 | ❌ 불필요 | 비교 통계 검정만 |
| **06** pct_sensitivity | ❌ 불필요 | ❌ 불필요 | BL 모듈만 사용 |

### 시나리오별 외부 폴더 요구사항

| 작업 범위 | Phase 1.5 | Phase 2 | 가능 여부 |
|---|---|---|---|
| **분석·시각화·통계만** (03~06, 현재 상태) | ❌ 불필요 | ❌ 불필요 | ✅ **Phase 3 단독 가능** |
| `01_universe_extended` 재실행 (yfinance) | ❌ 불필요 | ❌ 불필요 | ✅ Phase 3 단독 가능 (인터넷 필요) |
| `02a/02b` 재학습 | ✅ **필수** | ⚠️ §6 시만 | ❌ Phase 1.5 폴더 필요 |
| `02a §6` Phase 2 비교 sanity check | ❌ 불필요 | ✅ 필요 | ⚠️ Phase 2 의 `ensemble_predictions_top50.csv` 필요 |

### Phase 1.5 필수 파일 (재학습 시)

`Phase1_5_Volatility/scripts/` 에 다음 4개 파일이 있어야 합니다:

```
Phase1_5_Volatility/
└── scripts/
    ├── dataset.py                  # build_fold_datasets, walk_forward_folds
    ├── models.py                   # LSTMRegressor
    ├── train.py                    # train_one_fold
    └── baselines_volatility.py     # fit_har_rv
```

### 공유 시 권장사항

| 공유 목적 | 포함 폴더 | 가능한 작업 |
|---|---|---|
| **분석 결과 검토** | Phase 3 만 | 03~06 모든 분석 (모델 재학습 불가) |
| **학습 재현 + 분석** | Phase 3 + Phase 1.5 | 모든 노트북 (02a/02b 재학습 포함) |
| **완전 재현** | Phase 3 + Phase 1.5 + Phase 2 | 위 + 02a §6 의 Phase 2 비교 검증 포함 |

---

## 데이터 파일 명세

> ⚠️ 모든 파일은 `data/` 폴더에 위치합니다.

### A. Raw 입력 데이터 (외부 소스)

| 파일 | 출처 | 내용 | 생성 노트북 |
|---|---|---|---|
| `prices_daily/{TICKER}.csv` | yfinance | 종목별 일별 OHLCV (Open, High, Low, Close, Volume), auto_adjust=True | 01 |
| `market_data.csv` | yfinance | SPY·VIX·TNX 일별 종가 (2002~2025) | 01 |
| `risk_free.csv` | yfinance ^IRX → FRED fallback | 무위험 수익률 (3개월 미국채, 일별) | 01 |
| `vix_daily.csv` | yfinance | VIX 일별 (panel 의 vix 컬럼 원본) | 01 |

### B. Universe / 멤버십

| 파일 | 형식 | 내용 |
|---|---|---|
| `sp500_membership.pkl` | dict | `{월말_Timestamp: frozenset(tickers)}` — 시점별 S&P 500 멤버 (Wikipedia 크롤) |
| `universe_full_history.csv` | CSV | 624 unique 종목 flat 리스트 (cutoff_date 기준) |
| `universe_top50_history_extended.csv` | CSV | (구 버전) 연도별 top-50 universe — Phase 2 호환용 |
| `splits_history.pkl` | dict | 종목별 분할 이벤트 (가격 보정 검증용) |
| `shares_outstanding.pkl` | dict | 종목별 유통 주식수 시계열 (시총 계산용) |
| `prices_close_universe.pkl` | DataFrame | universe 전체 종가 wide 매트릭스 (date × ticker) |

### C. Panel (학습·백테스트 입력)

| 파일 | 형식 | 컬럼 |
|---|---|---|
| `daily_panel.csv` | long-format CSV (3.34M 행) | `date, ticker, log_ret, vol_21d, mcap_value, log_mcap, spy_close, rf_daily, vix` 등 |
| `daily_panel_extended.csv` | long-format CSV | (확장 버전) 추가 피쳐 포함 |
| `daily_panel.csv.bak`, `market_data.csv.bak`, `vix_daily.csv.bak` | 백업 | 직전 버전 자동 백업 |

`daily_panel.csv` 컬럼 상세:

| 컬럼 | 정의 |
|---|---|
| `date` | 거래일 |
| `ticker` | 종목 코드 |
| `log_ret` | 로그 수익률 = `log(close_t / close_{t-1})` |
| `vol_21d` | 21일 trailing 표준편차 (BL_trailing 의 P 행렬 입력) |
| `mcap_value` | 시가총액 (USD) |
| `log_mcap` | log(시가총액) |
| `spy_close` | SPY 종가 (시장 벤치마크) |
| `rf_daily` | 무위험 일별 수익률 (^IRX/365) |
| `vix` | VIX 종가 (LSTM 의 4번째 채널) |

### D. Sector / 부가 정보

| 파일 | 내용 |
|---|---|
| `sector_map_combined.pkl` | `{ticker: GICS_sector}` — Wikipedia + yfinance 보완 매핑 |

### E. 학습 결과 (Phase 3 핵심 산출물)

| 파일 | 생성 노트북 | 형식 | 내용 |
|---|---|---|---|
| `ensemble_predictions_stockwise.csv` | 02a / 02a_v2 | long-format (2.47M 행) | 종목별 LSTM 예측: `date, ticker, y_pred_lstm, y_pred_har, y_pred_ensemble, y_true, weight_lstm, weight_har, best_model` |
| `ensemble_predictions_crosssec.csv` | 02b | long-format (2.64M 행) | Cross-Sectional LSTM 예측: 위와 동일 + Ticker Embedding 학습 결과 반영 |
| `ensemble_predictions_partial.csv` | 02a/02b | long-format | 학습 진행 중 매 20 종목마다 중간 저장 (재시작 안전망) |
| `fold_predictions_stockwise.csv` | 02a | long-format | fold 단위 예측 (앙상블 전 단계, 재학습 검증용) |

### F. BL 백테스트 캐시

| 파일 | 생성 노트북 | 내용 |
|---|---|---|
| `scenario_weights_03_v2.pkl` | 03_v2 | dict `{scenario: {date: pd.Series(weights)}}` — 9 시나리오 × 192 리밸런싱 시점 가중치 + diagnostics |
| `scenario_weights_03_dynamic.pkl` | 03_BL_backtest_extended | 동일 형식 (구 버전, Dynamic-Membership Step 7 적용) |
| `bl_weights_sanity_check.pkl` | 02a | 02a 학습 직후 BL 가중치 sanity check (학습 직후 빠른 검증용) |
| `bl_weights_v2_sanity_check.pkl` | 02a_v2 | 동일 (v2) |
| `bl_weights_sanity_check_legacy_static.pkl` | 03_legacy_static | 구 버전 Static-Membership 결과 |
| `bl_weights_sanity_check_step7_dynamic.pkl` | 03 | Step 7 Dynamic-Membership 결과 |
| `bl_metrics_sanity_check.pkl` | 02a | sanity check 메트릭 (Sharpe, MDD 등) |
| `bl_metrics_v2_sanity_check.pkl` | 02a_v2 | 동일 (v2) |

### G. 분석 캐시 (시각화·리포트용)

| 파일 | 내용 |
|---|---|
| `sec7_k_recovery_analysis.pkl` | 02a_v2 §7-k: 누락 종목 sector 회복 분석 |
| `sec7_v2_analyses.pkl` | 02a_v2 §7: 생존편향 정량화 (Wikipedia 809 vs panel 646) |

---

## 노트북 파일 명세

> 노트북 명명 규칙:
> - `XX_*.ipynb`: 원본 또는 가장 최신 버전
> - `XX_v2.ipynb`: 개선판 (Dynamic-Membership·Stale 필터 적용 등)
> - `XX_*_legacy_*.ipynb`: 구 버전 (재현 검증용 보존)

---

### Step 1 — Universe & Panel 구축

#### 📓 `01_universe_extended.ipynb`

| 항목 | 내용 |
|---|---|
| **목적** | OOS 2009~2025 (17년) 전체 S&P 500 멤버 universe (~624 종목) + daily_panel (2002~2025) 구축 |
| **입력** | (외부) yfinance API, Wikipedia |
| **출력** | `universe_full_history.csv`, `daily_panel.csv`, `prices_daily/*.csv`, `market_data.csv` |
| **소요시간** | 첫 실행 30~60분 (yfinance), 캐시 활용 시 1~2분 |
| **인터넷 필수** | ✅ (캐시 없을 시) |
| **재실행 권장** | ❌ — 데이터 일관성 보호 (학습 결과와 mismatch 방지) |

핵심 동작: `build_full_universe_for_panel()` 로 멤버십 추출 → `build_daily_panel_fast()` 로 OHLCV 다운로드·집계.

---

### Step 2 — ML 모델 학습 (LSTM + HAR Ensemble)

#### 📓 `02a_phase15_stockwise_extended.ipynb` & `02a_v2.ipynb`

| 항목 | 내용 |
|---|---|
| **목적** | Phase 1.5 v8 Performance-Weighted Ensemble (LSTM + HAR) 을 624 종목 각각에 별도 학습 |
| **모델 구성** | LSTM v4 (3ch_vix: rv_d, rv_w, rv_m + vix_log, IS=1250, embargo=63, hidden=32) + HAR-RV (1d/5d/22d 이동평균) |
| **앙상블 방식** | Performance-Weighted (Diebold-Pauly rolling) |
| **입력** | `universe_full_history.csv`, `daily_panel.csv` |
| **출력** | `ensemble_predictions_stockwise.csv` (2.47M 행) |
| **소요시간** | 624 종목 × 4-way 병렬 (RTX 4090) → 3~5시간 |
| **GPU 필수** | ✅ |
| **v2 차이** | §7 추가 (생존편향 분석, sector 회복 분석) — 학습 결과는 동일 |

> 4-way 병렬 (8-way 대비 CUDA 컨텍스트 overhead 감소).

#### 📓 `02b_phase15_cross_sectional.ipynb`

| 항목 | 내용 |
|---|---|
| **목적** | 624 종목 전체를 단일 LSTM 으로 학습 (Cross-Sectional, Ticker Embedding 사용) |
| **모델 구성** | 단일 LSTM + 8차원 Ticker Embedding (Gu, Kelly, Xiu 2020) + HAR Ensemble |
| **02a 대비 차이** | 종목 A 의 패턴이 모든 종목 학습에 기여 (정보 공유). 파라미터 수 ~57K (02a 의 1/47) |
| **입력** | `universe_full_history.csv`, `daily_panel.csv` |
| **출력** | `ensemble_predictions_crosssec.csv` (2.64M 행) |
| **소요시간** | 1~2시간 (vectorized, GPU 친화적) |
| **GPU 필수** | ✅ |

> 핵심 가설: Cross-Sectional 이 정보 공유 효과로 종목별 LSTM 보다 RMSE 가 낮은가? → 04 노트북에서 검정.

---

### Step 3 — Black-Litterman 백테스트

#### 📓 `03_v2.ipynb` (현재 권장)

| 항목 | 내용 |
|---|---|
| **목적** | 02a (Stockwise) + 02b (Cross-Sectional) 예측을 BL 에 통합하여 9 시나리오 17년 백테스트 + 서윤범 99 재현 검증 |
| **시나리오 9개** | BL_ml_sw × {mcap, eq, rp} (3) + BL_ml_cs (1) + BL_trailing × {mcap, eq, rp} (3) + EqualWeight (1) + McapWeight (1) |
| **BL 파라미터** | Q_FIXED=0.003, DEFAULT_TAU=0.1, LAM_FIXED=2.5, PCT_GROUP=0.30 (서윤범 99 일관) |
| **Universe 모드** | Dynamic-Membership (시점 t 의 S&P 멤버) ∩ panel ∩ 학습 615 ∩ non-stale (look-ahead bias 차단) |
| **입력** | `daily_panel.csv`, `ensemble_predictions_*.csv` × 2, `market_data.csv`, `sp500_membership.pkl` |
| **출력** | `outputs/03_bl_backtest/` (`metrics_fair.csv`, `returns_*.csv` × 10, `backtest_comparison.png`), `data/scenario_weights_03_v2.pkl` |
| **소요시간** | 약 80분 (192 리밸런싱 × LedoitWolf + 9 BL × SLSQP) |
| **캐시 활용** | ✅ `FORCE_RECOMPUTE=False` 시 `scenario_weights_03_v2.pkl` 로드 → 즉시 결과 |

> Phase 3 결정: 위험회피 계수 λ=2.5 고정 (서윤범 99 일관, He-Litterman 1999 표준).

#### 📓 `03_BL_backtest_extended.ipynb` (구 메인)
- v2 와 동일 구조이나 **6 시나리오**만 (mcap-only). v2 가 weighting 변형 3종 (mcap/eq/rp) 추가하여 더 완전.

#### 📓 `03_BL_backtest_extended_legacy_static.ipynb` (legacy 보존)
- Static-Membership (panel 가용 ∩ 학습) 사용. **look-ahead bias 잠재** — 비교 검증용으로만 보존.

---

### Step 4 — 종합 비교

#### 📓 `04_compare_stockwise_vs_cross.ipynb` & `04_v2.ipynb`

| 항목 | 내용 |
|---|---|
| **목적** | 종목별 (02a) vs Cross-Sectional (02b) 의 예측 정확도 + 포트폴리오 성과 통합 비교 |
| **검정 방법** | RMSE 비교 (Paired t-test + Wilcoxon), Sharpe 비교 (Jobson-Korkie + Block Bootstrap), 시기별 분해 (3 시기 × 6 시나리오) |
| **핵심 질문** | (1) CS 가 종목별보다 RMSE 낮은가? (2) BL_ml_cs 가 BL_ml_sw 보다 Sharpe 우위? (3) ML 통합이 서윤범 BL_trailing 능가? |
| **입력** | `ensemble_predictions_*.csv` × 2, `outputs/03_bl_backtest/returns_*.csv`, `metrics_fair.csv` |
| **출력** | `outputs/04_v2_compare/` 비교 표·시각화 |
| **소요시간** | 5~10분 |

---

### Step 5 — 단독·통합 평가 (4-Layer 진단)

#### 📓 `05a_eval_stockwise.ipynb` & `05a_v2.ipynb`

| 항목 | 내용 |
|---|---|
| **목적** | 02a Stockwise + BL_ml_sw 단독 평가 (다른 시나리오와 비교 없이) |
| **4-Layer 평가** | (1) 변동성 예측 진단: RMSE, QLIKE (Patton 2011), Mincer-Zarnowitz, DM-test, Best model 분포<br>(2) 포트폴리오: Sharpe, CAPM α, IR, Sortino, CVaR, turnover<br>(3) ML→BL 인과: low/high vol hit rate, P 행렬 안정성<br>(4) 시기별 분해: 5 시기 × 모든 메트릭 |
| **외부 호출** | ⚠️ yfinance (sector mapping 보완) — `sector_map_combined.pkl` 캐시 있으면 SKIP |
| **입력** | `ensemble_predictions_stockwise.csv`, `daily_panel.csv`, `outputs/03_bl_backtest/returns_BL_ml_sw.csv` |
| **출력** | `outputs/05a_v2_eval_stockwise/` |

#### 📓 `05a_v2_lstm.ipynb` (2026-05-02 신설 — 종목별 LSTM 12 분석)

| 항목 | 내용 |
|---|---|
| **목적** | 02a Stockwise LSTM 모델의 **변동성 예측 자체** 다차원 진단 (백테스트 무관, 가중치 무관) |
| **12 분석** | §2-A 월별 RMSE / B 종목×시기 RMSE / C Forecast Bias / D Vol Regime / E VIX Tier / F Diebold-Mariano test / G Sector × Best Model / H 가중치 안정성 / I 시기별 산점도 / J Top/Bottom 5 case / K CV fold별 / L y 분포 |
| **공정성 보강** | §2-B/J: **5 시기 cover 503 종목만** (인수/파산 110 종목 제외) |
| **셀 수** | 31 (16 MD + 15 code) |
| **입력** | `ensemble_predictions_stockwise.csv`, `daily_panel.csv`, `vix_daily.csv`, `ticker_sector_mapping.csv` |
| **출력** | `outputs/05a_v2_lstm_diag/` (A_*, B_*, ... L_* 산출 17개 + summary.json) |
| **핵심 결과** | LSTM 0.4298 > HAR 0.3922 > Ensemble 0.3815, Best model: Ensemble 69.7% / HAR 27.2% / LSTM 3.1% |

#### 📓 `05a_v2_lstm_2b_deep.ipynb` (2026-05-02 신설 — §2-B 학술 심화 + 효과크기)

| 항목 | 내용 |
|---|---|
| **목적** | §2-B 학술 심화 — 통계 검정 + 효과크기 + 시각화 (학술 보고서 인용용) |
| **통계 검정** | Heavy-tail (Skew/Kurt/JB/AD), ANOVA Variance Decomposition, Kruskal-Wallis, Pairwise Mann-Whitney (Bonferroni), Welch ANOVA (이분산 robust) |
| **효과크기** | η² (eta), ε² (epsilon), Cohen's d, r (rank-biserial) — Lin 2013 large-n 함정 보강 |
| **시각화 5종** | Sector Boxplot / Sector × Period Heatmap / COVID Impact / Heavy-tail KDE+QQ / Variance Decomp Pie |
| **셀 수** | 24 (12 MD + 12 code) |
| **출력** | `outputs/05a_v2_lstm_diag/B2_*, B3_*, B4_*` |
| **핵심 결과** | Period η²=0.450 LARGE / Ticker η²=0.194 LARGE / Sector ε²=0.121 medium / 14 sig pair 모두 LARGE Cohen's d (가짜 유의 0건) |

#### 📓 `05a_v2_weighting.ipynb` (2026-05-02 신설 — eq/rp/mcap 가중치 비교)

| 항목 | 내용 |
|---|---|
| **목적** | 6 시나리오 (3 가중치 × 2 vol input: BL_ml_sw_mcap/eq/rp + BL_trailing_mcap/eq/rp) Layer 2-4 비교 + ML 효과 분해 |
| **분석 영역** | §2 Layer 2 (9 시나리오 메트릭, OOS+Hold-out 분리) / §3 누적수익 + Drawdown / §4 ML 효과 robustness / §5 Layer 3 인과 / §6 Layer 4 시기별 |
| **셀 수** | 16 (8 MD + 8 code) |
| **입력** | `outputs/03_bl_backtest/returns_BL_*.csv` (9개), `data/scenario_weights_03_v2.pkl` |
| **출력** | `outputs/05a_v2_weighting/` (메트릭 + ML 효과 + 시각화) |
| **핵심 결과** | BL_trailing_mcap Sharpe 1.225 (OOS 1위), Hold-out mcap 환경 ML 효과 +0.643 (부호 역전) |

#### 📓 `05b_eval_crosssec.ipynb` & `05b_v2.ipynb`
- 05a 와 동일 4-Layer 형식이나 **02b CS 모델 + BL_ml_cs** 평가.
- **§2-7 CS 특화**: Ticker Embedding 의 PCA 시각화 + Cosine Similarity 행렬 (학습된 종목 표현 분석).

#### 📓 `05c_eval_compare.ipynb` & `05c_v2.ipynb`

| 항목 | 내용 |
|---|---|
| **목적** | 05a/05b 단독 평가 결과 통합 + 시나리오 간 통계적 검정 |
| **Layer 5 통계 검정** | Paired t-test, Wilcoxon, DM-test (예측), Jobson-Korkie 1981, Memmel 2003 (Sharpe), Hansen MCS 2011 (다중 시나리오) |
| **사전 조건** | 05a, 05b 실행 완료 (`eval_metrics_*.json` 생성), 03 실행 완료 |
| **출력** | `outputs/05c_v2_eval_compare/` 통합 비교 표·시각화 |

---

### Step 6 — 민감도 실험

#### 📓 `06_pct_sensitivity.ipynb`

| 항목 | 내용 |
|---|---|
| **목적** | `pct` 파라미터 (BL 변동성 양극단 long/short 비율, 기본 0.30) 를 0.10~0.40 범위로 변경 시 포트폴리오 성능 변화 측정 |
| **실험값** | pct ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40} (7개) |
| **시나리오** | BL_ml_sw, BL_ml_cs (선택), BL_trailing, EqualWeight (기준선) |
| **메트릭** | Sharpe, 연환산 수익률, Sortino, MDD |
| **핵심 최적화** | Step 1 — 204 시점 × LedoitWolf Sigma 1회 캐싱 (~7분)<br>Step 2 — 7 pct × Sigma 재사용 BL 계산 (~3분)<br>총 ~10분 |
| **격리성** | 기존 02~05 산출물에 영향 없음 (read-only 접근, `outputs/06_pct_sensitivity/` 만 신규 생성) |
| **출력** | Figure 3종 (`fig1_metrics_vs_pct.png`, `fig2_cumulative_selected.png`, `fig3_heatmap_sharpe.png`) + CSV 2종 + `summary.json` |

---

## 실행 가이드 — 권장 순서

### 시나리오 A: 모든 캐시 보유 (현재 상태) — **인터넷·GPU 불필요**

학습 결과 (`ensemble_predictions_*.csv`) 와 BL 캐시 (`scenario_weights_03_v2.pkl`) 가 모두 있으므로 **01, 02 SKIP** 가능합니다.

| 순서 | 노트북 | 캐시 사용 시 |
|---|---|---|
| 1 | `03_v2.ipynb` | ⚡ 즉시 (cache hit) |
| 2 | `04_v2.ipynb` | ⚡ 5~10분 |
| 3 | `05a_v2.ipynb` | ⚡ 5~10분 (sector_map_combined.pkl 캐시 있음) |
| 4 | `05b_v2.ipynb` | ⚡ 5~10분 |
| 5 | `05c_v2.ipynb` | ⚡ 5~10분 |
| 6 | `06_pct_sensitivity.ipynb` | ⚡ ~10분 |

> **03_v2.ipynb 는 캐시가 손상되었거나 재계산이 필요하면** `FORCE_RECOMPUTE = True` 로 변경 후 재실행 (~80분).

### 시나리오 B: Raw 데이터부터 재구축 — **인터넷·GPU·Phase 1.5 폴더 필수**

> ⚠️ 02a/02b 재학습 시 `../Phase1_5_Volatility/scripts/` 의 4개 모듈 (`dataset.py`, `models.py`, `train.py`, `baselines_volatility.py`) 이 필요합니다. 자세한 사항은 위 [외부 Phase 폴더 의존성](#️-외부-phase-폴더-의존성) 섹션 참조.

| 순서 | 노트북 | 소요 시간 | 추가 요구 |
|---|---|---|---|
| 1 | `01_universe_extended.ipynb` | 30~60분 (yfinance) | 인터넷 |
| 2 | `02a_v2.ipynb` (또는 02a) | 3~5시간 (RTX 4090) | Phase 1.5 폴더, GPU |
| 3 | `02b_phase15_cross_sectional.ipynb` | 1~2시간 (RTX 4090) | Phase 1.5 폴더, GPU |
| 4~9 | 03~06 (시나리오 A 와 동일) | ~2시간 합계 | (외부 폴더 불필요) |

---

## 캐시 사용 (FORCE_RECOMPUTE 플래그)

| 노트북 | 플래그 | 캐시 파일 |
|---|---|---|
| 02a, 02a_v2 | `FORCE_RECOMPUTE` (~6 곳) | `ensemble_predictions_stockwise.csv` |
| 02b | `FORCE_RECOMPUTE` (~2 곳) | `ensemble_predictions_crosssec.csv` |
| 03_v2 | `FORCE_RECOMPUTE = False` (default) | `scenario_weights_03_v2.pkl` |
| 03 (legacy) | 동일 | `scenario_weights_03_dynamic.pkl` |
| 05a/05a_v2 | 자동 (sector cache 사용) | `sector_map_combined.pkl` |

`FORCE_RECOMPUTE = True` 로 설정하면 캐시 무시하고 재계산합니다.

---

## 트러블슈팅

### 1. 한글 폰트 깨짐 (□□□)
- `scripts/setup.py` 의 `bootstrap()` 호출 확인
- Linux 환경: `pip install koreanize-matplotlib --break-system-packages`

### 2. `estimate_covariance() missing 2 required positional arguments`
- 03_v2.ipynb 의 §4 에서 `estimate_covariance(sub)` 사용 시 발생
- 수정: `daily_to_monthly(compute_sigma_daily(sub))` 직접 호출 (이미 적용됨)

### 3. cp949 인코딩 에러 (Windows)
- 백그라운드 실행 시 `python -X utf8 script.py` 권장
- `bootstrap(verbose=False)` 로 호출하면 em dash (—) 출력 SKIP

### 4. Jupyter 커널 무한로딩
- 메모리 부족 가능성 (panel 3.3M 행 + 학습) → `python -u -X utf8 03_v2.py` 백그라운드 실행 권장 (`03_v2.py` 가 변환본으로 제공됨)

### 5. 02a/02b 학습 결과 없음
- `data/ensemble_predictions_*.csv` 가 없으면 03 이후 노트북 실행 불가
- 02a (3~5시간) 또는 02b (1~2시간) 학습 필요 — GPU 필수

### 6. `prices_daily/` 폴더 일부 누락
- 01 노트북 재실행 시 누락된 종목만 yfinance 에서 보충 다운로드
- ⚠️ 단, `daily_panel.csv` 를 `overwrite=True` 로 덮어쓰므로 02 학습 결과와 mismatch 위험 — 가능하면 01 재실행하지 마세요.

### 7. `FileNotFoundError: Phase 1.5 모듈 없음`
- 02a/02b 학습 노트북 또는 05b_v2 의 Ticker Embedding 분석 셀 실행 시 발생
- 원인: `scripts/volatility_ensemble.py` 가 Phase 1.5 의 4개 모듈 (`dataset.py`, `models.py`, `train.py`, `baselines_volatility.py`) 을 동적 import 시도
- 해결:
  - **분석 목적**: 03~06 노트북만 실행하면 우회 가능 (`volatility_ensemble` 미사용)
  - **재학습 목적**: `Phase1_5_Volatility/scripts/` 폴더 복원 필요
- 자세한 의존성: 위 [⚠️ 외부 Phase 폴더 의존성](#️-외부-phase-폴더-의존성) 섹션 참조

### 8. `02a §6 sanity check 실패` (PHASE2_DIR 관련)
- 02a 또는 02a_v2 노트북의 §6 BL sanity check 실행 시 발생
- 원인: `Phase2_BL_Integration/data/ensemble_predictions_top50.csv` 비교 검증 read 실패
- 해결: §6 셀 SKIP 또는 Phase 2 폴더 복원
- 핵심 학습 결과 (`ensemble_predictions_stockwise.csv`) 는 §3~§5 에서 이미 생성되므로 §6 SKIP 해도 후속 노트북 (03~06) 에는 영향 없음

---

## Phase 3 핵심 결론 (현재 시점 기준)

1. **서윤범 99 재현 PASS**: Phase 3 BL_trailing_mcap Sharpe = 1.203 (서윤범 1.157 대비 +3.96% 차이, ±5% 기준 통과)
2. **ML 통합 효과 미확인**: BL_ml_sw_mcap, BL_ml_cs 모두 BL_trailing_mcap 대비 Sharpe 하락 (약 -0.11)
3. **pct 민감도**: 현행 pct=0.30 이 최적값 아님 — BL_ml_sw 는 0.15, BL_trailing 은 0.20 이 Sharpe 최적
4. **Weighting 효과**: ML 전략에서 eq/rp 가 mcap 보다 우수 — ML 의 변동성 ranking 신호가 시총 가중에 의해 오염됨
5. **시기별**: 정상 강세장 (2012~2019) BL_trailing_mcap 압도, COVID+AI (2020~2025) 시총 가중 (McapWeight, SPY) 우위

상세 결과: `outputs/03_bl_backtest/metrics_fair.csv`, `outputs/06_pct_sensitivity/summary.json` 참조.

---

## 참고 문헌

- **Black & Litterman (1992)** — 원조 BL 모델
- **He & Litterman (1999)** — Goldman Sachs 백서 (TAU 표준)
- **Pyo & Lee (2018)** — BAB factor + 30% 양극단 분류
- **Markowitz (1952)** — 평균-분산 최적화
- **Ledoit & Wolf (2004)** — Shrinkage 공분산 추정
- **DeMiguel, Garlappi, Uppal (2009)** — 1/N 의 강력한 baseline
- **Maillard et al. (2010)** — Risk Parity (inverse-vol)
- **Diebold & Mariano (1995)** — 예측 정확도 검정
- **Patton (2011)** — QLIKE 손실 함수
- **Jobson & Korkie (1981) / Memmel (2003)** — Sharpe 차이 검정
- **Hansen, Lunde & Nason (2011)** — Model Confidence Set
- **Gu, Kelly & Xiu (2020)** — Empirical Asset Pricing via ML (Ticker Embedding)
- **서윤범 99_baseline** — 본 프로젝트의 직접 비교 대상

---

## 변경 이력 (주요)

| 날짜 | 변경 |
|---|---|
| 2026-04-29 | Phase 3 시작: 전체 S&P 500 모드 (top-50 → 624 종목 확장), TAU=0.1·LAM=2.5 고정 |
| 2026-04-30 | Dynamic-Membership Step 7, Stale 필터 Step 8 추가 (look-ahead bias 차단) |
| 2026-04-30 | Phase 3-2: weighting 변형 3종 (mcap/eq/rp) 추가 — 9 시나리오 |
| 2026-05-01 | 03_v2 estimate_covariance() 인자 불일치 수정 + 03_v2.py 변환본 추가 |
| 2026-05-01 | 06_pct_sensitivity 추가 (pct 0.10~0.40 민감도) |
| 2026-05-02 | `05a_v2_lstm.ipynb` (12 분석) + `05a_v2_lstm_2b_deep.ipynb` (학술 심화 + 효과크기) + `05a_v2_weighting.ipynb` (3 가중치 비교) 신설 |
| 2026-05-02 | `data/ticker_sector_mapping.csv` 신설 (Wikipedia 503 + Hardcoded historical 25 = 528 종목 매핑) |
| 2026-05-02 | §2-B 효과크기 검정 추가 — η²(Period)=0.450 LARGE, η²(Ticker)=0.194 LARGE, Welch ANOVA robust 검증 |
| 2026-05-01 | README.md 종합 명세서로 재작성 |
| 2026-05-01 | README 에 외부 Phase 폴더 의존성 섹션 추가 (분석 vs 재학습 분리 명시) |

---

## 작성자 / 라이선스

- 작성: 본 프로젝트 분석 코드 기반 자동 생성
- 외부 데이터 출처: yfinance (Yahoo Finance), Wikipedia (S&P 500 멤버십), FRED (무위험 수익률 fallback), Ken French Library (Fama-French 3팩터)
- 학술 비교 대상: 서윤범 `99_baseline.ipynb`
