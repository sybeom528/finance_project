# final_pt/ — 분석 · 백테스트 파이프라인

Adaptive VolControl Fund 프로젝트의 **연구 코어** 폴더. 9개 메인 노트북 + 3개 부록 노트북 + `lib/` 모듈(7개) + 메서드 문서(7개)로 구성된 7-step quant 파이프라인.

> 프로젝트 전체 개요는 [상위 README](../README.md), 자세한 메서드는 [`docs/`](docs/) 참고.

## 빠른 시작

```bash
cd final_pt
uv run jupyter lab
```

권장 실행 순서: **01 → 02a → 02b → 03a → 03b → 04 → 05a → 05b → 06** (파일명 순)

- `04`는 `03a/03b` LSTM 산출물(`ensemble_predictions_stockwise.csv`)이 있어야 실행됨
- `05a` HMM 레짐은 `05b/06` 분석에서만 사용 (04 walk-forward에는 의존성 없음)
- `06`은 winner 검증용이므로 `05b` 다음에 실행

## 메인 노트북 (9개)

| # | 파일 | 역할 | 핵심 산출물 |
|---|---|---|---|
| 1 | [`01_DataCollection.ipynb`](01_DataCollection.ipynb) | S&P500 역사적 유니버스 + 월별 패널 생성 | `data/monthly_panel.csv` (617종목 × 13변수), `data/daily_returns.pkl` |
| 2a | [`02a_EDA_Returns_Volatility.ipynb`](02a_EDA_Returns_Volatility.ipynb) | 시계열 EDA — 수익률 예측 불가 + 변동성 예측 가능 4단 검증 (산점도/ACF/CUSUM/Chow) | LSTM 모델링 정당화 |
| 2b | [`02b_LowVol_PortfolioSort.ipynb`](02b_LowVol_PortfolioSort.ipynb) | 횡단면 EDA — 저변동 30% Sharpe 0.96 vs 고변동 0.73 forward sort (저변동 그룹의 위험조정 우위 확인) | BL spread view 정당화 |
| 3a | [`03a_LSTM_Optuna_GridSearch.ipynb`](03a_LSTM_Optuna_GridSearch.ipynb) | Optuna 12-trial HPO → V4_BEST_CONFIG | `_evidence/lstm_optuna_v4/best_metrics.json` |
| 3b | [`03b_Volatility_Forecasting.ipynb`](03b_Volatility_Forecasting.ipynb) | LSTM + HAR + Diebold-Pauly ensemble (617종목 각각 개별 학습) | `data/03b_lstm/data/ensemble_predictions_stockwise.csv` |
| 4 | [`04_BL_Walkforward.ipynb`](04_BL_Walkforward.ipynb) | 90개 슬롯 walk-forward 백테스트 (3 prior × 3 p_weight × 5 q_mode × 2 omega_mode, 월별 train=60m rolling) | `results/*.pkl` (90개) |
| 5a | [`05a_HMM_Regime.ipynb`](05a_HMM_Regime.ipynb) | 3-state HMM 시장 레짐 분류 (회복/확장/변동) | 레짐별 매크로 인수 분석 |
| 5b | [`05b_Analyze.ipynb`](05b_Analyze.ipynb) | **메인 분석** — K_CUT + 5단 심층분석 (한계/매트릭스/위기/벤치마크/winner) | Winner 자동 식별 (`sortino_ir≥10` 필터 + 전체기간 sortino 1위) |
| 6 | [`06_Regime_Analysis.ipynb`](06_Regime_Analysis.ipynb) | Winner 4-레짐 검증 — 3-레짐(K_CUT까지) + R4 hold-out(2024-2025) | Frazzini-Pedersen 2014 hold-out 검증 |

## 부록 노트북 (`appendix/`)

| 파일 | 역할 |
|---|---|
| [`99_explore.ipynb`](appendix/99_explore.ipynb) | 저변동 종목군 세부 특성 탐색 (관성, 분산 지속성 등) |
| [`99_lstm_statistics.ipynb`](appendix/99_lstm_statistics.ipynb) | LSTM 예측값 분포·통계 검증 |
| [`99_slot_effects.ipynb`](appendix/99_slot_effects.ipynb) | BL 슬롯별 한계효과 OAT 분리 (prior / p_weight / q_mode / omega) |

## `lib/` — 핵심 모듈 (7개)

2026-05-19 패키지화 (`lib/__init__.py` 추가, 노트북은 `from lib.X import ...` 로 import).

| 모듈 | 역할 | 주요 export |
|---|---|---|
| [`bl_config.py`](lib/bl_config.py) | 90개 실험 매트릭스 정의 + 공통 파라미터 + 평가 기간 | `EXPERIMENTS`, `BASELINE`, `EVAL_PERIODS` |
| [`bl_functions.py`](lib/bl_functions.py) | BL 수식 함수 모음 (Σ, π, P, q, Ω, BL, MVO, TC, Metrics) | `compute_sigma`, `compute_pi`, `build_P`, `compute_Q_*`, `black_litterman`, `optimize_portfolio`, `compute_metrics` |
| [`bl_runner.py`](lib/bl_runner.py) | walk-forward 실행 엔진 (월별 캐시 + 192개월 순차) | `load_lstm_pred`, `build_monthly_cache`, `walk_forward` |
| [`master_table.py`](lib/master_table.py) | 90개 pkl → 단일 DataFrame 통합 + winner 자동 식별 + 3-레짐 정의 | `build_master_table`, `build_regime_table`, `identify_winner`, `REGIMES`, `parse_config` |
| [`analyze_plots.py`](lib/analyze_plots.py) | 05b_Analyze의 6개 대시보드 시각화 | `plot_marginal_effects`, `plot_matrix_heatmap`, `plot_styled_regime_dashboard` |
| [`lstm_pipeline.py`](lib/lstm_pipeline.py) | LSTM universe × 8-way 병렬 학습 (03b에서 호출) | `V4_BEST_CONFIG`, `build_daily_panel`, `run_ensemble_for_universe_parallel` |
| [`timeseries_lib.py`](lib/timeseries_lib.py) | 시계열·통계 기본 함수 (walk-forward CV, LSTM, HAR-RV, ensemble, 검정) | `LSTMRegressor`, `walk_forward_folds`, `fit_har_rv`, `diebold_pauly_weights`, `rmse`, `welch_anova` |

## `docs/` — 메서드 문서 (7개)

| 문서 | 내용 |
|---|---|
| [`PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | 전체 파이프라인 + 결과 요약 (메인 entry-point) |
| [`BL_EXPERIMENT_GUIDE.md`](docs/BL_EXPERIMENT_GUIDE.md) | 90개 슬롯 매트릭스 상세 + 슬롯 추가법 + 실행 가이드 |
| [`DATA_COLLECTION.md`](docs/DATA_COLLECTION.md) | 데이터 수집 상세 (S&P500 멤버십, 피처 정의, NaN 처리) |
| [`ANOMALY_ANALYSIS.md`](docs/ANOMALY_ANALYSIS.md) | 저변동 그룹 forward portfolio sort EDA (Sharpe·MDD 우위 검증) |
| [`SENSITIVITY_ANALYSIS.md`](docs/SENSITIVITY_ANALYSIS.md) | q (6 sweep) + PCT (5 sweep) 민감도 + JK z-test + Bootstrap CI |
| [`WINNER_SLOT_TIMESERIES.md`](docs/WINNER_SLOT_TIMESERIES.md) | Winner 슬롯 4-패널 시계열 동학 (q/Ω/노출/Turnover) |
| [`Exploiting_LowRisk_Anomaly_BL_Summary.md`](docs/Exploiting_LowRisk_Anomaly_BL_Summary.md) | 학술 reference — Pyo & Lee 2018 |

## 데이터 폴더 (gitignored)

| 폴더 | 내용 |
|---|---|
| `data/` | `monthly_panel.csv`, `daily_returns.pkl`, `ff_factors_daily.csv`, `ff3_monthly.csv`, `ff5_monthly.csv`, `macro_daily.csv`, `prices_raw.pkl`, `sector_etf.pkl`, `shares_outstanding.pkl`, `sp500_membership.pkl`, `universe.csv` |
| `data/03b_lstm/` | LSTM 학습 산출물 (`ensemble_predictions_stockwise.csv`, 모델 체크포인트) |
| `results/` | BL walk-forward 결과 90개 `.pkl` (각 슬롯) |
| `outputs/` | 분석 노트북 산출 PNG (winner_slot_dynamics.png 등) |
| `_evidence/` | Optuna HPO 결과 캐시 |

## 핵심 결과 (Winner: `mat_eq_eq_raw_pap`)

| 메트릭 | Winner | SPY |
|---|---:|---:|
| Sharpe | 1.096 | 0.731 |
| Sortino | 1.826 | 1.108 |
| CAGR | 16.2% | 12.7% |
| MDD | −13.6% | −33.7% |

**검정 통과**:
- q 민감도: 6개 변형 [0.001~0.010] 모두 JK z-test p>0.5 (BAB 학술값 0.0055·0.0064 포함)
- PCT 민감도: [0.20~0.35] 4개 완전 robust, 0.15만 10% marginal
- 3-레짐 HMM: 모든 레짐에서 winner 상위 (`sortino_ir≥10` 필터 13개 robust 후보 중 1위)
- Block Bootstrap (block=6, 1000회) 95% CI 0 포함

## 주의사항

| 항목 | 설명 |
|---|---|
| **√252 단위 보정** | LSTM 예측값(log-daily-RV) → `np.exp() × √252` 연환산 필수. 누락 시 `vol_21d`와 단위 혼합 → P 랭킹 왜곡 |
| **K_CUT 강제성** | `05b_Analyze` 메인 분석은 2023-12-31 cutoff mandatory. Hold-out 2024-2025는 `06_Regime_Analysis`로 별도 검증 |
| **omega_mode `ff3_paper`** | 역사적 코드명. 실제 동작은 **Bayesian rolling view variance** (직전월 (q−P·r)² 적응). FF3 회귀 잔차와 무관 |
| **Look-ahead bias** | `fwd_ret_1m`은 forward portfolio sort 평가용만. BL 입력·LSTM 학습 절대 금지 |
| **Winner 자동 동기화** | `identify_winner()`가 `sortino_ir≥10` 필터 + 전체기간 sortino 1위로 자동 식별. 슬롯 변경 시 winner 재계산 |
