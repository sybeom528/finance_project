# 성향별 액티브 ETF 펀드 — Low-Risk BL 프로젝트

> **저변동성 anomaly + LSTM σ 예측 + Black-Litterman 단일 view 프레임워크로 위험성향별 펀드 후보를 탐색하는 quant 프로젝트.**
>
> 최종 갱신: 2026-05-11

---

## 파이프라인 (7-Step)

```
1. Data Collection      ✅ 01_DataCollection.ipynb              — DATA_COLLECTION.md
2. EDA (시계열)          ✅ 02a_EDA_Returns_Volatility.ipynb     — 수익률 예측 불가 / 변동성 예측 가능 증명
2. EDA (횡단면)          ✅ 02b_Low_Risk_Anomaly.ipynb           — ANOMALY_ANALYSIS.md
3. LSTM HPO             ✅ 03a_LSTM_Optuna_GridSearch.ipynb     — V4_BEST_CONFIG 정당화 (Optuna 12-trial)
3. LSTM σ 예측           ✅ 03b_Volatility_Forecasting.ipynb     — LSTM + HAR + Performance ensemble (617 종목 stockwise)
4. Black-Litterman       ✅ 99_run.ipynb (walk_forward)
5. MVO + 위험성향 매핑    🟡 99_analyze.ipynb (분석 진행 중)
6. 3-레짐 안정성 + 민감도  ✅ 99_analyze.ipynb (I/J/K/L/M/N)
7. Streamlit 대시보드     🟡 streamlit_dashboard/ (초안 진행 중)
```

---

## 파일 구조

```
final_pt/
├── 01_DataCollection.ipynb            ← 데이터 수집 (S&P500 멤버십 + 가격 + 패널 + 보조)
├── 02a_EDA_Returns_Volatility.ipynb   ← 시계열 EDA (수익률 vs 변동성 예측성)
├── 02b_Low_Risk_Anomaly.ipynb         ← 횡단면 EDA (저변동 anomaly 6단 검증)
├── 03a_LSTM_Optuna_GridSearch.ipynb   ← HPO 보조 (12-trial Optuna → V4_BEST_CONFIG)
├── 03b_Volatility_Forecasting.ipynb   ← LSTM 학습 + HAR baseline + Diebold-Pauly ensemble
├── 04_Statistical_Validation.ipynb    ← 학술 통계 심화 분석
├── HMM_Regime.ipynb                   ← 3-레짐 HMM 분류
│
├── 99_run.ipynb                       ← walk_forward 실행 → results/*.pkl
├── 99_analyze.ipynb                   ← 분석 (K_CUT → I → J → K → L → M → N)
├── 99_slot_effects.ipynb              ← 슬롯 차원 효과 라인플롯 (pivot CSV 자동 생성)
│
├── timeseries_lib.py                  ← 시계열·통계 함수 (LSTM, HAR-RV, ensemble, 통계 검정)
├── lstm_pipeline.py                   ← LSTM high-level orchestration (V4_BEST_CONFIG, walk_forward)
├── bl_config.py                       ← EXPERIMENTS 정의 (156개)
├── bl_functions.py                    ← BL 핵심 함수 (Σ/π/P/Q/Ω/BL/TC/Metrics)
├── master_table.py                    ← results/*.pkl → mt/rt 빌더
├── analyze_plots.py                   ← 시각화 모듈
│
├── _evidence/                         ← 03a Optuna 캐시 (lstm_optuna_v4/best_metrics.json)
├── data/                              ← 01 산출물 (monthly_panel, daily_returns, macro, FF 등)
├── phase3(data_outputs)/data/         ← 03b 산출물 (ensemble_predictions_stockwise.csv)
├── results/                           ← 156 BL pkl
├── outputs/                           ← 차트 PNG
└── streamlit_dashboard/               ← Streamlit 앱 (초안)
```

---

## 문서 가이드

| 목적 | 참고 |
|---|---|
| 실험 슬롯 수식 + 추가법 + 실행법 | [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md) |
| 데이터 수집 파이프라인 | [DATA_COLLECTION.md](DATA_COLLECTION.md) |
| 저변동 anomaly 검증 EDA | [ANOMALY_ANALYSIS.md](ANOMALY_ANALYSIS.md) |
| Q·PCT 민감도 분석 (winner 슬롯) | [SENSITIVITY_ANALYSIS.md](SENSITIVITY_ANALYSIS.md) |
| 학술 reference (Pyo & Lee 2018) | [Exploiting_LowRisk_Anomaly_BL_Summary.md](Exploiting_LowRisk_Anomaly_BL_Summary.md) |
| 4차 LSTM 재학습 검증 | [_validation_report_2026_05_08.md](_validation_report_2026_05_08.md) |
| 분석 노트북 셀 해설 | `99_analyze.ipynb` 안 markdown 셀 |

---

## 핵심 결과 (현 시점)

### 02 EDA 정당화
- **02a**: 수익률 R²≈0 / 변동성 ACF lag 60까지 유의 → **LSTM 변동성 예측 모델 정당화** (→ 03b)
- **02b** (forward portfolio sort, Frazzini-Pedersen 2014 형식): 저변동 Sharpe 0.96 vs 고변동 0.73, MDD -16.7% vs -34.1% → **risk-adjusted 우위 일관 + BL spread view 정당화**

### 03 모델
- **03a** Optuna 12-trial: best `3ch_vix / IS=1250 / embargo=63` → avg RMSE 0.3107 (HAR-RV 0.3477 대비 −10.6% 개선)
- **03b** 617 종목 stockwise walk-forward (225 fold): LSTM 0.5185 / HAR 0.3914 / **Ensemble 0.3822** (Best ensemble 403 종목, 65.3%)

### BL 백테스트
- **Top 후보** (sortino_ir 기준): `mcap_ls_eq_lam_pap` (Sharpe 1.140, CAGR 16.23%, MDD -14.68%)
- **baseline 대비**: Sharpe +0.03, CAGR +2.7%p, MDD 비슷한 수준 유지
- **SPY 대비**: 모든 위험조정 지표에서 우위 (특히 위기 방어)
- **Q 민감도**: q=0.003이 우리 setup의 robust optimum (BAB 학술값 0.0064는 단조 감소)
- **레짐 안정성**: 3-레짐 (R1 회복 / R2 확장 / R3 변동) HMM 기반

---

## 명명 체계 — canonical 5-token

```
{prior}_{p_mode}_{p_weight}_{q}_{omega}
mcap   _ls    _eq      _lam_pap  ← 예
```

토큰 정의 + 슬롯 수식 → [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md) §4 참고
