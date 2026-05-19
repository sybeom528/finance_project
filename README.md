# Adaptive VolControl Fund

> **저변동성 anomaly + LSTM 변동성 예측 + Black-Litterman 단일 view 프레임워크로 위험성향별 액티브 ETF 펀드 후보를 구축한 quant 프로젝트.**

## TL;DR

- **목표**: 학계 검증된 저변동성 anomaly 를 ML 변동성 예측으로 강화한 액티브 ETF 펀드 전략
- **방법**: LSTM σ 예측 → P/Q/Ω 슬롯 156 조합 walk-forward 백테스트 → 위험조정 winner 선정 + 통계 검정
- **결과 (winner slot, 2010-01 ~ 2024-12, 180개월)**:

  | 메트릭 | Winner (`mat_eq_eq_raw_pap`) | SPY |
  |---|---:|---:|
  | Sharpe | **1.096** | 0.731 |
  | Sortino | **1.826** | 1.108 |
  | CAGR | **16.2%** | 12.7% |
  | MDD | **−13.6%** | −33.7% |

- **검증**: Q 민감도 (Memmel JK z-test, p>0.5, BAB 학술값 0.0055·0.0064 동등) + 3-레짐 안정성 (R1 회복 / R2 확장 / R3 변동) + Fama-French 5+Mom 알파 분해

## Live Demo

- 🌐 **랜딩 페이지** (Vercel): _TBD — 배포 후 URL 추가_
- 📊 **인터랙티브 대시보드** (Streamlit): _TBD — 배포 후 URL 추가_

## Repo Structure

```
.
├── final_pt/              ← 분석·백테스트 노트북 + 라이브러리
│   ├── 01_DataCollection.ipynb            데이터 수집
│   ├── 02a_EDA_Returns_Volatility.ipynb   시계열 EDA (수익률 vs 변동성 예측성)
│   ├── 02b_LowVol_PortfolioSort.ipynb     횡단면 EDA (저변동 anomaly 6단 검증)
│   ├── 03a_LSTM_Optuna_GridSearch.ipynb   HPO (Optuna 12-trial)
│   ├── 03b_Volatility_Forecasting.ipynb   LSTM + HAR + Diebold-Pauly ensemble
│   ├── 04_BL_Walkforward.ipynb            BL walk-forward (156개 슬롯)
│   ├── 05a_HMM_Regime.ipynb               3-레짐 HMM 분류
│   ├── 05b_Analyze.ipynb                  분석 (K_CUT · 민감도 · α 분해)
│   ├── 06_Regime_Analysis.ipynb           4-레짐 hold-out winner 검증
│   ├── lib/                               핵심 모듈 (bl_*, lstm_pipeline, master_table 등)
│   ├── data/                              monthly_panel, daily_returns, FF factors 등
│   ├── results/                           BL 백테스트 결과 (90 슬롯 pkl)
│   └── docs/                              메서드 문서 (PROJECT_OVERVIEW, BL_EXPERIMENT_GUIDE 등)
│
├── streamlit_dashboard/   ← Streamlit 대시보드 (lib/, pages/, data/, scripts/)
├── vercel_deploy/         ← 정적 랜딩 페이지 (Vercel)
├── pyproject.toml         ← UV 패키지 정의
└── README.md
```

> 전체 협업 히스토리(개인 작업 디렉터리·초기 EDA·폐기된 실험 등)는 [**`archive` 브랜치**](https://github.com/sybeom528/finance_project/tree/archive)에 보존되어 있습니다.

## 핵심 방법론

### 7-Step Pipeline

```
1. Data Collection      30개 ETF/주식 + 12개 외부지표 + 8개 거시지표 (2010-2024)
2. EDA                  수익률 예측 불가(R²≈0) + 변동성 예측 가능(ACF lag 60) 검증
3. LSTM σ 예측           자산별 일별 RV → 다음월 변동성 (HAR baseline + Ensemble)
4. Black-Litterman      저변동 30% long / 고변동 30% short, 단일 spread view
5. MVO                  위험성향별 max_weight 차별화
6. Walk-forward         156개 슬롯 조합, 60개월 학습창
7. Streamlit Dashboard  리밸런싱 추천 UI
```

### Black-Litterman 사후분포 (단일 view, K=1)

```
μ_BL = π + (τΣPᵀ)·(Q − Pπ) / (PτΣPᵀ + Ω)
```

- **π** (prior): λΣw_mkt (CAPM 균형수익률)
- **P** (1×N): 저변동 long / 고변동 short (LSTM σ_pred 기준 30/30 그룹)
- **Q** (스칼라): "저변동이 고변동을 능가" view 강도
- **Ω** (스칼라): view 분산 (예측오차² 적응형 갱신)

→ Sherman-Woodbury 단일 view 닫힌해. 자세한 슬롯 수식은 [`final_pt/docs/BL_EXPERIMENT_GUIDE.md`](final_pt/docs/BL_EXPERIMENT_GUIDE.md)

## 실행 방법

### 환경 셋업

```bash
uv sync   # Python 3.12, 의존성 설치
```

### Streamlit 대시보드

```bash
cd streamlit_dashboard
uv run streamlit run app.py
```

### 분석 노트북

```bash
cd final_pt
uv run jupyter lab
```

권장 실행 순서: `01 → 02a → 02b → 03a → 03b → 05a → 04 → 05b → 06`

### Vercel 랜딩 페이지 로컬 미리보기

```bash
cd vercel_deploy
python -m http.server 8000
```

## Tech Stack

- **언어**: Python 3.12 (UV 패키지 관리)
- **ML**: PyTorch (LSTM), Optuna (HPO)
- **Quant**: NumPy, Pandas, SciPy
- **최적화**: CVXPY (MVO)
- **시각화**: Matplotlib, Plotly, Streamlit
- **데이터**: yfinance, FRED API

## 팀 협업 프로젝트

본 프로젝트는 4인 팀의 공동 작업입니다.

| 역할 | 담당 |
|---|---|
| 김윤서 | 데이터 수집, GDELT 추출, EDA, 프로세스 문서화 |
| 서윤범 | 아키텍처 리드, XGBoost Step 3, 설계 문서 |
| 김재천 | GDELT API 탐색, yfinance 검증 |
| 김하연 | EDA 노트북 |

main 브랜치는 포트폴리오 view 로 정리된 최종 산출물만 포함합니다. 개인 작업 디렉터리·초기 EDA·미채택 실험 등 전체 프로세스는 [`archive` 브랜치](https://github.com/sybeom528/finance_project/tree/archive)에 보존되어 있습니다.
