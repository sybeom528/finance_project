# 📊 Adaptive VolControl Fund

> LSTM 변동성 예측과 Black-Litterman 단일 view 프레임워크를 결합한 **저변동 종목 중심의 액티브 ETF 펀드** 구축 프로젝트

| | |
|---|---|
| **기간** | 2026.04 - 2026.05 (약 5주) |
| **역할** | 데이터 분석가 (팀 프로젝트, 팀원 4명) |
| **분석 영역** | Quant Finance, Time-series Forecasting, Portfolio Optimization |
| **분석 기간** | 2010-01 ~ 2025-12 (총 192개월) |

---

## 📌 Project Overview

### 배경
- 국내 ETF 시장 규모 2002년 3,500억 → 2026년 5월 **456조 원**으로 급성장
- 분산·안정성 추구 흐름 강화 + 변동성 경계 증가
- → **저변동 전략**의 중요성 부각 (Cf. 워렌 버핏 *"절대 돈을 잃지 마라"*)

### 목표
1. **장기적 균형 추구** — 단기 시장 변동에 흔들리지 않는 포트폴리오 구성
2. **변동성 최소화** — 수익률 극대화가 아닌 변동성 통제에 집중

저변동 종목군 중심의 액티브 ETF 펀드 후보를 **데이터·통계적으로** 설계·검증.

---

## 📊 Dataset

### 데이터 출처

| 출처 | 기간 | 용도 |
|---|---|---|
| `yfinance` | 2004 ~ 2026 | 833종목 OHLCV, 발행주식수 |
| `Wikipedia` | 2005 ~ 2025 (월별) | S&P500 멤버십 히스토리 (생존편향 제거) |
| `FRED API` | 2010 ~ 2024 | VIX, HY spread, yield curve, Sahm 등 거시지표 |
| `Fama-French Data Library` | 1926 ~ 현재 | FF5 + Momentum (월별·일별) |

### 데이터 규모
- **투자 유니버스**: **S&P500 역사적 멤버십 기반 개별 주식 617종목** (2005-2025 월별 332~498개 활성)
  - Wikipedia S&P500 변경 히스토리 역방향 재구성 → 시점별 정확한 멤버십 적용 (생존편향 완화)
- **외부 비교·보조 자산** (유니버스 아닌 외생 데이터)
  - `SPY`: 시장 벤치마크
  - `USMV` / `SPLV`: 저변동 ETF 카테고리 비교군 (2011년 출시 이후)
  - `^IRX`: 무위험금리 (13주 T-bill)
  - 11개 GICS 섹터 ETF (`XLE/XLB/XLI/XLY/XLP/XLV/XLF/XLK/XLC/XLU/XLRE`): 섹터 12개월 모멘텀(`indmom`) 계산용
- **일별 수익률**: 824 ticker × 5,595 영업일 (`daily_returns.pkl`)
- **월별 패널**: 617 종목 × 192개월 × 13개 변수 (`monthly_panel.csv`)

### 주요 전처리
- **생존편향 제거**: 시점별 S&P500 멤버십 필터 적용
- **Look-ahead bias 방지**: `fwd_ret_1m`(평가용)과 BL 입력(`ret_1m`, `vol_252d`) 코드 차원 분리
- **공분산 추정**: Ledoit-Wolf shrinkage (자산 수 > 일수 상황 대응)
- **LSTM 입력**: 일별 RV → log 변환 → 시퀀스 학습 → `np.exp() × √252` 연환산

---

## 🛠 Tech Stack

- **Language**: Python 3.12 (UV 패키지 관리)
- **ML / 시계열**: PyTorch (LSTM), HAR-RV (전통 시계열), Optuna (HPO), Diebold-Pauly (모델 결합)
- **Quant / 통계**: NumPy, Pandas, SciPy, statsmodels
- **포트폴리오 최적화**: CVXPY (long-only MVO + max_weight 제약)
- **레짐 분류**: hmmlearn (3-state HMM)
- **시각화 / 대시보드**: Matplotlib, Plotly, Streamlit
- **배포**: Vercel (정적), Streamlit Community Cloud
- **Tool**: Git, GitHub, Jupyter Lab, VS Code

상세 라이브러리 버전: [pyproject.toml](pyproject.toml)

---

## 🔍 Analysis & Methodology

### 7-Step 파이프라인

```
1. Data Collection      yfinance + FRED + Wikipedia 멤버십 → 617종목 월별 패널
2. EDA                  수익률 예측 불가(R²≈0) vs 변동성 예측 가능(ACF lag 60) 검증
3. LSTM σ 예측           LSTM + HAR-RV 두 모델 → Diebold-Pauly 가중평균 Ensemble
4. Black-Litterman      저변동 long / 고변동 short 단일 spread view 주입
5. MVO Walk-forward     90개 슬롯 × 192개월, 매월 60개월 rolling
6. 분석 & 검정           조합 선택 + q/PCT 민감도 + 3-레짐 안정성
7. 대시보드·배포          Streamlit + Vercel
```

### 핵심 방법론

**LSTM + HAR-RV Ensemble (변동성 예측)**
- LSTM (딥러닝, 3채널 입력: RV/Return/VIX) + HAR-RV (전통 시계열 baseline)
- 두 모델을 Diebold-Pauly 가중평균으로 결합 → 단일 모델 대비 RMSE 우위
- Optuna 12-trial HPO로 하이퍼파라미터 탐색

**Black-Litterman 단일 view (포트폴리오 구성)**
```
μ_BL = π + (τΣPᵀ)·(q − Pπ) / (PτΣPᵀ + Ω)
```
- 단일 view (K=1): "저변동 30% 그룹이 고변동 30% 그룹보다 우수" 1개 view 주입
- 90개 슬롯 매트릭스: `prior(3) × p_weight(3) × q_mode(5) × omega_mode(2)`
- Sherman-Woodbury 닫힌해로 사후분포 계산

**Walk-forward Backtest (시계열 정합성)**
- 매월 60개월 rolling window → 다음달 수익률 평가
- Look-ahead bias 완전 차단
- 거래비용 20bp/side, max_weight 10% 제약

---

## 💡 Key Findings

### 1. EDA — 수익률은 예측 불가, 변동성은 예측 가능
- 수익률 자기상관 R² ≈ 0 → 효율적 시장 가설과 일치
- 변동성 ACF lag 60까지 유의 → **변동성 클러스터링** 존재
- → LSTM 모델링 대상을 **수익률 → 변동성으로 피봇** 결정의 데이터 근거

### 2. 횡단면 저변동 그룹의 위험조정 우위 (Forward Portfolio Sort)
- 저변동(Q1) Sharpe **0.96** vs 고변동(Q5) **0.73** (+0.23 격차)
- 저변동 MDD **−16.7%** vs 고변동 **−34.1%** (위기 방어력 **2배** 차이)
- → BL spread view 구성의 정량 근거

### 3. LSTM + HAR Ensemble — Forecast Combination의 효과
- **Ensemble RMSE 0.3822** vs HAR 단독 0.3914 vs LSTM 단독 0.5185
- 403 종목 (65.3%)에서 Ensemble이 단독 모델 대비 우위

### 4. 90개 슬롯 walk-forward → 가장 타당한 조합 선택
90개 조합 중 위험조정 메트릭 기준(`sortino_ir ≥ 10` 필터 + 전체기간 `sortino` 1위)에서 **가장 적절한 조합으로 선택된 `mat_eq_eq_raw_pap`** 의 성과 (2010-01 ~ 2023-12, K_CUT):

| 메트릭 | 선택 조합 | SPY benchmark | 차이 |
|---|---:|---:|---:|
| Sharpe | **1.096** | 0.731 | +0.365 |
| Sortino | **1.826** | 1.108 | +0.718 |
| CAGR | **16.2%** | 12.7% | +3.5%p |
| MDD | **−13.6%** | −33.7% | **+20.1%p (방어)** |

> Sharpe·Sortino·MDD 모두 **변동성·하방 리스크 관리**를 측정하는 메트릭이라 1차 목표와 직접 정합. SPY는 시장 평균 대비 정합성을 확인하기 위한 benchmark.

### 5. Robustness 검증
- **q 민감도**: Memmel JK z-test 6개 변형 모두 p > 0.5 (BAB 학술값 0.0055·0.0064 통계적 동등) → 운(luck) 가설 기각
- **PCT 민감도**: [0.20~0.35] 4개 완전 robust
- **3-레짐 HMM 안정성**: 회복/확장/변동 전 구간 선택 조합 상위 유지
- **Block Bootstrap**: block=6개월, 1,000회 → 95% CI 0 포함

---

## 🌐 Live Demo & Outputs

| 자료 | 설명 | 링크 |
|---|---|---|
| 🌐 프로젝트 소개 웹페이지 | 정적 단일 페이지 (Vercel 배포) | _배포 후 URL 추가_ |
| 📊 인터랙티브 분석 대시보드 | Streamlit 7페이지 (직접 필터·시뮬 조작) | _배포 후 URL 추가_ |
| 📥 발표 PDF (11MB) | 프로젝트 발표 슬라이드 | [다운로드](https://github.com/sybeom528/finance_project/releases/download/v1.0/finance.project.pdf) |

---

## ⚙️ How to Run

### 사용 환경

| 항목 | 권장 사양 |
|---|---|
| OS | macOS 12+ / Ubuntu 20.04+ / Windows 10+ (WSL2) |
| Python | 3.12.x |
| 패키지 매니저 | UV (https://docs.astral.sh/uv/) |
| 메모리 | 8GB+ (LSTM 학습 시 16GB+) |
| GPU | 선택 (CUDA 가능, CPU도 동작) |

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/sybeom528/finance_project.git
cd finance_project

# 2. UV 설치 (미설치 시)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 의존성 일괄 설치
uv sync
```

### 실행

**A. Streamlit 대시보드** (가장 빠른 데모)
```bash
cd streamlit_dashboard
uv run streamlit run app.py
```
→ 브라우저 자동 열림 (http://localhost:8501)

**B. 분석 노트북**
```bash
cd final_pt
uv run jupyter lab
```
→ 브라우저에서 노트북 열기 → 상단 메뉴 `Run > Run All Cells`

권장 실행 순서: `01 → 02a → 02b → 03a → 03b → 04 → 05a → 05b → 06`

**C. Vercel 소개 페이지 로컬 미리보기**
```bash
cd vercel_deploy
python -m http.server 8000
```

---

## 📁 Project Navigation

| 폴더 | 역할 | 상세 README |
|---|---|---|
| `final_pt/` | 분석·백테스트 파이프라인 (7-step 노트북 + lib + docs) | [final_pt/README.md](final_pt/README.md) |
| `streamlit_dashboard/` | 인터랙티브 분석 대시보드 (7 페이지) | [streamlit_dashboard/README.md](streamlit_dashboard/README.md) |
| `vercel_deploy/` | Vercel 정적 소개 페이지 | [vercel_deploy/README.md](vercel_deploy/README.md) |

---

## 🌿 Repository Branches

- **`main`**: 포트폴리오 view (`final_pt/`, `streamlit_dashboard/`, `vercel_deploy/` 만 포함)
- **[`archive`](https://github.com/sybeom528/finance_project/tree/archive)**: 전체 개발 히스토리 (초기 EDA, 미채택 실험, 학습 자료, 이전 매트릭스 등)

archive 브랜치에 보존된 주요 자료:
- 방향 전환 흔적 (자산별 q 직접 예측 실패 R²≈−0.95 → 변동성 σ 예측으로 피봇한 과정)
- 학습 자료 (시계열 분석·LSTM 구조·look-ahead bias 처리 study notes)

---

> ⚠️ **면책**: 본 프로젝트의 분석 결과는 학술·교육 목적이며, 실제 투자 권유가 아닙니다. 과거 성과가 미래 수익을 보장하지 않습니다.
