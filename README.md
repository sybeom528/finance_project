# Adaptive VolControl Fund

> 저변동성 anomaly를 LSTM 변동성 예측과 Black-Litterman 단일 view 프레임워크로 체계화하여, 위험성향별 액티브 ETF 후보 펀드를 구축한 정량 포트폴리오 연구 프로젝트.

## 한눈에 보기

**문제 설정**: 학계가 검증한 저위험 이상현상(low-risk anomaly)을 실제 운용 가능한 펀드 전략으로 변환한다.

**접근법**: LSTM으로 자산별 변동성을 예측 → 저변동/고변동 그룹 spread를 단일 Black-Litterman view로 주입 → 90개 슬롯 조합(3 prior × 3 p_weight × 5 q_mode × 2 omega_mode)을 walk-forward로 백테스트 → 위험조정 메트릭 기준(`sortino_ir ≥ 10` 필터 + 전체기간 `sortino` 1위)으로 winner 선정 → 통계 검정으로 robustness 검증.

**기간**: 2010-01 ~ 2025-12 (총 192개월). K_CUT 2023-12-31 기준 분할:
- **TEST** 2010-01 ~ 2023-12 (168개월, winner 선정·민감도 분석)
- **HOLD_OUT** 2024-01 ~ 2025-12 (24개월, 실전 운용 검증)

**Winner 슬롯** (`mat_eq_eq_raw_pap`) vs SPY:

| 메트릭 | Winner | SPY | 차이 |
|---|---:|---:|---:|
| Sharpe | **1.096** | 0.731 | +0.365 |
| Sortino | **1.826** | 1.108 | +0.718 |
| CAGR | **16.2%** | 12.7% | +3.5%p |
| MDD | **−13.6%** | −33.7% | **+20.1%p (방어)** |

**검증**: Q 민감도 (Memmel JK z-test, p>0.5에서 BAB 학술값 0.0055·0.0064와 통계적 동등) + PCT 민감도 (20~35% 임계값에서 robust) + 3-레짐 HMM 안정성 (회복/확장/변동 전 구간 winner 유지) + Block Bootstrap 95% CI 0 포함 + Fama-French 5+Mom α 분해.

## Live Demo

- 🌐 **랜딩 페이지** (Vercel): _배포 후 URL 추가_
- 📊 **인터랙티브 대시보드** (Streamlit): _배포 후 URL 추가_

## 핵심 방법론

### 7-Step 파이프라인

```
1. Data Collection      yfinance + FRED + Wikipedia 멤버십 → 617종목 월별 패널
2. EDA                  수익률 예측 불가(R²≈0) + 변동성 예측 가능(ACF lag 60) 입증
3. LSTM σ 예측           Optuna HPO → V4_BEST_CONFIG → LSTM + HAR + Ensemble (RMSE 0.3822)
4. Black-Litterman      단일 spread view (저변동 long / 고변동 short), Sherman-Woodbury 닫힌해
5. MVO Walk-forward     90개 슬롯 × 192개월, TC 20bp/side, max_weight 10%
6. 분석·검정              K_CUT cutoff + Q/PCT 민감도 + 3-레짐 안정성 + α 분해
7. 배포                  Streamlit 대시보드 + Vercel 랜딩 페이지
```

### Black-Litterman 사후분포 (단일 view, K=1)

```
μ_BL = π + (τΣPᵀ)·(Q − Pπ) / (PτΣPᵀ + Ω)
```

| 슬롯 | 의미 | Winner 구성 |
|---|---|---|
| **π** (prior) | CAPM 균형수익률 (λΣw_mkt) | `capm_mcap` (시가가중) |
| **P** (1×N) | view 행렬 (저변동 +1/n, 고변동 −1/n) | `eq` (동일가중) |
| **Q** (스칼라) | view 강도 | `raw_lam` (SPY excess/σ²_mkt, 자연 게이팅) |
| **Ω** (스칼라) | view 분산 (신뢰도 역수) | `ff3_paper` (직전월 예측오차² 적응형) |

> ⚠️ `omega_mode='ff3_paper'`는 역사적 코드명. 실제 동작은 **Bayesian rolling view variance** (직전월 (Q−P·r)² 적응 갱신)이며, FF3 회귀 잔차와 무관.

### 슬롯 명명 체계 (canonical 5-token)

```
{prior}_{p_mode}_{p_weight}_{q}_{omega}
mat_eq_eq_raw_pap  ← Winner
```

자세한 슬롯 정의·수식: [`final_pt/docs/BL_EXPERIMENT_GUIDE.md`](final_pt/docs/BL_EXPERIMENT_GUIDE.md)

## 학술 근거

저변동 anomaly: Frazzini & Pedersen (2014) *Betting Against Beta*, Pacific-Basin Finance Journal, 51.
BL 응용: Pyo & Lee (2018) — ANN으로 변동성 예측 → BL view 주입 → KOSPI200 Sharpe 0.70, α 2.46% / 2008 위기 α +1.46% (CAPM −22.6%).

본 프로젝트는 학술적 baseline (KOSPI200 LowVol Sharpe 1.80, HighVol 0.33)을 미국 시장으로 확장하고, LSTM σ 예측 + 단일 spread view BL을 walk-forward로 검증.

## 저장소 구조

```
.
├── final_pt/                ← 분석·백테스트 (포트폴리오 핵심)
│   ├── 01~06_*.ipynb        7-step 파이프라인 노트북
│   ├── appendix/            보조 분석 (slot effects, LSTM 통계, 탐색)
│   ├── lib/                 핵심 모듈 (bl_*, lstm_*, master_table 등 7개)
│   ├── data/, results/      데이터·실험 산출물 (gitignored, archive 참조)
│   ├── outputs/             차트 PNG
│   └── docs/                메서드 문서 7종
├── streamlit_dashboard/     ← Streamlit 대시보드 (6 페이지)
├── vercel_deploy/           ← Vercel 정적 랜딩 페이지
├── pyproject.toml, uv.lock  ← UV 환경 정의
└── README.md
```

각 폴더에 더 자세한 README가 있습니다:
- 분석 파이프라인 → [final_pt/README.md](final_pt/README.md)
- 대시보드 → [streamlit_dashboard/README.md](streamlit_dashboard/README.md)
- 랜딩 페이지 → [vercel_deploy/README.md](vercel_deploy/README.md)

## 실행 방법

### 환경 셋업

```bash
uv sync   # Python 3.12, 의존성 일괄 설치
```

`.env` 파일에 다음 키 필요 (재실행 시):
- `FRED_API_KEY` — 거시지표 수집용
- Google BigQuery 인증 — GDELT 탐색 단계 (최종 모델은 미사용)

### Streamlit 대시보드 실행

```bash
cd streamlit_dashboard
uv run streamlit run app.py
```

### 분석 노트북 실행

```bash
cd final_pt
uv run jupyter lab
```

권장 실행 순서: `01 → 02a → 02b → 03a → 03b → 04 → 05a → 05b → 06` (파일명 순)

### Vercel 랜딩 페이지 로컬 미리보기

```bash
cd vercel_deploy
python -m http.server 8000
# 브라우저에서 http://localhost:8000
```

## 데이터 출처

| 출처 | 기간 | 용도 |
|---|---|---|
| yfinance | 2004 ~ 2026 | 833종목 OHLCV, 발행주식수 |
| Wikipedia | 2005 ~ 2025 (월별) | S&P500 멤버십 히스토리 (생존편향 제거) |
| FRED API | 2010 ~ 2024 | VIX, HY spread, yield curve, Sahm, WEI 등 거시 |
| Fama-French | 1926 ~ 현재 | FF5 + Momentum (월별·일별) |
| GDELT (via BigQuery) | 2016 ~ 2025 | 뉴스 sentiment (탐색 단계, 최종 모델 미사용) |

**투자 유니버스**: 617종목 (멤버십 필터 후, 월별 332~498개) — 5개 broad ETF (SPY/QQQ/IWM/EFA/EEM) + 4개 bond ETF (TLT/AGG/SHY/TIP) + 2개 alternative (GLD/DBC) + 11개 sector ETF + 개별 주식.

## 기술 스택

- **언어**: Python 3.12 (UV 패키지 관리)
- **ML / 시계열**: PyTorch (LSTM), Optuna (HPO), HAR-RV (HAR baseline)
- **Quant / 통계**: NumPy, Pandas, SciPy, statsmodels
- **포트폴리오 최적화**: CVXPY (long-only MVO + max_weight 제약)
- **HMM 레짐 식별**: hmmlearn
- **시각화 / 대시보드**: Matplotlib, Plotly, Streamlit
- **데이터 수집**: yfinance, fredapi

## 저장소 브랜치 구성

`main` 브랜치는 포트폴리오 view 로 정리된 최종 산출물만 포함합니다. 개인 작업 디렉터리·초기 EDA·미채택 실험 등 전체 프로세스는 [`archive` 브랜치](https://github.com/sybeom528/finance_project/tree/archive)에 보존되어 있습니다.

| 브랜치 | 용도 | 내용 |
|---|---|---|
| `main` | 포트폴리오 view | `final_pt/`, `streamlit_dashboard/`, `vercel_deploy/`, 환경 설정 파일 |
| `archive` | 전체 개발 히스토리 | 개인 작업 디렉터리, 초기 EDA, 폐기된 실험(XGBoost/TabPFN Q 직접 예측 등), 시계열 학습 자료, 미공개 보고서 등 |

archive 브랜치에 보존된 주요 자료:
- **방향 전환 흔적**: 자산별 Q 직접 예측 실패(R²≈−0.95) → 변동성 σ 예측으로 피봇한 과정
- **초기 EDA 노트북**: 모멘텀·횡단면 랭크 등 시도된 신호들
- **학습 자료**: 시계열 분석·LSTM 구조·look-ahead bias 처리 등 study notes
- **이전 매트릭스**: 90개로 축소되기 전 156개 슬롯 구성 (`omega=scaled`, sparse `vol_mcap` 변형 포함)

