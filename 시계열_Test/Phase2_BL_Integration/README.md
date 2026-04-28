# Phase 2 — Black-Litterman 통합 단계

> **협업 진입점 문서**. 처음 합류한 팀원은 이 README → [PLAN.md](PLAN.md) → [재천_WORKLOG.md](재천_WORKLOG.md) → 노트북 순으로 읽으십시오.
>
> - **PLAN.md**: 전체 구현 계획서 + 8 결정사항 + 데이터 요구량
> - **재천_WORKLOG.md**: 작업·판단 일지 (시간순 누적)
> - **노트북**: 실제 분석·학습·백테스트 흐름

---

## 1. 단계 위치 및 목적

- **상위 프로젝트**: COL-BL (Su et al. 2026 ESWA 295 + Pyo & Lee 2018 PBFJ 51 결합)
- **현 단계**: **Phase 2** — [Phase 1.5](../Phase1_5_Volatility/) 변동성 예측 결과를 Black-Litterman 포트폴리오로 통합
- **Phase 1, 1.5 와의 관계**:
  - [Phase 1](../Phase1_LSTM/) (수익률 예측, 보존, 변경 금지)
  - [Phase 1.5](../Phase1_5_Volatility/) (변동성 예측 v8 ensemble, 본 단계의 핵심 입력)
  - **Phase 2 (본 폴더)**: ensemble 변동성 예측 → BL 의 P 행렬 → 포트폴리오 가중치
- **유일한 목적**: **"Phase 1.5 ensemble 의 변동성 예측 정확도 향상이 BL 포트폴리오의 위험조정 수익으로 이전되는가?"** 단일 질문에 답

### Phase 1.5 ⇄ Phase 2 비교

| 항목 | Phase 1.5 | **Phase 2** |
|---|---|---|
| 폴더 | `Phase1_5_Volatility/` (보존) | `Phase2_BL_Integration/` (본 폴더) |
| 자산군 | 7 종목 (SPY, QQQ, DIA, EEM, XLF, GOOGL, WMT) | **S&P 500 Top 50 (매년 갱신)** ⭐ |
| 데이터 빈도 | 일별 (input) → 21일 forward (target) | **일별 ret 으로 Σ 추정 → × 21 월별 환산 (BL 월별 단위 통일)** |
| 평가 지표 | RMSE / QLIKE / R²_train_mean | **Sharpe / Alpha / MDD / CumRet** ⭐ |
| 베이스라인 | HAR-RV / EWMA / Naive / Train-Mean | **SPY / 1/N / Mcap / BL** ⭐ |
| 분석 기간 | 2016 ~ 2025 (10년) | 2013-04 ~ 2025-12 (12.7년) |
| Walk-Forward | IS=1250 / OOS=21 / 55 fold | **동일** (Phase 1.5 일관) |

### 본 단계 평가 비대상 (의도적 분리)

추후 별도 단계로 미룸:
- 공매도 제약 (long-only) 실험
- τ 정밀 민감도 (6 점)
- Q 동적 추정 (FF3 회귀, Phase 1 LSTM)
- Ω 동적 추정 (전월 잔차 분산, Pyo & Lee 정합)
- 종목 수 확장 (Top 100, S&P 500 전체)

---

## 2. 핵심 의사결정 요약 (8 결정, 2026-04-28 사용자 확정)

자세한 근거는 [PLAN.md](PLAN.md) §2 및 [재천_WORKLOG.md](재천_WORKLOG.md) 참조.

| # | 영역 | 결정 |
|---|---|---|
| 1 | 거래비용 | **0 default, 인자화** (`transaction_cost: float = 0.0`) |
| 2 | 부족 종목 | **51위 이하 자동 대체** |
| 3 | 시총 데이터 출처 | **서윤범 01_DataCollection 로직 재사용** |
| 4 | Performance ensemble warmup | **신규 편입 종목만 reset, 기존은 history 유지** |
| 5 | BL Σ (공분산) | **일별 ret 으로 추정 후 × 21 월별 환산** + LedoitWolf shrinkage |
| 6 | 벤치마크 | **SPY + 1/N + Mcap + BL_Phase15 (4종)** |
| 7 | OOS 구조 | **Phase 1.5 와 동일 — 21일 forward, 매월 walk-forward** |
| 8 | 상장폐지 처리 | **남은 종목 시가총액 비중으로 비례 전이** |

---

## 3. 폴더 구조

```
Phase2_BL_Integration/
├── README.md                              ← 이 문서
├── PLAN.md                                ← ⭐ 전체 구현 계획 + 8 결정사항
├── 재천_WORKLOG.md                         ← 작업·판단 일지
│
├── 00_setup_and_utils.ipynb               ← Phase 1.5 복사 — 환경 부트스트랩
├── 01_universe_construction.ipynb         ← ⭐ 매년 시총 상위 50 산정 (예정)
├── 02_data_collection.ipynb               ← ⭐ 일별 데이터 수집 + 통합 패널 (예정)
├── 03_phase15_ensemble_top50.ipynb        ← ⭐ Phase 1.5 ensemble 50 종목 확장 (예정)
├── 04_BL_yearly_rebalance.ipynb           ← ⭐ BL 백테스트 (예정)
├── 05_comparison.ipynb                    ← ⭐ 4 비교군 + 시각화 + 보고서 (예정)
│
├── scripts/                               ← 재사용 모듈 (예정)
│   ├── universe.py                        ← 시총 상위 50 + fallback (결정 2, 3)
│   ├── volatility_ensemble.py             ← Performance ensemble + 신규 reset (결정 4)
│   ├── covariance.py                      ← 일별 ret + LedoitWolf (결정 5)
│   ├── black_litterman.py                 ← 서윤범 99_baseline 함수 import
│   ├── backtest.py                        ← transaction_cost 인자화 + 폐지 처리 (결정 1, 8)
│   └── benchmarks.py                      ← SPY/1N/Mcap/BL (결정 6)
│
├── data/                                  ← 원천·중간 데이터 (예정)
│   ├── universe_top50_history.csv         ← 매년 universe (50×N)
│   ├── prices_daily/{ticker}.csv          ← 종목별 일별 OHLCV
│   ├── ensemble_predictions_top50.csv     ← Phase 1.5 ensemble 50 종목 확장
│   ├── market_data.csv                    ← SPY, VIX, ^TNX 일별
│   ├── ff3_monthly.csv                    ← Fama-French
│   ├── risk_free.csv                      ← FRED DGS3MO
│   └── daily_panel.csv                    ← 통합 일별 패널
│
└── outputs/                               ← 노트북 산출물만 (예정)
    ├── 03_ensemble_top50/
    ├── 04_BL_yearly/
    └── 05_comparison/
```

---

## 4. 실행 순서

### 4.1 환경 설치 (OS 호환)

본 단계는 **Windows / macOS / Linux** 모두에서 실행 가능하도록 설계되었습니다 (Phase 1.5 와 동일 환경).

```bash
# 1. Jupyter 가 사용하는 Python 확인 (모든 OS 공통)
python -c "import sys; print(sys.executable)"

# 2. 의존성 설치 (모든 OS 공통)
python -m pip install yfinance statsmodels scipy scikit-learn torch \
                      pandas numpy matplotlib jupyter nbconvert \
                      pandas-datareader

# 3. (Linux 전용) 한글 폰트 패키지 추가 — matplotlib 한글 깨짐 방지
python -m pip install koreanize-matplotlib --break-system-packages
```

> ⚠️ **함정 주의**: `pip install` 이 다른 Python 환경(예: MiniConda)으로 잘못 가는 경우가 있습니다. 반드시 Jupyter 가 쓰는 Python 에서 `python -m pip` 로 설치하십시오.

> 한글 폰트는 `00_setup_and_utils.ipynb` 가 OS 자동 분기 처리합니다 (Windows: Malgun Gothic / macOS: AppleGothic / Linux: NanumGothic via koreanize-matplotlib).

### 4.2 노트북 실행 순서

```
00_setup_and_utils.ipynb              ← 환경 부트스트랩 (% run 으로 호출됨)
        ↓
01_universe_construction.ipynb        ← 매년 시총 상위 50 산정 (예정)
        ↓
02_data_collection.ipynb              ← 일별 데이터 수집 (예정)
        ↓
03_phase15_ensemble_top50.ipynb       ← Phase 1.5 ensemble 50 종목 학습 ⚠️ 가장 무거움 (예정)
        ↓
04_BL_yearly_rebalance.ipynb          ← BL 백테스트 (예정)
        ↓
05_comparison.ipynb                   ← 4 비교군 비교 + 보고서 (예정)
```

---

## 5. 주요 학술 출처

- **Pyo, S., & Lee, J. (2018).** Exploiting the low-risk anomaly using machine learning to enhance the Black-Litterman framework. *Pacific-Basin Finance Journal*, 51, 1-12.
- **Black, F., & Litterman, R. (1992).** Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28-43.
- **He, G., & Litterman, R. (1999).** The intuition behind Black-Litterman model portfolios. *Goldman Sachs Investment Management Research*.
- **Idzorek, T. M. (2005).** A step-by-step guide to the Black-Litterman model: Incorporating user-specified confidence levels. *Ibbotson Associates*.
- **Frazzini, A., & Pedersen, L. H. (2014).** Betting against beta. *Journal of Financial Economics*, 111(1), 1-25.
- **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915-1953.
- **Ledoit, O., & Wolf, M. (2004).** A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
- **Corsi, F. (2009).** A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.

---

## 6. 진행 현황 (2026-04-28 기준)

| Step | 작업 | 상태 |
|---|---|---|
| 0 | 폴더 + 3 문서 (README/PLAN/WORKLOG) | ✅ 완료 |
| 1 | Universe construction (매년 시총 상위 50) | ⏳ 예정 |
| 2 | Data collection (일별 패널) | ⏳ 예정 |
| 3 | Phase 1.5 ensemble → 50 종목 확장 | ⏳ 예정 |
| 4 | BL Yearly rebalance backtest | ⏳ 예정 |
| 5 | 4 비교군 + 시각화 + 보고서 | ⏳ 예정 |

다음 작업: **Step 1 (Universe Construction)** — 사용자 승인 대기.
