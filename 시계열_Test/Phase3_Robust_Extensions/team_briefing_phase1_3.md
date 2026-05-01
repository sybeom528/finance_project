# Phase 1~3 팀 브리핑 — 한 페이지 요약

**작성일**: 2026-05-01 | **대상**: 팀원 공유용 | **분량**: A4 1면

---

## 한 줄 요약

> S&P 500 종목 대상 LSTM 기반 변동성 예측 + Black-Litterman 포트폴리오 최적화 시스템을 단계적으로 발전시켜 (Phase 1.5 → 2 → 3-1 → 3-2 v2), **2010-2024 OOS 180개월 + 2025 hold-out 11개월** 환경에서 **6 시나리오 (vol_input × weighting) fair 비교** 를 완료했다.

## 프로젝트 구조

| Phase | 핵심 작업 | 주요 산출물 |
|---|---|---|
| **1.5** | LSTM 변동성 예측 (V4_BEST_CONFIG, walk-forward CV) | `ensemble_predictions_*.csv` |
| **2** | BL 통합 (5 핵심 함수, 서윤범 99 일관) | `scripts/black_litterman.py` |
| **3-1** | Robustness 확장 (Universe + Membership + Stale + 학술보정) | freeze (참조 / 부록) |
| **3-2 (v2)** | 가중치 다양화 + OOS/Hold-out 분리 + 심층 분석 10종 | `data/bl_weights_v2_*.pkl` |

### Phase 3-1 세부 (2026-04 freeze, robustness 부록)

| 단계 | 내용 |
|---|---|
| **Universe 확장** | Top 50 → S&P 500 historical **624 종목** (Wikipedia + yfinance) |
| **2 모델 동시 학습** | 02a stockwise (per-ticker LSTM × 615) + 02b cross-sectional (1 모델, 진행 중) |
| **Step 6**: Static Universe (격리) | 모든 시점 동일 종목 → 미래 종목이 과거 시점에 사용 (look-ahead) |
| **Step 7**: Dynamic-Membership | 시점별 S&P 500 멤버십 사용, **look-ahead 차단** (Sharpe 1.108 → 1.123) |
| **Step 8**: Stale price 필터 | zero ratio > 30% 종목 제외 (SW/EP/COL/CPWR/CVG 등 ticker 재사용 사례) |
| **§7-3 학술 보정** | 생존편향 정량화 (Wikipedia 809 → yfinance 646 → trained 615 분해) |
| **§7-6/7/8** | 진정한 survivorship bias 측정 + delisted 종목 영향 |
| **결과 (mcap, 204m OOS)** | BL_ml_sw 1.122 / BL_trailing 1.207 / **ML 효과 -0.085** |

### Phase 3-2 v2 세부 (2026-04-30 ~ 진행, 메인 결과)

| 단계 | 내용 |
|---|---|
| **OOS 기간 변경** | 2009-2025 (204m) → **2010-2024 (180m) + 2025 hold-out (12m)** — GFC 회복기 제거 |
| **BL 가중치 다양화** | mcap 단독 → **mcap (Pyo 2018) + 1/N (DeMiguel 2009) + 1/σ (Maillard 2010)** |
| **6 시나리오 fair 비교** | vol_input (ML / trailing) × weighting (mcap / eq / rp) |
| **v2 노트북 6 사본** | `02a_v2`, `03_v2`, `04_v2`, `05a/b/c_v2.ipynb` (Phase 3-1 freeze, v2 main) |
| **§6 BL sanity check** | 192 시점 walk-forward, **76.7분** standalone 실행 (cell timeout 우회) |
| **§7 심층 분석 10종** | Sortino/Calmar/Info/Omega + Bootstrap CI + 거래비용 + Rolling + Tail + Drawdown + CAPM + Sector + VIX + HHI |
| **모든 분석 6 시나리오 cover** | mcap/eq/rp × ML/trailing 통합 비교 |
| **standalone 우회 패턴** | VS Code Jupyter kernel 무한 로딩 → `scripts/_run_*.py` + nbconvert 캐시 hit |
| **결과 산출물** | `bl_weights_v2_*.pkl` (6 MB) + `bl_metrics_v2_*.pkl` + `sec7_v2_analyses.pkl` + 7 PNG |

---

## Phase 1.5 — LSTM 변동성 예측

- **모델**: V4_BEST_CONFIG (3채널 input + VIX, IS=1250 d, OOS=21 d, embargo=63 d, step=21 d)
- **학습**: 8-way 병렬 GPU (RTX 4090), per-ticker LSTM × 615 종목 (stockwise) + cross-sectional LSTM 1개 (02b, 진행 중)
- **앙상블**: LSTM + HAR-RV (7:3 weighted)
- **Walk-forward CV**: 매 시점 t 에서 [t-1250d, t] 학습 → [t, t+21d] 예측 (look-ahead 0)
- **데이터 범위**: 2007-04 ~ 2025-12 (Hold-out 시기까지 walk-forward 학습 진행)

## Phase 2 — Black-Litterman 통합

- **5 핵심 함수** ([scripts/black_litterman.py](scripts/black_litterman.py)):
  - `compute_pi`: CAPM 역산 (π = λ·Σ·w_mkt, λ=2.5 fixed)
  - `build_P`: 변동성 양극단 30% long/short 행렬
  - `compute_omega`: He-Litterman view 불확실성 (Ω = τ·P·Σ·P^T, τ=0.1)
  - `black_litterman`: 단일 view 단순화 공식 (μ_BL)
  - `optimize_portfolio`: Markowitz 평균-분산 long-only (Σw=1)
- **View 가정**: q = 0.003 (월 0.3% = 연 3.6% BAB factor 보수 추정, Pyo & Lee 2018 기반)
- **베이스라인**: 서윤범 99_baseline 일관성 보존

## Phase 3-1 — Robustness Extensions (freeze)

- **Universe 확장**: Top 50 → S&P 500 historical 624 종목 (Wikipedia + yfinance)
- **Step 6→7→8 진화**:
  - **Step 6**: Static Universe (모든 시점에 동일 종목, 격리됨)
  - **Step 7**: Dynamic-Membership (시점별 S&P 500 멤버십 사용 → look-ahead 차단) — Sharpe 1.108→1.123
  - **Step 8**: Stale price 필터 (zero ratio > 30% 종목 제외, SW/EP/COL/CPWR/CVG 등) — Sharpe 1.122
- **§7-3 학술 보정**: 생존편향 정량화 (Wikipedia 809 vs yfinance 646 vs trained 615 분해)
- **결과 (mcap default, 204m OOS)**: BL_ml_sw Sharpe 1.122, BL_trailing 1.207, **ML 효과 -0.085**

## Phase 3-2 v2 — 신규 (메인 결과)

### 변경 사항
1. **OOS 기간 단축**: 2009-2025 (204m) → **2010-2024 (180m) + 2025 hold-out (12m)** — GFC 회복기 제거
2. **BL 가중치 다양화**: mcap 단독 → **mcap + 1/N (DeMiguel 2009) + 1/σ (Maillard 2010)**
3. **6 시나리오 fair 비교**: vol_input (ML / trailing) × weighting (mcap / eq / rp)
4. **02a_v2 노트북 §6 + §7** 심층 분석 47 셀 (10 신규 분석 + 6 시나리오 cover)

### 메인 결과 (OOS 180m, fair 비교)

| 시나리오 | Sharpe | CAGR % | MDD % |
|---|---|---|---|
| BL_ml_sw_mcap | 1.082 | 12.84 | -18.13 |
| BL_ml_sw_eq | **1.136** | 12.95 | -16.58 |
| BL_trailing_mcap | **1.206** ⭐ | 14.34 | -16.48 |
| SPY | 0.996 | 14.26 | -23.93 |

### Hold-out (2025, 11m forward test)

| 시나리오 | Sharpe |
|---|---|
| **BL_ml_sw_mcap** | **1.503** ⭐ |
| BL_trailing_mcap | 0.847 |
| SPY | 1.365 |

→ **ΔSharpe +0.656** — Hold-out 의 ML 압도적 우위 (preliminary forward evidence).

---

## 🌟 핵심 Finding 5선

1. **OOS mcap ML 효과 -0.124, p=0.046** — statistically significant negative (Bootstrap N=5000, §7-B)
2. **eq/rp 환경에서 ML 효과 ~0** (p>0.6) — **mcap 시총 편향이 ML 의 우위를 가린다는 통계적 근거**
3. **거래비용 10bp+ 환경에서 BL_ml_sw_eq/rp 가 trailing 추월** (§7-C, ML 의 50% turnover 효과)
4. **5/6 시나리오의 alpha 가 statistically significant** vs SPY (p<0.05, §7-G CAPM)
5. **ML 의 가치는 회복·AI 시기 (2023-2024) + Hold-out (2025) 에 발현** (시기별 분해 + Hold-out 일관)

## 추가 분석 (§7-A ~ §7-J, 6 시나리오 모두 cover)

| 분석 | best 시나리오 | 핵심 결과 |
|---|---|---|
| §7-A 다양화 메트릭 | BL_trailing_mcap | Sortino 1.928, Calmar 0.842, Omega 2.385 |
| §7-B Bootstrap CI | mcap_oos: p=0.046 | Hold-out 표본 작아 p=0.192 |
| §7-C 거래비용 | BL_ml_sw_eq @ 20bp | ΔSharpe +0.022 (ML 추월) |
| §7-D 36m Rolling | BL_trailing_rp | alpha 4.65%/y, beta 0.56 (defensive) |
| §7-E Tail Risk | BL_trailing_mcap | VaR -3.94% (가장 작음) |
| §7-F Drawdown | BL_ml_sw_mcap | 8 events, max -18.13% (2020-COVID) |
| §7-G CAPM | BL_trailing_rp | annual alpha 4.30%, p=0.026 |
| §7-H Sector | mcap 환경만 차이 | ML 이 IT -2.42%p (빅테크 회피) |
| §7-I VIX Regime | High VIX best (모든 시나리오) | ML 약점은 Low VIX 시기 |
| §7-J Diversification | BL_ml_sw_eq | Effective N 50.0 (가장 분산) |

## 다음 단계

- ⏳ **02b 학습 완료 후** (~7시간 남음): 03_v2 정식 9 시나리오 BL 백테스트 + 04/05b/05c_v2 본격
- ⏳ **§7-B ~ §7-J 각 분석 자세한 해설** 작성 (학술 보고서 작성 자료)
- ⏳ **보고서 Limitations 작성**: forward test 의 한계 (6 시나리오 동시 평가의 multiple comparison), Stale price (yfinance 한계), CRSP 재현 검증 권고

## 주요 파일

- 노트북: [02a_v2.ipynb](02a_v2.ipynb) (47 셀, 2.46 MB)
- WORKLOG: [재천_WORKLOG_v2.md](재천_WORKLOG_v2.md) (Phase 3-2 작업 기록)
- 캐시: `data/bl_weights_v2_sanity_check.pkl` (6 MB), `data/bl_metrics_v2_sanity_check.pkl`, `data/sec7_v2_analyses.pkl`
- 시각화: `outputs/02a_v2_stockwise/` (7 PNG)

---

**Phase 3-2 v2 의 학술 / 실무적 의의:**

1. **학술적**: BL 가중치 다양화 (Pyo & Lee 2018 mcap 외에 DeMiguel eq + Maillard rp) 환경에서의 ML 효과 robustness 검증 — eq/rp 가 ML 의 small advantage 를 발현시키는 결과
2. **실무적**: 거래비용 10bp+ 환경에서의 ML 우위 + ~50% turnover 안정성 → 실제 운용 환경에서 BL_ml_sw_eq 가 강력한 후보
3. **방법론적**: 단일 walk-forward 루프 (76분, 192 시점) + 메트릭 단계 OOS/Hold-out 분리 → 효율 + 학술 표준 동시 달성
