# Phase 3 — Robust Extensions

> Phase 1.5 v8 ensemble (LSTM + HAR-RV) 의 변동성 예측 정확도 향상이, 다양한 가정 변경에 대해 BL 포트폴리오 성과로 이전되는지 검증하는 단계.

## 단일 질문

> **"본 Phase 2 의 BL_ml 우위가 시기 확장 (6년 → 17년) + universe 확장 (74 → 615 종목) + 학습 방식 (종목별 vs Cross-Sec) 의 변경에 robust 한가?"**

## 진입 조건

✅ Phase 1.5 ~ 2 정합성 검증 완료 (Issue #1, #1B, #2 모두 수정).

검증 결과:
1. Phase 1.5 의 Walk-Forward / 누수 방지 / 메트릭 / 단위 — 정합성 OK
2. Phase 2 의 BL 공식 / 백테스트 / Σ 환산 / 시나리오 비교 — Issue 수정 후 정합성 OK
3. Phase 1.5 ↔ Phase 2 연결 (단위, 시점, target) — 정합성 OK

---

## 현재 진행 상태 (2026-04-29 갱신) ⭐

| Step | 노트북 | 상태 |
|---|---|---|
| 1 | `01_universe_extended.ipynb` | ✅ 완료 (universe 809, panel 646) |
| 2a | `02a_phase15_stockwise_extended.ipynb` | ✅ 완료 (615 학습, RMSE 0.391) |
| 2a-§6 | (02a 내부) BL sanity check | ✅ 완료 (3 시나리오 + 패러독스 발견) |
| 2b | `02b_phase15_cross_sectional.ipynb` | ⏳ 대기 |
| 3 | `03_BL_backtest_extended.ipynb` | ⏳ 02b 후 |
| 4 | `04_compare_stockwise_vs_cross.ipynb` | ⏳ 03 후 |
| 5a/b/c | `05a/b/c_eval_*.ipynb` | 빌드 완료, 실행 대기 |

---

## ⭐ 핵심 발견 (Step 2a 완료 후, 2026-04-29)

### 1. 02a 학습 결과

- **615 종목 × 17 년 학습** 완료 (RTX 4090, 약 15h)
- Ensemble RMSE 0.391 (LSTM 0.529, HAR 0.401)
- Best 모델 분포: Ensemble 65% / HAR 32.6% / LSTM 2.4%
- Phase 1.5 v8 (74 종목) 의 64% 와 거의 동등 → 학습 패턴 재현성 검증

### 2. §6 BL Sanity Check — 3 시나리오 (203 개월, 17 년)

| 시나리오 | Sharpe | CAGR | Vol | MDD |
|---|---|---|---|---|
| **BL_trailing** | **1.222** ⭐ | 14.52% | 11.71% | -15.88% |
| BL_ml_sw | 1.108 | 13.41% | 12.07% | -18.56% |
| SPY | 1.050 | 15.37% | 14.72% | -23.93% |

- **서윤범 99 재현 검증**: BL_trailing 1.222 vs 서윤범 재계산 1.157 (+5.62%, 양호)
- **ML 통합 효과**: Sharpe -0.114 (NEGATIVE)

### 3. ⭐ 패러독스 — Hit Rate ↑ 인데 BL 성과 ↓

| 측정 | ML | Trailing | 차이 |
|---|---|---|---|
| Low vol hit rate | **0.634** | 0.590 | +4.4%p |
| High vol hit rate | **0.663** | 0.626 | +3.7%p |
| Spearman rank corr | **0.688** | 0.616 | +0.072 |
| **BL Sharpe** | 1.108 | **1.222** | -0.114 |

→ **모든 vol ranking 측정에서 ML > Trailing 인데 BL 성과는 ML < Trailing**

### 4. LS Spread 분석 — BAB Anomaly 17 년 평균 미작동

| 측정 (mcap-w, 연환산 %) | ML | Trailing |
|---|---|---|
| LS spread (전체) | -9.53% | -4.84% |
| 강세장 (12~19) | -3.55% | **+3.17%** |
| 긴축 (21~22) | +1.88% | **+8.91%** |

→ Trailing 의 BAB 활용도 ML 능가 (132 개월 vs 24 개월).

### 5. 진단 — Trailing vol = 방어주 식별 proxy 가설

```
Trailing vol_21d 의 진정한 가치:
  "최근 vol 이 낮음" = "안정적 cash flow 회사" 의 proxy
  Utilities, Consumer Staples, Healthcare 식별
  → BAB anomaly 의 underlying 회사 특성과 일치

ML forward vol prediction 의 한계:
  정확한 vol 예측 ≠ 회사 특성 식별
  "이번 달 vol 이 낮을 종목" ≠ "구조적으로 안정적인 회사"
  → BAB anomaly 활용에는 부적합
```

### 6. 학술 기여 (잠정)

> **"Volatility prediction accuracy improvement (RMSE↓, hit rate↑) does NOT
>  translate to Black-Litterman portfolio alpha when used as P-matrix sorter."**

→ Pyo & Lee (2018) "ML > Trailing" 주장 **부분 반증** (KOSPI vs 미국 17 년 환경).
→ "Vol prediction" 과 "BAB anomaly" 의 **분리** — 학술 보고 핵심 결론 가능.

---

## 다음 단계

1. **02b cross-sectional 학습** (1-2 시간)
   - 핵심 검증: CS 가 BAB anomaly 를 더 잘 잡을 수 있는가?
   - Cross-sectional 비교가 회사 특성 식별에 유리할 가능성
2. **03 BL 백테스트** (02b 후)
   - 6 시나리오: BL_trailing, BL_ml_sw, BL_ml_cs, SPY, 1/N, McapWeight
3. **04 + 05a/b/c 평가** (Layer 1~5)
   - 모델별 단독 평가 + 시나리오 간 통계 검정
   - 최종 학술 보고서

---

## 폴더 구조

```
Phase3_Robust_Extensions/
├── README.md (본 파일)
├── PLAN.md
├── NOTEBOOK_TODO.md
├── 재천_WORKLOG.md
│
├── 01_universe_extended.ipynb       ✅
├── 02a_phase15_stockwise_extended.ipynb  ✅ (§6 sanity check 포함)
├── 02b_phase15_cross_sectional.ipynb     ⏳
├── 03_BL_backtest_extended.ipynb         ⏳
├── 04_compare_stockwise_vs_cross.ipynb   ⏳
├── 05a_eval_stockwise.ipynb              📋 빌드 완료
├── 05b_eval_crosssec.ipynb               📋 빌드 완료
├── 05c_eval_compare.ipynb                📋 빌드 완료
│
├── _build_*.py                       (빌드 스크립트들)
│
├── scripts/
│   ├── setup.py
│   ├── black_litterman.py            (TAU=0.1, LAM_FIXED=2.5)
│   ├── covariance.py
│   ├── backtest.py
│   ├── benchmarks.py
│   ├── universe.py
│   ├── data_collection.py
│   ├── volatility_ensemble.py        (Issue 3 수정 적용)
│   ├── models_cs.py                  (Cross-Sectional LSTM)
│   ├── universe_extended.py          (universe 17년 확장)
│   └── diagnostics.py                (Layer 1~5 평가 모듈)
│
├── data/
│   ├── universe_full_history.csv
│   ├── daily_panel.csv
│   ├── fold_predictions_stockwise.csv
│   └── ensemble_predictions_stockwise.csv  ⭐
└── outputs/
    └── 02a_stockwise/
        ├── bl_sanity_check.png
        ├── hit_rate_analysis.png
        └── paradox_analysis.png
```

---

## 주요 산출물

- `data/ensemble_predictions_stockwise.csv` (613 종목, 약 2.47M 행, RMSE 0.391)
- `outputs/02a_stockwise/bl_sanity_check.png` (3 panel: 누적/DD/Rolling Sharpe)
- `outputs/02a_stockwise/hit_rate_analysis.png` (4 panel: hit rate, Spearman, confusion)
- `outputs/02a_stockwise/paradox_analysis.png` (4 panel: LS spread 시기별, 시총)

---

## 학술 메시지

> "Phase 1.5 v8 ensemble 의 변동성 예측 정확도 향상은, BL 포트폴리오의 P 행렬 정렬 기준으로 사용될 때 alpha 로 이전되지 않는다.
>  Trailing vol 이 BAB anomaly 의 underlying 회사 특성 (방어주) 식별 proxy 로 작동하기 때문.
>  Cross-Sec LSTM 의 BAB 활용도 검증이 다음 핵심 단계."

---

## 참조 문헌

- Pyo, S., & Lee, J. (2018). *Pacific-Basin Finance Journal*, 51 (BL + ML 변동성 예측)
- Frazzini, A., & Pedersen, L. H. (2014). *Journal of Financial Economics* (Betting Against Beta)
- DeMiguel, Garlappi, Uppal (2009). *Review of Financial Studies* (1/N vs Mean-Variance)
- Gu, S., Kelly, B., & Xiu, D. (2020). *Review of Financial Studies* (Cross-Sectional ML)
- 서윤범 99_baseline (2026, 본 프로젝트 동료) — Sharpe 1.157 reference
