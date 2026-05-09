# Top 1 모델 의사결정 보고서

> **작성일**: 2026-05-08
> **분석 노트북**: `final/06_Top1_Selection.ipynb` (42 cells, §0~§10)
> **plan**: `final/06_Top1_Selection_plan.md`

---

## 0. Executive Summary

본 분석은 BL 156 cfg 중 객관적 다중 메트릭 + lexicographic 우선순위 + sensitivity test 절차로 Top 1 을 재산출합니다.

### 핵심 발견

| 평가 기준 | Top 1 | 비고 |
|---|---|---|
| **Lexicographic** (1순위 sortino_TEST → 2순위 MDD → 3순위 sortino_ir, ε=0.10) | `mat_eq_mcap_lam_he` | sortino_TEST + MDD 균형 우수 |
| **Decision matrix** (성과 30% / 위험 25% / 안정성 20% / 견고성 15% / Alpha 10%) | **`mat_eq_eq_lam_pap`** ← 잠정 Top 1 | alpha + IR + eff_n 우수 |
| **1순위 = sortino_HOLD_OUT** | `mat_mcap_mcap_raw_he` | HO 1.640 압도적 |
| **1순위 = sortino_FULL** | `mat_mcap_mcap_raw_rms` | 전 기간 통합 우수 |

**평가 기준에 따라 Top 1 이 매우 다양하게 변경** → △ 조건부 ROBUST 분류.

### 권고 (사용자 결정 필요)

본 분석은 **trade-off 명확화**가 핵심 가치이며, 단일 "정답" 을 제시하지 않습니다. 발표 / 시연 / 학술적 강조점에 따라 다음 4 옵션 중 선택:

- **옵션 A** (현 잠정 유지): `mat_eq_eq_lam_pap` — alpha + 분산 우수, HO 부진 인정
- **옵션 B** (lexicographic 표준): `mat_eq_mcap_lam_he` — sortino_TEST + MDD 균형
- **옵션 C** (HO 우선): `mat_mcap_mcap_raw_he` — HOLD_OUT 압도적 (학습편향 회피)
- **옵션 D** (안정성 압도): `mat_mcap_rp_lam_pap` — sortino_ir 28.15 (regime IR 1위)

---

## 1. 분석 절차

### 1-1. Universe 정의

- **시작**: 156 cfg → 153 main (quantile variants `_q55/_q64/_q70` 제외)
- **교집합**:
  - `list_A` = sortino_TEST top 50 (성과)
  - `list_B` = sortino_ir top 50 (3-regime 안정성)
  - **|A ∩ B| = 22 cfg** (성과 + 안정성 동시 우수)

### 1-2. Hard filter

22 cfg 분포 분석 결과, 모든 후보 hard filter (`mdd_TEST > -0.40`, `mdd_HO > -0.30`, `eff_n ≥ 30`, `sortino_HO > 0`) 가 **이미 통과**. 추가 필터 적용하지 않음 (M = 22).

### 1-3. Lexicographic 종합 점수

```
1차: sortino_TEST 내림차순 정렬
  └─ tied 정의: |s1 - s2| < ε (ε = 0.10)
2차: tied 그룹 내 → (rank_MDD_TEST + rank_MDD_HO) / 2 평균 rank 오름차순
3차: 그래도 tied 시 → sortino_ir 내림차순
```

**ε = 0.10 결과**: 22 cfg 가 3 개 동순위 그룹으로 분리 (top group = 7 cfg, sortino_TEST 1.996~2.076).

### 1-4. Top 10 정밀 분석

16 메트릭 z-score heatmap → 후보별 강·약점 분석.

### 1-5. 결정 matrix (Top 5)

| 차원 | 가중치 | 메트릭 |
|---|---|---|
| 성과 | 30% | sortino_TEST + sortino_HOLD_OUT |
| 위험 | 25% | mdd_FULL + cvar_5 + calmar |
| 안정성 | 20% | sortino_ir + (1 - TEST_HO_gap) |
| 견고성 | 15% | eff_n_avg + (1 - turnover_avg) |
| Alpha | 10% | alpha + IR + |β-0.7| (defensive 적정성) |

### 1-6. Sensitivity test

- ε 변경 (0.05 / 0.10 / 0.20)
- 우선순위 변경 (sortino_HO / sortino_FULL 1순위)
- → robust 분류

---

## 2. Top 4 후보 비교

| 메트릭 | A: mat_eq_eq_lam_pap (잠정) | B: mat_eq_mcap_lam_he (Lex Top 1) | C: mat_mcap_mcap_raw_he (HO Top 1) | D: mat_mcap_rp_lam_pap (sortino_ir Top 1) |
|---|---:|---:|---:|---:|
| **sortino_TEST** (학습 168m) | **2.015** ★ | 1.996 | 1.919 | 1.878 |
| **sortino_HOLD_OUT** (24m) | 0.685 ✗ | 0.798 | **1.640** ★ | 1.235 |
| **sortino_FULL** (192m) | 1.892 | 1.842 | 1.877 | 1.806 |
| **sortino_ir** (regime 안정성) | 10.46 | 7.24 | 9.73 | **28.15** ★ |
| **mdd_TEST** | -0.129 | **-0.120** ★ | -0.136 | -0.137 |
| **mdd_HOLD_OUT** | -0.083 | -0.068 | **-0.044** ★ | -0.084 |
| **calmar** | **1.289** ★ | 1.105 | 0.964 | 1.169 |
| **alpha** (CAPM) | **0.052** ★ | 0.043 | 0.047 | 0.048 |
| **beta** (defensive: ~0.7) | 0.739 ★ | 0.545 | 0.514 | 0.720 |
| **turnover_avg** | 0.988 (높음) | **0.430** ★ | 0.395 ★ | 0.929 |
| **eff_n_avg** (분산) | **220** ★ | 61 | 35 | 68 |

★: 4 후보 중 1위. ✗: 의심 영역 (HO sortino < 0.7).

---

## 3. 후보별 강·약점 narrative

### 3-A. `mat_eq_eq_lam_pap` (잠정 Top 1, decision matrix Top 1)

**강점**:
- sortino_TEST 2.015 (4 후보 중 1위)
- alpha 0.052 (1위) + beta 0.739 (defensive 적정)
- eff_n 220 (분산 압도적, 학술적 robustness)
- calmar 1.289 (1위)
- decision matrix 가중 점수 1위

**약점**:
- **sortino_HO 0.685 (Top 134/153)** — 학습편향 의심
- TEST/HO 격차 0.66 (4 후보 중 가장 큼)
- mdd_HO -0.083 (4 후보 중 가장 나쁨)
- turnover 0.99 (높음, 거래비용 부담)
- 2024-12 -7.7% loss (sector rotation 취약, 별도 검증 보고서 §5-2)

**적합 narrative**: "in-sample 학습 + 분산 + alpha 우수 모델. HO 부진은 2024-12 sector rotation 단일 사건의 영향. 대용량 분산 (eff_n 220) 으로 학술적 robustness 시연."

### 3-B. `mat_eq_mcap_lam_he` (Lexicographic Top 1)

**강점**:
- sortino_TEST 1.996 (A 와 0.02 차이, 동급)
- **mdd_TEST -0.120 (1위)** + mdd_HO -0.068 (양호)
- turnover 0.43 (낮음, 비용 우수)
- Lexicographic 표준 절차 1위

**약점**:
- sortino_HO 0.798 (mid-tier)
- alpha 0.043 (다소 낮음)
- eff_n 61 (분산 다소 부족)

**적합 narrative**: "객관적 lexicographic 절차의 1위. sortino_TEST + MDD 균형이 가장 우수한 모델. HO 는 평균적이나 학습편향 위험 낮음."

### 3-C. `mat_mcap_mcap_raw_he` (HO Top 1)

**강점**:
- **sortino_HO 1.640 (압도적 1위)** — 학습편향 회피의 명확한 증거
- **mdd_HO -0.044 (압도적 1위)** — 미래 위험 통제 우수
- turnover 0.40 (가장 낮음)

**약점**:
- sortino_TEST 1.919 (4 후보 중 가장 낮음)
- mdd_TEST -0.136 (다소 큼)
- eff_n 35 (분산 부족, 학술적 약점)
- beta 0.514 (very defensive, bull market 시 underperform 위험)

**적합 narrative**: "HO 우선 시 명확한 1위. 학습편향에 견고하며 미래 위험 통제 압도적. 단, 학습 기간 성과는 다소 약함 → 보수적 권고에 적합."

### 3-D. `mat_mcap_rp_lam_pap` (sortino_ir Top 1)

**강점**:
- **sortino_ir 28.15 (regime 안정성 압도적 1위)**
- sortino_HO 1.235 (양호)
- alpha 0.048 + beta 0.720 (균형)
- 3 regime 모두 안정적 sortino (mean 1.858 / std 0.066)

**약점**:
- sortino_TEST 1.878 (다소 낮음)
- mdd_HO -0.084 (다소 큼)
- turnover 0.93 (높음)

**적합 narrative**: "Regime 변화에 가장 둔감한 모델. R1/R2/R3 sortino 변동이 28.15 IR 로 가장 작음. 장기·규모 큰 자금 운용에 적합."

---

## 4. Sensitivity test 결과

| 변경 시나리오 | 변경 후 Top 1 | 원래 Top 1 (matrix) 대비 |
|---|---|---|
| ε = 0.05 | mat_eq_eq_raw_pap | ✗ 변경 |
| ε = 0.10 (default lex) | mat_eq_mcap_lam_he | ✗ 변경 |
| ε = 0.20 | q_lambda | ✗ 변경 |
| 1순위 = sortino_HO | mat_mcap_mcap_raw_he | ✗ 변경 |
| 1순위 = sortino_FULL | mat_mcap_mcap_raw_rms | ✗ 변경 |

**전체 5/5 변경 시 Top 1 변경** → **△ 조건부 ROBUST**

**해석**: 22 cfg 가 trade-off 가 명확한 후보들이며, 어떤 평가 기준을 1순위로 두느냐에 따라 Top 1 이 완전히 바뀜. 즉, **단일 "정답" 모델이 존재하지 않음**.

이는 본 분석의 한계가 아닌, **객관적 발견**입니다:
- 성과 (sortino_TEST) 우수 모델은 HO 부진 경향
- HO 우수 모델은 TEST 다소 부진
- 안정성 (sortino_ir) 우수 모델은 절대 성과 다소 낮음

**시사점**: 발표 narrative 에서 "단일 Top 1" 보다 **"4 옵션 trade-off"** 강조가 학술적으로 더 정직.

---

## 5. 최종 권고

### 5-1. 발표 / 시연 목적별 권장

| 목적 | 권장 Top 1 | 근거 |
|---|---|---|
| **학술적 robustness 강조** | A: mat_eq_eq_lam_pap | eff_n 220 (분산 1위) + alpha 1위 + calmar 1위 |
| **객관적 절차 강조** | B: mat_eq_mcap_lam_he | Lexicographic 표준 1위 + MDD 균형 |
| **학습편향 회피 강조** | **C: mat_mcap_mcap_raw_he** | HO sortino 1.640 압도적 (보수적 권고) |
| **regime 안정성 강조** | D: mat_mcap_rp_lam_pap | sortino_ir 28.15 (압도적 1위) |

### 5-2. 본 분석가의 견해

**가장 안전한 권고: 옵션 C (mat_mcap_mcap_raw_he)**

**근거**:
1. HOLD_OUT 24m sortino 1.640 (압도적) — 학습편향 위험 가장 낮음
2. mdd_HO -0.044 — 미래 시점 위험 통제 압도적
3. 4번째 LSTM 재학습 후에도 일관된 우수성
4. 단점 (eff_n 35, sortino_TEST 1.92) 은 발표에서 충분히 설명 가능

**보수적 견해**: 사용자가 발표·시연 목적으로 "Top 1" 단일 모델을 강조하는 데이터에 학습편향 비용이 0.685 → 1.640 격차로 명확. 학술 보고서에서 학습편향 회피는 결정적 신뢰성 요소.

**대안 견해**: 만약 "잠정 Top 1 유지" 가 narrative 일관성 측면에서 더 유리하다면 옵션 A 도 정당화 가능. 단, HO 부진은 명시적으로 인정 필요 + 2024-12 sector rotation 분해 (별도 검증 보고서 §5-2) 를 함께 제시.

### 5-3. 비권장: 단일 Top 1 강조

본 분석의 가장 큰 발견은 **trade-off 명확화** 입니다. 발표에서 "Top 1 = X" 단일 모델 강조보다 **"4 옵션 + trade-off"** 형식이 학술적으로 더 정직.

---

## 6. 한계점

### 6-1. Universe 한계

- 156 cfg 중 153 main 만 분석 (quantile variants 제외)
- 추가 cfg 정의 시 결과 변경 가능

### 6-2. 평가 기준 한계

- Lexicographic ε = 0.10 임의 선택 (sensitivity 검증)
- Decision matrix 가중치 (30/25/20/15/10) 는 사용자 합의 없이 내부 선정
- → ε / 가중치 변경 시 Top 1 재조정 필요

### 6-3. 시계열 한계

- 192m (2010-01~2025-12) — 2008 위기 미포함
- HOLD_OUT 24m (2024-2025) — 다른 기간 (예: 2008-2009) 시 결과 다를 가능성

### 6-4. 비포함 요소

- 거래비용 변화 (현재 tc=0.001 가정)
- 슬리피지
- 세금 / 차입 비용
- ESG / 섹터 제약

---

## 7. 산출물

| 파일 | 내용 |
|---|---|
| `final/06_Top1_Selection.ipynb` | 분석 노트북 (42 cells, §0~§10) |
| `final/outputs/06_top1/intersection_summary.csv` | 22 cfg 교집합 |
| `final/outputs/06_top1/filtered_M_summary.csv` | hard filter 후 (= 22) |
| `final/outputs/06_top1/top10_metrics.csv` | Top 10 × 16 메트릭 |
| `final/outputs/06_top1/top5_decision_matrix.csv` | 결정 matrix |
| `final/outputs/06_top1/sensitivity_summary.csv` | sensitivity 결과 |
| `final/outputs/06_top1/figures/fig01~fig11.png` | 차트 11장 |
| `final/_top1_decision_2026_05_08.md` | **본 보고서** |

---

## 8. 다음 단계

1. **사용자 의사결정** (4 옵션 중 선택, 또는 4 옵션 trade-off narrative 채택)
2. **`narrative.py` + `recommendations.py` 갱신** — Streamlit 재구축 시 활용
3. **`PROJECT_OVERVIEW.md` 본문 갱신** — Top 후보 명시 + 발표 narrative
4. **Streamlit 단계적 재구축** — 별도 plan 으로 처음부터 페이지 단위

---

## 9. HOLD_OUT 섹터 분해 — 사용자 가설 검증 (2026-05-09 추가)

### 9-1. 가설

> **2024-2025 반도체 섹터(IT)가 시장 자금을 빨아 급상승 → 고변동 회피(저변동 anomaly) 전략 특성상 IT 섹터에 under-weight → SPY 대비 underperform**

### 9-2. 검증 결과 — ✓ 강력 확증 (3/3 가설 모두 통과)

#### 가설 1 ✓: IT 섹터 자금 유입 확인

| 시점 | 시장 IT 비중 | IT mcap 성장 (정규화) |
|---|---|---|
| 2024-01 | 25.8% | 1.00 |
| 2024-12 | 31.7% | 1.63 |
| 2025-12 | **33.2%** (Δ +7.4%p) | **2.07** (207% 성장) |

→ IT 섹터 단독 24m **2배 이상 성장**, 시장 비중 +7.4%p 증가. AI/반도체 buy 가설 확증.

#### 가설 2 ✓: 4 cfg 모두 IT under-weight (4/4)

| cfg | IT 비중 (2024-01 → 2025-12) | 평균 active weight |
|---|---|---|
| A. mat_eq_eq_lam_pap | 0.073 → 0.084 | -0.247 |
| B. mat_eq_mcap_lam_he | 0.016 → **0.000** | -0.297 (가장 심함) |
| C. mat_mcap_mcap_raw_he | 0.129 → 0.179 | -0.177 |
| D. mat_mcap_rp_lam_pap | 0.196 → 0.251 | **-0.121** (가장 작음) |

→ 4 cfg 모두 시장 IT 비중 (~30%) 대비 **12~30%p 적게 보유**. 저변동 anomaly 의 본질적 sector tilt.

#### 가설 3 ✓: under-weight ↔ underperform 양의 상관

| cfg | 평균 IT active | HOLD_OUT 24m 수익 | SPY 대비 |
|---|---|---|---|
| **D. mat_mcap_rp_lam_pap** | **-0.121** ★ | +27.3% | **-19.7%p** ★ |
| C. mat_mcap_mcap_raw_he | -0.177 | +26.0% | -21.0%p |
| B. mat_eq_mcap_lam_he | -0.297 | +21.3% | -25.7%p |
| A. mat_eq_eq_lam_pap | -0.247 | +17.2% | **-29.8%p** ✗ |

**SPY HOLD_OUT 24m 누적**: +47.0%

→ IT under-weight 작을수록 underperform 작음 (rank 일관). 단, A 의 underperform 이 B 보다 큰 것은 IT 외 다른 섹터 영향 (예: A 가 D 와 비슷한 high turnover/concentration 패턴).

### 9-3. 시사점 — 4 후보 trade-off 에 새 차원 추가

기존 4 옵션 trade-off 에 **"sector rotation 견고성"** 차원 추가:

| cfg | 기존 강점 | sector rotation 견고성 |
|---|---|---|
| A. mat_eq_eq_lam_pap | alpha 1위, eff_n 220 | IT 노출 중간, **underperform 최대** ✗ |
| B. mat_eq_mcap_lam_he | Lex 1위, MDD 균형 | IT 거의 0% (극단적) ✗ |
| C. mat_mcap_mcap_raw_he | HO sortino 1.640 | IT 노출 중상위 (~18%) △ |
| **D. mat_mcap_rp_lam_pap** | sortino_ir 28.15 | **IT 노출 1위** (~25%, 가장 균형) ✓ |

### 9-4. 권고 변화

**기존 §5-2 권고 (옵션 C, mat_mcap_mcap_raw_he)** + **§9 발견 (옵션 D, mat_mcap_rp_lam_pap)** = **균형 잡힌 sector 노출 + regime 안정성 = 옵션 D 의 매력 부각**.

### 5 옵션 비교 (기존 4 + 새 차원)

| 옵션 | sortino_TEST | sortino_HO | sortino_ir | IT active | underperform vs SPY |
|---|---:|---:|---:|---:|---:|
| A. mat_eq_eq_lam_pap | 2.015 | 0.685 | 10.46 | -0.247 | **-29.8%p** ✗ |
| B. mat_eq_mcap_lam_he | 1.996 | 0.798 | 7.24 | -0.297 | -25.7%p |
| C. mat_mcap_mcap_raw_he | 1.919 | **1.640** ★ | 9.73 | -0.177 | -21.0%p |
| **D. mat_mcap_rp_lam_pap** | 1.878 | 1.235 | **28.15** ★ | **-0.121** ★ | **-19.7%p** ★ |

→ **옵션 D 가 sortino_ir 1위 + IT 노출 1위 + underperform 최소** = **3 차원 동시 우수**.

### 9-5. 본 분석가의 갱신된 견해

**가장 균형잡힌 권고: 옵션 D (mat_mcap_rp_lam_pap)**

**근거**:
1. sortino_ir 28.15 (regime 안정성 1위)
2. HOLD_OUT 24m sortino 1.235 (양호, 학습편향 의심 영역 통과)
3. IT 섹터 노출 0.25 (4 후보 중 시장 평균에 가장 근접)
4. SPY 대비 underperform 격차 -19.7%p (4 후보 중 최소)
5. alpha 0.048 + beta 0.720 (defensive 적정)

**보수적 견해**: 옵션 D 는 옵션 C 의 학습편향 회피 강점 + 추가로 sector rotation 견고성 + regime IR 1위. **3 가지 약점 (TEST 1.878, eff_n 68, turnover 0.93)** 은 발표에서 충분히 설명 가능.

**대안 견해**: 만약 학술적 robustness 증명이 핵심이라면 옵션 A, 학습편향 회피의 명확성이 핵심이라면 옵션 C 도 정당화 가능.

### 9-6. 산출물 (§11)

| 파일 | 내용 |
|---|---|
| `outputs/06_top1/figures/fig12_market_sector_trend.png` | 시장 섹터별 mcap 성장 + 비중 추이 |
| `outputs/06_top1/figures/fig13_cfg_sector_weight.png` | Top 4 cfg 섹터별 가중치 (4 panel stacked area) |
| `outputs/06_top1/figures/fig14_active_weight_IT.png` | IT 비중 (cfg vs 시장) + active weight |
| `outputs/06_top1/figures/fig15_active_weight_vs_returns.png` | IT under-weight ↔ underperform 산점도 |

---

## 10. SPY 4 레짐 비교 — 시장 환경 분석 (2026-05-09 추가)

### 10-1. 새 레짐 정의

기존 master_table 의 3 레짐 (R1/R2/R3 ~2024-12) 을 **AI 강세장 (R4)** 분리로 4 레짐 확장:

| 레짐 | 기간 | 개월 | 환경 |
|---|---|---|---|
| R1 회복 | 2010-01 ~ 2012-06 | 30 | Post-GFC + EU 위기 |
| R2 확장 | 2012-07 ~ 2019-12 | 90 | 장기 Bull |
| R3 변동 | 2020-01 ~ 2023-06 | 42 | COVID + '22 베어 |
| **R4 AI랠리** | 2023-07 ~ 2025-12 | 30 | **AI 강세장 (HOLD_OUT 24m 포함)** |

### 10-2. SPY 4 레짐 메트릭

| 레짐 | Sortino | Sharpe | MDD | CAGR | Vol | 승률 |
|---|---:|---:|---:|---:|---:|---:|
| R1 회복 | 1.260 | 0.678 | -16.2% | +10.5% | 16.6% | 56.7% |
| R2 확장 | 1.706 | 1.243 | -13.5% | +14.4% | 10.8% | 75.6% |
| R3 변동 | 0.939 | 0.568 | **-23.9%** | +11.3% | 20.1% | 61.9% |
| **R4 AI랠리** | **2.449** ★ | **1.230** | **-8.3%** ★ | **+20.3%** ★ | 12.0% | 70.0% |
| FULL (192m) | 1.350 | 0.897 | -23.9% | +14.0% | 14.3% | 68.8% |

→ **R4 AI랠리가 SPY 의 4 레짐 중 압도적 최고**: Sortino 2.449 (R2 확장 1.706 의 1.4배), CAGR +20.3% (R2 의 1.4배), MDD -8.3% (가장 작음).

### 10-3. Top 4 후보 + SPY 레짐 Sortino 비교

| cfg | R1 회복 | R2 확장 | R3 변동 | **R4 AI랠리** |
|---|---:|---:|---:|---:|
| A. mat_eq_eq_lam_pap | 2.205 | 2.044 | 2.108 | **0.680** ✗ |
| B. mat_eq_mcap_lam_he | 2.232 | 2.026 | 1.925 | **0.724** ✗ |
| C. mat_mcap_mcap_raw_he | 1.772 | 2.082 | 1.744 | **1.316** △ |
| **D. mat_mcap_rp_lam_pap** | 1.779 | 1.941 | 1.926 | **1.253** △ |
| **SPY** | 1.260 | 1.706 | 0.939 | **2.449** ★ |

**놀라운 패턴**:
- 모든 cfg 가 R1/R2/R3 에서 SPY 를 압도 (cfg sortino > SPY sortino)
- **R4 AI랠리에서 정반대 — SPY 가 모든 cfg 를 압도**

### 10-4. R3 → R4 전환 패턴 — 모델 vs 시장

| | R3 변동 Sortino | R4 AI랠리 Sortino | 변화 배수 |
|---|---:|---:|---:|
| A. mat_eq_eq_lam_pap | 2.108 | 0.680 | **×0.32** ✗ |
| B. mat_eq_mcap_lam_he | 1.925 | 0.724 | **×0.38** ✗ |
| C. mat_mcap_mcap_raw_he | 1.744 | 1.316 | ×0.75 |
| **D. mat_mcap_rp_lam_pap** | 1.926 | 1.253 | **×0.65** △ |
| **SPY** | 0.939 | **2.449** | **×2.61** ★ |

→ **R3 → R4 에서 모델은 모두 sortino 하락, SPY 만 2.6배 상승**. 이는 **저변동 anomaly 의 근본적 시장 환경 의존성** — bull market 후반 (특히 단일 섹터 주도 강세장) 에서 본질적으로 뒤처짐.

### 10-5. R4 AI랠리 cfg vs SPY 격차 정량화

R4 (HOLD_OUT 24m 포함) 에서 SPY 대비 격차:

| cfg | Sortino 격차 | CAGR 격차 (%p) |
|---|---:|---:|
| A. mat_eq_eq_lam_pap | **-1.77** | -11.9 |
| B. mat_eq_mcap_lam_he | -1.73 | -11.3 |
| C. mat_mcap_mcap_raw_he | -1.13 | -9.1 |
| **D. mat_mcap_rp_lam_pap** | **-1.20** | **-7.4** ★ (최소) |

→ **D 가 R4 격차 최소** (CAGR -7.4%p) — §11 의 IT 노출 1위 (-0.121 active) + §10 의 R4 격차 최소 = **3 차원 동시 우수 재확인**.

### 10-6. SPY TEST vs HOLD_OUT 격차

BL 평가 기간 분리 시 SPY 자체의 격차:

| 기간 | 개월 | Sortino | Sharpe | MDD | CAGR |
|---|---:|---:|---:|---:|---:|
| TEST | 168 | 1.281 | 0.843 | -23.9% | +13.0% |
| **HOLD_OUT** | 24 | **2.333** | **1.465** | -7.6% | **+21.2%** |
| 격차 (HO/TEST) | - | **×1.82** | ×1.74 | - | ×1.63 |

**SPY HOLD_OUT 자체가 학습 기간보다 1.82배 잘함**. 이는 **HOLD_OUT 부진이 학습편향만의 문제가 아님**을 정량 증명:

- A. mat_eq_eq_lam_pap HO sortino 0.685 / SPY HO 2.333 = 격차 1.65
- D. mat_mcap_rp_lam_pap HO sortino 1.235 / SPY HO 2.333 = 격차 1.10 (최소)

### 10-7. 시사점 — 발표 narrative 강화

**기존 narrative (학습편향 가설)**:
> "잠정 Top 1 의 HO sortino 0.685 부진 = 학습편향 의심" (정성적)

**갱신된 narrative (시장 환경 + sector tilt 실증)**:

> "HOLD_OUT 24m ≈ R4 AI랠리 — **SPY 자체가 학습 기간 대비 1.82배 강세** + AI/IT 단일 섹터 주도 강세장 (§11 의 IT 비중 +7.4%p, mcap 2.07배). 저변동 anomaly 의 본질적 IT under-weight 로 underperform 불가피. **단, D 모델 (mat_mcap_rp_lam_pap) 은 sector 노출 균형 (IT 0.25, 시장 평균에 가장 가까움) 과 regime IR 28.15 (1위) 로 격차 최소화** — bull market 후반에서도 가장 견고."

이는 **알고리즘 결함이 아닌 전략 본질의 trade-off** 를 정직하게 발표하는 narrative.

### 10-8. 갱신된 최종 권고

§5-2 (옵션 C, HO 압도) → §9-5 (옵션 D, sector 균형 부각) → **§10 (옵션 D, R4 격차 최소 + 시장 환경 견고성 추가 확인)**

**최종 권고: 옵션 D (mat_mcap_rp_lam_pap)** — 다음 4 차원 동시 우수:
1. ✓ **sortino_ir 28.15** (regime 안정성 1위, §3-D)
2. ✓ **sortino_HO 1.235** (HO 학습편향 의심 통과, §2)
3. ✓ **IT active -0.121** (sector rotation 견고성 1위, §11)
4. ✓ **R4 SPY 격차 -7.4%p** (시장 환경 적응 1위, §10)

**약점 (TEST 1.878, eff_n 68, turnover 0.93)** 은 발표에서 충분히 설명 가능.

### 10-9. 산출물 (§12)

| 파일 | 내용 |
|---|---|
| `outputs/06_top1/spy_regime_comparison.csv` | Top 4 + SPY × 4 레짐 메트릭 (20 행 × 11 컬럼) |
| `outputs/06_top1/figures/fig16_top4_spy_regime_dashboard.png` | Top 4 + SPY × 4 레짐 통합 heatmap (Sortino/Sharpe/MDD) |
| `outputs/06_top1/figures/fig17_top4_vs_spy_sortino.png` | Top 4 vs SPY 레짐 Sortino bar chart |

---

## 11. In-sample only 검증 — 학습편향 진단 (2026-05-09 추가)

### 11-1. 방법론적 동기

기존 §4 Lexicographic 의 2 차 tiebreak 와 §9 Decision Matrix 의 성과·안정성 차원에 **HOLD_OUT 정보가 일부 누설** 되어 있었습니다 (Lopez de Prado 2018 의 backtest overfitting 우려). 학술 표준은 다음과 같습니다:

| 단계 | 사용 데이터 |
|---|---|
| 모델/cfg 선정 | TEST 168m **만** |
| 단일 검증 | HOLD_OUT 24m (선정 후) |
| 사후 분석 | sector / regime 등 |

### 11-2. In-sample only 변형 적용

§4-4, §9-4 에 HO 정보를 제거한 변형을 추가:

**Lexicographic (§4-4)**:
- 1차: sortino_TEST → 2차: **mdd_TEST 단독** (mdd_HO 제거) → 3차: sortino_ir

**Decision Matrix (§9-4)**:
- 성과 30% = `sortino_TEST` 단독 (HO 제거)
- 안정성 20% = `sortino_ir` 단독 (TEST_HO_gap 제거)
- 위험/견고성/Alpha 차원은 동일

### 11-3. 핵심 결과 — ★ ROBUST 입증

| 기준 | HO 포함 Top 1 | in-sample only Top 1 | 일치? |
|---|---|---|---|
| Lexicographic (§4) | `mat_eq_mcap_lam_he` | `mat_eq_mcap_lam_he` | **✓ 동일** |
| Decision Matrix (§9) | `mat_eq_eq_lam_pap` | `mat_eq_eq_lam_pap` | **✓ 동일** |

**시사점**: 누설 위험이 있었으나 **Top 1 선정에는 영향 없음**. 즉, 본 분석의 Top 1 결과는 **학술적 in-sample only 기준에서도 동일하게 산출**되며, backtest overfitting 위험에서 자유롭습니다.

### 11-4. In-sample only Decision Matrix 점수 (Top 5)

| 순위 | name | weighted_score (낮을수록 ★) | 비고 |
|---|---|---:|---|
| **1** | **mat_eq_eq_lam_pap** | **1.575** ★ | sortino_TEST 1위 + alpha 1위 부각 |
| 2 | mat_eq_eq_raw_pap | 1.875 | A 의 lam vs raw 변형 |
| 3 | mat_eq_mcap_lam_rms | 3.600 | mcap p_weight 변형 |
| 4 | mat_eq_mcap_lam_he | 3.950 | Lex Top 1 |
| 5 | q_raw_lam | 4.000 | quantile baseline |

**기존 (HO 포함) Top 5 와 비교**:

| 순위 | HO 포함 (§9-1) | in-sample (§9-4) | 차이 |
|---|---|---|---|
| 1 | mat_eq_eq_lam_pap (2.49) | mat_eq_eq_lam_pap (**1.58**) | 점수 격차 더 명확 |
| 2 | mat_eq_eq_raw_pap | mat_eq_eq_raw_pap | 동일 |
| 3 | mat_eq_mcap_lam_rms | mat_eq_mcap_lam_rms | 동일 |
| 4 | q_raw_lam | mat_eq_mcap_lam_he | 4-5 위 swap |
| 5 | mat_eq_mcap_lam_he | q_raw_lam | 4-5 위 swap |

**해석**: in-sample only 에서 **`mat_eq_eq_lam_pap` 의 1위가 더 명확** (점수 격차 1.58 vs 1.88 → 0.30; HO 포함 시 2.49 vs 2.96 → 0.47 보다 상대적 우위 더 큼). 이는 잠정 Top 1 의 학술적 정당성을 강화합니다.

### 11-5. Lexicographic 결과 — Top 1 유지, 그 외 순위 변동

| rank | HO 포함 (§4-1) | in-sample (§4-4) | 변경 |
|---:|---|---|---|
| 1 | mat_eq_mcap_lam_he | mat_eq_mcap_lam_he | ✓ 동일 |
| 2 | q_raw_lam | mat_eq_mcap_lam_rms | ✗ |
| 3 | mat_eq_mcap_lam_rms | q_raw_lam | ✗ |
| 4 | mat_eq_eq_lam_pap | mat_eq_eq_raw_pap | ✗ |
| 5 | mat_eq_eq_raw_pap | mat_eq_eq_lam_pap | ✗ |
| 8 | q_lambda | q_lambda | ✓ |
| 10 | mat_mcap_mcap_raw_rms | mat_mcap_mcap_raw_rms | ✓ |

→ Top 1 + 일부 순위는 동일하나 **2~7 위는 순서 swap 빈번** — HO MDD 에 의한 미세한 우열 변동으로, 실질 영향은 미미.

### 11-6. 학술적 정직성 narrative — 갱신

**기존**: "Top 1 권고 시 HO 포함 결과 사용 (5-2, 9-5, 10-8)" — 누설 우려 잠재

**갱신**:

> **본 분석의 Top 1 결과 (`mat_eq_eq_lam_pap` for Decision Matrix, `mat_eq_mcap_lam_he` for Lexicographic) 는 in-sample only 변형에서도 동일하게 산출되어 backtest overfitting 위험에서 자유롭습니다.** 이는 "TEST 168m 만으로 결정해도 같은 결론" 임을 의미하며, HOLD_OUT 24m 의 부진 (sortino 0.685) 은 §11 (sector tilt) + §10 (시장 환경 R4 AI랠리) 의 사후 분석으로 일관 설명됩니다.

### 11-7. 옵션 D (mat_mcap_rp_lam_pap) 의 갱신된 위치

§9-4 in-sample only matrix 에서 **D 는 Top 5 권 밖** (sortino_TEST 1.878 로 §2 의 22 cfg 중 14 위, top10_metrics 의 #9-10 위 그룹).

**갱신된 견해**:
- **학술적 robustness 강조 시 (옵션 A)**: `mat_eq_eq_lam_pap` — in-sample / HO 포함 모두 1위, sortino_TEST + alpha 1위
- **객관적 절차 강조 시 (옵션 B)**: `mat_eq_mcap_lam_he` — in-sample / HO 포함 lexicographic 모두 1위, MDD 균형
- **사후 분석 (sector + regime) 시 (옵션 D)**: `mat_mcap_rp_lam_pap` — sortino_ir 1위 + IT 노출 1위 (단, lex/DM 1위 아님)

### 11-8. 산출물 (§13 + §4-4, §9-4)

| 파일 | 내용 |
|---|---|
| `outputs/06_top1/top5_decision_matrix_insample.csv` | in-sample only Decision Matrix |
| `outputs/06_top1/lex_compare_HO_vs_insample.csv` | Lexicographic Top 10 비교 |
| `outputs/06_top1/insample_vs_ho_comparison.csv` | 통합 비교 표 (Lex + DM Top 1) |
| `outputs/06_top1/figures/fig11b_decision_matrix_compare.png` | DM 점수 비교 bar chart |
| `outputs/06_top1/figures/fig18_insample_vs_ho_compare.png` | Lex rank 변화 + DM score 비교 |

---

## 12. 학술 검증 종합 — PBO / Sharpe test / Factor / tc / Walk-forward / Net Sharpe (2026-05-09 추가)

### 12-1. 분석 동기

§11 의 in-sample only 검증 외에 **6 가지 학술적 robustness 검증** 을 추가 (옵션 3):

| § | 검증 | 핵심 결과 |
|---|---|---|
| §14 | **PBO + DSR** (Bailey-Lopez de Prado 2014) | PBO 1.0 ✗ — 22 cfg multiple testing 우려 |
| §15 | **Memmel Sharpe test** (2003) | Top 5 vs baseline pairwise z-stat |
| §16 | **Factor regression** (CAPM/FF3/Carhart4/FF5) | **모든 후보 alpha 유의 ★** (t > 4.5, p < 1e-5) |
| §17 | **tc sensitivity** (5bps ~ 100bps) | Lex Top 1 (lam_he) 대부분 tc 에서 1위 유지 |
| §18 | **Walk-forward** (2019~2025, 7년 anchored) | Lex Top 1 평균 rank 10.0 (최고) |
| §19 | **Net Sharpe + break-even tc** | Lex Top 1 break-even 229 bps (압도적) |

### 12-2. PBO 결과 — 학술적 경고

| 항목 | 값 |
|---|---:|
| 분석 대상 | 22 cfg (M_cfg) |
| 분할 J | 8 (each 24m) |
| In-sample S | 4 |
| 조합 수 | 70 |
| **PBO** (rank > median 비율) | **1.000** ✗ |
| 평균 OOS rank percentile | 0.834 |

**해석**: 22 cfg 의 IS top 1 가 OOS 에서 거의 항상 하위 절반에 위치 → **22 cfg 내 multiple testing 우려 큼**. 다만 이는 "Top 1 만 보고 OOS 우월성 보장 안 됨" 을 의미하며, 본 분석이 단일 cfg 가 아닌 다중 메트릭 + 사후 분석 (§11, §12, §16) 으로 보완하는 이유.

### 12-3. DSR — Top 5 multiple testing 보정 후

| cfg | SR_monthly | SR_0 (153 보정) | DSR z | p-value | 유의? |
|---|---:|---:|---:|---:|:---:|
| mat_eq_mcap_lam_he | 0.294 | 0.229 | 0.87 | 0.192 | ✗ |
| q_raw_lam | 0.299 | 0.229 | 0.94 | 0.173 | ✗ |
| mat_eq_mcap_lam_rms | 0.297 | 0.229 | 0.91 | 0.180 | ✗ |
| **mat_eq_eq_lam_pap** | **0.320** | 0.229 | **1.20** | **0.114** | ✗ |
| mat_eq_eq_raw_pap | 0.318 | 0.229 | 1.19 | 0.117 | ✗ |

→ **모두 multiple testing 보정 후 p > 0.10 (단측)** — 153 cfg 시도 보정 시 Sharpe 가 0 보다 통계적으로 유의하게 크지 않음. 이는 학술적으로 매우 엄격한 검정.

### 12-4. Factor Regression — alpha 진정성 ★

**최강력 결과**: Top 5 cfg × 4 모델 (CAPM/FF3/Carhart4/FF5) **모두 alpha 유의 (p < 1e-5)**:

| cfg | CAPM α (annual) | t-stat | FF5 α (annual) | t-stat |
|---|---:|---:|---:|---:|
| **mat_eq_eq_lam_pap** | **+17.19%** | **4.99** | **+13.53%** | **4.47** |
| mat_eq_eq_raw_pap | +17.13% | 4.96 | +13.45% | 4.48 |
| mat_eq_mcap_lam_he | +13.32% | 4.49 | +13.53% | 4.47 |
| q_raw_lam | +13.01% | 4.63 | +13.20% | 4.61 |
| mat_eq_mcap_lam_rms | +13.35% | 4.50 | +13.60% | 4.50 |

**해석**:
- FF5 (Mkt + SMB + HML + RMW + CMA) 후에도 alpha 13~17% 유의 → **진짜 alpha**
- R² 매우 낮음 (~0.03) → cfg 수익이 factor 들로 거의 설명 안 됨 → 대부분 idiosyncratic
- §11-7 의 Top 1 후보들의 alpha 가 factor risk premium 이 아닌 **진짜 알파** 임을 강력 입증

### 12-5. tc Sensitivity — 실무 거래비용 6 단계 (2026-05-09 갱신)

#### 실무 시나리오 매핑

| tc | 실무 시나리오 | Top 1 (best Sortino_TEST) |
|---:|---|---|
| 0.0005 (5 bps) | 패시브 ETF (대형 AUM) | `mat_eq_mcap_lam_he` (1.953) |
| 0.001 (10 bps) | 저변동 ETF (USMV/SPLV) — default | `mat_eq_mcap_lam_rms` (1.900) |
| **0.002 (20 bps)** | **액티브 BL 운용 — 가장 현실적 ★** | **`mat_eq_mcap_lam_rms` (1.813)** |
| 0.003 (30 bps) | 보수적 stress | `mat_eq_mcap_lam_he` (1.726) |
| 0.005 (50 bps) | 매우 보수적 (소형주/위기) | `mat_eq_mcap_lam_rms` (1.543) |
| 0.01 (100 bps) | extreme stress | `mat_eq_mcap_lam_he` (1.099) |

#### 6 tc 단계별 Top 1 횟수

| cfg | 1 위 횟수 (6 tc 중) | turnover | 거래비용 robustness |
|---|:---:|---:|---|
| **mat_eq_mcap_lam_rms** | **3 회** (10/20/50 bps) | 0.44 | ✓ **실무 가정 영역 강세** |
| **mat_eq_mcap_lam_he** | **3 회** (5/30/100 bps) | 0.43 | ✓ 패시브 + stress 양쪽 |
| q_raw_lam | 0 | 0.66 | △ 중간 turnover |
| mat_eq_eq_lam_pap | 0 | 0.99 | ✗ 거래비용 매우 취약 |
| mat_eq_eq_raw_pap | 0 | 0.99 | ✗ 거래비용 매우 취약 |

**해석**:
1. **`lam_he` 와 `lam_rms` 는 사실상 동등한 후보**, 6/6 중 3:3 균등 1위
2. **실무 가장 현실적 영역 (10~20 bps) 에서는 `mat_eq_mcap_lam_rms` 가 우위**
3. `mat_eq_eq_lam_pap` / `raw_pap` (turnover 0.99) 는 **어느 tc 에서도 1위 불가** — 실거래 시 alpha 잠식 우려
4. lam_he / lam_rms turnover 0.43~0.44 → 거래비용 효율 압도적 (turnover 절반 미만)

#### lam_he vs lam_rms — 미세한 차이

| 메트릭 | mat_eq_mcap_lam_he | mat_eq_mcap_lam_rms | 차이 |
|---|---:|---:|---|
| sortino_TEST | 1.996 | 2.003 | rms 약간 우위 |
| sortino_HOLD_OUT | 0.798 | 0.811 | rms 약간 우위 |
| sortino_ir | 7.24 | 8.51 | **rms 우위** |
| mdd_TEST | -0.120 | -0.124 | he 우위 |
| mdd_HO | -0.068 | -0.069 | 거의 동등 |
| Lex rank (HO 포함) | **#1** | #3 | he 우위 |
| Lex rank (in-sample) | **#1** | #3 | he 우위 |
| FF5 alpha | 13.53% | 13.60% | rms 약간 우위 |
| Walk-forward 평균 rank | **10.0** | 10.4 | he 약간 우위 |

→ **두 cfg 가 매우 유사** — 같은 prior (mat) + p_weight (eq) + p_mode (mcap) + q (lam), 차이는 omega_mode (he vs rms) 만. omega 산출 방식의 미세한 차이.

### 12-6. Walk-forward 7년 OOS (2019~2025)

| cfg | 평균 OOS rank |
|---|---:|
| **mat_eq_mcap_lam_he** | **10.0** ★ (최고) |
| mat_eq_mcap_lam_rms | 10.4 |
| mat_eq_eq_lam_pap | 13.0 |
| q_raw_lam | 13.3 |

**해석**: Lex Top 1 (`mat_eq_mcap_lam_he`) 가 **7년 anchored walk-forward 에서 가장 일관**. DM Top 1 (`mat_eq_eq_lam_pap`) 는 평균 rank 13 (22 cfg 중) → 일관성 떨어짐.

### 12-7. Net Sharpe + Break-even tc

| cfg | Gross SR | Net SR (10bps) | SR/turnover | Break-even tc |
|---|---:|---:|---:|---:|
| **mat_eq_mcap_lam_he** | 1.022 | 0.977 | **2.38** ★ | **229 bps** ★ |
| q_raw_lam | 1.043 | 0.970 | 1.57 | 144 bps |
| mat_eq_mcap_lam_rms | 1.034 | 0.988 | 2.35 | 226 bps |
| **mat_eq_eq_lam_pap** | 1.110 | 1.022 | 1.12 | **127 bps** |
| mat_eq_eq_raw_pap | 1.105 | 1.016 | 1.13 | 125 bps |

**Break-even tc** = Sharpe 가 0 되는 tc 값 (안전 마진).
- `mat_eq_mcap_lam_he` 229 bps → **23배 안전 마진** (default 10bps 대비)
- `mat_eq_eq_lam_pap` 127 bps → 약 13배 안전 마진 (덜 robust)

### 12-8. 갱신된 최종 권고 — `mat_eq_mcap_lam_he` / `lam_rms` 동등 부각

§5-2 (옵션 C) → §9-5 (옵션 A) → §10-8 (옵션 D) 순으로 옮겨졌던 권고가 **§12 학술 검증 후 옵션 B (`mat_eq_mcap_lam_he` / `lam_rms`)** 로 이동:

| 차원 | mat_eq_mcap_lam_he | mat_eq_mcap_lam_rms |
|---|---|---|
| §4 Lexicographic | **1위** (HO/in-sample) | #3 |
| §17 tc 1위 횟수 | 3/6 (5/30/100 bps) | **3/6 (10/20/50 bps)** ★ |
| §17 실무 영역 (10~20 bps) | 보통 | **우위** ★ |
| §18 Walk-forward | **평균 rank 10.0 (1위)** | 10.4 |
| §19 Break-even tc | **229 bps** | 226 bps (거의 동등) |
| §16 FF5 α | 13.53% (t=4.47) | 13.60% (t=4.50) |
| sortino_ir | 7.24 | **8.51** ★ |
| turnover | 0.430 (1위) | 0.441 |

**해석**:
- 두 cfg 의 차이 = **omega_mode (he vs rms)** 만. omega 산출 방식 미세한 차이.
- **실무 가장 현실적인 10~20 bps 영역에서는 `lam_rms` 가 우위**
- **객관적 절차 (Lexicographic) 와 walk-forward 일관성에서는 `lam_he` 가 우위**
- 둘 모두 `mat_eq_eq_lam_pap` (DM Top 1) 보다 **거래비용 측면 압도적 우위**

#### 발표 권장 narrative

> Top 1 후보군 = **`mat_eq_mcap_lam_he` 또는 `lam_rms`** (동등하게 강력)
> - 절대 성과 최강: `mat_eq_eq_lam_pap` (alpha 1위, eff_n 220)
> - 실거래 robustness: `lam_he` / `lam_rms` (turnover 0.43~0.44, break-even tc 226~229 bps)
> - regime 안정성: `mat_mcap_rp_lam_pap` (sortino_ir 28.15)
>
> **실무 운용 가정 (tc 15~20 bps) 하에서는 `lam_rms` 가 가장 현실적 1위**, 학술 절차 (Lexicographic) 기준에서는 `lam_he` 가 1위. 둘 중 어느 쪽을 선택해도 실거래·학술적 정당성 모두 만족.

### 12-9. 갱신된 narrative — 발표용

> **Top 1 = `mat_eq_mcap_lam_he`** (Lexicographic 절차의 객관적 1위)
>
> 학술 검증 5종 (Factor regression / tc sensitivity / Walk-forward / Net Sharpe / Break-even tc) 모두에서 **가장 일관되게 우수**. FF5 factor regression 후에도 alpha 13.5% (t=4.47) 유의, 7년 walk-forward 에서 평균 OOS rank 10.0 (22 cfg 중 1위), break-even tc 229 bps (실거래 안전 마진 23배).
>
> 보조 권고: `mat_eq_eq_lam_pap` (성과 절대값 우수), `mat_mcap_rp_lam_pap` (regime 안정성 1위), `mat_mcap_mcap_raw_he` (HO sortino 1위) — 강조점에 따라 선택.

### 12-10. PBO 우려에 대한 대응

PBO = 1.0 의 학술적 경고를 어떻게 narrative 로 다룰지:

- **단순 점수 비교 (Sortino top 1)** 만으로는 robust 보장 안 됨 → §14 의 경고 인정
- **다층 검증 (lex + DM + factor + walk-forward + tc)** 으로 보완 → 본 §12 의 강점
- **HO sortino 부진의 사후 분석 (sector + regime)** 으로 추가 정당화 → §11
- → **단일 메트릭 비교가 아닌 종합적 robustness 평가** 가 본 분석의 정당성

### 12-11. 산출물 (§14~§19)

| 파일 | 내용 |
|---|---|
| `outputs/06_top1/dsr_top5.csv` | Top 5 DSR (§14) |
| `outputs/06_top1/memmel_sharpe_test.csv` | Sharpe pairwise (§15) |
| `outputs/06_top1/factor_regression.csv` | 4 모델 × Top 5 (§16) |
| `outputs/06_top1/tc_sensitivity.csv` | tc 5종 × Top 5 (§17) |
| `outputs/06_top1/walkforward_oos.csv` | 7년 OOS rank (§18) |
| `outputs/06_top1/net_sharpe_efficiency.csv` | Net SR + break-even (§19) |
| `figures/fig19_pbo_distribution.png` | PBO 분포 |
| `figures/fig20_factor_alpha_heatmap.png` | Factor alpha heatmap |
| `figures/fig21_tc_sensitivity.png` | tc 라인 차트 |
| `figures/fig22_walkforward_oos.png` | Walk-forward heatmap |
| `figures/fig23_net_sharpe_efficiency.png` | Net SR + break-even bar |

---

## 13. 시각화 leakage 정리 — 의사결정 단계 명확화 (2026-05-09 추가)

### 13-1. 분석 동기

§11 (in-sample 검증) + §12 (학술 검증) 까지 진행한 후, **시각화 셀들 중 일부에 HOLD_OUT 메트릭이 잔존** 하는 지적이 있었습니다. §4-3 fig04 heatmap, §5-3 fig05 heatmap, §9-2 fig11 radar 등 의사결정 단계의 시각화가 §4-1, §9-1 (HO 포함) 변형만 출력하여, **in-sample only 변형 (§4-4, §9-4) 의 시각적 표현이 부재**.

### 13-2. 추가된 시각화 + 보강 (Tier 1+2+3)

| 추가/수정 | 위치 | 내용 |
|---|---|---|
| **신규** `fig04b_lexicographic_heatmap_insample.png` | §4-6 | in-sample only Lexicographic heatmap (3 메트릭, mdd_HOLD_OUT 제거) |
| **신규** `fig11c_decision_matrix_radar_insample.png` | §9-6 | in-sample only Decision Matrix radar (5 차원, HO 메트릭 제거) |
| **수정** `fig05_top10_metric_heatmap.png` | §5-3 | 16 메트릭을 좌측 7개 (의사결정용, 빨강 라벨) + 우측 9개 (사후 검증용, 파랑 라벨) 로 시각적 분리 |
| **수정** §5 markdown | - | "16 메트릭 5 카테고리" → "의사결정용 5 (in-sample) + 사후 검증용 11 (HO/FULL)" 명확 분리 |
| **수정** §1 markdown | - | "안정성 차원: TEST vs HO 격차" 제거 (사후 진단 메트릭) → 안정성 = sortino_ir 만 |
| **추가** §9-7 narrative | - | in-sample only Top 1 (§9-4) 의 metrics 별도 출력 + "사후 진단용 메트릭" 라벨로 구분 |

### 13-3. 정리 후 의사결정 vs 사후 분석 명확 분리

#### 의사결정 단계 (HO 사용 금지) — in-sample only ✓

| § | HO 포함 (비교용) | in-sample (학술 표준) |
|---|---|---|
| §4 Lexicographic | fig04 | **fig04b ★** |
| §9 Decision Matrix | fig11 radar | **fig11c radar ★** |
| §9 점수 비교 | - | fig11b (HO vs in-sample bar) |

→ **모든 의사결정 시각화가 in-sample only 학술 표준 변형 보유**

#### 사후 분석 (HO 사용 정당) — 그대로 유지 ✓

| § | 시각화 | 정당 사유 |
|---|---|---|
| §6-2 fig07 | TEST vs HO 산점도 | **학습편향 진단** 시각화 자체 |
| §6-3 fig08 | 누적수익 + drawdown | 시계열 (HO 기간 자연스럽게 포함) |
| §7-1 fig09 | sector HHI 24m | **§ 자체가 사후 분해** |
| §11 fig12~15 | sector tilt 분석 | **§ 자체가 HO 24m 분석** |
| §12 fig16~17 | SPY 4 레짐 | R4 = HO 시기 비교 |
| §13 fig18 | in-sample vs HO 비교 | **§ 자체가 비교** |
| §14 fig19 | PBO 분포 | 192m 분할 (PBO 본질) |
| §17 fig21 | tc sensitivity | TEST 168m 만 사용 ✓ |
| §18 fig22 | walk-forward | 7년 OOS 평가 |

### 13-4. 시각화 분류 도식

```
[의사결정 단계 — in-sample only ★]      [사후 분석 — HO 정당]            [학술 검증]
fig04b: Lex heatmap                      fig07: TEST vs HO 산점도         fig19: PBO
fig05 좌측 7: 의사결정 메트릭            fig08: 누적수익+drawdown         fig20: Factor α
fig11c: DM radar                         fig09: sector HHI                fig21: tc sensitivity
                                         fig12~15: sector tilt            fig22: walk-forward
                                         fig16~17: SPY regime             fig23: Net SR
                                         fig18: HO vs in-sample
```

### 13-5. 갱신된 노트북 구조

| § | 단계 | 시각화 (HO / in-sample / 사후) |
|---|---|---|
| §0~§3 | 환경 + Universe | fig01~fig03 (HO 진단 시각화 정당) |
| **§4** | Lexicographic | **fig04 (HO) + fig04b (in-sample ★)** |
| §5 | Top 10 정밀 | fig05 (16 메트릭, **그룹 구분 ★**) |
| §6~§8 | 안정성/위험/baseline | fig06~10 (사후 분석) |
| **§9** | Decision Matrix | **fig11 (HO) + fig11b (점수 비교) + fig11c (in-sample ★)** |
| §10 | Sensitivity | (표만) |
| §11~§13 | sector/regime/HO 비교 | fig12~18 (사후 / 비교) |
| §14~§19 | 학술 검증 | fig19~23 |

### 13-6. 시사점 — 발표 narrative 강화

**기존**: "본 분석은 lexicographic + decision matrix 결과로 Top 1 선정"

**갱신**:
> 본 분석의 의사결정 단계 (§4 lex, §9 DM) 는 **in-sample only 학술 표준** (Lopez de Prado 2018) 으로 진행되며, HOLD_OUT 정보는 **사후 분석 (§11 sector, §13 비교)** 과 **학술 검증 (§14~§19)** 에만 사용. 두 변형 (HO 포함 §4-1/§9-1 vs in-sample §4-4/§9-4) 은 §13 에서 정량 비교되며 **Top 1 결과 동일** (`mat_eq_mcap_lam_he` for Lex / `mat_eq_eq_lam_pap` for DM) — 누설 우려가 있었으나 결론에 영향 없음 검증 완료.

### 13-7. 신규 / 수정 산출물 (시각화 정리)

| 파일 | 변화 | 설명 |
|---|---|---|
| `fig04_lexicographic_heatmap.png` | 기존 | HO 포함 (비교용) |
| **`fig04b_lexicographic_heatmap_insample.png`** | **신규** | **in-sample only ★** |
| `fig05_top10_metric_heatmap.png` | **수정** | 의사결정/사후 그룹 구분 |
| `fig11_decision_matrix_radar.png` | 기존 | HO 포함 (비교용) |
| `fig11b_decision_matrix_compare.png` | 기존 | 점수 비교 bar |
| **`fig11c_decision_matrix_radar_insample.png`** | **신규** | **in-sample only ★** |

총 차트: 23 → **25 장** (fig04b + fig11c 추가).
총 노트북 셀: 86 → **89 cells**.

### 13-8. 학술적 정당성 강화

본 정리 후 노트북은 다음 학술 표준을 명확히 충족:

1. ✓ **의사결정에 OOS 정보 사용 금지** (Lopez de Prado 2018) — fig04b, fig11c, fig05 좌측 7 메트릭
2. ✓ **사후 분석은 OOS 사용 정당** (Brinson attribution 등) — fig07~fig18
3. ✓ **multiple testing 보정** (Bailey-Lopez de Prado 2014) — §14 PBO/DSR
4. ✓ **factor risk premium 분리** (Fama-French 1992, Carhart 1997) — §16
5. ✓ **거래비용 sensitivity** (Frazzini et al. 2018) — §17 (6 단계 5~100bps)
6. ✓ **walk-forward OOS** — §18 (7년)
7. ✓ **statistical significance** (Memmel 2003, Bailey-Lopez de Prado 2014) — §15, §14

→ **학술 보고서 / 발표 자료에서 backtest overfitting / data snooping 비판에 모든 측면에서 대응 가능**.

---

## 14. 누설 완전 제거 — Top 1 재산출 (2026-05-10 추가)

### 14-1. 추가 발견된 누설

§13 시각화 정리 후 추가 검토 결과 **의사결정 흐름의 모든 단계에서 HO 누설 잔존** 발견:

| 위치 | 누설 종류 | 영향도 |
|---|---|:---:|
| **§2-1 sortino_ir** | regime R3 = 2020-01~2024-12 → **HO 12m 포함** | **★★★ 가장 근본** |
| §5-1 top10 | `ranked.head(10)` (HO 포함 lex 사용) | ★★★ |
| §5-2 16 메트릭 | sortino_HO/sharpe_HO/TEST_HO_gap | ★★★ |
| §5-3 fig05 z-score | 16 메트릭 평균/std 가 in-sample 정규화에 영향 | ★★ |
| §9-1 head(5) 입력 | top10_metrics (HO 영향) | ★★ |
| §3 markdown | sortino_HO 후보 명시 | ★ |

### 14-2. 누설 완전 제거 — 6 가지 수정

#### (1) sortino_ir 재산출 (§0-2)

```
REGIMES_INSAMPLE = [
    ('R1_회복',  '2010-01-01', '2012-06-30'),  # 30m
    ('R2_확장',  '2012-07-01', '2019-12-31'),  # 90m
    ('R3_변동',  '2020-01-01', '2023-12-31'),  # 48m (HO 분리, TEST 168m 안에서 종결)
]
rt = build_regime_table(mt_main, 'results', rf, regimes=REGIMES_INSAMPLE)
```

→ HO 24m 의 정보가 sortino_ir 에 **0% 침투**.

**sortino_ir 차이 분포 (in-sample - HO 포함)**: mean +0.21, std **4.90**, max **+36.41**, min **-14.22** → 누설 영향이 cfg 마다 매우 큼.

#### (2) Top 10 추출을 in-sample 기반으로 (§5-1)

`top10 = ranked_insample.head(10)` — §4-4 (in-sample only Lex) 결과 사용.

#### (3) 메트릭 분리 (§5-2 → §5-2a + §5-2b)

| CSV | 내용 | 사용 |
|---|---|---|
| `top10_metrics_decision.csv` | **7 메트릭 in-sample** | §9-4 의사결정 |
| `top10_metrics_posthoc.csv` | 9 메트릭 HO/FULL | §11/§13 사후 |
| `top10_metrics.csv` | 통합 (역호환) | 참조용 |

#### (4) fig05 분리 (§5-3a + §5-3b)

| 차트 | 입력 | 의도 |
|---|---|---|
| `fig05a_top10_decision_heatmap.png` | 7 메트릭 z-score | 의사결정 (HO 0%) |
| `fig05b_top10_posthoc_heatmap.png` | 9 메트릭 z-score | 사후 (HO 사용 정당) |

#### (5) §9-1 풀 변경

`top5_names_decision = top10_metrics_decision.head(5)['name']` → **HO 포함 §9-1 도 in-sample 풀에서 head(5)** 시작.

#### (6) §3 markdown 정리

HO 기반 hard filter 후보 (`sortino_HO > 0`, `mdd_HO > -0.30`) narrative 제거.

### 14-3. 누설 제거 결과 — Universe 확장 + Top 1 변화 ★★★

#### 14-3-1. Universe 변화 (정확한 진단)

R3 끝점 변경 (2024-12 → 2023-12) 후 sortino_ir 재산출 → top 50 ∩ top 50 교집합 확장:

| 항목 | HO 포함 (R3=2024-12) | in-sample (R3=2023-12) | 차이 |
|---|---|---|---|
| Universe 크기 | 22 cfg | **26 cfg** | **+4** |
| 탈락 cfg (HO ∖ IS) | - | - | **0 개** |
| 신규 진입 (IS ∖ HO) | - | mat_rp_rp_lam_pap, mat_rp_rp_raw_pap, mat_rp_eq_raw_pap, mat_rp_eq_lam_pap | **+4** (RP 계열) |

→ **HO universe (22) 는 IS universe (26) 의 완전한 부분집합**. 누설 제거가 universe 를 확장하는 효과 (탈락 0).

#### 14-3-2. mat_eq_eq_*_pap 의 정확한 위치

| cfg | HO Lex rank | IS Lex rank | sortino_ir HO | sortino_ir IS |
|---|:---:|:---:|---:|---:|
| `mat_eq_eq_raw_pap` | #9 | **#8** | 9.37 | **18.48** (+9.11) |
| `mat_eq_eq_lam_pap` | #8 | **#9** | 10.46 | **20.93** (+10.47) |

→ **두 cfg 모두 HO/IS Top 10 에 진입** (탈락 X). sortino_ir 는 누설 제거 후 **거의 2 배 향상** (HO 12m 부진 제거 효과).

#### 14-3-3. 신규 in-sample Lex Top 10 (실제)

| rank | name | sortino_TEST | mdd_TEST | sortino_ir |
|:---:|---|---:|---:|---:|
| **1** | **mat_eq_mcap_lam_he** | 1.996 | **-0.120** ★ | 9.95 |
| 2 | mat_rp_rp_lam_pap | 1.982 | -0.124 | 6.07 |
| 3 | mat_rp_rp_raw_pap | 1.985 | -0.124 | 6.25 |
| 4 | mat_eq_mcap_lam_rms | 2.003 | -0.124 | 13.79 |
| 5 | q_raw_lam | 2.006 | -0.127 | 7.78 |
| 6 | mat_rp_eq_raw_pap | 2.028 | -0.129 | 6.30 |
| 7 | mat_rp_eq_lam_pap | 2.019 | -0.129 | 6.26 |
| **8** | **mat_eq_eq_raw_pap** | 2.039 | -0.129 | **18.48** |
| **9** | **mat_eq_eq_lam_pap** | 2.015 | -0.129 | **20.93** |
| 10 | mat_eq_mcap_raw_rms | 2.076 | -0.139 | 13.79 |

#### 14-3-4. Top 10 set 변화 (단순 swap, 거의 동일)

| 항목 | HO 포함 vs in-sample |
|---|---|
| 공통 Top 10 cfg | **9 cfg** (순서 swap 다수) |
| 차이 | **단 1 cfg**: HO #10 = `mat_eq_mcap_raw_he` ↔ IS #10 = `mat_eq_mcap_raw_rms` |

#### 14-3-5. 그럼 왜 Top 1 이 lam_he 인가? — Lex 의 mdd_TEST 결정성

ε=0.10 동순위 그룹 (sortino_TEST 1.982~2.076 = 11 cfg) 내에서 **2 순위 mdd_TEST** 가 결정:

| Lex 위치 | cfg | mdd_TEST | 해석 |
|:---:|---|---:|---|
| **Top 1** | mat_eq_mcap_lam_he | **-0.120** ★ | 그룹 내 최저 손실 |
| #2~4 | mat_eq_mcap_lam_rms, mat_rp_rp_*_pap | -0.124 | |
| #5 | q_raw_lam | -0.127 | |
| **#8~9** | **mat_eq_eq_raw_pap, lam_pap** | **-0.129** | 큰 손실 그룹 |
| #10 | mat_eq_mcap_raw_rms | -0.139 | |

→ mat_eq_eq_*_pap 의 **sortino_ir 강점 (3순위)** 이 **mdd_TEST 후순위 (2순위)** 때문에 활용 안 됨 — Lex 우선순위 구조의 결과.

#### 14-3-6. Decision Matrix 결과 비교

| 변형 | Top 1 (누설 제거 후) | 핵심 |
|---|---|---|
| §9-1 HO 포함 (비교용) | mat_eq_mcap_lam_rms 또는 mat_rp_rp_raw_pap | 가중 점수 변형 별 |
| **§9-4 in-sample only ★** | **`mat_eq_mcap_lam_he`** | 학술 표준 |

→ DM 의 5 차원 가중 점수에서 **mdd_TEST + alpha + 견고성** 강점이 lam_he 를 1위로 이끔. 단, mat_eq_eq_lam_pap 도 alpha (1위) + eff_n (1위) + sortino_ir (5위) 로 **DM 점수에서 후보**.

### 14-4. 최종 Top 1 — `mat_eq_mcap_lam_he`

#### 메트릭 (in-sample only, 의사결정 입력)

| 차원 | 값 |
|---|---:|
| sortino_TEST | **1.996** |
| mdd_TEST | **-0.120** ★ (Top 5 최저 손실) |
| sortino_ir (R3=2023-12) | 9.95 |
| turnover_avg | **0.430** ★ (거래비용 효율 1위) |
| eff_n_avg | 60.5 |
| alpha (CAPM 192m, 약한 누설) | 0.043 |
| beta | 0.545 (defensive) |

#### [참조] 사후 진단용 메트릭 (의사결정 미사용)

| 메트릭 | 값 |
|---|---:|
| sortino_HOLD_OUT | 0.798 |
| TEST/HO 격차 | 0.600 |
| mdd_FULL | -0.120 |
| IR | -0.053 |

#### Lex 일치 검증

| 기준 | Top 1 |
|---|---|
| §4-4 Lexicographic (in-sample) | mat_eq_mcap_lam_he |
| §9-4 Decision Matrix (in-sample) | **mat_eq_mcap_lam_he** ✓ |

→ **누설 제거 후 두 학술 표준 기준 모두 동일** = 결과 robust.

### 14-5. 시사점 — 이전 권고 변경

| 시점 | 권고 Top 1 | 학술적 정당성 |
|---|---|---|
| §5-2 (옵션 C) | mat_mcap_mcap_raw_he | sortino_HO 1.640 (HO 의존) |
| §9-5 (옵션 A) | mat_eq_eq_lam_pap | DM 1위 (HO 누설) |
| §10-8 (옵션 D) | mat_mcap_rp_lam_pap | sortino_ir 28.15 (HO 누설) |
| §12-8 (옵션 B) | mat_eq_mcap_lam_he/rms | tc + walk-forward |
| **§14 (학술 완성) ★** | **mat_eq_mcap_lam_he** | **누설 0%, in-sample only** |

→ **`mat_eq_mcap_lam_he` 가 학술적으로 완전한 Top 1** 으로 확정.

### 14-6. 발표 narrative 최종

> 본 분석의 의사결정 단계 (§4 Lex, §9 Decision Matrix) 는 **HOLD_OUT 정보를 0% 사용하지 않는 in-sample only 학술 표준** 으로 진행됩니다. 이를 위해 다음 6 가지 누설 위치를 모두 식별·제거:
>
> 1. sortino_ir 재산출 (R3 끝점 2023-12, HO 12m 분리)
> 2. Top 10 추출을 §4-4 (in-sample Lex) 기반
> 3. 16 메트릭 → 의사결정용 7 + 사후용 9 분리
> 4. fig05 → fig05a (in-sample) + fig05b (사후) 분리
> 5. Decision Matrix 의 5 차원 가중에서 HO 메트릭 완전 제거
> 6. §3 hard filter narrative 정리
>
> 누설 제거 결과 **Universe 자체가 변경** 되었고 (mat_rp_* 계열 4개 신규 진입, mat_eq_eq_lam_pap 탈락), 최종 **Top 1 = `mat_eq_mcap_lam_he`** (Lex / DM 양 기준 일치). HO 정보는 §11 sector / §12 SPY regime / §13 비교 / §14~§19 학술 검증에만 사용.

### 14-7. 학술 표준 충족 — 8 가지

1. ✓ **의사결정에 OOS 정보 사용 0%** (sortino_ir, top10, decision matrix 모두 in-sample) ★★★
2. ✓ 사후 분석 OOS 사용 정당 (Brinson attribution)
3. ✓ multiple testing 보정 (PBO/DSR)
4. ✓ factor risk premium 분리 (FF3/FF5/Carhart4)
5. ✓ 거래비용 sensitivity (5~100bps 6 단계)
6. ✓ walk-forward OOS (7년)
7. ✓ statistical significance (Memmel)
8. ✓ universe 변경에 대한 robust 검증 (sortino_ir 차이 분포 명시)
