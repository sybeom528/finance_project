# 4번째 LSTM 재학습 결과 검증 보고서

**작성일**: 2026-05-08
**범위**: fold csv / ensemble csv / 156 BL pkl 정합성 + Top 1 후보 비교 진단
**관련 commit**: c5dabba (4번째 LSTM 재학습), dcc6009 (Streamlit 초안 백업)

---

## §1. 4번째 학습 결과 요약

### 1-1. fold csv 검증

| 항목 | 결과 | 검증 |
|---|---|---|
| max fold | **224** | ✅ panel 끝점 -21d n_safe 안전장치 적용 (fold 225 미학습은 정상) |
| fold 0~220 보존 | ✅ | 이전 학습 결과 그대로 |
| fold 221~224 신규 | ✅ | 4 folds 추가 (incremental, GPU 8-way 병렬 약 5~10분) |
| ticker per fold (신규) | 446 (221, 222, 223), 445 (224) | 일관성 OK |
| fold 일자 매핑 (panel start 2002-01-02) | OK | OOS 21 영업일 |
| 39 IS 부족 종목 | walk_forward_folds 자동 skip | ABNB, COIN, CEG 등 신규 상장 |

**fold 221~224 OOS 일자**:
- fold 221: 2025-10-01 ~ 2025-10-29 (10월)
- fold 222: 2025-10-30 ~ 2025-11-28 (11월)
- fold 223: 2025-12-01 ~ 2025-12-30 (12월)
- fold 224: 2025-12-31 ~ 2026-01-30 (12-31일 + 1월)

### 1-2. ensemble csv 검증

| 월 | 이전 ticker | 현재 ticker | 정상화 |
|---|---|---|---|
| 2025-09 | 600 | 600 | ✅ 변화 없음 |
| **2025-10** | **157** ⚠️ | **600** | ✅ +443 (3.8x) |
| **2025-11** | **157** ⚠️ | **600** | ✅ +443 |
| **2025-12** | **154** ⚠️ | **598** | ✅ +444 |
| 12-30, 12-31 | 1~5 | 451 | ✅ 정상 cover |
| 2026-01 (신규) | — | 592 | ✅ |
| 2026-02 (신규) | — | 591 | ✅ |

- y_pred_lstm finite 비율 100% (월별)
- Diebold-Pauly 가중치 (w_v4, w_har) 정상

### 1-3. 156 BL pkl 검증

- 156/156 모두 오늘 (05-08 13:03~14:05) 갱신
- **모든 ret 시리즈 192 months** (2010-01-31 ~ 2025-12-31)
- 192m 미만 cfg 0개

### 결론
> **데이터 무결성 검증 통과**. 4번째 LSTM 재학습이 의도한 결과로 완료됨.

---

## §2. Top 1 후보 비교 진단

### 2-1. 핵심 cfg TEST vs HOLD_OUT 격차

| cfg | sortino_TEST | sortino_HO | gap | rank_TEST | rank_HO | 해석 |
|---|---|---|---|---|---|---|
| **mat_eq_eq_lam_pap** (사용자 Top 1) | **2.015** | **0.685** | **+1.330** | **9** | **134** | **outlier overfitting** |
| mat_mcap_rp_lam_pap (안정성 1위) | 1.878 | 1.235 | +0.643 | 30 | 47 | 평균 가까움 |
| mat_mcap_mcap_inv_rms (HO 1위) | 1.459 | **2.278** | -0.820 | 136 | **1** | 반대 outlier |
| baseline | 1.761 | 1.464 | +0.298 | 65 | 35 | 정상 |
| capm_no_bl | 1.252 | 1.825 | -0.573 | **151** | 19 | TEST 꼴찌급, HO 강세 |
| naive_lowvol | 1.547 | 1.195 | +0.352 | 111 | 52 | 정상 |

**156 cfg gap 분포**: mean=0.585, median=0.657, std=0.554

**관찰**:
1. 평균 gap +0.585 → 156 cfg 다수가 TEST > HO (2024-2025 시장이 학습 기간 대비 어려움)
2. mat_eq_eq_lam_pap 의 gap +1.330 = 평균의 **2.27배** — outlier 수준
3. mat_mcap_mcap_inv_rms 의 gap -0.820 = 반대 방향 outlier — 단순 시총 기반 BL 이 실전에서 강함
4. capm_no_bl (BL 사용 안 한 baseline) 이 TEST 151위 → HO 19위 — **BL 가치 의심 가능 시그널**

### 2-2. mat_eq_eq_lam_pap HO 부진 원인 분석

| 지표 | TEST 168m | HO 24m | 변화 |
|---|---|---|---|
| 월평균 수익률 | 1.46%/m | 0.71%/m | **-51%** |
| 월별 std | 4.02% | 2.97% | -26% |
| 음수 월 비율 | 33% (55/168) | 33% (8/24) | 동일 |
| down_std (annualized) | 0.0823 | 0.0596 | -28% |

**HO worst 3 months**:
- 2024-11: **-6.68%** (단일 월 큰 손실)
- 2025-09: -3.33%
- 2025-02: -2.89%

**HO best 3 months**:
- 2024-06: +5.88%
- 2025-05: +4.05%
- 2024-07: +3.75%

**HO 부진의 주요 원인**:
1. **평균 수익률 절반으로 하락** (1.46% → 0.71%/m)
2. **rf (무위험수익률) 상승**: 2024-2025 미국 기준금리 상승으로 분자 (excess return) 압축
3. 변동성은 낮아졌으나 (std -26%) 평균 압축이 더 커서 sortino 부진
4. 단일 월 -6.68% (2024-11) 손실이 down_std 대비 큰 영향

> **결론**: mat_eq_eq_lam_pap 의 HO 부진은 단순 overfitting 보다는 **시장 환경 변화 (rf 상승 + 시장 안정화)** 의 영향이 큼. 그러나 156 cfg 평균 대비 2.27배 outlier 라는 점은 부분적 학습편향 가능성을 배제할 수 없음.

### 2-3. mat_mcap_rp_lam_pap (안정성 1위) 검증

| 메트릭 | 값 |
|---|---|
| sortino_FULL | 1.787 (Top 17) |
| sortino_TEST | 1.878 (Top 30) |
| sortino_HOLD_OUT | 1.235 (Top 47) |
| **sortino_ir** (3 레짐) | **28.15** ← 1위 |
| sortino_mean (3 레짐) | 1.858 |
| sortino_std (3 레짐) | **0.066** ← 압도적 |

**해석**:
- sortino_ir 28.15 = mean 1.858 / std 0.066, **3 레짐 (R1 회복 / R2 확장 / R3 변동) 매우 robust**
- TEST/HO 모두 평균 수준 (Top 30~47), **outlier 없는 안정성**
- 발표 narrative: "regime 무관 robust 모델"

### 2-4. mat_mcap_mcap_inv_rms (HO 1위) 검증

| 메트릭 | 값 |
|---|---|
| sortino_FULL | 1.457 |
| sortino_TEST | 1.459 (Top 136 — TEST 매우 부진) |
| sortino_HOLD_OUT | **2.278** ← HO 1위 |

**해석**:
- TEST 부진 + HO 1위 → **반대 방향 outlier**
- 발표 narrative 어려움 (학술 검증 약함)
- 단순 시총 기반 BL + inv_lambda Q + RMSE Ω 의 조합이 2024-2025 시장에서 운 좋게 잘 동작했을 가능성

---

## §3. 의사결정 권고 (4 옵션 trade-off)

| Option | Top 1 cfg | TEST 강점 | HO 강점 | 안정성 | 발표 메시지 |
|---|---|---|---|---|---|
| **A** | mat_eq_eq_lam_pap | ✅✅ Top 9 | ❌ Top 134 | ⚠️ outlier | "학습 기간 우월, 실전 시장 환경 변화로 부진 — 학술 valid 모델" |
| **B** | mat_mcap_rp_lam_pap | ⚪ Top 30 | ⚪ Top 47 | ✅✅ regime ir 28.15 | "regime 무관 안정 robust 모델" |
| **C** | mat_mcap_mcap_inv_rms | ❌ Top 136 | ✅✅ Top 1 | ❌ outlier 반대 방향 | "실전 검증 우수 (학술 약함 — 발표 부적합)" |
| **D** | baseline | ⚪ Top 65 | ⚪ Top 35 | ⚪ 정상 | "단순 강함 (BL 가치 부정)" |

### 권고

**Option B (mat_mcap_rp_lam_pap) 가 가장 정직한 선택**:
- TEST/HO 둘 다 평균 수준 — outlier 없음
- regime 안정성 28.15 (압도적) 는 학술 가치 있음
- 발표 narrative 작성 용이

**Option A (mat_eq_eq_lam_pap 유지)**: 정직한 서술로 발표 가능
- "TEST 168m sortino 2.015 (Top 9, 학술 우수) / HO 24m 0.685 (Top 134, 시장 환경 변화)"
- 학습편향 가능성 인지 + 정직한 한계 서술

**Option C, D 는 비추천** (학술 의의 약화 또는 BL 가치 부정)

---

## §4. 다음 단계

### 4-1. 등록된 spawn_task 진행
**"Top 1 추천 모델 재검토 (HO 부진)"** chip 클릭 후 별도 worktree 에서 진행:
- 본 보고서 검토
- 4 옵션 중 Top 1 결정
- PROJECT_OVERVIEW.md / 발표 narrative 갱신

### 4-2. 추가 검증 (Optional)
- **2024-11 단일 월 -6.68% 의 원인**: 어떤 종목/sector 손실?
- **HO 24m 의 portfolio turnover**: 안정성 추가 확인
- **regime별 행동 dashboard**: `analyze_plots.plot_styled_regime_dashboard` 활용

### 4-3. Streamlit 재구축 (별도 plan)
- 본 결정 결과를 반영한 새 Streamlit 단계적 재구축
- 기존 백업 (`final/_app_draft_backup/`) 참조

---

## §5. 정밀 분석 (학술 보강)

### 5-1. ⚠ Convention 정정 — `ret[t]` 의 시점 의미

| 표현 | 실제 의미 |
|---|---|
| `ret[2024-11-30] = -6.68%` | **2024-12 의 strategy 수익률** (2024-11-30 weights 기반 forward 1m) |
| 보고서 §2-2 "HO worst 2024-11" | 사실 **2024-12 손실** (정정) |

`bl_functions.walk_forward()` 의 `actual_ret = month_df['fwd_ret_1m']` 로 인해 `ret` series 의 인덱스 t 는 **rebalance 시점**이고 값은 **그 다음 달 수익률**.

### 5-2. 2024-12 -6.68% 손실의 sector 분해

**weights[2024-11-30] × fwd_ret_1m[2024-11-30] 분해 결과**:

| Sector | weight | avg_ret | contrib |
|---|---|---|---|
| Industrials | 16.6% | -8.39% | **-1.38%** |
| Utilities | 17.1% | -6.99% | **-1.18%** |
| Financials | 18.3% | -6.55% | **-1.10%** |
| Consumer Staples | 17.1% | -4.46% | -0.85% |
| Health Care | 12.9% | -6.04% | -0.81% |
| Real Estate | 7.6% | -8.08% | -0.58% |
| Information Technology | 1.8% | -3.70% | -0.11% |
| Communication Services | 1.5% | -4.19% | -0.11% |
| **합계 (gross)** | — | — | **-6.53%** |

**핵심 발견 — 본질적 약점 노출**:
1. **방어주 sector 6 개 합계 77.6% 집중** (Industrials + Utilities + Financials + Consumer Staples + Health Care + Real Estate)
2. AI/Tech sector underweight (IT 1.8%, Comm Services 1.5%)
3. 2024-12 = 미국 대선 후 sector rotation 환경, AI/Tech rally 시기 — **low-vol anomaly portfolio 의 cyclical underperformance**
4. 비교: baseline (균등가중) -4.05%, SPY -2.41% — mat_eq_eq_lam_pap 가 두 benchmark 대비 모두 underperform

**결론**: HO 부진의 핵심 원인은 단순 시장 환경 변화가 아닌 **strategy 의 sector exposure 본질적 약점** 임이 확인됨.

### 5-3. Regime별 sortino 직접 비교 (4 cfg)

| cfg | R1 회복 | R2 확장 | R3 변동 | mean | std | sortino_ir |
|---|---|---|---|---|---|---|
| **mat_eq_eq_lam_pap** | 2.205 | 2.044 | 1.745 | 1.998 | 0.191 | 10.46 |
| **mat_mcap_rp_lam_pap** | 1.779 | 1.941 | **1.854** | 1.858 | **0.066** | **28.15** |
| baseline | 1.952 | 1.831 | 1.525 | 1.769 | 0.180 | 9.83 |
| mat_mcap_mcap_inv_rms | 1.951 | 1.609 | 1.129 | 1.563 | 0.337 | 4.64 |

**관찰**:
- mat_eq_eq_lam_pap: R1/R2 강함, R3 변동 시기 (2020-2024) 약간 약화 — **변동 시기 약점**
- mat_mcap_rp_lam_pap: 모든 regime 1.78~1.94, std 0.066 — **압도적 robust**
- mat_mcap_mcap_inv_rms: R1 1.951 → R3 1.129, std 0.337 — 매우 불안정

### 5-4. Sortino vs Sharpe 일관성 — outlier 검증

| cfg | sharpe_TEST rank | sharpe_HO rank | sortino_TEST rank | sortino_HO rank |
|---|---|---|---|---|
| mat_eq_eq_lam_pap | **3** | **145** ⚠️ | 9 | 134 |
| mat_mcap_rp_lam_pap | 12 | 38 | 30 | 47 |
| mat_mcap_mcap_inv_rms | 102 | 7 | 136 | 1 |
| baseline | 31 | 45 | 65 | 35 |

**관찰**: mat_eq_eq_lam_pap 의 HO outlier 가 **Sortino 만의 문제가 아니라 Sharpe 도 동일 패턴** — 단순 down-side specific 이슈가 아닌 **전반적 HO 약화**.

### 5-5. 학술 강점 (mat_eq_eq_lam_pap 의 trade-off)

| metric | 값 | 전체 rank (n=153) |
|---|---|---|
| **CAGR** | 16.6% | **1위** |
| **Calmar** | 1.289 | **1위** |
| **Alpha (Jensen)** | 5.19% | **3위** |
| Sharpe | 1.108 | 9위 |
| Sortino | 1.850 | 8위 |
| MDD | -12.89% | 118위 |

**해석**: mat_eq_eq_lam_pap 은 **CAGR / Calmar / Alpha 의 학술 핵심 지표 1~3위** 를 차지하는 **학술적으로 매우 강한 모델**. 그러나 hold-out 24m sector rotation 환경에서는 약점 노출. **trade-off 관계**.

### 5-6. Portfolio composition 변화 (HO vs TEST)

| cfg | 항목 | TEST 168m | HO 24m | 변화 |
|---|---|---|---|---|
| mat_eq_eq_lam_pap | turnover | 1.011 | 0.829 | -18% |
| | eff_n | 213 | 270 | **+27%** |
| | top10_share | 23% | 16% | -7%p |
| mat_mcap_rp_lam_pap | turnover | 0.957 | 0.731 | -24% |
| | eff_n | 71 | 47 | -33% |
| | top10_share | 34% | 39% | +5%p |

**관찰**: HO 시 mat_eq_eq_lam_pap 의 eff_n 이 213 → 270 으로 증가 = **portfolio 가 더 균등해짐** (BL view 가 약해져 prior 균등 가까이 회귀). 이는 sector rotation 환경에서 결국 손실 확대.

### 5-7. 최종 권고 — 정밀 분석 결과 반영

| Option | 추천 | 사유 |
|---|---|---|
| **A** | **🟢 가능** | mat_eq_eq_lam_pap 의 학술 강점 (cagr 1위, calmar 1위, alpha 3위) 인지 + HO 부진의 본질 (sector rotation 약점) 정직히 서술 |
| **B** | **🟢 강력 추천** | mat_mcap_rp_lam_pap regime IR 28.15 + 모든 regime 1.78~1.94 압도적 안정성. 전 sector 분산. 발표 메시지 단순 |
| **C** | 🔴 비추천 | TEST 부진 (Sharpe 102위, Sortino 136위) 학술 약점 |
| **D** | 🔴 비추천 | BL 가치 부정 |

**최종 권고**:
- **학술 발표용**: Option A (mat_eq_eq_lam_pap) — cagr/calmar/alpha 1~3위 강조 + HO 부진 정직 서술
- **실전 robust 강조**: Option B (mat_mcap_rp_lam_pap) — regime 안정성 28.15 강조
- **두 옵션 모두 정당** — 사용자 가치 판단 영역

---

## §6. 시각화 해석 (5 차트 상세)

### 6-1. fig1 — TEST vs HOLD-OUT 산점도 (156 cfg overfitting 패턴)

**무엇을 보여주는가**:
- X축: Sortino TEST 168m, Y축: Sortino HOLD_OUT 24m
- 156개 점 = 156개 BL cfg
- 색상: Sortino FULL 192m (밝을수록 우수)
- y=x 점선: 정합 라인 (TEST = HO)
- 빨간 영역: TEST > HO (overfitting 의심), 파란 영역: TEST < HO (HO 강세)

**핵심 발견**:
1. **점들이 빨간 영역에 다수 분포** — 156 cfg 의 약 70% 이상이 TEST > HO. 즉 **2024-2025 시장이 학습 기간보다 어려웠다**는 시장 환경 효과
2. **mat_eq_eq_lam_pap (빨강 ★)** 이 그래프 우상단에서 **수직 아래로 떨어진 위치** (TEST 2.015, HO 0.685) — 156 cfg 평균 gap (0.585) 대비 2.27배 격차
3. **mat_mcap_mcap_inv_rms (녹색)** 이 좌상단 — TEST 부진하지만 HO 1위 (반대 outlier)
4. **mat_mcap_rp_lam_pap (주황)** 은 y=x 라인 가까이 — 정합성 양호
5. **baseline (보라)** 도 정합성 양호

**의사결정 함의**:
- mat_eq_eq_lam_pap 의 outlier 위치 = 학술 발표에서 정직 서술 필요
- mat_mcap_rp_lam_pap 의 정합 위치 = "robust 모델" narrative 정당화

### 6-2. fig2 — Top 1 후보 4개 누적수익률 (2010-01 ~ 2025-12)

**무엇을 보여주는가**:
- y축 log scale 누적수익률 (start=1)
- 4개 cfg + SPY (회색 점선)
- 빨간 점선 = 2024-01 HOLD-OUT 시작 시점

**핵심 발견**:
1. **TEST 168m 구간 (~2024-01 까지)**: mat_eq_eq_lam_pap (빨강) 이 가장 높이 상승 — cagr 1위 데이터의 시각적 증거
2. **HOLD-OUT 24m 구간 (2024-01 이후)**:
   - mat_eq_eq_lam_pap 곡선 **기울기 둔화** + 2024-12 무렵 큰 하락 (-6.68%)
   - mat_mcap_mcap_inv_rms (녹색) 가 가파르게 상승 (HO 1위)
   - mat_mcap_rp_lam_pap (주황) 안정적 상승
3. **2024-12 dip**: mat_eq_eq_lam_pap 이 가장 큰 하락폭 (수직 절벽 모양)
4. **SPY (회색)**: 전체 기간 4 cfg 모두 SPY 대비 우수 — BL 가치 입증

**의사결정 함의**:
- mat_eq_eq_lam_pap 의 TEST 우월성은 **시각적으로 압도적**
- HO 24m 의 부진도 명확히 시각화됨 — 정직 서술 가능

### 6-3. fig3 — mat_eq_eq_lam_pap HO 24m return 분포

**무엇을 보여주는가**:
- 좌: 24개월 monthly return 막대 (녹색=양수, 빨강=음수) + HO 평균선
- 우: TEST 168m vs HO 24m 분포 비교 (히스토그램)

**핵심 발견 (좌측)**:
1. **2024-11-30 막대 (빨강 큰 하락)**: -6.68% — 가장 큰 음수 = 2024-12 손실 (convention 정정 후)
2. **HO 평균 0.71%/m** (파란 점선) — TEST 평균 1.46% 대비 절반
3. **음수 월 8/24 = 33%** — TEST 와 동일 비율이지만 평균 압축으로 sortino 부진

**핵심 발견 (우측)**:
1. **분포 자체는 유사** — HO (빨강) 가 TEST (파랑) 대비 좀 더 narrow (변동성 -26%)
2. **HO 분포가 좌측으로 살짝 이동** = 평균 수익률 압축
3. **정규성 유지** — outlier 단일 월 영향이 아니라 분포 전체 이동

**의사결정 함의**:
- HO 부진의 원인은 **분포 자체 이동** (단순 outlier 아님)
- 시장 환경 변화 (rf 상승 + sector rotation) 가 분포 mean 을 끌어내림

### 6-4. fig4 — Regime별 sortino 비교 + sortino_ir

**무엇을 보여주는가**:
- 좌 패널: 4 cfg × 3 regime (R1 회복 / R2 확장 / R3 변동) sortino
- 우 패널: 각 cfg 의 sortino_ir (mean/std, regime 안정성)
- 점선: sortino=1.5 기준선

**핵심 발견 (좌측)**:
1. **mat_eq_eq_lam_pap**: R1 (2.21) ↘ R2 (2.04) ↘ R3 (1.75) — **단조 감소 패턴**. R3 변동 시기 (2020-2024) 가장 약함
2. **mat_mcap_rp_lam_pap**: R1 (1.78) → R2 (1.94) → R3 (1.85) — **거의 평탄** ★ 압도적 안정
3. **baseline**: R1 (1.95) ↘ R3 (1.53) — 약화 패턴
4. **mat_mcap_mcap_inv_rms**: R1 (1.95) ↘ R3 (1.13) — 매우 큰 감소, 불안정

**핵심 발견 (우측, sortino_ir)**:
- **mat_mcap_rp_lam_pap = 28.15** (압도적 1위)
- mat_eq_eq_lam_pap = 10.46 (3위)
- baseline = 9.83 (4위)
- mat_mcap_mcap_inv_rms = 4.64 (꼴찌)

**의사결정 함의**:
- mat_mcap_rp_lam_pap 의 안정성이 **시각적으로 명확** — Option B 강력 추천 근거
- mat_eq_eq_lam_pap 의 R3 약화 = HO 부진과 일관 (R3 종료 후 hold-out 도 약화)
- mat_mcap_mcap_inv_rms 의 R3 큰 약화 (1.95 → 1.13) = HO 1위는 **운 좋은 결과** 가능성

### 6-5. fig5 — 2024-12 sector decomposition

**무엇을 보여주는가**:
- 좌: 2024-11-30 mat_eq_eq_lam_pap weight (sector별)
- 우: 2024-12 sector 별 contribution (weight × return)

**핵심 발견 (좌측, 빨강 = >10% 집중)**:
1. **방어주 6 sector 가 모두 >10%**:
   - Financials 18.3%
   - Consumer Staples 17.1%
   - Utilities 17.1%
   - Industrials 16.6%
   - Health Care 12.9%
   - 합계 약 82% (방어주 절대 다수)
2. **Real Estate 7.6%** (방어주 7번째)
3. **공격주 sector 매우 낮음**: Information Technology 1.8%, Communication Services 1.5%, Consumer Discretionary 3.5%

**핵심 발견 (우측, 빨강 = 손실)**:
1. **모든 sector 가 손실** — 12월 시장 전반 하락
2. **Industrials 가장 큰 contribution -1.38%**, Utilities -1.18%, Financials -1.10%
3. **합계 -6.53%** = ret pkl 의 gross_ret 와 정확 일치 (검증 통과)
4. **AI/Tech sector contribution 미미 (-0.11% 수준)** — weight 작아서 손실 방어/회피 둘 다 못 함

**의사결정 함의**:
- **mat_eq_eq_lam_pap 는 low-volatility anomaly portfolio 의 전형적 구조** — 방어주 집중
- 2024-12 같은 sector rotation (모든 sector 동시 약세) 환경에서 분산 효과 미미
- AI/Tech rally 환경에 underweight — 2024-2025 의 큰 흐름 참여 못 함
- **본질적 strategy 약점이 명확히 드러남** — 단순 일회성 event 아님

### 6-6. 5 차트 종합 메시지

| 발견 | 차트 | 함의 |
|---|---|---|
| 156 cfg 다수 overfitting 패턴 | fig1 | 시장 환경 영향 (공통) |
| mat_eq_eq_lam_pap outlier 위치 | fig1 | 학술 정직 서술 필요 |
| mat_eq_eq_lam_pap TEST 압도적 우월 | fig2 | cagr/calmar 1위 시각 증거 |
| 2024-12 단일 큰 dip | fig2, fig3 | 본질 약점 식별 |
| HO 분포 전체 좌측 이동 | fig3 | rf 상승 + 평균 압축 |
| mat_mcap_rp_lam_pap 평탄 regime | fig4 | Option B 강력 추천 |
| mat_eq_eq_lam_pap R3 단조 감소 | fig4 | 변동 시기 약점 일관 |
| 방어주 77.6% 집중 | fig5 | low-vol anomaly 본질 약점 |
| 모든 sector 12월 손실 | fig5 | sector rotation 환경 |

**최종 결론** (시각화 종합):
- **Option A (mat_eq_eq_lam_pap 유지)**: 학술 강점 (cagr/calmar/alpha 1~3위) **시각적 증거 명확** (fig2), HO 부진의 본질 (방어주 집중) **정직 서술 가능** (fig5)
- **Option B (mat_mcap_rp_lam_pap)**: regime 안정성 **시각적으로 압도적** (fig4) + 정합 위치 (fig1)
- **두 옵션 모두 학술 정당** — 발표 narrative 방향성 차이만 있음

---

## 첨부 파일

### 시각화 (`final/outputs/04_validation/`)
- `fig1_test_vs_holdout_scatter.png` — 156 cfg TEST vs HO 산점도 (overfitting 패턴)
- `fig2_cumulative_return_top4.png` — Top 1 후보 4개 누적수익률 + HOLD-OUT 구분선
- `fig3_mat_eq_eq_lam_pap_ho_returns.png` — mat_eq_eq_lam_pap HO 24m return 분포
- `fig4_regime_sortino_4cfg.png` — regime별 sortino 비교 + sortino_ir
- `fig5_2024_12_sector_decomposition.png` — 2024-12 손실 sector 분해

### 자동 검증 명령
plan 파일 V1, V2 참조: `C:\Users\gorhk\.claude\plans\c-users-gorhk-finance-project-test-atomic-comet.md`
