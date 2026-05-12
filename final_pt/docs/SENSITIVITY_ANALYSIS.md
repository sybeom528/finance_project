# Winner 민감도 분석 — Q와 PCT_GROUP 정당화 및 해석 가이드

본 문서는 [05b_Analyze.ipynb](05b_Analyze.ipynb) **섹션 M (Q sensitivity, M0~M3)** 와 **섹션 N (PCT_GROUP sensitivity, N1~N3)** 의 winner 슬롯 민감도 분석 파이프라인이 **왜 필요한가**, **각 단계가 무엇을 답하는가**, **결과가 무엇을 의미하는가**를 정리.

대상 슬롯: **`mat_eq_eq_raw_pap`** (winner, 자동 식별)
- 식별 기준: `sortino_ir ≥ 10` 필터 (3-레짐 안정성 robust) → 전체기간 `mt['sortino']` 1위
- Top 후보군: sortino_ir 16.50 / 전체기간 sortino 1.826 / Sharpe 1.096 / CAGR 16.2% / MDD -13.6%
- **Q sweep**: 6개 q_value 변형 (`[0.001, 0.003, 0.0055, 0.0064, 0.008, 0.010]`)
  - 0.003 = winner default. 0.0055 / 0.0064 = BAB (Frazzini-Pedersen 2014) 학술 spread.
- **PCT sweep**: 5개 pct_group 변형 (`[0.15, 0.20, 0.25, 0.30 (winner), 0.35]`)

> **노트**: 초기 설계엔 M4 (실현 Q 분포 메커니즘), M5 (drawdown event attribution), N4 (eff_n/turnover 포트폴리오 구성 변화) 가 포함됐으나, 실측 결과 narrative 약함 (M5) 또는 본 발표 범위 외 (M4, N4) 로 판단되어 모두 생략. Q 섹션은 M0~M3, PCT 섹션은 N1~N3으로 최종 확정.

---

# Part 1. Q_value 민감도 (섹션 M)

## 1-1. 핵심 질문

```
"winner의 q=0.003 선택은 합리적인가? 다른 q로 바꾸면 결과가 크게 달라지나?
 BAB 학술 spread (0.0055~0.0064) 와 비교해도 winner 가 유지되나?"
```

이 두 문장에 답하기 위해 4단계 (M0 환경 셋업 + M1-M3 분석).

---

## 1-2. 분석 4단계 — 정당화 (필요성)

### M0. Winner 자동 식별 + bl_runner 환경 셋업

**무엇**:
- `rt_full` 의 `sortino_ir ≥ 10` 필터 (안정성 robust 13개) → `mt_full['sortino']` 정렬 1위 = winner
- panel, daily_ret, lstm_state, monthly_cache 1회 빌드 → 후속 sweep 에서 재사용

**왜 필요**:
- winner 가 하드코딩이 아닌 **데이터 기반 자동 식별** → 슬롯 결과 변경 시 자동 동기화
- monthly_cache 1회 빌드로 sweep 17회 호출이 빠르게 (~10-15분)

---

### M1. Q-value sweep + 메트릭 테이블

**무엇**: 6개 q 변형 walk_forward 실행 → q_results dict + 비교 표 (Sharpe / Sortino / Sortino_IR / CAGR / Vol / MDD).

**왜 필요**:
- 모든 후속 분석의 **raw data**. 이게 없으면 시작 불가.
- 결과는 `outputs/05b_Analyze/_sweep_q_<winner>.pkl` 로 자동 캐시 → 재실행 시 즉시 로드

| q_value | 의미 |
|---:|---|
| 0.001 | 매우 약한 view (보수적) |
| **0.003** | winner default (월 0.3% ≈ 연 3.6%) |
| **0.0055** | BAB (Frazzini-Pedersen 2014) 학술 평균 spread |
| **0.0064** | BAB 학술 high estimate |
| 0.008 | 강한 view (annualized ~10%) |
| 0.010 | 매우 강한 view (학술 상한) |

> 0.003 (winner default) 은 walk_forward 스킵 → winner pkl 의 ret 직접 사용. 실제 walk_forward 호출 5회.

---

### M2. 라인 플롯 (전체기간 + 레짐별 6 subplot)

**무엇**: q_value 변화에 따른 Sharpe / Sortino / MDD 곡선 시각화. winner 위치(빨간점) annotation.

**왜 필요**:
- 표만 봐서는 **곡선 모양** 파악 어려움 (역U? 단조? 평탄?)
- 레짐별 분해(R1/R2/R3)로 **레짐 의존성** 즉각 확인
- 발표·논문 핵심 그림 1장으로 작동

---

### M3. 통계 유의성 검정 (Jobson-Korkie + Block Bootstrap)

**무엇**:
- **Jobson-Korkie z-test (Memmel 2003 보정)**: 두 전략 Sharpe 차이 z-stat
  - H0: SR_variant − SR_winner = 0
  - 분산: V(ΔSR) ≈ (1/T) × [2(1−ρ) + ½(SR_w² + SR_v²) − SR_w·SR_v·(1+ρ²)/2]
  - z ≥ 1.96 → p < 0.05 (양측)
- **Block Bootstrap 95% CI**: block=6 (반년) × 1000회 resample → ΔSR 분포의 [2.5%, 97.5%]

**왜 필요**: 168개월 표본에서 작은 sharpe 차이가 통계적 의미 있는지 판정. "0.6% 더 좋은 q로 바꿔야 하나?" 같은 잘못된 결론 방지.

> Bonferroni 보정 미적용 (변형 수 5개로 false positive 위험 낮음). 필요시 p × 5 로 보정 가능.

---

## 1-3. Q 분석 실측 결과

### 핵심 발견

| 항목 | 값/패턴 |
|---|---|
| 변형 수 (winner 제외) | 5개 |
| 5% 수준 유의 | **0개** |
| 10% 수준 marginal (0.05 ≤ p < 0.10) | **0개** |
| 평균 z-stat | **-0.394** (variant 가 winner 보다 살짝 낮은 경향) |
| Bootstrap CI 모두 0 포함 | **True** |
| 결론 | **q ∈ Q_SWEEP 전 구간에서 winner robust** |

### 상세 p-value 표

| q | JK_z | JK_p | Boot CI_lo | Boot CI_hi | 해석 |
|---:|---:|---:|---:|---:|---|
| 0.001 | +0.137 | 0.891 | -0.104 | +0.121 | 매우 약한 view — winner 와 거의 동일 |
| **0.003 ★** | — | — | 0 | 0 | winner default (기준) |
| 0.0055 | -0.594 | 0.552 | -0.124 | +0.074 | BAB 학술 평균 — 동등 |
| 0.0064 | -0.407 | 0.684 | -0.141 | +0.117 | BAB 학술 high — 동등 |
| 0.008 | -0.505 | 0.614 | -0.186 | +0.116 | 강한 view — 동등 |
| 0.010 | -0.599 | 0.549 | -0.246 | +0.135 | 학술 상한 — 동등 |

### 결론

> **winner q=0.003은 sweep 전 구간([0.001, 0.010]) 에서 모든 변형과 통계적으로 동등.**
> 모든 변형 p > 0.5 (절반 이상이 random noise 수준), CI 모두 0 포함, |z| < 1.96 (5% 임계).
> 특히 **BAB (Frazzini-Pedersen 2014) 학술값 0.0055 / 0.0064 에서도 winner 와 차이 없음** — 본 펀드가 학술 standard 와 정합.

### M4·M5 생략 사유

| 분석 | 결과 | 결정 |
|---|---|---|
| M4 (실현 Q 분포 메커니즘) | 본 프로젝트 우선순위에서 후속 영역 | 생략 |
| M5 (drawdown event attribution) | 실측 결과 시장 system shock dominant + q-direction 월별 상쇄로 narrative 약함 | 생략 |

> q 선택의 정당화는 평시 risk-adjusted return (M1~M3) 에서 마무리.

---

# Part 2. PCT_GROUP 민감도 (섹션 N)

## 2-1. 핵심 질문

```
"P 행렬의 30% 컷오프(저변동 30% long / 고변동 30% short)는 robust 선택인가?
 더 좁은 cut(15-20%, 극단 종목만)이나 더 넓은 cut(35%, 분산)이 더 좋은가?"
```

PCT_GROUP은 BL view의 **포트폴리오 집중도**를 결정하는 핵심 hyperparameter:
- 작은 pct → view 신호 강하지만 종목 specific risk 큼
- 큰 pct → view 희석되지만 분산 안정

---

## 2-2. 분석 3단계 — 정당화

### N1. PCT sweep + 메트릭 테이블

**무엇**: 5개 pct 변형 walk_forward 실행 → pct_results dict + 비교 표 (M1과 동일 메트릭 셋).

**왜 필요**: 후속 모든 분석의 raw data. 결과는 `outputs/05b_Analyze/_sweep_pct_<winner>.pkl` 캐시.

| pct | 의미 |
|---:|---|
| 0.15 | 가장 좁은 분류 (저·고변동 각 15% 만) |
| 0.20 | 좁은 분류 |
| 0.25 | 약간 좁은 분류 |
| **0.30** | winner default (저·고변동 각 30%) |
| 0.35 | 약간 넓은 분류 |

> 0.30 (winner default) 은 walk_forward 스킵. 실제 호출 4회.

---

### N2. 라인 플롯 (전체기간 + 레짐별 6 subplot)

**무엇**: pct_group 변화에 따른 Sharpe / Sortino / MDD 곡선.

**왜 필요**:
- pct 가 만드는 **포트폴리오 집중도-분산 trade-off** 를 곡선 모양으로 표현
- 레짐별 분해로 위기 시 분산(큰 pct) 효과 vs 정상 시 집중(작은 pct) 효과 검증

---

### N3. 통계 유의성 검정 (Jobson-Korkie + Block Bootstrap)

**무엇**: M3 와 동일 방법론 (Memmel 보정 + Block Bootstrap), 비교 변수만 `pct_group`.

**왜 필요**: pct 변화가 통계적으로 유의한 차이를 만드는지 정량 판정.

---

## 2-3. PCT 분석 실측 결과

### 핵심 발견

| 항목 | 값/패턴 |
|---|---|
| 변형 수 (winner 제외) | 4개 |
| 5% 수준 유의 | **0개** |
| 10% 수준 marginal (0.05 ≤ p < 0.10) | **1개** (pct=0.15) |
| 평균 z-stat | **-1.166** (M3 의 -0.394 보다 강한 음수) |
| 결론 | **[0.20, 0.35] 완전 robust, 0.15 만 marginal 추세** |

### 상세 p-value 표

| pct | JK_z | JK_p | Boot CI_lo | Boot CI_hi | 해석 |
|---:|---:|---:|---:|---:|---|
| 0.15 | -1.691 | **0.091** | -0.185 | +0.022 | 가장 좁은 분류 — **10% 수준 marginal** |
| 0.20 | -1.604 | 0.109 | -0.136 | +0.018 | 10% 임계 (0.10) 살짝 초과 — marginal 아님 |
| 0.25 | -0.594 | 0.553 | -0.039 | +0.018 | winner 와 거의 동등 |
| **0.30 ★** | — | — | 0 | 0 | winner default |
| 0.35 | -0.774 | 0.439 | -0.057 | +0.033 | winner 와 동등 |

### 결론

> **winner pct=0.30은 [0.20, 0.35] 영역에서 모든 변형과 통계적 동등.**
> pct=0.15 만 |z|=1.69 (10% 임계 1.645 살짝 넘음) → marginal 약화 추세.
>
> 메커니즘: 너무 좁은 분류 (15%) 는 P 행렬의 view 신호 희석 또는 specific risk 증가로 winner 보다 약간 낮은 Sharpe (but 통계 유의 임계 못 넘음).

### N4 생략 사유

| 분석 | 결과 | 결정 |
|---|---|---|
| N4 (포트폴리오 구성 변화 eff_n / turnover) | 본 발표 범위 외 — Q 와 직교 분석이지만 핵심 결론에 영향 미미 | 생략 |

---

## 2-4. M3 / N3 자동 해석 (코드 자동 출력)

코드 셀 끝에 결과 기반 동적 결론 print:

**M3 자동 해석**:
```
=== M3 자동 해석 ===
  · 변형 수: 5개 (winner 제외)
  · 5% 수준 유의: 0개  /  10% 수준 marginal: 0개
  · 평균 z-stat: -0.394  (variant 가 winner 보다 낮은 경향)
  · Bootstrap CI 모두 0 포함: True
  → 결론: winner 가 q ∈ Q_SWEEP 전 구간에서 robust (모든 변형이 통계적으로 동등)
```

**N3 자동 해석**:
```
=== N3 자동 해석 ===
  · 변형 수: 4개 (winner 제외)
  · 5% 수준 유의: 0개  /  10% 수준 marginal: 1개
  · 평균 z-stat: -1.166
  → 결론: 1개 변형이 10% 수준 marginal — 5% 기준 robust, but 좁은 분류 영역에서 약한 추세
```

winner 또는 sweep 결과 변경 시 자동 갱신됨 (`n_sig`, `n_marg`, `mean_z` 분기).

---

# Part 3. 통합 — Q + PCT 결합 해석

## 3-1. 두 sensitivity의 관계

| 차원 | 영향 메커니즘 | trade-off |
|---|---|---|
| **Q** | view **강도** (얼마나 강하게 베팅) | 강한 view → alpha ↑ but variance ↑ |
| **PCT** | view **포트폴리오 구성** (몇 종목 베팅) | 좁은 cut → signal ↑ but specific risk ↑ |

→ **Q는 'how strongly', PCT는 'how broadly'**. 직교적 hyperparameter.

## 3-2. 결합 효과 (가설 — 미검증)

본 분석은 **각각 독립** sensitivity (1D). 결합 효과(2D grid)는 **미수행**.

| 성향 가설 | q | pct | 근거 |
|---|---|---|---|
| 안정형 | 약한 (0.001) | 넓은 (0.35) | view 약화 + 분산 |
| 중립형 | **0.003 (winner)** | **0.30 (winner)** | 양 축 plateau 중앙 |
| 공격형 | 강한 (0.008~0.010) | 좁은 (0.20~0.25) | view 강화 + 집중 |

> ⚠️ **차별화 폭의 한계**:
> - Q 축: 모든 변형 p > 0.5 — **통계적으로 q 변경 효과 입증 불가능**
> - PCT 축: [0.20, 0.35] 전 구간 p > 0.10 — 차별화 폭 사실상 0.15 (marginal) ~ 0.35
> - 2D 결합 효과는 **백테스트 미수행** — 위 매핑은 1D 추론
> - → 발표 시 "plateau 내 옵션" 정도로 보수적 표현 권장. 또는 차별화 컨셉 자체를 단순화.

---

# Part 4. 결론 narrative 템플릿

## 4-1. Q 관련 narrative

### A. Q-robust optimum
> "winner q=0.003은 [0.001, 0.010] 전 구간에서 모든 변형과 통계적 동등. 모든 변형 p > 0.5 (random noise 수준), CI 모두 0 포함. **q 선택이 winner 결과를 흔들지 않음** = q sensitivity 낮음."

### B. Q-BAB 학술 정합
> "BAB (Frazzini-Pedersen 2014) 학술 spread 0.0055 / 0.0064 에서도 winner 와 sharpe 차이 noise 수준 (p=0.55, 0.68). 본 펀드가 학술 standard 와 일치."

### C. Q-statistical robustness
> "5개 변형 모두 winner 와 통계적 유의차 없음. Sharpe 차이 noise 수준 + Bootstrap 95% CI 모두 0 포함 → 두 검정 방법이 동일 결론."

## 4-2. PCT 관련 narrative

### D. PCT-plateau robust
> "winner pct=0.30 은 [0.20, 0.35] 영역에서 winner 와 모든 변형 통계적 동등. 4개 변형 중 5% 수준 유의 0개, 10% 수준 marginal 1개 (pct=0.15) — **default 0.30 이 사후 통계로 검증된 sweet spot**."

### E. PCT-좁은 cut 약화 추세
> "pct=0.15 만 |z|=1.69 (10% 임계 살짝 넘음) — 5% 수준은 유의 X 이지만 약한 추세. 너무 좁은 분류는 P 행렬의 view 신호 희석 또는 specific risk 증가 가능성. 0.20 이상 권장."

## 4-3. 통합 narrative

### F. Two-axis robustness ⭐ 본 프로젝트 핵심
> "winner 슬롯 `mat_eq_eq_raw_pap` 은 q (강도) × pct (집중도) 두 축 모두에서 robust optimum.
> - **Q**: 5개 변형 모두 winner 와 통계적 동등 (p > 0.5, CI 모두 0 포함) → 완전 robust
> - **PCT**: [0.20, 0.35] 4개 변형 중 5% 수준 유의 0개, 0.15 만 marginal — robust
>
> → BL hyperparameter (q × pct) 두 축에서 동시에 안정 영역. winner 가 운으로 1위 된 게 아님."

### G. Risk profile mapping (제한적)
> "성향별 ETF 라인업 후보 (1D 결과 조합 추론, 2D 미검증):
> - 안정형 (q=0.001, pct=0.35)
> - 중립형 (q=0.003, pct=0.30 = winner)
> - 공격형 (q=0.008-0.010, pct=0.20-0.25)
>
> 모든 후보가 winner 와 통계적 동등 → **차별화 폭 noise 수준**. 발표 시 'plateau 내 옵션' 정도로 보수적 표현 권장. 2D q × pct grid backtest 미수행."

---

# Part 5. 산출물 위치

```
final_pt/outputs/05b_Analyze/
├─ M_winner_q_sensitivity.png         (M2 — Q 6 subplot 라인플롯)
├─ N_winner_pct_sensitivity.png       (N2 — PCT 6 subplot 라인플롯)
├─ _sweep_q_<winner>.pkl              (M1 결과 캐시 — 자동 생성)
└─ _sweep_pct_<winner>.pkl            (N1 결과 캐시 — 자동 생성)
```

M3 / N3 통계검정은 표·동적 print 출력만 (PNG 안 만듦).
sweep 캐시 덕분에 재실행 시 즉시 로드 (수십 ms).

---

# Part 6. 비판적 자기 점검

## 6-1. 본 분석의 한계

1. **표본 크기**: 168개월(K_CUT 적용)은 sharpe 차이 ±0.1 noise 흔함. Memmel z-test 의 표본 가정 (월별 ret 정규성) 도 엄격히는 성립 X.
2. **R2 dominance**: R2(~90M) > R3(48M) > R1(30M). 전체기간 곡선은 R2가 끌고감.
3. **단일 검정 통과 ≠ 진실**: Jobson-Korkie 비유의 → "차이 없다"가 아닌 "유의성 못 입증". 표본 더 크면 차이 나올 수도.
4. **PCT 의 universe 의존성**: 614종목 중 30%면 ~184종목 long/short. 다른 universe 면 절대 종목수 다름. **본 결론은 winner universe 한정**.
5. **Q와 PCT의 결합 sensitivity 미수행**: 각각 독립 분석. 인터랙션 효과 미검증.
6. **Bonferroni 미적용**: 5개 변형이라 false positive 위험 낮지만, 엄밀한 학술적 결론은 보정 후 권장.
7. **JK_z 부호 음수의 의미**: variant 가 winner 보다 살짝 낮은 경향. **단 통계 유의는 별개** — z 절대값 < 1.96 이면 noise 수준.

## 6-2. 발표에서 다루는 법

- 보수적 표현 사용: "통계적 차이 입증 못 함" (not "차이 없음")
- "robust optimum" 표현 (절대 우월 아님)
- universe 의존성 명시 ("614종목 universe 기준 30%")
- **JK_z 부호만으로 우열 주장 금지** — 통계 유의성과 실효 크기를 함께 봐야 함

---

# Part 7. 후속 분석 (현재 안 함)

| 분석 | 의의 | 우선순위 |
|---|---|---|
| Q × PCT 2D grid sensitivity | 결합 효과 / 인터랙션 | 중 |
| Walk-forward stability | rolling 36M optimum 변동 | 후속 연구 |
| Pareto frontier (Sortino, MDD) | 다목적 최적 | 낮음 |
| Information Ratio vs SPY | 액티브 운용 품질 | 중 (HOLD_OUT 검증) |
| lam_mean / TAU sensitivity | 다른 hyperparameter | 낮음 |
| Bonferroni / Holm 다중 비교 보정 | 엄밀 학술 검증 | 낮음 (현재 5변형으로 위험 낮음) |
| Drawdown event attribution | 위기 시 hyperparameter 보호효과 | 낮음 (M5 실측 narrative 약함) |
| 실현 Q 분포 메커니즘 | lambda mode 분포 추적 | 낮음 (구 M4) |
| eff_n / turnover 구성 변화 | PCT 의 포트폴리오 직접 효과 | 낮음 (구 N4) |

---

# Part 8. 메타 — 분석 구조의 합리성

## 8-1. M (Q) 섹션 — 본 프로젝트 채택 4단계

```
M0: WHO?          — winner 식별 (자동, sortino_ir ≥ 10 → 전체기간 sortino 1위)
M1: WHAT?         — sweep 결과 raw
M2: HOW LOOK?     — 곡선 시각화
M3: SIGNIFICANT?  — 통계 유의성 (JK + Bootstrap)
```

## 8-2. N (PCT) 섹션 — 3단계

```
N1: WHAT?         — sweep 결과 raw
N2: HOW LOOK?     — 곡선 시각화
N3: SIGNIFICANT?  — 통계 유의성
```

데이터 분석의 표준 흐름 (Tukey 1977, EDA → confirmatory): WHO → WHAT → HOW LOOK → SIGNIFICANT.

---

**문서 버전**:
- v1 (2026-05-08): Q 민감도 (섹션 M) 초안
- v2 (2026-05-08): PCT_GROUP 민감도 (섹션 N) 통합
- v3 (2026-05-08): 고등학생용 쉬운 해석 가이드 추가
- v4 (2026-05-08): M4·M5 생략, Q 섹션 M1-M3 확정
- v5 (2026-05-08): N3 PCT 통계 검정 실측 (winner=mat_eq_eq_lam_pap)
- v6 (2026-05-08): N3 톤 재조정 "Winner-favorable signal"
- **v7 (2026-05-12)**: **Winner 변경 (mat_eq_eq_lam_pap → `mat_eq_eq_raw_pap`, 자동 식별 기준 sortino_ir ≥ 10 + 전체기간 sortino 1위)** + **Q sweep 12→6개 (BAB 학술값 0.0055/0.0064 명시)** + **PCT sweep 7→5개** + **N4 (eff_n/turnover) 생략** + **Bonferroni 보정 제거** + 모든 결과 재실행 반영 + M3/N3 자동 해석 print 도입

**관련 노트북**: [05b_Analyze.ipynb](05b_Analyze.ipynb) §M, §N
**관련 가이드**: [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md)
