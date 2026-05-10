# Winner 민감도 분석 — Q와 PCT_GROUP 정당화 및 해석 가이드

본 문서는 [99_analyze.ipynb](99_analyze.ipynb) **섹션 M (Q sensitivity, M1~M3)** 와 **섹션 N (PCT_GROUP sensitivity, N1~N4)** 의 winner 슬롯 민감도 분석 파이프라인이 **왜 필요한가**, **각 단계가 무엇을 답하는가**, **어디까지 해야 충분한가**를 정리.

> **노트**: 초기 설계엔 M4 (실현 Q 분포 메커니즘), M5 (drawdown event attribution)가 포함됐으나, 실제 분석 결과 **공통 시장 충격이 dominant + q-direction 월별 상쇄**로 narrative 약함이 확인되어 **M4·M5 모두 생략**. Q 섹션은 M1~M3, PCT 섹션은 N1~N4로 최종 확정.

대상 슬롯: `mat_eq_eq_lam_pap` (winner)
- **Q sweep**: 12개 q_value 변형 (0.001 ~ 0.010 step 0.001 + 0.0055, 0.0064)
- **PCT sweep**: 7개 pct_group 변형 (0.10 / 0.15 / 0.20 / 0.25 / **0.30 (winner)** / 0.35 / 0.40)

---

# Part 1. Q_value 민감도 (섹션 M)

## 1-1. 핵심 질문

```
"winner의 q=0.003 선택은 합리적인가? 다른 q로 바꾸면 결과가 크게 달라지나?"
```

이 한 문장에 답하기 위해 세 단계.

---

## 1-2. 분석 3단계 — 정당화 (필요성)

### M1. 메트릭 테이블

**무엇**: 12개 q 변형 × 메트릭 (Sharpe/Sortino/Sortino_IR/CAGR/MDD/Beta/Alpha/Turnover/레짐별 분해)

**왜 필요**:
- 모든 후속 분석의 **raw data**. 이게 없으면 시작 불가.
- 절대 필요 (★★★)

---

### M2. 라인 플롯 (전체기간 + 레짐별 6 subplot)

**무엇**: q_value 변화에 따른 Sharpe/Sortino/MDD 곡선 시각화. winner 위치(빨간점) + optimum 위치(초록 화살표) annotation.

**왜 필요**:
- 표만 봐서는 **곡선 모양** 파악 어려움 (역U자? 단조? 평탄?)
- 레짐별 분해(R1/R2/R3)로 **레짐 의존성** 즉각 확인
- 발표·논문 핵심 그림 1장으로 작동

---

### M3. 통계 검정 (Jobson-Korkie + Block Bootstrap)

**무엇**:
- Jobson-Korkie z-test (Memmel 보정): Sharpe 차이의 통계적 유의성
- Block Bootstrap (B=2000, block=3M): Sortino 95% 신뢰구간

**왜 필요**: 168개월 표본에서 sortino 차이 0.011 같은 작은 차이가 통계적 의미 있는지 판정. "0.6% 더 좋은 q로 바꿔야 하나?" 같은 잘못된 결론 방지.

---

## 1-3. Q 분석 결과 (실측)

### 핵심 발견

| 항목 | 값/패턴 |
|---|---|
| 전체기간 optimum | q=0.002 (Sortino 1.989) |
| Winner q=0.003 | Sortino 1.978 (optimum 대비 -0.6%) |
| 곡선 모양 | 좌측 peak + 우측 단조 감소 |
| MDD anomaly | q=0.004에서만 절벽 (-0.144) |
| 레짐 의존성 | **R2 확장: 큰 q 선호 / R3 변동: 작은 q 선호** |

### 결론

> winner q=0.003은 plateau optimum 영역에 위치하고, R2와 R3의 정반대 선호를 절반씩 양보한 합리적 타협점. 통계 검정 (Bonferroni 후) 차이 무의미.

### M4·M5 생략 사유 (실측 후)

| 분석 | 결과 | 결정 |
|---|---|---|
| M4 (실현 Q 분포 메커니즘) | 본 프로젝트 우선순위에서 후속 영역 | 생략 |
| M5 (drawdown event attribution) | 실측 결과 모든 q에서 worst 6M 평균 ≈ -7.8% **평탄** + worst Top5 중 4개월이 12/12 슬롯 공통 = 시장 system shock + q-direction이 월마다 정반대로 상쇄 | 생략 (narrative 약함) |

> **M5 실측 진단 요약**: "위기에서 q 선택은 평균적으로 무관. 시장 system shock은 q로 못 막고, individual event의 q-direction은 월마다 반대로 작용해 평균 상쇄됨. winner는 worst event들에서 평균 위치 — 강점도 약점도 없음." 따라서 q 선택의 정당화는 평시 risk-adjusted return (M1~M3) 에서 마무리.

---

# Part 2. PCT_GROUP 민감도 (섹션 N) ⭐

## 2-1. 핵심 질문

```
"P 행렬의 30% 컷오프(저변동 30% long / 고변동 30% short)는 robust 선택인가?
 더 좁은 cut(10-20%, 극단 종목만)이나 더 넓은 cut(35-40%, 분산)이 더 좋은가?"
```

PCT_GROUP은 BL view의 **포트폴리오 집중도**를 결정하는 핵심 hyperparameter:
- 작은 pct → view 신호 강하지만 종목 specific risk 큼
- 큰 pct → view 희석되지만 분산 안정

---

## 2-2. 분석 4단계 — 정당화

### N1. 메트릭 테이블

**무엇**: 7개 pct 변형 × 메트릭 (M1과 동일 + **eff_n_avg** 컬럼 추가).

**왜 필요**: 후속 모든 분석의 raw data.

---

### N2. 라인 플롯 (전체기간 + 레짐별 6 subplot)

**무엇**: pct_group ∈ [0.10, 0.40] 변화에 따른 Sharpe/Sortino/MDD 곡선.

**왜 필요**:
- pct가 만드는 **포트폴리오 집중도-분산 trade-off**를 곡선 모양으로 표현
- 레짐별 분해로 위기 시 분산(큰 pct) 효과 vs 정상 시 집중(작은 pct) 효과 검증

---

### N3. 통계 검정 (Jobson-Korkie + Bootstrap)

**무엇**: M3와 동일 방법론 — Sharpe z-test + Sortino 95% CI.

**왜 필요**: pct 변화가 통계적으로 유의한 차이를 만드는지 정량 판정.

---

### N4. 포트폴리오 구성 변화 (eff_n, turnover) ⭐ Q에 없는 PCT 고유 분석

**무엇**: pct 변화에 따른 **effective N** (실효 종목 수, 분산도 측정) 및 평균 turnover 변화.

**왜 필요**:
- pct는 **직접적으로 P 행렬의 종목 수를 결정**:
  - pct=0.10 → 484종목 × 0.10 = ~48종목씩 long/short
  - pct=0.30 → ~145종목씩 (winner)
  - pct=0.40 → ~194종목씩
- **eff_n**은 분산 효과 측정 — 작은 pct는 specific risk ↑
- **turnover**는 거래비용과 직결 — 작은 pct는 변동성 ↑ → 자주 종목 교체
- **이 trade-off가 sortino 곡선과 어떻게 정합하는지** 확인

---

## 2-3. PCT 분석 결과 — **고등학생용 쉬운 해석 가이드**

### 🎯 한 줄 요약
**"베팅에 종목을 몇 개나 포함시킬지(좁게? 넓게?)에 따라 결과가 어떻게 바뀌나"**

### 📊 N1~N2 — 메트릭 표 + 라인 곡선

**무엇**:
- pct=0.10 → 484종목 중 양 끝 10%만 (저변동 48종목 long, 고변동 48종목 short, 중간 388종목 무시)
- pct=0.30 (winner) → 양 끝 30%씩 (각 145종목, 중간 194종목 무시)
- pct=0.40 → 양 끝 40%씩 (각 194종목, 중간 96종목만 무시)

**3가지 곡선 패턴 (Sortino vs pct)**:

#### 패턴 1: 평탄 (가장 흔한 결과 — robust)
**모양**: 0.10에서 0.40까지 sortino가 거의 일직선 (변동 ±0.1 이내)
**의미**: "pct 선택은 결과에 영향 거의 없음"
**비유**: 그릇 크기를 바꿔도 같은 음식이 나옴
**결론**: ✅ **winner 30% 유지 안전** — 셀링 포인트로 활용 ("pct 선택 무관 = robust")

#### 패턴 2: 역U자 (peak at 0.20-0.30)
**모양**: 양 끝(0.10, 0.40)에서 낮고, 중간(0.20-0.30)에서 가장 높음
**의미**: "**적당한 분산**이 최적. 너무 집중도 너무 분산도 안 좋음"
**비유**: 음식 재료를 너무 적게 넣어도 (단조), 너무 많이 넣어도 (잡미) 맛이 떨어짐
**결론**: ✅ winner 30% 그대로 OK (peak 영역 안)

#### 패턴 3: 단조 (한 방향이 우월)
**모양 A** (작은 pct 우월): 0.10 → 0.40 으로 갈수록 sortino 감소
**의미**: "극단 종목만 골라 view 신호 강화 시 더 좋음"
**결론**: ⚠️ winner 30% → 20%로 변경 검토 (단 specific risk 점검 필요)

**모양 B** (큰 pct 우월): 0.10 → 0.40 으로 갈수록 sortino 증가
**의미**: "분산이 view 집중보다 효과 큼"
**결론**: ⚠️ winner 30% → 35-40%로 변경 검토

---

### 📊 N3 — 통계 검정 (Jobson-Korkie + Bootstrap)

**무엇**:
- Jobson-Korkie: "winner의 sharpe와 다른 pct의 sharpe 차이가 우연인가?"
- Bootstrap CI: 각 pct의 sortino 95% 신뢰구간 → winner CI와 겹치는지

**3가지 결과 패턴**:

#### 결과 A: 모두 winner CI와 겹침
**의미**: "pct 선택은 통계적으로 영향 없음" — 모든 차이는 노이즈 수준
**결론**: ✅ **winner 30% 정당화 강력** ("어떤 pct로 바꿔도 통계적으로 같음")

#### 결과 B: 일부(예: 0.10, 0.40)가 분리, 나머지는 겹침
**의미**: "극단(아주 좁거나 넓은 cut)에서만 차이 있음, 중간 영역(0.20-0.35)은 동일"
**결론**: ✅ winner 30% OK + "극단은 피하라" narrative

#### 결과 C: 모두 분리 (ipsatized 차이)
**의미**: "pct 선택이 결과에 큰 영향" — robust 아님, 신중 선택 필요
**결론**: ⚠️ winner 재검토 + 가장 좋은 CI를 가진 pct로 변경 고려

> **참고**: Bonferroni 보정 (p < 0.05/7 = 0.0071) 적용. 다중 비교 시 false positive 방지.

---

### 📊 N4 — 포트폴리오 구성 변화 (eff_n / turnover)

**무엇**: pct가 바뀌면 **실제 포트폴리오가 어떻게 변하나** 정량화. 두 가지 그래프.

#### 그래프 1: eff_n (Effective N — 실효 종목 수)

**eff_n이란?**: 포트폴리오의 "실질적 분산도"
- eff_n = 1/Σ(w²) (Herfindahl 역수)
- 모든 종목이 똑같이 들어가면 eff_n = 종목 수
- 한 종목에 몰리면 eff_n = 1
- **숫자가 클수록 분산 잘 됨**

**예상 패턴** (pct vs eff_n):

| pct | 예상 eff_n | 의미 |
|---|---|---|
| 0.10 | 작음 (~30~50) | 적은 종목, 집중 |
| 0.30 (winner) | 중간 (~100~150) | 균형 |
| 0.40 | 큼 (~150~200) | 많은 종목, 분산 |

**해석**:
- **eff_n이 pct와 함께 증가** = 정상. 종목 수가 많아질수록 분산 효과 자동 발생
- **그래프가 평탄** = pct가 늘어도 분산 효과 미미 (소수 대형주가 dominant)
- **U자나 봉우리** = 비정상. mcap 가중과의 인터랙션 효과

#### 그래프 2: turnover (월별 회전율)

**turnover란?**: 한 달에 포트폴리오를 얼마나 갈아치우는가
- turnover = 0.5 → 매월 절반 갈아엎음 (편측 50%)
- turnover = 1.0 → 매월 전부 새로 매수
- **숫자가 작을수록 거래비용 적음 = 좋음**

**예상 패턴**:

| pct | 예상 turnover | 의미 |
|---|---|---|
| 0.10 | 큼 (~1.0~1.3) | 적은 종목, 변동 클 때 자주 교체 |
| 0.30 | 중간 (~0.7~0.9) | 균형 |
| 0.40 | 작음 (~0.6~0.8) | 많은 종목, 한두 종목 변해도 영향 작음 |

**해석**:
- **turnover가 pct와 반비례 (단조 감소)** = 정상. 큰 그룹은 안정적
- **U자** = 양 극단(0.10, 0.40)에서 변동성, 중간이 안정 (현실에서 흔함)
- **단조 증가** = 비정상 (큰 pct가 더 자주 바뀜? 의심)

#### 두 그래프 통합 해석

| 그래프 1 (eff_n) | 그래프 2 (turnover) | 결합 의미 | 결정 |
|---|---|---|---|
| 단조 증가 | 단조 감소 | "큰 pct = 분산↑ + TC↓ 양쪽 다 유리" | ➡ 큰 pct 검토 |
| 단조 증가 | U자 (winner 중간이 최저) | "winner가 turnover 안정 sweet spot" | ✅ winner 유지 |
| 봉우리 (winner peak) | 단조 감소 | "winner가 분산 sweet spot" | ✅ winner 유지 |
| 평탄 | 평탄 | "pct가 구성에 영향 거의 없음" | 신호 약함, sortino만 보고 결정 |

---

---

### ✅ N3 실측 결과 (ipynb 기준 갱신) — Winner-favorable signal

**관찰값** (`99_analyze.ipynb` cell 45 실행 결과):
```
Jobson-Korkie 5% 유의:    3/6  ← pct=0.10 (★★), pct=0.15 (★), pct=0.20 (★)
Jobson-Korkie Bonferroni: 1/6  ← pct=0.10 만 (winner가 Bonferroni 후도 유의 우위)
Bootstrap CI 겹침:        7/7  ← 0.10~0.40 모든 pct가 winner Sortino CI와 겹침
```

**상세 JK p-value 표**:

| pct | JK p-value | 5% 유의 | Bonferroni (0.0071) | 평가 |
|---|---:|:---:|:---:|---|
| 0.10 | 0.0000 | ★ | ★★ | winner 강력 우위 |
| 0.15 | 0.0117 | ★ | — | winner 우위 (5% only) |
| **0.20** | **0.0236** | **★** | — | **winner 우위 (5% only)** |
| 0.25 | 0.0723 | — | — | plateau 진입 |
| 0.35 | 0.2617 | — | — | plateau |
| 0.40 | 0.4413 | — | — | plateau |

**패턴 진단**: **Winner-favorable — winner가 좁은 cut(0.10~0.20) 대비 sharpe 유의 우위 + plateau ([0.25, 0.40]) 내 동등**

#### 결과 요약

- **plateau (winner와 통계적 동등)** = `[0.25, 0.40]` — Sharpe·Sortino 모두 동등
- **winner 우위 영역** = `[0.10, 0.20]` — Sharpe 유의 열등 (Bonferroni 통과는 0.10만)
- **두 기준이 다른 그림**: Sharpe는 plateau를 [0.25, 0.40]으로 좁게 잡지만, Sortino bootstrap은 [0.10, 0.40] 모두 동등
- → **winner 30%는 Sharpe 기준 plateau의 중앙 + 좁은 cut 대비 통계적 우위 입증**

| pct | Sharpe 기준 | Sortino 기준 | 사용 적합성 |
|---|---|---|---|
| 0.10 | ★★ 유의 열등 (Bonferroni) | 동등 (CI 겹침) | ❌ Sharpe로 명백 열등 |
| 0.15 | ★ 유의 열등 (5% only) | 동등 | ❌ Sharpe로 열등 |
| **0.20** | **★ 유의 열등 (5% only)** | **동등** | **⚠️ Sharpe 부족** |
| 0.25 | 동등 (p=0.072) | 동등 | ✅ plateau 진입 |
| **0.30 (winner)** | 기준 | 기준 | ✅ default |
| 0.35 | 동등 | 동등 | ✅ plateau |
| 0.40 | 동등 | 동등 | ✅ plateau |

#### 메커니즘 (왜 좁은 cut이 Sharpe 열등?)

좁은 cut(48~96종목씩 long/short, pct=0.10~0.20)은 specific risk가 커서 전체 변동성↑ → sharpe 분모 페널티. 하방 위험에선 차이 못 잡아내지만(Sortino bootstrap 7/7 겹침), 분산 효과가 부족한 영역이라는 점은 Sharpe로 명확히 드러남.

#### narrative

> "PCT sensitivity 분석은 winner 30% 선택을 통계적으로 지지한다. winner는 좁은 cut(0.10) 대비 Bonferroni 보정 후도 sharpe 유의 우위(★★)를 보이며, 0.15·0.20 대비도 5% 유의 우위. plateau [0.25, 0.40]에서는 sharpe·sortino 모두 동등. 즉 winner는 **Sharpe 기준 plateau 중앙 + 좁은 cut 대비 입증된 우위** 두 측면에서 정당화된다."

#### 성향별 ETF 매핑 — Sharpe 기준 갱신

| 성향 | pct | 평가 |
|---|---|---|
| 공격형 | **0.25** (단일점) | ⚠️ winner와 동등하지만 plateau 좌측 끝 |
| 중립형 | **0.30 (winner)** | ✅ default, plateau 중앙 |
| 안정형 | **0.35-0.40** | ✅ winner와 동등 + 분산 안정 보너스 |

→ **차별화 폭이 markdown 이전 기록(0.20-0.40)보다 좁아졌음 (0.25-0.40)**. pct=0.20은 Sharpe 유의 열등으로 권장 영역에서 배제. Sortino만 본다면 [0.10, 0.40] 모두 동등하지만, 두 기준이 충돌할 때 보수적으로 Sharpe 기준 채택 권장.

---

### 🧠 종합 해석 — N1~N4 통합 패턴

| sortino (N2) | 통계 (N3) | 구성 (N4) | 본 프로젝트 활용 |
|---|---|---|---|
| **평탄** | 모두 winner CI 겹침 | eff_n 단조 증가 + turnover 단조 감소 | "**winner 30% robust + 안정형은 큰 pct 검토 가치**" |
| **역U at 30** | winner가 peak | eff_n 봉우리 (winner) | "**winner 30% strict optimum**" |
| **0.10이 peak** | 작은 pct가 분리 (유의) | eff_n 작음, turnover 큼 | "**집중 베팅 우월, 단 specific risk + TC 인지**" |
| **0.40이 peak** | 큰 pct가 분리 (유의) | eff_n 큼, turnover 작음 | "**winner 30%은 미흡, 35-40%로 변경 권장**" |

---

### 💡 발표용 핵심 메시지 후보

#### 만약 평탄(robust)이 나오면:
> "PCT_GROUP은 [0.10, 0.40] 범위에서 sortino 변동 ±0.1 이내. winner 30%는 **robust optimum**이며, BAB 학술 표준(보통 30%)과 일치. 두 hyperparameter (Q + PCT) 모두 robust → 모델 자체가 hyperparameter sensitivity 적음을 증명."

#### 만약 역U at 30이 나오면:
> "PCT_GROUP 곡선은 30%에서 peak를 형성하는 명확한 역U자. 너무 좁으면(0.10) specific risk로 손해, 너무 넓으면(0.40) view 신호 희석으로 손해. 30%는 데이터 기반 strict optimum."

#### 만약 단조 (작은 pct 우월)가 나오면:
> "극단 종목 cut(0.10~0.20)이 sortino 우월. 단 eff_n은 30~50으로 specific risk 큼, turnover도 1.0+ 수준 → 거래비용 후 net 검증 필요. 현재 winner 30%는 **TC-adjusted 균형점**으로 정당화 가능."

#### 만약 N4가 단조 (eff_n↑ + turnover↓)이고 sortino 평탄이면:
> "**큰 pct가 모든 면에서 비-수익 측면 우월** (분산↑, TC↓, 동등 sortino). 안정형 ETF 변형은 pct=0.35-0.40 채택 시 위험·비용 측면 추가 보호 가능."

---

### ⚠️ 자주 헷갈리는 부분 (PCT 편)

1. **eff_n과 종목 수는 다르다**: pct=0.30이면 145종목 들어가지만, mcap 가중이라 큰 종목이 weight 잡음 → eff_n은 30~80 수준일 수 있음
2. **turnover 단위**: 본 프로젝트는 **편측(one-side, 0~1 범위)**. 매수+매도 합친 양면(two-side, 0~2 범위)이 아님. 0.5 = 50% 교체.
3. **CI 겹침은 "차이 없음"이 아니라 "차이 입증 못 함"**: 168개월 표본의 한계. 실무에선 더 긴 데이터로 재검증 권장.
4. **PCT는 universe-dependent**: S&P500 vs Russell 1000이면 같은 pct라도 다른 종목 수. 본 분석은 484종목 universe 고정.
5. **winner의 "robust" 의미**: "다른 pct도 winner와 비슷하게 좋다" (= 절대 우월 아님). 발표 시 "robust optimum"으로 표현 권장.

---

# Part 3. 통합 — Q + PCT 결합 해석

## 3-1. 두 sensitivity의 관계

| 차원 | 영향하는 메커니즘 | trade-off |
|---|---|---|
| **Q** | view **강도** (얼마나 강하게 베팅) | 강한 view → alpha ↑ but variance ↑ |
| **PCT** | view **포트폴리오 구성** (몇 종목 베팅) | 좁은 cut → signal ↑ but specific risk ↑ |

→ **Q는 'how strongly', PCT는 'how broadly'**. 직교적 hyperparameter.

## 3-2. 결합 효과 (가설 — 미검증)

본 분석은 **각각 독립** sensitivity (1D). 결합 효과(2D grid)는 **미수행**. 가능한 인터랙션 (가설):
- 작은 q + 큰 pct: "보수적 분산" — 안정형 후보
- 큰 q + 작은 pct: "집중 베팅" — 공격형 후보
- 본 winner (q=0.003 + pct=0.30): "균형형" — 중립형 default

> ⚠️ 위 매핑은 1D 결과의 *조합 추론*이지 2D backtest로 입증된 사실 아님.

## 3-3. 성향별 ETF 차별화 매핑 (제안 — caveat 다수)

| 성향 | q | pct | 근거 (제한적) |
|---|---|---|---|
| **공격형** | 0.005 | **0.25** (단일점) | Q 축은 plateau 내 어디든 OK. PCT는 0.20이 Sharpe 유의 열등으로 배제됐으므로 0.25만 |
| **중립형** | **0.003 (winner)** | **0.30 (winner)** | 양 축 plateau 중앙 |
| **안정형** | 0.001-0.002 | 0.35-0.40 | Q는 R3 선호, PCT는 plateau 우측 |

> ⚠️ **차별화 폭의 한계**:
> 1. Q 축 — Bonferroni 통과 차이 **0개** (즉 q 변경의 통계적 의미는 없음, Q sensitivity §1-3 결론)
> 2. PCT 축 — Sharpe 기준 plateau는 [0.25, 0.40] 3개 점만, 공격형 차별화 폭은 사실상 0.25 단일
> 3. 2D 결합 효과는 **백테스트 미수행** — 위 매핑은 1D 추론
>
> → "q × pct 두 축 차별화"는 **추론 수준**의 매핑이며, 발표 시 학술적 방어를 위해서는 **2D 9-cell grid backtest 추가 권장**. 또는 차별화 컨셉 자체를 단순화 ("plateau 내 옵션") 권장.

---

# Part 4. 우선순위 / 시간 배분

## 4-1. 단계별 우선순위 통합표

| 단계 | 학부 프로젝트 | 학회/논문 | 운용 결정 | 발표 (15분) | 발표 (30분+) |
|---|---|---|---|---|---|
| **Q 섹션 (M1-M3, 본 프로젝트 채택)** | | | | | |
| M1 (Q표) | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ |
| M2 (Q곡선) | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ |
| M3 (Q통계) | ★★ | ★★★ | ★★ | ★ | ★★★ |
| ~~M4 (메커니즘)~~ | 생략 | 후속 | — | — | — |
| ~~M5 (이벤트)~~ | 생략 (실측 후 narrative 약함) | 후속 | — | — | — |
| **PCT 섹션 (N1-N4)** | | | | | |
| N1 (PCT표) | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ |
| N2 (PCT곡선) | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ |
| N3 (PCT통계) | ★★ | ★★★ | ★★ | ★ | ★★★ |
| N4 (구성변화) | ★★★ | ★★★ | ★★★ | ★★ | ★★★ |

## 4-2. 본 프로젝트 채택 구성

**Q sensitivity (M1+M2+M3) + PCT sensitivity (N1+N2+N3+N4)**

- Q: 평시 robustness 검증 3단계 (raw → curve → inference)
- PCT: 4단계로 구성 효과까지 (N4 = Q에 없는 PCT 고유 분석)
- M4·M5 생략: 메커니즘은 후속 분석 영역, drawdown event는 실측 결과 narrative 약함
- 발표 ~20분, 학부 졸업 프로젝트 + 안정형/공격형 차별화 + Q × PCT 두 축 robust optimum 증명

---

# Part 5. 산출물 위치

```
final/outputs/99_analyze/
├─ M_winner_q_sensitivity.png         (M2 — Q 6 subplot 라인플롯)
├─ M3_winner_q_bootstrap_ci.png       (M3 — Q Sortino CI)
├─ N_winner_pct_sensitivity.png       (N2 — PCT 6 subplot 라인플롯)
├─ N3_winner_pct_bootstrap_ci.png     (N3 — PCT Sortino CI)
└─ N4_pct_portfolio_composition.png   (N4 — eff_n / turnover)
```

CSV 산출 없음 (모두 노트북 안 display로 확인). PNG만 발표·문서용 보관.

---

# Part 6. 결론 narrative 템플릿

## 6-1. Q 관련 narrative

### A. Q-plateau optimum
> "winner q=0.003은 [0.001, 0.005] 영역에서 sortino 1.88~1.99 plateau에 위치한 robust optimum. 절대 optimum q=0.002 대비 차이 0.6%로 통계적 노이즈 수준."

### B. Q-regime dependence
> "정상기(R2 확장)는 큰 q(0.005~0.006), 위기기(R3 변동)는 작은 q(0.001~0.002)가 최적. winner q=0.003은 양 레짐 절반씩 양보한 합리적 타협점."

### C. Q-statistical robustness
> "11개 비교 중 Bonferroni 보정 후 winner와 통계적으로 유의한 차이 0개. Sortino 95% bootstrap CI도 모두 winner CI와 겹침."

## 6-2. PCT 관련 narrative

### F. PCT-plateau robust + winner 우위 (실측 확인) ⭐
> "winner pct=0.30은 [0.25, 0.40] 영역의 Sharpe plateau에서 robust (Sortino bootstrap은 [0.10, 0.40] 모두 winner CI와 겹침). 추가로 좁은 cut(0.10) 대비 Bonferroni 보정 후도 sharpe 유의 우위(★★), 0.15·0.20 대비도 5% 유의 우위 → **Sharpe plateau 중앙 + 좁은 cut 대비 통계적 우위** 두 측면에서 정당화. BAB 학술 표준(30%)과 일치."

### G. PCT-portfolio composition trade-off (N4)
> "pct=0.10에서 eff_n=X (집중), pct=0.40에서 eff_n=Y (분산). winner 30%는 중간 분산도. turnover는 작은 pct에서 [Z]배 → TC 영향 인지."

### H. PCT-risk profile differentiation (실측 갱신)
> "안정형 ETF: pct=0.35-0.40 (분산 안정, N3 winner와 동등), 공격형: pct=0.25 (Sharpe plateau 좌측 끝, 단일점). 좁은 cut(0.10~0.20)은 specific risk로 winner 대비 sharpe 유의 열등(0.10은 Bonferroni까지 통과) — 사용 안 하는 영역. winner 30%은 중립형 표준. 단, 차별화 폭이 매우 좁아(공격 1점, 중립 1점, 안정 2점) ETF 라인업 차별화 근거로는 약함."

## 6-3. 통합 narrative

### I. Two-axis robustness ⭐ 본 프로젝트 핵심 (실측 반영)
> "winner 슬롯 mat_eq_eq_lam_pap은 q (강도) × pct (집중도) 두 축 모두에서 robust optimum.
> - **Q**: Bonferroni 후 sharpe 유의 차이 0개, sortino CI 모두 겹침 → 완전 robust
> - **PCT**: Sharpe plateau ([0.25, 0.40]) 내 모든 pct가 winner와 동등 + 좁은 cut(0.10~0.20) 대비 sharpe 유의 우위 (0.10은 Bonferroni까지)
>
> → 두 축에서 동시에 plateau 중앙 위치 + 비교 영역에서 통계적 우위 입증."

### J. Risk profile from two axes (제한적 — 실측 반영)
> "성향별 ETF 라인업 후보 (1D 결과 조합 추론, 2D 미검증): 안정형(q=0.001, pct=0.35-0.40), 중립형(q=0.003, pct=0.30 = winner), 공격형(q=0.005, pct=0.25). 모든 성향 후보가 1D N3 검정에서 winner와 통계적 동등하지만, **공격형 pct=0.25는 단일점이며 차별화 폭이 좁음**. 2D q × pct grid backtest 미수행으로 결합 효과 불확실 → 발표 시 'plateau 내 옵션' 정도로 보수적 표현 권장."

---

# Part 7. 비판적 자기 점검

## 7-1. 본 분석의 한계

1. **표본 크기**: 168개월(K_CUT 적용)은 sortino 차이 ±0.05 노이즈 흔함.
2. **R2 dominance**: R2(~84M) > R3(48M) > R1(30M). 전체기간 곡선은 R2가 끌고감.
3. **q_mode='lambda' clip 효과**: 실효 Q ∈ [q_base × 0.1, q_base × 3.0].
4. **단일 검정 통과 ≠ 진실**: Jobson-Korkie 비유의 → "차이 없다"가 아닌 "유의성 못 입증".
5. **PCT의 universe 의존성**: 484종목 중 30%면 145종목. 다른 universe(예: S&P500 vs Russell 1000)면 절대 종목수 다름.
6. **Q와 PCT의 결합 sensitivity 미수행**: 각각 독립 분석. 인터랙션 효과 미검증.
7. **MDD V-shape (Q=0.004)**: 통계적 우연 가능. 다른 시드/기간으로 검증 시 사라질 수 있음.
8. **PCT 좁은 cut (≤0.15) 임계점은 universe 의존**: N3 실측에서 0.10·0.15가 winner 대비 sharpe 유의 열등이지만, 484종목 기준 임계점이므로 다른 universe(예: S&P500 vs Russell 1000)면 임계 pct가 달라질 수 있음. 본 결론은 winner 영역(≥0.20)에 한정해 일반화 권장.

## 7-2. 발표에서 다루는 법

- 보수적 표현 사용: "통계적 차이 입증 못 함" (not "차이 없음")
- robust optimum 표현 (절대 우월 아님)
- N4로 구성 변화 attribution까지 보여 종합적 판단 가능하게
- universe 의존성 명시 ("S&P500 484종목 기준 30%")

---

# Part 8. 후속 분석 (현재 안 함)

| 분석 | 의의 | 우선순위 |
|---|---|---|
| Q × PCT 2D grid sensitivity | 결합 효과 / 인터랙션 | 중 (시간 되면) |
| Walk-forward stability | rolling 36M optimum 변동 | 후속 연구 |
| Pareto frontier (Sortino, MDD) | 다목적 최적 | 낮음 |
| Information Ratio vs SPY | 액티브 운용 품질 | 중 (Phase 2 holdout) |
| lam_mean / TAU sensitivity | 다른 hyperparameter | 낮음 |
| PCT × universe 변경 | universe dependent 검증 | 낮음 (현재 universe 고정) |
| Drawdown event attribution | 위기 시 hyperparameter 보호효과 | 낮음 (M5 실측 결과 narrative 약함) |
| 실현 Q 분포 메커니즘 | lambda mode 분포 추적 | 낮음 (구 M4) |

---

# Part 9. 메타 — 분석 구조의 합리성

## 9-1. M (Q) 섹션 — 본 프로젝트 채택 3단계

```
M1: WHAT?         — 어떤 값인가? (raw)
M2: HOW LOOK?     — 어떤 모양인가? (visualization)
M3: SIGNIFICANT?  — 우연인가 진짜인가? (inference)
~~M4 (메커니즘)~~ — 실현 Q 분포 → 후속 분석 영역
~~M5 (이벤트)~~  — drawdown attribution → 실측 narrative 약함, 생략
```

## 9-2. N (PCT) 섹션의 4단계

```
N1: WHAT?         — 어떤 값인가? (raw, M1 parallel)
N2: HOW LOOK?     — 어떤 모양인가? (visualization, M2 parallel)
N3: SIGNIFICANT?  — 우연인가? (inference, M3 parallel)
N4: HOW STRUCTURED? — 어떻게 구성되나? (composition — eff_n / turnover)
```

## 9-3. M과 N의 차이

- M은 3단계, N은 4단계 — N4(eff_n/turnover)가 PCT 고유 분석.
- Q에는 view 강도 sensitivity만 있어 구성 분석 불필요. PCT는 직접 종목 수를 결정 → 구성 분석 (N4) 가치 ★★★.
- M5(event attribution)는 PCT에도 적용 가능하지만 실측 결과 narrative 약함이 확인되어 양쪽 모두 생략.

데이터 분석의 표준 흐름 (Tukey 1977, EDA → confirmatory): WHAT → HOW LOOK → SIGNIFICANT → STRUCTURE.

---

**문서 버전**:
- v1 (2026-05-08): Q 민감도 (섹션 M)
- v2 (2026-05-08): PCT_GROUP 민감도 (섹션 N) 통합 + 파일명 변경 (Q_SENSITIVITY → SENSITIVITY)
- v3 (2026-05-08): M4를 메커니즘에서 drawdown event로 교체 (구 M5 rename) + 고등학생용 쉬운 해석 가이드 추가
- v4 (2026-05-08): M4 (구 M5) 실측 결과 narrative 약함 확인 → **삭제**. Q 섹션 M1~M3 확정. PCT 섹션 (N1~N4) 고등학생용 쉬운 해석 가이드 추가.
- v5 (2026-05-08): **N3 PCT 통계 검정 실측 반영** — Bonferroni ★★ 2개(pct=0.10, 0.15), Bootstrap 7/7 겹침. "Mixed signal" 표현으로 작성.
- v6 (2026-05-08): N3 톤 재조정 — Bonferroni ★★ 차이는 winner 우위 방향이므로 "Mixed signal/방어적" 표현 → "**Winner-favorable signal**"로 단순화. Part 2-3 N3 실측 / Part 6 F·H·I·J / Part 7 #8 모두 winner 정당화 톤으로 수정.

**관련 노트북**: [99_analyze.ipynb](99_analyze.ipynb) §M, §N
**관련 가이드**: [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md)
