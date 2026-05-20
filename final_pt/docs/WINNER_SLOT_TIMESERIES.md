# Winner BL 슬롯 시계열 추이 — 상세 해석

> **대상**: `mat_eq_eq_raw_pap` (05b §4.2 자동 식별 winner)
> **기간**: 2010-01 ~ 2025-12 (192개월)
> **시각화**: `final_pt/outputs/05b_Analyze/winner_slot_dynamics.png` (4 패널)
> **노트북**: [05b_Analyze.ipynb](../05b_Analyze.ipynb) §8

---

## 1. Black-Litterman 프레임워크 복습

BL 사후분포 공식:

```
μ_BL = π + τΣPᵀ · (Q − Pπ) / (PτΣPᵀ + Ω)
        ↑      ↑                ↑
      prior  관측-기대          분산 가중
```

각 슬롯의 역할:

| 슬롯 | 역할 | Winner 모드 |
|---|---|---|
| **π** (prior) | view 없을 때 균형 기대수익 | `capm_eq` (등가중 시장) |
| **P** | view 행렬 (어느 자산에 거나) | `pw=eq` (저변동 30% long, 고변동 30% short, 동일가중) |
| **Q** | view 강도 (얼마나 벌 것 예측) | `raw_lam` (자연 게이팅) |
| **Ω** | view 분산 (얼마나 불확실) | `ff3_paper` (적응형) |

---

## 2. Winner 의 BL 파라미터 (BASELINE)

| 파라미터 | 값 | 의미 |
|---|---:|---|
| `q_base` | 0.003 | Q 기본 강도 (월 0.3%) |
| `lam_mean` | 2.5 | raw_lam Q 정규화 기준 |
| `lam_fixed` | 2.5 | prior `π = λ·Σ·w_mkt` 의 λ |
| `pct_group` | 0.30 | P 분류 컷오프 (저/고 각 30%) |
| `τ` (tau) | 0.1 | He-Litterman scale |
| `tc` | 0.002 | 편측 거래비용 (20bp/side) |
| `max_weight` | 0.10 | 단일 종목 상한 |

---

## 3. Panel 1: Q (view 강도) 추이

### 공식

```
Q = max(0, q_base · λ_raw / lam_mean)
  = max(0, 0.003 · λ_raw / 2.5)
where λ_raw = spy_excess / σ²_mkt   (clip 없음, 음수 가능)
```

### 측정값 (192개월)

| 통계 | 값 | 해석 |
|---|---:|---|
| 평균 | **0.0073** | 월별 평균 view 강도 0.73% spread 예측 |
| std | 0.0052 | 변동성 큼 (평균의 약 70%) |
| min | **0.0000** | Q=0 자연 게이팅 발생 |
| max | **0.0197** | base (0.003) 의 약 6.6배 폭증 |

### 추이 — 발생 시점 분석

#### Q = 0 구간 (9개월, 모두 2010 초기)

- 2010-01, 02, 03, 06, 07 등
- **원인**: SPY excess return < 0 (월별 SPY 수익이 rf 미달)
- **메커니즘**: λ_raw = spy_excess / σ²_mkt 가 음수 → `max(0, neg) = 0`
- **결과**: BL 식에서 `Q − Pπ ≈ -Pπ` → view 차단, μ_BL ≈ π (prior 만 사용)

> 💡 raw_lam 의 핵심 특성: **하드 stop 없이 자연 게이팅** — SPY 하락 시 자동으로 anomaly view 꺼짐

#### Q 폭증 Top 5 (모두 2017-2018)

| 시점 | Q 값 |
|---|---:|
| 2018-01 | 0.0197 |
| 2017-12 | 0.0196 |
| 2018-02 | 0.0196 |
| 2017-11 | 0.0191 |
| 2017-06 | 0.0188 |

- **원인**: σ²_mkt 가 역사적 저점 (VIX ~10 수준의 저변동성 시기)
- **메커니즘**: 분모 σ²_mkt 폭락 → λ_raw 폭등 → Q 폭증
- **결과**: 그 달 BL view 강하게 채택, 저변동 종목 가중치 ↑

> ⚠️ 주의: Q 폭증은 **위기 시 아닌 안정 시장** 에서 발생. raw_lam 은 시장 *안정도* 시그널

### 시계열 narrative

```
2010 초반: Q=0 자주 (회복 초기 변동 시기)
2011-2016: Q 점진 상승, base (0.003) 부근
2017-2018: Q 폭증 구간 (저변동성 안정기)
2019-2020: COVID 전후 변동
2020-2024: 변동성 회복, Q 진동
2025: K_CUT 이후 hold-out 구간
```

---

## 4. Panel 2: Ω (view 분산) 추이 [재계산]

### 공식

```
Ω_t = max((Q_{t-1} − P_{t-1}·r_t)², 1e-8)
              ↑              ↑
           예측치       실제 발생 P-spread
```

`P_{t-1}·r_t` 의 의미:
- `P_{t-1}` = 직전월 view 행렬 (저변동 +1/n, 고변동 -1/n)
- `r_t` = 당월 실현 수익률
- 곱 = **저변동 그룹 평균 r − 고변동 그룹 평균 r** = 실현 spread

### 재계산 방법

[bl_runner.py L380-391](../lib/bl_runner.py) 와 동일 로직 재현:
1. 매월 `vol_series = vol_21d + LSTM vol_pred overlay` 정렬
2. bottom 30% → P[low] = +1/n_g, top 30% → P[high] = -1/n_g
3. `actual_p_t = month_df['ret_1m'] @ P_prev` 계산
4. `Ω_t = max((Q_prev − actual_p_t)², 1e-8)`

→ pkl 없이 `monthly_panel.csv` + `ensemble_predictions_stockwise.csv` + `meta['Q']` 만으로 재현.

### 측정값 (191개월, 첫 달 skip)

| 통계 | 값 | 해석 |
|---|---:|---|
| 평균 | **0.0020** | 평균 예측오차² 0.002 |
| std | 0.0034 | 큰 변동 |
| min | **1e-8 (floor)** | 예측 거의 정확한 달들 |
| max | **0.0270** | 예측 크게 빗나간 달 |

log scale 로 시각화 — 예측오차가 **4~5 자릿수 범위** 변동.

### 추이 — 폭증 시점 분석

#### Ω Top 5 (모두 회복랠리·변동 구간)

| 시점 | Ω | Q_{t-1} | actual_p | 해석 |
|---|---:|---:|---:|---|
| 2020-11 | 0.0270 | 0.0072 | **-0.1571** | COVID 회복랠리 (백신 발표) |
| 2023-01 | 0.0196 | 0.0038 | **-0.1361** | 베어 회복 (AI 랠리 시작) |
| 2011-10 | 0.0170 | 0.0003 | **-0.1299** | EU 위기 후 반등 |
| 2021-02 | 0.0159 | 0.0074 | **-0.1185** | 회복 후 변동 |
| 2011-09 | 0.0126 | 0.0009 | +0.1132 | (반대 방향, +) |

**공통 패턴 (1~4번)**: `actual_p` 가 큰 **음수** (-10% ~ -16%)
- 의미: 저변동 그룹 평균이 고변동 그룹 평균보다 *10%+ 적게* 벌었음
- 즉 **P 뷰 (저변동 long, 고변동 short) 가 정반대로 작동**
- 회복랠리에서 고변동 종목 (회복 빔주, 성장주) 폭등 → anomaly 실패

#### 메커니즘

```
회복랠리 발생:
  → 고변동 종목 폭등
  → P_{t-1}·r_t = 저 - 고 = 큰 음수
  → (Q_{t-1} - actual_p)² = (0.007 - (-0.157))² = 큰 값
  → Ω_t 폭증
  → 다음 BL 가 view 영향력 감쇠 (분모 키움)
  → μ_BL 이 π 쪽으로 회귀
  → 다음 달 anomaly tilt 약화
```

> 💡 ff3_paper 의 핵심: **직전월 예측오차로 다음월 신뢰도 자동 조정** (Bayesian 학습)

### 시계열 narrative

```
2010-2011: 초기 변동 (Ω 진동)
2011-09~10: EU 위기 후 반등 → Ω 폭증
2012-2019: 안정 강세장 → Ω 낮음 (예측 잘 맞음)
2020-03: COVID 폭락 → Ω 큰 변동
2020-11: COVID 회복랠리 → Ω 최대 (백신 빔주 폭등)
2021-02: 후속 변동
2022: 베어 진입 (변동)
2023-01: 베어 복구 + AI 랠리 시작 → Ω 폭증
2024-2025: 안정화 (K_CUT 이후)
```

---

## 5. Panel 3: P 그룹 포트폴리오 노출 추이

### ⚠️ 측정값이 P 자체가 아닌 이유

| | P (BL 입력) | low_weight / high_weight (측정값) |
|---|---|---|
| 무엇? | view 행렬 | **MVO 최종 포트폴리오 가중치 합** |
| 부호 | +1/n (long), -1/n (short) | 0~1 (long-only 제약) |
| 합 | 0 | < 1.0 (cash 가능) |
| 시점 | BL 입력 단계 | BL + MVO 출력 후 |

`low_weight_t` 정의:
```
low_weight_t = Σ w[i]_t   for i in (vol 정렬) 하위 30% 그룹
            = "월 t 의 최종 포트폴리오 중 저변동 그룹 비중"
```

### 측정값 (192개월)

| 항목 | 평균 | std | min | max |
|---|---:|---:|---:|---:|
| **Low-vol 그룹** | **54.2%** | 20.3% | 18.7% | **100%** |
| **High-vol 그룹** | **11.9%** | 9.9% | **0%** | 30.0% |
| Net tilt (Low − High) | **42.3%p** | — | — | — |

### 추이 — Low-vol = 100% 도달 시점 Top 5

| 시점 | low | high | 시장 환경 |
|---|---:|---:|---|
| 2018-11 | 1.000 | 0.000 | 강세장 안정기 |
| 2017-08 | 0.985 | 0.000 | 저변동성 시기 (Q 폭증 인접) |
| 2012-03 | 0.981 | 0.006 | EU 위기 후 회복 |
| 2019-02 | 0.971 | 0.000 | 강세장 |
| 2016-02 | 0.969 | 0.000 | 변동 회복 |

### 해석

**low_weight = 100% 의 의미**:
- 포트폴리오 전체 30~50개 종목이 저변동 30% 그룹 안에서만 선택됨
- 중간 40% + 고변동 30% 그룹 **완전 회피**
- "BL view 의 *극단적 채택*" — anomaly tilt 가 max 까지 도달

**왜 그렇게 됨**:
1. 안정 강세장 → Q 큼 (Panel 1 의 2017-2018 와 일치)
2. μ_BL 이 저변동 종목 강하게 ↑, 고변동 강하게 ↓
3. MVO + long-only → 고변동 가중치 강제 0, 저변동에 max_weight 까지 몰빵
4. 결과: 저변동 100%

**Net tilt 42%p 의 의미**:
> "이 포트폴리오는 평균적으로 시장 (등가중) 대비 저변동 그룹에 42%p 더 노출되어 있다"

→ winner 의 **지속적·일관된 저변동 anomaly tilt** 입증.

---

## 6. Panel 4: Turnover 추이

### 정의

```
turnover_t = Σ |w_new[i] − w_prev[i]|   for all i
```

### 측정값

| 통계 | 값 | 해석 |
|---|---:|---|
| 평균 | **0.99** | 매월 100% two-way (약 50% 종목/가중치 교체) |
| std | 0.64 | 큰 변동 |
| min | 0.00 | 첫 달 (이전 포지션 없음) |
| max | **1.93** | 거의 완전 교체 |

### 거래비용 계산

```
TC_t = turnover_t × tc(per-side)
     = 0.99 × 0.002
     ≈ 0.2%/월
     × 12 = 2.4%/년
     = 연 240bp 거래비용
```

→ winner 의 **net return 에 이미 baked-in** (gross_ret - turnover × tc = net_ret)

### 학술 baseline 비교

**Frazzini & Pedersen (2014) "Betting Against Beta"**: BAB 팩터 평균 월간 turnover ≈ 1.2

우리 winner 0.99 → **학술 baseline 의 약 83%** 수준 → 운용 현실성 확인.

### 추이

- 폭증 (>1.5) 시점: vol_pred 급변 구간 (위기 진입·회복)
- 안정 (~0.5) 시점: 강세장 후반
- COVID 2020 직전·직후 폭증

---

## 7. Q–Ω 상관 분석 (cross-panel)

### 측정

| 변수 쌍 | Pearson r | p-value | 유의성 |
|---|---:|---:|---|
| Q_{t-1} vs Ω_t | **−0.123** | 0.089 | p > 0.05 (유의 X) |

### 해석

**약한 음의 상관**:
- Q 가 큰 달에 다음 Ω 가 *작은* 경향 (약함)
- 단순 "자기조절 (Q ↑ → Ω ↑)" 가설 **데이터로 입증 안 됨**

**진짜 동학**:
- Ω 는 Q 크기보다 **`actual_p` 부호·크기** 에 강하게 의존
- 핵심 변수: 예측오차 (Q − actual_p)
- Q 가 작아도 actual_p 가 크게 빗나가면 Ω 폭증 (예: 2011-09: Q_prev=0.0009, actual_p=+0.1132 → Ω=0.013)

> 💡 ff3_paper 적응형 Ω 의 본질: "Q 의 크기" 가 아닌 "P 뷰 실현 방향" 의 학습

---

## 8. 발표 활용 가이드

| 슬라이드 | 활용 포인트 |
|---|---|
| **Methodology** | "BL 입력 4 슬롯 모두 동적 — 단순 정적 룰 아님" 시각적 근거 |
| **raw_lam 자연 게이팅** | Panel 1 Q=0 9개월 (2010 초기) — 하드스톱 없이 자동 차단 |
| **저변동성 시장 → Q 폭증** | Panel 1 의 2017-2018 — σ²_mkt 분모 효과 (직관 반대) |
| **ff3_paper 적응형 Ω** | Panel 2 + Ω Top 5 표 — 회복랠리에서 anomaly 실패 → 자동 학습 |
| **Robustness** | Panel 3 Low 평균 54%, net tilt 42%p — winner 의 지속적 anomaly tilt |
| **TC realism** | Panel 4 turnover 0.99 → FP 2014 BAB ~1.2 와 동급 수준 |
| **Q&A 백업** | Ω Top 시점 = 회복랠리 실패 → §3 R4 hold-out narrative 와 정합 |

---

## 9. R4 Hold-out (§3, 06) 과의 연결

§8 의 Ω 분석이 **§3 의 R4 hold-out underperform narrative** 와 자연스럽게 연결:

```
§8 Panel 2 발견:
  Ω 폭증 시점 = 회복랠리에서 저변동 anomaly 실패
                (고변동 종목 outperform)

§3/§6 R4 hold-out 결과:
  2024-2025 AI 랠리에서 winner < SPY (sortino gap -1.79)
  → 저변동 anomaly cyclical 한계

→ 같은 메커니즘:
  Frazzini-Pedersen (2014) §5 명시:
  "고변동 모멘텀 강세장 후반에 저변동 전략 underperform"
```

§8 Ω = 매월 단위 anomaly 실패 진단
§3/§6 R4 = 2년 단위 anomaly cycle 진단
**두 분석이 같은 학술 한계를 다른 시간 척도로 입증.**

---

## 10. 미표시 슬롯 (FAQ 대응)

### λ (lam_fixed = 2.5 상수)

- `compute_pi(lam_fixed=2.5)` 에서 prior `π = λ·Σ·w_mkt` 의 λ
- 상수 → 시각화 의미 없음
- 단, raw_lam Q 공식 내부의 `λ_raw = spy_excess/σ²_mkt` 는 *동적* 이지만, Q 값 자체에 흡수되어 Panel 1 에서 확인 가능

### prior (capm_eq 등가중)

- `w_mkt = 1/n_t` (등가중)
- n_t 만 시점별 변동 (가용 종목 수 변동)
- 시각화 의미 약함 — 매월 거의 동일 구조

---

## 11. 재현 방법

```python
import pickle, pandas as pd, numpy as np
from lib.bl_runner import load_lstm_pred
from lib.bl_config import BASELINE

WINNER = 'mat_eq_eq_raw_pap'

# 1. pkl 로드 (Q, P 그룹 효과, Turnover)
with open(f'results/{WINNER}.pkl', 'rb') as f:
    d = pickle.load(f)
meta, comp = d['meta'], d['comp']

# 2. Ω 재계산 (필요 시)
panel = pd.read_csv('data/monthly_panel.csv', parse_dates=['date']).set_index(['date','ticker'])
lstm_state = load_lstm_pred(BASELINE['lstm_pred_path'], pred_dates_full)

# (이후 §8 cell 37 코드와 동일 로직)
```

전체 실행 가능 코드: [05b_Analyze.ipynb](../05b_Analyze.ipynb) §8 cell 37.

---

**작성일**: 2026-05-13
**버전**: §8 final (Q 수식 정정, Ω 재계산 추가, narrative 실측 검증 후)
**관련 문서**:
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — 전체 파이프라인
- [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md) — 슬롯 정의 + 수식
- [SENSITIVITY_ANALYSIS.md](SENSITIVITY_ANALYSIS.md) — §5·§6 Q·PCT 민감도
- [Exploiting_LowRisk_Anomaly_BL_Summary.md](Exploiting_LowRisk_Anomaly_BL_Summary.md) — Pyo & Lee 2018 학술 reference
