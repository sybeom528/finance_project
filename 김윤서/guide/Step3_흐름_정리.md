# Step 3: 포트폴리오 최적화 (계층형 배분) 흐름 정리

> 파일: `Step3_Portfolio_Optimization.ipynb`  
> 작성자: 김윤서  
> 작성일: 2026-04-19  
> 목적: Step3 전체 흐름 + HRP 알고리즘 설계 결정 근거 문서화

---

## 목차

1. [전체 흐름 개요](#1-전체-흐름-개요)
2. [계층형 2단계 배분 설계 원리](#2-계층형-2단계-배분-설계-원리)
3. [3-1. 투자자 성향 프로파일](#3-3-1-투자자-성향-프로파일)
4. [3-2. Level 1 — 자산군 배분 (블록 대각)](#4-3-2-level-1--자산군-배분-블록-대각)
5. [3-3. Level 2 — 그룹 내 배분](#5-3-3-level-2--그룹-내-배분)
6. [HRP 알고리즘 완전 해부](#6-hrp-알고리즘-완전-해부)
7. [3가지 전략 비교 (MV / RP / HRP)](#7-3가지-전략-비교-mv--rp--hrp)
8. [EDA 시각화 (5개 차트)](#8-eda-시각화-5개-차트)
9. [저장 파일 및 Step 연계](#9-저장-파일-및-step-연계)
10. [주요 설계 결정 요약](#10-주요-설계-결정-요약)

---

## 1. 전체 흐름 개요

```
Step3_Portfolio_Optimization.ipynb
│
├── [설정] 데이터 로드 (Cell 1)
│       portfolio_prices.csv → 로그 수익률 계산
│       2016-01-01~ 슬라이싱 (ANALYSIS_START 기준)
│       자산 그룹 정의: Equity 24개 / Bond 4개 / Alt 2개
│
├── [3-1] 투자자 프로파일 (Cell 3)
│       generate_profile(γ)  → 4개 제약조건 자동 생성
│       get_weight_bounds(γ) → 종목별 비중 상한 역산
│       4가지 사전 프로파일: 보수형(γ=8), 중립형(γ=4), 적극형(γ=2), 공격형(γ=1)
│
├── [3-2] Level 1: 자산군 배분 (Cell 7~9)
│       대표 ETF 3개 (SPY / AGG / GLD)로 3×3 블록 대각 공분산 구성
│       Ledoit-Wolf 수축 추정 → Mean-Variance 효용 극대화
│       → l1_weights: 4 프로파일별 [Equity, Bond, Alt] 비중
│
├── [3-3] Level 2: 그룹 내 배분 (Cell 11~13)
│       Equity (24개): PCA Factor Covariance + MV / RP / HRP
│       Bond   (4개):  Ledoit-Wolf + MV
│       Alt    (2개):  역변동성 비율 (GLD vs DBC)
│
├── [결합] Level 1 × Level 2 (Cell 13)
│       최종 30개 자산 비중 = L1 자산군 비중 × L2 종목 내 비중
│       4 프로파일 × 3 전략 = 12개 포트폴리오
│
├── [시각화] EDA (Cell 5, 9, 15~17)
│       step3_01_profiles.png       — 투자자 프로파일 비교
│       step3_02_level1_allocation.png — Level 1 자산군 배분
│       step3_03_efficient_frontier.png — 효율적 프론티어
│       step3_04_final_weights.png  — 최종 비중 적층 막대
│       step3_05_risk_contribution.png — 위험 기여도 분석
│
└── [저장] (Cell 18)
        data/optimal_weights.csv (360개 레코드: 4×3×30)
        data/profiles.csv        (4개 프로파일 파라미터)
```

---

## 2. 계층형 2단계 배분 설계 원리

### 왜 30×30 직접 최적화를 하지 않는가

```
30×30 공분산 행렬 추정 문제:
  - 추정해야 할 파라미터 수: 30×31/2 = 465개
  - 관측치 수: ~2,491일
  - 비율: 2,491 / 465 ≈ 5배 → "충분하다"처럼 보이지만

Michaud(1989) 연구: MVO는 추정 오차를 극대화하는 경향
  → 공분산이 조금만 틀려도 최적 비중이 극단적으로 바뀜
  → 결과: 2~3개 자산에 과도한 집중, 나머지 0% 배분
```

### 계층형 2단계의 해법

```
Level 1: 3×3 블록 대각 공분산 (SPY, AGG, GLD)
  → 추정 파라미터: 3×4/2 = 6개
  → 추정 오차 최소, 자산군 수준 제약 직접 반영 가능

Level 2: 그룹 내 최적화 (24개, 4개, 2개 분리)
  → 그룹 내 상관이 높아 구조가 명확 → 추정이 상대적으로 안정
  → 그룹별 전문화된 추정 방법 적용 가능
```

| 단계 | 대상 | 방법 | 파라미터 수 |
|------|------|------|----------|
| Level 1 | 3개 대표 ETF | 블록 대각 + Ledoit-Wolf | 6개 |
| Level 2-Equity | 24개 종목 | PCA Factor Covariance | 팩터 수 × 24 + 24 |
| Level 2-Bond | 4개 ETF | Ledoit-Wolf | 10개 |
| Level 2-Alt | 2개 ETF | 역변동성 | 2개 |

---

## 3. 3-1. 투자자 성향 프로파일

### generate_profile(γ) — γ 하나로 4개 제약 자동 생성

```python
t = (gamma - 1) / 9   # [0, 1] 정규화 (γ=1 → t=0, γ=10 → t=1)

target_vol = 0.22 - 0.15 * t   # 목표 변동성: 22% → 7%
max_equity = 0.90 - 0.60 * t   # 최대 주식 비중: 90% → 30%
min_bond   = 0.00 + 0.40 * t   # 최소 채권 비중: 0% → 40%
max_mdd    = -(0.35 - 0.25 * t) # 최대 허용 MDD: -35% → -10%
```

**γ (위험회피계수)의 경제학적 근거 — CRRA 효용함수:**

$$U(W) = \frac{W^{1-\gamma}}{1-\gamma}, \quad \text{최적 비중} = \frac{\mu - r_f}{\gamma \cdot \sigma^2}$$

| γ | 프로파일 | 목표변동성 | 최대주식 | 최소채권 | 최대MDD |
|---|---------|---------|--------|--------|--------|
| 1 | 공격형 | 22% | 90% | 0% | -35% |
| 2 | 적극형 | 20.3% | 77% | 8.9% | -29.4% |
| 4 | 중립형 | 17% | 70% | 13.3% | -26.7% |
| 8 | 보수형 | 10.5% | 43% | 31.1% | -16.7% |

### get_weight_bounds(γ) — MDD 기반 단일 종목 상한 역산

**3단계 논리:**

```
Step 1: MDD 예산의 1/3을 단일 종목에 배정
  → 자산군 3개가 독립적으로 MDD에 기여한다고 가정

Step 2: 극단 충격 시나리오로 역산
  → 종목이 -50% 폭락 시에도 내 MDD 한도 내에 있으려면?
  w_i × 0.50 ≤ max_mdd × (1/3)
  → w_i ≤ max_mdd / 1.5

Step 3: 종목 유형별 승수 + HHI 상한으로 보정
```

| 종목 유형 | 승수 | 근거 |
|---------|------|------|
| 인덱스 ETF (SPY) | 1.4배 완화 | 500종목 내재 분산 — 단일 종목 -50% 사실상 불가 |
| 섹터 ETF (XLK) | 1.1배 완화 | 50~70종목 — 충격 확률 낮음 |
| 개별 종목 (AAPL) | 기본 1.0 | 단일 종목 -50%+ 충분히 가능 |

**HHI 집중도 상한 (이중 안전장치):**

| γ 범위 | HHI 상한 | 의미 |
|--------|---------|------|
| 4~8 (보수·중립) | 15% | 어떤 종목도 15% 초과 금지 |
| 2~3 (적극형) | 20% | |
| 1 (공격형) | 25% | |

**중립형(γ=4) 구체 계산 예시:**

```
max_mdd      = 26.7% (절대값)
raw_max      = (26.7% × 1/3) / 50% = 17.8%
개별 종목 상한 = min(17.8% × 1.0, 15%) = 15%  ← HHI가 바인딩
인덱스 ETF 상한 = min(17.8% × 1.4, 15%) = 15%  ← 동일하게 HHI 바인딩
```

---

## 4. 3-2. Level 1 — 자산군 배분 (블록 대각)

### 대표 ETF 3개 선정

| 대표 ETF | 자산군 | 역할 |
|---------|--------|------|
| **SPY** | Equity | S&P 500 — 전체 주식군 대표 |
| **AGG** | Bond | 미국 종합채권 — 전체 채권군 대표 |
| **GLD** | Alternative | 금 ETF — 대안자산군 대표 |

### 최적화: Mean-Variance 효용 극대화

$$\max_w \left[ w^\top \mu - \frac{\gamma}{2} w^\top \Sigma w \right]$$

**제약조건:**
- $\sum w_i = 1$
- $w_\text{equity} \leq \text{max\_equity}$ (프로파일 제약)
- $w_\text{bond} \geq \text{min\_bond}$ (프로파일 제약)

**공분산 추정 — Ledoit-Wolf 수축:**

```python
l1_cov = LedoitWolf().fit(l1_ret).covariance_ * 252
```

3×3 블록이므로 추정 파라미터가 6개에 불과 → 표본 공분산만으로도 충분하지만,  
Ledoit-Wolf 적용으로 조건수(condition number)를 추가 개선해 수치 안정성 확보.

---

## 5. 3-3. Level 2 — 그룹 내 배분

### 그룹별 공분산 추정 방법

| 그룹 | 종목 수 | 공분산 추정 | 이유 |
|------|--------|-----------|------|
| Equity | 24개 | **PCA Factor Covariance** | 종목 수 대비 팩터 구조가 명확, 노이즈 필터링 필요 |
| Bond | 4개 | **Ledoit-Wolf** | 적은 종목 수 → 수축 추정만으로 충분 |
| Alt | 2개 | **역변동성 비율** | 2개 자산 → 최적화보다 단순 비율이 견고 |

### PCA Factor Covariance (Equity 그룹)

**왜 표본 공분산이 아닌가:**

```
24×24 표본 공분산의 문제:
  - 조건수(condition number)가 수천~수만 → 역행렬 계산 불안정
  - 노이즈 고유벡터가 최적화에 영향 → 극단적 비중 발생
```

**PCA 팩터 공분산 구조:**

$$\Sigma_\text{factor} = B \cdot \Sigma_F \cdot B^\top + \text{diag}(\sigma^2_\epsilon)$$

- $B$ (N×K): 팩터 로딩 행렬 (각 종목의 팩터 노출도)
- $\Sigma_F$ (K×K): 팩터 간 공분산
- $\text{diag}(\sigma^2_\epsilon)$: 잔차(고유) 분산 — 종목별 독립 위험

**팩터 수 자동 선택:**

```python
def select_n_factors(returns, min_var=0.80, max_k=10):
    # 누적 설명분산 80% 이상 되는 최소 팩터 수 (최소 3개)
```

```
결과: Equity 24개 종목 → PCA 팩터 k개 선택
      표본 공분산 조건수: 수천
      PCA 팩터 공분산 조건수: 수십~수백 (수십 배 개선)
```

---

## 6. HRP 알고리즘 완전 해부

> **참고**: Lopez de Prado (2016), "Building Diversified Portfolios that Outperform Out-of-Sample"

HRP는 **역공분산 행렬을 전혀 사용하지 않는** 포트폴리오 최적화 알고리즘입니다.  
전통적 MVO의 역행렬 추정 오차 문제를 근본적으로 우회합니다.

### HRP 3단계 절차

```
Step 1. 상관 거리 행렬 구성
         ↓
Step 2. 계층적 클러스터링 → 자산 재정렬 (준대각화)
         ↓
Step 3. 재귀적 이분할 역분산 배분
```

---

### Step 1. 상관 거리 행렬 구성

**목적**: 상관관계를 '거리'로 변환해 유사한 자산끼리 가깝게 묶는다.

$$d_{ij} = \sqrt{\frac{1 - \rho_{ij}}{2}}$$

```python
corr = returns.corr()
dist = np.sqrt((1 - corr) / 2)
np.fill_diagonal(dist.values, 0)
```

**이 공식이 올바른 거리인 이유:**

| 상관계수 ρ | 거리 d | 의미 |
|----------|-------|------|
| ρ = +1.0 | d = 0 | 완전 동일 움직임 → 거리 0 |
| ρ = 0.0 | d = 0.707 | 무관계 |
| ρ = -1.0 | d = 1.0 | 완전 반대 움직임 → 거리 최대 |

> `(1 - ρ) / 2` 로 나누는 이유: ρ ∈ [-1, 1] → d ∈ [0, 1] 범위로 정규화.  
> 단순히 `1 - ρ`를 쓰면 d ∈ [0, 2] 이 되어 삼각 부등식을 위반할 수 있다.

---

### Step 2. 계층적 클러스터링 → 준대각화 (Quasi-Diagonalization)

**Ward 연결법(Linkage)으로 군집 형성:**

```python
link = linkage(squareform(dist.values), method='ward')
sort_idx = leaves_list(link)
sorted_tickers = [returns.columns[i] for i in sort_idx]
```

**Ward 연결법이란:**  
두 클러스터를 합칠 때 **합친 후의 클러스터 내 분산 증가량이 최소**가 되는 쌍을 선택.  
→ 각 클러스터가 내부적으로 가장 균일(compact)하게 유지됨.

**준대각화(Quasi-Diagonalization):**

```
클러스터링 전 공분산 행렬:
  [SPY  XLK  TLT  AGG  GLD  XOM  ...]  ← 임의 순서

클러스터링 후 재정렬 (leaves_list):
  [XLK  XLY  AAPL  MSFT  AMZN  |  SPY  QQQ  |  TLT  AGG  |  GLD  DBC]
   ─── 기술/성장 클러스터 ────   ─ 광역주식 ─  ── 채권 ──   ─ 대안 ─
```

**왜 재정렬이 필요한가:**

```
재정렬 후 공분산 행렬은 '준대각(quasi-diagonal)' 구조를 가짐
→ 비슷한 자산끼리 서로 인접 → 블록 구조가 대각선을 따라 집중
→ 이후 이분할 시 "왼쪽 = 유사 자산끼리, 오른쪽 = 유사 자산끼리" 분리됨
→ 각 분할이 의미있는 자산군을 형성
```

---

### Step 3. 재귀적 이분할 역분산 배분 (Recursive Bisection)

**핵심 아이디어**: 재정렬된 자산 목록을 절반씩 쪼개면서, 각 절반에 분산 역수 비율로 가중치를 배분한다.

```python
def bisect(items):
    if len(items) <= 1:
        return
    mid = len(items) // 2
    left, right = items[:mid], items[mid:]

    var_left  = cov.loc[left, left].values
    var_right = cov.loc[right, right].values

    inv_l = 1 / np.diag(var_left).sum()   # 왼쪽 역분산
    inv_r = 1 / np.diag(var_right).sum()  # 오른쪽 역분산
    alloc = inv_l / (inv_l + inv_r)       # 왼쪽 배분 비율

    weights[left]  *= alloc
    weights[right] *= (1 - alloc)

    bisect(left)   # 재귀
    bisect(right)  # 재귀
```

**단계별 시각적 예시 (8개 자산 기준):**

```
초기: [A  B  C  D  E  F  G  H]  모두 가중치 = 1.0
      ├─ 왼쪽 [A B C D]         오른쪽 [E F G H]
      │   var_L=0.5, var_R=1.0
      │   inv_L=2, inv_R=1
      │   alloc_L = 2/(2+1) = 0.667
      │   alloc_R = 1/(2+1) = 0.333
      │
      ├─ 왼쪽 [A B C D]: 각 1.0 × 0.667 = 0.667
      │   ├─ [A B]               [C D]
      │   │   var_AB=0.4, var_CD=0.6
      │   │   alloc_AB = 0.6/(0.6+0.4) = 0.6
      │   │   A, B: 0.667 × 0.6 / 2 = 0.200
      │   │   C, D: 0.667 × 0.4 / 2 = 0.133
      │
      └─ 오른쪽 [E F G H]: 각 1.0 × 0.333 = 0.333
          └─ ... (동일하게 반복)
```

**역분산 비율의 의미:**

$$\text{alloc}_\text{left} = \frac{1 / \text{Var}_\text{left}}{1/\text{Var}_\text{left} + 1/\text{Var}_\text{right}}$$

- 왼쪽 분산이 작을수록 → 왼쪽 역분산이 크다 → 왼쪽에 더 많이 배분
- "변동성이 낮은 쪽에 더 많은 자본을 배분" = **위험 균등화**의 직접적 표현

**코드에서 `np.diag(var).sum()`을 쓰는 이유:**

```
이상적으로는 클러스터 포트폴리오 분산을 써야 함:
  var_cluster = w^T Σ_cluster w  (역분산 비중으로 구성)

그러나 이는 클러스터 내 재귀 호출이 필요해 복잡도 증가
→ 단순화: 대각 원소 합 (개별 분산의 합) 으로 근사
→ 클러스터 내 상관이 높으면 근사 오차 작음 (이미 클러스터링으로 보장)
```

---

### HRP 전체 흐름 요약

```
수익률 DataFrame
     │
     ▼
[Step 1] 상관 거리 행렬: d = √((1-ρ)/2)
     │
     ▼
[Step 2] Ward 계층적 클러스터링 → dendrogram
         leaves_list()로 준대각 순서 추출
     │
     ▼
[Step 3] 재귀적 이분할
         각 분할에서 역분산 비율로 가중치 배분
     │
     ▼
최종 weights (합 = 1.0)
```

### HRP가 MVO·RP보다 나은 상황

| 상황 | MVO | RP | HRP |
|------|-----|-----|-----|
| 공분산 추정 오차가 클 때 | 취약 (역행렬 오차 증폭) | 보통 | **강건** (역행렬 불사용) |
| 자산 간 클러스터 구조가 뚜렷할 때 | 무시 | 무시 | **활용** |
| 기대수익률 추정 불확실 | 취약 | 불필요 | **불필요** |
| 볼록 최적화 수렴 문제 | 발생 가능 | 발생 가능 | **없음** (휴리스틱) |

### HRP의 한계

```
1. 비중 상한(bounds) 제약을 직접 반영하지 않음
   → 이 코드에서는 Level 2 내에서 HRP를 쓰고
     Level 1 비중으로 곱하는 방식으로 간접 제어

2. 클러스터 분할이 절반(mid = len//2)으로 고정
   → 자산 수가 홀수이면 비대칭 분할 발생

3. 분산 근사 (대각 합)가 공분산 구조를 일부 무시
   → 클러스터 내 상관이 낮으면 오차 증가
```

---

## 7. 3가지 전략 비교 (MV / RP / HRP)

### 전략별 특성 비교

| 항목 | Mean-Variance (MV) | Risk Parity (RP) | HRP |
|------|-------------------|----------------|-----|
| **목적함수** | $U = w^\top\mu - \frac{\gamma}{2}w^\top\Sigma w$ | 동일 위험 기여도 | 재귀적 역분산 |
| **기대수익률 필요** | O | X | X |
| **역공분산 필요** | O | O (공분산만) | X |
| **볼록 최적화** | O (SLSQP) | O (SLSQP) | X (휴리스틱) |
| **추정 오차 민감도** | 높음 | 중간 | 낮음 |
| **비중 상한 제약** | 직접 반영 | 직접 반영 | 간접 반영 (L1 곱) |
| **해석 용이성** | 중간 | 높음 | 높음 |

### RP 알고리즘 보충

```python
def risk_budget_obj(w):
    port_vol = np.sqrt(w @ cov @ w)
    mrc = cov @ w / port_vol     # 한계 위험 기여도 (Marginal Risk Contribution)
    rc  = w * mrc                 # 위험 기여도 (Risk Contribution)
    target_rc = port_vol / n      # 목표: 모든 종목 동일 기여
    return np.sum((rc - target_rc) ** 2)
```

**RC(Risk Contribution)의 의미**: 종목 i가 포트폴리오 전체 변동성에 기여하는 양.

$$RC_i = w_i \cdot \frac{\partial \sigma_P}{\partial w_i} = w_i \cdot \frac{(\Sigma w)_i}{\sigma_P}$$

RP는 이 값을 모든 종목이 동일하게 갖도록 비중을 결정.

---

## 8. EDA 시각화 (5개 차트)

| 파일명 | 내용 | 주요 포인트 |
|--------|------|-----------|
| `step3_01_profiles.png` | 4개 프로파일 파라미터 비교 | γ↑ 시 변동성↓, 채권↑, MDD↑(완화) |
| `step3_02_level1_allocation.png` | Level 1 자산군 배분 적층 막대 | 공격형 Equity 90% → 보수형 40% |
| `step3_03_efficient_frontier.png` | 중립형(γ=4) Equity 효율적 프론티어 | MV 최적점, RP·HRP 포지션 표시 |
| `step3_04_final_weights.png` | 4 프로파일 × MV 전략 최종 비중 | 자산군별 색상 구분 적층 막대 |
| `step3_05_risk_contribution.png` | 4 프로파일 위험 기여도 분석 | Equity가 위험의 ~80% 기여 |

---

## 9. 저장 파일 및 Step 연계

### 저장 파일 목록

| 파일 | 크기 | 내용 |
|------|------|------|
| `data/optimal_weights.csv` | 360개 레코드 | 4 프로파일 × 3 전략 × 30 종목 비중 |
| `data/profiles.csv` | 4개 레코드 | γ, target_vol, max_equity, min_bond, max_mdd, L1 비중 |

### optimal_weights.csv 컬럼

```
profile  | gamma | strategy | ticker | weight
보수형   |  8    | MV       | SPY    | 0.0423
보수형   |  8    | MV       | QQQ    | 0.0318
...      |  ...  | RP       | SPY    | 0.0512
...      |  ...  | HRP      | SPY    | 0.0389
```

### Step 간 연계

```
Step3 입력                     →  사용 방식
─────────────────────────────────────────────────────
portfolio_prices.csv          →  log_returns 계산
                                 2016-01-01~ 슬라이싱
─────────────────────────────────────────────────────
Step3 출력                     →  Step4+ 사용처
─────────────────────────────────────────────────────
optimal_weights.csv           →  Step6 백테스트: 비중 × 수익률
profiles.csv                  →  Step7 대시보드: 투자자 프로파일 매핑
                                  Step5 BL: 사전 비중(π) 역산에 활용
─────────────────────────────────────────────────────
주의: 이 파일은 10년 전체 데이터 기반 정적 최적화
      walk-forward 방식은 Step6에서 별도 구현 필요
```

---

## 10. 주요 설계 결정 요약

| 결정 항목 | 이전 방식 | 최종 방식 | 근거 |
|---------|---------|---------|------|
| 최적화 구조 | 30×30 직접 MVO | **계층형 2단계** (L1→L2) | 추정 오차 최소화, 자산군 제약 직접 반영 |
| Level 1 공분산 | 표본 공분산 | **Ledoit-Wolf 수축** | 3×3에도 수치 안정성 추가 확보 |
| Equity 공분산 | 표본 24×24 | **PCA Factor Covariance** | 조건수 수십 배 개선, 노이즈 필터링 |
| Bond 공분산 | 표본 4×4 | **Ledoit-Wolf** | 4개 종목 → 수축 추정으로 충분 |
| Alt 배분 | MVO | **역변동성 비율** | 2개 자산 → 최적화보다 단순 비율이 더 견고 |
| HRP 공분산 | PCA Factor | **표본 공분산** | HRP는 역행렬 불사용 → 추정 정확도보다 안정성이 우선 |
| 투자자 제약 | 단일 전역 상한 | **γ 기반 MDD 역산 + HHI** | 프로파일별 일관성 자동 보장 |
| 전략 수 | MVO 단일 | **MV / RP / HRP 3종** | 추정 오차 민감도가 다른 전략 비교 → Step6에서 성과 검증 |

---

> **참고 문서**  
> - `서윤범/project_design_v3.md` — 전체 7단계 파이프라인 아키텍처  
> - `김윤서/Step1_흐름_정리.md` — 데이터 수집 설계  
> - `김윤서/Step2_흐름_정리.md` — 전처리 + 피처 엔지니어링  
> - Lopez de Prado (2016), "Building Diversified Portfolios that Outperform Out-of-Sample"
