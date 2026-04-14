## 프로젝트 전체에서 사용된 ML/통계 기법 정리

---

### 한눈에 보기

| Step | 기법 | 분류 | 목적 |
|------|------|------|------|
| Step2 | **Granger 인과 검정** | 통계 검정 | 대안데이터의 선행성 입증 |
| Step3 | **PCA (주성분 분석)** | 비지도 ML (차원 축소) | 공분산 행렬 안정화 |
| Step3 | **Ledoit-Wolf 축소 추정** | 통계 추정 | 공분산 노이즈 제거 |
| Step3 | **Mean-Variance 최적화** | 수리 최적화 | 포트폴리오 비중 결정 |
| Step3 | **Risk Parity** | 수리 최적화 | 동일 리스크 기여 배분 |
| Step3 | **HRP (계층적 클러스터링)** | 비지도 ML (클러스터링) | 역행렬 없는 강건 배분 |
| Step4 | **Walk-Forward 교차검증** | 모델 검증 | 과적합 방지, 시계열 IS/OOS 분리 |
| Step5 | **Historical VaR/CVaR** | 비모수 통계 | 극단 손실 추정 |
| Step5 | **Parametric VaR/CVaR** | 모수 통계 (정규분포) | 극단 손실 추정 (분포 가정) |
| Step6 | **Hidden Markov Model** | 비지도 ML (상태 모델) | 시장 레짐 분류 |
| Step6 | **BIC 모델 선택** | 통계 모델 선택 | 최적 레짐 수 결정 |
| Step6 | **StandardScaler** | 전처리 | HMM 입력 변수 표준화 |
| Step7 | **Bootstrap 재표본 검정** | 비모수 통계 검정 | Sharpe 차이 유의성 |

---

### 기법별 상세

#### 1. Granger 인과 검정 (Step2)

```
분류: 시계열 통계 검정 (F-검정 기반)
라이브러리: statsmodels.tsa.stattools.grangercausalitytests

질문: "X의 과거 값이 Y의 미래 값 예측에 통계적으로 유의한 추가 정보를 제공하는가?"

구체적 적용:
  Y = rv_neutral (포트폴리오 변동성)
  X = 43개 후보 변수 (외부 지표 + 대안데이터 파생 변수)
  lag = 1~10일 (각 변수별 최적 lag 자동 선택)

  내부 원리:
    제한 모형:  Y_t = α + β₁Y_{t-1} + ... + βₚY_{t-p} + ε   (Y의 과거만)
    비제한 모형: Y_t = α + β₁Y_{t-1} + ... + βₚY_{t-p}
                     + γ₁X_{t-1} + ... + γₚX_{t-p} + ε   (Y + X의 과거)
    F-검정: "γ₁=γ₂=...=γₚ=0" (X가 아무 정보도 없다)을 귀무가설로 검정
    p-value < 0.05이면 "X가 Y를 Granger-cause한다" (선행 예측력 있음)

결과: 43개 중 34개 유의. Top: HY_spread_chg (p=7.2e-65)
```

#### 2. PCA — 주성분 분석 (Step3)

```
분류: 비지도 ML (차원 축소)
라이브러리: sklearn.decomposition.PCA

질문: "24개 주식 자산의 움직임을 몇 개의 공통 원인으로 설명할 수 있는가?"

구체적 적용:
  입력: 24개 주식의 일별 수익률 행렬 (T×24)
  출력: K개 팩터 (T×K, K=4 자동 선택)

  내부 원리:
    수익률 행렬의 공분산을 고유값 분해(eigendecomposition)
    → 가장 큰 고유값에 대응하는 고유벡터 = 제1주성분 (시장 전체 움직임)
    → 제2주성분 = 시장 제거 후 가장 큰 변동 (금리 민감도 등)
    → K개 주성분으로 원래 24×24 공분산을 복원

  효과:
    표본 공분산 조건수: 3,985 (불안정)
    PCA 공분산 조건수:   825 (4.8배 개선)
    → 최적화 결과가 안정적으로 변함

K 선택 기준: 누적 설명 분산 >= 80%인 최소 K (하한 3, 상한 10)
```

#### 3. Ledoit-Wolf 축소 추정 (Step3)

```
분류: 통계 추정 (공분산 행렬 정규화)
라이브러리: sklearn.covariance.LedoitWolf

질문: "제한된 데이터(500일)로 안정적인 공분산 행렬을 어떻게 추정하는가?"

구체적 적용:
  채권 그룹 (4개): LedoitWolf().fit(bond_returns).covariance_
  대안 그룹 (2개): LedoitWolf().fit(alt_returns).covariance_
  Level 1 (3개):  LedoitWolf().fit(l1_returns).covariance_

  내부 원리:
    Σ_shrunk = (1-α) × Σ_sample + α × Σ_target
    Σ_sample = 표본 공분산 (노이즈 많지만 정보 풍부)
    Σ_target = 대각 행렬 (노이즈 없지만 "상관=0" 가정)
    α = 최적 축소 강도 (해석적 공식으로 자동 산출, 교차검증 불필요)

  효과:
    α가 높을수록 (자산 많을수록) → 표본 공분산의 노이즈를 더 강하게 억제
    극단적 상관 추정값(예: 우연히 0.95)을 합리적 범위로 당겨줌
```

#### 4. Mean-Variance 최적화 (Step3)

```
분류: 수리 최적화 (볼록/비선형)
라이브러리: scipy.optimize.minimize (method='SLSQP')

질문: "주어진 리스크 허용도(γ)에서 효용을 최대화하는 비중은?"

구체적 적용:
  효용 함수: U = μ'w - (γ/2) × w'Σw
    μ = 기대수익률 벡터 (IS 구간 표본 평균 × 252)
    Σ = 공분산 행렬 (PCA 또는 Ledoit-Wolf)
    w = 비중 벡터 (결정 변수)
    γ = 위험 회피 계수 (보수형 8, 공격형 1)

  제약조건:
    Σw_i = 1 (비중 합 = 100%)
    w_i >= 0 (공매도 금지)
    w_i <= HHI 기반 상한 (성향별 차등)
    Σ(equity) <= max_equity (성향별 주식 상한)
    Σ(bond) >= min_bond (성향별 채권 하한)

  SLSQP: Sequential Least Squares Programming
    비선형 제약조건을 처리하는 반복 최적화 알고리즘
```

#### 5. Risk Parity (Step3)

```
분류: 수리 최적화
라이브러리: scipy.optimize.minimize

질문: "모든 자산이 포트폴리오 위험에 동일하게 기여하는 비중은?"

구체적 적용:
  목적함수: min Σ(RC_i - σ_p/N)²
    RC_i = w_i × (Σw)_i / σ_p  (자산 i의 리스크 기여도)
    σ_p/N = 목표 기여도 (전체 변동성을 N등분)

  기대수익률(μ)을 사용하지 않음 → μ 추정 오차에 면역
  대신 저변동성 자산(채권)에 과다 배분되는 경향
```

#### 6. HRP — 계층적 리스크 패리티 (Step3)

```
분류: 비지도 ML (계층적 클러스터링 + 재귀 배분)
라이브러리: scipy.cluster.hierarchy (linkage, leaves_list)

질문: "공분산 역행렬 없이도 안정적인 배분이 가능한가?"

구체적 적용:
  Step 1: 상관 거리 행렬 → Ward 연결법 계층적 클러스터링
    거리 = √((1-ρ)/2), 상관 높을수록 가까움
    → 비슷한 자산끼리 묶임 (예: SPY-QQQ, TLT-AGG)

  Step 2: 클러스터 순서대로 재귀적 이등분 배분
    각 분할에서 역분산(1/σ²) 비율로 비중 배분
    → 변동성 낮은 쪽에 더 많이 배분

  핵심 장점: 공분산 역행렬(Σ⁻¹) 불필요
    → MV는 Σ⁻¹이 필요하여 N이 크면 불안정
    → HRP는 대각 원소(분산)만 사용 → 가장 강건
```

#### 7. Walk-Forward 교차검증 (Step4)

```
분류: 시계열 모델 검증 (Time Series Cross-Validation)
구현: 커스텀 (generate_walk_forward_windows)

질문: "모델이 과적합 없이 미래에도 작동하는가?"

구체적 적용:
  일반 교차검증(K-fold)은 시계열에 사용 불가
    → 미래 데이터가 학습에 섞이면 look-ahead bias

  Walk-Forward:
    IS(학습) 24개월 → OOS(검증) 3개월 → 슬라이딩
    31개 윈도우, 각 윈도우에서 독립적으로 최적화 후 OOS 성과 측정
    → "2018년 1월에 앉아서 2016~2017년만 보고 결정했으면 어땠을까"를 시뮬레이션
```

#### 8. VaR / CVaR (Step5)

```
Historical VaR (비모수):
  분류: 비모수 통계 (경험적 분위수)
  방법: 수익률 분포의 하위 α% 분위수
    95% VaR = percentile(returns, 5%)
    "100일 중 5일은 이 손실보다 더 클 수 있다"

Parametric VaR (모수):
  분류: 모수 통계 (정규분포 가정)
  방법: VaR = -(μ + z_α × σ)
    z_α = 정규분포 역함수 (95%: -1.645, 99%: -2.326)
    정규분포를 가정하므로 팻테일(fat tail)을 과소 추정하는 한계

CVaR (Conditional VaR = Expected Shortfall):
  "VaR를 넘어선 손실의 평균"
  → VaR보다 보수적, 꼬리 위험을 더 잘 반영
  → 항상 CVaR >= VaR
```

#### 9. Hidden Markov Model (Step6)

```
분류: 비지도 ML (확률적 상태 모델)
라이브러리: hmmlearn.hmm.GaussianHMM

질문: "시장이 현재 어떤 '상태(레짐)'에 있는가?"

구체적 적용:
  관측 변수 (5개): VIX_level, VIX_contango, HY_spread, yield_curve, Cu_Au_ratio_chg
  숨겨진 상태: N개 레짐 (BIC로 N=4 선택)

  내부 원리:
    각 레짐 k는 고유한 평균 벡터(μ_k)와 공분산(Σ_k)을 가짐
    → 레짐 0: μ = [VIX 12.3, HY 3.63, ...] (안정)
    → 레짐 3: μ = [VIX 22.3, HY 4.48, ...] (위기)

    전이 확률 행렬 A:
      A[i,j] = P(내일 레짐 j | 오늘 레짐 i)
      → 안정 유지 99.3%, 위기 유지 96.8%

    학습: Baum-Welch 알고리즘 (EM 알고리즘의 HMM 특화 버전)
      E-step: 각 시점의 레짐 확률 추정 (forward-backward)
      M-step: μ, Σ, A 업데이트
      반복 수렴

  BIC 모델 선택:
    BIC = -2×LL + k×ln(n)
    LL = log-likelihood (적합도, 높을수록 좋음)
    k = 파라미터 수 (복잡도, 낮을수록 좋음)
    → 적합도와 복잡도의 트레이드오프에서 최적 N 선택
    → N=4가 최소 BIC(19,976)
```

#### 10. Bootstrap 재표본 검정 (Step7)

```
분류: 비모수 통계 검정
구현: 커스텀 (bootstrap_sharpe_diff)

질문: "Config B의 Sharpe 개선이 우연이 아닌 진짜인가?"

구체적 적용:
  원래 데이터: Config A 일별 수익률, Config B 일별 수익률 (각 2,328일)

  Bootstrap 과정:
    5,000번 반복:
      2,328일에서 복원추출(with replacement)로 2,328일 재추출
      재추출 데이터로 Sharpe_A, Sharpe_B 각각 계산
      차이 = Sharpe_B - Sharpe_A 저장
    5,000개 차이값의 2.5%, 97.5% 분위수 = 95% 신뢰구간

  판정:
    CI 하한 > 0 → "B가 A보다 유의하게 우수" (p < 0.05)
    CI가 0을 포함 → "차이가 통계적으로 유의하지 않음"

  왜 Bootstrap인가:
    Sharpe Ratio의 분포는 정규분포가 아님 (비율 통계량)
    → t-검정 부적합
    → Bootstrap은 분포 가정 없이 신뢰구간 산출 가능
```

---

### 기법 간 의존 관계

```
Granger (Step2)  ──→  "어떤 대안데이터가 유의한가?" → 경보 지표 선정 근거
     │
PCA + LW (Step3) ──→  공분산 추정 → MV/RP/HRP 최적화
     │
Walk-Forward (Step4)──→  IS/OOS 분리 → 과적합 방지 프레임워크
     │
HMM (Step6)      ──→  레짐 분류 → 경보 시스템 입력 → 공분산 전환(Σ_stable/Σ_crisis)
     │
Bootstrap (Step7) ──→  "대안데이터가 실제로 성과를 개선했는가?" 최종 판정
```

report에도 추가하시겠습니까?

아직 추가하지 않았습니다. 지금 바로 추가하겠습니다.