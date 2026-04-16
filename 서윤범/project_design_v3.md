# 성향별 액티브 ETF 펀드 상품 구축
## 프로젝트 설계 문서

> 최초 작성: 2026-04-15  
> 분석 기간: 2016-01-01 ~ 2025-12-31 (10년, 약 2,520 영업일)  
> 투자 유니버스: ETF 22개 (인덱스5 + 채권4 + 대안2 + 섹터11)  
> 노트북 구성: Step1 ~ Step7.ipynb

---

## 목차

1. [데이터 수집 (Step1)](#1-데이터-수집-step1)
2. [데이터 전처리 / EDA / Feature Engineering (Step2)](#2-데이터-전처리--eda--feature-engineering-step2)
3. [Modeling (Step3)](#3-modeling-step3)
4. [백테스트 (Step4)](#4-백테스트-step4)
5. [리스크 분석 (Step5)](#5-리스크-분석-step5)
6. [동적 포트폴리오 구현 & 대시보드 구축 (Step6~7)](#6-동적-포트폴리오-구현--대시보드-구축-step67)

---

## 1. 데이터 수집 (Step1)

### 수집 대상 요약

| 구분 | 개수 | 소스 | 역할 |
|------|------|------|------|
| 인덱스 ETF | 5 | yfinance | 자산군 대표 (미국대형/기술/소형/선진/신흥) |
| 채권 ETF | 4 | yfinance | 듀레이션 스펙트럼 (장기/종합/단기/물가연동) |
| 대안 ETF | 2 | yfinance | 원자재·금 |
| 섹터 ETF | 11 | yfinance | GICS 11개 섹터 전체 |
| 외부 시장 지표 | 12 | yfinance | VIX 기간구조, 원자재, 암호화폐, 달러 등 |
| FRED 매크로 | 8 | FRED API | 금리·스프레드·고용·성장 |
| GDELT 뉴스 | - | GDELT API | 기사 원문, 감성 점수, 기사량 *(수집 진행 중)* |

### 1-1. 투자 자산 (ETF 22개)

| 티커 | 분류 | 역할 |
|------|------|------|
| SPY, QQQ, IWM, EFA, EEM | 인덱스 ETF | 주요 시장 대표 |
| TLT, AGG, SHY, TIP | 채권 ETF | 듀레이션·물가 스펙트럼 |
| GLD, DBC | 대안 ETF | 금·원자재 |
| XLK, XLF, XLE, XLV, VOX, XLY, XLP, XLI, XLU, XLRE, XLB | 섹터 ETF | GICS 11개 섹터 |

> **개별주 제외 결정 논의 필요**: 생존편향(survivorship bias) 및 섹터 ETF와의 중복 노출 문제로 제외.

### 1-2. 외부 시장 지표 (12개 — yfinance)

| 티커 | 설명 | Step2 활용 |
|------|------|-----------|
| ^VIX | 공포 지수 (수준값 유지) | VIX 기간구조 파생변수 기준 |
| ^VIX9D, ^VIX3M, ^VIX6M | 단기·중기·장기 내재변동성 | VIX_contango, slope 계산 |
| ^SKEW | 꼬리 위험 지수 | SKEW_zscore 파생변수 |
| HG=F | 구리 선물 | Cu/Au ratio 계산 |
| CL=F, GC=F, SI=F | WTI·금·은 선물 | 원자재 수익률 피처 |
| BTC-USD, ETH-USD | 비트코인·이더리움 | 시장 위험선호 proxy |
| DX-Y.NYB | 달러 인덱스 | 글로벌 유동성 지표 |

### 1-3. FRED 매크로 (8개)

| 시리즈 | 설명 | 처리 방식 |
|--------|------|----------|
| BAMLH0A0HYM2 | HY OAS 신용스프레드 | 수준값 유지 |
| T10Y2Y | 10Y-2Y 수익률 곡선 스프레드 | 수준값 유지 |
| ICSA | 신규 실업수당 청구 (주간) | 수준값·Z-score 파생 |
| WEI | Weekly Economic Index | 수준값 유지 (2020-03 이전 제외) |
| SAHMREALTIME | Sahm 경기침체 지표 | 수준값 유지 |
| DGS10 | 미국 10Y 국채 수익률 | 차분(`diff`) |
| CPIAUCSL | 소비자물가지수 (월간) | 희소성 95% → 제외 검토 |
| UNRATE | 실업률 (월간) | 차분, SAHMREALTIME으로 대체 검토 |

### 1-4. GDELT 뉴스 데이터

- **ML용**: 감성 점수(tone), 기사량(volume) → 수치 파생변수로 Step2에서 처리
- **LLM용**: 뉴스 원문 요약 → Step3 LLM Agent 프롬프트에 직접 투입
- 수집 단위: 자산별(ETF 티커 기준), 월별 집계
- **저장 파일**: `data/portfolio_prices.csv`, `data/external_prices.csv`, `data/fred_data.csv`, `data/gdelt_news.csv`

---

## 2. 데이터 전처리 / EDA / Feature Engineering (Step2)

### 2-1. 수익률 계산 및 정상성 처리

| 변수 유형 | 처리 방법 | 근거 |
|----------|----------|------|
| 자산 가격 22개, 원자재·암호화폐 | 로그 수익률 `ln(P_t / P_{t-1})` | 시계열 합산 가능, 정규성 근사 |
| VIX 계열 수준값 (`^VIX`) | 수준값 유지 + 기간구조 파생변수로 대체 | VIF 1000+ → 직접 투입 제외 |
| T10Y2Y, BAMLH0A0HYM2 스프레드 | 수준값 유지 | 스프레드 수준 자체가 신호 |
| DGS10, UNRATE | 차분 `diff()` | ADF 검정 비정상 확인 |
| SAHMREALTIME, WEI | 수준값 유지 | 수준 자체가 경기 지표 |

### 2-2. 15개 파생변수 (Feature Engineering)

| # | 변수명 | 수식 | 카테고리 |
|---|--------|------|----------|
| 1 | `VIX_contango` | `^VIX3M / ^VIX - 1` | VIX 기간구조 |
| 2 | `VIX_slope_9d_3m` | `^VIX3M - ^VIX9D` | VIX 단기 기울기 |
| 3 | `VIX_slope_3m_6m` | `^VIX6M - ^VIX3M` | VIX 장기 기울기 |
| 4 | `SKEW_level` | `^SKEW` 수준값 | 꼬리 위험 |
| 5 | `SKEW_zscore` | `(SKEW - μ_63d) / σ_63d` | 꼬리 위험 상대적 수준 |
| 6 | `Cu_Au_ratio` | `HG=F / GC=F` | 실물 경기 |
| 7 | `Cu_Au_ratio_chg` | `Cu_Au_ratio.diff(21)` | 실물 경기 변화율 |
| 8 | `HY_spread` | `BAMLH0A0HYM2` | 신용 시장 |
| 9 | `HY_spread_chg` | `HY_spread.diff(5)` | 신용 급변 감지 |
| 10 | `yield_curve` | `T10Y2Y` | 경기선행 |
| 11 | `yield_curve_inv` | `T10Y2Y < 0` (역전 더미) | 경기침체 신호 |
| 12 | `claims_4wma` | `ICSA` 4주 이동평균 | 노동시장 |
| 13 | `claims_zscore` | `(claims - μ_260d) / σ_260d` | 노동시장 상대 수준 |
| 14 | `WEI_level` | `WEI` 수준값 | 주간 경기 |
| 15 | `sahm_indicator` | `SAHMREALTIME` | 경기침체 트리거 |

### 2-3. 자산별 피처 생성 (ML Long-Panel용)

각 ETF 자산별로 아래 피처를 추가 생성한다. 매크로 피처(2-2)는 모든 자산에 동일한 값을 사용한다.

| 피처명 | 수식 | 카테고리 |
|--------|------|----------|
| `ret_1m` | `log(P_t / P_{t-21})` | 모멘텀 |
| `ret_3m` | `log(P_t / P_{t-63})` | 모멘텀 |
| `ret_6m` | `log(P_t / P_{t-126})` | 모멘텀 |
| `ret_12m` | `log(P_t / P_{t-252})` | 모멘텀 |
| `vol_21d` | `std(daily_ret, 21일) × √252` | 변동성 |
| `vol_63d` | `std(daily_ret, 63일) × √252` | 변동성 |
| `vol_ratio` | `vol_21d / vol_63d` | 변동성 가속 |
| `volume_zscore` | `(vol_t - μ_63d) / σ_63d` | 거래량 이상 |
| `ret_vs_sector` | `ret_1m - sector_mean_ret_1m` | 상대 모멘텀 |
| `sector_code` | EQUITY / BOND / ALT | 자산군 구분 |

### 2-4. HMM 레짐 확률 피처(초안. 아직 피처로 넣을지에 대한 고민 중)

BIC로 최적 레짐 수(K=3~4) 결정 후 위기 국면 확률을 연속형 피처로 생성한다.

```
입력: VIX_level, VIX_contango, HY_spread, yield_curve, Cu_Au_ratio_chg
출력: hmm_crisis_prob  ∈ [0, 1]  — P(regime = crisis | 관측값)
```

- 롤링 윈도우 안에서 매 IS 구간마다 HMM 재학습 (미래 레짐 정보 사용 금지)
- 이 확률값을 ML 피처로 투입 → 모델이 레짐별 가중치를 스스로 학습

### 2-5. GDELT 감성 피처(재천, 윤서 작업 중)

```python
# 아래의 변수들은 예시. 변할 수 있음
gdelt_avg_tone_1m      # 1개월 평균 감성 점수 (양수=긍정, 음수=부정)
gdelt_tone_momentum    # (tone_1m - tone_3m)  감성 변화율
gdelt_article_volume   # 기사량 (시장 불확실성 proxy)
```

### 2-6. EDA 및 통계 검정

| 검정 항목 | 목적 | 기준 |
|----------|------|------|
| ADF 정상성 검정 | 변수 변환 방식 결정 | p < 0.05 → 정상, 아니면 차분 |
| VIF 다중공선성 | 중복 변수 제거 | VIF > 10 제거 (VIX 계열 파생변수로 대체) |
| 상관관계 히트맵 | 자산·피처 간 관계 파악 | - |
| Granger 인과 검정 | 피처 → 수익률 예측력 확인 | maxlag=10, 최소 p-value |
| ARCH 효과 검정 | 변동성 군집 존재 여부 | 공분산 추정 방식 결정 근거 |
| **IC 분석** | ML 피처 유효성 평가 | IC > 0.02, ICIR > 0.3 유지 |
| 월간 주기 유효성 | 포워드 기간 최적화 | R²_oos 비교 (5/10/21/63일) |

> IC 분석의 경우 좀 더 찾아봐야 함

### 2-7. Long-Panel 데이터셋 구성

**최종 구조**: 인덱스 `[date(월말), ticker]`

```
date       | ticker | ret_1m | vol_21d | VIX_level | HY_spread | gdelt_tone | hmm_prob | target_fwd
2016-08-31 | SPY    |  0.032 |  0.109  |    12.1   |    3.51   |    0.12    |   0.04   |   0.014
2016-08-31 | QQQ    |  0.048 |  0.134  |    12.1   |    3.51   |    0.12    |   0.04   |   0.021
2016-08-31 | TLT    | -0.011 |  0.082  |    12.1   |    3.51   |    0.12    |   0.04   |  -0.008
```

- **자산별 피처**: 각 행마다 자산 고유값 (모멘텀, 변동성 등)
- **매크로 피처**: 같은 날짜의 모든 자산에 동일한 값
- **타겟**: 다음 21 영업일 포워드 로그 수익률

---

## 3. Modeling (Step3)

### 전체 파이프라인

```
[Long-Panel 데이터 (date × ticker)]
              ↓
  ┌──────────────────────────────────────────────────────┐
  │     Step 3-1. 기대수익률 & 확신도 추출               │
  │                                                      │
  │   분류 (XGBoost, TabPFN)   회귀 (RF)   LLM Agent    │
  │           ↓                    ↓            ↓        │
  │       Q_cls, Ω_cls        Q_reg, Ω_reg  Q_llm, Ω_llm│
  └──────────────────────────────────────────────────────┘
              ↓  모델별 Q, Ω 독립 평가 후 최적 선택
  ┌──────────────────────────────────────────────────────┐
  │     Step 3-2. Black-Litterman                        │
  │   Prior π  +  선택된 Q, Ω → μ_BL, Σ_BL              │
  └──────────────────────────────────────────────────────┘
              ↓
  ┌──────────────────────────────────────────────────────┐
  │     Step 3-3. MVO — 성향별 최적 비중                 │
  │   γ_공격형 / γ_중립형 / γ_보수형 → w*               │
  └──────────────────────────────────────────────────────┘
```

> **모델 앙상블 전략**: 각 모델(RF, XGBoost, TabPFN, LLM)을 독립적으로 평가한 뒤 성과가 가장 좋은 모델의 Q, Ω를 BL에 투입한다. 무조건 앙상블할 경우 신호가 상쇄되어 시장 추종으로 수렴할 위험이 있다.

---

### 3-1. 기대수익률과 확신도 계산 (Q, Ω)

> **핵심 아이디어**  
> 주식 수익률은 SNR이 낮아 정확한 점 예측이 어렵다. 따라서 예측값(Q)과 함께 그 불확실성(Ω)을 반환하고, BL이 불확실한 뷰는 자동으로 할인하도록 설계한다.  
> Ω가 작을수록 → 모델 확신 → BL이 해당 뷰를 강하게 반영  
> Ω가 클수록 → 모델 불확실 → BL이 Prior(시장 균형)로 회귀

---

#### (A) 분류 모델 — XGBoost / TabPFN

**참고 논문**: Min, Dong et al. (2021) *A Black-Litterman Portfolio Selection Model with Investor Opinions Generating from Machine Learning Algorithms*

**Step A-1. 레이블 이산화**

IS 구간 수익률의 분위수를 기준으로 N=5 등급 분류.  
⚠️ **분위수 경계는 IS 구간 데이터만으로 결정** — look-ahead bias 방지

```python
bins = pd.qcut(is_forward_returns, q=5, retbins=True)[1]  # IS만 사용
labels_is  = pd.cut(is_forward_returns,  bins=bins, labels=[1,2,3,4,5])
labels_oos = pd.cut(oos_forward_returns, bins=bins, labels=[1,2,3,4,5])
```

| 클래스 | 의미 | 범위 |
|--------|------|------|
| C1 | 강한 하락 | 하위 20% |
| C2 | 약한 하락 | 20~40% |
| C3 | 중립 | 40~60% |
| C4 | 약한 상승 | 60~80% |
| C5 | 강한 상승 | 상위 20% |

**Step A-2. XGBoost 학습**

```
입력 X: [자산별 피처] + [매크로 15개] + [GDELT 감성] + [hmm_crisis_prob] + [sector_code]
출력 Y: 클래스 확률 [P(C1), P(C2), P(C3), P(C4), P(C5)]
```

XGBoost 목적함수 (K번째 트리):

$$L^{(K)} \approx \sum_{i=1}^{m}\left[g_i f_K(x_i) + \frac{1}{2}h_i f_K^2(x_i)\right] + \Omega(f_K), \quad \Omega(f_K) = \gamma T + \frac{1}{2}\lambda\|w\|^2$$

$$g_i = \frac{\partial\, l(y_i, \hat{y}^{(K-1)})}{\partial\, \hat{y}^{(K-1)}}, \qquad h_i = \frac{\partial^2 l(y_i, \hat{y}^{(K-1)})}{\partial\, (\hat{y}^{(K-1)})^2}$$

**Step A-3. Q 도출** — 클래스 가중 기댓값

$$Q_i = \sum_{k=1}^{5} \bar{r}_k \cdot P(C_k \mid z_{i,t})$$

($\bar{r}_k$: IS 구간 클래스 $k$ 내 평균 수익률)

**Step A-4. Ω 도출** — 예측 분포의 분산

$$\Omega_{ii} = \sum_{k=1}^{5} P(C_k \mid z_{i,t}) \cdot (\bar{r}_k - Q_i)^2$$

$$\Omega = \text{diag}(\Omega_{11}, \ldots, \Omega_{NN})$$

> TabPFN은 동일한 입력·출력 구조에서 적용 (단, 피처 수 100개 이하 제한 → IC 기준 사전 선별 필요)

---

#### (B) 회귀 모델 — Random Forest

**아이디어**: B개 개별 트리의 예측값 분포에서 Q(평균)와 Ω(분산)을 동시에 추출한다.

> **개별 트리가 너무 랜덤하지 않나?**  
> 개별 트리는 노이즈가 많다. 그러나 핵심은 다음과 같다.  
> - **Q** = B개 트리의 **평균** → 대수의 법칙으로 안정적인 중심 추정값  
> - **Ω** = B개 트리의 **분산** → 트리들이 동의하면 작고, 갈릴수록 커진다  
> 이 분산이 바로 모델의 **epistemic uncertainty(인식적 불확실성)** 이다.  
> 랜덤성이 문제가 아니라, 그 랜덤성의 크기가 곧 불확실성의 척도가 된다.

이 방식은 BL+ML 논문의 분해에 근거한다:

$$E(f, D) = \text{bias}^2(x) + \text{var}(x)$$

**Step B-1. 개별 트리 OOB 예측값 수집**

```python
tree_preds = np.array([tree.predict(X_oos) for tree in rf_model.estimators_])
# shape: (B개 트리, N개 자산)
```

**Step B-2. Q 도출** — 트리 앙상블 평균

$$Q_i = \frac{1}{B}\sum_{b=1}^{B} f_b(z_{i,t})$$

**Step B-3. Ω 도출** — 트리 간 분산

$$\Omega_{ii} = \frac{1}{B-1}\sum_{b=1}^{B}(f_b(z_{i,t}) - Q_i)^2$$

| 상황 | 트리 분산 | Ω | BL 영향력 |
|------|-----------|---|-----------|
| 트리들이 일치 | 작음 | 작음 → 강하게 반영 | ML 예측 우선 |
| 트리들이 분산 | 큼 | 큼 → 약하게 반영 | Prior(시장 균형)로 회귀 |

---

#### (C) LLM Agent

**참고 논문**: Lee, Kim et al. (2025) *LLM-Enhanced Black-Litterman Portfolio Optimization*

**Step C-1. 입력 구성 (구조화 프롬프트)**

```
[시스템 프롬프트]
{DATE}: 향후 21 영업일(1개월)의 평균 일별 수익률을 예측하라.
제공된 시계열·뉴스를 분석하고 단일 부동소수점 숫자로 반환하라.

[유저 프롬프트 — 자산별 반복]
- 일별 수익률: 최근 5주 시계열
- 섹터 수익률: 해당 GICS 섹터 일별 수익률
- 시장 수익률: SPY 일별 수익률
- 종목 정보: 티커, 섹터명, GICS 서브섹터
- GDELT 뉴스 요약: 최근 1개월 감성 점수 + 주요 이슈
```

**Step C-2. Q 도출** — LLM 반복 호출 평균 (temperature=0, N=100회)

$$Q_i = \frac{1}{N}\sum_{j=1}^{N} \hat{r}_{i,j}^{LLM}$$

모든 호출 결과를 JSON으로 저장 (재현성 확보)

**Step C-3. Ω 도출** — 검증셋 예측 오차 기반 사후 보정

LLM 자체 confidence는 사용하지 않는다. 검증 구간의 실제 예측 오차 MSE로 신뢰도를 역산한다:

$$\Omega_{ii} = \frac{1}{|T_{val}|}\sum_{t \in T_{val}}\left(\hat{r}_{i,t}^{LLM} - r_{i,t}^{actual}\right)^2$$

---

### 3-2. Black-Litterman — μ_BL, Σ_BL 도출

#### 기본 기호 정의

| 기호 | 의미 |
|------|------|
| $N$ | 자산 수 (ETF 22개) |
| $\pi$ | CAPM 균형 수익률 Prior |
| $\Sigma$ | 자산 수익률 공분산 행렬 (Ledoit-Wolf) |
| $\tau$ | 스케일링 파라미터 — Prior 대비 뷰 신뢰도 |
| $P$ | 픽 행렬 ($K \times N$) |
| $Q$ | 뷰 벡터 — ML 예측 기대수익률 |
| $\Omega$ | 뷰 불확실성 행렬 (대각) |
| $\mu_{BL}$ | BL 사후 기대수익률 |
| $\Sigma_{BL}$ | BL 사후 공분산 행렬 |

#### Step 1. Prior 균형 수익률 (π)

$$\pi = \lambda \cdot \Sigma \cdot w_{mkt}, \qquad \lambda = \frac{\bar{r}_{mkt} - r_f}{\sigma_{mkt}^2}$$

- $w_{mkt}$: 동일 비중(1/N) 또는 ETF AUM 기반 비중  
- $r_f$: DGS10 (미국 10년물)  

> **Prior 선택 전략**: 동일 비중(1/N)이 가장 방어하기 쉽다. AUM 방식과 비교해 ablation으로 처리하여 논문 기여점으로 활용 가능.

#### Step 2. 뷰 행렬 구성

절대적 뷰(Absolute Views) — 자산별 독립 뷰, $P = I_N$:

$$Q = P\mu + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \Omega)$$

#### Step 3. BL 사후 분포

$$\boxed{\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1} \left[(\tau\Sigma)^{-1}\pi + P^\top\Omega^{-1}Q\right]}$$

$P = I$ 적용 시:

$$\mu_{BL} = \left[(\tau\Sigma)^{-1} + \Omega^{-1}\right]^{-1} \left[(\tau\Sigma)^{-1}\pi + \Omega^{-1}Q\right]$$

$$\Sigma_{BL} = \Sigma + \left[(\tau\Sigma)^{-1} + \Omega^{-1}\right]^{-1}$$

**직관**:

| 조건 | 결과 |
|------|------|
| $\Omega_{ii} \to 0$ (확신) | $\mu_{BL,i} \to Q_i$ (ML 예측 우선) |
| $\Omega_{ii} \to \infty$ (불확실) | $\mu_{BL,i} \to \pi_i$ (시장 균형 회귀) |
| $\tau \to 0$ | Prior에 전적으로 의존 |
| $\tau \to \infty$ | 뷰에 전적으로 의존 |

#### Step 4. τ (tau) 추정

**단계 1** — 이론적 초기값:
$$\tau_{default} = \frac{1}{T} \quad (T = \text{IS 관측치 수, 150일 기준} \approx 0.0067)$$

**단계 2** — 검증셋 기반 초기 추정:
$$\tau_{init} = \frac{1}{|T_{val}|}\sum_{t \in T_{val}} \frac{\text{mean}(\Omega_t)}{\text{mean}(\Sigma_t)}$$

**단계 3** — 그리드 서치:
$$\mathcal{T} = \{0.5\tau_{init},\ 0.75\tau_{init},\ \tau_{init},\ 1.25\tau_{init},\ 1.5\tau_{init}\}$$
$$\tau^* = \underset{\tau \in \mathcal{T}}{\arg\max}\ \text{Sharpe}(\tau \mid D_{val})$$

---

### 3-3. MVO — 성향별 최적 포트폴리오

#### 목적함수

$$\underset{w}{\text{maximize}} \quad w^\top\mu_{BL} - \frac{\gamma}{2}\,w^\top\Sigma_{BL}\,w - \lambda_{TO}\sum_{i}|w_i - w_{i,\text{prev}}|$$

#### 제약 조건

$$\sum_{i} w_i = 1, \quad w_i \geq 0, \quad w_i \leq w_{i,\text{max}}\ (\text{기본 30\%})$$

#### 성향별 위험회피계수 γ

| 성향 | γ | 특징 |
|------|---|------|
| 공격형 | 0.5 ~ 1.5 | 고수익 추구, 주식 비중 ↑ |
| 중립형 | 2.0 ~ 4.0 | 수익-위험 균형 |
| 보수형 | 5.0 ~ 10.0 | 안정성 우선, 채권 비중 ↑ |

---

## 4. 백테스트 (Step4)

### Walk-Forward 구조

```
┌────────────────────────┬────────────┐
│  IS (학습): 150 영업일  │ OOS: 30일  │ → 슬라이딩
└────────────────────────┴────────────┘
총 윈도우 수: (2,520 - 150) / 30 ≈ 79개
Purged + Embargo: IS-OOS 경계 21일 embargo (Lopez de Prado)
```

### ML 예측력 평가 (Statistical)

| 지표 | 적용 모델 | 설명 |
|------|----------|------|
| Accuracy | XGBoost, TabPFN | 5등급 분류 정확도 |
| MSE | RF | 평균제곱오차 |
| $R^2_{oos}$ | 전체 | 샘플 외 결정계수 (양수 → ML이 랜덤보다 우수) |
| Diebold-Mariano Test | 모델 간 비교 | 예측력 차이 유의성 검정 |

### BL 운용 성과 평가 (Economic)

| 지표 | 설명 |
|------|------|
| Sharpe Ratio | 위험조정 수익률 (주 지표) |
| Sortino Ratio | 하방 변동성만 고려 |
| CAGR | 연평균 복리 성장률 |
| MDD | 최대 낙폭 |
| Calmar Ratio | CAGR / \|MDD\| |
| Net Return | 거래비용 왕복 30bps 차감 후 순수익률 |
| 팩터 알파(α) | Fama-French 3~5팩터 대비 초과 수익 |
| 회전율 | 월평균 포트폴리오 교체 비율 |

### 벤치마크

- SPY 100% (시장 수익률)
- 동일가중(EW) 22자산
- BL 미적용 순수 MVO
- 각 모델(RF/XGB/TabPFN/LLM) 독립 성과 비교

---
# 여기부터는 아직 미정. 재천님 초안 가이로 적어둠

## 5. 리스크 분석 (Step5)

### 5-1. VaR / CVaR

| 지표 | 정의 |
|------|------|
| VaR (95%, 99%) | 특정 신뢰수준에서의 최대 예상 손실 |
| CVaR (95%, 99%) | VaR 초과 손실들의 평균 (Expected Shortfall) |

측정 방법: Historical / Parametric / Monte Carlo

### 5-2. 리스크 기여도 분석

$$RC_i = w_i \times \frac{(\Sigma\mathbf{w})_i}{\sigma_p}, \qquad \sum_i RC_i = \sigma_p$$

성향별로 각 자산의 리스크 기여 비율을 분해하여 집중도 확인

### 5-3. 스트레스 테스트

| 유형 | 시나리오 |
|------|---------|
| 역사적 | 2020 코로나 폭락, 2022 긴축 충격, 2018 VIX 폭발 |
| 가상 | 유가 +50%, VIX 80, 달러 급등, 신용스프레드 급등 |

### 5-4. 거래비용 민감도 분석

왕복 10 ~ 50bps 구간에서 Net Return / Sharpe 변화 시뮬레이션  
→ 전략의 실용 가능 거래비용 상한 추정

---

## 6. 동적 포트폴리오 구현 & 대시보드 구축 (Step6~7)

### 6-1. HMM 기반 레짐 감지 및 경보 시스템 (Step6)

**입력 변수**: VIX_level, VIX_contango, HY_spread, yield_curve, Cu_Au_ratio_chg  
**BIC로 최적 레짐 수 결정** (예상 K=3~4)

| 경보 레벨 | 의미 | 트리거 기준 |
|-----------|------|------------|
| L0 | 정상 | P(crisis) < 0.3 |
| L1 | 주의 | P(crisis) ≥ 0.3 |
| L2 | 경계 | P(crisis) ≥ 0.5 또는 수익률 곡선 역전 |
| L3 | 위기 | P(crisis) ≥ 0.7 또는 Sahm Rule 발동 |

**4개 Config (Ablation Study용)**:
- Config A: VIX 기반 단순 경보
- Config B: VIX + HY Spread
- Config C: VIX + HY Spread + Yield Curve
- Config D: 전체 대안 데이터 통합

### 6-2. 동적 리밸런싱 규칙 (Step7)

경보 레벨에 따라 주식 비중을 차등 축소, 해제 비중은 채권(70%) + 대안(30%)으로 재분배:

| 성향 | γ | L1 축소 | L2 축소 | L3 축소 |
|------|---|---------|---------|---------|
| 보수형 | 8 | -15% | -35% | -60% |
| 중립형 | 4 | -10% | -25% | -50% |
| 적극형 | 2 | -5% | -15% | -35% |
| 공격형 | 1 | 0% | -5% | -20% |

### 6-3. Ablation Study

4 Config × 4 성향 = **16개 시뮬레이션** + EW·SPY 벤치마크 비교  
Bootstrap 유의성 검정 (5,000회): Config A 대비 B/C/D의 Sharpe 차이가 통계적으로 유의한지 확인

### 6-4. Streamlit 대시보드

- 성향 선택 → 최적 포트폴리오 비중 파이차트
- 누적 수익률 vs 벤치마크(SPY, EW) 비교 차트
- 위험 지표 테이블 (Sharpe, MDD, VaR, CVaR)
- HMM 경보 레벨 실시간 표시
- 리밸런싱 이력 및 회전율 로그

---

## 부록. 주요 설계 결정 사항

| 항목 | 결정 | 근거 |
|------|------|------|
| 개별주 제외 | ETF 22개만 사용 | 생존편향 + 섹터 ETF 중복 |
| Long-Panel 구조 | 단일 모델, 전 자산 통합 학습 | Gu et al. (2020) 방법론 |
| 날짜를 피처로 미사용 | 롤링 윈도우 + hmm_crisis_prob로 대체 | 과적합 방지 |
| Prior = 동일 비중 | 1/N | 방어 용이, AUM과 ablation 비교 |
| 모델 독립 평가 후 최적 선택 | 앙상블 미사용 | 신호 상쇄 → 시장 추종 위험 방지 |
| τ 그리드 서치 | 검증셋 Sharpe 최대화 | LLM 논문 방식 |
| Ω_LLM = 검증셋 MSE | LLM 자체 confidence 미사용 | LLM confidence 신뢰성 낮음 |

---

*본 문서는 프로젝트 진행에 따라 단계별로 업데이트됩니다.*
