> **이 문서는 팀 공유용 사본입니다.**
>
> - **진실원(원본)**: Claude Code 의 plan 파일 (`C:\Users\gorhk\.claude\plans\sharded-mapping-puffin.md`)
> - **마지막 동기화**: 2026-04-27 (v2 산출물 명세 확정 반영)
> - **동기화 정책**: 진실원이 갱신될 때마다 본 사본도 함께 갱신합니다. 본 사본을 직접 수정하지 마시고, 변경 사항은 `재천_WORKLOG.md` 또는 본인 prefix `<이름>_WORKLOG.md` 에 제안 형태로 기록 후 협의하십시오.
> - **목적**: 팀원이 Claude 의 plan 파일에 직접 접근하지 않고도 같은 계획을 한곳에서 확인하기 위함입니다.

---

# Phase 1.5 — 변동성 예측 분기 구축 계획

## Context

### 왜 이 변경이 필요한가

Phase 1 LSTM 베이스라인이 5회 Run을 거쳐 다음 결과로 마감되었습니다.

- **3차 Run (현재 best)**: SPY hit_rate=0.6313 ✅ / R²_OOS=-0.2118 ❌
- **4차 Run (Y_trailing 다변량)**: 대폭 악화 (R²_OOS -2.15)
- **5차 Run (VIX 추가)**: 더 악화 (R²_OOS -1.13)

윤서님의 진단으로 **134 훈련 샘플/fold 환경의 절대 부족**이 root cause로 확인되었고, 다변량 피처 추가는 모두 역효과를 냈습니다. 한편 김하연님의 §10 ARCH-LM 검정에서 **변동성에는 강한 자기상관** (SPY LM=754, p≈0)이 확인되어, 동일 시계열에서 **수익률 방향(ACF max 0.13)보다 변동성(ACF max 0.30)이 훨씬 풍부한 신호**임이 정량적으로 입증되었습니다.

**사용자 결정 (확정)**: Phase 2(GRU)로 진행하지 않고, Phase 1 결과는 그대로 보존한 채 **Phase 1.5 분기**를 신설하여 예측 대상을 **누적수익률 → 실현변동성(realized volatility)** 으로 변경한 새 LSTM을 학습·평가합니다.

### Phase 1.5의 핵심 목표 (단일 질문에 집중)

> **"변동성 예측이 가능한가?"**

본 단계는 위 질문에 명확한 답을 내는 것이 유일한 목표입니다. 즉 수익률 방향 예측보다 신호가 풍부한 변동성을 LSTM으로 예측했을 때, **학술 표준 베이스라인(HAR-RV, EWMA, Naive)을 능가하는가**를 검증합니다.

**본 단계 평가 대상이 아닌 사항**:
- 포트폴리오 구축 (Mean-Variance / Black-Litterman)
- 벤치마크 대비 alpha / Sharpe / drawdown
- BL의 Q/Ω 입력 통합

→ 위 사항들은 **추후 별도 단계**에서 검증. Phase 1.5의 결과(변동성 예측 가능성·정확도)는 그 단계의 입력 자료로만 사용.

> 변동성 예측의 포트폴리오 활용 논리·학술적 한계·벤치마크 우위 가능성은 본 plan **Appendix**에 배경 정보로 보관 (본 단계 진행 결정에 영향 없음).

---

## 추천 접근법

### 1. 폴더 구조 — `Phase1_5_Volatility/` 신규 (Phase 1과 형제)

```
시계열_Test/
├── Phase1_LSTM/                   # 보존, 변경 금지
└── Phase1_5_Volatility/           # ⭐ 신규
    ├── README.md, PLAN.md, 재천_WORKLOG.md
    ├── 00_setup_and_utils.ipynb           # Phase1 복사
    ├── 01_volatility_eda.ipynb            # 신규 — RV 분포·ACF·정상성·체제 진단
    ├── 02_volatility_lstm.ipynb           # 신규 — 변동성 LSTM 본 실험 (105 fold × 2 ticker)
    ├── 03_baselines_and_compare.ipynb     # 신규 — HAR-RV/EWMA/Naive vs LSTM
    ├── scripts/
    │   ├── setup.py / dataset.py / models.py / train.py     # Phase1 그대로 복사 (격리 보존)
    │   ├── targets_volatility.py          # 신규 — Log-RV 빌더 + 누수 검증
    │   ├── metrics_volatility.py          # 신규 — RMSE/QLIKE/R²_train_mean/MZ
    │   └── baselines_volatility.py        # 신규 — HAR-RV/EWMA/Naive
    └── results/                           # 노트북 산출물만
```

**scripts 전략**: Phase1_LSTM/scripts를 **그대로 복사** (변경 없음). 변동성 전용 신규 3개 파일만 추가. Phase 1 자산 격리 보존이 최우선.

---

### 2. 타깃 정의 — **Log Realized Volatility (Log-RV)**

#### 2-1. 타깃이란?
머신러닝 모델이 **예측해야 할 목표 변수**(y 변수)입니다.
- Phase 1: "다음 21영업일 동안의 **누적 수익률**" — 즉 "한 달 후까지 얼마나 올랐을까?"
- Phase 1.5: "다음 21영업일 동안의 **변동성 크기**" — 즉 "한 달 후까지 얼마나 출렁였을까?"

#### 2-2. Realized Volatility (실현 변동성)란?

**정의**: 실제 시장에서 관측된 일별 로그수익률들의 **21일 표준편차**.

```
RV[t] = std(log_ret[t-20], log_ret[t-19], ..., log_ret[t])   # trailing 21일
```

- **단위**: 일별 표준편차 (예: 0.01 = 일변동성 1%)
- **의미**: "이 21일 동안 일별 수익률이 평균 ±몇 % 흔들렸는가"
- **항상 양수** (표준편차이므로)
- **분포**: right-skewed (대부분 작은 값, 가끔 시장 충격으로 매우 큰 값 — COVID, 금리 인상 등)

#### 2-3. 왜 `log(RV)`를 사용하는가?

| 단계 | 효과 |
|---|---|
| 원본 RV 분포 | right-skewed, fat-tail (예측 어려움) |
| `log(RV)` 변환 후 | **거의 정규 분포** (대칭, 정상적 꼬리) |

**3가지 이점**:
1. **MSE loss와 정합**: MSE는 정규분포 가정 하에서 최대우도 추정 → log 변환으로 가정 충족
2. **양수 보장**: 예측 후 `exp(pred)`로 역변환하면 자동으로 양수 (변동성 음수 불가능 제약 자동 충족)
3. **학술 표준**: Corsi(2009) HAR-RV 모델이 log-RV에서 작동 → 베이스라인과 직접 비교 가능

#### 2-4. 21일 forward window의 의미

```
target[t] = log( std(log_ret[t+1], log_ret[t+2], ..., log_ret[t+21]) )
                    └─────── 미래 21영업일 ───────┘
```

- 시점 t에서 모델은 "**향후 1개월(21영업일) 동안 시장이 얼마나 출렁일 것인가**"를 예측
- `.shift(-21)`로 trailing RV를 forward 방향으로 이동 → forward 타깃 생성

#### 2-5. 코드

```python
def build_daily_target_logrv_21d(adj_close: pd.Series) -> pd.Series:
    """21일 forward log-realized-volatility 타깃."""
    log_ret = np.log(adj_close).diff()       # 일별 로그수익률
    rv = log_ret.rolling(21).std()           # trailing 21일 표준편차
    target = np.log(rv).shift(-21)            # forward 21일로 이동
    return target
```

---

### 3. 입력 정의 — **`log_ret²` univariate (단일 채널)**

#### 3-1. 입력이란?
모델이 예측에 사용할 **정보 변수**(X 변수). 시점 t의 입력으로부터 시점 t+21의 타깃을 예측합니다.

#### 3-2. `log_ret²` (제곱 로그수익률)란?

```python
log_ret = np.log(adj_close).diff()    # 예: -0.012, +0.008, -0.005, ...
log_ret_squared = log_ret ** 2         # 예: 0.000144, 0.000064, 0.000025, ...
```

- **직관**: "오늘 가격이 얼마나 크게 움직였는가의 **제곱**" (방향은 무시하고 크기만)
- **금융학적 의미**: instantaneous variance proxy — "그날의 분산 추정치"
- **항상 양수** (제곱이므로)
- **타깃과 차원 일치**: 분산(variance) = 변동성(std)의 제곱 → log_ret²의 누적·평균이 RV²와 직접 연결

#### 3-3. 왜 univariate (단 1개 채널)?

**Phase 1 Run 4·5 결정적 교훈**:
- Run 4: 4채널 입력(log_ret, Y_trailing, std_21, Y_trailing_5) → R²_OOS -2.15 (대폭 악화)
- Run 5: VIX 추가 (2채널) → R²_OOS -1.13 (대폭 악화)
- **원인**: 134 훈련 샘플/fold에서 다채널 입력은 모델이 fold-specific 패턴을 암기(과적합)

→ Phase 1.5에서도 **단일 채널 원칙 유지**.

#### 3-4. 왜 `log_ret²`인가? (`log_ret`이 아닌)

| 후보 | 장점 | 단점 |
|---|---|---|
| `log_ret` | Phase 1과 동일 (비교 공정성) | 타깃은 변동성 → 입력은 방향 정보까지 포함 (불일치) |
| **`log_ret²`** ⭐ | **타깃과 dimensional match** (variance proxy) | Phase 1과 입력 다름 |
| `\|log_ret\|` | 절댓값, 분포 평이 | 단조 변환만 차이 |

**선택 근거**: 타깃이 RV(=variance의 sqrt)이므로 입력으로 instantaneous variance proxy(`log_ret²`)를 주면 LSTM이 단순 누적·평활화만 학습하면 됨. 학습 부담 최소화.

---

### 4. 평가 지표 — 수익률용 Hit Rate/R²_OOS 폐기

**Hit Rate 폐기 이유**: 변동성은 정의상 항상 양수 → `sign()`이 모두 양수 → hit_rate=1.0 (trivially 통과). 의미 없음.

**R²_OOS (zero baseline) 폐기 이유**: 변동성=0은 비현실적이며 분모 `sum(y²)`가 변동성 제곱합이라 매우 큰 값 → R²가 인공적으로 1에 가까움. Phase 1의 R²_OOS 함정(윈도우 겹침에 의한 0.89 baseline) 회피.

#### 4-1. RMSE on Log-RV (1차 지표)

```
RMSE = sqrt( mean( (y_pred - y_true)² ) )
```

- **단위**: 타깃과 동일 (Log-RV scale, 약 -5 ~ -3 범위)
- **의미**: 예측 오차의 평균 크기 (제곱 평균의 제곱근)
- **방향**: 작을수록 좋음
- **제곱**: 큰 오차에 더 큰 페널티 (이상치에 민감)
- **사용**: 모델 간 비교, 베이스라인 대비 우위 판정

#### 4-2. QLIKE — Quasi-Likelihood (1차 지표, 변동성 전용)

```
QLIKE = mean( σ²_true / σ²_pred  -  log(σ²_true / σ²_pred)  -  1 )
        (여기서 σ² = exp(2 · logrv))
```

- **출처**: Patton (2011) "Volatility forecast comparison using imperfect volatility proxies"
- **변동성 예측의 학술 표준 손실 함수**
- **핵심 특징: 비대칭** — under-prediction(과소예측)을 over-prediction(과대예측)보다 **더 크게 처벌**
- **왜 비대칭?**: 위험 관리 관점에서 변동성을 과소평가하는 것이 더 위험함 (실제는 더 출렁이는데 안전하다고 판단 → 큰 손실 위험)
- **방향**: 작을수록 좋음 (0이 이상적)

#### 4-3. R²_train_mean (1차 지표, 명확한 sanity check)

```
R²_train_mean = 1 - SSE_model / SSE_train_mean

  SSE_model      = sum( (y_pred - y_true)² )
  SSE_train_mean = sum( (mean(y_train) - y_true)² )
```

- **의미**: "**train 평균값으로만 예측하는 trivial baseline**" 대비 모델이 얼마나 개선했는가
- **0 이상이면 PASS**: 적어도 평균보다는 나음
- **음수면 FAIL**: trivial baseline에도 못 미침 → 학습 자체 실패
- **왜 `train_mean`이 분모인가?**: zero baseline은 변동성에서 비현실적, Phase 1의 R²_OOS 함정(윈도우 겹침으로 분모 부풀림) 회피

#### 4-4. MZ Regression — Mincer-Zarnowitz (2차 지표, 편향 진단)

```
y_true[i] = α + β · y_pred[i] + ε[i]   ← OLS 적합
```

| 검정 결과 | 해석 |
|---|---|
| α=0 AND β=1 | **Unbiased forecast** (이상적) |
| α≠0 | 시스템적 편향 (모델이 일관되게 과/저예측) |
| β<1 | 모델이 변동성 변화폭을 과대 예측 (실제는 덜 출렁임) |
| β>1 | 모델이 변동성 변화폭을 과소 예측 (실제는 더 출렁임) |
| R²<<1 | 예측-실제 상관 약함 |

- **사용**: Pass/Fail 판정이 아닌 **편향 진단** 도구

---

### 5. 베이스라인 (필수 비교 대상)

LSTM이 능가해야 의미 있는 4종 baseline. 각각이 변동성 예측의 어떤 측면을 대표.

#### 5-1. HAR-RV — Heterogeneous Autoregressive Realized Volatility (Corsi 2009)

```
RV[t+h] = β₀ + β_d · RV_d[t] + β_w · RV_w[t] + β_m · RV_m[t]

  RV_d (daily)   = 1일 RV (가장 최근)
  RV_w (weekly)  = 5일 평균 RV
  RV_m (monthly) = 22일 평균 RV
```

- **이론적 근거**: 시장 참가자가 단기·중기·장기 시야로 변동성을 본다는 가설을 OLS로 단순화
- **구조**: 고작 4개 계수의 선형 회귀이지만 **변동성 예측의 사실상 학술 표준**
- **강력함**: 단순한 선형 모델이지만 LSTM이 이걸 못 이기는 경우가 흔함
- **이걸 능가해야 LSTM 도입 정당화 가능**

#### 5-2. EWMA — Exponentially Weighted Moving Average (RiskMetrics, J.P. Morgan 1996)

```
σ²[t] = λ · σ²[t-1] + (1-λ) · log_ret[t-1]²

  λ = 0.94  (J.P. Morgan RiskMetrics 표준값)
```

- **직관**: "최근 수익률 제곱에 더 큰 가중치, 과거는 지수적으로 감소"
- **재귀 구조**: 변동성의 자기상관(persistence)을 단순 재귀로 포착
- **산업 표준**: 모든 risk management system이 EWMA를 기본 baseline으로 보유

#### 5-3. Naive — 직전 RV 유지

```
pred[t+h] = RV[t]   (가장 최근 RV를 그대로 미래로 broadcast)
```

- **가장 단순한 baseline**
- **변동성의 강한 자기상관 활용**: ACF lag 1 ≈ 0.9 (Persistence가 매우 강함)
- **이걸 못 이기면 모델 무의미**: 가장 무지성(naive)인 예측보다 못함

#### 5-4. Train-Mean — 훈련셋 평균값

```
pred[t+h] = mean( RV_train )
```

- **가장 단순한 statistical baseline**
- **R²_train_mean의 분모 정의에 사용**
- **최소 sanity check**: 이걸 못 이기면 학습이 작동 안 함

---

### 6. 관문 (Phase 1.5 PASS 조건)

다음 **3개 모두 충족** 시에만 PASS. 어느 하나라도 실패 시 분기 분석 진행.

#### 6-1. 관문 1: `LSTM RMSE < HAR-RV RMSE` (105 fold 평균 기준)

- **의미**: 학술 표준 베이스라인을 능가
- **왜 필요한가?**: HAR-RV는 단순한 선형 회귀이지만 매우 강력. LSTM은 비선형 모델이면서 추가 복잡도(파라미터 4,513개)를 가지고 있는데, 이걸 능가 못 하면 추가 복잡도가 정당화 안 됨
- **통과 시 의미**: LSTM이 비선형 패턴 포착에 가치 있음 → "변동성 예측 가능"의 명확한 증거

#### 6-2. 관문 2: `R²_train_mean > 0`

- **의미**: 가장 단순한 baseline (train 평균)을 능가
- **왜 필요한가?**: 음수면 모델이 단순 평균 예측보다 못함 → 학습 자체 실패 신호
- **통과는 최소 sanity check**: 통과 못하면 하이퍼파라미터·loss·아키텍처 재검토 필요

#### 6-3. 관문 3: `pred_std / true_std > 0.5`

```
ratio = std(y_pred) / std(y_true)
```

- **의미**: 예측값의 분산이 실제값 분산의 50% 이상
- **0.5 미만이면 mean-collapse**: 모델이 평균값 근처만 출력 (학습 실패의 흔한 패턴)
- **통과 시 의미**: 모델이 actual variability의 일부를 포착하고 있음
- **Phase 1 §9.C 진단 셀 패턴 그대로 재사용**

---

### 7. 모델·학습·Walk-Forward — 정밀도 우선 (2026-04-26 갱신)

> **우선순위 변경 (2026-04-26, 사용자 결정)**: Phase 1 과의 비교 공정성보다 **변동성 예측의 정밀도 최대화** 가 우선. EDA §5-확장에서 lag 60 ACF 0.23, lag 126 ACF 0.13, 95% CI 진입 첫 lag = 220 (long-memory 시그니처) 확인 후 Walk-Forward 파라미터 재조정.

#### 7-1. 모델 구조

```python
LSTMRegressor(
    input_size=1,        # log_ret² 단일 채널 (1차 시도; HAR 다채널은 추후 별도 검증)
    hidden_size=32,      # LSTM 은닉 차원
    num_layers=1,        # 단일 레이어
    dropout=0.3,         # 출력 직전 dropout
    batch_first=True,    # (Batch, Time, Feature) 축 순서
)
```

- **파라미터 수**: 4,513개 (Phase 1 3차 Run과 동일 capacity 유지)
- **왜 모델 구조는 그대로?**: Walk-Forward 파라미터 변경만으로 정밀도 향상 효과를 분리 측정. 모델 capacity 도 동시에 키우면 어느 변경이 효과인지 분리 불가.

#### 7-2. 손실 함수 — MSE Loss (Huber에서 변경)

##### MSE Loss (Mean Squared Error)

```
loss = mean( (y_pred - y_true)² )
```

- **정규분포 가정**: 가우시안 가정 하 최대우도 추정과 등가
- **직관**: 예측 오차의 제곱 평균
- **모든 오차에 동일한 페널티 (gradient는 오차에 비례, linear)**

##### Huber Loss (Phase 1 사용)

```
loss = {
    0.5 × e²              if |e| < δ        ← 작은 오차: MSE처럼 작동 (제곱)
    δ × (|e| - 0.5×δ)     if |e| >= δ      ← 큰 오차: MAE처럼 작동 (선형)
}
  e = y_pred - y_true
  δ = 임계값 (Phase 1: 0.01)
```

- **이상치 강건성**: 큰 오차의 영향 제한
- **Phase 1에서 필요했던 이유**: 수익률 fat-tail (첨도 15) → outlier 방어

##### **왜 Phase 1.5에서 MSE로 변경?**

1. **Log-RV는 거의 정규 분포** (log 변환의 핵심 효과) → MSE의 가우시안 가정 충족
2. **Phase 1의 δ=0.01은 수익률 ~1% 스케일에 맞춤**. Log-RV 스케일은 약 -4.5 ± 0.4로 다름 → δ 재튜닝 필요
3. **MSE가 학술 표준과 정합** (HAR-RV도 MSE 최소화로 OLS 적합)
4. **단순성**: 추가 하이퍼파라미터 δ 제거

##### Fallback 옵션

Log-RV 변환 후에도 점프(COVID 2020, 2022 긴축 등) 잔존이 발견되면 `train.py`에 Huber(δ=0.1) fallback 옵션 보존.

#### 7-3. 옵티마이저 — AdamW

##### Adam (Adaptive Moment Estimation)

```
m[t] = β₁ · m[t-1] + (1-β₁) · ∇L         ← 1차 모멘트 (gradient의 지수이동평균, 모멘텀 효과)
v[t] = β₂ · v[t-1] + (1-β₂) · (∇L)²      ← 2차 모멘트 (gradient² 의 지수이동평균, 분산 효과)
θ[t] = θ[t-1] - lr · m_hat / (sqrt(v_hat) + ε)   ← parameter별 adaptive lr
```

- **장점**: 모멘텀 + parameter별 적응적 학습률 → 안정적 수렴

##### AdamW (Decoupled Weight Decay, Loshchilov & Hutter 2019)

- **차이**: L2 정규화(weight decay)를 옵티마이저 step에서 **분리**하여 적용
- **이유**: Adam의 adaptive lr이 L2 효과를 왜곡 → AdamW가 더 정확한 정규화 적용
- **Transformer·LSTM 학습의 사실상 표준**

##### 설정값

| 인자 | 값 | 의미 |
|---|---|---|
| `lr` (learning rate) | 1e-3 (0.001) | PyTorch 기본값. 적당히 안정적인 학습 속도 |
| `weight_decay` | 1e-3 (0.001) | L2 정규화 강도. **Phase 1 3차 Run에서 1e-4 → 1e-3으로 강화** (134 샘플 환경 과적합 방지) |
| `betas` | (0.9, 0.999) | β₁(모멘텀), β₂(분산) 기본값 |

#### 7-4. 학습률 스케줄러 — ReduceLROnPlateau

```
if val_loss가 patience epoch 동안 개선 없음:
    lr ← lr × factor
```

| 인자 | 값 | 의미 |
|---|---|---|
| `patience` | 3 | 3 epoch 동안 val_loss 개선 없으면 발동 |
| `factor` | 0.5 | lr을 절반으로 감소 |
| `mode` | 'min' | val_loss를 최소화 (작아질수록 좋음) |

- **효과**: 학습 후반부 fine-tuning. 큰 lr로 빠르게 수렴 후, 정체 시 작은 lr로 미세 조정
- **EarlyStopping과의 차이**: 스케줄러는 **lr 조정**만, EarlyStopping은 **학습 중단**

#### 7-5. EarlyStopping

```
if val_loss가 patience epoch 동안 개선 없음:
    학습 중단
```

| 인자 | 값 | 의미 |
|---|---|---|
| `patience` | 5 | 5 epoch 정체 시 stop |

- **목적**: 과적합 방지 + 학습 시간 절약
- **best 체크포인트**: 학습 중 val_loss 최소 시점의 weights를 최종 사용

#### 7-6. Walk-Forward 구조 — 정밀도 우선 재조정 (2026-04-26)

| 구성 | 변경 전 (Phase 1 동일) | **변경 후 (Phase 1.5)** | 근거 |
|---|---|---|---|
| IS (In-Sample) | 231 영업일 (~11개월) | **504 영업일 (~24개월)** | 훈련 샘플 fold 당 134 → **441** (3.3배). Run 4·5 다변량 실패 root cause(샘플 부족) 직접 해결 |
| Purge | 21 영업일 | **21** (유지) | 타깃 forward 21일 윈도우 → 21일 누수 차단 필수 |
| Embargo | 21 영업일 | **63 영업일** | log(RV) ACF lag 21=0.55, lag 42=0.37, **lag 63=0.21** (95% CI 0.04 의 5배) → embargo=63 으로 자기상관 잔존 강화 차단 |
| OOS (test) | 21 영업일 | **21** (유지) | 1개월 단위 평가 |
| Step | 21 영업일 | **21** (유지) | rolling sliding |
| 예상 fold 수 | 105 | **90** | 105 → 90 (-15, 약 14% 감소). 통계 신뢰도는 충분 (학술 권장 30+ fold) |
| OOS 총 샘플 | 2,205 | **1,890** | 90 fold × 21 |
| seq_len | 63 | **63** (유지) | PACF lag 22 spike (0.34) 충분 포함, 변경 시 추가 변수와 분리 측정 어려움 |

#### 7-6-1. 본 변경의 정량 근거 (EDA §5-확장 결과)

**Long-memory 시그니처 확인** (Andersen, Bollerslev, Diebold, Labys 2003 와 일치):

| lag | SPY ACF | QQQ ACF | 95% CI 밖? |
|---|---|---|---|
| 21 | 0.5566 | 0.5848 | ✓ (강한 잔존) |
| 42 | 0.3695 | 0.3667 | ✓ |
| **63** | **0.2075** | **0.1921** | ✓ |
| 126 | 0.1292 | 0.1866 | ✓ |
| 252 | -0.1188 | -0.1448 | ✓ (1년 mean-reversion) |
| **95% CI 진입 첫 lag** | **222** | **208** | — |

이론적으로 자기상관 완전 차단은 embargo ≥ 200 필요. 그러나 OOS=21, IS=504 환경에서 그렇게 큰 embargo 는 fold 수 급감. **실무적 타협점**: embargo=63 으로 lag 63 ACF 0.21 까지 차단 → 부분적이지만 의미 있는 강화.

#### 7-6-2. 입력 채널 — 1ch 유지 (1차 시도)

`log_ret²` univariate. **이유**: Walk-Forward 파라미터 변경 효과를 먼저 분리 측정. 결과 보고 후 HAR 3ch (`[RV_d, RV_w, RV_m]`) 별도 시도 가능 (PACF lag 22 spike 직접 활용 후보).

---

### 8. Step별 작업 흐름 (사용자 체크포인트 명시)

| Step | 작업 | 사용자 체크포인트 |
|---|---|---|
| 0 | 폴더 생성·scripts 복사 | 폴더 구조·파일 목록 합의 |
| 1 | `01_volatility_eda.ipynb` (§1~§9) | RV 분포·ACF 결과 보고 → Log-RV 채택 합의, embargo=21 충분성 확인 |
| 2 | `targets_volatility.py`·`metrics_volatility.py`·`baselines_volatility.py` 작성 + 단위 테스트 (4+8+4건) | 모듈 API 정의서 합의 |
| 3 | `02_volatility_lstm.ipynb` 105 fold × 2 ticker 학습 | 첫 fold 결과(loss curve) 보고, 105 fold 종합 결과 보고 |
| 4 | `03_baselines_and_compare.ipynb` HAR-RV/EWMA/Naive 비교 + 관문 판정 | PASS/FAIL 보고 — "변동성 예측이 가능한가?" 질문에 명확한 답변 |

### 9. 누수 방지 체크리스트 (변동성 특수 함정)

| # | 함정 | 방어 |
|---|---|---|
| 1 | `shift(-21)` 부호 누락 → 미래 참조 | §4 검증 셀 (assert + 5행 표) |
| 2 | `np.log(0)` 또는 `np.log(NaN)` → -inf/NaN 전파 | rolling 직후 NaN drop, `assert (rv > 0).all()` |
| 3 | HAR-RV의 monthly(22) lookback이 fold 외부 참조 | `fit_har_rv` 내부 `train_idx` 한정 슬라이싱 |
| 4 | EWMA seed가 test 데이터 포함 → 누수 | seed = train의 마지막 EWMA 값으로 고정 |
| 5 | ddof 불일치 (pandas 기본 1, numpy 기본 0) | `assert` 시 `ddof=1` 명시 통일 |

### 10. 함정·리스크

- **HAR-RV는 매우 강력한 baseline** — LSTM이 못 이길 가능성 실재. 이 경우 결과는 "변동성 예측은 가능하지만 LSTM은 HAR-RV 수준에 못 미침"으로 명확히 보고. Negative result도 본 단계 목표(질문에 답하기) 달성.
- **체제 변화** (COVID 2020, 2022 긴축): fold별 metric 분리 보고로 진단 (§9.E 박스플롯).
- **Log-RV 변환 후 점프 잔존** 시: Huber(δ=0.1) fallback 옵션을 `train.py`에 보존.
- **Long-memory 잔존 차단 불완전**: embargo=63 으로도 lag 63+ ACF 잔존 (lag 126 ACF 0.13~0.19). 첫 fold 등 일부 fold OOS metric 이 부풀려질 가능성 — §9.E 박스플롯의 IQR 분포와 fold 0~5 의 outlier 여부로 사후 진단.
- **PACF lag 22 spike (0.34)**: monthly cycle 의 강한 직접 의존성. seq_len=63 은 lag 22 를 충분히 포함하므로 LSTM 이 자연스럽게 학습 가능. 그러나 명시적 활용을 위해 추후 HAR 3ch 입력 시도 가치 ↑.

---

## Critical Files (수정·신규 작성 대상)

**복사 (변경 없음, Phase 1 격리 보존)**:
- 시계열_Test/Phase1_5_Volatility/scripts/setup.py — 출처: [Phase1_LSTM/scripts/setup.py](시계열_Test/Phase1_LSTM/scripts/setup.py)
- 시계열_Test/Phase1_5_Volatility/scripts/dataset.py — 출처: [Phase1_LSTM/scripts/dataset.py](시계열_Test/Phase1_LSTM/scripts/dataset.py)
- 시계열_Test/Phase1_5_Volatility/scripts/models.py — 출처: [Phase1_LSTM/scripts/models.py](시계열_Test/Phase1_LSTM/scripts/models.py)
- 시계열_Test/Phase1_5_Volatility/scripts/train.py — 출처: [Phase1_LSTM/scripts/train.py](시계열_Test/Phase1_LSTM/scripts/train.py) (loss_type='mse' 옵션 추가)
- 시계열_Test/Phase1_5_Volatility/00_setup_and_utils.ipynb — 출처: [Phase1_LSTM/00_setup_and_utils.ipynb](시계열_Test/Phase1_LSTM/00_setup_and_utils.ipynb)

**신규 작성**:
- 시계열_Test/Phase1_5_Volatility/scripts/targets_volatility.py — Log-RV 빌더 + 누수 검증 (참조 패턴: [Phase1_LSTM/scripts/targets.py](시계열_Test/Phase1_LSTM/scripts/targets.py))
- 시계열_Test/Phase1_5_Volatility/scripts/metrics_volatility.py — RMSE/QLIKE/R²_train_mean/MZ (참조 패턴: [Phase1_LSTM/scripts/metrics.py](시계열_Test/Phase1_LSTM/scripts/metrics.py))
- 시계열_Test/Phase1_5_Volatility/scripts/baselines_volatility.py — HAR-RV/EWMA/Naive
- 시계열_Test/Phase1_5_Volatility/01_volatility_eda.ipynb — RV 분포·ACF·정상성·체제 진단 (참조: [Phase1_LSTM/01_eda_statistics.ipynb](시계열_Test/Phase1_LSTM/01_eda_statistics.ipynb))
- 시계열_Test/Phase1_5_Volatility/02_volatility_lstm.ipynb — 변동성 LSTM 본 실험 (참조: [Phase1_LSTM/02_setting_A_daily21.ipynb](시계열_Test/Phase1_LSTM/02_setting_A_daily21.ipynb) §1~§10 + §9.A~F)
- 시계열_Test/Phase1_5_Volatility/03_baselines_and_compare.ipynb — HAR-RV/EWMA/Naive vs LSTM 통합 비교
- 시계열_Test/Phase1_5_Volatility/README.md, PLAN.md, 재천_WORKLOG.md

**재사용 (변경 없음)**:
- [results/raw_data/SPY.csv](시계열_Test/Phase1_LSTM/results/raw_data/SPY.csv), [QQQ.csv](시계열_Test/Phase1_LSTM/results/raw_data/QQQ.csv) — Phase 1 다운로드 결과 그대로 활용

---

## Verification (End-to-End)

1. **Step 1 누수 검증** (`01_volatility_eda.ipynb` §8)
   - `verify_no_leakage_logrv(log_ret, target)` PASS
   - 첫 5행 육안 표 출력
   - 단위 테스트: `target[t] == log( log_ret[t+1:t+22].std(ddof=1) )` (3개 무작위 시점)

2. **Step 2 모듈 단위 테스트** (Phase 1 패턴 동일)
   - `targets_volatility.py`: 4건 (누수 검증, ddof, NaN 처리, log domain)
   - `metrics_volatility.py`: 8건 (rmse, qlike, r2_train_mean, mz_regression, baseline_metrics, summarize_folds, edge cases)
   - `baselines_volatility.py`: 4건 (HAR-RV 계수, EWMA recursion, Naive shift, fold 외부 참조 차단)

3. **Step 3 노트북 Run All 재현성**
   - seed=42 고정, GPU/CPU 동일 결과
   - `results/volatility_lstm/{SPY,QQQ}/metrics.json`에 hyperparams + 105 fold 메트릭 직렬화

4. **Step 4 관문 판정 보고** — 본 단계의 핵심 산출물
   ```
   "변동성 예측이 가능한가?" 답변:
     PASS 조건 (모두 충족 시):
       ✓ LSTM RMSE < HAR-RV RMSE
       ✓ R²_train_mean > 0
       ✓ pred_std / true_std > 0.5
   ```
   `comparison_report.md` 자동 생성, 8행 비교 표 (SPY/QQQ × LSTM/HAR/EWMA/Naive)

5. **시각화 산출물 한글 폰트 OK 육안 확인**
   - 학습곡선 갤러리(§9.A) / best_epoch 분포(§9.B) / 예측 분포 sanity(§9.C) / 잔차 시계열·MZ 산점도(§9.D) / 박스플롯(§9.E) / Train/Val/Test 갭(§9.F)

6. **사용자 단계별 승인** — Step 0~4 종료 시점마다 결과 보고 + 다음 진행 명시 승인 (CLAUDE.md "단계별 대화형 피드백" 원칙)

---

## Appendix: 변동성 예측의 포트폴리오 활용 (배경 참고용 — 본 단계 평가 대상 아님)

> ⚠️ 아래는 Phase 1.5의 **결과를 추후 어떻게 활용할 수 있는지**에 대한 배경 정보입니다. **본 단계의 PASS/FAIL 판정·진행에는 영향 없음**. Phase 1.5는 오직 "변동성 예측 가능성"만 검증합니다.

### A. 효율적 프론티어 — 포트폴리오 최적화 출발점

Markowitz(1952) 평균-분산 최적화의 핵심:

```
                     ↑ 기대수익률
                     |
                     |       . . . .
                     |     .  · ·  ← 효율적 프론티어
                     |   .  ·  ·     (같은 위험에서 최대 수익)
                     | .  ·  ·
                     +─────────────→ 위험 (변동성, σ)
```

포트폴리오 최적화는 본질적으로 **수익률(μ)** 과 **변동성(σ)** 두 입력이 모두 필요.

### B. Black-Litterman 모델의 두 입력

| 입력 | 의미 | 본 프로젝트에서 |
|---|---|---|
| **Q (View 벡터)** | "각 자산의 기대 수익률" | Phase 1 LSTM (수익률 예측) |
| **Ω (불확실성 행렬)** | "각 view의 신뢰도(분산)" | Phase 1.5 LSTM (변동성 예측) — **본 단계 산출물의 잠재적 입력처** |

→ Phase 1.5의 결과(변동성 예측 정확도)는 추후 BL 통합 단계에서 Ω 입력으로 활용 가능.

### C. 변동성 예측만으로 벤치마크를 이길 수 있는가? (학술적 답변)

**결론**: "불가능"이 아니라 **"평가 기준에 따라 다름"**.

| 비교 기준 | 변동성 단독 우위? | 학술 근거 |
|---|---|---|
| **Sharpe 비율** | ✅ **증명** | Moreira & Muir (2017, *Journal of Finance*) — 시장 0.45→0.62 |
| **Max Drawdown** | ✅ **증명** | Harvey et al. (2018) — 90년 multi-asset |
| **하락장 방어** | ✅ **증명** | 2008, 2020 실증 다수 |
| **S&P 500 절대수익** | ⚠️ **시장 환경 의존** | 2010s 강세장 underperform 다수 |
| **1/N 등배 능가** | ⚠️ **혼재** | DeMiguel, Garlappi, Uppal (2009, *Review of Financial Studies*) — 추정 정확도 충분히 높지 않으면 1/N에 패배 |

#### 핵심 학술 인용

1. **Moreira & Muir (2017) "Volatility-Managed Portfolios"** *Journal of Finance*
   - 변동성 역수에 비례한 vol-scaling
   - 1926~2015 미국 주식 + 다양한 팩터
   - **Sharpe 비율 0.45 → 0.62 (38% 향상)**

2. **Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert (2018) "The Impact of Volatility Targeting"** Man Group / *Journal of Portfolio Management*
   - 다양한 자산군에 vol targeting 적용 (1926~2017)
   - **Sharpe 일관 향상, drawdown 유의미 감소**

3. **DeMiguel, Garlappi, Uppal (2009) "Optimal Versus Naive Diversification"** *Review of Financial Studies*
   - **1/N 등배가 14가지 정교한 평균-분산 모델을 out-of-sample에서 모두 능가**
   - 함의: 추정 정확도가 충분치 않으면 1/N이 더 나음

4. **Asness, Frazzini, Pedersen (2012) "Leverage Aversion and Risk Parity"** *Financial Analysts Journal*
   - Risk Parity가 60/40 대비 risk-adjusted 우위

### D. Phase 1.5 결과의 잠재적 활용처 (추후 단계)

| # | 활용 | 필요 정확도 |
|---|---|---|
| 1 | **BL의 Ω 입력** | LSTM이 EWMA·Naive보다 의미 있게 우위 |
| 2 | **Volatility Targeting** | 절대 변동성 예측 정확도 (RMSE 기준) |
| 3 | **Risk Parity** | 자산 간 변동성 비율 정확도 |
| 4 | **Regime Detection** | 분위수 기반 분류 정확도 (저/고 변동) |

### E. 본 plan의 평가 범위 (다시 강조)

**평가 대상**: "변동성 예측이 가능한가?" 단일 질문에 대한 명확한 답변
**평가 비대상** (추후 단계로 미룸):
- 포트폴리오 구축 (BL, Mean-Variance, Risk Parity)
- 벤치마크(S&P 500, 1/N, 60/40) 대비 alpha/Sharpe/drawdown
- BL의 Q/Ω 통합

Phase 1.5 결과가 "변동성 예측 PASS"이면 위 추후 단계의 입력 자료가 되고, "FAIL"이면 변동성 예측을 대체할 다른 신호(VIX 직접 사용, GARCH 등)를 검토하는 단계로 이행.

---

# v2 분기 신설 (2026-04-27, HAR 3ch 입력 검증)

## v1 결과 요약 (2026-04-27 학습 완료)

90 fold 학습 결과 — 관문 2, 3 모두 FAIL:

| metric | SPY | QQQ | 관문 |
|---|---|---|---|
| rmse | 0.4688 | 0.4329 | (§03 비교) |
| **r2_train_mean** | **-2.93** | **-2.26** | ❌ FAIL (>0) |
| **pred_std_ratio** | **0.35** | **0.35** | ❌ FAIL (>0.5) |
| mz_beta | -0.68 | -0.31 | (편향 진단 — 음의 상관) |
| best_epoch | 9.7 | 9.1 | ✅ 학습 자체 수렴 |

**진단**: "학습 코드 정상, 모델은 mean-collapse + underfit 상태로 수렴, fold 마다 결과 매우 다름."

## v2 진입 결정 (사용자 가설 검증)

### 검증 가설
> "log_ret² 1ch 의 정보 빈약이 mean-collapse 의 주범인가?"
> (가설 a — 변수 부족, 재평가 비중 50~60%)

### v1 ⇄ v2 차이 (단 1가지 변경 — 통제 실험)

| 항목 | v1 | **v2** |
|---|---|---|
| **입력 채널** | `log_ret²` (1ch) | **HAR 3채널 `[\|log_ret\|, RV_w, RV_m]`** |
| `input_size` | 1 | **3** |
| Walk-Forward / 모델 / 손실 / 옵티마 / seed / 분석기간 등 | — | **v1 동일** |

→ 변수만 변경 → 가설 a 의 직접 검증 (다른 가설 b, c, d 의 영향 분리).

### v2 입력 정의 (Corsi 2009 학술 표준 일간 적응)

```python
log_ret = log(adj_close).diff()

# 채널 1: 1일 변동성
RV_d[t] = |log_ret[t]|

# 채널 2: 5일 평균 변동성
RV_w[t] = sqrt( mean(log_ret²[t-4 : t+1]) )

# 채널 3: 22일 평균 변동성
RV_m[t] = sqrt( mean(log_ret²[t-21 : t+1]) )

# 시퀀스 shape: (Batch, 63, 3)
```

### v2 산출물 (격리 보존) — 사용자 확정 (2026-04-27)

```
Phase1_5_Volatility/
├── 02_volatility_lstm.ipynb              ← v1 (보존, 변경 X)
├── 02_v2_volatility_lstm_har3ch.ipynb    ← v2 신규 ✓
├── _build_02_v2_har_nb.py                ← v2 빌드 스크립트 ✓
└── results/
    ├── volatility_lstm/                  ← v1 결과 (보존)
    └── volatility_lstm_har3ch/            ← v2 결과 (신규) ✓
```

**확정 시점**: 2026-04-27 (사용자 권장안 채택). 다음 세션 (compact 후) 에서 빌드·학습 시작 예정.

### v2 결과 → §03 진입 결정 분기

```
v2 결과    | 해석
─────────────────────────────────────────────────────
v2 PASS    | 변수 부족 가설 a 확정 → §03 베이스라인 비교 진행
v2 FAIL    | LSTM 자체 한계 또는 CV 구조 영향 ↑ → 추가 실험 검토
v2 borderline | §03 비교로 HAR-RV 와 비교 후 종합 판정
```

## 미적용 옵션 (v2 결과 따라 추후 검토)

| 옵션 | 동기 | 우선순위 |
|---|---|---|
| IS 504→252 | 체제 일관성 ↑ | v2 효과 제한적이면 |
| Huber(δ=0.1) loss | outlier 강건 | log 변환 효과로 후순위 |
| dropout 0.3→0.5 | 정규화 강화 | 과적합 X 확인되어 후순위 |
| seq_len 63→126 | lookback 확장 | 신호 풍부 후 검토 |
