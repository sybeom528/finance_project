# HMM + Bayesian Q 동적 업데이트 실험 계획

> 작성일: 2026-05-01  
> 담당: 서윤범 (BL 모델) × 김하연 (HMM 레짐)  
> 목적: 고정 Q=0.003의 한계 극복 → 레짐 기반 동적 Q 설정

---

## 1. 배경 및 동기

### 현재 구조의 한계

현재 BL 모델은 Q=0.003 (월 0.3% 상승 기대)을 **모든 시장 상황에서 고정** 사용.

```
코로나 2020-03: 시장 -30% 급락 중에도 Q=0.003 유지 → 방어 기능 없음
GFC 2008-09:   동일 문제
```

### 해결 아이디어

변동성 예측(LSTM)을 단순 순위 구조(P 행렬)에만 쓰지 않고,  
HMM 레짐 posterior와 결합해 **Q를 동적으로 조정**.

---

## 2. 구현 구조 (4단계)

### 2-1. HMM 학습 (김하연 모델 재사용)

```python
# 김하연 Phase 2 모델 (6 features, n=4 또는 n=5)
features = ['mkt_rf', 'vol_21d', 'vol_63d', 'vix', 'vix_chg', 't10y2y']
model = GaussianHMM(n_components=4, covariance_type='full', n_iter=500)
model.fit(X_scaled)  # StandardScaler 전처리 필수
```

**주의: 현재 In-Sample 학습 (2004-2026 전체)**  
→ HMM 파라미터(transmat_, means_, covariances_)는 레짐 구조 파악용으로만 사용  
→ posterior 계산은 pred_date까지 데이터만으로 실시간 수행 (look-ahead bias 제어)

### 2-2. t 시점에서 P(레짐_{t+1}) 계산

```python
# pred_date까지 관측값만 사용 (walk-forward)
X_past = macro_df.loc[:pred_date, features]
X_scaled = scaler.transform(X_past)

_, posteriors = hmm_model.score_samples(X_scaled)
p_current = posteriors[-1]           # P(레짐_t | obs_1:t)

A = hmm_model.transmat_
p_next = p_current @ A               # P(레짐_{t+1}) — 전이행렬 적용
```

### 2-3. LSTM predicted vol로 베이즈 보정

```python
from scipy.stats import norm

# LSTM 전종목 예측 vol 중앙값 (시장 수준 신호, walk-forward)
lstm_vol_pred = lstm_preds.loc[pred_date].median()

# 6개 feature 중 vol_21d 인덱스 = 1
vol_idx = 1
likelihoods = np.array([
    norm.pdf(lstm_vol_pred,
             hmm_model.means_[i, vol_idx],
             np.sqrt(hmm_model.covariances_[i, vol_idx, vol_idx]))
    for i in range(n_regimes)
])

# Bayesian update: prior × likelihood → posterior
posterior = p_next * likelihoods
posterior /= posterior.sum()
```

### 2-4. Q 결정 (가중 평균)

```python
# n=4 기준 (Bull / Recovery / Neutral / Bear)
Q_by_regime = np.array([0.005, 0.003, 0.002, 0.000])

q_dynamic = float(posterior @ Q_by_regime)   # 기대 Q
```

---

## 3. 설계 결정 사항

### 레짐 수 선택

| n | BIC | 해석 | 포트폴리오 적합성 |
|---|-----|------|---------------|
| n=3 | 245M | Bull / Neutral / Bear | 단순, Crisis 포착 제한 |
| **n=4** | **220M** | **Bull / Recovery / Neutral / Bear** | **권장 — 레짐 균등 분포, 회복기 포착** |
| n=5 | 204M | +Mild Bull + Volatile + Crisis | BIC 최적, 해석 복잡 |

→ **n=4 사용** (포트폴리오 레짐 전환 목적에 최적)

### Q 값 설정 근거

| 레짐 | Q 값 | 경제적 해석 |
|------|------|-----------|
| Bull | 0.005 | 저변동 강세 → 강한 상향 뷰 |
| Recovery | 0.003 | 반등 구간 → 기본 뷰 유지 |
| Neutral | 0.002 | 관망세 → 뷰 약화 |
| Bear | 0.000 | 하락장 → 뷰 포기, Prior에 의존 |

**Q=0 (Bear)** 의미: BL 업데이트 없음 → CAPM prior π만으로 최적화  
→ 자동으로 방어적 포지션 (시총가중 전체 유니버스)

### bl_config.py 추가 실험

```python
{**BASELINE, 'name': 'q_hmm_bayesian',
 'q_mode': 'regime_bayesian',
 'hmm_n': 4,
 'Q_by_regime': [0.005, 0.003, 0.002, 0.000]},
```

---

## 4. Look-ahead Bias 처리

| 단계 | Bias 여부 | 처리 방법 |
|------|---------|---------|
| HMM 학습 (파라미터) | ⚠️ In-sample | 레짐 구조 파악용으로만 사용 (파라미터 고정) |
| posterior 계산 | ✅ WF | pred_date까지 데이터만 입력 |
| LSTM vol 예측 | ✅ WF | 이미 walk-forward 생성됨 |
| 전이행렬 적용 | ✅ | 과거 학습된 전이 구조 그대로 사용 |

**발표 시 설명**: "HMM 파라미터는 레짐의 통계적 특성 파악에 사용했고, 실제 Q 결정은 pred_date 이전 관측값 기반 posterior와 walk-forward LSTM 예측의 Bayesian 결합으로 이루어집니다."

---

## 5. 검증 필요 항목 (레짐 품질)

### 5-1. 반드시 확인해야 할 것

- [ ] **위기 기간 매핑 확인**: Crisis/Bear 레짐이 실제로 GFC(2008-09), 코로나(2020-02~04)를 포착하는가?
- [ ] **n=4 Recovery 전체 매핑 문제**: 이전에 n=4가 모두 Recovery로 나온 이슈가 실제 모델 문제인지 시각화 오류인지 확인 필요
- [ ] **월별 레짐 집계 방식**: 일별 HMM → 월별 BL 결정 시 월말 값 사용 vs 월평균 posterior 사용

### 5-2. 이론적 불일치 (인지하고 넘어갈 것)

- **t10y2y**: Ablation에서 BIC 악화(-58M)이지만 경제적 의미로 유지 결정 → 발표에서 질문 받을 수 있음
- **mkt_rf**: KW 검정에서 유의하지 않음(p=0.280)이지만 Ablation에서 핵심 기여 → 상충
- **In-sample HMM**: 완전한 WF가 아님 → 한계로 명시 필요

### 5-3. 성능 비교 실험

```
baseline (Q=0.003 고정)
  vs
q_hmm_bayesian (Q 동적)
```

기대 효과:
- 2020년 구간 최대낙폭(MDD) 개선
- Sharpe ratio 개선 (변동성 대비 수익)
- 유효 종목수 변화 (Bear 레짐 시 포지션 희석)

---

## 6. 구현 순서

1. **김하연에게 요청**: 학습된 모델 객체 pickle 저장
   ```python
   import pickle
   with open('hmm_n4_model.pkl', 'wb') as f:
       pickle.dump({'model': model, 'scaler': scaler}, f)
   ```

2. **macro feature 준비**: 99_run.ipynb에서 pred_date별 mkt_rf, VIX, vol 계산 (이미 일부 있음)

3. **bl_functions.py 추가**: `compute_q_regime_bayesian()` 함수

4. **99_run.ipynb**: monthly_cache에 레짐 posterior 추가

5. **99_analyze.ipynb**: 레짐별 Q 추이 시각화 추가 (시간축 × Q_t)

---

## 7. 미결 사항

- [ ] LSTM vol 예측 파일과 HMM feature의 날짜 정합성 확인
- [ ] Q_by_regime 값 sensitivity 분석 (0.005/0.003/0.002/0.000 vs 다른 설정)
- [ ] n=4 Recovery 레짐 문제 재확인 후 n=4/n=5 최종 선택
