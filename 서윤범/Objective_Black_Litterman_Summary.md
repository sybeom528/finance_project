# 논문 요약: Objective Black-Litterman Views through Deep Learning

**제목:** Objective Black-Litterman views through deep learning: A novel hybrid model for enhanced portfolio returns  
**저자:** Xianran Su, Ke Lu, Jerome Yen (University of Macau)  
**저널:** Expert Systems With Applications, Vol. 295 (2026)  
**DOI:** 10.1016/j.eswa.2025.128868

---

## 1. 연구 배경 및 문제의식

### Black-Litterman 모델의 한계
- Black-Litterman(BL) 모델은 **시장 균형 수익률(Market Equilibrium Return)** 과 **투자자 관점(Investor Views)** 을 Bayesian 방식으로 결합하여 포트폴리오를 최적화하는 모델
- 마코위츠의 Mean-Variance 모델의 단점(입력값 민감성, 파라미터 불확실성 무시)을 극복하기 위해 설계됨
- **핵심 문제:** BL 모델의 성능은 투자자 관점(Q, Ω)의 품질에 크게 의존하는데, 이 관점이 인간의 편향(bias)·주관성(subjectivity)에 의해 오염될 수 있음

### 목표
> "투자자의 주관적 판단 없이, 딥러닝으로 **객관적·데이터 기반** 투자자 관점을 자동 생성하는 것"

---

## 2. 관련 연구

### 시계열 분해 알고리즘의 발전
| 알고리즘 | 특징 |
|---|---|
| EMD (Empirical Mode Decomposition) | 비선형·비정상 시계열 분해의 시초 |
| EEMD | EMD의 모드 혼합(mode mixing) 문제 완화 |
| CEEMD | 잡음 처리 개선 |
| **CEEMDAN** | 적응적 노이즈 조정으로 더 안정적·정확한 분해 달성 |

### CEEMDAN + LSTM 기반 선행 연구
- Cao et al. (2019): CEEMDAN-LSTM으로 S&P500, HSI, DAX, SSE 예측 - 단순 LSTM 대비 우위
- Lin et al. (2020): 환율 예측에 CEEMDAN-LSTM 적용
- Barua & Sharma (2023): CEEMDAN-GRU로 Fear/Greed 지수 예측 후 BL 모델에 통합
- Gao et al. (2024): CEEMDAN + 4개 신경망 조합으로 BL 관점 생성

### 기존 모델의 두 가지 한계
1. **LSTM 하이퍼파라미터 일반화 적용**: 각 IMF(고유 모드 함수)별로 개별 최적화 없이 동일 하이퍼파라미터 사용 → 예측력 손실
2. **선형 앙상블**: IMF 예측값을 단순 합산 → 오차 누적 및 비선형 관계 포착 실패

---

## 3. 이론적 배경

### 3.1 Black-Litterman 모델 수식

**단계 1: 시장 균형 수익률 산출**
$$\Pi = \lambda \Sigma w_{mkt}$$
- $\Pi$: 내재 균형 초과수익률 벡터 (N×1)
- $\lambda$: 위험 회피 계수 (논문에서 2.5 사용)
- $\Sigma$: 자산 수익률 공분산 행렬
- $w_{mkt}$: 시장 자산 배분 비중

**단계 2: 투자자 관점 파라미터**
- $P$: 관점 행렬 (K×N), 관점 대상 자산은 1, 그 외 0
- $Q$: 관점 벡터 (K×1) — **CGL 딥러닝 모델이 예측하여 생성**
- $\Omega$: 관점 불확실성 행렬 (대각행렬), $\Omega_{i,i} \approx p_i \Sigma p_i' \cdot \tau$

**단계 3: 사후 기대수익률 (Bayesian 결합)**
$$\mu = \left[(\tau\Sigma)^{-1} + P'\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\Pi + P'\Omega^{-1}Q\right]$$
- $\tau$: 균형수익률 불확실성 스칼라 (논문에서 0.025 사용)

**단계 4: 최적 포트폴리오 비중**
$$w_{new} = (\lambda\Sigma)^{-1}\mu$$

### 3.2 LSTM (Long Short-Term Memory)
- RNN의 기울기 소실 문제 해결
- 입력 게이트($i_t$), 출력 게이트($o_t$), 망각 게이트($f_t$) 세 가지 게이트로 정보 흐름 제어
- 장기·단기 의존성을 동시에 캡처하는 데 강점

### 3.3 CEEMDAN 알고리즘
- 원본 시계열 $y(t)$에 가우시안 백색 노이즈를 반복 추가하여 EMD 분해
- N번 추가 후 평균을 취해 IMF(Intrinsic Mode Function) 추출
- 분해 결과: $y(t) = \sum_{i=1}^{K} C_i(t) + r_K(t)$
  - 고주파 IMF: 단기 변동성·노이즈
  - 중주파 IMF: 중·단기 추세
  - 저주파 IMF/잔차: 장기 추세

### 3.4 유전 알고리즘 (GA)
- 자연선택 원리 모방: 선택(Selection) → 교차(Crossover) → 변이(Mutation)
- LSTM 하이퍼파라미터 최적화에 적용
- 최적화 대상: 학습률, 은닉층 수, 뉴런 수, 배치 크기

---

## 4. 제안 모델: CGL-BL

### 4.1 전체 구조

```
주가 수익률 데이터
        ↓
   [CEEMDAN 분해]
   IMF1, IMF2, ..., IMFK, 잔차
        ↓
 [Stage 2: GA-최적화 LSTM (GLSTM)]
  각 IMF에 개별 하이퍼파라미터 최적화 적용
        ↓
 [Stage 3: 앙상블 LSTM]
  IMF별 예측값을 비선형 통합 → 주식 수익률 예측 (Q 벡터)
        ↓
   [Black-Litterman 모델]
  시장 균형수익률(Π) + 투자자 관점(Q) → 사후 기대수익률(μ)
        ↓
   [최적 포트폴리오 비중 w_new]
```

### 4.2 세 가지 핵심 기여

| 기여 | 내용 | 해결하는 문제 |
|---|---|---|
| **GLSTM** | 각 IMF마다 GA로 개별 LSTM 하이퍼파라미터 최적화 | 주파수별 패턴을 제대로 학습 못하는 문제 |
| **앙상블 LSTM** | IMF 예측값들을 LSTM으로 비선형 통합 | 단순 합산으로 인한 오차 누적 |
| **CGL-BL 통합** | CGL 예측을 BL 모델의 투자자 관점(Q)으로 변환 | 주관적 투자자 관점 의존성 제거 |

### 4.3 투자자 관점 생성 방법
- 10개 구성 종목 수익률을 CGL로 예측
- 예측 결과 벡터 $[R_1, R_2, ..., R_k]$ → **절대적 관점(Absolute View)** 방식으로 Q 행렬 구성
- P 행렬: k×N, 각 종목에 대한 관점이 독립적으로 설정됨

---

## 5. 실험 설계

### 5.1 데이터
| 시장 | 지수 | 데이터 | 학습 기간 | 테스트 기간 |
|---|---|---|---|---|
| 중국 | SSE 50 | 일간 | 2022.01–2023.12 (Period 1) | 2024.01–09 (3구간) |
| 미국 | DJIA | 주간 | 2019.01–2023.12 | 2024.01–12 |

- SSE 50 테스트: 상승(Period 1), 변동(Period 2), 하락(Period 3) 세 시장 국면으로 분리
- 각 시장에서 10개 대표 종목 선정하여 관점 생성

### 5.2 벤치마크 포트폴리오
1. **Index**: 지수 자체
2. **Market Weight Portfolio**: 시총 기반 패시브 포트폴리오
3. **Mean-Variance Portfolio**: 마코위츠 최적화
4. **CL-BL**: CEEMDAN-LSTM 기반 BL 모델 (비교 기준)

### 5.3 평가 지표
- **예측 성능**: MAE, RMSE, R²
- **포트폴리오 성능**: Sharpe Ratio, Sortino Ratio, Maximum Drawdown, 누적 수익률

---

## 6. 실험 결과

### 6.1 예측 성능 (CGL vs 벤치마크)
- CGL은 SSE 50 전 구간과 DJIA에서 **MAE, RMSE 기준 최저**, **R² 기준 최고**를 대부분 달성
- 단순 LSTM의 R²는 대부분 음수 → BL에 활용 불가 수준
- **절제 연구(Ablation Study)** 결과: GLSTM과 앙상블 LSTM 두 요소 모두 예측력에 독립적·시너지적으로 기여

### 6.2 포트폴리오 성능 요약

**SSE 50 (일간 수익률, 3일 리밸런싱)**
| 포트폴리오 | 수익률(거래비용 후) | Sharpe | Max Drawdown |
|---|---|---|---|
| **CGL-BL** | **86.95% (66.28%)** | **4.691** | 4.25% |
| CL-BL | 81.97% (61.83%) | 4.482 | 4.47% |
| Mean-Variance | 26.59% (12.51%) | 1.601 | 11.00% |
| SSE 50 Index | −3.99% | −0.756 | 12.91% |

**DJIA (주간, 1주 리밸런싱)**
| 포트폴리오 | 수익률(거래비용 후) | Sharpe | Max Drawdown |
|---|---|---|---|
| **CGL-BL** | **122.49% (100.81%)** | **4.443** | 3.56% |
| CL-BL | 95.02% (75.97%) | 3.776 | 4.16% |
| Mean-Variance | 53.07% (38.04%) | 1.918 | 9.31% |
| DJIA Index | 24.00% | 1.432 | 4.61% |

### 6.3 SHAP 해석 가능성 분석
- **고주파 IMF(IMF1)**: SHAP 값 낮음 → 노이즈 성분, 모델이 무시
- **중주파 IMF(IMF3-6)**: 최근 시점 SHAP 높음 → 중·단기 추세 포착
- **저주파 IMF/잔차**: 시점 근접할수록 SHAP 증가 → 장기 추세 반영

### 6.4 롤링 윈도우 분석
- 보유 기간이 길어질수록 CGL-BL 초과수익률 감소
- 3일 → 6일로 연장 시 연환산 초과수익률 **36.63% 감소**
- 단기 알파 신호를 활용하므로 **짧은 리밸런싱 주기**가 더 유리

### 6.5 장기 투자 (DJIA 월별 리밸런싱, 2021–2024)
- 월간 리밸런싱: CGL-BL 44.85% vs DJIA Index 15.83%
- 장기에서도 초과수익 유지하나, 단·중기 대비 상대적 우위 약화
- 장기 적용 개선 방향: 거시경제 변수(경기 사이클, 인플레이션) 추가 필요

### 6.6 파라미터 민감도 분석
- **τ (0.02 → 0.03)**: 투자자 관점 영향력 증가, 포트폴리오 비중 변화 소폭 확대 (약 5% 이내)
- **λ (2.0 → 3.0)**: 위험 회피 증가 → 시장 균형 비중 수렴, 변동성 감소
- 두 파라미터 모두 합리적 범위 내에서 안정적 → **모델의 강건성 확인**

---

## 7. 결론

### 핵심 성과
- **CGL(CEEMDAN-GLSTM-LSTM)** 모델로 객관적 BL 투자자 관점 자동 생성
- SSE 50 (9개월): 3일 리밸런싱 기준 초과수익률 **+70.27%** (거래비용 전)
- DJIA (1년): 1주 리밸런싱 기준 초과수익률 **+76.81%**
- 모든 시장 국면(상승·변동·하락)에서 벤치마크 대비 우위

### 두 가지 핵심 개선점
1. **GA 기반 개별 LSTM 최적화** → 주파수별 패턴 학습 강화
2. **비선형 앙상블 LSTM** → IMF 간 복잡한 관계 포착, 오차 누적 방지

### 한계 및 향후 연구 방향
1. **이종 알고리즘 조합**: IMF별로 LSTM 외 다른 모델(Transformer, CNN 등) 탐색
2. **추가 팩터 통합**: 거시경제 지표, 펀더멘털 변수, 고차 통계량(왜도·첨도)을 BL 관점 생성에 통합
3. **장기 투자 적용성**: 월별·분기별 리밸런싱에서의 성능 개선

---

## 8. 모델 구현 요약 (빠른 참조용)

```python
# CGL-BL 파이프라인 개요

# Step 1: 수익률 시계열 CEEMDAN 분해
imfs = ceemdan(stock_returns)  # [IMF1, IMF2, ..., IMFK, residual]

# Step 2: 각 IMF에 GA로 LSTM 하이퍼파라미터 최적화 후 예측
predictions = []
for imf in imfs:
    best_params = genetic_algorithm_optimize(imf)
    lstm = LSTM(**best_params)
    predictions.append(lstm.predict(imf))

# Step 3: 앙상블 LSTM으로 비선형 통합
ensemble_lstm = LSTM(...)
Q = ensemble_lstm.predict(predictions)  # 투자자 관점 (예측 수익률)

# Step 4: Black-Litterman 모델 적용
Pi = risk_aversion * cov_matrix @ market_weights  # 균형 수익률
mu = bl_posterior(Pi, P, Q, Omega, tau)           # 사후 기대수익률
w_new = (risk_aversion * cov_matrix).inv() @ mu  # 최적 비중
```

**주요 하이퍼파라미터 설정값**
| 파라미터 | 값 | 의미 |
|---|---|---|
| λ (risk aversion) | 2.5 | 세계 평균 위험 회피 계수 |
| τ (prior uncertainty) | 0.025 | 균형수익률 불확실성 |
| GA 세대 수 | 20 | 유전 알고리즘 반복 횟수 |
| GA 개체 수 | 30 | 각 세대 후보 수 |
| 거래 비용 | 0.2% | 현실 반영 |

---

*요약 작성일: 2026-04-24*
