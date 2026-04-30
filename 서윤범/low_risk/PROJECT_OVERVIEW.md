# Low-Risk BL + ML Q 예측 프로젝트 개요

---

## 1. 핵심 목표

> **`99_baseline.ipynb`의 `Q = 0.003` (근거 없는 고정값)을 ML로 예측한 데이터 기반 Q로 교체한다.**

---

## 2. Black-Litterman에서 Q란?

```
π (CAPM 균형 수익률)
+
Q (투자자 뷰 = P 포트폴리오의 기대 초과수익률)
─────────────────────────────────────────────
→ μ_BL (사후 기대 수익률) → 포트폴리오 최적화
```

**P 포트폴리오**:
- Long: `vol_21d` 하위 30% 종목 (저위험, 시총 가중)
- Short: `vol_21d` 상위 30% 종목 (고위험, 시총 가중)
- `sum(P) = 0` (Long 합 +1, Short 합 -1)

**Q = P 포트폴리오의 기대 수익률** = "다음 달에 저위험이 고위험을 얼마나 아웃퍼폼할 것인가?"

현재: `Q = 0.003` (근거 없음) → 목표: ML로 데이터 기반 Q 계산

---

## 3. ML Q 예측 방식 (Step3 → 분류)

### 원칙

Step3_IndividualStocks의 XGBoost **회귀**를 **분류**로 전환:

| | Step3 (기존) | 05_ML_Q (신규) |
|---|---|---|
| 타겟 | `fwd_excess_ret_1m` 연속값 (회귀) | 방향: up/neutral/down (분류) |
| 단위 | 종목별 예측 수익률 | 종목별 방향 → 포트폴리오 방향 |
| 피처 정규화 | GKX 횡단면 rank | 동일 |
| walk-forward | 연간 expanding window | 동일 |

### 흐름

```
Step3 monthly_panel.csv (종목 × 월 패널)
         ↓
  GKX 방식 피처 정규화
  - 주식 피처: 날짜별 횡단면 rank → [-1, 1]
  - 매크로 피처: rolling z-score
         ↓
  종목별 방향 분류 (XGBClassifier, 연간 expanding window)
  - 타겟: 각 종목의 fwd_excess_ret_1m 부호 (up/neutral/down)
  - 훈련: 현재 연도 이전 전체 패널 (N종목 × T월, ~125,000 행)
         ↓
  P 포트폴리오 수준으로 집계
  - portfolio_score = Σ P_i × (prob_up_i - prob_down_i)
  - portfolio_direction = "up" if score > 0 else "down"
         ↓
  Q_conditional = 훈련 기간 내 해당 방향의 P_ret 평균
  - Q_up   = mean(P_ret | 과거 up 달들)
  - Q_down = mean(P_ret | 과거 down 달들)
         ↓
  99_baseline.ipynb에서 Q 대체
```

---

## 4. 데이터 흐름

```
[Step3_IndividualStocks/data/monthly_panel.csv]
  이미 존재. 별도 재수집 불필요.
  (628종목, 2004~2025, 108,328행, ~68개 피처 + 매크로)

         ↓ 직접 로드

[05_ML_Q_predict.ipynb]
  → outputs/05_ml_q/ml_q_predictions.csv

         ↓ 로드

[99_baseline.ipynb 수정]
  기존: q = Q_FIXED (0.003)
  변경: q = ml_q.loc[pred_date, 'Q_conditional']
```

---

## 5. 각 노트북 역할

| 노트북 | 역할 | 상태 |
|---|---|---|
| `01_DataCollection.ipynb` | 기존 low_risk 데이터 | 변경 없음 |
| `02_DataCollection_ML.ipynb` | 확장 피처 수집 (미래 사용 대비) | 부차적 (지금 당장 불필요) |
| `05_ML_Q_predict.ipynb` | Step3 분류 → Q 예측 | **핵심** |
| `97/98/99_baseline.ipynb` | 기존 BL 백테스트 | Q 교체 대상 |

---

## 6. 핵심 용어 정리

| 용어 | 의미 |
|---|---|
| Q | P 포트폴리오의 기대 수익률. BL 투자자 뷰 입력값 |
| P | 저위험 Long + 고위험 Short. `sum(P) = 0` |
| P_ret | 실현된 Q. `P @ 실제수익률` |
| portfolio_score | `Σ P_i × (prob_up_i - prob_down_i)`. 양수면 포트폴리오 방향 up |
| Q_conditional | 예측 방향에 따른 역사적 P_ret 조건부 평균 |
| vol_20d | `rolling(20)`. Step3(GKX) 기준 |
| vol_21d | `rolling(21)`. 99_baseline P 분류 기준 |

---

## 7. 다음 단계

1. **`05_ML_Q_predict.ipynb` 실행**
   - Step3 패널(`monthly_panel.csv`) 그대로 사용
   - 분류 성능 확인 (Balanced Accuracy 기준)
   - Q 스프레드 확인 (Q_up - Q_down이 의미있게 다른가)

2. **성능이 부족하면**
   - DIR_THR 조정 (0.005 → 0 또는 더 크게)
   - 피처 추가 (02_DataCollection_ML 패널 사용)
   - 방식 2로 전환 (그룹 차분 시계열)

3. **성능 확인 후 `99_baseline.ipynb` 수정**
   - Q_FIXED 대신 ml_q_predictions.csv 로드
