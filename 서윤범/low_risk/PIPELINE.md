# 저위험 이상현상 기반 Black-Litterman 포트폴리오 파이프라인

## 전체 흐름

```
01_DataCollection
      ↓  monthly_panel.csv
02_LowRiskAnomaly
      ↓  (검증 완료 확인 후)
03_VolatilityEDA
      ↓  (예측 가능성 확인 후)
04_VolatilityPrediction
      ↓  vol_predicted.csv
05_BlackLitterman
      ↓  bl_weights.csv
06_Comparison
```

---

## 노트북별 명세

### 01_DataCollection.ipynb
**목적**: 생존편향을 줄인 20년치 S&P500 월별 패널 구성

| 구분 | 내용 |
|------|------|
| 입력 | Wikipedia S&P500 변경 히스토리, yfinance 일별 가격 |
| 처리 | 역방향 멤버십 재구성 → 각 월에 실제 편입된 종목만 포함 |
| 출력 | `data/monthly_panel.csv` |

**출력 컬럼:**

| 변수 | 설명 | 다음 단계 역할 |
|------|------|--------------|
| `ret_1m` | 당월 수익률 | 포트폴리오 성과 계산 |
| `vol_20d` | 20일 실현변동성 (연환산) | **EDA 검정 대상 (≈ 월별 변동성)** |
| `vol_60d` | 60일 실현변동성 | ML 입력 피처 |
| `vol_252d` | 252일 실현변동성 | ML 입력 피처 (장기 레짐) |
| `beta_252d` | 252일 CAPM 베타 | 저위험 분류 참고 |
| `log_mcap` | 로그 시가총액 | P 행렬 가중치 (BL) |
| `gics_sector` | GICS 섹터 | 섹터 분석 |
| `spy_ret` | SPY 월별 수익률 | 벤치마크, π 계산 |
| `rf_1m` | 월별 무위험수익률 | 초과수익 계산 |
| `fwd_ret_1m` | 다음 달 수익률 | (현재 미사용) |

> ⚠ **미완료**: `vol_21d` (21일 = 1개월 거래일) 추가 필요 → 04에서 예측 타겟으로 사용

---

### 02_LowRiskAnomaly.ipynb
**목적**: 저위험 이상현상 검증 (BL 전략의 동기 부여)

| 구분 | 내용 |
|------|------|
| 입력 | `monthly_panel.csv` |
| 분석 | `vol_252d`, `beta_252d` 기준 5분위 포트폴리오 정렬 |
| 기간 | 전체(2006~2025), 최근(2016~2025) 두 구간 |
| 출력 | 누적수익 그래프, Sharpe Ratio 비교 (시각화만, CSV 없음) |

**결론:**
- 변동성 기준: Q1(저변동) Sharpe > Q5(고변동) Sharpe → 이상현상 명확
- 베타 기준: 절대수익은 고베타가 높으나 Sharpe는 저베타가 우위 → 이상현상 약하게 확인

---

### 03_VolatilityEDA.ipynb
**목적**: ML 예측 모델 적용 전, 변동성 예측 가능성 통계 검정

| 구분 | 내용 |
|------|------|
| 입력 | `monthly_panel.csv` |
| 검정 대상 | **`vol_20d`** (예측 타겟 ≈ 월별 변동성) |
| 참고 비교 | `vol_252d` (입력 피처 후보) |
| 출력 | 검정 결과 요약 (시각화만, CSV 없음) |

**검정 항목 및 목적:**

| 검정 | 목적 |
|------|------|
| ADF | 정상성 확인 → ML 회귀 적용 가능 여부 |
| Ljung-Box (lag-1) | 자기상관 → 예측 가능성 존재 여부 |
| ARCH LM | 변동성 군집 → ML이 단순 평균보다 나은 근거 |
| AR(1) R² | 선형 baseline → ML이 이걸 넘어야 의미 있음 |

> ⚠ **핵심 구분**: `vol_20d` (≈ 월별 실현변동성)가 **예측 타겟이자 입력 피처**. 다른 피처(vol_252d, beta 등) 없음 — 논문 그대로

---

### 04_VolatilityPrediction.ipynb
**목적**: 매월 말 기준, 다음 달 개별 종목 변동성 예측

| 구분 | 내용 |
|------|------|
| 입력 | `monthly_panel.csv` |
| 예측 타겟 | `vol_21d` (다음 달 월별 실현변동성) |
| 입력 피처 | **직전 10개월 월별 변동성만** (논문 그대로) |
| 모델 | **ANN** (논문 최종 선택) — GPR/SVR/GARCH는 비교용 |
| 검증 방식 | Walk-forward validation (look-ahead bias 방지) |
| 출력 | `data/vol_predicted.csv` — (date, ticker, vol_pred) |

**학습 설계 (논문 기반):**
- 입력 X: [vol_{t-10}, vol_{t-9}, ..., vol_{t-1}] — 10개 월별 변동성
- 타겟 y: vol_t (다음 달 월별 실현변동성)
- 학습 윈도우: 롤링 60개월
- 리밸런싱: 월별

> ⚠ **논문에 없는 것**: XGBoost, LightGBM, vol_252d/beta/ret_1m 입력 피처 — 추가하지 않음

---

### 05_BlackLitterman.ipynb
**목적**: 예측 변동성 기반 BL 포트폴리오 구성

| 구분 | 내용 |
|------|------|
| 입력 | `monthly_panel.csv`, `data/vol_predicted.csv` |
| 출력 | `data/bl_weights.csv` — (date, ticker, weight) |

**BL 구성 요소:**

| 요소 | 내용 |
|------|------|
| π (사전 균형수익률) | CAPM 역산: λ·Σ·w_mkt |
| P (뷰 포트폴리오) | 예측 변동성 기준 하위 30%(long) vs 상위 30%(short), 시총 가중 |
| Q (뷰 수익률) | Fama-French 3팩터로 추정한 기대수익률 |
| Ω (뷰 불확실성) | 전월 예측 오차의 분산 (동적 설정) |
| τ | 민감도 분석 (0.001 ~ 1.0) |

---

### 06_Comparison.ipynb
**목적**: BL 포트폴리오 vs 벤치마크 성과 비교

| 구분 | 내용 |
|------|------|
| 입력 | `bl_weights.csv`, `monthly_panel.csv` |
| 비교 대상 | BL 포트폴리오 vs CAPM 균형 vs S&P500(SPY) |
| 성과 지표 | 누적수익, Sharpe Ratio, MDD, 연환산 알파 |
| 기간 | 전체(2006~2025), 최근(2016~2025) |

---

## 핵심 변수 역할 요약

| 변수 | 역할 | 생성 위치 | 사용 위치 |
|------|------|----------|----------|
| `vol_20d` | EDA 검정 대상 (≈월별 변동성) | 01 | 03 |
| `vol_21d` | **ML 예측 타겟** | 01 (추가 필요) | 04 |
| `vol_60d` | ML 입력 피처 | 01 | 04 |
| `vol_252d` | ML 입력 피처 (장기 레짐) | 01 | 04 |
| `vol_predicted` | BL 뷰 분류 기준 | 04 | 05 |
| `log_mcap` | P 행렬 가중치 | 01 | 05 |
| `ret_1m` | 포트폴리오 수익 계산 | 01 | 05, 06 |
| `spy_ret` | 벤치마크, π 계산 | 01 | 05, 06 |
| `rf_1m` | 초과수익 계산 | 01 | 02, 05, 06 |

---

## 미완료 / 결정 필요 사항

- [x] `vol_21d` 컬럼 추가 → `01_DataCollection` 수정 완료, **`monthly_panel.csv` 삭제 후 패널 재구성 필요**
- [ ] 04 모델 선택: XGBoost vs LightGBM vs ANN (논문은 ANN)
- [ ] 05 공매도 허용 여부: P 행렬에 음의 가중치 포함할지
- [ ] 06 비교 기간 확정 (전체 / 최근 10년)
