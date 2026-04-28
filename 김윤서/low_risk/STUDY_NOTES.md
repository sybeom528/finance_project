# Low-Risk Anomaly × Black-Litterman 학습 노트

---

## 전체 파이프라인

```
01_DataCollection      → monthly_panel.csv
02_LowRiskAnomaly      → 저위험 이상현상 검증 (EDA)
03_VolatilityEDA       → ML 적용 타당성 통계 검정
04_VolatilityPrediction → vol_predicted.csv (GARCH 예측)
05_BlackLitterman      → bl_weights.csv
06_Comparison          → 성과 비교
```

---

## 핵심 개념 정리

### 월별 패널 구성 (01_DataCollection)

매월 유니버스가 다르게 구성된다. **생존편향(Survivorship Bias)** 을 줄이기 위해 Wikipedia S&P500 변경 히스토리를 역방향으로 재구성하여, 각 월에 실제로 S&P500에 편입되어 있던 종목만 패널에 포함한다.

```python
# 역방향 재구성: 현재 구성에서 시작해 변경 이벤트를 거꾸로 적용
mask = [(ticker in get_members_at(date))
        for (date, ticker), _ in monthly_df.iterrows()]
monthly_df = monthly_df[mask]
```

- `get_members_at(date)`: 해당 날짜에 S&P500에 편입되어 있던 종목 집합 반환
- `searchsorted` + `side='right' - 1`: date 이전 마지막 이벤트를 찾는 이진 탐색
- S&P500 멤버십 크기: 특정 시점에 S&P500에 편입된 종목 수 (평균 약 508개)

---

### ffill().bfill()

결측값(NaN) 처리 패턴.

```
ffill(): 직전 유효한 값으로 NaN 채우기 (Forward Fill)
bfill(): 직후 유효한 값으로 NaN 채우기 (Backward Fill)
ffill().bfill(): ffill로 먼저 채우고, 맨 앞 남은 NaN은 bfill로 처리
```

금융 데이터에서 주말/공휴일 빠진 날짜를 직전 거래일 값으로 채울 때 주로 사용.
look-ahead bias가 없는 `ffill()`만 쓰는 경우도 많다.

---

### 주요 변수

| 변수 | 의미 | 계산 방식 |
|------|------|----------|
| `ret_1m` | 월별 수익률 | 이번 달 말 종가 vs 지난달 말 종가의 단순 비율 변화 (`pct_change`) |
| `vol_21d` | 21일 실현변동성 | 직전 21일 일별 로그 수익률의 표준편차 × √252 (연환산) |
| `vol_252d` | 252일 실현변동성 | 직전 252일 기준, 장기 레짐 파악용 |
| `beta_252d` | 252일 CAPM 베타 | cov(종목, SPY) / var(SPY), rolling 252일 |
| `log_mcap` | 로그 시가총액 | log(주가 × 발행주식수) |
| `rf_1m` | 월별 무위험수익률 | ^IRX(13주 T-bill 금리) → 월별 변환 |
| `spy_ret` | SPY 월별 수익률 | 시장 벤치마크 |
| `fwd_ret_1m` | 다음 달 수익률 | shift(-1)로 미래 참조 (타겟 변수 전용) |

**vol_21d는 누적 합이 아님.** 수익률이 매일 얼마나 들쑥날쑥했는가(표준편차)를 측정한다. 수익률이 높아도 일정하면 vol은 낮고, 수익률이 낮아도 매일 크게 흔들리면 vol은 높다.

**ret_1m은 복리 누적 계산과 거의 동일.** 주가 자체가 복리 누적을 반영하므로 월말 종가 비교만으로도 그 달의 누적 수익률이 된다.

---

### 베타 (Beta, β)

시장 전체가 움직일 때 개별 종목이 얼마나 민감하게 반응하는지를 나타내는 지표.

```
β = Cov(r_종목, r_시장) / Var(r_시장)

β = 1.0 → 시장과 동일하게 움직임
β = 0.5 → 시장이 1% 오를 때 0.5% 오름 (저위험)
β = 2.0 → 시장이 1% 오를 때 2% 오름 (고위험)
```

- **변동성 vs 베타**: 변동성은 절대적 흔들림, 베타는 시장 대비 상대적 민감도
- **CAPM의 예측**: 고베타 = 고수익 → 실제로는 반대 (저위험 이상현상)

---

### 저위험 이상현상 (Low-Risk Anomaly)

CAPM 이론: 고위험 → 고수익  
실제 데이터: 저위험이 위험 대비 수익(Sharpe)이 더 높음

```
[02_LowRiskAnomaly 결과]
변동성 기준  Q1(저변동) Sharpe: 0.940  vs  Q5(고변동) Sharpe: 0.478  → 스프레드: 0.461
베타 기준    Q1(저베타) Sharpe: 0.775  vs  Q5(고베타) Sharpe: 0.479  → 스프레드: 0.296
```

이 검증이 있어야 BL에서 "저변동 long, 고변동 short" 뷰를 정당화할 수 있다.

**스프레드**: Q1과 Q5의 Sharpe Ratio 차이. 클수록 저위험 이상현상이 강하게 존재.

---

### 포트폴리오 정렬 (Portfolio Sort)

```python
def portfolio_sort(df, sort_col, ret_col='ret_1m', n_quantiles=5):
    for date, group in df.groupby(level='date'):  # 매월
        valid['quintile'] = pd.qcut(valid[sort_col], 5, labels=[1,2,3,4,5])
        port_ret = valid.groupby('quintile')[ret_col].mean()  # 동일가중 수익률
```

매월 종목을 베타/변동성 기준으로 5분위 분류 → 각 분위의 동일가중 수익률 측정 → 전체 기간 시계열로 누적.

---

### Long / Short

| 용어 | 의미 |
|------|------|
| **Long (매수)** | 자산을 사서 보유. 가격 상승 시 수익 |
| **Short (공매도)** | 없는 주식을 빌려서 팔고 나중에 사서 갚음. 가격 하락 시 수익 |

이 프로젝트 BL의 P 행렬:
```
P[저변동 그룹] = +양수  ← Long
P[고변동 그룹] = -음수  ← Short (뷰 인코딩용)
```
최종 포트폴리오는 `bounds=[(0,1)]`로 **Long-only**. 음수 P는 수학적 표현이고 실제 공매도는 없다.

---

### 공분산 행렬 (Σ)과 Ledoit-Wolf

**공분산 행렬**: 두 종목이 얼마나 같이 움직이는지를 모든 종목 쌍에 대해 정리한 표.
- 대각선: 각 종목 자체의 분산 (변동성²)
- 비대각선: 두 종목의 공분산 (함께 움직이는 정도)
- 포트폴리오 분산 = `w^T · Σ · w`

**문제**: 종목 수 n=400이면 파라미터 160,000개, 학습 데이터 60개월로는 추정 오차 폭발.

**Ledoit-Wolf 수축**: 과장된 표본 공분산을 단순 구조로 당겨오는 기법.
```
Σ_LW = (1 - α) · Σ_표본  +  α · Σ_타겟
```
α는 데이터가 적을수록 커짐 → 자동 계산, 튜닝 불필요. 수치 안정성 + 추정 정확도 동시 확보.

---

### RF (무위험수익률)

위험 없는 투자(미국 단기 국채)의 수익률. 이 프로젝트에서는 `^IRX` (13주 T-bill).

```
Sharpe Ratio = (포트폴리오 수익률 - RF) / 변동성
초과수익      = r_i - RF
```

국채에 넣어도 4% 버는데 포트폴리오가 5% 벌었다면 실질 알파는 1%.

---

### FF3 (Fama-French 3 Factor Model)

CAPM의 확장. 3가지 팩터로 주식 수익률을 설명.

```
r_i - RF = α + β1·(Mkt-RF) + β2·SMB + β3·HML + ε
```

| 팩터 | 의미 |
|------|------|
| **Mkt-RF** | 시장 초과수익률 |
| **SMB** (Small Minus Big) | 소형주 수익률 - 대형주 수익률 |
| **HML** (High Minus Low) | 가치주 수익률 - 성장주 수익률 |

---

## Black-Litterman 구조

### 전체 흐름

```
π (CAPM 균형수익률, 사전)
    ↓
P (뷰 포트폴리오 행렬): 저변동 long / 고변동 short, 시총 가중
Q (뷰 수익률, 스칼라): 저위험 그룹 - 고위험 그룹 기대수익 차이
Ω (뷰 불확실성): τ·P·Σ·P^T
    ↓
μ_BL = π + τΣP^T[P(τΣ)P^T + Ω]^{-1}(Q - Pπ)
    ↓
포트폴리오 최적화: min (λ/2)w^T Σ w - w^T μ_BL
                   s.t. Σw=1, w≥0 (Long-only)
```

### P 행렬

`k×n` 행렬 (k=1, n=유니버스 종목 수). 한 행이 하나의 뷰를 나타냄.

```python
def build_P(vol_series, mcap_series, pct=0.30):
    low_risk  = 변동성 하위 30%  →  P[low_risk]  = +시총가중치
    high_risk = 변동성 상위 30%  →  P[high_risk] = -시총가중치
    # 행 합 = 0 (relative view)
```

### Q의 역할

`diff = Q - P·π` 가 BL 업데이트의 핵심.

- Q 크면 → 저위험 종목 기대수익률 크게 상향 → 비중 더 배분
- Q = 0 → CAPM π와 동일
- Q 음수 → 오히려 고위험 쪽으로 쏠림

**Q가 작아도 조정이 일어나는 이유**: CAPM은 고변동 종목에 더 높은 기대수익을 부여하므로 `P·π < 0`. 따라서:
```
diff = Q - P·π = 0.003 - (음수) = 양수 (Q보다 큰 값)
```
Q=0.003이 특별한 게 아니라, "방향(저변동 우대)은 P·π 보정에서 이미 나오고, Q는 강도를 추가로 조절하는 하이퍼파라미터"다.

### Q 추정 방식 비교

| 방식 | 계산 | 특징 |
|------|------|------|
| **Q_FIXED** | 0.003 (0.3%/월) | 문헌 기반 고정값, 단순 |
| **Q_hist** | 학습 기간 P 포트폴리오 평균 실현수익률 | 과거 패턴 반영, 안정적 |
| **Q_ff3** | FF3 회귀 → 종목별 r̂ → P·r̂ | forward-looking, 복잡 |

`05_BL_Q_Comparison.ipynb` 실행 결과 (2009~2025, 204개월):

| 전략 | Sharpe | MDD | 누적수익률 |
|------|--------|-----|-----------|
| Q_FIXED | **1.144** | **-13.70%** | 1012% |
| Q_hist | 0.962 | -21.57% | 1111% |
| Q_ff3 | 0.698 | -51.48% | 1215% |
| CAPM (BL 없음) | 0.985 | -22.17% | **1330%** |
| SPY | 0.965 | -23.93% | — |

**패턴**: Q가 클수록 포트폴리오가 저변동 종목에 과집중 → 변동성 급등 → Sharpe 하락, MDD 폭발.  
Q_FIXED=0.003이 좋은 이유는 값이 정확해서가 아니라 "충분히 작아서" 과집중을 막기 때문.  
→ 세부 내용: `05_BL_Q_Comparison_RESULTS.md`

### Ω (오메가) 계산 방법

Ω = 뷰 Q에 대한 불확실성. Ω가 클수록 Q를 못 믿는다 → BL이 CAPM(π)에서 덜 벗어남.

| 방법 | 공식 | 특징 |
|------|------|------|
| **He-Litterman** (현재) | `Ω = τ·P·Σ·P^T` | 자동 계산, 튜닝 불필요 |
| **Idzorek** | 신뢰도 % → Ω 역산 | 직관적이나 주관적 |
| **역사적 분산** | `Var(P·r)` over train window | Q_hist와 논리적 일관성 |
| **GARCH 기반** | GARCH 조건부 분산 | 레짐 반응, 구현 복잡 |

현재 He-Litterman 방식에서 `Ω = τ·P·Σ·P^T`이므로 분모가 `2·τ·P·Σ·P^T` → 조정량이 Ω=0 대비 절반. Q를 50% 신뢰하는 것과 동일한 효과.

---

## 유니버스 구성 및 필터링

### 월별 유니버스 흐름

```
monthly_panel.csv (S&P500 멤버십 필터 완료, ~508개/월)
    ↓ 필터 1: dropna(vol_21d, log_mcap, ret_1m)
이번 달 피처가 모두 있는 종목 (~480~500개)
    ↓ 필터 2: ret_slice.dropna(axis=1, thresh=42)
직전 60개월 중 42개월 이상 수익률 데이터 있는 종목
최종 valid_tix (~400~480개) → 공분산 추정 + MVO 대상
```

| 필터 | 시점 | 제거 이유 | 대표 케이스 |
|------|------|----------|------------|
| `dropna(vol_21d, log_mcap, ret_1m)` | 이번 달 | 피처/수익률 없음 | 거래정지, 상장폐지 직전 |
| `dropna(thresh=42)` | 과거 60개월 | 학습 데이터 부족 → 공분산 불안정 | 신규 편입 종목 |

### PCT_GROUP vs thresh 구분

- **thresh**: 공분산 추정 유니버스 크기 조절. 올리면 생존편향 위험(오래된 대형주 편향).
- **PCT_GROUP**: BL 뷰에 들어가는 저위험/고위험 버킷 종목 수 조절.

```
PCT_GROUP=0.30 → 하위/상위 30% (~120~140개) 버킷
PCT_GROUP=0.20 → 하위/상위 20% (~80~90개) 버킷
→ 공분산 유니버스(~400개)는 그대로, 뷰 강도만 조절
```

### 포트폴리오 내 종목 구성

최종 포트폴리오는 상위/하위 30%만이 아니라 **valid_tix 전체 (~400개)** 에 MVO가 weight를 배분한 결과다.

```
하위 30% (저변동) → P = +시총가중  → μ_BL 상향 → MVO가 더 많이 배분
중간 40%          → P = 0          → μ_BL ≈ π  → CAPM 기대수익 그대로
상위 30% (고변동) → P = -시총가중  → μ_BL 하향 → MVO가 덜 배분 (0에 수렴 가능)
```

MVO 제약: `Σw=1` (비중 합 = 100% 항상 만족), `w≥0` (Long-only).

---

## CAPM 균형 포트폴리오 vs SPY

- **SPY**: 시가총액 가중 S&P500 지수 그대로 보유
- **CAPM 균형 (코드 내)**: `π = λ·Σ·w_mkt`를 기대수익으로 넣고 MVO 최적화한 결과. 시가총액 가중과 다름 — 고베타 종목에 π가 높게 책정되어 MVO가 집중 배분 → SPY보다 수익 높지만 변동성도 높음.

```
[2009~2025 결과]
SPY:  연수익 15.46%, 변동성 14.72%, Sharpe 0.965
CAPM: 연수익 17.02%, 변동성 15.96%, Sharpe 0.985  ← MVO 고베타 집중
BL:   연수익 14.96%, 변동성 11.90%, Sharpe 1.144  ← 저변동 틸팅
```

BL은 수익이 낮지만 변동성이 훨씬 낮아 Sharpe 우위. 저변동 이상현상의 실증 결과.

---

## GARCH 변동성 예측 (04_VolatilityPrediction)

### 역할

GARCH는 **P 행렬의 품질을 높이는 모델**이다.

```
Baseline: vol_21d (과거 실현변동성) → P 구성 → "지난달에 낮았던 종목을 저위험으로"
GARCH:    vol_pred (예측 변동성)    → P 구성 → "다음달에 낮을 종목을 저위험으로"
```

GARCH를 LSTM 등 다른 모델로 교체해도 동일한 구조. 핵심은 **다음달 변동성 순위**를 얼마나 정확히 예측하는가 → Rank IC로 측정.

### 실행 결과 (04_VolatilityPrediction.ipynb)

```
GARCH(1,1) Rank IC (예측 vol vs 실제 vol_21d)
  평균 IC:    0.5874   ← 횡단면 순위 상관계수
  중앙값 IC:  0.5855
  양수 비율:  100.0%   ← 180개월 전부 양수
  ICIR:       7.2854

  예측 기간: 2011-01 ~ 2025-12 (180개월)
  예측 건수: 79,135개 (fallback 0개)
```

**Rank IC**: 매월 ~500개 종목의 GARCH 예측 vol 순위 vs 실제 vol_21d 순위의 Spearman 상관계수. 절대값이 아닌 순위를 맞히는 것 → P 분류에 직접 필요한 정보.

### GARCH가 Baseline보다 Sharpe가 낮은 이유 (06_BL_VolQ_Grid 결과)

```
Baseline+Q_FIXED: Sharpe 1.074, MDD -13.70%
GARCH+Q_FIXED:    Sharpe 1.019, MDD -16.73%  ← 오히려 낮음
```

변동성은 강한 자기상관을 가지므로 현재 vol_21d 자체가 이미 다음 달 vol의 좋은 예측치다. GARCH와 Baseline이 비슷한 종목을 저위험으로 분류하는 경우가 많아 P 행렬에서 실질적 차이가 작다. 나머지 IC 오차(1-0.587)에서 오는 간헐적 오분류가 오히려 포트폴리오를 살짝 불안정하게 만든다.  
→ **단순하고 안정적인 신호(baseline)가 복잡한 예측(GARCH)을 이기는 사례.**

### GARCH vs LSTM

| | GARCH(1,1) | LSTM |
|--|-----------|------|
| 학습 데이터 | 종목당 60개월 | 종목당 60개월 + 추가 피처 가능 |
| IC 잠재력 | 0.587 (확인) | 더 높을 수도, 비슷할 수도 |
| 과적합 위험 | 낮음 | 60개 샘플로 위험, 패널 학습 필요 |

GARCH도 Baseline을 이기지 못했으므로 LSTM이 의미 있는 개선을 만들려면 **기존 baseline 대비 상당히 다른 종목을 분류해야** 한다. 단순히 IC가 높은 것만으로는 부족.

---

## 리밸런싱

**매달** 이루어짐. 매월 말마다:

1. 유니버스 재구성 (그달 S&P500 편입 종목)
2. Σ 재추정 (Ledoit-Wolf, 직전 60개월)
3. 저위험/고위험 재분류 (vol_21d 또는 vol_pred 기준 상위·하위 30%)
4. 가중치 재최적화

종목이 저위험 ↔ 중간 ↔ 고위험 그룹 간 이동해도 매월 P를 처음부터 재구성하므로 자동 반영.  
거래비용은 이 코드에서 반영하지 않음.

---

## Baseline의 의미

변동성 예측 ML 없이 **현재 vol_21d를 그대로** 저위험 분류에 사용한 버전.

```
Baseline  → 현재 vol_21d로 분류 → BL
ML 버전   → 예측 vol로 분류    → BL
```

Baseline보다 ML 버전 Sharpe가 높아야 "ML 예측이 의미 있다"고 말할 수 있음.

| | 99_baseline | 98_2006_baseline |
|--|-------------|-----------------|
| TRAIN_WINDOW | 60개월 | 24개월 |
| 예측 시작 | 2009-01 | 2006-01 |
| BL Sharpe | 1.145 | 0.612 |
| BL MDD | -13.70% | -55.97% |

99_baseline Sharpe가 높은 이유: 2008 금융위기 직후부터 시작 → 출발점이 유리.

---

## ML 적용 전 통계 검정 (03_VolatilityEDA)

ML 모델을 돌리기 전에 **타겟 변수 자체의 예측 가능성**을 먼저 확인해야 한다.

| 검정 | 귀무가설 | 의미 |
|------|---------|------|
| **ADF** | 단위근 존재 (비정상) | 정상성 확인 → ML 회귀 적용 가능 여부 |
| **Ljung-Box** | 자기상관 없음 | 자기상관 없으면 과거로 미래 예측 자체 불가 |
| **ARCH LM** | 변동성 군집 없음 | 군집 있으면 ML이 단순 평균보다 나은 근거 |
| **AR(1) R²** | - | 선형 baseline → ML이 이걸 넘어야 의미 있음 |

**수익률 예측 실패(R² ≈ -0.95)의 원인**: Ljung-Box 검정을 사전에 했다면 수익률 자체의 예측 가능성이 없음을 미리 알 수 있었음. 수익률보다 **변동성**이 자기상관이 있어 예측하기 더 용이하다 (ARCH LM으로 확인 가능).

---

## 파일 구조

```
김윤서/low_risk/
├── 01_DataCollection.ipynb       → monthly_panel.csv 생성
├── 02_LowRiskAnomaly.ipynb       → 저위험 이상현상 검증
├── 04_VolatilityPrediction.ipynb → GARCH(1,1) vol 예측 → vol_predicted.csv
├── 99_baseline.ipynb             → BL Baseline (Q_FIXED, 2009~)
├── 98_2006_baseline.ipynb        → BL Baseline (Q_FIXED, 2006~, 금융위기 포함)
├── 05_BL_Q_Comparison.ipynb      → Q_FIXED / Q_hist / Q_ff3 동시 비교 (baseline vol)
├── 06_BL_VolQ_Grid.ipynb         → 2 vol × 3 Q = 6조합 전체 비교 (2011~)
├── 07_Q_Sensitivity.ipynb        → GARCH + Q_FIXED 최적 Q 민감도 분석
├── 05_BL_Q_Comparison_RESULTS.md → Q 비교 실행 결과 해석 + 개선 방향
├── 06_BL_VolQ_Grid_RESULTS.md    → 6조합 비교 결과 해석 (GARCH가 Baseline보다 낮은 이유 포함)
├── data/
│   ├── monthly_panel.csv         → 월별 패널 (date × ticker)
│   ├── sp500_membership.pkl      → S&P500 멤버십 히스토리
│   ├── universe.csv              → 역사적 유니버스 전체
│   ├── ff3_monthly.csv           → Fama-French 3팩터
│   ├── bl_baseline_returns.csv   → BL 백테스트 결과
│   ├── vol_predicted.csv         → GARCH 예측 vol (date, ticker, vol_pred)
│   └── q_comparison_returns.csv  → Q 비교 수익률 시계열
└── STUDY_NOTES.md                → 이 파일
```
