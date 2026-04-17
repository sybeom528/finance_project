# Step2 · Step3 검토 및 HMM 레짐 설계 보완

> 작성일: 2026-04-17  
> 대상 파일: `김윤서/Step2_Preprocessing_yoonbeom.ipynb`, `김윤서/Step3_XGBoost_walkForward_yoonbeom.ipynb`  
> 참고 문서: `김윤서/decision_log_jaecheon.md`

---

## 목차

1. [Step2 전체 파이프라인 흐름](#1-step2-전체-파이프라인-흐름)
2. [타겟 변수 생성 메커니즘 (2-6)](#2-타겟-변수-생성-메커니즘-2-6)
3. [qcut으로 분류 타겟을 만드는 이유](#3-qcut으로-분류-타겟을-만드는-이유)
4. [Step2 오류 및 주의사항](#4-step2-오류-및-주의사항)
5. [EDA 시각화 해석](#5-eda-시각화-해석)
6. [전체 피처 목록](#6-전체-피처-목록)
7. [모멘텀 수식 검토](#7-모멘텀-수식-검토)
8. [Step3 논리 흐름 검토 및 오류](#8-step3-논리-흐름-검토-및-오류)
9. [ValueError 수정](#9-valueerror-수정)
10. [embargo_days 기본값 이슈](#10-embargo_days-기본값-이슈)
11. [HMM 레짐 피처 설계](#11-hmm-레짐-피처-설계)

---

## 1. Step2 전체 파이프라인 흐름

```
2-1 ETF 유니버스 정의 (22종, 개별주 8종 생존편향으로 제외)
  ↓
2-2 수익률 계산
    - ETF 22종: 로그 수익률 ln(P_t / P_{t-1})
    - VIX 계열 (^VIX, ^VIX9D, ^VIX3M, ^VIX6M, ^SKEW): .diff() 차분
    - FRED 지표: 수준 또는 .diff(5)
  ↓
2-3 거시 파생변수 17개 생성
    (VIX 기간구조, SKEW, Cu/Au, HY, yield curve, claims, WEI, sahm, DGS10_chg5)
  ↓
2-4 자산별 특성 피처 11개 생성
    (모멘텀 4개, 변동성 3개, beta 2개, rate_corr, RSI)
  ↓
2-5 Long-Panel 구축 [Date × Ticker] → 57,398행
    (22 ETF × ~2,609일)
  ↓
2-6 타겟 변수 생성
    - fwd_ret_5d / fwd_ret_21d / fwd_ret_63d (연속형)
    - fwd_label_21d (ticker별 quintile 1~5)
  ↓
2-7 워밍업 행 제거 + CSV 저장 → 51,612행 (df_panel.csv)
  ↓
2-8 EDA 시각화 (Beta, Rate Sensitivity, Forward Return 분포, 거시지표 시계열)
```

---

## 2. 타겟 변수 생성 메커니즘 (2-6)

### fwd_ret_21d 계산

```python
g[f'fwd_ret_{h}d'] = np.log(p.shift(-h) / p)
```

`p.shift(-h)`는 시계열을 h행 앞으로 당기므로, t 시점에서 t+h일 가격을 가져옵니다.

```
fwd_ret_21d_t = ln(P_{t+21} / P_t)
```

마지막 21거래일은 NaN → 학습에서 제외 (51,150행 유효).

### fwd_label_21d 계산

```python
df_panel['fwd_label_21d'] = (
    df_panel.groupby('Ticker')['fwd_ret_21d']
    .transform(safe_qcut)
)
```

- **ticker별 개별 qcut** 적용 → TLT와 QQQ는 각자의 수익률 분포를 기준으로 5분위
- Label 1 = bottom 20% (강한 하락), Label 5 = top 20% (강한 상승)

---

## 3. qcut으로 분류 타겟을 만드는 이유

XGBoost를 **5-class 분류기**로 사용하기 때문입니다.

| 이유 | 설명 |
|------|------|
| 낮은 SNR | 금융 수익률의 절대값 예측은 노이즈가 너무 심해 회귀로는 사실상 불가능 |
| 상대 서열 | "어느 자산이 더 오를지" 서열 문제는 노이즈에 덜 민감 |
| B-L 연결 | 분류 확률 P(C1)~P(C5)로 Q와 Ω를 동시에 계산 가능 |

**Black-Litterman 연결 공식:**

```
Q  = Σ r̄_k × P(C_k)   # 기대수익률
Ω  = Σ P(C_k) × (r̄_k - Q)²   # 예측 불확실성
```

회귀 모델은 점추정값 하나만 반환해 Ω를 구성할 수 없습니다.

---

## 4. Step2 오류 및 주의사항

### 오류 1 (Major) — ext_ret, ext_diff 미사용 (dead code)

2-2에서 계산된 `ext_ret`, `ext_diff`가 이후 어디에도 사용되지 않습니다.  
2-3에서는 raw `external_prices`를 직접 참조합니다.  
외부 지표(HG=F, GC=F 등)의 **일별 수익률 자체**를 피처로 사용할지 팀 재검토 필요.

### 오류 2 (Critical) — fwd_label_21d 전체 샘플 look-ahead bias

```python
# 노트북 내 경고 주석
# 주의: fwd_label_21d는 EDA용 full-sample qcut. 실제 학습 시 IS 윈도우 내 재계산 필수
```

저장된 `fwd_label_21d`는 2016~2025년 전체 분포를 알고 나서 분위수를 나눈 것입니다.  
Step3에서는 이미 `discretize_labels()` 함수로 IS 윈도우 내 재계산하므로 올바르게 처리됩니다.  
단, `df_panel.csv`의 `fwd_label_21d` 컬럼을 Step3에서 직접 사용하면 안 됩니다.

### 주의 3 — 2-5 positional join의 위험성

```python
df_t = df_t.join(char_frames[ticker].reset_index(drop=True))
df_t = df_t.join(macro.reset_index(drop=True))
```

날짜 인덱스 기반이 아닌 **정수 인덱스 기반 join**입니다.  
현재는 모두 2,609행으로 일치하므로 무해하지만, 향후 데이터 업데이트 시 길이 불일치 발생 시 날짜 오정렬이 조용히 발생할 수 있습니다.  
안전한 방법: `pd.merge(..., on='Date')` 사용 권장.

### 주의 4 — fwd_ret_5d, fwd_ret_63d에 quintile label 없음

현재 `fwd_label_21d`만 존재합니다.  
5d 또는 63d horizon으로 분류 모델을 돌리려면 해당 label 별도 생성 필요.

---

## 5. EDA 시각화 해석

### Rolling Beta 60d by Ticker

| 자산군 | 특징 |
|--------|------|
| 채권 (파란색) | Beta ≈ 0 → 주식과 독립. TLT는 위기 시 음의 Beta (flight-to-quality) |
| 대안 (주황색) | GLD: Beta ≈ 0. DBC: Beta ≈ 0.3 (약한 양 상관) |
| 주식 (초록색) | QQQ: ~1.25, IWM: ~1.2 (시장보다 크게 움직임) |
| 섹터 (빨간색) | XLE 최고(~1.4), XLP/XLU/XLRE 방어적(~0.5) |

### Rate Sensitivity (corr with DGS10_chg5)

| 자산군 | 특징 |
|--------|------|
| 채권 (파란색) | -0.3 ~ -0.4 → 금리 상승 시 하락 (채권 가격 역관계) |
| GLD (주황색) | -0.1 ~ -0.15 → 금리 상승 시 약세 (실질금리 상승) |
| XLF (빨간색) | +0.1 → 금리 상승 시 수혜 (예대마진 확대) |

> **Beta와 rate_corr_60d 두 피처만으로 채권/주식/대안을 명확히 구분 가능**  
> → ticker 원-핫 인코딩 대신 수치 피처를 쓰는 이유가 여기서 검증됨.

### Forward 21d Log Return 분포 (Violin Plot)

- **채권 ETF**: 바이올린이 극도로 좁음 → ±5% 이내 (quintile 경계가 촘촘해 분류 난이도 높음)
- **XLE (에너지)**: 하락 꼬리 -0.6까지 → 극단적 하방 리스크 (2020 유가 폭락)
- **대부분 자산**: 좌편향 (negative skew) → 평상시 완만 상승, 위기 시 급락
- **QQQ/XLK**: 가장 넓은 분포 → 기술주 고변동성

### 자산별 평균 특성 피처 히트맵

| 피처 | 자산 구분력 | 주요 정보 |
|------|-----------|---------|
| ret_1m ~ ret_12m | 낮음 | TLT ret_12m = -0.02 (2022 금리인상 반영) |
| vol_21d / vol_63d | 높음 | XLE 최고(0.25~0.26), SHY 최저(0.01) |
| vol_ratio | 낮음 (평균) | 시계열 변동에 정보 (위기 시 >1 급등) |
| beta_60d | 매우 높음 | 공격/방어/독립 자산 완벽 분리 |
| rate_corr_60d | 높음 | 채권/주식 분리 |
| rsi_14d | 낮음 (평균) | 시계열 과매수·과매도 신호 |

### 거시지표 시계열 (2017~2025)

| 지표 | 주요 이벤트 |
|------|------------|
| VIX + Contango | 2020 COVID: VIX 80 급등 + Backwardation 전환 (위기 레짐 신호) |
| HY Credit Spread | 2020: 11%까지 급등, 2023~2025: 역대 최저 2~3% |
| Yield Curve (T10Y2Y) | 2022~2023: 역사상 가장 깊고 긴 역전 (빨간 음영). 2024~2025: 해소 |
| 실업수당 Z-Score | 2020 COVID: Z=12+ 극단 스파이크. 2023~2025: -1 (타이트한 노동시장) |

---

## 6. 전체 피처 목록

### 투자 자산 유니버스 (22 ETF)

| 분류 | 티커 |
|------|------|
| 지수 ETF (5) | SPY, QQQ, IWM, EFA, EEM |
| 채권 ETF (4) | TLT, AGG, SHY, TIP |
| 대안 ETF (2) | GLD, DBC |
| 섹터 ETF (11) | XLK, XLF, XLE, XLV, VOX, XLY, XLP, XLI, XLU, XLRE, XLB |

### 자산별 특성 피처 (11개, 티커별 시계열)

| 피처 | 계산식 | 의미 |
|------|--------|------|
| ret_1m | ln(P_t / P_{t-21}) | 1개월 수익률 |
| ret_3m | ln(P_t / P_{t-63}) | 3개월 수익률 |
| ret_6m | ln(P_t / P_{t-126}) | 6개월 수익률 |
| ret_12m | ln(P_t / P_{t-252}) | 12개월 수익률 |
| vol_21d | std(ret, 21) × √252 | 단기 연환산 변동성 |
| vol_63d | std(ret, 63) × √252 | 중기 연환산 변동성 |
| vol_ratio | vol_21d / vol_63d | 변동성 가속도 |
| beta_60d | Cov(r, r_SPY, 60d) / Var(SPY) | 60일 시장 베타 |
| beta_120d | 동일, 120d | 장기 시장 베타 |
| rate_corr_60d | corr(ret, DGS10_chg5, 60d) | 금리 민감도 |
| rsi_14d | RSI(14) | 과매수/과매도 신호 |

### 거시 파생 피처 (17개, 모든 자산에 공통)

| 피처 | 원본 | 의미 |
|------|------|------|
| VIX_level | ^VIX | 변동성 절대 수준 |
| VIX_contango | ^VIX3M / ^VIX - 1 | 양수=정상, 음수=위기 |
| VIX_slope_9d_3m | ^VIX3M - ^VIX9D | 단기 VIX 기간 기울기 |
| VIX_slope_3m_6m | ^VIX6M - ^VIX3M | 중기 VIX 기간 기울기 |
| SKEW_level | ^SKEW | 꼬리 위험 절대값 |
| SKEW_zscore | ^SKEW 63일 z-score | 과거 대비 꼬리 위험 |
| Cu_Au_ratio | HG=F / GC=F | 구리/금 비율 (경기 낙관 지표) |
| Cu_Au_ratio_chg | Cu_Au_ratio.diff(21) | 21일 변화량 |
| HY_spread | FRED BAMLH0A0HYM2 | 하이일드 스프레드 |
| HY_spread_chg | HY_spread.diff(5) | 5일 스프레드 변화 |
| yield_curve | FRED T10Y2Y | 10년-2년 금리 차 |
| yield_curve_inv | T10Y2Y < 0 → 1 | 역전 이진 플래그 |
| claims_4wma | FRED ICSA 28일 이동평균 | 실업수당 청구 추세 |
| claims_zscore | claims_4wma 260일 z-score | 노동시장 긴장도 |
| WEI_level | FRED WEI | 주간 경제활동 지수 |
| sahm_indicator | FRED SAHMREALTIME | 침체 진입 신호 (≥0.5) |
| DGS10_chg5 | FRED DGS10.diff(5) | 10년 금리 5일 변화 |

---

## 7. 모멘텀 수식 검토

### 코드 수식은 수학적으로 정확

```python
ch['ret_1m']  = np.log(p / p.shift(21))   # = ln(P_t / P_{t-21})
ch['ret_12m'] = np.log(p / p.shift(252))  # = ln(P_t / P_{t-252})
```

### 학술 관례와의 차이 (skip-month 부재)

Jegadeesh-Titman (1993) 전통적 모멘텀 팩터는 **최근 1개월을 제외**합니다.

| | 현재 코드 | 학술 관례 |
|---|---|---|
| ret_12m | ln(P_t / P_{t-252}) — 오늘~12개월 전 | ln(P_{t-21} / P_{t-252}) — skip-month |

최근 1개월을 포함하면 **단기 반전(short-term reversal)** 효과가 모멘텀 신호에 혼재합니다.  
단, XGBoost 같은 ML 모델에서는 단기 반전도 하나의 신호로 학습 가능하므로 치명적 오류는 아닙니다.

---

## 8. Step3 논리 흐름 검토 및 오류

### Walk-Forward 구조 (IS=150일 기준)

```
전체 1개 윈도우 = 192 거래일

┌──────────────────────────────┐  ┌──────────┐  ┌──────────┐
│         IS: 150일            │  │Embargo   │  │  OOS     │
│  ┌─────────────┐  ┌────────┐ │  │  21일    │  │  21일    │
│  │ Optuna 학습  │  │ Purge  │ │  │          │  │          │
│  │  82일 train │  │ 21일   │ │  │          │  │          │
│  │  21일 embargo  제거    │ │  │          │  │          │
│  │  26일 val   │  │        │ │  │          │  │          │
│  └─────────────┘  └────────┘ │  │          │  │          │
└──────────────────────────────┘  └──────────┘  └──────────┘
```

| 구간 | 날짜 수 | 행 수 (×22 ETF) |
|------|---------|-----------------|
| Optuna 내부 Train | 82일 | ~1,804행 |
| Optuna 내부 Embargo | 21일 | 제외 |
| Optuna 내부 Val | 26일 | ~572행 |
| 최종 IS 학습 (Purge 후) | 129일 | ~2,838행 |
| Embargo | 21일 | 제외 |
| OOS 예측 | 21일 | ~462행 |

### 오류 1 (Critical) — IS_DAYS 코드·markdown 불일치

```python
IS_DAYS = 252   # 코드 상단 주석
```
```
✅ 설정 완료
  IS=150일 | ...  ← 이전 실행 결과가 캐시됨
```

상단 markdown에 `IS: 252일`로 표기되어 있으나, CLAUDE.md 설계 문서는 **IS=150**을 명시합니다.  
**수정 필요**: 코드를 `IS_DAYS = 150`으로 확정하고 markdown 주석도 동기화.

### 오류 2 (Fixed) — make_objective의 embargo_days 미전달

이전 코드:
```python
study.optimize(make_objective(X_is, y_is, N_CLASSES), ...)
```

수정 후:
```python
study.optimize(make_objective(X_is, y_is, N_CLASSES, embargo_days=EMBARGO_DAYS), ...)
```

단, 기본값이 21로 동일해서 실제 학습 동작은 변화 없었음.  
향후 `EMBARGO_DAYS` 변경 시 자동 반영을 위해 명시적 전달이 올바른 방식.

### 주의 — long_panel.parquet 미존재

```
✅ csv 로드 성공  ← parquet 없어서 fallback
```

Step2에서 `.to_parquet()` 추가를 권장합니다 (CSV 대비 10배 이상 로딩 속도).

### 현재 성과 (GDELT/HMM 피처 없는 베이스라인)

```
OOS Accuracy : 0.2805  (랜덤 기준선 0.20 대비 +40%)
R²_oos 평균  : -0.9588  (절대값 예측 지표라 음수는 정상)
R²>0 비율    : 14.3%
```

GDELT 3개 + HMM 1개 피처 추가 후 개선 여지 있음 (특히 레짐 전환 구간 2020, 2022).

---

## 9. ValueError 수정

### 오류 내용

```
ValueError: y_true and y_prob contain different number of classes: 4 vs 5.
Classes found in y_true: [1 2 3 4]
```

### 원인

Optuna 내부 검증 세트(IS 마지막 20%, 약 26일)가 강세장 구간이면  
22개 ETF 중 하위 quintile(`label=0`)에 해당하는 자산이 한 건도 없을 수 있습니다.  
sklearn `log_loss`는 `y_true`에 없는 클래스를 자동 처리하지 않아 에러 발생.

### 수정 (셀 `4bc79815`)

```python
# 수정 전
return log_loss(y_v, proba_v)

# 수정 후
return log_loss(y_v, proba_v, labels=list(range(n_classes)))
```

`labels` 인자를 명시하면 `y_v`에 없는 클래스도 포함해 올바르게 log_loss를 계산합니다.

---

## 10. embargo_days 기본값 이슈

기존 코드에서 `embargo_days=EMBARGO_DAYS`가 빠져있었지만,  
함수 기본값 `embargo_days=21`이 `EMBARGO_DAYS=21`과 동일하여 **실제 학습 동작은 변화 없었습니다.**

낮은 모델 성과의 실제 원인은 embargo가 아니라 **누락된 피처**입니다:

```python
⚠️ 아직 없는 피처: ['gdelt_avg_tone_1m', 'gdelt_tone_momentum',
                    'gdelt_article_volume', 'hmm_crisis_prob']
```

---

## 11. HMM 레짐 피처 설계

### GDELT만으로 HMM을 돌리면 안 되는 이유

```
2020-02-20: fin_sentiment = -0.15 (뉴스 부정적)
            VIX = 17 (시장은 아직 평온)

→ GDELT만 보면 이미 "위기 레짐"
→ 실제 시장 레짐 전환은 2월 24일
→ VIX 없이는 레짐 경계를 시장 현실에 맞게 정의 불가
```

GDELT는 뉴스 감성을 측정하지, 시장 공포 상태를 직접 측정하지 않습니다.  
**느린 위기(2022 금리인상)**에서 더욱 취약합니다.

### GDELT 피처별 HMM 적합성

| 피처 | 출처 | HMM 적합성 | 이유 |
|------|------|-----------|------|
| fin_sentiment | GKG 쿼리 | ✅ 적합 | 일별 집계, 레짐 지속 상태 반영 |
| article_count | GKG 쿼리 | ✅ 적합 | 뉴스 볼륨 급증이 레짐 전환 신호 |
| min_shock | EVENTS 쿼리 | ❌ 부적합 | 일회성 사건 충격, 레짐 상태 아님 |

**min_shock이 HMM에 부적합한 이유:**

```
안정 레짐 중에도 min_shock=-7 발생 가능 (지진, 테러 등 비금융 사건)
위기 레짐 중에도 min_shock=-1 가능 (외교 회의 등 루틴 이벤트가 최저값)
```

또한 p≤7 제약에서 min_shock을 추가하면 p=8이 되어 위기 레짐 실효 비율이 5.7:1로 떨어집니다.

```
p=7 유지: 위기 레짐 파라미터 35개 → 250/35 = 7.1:1  ✅
p=8 추가: 위기 레짐 파라미터 44개 → 250/44 = 5.7:1  경계선
```

### 최종 권장 피처 배분

#### HMM 입력 (p=7, decision_log 17-3 기준)

| 그룹 | 피처 | 출처 |
|------|------|------|
| A. 주식시장 공포 | VIX_level | yfinance |
| B. 신용·유동성 | HY_spread | FRED |
| C. 금리·경기 | T10Y2Y, MOVE_index | FRED |
| D. 실물 경기 | sahm_indicator | FRED |
| E. GDELT 감성 | fin_sentiment | GKG 쿼리 |
| E. GDELT 볼륨 | article_count | GKG 쿼리 |

#### XGBoost 입력

```python
ASSET_FEATURES  = [...]                      # 11개 (모멘텀, 변동성, beta, RSI 등)
MACRO_FEATURES  = [...]                      # 17개 (거시지표 전체 유지)
GDELT_FEATURES  = ["fin_sentiment",          # GKG → HMM에도 투입
                   "article_count",          # GKG → HMM에도 투입
                   "min_shock"]              # EVENTS → XGBoost에만 (HMM 제외)
HMM_FEATURES    = ["hmm_crisis_prob"]        # HMM 출력 (regime 0/1/2 확률)
```

> **핵심 원칙**: 거시지표는 레짐 "강도"를 담당하고, HMM regime은 지표들의 "동시 폭발 패턴"을 담당합니다.  
> 두 역할은 겹치지 않고 서로 보완적이므로, 레짐으로 거시지표를 대체하는 것은 잘못된 방향입니다.

### 레짐 설계 최종 아키텍처 (decision_log 16장)

```
[GDELT 집계 + VIX + HY_spread + T10Y2Y + MOVE + Sahm]
                     ↓
      HMM (K=3, covariance_type='full', 전체 시계열 1회 학습)
      재보정: 매년 1월 또는 KL-divergence 트리거
                     ↓
      regime ∈ {0: 안정, 1: 중립, 2: 위기}
                     ↓
 ┌─────────────────────────────────────────────────┐
 │  XGBoost Walk-Forward (IS=150일 / OOS=21일)     │
 │  피처: 자산별 특성 + 거시지표 + GDELT + regime  │
 └─────────────────────────────────────────────────┘
                     ↓
               Q, Ω → Black-Litterman
```

---

## 액션 아이템 요약

| 우선순위 | 항목 | 담당 파일 |
|----------|------|----------|
| 긴급 | IS_DAYS = 150으로 확정 + markdown 동기화 | Step3 셀 2, 셀 1 |
| 긴급 | log_loss에 labels 인자 추가 (완료) | Step3 셀 `4bc79815` |
| 권장 | Step2에서 df_panel.to_parquet() 추가 | Step2 2-7 |
| 권장 | GDELT 쿼리 실행 후 fin_sentiment, article_count, min_shock 컬럼 추가 | Step2 또는 별도 수집 셀 |
| 권장 | HMM 모델 구현 후 hmm_crisis_prob 컬럼 Long-Panel에 merge | HMM 구현 노트북 |
| 참고 | IC 결과 확인 (GDELT/HMM 추가 후 재실행) | Step3 셀 `3ae201b7` |
