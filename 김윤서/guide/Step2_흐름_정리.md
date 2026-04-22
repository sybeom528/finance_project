# Step 2: 전처리 + 피처 엔지니어링 + EDA 흐름 정리

> 파일: `Step2_Preprocessing_EDA.ipynb`  
> 작성자: 김윤서  
> 작성일: 2026-04-19  
> 목적: Step2 전체 흐름 + 설계 결정 근거 문서화

---

## 목차

1. [전체 흐름 개요](#1-전체-흐름-개요)
2. [입력 데이터 및 분석 기간 설정](#2-입력-데이터-및-분석-기간-설정)
3. [2-1. 수익률 계산](#3-2-1-수익률-계산)
4. [2-2. 13개 파생 변수 생성](#4-2-2-13개-파생-변수-생성)
5. [2-3. df_reg_v2 구축 (확장 회귀 데이터셋)](#5-2-3-df_reg_v2-구축-확장-회귀-데이터셋)
6. [2-4. EDA 시각화 (6개 차트)](#6-2-4-eda-시각화-6개-차트)
7. [2-5. Granger 인과 검정](#7-2-5-granger-인과-검정)
8. [WEI·SAHMREALTIME 제거 — PIT 소급창조 문제](#8-weisahmrealtime-제거--pit-소급창조-문제)
9. [주요 설계 결정 요약](#9-주요-설계-결정-요약)

---

## 1. 전체 흐름 개요

```
Step2_Preprocessing_EDA.ipynb
│
├── [설정] 분석 기간 상수
│       ANALYSIS_START = 2016-01-01  ← 슬라이싱 기준 (워밍업 2014~2015 제외)
│       입력 데이터는 WARMUP_START(2014-01-01)부터 로드
│
├── [Cell 1] 데이터 로드
│       portfolio_prices.csv  (3,017행 × 30열)
│       external_prices.csv   (3,017행 × 11열)
│       fred_data.csv         (3,017행 × 8열)
│
├── [2-1] 수익률 계산 (Cell 3)
│       포트폴리오 30종: 로그 수익률 ln(P_t / P_{t-1})
│       외부 지표: 로그 수익률 (VIX 계열은 .diff() 수준 차분)
│
├── [2-2] 13개 파생 변수 생성 (Cell 5)
│       변동성 기간구조 3개 (VIX_contango, VIX_slope_9d_3m, VIX_slope_3m_6m)
│       테일 리스크 2개 (SKEW_level, SKEW_zscore)
│       경기 심리 2개 (Cu_Au_ratio, Cu_Au_ratio_chg)
│       신용 스프레드 2개 (HY_spread, HY_spread_chg) ← BAA10Y 소스
│       수익률 곡선 2개 (yield_curve, yield_curve_inv)
│       고용 지표 2개 (claims_4wma, claims_zscore)
│       → features.csv 저장 (전체 기간 포함, 슬라이싱 전)
│
├── [2-3] df_reg_v2 구축 (Cell 7)
│       rv_neutral (종속변수) + 외부 수익률 11 + 외부 롤링 변동성 11
│       + VIX 수준 1 + FRED 매크로 4 + 파생변수 13 = 41컬럼
│       ANALYSIS_START(2016-01-01) 슬라이싱 후 저장
│       → df_reg_v2.csv (2,491행 × 41열)
│
├── [2-4] EDA 시각화 (Cell 9~14)
│       step2_01_dashboard.png     — 4-panel 대안데이터 대시보드
│       step2_02_correlation.png   — 확장 상관관계 히트맵
│       step2_03_vix_contango.png  — VIX Contango vs rv_neutral scatter/boxplot
│       step2_04_yield_curve.png   — 수익률 곡선 역전 타임라인
│       step2_05_macro_nowcast.png — 주간 매크로 Nowcasting 3-panel
│       step2_06_granger.png       — Granger Top 15 막대 차트
│
└── [2-5] Granger 인과 검정 (Cell 16)
        40개 변수 → rv_neutral, maxlag=10
        유의 변수 33개 / 40개 (p < 0.05)
        → granger_results.csv 저장
```

---

## 2. 입력 데이터 및 분석 기간 설정

### 2-1. 입력 파일 3종

| 파일 | 크기 | 기간 | 내용 |
|------|------|------|------|
| `portfolio_prices.csv` | 3,017행 × 30열 | 2014-01 ~ 2025-12 | 투자 자산 30종 종가 |
| `external_prices.csv` | 3,017행 × 11열 | 2014-01 ~ 2025-12 | 외부 지표 11종 종가 |
| `fred_data.csv` | 3,017행 × 8열 | 2014-01 ~ 2025-12 | FRED 매크로 8종 (PIT 적용) |

### 2-2. 자산 그룹 정의

```python
INDEX_ETF  = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']           # 5개
BOND_ETF   = ['TLT', 'AGG', 'SHY', 'TIP']                   # 4개
ALT_ETF    = ['GLD', 'DBC']                                  # 2개
SECTOR_ETF = ['XLK','XLF','XLE','XLV','VOX','XLY',
              'XLP','XLI','XLU','XLRE','XLB']                # 11개
STOCKS     = ['AAPL','MSFT','AMZN','GOOGL','JPM','JNJ','PG','XOM']  # 8개

EQUITY_TICKERS = INDEX_ETF + SECTOR_ETF + STOCKS  # 24개
ALL_TICKERS    = INDEX_ETF + BOND_ETF + ALT_ETF + SECTOR_ETF + STOCKS  # 30개
```

### 2-3. 슬라이싱 전략 — 워밍업 기간 처리

```
파생변수 계산: 전체 기간(2014~2025) 기준으로 수행
  → 260일 롤링(claims_zscore) 등이 2016-01 기준에서 이미 수렴된 상태

최종 슬라이싱: ANALYSIS_START = '2016-01-01'
  → df_reg_v2 = df_reg_v2.loc[ANALYSIS_START:]
  → 2,491 영업일 (2016-01-04 ~ 2025-12-30)
```

> Step1 설계와 연동: 워밍업 기간(2014~2015)은 Step1에서 수집되어 CSV에 포함되어 있고,  
> Step2에서 파생변수 계산 후 ANALYSIS_START로 슬라이싱해 제거한다.

---

## 3. 2-1. 수익률 계산

### 포트폴리오 30종 — 로그 수익률

```python
port_ret = np.log(portfolio_prices / portfolio_prices.shift(1)).dropna()
# 결과: (2,571행 × 30열)
```

**로그 수익률을 사용하는 이유:**
- 시계열 합산 가능 (기간 수익률 = 일별 로그 수익률의 합)
- 정규성 근사로 통계 모형(XGBoost, HMM) 입력에 적합
- 분할·배당 조정 가격(`auto_adjust=True`)과 함께 시계열 연속성 보장

### 외부 지표 11종 — VIX 계열 분리 처리

```python
VIX_COLS = [c for c in external_prices.columns if 'VIX' in c.upper()]
# → ['^VIX', '^VIX9D', '^VIX3M', '^VIX6M']

NON_VIX  = [c for c in external_prices.columns if c not in VIX_COLS]
# → ['CL=F', 'GC=F', 'SI=F', 'BTC-USD', 'DX-Y.NYB', '^SKEW', 'HG=F']

ext_ret_logpart = np.log(external_prices[NON_VIX] / external_prices[NON_VIX].shift(1))
ext_ret_vixpart = external_prices[VIX_COLS].diff()  # 수준 차분
```

| 처리 방법 | 대상 | 이유 |
|---------|------|------|
| 로그 수익률 `ln(P_t / P_{t-1})` | 원유·금·은·BTC·DXY·SKEW·구리 | 가격 변수 — 수익률로 정상화 |
| 수준 차분 `.diff()` | ^VIX, ^VIX9D, ^VIX3M, ^VIX6M | 이미 % 단위 — 로그 수익률 적용 시 왜곡 |

---

## 4. 2-2. 13개 파생 변수 생성

### 파생 변수 전체 목록

| # | 변수명 | 수식 / 설명 | 카테고리 | 롤링 윈도우 |
|---|--------|------------|----------|------------|
| 1 | `VIX_contango` | `^VIX3M / ^VIX - 1` | 변동성 기간구조 | 없음 |
| 2 | `VIX_slope_9d_3m` | `^VIX3M - ^VIX9D` | VIX 단기 기울기 | 없음 |
| 3 | `VIX_slope_3m_6m` | `^VIX6M - ^VIX3M` | VIX 장기 기울기 | 없음 |
| 4 | `SKEW_level` | `^SKEW` 원시값 | 테일 리스크 | 없음 |
| 5 | `SKEW_zscore` | `(^SKEW - 63일 MA) / 63일 σ` | SKEW 표준화 | **63일** |
| 6 | `Cu_Au_ratio` | `HG=F / GC=F` | 경기 심리 지표 | 없음 |
| 7 | `Cu_Au_ratio_chg` | `Cu_Au_ratio.pct_change(21)` | 구리/금 모멘텀 | **21일** |
| 8 | `HY_spread` | `BAA10Y` | 신용 스프레드 | 없음 |
| 9 | `HY_spread_chg` | `BAA10Y.diff(5)` | 스프레드 변화 | 5일 차분 |
| 10 | `yield_curve` | `T10Y2Y` | 수익률 곡선 | 없음 |
| 11 | `yield_curve_inv` | `(T10Y2Y < 0).astype(int)` | 역전 더미 | 없음 |
| 12 | `claims_4wma` | `ICSA.rolling(20).mean()` | 실업 4주 이동평균 | **20일** |
| 13 | `claims_zscore` | `(ICSA - 260일 MA) / 260일 σ` | 실업 표준화 | **260일** |

### 핵심 파생변수 설계 근거

#### VIX 기간구조 (VIX_contango)

```
정상 시장: ^VIX3M > ^VIX  → VIX_contango > 0 (Contango)
           장기 불확실성이 단기보다 크게 가격 반영
위기 국면: ^VIX3M < ^VIX  → VIX_contango < 0 (Backwardation)
           단기 공포가 극대화되어 기간구조 역전
```

Granger 검정 결과 `VIX_slope_9d_3m`이 p=1.56e-62 (lag=1)로 2위, `VIX_contango`가 p=4.86e-26 (lag=1)으로 6위 → rv_neutral의 가장 강력한 단기 예측 변수 그룹.

#### 구리/금 비율 (Cu_Au_ratio)

```
Cu/Au ↑ → 구리(산업수요) > 금(안전자산) → 경기 확장 기대
Cu/Au ↓ → 금 강세, 구리 약세 → 위험 회피, 경기 수축 신호
Cu_Au_ratio_chg = 21일 변화율 → 방향 전환 모멘텀 포착
```

#### HY_spread (BAA10Y 소스)

```
원래 목표: BAMLH0A0HYM2 (ICE BofA HY 스프레드)
문제: ICE 라이선스 → 최근 3년만 FRED 공개 (2016~2022 구간 불가)
대안: BAA10Y (Moody's Baa - 10Y Treasury)
  - Δ 기준 상관 0.70+ (HY와 유사한 움직임)
  - 전체 기간 2014~2025 수집 가능
  - 변수명 'HY_spread' 유지 → Step6 이후 코드 참조명 일관성 보장
  - 단, 반응 강도 HY 대비 약 1/2 → Step6 Config C 임계값 비례 조정
```

#### claims_zscore (가장 긴 롤링 윈도우)

```python
icsa_roll = fred_data['ICSA'].rolling(260)  # ≈ 1 거래연도
claims_zscore = (fred_data['ICSA'] - icsa_roll.mean()) / icsa_roll.std()
```

- **260일 롤링 = Step2 최장 윈도우** → Step1 워밍업 기간(2년) 설계의 핵심 근거
- 단순 ICSA 수준 대신 z-score를 쓰는 이유: 절대값은 경기 사이클에 따라 수준이 달라지므로, 1년 평균 대비 편차로 이상 신호 포착

### features.csv 저장

```python
feat.to_csv(DATA / 'features.csv')
# (3,017행 × 13열) — 전체 기간 포함, 슬라이싱 전 원본
```

> **슬라이싱 전에 저장하는 이유**: 워밍업 구간(2014~2015) 포함 원본을 보존해  
> 향후 롤링 윈도우 검증·재계산 시 참고 가능.

---

## 5. 2-3. df_reg_v2 구축 (확장 회귀 데이터셋)

### 종속변수: rv_neutral

```python
ew_ret    = port_ret.mean(axis=1)           # 30종 동일가중 (1/30)
rv_neutral = ew_ret.rolling(21).std() * np.sqrt(252)  # 연환산 실현변동성
```

- **동일가중(1/30) 포트폴리오**: 자산 선택 편향 없이 전체 포트폴리오의 변동성 대표
- **21일 롤링 × √252**: 월간 단위 변동성을 연환산으로 환산

> rv_neutral은 **가격이 얼마나 하락했는가**가 아니라 **가격이 얼마나 불규칙하게 흔들렸는가**를 측정.  
> 이 값이 높을수록 리스크 관리 신호로 활용된다.

### 독립변수 구성 (40개)

| 블록 | 컬럼 수 | 내용 |
|------|---------|------|
| 외부 수익률 | 11 | `ext_ret` (로그 or 차분) |
| 외부 롤링 변동성 | 11 | `ext_vol21 = ext_ret.rolling(21).std() × √252` |
| VIX 수준 | 1 | `^VIX` (수준값 직접 사용) |
| FRED 매크로 | 4 | `DGS10`, `DGS10_chg`, `CPI_MoM`, `UNRATE` |
| 파생 변수 | 13 | features.csv와 동일 |
| **합계** | **40** | + rv_neutral 포함 시 **41열** |

### FRED 매크로 4종 선택 상세

```python
fred_macro['DGS10']     = fred_data['DGS10']           # 10Y 금리 수준
fred_macro['DGS10_chg'] = fred_data['DGS10'].diff(5)   # 5일 차분 (금리 서프라이즈)
fred_macro['CPI_MoM']   = fred_data['CPIAUCSL'].pct_change() * 100  # 월 변화율
fred_macro['UNRATE']    = fred_data['UNRATE']           # 실업률 수준
```

| 변수 | 변환 방식 | 이유 |
|------|---------|------|
| `DGS10` | 수준 유지 | 절대 금리 수준 자체가 할인율·밸류에이션 영향 |
| `DGS10_chg` | 5일 차분 | DGS10은 비정상 시계열 → 차분으로 금리 서프라이즈 포착 |
| `CPI_MoM` | 월 변화율 | 절대 수준보다 변화율이 인플레이션 충격 반영 |
| `UNRATE` | 수준 유지 | 이미 % 단위, 저변동 — 수준이 경기 상태 직접 표현 |

> WEI·SAHMREALTIME은 df_reg_v2에서 **제외**. 제거 이유는 [섹션 8](#8-weisahmrealtime-제거--pit-소급창조-문제) 참조.

### 최종 산출물

```
df_reg_v2.csv
  기간  : 2016-01-04 ~ 2025-12-30 (ANALYSIS_START 슬라이싱 후)
  크기  : 2,491행 × 41열
  구성  : rv_neutral(1) + 외부수익률(11) + 외부변동성(11)
          + VIX수준(1) + FRED매크로(4) + 파생변수(13)
```

---

## 6. 2-4. EDA 시각화 (6개 차트)

### 차트 목록

| 파일명 | 내용 | 주요 인사이트 |
|--------|------|------------|
| `step2_01_dashboard.png` | 4-panel 대안데이터 대시보드 | VIX 백워데이션·신용 스프레드·YC 역전·Cu/Au 동시 확인 |
| `step2_02_correlation.png` | 확장 상관관계 히트맵 | HY_spread(+0.74), VIX_slope(-0.69) rv와 강상관 |
| `step2_03_vix_contango.png` | VIX Contango vs rv_neutral scatter/boxplot | Backwardation 국면에서 rv 중앙값 약 2배 높음 |
| `step2_04_yield_curve.png` | 수익률 곡선 역전 + rv_neutral 이중 축 | 역전 구간(2019~2023)과 변동성 관계 시각화 |
| `step2_05_macro_nowcast.png` | 실업수당 4wma + WEI + Sahm 3-panel | WEI·Sahm은 fred_data.csv에서 직접 로드 (df_reg_v2 미포함) |
| `step2_06_granger.png` | Granger Top 15 수평 막대 | HY_spread_chg·VIX_slope가 rv 예측력 최상위 |

### 주요 상관 관계 해석

#### rv_neutral과 양의 상관 (경제 스트레스 지표)

| 변수 | 상관계수 | 메커니즘 |
|------|---------|---------|
| `HY_spread` | **+0.74** | 신용 스트레스 → 기업 이익 불확실성 → 주식 변동성 직접 전파 |
| `claims_4wma` | +0.41 | 실업↑ → 소비·이익 불확실성↑ → 시장 흔들림 |
| `claims_zscore` | +0.37 | claims_4wma의 정규화 버전 |

> 이 포트폴리오는 주식 관련 자산이 24/30(80%)이므로, 주식 변동성과 경제 스트레스 지표의 양의 상관이 증폭되어 나타남.

#### rv_neutral과 음의 상관 (VIX 기간구조)

| 변수 | 상관계수 | 메커니즘 |
|------|---------|---------|
| `VIX_slope_3m_6m` | **-0.69** | 장기 VIX가 가파를수록 현재 급박한 위기 아님 → rv 낮음 |
| `VIX_contango` | -0.54 | Contango(정상) → 실현 변동성 안정 |
| `VIX_slope_9d_3m` | -0.51 | 단기 기울기 동일 방향 |

> Backwardation(단기 VIX > 장기 VIX)이면 즉각적 공포 상태 → rv 폭발.  
> 이것이 `VIX_contango`를 Step6 Config B 보정 트리거로 사용하는 근거.

#### SKEW_level 음의 상관 (-0.38) — 직관과 반대인 이유

```
SKEW ↑ = 투자자가 꼬리 리스크 보험을 구매하는 "사전 헤징" 상태
       → 아직 실제 위기 미발생 → rv_neutral 낮음

실제 위기 발생 시 → VIX 폭발, 모두 패닉 → SKEW 구조 붕괴·하락
```

#### 다중공선성 주의 쌍

| 변수 쌍 | 상관 | 처리 |
|---------|------|------|
| `yield_curve` ↔ `yield_curve_inv` | -0.79 | 파생 관계 — 모형에 동시 투입 금지 |
| `VIX_slope_9d_3m` ↔ `VIX_slope_3m_6m` | +0.81 | 같은 기간구조 계열 — 선택적 사용 |
| `sahm` ↔ `claims_4wma` | +0.79 | Sahm이 실업률 기반 — 중복 신호 |

---

## 7. 2-5. Granger 인과 검정

### 검정 설계

```python
from statsmodels.tsa.stattools import grangercausalitytests

target     = 'rv_neutral'
candidates = [c for c in df_reg_v2.select_dtypes(include=[np.number]).columns if c != target]
# 40개 변수

gc = grangercausalitytests(sub, maxlag=10, verbose=False)
# 각 변수별 lag 1~10 F-test p-value 중 최소 p-value와 해당 lag 추출
```

**Granger 인과 검정의 의미**: "X의 과거값이 Y 예측에 통계적으로 유의미한 정보를 추가하는가?"  
H₀(귀무): X는 Y를 Granger-cause하지 않는다 / p < 0.05이면 기각 → X가 Y에 선행 정보 제공

### 검정 결과 요약

```
총 40개 변수 검정
유의미 (p < 0.05): 33개
유의하지 않음    :  7개 (SKEW_level, HG=F_vol21, ^SKEW_vol21, DX-Y.NYB, CPI_MoM, yield_curve, yield_curve_inv)
```

### Top 10 결과

| 순위 | 변수 | best_lag | p-value |
|------|------|---------|---------|
| 1 | `HY_spread_chg` | 1 | 2.51e-63 |
| 2 | `VIX_slope_9d_3m` | 1 | 1.56e-62 |
| 3 | `VIX_level` | 1 | 7.53e-51 |
| 4 | `VIX_slope_3m_6m` | 1 | 9.91e-48 |
| 5 | `claims_4wma` | 2 | 2.26e-47 |
| 6 | `VIX_contango` | 1 | 4.86e-26 |
| 7 | `HY_spread` | 5 | 3.14e-23 |
| 8 | `^VIX9D_vol21` | 4 | 2.37e-20 |
| 9 | `^VIX_vol21` | 4 | 6.96e-19 |
| 10 | `CL=F_vol21` | 7 | 5.17e-18 |

### 유의하지 않은 변수 (p ≥ 0.05) 및 처리 방향

| 변수 | p-value | 해석 |
|------|---------|------|
| `yield_curve` | 0.500 | 수준 자체보다 역전 여부(더미)가 중요 — `yield_curve_inv`로 대체 |
| `yield_curve_inv` | 0.708 | 더미 변수 자체도 유의하지 않음 — Granger 검정 한계 (비선형 관계) |
| `CPI_MoM` | 0.408 | 월별 발표, 주간 변동성 예측에 즉각 반응 미약 |
| `DX-Y.NYB` | 0.248 | 달러 수준 자체보다 급격한 달러 강세 이벤트가 변동성에 영향 |

> Granger 검정 결과는 **선형 단변량** 관계만 포착한다.  
> yield_curve 등은 XGBoost(비선형, 상호작용 포착)에서 여전히 유용할 수 있어 df_reg_v2에 보존.

### granger_results.csv 저장

```python
granger_df.to_csv(DATA / 'granger_results.csv', index=False)
# 컬럼: variable, best_lag, p_value, significant
```

---

## 8. WEI·SAHMREALTIME 제거 — PIT 소급창조 문제

### 문제 발견 경위

Step1에서 ALFRED vintage PIT 적용 후 두 지표의 실제 데이터 시작 시점을 확인:

```
WEI          : 2020-04-03 최초 발표 (그 이전 값은 소급 추정치)
SAHMREALTIME : 2019-09 최초 발표 (그 이전 값은 소급 추정치)
```

### 영향

```
WEI를 df_reg_v2에 포함할 경우:
  → 2020-04 이전 행에서 NaN 발생
  → dropna() 적용 시 2016~2020-03 구간 (약 4년) 제거
  → 분석 기간 2016~2025 → 2020~2025 (6년)으로 축소

SAHMREALTIME도 동일 문제 (2019-09 이전 NaN)
```

### 처리 방법

| 위치 | 처리 | 이유 |
|------|------|------|
| `df_reg_v2.csv` | **제거** | 분석 기간 4년 축소 방지 |
| `features.csv` | **보존** | 원본 시계열 참고용 |
| `fred_data.csv` | **보존** | Step6 Config C 독립 트리거에서 별도 로드 |

```python
# Step6 Config C에서 sahm을 별도 조건으로 활용할 때:
sahm = pd.read_csv(DATA / 'fred_data.csv', index_col=0, parse_dates=True)['SAHMREALTIME']
# 2019-09 이후 구간에서만 트리거로 사용
```

> **핵심 원칙**: PIT 적용 결과 데이터가 없는 시점의 값은 "그 시점에는 존재하지 않았던 값"이다.  
> 이를 회귀 모형에 소급 적용하면 **look-ahead bias**와 동일한 효과.

---

## 9. 주요 설계 결정 요약

| 결정 항목 | 이전 방식 | 최종 방식 | 근거 |
|---------|---------|---------|------|
| 파생변수 계산 기간 | 분석 기간(2016~)만 | **전체 기간(2014~)으로 계산 후 슬라이싱** | 260일 롤링 초기값 안정화 |
| VIX 수익률 계산 | 로그 수익률 | **수준 차분 `.diff()`** | VIX는 이미 % 단위 — 로그 적용 시 왜곡 |
| HY 스프레드 소스 | BAMLH0A0HYM2 (ICE, 3년 한계) | **BAA10Y** (전체 기간) | ICE 라이선스 제약 |
| DGS10 변환 | 수준만 | **수준 + 5일 차분(DGS10_chg)** | 비정상 시계열 보정, 금리 서프라이즈 포착 |
| WEI·SAHMREALTIME | df_reg_v2 포함 | **df_reg_v2 제외, fred_data.csv 보존** | PIT 적용 시 분석 기간 4년 축소 방지 |
| HY_spread 변수명 | 소스 변경 시 변수명도 변경 | **변수명 'HY_spread' 유지** | Step6 이후 코드 참조명 일관성 |
| features.csv 저장 시점 | 슬라이싱 후 저장 | **슬라이싱 전 전체 기간 저장** | 워밍업 구간 원본 보존 |
| Granger 유의하지 않은 변수 | 제거 | **df_reg_v2에 보존** | XGBoost 비선형 모형에서 여전히 유용 가능 |

---

### Step 간 연계 구조

```
Step2 입력                  →  처리 방식
─────────────────────────────────────────────────────────────
portfolio_prices.csv        →  port_ret (로그 수익률)
                                rv_neutral (21일 롤링 변동성)
external_prices.csv         →  ext_ret (로그/차분 수익률)
                                ext_vol21 (21일 롤링 변동성)
                                VIX_contango, Cu_Au_ratio 등 파생변수
fred_data.csv               →  HY_spread(BAA10Y), yield_curve(T10Y2Y)
                                claims_zscore(ICSA 260일 z-score)
                                DGS10, DGS10_chg, CPI_MoM, UNRATE
─────────────────────────────────────────────────────────────
Step2 출력:
  df_reg_v2.csv   (2,491행 × 41열) → Step3 XGBoost walk-forward 입력
  features.csv    (3,017행 × 13열) → 전체 기간 파생변수 원본
  granger_results.csv              → 피처 중요도 참고
  images/step2_0*.png              → EDA 시각화 6종
```

---

> **참고 문서**  
> - `서윤범/project_design_v3.md` — 전체 7단계 파이프라인 아키텍처  
> - `김윤서/Step1_흐름_정리.md` — 데이터 수집 설계 (워밍업 기간, PIT, FRED 수집)  
> - `김윤서/전체_프로젝트_프로세스.md` — 단계별 프로세스 + GDELT 피처 매핑
