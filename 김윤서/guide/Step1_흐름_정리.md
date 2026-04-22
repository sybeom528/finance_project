# Step 1: 데이터 수집 흐름 정리

> 파일: `Step1_Data_Collection.ipynb`  
> 작성자: 김윤서  
> 작성일: 2026-04-19  
> 목적: Step1 전체 흐름 + 설계 결정 근거 문서화

---

## 목차

1. [전체 흐름 개요](#1-전체-흐름-개요)
2. [워밍업 기간 설계 — 왜 2014년부터 수집하는가](#2-워밍업-기간-설계--왜-2014년부터-수집하는가)
3. [수집 대상 전체 목록](#3-수집-대상-전체-목록)
4. [1-1. 투자 자산 정의 (30개)](#4-1-1-투자-자산-정의-30개)
5. [1-2. yfinance 시세 수집 (41개)](#5-1-2-yfinance-시세-수집-41개)
6. [1-3. FRED 수집 — 하이브리드 PIT](#6-1-3-fred-수집--하이브리드-pit)
7. [1-4. 데이터 검증 + 기초 통계](#7-1-4-데이터-검증--기초-통계)
8. [1-5. CSV 저장 및 Step 연계](#8-1-5-csv-저장-및-step-연계)
9. [주요 설계 결정 요약](#9-주요-설계-결정-요약)

---

## 1. 전체 흐름 개요

```
Step1_Data_Collection.ipynb
│
├── [설정] 기간 정의
│       WARMUP_START   = 2014-01-01  ← 롤링 윈도우 안정화용 사전 수집
│       ANALYSIS_START = 2016-01-01  ← 실제 분석 기준일
│       END            = 2025-12-31
│
├── [Cell 4] BAA10Y 실용성 검증
│       BAMLH0A0HYM2(ICE 독점, 3년만 가능) → BAA10Y(전체 기간)로 대체 검증
│
├── [Cell 5] 티커 정의
│       투자 자산 30개 + 외부 지표 6개 + 대안 yfinance 5개 + FRED 8개
│
├── [1-2] yfinance 수집 (Cell 7)
│       41개 티커 개별 수집 (WARMUP_START~END)
│       실패 티커 별도 수집
│
├── [1-2] DataFrame 병합 (Cell 8)
│       SPY 실거래일을 기준 인덱스로 사용
│       ffill() 단독 적용 (bfill 금지 — look-ahead bias)
│
├── [1-2] 투자 자산 / 외부 지표 분리 (Cell 9)
│       df_portfolio (30개), df_ext_alt (12개)
│
├── [1-3] FRED 수집 — 하이브리드 PIT (Cell 13)
│       일별 시리즈: observation + 발표 시차 lag
│       주간·월간: ALFRED vintage 기반 완전 PIT 재구성
│
├── [1-3] FRED 달력일 ffill → SPY 영업일 필터 (Cell 16)
│       ffill만 사용, bfill 금지
│
├── [1-4] 데이터 검증 (Cell 18~20)
│       결측률 리포트, 투자 자산 10년 수익률 시각화, 대안 데이터 시계열
│
└── [1-5] CSV 저장 (Cell 22)
        portfolio_prices.csv, external_prices.csv, fred_data.csv
```

---

## 2. 워밍업 기간 설계 — 왜 2014년부터 수집하는가

### 2-1. 문제: 롤링 윈도우의 "초기 불안정 구간"

Step2 피처 엔지니어링에서 여러 지표를 **롤링(이동) 윈도우**로 계산한다.

```
claims_zscore = (ICSA - rolling_mean(260)) / rolling_std(260)
```

**롤링 윈도우의 구조적 문제**: 윈도우 길이 N일인 지표를 계산할 때,  
**처음 N-1일은 값을 계산할 충분한 과거 데이터가 없다.**

```
예: claims_zscore (260일 롤링)

시작일이 2016-01-01이면:
  2016-01-01 ~ 2016-12-31 (약 260 영업일) 동안
  → 롤링 평균/표준편차가 수렴하지 않은 상태 (초기 샘플 부족)
  → 2016년 전체가 사실상 사용 불가능한 데이터

시작일을 2014-01-01로 당기면:
  2014-01-01 ~ 2015-12-31 (약 504 영업일)이 워밍업 구간
  → 2016-01-01 시점에서 이미 260일의 2배 이상 확보
  → 롤링 통계가 완전 수렴한 안정적인 값 제공
```

### 2-2. 워밍업 기간이 필요한 피처 목록

| 피처 | 롤링 윈도우 | 2016년 기준 필요 과거 데이터 | 워밍업 기간 (2년) 필요 이유 |
|------|-----------|---------------------------|--------------------------|
| `claims_zscore` | **260일** (≈ 1 거래연도) | 260 영업일 | 가장 긴 윈도우 — 워밍업 설계 기준이 됨 |
| `SKEW_zscore` | 63일 (≈ 3개월) | 63 영업일 | 260일의 부분 집합 — 워밍업으로 커버 |
| `Cu_Au_ratio_chg` | 21일 (≈ 1개월) | 21 영업일 | 260일의 부분 집합 — 커버 |
| `rv_neutral` | 21일 | 21 영업일 | 260일의 부분 집합 — 커버 |
| `VIX_contango` | 단순 비율 | — | 롤링 없음, 워밍업 불필요 |
| HMM (Step3) | IS=150일 | 150 영업일 | 워밍업으로 커버 |
| XGBoost walk-forward | IS=150일 | 150 영업일 | 워밍업으로 커버 |

> 가장 긴 롤링 윈도우 = **260일** (`claims_zscore`).  
> 워밍업 기간 = 2014-01 ~ 2015-12 = 약 **504 영업일** = 260일의 **약 2배** 확보.  
> 이 여유 덕분에 2016-01-01부터 **모든 피처가 완전 수렴된 상태**로 분석 시작 가능.

### 2-3. 워밍업 없이 분석을 시작하면 어떤 일이 생기나

```
2016-01-04 (첫 거래일) claims_zscore 계산:
  → 현재까지 ICSA 데이터 1개 (2016-01-07 발표)
  → rolling(260).std() = NaN  (데이터 부족)
  → claims_zscore = NaN

→ Step2 패널에 NaN 발생
→ Step3 XGBoost 학습에서 해당 행 drop
→ 초기 거래일 데이터가 누락되어 백테스트 왜곡
```

### 2-4. 워밍업 데이터의 최종 처리 방법

```python
# Step1에서 수집: 2014-01-01 ~ 2025-12-31 전체 저장
WARMUP_START   = '2014-01-01'
END            = '2025-12-31'

# Step2에서 피처 계산 후 슬라이싱:
ANALYSIS_START = '2016-01-01'
long_panel = long_panel.loc[ANALYSIS_START:]  # 워밍업 구간 제거
```

→ Step1 CSV에는 2014년 데이터가 포함되어 있지만, **Step2 이후에서는 2016년 이후만 사용**한다.  
→ 이 방식으로 피처는 완전히 수렴하되, 분석 기간은 2016-2025 10년으로 명확히 제한된다.

### 2-5. BTC-USD 특수 처리

```python
# BTC-USD는 2014-09 이후 데이터만 존재 → 워밍업 8개월 결측
# WARMUP_START(2014-01) ~ 2014-08: NaN
# 처리: ffill() 적용 시 BTC 시작 시점 이전은 NaN 유지 → dropna로 제거
# 결론: ANALYSIS_START(2016-01) 기준으로는 결측 없음 → 무해
```

---

## 3. 수집 대상 전체 목록

```
총 49개 시리즈

yfinance (41개)
├── 투자 자산 (30개) — 실제 매매 대상
│   ├── 인덱스 ETF  (5): SPY, QQQ, IWM, EFA, EEM
│   ├── 채권 ETF    (4): TLT, AGG, SHY, TIP
│   ├── 대안 ETF    (2): GLD, DBC
│   ├── 섹터 ETF   (11): XLK, XLF, XLE, XLV, VOX, XLY, XLP, XLI, XLU, XLRE, XLB
│   └── 개별 종목   (8): AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, PG, XOM
│
├── 외부 지표 (6개) — 관찰 전용, 투자 대상 아님
│   CL=F, GC=F, SI=F, BTC-USD, ^VIX, DX-Y.NYB
│
└── 대안 yfinance (5개)
    ^VIX9D, ^VIX3M, ^VIX6M, ^SKEW, HG=F

FRED (8개)
├── 대안 매크로 (5): BAA10Y, T10Y2Y, ICSA, WEI, SAHMREALTIME
└── 기존 매크로 (3): DGS10, CPIAUCSL, UNRATE
```

---

## 4. 1-1. 투자 자산 정의 (30개)

### 인덱스 ETF (5개) — 자산군 대표

| 티커 | 자산군 | 역할 |
|------|--------|------|
| SPY | 미국 대형주 | S&P 500 대표, 기준 인덱스 역할도 겸임 |
| QQQ | 미국 기술주 | NASDAQ-100, 성장주 노출 |
| IWM | 미국 소형주 | Russell 2000, 내수 경기 민감 |
| EFA | 선진국 주식 | MSCI EAFE (유럽/일본), 지역 분산 |
| EEM | 신흥국 주식 | MSCI EM, 달러/원자재 민감 |

### 채권 ETF (4개)

| 티커 | 설명 | 역할 |
|------|------|------|
| TLT | 미국 20Y+ 국채 | 금리 위험 대표, 주식과 음의 상관 |
| AGG | 미국 종합 채권 | 중간 듀레이션 채권 시장 |
| SHY | 미국 1-3Y 단기채 | 금리 민감도 낮음, 현금 대체 |
| TIP | 물가연동채 (TIPS) | 인플레이션 헷지 |

### 대안 ETF (2개)

| 티커 | 설명 | 역할 |
|------|------|------|
| GLD | 금 ETF | 실물 자산, 달러 약세/위기 헷지 |
| DBC | 원자재 ETF | 에너지·농산물·금속 복합 노출 |

### 섹터 ETF (11개) — GICS 기반

| 티커 | 섹터 |
|------|------|
| XLK | 기술 |
| XLF | 금융 |
| XLE | 에너지 |
| XLV | 헬스케어 |
| VOX | 통신 |
| XLY | 경기소비재 |
| XLP | 필수소비재 |
| XLI | 산업재 |
| XLU | 유틸리티 |
| XLRE | 부동산 |
| XLB | 소재 |

### 개별 종목 (8개)

| 티커 | 기업 | 선택 이유 |
|------|------|---------|
| AAPL | Apple | 기술 섹터 대형주 |
| MSFT | Microsoft | 클라우드 중심 기술 대형주 |
| AMZN | Amazon | 이커머스 + 클라우드 |
| GOOGL | Alphabet | 디지털 광고 대표 |
| JPM | JPMorgan | 금융 섹터 대형주 |
| JNJ | J&J | 헬스케어 방어 종목 |
| PG | P&G | 필수소비재 방어 종목 |
| XOM | ExxonMobil | 에너지 섹터 대표 |

### 외부 지표 (6개) — 관찰 전용

| 티커 | 설명 | 역할 |
|------|------|------|
| CL=F | WTI 원유 선물 | 에너지 경기 선행 |
| GC=F | 금 선물 (COMEX) | 위험 회피 심리 |
| SI=F | 은 선물 (COMEX) | 산업 수요 + 귀금속 |
| BTC-USD | 비트코인 | 위험 선호/회피 지표 |
| ^VIX | CBOE 공포 지수 | 단기 변동성 기대치 |
| DX-Y.NYB | 달러 인덱스 (DXY) | 글로벌 달러 강세 |

### 대안 yfinance (5개)

| 티커 | 설명 | 피처 역할 |
|------|------|---------|
| ^VIX9D | 9일 내재변동성 | VIX 기간구조 (초단기) |
| ^VIX3M | 3개월 내재변동성 | VIX 기간구조 (중기) |
| ^VIX6M | 6개월 내재변동성 | VIX 기간구조 (장기) |
| ^SKEW | CBOE SKEW 지수 | 꼬리 위험 (tail risk) |
| HG=F | 구리 선물 | 구리/금 비율 (경기 선행 지표) |

> **VIX 기간구조의 의미**: `^VIX` (30일) vs `^VIX3M` (90일) vs `^VIX6M` (180일)을 동시에 보유하면  
> Step2에서 `VIX_contango = ^VIX3M / ^VIX` (정상 시장: >1, 위기: <1) 등의 기간구조 피처를 계산할 수 있다.

---

## 5. 1-2. yfinance 시세 수집 (41개)

### 수집 방식 — 개별 루프 (안정성 우선)

```python
for ticker in all_yf_tickers:
    df = yf.download(ticker, start=WARMUP_START, end=END,
                     auto_adjust=True, progress=False)
    series = df['Close'].dropna()
    yf_data[ticker] = series
```

**왜 일괄 수집(batch) 대신 개별 수집인가:**

- yfinance 일괄 수집은 일부 티커 실패 시 **전체 DataFrame이 비어있을 가능성**이 있음
- 개별 수집은 한 티커가 실패해도 나머지 39개는 정상 유지
- `failed` 리스트로 실패 티커를 별도 추적

**`auto_adjust=True`의 의미:**

주식 분할(split)이나 배당락이 발생하면 과거 가격이 조정된다.  
`auto_adjust=True`는 **Adjusted Close** 가격을 사용해 시계열 연속성을 보장한다.  
예: AAPL이 4:1 분할 시 분할 이전 가격을 모두 1/4로 조정 → 수익률 계산 오류 방지.

### DataFrame 병합 — SPY 실거래일 기준 인덱스

```python
# pd.bdate_range(freq='B') 사용 금지 이유:
# → NYSE 공휴일(MLK Day, Good Friday, Thanksgiving 등 9개)을 포함해버림
# → 해당 날의 Close가 NaN → ffill로 전일 값 복사 → 데이터 정합성 문제

# 올바른 방법: SPY는 NYSE에 상장된 ETF이므로
# SPY.dropna().index = 실제 NYSE 영업일 그 자체
spy_trading_days = yf_data['SPY'].dropna().index
df_all_prices = df_all_prices.reindex(spy_trading_days)
```

### ffill() 단독 사용 — bfill() 금지

```python
df_all_prices = df_all_prices.ffill()   # ← OK
# df_all_prices = df_all_prices.bfill() # ← 절대 금지
```

| 방법 | 동작 | 경제적 해석 | 백테스트 영향 |
|------|------|-----------|------------|
| `ffill()` | 결측일에 직전 종가 유지 | "공휴일엔 가격이 변하지 않는다" | 정상 — 실제와 동일 |
| `bfill()` | 결측일에 **다음 날** 가격을 소급 | 미래 가격을 과거에 삽입 | **look-ahead bias 발생** |

> look-ahead bias: 2020-01-20(MLK Day)이 결측이면 `bfill()`은 1월 21일 종가를 삽입 →  
> 1월 20일에 1월 21일 가격을 이미 알고 있는 것처럼 되어 백테스트 성과가 비현실적으로 좋게 나옴.

### 투자 자산 / 외부 지표 분리 저장

```python
df_portfolio = df_all_prices[ALL_PORTFOLIO]          # 30개 — 실제 매매 대상
df_ext_alt   = df_all_prices[EXTERNAL + ALT_YF]      # 12개 — 관찰 전용
```

투자 자산과 외부 지표를 분리하는 이유:
- **Step4 Black-Litterman**: `portfolio_prices.csv`에서 공분산행렬(Σ) 계산
- **Step2 피처 엔지니어링**: `external_prices.csv`에서 VIX 기간구조, Cu/Au 비율 등 파생 피처 생성
- 두 역할을 섞으면 이후 단계에서 자산 선택 로직이 복잡해짐

---

## 6. 1-3. FRED 수집 — 하이브리드 PIT

### 6-1. FRED 수집 대상 (8개)

| 시리즈 ID | 설명 | 빈도 | 수집 방식 |
|----------|------|------|---------|
| `BAA10Y` | Moody Baa − 10Y 신용스프레드 | 일별 | obs + lag 1일 |
| `T10Y2Y` | 10Y-2Y 수익률 곡선 스프레드 | 일별 | obs + lag 0일 |
| `DGS10` | 미국 10년 국채 수익률 | 일별 | obs + lag 0일 |
| `ICSA` | 신규 실업수당 청구 (주간) | 주간 (목요일) | ALFRED vintage PIT |
| `WEI` | Weekly Economic Index | 주간 | ALFRED vintage PIT |
| `SAHMREALTIME` | Sahm 경기침체 지표 | 월간 | ALFRED vintage PIT |
| `CPIAUCSL` | 소비자물가지수 (CPI) | 월간 | ALFRED vintage PIT |
| `UNRATE` | 실업률 | 월간 | ALFRED vintage PIT |

### 6-2. BAA10Y 선택 이유 — BAMLH0A0HYM2 대체

```
원래 목표: BAMLH0A0HYM2 (ICE BofA US High Yield Index, HY 스프레드)
문제: ICE 데이터 라이선스 제약으로 FRED에서 최근 3년(2023~)만 공개
      → 2016-2022 구간 수집 불가 → 전체 기간 사용 불가

대안: BAA10Y (Moody's Seasoned Baa Corporate Bond Yield - 10Y Treasury)
검증 결과:
  - 변화 상관(Δ 기준): ~0.70+ (HY 스프레드와 매우 유사한 움직임)
  - 주요 위기 이벤트(SVB, 엔캐리, 관세 쇼크)에서 방향 일치
  - 전체 기간 2014-2025 수집 가능
결론: BAA10Y 채택 확정
```

### 6-3. Point-In-Time (PIT) 처리 — look-ahead bias 제거

**문제**: FRED `get_series()`는 **관측 일자(observation date)** 기준으로만 데이터를 반환한다.

```
예: ICSA (신규 실업수당 청구, 주간)
  - 1월 3주차 결과 관측일: 2020-01-18 (토요일)
  - 실제 FRED 발표일: 2020-01-23 (목요일)
  - 시차: 5일

기존 방식 문제:
  get_series()로 수집 시 → 2020-01-18에 값이 존재
  백테스트에서 2020-01-21(화요일)에 이 값을 사용
  → 실제로는 아직 발표 전인 데이터를 이미 알고 사용
  → look-ahead bias
```

**PIT 방식**: 각 날짜 T에서 **T 시점까지 실제로 발표된** 값만 사용

```
ALFRED vintage API: 모든 수정 이력 포함
  2020-01-23 발표: 220만 건 (속보치)
  2020-01-30 수정: 222만 건 (잠정치)
  2020-02-20 수정: 223만 건 (확정치)

PIT 재구성:
  2020-01-21 (화)에서 사용 가능한 값 = 없음 (아직 발표 전)
  2020-01-23 (목) 이후 = 220만 건 (속보치)
  2020-01-30 (목) 이후 = 222만 건 (잠정치)
  2020-02-20 (목) 이후 = 223만 건 (확정치)
```

### 6-4. 하이브리드 방식 선택 근거

| 시리즈 유형 | 방식 | 이유 |
|-----------|------|------|
| 일별 (`DGS10`, `T10Y2Y`, `BAA10Y`) | `obs + lag` | ALFRED vintage API가 ~2,000개 한도 초과로 실패; 발표 시차 0~1일로 미미 |
| 주간·월간 (`ICSA`, `WEI`, `SAHMREALTIME`, `CPIAUCSL`, `UNRATE`) | `ALFRED vintage PIT` | 수정 빈도 높음 (ICSA 최대 21배 수정 사례 있음), 발표 시차 수일~수주 |

### 6-5. `get_fred_pit()` 알고리즘 상세

```python
def get_fred_pit(series_id, start, end):
    # 1) 모든 vintage (수정 이력 포함) 수집
    all_rel = fred_client.get_series_all_releases(series_id)

    # 2) 발표일(realtime_start) 순으로 정렬
    all_rel = all_rel.sort_values(['realtime_start', 'date'])

    # 3) 누적 snapshot 유지
    #    snapshot: {obs_date → 해당 날짜의 가장 최근 발표 값}
    snapshot = {}
    for rt, obs_date, value in zip(...):
        snapshot[obs_date] = value
        # 현재까지 발표된 것 중 가장 최근 관측일의 값을 timeline에 기록
        timeline.append((rt, snapshot[latest_obs]))

    # 4) asof-merge로 각 거래일 T에 "T 시점에서 사용 가능했던 값" 할당
    pd.merge_asof(daily, timeline, direction='backward')
```

```
시간축:
  1월  5일 (일요일) T  → 아직 발표 없음 → NaN
  1월  9일 (목요일) T  → 1/9 발표 값 적용
  1월 16일 (목요일) T  → 1/16 발표 값 적용 (1/9 수정치 포함 가능)
  ───────────────────────────────────────────────────────
  각 날짜가 실제로 알 수 있었던 가장 최신 값만 사용 → look-ahead bias 완전 차단
```

### 6-6. FRED → SPY 영업일 정렬

```python
# 주간·월간 데이터는 발표일에만 값이 있음 → ffill로 다음 발표 전까지 유지
daily_range  = pd.date_range(start=WARMUP_START, end=END, freq='D')
df_fred_daily = df_fred_raw.reindex(daily_range).ffill()   # 달력일 기준 전파

# NYSE 영업일만 추출
df_fred = df_fred_daily.reindex(spy_trading_days)
```

```
ICSA 예시 (목요일 발표):
  Date        ICSA(PIT)
  2020-01-21  220.0   ← 1/23 발표 전이므로 이전 값 ffill
  2020-01-22  220.0
  2020-01-23  220.0   ← 1/23 발표일 → 신규 값 220.0 반영
  2020-01-24  220.0   ← 다음 발표 전까지 ffill 유지
```

---

## 7. 1-4. 데이터 검증 + 기초 통계

### 7-1. 결측률 리포트 (Cell 18)

```python
all_data = pd.concat([df_all_prices, df_fred], axis=1)
report = DataFrame({'결측 수': all_data.isnull().sum(), '결측률(%)': ...})
```

PIT 적용 후 초기 구간에서 결측이 자연 발생한다:  
`SAHMREALTIME`, `WEI` 등은 FRED 데이터가 일정 기간 이후에만 존재 → 2014~2015 구간 일부 NaN.  
이는 "아직 발표되지 않은 데이터를 사용하지 않는다"는 PIT의 올바른 동작.  
ANALYSIS_START(2016-01) 이후에는 모든 시리즈가 충분히 수집되어 NaN 없음.

### 7-2. 투자 자산 기초 통계 (Cell 19)

2016-01 ~ 2025-12 기간의 **10년 총 수익률** 막대그래프 시각화.  
상위권: AAPL, MSFT, AMZN, GOOGL 등 기술주.  
하위권: EEM, XLE, DBC 등 신흥국·원자재.  
저장: `images/step1_01_total_returns.png`

### 7-3. 대안 데이터 시계열 미리보기 (Cell 20)

6개 서브플롯으로 구성:

| 서브플롯 | 내용 | 주요 시각적 포인트 |
|---------|------|----------------|
| VIX 기간구조 | ^VIX9D / ^VIX3M / ^VIX6M 동시 시각화 | 위기 시 단기 VIX 폭등으로 역전 발생 |
| CBOE SKEW | ^SKEW 시계열 | 꼬리 위험 고조 시 140+ |
| 신용 스프레드 | BAA10Y | 2020 코로나, 2022 금리인상 시 급등 |
| 수익률 곡선 | T10Y2Y + 반전 구간 강조 | 2019~2023 역전 구간 주황색 음영 |
| 구리 선물 | HG=F | 경기 선행 지표로 금과의 비율 시각화 |
| Sahm + WEI | SAHMREALTIME (빨강) + WEI (파랑) | 0.5 트리거 선 + 이중 축 비교 |

저장: `images/step1_02_alt_data_preview.png`

---

## 8. 1-5. CSV 저장 및 Step 연계

### 저장 파일 목록

| 파일 | 행수 × 열수 | 기간 | 사용처 |
|------|-----------|------|--------|
| `data/portfolio_prices.csv` | ~3,000행 × 30열 | 2014-01 ~ 2025-12 | Step2~7 전체 |
| `data/external_prices.csv` | ~3,000행 × 12열 | 2014-01 ~ 2025-12 | Step2 피처 엔지니어링 |
| `data/fred_data.csv` | ~3,000행 × 8열 | 2014-01 ~ 2025-12 | Step2 피처 엔지니어링 |

### Step 간 연계 구조

```
Step1 출력                  →  Step2 사용 방식
─────────────────────────────────────────────────────────────
portfolio_prices.csv        →  수익률(ret_1d, ret_21d) 계산
                                각 자산의 롤링 변동성 계산
external_prices.csv         →  VIX_contango = ^VIX3M / ^VIX
                                Cu_Au_ratio  = HG=F / GC=F
                                SKEW_zscore  = rolling_z(^SKEW)
fred_data.csv               →  yield_curve = T10Y2Y
                                HY_spread   = BAA10Y
                                claims_zscore = rolling_z(ICSA, 260)
                                sahm_indicator = SAHMREALTIME
                                WEI         = WEI
                                CPI, UNRATE, DGS10 → 매크로 피처
─────────────────────────────────────────────────────────────
Step2 출력: long_panel.parquet (date × ticker, 모든 피처 포함)
                            →  Step3 XGBoost + HMM 입력
```

---

## 9. 주요 설계 결정 요약

| 결정 항목 | 이전 방식 | 최종 방식 | 근거 |
|---------|---------|---------|------|
| 수집 시작일 | 2016-01 | **2014-01** (워밍업 2년) | claims_zscore 260일 롤링 안정화 |
| yfinance 수집 방식 | 일괄(batch) | **개별 루프** | 일부 실패 시 전체 보호 |
| 인덱스 기준 | `pd.bdate_range(freq='B')` | **SPY 실거래일** | NYSE 공휴일 자동 반영 |
| 결측 처리 | `ffill() + bfill()` | **ffill() 단독** | bfill() = look-ahead bias |
| HY 스프레드 소스 | BAMLH0A0HYM2 (ICE, 3년 한계) | **BAA10Y** (FRED 전체 기간) | ICE 라이선스 제약 |
| FRED 일별 시리즈 | ALFRED vintage | **obs + lag** | ALFRED API 2,000개 한도 초과 |
| FRED 주간·월간 | `get_series()` | **ALFRED vintage PIT** | 발표 시차·수정 이력 반영 필요 |
| ETH-USD | 수집 | **제거** | 2015-08 이전 없음 + 후속 단계 미사용 |
| 투자 자산 저장 분리 | 단일 파일 | **portfolio / external 분리** | Step4 BL 공분산 계산 구조 명확화 |

---

> **참고 문서**  
> - `서윤범/project_design_v3.md` — 전체 7단계 파이프라인 아키텍처  
> - `김윤서/전체_프로젝트_프로세스.md` — 단계별 프로세스 + GDELT 피처 매핑  
> - `김윤서/GDELT_쿼리_설계_보완.md` — GDELT 수집 설계 (Step1과 별개로 BigQuery 수집)
