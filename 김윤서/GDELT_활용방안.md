# GDELT 뉴스 감성 분석 활용 방안
> 프로젝트: 성향별 액티브 펀드 구성 및 리스크 관리  
> 작성자: 김윤서  
> 최종 수정: 2025.04

---

## 0. 결론 요약

GDELT **단독** 사용으로 NewsAPI / yfinance 뉴스 수집을 대체한다.  
GKG 테이블의 **금융 특화 GCAM(c15, Loughran-McDonald) 사전**으로 FinBERT 역할을 대체하고,  
EVENTS의 `GoldsteinScale`로 충격 강도를, MENTIONS의 `tone_std`로 Ω 재료를 추출한다.  
이 피처들을 기존 yfinance / FRED 데이터와 병합하여 **레짐 분류 + ML 입력 피처**로 동시에 활용한다.

---

## 1. GDELT 단독 사용 결정 근거

| 비교 항목 | yfinance + NewsAPI | GDELT 단독 |
|---|---|---|
| 데이터 기간 | NewsAPI 무료 = 최근 1개월 | 2015년~ 전체 무료 |
| 금융 도메인 특화 | FinBERT 활용 가능 | GCAM c15(Loughran-McDonald) 사전으로 대체 |
| 중복 기사 문제 | Reuters 등 언론사 겹침 | 단일 소스로 중복 없음 |
| 구현 복잡도 | 3개 파이프라인 필요 | 1개 파이프라인으로 통합 |
| 백테스트 적합성 | 과거 데이터 한계 | 10년치 일괄 수집 가능 |

**핵심 판단:**  
GDELT GKG 테이블 안에 이미 67개 감성 사전(GCAM)이 포함되어 있고,  
그 중 `c15(Loughran-McDonald Financial)`가 금융 도메인 특화 사전으로  
FinBERT 없이도 금융 맥락의 감성 점수를 추출할 수 있다.

---

## 2. GDELT 3개 테이블 구조 및 역할

```
GDELT 데이터셋
├── GKG      : 기사의 감성 / 테마 / 소스 분석  ← 메인
├── EVENTS   : 사건의 충격 강도 / 행위자 정보  ← 보조
└── MENTIONS : 사건의 언론 확산도 / 언급 빈도  ← 보조
```

세 테이블은 `GLOBALEVENTID`로 연결된다.

---

## 3. 테이블별 활용 컬럼 상세

### 3.1 GKG — 감성 분석 메인 테이블

| 컬럼 | 내용 | 활용 방식 |
|---|---|---|
| `V2Tone` | 전체감성, 긍정비율, 부정비율 등 콤마 구분 문자열 | 전체 시장 감성 점수 (`avg_tone`) |
| `GCAM` | 67개 감성 사전별 점수 (금융 특화 c15 포함) | **금융 특화 감성 점수** (`fin_sentiment`) |
| `Themes` | ECON_STOCKMARKET 등 테마 태그 | 금융 기사 필터링 |
| `SourceCommonName` | 언론사명 | Reuters 등 고품질 소스 필터 |
| `DATE` | 날짜 | 날짜 인덱스 |
| `DocumentIdentifier` | 기사 URL | 중복 기사 제거 |

**GCAM 금융 특화 사전 목록:**

| 사전 ID | 사전명 | 특징 |
|---|---|---|
| `c15.1` | Loughran-McDonald Financial Positive | 금융 문서 특화 긍정 사전 |
| `c15.2` | Loughran-McDonald Financial Negative | 금융 문서 특화 부정 사전 |
| `c16.1` | Henry Financial Positive | 금융 보도 특화 |
| `c16.2` | Henry Financial Negative | 금융 보도 특화 |

> **왜 c15(Loughran-McDonald)인가?**  
> 일반 감성 사전에서 "liability", "loss" 같은 단어는 중립이지만  
> 금융 문서에서는 명백히 부정적이다.  
> Loughran-McDonald 사전은 이런 금융 도메인 특수성을 반영하여  
> FinBERT와 가장 유사한 금융 감성 점수를 제공한다.

```python
# GKG에서 금융 특화 감성 추출 쿼리
query = """
SELECT
    PARSE_DATE('%Y%m%d', CAST(DATE AS STRING)) AS date,
    
    -- V2Tone: 전체 감성 (보조)
    AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS avg_tone,
    STDDEV(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS tone_std,
    
    -- GCAM c15: Loughran-McDonald 금융 특화 감성 (메인)
    AVG(
        COALESCE(CAST(REGEXP_EXTRACT(GCAM, r'c15\\.1:([0-9.]+)') AS FLOAT64), 0) -
        COALESCE(CAST(REGEXP_EXTRACT(GCAM, r'c15\\.2:([0-9.]+)') AS FLOAT64), 0)
    ) AS fin_sentiment,
    
    COUNT(*) AS article_count

FROM `gdelt-bq.gdeltv2.gkg`
WHERE DATE BETWEEN 20160101 AND 20251231
  AND (
        Themes LIKE '%ECON_STOCKMARKET%'
     OR Themes LIKE '%ECON_FRBRESERVE%'
     OR Themes LIKE '%ECON_INFLATION%'
     OR Themes LIKE '%ECON_UNEMPLOYMENT%'
  )
  AND SourceCommonName IN (
        'reuters.com', 'cnbc.com', 'wsj.com',
        'ft.com', 'bloomberg.com'
  )
GROUP BY date
ORDER BY date
"""
```

---

### 3.2 EVENTS — 시장 충격 이벤트 감지 (보조)

| 컬럼 | 내용 | 활용 방식 |
|---|---|---|
| `GoldsteinScale` | 사건 충격 강도 (-10 ~ +10) | 시장 충격 강도 피처 (`avg_shock`, `min_shock`) |
| `NumMentions` | 언급 횟수 | 이벤트 확산도 |
| `EventCode` | CAMEO 이벤트 코드 | 경제 관련 이벤트 필터 (10xx대) |
| `Actor1CountryCode` | 행위자 국가코드 | 미국 관련 이벤트만 필터 (`USA`) |
| `AvgTone` | 해당 이벤트 평균 감성 | GKG Tone 보완 |

```python
# EVENTS에서 미국 경제 충격 이벤트 추출 쿼리
events_query = """
SELECT
    PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
    AVG(GoldsteinScale)  AS avg_shock,    -- 충격 강도 평균
    MIN(GoldsteinScale)  AS min_shock,    -- 최대 부정 충격 (스트레스 감지용)
    SUM(NumMentions)     AS total_mentions
FROM `gdelt-bq.gdeltv2.events`
WHERE Actor1CountryCode = 'USA'
  AND EventCode LIKE '1%'               -- CAMEO 경제 관련 코드
  AND SQLDATE BETWEEN 20160101 AND 20251231
GROUP BY date
ORDER BY date
"""
```

---

### 3.3 MENTIONS — 뉴스 확산도 / 감성 일관성 (보조)

| 컬럼 | 내용 | 활용 방식 |
|---|---|---|
| `Confidence` | 이벤트 감지 신뢰도 (0~100) | 노이즈 필터링 (50 이상만) |
| `MentionDocTone` | 해당 언급의 감성 점수 | 기사 간 감성 분산 → Ω 계산 재료 |
| `MentionTimeDate` | 언급 시각 | 감성 확산 속도 측정 |

```python
# MENTIONS에서 감성 일관성 추출 쿼리
mentions_query = """
SELECT
    PARSE_DATE('%Y%m%d', CAST(MentionTimeDate/100 AS STRING)) AS date,
    STDDEV(MentionDocTone) AS mention_tone_std,   -- 낮을수록 기사 간 의견 일치 → Ω 낮게 설정
    COUNT(*)               AS mention_count
FROM `gdelt-bq.gdeltv2.mentions`
WHERE Confidence > 50                              -- 신뢰도 50 이상만 사용
  AND MentionTimeDate BETWEEN 20160101000000 AND 20251231235959
GROUP BY date
ORDER BY date
"""
```

---

## 4. 전체 파이프라인

```
[Step 1] GDELT 3개 테이블 수집
    GKG      → fin_sentiment, avg_tone, tone_std, article_count
    EVENTS   → avg_shock, min_shock, total_mentions
    MENTIONS → mention_tone_std, mention_count
            ↓
[Step 2] 날짜 기준 병합 → gdelt_master 테이블
            ↓
[Step 3] 파생변수 생성 (모멘텀 / 변동성 / Z-score 등)
            ↓
[Step 4] yfinance / FRED 데이터와 최종 병합 → master_df
            ↓
[Step 5] 레짐 분류 (Bull / Bear / Crisis / Neutral)
            ↓
    ┌───────────────────────┐
    ↓                       ↓
[Step 6a]               [Step 6b]
ML 모델 입력 피처       스트레스 테스트 구간 식별
(RandomForest /         (fin_sentiment < 평균 - 2σ)
 XGBoost / AI Agent)
    ↓                       ↓
Q, Ω 생성               MDD, Sharpe 집중 측정
    ↓
[Step 7] 블랙-리터만 → 사후 기대수익률 → MVO → 포트폴리오 비중
```

---

## 5. 코드 구현

### 5.1 3개 테이블 수집 및 병합

```python
import numpy as np
import pandas as pd
from google.cloud import bigquery

# ============================================================
# BigQuery 클라이언트 초기화
# 사전 준비:
#   1. Google Cloud 프로젝트 생성 (무료)
#   2. BigQuery API 활성화
#   3. 터미널: gcloud auth application-default login
# ============================================================
client = bigquery.Client()

START, END = '20160101', '20251231'


def fetch_gkg(start: str, end: str) -> pd.DataFrame:
    """
    GKG 테이블: 금융 특화 감성 점수 추출 (메인)
    - fin_sentiment : Loughran-McDonald 금융 사전 기반 감성 (FinBERT 대체)
    - avg_tone      : 전체 감성 점수 (보완용)
    - tone_std      : 감성 변동성 → Ω 계산 재료
    - article_count : 시장 주목도 피처
    """
    query = f"""
    SELECT
        PARSE_DATE('%Y%m%d', CAST(DATE AS STRING)) AS date,
        AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64))    AS avg_tone,
        STDDEV(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS tone_std,
        AVG(
            COALESCE(CAST(REGEXP_EXTRACT(GCAM, r'c15\\.1:([0-9.]+)') AS FLOAT64), 0) -
            COALESCE(CAST(REGEXP_EXTRACT(GCAM, r'c15\\.2:([0-9.]+)') AS FLOAT64), 0)
        ) AS fin_sentiment,
        COUNT(*) AS article_count
    FROM `gdelt-bq.gdeltv2.gkg`
    WHERE DATE BETWEEN {start} AND {end}
      AND (
            Themes LIKE '%ECON_STOCKMARKET%'
         OR Themes LIKE '%ECON_FRBRESERVE%'
         OR Themes LIKE '%ECON_INFLATION%'
         OR Themes LIKE '%ECON_UNEMPLOYMENT%'
      )
      AND SourceCommonName IN (
            'reuters.com', 'cnbc.com', 'wsj.com',
            'ft.com', 'bloomberg.com'
      )
    GROUP BY date
    ORDER BY date
    """
    df = client.query(query).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def fetch_events(start: str, end: str) -> pd.DataFrame:
    """
    EVENTS 테이블: 미국 경제 관련 시장 충격 강도 추출 (보조)
    - avg_shock : 충격 강도 평균 (-10 ~ +10)
    - min_shock : 최대 부정 충격 → 스트레스 구간 식별
    """
    query = f"""
    SELECT
        PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
        AVG(GoldsteinScale) AS avg_shock,
        MIN(GoldsteinScale) AS min_shock,
        SUM(NumMentions)    AS total_mentions
    FROM `gdelt-bq.gdeltv2.events`
    WHERE Actor1CountryCode = 'USA'
      AND EventCode LIKE '1%'
      AND SQLDATE BETWEEN {start} AND {end}
    GROUP BY date
    ORDER BY date
    """
    df = client.query(query).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def fetch_mentions(start: str, end: str) -> pd.DataFrame:
    """
    MENTIONS 테이블: 기사 간 감성 일관성 추출 (보조)
    - mention_tone_std : 낮을수록 기사 간 의견 일치 → Ω 낮게 설정 가능
    """
    query = f"""
    SELECT
        PARSE_DATE('%Y%m%d', CAST(MentionTimeDate/100 AS STRING)) AS date,
        STDDEV(MentionDocTone) AS mention_tone_std,
        COUNT(*)               AS mention_count
    FROM `gdelt-bq.gdeltv2.mentions`
    WHERE Confidence > 50
      AND MentionTimeDate BETWEEN {start}000000 AND {end}235959
    GROUP BY date
    ORDER BY date
    """
    df = client.query(query).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def build_gdelt_master(start: str, end: str) -> pd.DataFrame:
    """
    3개 테이블 날짜 기준 병합 → gdelt_master
    
    최종 컬럼:
    fin_sentiment, avg_tone, tone_std, article_count,  ← GKG
    avg_shock, min_shock, total_mentions,               ← EVENTS
    mention_tone_std, mention_count                     ← MENTIONS
    """
    print("GKG 수집 중...")
    gkg_df      = fetch_gkg(start, end)
    print("EVENTS 수집 중...")
    events_df   = fetch_events(start, end)
    print("MENTIONS 수집 중...")
    mentions_df = fetch_mentions(start, end)

    master = (gkg_df
              .join(events_df,   how='left')
              .join(mentions_df, how='left')
              .fillna(method='ffill'))  # 주말 / 공휴일 결측 forward fill

    print(f"GDELT master 테이블 생성 완료: {master.shape}")
    print(master.head())
    return master
```

---

### 5.2 GDELT 파생변수 생성

```python
def create_gdelt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    GDELT 원시 피처 → 모델 입력용 파생변수 생성
    기존 EDA의 모멘텀 / 변동성 파생변수 생성 방식과 동일하게 적용
    
    생성 피처 목록:
    ┌─────────────────────┬─────────────────────────────────────────┐
    │ 피처명              │ 의미                                    │
    ├─────────────────────┼─────────────────────────────────────────┤
    │ sent_momentum       │ 현재 감성 - 20일 이동평균               │
    │ sent_volatility     │ 20일 감성 표준편차 (불확실성 지표)      │
    │ sent_zscore         │ 60일 기준 감성 Z-score (이상치 감지)    │
    │ news_spike          │ 기사량 급증 여부 (0/1)                  │
    │ crisis_signal       │ 충격강도 × 감성 복합 위기 지표          │
    │ omega_proxy         │ 감성 일관성 기반 Ω 추정값               │
    └─────────────────────┴─────────────────────────────────────────┘
    """
    df = df.copy()

    # 감성 모멘텀: 현재 감성이 최근 추세보다 얼마나 이탈했는지
    df['sent_momentum']   = (df['fin_sentiment']
                             - df['fin_sentiment'].rolling(20).mean())

    # 감성 변동성: 높을수록 시장 불확실성 ↑ → Ω 높게 설정
    df['sent_volatility'] = df['fin_sentiment'].rolling(20).std()

    # 감성 Z-score: 60일 기준 현재 감성이 얼마나 극단적인지
    # Z < -2 구간이 스트레스 테스트 대상
    roll_mean = df['fin_sentiment'].rolling(60).mean()
    roll_std  = df['fin_sentiment'].rolling(60).std()
    df['sent_zscore']     = (df['fin_sentiment'] - roll_mean) / roll_std

    # 기사량 급증 여부: 평소 대비 2배 이상 = 시장 주목 이벤트 발생
    df['news_spike']      = (
        df['article_count'] > df['article_count'].rolling(20).mean() * 2
    ).astype(int)

    # 충격-감성 복합 지표: 충격 강하고(낮은 GoldsteinScale) 감성 부정 = 위기
    df['crisis_signal']   = df['avg_shock'] * df['fin_sentiment']

    # Ω 추정값: mention_tone_std 낮을수록(기사 간 의견 일치) 확신도 높음
    df['omega_proxy']     = df['mention_tone_std'].apply(
        lambda x: 0.01 if x < 0.3 else (0.05 if x < 0.6 else 0.10)
    )

    return df
```

---

### 5.3 레짐 분류

```python
def classify_regime(gdelt_df: pd.DataFrame,
                    vix: pd.Series,
                    spread: pd.Series) -> pd.Series:
    """
    GDELT 감성 + VIX + 하이일드 스프레드 결합 → 레짐 분류
    
    레짐 분류 기준:
    ┌─────────┬──────────────────┬──────────┬─────────────────────┐
    │ 레짐    │ 감성 조건        │ VIX 조건 │ 포트폴리오 반응     │
    ├─────────┼──────────────────┼──────────┼─────────────────────┤
    │ Crisis  │ 평균 - 2σ 이하   │  > 30    │ 채권 / 원자재 확대  │
    │ Bear    │ 평균 - 1σ 이하   │    -     │ 방어주 확대         │
    │ Bull    │ 평균 + 1σ 이상   │  < 18    │ 공격형 주식 확대    │
    │ Neutral │ 그 외            │    -     │ 기본 비중 유지      │
    └─────────┴──────────────────┴──────────┴─────────────────────┘
    
    스트레스 테스트 구간:
    sent_zscore < -2 인 날짜 자동 식별
    → 해당 구간의 MDD, Sharpe 집중 측정
    """
    df = gdelt_df.copy()
    df['VIX']    = vix
    df['spread'] = spread

    mean = df['fin_sentiment'].mean()
    std  = df['fin_sentiment'].std()

    conditions = [
        (df['fin_sentiment'] < mean - 2*std) & (df['VIX'] > 30),
        (df['fin_sentiment'] < mean - std)   & (df['spread'] > 5),
        (df['fin_sentiment'] > mean + std)   & (df['VIX'] < 18),
    ]
    choices = ['Crisis', 'Bear', 'Bull']
    regime  = pd.Series(
        np.select(conditions, choices, default='Neutral'),
        index=df.index
    )

    # 스트레스 테스트 구간 출력
    stress = df[df['sent_zscore'] < -2].index
    print(f"\n스트레스 테스트 구간 {len(stress)}일 식별")
    print("주요 구간 (상위 5개):", stress[:5].tolist())

    return regime
```

---

### 5.4 최종 master_df 병합 및 모델 연결

```python
def build_master_df(gdelt_master, yf_data, fred_data):
    """
    GDELT + yfinance 가격 + FRED 매크로 최종 병합
    → ML 모델 (RandomForest / XGBoost / AI Agent) 입력 피처로 사용
    
    최종 master_df 컬럼 구성:
    ┌──────────────────────────────────────────────────┐
    │ 가격/수익률   : SPY_ret, QQQ_ret, XLK_ret ...   │
    │ 외부 지표     : VIX, DXY, BTC-USD ...           │
    │ FRED 매크로   : DGS10, CPI, UNRATE ...          │
    │ GDELT 감성    : fin_sentiment, avg_tone ...      │
    │ GDELT 파생    : sent_momentum, sent_zscore ...   │
    │ 레짐 레이블   : regime (Bull/Bear/Crisis/Neutral)│
    └──────────────────────────────────────────────────┘
    """
    # 가격 데이터 → 수익률 변환
    price_df = pd.DataFrame(yf_data).pct_change().add_suffix('_ret')

    # GDELT 파생변수 생성
    gdelt_feat = create_gdelt_features(gdelt_master)

    # VIX, 스프레드 추출 (yfinance / FRED에서)
    vix_series    = pd.DataFrame(yf_data)['^VIX']
    spread_series = fred_data['BAMLH0A0HYM2']

    # 레짐 분류
    regime_series = classify_regime(gdelt_feat, vix_series, spread_series)

    # 전체 병합
    master = (price_df
              .join(gdelt_feat,  how='left')
              .join(fred_data,   how='left')
              .join(regime_series.rename('regime'), how='left')
              .fillna(method='ffill'))

    return master


# ============================================================
# 모델 연결: GDELT 피처 → Q, Ω → 블랙-리터만
# ============================================================
#
# Rolling Window 방식 (150일 학습 → 30일 예측):
#
# [XGBoost 분류]
#   입력: sent_momentum, sent_zscore, crisis_signal 등 GDELT 파생변수
#       + 기존 모멘텀, 변동성 피처
#   출력: 익월 수익률 등급 예측 확률
#   → 예측 확률 = Q (전망값)
#   → 예측 확률의 확신도 = Ω (omega_proxy로 조정)
#
# [RandomForest 회귀]
#   입력: 동일
#   출력: 익월 수익률 예측값 (트리별 예측의 평균)
#   → 예측값 평균 = Q
#   → 예측값 표준편차 = Ω
#
# [AI Agent]
#   입력: 동일
#   출력: 여러 번 호출 후 예측값 평균 = Q, 표준편차 = Ω
#
# → Q, Ω → 블랙-리터만 → 사후 기대수익률(mu) → MVO → 포트폴리오 비중
```

---

## 6. 피처 역할 최종 정리

| 테이블 | 추출 피처 | 파생변수 | 모델 연결 |
|---|---|---|---|
| GKG (GCAM c15) | `fin_sentiment` | `sent_momentum`, `sent_zscore` | ML 입력 피처 + 레짐 분류 |
| GKG (V2Tone) | `avg_tone`, `tone_std` | `sent_volatility` | ML 입력 피처 보완 |
| EVENTS | `avg_shock`, `min_shock` | `crisis_signal` | 스트레스 구간 식별 |
| MENTIONS | `mention_tone_std` | `omega_proxy` | Ω 추정 재료 |
| 3개 공통 | `article_count` | `news_spike` | 시장 주목도 피처 |

---

## 7. 스트레스 테스트 연결

```
sent_zscore < -2 구간 자동 식별
        ↓
해당 날짜 범위에서 포트폴리오 낙폭(MDD) 집중 측정
        ↓
레짐 변화 감지 전 vs 후 MDD 비교
        ↓
성과 지표: MDD, Sharpe, Sortino, Calmar Ratio
```

**예상 식별 구간 (참고):**
- 2020-02 ~ 2020-04 : 코로나19 쇼크
- 2022-02 ~ 2022-03 : 러시아-우크라이나 침공
- 2022-06 ~ 2022-10 : 연준 급격한 금리 인상
- 2023-03           : SVB 파산 사태

---

## 8. 사전 준비 체크리스트

- [ ] Google Cloud 프로젝트 생성
- [ ] BigQuery API 활성화
- [ ] `gcloud auth application-default login` 실행
- [ ] `pip install google-cloud-bigquery pandas-gbq` 설치
- [ ] GDELT BigQuery 무료 한도 확인 (월 1TB 무료)
- [ ] 쿼리 비용 최소화: `SELECT` 컬럼 최소화, `WHERE` 날짜 필터 필수
