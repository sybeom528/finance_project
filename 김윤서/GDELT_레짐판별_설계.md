# GDELT 레짐 판별 설계 문서

> 프로젝트: 성향별 액티브 ETF 펀드 상품 구축  
> 작성자: 김윤서  
> 작성일: 2026.04.16  
> 목적: GDELT에서 레짐 판별(Bull / Bear / Crisis / Neutral)에 필요한 피처만 추출하는 설계 정리

---

## 0. 결론 요약

GDELT를 **레짐 판별 전용**으로 사용한다.  
GKG 테이블에서 3개 컬럼, EVENTS 테이블에서 1개 컬럼 = **총 4개 컬럼**으로 충분하다.

| 컬럼 | 원천 테이블 | 역할 |
|------|-----------|------|
| `fin_sentiment` | GKG (GCAM c6.5-c6.4) / wc | 레짐 경계선 핵심 신호 |
| `fin_uncertainty` | GKG (GCAM c6.6) / wc | 레짐 전환 선행 신호 (단, VIX와 중복 가능) |
| `article_count` | GKG COUNT(*) | 시장 주목도 — 현재 데이터셋에 없는 정보 |
| `min_shock` | EVENTS GoldsteinScale MIN | Bear / Crisis 구분 보완 |

> **fin_uncertainty 주의**: 이미 ^VIX, ^VIX3M 등 4개 변동성 지표를 보유 중.  
> HMM 투입 피처 결정 시 VIF 검증 후 중복이면 제거 검토.

---

## 1. GDELT 테이블 역할 분담

```
GDELT 데이터셋
├── GKG      : 기사의 감성 / 불확실성 / 기사량  ← 메인
├── EVENTS   : 사건의 충격 강도                ← 보조 (min_shock만)
└── MENTIONS : 기사 간 감성 분산               ← 미사용 (Ω 추정용이지 레짐 판별용 아님)
```

---

## 2. GCAM 코드 정의 (공식 문서 기준)

> 출처: GDELT 공식 GCAM Master Codebook  
> `http://data.gdeltproject.org/documentation/GCAM-MASTER-CODEBOOK.TXT`

### 2-1. 사용하는 코드: c6 — Loughran and McDonald Financial Sentiment Dictionaries

> 출처 논문: Loughran, T. and McDonald, B. (2011).  
> *"When is a Liability not a Liability?"* Journal of Finance, V66, pp. 35–65.

| 코드 | 카테고리 | 설명 | 사용 여부 |
|------|---------|------|---------|
| `c6.1` | Litigious | 소송·법적 분쟁 언어 ("lawsuit", "regulatory") | 미사용 |
| `c6.2` | ModalStrong | 강한 단정 표현 ("will", "must") | 미사용 |
| `c6.3` | ModalWeak | 약한 추측 표현 ("may", "might") | 미사용 |
| `c6.4` | **Negative** | 금융 부정어 ("loss", "default", "liability") | **사용** |
| `c6.5` | **Positive** | 금융 긍정어 ("profit", "growth", "gain") | **사용** |
| `c6.6` | **Uncertainty** | 불확실성어 ("uncertain", "unclear", "risk") | **사용** |

**왜 c6(Loughran-McDonald)인가:**  
일반 감성 사전은 금융 텍스트에서 오류를 냅니다.

| 단어 | 일반 사전 | 금융 맥락 실제 의미 |
|------|---------|-------------------|
| "liability" | 중립 | 부채 → 부정 |
| "loss" | 부정 | 손실 → 강하게 부정 |
| "capital" | 긍정 | 맥락에 따라 중립 |
| "risk" | 부정 | 리스크 관리 맥락에선 중립 |

LM 사전은 SEC 공시, 애널리스트 보고서, 재무제표를 기반으로 만들어져 이 문제가 없다.  
**FinBERT의 금융 도메인 특화를 사전 기반으로 대체**하는 것이 핵심.

### 2-2. 사용하지 않는 코드

| 코드 | 정의 (공식 문서 확인) | 미사용 이유 |
|------|-------------------|-----------|
| `c15` | WordNet Affect 1.1 (감정 분류: abashment, admiration 등) | 심리학적 감정 범주 — 금융 도메인 무관 |
| 기타 (c1, c2, c3, c5, c9 등) | 공식 코드북 확인 필요 | 정의 미확인 상태에서 사용 보류 |

> **주의**: 이전 버전 문서의 `c15(Henry Financial)`, `c6(ANEW)` 표기는 오류.  
> 공식 문서 기준: c6 = Loughran-McDonald, c15 = WordNet Affect.

---

## 3. GCAM 값 계산 방식

### 3-1. GCAM 컬럼 형식

실제 GKG 행에서 GCAM 값 예시:
```
wc:104,c12.1:9,c12.10:10,...,c6.1:1,c6.4:2,...
```

- `wc:104` = 총 단어 수 104
- `c6.4:2` = LM Negative 단어 2개
- `c6.5` 없음 = LM Positive 단어 0개

### 3-2. wc 정규화가 필요한 이유

GCAM 값은 **절대 단어 카운트**이므로 기사 길이에 따라 달라집니다.

```
10단어 기사:   c6.4 = 2  →  비율 = 2/10  = 20% (매우 부정적)
1000단어 기사: c6.4 = 2  →  비율 = 2/1000 = 0.2% (거의 중립)
```

→ 반드시 `wc`로 나눠서 비율(0~1)로 표준화해야 기사 간 공정한 비교 가능.

### 3-3. 파생 피처 계산

```
fin_pos_rate    = c6.5 / wc
fin_neg_rate    = c6.4 / wc
fin_sentiment   = (c6.5 - c6.4) / wc   = fin_pos_rate - fin_neg_rate
fin_uncertainty = c6.6 / wc
```

---

## 4. 컬럼 선택 근거 (제거된 컬럼 포함)

| 컬럼 | 판단 | 이유 |
|------|------|------|
| `fin_sentiment` | **유지** | 레짐 경계선 핵심 신호 |
| `fin_uncertainty` | **유지 (검토 필요)** | VIX와 중복 가능, HMM 투입 전 VIF 확인 |
| `article_count` | **유지** | 시장 주목도 — 기존 데이터에 없는 정보 |
| `min_shock` | **유지** | Bear/Crisis 구분 유일한 신호 |
| `avg_tone` | **제거** | V2Tone 일반 사전 기반, fin_sentiment와 구조 동일하며 금융 정확도 낮음 |
| `fin_pos_rate` | **제거** | fin_sentiment = fin_pos_rate - fin_neg_rate, 중복 |
| `fin_neg_rate` | **제거** | 동일 이유 |
| `avg_shock` | **제거** | min_shock과 역할 유사, min_shock이 Crisis 감지에 더 적합 |

---

## 5. BigQuery 쿼리

```python
START_GKG = 20160101000000   # GKG DATE 형식: YYYYMMDDHHMMSS (INT64)
END_GKG   = 20251231235959
START_INT = 20160101          # EVENTS SQLDATE 형식: YYYYMMDD (INT64)
END_INT   = 20251231

# ── Query 1: GKG ──────────────────────────────────────────────
GKG_QUERY = f"""
SELECT
    PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,

    AVG(SAFE_DIVIDE(
        COALESCE(CAST(REGEXP_EXTRACT(GCAM, r'c6\\.5:([0-9]+)') AS FLOAT64), 0) -
        COALESCE(CAST(REGEXP_EXTRACT(GCAM, r'c6\\.4:([0-9]+)') AS FLOAT64), 0),
        COALESCE(CAST(REGEXP_EXTRACT(GCAM, r'wc:([0-9]+)')     AS FLOAT64), 1)
    )) AS fin_sentiment,

    AVG(SAFE_DIVIDE(
        COALESCE(CAST(REGEXP_EXTRACT(GCAM, r'c6\\.6:([0-9]+)') AS FLOAT64), 0),
        COALESCE(CAST(REGEXP_EXTRACT(GCAM, r'wc:([0-9]+)')     AS FLOAT64), 1)
    )) AS fin_uncertainty,

    COUNT(*) AS article_count

FROM `gdelt-bq.gdeltv2.gkg`
WHERE DATE BETWEEN {START_GKG} AND {END_GKG}
  AND (
        Themes LIKE '%ECON_STOCKMARKET%'
     OR Themes LIKE '%ECON_FRBRESERVE%'
     OR Themes LIKE '%ECON_INFLATION%'
     OR Themes LIKE '%ECON_UNEMPLOYMENT%'
  )
  AND SourceCommonName IN (
        'reuters.com', 'wsj.com', 'ft.com',
        'bloomberg.com', 'cnbc.com'
  )
GROUP BY date
ORDER BY date
"""

# ── Query 2: EVENTS ───────────────────────────────────────────
EVENTS_QUERY = f"""
SELECT
    PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
    MIN(GoldsteinScale) AS min_shock

FROM `gdelt-bq.gdeltv2.events`
WHERE SQLDATE BETWEEN {START_INT} AND {END_INT}
  AND Actor1CountryCode = 'USA'
  AND IsRootEvent = 1
GROUP BY date
ORDER BY date
"""
```

**DATE 형식 주의:**
- GKG `DATE`: `20250414234500` (INT64, YYYYMMDDHHMMSS) → `SUBSTR(..., 1, 8)`로 날짜만 추출
- EVENTS `SQLDATE`: `20250414` (INT64, YYYYMMDD) → 직접 `CAST AS STRING`

---

## 6. Python 수집 코드

```python
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

def fetch_gkg() -> pd.DataFrame:
    df = client.query(GKG_QUERY).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()

def fetch_events() -> pd.DataFrame:
    df = client.query(EVENTS_QUERY).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()

# 수집 및 병합
gkg_df    = fetch_gkg()
events_df = fetch_events()
gdelt_raw = gkg_df.join(events_df, how='outer')

# NYSE 영업일 정렬 (Step1의 nyse_dates 재사용)
nyse_dates  = pd.bdate_range(start='2016-01-01', end='2025-12-31', freq='B')
gdelt_daily = gdelt_raw.reindex(
    pd.date_range(start='2016-01-01', end='2025-12-31', freq='D')
).ffill().bfill()

df_gdelt = gdelt_daily.reindex(nyse_dates)
df_gdelt.index.name = 'Date'

# 저장
df_gdelt.to_csv('data/gdelt_data.csv')
```

---

## 7. 패널 데이터 구조에서의 위치

GDELT 피처는 **날짜 레벨 신호**이므로 같은 날의 모든 티커에 동일한 값이 브로드캐스트됩니다.  
VIX, HY spread 등 매크로 피처와 동일한 방식입니다.

```
date       | ticker | ret_1m | vol_21d | VIX  | fin_sentiment | article_count | min_shock
2020-02-26 | SPY    | -0.031 |  0.22   | 47.3 |    -0.041     |     621       |   -7.8
2020-02-26 | QQQ    | -0.038 |  0.25   | 47.3 |    -0.041     |     621       |   -7.8   ← 동일
2020-02-26 | TLT    | +0.018 |  0.12   | 47.3 |    -0.041     |     621       |   -7.8   ← 동일
```

```python
# Step2에서 패널에 합치는 방법
panel_df = panel_df.join(df_gdelt, on='date')
```

---

## 8. HMM 레짐 판별 설계

### 8-1. HMM이 적절한 이유

레짐(Bull/Bear/Crisis)은 직접 관측할 수 없는 **숨겨진 상태(hidden state)**이고,  
VIX, fin_sentiment 등은 그 상태가 만들어내는 **관측값(observation)**입니다.  
이 구조가 HMM의 가정과 정확히 일치합니다.

```
숨겨진 레짐:  Bull ──→ Neutral ──→ Bear ──→ Crisis
                ↓          ↓          ↓         ↓
관측값:     (VIX낮음,  (VIX보통,  (VIX높음,  (VIX폭등,
             감성긍정)  감성중립)   감성부정)   감성급락)
```

### 8-2. 입력 데이터 형식

HMM(GaussianHMM)은 `(T, N_features)` shape의 numpy 배열을 받습니다.

**투입 피처 후보:**
```python
features = [
    'VIX',             # 현재 공포 수준
    'VIX_contango',    # VIX 기간구조
    'HY_spread',       # 신용 위험
    'yield_curve',     # 경기선행 신호
    'fin_sentiment',   # 뉴스 감성 (GDELT)
    'article_count',   # 시장 주목도 (GDELT)
    'min_shock',       # 극단 충격 (GDELT)
]
```

**z-score 표준화 필수** (피처 스케일 차이가 크기 때문):

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df[features].values)
# shape: (2609, 7)
```

표준화 전후 예시:
```
             VIX    HY_sp  fin_sent  art_cnt  min_shock
2020-02-24   25.0   3.5   -0.015    312      -3.2
2020-02-27   49.5   5.8   -0.052    703      -8.9

             VIX    HY_sp  fin_sent  art_cnt  min_shock
2020-02-24   0.8    0.1   -0.5       0.6     -0.8
2020-02-27   3.6    2.0   -4.0       4.5     -5.0
```

### 8-3. 출력 데이터 형식

**① 상태 시퀀스 (정수 배열):**
```python
state_seq = model.predict(X)
# shape: (2609,)
# 예: [0, 0, 1, 1, 2, 0, ...]
# 숫자는 의미 없음 → 학습 후 수동 레이블링 필요
```

레이블링 방법:
```python
for k in range(n_states):
    mask = (state_seq == k)
    print(f"State {k}: VIX={df.loc[mask,'VIX'].mean():.1f}, "
          f"fin_sentiment={df.loc[mask,'fin_sentiment'].mean():.4f}")
# State 0: VIX=14.2, fin_sentiment=+0.003  → Bull
# State 1: VIX=22.1, fin_sentiment=-0.008  → Bear
# State 2: VIX=38.5, fin_sentiment=-0.031  → Crisis
```

**② 상태별 확률 (연속값):**
```python
state_probs = model.predict_proba(X)
# shape: (2609, K)

# K=3 예시
Date        P(Bull)  P(Bear)  P(Crisis)
2020-02-24   0.55     0.38      0.07
2020-02-26   0.04     0.21      0.75
2020-02-27   0.02     0.08      0.90
```

**③ hmm_crisis_prob (패널에 추가되는 피처):**
```python
crisis_state_idx = 2  # 수동 레이블링으로 확인
df['hmm_crisis_prob'] = state_probs[:, crisis_state_idx]
# 0~1 연속값 → XGBoost 입력 피처로 사용
```

### 8-4. Look-ahead Bias 방지

```
잘못된 방법: 전체 2609일로 HMM 1회 학습 → 과거 예측에 미래 정보 반영
올바른 방법: 롤링 윈도우 IS 구간(150일)마다 HMM 재학습
            → OOS 30일에 predict_proba 적용
```

---

## 9. 참고 자료

| 자료 | 링크 |
|------|------|
| GCAM Master Codebook | http://data.gdeltproject.org/documentation/GCAM-MASTER-CODEBOOK.TXT |
| GKG 2.1 Codebook | http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf |
| EVENTS Codebook | http://data.gdeltproject.org/documentation/GDELT-Data_Format_Codebook.pdf |
| LM 논문 (c6 출처) | Loughran & McDonald (2011), *When is a Liability not a Liability?*, Journal of Finance |
| WordNet Affect (c15 출처) | Strapparava & Valitutti (2004), LREC 2004 |
