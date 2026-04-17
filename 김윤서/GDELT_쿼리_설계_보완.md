# GDELT 쿼리 설계 보완 문서

> 프로젝트: 성향별 액티브 ETF 펀드 상품 구축  
> 작성자: 김윤서  
> 작성일: 2026.04.17  
> 목적: GDELT_레짐판별_설계.md 작성 이후 추가 검토된 내용 정리  
> 선행 문서: `GDELT_레짐판별_설계.md`, `decision_log_jaecheon.md`

---

## 목차

1. [EVENTS 테이블 컬럼 설명](#1-events-테이블-컬럼-설명)
2. [SQLDATE vs DATEADDED — 날짜 컬럼 선택 근거](#2-sqldate-vs-dateadded--날짜-컬럼-선택-근거)
3. [WHERE 절 필터 비판적 검토](#3-where-절-필터-비판적-검토)
4. [최종 개선 쿼리](#4-최종-개선-쿼리)
5. [검토 결과 요약](#5-검토-결과-요약)

---

## 1. EVENTS 테이블 컬럼 설명

### 1-1. GoldsteinScale — 컬럼의 본질

GDELT EVENTS 테이블은 전 세계 뉴스 기반 사건을 하루 수십만 건씩 저장한다.  
각 사건에는 **CAMEO(Conflict and Mediation Event Observations) 코드**가 부여되고,  
이 코드에 따라 GoldsteinScale이 자동 결정된다.

```
범위: -10.0 ~ +10.0

-10.0  ──── 가장 적대적 (전쟁, 대량학살, 핵 공격)
 -7.0  ──── 강한 충격 (경제 제재, 시위 강제 진압)
 -5.0  ──── 중간 부정 (비난, 협박, 경제 압박)
  0.0  ──── 중립 (단순 사실 보도)
 +3.0  ──── 중간 긍정 (협상, 협력 합의)
 +7.0  ──── 강한 긍정 (평화 협정, 원조 제공)
+10.0  ──── 가장 우호적
```

금융 레짐 판별에서 중요한 것은 **음수 방향의 극단값**이다.  
Bear/Crisis 레짐에서는 미국 관련 극단적 충격 사건이 집중 발생하기 때문이다.

---

### 1-2. min_shock vs avg_shock — 제거 이유

| | `min_shock` | `avg_shock` |
|--|------------|------------|
| 계산 | 당일 GoldsteinScale **최솟값** | 당일 GoldsteinScale **평균** |
| 예시 (2020-03-16) | -9.0 | -1.3 |
| 특징 | **극단 충격 하나**를 포착 | 수천 건 평균 → 희석 |
| 레짐 신호 | Crisis 구간에서 뚜렷한 음수 스파이크 | 항상 중립 근처, 레짐 간 차이 작음 |

2020년 COVID 충격처럼 레짐 전환은 극단적 사건 **하나**에서 촉발된다.  
평균을 내면 그날 발생한 수천 건의 평범한 사건에 묻혀버린다.  
→ `avg_shock` 제거, `min_shock` 유지.

---

### 1-3. 쿼리 결과 형식 예시

```
date        min_shock   event_count
──────────────────────────────────
2016-01-04   -4.0        1,823
2016-01-05   -3.8        1,654
...
2020-02-24   -6.2        2,104
2020-02-27   -7.8        3,891      ← 급격히 악화
2020-03-02   -8.7        5,203      ← 위기 레짐 진입 신호
2020-03-16   -9.0        6,741      ← COVID 최고조
2020-03-17   -8.3        5,983
...
2022-01-24   -5.5        2,876      ← 러시아-우크라이나 긴장
2022-02-24   -8.9        7,102      ← 전쟁 발발
...
2025-12-31   -3.1        1,901
```

---

### 1-4. GKG와 병합 후 GDELT 피처 전체 구조

```python
gdelt_raw = gkg_df.join(events_df, how='outer')
```

```
date        fin_sentiment   article_count   min_shock
────────────────────────────────────────────────────
2016-01-04    +0.0021           312           -4.0
...
2020-02-27    -0.0523           703           -7.8   ← 세 신호 동시 악화
2020-03-16    -0.0891           984           -9.0   ← 위기 레짐 확정
...
```

| 피처 | 정상 레짐 범위 | 위기 레짐 범위 |
|------|-------------|-------------|
| `fin_sentiment` | -0.005 ~ +0.005 | -0.05 ~ -0.10 |
| `article_count` | 200 ~ 400건 | 600 ~ 1,000건+ |
| `min_shock` | -3.0 ~ -5.0 | -7.0 ~ -10.0 |

세 피처가 **동시에 극단값**을 보일 때 HMM이 위기 레짐(regime=2)으로 전환한다.  
`covariance_type='full'`을 선택한 이유가 바로 이 동시성 패턴을 포착하기 위해서다.

---

## 2. SQLDATE vs DATEADDED — 날짜 컬럼 선택 근거

### 2-1. 두 컬럼의 정의

| 컬럼 | 정의 |
|------|------|
| `SQLDATE` | 사건이 발생한 날짜 (기사 본문에서 추출) |
| `DATEADDED` | 기사가 GDELT 데이터베이스에 수집된 날짜 |

### 2-2. DATEADDED를 쓰면 발생하는 문제

DATEADDED 기준으로 집계하면 min_shock이 시장 반응보다 **1~2일 늦게** 나타난다.

```
사건 발생:     2020-02-24  (SQLDATE)
기사 수집:     2020-02-26  (DATEADDED)

DATEADDED 기준 집계 결과:
date       VIX     min_shock
2020-02-24  35.0     없음      ← VIX는 폭등인데 충격 신호 없음
2020-02-25  38.0     없음
2020-02-26  40.0     -8.7     ← 2일 뒤에야 충격 신호 등장
```

HMM 관측 벡터가 날짜별로 어긋나게 된다.

### 2-3. SQLDATE를 써야 하는 근거

VIX, HY_spread 등 시장 지표는 **사건 발생 당일**에 즉시 반응한다.  
SQLDATE를 쓰면 min_shock이 VIX 스파이크와 같은 날짜에 정렬된다.

```
SQLDATE 기준 집계 결과:
date       VIX     min_shock
2020-02-24  35.0     -8.7     ← VIX 폭등과 충격 신호가 같은 날 정렬
2020-02-25  38.0     -7.2
2020-02-26  40.0     -6.8
```

HMM의 `covariance_type='full'`이 포착해야 할 **위기 시 피처 간 동시 급등 패턴**이  
제대로 형성되려면 모든 피처가 같은 날짜 기준으로 정렬되어야 한다.

| | SQLDATE | DATEADDED |
|--|---------|-----------|
| 기준 | 사건 발생일 | 기사 수집일 |
| VIX와 정렬 | 사건 당일 정렬 ✅ | 1~2일 후 ❌ |
| HMM 공분산 학습 | 위기 피처 동시성 포착 ✅ | 시차로 인해 희석 ❌ |

**결론: SQLDATE 채택 확정.**

---

## 3. WHERE 절 필터 비판적 검토

### 3-1. GKG — Themes 필터

**현재:**
```sql
AND (
    Themes LIKE '%ECON_STOCKMARKET%'
 OR Themes LIKE '%ECON_FRBRESERVE%'
 OR Themes LIKE '%ECON_INFLATION%'
 OR Themes LIKE '%ECON_UNEMPLOYMENT%'
)
```

**각 테마 선택 근거:**

| 테마 | 선택 이유 |
|------|---------|
| `ECON_STOCKMARKET` | 주식시장 직접 언급 → fin_sentiment의 핵심 타겟 |
| `ECON_FRBRESERVE` | 연준 통화정책 → 금리 기반 레짐 전환의 선행 신호 |
| `ECON_INFLATION` | 인플레이션 → 2022년 금리인상 레짐의 핵심 지표 |
| `ECON_UNEMPLOYMENT` | 실업 → 경기침체 레짐 전환 신호 |

**문제점 — 누락된 중요 테마:**

```
ECON_TRADE       — 무역 분쟁 (2018 미중 관세전쟁 레짐에 영향)
ECON_BANKING     — 은행 위기 (2023 SVB 사태: 현재 쿼리로 대부분 누락)
ECON_RECESSION   — 경기침체 직접 언급 기사
```

특히 2023 SVB 사태는 `ECON_BANKING` 테마가 핵심인데, 현재 4개 테마에 포함되지 않아  
해당 시기 fin_sentiment가 위기 신호를 제때 포착하지 못할 수 있다.

**개선안:**
```sql
AND (
    Themes LIKE '%ECON_STOCKMARKET%'
 OR Themes LIKE '%ECON_FRBRESERVE%'
 OR Themes LIKE '%ECON_INFLATION%'
 OR Themes LIKE '%ECON_UNEMPLOYMENT%'
 OR Themes LIKE '%ECON_TRADE%'       -- 추가
 OR Themes LIKE '%ECON_BANKING%'     -- 추가
 OR Themes LIKE '%ECON_RECESSION%'   -- 추가
)
```

---

### 3-2. GKG — SourceCommonName 필터

**현재:**
```sql
AND SourceCommonName IN (
    'reuters.com', 'wsj.com', 'ft.com',
    'bloomberg.com', 'cnbc.com'
)
```

**선택 근거:**
- 5개 모두 글로벌 금융 전문 매체
- 일반 매체 대비 금융 용어 밀도가 높아 LM 사전 적용 시 정확도 높음
- 스포츠·연예 등 노이즈 기사 자동 배제

**문제점 ①: 정확한 SourceCommonName 값 미검증**

GDELT에서 Bloomberg가 `bloomberg.com`이 아니라 `bloomberg.com/news`,  
`www.bloomberg.com` 등으로 저장되어 있으면 **매칭이 0건**이 된다.

사전 확인 쿼리:
```sql
SELECT SourceCommonName, COUNT(*) AS cnt
FROM `gdelt-bq.gdeltv2.gkg`
WHERE DATE BETWEEN 20200101000000 AND 20200201000000
  AND SourceCommonName LIKE '%bloomberg%'
GROUP BY SourceCommonName
ORDER BY cnt DESC
LIMIT 20
```

**문제점 ②: Bloomberg 페이월로 인한 수집량 부족 가능성**

Bloomberg는 페이월 기사가 많아 GDELT 수집량이 다른 매체 대비 적을 수 있다.

**개선안 — LIKE로 변경:**
```sql
AND (
    SourceCommonName LIKE '%reuters%'
 OR SourceCommonName LIKE '%wsj%'
 OR SourceCommonName LIKE '%ft.com%'
 OR SourceCommonName LIKE '%bloomberg%'
 OR SourceCommonName LIKE '%cnbc%'
)
```

---

### 3-3. GKG — Themes AND Source 조합 문제

현재 구조: **(테마 조건) AND (언론사 조건)** → 둘 다 만족해야 포함.

```
WSJ의 ECON_STOCKMARKET 기사  → 포함 ✅
WSJ의 ECON_BANKING 기사      → 제외 ❌ (테마 미포함)
AP통신의 ECON_STOCKMARKET 기사 → 제외 ❌ (언론사 미포함)
```

2023 SVB 사태 당시 AP통신·CNBC의 ECON_BANKING 보도가 대량 누락된다.  
테마 확장 + 언론사 LIKE 변경으로 이중으로 보완해야 한다.

---

### 3-4. EVENTS — Actor1CountryCode = 'USA'

**현재:**
```sql
AND Actor1CountryCode = 'USA'
```

**선택 근거:** 미국 시장 레짐 판별이 목적이므로 미국이 주체인 사건만 수집

**문제점: 미국이 대상(Actor2)인 사건 전부 누락**

```
"중국이 미국에 제재 부과"    → Actor1=CHN, Actor2=USA → 현재 쿼리에서 제외
"OPEC이 미국 압박"           → Actor1=OPEC              → 제외
"러시아가 미국 기업 제재"     → Actor1=RUS, Actor2=USA  → 제외
```

미국 시장에 충격을 줬던 2020 COVID 초기(Actor1=CHN), 2011 유럽 재정위기도 제외된다.

**개선안 — ActionGeo_CountryCode 사용:**

```sql
AND ActionGeo_CountryCode = 'US'
```

`ActionGeo_CountryCode`는 사건이 발생한 **지역** 기준이다.  
미국에서 발생한 사건을 포착하는 데 `Actor1CountryCode`보다 더 직접적이고,  
미국이 대상이거나 미국 내에서 발생한 사건을 모두 포함한다.

---

### 3-5. EVENTS — IsRootEvent = 1

**선택 근거:** 같은 사건이 여러 기사에서 언급될 때 원본 1건만 집계해서 중복 방지

**검토:**

`MIN(GoldsteinScale)` 집계에서 중복 사건이 있어도 결과값은 동일하다.  
IsRootEvent 필터는 `event_count` 집계에는 의미 있으나 `min_shock`에는 불필요하다.  
단, event_count를 별도로 활용하지 않으므로 유지해도 무방하다.

---

## 4. 최종 개선 쿼리

```python
START_GKG = 20160101000000
END_GKG   = 20251231235959
START_INT = 20160101
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

    COUNT(*) AS article_count

FROM `gdelt-bq.gdeltv2.gkg`
WHERE DATE BETWEEN {START_GKG} AND {END_GKG}
  AND (
        Themes LIKE '%ECON_STOCKMARKET%'
     OR Themes LIKE '%ECON_FRBRESERVE%'
     OR Themes LIKE '%ECON_INFLATION%'
     OR Themes LIKE '%ECON_UNEMPLOYMENT%'
     OR Themes LIKE '%ECON_TRADE%'
     OR Themes LIKE '%ECON_BANKING%'
     OR Themes LIKE '%ECON_RECESSION%'
  )
  AND (
        SourceCommonName LIKE '%reuters%'
     OR SourceCommonName LIKE '%wsj%'
     OR SourceCommonName LIKE '%ft.com%'
     OR SourceCommonName LIKE '%bloomberg%'
     OR SourceCommonName LIKE '%cnbc%'
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
  AND ActionGeo_CountryCode = 'US'   -- Actor1 → 사건 발생 지역 기준으로 변경
  AND IsRootEvent = 1
GROUP BY date
ORDER BY date
"""
```

> **실행 전 필수 확인**: SourceCommonName 실제 값 검증 쿼리를 먼저 실행하여  
> `bloomberg`, `reuters` 등의 정확한 저장 형식을 확인할 것.

---

## 5. 검토 결과 요약

| 필터 | 기존 | 문제점 | 변경 내용 |
|------|------|--------|---------|
| GKG Themes | 4개 OR | SVB 등 banking 위기 누락 | 7개로 확장 |
| GKG Source IN | 정확 매칭 | 서브도메인 불일치 가능 | LIKE로 변경 |
| Themes AND Source | 둘 다 필수 조건 | 과도하게 제한적 | 각각 확장으로 보완 |
| EVENTS Actor1='USA' | 주체 국가만 | 피대상 미국 사건 누락 | ActionGeo_CountryCode = 'US' |
| EVENTS IsRootEvent | 중복 제거 | min_shock에는 불필요 | 유지 (해가 없음) |
| EVENTS 날짜 기준 | SQLDATE | — | SQLDATE 유지 확정 (VIX와 정렬) |
