# GDELT 설계 통합 문서

> 프로젝트: 성향별 액티브 ETF 펀드 상품 구축  
> 작성자: 김윤서  
> 최초 작성: 2026.04.16  
> 최종 수정: 2026.04.17 (GDELT_레짐판별_설계.md 통합 / 전면 재정리)  
> 목적: GDELT 레짐판별 설계 + 쿼리 보완 검토 내용을 한 파일로 통합  
> 통합 전 문서: `GDELT_레짐판별_설계.md` (초기 설계, 현재 파일로 대체됨), `GDELT_쿼리_설계_보완.md` (보완 검토)

---

## 목차

1. [결론 요약](#0-결론-요약)
2. [GDELT 테이블 역할 분담](#1-gdelt-테이블-역할-분담)
3. [GCAM 코드 정의 — c15 (Loughran-McDonald)](#2-gcam-코드-정의--c15-loughran-mcdonald)
4. [GCAM 값 계산 방식](#3-gcam-값-계산-방식)
5. [컬럼 선택 근거 (제거된 컬럼 포함)](#4-컬럼-선택-근거-제거된-컬럼-포함)
6. [EVENTS 테이블 컬럼 설명](#5-events-테이블-컬럼-설명)
7. [SQLDATE vs DATEADDED — 날짜 컬럼 선택 근거](#6-sqldate-vs-dateadded--날짜-컬럼-선택-근거)
8. [WHERE 절 필터 설계 및 결정 근거](#7-where-절-필터-설계-및-결정-근거)
9. [최종 확정 쿼리](#8-최종-확정-쿼리)
10. [Python 수집 코드 (월별 캐시 분할)](#9-python-수집-코드-월별-캐시-분할)
11. [패널 데이터 구조에서의 위치](#10-패널-데이터-구조에서의-위치)
12. [HMM 레짐 판별 설계](#11-hmm-레짐-판별-설계)
13. [BigQuery 쿼터 관리 — 월별 캐시 분할 전략](#12-bigquery-쿼터-관리--월별-캐시-분할-전략)
14. [HMM 입력 피처 — TYVIX 채택](#13-hmm-입력-피처--tyvix-채택)
15. [최종 설계 결정 요약](#14-최종-설계-결정-요약)
16. [참고 자료](#15-참고-자료)

---

## 0. 결론 요약

GDELT를 **레짐 판별 전용**으로 사용한다.  
GKG 테이블에서 2개 컬럼, EVENTS 테이블에서 1개 컬럼 = **총 3개 피처**를 수집한다.

| 컬럼 | 원천 테이블 | 역할 | 사용 여부 |
|------|-----------|------|---------|
| `fin_sentiment` | GKG (GCAM c15.1 − c15.2) | 레짐 경계선 핵심 신호 | **사용** |
| `article_count` | GKG COUNT(*) | 시장 주목도 — 기존 데이터셋에 없는 정보 | **사용** |
| `min_shock` | EVENTS GoldsteinScale MIN | Bear/Crisis 구분 보완 — 수집은 하되 HMM 직접 투입 여부는 아래 참고 | **수집** |
| ~~`fin_uncertainty`~~ | ~~GKG (GCAM c15.3 계열)~~ | ~~레짐 전환 선행 신호~~ | **제거** |

> **fin_uncertainty 제거 확정**  
> 제거 이유 ①: 이미 ^VIX, ^VIX3M 등 VIX 계열 변동성 지표를 보유하고 있어 중복.  
> 제거 이유 ②: HMM 위기 레짐 추정 안정성 확보를 위해 피처 수를 p ≤ 7로 제한해야 함.

> **min_shock과 HMM 투입 관계**  
> 초기 설계에서는 fin_sentiment + article_count + min_shock 세 피처를 모두 HMM에 투입할 계획이었다.  
> 그러나 TYVIX 및 sahm_indicator 추가로 p=7 제약 하에서 min_shock이 HMM 피처 목록에서 제외되었다.  
> min_shock은 수집 및 저장하되, HMM 피처 대신 XGBoost 직접 입력 피처로 활용 가능하다.

---

## 1. GDELT 테이블 역할 분담

```
GDELT 데이터셋
├── GKG      : 기사의 감성 / 기사량        ← 메인 (fin_sentiment, article_count)
├── EVENTS   : 사건의 충격 강도            ← 보조 (min_shock 수집)
└── MENTIONS : 기사 간 감성 분산           ← 미사용 (Ω 추정용이지 레짐 판별용 아님)
```

MENTIONS 테이블을 사용하지 않는 이유:
- Ω(불확실성) 추정 목적이나 HMM의 `covariance_type='full'`이 피처 간 공분산으로 이미 커버
- 데이터 복잡도 증가 대비 레짐 판별 기여도 낮음

---

## 2. GCAM 코드 정의 — c15 (Loughran-McDonald)

> 출처: GDELT 공식 GCAM Master Codebook  
> `http://data.gdeltproject.org/documentation/GCAM-MASTER-CODEBOOK.TXT`

> **⚠️ 초기 설계 문서 오류 수정 (GDELT_레짐판별_설계.md)**  
> 초기 설계에서 `c6`을 Loughran-McDonald로 표기했으나 이는 오류였다.  
> GCAM 공식 코드북 기준:
> - `c6` = **General Inquirer (Harvard IV)** — 일반 감성 사전  
> - `c15` = **Loughran-McDonald Financial** — 금융 도메인 특화 사전 ✅  
>
> 최종 쿼리는 `c15.1 / c15.2`를 사용한다. 초기 문서의 c6 기반 분석은 모두 c15로 대체된다.

### 2-1. 사용하는 코드: c15 — Loughran and McDonald Financial Sentiment Dictionaries

> 출처 논문: Loughran, T. and McDonald, B. (2011).  
> *"When is a Liability not a Liability?"* Journal of Finance, V66, pp. 35–65.

| 코드 | 카테고리 | 설명 | 사용 여부 |
|------|---------|------|---------|
| `c15.1` | **Positive** | 금융 긍정어 ("profit", "growth", "gain") | **사용** |
| `c15.2` | **Negative** | 금융 부정어 ("loss", "default", "liability") | **사용** |
| `c15.3` 이하 | Uncertainty 등 | LM 사전 기타 하위 카테고리 | 미사용 (VIX 중복 또는 역할 불명) |

### 2-2. 왜 c15(Loughran-McDonald)인가 — c6(General Inquirer) 대비 우위

일반 감성 사전(c6)은 금융 텍스트에서 오류를 낸다.

| 단어 | 일반 사전(c6) 해석 | 금융 맥락 실제 의미 |
|------|------------------|-------------------|
| "liability" | 중립 | 부채 → 부정 |
| "loss" | 부정 | 손실 → 강하게 부정 |
| "capital" | 긍정 | 맥락에 따라 중립 |
| "risk" | 부정 | 리스크 관리 맥락에선 중립 |

LM 사전(c15)은 SEC 공시, 애널리스트 보고서, 재무제표를 기반으로 구축되어 이 문제가 없다.  
**FinBERT의 금융 도메인 특화를 사전 기반으로 대체**하는 것이 핵심.

### 2-3. 사용하지 않는 주요 코드

| 코드 | 실제 정의 | 미사용 이유 |
|------|---------|-----------|
| `c6.x` | General Inquirer (Harvard IV) — 일반 감성 | 금융 도메인 정확도 낮음, c15로 대체 |
| `c15.3+` | LM Uncertainty 등 하위 카테고리 | fin_uncertainty 제거 결정으로 미사용 |
| `c1, c2, c3` 등 | 공식 코드북 확인 필요 | 정의 미확인 상태에서 보류 |

---

## 3. GCAM 값 계산 방식

### 3-1. GCAM 컬럼 형식

실제 GKG 행에서 GCAM 값 예시:
```
wc:104,c12.1:9,c12.10:10,...,c15.1:3,c15.2:2,...
```

- `wc:104` = 총 단어 수 104
- `c15.1:3` = LM Positive 단어 3개
- `c15.2:2` = LM Negative 단어 2개

### 3-2. wc 정규화가 필요한 이유

GCAM 값은 **절대 단어 카운트**이므로 기사 길이에 따라 달라진다.

```
10단어 기사:   c15.2 = 2  →  비율 = 2/10   = 20%  (매우 부정적)
1000단어 기사: c15.2 = 2  →  비율 = 2/1000 = 0.2% (거의 중립)
```

→ 정확한 비교를 위해 `wc`로 나눠서 비율(0~1)로 표준화해야 기사 간 공정한 비교가 가능하다.

### 3-3. 파생 피처 계산 (wc 정규화 기준)

```
fin_pos_rate  = c15.1 / wc
fin_neg_rate  = c15.2 / wc
fin_sentiment = (c15.1 - c15.2) / wc   ← 이상적 계산식
```

> **현재 쿼리 적용 방식 (단순화):**  
> 최종 확정 쿼리(섹션 8)에서는 `AVG(c15.1 - c15.2)` (wc 나누기 없음)를 사용한다.  
> 하루 수백~천 개 기사의 AVG를 내는 과정에서 극단 길이 기사의 영향이 어느 정도 희석되므로  
> 실용적으로 허용 가능한 단순화다.  
> 더 정밀한 버전을 원한다면 쿼리의 AVG 블록을 아래로 교체한다:
> ```sql
> AVG(SAFE_DIVIDE(
>     COALESCE(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c15\\.1:([0-9.]+)') AS FLOAT64), 0.0)
>   - COALESCE(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c15\\.2:([0-9.]+)') AS FLOAT64), 0.0),
>     NULLIF(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'wc:([0-9]+)') AS FLOAT64), 0)
> )) AS fin_sentiment
> ```

---

## 4. 컬럼 선택 근거 (제거된 컬럼 포함)

| 컬럼 | 판단 | 이유 |
|------|------|------|
| `fin_sentiment` | **유지** | 레짐 경계선 핵심 신호 |
| `article_count` | **유지** | 시장 주목도 — 기존 데이터셋에 없는 정보 |
| `min_shock` | **수집** | Bear/Crisis 구분 신호 — 수집 후 XGBoost 피처로 활용 가능 |
| `fin_uncertainty` | **제거** | VIX 계열(^VIX, ^VIX3M 등)과 중복. p≤7 유지 필요 |
| `avg_tone` | **제거** | V2Tone 일반 사전 기반, fin_sentiment와 구조 동일하며 금융 정확도 낮음 |
| `fin_pos_rate` | **제거** | fin_sentiment = fin_pos_rate − fin_neg_rate이므로 중복 |
| `fin_neg_rate` | **제거** | 동일 이유 |
| `avg_shock` | **제거** | min_shock과 역할 유사, min_shock이 Crisis 감지에 더 적합 |

---

## 5. EVENTS 테이블 컬럼 설명

### 5-1. GoldsteinScale — 컬럼의 본질

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

### 5-2. min_shock vs avg_shock — 제거 이유

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

### 5-3. 쿼리 결과 형식 예시

```
date        min_shock
─────────────────────
2016-01-04   -4.0
2016-01-05   -3.8
...
2020-02-24   -6.2
2020-02-27   -7.8      ← 급격히 악화
2020-03-02   -8.7      ← 위기 레짐 진입 신호
2020-03-16   -9.0      ← COVID 최고조
2020-03-17   -8.3
...
2022-01-24   -5.5      ← 러시아-우크라이나 긴장
2022-02-24   -8.9      ← 전쟁 발발
...
2025-12-31   -3.1
```

---

### 5-4. GKG와 병합 후 GDELT 피처 전체 구조

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

## 6. SQLDATE vs DATEADDED — 날짜 컬럼 선택 근거

### 6-1. 두 컬럼의 정의

| 컬럼 | 정의 |
|------|------|
| `SQLDATE` | 사건이 발생한 날짜 (기사 본문에서 추출) |
| `DATEADDED` | 기사가 GDELT 데이터베이스에 수집된 날짜 |

### 6-2. DATEADDED를 쓰면 발생하는 문제

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

### 6-3. SQLDATE를 써야 하는 근거

VIX, HY_spread 등 시장 지표는 **사건 발생 당일**에 즉시 반응한다.  
SQLDATE를 쓰면 min_shock이 VIX 스파이크와 같은 날짜에 정렬된다.

```
SQLDATE 기준 집계 결과:
date       VIX     min_shock
2020-02-24  35.0     -8.7     ← VIX 폭등과 충격 신호가 같은 날 정렬
2020-02-25  38.0     -7.2
2020-02-26  40.0     -6.8
```

| | SQLDATE | DATEADDED |
|--|---------|-----------|
| 기준 | 사건 발생일 | 기사 수집일 |
| VIX와 정렬 | 사건 당일 정렬 ✅ | 1~2일 후 ❌ |
| HMM 공분산 학습 | 위기 피처 동시성 포착 ✅ | 시차로 인해 희석 ❌ |

**결론: SQLDATE 채택 확정.**

---

## 7. WHERE 절 필터 설계 및 결정 근거

### 7-1. GKG — Themes 필터

**최종 확정 (7개 테마):**
```sql
REGEXP_CONTAINS(Themes,
    r'(ECON_STOCKMARKET|ECON_FRBRESERVE|ECON_INFLATION|ECON_UNEMPLOYMENT|ECON_RECESSION|ECON_TRADE|ECON_BANKING)')
```

| 테마 | 선택 이유 |
|------|---------|
| `ECON_STOCKMARKET` | 주식시장 직접 언급 → fin_sentiment의 핵심 타겟 |
| `ECON_FRBRESERVE` | 연준 통화정책 → 금리 기반 레짐 전환의 선행 신호 |
| `ECON_INFLATION` | 인플레이션 → 2022년 금리인상 레짐의 핵심 지표 |
| `ECON_UNEMPLOYMENT` | 실업 → 경기침체 레짐 전환 신호 |
| `ECON_RECESSION` | 경기침체 직접 언급 기사 추가 |
| `ECON_TRADE` | 무역 분쟁 → 2018 미중 관세전쟁 레짐 포착 |
| `ECON_BANKING` | 은행 위기 → 2023 SVB 사태 포착 (기존 4개 테마로는 대부분 누락됨) |

**검토 과정에서 제거한 테마:**

| 제거 테마 | 제거 이유 |
|---------|---------|
| `ECON_RATECUT` | Bull 레짐 전환 신호로 유용할 수 있으나, `ECON_FRBRESERVE`와 기사 중복 비율이 높아 중복 효과 최소화 원칙 적용 |
| `ECON_DEBT` | 부채 위기 (2011 유럽 재정위기 등) — 미국 시장 중심 프로젝트에서 직접 영향 제한적 |

---

### 7-2. GKG — SourceCommonName 필터

**최종 확정:**
```sql
REGEXP_CONTAINS(SourceCommonName, r'(reuters|cnbc|wsj|ft|bloomberg)')
```

**선택 근거:**
- 5개 모두 글로벌 금융 전문 매체
- 일반 매체 대비 금융 용어 밀도가 높아 LM 사전(c15) 적용 시 정확도 높음
- 스포츠·연예 등 노이즈 기사 자동 배제
- 기존 `IN ('bloomberg.com', ...)` 방식 대신 `REGEXP_CONTAINS` 사용으로 서브도메인 불일치 방지  
  (`bloomberg.com`, `www.bloomberg.com`, `bloomberg.com/news` 등 모두 포함)

> **SourceCommonName 사전 검증 쿼리** (최초 실행 전 실제 저장 형식 확인):
> ```sql
> SELECT SourceCommonName, COUNT(*) AS cnt
> FROM `gdelt-bq.gdeltv2.gkg`
> WHERE DATE BETWEEN 20200101000000 AND 20200201000000
>   AND SourceCommonName LIKE '%bloomberg%'
> GROUP BY SourceCommonName ORDER BY cnt DESC LIMIT 20
> ```

---

### 7-3. GKG — Themes AND Source 이중 필터

현재 구조: **(테마 조건) AND (언론사 조건)** → 둘 다 만족해야 포함.

```
Reuters의 ECON_BANKING 기사  → 포함 ✅
AP통신의 ECON_BANKING 기사   → 제외 ❌ (언론사 미포함)
Reuters의 스포츠 기사        → 제외 ❌ (테마 미포함)
```

이중 필터의 효과: 금융 전문 매체의 금융 주제 기사만 선별 → fin_sentiment 노이즈 최소화.  
테마 7개 확장 + REGEXP_CONTAINS로 기존 이중 필터의 누락을 보완.

---

### 7-4. EVENTS — 지역 필터

**최종 확정: `ActionGeo_CountryCode = 'US'`**

| 필터 | 설명 | 한계 |
|------|------|------|
| `Actor1CountryCode = 'USA'` | 미국이 주체인 사건만 | 외국이 미국에 가한 충격 누락 |
| `ActionGeo_CountryCode = 'US'` ✅ | 미국 지역에서 발생한 사건 | 미국 직접 관련 포괄 |

```
"중국이 미국에 제재 부과"  → Actor1=CHN, ActionGeo=US → ActionGeo 필터에서 포함 ✅
"러시아가 미국 기업 제재"  → Actor1=RUS, ActionGeo=US → 포함 ✅
```

---

### 7-5. EVENTS — IsRootEvent 및 EventCode

| 조건 | 결정 | 이유 |
|------|------|------|
| `IsRootEvent = 1` | **유지** | MIN에 영향 없으나 중복 기사 방지, 집계 일관성 유지 |
| `EventCode LIKE '1%'` 등 경제 코드 필터 | **제거** | CAMEO 코드 체계상 경제 이벤트가 명확히 분리되지 않아 오히려 포착 범위를 좁힘 |

---

## 8. 최종 확정 쿼리

> 아래는 `Step3_GDELT_HMM_XGBoost_myun.ipynb`에 실제 반영된 코드와 동일한 버전이다.

```python
# ── Query 1: GKG ──────────────────────────────────────────────────────────────
# gkg_partitioned: _PARTITIONTIME 필터로 해당 월만 스캔 → 비용 최소화
# c15.1/c15.2: Loughran-McDonald Financial 사전 (금융 도메인 특화, c6 대체)
# REGEXP_CONTAINS: 서브도메인 불일치 방지 (bloomberg.com, www.bloomberg.com 등 포함)
GKG_QUERY = """
SELECT
    PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
    AVG(
        COALESCE(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c15\\.1:([0-9.]+)') AS FLOAT64), 0.0)
      - COALESCE(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c15\\.2:([0-9.]+)') AS FLOAT64), 0.0)
    ) AS fin_sentiment,
    COUNT(*) AS article_count
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME BETWEEN TIMESTAMP('{start_dash}') AND TIMESTAMP('{end_dash}')
    AND REGEXP_CONTAINS(Themes,
        r'(ECON_STOCKMARKET|ECON_FRBRESERVE|ECON_INFLATION|ECON_UNEMPLOYMENT|ECON_RECESSION|ECON_TRADE|ECON_BANKING)')
    AND REGEXP_CONTAINS(SourceCommonName, r'(reuters|cnbc|wsj|ft|bloomberg)')
GROUP BY date
ORDER BY date
"""

# ── Query 2: EVENTS ───────────────────────────────────────────────────────────
# events_partitioned: _PARTITIONTIME 필터 사용 (events 테이블보다 스캔 비용 저렴)
# ActionGeo_CountryCode='US': 사건 발생 지역 기준 → 외국발 미국 충격도 포착
# IsRootEvent=1: 중복 기사 방지 (MIN 결과에는 영향 없으나 유지)
# EventCode 필터 없음: CAMEO 코드 체계상 경제 이벤트가 명확히 분리되지 않아 제외
EVENTS_QUERY = """
SELECT
    PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
    MIN(GoldsteinScale) AS min_shock
FROM `gdelt-bq.gdeltv2.events_partitioned`
WHERE _PARTITIONTIME BETWEEN TIMESTAMP('{start_dash}') AND TIMESTAMP('{end_dash}')
    AND ActionGeo_CountryCode = 'US'
    AND IsRootEvent = 1
GROUP BY date
ORDER BY date
"""
```

**DATE 형식 주의:**
- GKG `DATE`: `20250414234500` (INT64, YYYYMMDDHHMMSS) → `SUBSTR(..., 1, 8)`로 날짜만 추출
- EVENTS `SQLDATE`: `20250414` (INT64, YYYYMMDD) → 직접 `CAST AS STRING`
- `_PARTITIONTIME`: `'{start_dash}'` = `'2025-12-01'` 형식 (YYYY-MM-DD)

---

## 9. Python 수집 코드 (월별 캐시 분할)

```python
from pathlib import Path
from google.cloud import bigquery
import pandas as pd
from dateutil.relativedelta import relativedelta

DATA_DIR  = Path("data")
CACHE_DIR = DATA_DIR / "gdelt_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

client = bigquery.Client()

START_DATE = pd.Timestamp("2016-01-01")
END_DATE   = pd.Timestamp("2025-12-31")

# 수집할 전체 월 목록 생성
def all_months_list(start: pd.Timestamp, end: pd.Timestamp) -> list:
    months = []
    cur = start.replace(day=1)
    while cur <= end:
        months.append(cur)
        cur += relativedelta(months=1)
    return months

# 이미 수집된 (year, month) 집합 — gkg_YYYY_MM.parquet 기준
def cached_months() -> set:
    result = set()
    for f in CACHE_DIR.glob("gkg_????_??.parquet"):
        parts = f.stem.split("_")
        result.add((int(parts[1]), int(parts[2])))
    return result

# 단일 월 수집 함수
def fetch_month(year: int, month: int) -> tuple:
    start_dash = f"{year}-{month:02d}-01"
    end_dt     = (pd.Timestamp(f"{year}-{month:02d}-01") + relativedelta(months=1) - pd.Timedelta(days=1))
    end_dash   = end_dt.strftime("%Y-%m-%d")

    gkg_df    = client.query(GKG_QUERY.format(start_dash=start_dash, end_dash=end_dash)).to_dataframe()
    events_df = client.query(EVENTS_QUERY.format(start_dash=start_dash, end_dash=end_dash)).to_dataframe()

    for df in [gkg_df, events_df]:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

    return gkg_df, events_df

# 전체 수집 루프 (최신→과거 역순)
all_months = all_months_list(START_DATE, END_DATE)
done       = cached_months()
to_collect = sorted(
    [m for m in all_months if (m.year, m.month) not in done],
    reverse=True   # 최신부터 수집 → 쿼터 소진 시 최신 데이터 확보 우선
)

for m in to_collect:
    print(f"수집 중: {m.year}-{m.month:02d} ...", end=" ")
    gkg_m, events_m = fetch_month(m.year, m.month)
    gkg_m.to_parquet(CACHE_DIR / f"gkg_{m.year}_{m.month:02d}.parquet")
    events_m.to_parquet(CACHE_DIR / f"events_{m.year}_{m.month:02d}.parquet")
    print("완료")

# 전체 캐시 로드 및 병합
def load_all_cached() -> tuple:
    gkg_parts, events_parts = [], []
    for f in sorted(CACHE_DIR.glob("gkg_????_??.parquet")):
        gkg_parts.append(pd.read_parquet(f))
    for f in sorted(CACHE_DIR.glob("events_????_??.parquet")):
        events_parts.append(pd.read_parquet(f))
    gkg_df    = pd.concat(gkg_parts).sort_index()
    events_df = pd.concat(events_parts).sort_index()
    return gkg_df, events_df

gkg_df, events_df = load_all_cached()
gdelt_raw = gkg_df.join(events_df, how='outer')

# NYSE 영업일 정렬
nyse_dates  = pd.bdate_range(start='2016-01-01', end='2025-12-31', freq='B')
gdelt_daily = gdelt_raw.reindex(
    pd.date_range(start='2016-01-01', end='2025-12-31', freq='D')
).ffill().bfill()

df_gdelt = gdelt_daily.reindex(nyse_dates)
df_gdelt.index.name = 'Date'

# 저장
df_gdelt.to_csv(DATA_DIR / 'gdelt_data.csv')
```

---

## 10. 패널 데이터 구조에서의 위치

GDELT 피처는 **날짜 레벨 신호**이므로 같은 날의 모든 티커에 동일한 값이 브로드캐스트된다.  
VIX, HY spread 등 매크로 피처와 동일한 방식이다.

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

## 11. HMM 레짐 판별 설계

### 11-1. HMM이 적절한 이유

레짐(Stable/Neutral/Crisis)은 직접 관측할 수 없는 **숨겨진 상태(hidden state)**이고,  
VIX, fin_sentiment 등은 그 상태가 만들어내는 **관측값(observation)**이다.  
이 구조가 HMM의 가정과 정확히 일치한다.

```
숨겨진 레짐:  Stable ──→ Neutral ──→ Crisis
                ↓            ↓           ↓
관측값:     (VIX낮음,   (VIX보통,   (VIX폭등,
             감성긍정)   감성중립)    감성급락)
```

### 11-2. 입력 데이터 형식

HMM(GaussianHMM)은 `(T, N_features)` shape의 numpy 배열을 받는다.

**최종 투입 피처 (p=7):**
```python
features = [
    'VIX_level',          # 현재 공포 수준
    'HY_spread',          # 신용 위험
    'yield_curve',        # 경기선행 신호
    'sahm_indicator',     # 경기침체 조기 경보
    'TYVIX',              # 채권 변동성 (CBOE 10Y 국채 옵션 내재변동성)
    'fin_sentiment',      # 뉴스 감성 (GDELT GKG c15.1/c15.2)
    'article_count_norm', # 시장 주목도 (GDELT GKG, z-score 정규화)
]
# TYVIX 수집 실패 시 p=6 (섹션 13 참고)
```

> **초기 설계 대비 변경점**  
> - `VIX_contango` 제거 → TYVIX로 대체 (채권 변동성이 2022 금리인상 레짐 식별에 더 결정적)  
> - `sahm_indicator` 추가 (FRED에서 직접 수집 가능, 경기침체 선행 신호)  
> - `min_shock` HMM 피처에서 제외 → p=7 제약으로 sahm + TYVIX 우선, min_shock은 수집 후 XGBoost 피처로 활용

**z-score 표준화 필수** (피처 스케일 차이가 크기 때문):

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df[features].values)
# shape 예시: (2509, 7)
```

표준화 전후 예시:
```
             VIX    HY_sp  fin_sent  art_cnt  TYVIX
2020-02-24   25.0   3.5   -0.015    312       45
2020-02-27   49.5   5.8   -0.052    703       89

             VIX    HY_sp  fin_sent  art_cnt  TYVIX
2020-02-24   0.8    0.1   -0.5       0.6      0.3
2020-02-27   3.6    2.0   -4.0       4.5      3.1
```

### 11-3. 출력 데이터 형식

**① 상태 시퀀스 (정수 배열):**
```python
state_seq = model.predict(X)
# shape: (T,)
# 예: [0, 0, 1, 1, 2, 0, ...]
# 숫자는 의미 없음 → 학습 후 수동 레이블링 필요
```

레이블링 방법:
```python
for k in range(n_states):
    mask = (state_seq == k)
    print(f"State {k}: VIX={df.loc[mask,'VIX_level'].mean():.1f}, "
          f"fin_sentiment={df.loc[mask,'fin_sentiment'].mean():.4f}")
# State 0: VIX=14.2, fin_sentiment=+0.003  → Stable
# State 1: VIX=22.1, fin_sentiment=-0.008  → Neutral
# State 2: VIX=38.5, fin_sentiment=-0.031  → Crisis
```

**② 상태별 확률 (연속값):**
```python
state_probs = model.predict_proba(X)
# shape: (T, K)

# K=3 예시
Date        P(Stable) P(Neutral) P(Crisis)
2020-02-24   0.55      0.38       0.07
2020-02-26   0.04      0.21       0.75
2020-02-27   0.02      0.08       0.90
```

**③ hmm_crisis_prob (패널에 추가되는 피처):**
```python
crisis_state_idx = 2  # 수동 레이블링으로 확인
df['hmm_crisis_prob'] = state_probs[:, crisis_state_idx]
# 0~1 연속값 → XGBoost 입력 피처로 사용
```

### 11-4. Look-ahead Bias 방지

```
잘못된 방법: 전체 T일로 HMM 1회 학습 → 과거 예측에 미래 정보 반영
올바른 방법: 롤링 윈도우 IS 구간(150일)마다 HMM 재학습
            → OOS 21일(= 영업일 기준 1개월)에 predict_proba 적용
```

Walk-forward 파라미터:
```
IS (In-Sample)     : 150 영업일
Embargo            : 21 영업일 (레이블 horizon과 동일, look-ahead 방지)
OOS (Out-of-Sample): 21 영업일 (영업일 기준 정확히 1개월, ~103 윈도우)
```

---

## 12. BigQuery 쿼터 관리 — 월별 캐시 분할 전략

### 12-1. 문제 배경

GKG 테이블은 월별 약 85~95 GB 스캔 → 10년(120개월) 전체 수집 시 약 10 TB 필요.  
BigQuery 무료 쿼터는 **결제 계정당 월 1 TB**.

### 12-2. 이전 방식의 한계

```python
# 기존: 단일 파일 캐시 → "전부 아니면 전무"
GKG_CACHE = DATA_DIR / "gdelt_gkg_cache.parquet"  # 있으면 통째로 로드, 없으면 전부 재수집
```

쿼터 초과로 중단되면 이미 수집된 달을 포함해 전부 재수집해야 했다.

### 12-3. 개선: 월별 캐시 분할

```
data/gdelt_cache/
    gkg_2025_12.parquet
    gkg_2025_11.parquet
    ...                   ← 수집 완료된 달
    events_2025_12.parquet
    events_2025_11.parquet
    ...
```

수집 완료된 달은 자동 스킵, 미수집 달만 최신→과거 역순으로 수집.

### 12-4. GCP 계정 전략

| 방법 | 가능 여부 | 비고 |
|------|----------|------|
| 같은 구글 계정, 새 프로젝트 | ❌ | 결제 계정 공유 → 쿼터 공유 |
| 새 구글 계정 | ✅ | 1 TB 추가 확보 가능 |
| 실행 순서 | 최신→과거 역순 | 쿼터 소진 시 중단해도 다음 실행에서 자동 이어받기 |

### 12-5. 현재 수집 현황

```
2025-12 ✅ | 2025-11 ✅ | 2025-10 ✅ | 2025-09 ✅ | 2025-08 ✅
2025-07 ✅ | 2025-06 ✅ | 2025-05 ✅ | 2025-04 ✅ | 2025-03 ✅
2025-02 ✅ | 2025-01 ⬜ (다음 실행 시작점)
2024-12 ⬜ ~ 2016-01 ⬜ (약 109개월 미수집)
```

---

## 13. HMM 입력 피처 — TYVIX 채택

### 13-1. MOVE Index 대체 배경

HMM 입력 p=7에 `MOVE_index`(ICE BofA MOVE)를 포함시키기로 했으나,  
MOVE는 ICE 독점 데이터로 **FRED API에서 제공하지 않는다**.

### 13-2. TYVIX 채택 근거

| | MOVE | TYVIX (`^TYVIX`) |
|--|------|-----------------|
| 정의 | ICE BofA 1개월 국채 옵션 내재변동성 | CBOE 10년 국채 옵션 내재변동성 |
| 상관관계 | — | MOVE와 0.85 ~ 0.90 |
| 데이터 접근 | FRED 미제공 (독점) | yfinance로 수집 가능 |
| 기간 | — | 2013년~ (프로젝트 기간 2016~2025 커버) |

### 13-3. TYVIX가 포착하는 레짐 패턴

2022년 금리인상 구간은 VIX만으로 위기 레짐 식별이 어렵다:

```
2022년 금리인상 레짐:
  VIX   : ~35  (주식 변동성, 중간 수준)
  TYVIX : ~130 (채권 변동성, 역사적 고점)
  → TYVIX 없이는 이 구간을 "중립" 레짐으로 오분류할 위험
```

`covariance_type='full'`이 포착하는 **VIX-TYVIX 동시 폭등 패턴**이  
금융위기(2020 코로나)와 금리위기(2022)를 구분하는 핵심 신호가 된다.

### 13-4. 수집 로직 (Step3 노트북)

```python
# 1순위: yfinance ^TYVIX / VXTYN / ^VXTYN
for ticker in ['^TYVIX', 'VXTYN', '^VXTYN']:
    raw = yf.download(ticker, ...)
    if len(raw) > 100:
        # 수집 성공 → tyvix_index.parquet 저장
        break

# 2순위 fallback: TLT 실현변동성 (TLT는 패널에 이미 존재)
tyvix_ts = panel.xs('TLT', level='ticker')['ret_1m'].rolling(21).std() * sqrt(252) * 100
```

| 시나리오 | HMM 피처 수 | 구성 |
|---------|------------|------|
| TYVIX 수집 성공 | p=7 | VIX + HY + yield + sahm + TYVIX + fin_sent + article_vol |
| TYVIX 실패, TLT proxy 성공 | p=7 | VIX + HY + yield + sahm + TLT_vol + fin_sent + article_vol |
| 모두 실패 | p=6 | TYVIX 제외 |

---

## 14. 최종 설계 결정 요약

| 항목 | 기존 초안 | 최종 확정 | 근거 |
|------|----------|----------|------|
| GKG 테이블 | `gkg` | **`gkg_partitioned`** + `_PARTITIONTIME` | 스캔 비용 절감 |
| 감성 사전 | `c6.4/c6.5` (General Inquirer, **초기 오류**) | **`c15.1/c15.2`** (Loughran-McDonald) | 금융 도메인 특화 |
| GKG Themes | 4개 | **7개** (ECON_RECESSION·ECON_TRADE·ECON_BANKING 추가) | 2018 관세전쟁·SVB 포착 |
| GKG Source | `IN (정확 매칭)` | **`REGEXP_CONTAINS`** | 서브도메인 불일치 방지 |
| EVENTS 테이블 | `events` | **`events_partitioned`** + `_PARTITIONTIME` | 스캔 비용 절감 |
| EVENTS 지역 필터 | `Actor1CountryCode='USA'` | **`ActionGeo_CountryCode='US'`** | 외국발 미국충격 포착 |
| EVENTS EventCode | 미지정 | **없음** | CAMEO 코드가 경제 이벤트 명확 분리 안 됨 |
| EVENTS IsRootEvent | — | **`= 1` 유지** | MIN에 영향 없으나 중복 방지 |
| EVENTS 날짜 기준 | — | **`SQLDATE` 확정** | VIX와 동일 날짜 정렬 |
| 채권 변동성 피처 | `MOVE_index` (FRED 미제공) | **`TYVIX`** | yfinance 수집 가능, MOVE 상관 0.85~0.90 |
| HMM 피처 구성 | VIX, VIX_contango, HY, yield, fin_sent, art_cnt, min_shock | **VIX, HY, yield, sahm, TYVIX, fin_sent, art_cnt** | TYVIX·sahm 추가, VIX_contango·min_shock 제외 |
| OOS 기간 | 30일 | **21일** | 영업일 기준 정확히 1개월 (~103 윈도우) |
| 데이터 수집 | 단일 캐시 | **월별 캐시 분할** | 쿼터 초과 시 이어받기 가능 |

---

## 15. 참고 자료

| 자료 | 링크 |
|------|------|
| GCAM Master Codebook | http://data.gdeltproject.org/documentation/GCAM-MASTER-CODEBOOK.TXT |
| GKG 2.1 Codebook | http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf |
| EVENTS Codebook | http://data.gdeltproject.org/documentation/GDELT-Data_Format_Codebook.pdf |
| CAMEO Codebook | http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf |
| LM 논문 (c15 출처) | Loughran & McDonald (2011), *When is a Liability not a Liability?*, Journal of Finance |
| CBOE TYVIX | CBOE 공식 사이트 (yfinance ticker: ^TYVIX / VXTYN / ^VXTYN) |
