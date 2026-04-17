# GDELT 활용 계획 및 판단 과정

> 최초 작성일: 2026-04-15  
> 최종 업데이트: 2026-04-16  
> 목적: ML + Black-Litterman 기반 포트폴리오 최적화에서 GDELT 데이터 활용 여부 및 방식에 대한 설계·판단 과정 기록

---

## 목차

1. [판단 과정 요약](#1-판단-과정-요약)
2. [실 데이터 테스트 결과](#2-실-데이터-테스트-결과)
3. [매핑 개선 시도 (v1 → v2)](#3-매핑-개선-시도)
4. [외부 키워드와의 비교](#4-외부-키워드와의-비교)
5. [근본적 한계 재검토](#5-근본적-한계-재검토)
6. [미결 의사결정](#6-미결-의사결정)
7. [GDELT 데이터 구조 개요](#7-gdelt-데이터-구조-개요)
8. [초기 매핑 설계](#8-초기-매핑-설계)
9. [집계 후 데이터 구조](#9-집계-후-데이터-구조)
10. [결측 처리 전략](#10-결측-처리-전략)
11. [ML → Black-Litterman 연결](#11-ml--black-litterman-연결)
12. [Ablation 축](#12-ablation-축)

---

## 1. 판단 과정 요약

### 1단계 — GDELT 활용 전제로 설계 시작

BL 모델의 Q 벡터(30차원 기대수익률 뷰)를 ML로 생성하기 위해,
GDELT GKG의 뉴스 감성 데이터를 종목별 피처로 사용하려 했음.

- (date, ticker) 패널 구조: `tone_avg`, `event_cnt`, `tone_std` 파생
- 3가지 매핑 타입 설계: `org` / `keyword` / `org_themed`

### 2단계 — 실 데이터 테스트 → 첫 번째 문제 발견

2025-04-14 하루치 전체 수집(126,375행) 결과: **14/30 종목(47%)만 커버**

- 채권 ETF 4종 전부 0건, 섹터 ETF 다수 0건, GLD 0건
- 원인: `ECON_INTEREST_RATES`, `ECON_GOLD`, `COMMODITY` 등 태그가 **실제 GKG에 존재하지 않음**

### 3단계 — v2 매핑으로 부분 개선 시도

- 채권 ETF: `ECON_INTEREST_RATES` → `EPU_ECONOMY` (동작 확인 태그)
- 섹터 ETF 미동작 테마 → `ECON_STOCKMARKET`으로 일괄 교체
- GDELT_Test.ipynb Step 6에 v1 vs v2 커버리지 비교 셀 추가

### 4단계 — 외부 키워드 매핑과 비교

`ECON_FRBRESERVE`(연준), `ECON_INFLATION` 등 더 정밀한 GDELT 태그 발견.  
단, 구조적으로 AND 로직 없음 → 노이즈 제어 부재.

### 5단계 — 근본적 신뢰도 질문 제기 → 한계 확인

**"개별주 매핑도 완벽한가?"** → 아니다. 3가지 구조적 한계 확인.

---

## 2. 실 데이터 테스트 결과

### 수집 환경

| 항목 | 값 |
|------|---|
| 테스트 날짜 | 2025-04-14 |
| 수집 모드 | coverage=True (하루치 전체 파일) |
| 수집 행 수 | 126,375행 |
| 실행 환경 | finance_project/.venv (Python 3.12.4) |
| 결과 파일 | `김재천/gdelt_test_result.txt` |

### 커버리지 결과

```
커버됨:  GOOGL(1,872건), QQQ(1,295건), MSFT(615건), JPM(419건),
         AAPL(254건), XOM(88건), AMZN(14건), XLK(13건),
         SPY(5건), EFA(4건), XLE(4건), EEM(2건), IWM(1건), XLF(1건)

0건:     TLT, AGG, SHY, TIP (채권 ETF 전체)
         GLD, DBC (대안 ETF)
         XLV, VOX, XLY, XLP, XLI, XLU, XLRE, XLB (섹터 ETF 다수)
         JNJ, PG (개별주인데도 0건)
```

### 0건 원인 진단

| 원인 | 해당 종목 |
|------|---------|
| `ECON_INTEREST_RATES` 태그 GKG 미존재 | TLT, AGG, SHY, TIP |
| `ECON_GOLD`, `COMMODITY` 태그 GKG 미존재 | GLD, DBC, XLB |
| `HEALTH`, `ECON_RETAIL`, `ECON_TRADE`, `ECON_REALESTATE` 미존재 | XLV, XLY, XLP, XLI, XLRE |
| `&` 기호 인코딩 누락 가능성 | JNJ, PG |

### 동작 확인된 GDELT GKG 태그

| 태그 | 확인 종목 |
|------|---------|
| `ECON_STOCKMARKET` | QQQ, SPY, XLF |
| `TECHNOLOGY` | QQQ, XLK |
| `ENV_OIL`, `ENERGY` | XLE |
| `EPU_ECONOMY`, `EPU_ECONOMY_HISTORIC` | AMZN, AAPL 기사 내 등장 |
| `EPU_UNCERTAINTY` | AAPL 기사 내 등장 |
| `WB_713_PUBLIC_FINANCE` | AMZN 기사 내 등장 |

---

## 3. 매핑 개선 시도

### v1 → v2 핵심 변경 사항

| 대상 | 기존 테마 (v1) | 수정 테마 (v2) | 수정 이유 |
|------|--------------|--------------|---------|
| TLT/AGG/SHY/TIP | `ECON_INTEREST_RATES` | `EPU_ECONOMY` | 미존재 확인 |
| GLD | `ECON_GOLD` + `COMMODITY` | `ECON_STOCKMARKET` | 두 태그 모두 미동작 |
| DBC | `COMMODITY` | `ECON_STOCKMARKET` | 미동작 확인 |
| XLV | `HEALTH` | `ECON_STOCKMARKET` | 미동작 확인 |
| XLY/XLP | `ECON_RETAIL` | `EPU_ECONOMY` | 미동작 확인 |
| XLI | `ECON_TRADE` | `ECON_STOCKMARKET` | 미동작 확인 |
| XLRE | `ECON_REALESTATE` | `ECON_STOCKMARKET` | 미동작 확인 |
| XLB | `COMMODITY` | `ECON_STOCKMARKET` | 미동작 확인 |
| JNJ/PG | `& JOHNSON` / `& GAMBLE` | `JOHNSON JOHNSON` 등 추가 | `&` 인코딩 누락 대비 |
| AMZN themes | `ECON_RETAIL` 포함 | `EPU_ECONOMY`로 교체 | ECON_RETAIL 미동작 |

### v2 매핑 구현 위치

`김재천/GDELT_Test.ipynb` — cell-25 (`TICKER_TO_GDELT_V2`)

---

## 4. 외부 키워드와의 비교

### 비교 대상: TICKER_KEYWORDS (외부 제안)

```python
# 구조: 단순 리스트, OR 로직, 테마코드 + 자연어 혼재
'TLT': ['treasury bonds', 'long-term bonds', 'ECON_FRBRESERVE', 'yield curve', '10-year treasury']
'TIP': ['ECON_INFLATION', 'TIPS', 'CPI', 'inflation expectations', 'real yield']
'JNJ': ['Johnson Johnson', 'JNJ', 'Kenvue', 'pharmaceutical giant', 'medical devices']
```

### 비교 결과

| 항목 | TICKER_KEYWORDS | TICKER_TO_GDELT_V2 |
|------|----------------|-------------------|
| 필터 구조 | 단일 OR | themes AND keywords |
| 컬럼 라우팅 | 불명확 | 명시적 분리 |
| 노이즈 제어 | 약함 | 강함 |
| TLT 테마 | `ECON_FRBRESERVE` (더 정밀) | `EPU_ECONOMY` (너무 넓음) |
| TIP 테마 | `ECON_INFLATION` (더 정밀) | `EPU_ECONOMY` (너무 넓음) |
| GLD 키워드 | `'gold'` 단독 포함 → GOLDMAN SACHS 오매핑 위험 | 구체적 구문 사용 |

### 시사점

`ECON_FRBRESERVE`(연방준비제도), `ECON_INFLATION`은 우리 v2보다 더 정밀한 태그.  
Step 6-A 실행 시 이 태그들이 실제 GKG에 존재하는지 확인 후 v2에 반영 예정.

---

## 5. 근본적 한계 재검토

### 개별주 매핑의 구조적 한계

#### 한계 1. 부분 문자열 노이즈 (관리 가능)

AAPL 254건 중 실제 샘플:
```
기관: "traffic safety administration; apple app"
테마: MANMADE_DISASTER_IMPLIED → 교통사고 기사에 "apple app" 언급

기관: "birmingham police headquarters; apple app"
테마: SECURITY_SERVICES → 경찰 기사에 "apple app" 언급
```
→ `'APPLE'` 단독 키워드 사용 시 심화. `'APPLE INC'`만 사용하면 어느 정도 완화되나 완전 차단 불가.

#### 한계 2. 기사 전체 감성 ≠ 기업 특정 감성 (구조적, 해결 불가)

```
기사: "Apple, Microsoft, Nvidia fell as White House announced tariff policies..."
GDELT Organizations: [apple inc, microsoft, nvidia, white house, dow jones]
V2Tone: -2.3  ← 기사 전체 감성

이 -2.3이 AAPL의 감성인가? MSFT의 감성인가? 관세 정책의 감성인가?
→ 구분 불가. AAPL, MSFT, NVDA 모두 동일한 -2.3을 공유.
```

결과적으로 대형 기술주들의 GDELT 감성은 **시장 전체 감성에 수렴**하는 경향.

#### 한계 3. 대형주 정보 효율성 (구조적, 해결 불가)

```
뉴스 발생
    ↓ (수 초 이내)
HFT/알고리즘 트레이딩 → 가격 반영
    ↓ (15분 이상 후)
GDELT GKG 업데이트
    ↓ (일별 집계)
우리 모델에 입력
```

GDELT가 뉴스를 포착할 시점엔 이미 가격에 반영된 이후.
이는 매핑 품질의 문제가 아닌 **데이터 자체의 구조적 한계**.

### 결론적 평가

| 항목 | 평가 |
|------|------|
| 부분 문자열 노이즈 | 관리 가능 |
| 기사 수준 감성 ≠ 기업 수준 감성 | ❌ 구조적 한계 |
| 대형주 정보 효율성 | ❌ 구조적 한계 |
| 매핑 유효성 검증 방법 | ❌ 없음 (블랙박스) |

---

## 6. 미결 의사결정

### 선택지

| 선택지 | 내용 | 장점 | 단점 |
|--------|------|------|------|
| **A** | GDELT 전면 제거 | 구현 간소화, 블랙박스 제거 | 기투자 작업 낭비, 차별화 감소 |
| **B** | 개별주(8종목)만 GDELT 유지, ETF는 FRED 대체 | 신뢰도 있는 영역에서만 사용 | 구조적 한계 2, 3은 여전히 존재 |
| **C** | GDELT 전체 유지, 역할 축소 (`event_count`만) | 현재 구조 유지 | 신호 약함 |
| **D** | GDELT 완전 제거 + FRED/매크로 지표 전환 | 가장 신뢰도 높음, 검증 용이 | 초기 설계 대폭 변경 필요 |

### 대안 데이터 후보 (선택지 B/D에서 ETF 피처로 활용)

| 자산군 | FRED/매크로 지표 |
|--------|---------------|
| 채권 ETF (TLT/AGG/SHY/TIP) | 국채 수익률, T10Y2Y(수익률 기울기), EFFR |
| GLD | 금 현물 모멘텀, DXY(달러 인덱스), 실질금리(TIPS 10Y) |
| 섹터 ETF | ISM 제조업/서비스업 PMI, 유가, 산업별 경제지표 |

> **현재 상태**: 의사결정 보류 중. 위 선택지 중 하나를 결정하면 `ML_DataPrep_Example.ipynb` 피처 섹션 및 BL 모델 Q 벡터 구성 방식이 확정됨.

---

## 7. GDELT 데이터 구조 개요

GDELT(Global Database of Events, Language, and Tone)는 3개 테이블로 구성됩니다.

| 테이블 | 주요 컬럼 | 활용 목적 |
|--------|----------|----------|
| **Events** | Actor1, Actor2, EventCode, GoldsteinScale, NumMentions | 사건 강도·행위자 추출 |
| **Mentions** | EventID, MentionDocTone, MentionSourceName | 기사별 감성 |
| **GKG** (지식그래프) | DATE, THEMES, ORGANIZATIONS, V2Tone | 핵심 활용 테이블 |

### V2Tone — 7차원 감성 벡터

```
V2Tone = "전체, 긍정, 부정, 양극화, 활동성, 자기참조, 외부참조"
예: "2.31, 3.50, -1.20, 4.70, 10.2, 0.3, 1.8"
```

### 실제 확인된 GKG 컬럼명 (gdelt Python 라이브러리 v2)

| 논리 컬럼 | 실제 컬럼명 | 인덱스 |
|----------|-----------|--------|
| 날짜 | `DATE` | 1 |
| 테마 | `Themes` | 7 |
| 기관 | `Organizations` | 13 |
| 감성 | `V2Tone` | 15 |

---

## 8. 초기 매핑 설계

### 매핑 타입 3가지

| 타입 | 대상 | 방식 | 비고 |
|------|------|------|------|
| `org` | 개별 대형주 | Organizations에서 기업명 직접 검색 | 가장 신뢰도 높음 |
| `org_themed` | AMZN | 기업명 AND 테마 필터 | Amazon River 지리명 오매핑 차단 |
| `keyword` | ETF류 | Themes 태그 AND 키워드 조합 | 커버리지 불안정 |

### TICKER_TO_GDELT v2 (현재 최신)

구현 위치: `김재천/GDELT_Test.ipynb` — cell-25

```python
TICKER_TO_GDELT_V2 = {
    # 개별 대형주 (org 타입)
    'AAPL': {'type':'org', 'names':['APPLE INC', 'APPLE COMPUTER']},
    'MSFT': {'type':'org', 'names':['MICROSOFT CORP', 'MICROSOFT CORPORATION', 'MICROSOFT']},
    'AMZN': {'type':'org_themed',
             'names':  ['AMAZON.COM', 'AMAZON INC', 'AMAZON WEB SERVICES', 'AWS'],
             'themes': ['ECON_STOCKMARKET', 'TECHNOLOGY', 'EPU_ECONOMY']},
    'GOOGL':{'type':'org', 'names':['ALPHABET INC', 'GOOGLE', 'ALPHABET']},
    'JPM':  {'type':'org', 'names':['JPMORGAN CHASE', 'JP MORGAN', 'JPMORGAN']},
    'JNJ':  {'type':'org', 'names':['JOHNSON & JOHNSON', 'JOHNSON AND JOHNSON', 'JOHNSON JOHNSON']},
    'PG':   {'type':'org', 'names':['PROCTER & GAMBLE', 'PROCTER AND GAMBLE', 'PROCTER GAMBLE']},
    'XOM':  {'type':'org', 'names':['EXXON MOBIL', 'EXXONMOBIL', 'EXXON']},

    # 채권 ETF (EPU_ECONOMY로 수정 — ECON_INTEREST_RATES 미존재 확인)
    'TLT':  {'type':'keyword', 'themes':['EPU_ECONOMY'],
             'keywords':['TREASURY', 'FEDERAL RESERVE', 'LONG-TERM BOND', 'BOND YIELD', '20-YEAR']},
    'AGG':  {'type':'keyword', 'themes':['EPU_ECONOMY'],
             'keywords':['INVESTMENT GRADE', 'BOND MARKET', 'CREDIT SPREAD', 'AGGREGATE BOND']},
    'SHY':  {'type':'keyword', 'themes':['EPU_ECONOMY'],
             'keywords':['2-YEAR TREASURY', 'SHORT-TERM BOND', 'FEDERAL RESERVE', 'FED FUNDS']},
    'TIP':  {'type':'keyword', 'themes':['EPU_ECONOMY'],
             'keywords':['TIPS', 'INFLATION PROTECTED', 'REAL YIELD', 'CPI', 'INFLATION RATE']},

    # 대안 ETF (ECON_STOCKMARKET으로 수정 — ECON_GOLD/COMMODITY 미존재 확인)
    'GLD':  {'type':'keyword', 'themes':['ECON_STOCKMARKET'],
             'keywords':['GOLD PRICE', 'WORLD GOLD COUNCIL', 'COMEX GOLD', 'GOLD FUTURES', 'SPOT GOLD']},
    'DBC':  {'type':'keyword', 'themes':['ECON_STOCKMARKET'],
             'keywords':['COMMODITY INDEX', 'RAW MATERIALS', 'COMMODITY MARKET', 'COMMODITY ETF']},

    # ... (전체 코드는 GDELT_Test.ipynb cell-25 참조)
}
```

---

## 9. 집계 후 데이터 구조

### (date, ticker) 기준 일별 집계

```python
gdelt_by_stock = (
    gdelt_mapped
    .groupby(['date', 'ticker'])
    .agg(
        gdelt_tone_avg   = ('_tone_avg',   'mean'),
        gdelt_tone_neg   = ('_tone_neg',   'mean'),
        gdelt_tone_polar = ('_tone_polar', 'mean'),
        gdelt_event_cnt  = ('_tone_avg',   'count'),
        gdelt_tone_std   = ('_tone_avg',   'std'),
    )
    .reset_index()
)
```

### ML 입력 패널 구조

```
date     | ticker | gdelt_tone_avg | mom_20d | vol_20d | VIX | HY_spread | fwd_ret_30d
---------|--------|---------------|---------|---------|-----|-----------|------------
24-01-01 | AAPL   |     2.31      |  0.042  |  0.018  |13.2 |    3.8    |   +0.032
24-01-01 | TLT    |    -0.45      | -0.005  |  0.008  |13.2 |    3.8    |   -0.005
```

- 학습 창 1개: 150일 × 30종목 = **4,500행**

---

## 10. 결측 처리 전략

```python
panel['has_gdelt']       = panel['gdelt_tone_avg'].notna().astype(int)  # 결측 플래그
panel['gdelt_tone_avg']  = panel['gdelt_tone_avg'].fillna(0)
panel['gdelt_tone_neg']  = panel['gdelt_tone_neg'].fillna(0)
panel['gdelt_tone_polar']= panel['gdelt_tone_polar'].fillna(0)
panel['gdelt_event_cnt'] = panel['gdelt_event_cnt'].fillna(0)
panel['gdelt_tone_std']  = panel['gdelt_tone_std'].fillna(0)
```

> 0 채움 + `has_gdelt` 플래그 → 모델이 "GDELT 신호 없음" 상태를 학습 가능

---

## 11. ML → Black-Litterman 연결

```
GDELT 파생 피처 (gdelt_tone_avg, gdelt_event_cnt, gdelt_tone_std)
     +
기존 피처 (VIX, HY스프레드, 모멘텀, 변동성)
     ↓
ML 모델 (RF / XGBoost / TabPFN)
학습: 150일 × 30종목 패널
     ↓
OOS 첫날 예측: 30종목 각 1행 → 예측 수익률
     ↓
Q 벡터 (30 × 1) + Ω 행렬 (불확실성)
     ↓
Black-Litterman → μ_BL → MV 최적화 → 비중
```

| 모델 | Q 산출 | Ω 산출 |
|------|--------|--------|
| RF 회귀 | 트리 평균 예측값 | 트리 간 표준편차 |
| XGBoost 분류 | 클래스별 기대수익률 가중합 | 1 - max(예측 확률) |
| TabPFN | 분류 확률 기반 기대값 | 분류 확률 엔트로피 |

---

## 12. Ablation 축

| Ablation | 내용 | 측정 지표 |
|----------|------|---------|
| No GDELT | GDELT 피처 전부 제거 | Sharpe 변화량 |
| GDELT only equity | 개별주만 GDELT, ETF는 0 | Sharpe 변화량 |
| GDELT keyword only | 키워드 방식만 (기업명 제거) | 커버리지·Sharpe |
| tone_avg only | 다차원 V2Tone → tone_avg 단일 | 피처 중요도 변화 |
