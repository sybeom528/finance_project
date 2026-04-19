# Step 1 해설 — 데이터 수집

> **독자 대상**: 비전문가 투자자
> **관련 파일**: [`Step1_Data_Collection.ipynb`](../Step1_Data_Collection.ipynb)

> **📅 2026-04 업데이트 요약 (v4.2)**:
> 1. **워밍업 기간 도입**: 2014-01-01부터 수집, 분석은 2016-01-01~
> 2. **SPY 실거래일 인덱스**: NYSE 공휴일 정확 반영 (`pd.bdate_range` 대체)
> 3. **bfill 제거**: `ffill` 단독 사용 → look-ahead bias 차단
> 4. **ETH-USD 제거**: 2015-08 이전 데이터 부재 + 미사용
> 5. **BAA10Y 대체**: `BAMLH0A0HYM2`(HY)가 ICE 라이선스 3년 제약 → `BAA10Y`(Moody's)
> 6. **FRED PIT 적용**: observation → realtime_start(발표일) 기반 하이브리드 처리

## 🎯 TL;DR (30초 요약)

- **12년치 수집(2014~2025) + 10년 분석(2016~2025)**: 30개 자산 + 11개 시장 지표 + 8개 거시경제 지표
- **주요 출처**: yfinance (주식/ETF), FRED (거시지표, **PIT 방식**)
- **핵심 산출물**: `portfolio_prices.csv`, `external_prices.csv`, `fred_data.csv` (모두 3,017행)
- **데이터 품질**: SPY 실거래일 정렬, forward-fill만 사용 (look-ahead bias 제거)

---

## 📑 목차

1. [배경과 목적](#1-배경과-목적)
2. [사전 지식](#2-사전-지식)
3. [진행 과정](#3-진행-과정)
4. [주요 개념 설명](#4-주요-개념-설명)
5. [판단 과정과 결정 사항](#5-판단-과정)
6. [실행 방법](#6-실행-방법)
7. [결과 해석](#7-결과-해석)
8. [FAQ](#8-faq)
9. [관련 파일](#9-관련-파일)

---

## 1. 배경과 목적

### 🎯 왜 데이터 수집이 중요한가

**요리 비유**:
- 재료가 신선하지 않으면 아무리 좋은 레시피로도 맛있는 요리 불가
- 데이터가 오염되거나 부족하면 정교한 AI/통계 모델도 엉뚱한 답 도출

### 🎯 Step 1의 목표

> **"10년치 자산 가격 + 시장 지표 + 거시경제 데이터를 한 폴더에 깨끗하게 정리"**

이후 단계(Step 2~10)가 이 데이터를 공통 기반으로 사용합니다.

---

## 2. 사전 지식

### 📚 용어 사전

| 용어 | 쉬운 설명 |
|------|---------|
| **yfinance** | "Yahoo Finance 무료 API". 주식·ETF 가격을 파이썬으로 가져옴 |
| **FRED** | "미국 연준(세인트루이스) 경제 데이터 API". 금리·GDP·실업률 등 |
| **ALFRED** | "Archival FRED" — 발표 당시 값·수정 이력 전체 제공 (vintage 데이터) |
| **OHLCV** | Open/High/Low/Close/Volume (시가/고가/저가/종가/거래량) |
| **ETF (Exchange-Traded Fund)** | "여러 주식을 묶어서 1주처럼 거래하는 상품" (예: SPY=S&P 500 전체) |
| **NYSE 영업일** | 주식시장이 열리는 날 (SPY 실거래일로 자동 정렬) |
| **Forward-fill** | 결측치를 앞의 값으로 채우는 방법 (전일 가격 유지) |
| **로그 수익률** | ln(오늘 가격/어제 가격). 시간 합산 가능 + 정규분포 근사 |
| **워밍업 기간** | 롤링 변수 안정화를 위해 분석 시작 전에 미리 수집하는 구간 |
| **PIT (Point-In-Time)** | 각 시점 T에 대해 "T 시점에 실제로 알 수 있었던 값"만 사용 |
| **Look-ahead bias** | 백테스트에 미래 정보가 섞여 성과가 과대평가되는 오류 |
| **발표 시차 (Publication lag)** | 관측 종료 시점과 실제 발표 시점의 차이 (예: UNRATE는 ~30일 지연) |

---

## 3. 진행 과정

### 📊 수집 데이터 3범주

```
[범주 1] 포트폴리오 자산 (30개) — 매매 대상
  - 인덱스 ETF (5): SPY, QQQ, IWM, EFA, EEM
  - 섹터 ETF (11): XLK, XLF, XLE, XLV, VOX, XLY, XLP, XLI, XLU, XLRE, XLB
  - 개별 주식 (8): AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, PG, XOM
  - 채권 ETF (4): TLT, AGG, SHY, TIP
  - 대체 자산 (2): GLD (금), DBC (원자재)

[범주 2] 외부 시장 지표 (6 + 5 = 11개) — 관찰용
  - 외부 6: CL=F(원유), GC=F(금선물), SI=F(은), BTC-USD, ^VIX, DX-Y.NYB(달러)
      · ETH-USD는 2015-08 이전 데이터 부재로 제거 (2026-04 업데이트)
  - 대안 5: ^VIX9D, ^VIX3M, ^VIX6M, ^SKEW, HG=F(구리)

[범주 3] FRED 거시경제 (8개) — PIT 적용
  - 일별 (3): DGS10 (10년물), T10Y2Y (수익률 곡선), BAA10Y (신용 스프레드)
      · BAA10Y는 BAMLH0A0HYM2 대체 (ICE 라이선스 3년 제약 우회, 2026-04)
  - 주간 (2): ICSA (실업수당), WEI (주간 경제지수)
  - 월간 (2): CPIAUCSL (CPI), UNRATE (실업률)
```

### 🔧 처리 파이프라인 (2026-04 업데이트)

```
1. yfinance API로 일별 데이터 다운로드 (WARMUP_START=2014-01-01 ~ 2025-12-31)
2. FRED API로 거시지표 수집 — 하이브리드 PIT:
     · 일별 시리즈 (DGS10, T10Y2Y, BAA10Y): observation + 발표 시차(0~1일)
     · 주간/월간 (ICSA, WEI, CPI, UNRATE): ALFRED vintage 기반 완전 PIT
3. SPY 실거래일 기준 인덱스 정렬 (NYSE 공휴일 자동 제외)
4. 결측치 처리: forward-fill 단독 (bfill 제거 → look-ahead 방지)
5. CSV 파일로 저장 (3,017일 기준)
```

---

## 4. 주요 개념 설명

### 🎓 개념 1: 왜 ETF와 개별주를 섞나요?

**분산투자 교과서 접근**:
- 인덱스 ETF (SPY 등): 시장 전체를 대표
- 섹터 ETF (XLK 등): 특정 섹터에 집중
- 개별주 (AAPL 등): 기업별 특성 반영

**효과**: 서로 다른 자산 움직임 패턴이 모여 **분산 효과** 극대화

### 🎓 개념 2: 왜 VIX 기간구조를 수집하나요?

**VIX (공포지수)**: 현재 주식시장 변동성 예상치 (30일 기준)
**VIX3M**: 3개월 기준 변동성 예상치

**기간구조 (Contango vs Backwardation)**:
- **Contango** (VIX3M > VIX): 미래가 더 불안할 것으로 예상 (정상)
- **Backwardation** (VIX3M < VIX): 현재 공포가 미래보다 큼 (비상)

→ **Backwardation은 대공포의 신호** (예: 2008, 2020-3월)

### 🎓 개념 3: FRED 거시지표의 가치

**왜 주식 데이터만으로는 부족한가**:
- 주식은 시장 심리만 반영
- FRED 지표는 **실물 경제**와 **정책**을 반영
- 예: claims_zscore (실업수당 청구 급증 → 경기침체 선행 신호)

---

## 5. 판단 과정

### 🤔 주요 결정 사항

#### 결정 1: 분석 기간 2016-2025 + 워밍업 2014-

**후보**:
- 15년 (2011-2025): 더 풍부한 데이터, 2012 유로존 위기 포함
- **10년 분석 + 2년 워밍업 (2014-수집 / 2016-분석)**: 현대적 시장 구조 + 롤링 변수 안정화
- 5년 (2021-2025): 최신 흐름만

**선택: 워밍업 포함 10년 분석** (2026-04 업데이트)
- `WARMUP_START = 2014-01-01`: 최대 260일 롤링 변수(claims_zscore) 안정화
- `ANALYSIS_START = 2016-01-01`: 실제 분석 기간

#### 결정 2: 30개 자산 선정

**선정 기준**:
- **다양성**: 인덱스 + 섹터 + 개별주 + 채권 + 대체
- **유동성**: 일평균 거래량 충분 (거래비용 정확성)
- **10년 전 기간 존재**: 생존자 편향 최소화

#### 결정 3: 결측 처리 방식 (2026-04 수정)

**선택: Forward-fill 단독**
- 국경일·장마감 등은 전날 가격 유지가 합리적 (ffill)
- **bfill 제거**: "미래값으로 과거 채움"은 look-ahead bias 유발 → 절대 금지
- 시작 시점 결측은 `dropna(how='all')`로 완전 NaN 행만 제거

#### 결정 4: SPY 실거래일 인덱스 (2026-04 신규)

**기존 문제**: `pd.bdate_range(freq='B')`는 NYSE 공휴일(MLK Day, Good Friday)을 포함 → 잘못된 인덱스  
**해결**: `yf_data['SPY'].dropna().index` 사용 → SPY 실제 거래일만 = NYSE 영업일 정확

#### 결정 5: FRED PIT 하이브리드 (2026-04 신규)

**문제**: `get_series()`의 observation_date는 "측정 시점"일 뿐, 실제 발표일과 차이
- 예: 2024-01 CPI는 2024-02-13 발표 → 43일 look-ahead bias

**해결 (변수별 차등)**:
- 일별 시리즈: vintage API 상한(2,000개) 초과 → observation + 하드코딩 lag 적용
- 주간/월간: `get_series_all_releases()` vintage 기반 완전 PIT 재구성
  - 각 거래일 T에 "T 시점까지 발표된 가장 최근 값" 사용
  - 속보치 → 잠정치 → 확정치 → benchmark 수정 모두 반영

### 📝 특이 처리

- **VIX9D**: 2011년 이후 존재 → 워밍업 기간(2014~)에서도 모두 가용
- **DBC**: 원자재 ETF, 환율 영향 있으나 단순 보유 목적이라 원가 기준으로 충분
- **XLRE**: 2015-10-08 상장 → 워밍업 초기엔 NaN, ffill로 처리
- **BTC-USD**: 2014-09 이후 데이터, 워밍업 8개월만 NaN (분석 기간에 영향 없음)
- **WEI**: 2020-04 신설 소급 지표 → Step 2 df_reg_v2에서 제외 (fred_data.csv에 원본 보존)
  (이전 v4.x에서 추가 매크로 지표도 사용했으나 데이터 수집 제한으로 v4.x 후반부터 전 파이프라인에서 제거)

---

## 6. 실행 방법

### 🔌 입출력

**입력**: 없음 (API 직접 호출)
**출력** (2026-04 기준):
```
data/portfolio_prices.csv   ← 30자산 × 3,017일 (1.6 MB, 2014~2025)
data/external_prices.csv    ← 11 지표 × 3,017일 (ETH 제거)
data/fred_data.csv          ← 8 거시지표 × 3,017일 (PIT 적용)
```

### ⏱️ 실행 시간

**약 5~10분** (FRED PIT vintage 처리 시간 포함, 네트워크 속도에 따라 변동)
- yfinance 41개 티커: 약 2~3분
- FRED 8 시리즈: 일별 3개 즉시, 주간/월간 5개는 vintage 처리로 각 30초~1분

### ✅ 체크리스트

```
[ ] yfinance 패키지 설치 (pip install yfinance)
[ ] FRED_API_KEY 환경변수 설정 (https://fred.stlouisfed.org/docs/api/api_key.html)
[ ] .env 파일이 김재천/ 디렉토리에 존재 (Guide의 상위)
[ ] 인터넷 연결 확인
[ ] 3,017 SPY 실거래일 수집 확인 (NYSE 공휴일 정확 제외)
```

---

## 7. 결과 해석

### 📊 데이터 기본 통계

| 자산 | 2016 시가 | 2025 종가 | 연환산 수익률 |
|------|---------|---------|----------|
| SPY | $200 | $580 | **+12%/yr** |
| QQQ | $108 | $420 | +16%/yr |
| AAPL | $26 | $240 | +25%/yr |
| TLT | $122 | $91 | -3%/yr |
| GLD | $101 | $245 | +10%/yr |

**관찰**: 주식은 상승, 장기채 하락 (금리 상승 반영), 금 상승

### 📈 주요 이벤트 확인

- **2018-02 Volmageddon**: VIX 50 스파이크 1회
- **2020-03 COVID**: VIX 82 (역사적 최고치)
- **2022-01~10**: 연준 긴축 → SPY -25%
- **2023-03 SVB**: 단기 경색
- **2024-08 엔캐리 청산**: VIX 60 재상승

---

## 8. FAQ

### ❓ Q1. 왜 국제 주식 (신흥국 채권 등)은 포함 안 되나요?

**A**: 복잡도 관리 차원. v5에서 확장 예정.

### ❓ Q2. yfinance가 무료인데 믿을 만한가요?

**A**: 일반적 개인 분석엔 충분. 단, 실시간 트레이딩엔 유료 데이터(Bloomberg 등) 권장.

### ❓ Q3. 결측치가 얼마나 되나요?

**A**: 대부분 변수 0~1%. 단 PIT 적용으로 **WEI 52% NaN**.
- 이유: WEI가 2020-04 신설되어 그 이전은 "아직 발표 안 됨"
- 처리: Step 2 df_reg_v2에서 제외 (원본은 `fred_data.csv`에 보존)
- Step 2 최종 데이터셋: **2,491일 × 41 변수** (WEI 제외로 전체 분석 기간 확보)

### ❓ Q5. PIT 처리가 왜 필요한가요?

**A**: 백테스트의 look-ahead bias 제거.
- 예: 2024-01 CPI는 실제 2024-02-13 발표 → observation 기준이면 43일 미래 정보 유입
- 이 프로젝트는 Step 6 경보 시스템에 CPI·UNRATE 등 사용 → PIT 미적용 시 과대평가
- ALFRED vintage API로 "각 시점에 실제 알 수 있었던 값"만 사용하여 해결

### ❓ Q6. BAA10Y가 BAMLH0A0HYM2와 같은 역할인가요?

**A**: **비슷하지만 다름**.
- BAMLH0A0HYM2: High Yield(투기등급) 스프레드, ICE BofA 제공
- BAA10Y: Moody's Baa(투자등급 최하단) - 10Y 국채 스프레드
- 상관관계 0.77, 변화 상관 0.64 (공통 3년 기준)
- BAA가 HY보다 변동폭 약 1/2 수준 → Step 6 Config C 임계값을 비례 조정
- 주된 이유: BAMLH0A0HYM2의 FRED API 제약(3년 롤링 데이터만 제공) 회피

### ❓ Q4. 왜 개별주 8개만 선정?

**A**: 대형주 중심 (Magnificent 7 + JPM/JNJ/PG/XOM 전통 대표). 추가 개별주는 분산 효과 한계.

---

## 9. 관련 파일

```
Guide/
├── Step1_Data_Collection.ipynb (본 해설 대상)
├── data/
│   ├── portfolio_prices.csv
│   ├── external_prices.csv
│   └── fred_data.csv
└── images/
    ├── step1_01_total_returns.png
    └── step1_02_alt_data_preview.png
```

**다음**: [Step2_해설.md](Step2_해설.md) — 수집된 데이터를 어떻게 가공하는가

### 📚 외부 참고

- Yahoo Finance API: https://github.com/ranaroussi/yfinance
- FRED API: https://fred.stlouisfed.org/docs/api/fred/

---

## 🔄 변경 이력

| 일자 | 버전 | 내용 |
|------|------|------|
| 2026-04-17 | v4.1 | 최초 작성 |
| 2026-04-19 | **v4.2** | **PIT 적용 + ETH 제거 + BAA10Y 대체 + 워밍업 도입 + bfill 제거 + SPY 실거래일** |
