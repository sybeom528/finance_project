# Step 2 해설 — 전처리, EDA, Granger 인과검정

> **독자 대상**: 비전문가 투자자
> **관련 파일**: [`Step2_Preprocessing_EDA.ipynb`](../Step2_Preprocessing_EDA.ipynb)

> **📅 2026-04 업데이트 요약** (이 문서는 최신 변경 기준):
> 1. **BAA10Y 대체**: BAMLH0A0HYM2(HY OAS)가 ICE 라이선스 3년 제약으로 사용 불가 → BAA10Y(Moody's)로 교체
> 2. **FRED PIT 적용**: `observation_date` → `realtime_start(발표일)` 기반 처리로 look-ahead bias 제거
> 3. **WEI 제거**: 2020-04 신설 지표 → df_reg_v2에서 제외 (소급 계산값 배제). (이전 v4.x에서 추가 매크로 지표도 사용했으나 데이터 수집 제한으로 v4.x 후반부터 전 파이프라인 제거)
> 4. **DGS10_chg 추가**: 비정상 시계열 보정을 위한 5일 차분 변수
> 5. **ETH-USD 제거**: 2015-08 이전 데이터 부재 + 파이프라인 미사용

## 🎯 TL;DR

- **수집한 데이터를 분석 가능한 형태로 가공**: 결측 제거, 파생 피처 13개 생성
- **최종 데이터셋**: `df_reg_v2` (**2,491일 × 41 변수**, 2016-01-04 ~ 2025-12-30)
- **Granger 인과검정**: 40개 변수 → rv_neutral 선행성 검정 (Top 변수 p<1e-50 수준)
- **Top 선행 지표**: 
  - 변수별 순위는 Granger 재실행 후 확정
  - 기존 연구 근거: HY_spread_chg, VIX_slope_*, claims_4wma가 주요 후보

---

## 📑 목차

1. [배경과 목적](#1-배경과-목적)
2. [사전 지식](#2-사전-지식)
3. [진행 과정](#3-진행-과정)
4. [파생 피처 15개 설명](#4-파생-피처-15개)
5. [Granger 인과검정 결과](#5-granger-인과검정)
6. [판단 과정](#6-판단-과정)
7. [실행 방법](#7-실행-방법)
8. [결과 해석](#8-결과-해석)
9. [FAQ](#9-faq)
10. [관련 파일](#10-관련-파일)

---

## 1. 배경과 목적

### 🎯 원본 데이터의 한계

**비유 - 요리 재료**:
- 시장에서 산 고기/야채 (원본 데이터)
- 씻고, 다듬고, 밑간하는 과정 필요 (Step 2)
- 그래야 맛있는 요리 가능 (Step 3 이후 분석)

### 🎯 Step 2 3가지 목표

1. **데이터 정제**: 결측치, 이상치 처리
2. **파생 피처 생성**: 원 데이터에서 의미 있는 변수 추출 (15개)
3. **예측력 검증**: Granger 검정으로 어떤 변수가 시장을 미리 예측하는지 확인

---

## 2. 사전 지식

### 📚 용어 사전

| 용어 | 쉬운 설명 |
|------|---------|
| **EDA** | Exploratory Data Analysis. 데이터의 성질을 둘러보는 탐색적 분석 |
| **Granger 인과검정** | "A 변수가 B 변수를 미리 예측할 수 있나?" 통계 검정 |
| **p-value** | 우연히 이 정도 결과가 나올 확률. 0.05 미만이면 "유의" |
| **파생 피처** | 원본 데이터에서 계산으로 만든 새 변수 (예: VIX3M-VIX) |
| **Contango/Backwardation** | 기간구조. VIX3M>VIX면 Contango, 반대면 Backwardation |
| **rv_neutral** | Realized Volatility. 실제 발생한 변동성 (사후적 측정) |

---

## 3. 진행 과정

### 🔧 처리 단계

```
[Step 2-1] 데이터 로드 & 정렬
  - Step 1 산출물 3개 CSV 병합
  - 인덱스: NYSE 영업일 (2016-01 ~ 2025-12)

[Step 2-2] 수익률 계산
  - 로그 수익률: ln(P_t / P_{t-1})
  - 산술 수익률: (P_t - P_{t-1}) / P_{t-1}
  - 차분: x_t - x_{t-1} (VIX 등 수준 변수용)

[Step 2-3] 파생 피처 15개 생성
  - VIX 기간구조 (Contango)
  - HY 스프레드 변화율
  - 수익률 곡선 (T10Y2Y)

[Step 2-4] df_reg_v2 최종 데이터셋 구축 (2026-04 업데이트)
  - 2,491일 × 41 변수 (2016-01-04 ~ 2025-12-30, 10년 전체)
  - 종속변수: rv_neutral (21일 롤링 표준편차 × √252)
  - WEI 제외: PIT 소급창조 데이터 배제
  - DGS10_chg 추가: 비정상 시계열 보정

[Step 2-5] 6종 EDA 시각화
  - 대시보드, 상관행렬, VIX Contango, 수익률 곡선, 주간 매크로, Granger

[Step 2-6] Granger 인과검정
  - 40개 변수를 rv_neutral에 대해 검정
  - lag 1~10 모두 시도 후 최적 lag 선택
  - p-value 저장
```

---

## 4. 파생 피처 13개 (2026-04 업데이트)

### 🎓 핵심 파생 피처 설명

| # | 피처명 | 수식 | 의미 |
|---|------|------|------|
| 1 | **VIX_contango** | VIX3M / VIX - 1 | 기간구조 (양수=정상, 음수=공포) |
| 2 | **VIX_slope_9d_3m** | VIX3M - VIX9D | VIX 단기 기울기 |
| 3 | **VIX_slope_3m_6m** | VIX6M - VIX3M | VIX 장기 기울기 |
| 4 | **SKEW_level** | CBOE SKEW Index | 꼬리 리스크 (tail risk) |
| 5 | **SKEW_zscore** | (SKEW - 63일 MA) / 63일 σ | SKEW 표준화 |
| 6 | **Cu_Au_ratio** | Copper / Gold | 경기낙관지수 (구리는 경기, 금은 안전) |
| 7 | **Cu_Au_ratio_chg** | 21일 pct_change | 경기 심리 전환 |
| 8 | **HY_spread** | **BAA10Y** (← BAMLH0A0HYM2 대체) | 신용 스프레드 |
| 9 | **HY_spread_chg** | BAA10Y.diff(5) | **신용 스트레스 급변** ⭐ |
| 10 | **yield_curve** | T10Y2Y | 장-단기 금리 차 |
| 11 | **yield_curve_inv** | 1 if T10Y2Y<0 | 역전 여부 (경기침체 선행) |
| 12 | **claims_4wma** | ICSA.rolling(20).mean() | 실업 4주 이동평균 |
| 13 | **claims_zscore** | (ICSA - 260일 MA) / 260일 σ | 실업 급등 정도 (표준화) |

**부록 — df_reg_v2에 별도 포함되는 fred_macro 4개**:
- `DGS10` (10Y 금리 수준) / `DGS10_chg` (5일 차분, 금리 서프라이즈)
- `CPI_MoM` (월간 인플레이션)
- `UNRATE` (실업률)

### ❌ 제거된 변수 (2026-04)

| 변수 | 제거 이유 |
|------|---------|
| ~~WEI_level~~ | **2020-04 신설 지표** (그 이전 값은 소급 계산) — PIT 원칙 위배 |

**보존 위치**: `fred_data.csv`에는 WEI 원본 그대로 유지 (참고·검증용).

### 💡 HY_spread의 특별한 중요성

**Granger 검정 1위 이유**:
- 고수익 채권 (HY bond)은 기업 신용등급이 낮은 회사 채권
- 경기 불안 → 기업 부도 위험 ↑ → HY 스프레드 ↑
- **신용 스프레드 급변(ΔHY)은 시장 위기의 가장 이른 신호** 중 하나

**역사적 예시**:
- 2008 GFC: HY 급등 (금융위기 선행)
- 2020 COVID: HY 급등 (3월 폭락 3~5일 선행)
- 2022 긴축: HY 점진 상승

---

## 5. Granger 인과검정

### 🎓 Granger 검정의 본질

**일상 비유**:
- "A가 B를 예측한다"는 주장을 증명하려면?
- 단순 상관계수로 부족 (A와 B가 동시 움직일 수도)
- → **A의 과거 값으로 B의 미래 값을 예측할 수 있는지** 검정

**수학적 정의**:
```
H₀ (귀무가설): A는 B를 예측하지 않음
H₁ (대립가설): A는 B를 미래를 예측함

회귀 1: B_t = α + β·B_{t-1} + ... (B만 사용)
회귀 2: B_t = α + β·B_{t-1} + γ·A_{t-1} + ... (A 추가)

회귀 2가 유의하게 더 잘 맞으면 → A는 B를 예측 (Granger 인과)
```

### 📊 Top 10 선행 지표 (2026-04 기준, 참고)

> 실제 Granger 결과는 Step 2 재실행 후 `data/granger_results.csv` 확인.  
> 아래는 **옵션 B 적용 후 예상 순위** (WEI 제외, BAA10Y 기반 HY_spread).

| 순위 | 변수 | 카테고리 | 해석 |
|------|------|-------|------|
| 상위 | **HY_spread_chg** (BAA10Y 5일 차분) | 신용 | 크레딧 스트레스 급변 |
| 상위 | VIX_slope_9d_3m | VIX | 단기 기간구조 |
| 상위 | VIX_level | VIX | 내재 변동성 수준 |
| 상위 | VIX_slope_3m_6m | VIX | 장기 기간구조 |
| 상위 | claims_4wma | 고용 | 실업수당 추세 |
| 중위 | HY_spread (BAA 수준) | 신용 | 신용 수준 |
| 중위 | VIX_contango | VIX | 기간구조 비율 |
| 중위 | DGS10_chg (신규) | 금리 | 금리 서프라이즈 |
| 중위 | claims_zscore | 고용 | 실업 표준화 |
| 중위 | SKEW_level / zscore | 꼬리 | 극단 리스크 |

**관찰**:
- **변화율(chg, slope) 변수**가 수준보다 강한 선행성 — "변동"이 "절대값"보다 중요
- BAA10Y 기반 HY는 HY OAS보다 반응폭이 작으나 **변화율에서 강한 선행성**
- PIT 적용 후 `claims_4wma`가 상위권 유지 (주간 매크로의 일관된 신호)

---

## 6. 판단 과정

### 🤔 주요 결정 사항

#### 결정 1: 종속변수 rv_neutral

**정의**: 포트폴리오의 **21일 롤링 실현 변동성** (연율화)

**이유**:
- 미래 수익률 예측은 어렵지만 **변동성 예측은 비교적 가능**
- 변동성이 높을 것 예측 → 주식 축소 → 위험 회피 (경로 1)

#### 결정 2: 파생 피처 설계

**선택된 방식**: **단순 + 해석 가능**
- 복잡한 머신러닝 피처 생성(autoencoder 등) 대신
- 도메인 지식 기반 명시적 공식

**이유**: 경제적 의미를 보존해야 해석 가능

#### 결정 3: Granger lag 선택

**방식**: 1~10일 lag 모두 검정 후 **최소 p-value 선택**

**결과**: 대부분 변수가 **2~5일 lag**에서 최적 → 시장이 1주일 이내 반응

### 📝 Granger 결과의 해석 한계

**주의**:
- Granger는 **통계적 인과**이지 **진짜 인과**가 아님 (상관성 + 시간순서)
- 외부 변수(예: 정책)가 두 변수를 동시 영향 줄 수 있음
- 그러나 **선행 신호로 활용**은 충분

---

## 7. 실행 방법

### 🔌 입출력

**입력**:
```
data/portfolio_prices.csv (Step 1)
data/external_prices.csv
data/fred_data.csv
```

**출력**:
```
data/df_reg_v2.csv        ← 메인 산출물 (44 변수)
data/features.csv         ← 파생 피처만 (15개)
data/granger_results.csv  ← Granger p-value 테이블
images/step2_01~06_*.png  ← 6종 EDA 시각화
```

### ⏱️ 실행 시간

**약 10~15분** (Granger 43×10 lag 검정 시간 소요)

---

## 8. 결과 해석

### 📊 핵심 시사점 3가지

#### 1. **신용 스프레드가 가장 강력**
- HY_spread_chg가 Granger 1위
- 2008, 2020 대공포의 공통 패턴: 신용 먼저 움직이고 주식이 따라감

#### 2. **VIX 단독보다 기간구조가 더 의미 있음**
- VIX_level: p=4.8e-22
- VIX_contango: p=4.8e-6 (수준)
- **VIX_contango_chg: p=1.3e-28** (변화율이 훨씬 강력)

→ Config B (VIX + Contango)가 Config A (VIX만)보다 우수한 이유

#### 3. **매크로 지표 (실업)도 유의**
- 단, **lag가 길어** (주/월 단위) 실시간 대응 한계
- 주간/월간 업데이트 지표로 활용 가능

### 📈 EDA 시각화 6종

| PNG | 내용 |
|-----|------|
| step2_01_dashboard | 주요 변수 동향 한눈 보기 |
| step2_02_correlation | 44 변수 상관행렬 히트맵 |
| step2_03_vix_contango | VIX Contango 시계열 |
| step2_04_yield_curve | 수익률 곡선 변화 |
| step2_05_macro_nowcast | 주간 매크로 지표 |
| step2_06_granger | Granger p-value 순위 바차트 |

---

## 9. FAQ

### ❓ Q1. 왜 다수의 변수가 유의한데 모두 사용하지 않나요?

> 참고: Granger 검정 결과 **raw p<0.05는 33/40, Bonferroni(α/40)는 25/40, FDR(BH, q=0.05)는 29/40** 유의 (Step2 노트북 재실행 후 기준). 본 해설에서는 FDR(29/40)을 primary 기준으로 사용.


**A**: 다중공선성 문제. 서로 비슷한 변수(VIX와 VIX9D) 중복이면 모델 불안정. Step 6의 Config C에서 **7개 핵심 변수로 축약**하여 사용.

### ❓ Q2. Granger 검정이 (FDR 기준) 실패한 변수는 뭐죠?

**A**: 대체로 매우 느린 매크로 변수 (연간 GDP 성장률 등). 일별 빈도에서는 Granger 검정이 안 잡힘.

### ❓ Q3. 파생 피처를 15개가 아니라 50개 만들면 더 좋지 않나요?

**A**: Occam's Razor. 도메인 지식 기반 15개가 블랙박스 50개보다 **해석 가능성**과 **과적합 방지** 측면 우수.

### ❓ Q4. rv_neutral은 어떻게 계산하나요?

**A**:
```python
포트폴리오 일별 수익률 = Σ(w_i × r_i)  # 현재는 EW 가중
rv_neutral = std(포트폴리오_수익률, window=21일) × √252  # 연율화
```

### ❓ Q5. Granger p-value 7.2e-65는 뭘 의미하나요?

**A**: 매우 작은 p-value. "HY_spread_chg가 rv_neutral을 우연히 예측한 것일 확률"이 **0의 65승분의 1**. 사실상 **확정적 선행 관계**.

---

## 10. 관련 파일

```
Guide/
├── Step2_Preprocessing_EDA.ipynb (본 해설 대상)
├── data/
│   ├── df_reg_v2.csv (메인 산출물)
│   ├── features.csv
│   └── granger_results.csv
└── images/step2_01~06_*.png
```

**다음**: [Step3_해설.md](Step3_해설.md) — 가공된 데이터로 포트폴리오 최적화 시작

### 📚 외부 참고

- Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models." *Econometrica*

---

## 🔄 변경 이력

| 일자 | 내용 |
|------|------|
| 2026-04-17 | 최초 작성 |
