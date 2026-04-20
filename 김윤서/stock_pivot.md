# 프로젝트 방향성 v4 — 개별 주식 기반 Black-Litterman 전환

> 작성자: 김윤서  
> 작성일: 2026-04-20  
> 근거: 2026-04-20 팀 회의 결정사항 + Claude 피처 엔지니어링 논의  
> 이전 버전: `서윤범/project_design_v3.md`, `김윤서/전체_프로젝트_프로세스.md`

---

## 목차

1. [핵심 방향 전환 요약](#1-핵심-방향-전환-요약)
2. [업데이트된 파이프라인](#2-업데이트된-파이프라인)
3. [종목 선정 기준](#3-종목-선정-기준)
4. [Prior 정의 — 시가총액 비중](#4-prior-정의--시가총액-비중)
5. [Embargo가 필요한 이유](#5-embargo가-필요한-이유)
6. [핵심 함수 이해: pct_change()와 rolling()](#6-핵심-함수-이해-pct_change와-rolling)
7. [피처 엔지니어링 전체](#7-피처-엔지니어링-전체)
8. [전체 피처 목록 및 BL 활용 맵](#8-전체-피처-목록-및-bl-활용-맵)
9. [미결 사항 및 액션 아이템](#9-미결-사항-및-액션-아이템)

---

## 1. 핵심 방향 전환 요약

### 변경 전 (v3)

| 항목 | 기존 설계 |
|------|----------|
| 투자 유니버스 | ETF 22개 + 개별주 8개 (30종) |
| 자산군 | 주식 ETF + 채권 ETF + 대안 ETF |
| Prior 정의 | 1/N 동일비중 또는 ETF AUM 역산 |
| 데이터 구조 | Long-Panel (date × ticker, 패널 전체 통합 학습) |
| 모델링 방식 | 단일 패널 모델 (모든 자산 동일 모델) |

### 변경 후 (v4 — 2026-04-20 회의 결정)

| 항목 | 새 설계 |
|------|---------|
| 투자 유니버스 | **개별 주식** — GICS 11개 섹터 기준 상위 종목 |
| 자산군 | **주식만** (채권·대안 제외) |
| Prior 정의 | **개별 주식 시가총액 비중** |
| 데이터 구조 | 자산별 개별 모델링 검토 (패널 방식과 병행 비교) |
| 모델링 방식 | 자산별 독립 또는 패널 (섹터 코드 피처로 구분) |

### 전환 근거

- **ETF Prior 문제**: ETF는 시가총액 비중 계산이 어려워 BL의 균형 Prior(π = λΣw_mkt) 정의가 불명확. 개별 주식은 시가총액으로 명시적 Prior 정의 가능
- **30개 산업 포트폴리오 논문 사례**: 섹터별 종목을 묶어 포트폴리오 구성 후 BL 적용 → 우리 접근법의 방법론적 근거
- **ETF NAV는 일별 업데이트 가능**하나, 내부 리밸런싱이 분기/반기 단위라 구성 종목 변화가 늦음

---

## 2. 업데이트된 파이프라인

```
[Step 1] 데이터 수집
    GICS 11개 섹터 × 섹터별 상위 N종목 (개별 주식)
    + 외부 지표 (SPY, 섹터 ETF 수익률 — 피처용)
    + FRED 매크로 8개
    + GDELT 뉴스 감성
            ↓
[Step 2] 전처리 & 피처 엔지니어링
    기존 11개 자산 피처 + 추가 14개 피처 (Layer 1~4)
    + 거시 파생변수 17개 + GDELT 6개 + HMM 1개
            ↓
[Step 3] 모델링 — Q, Ω 도출
    XGBoost / TabPFN / RF + LLM Agent
    Walk-Forward (IS=150일 / Embargo=21일 / OOS=21일)
            ↓
[Step 4] Black-Litterman
    Prior π = λΣw_mkt (시가총액 비중)
    Q, Ω → μ_BL, Σ_BL
            ↓
[Step 5] MVO — 성향별 최적 비중
    γ별 5개 투자 성향 포트폴리오
            ↓
[Step 6] 백테스트 & 리스크 분석
            ↓
[Step 7] 동적 리밸런싱 & Streamlit 대시보드
```

---

## 3. 종목 선정 기준 (미확정 — 추가 논의 필요)

### GICS 11개 섹터

| 섹터 | 대표 종목 예시 |
|------|---------------|
| Information Technology | AAPL, MSFT, NVDA |
| Health Care | JNJ, UNH, LLY |
| Financials | JPM, BAC, WFC |
| Consumer Discretionary | AMZN, TSLA, HD |
| Communication Services | GOOGL, META, NFLX |
| Industrials | CAT, BA, HON |
| Consumer Staples | PG, KO, WMT |
| Energy | XOM, CVX, COP |
| Utilities | NEE, DUK, SO |
| Real Estate | PLD, AMT, EQIX |
| Materials | LIN, APD, SHW |

### 종목 수 결정 이슈

- 섹터별 **3~5종목**: 총 33~55개 → BL 공분산 행렬 연산 부담 검토 필요
- 섹터별 **1종목 (시총 1위)**: 총 11개 → 단순하나 분산 부족
- **결정 기준**: 학습 데이터 충분성 + 행렬 연산 안정성 + 섹터 대표성

### 종목 변경 규칙

```
매 IS 구간 시작 시점 기준, 각 섹터 시가총액 상위 N위 종목 자동 선정
→ 기간별 생존편향 완화 (현 시점 시총 상위종목만 사용하는 게 아님)
→ 단, 상장폐지 종목은 해당 윈도우에서 제외
```

---

## 4. Prior 정의 — 시가총액 비중

```python
import yfinance as yf
import numpy as np

def get_market_cap_weights(tickers: list[str], date: str) -> np.ndarray:
    """
    특정 날짜 기준 시가총액 비중.
    실무적 한계: yfinance가 과거 발행주식수를 제공하지 않아
    현재 발행주식수 × 과거 주가로 근사 (단순화).
    """
    market_caps = []
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        shares = info.get('sharesOutstanding', 0)
        hist = yf.download(ticker, start=date, end=date, progress=False)
        price = hist['Close'].iloc[0] if not hist.empty else 0
        market_caps.append(shares * price)
    caps = np.array(market_caps, dtype=float)
    return caps / caps.sum()


def compute_equilibrium_returns(Sigma: np.ndarray, w_mkt: np.ndarray,
                                 delta: float = 2.5) -> np.ndarray:
    """CAPM 역산: π = δ × Σ × w_mkt"""
    return delta * Sigma @ w_mkt
```

---

## 5. Embargo가 필요한 이유

### 핵심 문제: 라벨 겹침(Label Overlap)

우리가 예측하는 타겟은 **21일 선행 수익률**이다.

```
t 시점 타겟: fwd_ret_21d_t = ln(P_{t+21} / P_t)
```

즉, t 시점의 타겟을 계산하려면 **t+21일까지의 가격 데이터**가 필요하다.

### Walk-Forward 구조에서의 문제

```
IS 구간 끝           OOS 구간 시작
    ↓                     ↓
[─────────── IS: t=1 ~ t=150 ───────────][── OOS: t=151 ~ t=171 ──]
                                    ↑
                              t=130일 타겟 = ln(P_{t+21}/P_t) = ln(P_151 / P_130)
                              → 이 라벨을 만들 때 OOS 구간 가격(P_151)이 사용됨!
```

IS 구간 마지막 21일(t=130~150)의 라벨을 계산하는 데 OOS 구간(t=151~171)의 가격이 포함된다.

**모델이 IS 구간에서 이 샘플들로 학습하면, 사실상 미래를 보고 학습하는 것이다.**

### Embargo의 역할

Embargo는 IS-OOS 경계 직전 21일을 학습에서 제거한다.

```
IS 구간               Embargo      OOS 구간
[─────── 학습 가능 (t=1~129) ───────][제외: 130~150][── 예측: 151~171 ──]
```

- t=130~150의 타겟은 P_{151}~P_{171}을 참조 → OOS 가격이 라벨에 포함 → 제거
- t=1~129의 타겟은 P_{22}~P_{150}만 참조 → IS 구간 내에서 완결 → 학습 허용

### 직관적 비유

> 시험(OOS)이 내일이다. Embargo가 없으면 선생님이 시험지를 미리 보고 수업(IS)에서 힌트를 줄 수 있는 상황이다. Embargo는 시험지가 완성된 후 수업에서 가르친 내용은 사용하지 못하게 막는 규칙이다.

### 코드에서의 구현

```python
IS_DAYS    = 150   # 학습 기간
EMBARGO    = 21    # 제거할 경계 구간 (타겟 horizon과 동일하게 설정)
OOS_DAYS   = 21    # 예측 기간

# 실제 학습에 사용되는 IS 구간: IS_DAYS - EMBARGO = 129일
# (마지막 21일은 라벨이 OOS를 참조하므로 제거)
train_end   = train_start + IS_DAYS - EMBARGO
oos_start   = train_start + IS_DAYS            # Embargo 이후
oos_end     = oos_start + OOS_DAYS
```

> **결론**: Embargo를 빠뜨리면 모델 성과가 지나치게 좋게 측정되고(낙관적 편향), 실제 운용 시 성과가 급락한다. 우리 프로젝트처럼 21일 선행 타겟을 쓰면 Embargo도 반드시 21일이어야 한다.

---

## 6. 핵심 함수 이해: pct_change()와 rolling()

### 6-1. pct_change() — 일별 수익률 추출

`pct_change()`는 pandas의 내장 함수로, 이전 행 대비 **퍼센트 변화율**을 계산한다.

```python
# 수식: pct_change() = (현재값 - 이전값) / 이전값 = P_t / P_{t-1} - 1
returns = df_price.pct_change()
```

**단계별 이해:**

```python
import pandas as pd

# 예시: AAPL 주가
prices = pd.Series([100, 102, 99, 105], name='AAPL')

# pct_change() 내부 동작
# (102 - 100) / 100 = 0.02  → 2% 상승
# (99  - 102) / 102 = -0.029 → 2.9% 하락
# (105 - 99)  / 99  = 0.061  → 6.1% 상승
print(prices.pct_change())
# 0      NaN   ← 첫 행은 이전값이 없어 NaN
# 1     0.020
# 2    -0.029
# 3     0.061
```

**왜 로그 수익률 대신 단순 수익률을 쓰는 경우도 있는가?**

```python
# 단순 수익률: pct_change()
simple_ret = df_price.pct_change()          # r_t = P_t/P_{t-1} - 1

# 로그 수익률: log(P_t / P_{t-1})
log_ret    = np.log(df_price / df_price.shift(1))  # r_t = ln(P_t/P_{t-1})
```

| 구분 | 단순 수익률 | 로그 수익률 |
|------|-----------|-----------|
| 계산 | pct_change() | np.log(...shift(1)) |
| 합산 | 불가 (기간 누적 시 곱셈 필요) | 가능 (단순 덧셈으로 누적) |
| 정규성 | 약간 낮음 | 더 높음 (ML 모델에 유리) |
| 피처 계산 | 롤링 Sharpe, Sortino 등 대부분 사용 | 모멘텀(ret_1m ~ ret_12m) 계산 시 사용 |

**DataFrame에 적용하면 모든 컬럼에 동시 적용된다:**

```python
# df_price: (날짜 × 티커) 행렬
#            AAPL   MSFT   SPY
# 2020-01-02  300    160   320
# 2020-01-03  306    158   322
# 2020-01-06  303    162   319

returns = df_price.pct_change()
#            AAPL    MSFT    SPY
# 2020-01-02  NaN     NaN    NaN
# 2020-01-03  0.020  -0.013  0.006
# 2020-01-06 -0.010   0.025 -0.009
# → 한 번의 호출로 모든 종목의 일별 수익률이 동시에 계산됨
```

---

### 6-2. rolling() — 롤링 윈도우 계산

`rolling(window)`는 **슬라이딩 윈도우** 방식으로 통계를 계산하는 pandas 함수다.  
window=21이면 "오늘 포함 최근 21일"의 데이터를 묶어서 통계를 계산하고, 하루씩 밀면서 반복한다.

```
날짜    수익률   rolling(21).mean()
1일     0.01     NaN   ← 21개 미만이라 NaN
2일     0.02     NaN
...
21일    0.03     0.018  ← 1~21일의 평균
22일   -0.01     0.017  ← 2~22일의 평균  (1일 데이터 빠지고 22일 데이터 추가)
23일    0.00     0.016  ← 3~23일의 평균
```

**Rolling Sharpe Ratio 코드 분해:**

```python
sharpe_21d = (ret.rolling(21).mean() / ret.rolling(21).std()) * np.sqrt(252)
```

이 한 줄을 단계별로 분리하면:

```python
# Step 1: 최근 21일 평균 수익률
# → "최근 21일 동안 하루 평균 얼마나 벌었나?"
mean_21d = ret.rolling(21).mean()

# Step 2: 최근 21일 수익률 표준편차
# → "최근 21일 동안 하루 수익률이 얼마나 들쭉날쭉했나?"
std_21d  = ret.rolling(21).std()

# Step 3: 평균 / 표준편차 = 일별 Sharpe
# → "위험(표준편차) 1단위당 얼마나 벌었나?"
daily_sharpe = mean_21d / std_21d

# Step 4: √252 곱해서 연환산
# → "이 효율이 1년 내내 지속된다면 연간 기준으로 얼마나 되나?"
# (거래일 기준 1년 ≈ 252일, 일별 → 연간 환산 시 √252 곱함)
sharpe_21d = daily_sharpe * np.sqrt(252)
```

**Rolling Sortino 코드 분해:**

```python
downside = ret.copy()
downside[downside > 0] = 0            # 양수 수익률 → 0으로 마스킹
sortino = (ret.rolling(63).mean() / downside.rolling(63).std()) * np.sqrt(252)
```

```python
# 왜 양수를 0으로 마스킹하는가?
# Sharpe: 표준편차 = 위아래 변동 모두 포함
# Sortino: 표준편차 = 하락일만 포함 (= 하방 변동성, Downside Deviation)

# 예시
daily_returns = [0.02, -0.03, 0.01, -0.05, 0.04]
downside_only = [0.00, -0.03, 0.00, -0.05, 0.00]  # 양수는 0 처리

# Sortino의 분모 = downside_only의 표준편차
# → 하락이 없거나 적으면 분모가 작아져 Sortino가 높아짐
# → "얼마나 올랐냐"보다 "얼마나 덜 빠졌냐"를 더 중시
```

**핵심 요약:**

```
rolling(21).mean()  → 최근 21일 평균 (= 수익률 추세)
rolling(21).std()   → 최근 21일 표준편차 (= 변동성)
rolling(21).skew()  → 최근 21일 왜도 (= 분포 비대칭성)
rolling(21).kurt()  → 최근 21일 첨도 (= 꼬리 두께)
rolling(21).max()   → 최근 21일 최대값 (= 고점)
rolling(21).min()   → 최근 21일 최소값 (= 저점)
rolling(63).apply(fn) → 최근 63일 데이터를 fn에 통째로 넘겨서 계산
```

---

## 7. 피처 엔지니어링 전체

### 기본 설정

```python
import pandas as pd
import numpy as np

# df_price:  index=날짜(DatetimeIndex), columns=티커, values=종가
# df_volume: index=날짜, columns=티커, values=거래량
# sector_map: dict — {티커: 섹터명}  예) {'AAPL': 'IT', 'JPM': 'Financials'}

returns = df_price.pct_change()       # 일별 단순 수익률 (Wide format)
log_ret = np.log(df_price / df_price.shift(1))  # 일별 로그 수익률
spy_ret = returns['SPY']              # 시장 기준 수익률
```

---

### Part A. 기존 자산별 피처 (11개)

#### A-1. 모멘텀 피처 (4개)

```python
ch['ret_1m']  = np.log(p / p.shift(21))    # 1개월(21거래일) 수익률
ch['ret_3m']  = np.log(p / p.shift(63))    # 3개월 수익률
ch['ret_6m']  = np.log(p / p.shift(126))   # 6개월 수익률
ch['ret_12m'] = np.log(p / p.shift(252))   # 12개월 수익률
```

**왜 필요한가?**

모멘텀(과거에 잘 오른 종목은 앞으로도 잘 오르는 경향)은 금융에서 가장 오래되고 강력한 알파 팩터다(Jegadeesh & Titman, 1993). ML 모델에 시간대별 모멘텀을 여러 개 주면 단기/중기/장기 추세를 각각 학습한다.

- `ret_1m`: 단기 추세 (단기 반전과 모멘텀 지속 신호 혼재)
- `ret_3m`: 가장 강한 모멘텀 신호로 알려진 구간
- `ret_6m`, `ret_12m`: 중장기 추세 (섹터 로테이션, 매크로 사이클 반영)

**Black-Litterman 연결**: 모멘텀이 높은 종목은 Q(기대수익률)가 높아질 가능성 → ML이 높은 Q를 예측하도록 학습하는 핵심 신호.

```
주의: 학술 관례(Jegadeesh-Titman)는 ret_12m 계산 시 최근 1개월을 제외(skip-month)함.
이유: 최근 1개월 수익률은 단기 반전 효과가 섞여 있어 장기 모멘텀 신호를 오염시킴.
우리 코드: skip-month 없음 → XGBoost는 단기 반전도 하나의 신호로 학습 가능하므로 치명적 오류는 아님.
필요 시: ret_12m_skip = np.log(p.shift(21) / p.shift(252)) 로 수정.
```

#### A-2. 변동성 피처 (3개)

```python
ch['vol_21d']  = r.rolling(21).std() * np.sqrt(252)   # 단기 연환산 변동성
ch['vol_63d']  = r.rolling(63).std() * np.sqrt(252)   # 중기 연환산 변동성
ch['vol_ratio'] = ch['vol_21d'] / ch['vol_63d']        # 변동성 가속도
```

**왜 필요한가?**

- `vol_21d`, `vol_63d`: 변동성은 Black-Litterman Ω(불확실성)과 직접 연결된다. 변동성이 높은 자산은 수익률 예측이 더 불확실하므로 Ω를 키워야 한다.
- `vol_ratio`: 단기/중기 변동성 비율. `> 1`이면 최근 변동성이 평소보다 급등(위기 신호), `< 1`이면 평온. 레짐 전환 감지에 유용하다.

```
vol_ratio > 1.5 → 최근 1개월 변동성이 3개월 평균의 1.5배 → 위기 레짐 진입 가능성
vol_ratio < 0.8 → 변동성 수렴 → 안정 레짐
```

#### A-3. 베타 피처 (2개)

```python
def rolling_beta(asset_ret, market_ret, window):
    """β = Cov(r_i, r_mkt) / Var(r_mkt)"""
    cov = asset_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    return cov / var

ch['beta_60d']  = rolling_beta(r, spy_ret, 60)
ch['beta_120d'] = rolling_beta(r, spy_ret, 120)
```

**왜 필요한가?**

Beta는 시장이 1% 움직일 때 이 종목이 얼마나 움직이는지를 나타낸다. ML 모델이 공격형/방어형 자산을 구분하는 핵심 피처다.

| Beta 값 | 의미 | 예시 |
|---------|------|------|
| > 1.5 | 시장보다 크게 움직임 (공격형) | NVDA, TSLA |
| ≈ 1.0 | 시장과 동행 | SPY |
| < 0.5 | 방어형, 시장과 낮은 상관 | PG, KO |
| < 0 | 위기 시 상승 (헤지 자산) | 금, TLT |

두 가지 윈도우를 쓰는 이유: 60일 beta는 최근 국면, 120일 beta는 구조적 특성을 포착. 괴리가 크면 최근 성격이 바뀌었다는 신호.

#### A-4. 금리 민감도 (1개)

```python
ch['rate_corr_60d'] = r.rolling(60).corr(fred_df['DGS10_chg5'])
# DGS10_chg5: 미국 10년 국채 금리 5일 변화량
```

**왜 필요한가?**

금리 상승 시 채권 가격은 하락하고, 금융주는 상승하고, 유틸리티는 하락한다. 이 상관관계를 피처로 넣으면 ML 모델이 종목의 금리 민감 특성을 직접 학습한다.

```
rate_corr_60d > 0.1  → 금리 상승 수혜 (XLF 계열)
rate_corr_60d < -0.2 → 금리 상승 피해 (채권, 유틸리티)
≈ 0                  → 금리 영향 중립
```

#### A-5. RSI (1개) — 기존 피처

```python
def calc_rsi(price, window=14):
    delta = price.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()   # 상승일 평균
    loss  = (-delta.clip(upper=0)).rolling(window).mean() # 하락일 평균
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

ch['rsi_14d'] = calc_rsi(p, 14)
```

**왜 필요한가?**

RSI는 과매수/과매도 구간을 감지한다. 모멘텀 피처(ret_1m 등)만 있으면 "많이 오른 종목"을 무조건 좋다고 학습하지만, RSI를 함께 주면 "이미 너무 오른 종목은 조심"이라고 학습할 수 있다.

```
RSI > 70 → 과매수 → 단기 반전(하락) 가능성 높음
RSI < 30 → 과매도 → 단기 반전(상승) 가능성 높음
```

---

### Part B. 추가 피처 Layer 1: 위험 대비 수익의 질 (Risk-Adjusted Quality)

**이 레이어가 필요한 근본 이유:**

모멘텀만 있으면 모델은 "많이 올랐다 = 좋은 자산"이라고 학습한다. 그런데 변동성이 크면서 많이 오른 자산(예: 단기 급등 종목)은 위험 대비 효율이 낮다. "위험 1단위당 얼마나 벌었나"를 추가로 주면 모델이 더 효율적인 자산을 구분하게 된다. BL 관점에서는 Sharpe가 높은 종목에 모델이 더 확신 → Ω가 낮아져 해당 종목 뷰가 강하게 반영된다.

#### B-1. Rolling Sharpe Ratio

```python
# Sharpe = E[r] / σ[r] × √252
ch['sharpe_63d'] = (
    r.rolling(63).mean() / r.rolling(63).std()
) * np.sqrt(252)

# 단기 버전도 함께
ch['sharpe_21d'] = (
    r.rolling(21).mean() / r.rolling(21).std()
) * np.sqrt(252)
```

**왜 두 가지 윈도우인가?** `sharpe_21d`는 최근 국면의 수익 효율, `sharpe_63d`는 좀 더 안정적인 중기 효율. 둘의 괴리가 크면 최근 국면이 급변했다는 신호.

| Sharpe 값 | 의미 |
|-----------|------|
| > 1.0 | 양호: 위험 대비 충분한 수익 |
| 0 ~ 1.0 | 보통: 위험을 감수했으나 수익이 적음 |
| < 0 | 주의: 위험을 감수하고도 손실 발생 |

#### B-2. Rolling Sortino Ratio

```python
downside = r.copy()
downside[downside > 0] = 0  # 양수 수익률 → 0으로 마스킹

ch['sortino_63d'] = (
    r.rolling(63).mean() / downside.rolling(63).std()
) * np.sqrt(252)
```

**Sharpe와의 차이:** Sharpe는 위아래 변동을 모두 "위험"으로 본다. Sortino는 하락일 변동성만 위험으로 취급한다. "위로 크게 오른 날"은 위험이 아니므로 제외하는 것이 더 현실적이다.

```
하락장에서 덜 빠지는 종목 = 분모(하방 변동성)가 작음 = Sortino 높음
→ 방어력이 있는 자산을 식별하는 데 유효
```

#### B-3. Rolling Information Ratio (vs Sector)

```python
def calc_rolling_ir(ret, sector_map, window=63):
    """IR = E[초과수익] / σ[초과수익]"""
    ir_frames = {}
    for ticker in ret.columns:
        sector = sector_map.get(ticker)
        if sector is None:
            continue
        peers = [t for t in ret.columns if sector_map.get(t) == sector and t != ticker]
        if not peers:
            continue
        sector_mean = ret[peers].mean(axis=1)
        excess = ret[ticker] - sector_mean
        ir_frames[ticker] = (
            excess.rolling(window).mean() / excess.rolling(window).std()
        ) * np.sqrt(252)
    return pd.DataFrame(ir_frames)

ch['ir_63d'] = calc_rolling_ir(returns, sector_map, 63)[ticker]
```

**왜 필요한가?** 같은 IT 섹터 내에서도 어떤 종목은 꾸준히 섹터 평균을 이기고, 어떤 종목은 가끔 크게 이기고 가끔 크게 진다. IR은 이 "꾸준함"을 측정한다. IR이 높은 종목은 섹터 흐름과 무관하게 자체 경쟁력이 있다는 신호.

---

### Part C. 추가 피처 Layer 2: 꼬리 위험과 비대칭성 (Tail Risk & Asymmetry)

**이 레이어가 필요한 근본 이유:**

Black-Litterman 모델이 공분산 행렬(Σ)을 계산할 때 정규분포를 가정하지만, 실제 주가 수익률은 **Fat Tail**(극단적 사건이 정규분포보다 자주 발생)을 가진다. 수익률 분포의 모양을 피처로 넣으면 ML 모델이 Ω를 더 정확하게 설정하는 데 도움이 된다.

#### C-1. Rolling Skewness (왜도)

```python
ch['skew_63d'] = r.rolling(63).skew()
```

**해석:**

```
음수(← 왼쪽 꼬리) → 평소 소소한 상승 + 가끔 폭락 → 하락 충격 위험
양수(→ 오른쪽 꼬리) → 평소 소소한 하락 + 가끔 급등
0에 가까움 → 대칭 분포 (정규분포에 가까움)
```

BL 활용: 왜도가 음수로 크면 실제 손실 위험이 분산 추정치보다 크다는 뜻 → Ω를 상향 조정해야 하는 신호.

#### C-2. Rolling Kurtosis (첨도)

```python
ch['kurt_63d'] = r.rolling(63).kurt()
```

**해석:**

```
정규분포의 excess kurtosis = 0 (pandas .kurt()는 excess kurtosis 반환)
높을수록 → Fat Tail → 극단적 이벤트가 정규분포보다 자주 발생
낮을수록(< 0) → 얇은 꼬리 → 극단값이 드물게 발생
```

BL 활용: 첨도가 높은 자산은 Σ가 실제 위험을 과소추정 → Ω 키워야 하는 신호.

#### C-3. Rolling Maximum Drawdown (최대 낙폭)

```python
def rolling_mdd(prices, window=63):
    """
    윈도우 내 최고점 대비 현재 가격의 낙폭 중 최소값(= 최대 낙폭)
    prices: 종가 시계열 (수익률이 아닌 가격)
    """
    roll_max  = prices.rolling(window).max()   # 최근 window일 내 최고점
    drawdown  = prices / roll_max - 1           # 최고점 대비 낙폭 (음수)
    return drawdown.rolling(window).min()       # 낙폭 중 가장 깊은 지점

ch['mdd_63d'] = rolling_mdd(p, 63)
```

**왜 prices를 쓰는가?** returns를 cumprod()로 변환하는 방식도 있지만, 실제 가격에서 직접 계산하면 더 직관적이고 연산이 간단하다.

**해석:**

```
mdd_63d = -0.30 → 최근 63일간 고점 대비 최대 30% 하락한 적 있음
mdd_63d ≈ 0    → 최근 63일간 거의 하락 없음 (신고가 갱신 중)
```

BL 활용: MDD가 크면 투자자가 중간에 손절하고 나갈 가능성이 높다. 최근 MDD가 큰 종목은 모멘텀이 약화된 신호로 볼 수 있다.

---

### Part D. 추가 피처 Layer 3: 시장 연동성 (Connectivity & Sensitivity)

**이 레이어가 필요한 근본 이유:**

포트폴리오의 핵심 목표는 분산(Diversification)이다. 서로 다른 방향으로 움직이는 자산을 섞어야 위험이 줄어드는데, 모든 자산이 함께 움직이면(상관관계 → 1) 분산 효과가 사라진다. 위기 시에는 평소 낮던 상관관계가 1에 수렴하는 "상관관계 붕괴(Correlation Breakdown)" 현상이 나타난다.

#### D-1. Rolling Beta (to SPY) — 기존 beta 확장

기존 `beta_60d`, `beta_120d`가 이미 이 역할을 한다. (Part A-3 참고)

#### D-2. Correlation Matrix Centrality

```python
def calc_avg_correlation(ret, window=63):
    """
    각 날짜에서 해당 종목과 나머지 전체 종목의 평균 상관계수.
    계산량 주의: 종목 수가 많을수록 느림 → 주기적으로 계산하거나 근사치 사용 검토.
    """
    result = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for i in range(window, len(ret)):
        window_ret = ret.iloc[i-window:i]
        corr_mat   = window_ret.corr().values
        n          = corr_mat.shape[0]
        mask       = ~np.eye(n, dtype=bool)   # 대각선(자기 자신) 제외
        avg_corr   = corr_mat[mask].reshape(n, n-1).mean(axis=1)
        result.iloc[i] = avg_corr
    return result

ch['avg_corr_63d'] = calc_avg_correlation(returns, 63)[ticker]
```

**왜 필요한가:**

```
평상시: avg_corr ≈ 0.3  (자산들이 어느 정도 독립적으로 움직임)
위기 시: avg_corr → 0.8 이상으로 급등 (모든 자산이 함께 빠짐 → 분산 효과 소멸)

→ avg_corr이 갑자기 높아지는 것 자체가 위기 레짐 진입 신호
→ avg_corr이 낮은 종목 = 포트폴리오에 분산 기여도 높음 (분산 투자 측면에서 유리)
```

---

### Part E. 추가 피처 Layer 4: 유동성 및 기술적 과열 (Liquidity & Mean-Reversion)

**이 레이어가 필요한 근본 이유:**

좋은 종목이라도 두 가지 이유로 투자하기 어려울 수 있다: (1) 살 수 없거나(유동성 부족), (2) 이미 너무 비싸거나(과매수). 이를 피처로 주면 모델이 실제로 투자 가능한 자산을 더 잘 선택한다.

#### E-1. Amihud Illiquidity

```python
def calc_amihud(ret, price, volume, window=21):
    """
    Amihud(2002): |일별 수익률| / 거래대금
    거래대금 = 종가 × 거래량
    """
    dollar_volume = price * volume
    illiquidity   = ret.abs() / dollar_volume
    return illiquidity.rolling(window).mean()

ch['amihud_21d'] = calc_amihud(r, p, vol, 21)
```

**해석:**

```
값이 클수록 → 적은 돈으로도 가격이 크게 흔들림 → 유동성 부족
값이 작을수록 → 많은 거래량에도 가격 안정 → 유동성 풍부

유동성 부족 종목은 실제 매매 시 시장충격(market impact)이 크고,
원하는 가격에 전량 매수/매도가 어려움 → MVO 비중 제약 강화 필요
```

#### E-2. 볼린저 밴드 %B

```python
bb_mid = p.rolling(20).mean()
bb_std = p.rolling(20).std()
ch['bb_pct'] = (p - (bb_mid - 2*bb_std)) / (4 * bb_std)
# 동일 표현: (p - lower) / (upper - lower)
```

**해석:**

```
bb_pct = 1.0 → 상단 밴드 (과매수)
bb_pct = 0.5 → 이동평균선 (중립)
bb_pct = 0.0 → 하단 밴드 (과매도)
bb_pct > 1.0 → 상단 밴드 돌파 (강한 모멘텀, 또는 과열 경보)
bb_pct < 0.0 → 하단 밴드 돌파 (강한 하락, 또는 반등 기회)
```

RSI와 함께 사용: 둘 다 과매수 신호면 더 강한 반전 위험 경보.

---

### Part F. 추가 피처 Layer 5: 구조적 신호 (Structure & Rank)

**이 레이어가 필요한 근본 이유:**

절대적 수익률 수치는 자산마다 스케일이 달라 직접 비교가 어렵다. "유니버스 내에서 상대적으로 얼마나 잘했냐"와 "추세의 구조가 어떤가"를 피처로 주면 모델이 더 안정적으로 학습한다.

#### F-1. 52주 신고가 대비 현재 위치

```python
ch['high52w_ratio'] = p / p.rolling(252).max()
```

**해석:**

```
1.0에 가까울수록 → 신고가 근처 (모멘텀 지속 신호)
0.7 이하          → 고점 대비 30% 이상 하락 (역모멘텀, 하락 추세)
```

단순 12개월 수익률(ret_12m)과 다른 점: ret_12m은 정확히 12개월 전 대비 수익률이지만, high52w_ratio는 "최근 1년 중 가장 좋았던 시점 대비 현재 위치"를 나타낸다. 투자자 심리에서 신고가는 중요한 심리적 저항/지지선이다.

#### F-2. 이동평균 괴리율 (MA Cross)

```python
ch['ma_gap_20_60'] = p.rolling(20).mean() / p.rolling(60).mean() - 1
```

**해석:**

```
양수 → 단기 MA가 장기 MA 위 (골든크로스 방향 → 상승 추세)
음수 → 단기 MA가 장기 MA 아래 (데드크로스 방향 → 하락 추세)
```

기존 `ret_1m`과의 차이: ret_1m은 특정 날 대비 수익률이지만, ma_gap은 "추세의 기울기"를 부드럽게 포착한다. 노이즈에 덜 민감하다.

#### F-3. 수익률 자기상관 (Autocorrelation)

```python
ch['autocorr_21d'] = r.rolling(63).apply(
    lambda x: x.autocorr(lag=21), raw=False
)
```

**해석:**

```
양수 → 21일 전 수익률이 오늘 수익률과 같은 방향 → 모멘텀 지속성
음수 → 21일 전 수익률이 오늘 수익률과 반대 방향 → 평균 회귀(mean-reversion)
```

```
성장주(NVDA, TSLA): 보통 양의 자기상관 → 모멘텀 지속
방어주/채권: 음의 자기상관 → 평균 회귀 경향
```

BL 활용: 자기상관이 양수인 종목은 모멘텀 신호(ret_1m 등)를 더 신뢰할 수 있음 → Ω 감소 요인.

#### F-4. 유니버스 내 수익률 순위 (Cross-Sectional Rank)

```python
# 패널 데이터에서 날짜별 전체 종목 수익률 순위 (0~1)
panel_df['ret_rank_1m'] = panel_df.groupby('date')['ret_1m'].rank(pct=True)
panel_df['ret_rank_3m'] = panel_df.groupby('date')['ret_3m'].rank(pct=True)

# 유니버스 내 변동성 순위
panel_df['vol_rank'] = panel_df.groupby('date')['vol_21d'].rank(pct=True)
```

**왜 필요한가?**

절대 수익률(ret_1m = 0.05)은 "5% 올랐다"는 정보만 주지만, 순위(ret_rank_1m = 0.9)는 "전체 종목 중 상위 10%"라는 상대 정보를 준다. 상대 정보가 더 안정적인 신호다.

```
2020년 3월: 모든 종목이 폭락 → ret_1m은 모두 -30% ~ -50%
그러나 ret_rank는 여전히 0~1로 분포 → 상대적으로 덜 빠진 종목을 구분할 수 있음
```

`vol_rank`: 유니버스 내에서 이 종목이 얼마나 위험한지를 상대적으로 표현. 성향별 포트폴리오(보수형은 vol_rank 낮은 종목 선호) 구성에 직접 활용 가능.

#### F-5. 고유 변동성 (Idiosyncratic Volatility)

```python
def idiosyncratic_vol(r, market_ret, window=63):
    """
    시장(SPY) 움직임으로 설명되지 않는 종목 고유 변동성.
    높을수록 종목 특수 리스크가 큼 → 예측 불확실성 높음.
    """
    resid = pd.Series(index=r.index, dtype=float)
    for i in range(window, len(r)):
        y    = r.iloc[i-window:i].values
        x    = market_ret.iloc[i-window:i].values
        beta = np.cov(y, x)[0, 1] / np.var(x)
        alpha = y.mean() - beta * x.mean()
        resid.iloc[i] = r.iloc[i] - (alpha + beta * market_ret.iloc[i])
    return resid.rolling(window).std() * np.sqrt(252)

ch['ivol_63d'] = idiosyncratic_vol(r, spy_ret, 63)
```

**왜 필요한가?**

일반 변동성(vol_21d)은 시장 전체가 흔들릴 때의 변동도 포함한다. ivol은 시장 움직임을 제거한 뒤 남은 변동성으로, 순수하게 해당 종목만의 리스크를 나타낸다.

```
이벤트 리스크: 실적 발표, FDA 승인, 규제 이슈 등 종목 특수 사건
= ivol이 높은 구간에서 집중 발생

BL 관점: ivol이 높으면 ML이 예측한 Q의 신뢰도 낮음 → Ω 상향 조정 신호
→ 고유 변동성이 클수록 시장 균형(Prior π)으로 회귀하는 것이 안전
```

**Amihud와의 차이:**

| 피처 | 측정 대상 |
|------|----------|
| `amihud_21d` | 거래량 대비 가격 변동 → 유동성 리스크 |
| `ivol_63d` | 시장 움직임 제거 후 잔차 → 고유 리스크 |

---

### 전체 피처 통합 — Long-Panel 구성

```python
def build_full_feature_panel(df_price, df_volume, fred_df, gdelt_df,
                              sector_map: dict) -> pd.DataFrame:
    """
    모든 피처를 날짜 × 종목 Long-Panel로 통합.
    출력: date, ticker, [피처들], sector
    """
    returns = df_price.pct_change()
    spy_ret = returns['SPY']

    feat = {}

    # ── Part A: 기존 피처 (11개) ────────────────────────────────
    feat['ret_1m']       = np.log(df_price / df_price.shift(21))
    feat['ret_3m']       = np.log(df_price / df_price.shift(63))
    feat['ret_6m']       = np.log(df_price / df_price.shift(126))
    feat['ret_12m']      = np.log(df_price / df_price.shift(252))
    feat['vol_21d']      = returns.rolling(21).std() * np.sqrt(252)
    feat['vol_63d']      = returns.rolling(63).std() * np.sqrt(252)
    feat['vol_ratio']    = feat['vol_21d'] / feat['vol_63d']
    feat['beta_60d']     = returns.apply(lambda r: r.rolling(60).cov(spy_ret) / spy_ret.rolling(60).var())
    feat['beta_120d']    = returns.apply(lambda r: r.rolling(120).cov(spy_ret) / spy_ret.rolling(120).var())
    feat['rsi_14d']      = calc_rsi(df_price, 14)

    # ── Part B: Layer 1 — 위험 대비 수익 (4개) ──────────────────
    feat['sharpe_21d']   = (returns.rolling(21).mean() / returns.rolling(21).std()) * np.sqrt(252)
    feat['sharpe_63d']   = (returns.rolling(63).mean() / returns.rolling(63).std()) * np.sqrt(252)
    downside = returns.copy(); downside[downside > 0] = 0
    feat['sortino_63d']  = (returns.rolling(63).mean() / downside.rolling(63).std()) * np.sqrt(252)
    feat['ir_63d']       = calc_rolling_ir(returns, sector_map, 63)

    # ── Part C: Layer 2 — 꼬리 위험 (3개) ──────────────────────
    feat['skew_63d']     = returns.rolling(63).skew()
    feat['kurt_63d']     = returns.rolling(63).kurt()
    feat['mdd_63d']      = df_price.apply(lambda p: rolling_mdd(p, 63))

    # ── Part D: Layer 3 — 시장 연동성 (1개) ─────────────────────
    feat['avg_corr_63d'] = calc_avg_correlation(returns, 63)

    # ── Part E: Layer 4 — 유동성/과열 (2개) ─────────────────────
    feat['amihud_21d']   = calc_amihud(returns, df_price, df_volume, 21)
    bb_mid = df_price.rolling(20).mean()
    bb_std = df_price.rolling(20).std()
    feat['bb_pct']       = (df_price - (bb_mid - 2*bb_std)) / (4 * bb_std)

    # ── Part F: Layer 5 — 구조적 신호 (5개) ─────────────────────
    feat['high52w_ratio'] = df_price / df_price.rolling(252).max()
    feat['ma_gap_20_60']  = df_price.rolling(20).mean() / df_price.rolling(60).mean() - 1
    feat['autocorr_21d']  = returns.rolling(63).apply(lambda x: x.autocorr(lag=21), raw=False)
    feat['ivol_63d']      = returns.apply(lambda r: idiosyncratic_vol(r, spy_ret, 63))

    # ── Wide → Long 변환 ──────────────────────────────────────
    long_frames = []
    for name, wide_df in feat.items():
        if isinstance(wide_df, pd.Series):
            wide_df = wide_df.to_frame(name=name)
        stacked = wide_df.stack().rename(name) if isinstance(wide_df, pd.DataFrame) else wide_df.rename(name)
        long_frames.append(stacked)

    df_long = pd.concat(long_frames, axis=1)
    df_long.index.names = ['date', 'ticker']

    # ── 패널에서 날짜별 순위 피처 (Part F 계속) ──────────────────
    df_long = df_long.reset_index()
    df_long['ret_rank_1m'] = df_long.groupby('date')['ret_1m'].rank(pct=True)
    df_long['ret_rank_3m'] = df_long.groupby('date')['ret_3m'].rank(pct=True)
    df_long['vol_rank']    = df_long.groupby('date')['vol_21d'].rank(pct=True)

    # ── 거시/GDELT 피처 merge ─────────────────────────────────
    df_long = df_long.merge(fred_df, on='date', how='left')
    if gdelt_df is not None:
        df_long = df_long.merge(gdelt_df, on='date', how='left')

    df_long['sector'] = df_long['ticker'].map(sector_map)

    return df_long.dropna(subset=['ret_1m'])
```

---

## 8. 전체 피처 목록 및 BL 활용 맵

### 자산별 특성 피처 (총 30개)

| 파트 | 피처명 | 윈도우 | 의미 | BL 역할 |
|------|--------|--------|------|---------|
| A 기본 | `ret_1m/3m/6m/12m` | 21~252d | 모멘텀 | Q 핵심 신호 |
| A 기본 | `vol_21d`, `vol_63d` | - | 변동성 | Ω 보조 |
| A 기본 | `vol_ratio` | - | 변동성 가속도 | 레짐 신호 |
| A 기본 | `beta_60d`, `beta_120d` | 60/120d | 시장 민감도 | 성향별 배분 |
| A 기본 | `rate_corr_60d` | 60d | 금리 민감도 | 금리 레짐 반응 |
| A 기본 | `rsi_14d` | 14d | 과매수/과매도 | 모멘텀 과열 필터 |
| B L1 | `sharpe_21d`, `sharpe_63d` | 21/63d | 위험 대비 수익 | Ω 감소 신호 |
| B L1 | `sortino_63d` | 63d | 하락 위험 대비 수익 | 방어력 측정 |
| B L1 | `ir_63d` | 63d | 섹터 대비 초과 수익 꾸준함 | Q 보조 |
| C L2 | `skew_63d` | 63d | 왜도 | Ω 상향 신호 |
| C L2 | `kurt_63d` | 63d | 첨도(Fat Tail) | Ω 상향 신호 |
| C L2 | `mdd_63d` | 63d | 최대 낙폭 | 모멘텀 약화 신호 |
| D L3 | `avg_corr_63d` | 63d | 평균 상관계수 | 위기 레짐 신호, 분산 기여 |
| E L4 | `amihud_21d` | 21d | 유동성 | MVO 비중 제약 |
| E L4 | `bb_pct` | 20d | 볼린저 밴드 위치 | 과열 필터 |
| F L5 | `high52w_ratio` | 252d | 52주 신고가 대비 위치 | 모멘텀 구조 |
| F L5 | `ma_gap_20_60` | - | 이동평균 괴리율 | 추세 기울기 |
| F L5 | `autocorr_21d` | 63d | 수익률 자기상관 | 모멘텀 지속성 |
| F L5 | `ivol_63d` | 63d | 고유 변동성 | Ω 상향 신호 |
| F L5 | `ret_rank_1m`, `ret_rank_3m` | - | 상대 수익률 순위 | 크로스섹셔널 신호 |
| F L5 | `vol_rank` | - | 상대 변동성 순위 | 성향별 배분 보조 |

> **총 자산별 피처: 30개** (기존 11개 + Layer 1~5 추가 19개)

### 거시 공통 피처 (17개, 기존 유지)

VIX_level, VIX_contango, VIX_slope_9d_3m, VIX_slope_3m_6m, SKEW_level, SKEW_zscore,  
Cu_Au_ratio, Cu_Au_ratio_chg, HY_spread, HY_spread_chg, yield_curve, yield_curve_inv,  
claims_4wma, claims_zscore, WEI_level, sahm_indicator, DGS10_chg5

### GDELT 감성 피처 (6개, 기존 유지)

fin_sentiment, sent_momentum, sent_volatility, sent_zscore, news_spike, crisis_signal

### HMM 레짐 피처 (1개)

hmm_crisis_prob (decision_log 17장 설계 기준, p=7)

**총 피처 수: 30 + 17 + 6 + 1 = 54개**

> TabPFN 피처 수 제한(≤100개)에는 여유. IC 분석 후 하위 피처 제거 검토.

---

## 9. 미결 사항 및 액션 아이템

| 담당 | 내용 | 기한 |
|------|------|------|
| @김재천 | 30개 산업 포트폴리오 논문 Prior 정의 방법 상세 확인 | 가능한 빨리 |
| @김재천 | 튜터님께 전달할 전체 프로세스 정리 및 컨펌 | 가능한 빨리 |
| @윤서 김, @김하연 | 섹터별 상위 종목 수 최종 결정 (3개 vs 5개) | 추가 논의 |
| @윤서 김, @김하연 | 개별 종목 데이터 수집 (Step1 업데이트) | 종목 확정 후 |
| @윤서 김, @김하연 | 추가 피처(Layer 1~5) 코드 Step2에 통합 | 데이터 수집 후 |
| 전원 | 패널 방식 vs 자산별 개별 모델링 최종 결정 | 추가 논의 |
| 전원 | 예측 기간(1개월 vs 3개월), 리밸런싱 주기 확정 | 추가 논의 |
| 전원 | `ivol_63d` 계산 속도 이슈 확인 (종목 수 많으면 느림) | 구현 시 |

---

## 피처 계층 구조 — "포트폴리오에 실제로 좋은 자산이란?"

```
수익률 (방향)
    └─ A. 모멘텀 (ret_1m/3m/6m/12m, high52w_ratio, ma_gap_20_60)
              ← "최근 얼마나 올랐나?"

         └─ B. L1 위험 보정 (sharpe, sortino, ir)
                   ← "잘 올랐나? (위험 대비)"

              └─ C. L2 꼬리 위험 (skew, kurt, mdd)
                        ← "안전하게 올랐나? (분포 모양)"

                   └─ D. L3 시장 연동성 (beta, avg_corr, ivol, autocorr)
                              ← "혼자 올랐나? (독립성)"

                        └─ E. L4 유동성/과열 (amihud, bb_pct, rsi)
                                   ← "살 수 있나? (실행 가능성)"

                             └─ F. L5 구조/순위 (ret_rank, vol_rank)
                                        ← "상대적으로 얼마나 좋은가?"
```

각 레이어를 추가할수록 단순히 **"많이 오른 자산"**이 아니라  
**"포트폴리오에 실제로 좋은 자산"**을 구분하는 모델을 만들 수 있다.

---

*이 문서는 2026-04-20 팀 회의 결정사항과 피처 엔지니어링 논의를 반영한 v4 설계 기준입니다.*  
*세부 파라미터(종목 수, 예측 기간 등)는 추가 논의 후 업데이트 예정.*
