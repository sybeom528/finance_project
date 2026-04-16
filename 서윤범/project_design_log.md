# 포트폴리오 최적화 프로젝트 설계 문서 (v2)
> 최초 작성: 2026-04-15 | 최종 수정: 2026-04-15 (v3)  
> Claude + NotebookLM 대화 기반 통합 정리

---

## 목차

1. [프로젝트 철학 및 전체 파이프라인](#1-프로젝트-철학-및-전체-파이프라인)
2. [확정 사항](#2-확정-사항)
   - 2-1. 투자 유니버스 및 데이터 수집
   - 2-2. 학습 구조: 롤링 윈도우
   - 2-3. 데이터셋 구조: Long-Panel
   - 2-4. Feature Engineering
   - 2-5. EDA 및 통계 검정
   - 2-6. ML 모델링 및 Q/Ω 도출
   - 2-7. Black-Litterman 통합
   - 2-8. 백테스팅 및 평가지표
3. [미결 고려 사항](#3-미결-고려-사항)
4. [약점 및 주의사항](#4-약점-및-주의사항)
5. [참고 논문 활용 매핑](#5-참고-논문-활용-매핑)
6. [단계별 진행 현황](#6-단계별-진행-현황)

---

## 1. 프로젝트 철학 및 전체 파이프라인

### 핵심 철학

주식 수익률의 낮은 SNR(신호 대 잡음비)을 인정하되,  
**"예측이 맞을 때는 과감하게, 틀릴 때는 시장에 맡긴다"**는 Black-Litterman의 베이지안 구조로 방어.

### 전체 파이프라인

```
[Step 1] 데이터 수집
  Yahoo Finance (30자산 + 외부12 + yf대안5) + FRED(8) + GDELT(텍스트/감성)
         ↓
[Step 2] 전처리 + Feature Engineering
  수익률 계산 → 정상성 검정 → 파생변수 생성(15개)
  + 자산별 모멘텀/변동성 피처 + 교차항(Interaction Terms)
  + GDELT 감성 피처 (→ ML 입력)
  + HMM 레짐 확률 피처 (→ ML 입력, 연속형 0~1)
         ↓
[Step 3] 모델링
  3.1 세 가지 뷰(View) 생성 엔진
      ① 정형데이터 기반: RF(회귀) + XGBoost(분류) + TabPFN(분류)
      ② LLM Agent: 뉴스 요약 + 수익률 → Q, Ω 직접 반환
  3.2 Q (기대수익률), Ω (불확실성) 통합
  3.3 BL 사후분포 계산 → MV 최적화
      + hmm_crisis_prob 피처 기반 (Σ 교체는 ablation으로 별도 비교)
      + 회전율 페널티(Turnover Penalty) 제약 추가
         ↓
[Step 4] Walk-Forward 백테스팅 (IS 150일, OOS 30일 롤링)
         ↓
[Step 5] Risk Analysis + HMM 기반 Circuit Breaker Alert
         ↓
[Step 6] 동적 리밸런싱 + Streamlit 대시보드
```

---

## 2. 확정 사항

### 2-1. 투자 유니버스 및 데이터 수집

| 구분 | 종목 | 개수 | 소스 |
|------|------|------|------|
| 인덱스 ETF | SPY, QQQ, IWM, EFA, EEM | 5 | yfinance |
| 채권 ETF | TLT, AGG, SHY, TIP | 4 | yfinance |
| 대안 ETF | GLD, DBC | 2 | yfinance |
| 섹터 ETF | XLK, XLF, XLE, XLV, VOX, XLY, XLP, XLI, XLU, XLRE, XLB | 11 | yfinance |
| 개별 종목 | 8개 GICS 섹터 시총 1위 (매년 1월 재구성) | 8 | yfinance |
| 외부 지표 | ^VIX, ^VIX9D, ^VIX3M, ^VIX6M, ^SKEW, HG=F, CL=F, GC=F, SI=F, BTC-USD, ETH-USD, DX-Y.NYB | 12 | yfinance |
| FRED 매크로 | BAMLH0A0HYM2, T10Y2Y, ICSA, WEI, SAHMREALTIME 외 3개 | 8 | FRED |
| GDELT | 뉴스 기사 텍스트, 감성 분석 결과 | - | GDELT API |

- 분석 기간: 2016-01-01 ~ 2025-12-31 (10년, ~2,520 영업일)
- ETH-USD: 투자 대상 아님, 외부 지표(감성 proxy)로만 포함
- WEI: 2020-03 이전 복합 스코어에서 제외 (소급 데이터 look-ahead bias 방지)

---

### 2-2. 학습 구조: 롤링 윈도우 ✅ 확정

```
┌────────────────────────┬────────────┐
│  학습(Train): 150일     │ 예측: 30일  │ → 슬라이딩
└────────────────────────┴────────────┘
                         ↓ 30일 경과
┌────────────────────────┬────────────┐
│  학습(Train): 150일     │ 예측: 30일  │ → 재학습(Rolling Update)
└────────────────────────┴────────────┘
```

- **학습 기간**: 가장 최근 150 영업일 (~7개월)의 패널 데이터
  - 150일 × 30자산 = 4,500개 관측치 (단일 모델 학습)
- **예측 기간**: 다음 30 영업일(~1개월) 동안 포트폴리오 비중 고정 운용
- **재학습**: 30일마다 모델 전체 재학습 (시장 트렌드 최신화)
- **총 윈도우 수**: (2,520 - 150) / 30 ≈ **79개 윈도우**

> 💡 **Purged Walk-Forward**: IS/OOS 경계 근처 21일 embargo 기간 설정  
> → 월간 수익률 레이블이 겹치는 구간의 look-ahead bias 방지 (Lopez de Prado 방법론)

---

### 2-3. 데이터셋 구조: Long-Panel ✅ 확정

**인덱스**: `[date(월말), ticker]` — Pooled Panel 방식

```
date       | ticker | ret_1m | ret_3m | vol_21d | vol_63d | volume_zscore | sector_code | VIX_level | HY_spread | VIX_contango | ... | gdelt_tone | hmm_crisis_prob | target_ret_fwd
2016-08-31 | SPY    |  0.032 |  0.056 |   0.109 |   0.121 |          0.3  |      EQUITY |      12.1 |      3.51 |        0.042 | ... |       0.12 |           0.04  |  0.014
2016-08-31 | QQQ    |  0.048 |  0.071 |   0.134 |   0.148 |          0.5  |      EQUITY |      12.1 |      3.51 |        0.042 | ... |       0.12 |           0.04  |  0.021
2016-08-31 | TLT    | -0.011 | -0.023 |   0.082 |   0.091 |         -0.2  |        BOND |      12.1 |      3.51 |        0.042 | ... |       0.12 |           0.04  | -0.008
...
2016-09-30 | SPY    |  0.014 |  0.052 |   0.112 |   0.118 |          0.1  |      EQUITY |      13.3 |      3.62 |        0.031 | ... |       0.09 |           0.07  |  0.022
```

- **자산별 피처**: 해당 자산마다 값이 다름 (모멘텀, 변동성 등)
- **매크로 피처**: 같은 날짜의 모든 자산에 동일한 값
- **target**: 다음 달 포워드 로그 수익률 (회귀용) 또는 등급 (분류용)

---

### 2-4. Feature Engineering

#### (A) 자산별 피처 (asset-level features)

| 피처명 | 수식 | 카테고리 |
|--------|------|----------|
| `ret_1m` | log(P_t / P_{t-21}) | 모멘텀 |
| `ret_3m` | log(P_t / P_{t-63}) | 모멘텀 |
| `ret_6m` | log(P_t / P_{t-126}) | 모멘텀 |
| `ret_12m` | log(P_t / P_{t-252}) | 모멘텀 |
| `vol_21d` | std(daily_ret, 21일) × √252 | 변동성 |
| `vol_63d` | std(daily_ret, 63일) × √252 | 변동성 |
| `volume_zscore` | (vol_t - vol_mean_63d) / vol_std_63d | 거래량 이상 |
| `ret_vs_sector` | ret_1m - sector_mean_ret_1m | 상대 모멘텀 |
| `vol_ratio` | vol_21d / vol_63d | 변동성 가속 |
| `sector_code` | EQUITY / BOND / ALT (카테고리 인코딩) | 자산군 |

#### (B) 매크로 파생변수 (기존 15개)

| 카테고리 | 피처 |
|----------|------|
| VIX 기간구조 | VIX_contango, VIX_slope_9d_3m, VIX_slope_3m_6m |
| 꼬리 위험 | SKEW_level, SKEW_zscore (63일 롤링) |
| 실물 경기 | Cu_Au_ratio, Cu_Au_ratio_chg (21일) |
| 신용 시장 | HY_spread, HY_spread_chg (5일), yield_curve, yield_curve_inv |
| 주간 매크로 | claims_4wma, claims_zscore (260일), WEI_level, sahm_indicator |

#### (C) GDELT 감성 피처 → Step 2에서 ML 입력으로 편입

```python
gdelt_avg_tone_1m       # 1개월 평균 감성 점수 (양수=긍정, 음수=부정)
gdelt_tone_momentum     # (gdelt_tone_1m - gdelt_tone_3m)  감성 변화율
gdelt_article_volume    # 기사량 (시장 불확실성 proxy)
gdelt_sector_tone_XLK   # 섹터별 감성 (섹터 ETF/개별주에 연결)
```

> 📄 **근거**: "Predicting Returns with Text Data" / "text-based-industry-momentum" 논문  
> 텍스트 감성 + 펀더멘털 데이터 결합 시 예측력 극대화

#### (D) HMM 레짐 확률 피처 → Step 2에서 ML 입력으로 편입

```python
hmm_crisis_prob   # P(regime=crisis | observations) — 연속형 0~1
```

- HMM 입력 변수 (5개): VIX_level, VIX_contango, HY_spread, yield_curve, Cu_Au_ratio_chg
- BIC로 최적 레짐 수 결정 (예상 N=3~4)
- 이 확률값을 ML의 피처로 넣으면 → AI가 레짐별 가중치를 스스로 학습

> 📄 **근거**: "심층강화학습 기반 경기순환 주기별 효율적 자산배분" 논문  
> 레짐 확률을 연속 피처로 입력 시 국면별 자동 가중치 학습 효과 증명

#### (E) 교차항 (Interaction Terms) — 논문 권장

```python
# 매크로 × 자산 특성 곱항 (비선형 관계 포착)
df['VIX_x_vol'] = df['VIX_level'] * df['vol_21d']
df['HY_x_ret3m'] = df['HY_spread'] * df['ret_3m']
df['Cu_Au_sq'] = df['Cu_Au_ratio'] ** 2
df['VIX_contango_x_SKEW'] = df['VIX_contango'] * df['SKEW_level']
```

> 📄 **근거**: BL+ML 논문 — "high-order terms and cross terms of basic features as preprocessing"

---

### 2-5. EDA 및 통계 검정

> **김하연 EDA 노트북(Step2_EDA)의 결과를 재활용 + 서윤범 방식에 맞게 확장**

#### ✅ 김하연 노트북에서 이미 완료된 항목 (재활용 가능)

| 항목 | 내용 | 주요 발견 |
|------|------|----------|
| 수익률 계산 | 포트폴리오 로그 수익률, VIX 차분, FRED 수준 유지 | 그대로 활용 |
| 기초통계 | 왜도/첨도 분석 | ICSA 첨도 58.6, ^VIX9D 20.6 — 극단값 주의 |
| 정상성 검정 (ADF) | 비정상 변수 변환 기준 확정 | 가격→로그수익률, 금리→차분 |
| VIF (다중공선성) | VIX 계열 4개 VIF 1000~3000 | **^VIX 하나만 유지, 나머지 제거 → 파생변수로 대체** |
| 상관관계 히트맵 | 30×30 자산 간 상관관계 | - |
| Granger 인과 검정 | AIC 최적 lag 보완 포함 | 최적 lag 9~10일 (약 2주 시차). DX-Y.NYB, CL=F, BAMLH0A0HYM2, ETH-USD 유의 |
| Granger 보완 | 희소성 확인 (CPIAUCSL 95% 희소 → 제외) | FRED 주간/월간 발표 변수 처리 기준 확정 |
| ARCH 효과 검정 | Engle ARCH LM Test | LM 통계량 857~911, p=0.000 → **변동성 군집 강하게 존재** |

**VIF 결과에 따른 변수 처리 확정**

- `^VIX9D`, `^VIX3M`, `^VIX6M` 직접 투입 제거 → `VIX_contango`, `VIX_slope_9d_3m`, `VIX_slope_3m_6m` 파생변수로 대체
- `UNRATE` 제거 → `SAHMREALTIME` 유지
- `CPIAUCSL` Granger 검정에서 제외 (월별 발표, 일별 차분 시 95% 희소)

---

#### 🆕 서윤범 방식을 위해 추가로 필요한 EDA/검정

**① 타겟 변수 변경에 따른 Granger 재실행**

김하연 노트북의 타겟(Y)은 `rv_neutral`(포트폴리오 변동성).  
서윤범 방식의 타겟은 **각 자산의 포워드 수익률** → Granger를 자산별로 재실행하거나,  
패널 전체를 대상으로 피처-수익률 관계 재확인 필요.

```python
# 서윤범 방식 Granger 타겟
Y = log(P_{t+21} / P_t)  # 개별 자산 포워드 수익률 (또는 패널 평균)
```

**② IC (Information Coefficient) 분석 — 피처 선별의 핵심**

```python
IC_i = spearmanr(feature_i_t, forward_return_{t+21}).correlation
# 롤링 윈도우마다 계산
# 기준: IC 평균 > 0.02, ICIR = IC평균/IC표준편차 > 0.3 → 유지
```

> 📄 "Empirical Asset Pricing via Machine Learning" (Gu et al., 2020)  
> 수천 개 피처 중 IC 기준으로 선별하는 것이 표준 — VIF/Granger만으로는 예측력 보장 불가

**③ 월간 주기 유효성 검증**

동일 모델로 포워드 기간만 바꿔 R²_oos 비교:

| 포워드 기간 | 예측 주기 | 비교 지표 |
|------------|---------|---------|
| 5영업일 | 주간 | R²_oos |
| 10영업일 | 격주 | R²_oos |
| **21영업일** | **월간 (본 설계)** | **R²_oos** |
| 63영업일 | 분기 | R²_oos |

→ 월간이 최적인지 확인 후 설계 확정

**④ 자산군별 특성 분석**

패널 모델의 이질성 문제를 사전에 확인:
- 매크로 피처-수익률 상관관계를 EQUITY / BOND / ALT 별로 분리해서 비교
- 방향이 반대인 관계(예: VIX↑ → 주식↓, 채권↑)가 있으면 `sector_code` 교차항 필수

**⑤ 레짐(HMM) 피처 유효성 확인**

```python
# HMM 학습 후 레짐별 자산 수익률 분포 비교
# stable 레짐 평균 수익률 vs crisis 레짐 평균 수익률
# BIC로 최적 레짐 수(N=3 or 4) 확정
```

**⑥ ARCH 효과 → 공분산 추정 방식 결정 참고**

김하연 결과(ARCH 효과 강하게 존재)에 따라:
- 공분산 추정 시 단순 표본공분산보다 **Ledoit-Wolf + PCA 팩터** 조합이 더 안정적
- GARCH 기반 동적 공분산은 구현 복잡도 대비 효과 미지수 → 우선 Ledoit-Wolf 사용

---

#### 정상성 검정 결과 요약 (확정)

| 변수 유형 | 처리 방법 |
|----------|---------|
| 자산 가격 30개, CL=F, GC=F, BTC 등 | 로그 수익률 `ln(P_t/P_{t-1})` |
| VIX 계열 수준값 | 그대로 유지 (단, ^VIX 하나만 직접 투입, 나머지는 파생변수로) |
| T10Y2Y, BAMLH0A0HYM2 등 스프레드 | 수준값 유지 (스프레드 수준 자체가 신호) |
| DGS10, UNRATE, CPIAUCSL 등 | 차분 `diff()` (단, CPIAUCSL은 희소성으로 제외) |
| SAHMREALTIME, WEI | 수준값 유지 |

---

### 2-6. ML 모델링 및 Q/Ω 도출

#### 뷰 생성 엔진 구조

```
엔진 ①: 정형데이터 기반 트리 모델
  ┌─ RandomForest (회귀)
  ├─ XGBoost (분류)
  └─ TabPFN (분류)

엔진 ②: LLM Agent
  ┌─ 입력: 뉴스 요약 (GDELT) + 최근 수익률 데이터
  └─ 출력: Q (기대수익률), Ω (확신도) 직접 반환
```

#### 타겟 변수 설계

| 모델 | 타겟 | 설계 방식 |
|------|------|----------|
| RandomForest | 연속형 수익률 | `log(P_{t+21} / P_t)` |
| XGBoost | 5등급 분류 | IS 구간 분위수 기반 `pd.qcut(ret, q=5)` |
| TabPFN | 5등급 분류 | XGBoost와 동일 (단, 샘플 10K 이하 제한 주의) |
| LLM Agent | 연속형 수익률 | 프롬프트로 각 자산 기대수익률 직접 생성 |

#### Q, Ω 도출 방법

| 모델 | Q 계산 | Ω 계산 |
|------|--------|--------|
| RF (회귀) | OOB 예측값 (트리 평균) | `bias² + var` 또는 트리 간 분산 |
| XGBoost (분류) | `Σ(grade_k × P(grade=k))` | `1 - max(class_prob)` |
| TabPFN (분류) | XGBoost와 동일 | XGBoost와 동일 |
| LLM Agent | 직접 반환 | 직접 반환 또는 앙상블 분산 |

> 📄 **근거**: BL+ML 논문 — `E(f,D) = bias²(x) + var(x)/σ`  
> 모델 불확실성 = 편향² + 분산으로 Ω 추정

#### 모델 앙상블 전략 (미결, 3-1 참조)

---

### 2-7. Black-Litterman 통합

#### BL 수식

```
μ_post = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ × [(τΣ)⁻¹π + PᵀΩ⁻¹Q]
```

- `π`: CAPM 균형 수익률 (prior)
- `Σ`: **레짐 조건부 공분산** (HMM 연동)
- `Q`: ML 예측 수익률 벡터
- `Ω`: 예측 불확실성 행렬 (대각)
- `τ`: 스케일링 파라미터 (보통 1/T)

#### 레짐 반영 방식 ✅ 확정: HMM 확률을 피처로 편입

```python
# Step 2에서 생성되는 연속형 피처
hmm_crisis_prob_t  # P(regime=crisis | 관측값) — 0~1 연속값
# → RF, XGBoost, TabPFN의 입력 피처로 사용
# → 모델이 "위기 확률 0.3일 때"와 "0.8일 때"를 스스로 다르게 학습
```

**공분산 교체(Σ_stable / Σ_crisis) 방식은 ablation으로 분리**  
→ 미결 고려사항 3-6 참조

#### 공분산 행렬 구조

```
Level 1 (자산군 간): 블록 대각 — SPY, AGG, GLD 간 3×3 (Ledoit-Wolf)
Level 2 주식 (24개): PCA 팩터 모형 (누적 분산 80% 기준 K 자동 선택)
Level 2 채권 (4개):  Ledoit-Wolf
Level 2 대안 (2개):  직접 추정

위기기 fallback:
  관측일 >= 48일: 정상 분리 추정
  20~47일: Σ_crisis = Σ_stable × 1.5
  < 20일:  단일 공분산 사용
```

#### MV 최적화 + 회전율 페널티

```python
# 목적함수 (회전율 페널티 포함)
maximize: μ_post'w - (γ/2) × w'Σw - λ × Σ|w_i - w_prev_i|

# 제약조건
Σw_i = 1             # 비중 합 = 100%
w_i >= 0             # 공매도 금지
w_i <= 상한 (성향별)  # HHI 기반 개별 비중 상한
```

- `λ`: 회전율 페널티 계수 → 사회초년생 실무 방어 논리 (거래비용 최소화)

---

### 2-8. 백테스팅 및 평가지표

#### ML 예측력 (Statistical)

| 지표 | 설명 |
|------|------|
| Accuracy | 분류 정확도 (XGBoost, TabPFN) |
| MSE | 평균제곱오차 (RF 회귀) |
| R²_oos | 샘플 외 결정계수 (핵심 — 값이 양수면 ML이 랜덤보다 나음) |
| Diebold-Mariano Test | 모델 간 예측력 차이 유의성 검정 |

#### BL 운용 성과 (Economic)

| 지표 | 설명 |
|------|------|
| Sharpe Ratio | 위험조정 수익률 (주 지표) |
| Sortino Ratio | 하방 변동성만 고려 |
| CAGR | 연평균 복리 성장률 |
| MDD | 최대 낙폭 |
| Calmar Ratio | CAGR / MDD |
| Net Return | 거래비용(왕복 30bps) 차감 후 순수익률 |
| 팩터 알파(α) | Fama-French 3~5 팩터 대비 초과 수익 |
| 회전율 | 월평균 포트폴리오 교체 비율 |

#### 벤치마크

- SPY 100% (시장 수익률)
- 동일가중(EW) 30자산
- BL 없이 MV 최적화만 (ML 기여도 분리)

---

## 3. 미결 고려 사항

### 3-1. 모델 앙상블 방식

RF, XGBoost, TabPFN, LLM Agent에서 각각 Q, Ω가 나올 때 이를 어떻게 통합할 것인가.

**옵션 A**: 단순 평균
```python
Q_final = (Q_RF + Q_XGB + Q_TabPFN + Q_LLM) / 4
```

**옵션 B**: Ω 역가중 평균 (신뢰도 높은 모델에 더 많은 가중치)
```python
w_i = (1/Ω_i) / Σ(1/Ω_j)
Q_final = Σ(w_i × Q_i)
```

**옵션 C**: 각 모델을 독립적 BL 뷰로 분리 → BL에 여러 view row로 투입  
→ 가장 이론적으로 깔끔하지만 구현 복잡도 높음

→ **현재 권장**: 옵션 B로 시작, 옵션 C는 이후 ablation

### 3-2. TabPFN 구체 설계

- 샘플 한계: 10,000개 이하 권장 → 150일 × 30자산 = 4,500개로 문제없음
- 단, 피처 수 제한도 있음 (기본 100개 이하) → 피처 선택 필요
- 분류 등급: XGBoost와 동일 5등급 사용 예정
- 논문이 없으므로 기존 모델과의 성능 비교 실험이 논문 기여점이 될 수 있음

### 3-3. LLM Agent 구체 설계

- 어떤 모델을 사용할 것인가? (GPT-4o, Claude, Gemini)
- 프롬프트 구조: 뉴스 요약 + 최근 수익률 + 매크로 지표 → Q, Ω 반환
- 재현성 문제: temperature=0으로 고정해도 완전 재현 어려움 → 결과 로깅 필수

> 📄 **근거**: "LLM-Enhanced Black-Litterman Portfolio Optimization" 논문 직접 참고

### 3-4. Step 5 Circuit Breaker — 임계값 설정

```
HMM P(crisis) > θ → 주식 비중 0%, 안전자산 100% 강제 전환
```

- θ 값을 얼마로 설정할 것인가? (논문: 60% 기준)
- 완전 청산 vs 부분 축소 중 선택
- 해제 조건: P(crisis) < θ - margin (히스테리시스 적용 여부)

### 3-5. 월간 주기 유효성 검증 결과에 따른 설계 변경

- R²_oos 비교 후 월간이 최적이 아닐 경우 주기 조정 가능성 열어둠
- 거래비용 차감 시 월간 vs 분기 순수익 비교 필요

### 3-6. 레짐-공분산 교체 방식 (Ablation 후보)

**기본 설계 (확정)**: `hmm_crisis_prob`을 ML 피처로 투입 (연속형)

이진 공분산 교체 방식(Σ_stable / Σ_crisis)은 다음 이유로 우선 순위를 낮춤:
- 위기기 데이터 부족(전체의 15~20%) → Σ_crisis 추정 불안정
- 레짐 이진 경계가 인위적 → 연속 확률이 더 자연스러움

**Ablation 비교 실험으로 남겨둠**:
- 방안 A (확정): `hmm_crisis_prob` 피처 편입
- 방안 B (ablation): 레짐 이진 판별 → Σ 교체 (fallback: Σ_crisis = Σ_stable × 1.5)
- Walk-Forward OOS Sharpe 차이로 우열 확인 → 논문 기여점 될 수 있음

---

## 4. 약점 및 주의사항 ⚠️

### (1) 월간 수익률 예측의 낮은 SNR
- 학술 연구 기준 R²_oos ≈ 1~5% 수준이 일반적
- 이것이 거래비용을 넘는 초과 수익으로 이어지지 않을 수 있음
- **대응**: 회전율 페널티(λ) 튜닝 + Net Return 중심으로 평가

### (2) Ω 추정의 부정확성
- RF 예측 구간 → calibrated되지 않는 경우 많음 (과소/과대 신뢰)
- XGBoost 확률 → 기본 설정에서 overconfident 경향
- **대응**: Platt Scaling 또는 Isotonic Regression으로 확률 보정 (Calibration)

### (3) 패널 모델의 자산군 이질성
- SPY(주식)와 TLT(채권)를 같은 모델로 학습하면 관계가 자산군마다 다를 수 있음
- **대응**: `sector_code`를 카테고리 피처로 투입 + SHAP으로 자산군별 피처 중요도 비교

### (4) Look-ahead Bias 위험
- GDELT 감성 피처: 데이터 수집 시 발표 시점 기준으로 정렬 필수
- Granger 검정: 반드시 IS 윈도우 내에서만 실행 (전체 데이터에서 하면 미래 정보 누출)
- HMM 레짐 확률: 각 IS 윈도우에서 재학습 (미래 레짐 정보 사용 금지)

### (5) 위기기 데이터 부족
- 10년 중 실제 위기 구간은 약 15~20% (약 300~500일)
- Σ_crisis 추정 시 샘플 부족 → fallback 로직 필수 (2-7 참조)
- **대응**: 스트레스 테스트 시나리오 (2008 GFC 등 역사적 데이터 보완 가능)

### (6) LLM Agent 재현성
- 동일 입력에도 출력이 달라질 수 있음
- **대응**: 모든 LLM 호출 결과 JSON으로 저장, temperature=0 고정

### (7) 거래비용 현실성
- 왕복 30bps 가정이 개인 투자자에게는 낙관적일 수 있음
- ETF는 현실적이나 개별 종목은 스프레드 추가 고려 필요

---

## 5. 참고 논문 활용 매핑

| 논문 | Step | 가져올 내용 |
|------|------|------------|
| A BL Portfolio Selection Model with ML Algorithms | 3.1, 3.2 | Q/Ω 도출, bias-variance Ω, 레이블 이산화, 고차 교차항 |
| LLM-Enhanced Black-Litterman Portfolio Optimization | 3.1 | LLM Agent 뷰 생성 구조, 프롬프트 설계 |
| Empirical Asset Pricing via Machine Learning (Gu et al.) | 2, 3.1 | 패널 데이터 표준, IC 필터링, R²_oos 해석 기준 |
| Predicting Returns with Text Data | 2 | GDELT 감성 피처 설계, 텍스트+정형 결합 방법론 |
| text-based-industry-momentum | 2 | 섹터별 텍스트 모멘텀 피처 |
| Explainable ML for Regime-Based | 2, 3.3 | 레짐 확률 피처 편입 vs 별도 모델 비교 |
| LEVERAGING LLMS FOR TOP-DOWN SECTOR | 3.1 | LLM 섹터 배분 뷰 생성 방식 |
| 심층강화학습 기반 경기순환 주기별 효율적 자산배분 | 2, 5 | 레짐 확률 연속 피처화 효과, Circuit Breaker 설계 근거 |
| Deep Learning in Finance | 3.1 | MLP 대안 모델 검토 시 참고 |

---

## 6. 단계별 진행 현황

| Step | 내용 | 상태 | 메모 |
|------|------|------|------|
| Step 1 | 데이터 수집 | 🔄 진행 중 | GDELT 수집 중 |
| Step 2 | 전처리 + Feature Engineering | ⏳ 대기 | 피처 목록 확정됨 |
| Step 3 | ML 모델링 + BL 통합 | ⏳ 대기 | 앙상블 방식 미결 |
| Step 4 | Walk-Forward 백테스팅 | ⏳ 대기 | |
| Step 5 | Risk Analysis + Circuit Breaker | ⏳ 대기 | θ 임계값 미결 |
| Step 6 | 동적 리밸런싱 + Streamlit | 🔲 미설계 | |

---

*이 문서는 새 대화 시작 시 Claude에게 전달하면 맥락 유지가 가능합니다.*  
*"project_design_log.md 읽고 [작업 내용] 시작해줘" 형태로 활용하세요.*
