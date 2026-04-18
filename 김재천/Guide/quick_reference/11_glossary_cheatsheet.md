# 📖 용어 치트시트

> **독자**: 학습 중인 모든 독자 (균형 톤)
> **목적**: 자주 등장하는 용어를 한 곳에 정리 (50+ 용어)

---

## 🔑 최핵심 용어 (10선)

| # | 용어 | 한 줄 정의 |
|---|------|---------|
| 1 | **Sharpe Ratio** | 연율수익률 / 연율변동성 (위험 조정 수익률) |
| 2 | **MDD (Maximum Drawdown)** | 최고점 대비 최대 낙폭 |
| 3 | **VIX** | S&P 500 옵션 기반 30일 내재변동성 (공포지수) |
| 4 | **Walk-Forward** | 과거 IS로 학습 → 미래 OOS로 검증 → 슬라이드 |
| 5 | **MV 최적화** | Markowitz의 평균-분산 최적화 (수익↑ 위험↓) |
| 6 | **HMM** | Hidden Markov Model, 시장 레짐 자동 분류 |
| 7 | **공분산 Σ** | 자산들이 함께 움직이는 패턴 (n×n 행렬) |
| 8 | **경보 레벨** | L0(정상)~L3(위기) 4단계 |
| 9 | **Ledoit-Wolf** | 공분산 추정 안정화 수축 기법 |
| 10 | **Bootstrap** | 복원추출 기반 통계 검정 |

---

## 📊 포트폴리오 이론

| 용어 | 정의 | 사용 Step |
|------|------|--------|
| **Equal Weight (EW)** | 30자산 균등 배분 (1/30) | Step 7, 벤치마크 |
| **MV (Mean-Variance)** | 평균-분산 최적화 | Step 3, 4, 9 |
| **RP (Risk Parity)** | 모든 자산 동일 위험 기여 | Step 3 |
| **HRP** | Hierarchical Risk Parity, 클러스터 기반 RP | Step 3 |
| **Rebalancing** | 비중 재조정 | 전반 |
| **Turnover** | 재조정 시 바뀌는 비중 총합 | Step 9 |
| **Efficient Frontier** | 최적 수익-위험 조합 곡선 | Step 3 |
| **γ (감마)** | 위험회피계수 (크면 보수, 작으면 공격) | Step 3 |

---

## 📈 성과 지표

| 용어 | 공식 | 해석 |
|------|-----|------|
| **Sharpe Ratio** | `ann_ret / ann_vol` | 위험 1단위당 수익. **>1.0 양호** |
| **Sortino Ratio** | `ann_ret / downside_vol` | 하방 변동성 기준 (상방은 위험 아님) |
| **Calmar Ratio** | `ann_ret / \|MDD\|` | MDD 대비 수익 효율 |
| **Information Ratio (IR)** | `mean(active) / std(active) × √252` | 초과수익/추적오차 |
| **CVaR 99%** | 최악 1% 일별 평균 손실 | VaR 보완 |
| **VaR 95%** | 95% 확률로 넘지 않을 손실 | 리스크 한도 |
| **ΔSharpe** | 두 전략의 Sharpe 차이 | 실무 임계: 0.2 이상 유의 |

---

## 🧠 통계·머신러닝

| 용어 | 정의 |
|------|------|
| **Granger 인과검정** | "A가 B를 예측하는가" 통계 검정 |
| **p-value** | 우연히 이 정도 결과 나올 확률 (<0.05 유의) |
| **BIC** | Bayesian Information Criterion, 모델 복잡도 페널티 |
| **Ledoit-Wolf 수축** | 공분산 추정 시 표본 + 구조화 블렌딩 |
| **HMM 레짐** | Hidden Markov가 분류한 시장 상태 |
| **Bonferroni 보정** | 다중 비교 시 α 축소 (α/K) |
| **FDR (Benjamini-Hochberg)** | False Discovery Rate 통제 (덜 보수적) |
| **Cohen's d** | 효과 크기 (재무에는 부적합, IR 선호) |

---

## 🔔 경보 시스템

| 용어 | 정의 |
|------|------|
| **Config A** | VIX 단독 기준 (20/28/35) |
| **Config B** | A + VIX Contango 조정 |
| **Config C** | 7지표 복합 스트레스 스코어 |
| **Config D** | C + 5일 디바운스 |
| **Contango** | VIX3M > VIX (정상) |
| **Backwardation** | VIX3M < VIX (공포) |
| **HY Spread** | BAA 회사채 - 국채 수익률 (신용 스트레스) |
| **Sahm Rule** | 실업률 3M avg - 12M min ≥ 0.5 → 침체 신호 |
| **Yield Curve** | T10Y - T2Y (음수면 역전 = 침체 선행) |
| **SKEW Index** | S&P 500 옵션의 꼬리 리스크 |

---

## 🔬 v4.1 신규 용어

| 용어 | 정의 |
|------|------|
| **Deployment Simulation** | "현재 모델을 과거에 적용" 철학 |
| **경로 1** | 경보 → 일별 주식 축소 |
| **경로 2** | HMM 레짐 → Σ 전환 (월별) |
| **Σ_stable / Σ_crisis** | 평상시/위기 공분산 |
| **Dynamic 4-stage Fallback** | Σ 추정 관측수 부족 시 대체 전략 |
| **separate** | Stable·Crisis 독립 추정 (≥48일) |
| **scaled** | Σ_crisis = Σ_stable × 1.5 |
| **scaled_reverse** | Σ_stable = Σ_crisis / 1.5 |
| **single** | 단일 Σ (레짐 분리 포기) |
| **M0/M1/M2/M3** | 4개 모드 (baseline/경로1/경로2/통합) |
| **Multi-criteria Decision** | 4지표(Sharpe+MDD+Calmar+Sortino) 가중 |
| **LOO (Leave-One-Out)** | 한 윈도우 제외 후 Sharpe 재계산 |

---

## 💰 자산 종류

### 주식 (24개)
| 구분 | 티커 | 의미 |
|------|------|------|
| **인덱스 ETF** | SPY, QQQ, IWM, EFA, EEM | 시장 대표 |
| **섹터 ETF** | XLK~XLB (11개) | 개별 섹터 |
| **개별 주식** | AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, PG, XOM | Magnificent + 전통 |

### 채권 (4개)
| 티커 | 의미 |
|------|------|
| TLT | 20년+ 장기채 |
| AGG | 종합 채권 인덱스 |
| SHY | 1-3년 단기채 |
| TIP | 물가연동채 |

### 대체 (2개)
| 티커 | 의미 |
|------|------|
| GLD | 금 ETF |
| DBC | 원자재 ETF |

---

## 📐 수식 모음

### Sharpe Ratio
```
Sharpe = E[R] / σ[R] × √252
```

### MV 최적화 목적함수
```
max  w'μ - (γ/2) × w'Σw
 w
s.t. Σw = 1, w ≥ 0, 제약조건
```

### Ledoit-Wolf 수축
```
Σ_hat = α × Σ_shrink + (1-α) × Σ_sample
```

### Granger 인과검정 (F-test)
```
H0: β_1 = β_2 = ... = β_p = 0  (A는 B 예측 못함)
F = (RSS_R - RSS_UR) / p ÷ RSS_UR / (n-2p-1)
```

### VaR 95% (역사적)
```
VaR_95% = percentile(returns, 5)
```

### BIC
```
BIC = -2 ln(L) + k × ln(n)
```
- L: 최대 우도
- k: 파라미터 수
- n: 관측 수

---

## 🎯 자주 헷갈리는 쌍

### Sharpe vs Sortino
- **Sharpe**: 전체 변동성 기준
- **Sortino**: 하방 변동성만 (상방은 좋은 변동)

### Config A vs B
- **A**: VIX만
- **B**: A + Contango 조정 (더 정교)

### M1 vs M3
- **M1**: 경로 1만 (경보 반응)
- **M3**: 경로 1 + 경로 2 (통합)
- 실증: **M1이 M3보다 우수**

### Ann Return vs CAGR
- **Ann Return**: 산술평균 × 252
- **CAGR**: (Total + 1)^(1/years) - 1
- 변동성 크면 CAGR < Ann Return

### Turnover vs 거래비용
- **Turnover**: 비중 변화량 (Σ|Δw|)
- **거래비용**: Turnover × 15bps

---

## 🔤 축약어 사전

| 축약 | 풀이 |
|------|------|
| MV | Mean-Variance |
| RP | Risk Parity |
| HRP | Hierarchical Risk Parity |
| EW | Equal Weight |
| WF | Walk-Forward |
| IS | In-Sample |
| OOS | Out-of-Sample |
| VaR | Value at Risk |
| CVaR | Conditional VaR |
| IR | Information Ratio |
| MDD | Maximum Drawdown |
| HMM | Hidden Markov Model |
| HY | High Yield |
| bps | Basis Point (0.01%) |
| FRED | Federal Reserve Economic Data |
| BIC | Bayesian Information Criterion |
| AIC | Akaike Information Criterion |
| FDR | False Discovery Rate |
| LOO | Leave-One-Out |
| CAGR | Compound Annual Growth Rate |

---

## 📚 심층 학습

용어를 더 깊이 이해하려면:

- `docs/Step1~11_해설.md` (각 Step별 상세 설명)
- `stats_model.md` (통계 기법 종합)
- `report_v3.md` 부록 D (113개 용어 사전 확장판)

---

## 🔍 검색 팁

**Streamlit 앱** Learn 페이지 이용 시 용어 검색 기능 제공 예정 (Phase 4-B-5).
