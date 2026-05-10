# C-1-6. Methodology 페이지

> **파일**: `07_methodology.md`
> **결정 시점**: 2026-05-10
> **상태**: 확정 (페이지 메타 Method M-1~M-4 + 영역 1~9)
> **포함**: 페이지 메타 결정 / Sub-header / Methodology Overview (Plotly Sankey) / Black-Litterman 상세 + 4-slot config / LSTM 변동성 예측 (walk-forward 명시) / Factor 분석 (CAPM + FF5) / 정규성 검정 (Jarque-Bera) / 한계 + 향후 개선 (5개 → 3개 한계 축소)

---

## C-1-6. Methodology 페이지

> **상태**: 진행 중 (메타 결정 확정 / 영역 1, 9 는 Overview 동일 / 2-8 미정)

### Methodology 페이지 통합 배경 (Context)

펀드 운용 방법론 설명 전담 페이지. BL (Black-Litterman) + LSTM 변동성 예측 + 4-slot config 의 학술/실무 토대 노출.

**핵심 청중**:
- 학술 심사위원: 학술 정확성 + 인용 근거
- 가상 투자자: 직관적 도식 + 운용 narrative

**vs 다른 페이지**:
- Performance / Risk / Holdings = 결과 위주
- **Methodology = 과정 / 토대 위주**

**메트릭 풀 (C-2) 활용**:
- Pool-6 (Factor 분석): CAPM Alpha / FF5 Alpha / Factor Exposures
- Performance S8-3 결정에서 이동: Jarque-Bera test (정규분포 검정)

### 페이지 메타 결정 (Method M-1 ~ M-4)

#### Method M-1. 영역 개수

**검토된 옵션**:
- (a) 압축 5 영역
- (b) 표준 7 영역
- (c) 풍부 8 영역

**결정**: (c) 풍부 8 영역

**근거**:
1. 학술 토대 (BL / LSTM / Factor / 정규성) 충분 활용
2. 한계 + 향후 개선 영역 = 학술 정직성 강화 필수
3. 자유 탐색 모드 부합

##### Method M-1 영역 8 (한계 + 향후 개선) — 표시 형식

**검토된 옵션**:
- (A) 단순 텍스트 나열
- (B) 카드 그리드 (5개 차원별 카드)
- (C) Expander (펼침/접기)
- (D) 표 형식 (한계/영향/개선)
- (E) 학술 보고서 형식
- (F) Q&A 형식

**결정**: (B) + (C) 결합 — 카드 그리드 + Expander

**근거**:
1. **카드 그리드 (B)** = 5가지 한계 차원 한눈에 시각
2. **Expander (C)** = 클릭 시 상세 + 학술 인용
3. 시각 효과 + 학술 정직성 + 청중 친화 모두 충족
4. 페이지 분량 균형 (접힘 상태 → 적정)

**5가지 한계 차원**:
1. 데이터 한계 (단일 OOS / Walk-forward 미적용 / Slippage 미반영)
2. 모델 한계 (LSTM 1차 / BL CAPM prior / Sector trend 미반영)
3. HO 24m 부진 인정 (SPY 대비 -12.9%p / Sector trade-off)
4. 향후 개선 방향 (Multi-factor / WFV / 실거래 시뮬레이션)
5. 실무 적용 제약 (가상 펀드 / 운용 규모 / Tax 미반영)

#### Method M-2. Sub-header 포함

**결정**: (a) 포함

**근거**: 다른 페이지 일관 — 인터랙션 일관성 원칙

#### Method M-3. 학술 인용 깊이

**검토된 옵션**:
- (a) 핵심 5-7개
- (b) 표준 10-15개
- (c) 풍부 15+

**결정**: (b) 표준 10-15개

**근거**:
1. **학술 정직성 + 청중 부담 균형**
2. 핵심 인용 (Markowitz / Black-Litterman / LSTM / Fama-French 등) + 보조 인용
3. (c) 풍부 15+는 학술 보고서 수준 → 가상 투자자 부담 ↑

**예상 인용 문헌**:
- Markowitz (1952) "Portfolio Selection" — 평균-분산 이론
- Black & Litterman (1990, 1992) — Black-Litterman 모델
- Hochreiter & Schmidhuber (1997) — LSTM
- Fama & French (1992, 1993, 2015) — FF3 / FF5
- Jensen (1968) — CAPM Alpha
- Sharpe (1966, 1994) — Sharpe Ratio
- Sortino & Price (1994) — Sortino Ratio
- Jarque & Bera (1980) — Jarque-Bera test
- Frazzini, Israel, Moskowitz (2018) — Trading costs (AQR)
- Lopez de Prado (2018) — Walk-forward validation
- Modigliani & Modigliani (1997) — M²
- Hill (1975) — Hill estimator (Risk Metrics 페이지)

#### Method M-4. 도식 / 시각화

**검토된 옵션**:
- (a) 텍스트 + 수식 위주
- (b) 도식 + 텍스트
- (c1) 인터랙티브 (st.expander)
- (c2) 인터랙티브 (Plotly Sankey)
- (c3) 둘 다

**결정**: (c2) Plotly Sankey

**근거**:
1. **워크플로우 흐름 시각** — Market Data → BL Prior → LSTM Vol → Optimizer → Weights
2. **인터랙티브** — 노드 hover 시 상세 정보
3. (c1) Expander 는 영역 8 (한계) 에서 사용 → 영역 3 (Overview) 는 Sankey 차별화
4. (b) 정적 도식보다 인터랙션 깊이 ↑

##### Plotly Sankey 구현 안

```python
import plotly.graph_objects as go

nodes = [
    "Market Data",      # 0
    "Returns Data",     # 1
    "Sector / Mcap",    # 2
    "BL Prior (CAPM eq)",  # 3
    "LSTM Vol Predict", # 4
    "View / Confidence",# 5
    "BL Posterior",     # 6
    "Optimizer (4-slot)", # 7
    "Portfolio Weights" # 8
]

links = {
    "source": [0, 0, 1, 1, 2, 3, 4, 5, 6],
    "target": [1, 2, 3, 4, 3, 6, 6, 6, 7],
    "value":  [1, 1, 1, 1, 1, 1, 1, 1, 1]
}

fig = go.Figure(data=go.Sankey(node=dict(label=nodes), link=links))
```

### Methodology 페이지 8 영역 구조

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트)
3. Methodology Overview         (BL+LSTM Plotly Sankey)
4. Black-Litterman 상세         (학술 설명 + 4-slot config)
5. LSTM 변동성 예측 상세        (학술 설명 + 모델 구조)
6. Factor 분석 (CAPM / FF5)     (alpha + exposures)
7. 정규성 검정 (Jarque-Bera)    (Performance S8-3 에서 이동)
8. 한계 + 향후 개선             (카드 + Expander)
9. Footer                       (Overview 동일)
```

### 영역 2: Sub-header — 확정

#### 결정 항목 M2-1: 텍스트 내용

**검토된 옵션**:
- (a) Performance 패턴 동일
- (b) 학술 narrative 강화 (Markowitz 인용)
- (c) 청중 친화 narrative

**결정**: (a) Performance 패턴 동일

**근거**:
1. 모든 페이지 Sub-header 일관 패턴 — 인터랙션 일관성 원칙
2. 학술 narrative (b) 는 영역 3 (Methodology Overview narrative 박스) 에서 자세히
3. 청중 친화 (c) 는 영역 4-7 의 학술 설명에 통합

**텍스트 안**:
```
Methodology (방법론)
Black-Litterman + LSTM 변동성 예측 + Factor 분석 + 한계 명시.
사이드바에서 기간 + 비교 토글 가능.
```

---

### 영역 3: Methodology Overview (Plotly Sankey) — 확정

#### 영역 3 통합 배경 (Context)

펀드 운용 워크플로우의 인터랙티브 Sankey diagram — 데이터 → BL → LSTM → Optimizer → Weights 흐름 시각.

#### 결정 항목 M3-1: Sankey 노드 구성

**결정**: (b) 표준 7-9 노드

**근거**:
1. **핵심 흐름 충분 시각** + 시각적 깔끔
2. (a) 5 노드는 BL Posterior / View Confidence 등 누락
3. (c) 12+ 노드는 시각 혼잡

**노드 안 (8개)**:
1. Market Data
2. Returns Data
3. Sector / Market Cap
4. BL Prior (CAPM equilibrium)
5. LSTM Vol Predict
6. View / Confidence
7. BL Posterior
8. Optimizer (4-slot)
9. Portfolio Weights

#### 결정 항목 M3-2: 노드 그룹화

**결정**: (b) 4그룹 (데이터 / BL / LSTM / Optimizer)

**근거**:
1. **색상 구분 명확** — 4가지 운용 단계 시각
2. (a) 3그룹은 BL/LSTM 통합 → 두 모델 차별화 ↓
3. (c) 단일 색은 운용 단계 구분 손실

**색상 안**:
- 데이터: Cobalt Blue `#3B82F6`
- BL: Green `#10B981`
- LSTM: Purple `#8B5CF6`
- Optimizer: Orange `#F59E0B`

#### 결정 항목 M3-3: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip (노드 detail + 학술 인용) ✓
- 노드 클릭 → 관련 영역 (4-7) 으로 navigation ✓
- Drag 조정 (Plotly 기본) ✓
- 다운로드 (PNG) ✓

#### 결정 항목 M3-4: 추가 표시

**결정**: (b) Sankey + 학술 narrative 박스

**근거**:
1. **학술 narrative** = Sankey 의 의미 즉시 안내 (Markowitz / BL / LSTM 인용)
2. (c) Step text 는 영역 4-5 에서 자세히
3. (d) KPI 카드는 4-slot config 정보 → 영역 4 에서 자세히

**narrative 박스 텍스트 안**:
```
Adaptive VolControl Fund 는 Black-Litterman (1990) 의 균형
prior 와 LSTM (Hochreiter 1997) 변동성 예측을 결합한 적응형
자산배분 펀드입니다. Markowitz (1952) 평균-분산 이론에
기반하여 4-slot config 로 robust 검증을 수행합니다.
```

#### 결정 항목 M3-5: 4-slot config 노출

**검토된 옵션**:
- (a) Sankey 노드 1개로 통합
- (b) Sankey + 별도 4-slot 도식
- (c) 영역 4 (BL 상세) 에서 자세히

**결정**: (a) + (c) — Sankey 통합 + 영역 4 자세히 (이중 노출)

**근거**:
1. **Sankey** = 운용 워크플로우 한눈 (4-slot 통합 노드 "Optimizer (4-slot)")
2. **영역 4** = 4-slot 자세한 도식 + 각 slot 의 의미
3. (b) 별도 도식은 영역 3 분량 ↑ → 영역 4 로 위임

#### 시각화 예시 (확정 사항 조합)

```
[Header — Overview 동일]

┌─ ℹ️ Methodology (방법론) ───────────────────────────────┐
│ Black-Litterman + LSTM 변동성 예측 + Factor 분석 + 한계 │
│ 사이드바에서 기간 + 비교 토글 가능.                      │
└─────────────────────────────────────────────────────────┘

┌─ Methodology Overview (Plotly Sankey) ──────────────────┐
│                                                         │
│ ┌─ 학술 narrative 박스 ────────────────────────────┐   │
│ │ Adaptive VolControl Fund 는 Black-Litterman      │   │
│ │ (1990) 의 균형 prior 와 LSTM (Hochreiter 1997)   │   │
│ │ 변동성 예측을 결합한 적응형 자산배분 펀드입니다. │   │
│ │ Markowitz (1952) 평균-분산 이론에 기반하여       │   │
│ │ 4-slot config 로 robust 검증.                    │   │
│ └─────────────────────────────────────────────────┘   │
│                                                         │
│ Plotly Sankey (4 그룹 색상):                            │
│                                                         │
│  🔵 Data Sources (blue)                                │
│  🟢 BL Pipeline (green)                                │
│  🟣 LSTM Pipeline (purple)                             │
│  🟠 Optimizer (orange)                                 │
│                                                         │
│  Market Data ═══➤ Returns Data ═══➤                    │
│       │                            BL Prior ═══➤       │
│       └═══➤ Sector/Mcap ═══➤              │            │
│                                           │            │
│  Returns ═══➤ LSTM Vol Predict ═══➤ View ════➤        │
│                                           │            │
│  View Confidence ═══➤ ↓                   │            │
│                       BL Posterior ═══➤    │            │
│                                                         │
│                              Optimizer (4-slot) ═══➤   │
│                                                         │
│                                  Portfolio Weights     │
│                                                         │
│ Hover: "BL Prior — CAPM equilibrium prior.             │
│         Black & Litterman (1990)"                       │
│ Click: 노드 → 영역 4-7 으로 navigation                  │
│ [⬇ PNG]                                                │
└─────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- Sankey 구현 = Plotly `go.Sankey`
- 노드 클릭 navigation = `st.plotly_chart(on_select='rerun')` + `st.session_state.scroll_to`
- 4-slot 통합 노드 → 영역 4 에서 자세히 분해 (이중 노출)
- 학술 인용 (Markowitz 1952, Black-Litterman 1990, Hochreiter 1997) = 부록 학술 근거 일람 추가

### 영역 4: Black-Litterman 상세 + 4-slot config — 확정

#### 영역 4 통합 배경 (Context)

Black-Litterman 모델의 학술 설명 + 4-slot config 의 자세한 분해.

**학술 토대**:
- Black & Litterman (1990) "Asset Allocation: Combining Investor Views with Market Equilibrium"
- Black & Litterman (1992) "Global Portfolio Optimization"
- He & Litterman (1999) "The Intuition Behind Black-Litterman Model Portfolios"
- Idzorek (2005) "A Step-by-Step Guide to the Black-Litterman Model"

**BL 핵심 수식**:
$$E[R] = \left[ (\tau\Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1} \left[ (\tau\Sigma)^{-1}\Pi + P^T \Omega^{-1} Q \right]$$

**4-slot config** = (prior × p_weight × q_mode × omega_mode)

#### 결정 항목 M4-1: 표시 형식

**결정**: (b) 도식 + 수식

**근거**:
1. **시각 (Bayesian update 도식) + 학술 (LaTeX 수식)** 균형
2. (a) 텍스트만은 청중 친화 ↓
3. (c) Step-by-step 은 영역 8 (한계 카드 Expander) 와 형식 중복
4. (d) Tab 전환은 정보 분산 ↑

#### 결정 항목 M4-2: 4-slot config 시각화

**결정**: (e) 표 + 우리 펀드 강조 결합

**표 구성**:
| Slot | Options | Our Fund |
|---|---|---|
| prior | mat / mcap / rp | **mat** ★ |
| p_weight | eq / mcap | **eq** ★ |
| q_mode | eq / mcap / lam / raw | **eq** ★ |
| omega_mode | pap / he / lam / rms | **pap** ★ |

**근거**:
1. **표** = 모든 옵션 한눈 시각
2. **우리 펀드 강조** (★) = mat_eq_eq_(raw)_pap 즉시 인지
3. (b) 트리는 단순 4 slot 에 과한 시각
4. (c) Heatmap 은 옵션별 결과 비교 → Backtesting 페이지로 위임

#### 결정 항목 M4-3: 학술 인용

**결정**: (b) 표준 4개 인용

**4개 인용**:
1. Black & Litterman (1990) — 원조
2. Black & Litterman (1992) — 글로벌 적용
3. He & Litterman (1999) — view confidence Ω 직관
4. Idzorek (2005) — Step-by-step guide (가상 투자자 친화 인용)

**근거**:
1. 핵심 2개 (Black-Litterman 1990/1992) 만은 부족
2. He-Litterman (1999) Ω intuition = LSTM 변동성 예측 통합 narrative 와 직접 연결
3. Idzorek (2005) = 실무 적용 가이드

#### 결정 항목 M4-4: 인터랙션

**결정**: hover + click 인용 (시뮬레이션 보류)

**채택 인터랙션**:
- Hover tooltip (수식 변수 정의: τ, Π, Σ, P, Q, Ω) ✓
- Click → 학술 인용 link (Google Scholar 등) ✓
- 4-slot 조합 시뮬레이션 = 보류 (구현 복잡 + Backtesting 페이지 위임)

#### 결정 항목 M4-5: 추가 시각화

**결정**: (a) Equilibrium prior 도식 + (b) View 통합 도식

**근거**:
1. **Equilibrium prior** = CAPM 시장 균형 직관 (Black-Litterman 1990 핵심)
2. **View 통합** = Bayesian update 시각 (BL 의 핵심 메커니즘)
3. (c) τ sensitivity 는 영역 8 (한계) 의 "BL prior 가정" 한계와 연결

#### 결정 항목 M4-6: 우리 펀드 (mat_eq_eq_raw_pap) 강조

**결정**: (b) 시각화 highlight + (c) 선택 이유 narrative

**근거**:
1. **시각화 highlight** = 4-slot 표 안에 ★ 마커 + Cobalt Blue accent
2. **선택 이유 narrative** = "별도 기준으로 선정된 Top 1 config (Backtesting 페이지 참조)" 박스
3. (a) 별도 박스만은 시각 강조 ↓

**narrative 박스 텍스트 안**:
```
✅ 본 펀드 config: mat_eq_eq_(raw)_pap

별도 기준으로 선정된 Top 1 config 입니다. 4-slot 조합 64개
(3 × 2 × 4 × 4 + 일부 제외) 중 학술/실무 다중 메트릭 검증을
통해 선정되었습니다.

자세한 선정 절차는 Backtesting 페이지의 Regime 분석 + Sensitivity
Test 참조.
```

#### 시각화 예시 (확정 사항 조합)

```
[Methodology Overview Sankey (영역 3)]

┌─ Black-Litterman 상세 ───────────────────────────────────┐
│                                                          │
│ ┌─ 1. Equilibrium Prior 도식 ───────────────────────┐   │
│ │ Market (SPY) → CAPM equilibrium → Π (prior)        │   │
│ │  수식: Π = δΣw_market                              │   │
│ │  • δ = risk aversion                               │   │
│ │  • Σ = covariance (Hover: 정의)                    │   │
│ │  • w_market = market cap weights                   │   │
│ │  ⓘ Black & Litterman (1990) [Click: 인용 link]     │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─ 2. View 통합 (Bayesian Update) ──────────────────┐   │
│ │ Π (prior) + P/Q/Ω (view) → E[R] (posterior)        │   │
│ │  E[R] = [(τΣ)^(-1) + P^T Ω^(-1) P]^(-1) ·          │   │
│ │         [(τΣ)^(-1)Π + P^T Ω^(-1) Q]                │   │
│ │  • τ = uncertainty (Hover)                         │   │
│ │  • P = view matrix                                 │   │
│ │  • Q = view returns                                │   │
│ │  • Ω = view confidence (LSTM 활용)                 │   │
│ │  ⓘ He & Litterman (1999), Idzorek (2005)           │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─ 3. 4-slot config (★ 본 펀드 강조) ──────────────┐   │
│ │ Slot      | Options                  | Our Fund   │   │
│ │ prior     | mat / mcap / rp          | mat ★      │   │
│ │ p_weight  | eq / mcap                | eq ★       │   │
│ │ q_mode    | eq / mcap / lam / raw    | eq ★       │   │
│ │ omega_mode| pap / he / lam / rms     | pap ★      │   │
│ │                                                    │   │
│ │ → 본 펀드 config: mat_eq_eq_(raw)_pap              │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─ ✅ 왜 mat_eq_eq_raw_pap 인가? ───────────────────┐   │
│ │ 별도 기준으로 선정된 Top 1 config. 4-slot 조합     │   │
│ │ 64개 중 학술/실무 다중 메트릭 검증 결과.           │   │
│ │ 자세한 절차는 Backtesting 페이지 참조.             │   │
│ └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- LaTeX 수식 = `st.latex` 또는 `st.markdown` 활용
- 도식 = matplotlib / Plotly 정적 이미지 또는 `st.image`
- 4-slot 표 = pandas DataFrame + Streamlit Styler (★ 마커 highlight)
- 학술 인용 link = Google Scholar / DOI link
- Backtesting 페이지의 Sensitivity Test 와 narrative 연결

### 영역 5: LSTM 변동성 예측 상세 — 확정 (walk-forward 명시)

#### 영역 5 통합 배경 (Context)

LSTM 모델의 학술 설명 + 변동성 예측 워크플로우 + **walk-forward 학습 절차** + 4번 재학습 검증.

**학술 토대**:
- Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"
- Gers, Schmidhuber, Cummins (2000) "Learning to Forget"
- Kim & Won (2018) "Forecasting the volatility of stock price index"
- **Lopez de Prado (2018) "Advances in Financial Machine Learning"** — walk-forward validation

#### ★ Walk-forward 구조 (사용자 지적 반영 — 명확 기록)

**모델 구조 정확 파악** (`final/lstm_pipeline.py` V4_BEST_CONFIG):
- **is_len: 1250일** (~5년 일별 데이터로 학습)
- **oos_len: 21일** (~1개월 OOS 예측)
- **step: 21일** (월별 슬라이딩)
- **embargo: 63일** (~3개월 buffer, 학습/예측 사이)
- **seq_len: 63일** (sequence length)
- → 615 종목 각각 walk-forward (~120 fold) + HAR-RV 앙상블

**LSTM walk-forward 흐름**:
```
[t=2010-01]
  ├─ 학습: 1250일 (2005-01 ~ 2010-01)
  ├─ Embargo: 63일 buffer
  └─ OOS 예측: 21일 (다음 1개월)

[t=2010-02 (21일 슬라이딩)]
  ├─ 학습: 1250일 (2005-02 ~ 2010-02)
  ├─ Embargo: 63일 buffer
  └─ OOS 예측: 21일 (다음 1개월)

... 192개월 동안 반복 ...
```

→ **LSTM 자체가 완전한 walk-forward 학습 구조** (Lopez de Prado 2018 표준)

#### BL Walk-forward 통합

**`final/bl_functions.py` 흐름**:
- 매월 (192개월) `walk_forward` 수행
- 각 월의 `pred_date` 에서:
  - `train_window=60개월` 일별 데이터로 Σ 추정
  - LSTM 예측 변동성 활용 (`p_mode='lstm_predicted'`)
  - BL 모델 → weight 결정
- → BL 도 완전한 walk-forward

#### TEST / HOLD_OUT 의 정확한 의미

**`final/master_table.py` EVAL_PERIODS**:
- **TEST 168m** (2010-01 ~ 2023-12) = walk-forward 결과의 in-sample evaluation (config selection 평가)
- **HOLD_OUT 24m** (2024-01 ~ 2025-12) = walk-forward 결과의 true out-of-sample (untouched)
- **FULL 192m** = 보조 통합 비교

→ **TEST/HOLD_OUT 은 학습/검증 분리가 아닌, walk-forward 결과의 평가 기간 분리**

**우리 펀드 적용**:
- Input: 과거 60일 일별 수익률 + 보조 feature (sector dummy / VIX)
- Output: 다음 월 변동성 (σ_next)
- BL 통합: σ_next → BL Ω (view confidence)
- 학습: 615 종목 × ~120 fold walk-forward

#### 결정 항목 M5-1: 표시 형식

**결정**: (b) 도식 + 수식

**근거**: 영역 4 (BL 상세) 일관 — 시각 + 학술 균형

#### 결정 항목 M5-2: LSTM 구조 시각화

**결정**: (c) Cell + Architecture 둘 다

**근거**:
1. **Cell 도식** = LSTM 학술 표준 (Forget/Input/Output gates)
2. **Architecture** = 우리 모델 구조 (Layer block)
3. 학술 일반성 + 우리 펀드 특수성 모두 노출

#### 결정 항목 M5-3: Input / Output 명시

**결정**: (a) 표 형식

**근거**:
1. **표** = 정확 (Input feature 목록 / Output 정의 명확)
2. (b) 도식은 영역 5 안 다른 도식 (Cell, Architecture) 와 중복
3. (c) 둘 다는 정보 과부하

**Input/Output 표**:
| Type | Item | Description |
|---|---|---|
| Input | Past 60d returns | 과거 60일 일별 수익률 |
| Input | Sector dummy | 11개 GICS 섹터 one-hot |
| Input | Market state | SPY rolling vol / VIX 등 |
| Output | σ_next | 다음 월 변동성 예측 |
| Downstream | → BL Ω | View confidence 입력 |

#### 결정 항목 M5-4: 학습 절차

**결정**: (c) 학습 곡선 + 4번 재학습 검증 보고서

**근거**:
1. **학습 곡선** (loss vs epoch / val_loss) = 학술 표준
2. **4번 재학습 검증 보고서** (2026-05-08) = 학술 정직성 ★★★
3. 학습편향 가능성 노출 → 영역 8 (한계) 와 narrative 연결
4. (d) 간략 설명은 학술 정직성 ↓

#### 결정 항목 M5-5: 학술 인용

**결정**: (b) 표준 3개

**3개 인용**:
1. Hochreiter & Schmidhuber (1997) — LSTM 원조
2. Gers, Schmidhuber, Cummins (2000) — Forget gate
3. Kim & Won (2018) — 금융 변동성 예측 적용

#### 결정 항목 M5-6: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip (LSTM gates 정의) ✓
- Click → 학술 인용 link ✓
- 학습 곡선 zoom ✓ (Plotly 기본)

#### 시각화 예시 (확정 사항 조합)

```
[Black-Litterman 상세 (영역 4)]

┌─ LSTM 변동성 예측 상세 ──────────────────────────────────┐
│                                                          │
│ ┌─ 1. LSTM Cell 도식 + 수식 ────────────────────────┐   │
│ │ [LSTM Cell 도식: Forget/Input/Output gates]         │   │
│ │  • Forget: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     │   │
│ │  • Input:  i_t = σ(W_i · ...)                      │   │
│ │  • Output: o_t = σ(W_o · ...)                      │   │
│ │  ⓘ Hochreiter & Schmidhuber (1997)                 │   │
│ │  ⓘ Gers, Schmidhuber, Cummins (2000) — Forget     │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─ 2. Network Architecture ─────────────────────────┐   │
│ │ Input (60d × N features)                           │   │
│ │   ↓                                                │   │
│ │ LSTM Layer (units=X)                               │   │
│ │   ↓                                                │   │
│ │ Dropout                                            │   │
│ │   ↓                                                │   │
│ │ Dense (units=Y)                                    │   │
│ │   ↓                                                │   │
│ │ σ_next 예측                                        │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─ 3. Input / Output 표 ────────────────────────────┐   │
│ │ Type    | Item             | Description          │   │
│ │ Input   | Past 60d returns | 과거 60일 일별 수익률│   │
│ │ Input   | Sector dummy     | GICS 11개 one-hot    │   │
│ │ Input   | Market state     | SPY vol / VIX        │   │
│ │ Output  | σ_next           | 다음 월 변동성 예측  │   │
│ │ → BL Ω  | View confidence  | 변동성 → confidence  │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─ 4. 학습 절차 + 4번 재학습 검증 보고서 ───────────┐   │
│ │ 학습 곡선:                                         │   │
│ │  Epoch  ────────────                              │   │
│ │  Loss   ╲╲╲╲╲____ (training)                       │   │
│ │  Val    ╲╲╲╲╲____ (validation)                     │   │
│ │  Click → zoom                                      │   │
│ │                                                     │   │
│ │ 4번 재학습 검증 결과 (2026-05-08 보고서):           │   │
│ │  • 재학습 1: Val sortino X.XX                       │   │
│ │  • 재학습 2: Val sortino X.XX                       │   │
│ │  • 재학습 3: Val sortino X.XX                       │   │
│ │  • 재학습 4: Val sortino X.XX                       │   │
│ │  → 학습편향 가능성 식별 (영역 8 한계 참조)          │   │
│ │  ⓘ Kim & Won (2018) — 금융 LSTM 적용                │   │
│ └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- LSTM Cell 도식 = 표준 LSTM diagram (참조 이미지 활용)
- Architecture = matplotlib Block diagram 또는 SVG
- 학습 곡선 = Plotly `go.Scatter` (training/val loss)
- 4번 재학습 보고서 = 이전 폐기된 노트북의 검증 보고서 참조
- 학술 인용 (Hochreiter 1997, Gers 2000, Kim-Won 2018) = 부록 학술 근거 일람 추가
- 영역 8 한계 narrative 와 연결 (학습편향 가능성)

### 영역 6: Factor 분석 (CAPM + FF5) — 확정 (조건부)

#### ★ Factor 분석 영역의 의미와 목적 (사용자 강조 — 명확 기록)

##### 학술적 의미

**핵심 질문**: "펀드의 outperformance 가 진정한 운용 skill 때문인가, 아니면 단순 factor 노출 때문인가?"

**예시**:
- 펀드 A 가 시장 대비 +5% 초과수익
- 그런데 펀드 A 가 small-cap 에 많이 투자
- SMB factor 가 +5% → 펀드 수익은 단순 factor 노출 결과
- 실제 alpha = 0 (skill 없음)
- → Factor 통제 후 잔여 alpha 가 진짜 skill

**Factor 모델 발전사**:
- Jensen (1968) CAPM Alpha — 시장 노출 (β_MKT) 통제 후 잔여
- Fama-French (1993) FF3 — + Size (SMB) + Value (HML)
- Carhart (1997) 4-factor — + Momentum
- Fama-French (2015) FF5 — + Profitability (RMW) + Investment (CMA)

→ Factor 가 많을수록 alpha 검증이 엄격

##### 우리 펀드 대시보드에서의 4가지 목적

**1. BL+LSTM 의 진정한 가치 검증**
- 우리 펀드 = sector 분산 + LSTM 변동성 예측 + BL 적응형
- Factor 통제 후 양의 alpha 가 나오는가?
  - **나온다면**: BL+LSTM 의 진정한 skill 입증
  - **안 나온다면**: 단순 factor 노출 결과 → 학술 정직성 인정

**2. 펀드의 factor tilt 명시**
- β_MKT < 1 기대 ("VolControl" 정체성 부합)
- β_SMB / HML / RMW / CMA 노출 패턴 노출

**3. HO 정당화 narrative 보완**
- HO 24m 부진을 factor 노출로 분해 가능
- 예: HO 시기 momentum factor 가 SPY 에 강력 → 우리 펀드 momentum 부족 → 부진
- "분산 운용 trade-off" narrative 의 학술 검증

**4. 학술 심사위원 어필**
- Factor 분석 = 학술 펀드 평가의 필수 절차
- CFA / 학술 논문 / 펀드 등급 평가 모두 사용
- 본 펀드의 학술 정확성 ↑↑

##### 위험 / 한계 (사용자 인지 사항)

| 위험 | 설명 |
|---|---|
| **결과가 부정적일 가능성** | alpha 가 양수가 아닐 수 있음 (단순 factor tilt 결과일 수도) |
| **회귀 결과 해석 어려움** | p-value, R², CI 등 통계 지식 필요 |
| **HO 부진 강조 가능** | Factor 통제 후 HO 부진이 더 부각될 수 있음 |

##### 영역 6 처리 결정

**검토된 옵션**:
- (A) 영역 6 전체 유지 (학술 깊이 max)
- (B) 단순화 (Alpha 카드 + Factor Exposure 만)
- (C) 제거 (About 또는 별도 위임)
- (D) 유지 + Plain English narrative 강화

**결정**: (A) 영역 6 전체 유지 (조건부)

**조건**:
1. **일단 생성** — 학술 깊이 우선
2. **결과 확인 후 재평가**:
   - alpha 가 양수 + 통계적 유의 → narrative 강화 + 유지
   - alpha 음수 / 통계적 무의 → 영역 축소 (B) 또는 제거 (C) 검토
3. **재평가 시점**: 구현 후 결과 확인 (D~L 섹션 / plan.md 작성 단계)

#### 영역 6 학술 토대 (참조)

**CAPM 회귀 모델**:
$$R_p - R_f = \alpha + \beta_{MKT}(R_m - R_f) + \epsilon$$

**FF5 회귀 모델**:
$$R_p - R_f = \alpha + \beta_{MKT}MKT + \beta_{SMB}SMB + \beta_{HML}HML + \beta_{RMW}RMW + \beta_{CMA}CMA + \epsilon$$

#### 결정 항목 M6-1: Factor 모델

**결정**: (c) CAPM + FF5 둘 다

**근거**:
1. **CAPM** = 단순 baseline (시장 노출만)
2. **FF5** = 학술 표준 (5 factor 통제)
3. 두 모델 비교 = alpha 의 강건성 검증
4. (d) Carhart 추가는 정보 과부하

#### 결정 항목 M6-2: 표시 형식

**결정**: (c) 표 + 막대 둘 다

**근거**:
1. **회귀 표** = 정확 수치 (α, β, R², p-value)
2. **Factor Exposure 막대** = β 시각화
3. (a) 표만은 시각 ↓ / (b) 막대만은 정확성 ↓

#### 결정 항목 M6-3: Alpha 시각화

**결정**: (c) Card + Rolling 둘 다

**근거**:
1. **Card** = 단일 값 (annualized alpha)
2. **Rolling Alpha** = 시간 안정성 (Performance 영역 5 와 차별화 — 여기는 alpha rolling)
3. HO 시기 alpha 변화 시각

#### 결정 항목 M6-4: 통계 보강

**결정**: (a) R²+p-value + (b) 95% CI

**근거**:
1. **R²** = 모델 설명력
2. **p-value α** = alpha 통계적 유의성
3. **95% CI α** = 신뢰구간 (Bootstrap 보다 단순)
4. (c) Bootstrap 은 구현 비용 ↑ → 보류

#### 결정 항목 M6-5: 비교

**결정**: (b) 펀드 vs SPY

**근거**:
1. SPY = 시장 baseline (CAPM 의 R_m)
2. 다중 토글 (EW/IVW) 은 영역 4 (BL) 에서 다룸
3. Factor 분석 자체가 vs SPY 비교 메트릭

#### 결정 항목 M6-6: 학술 인용

**결정**: (b) 표준 4개

**4개 인용**:
1. Jensen (1968) — CAPM Alpha 원조
2. Fama-French (1993) — FF3
3. Carhart (1997) — 4-factor (momentum)
4. Fama-French (2015) — FF5 (현재 표준)

#### 시각화 예시 (확정 사항 조합)

```
[LSTM 변동성 예측 상세 (영역 5)]

┌─ Factor 분석 (CAPM + FF5) ───────────────────────────────┐
│                                                          │
│ ┌─ 1. Annualized Alpha Card ────────────────────────┐   │
│ │ CAPM α: +X.XX% / yr (95% CI: [X, X], p=0.XX)       │   │
│ │ FF5 α:  +X.XX% / yr (95% CI: [X, X], p=0.XX)       │   │
│ │ ⓘ Jensen (1968), Fama-French (1993, 2015)          │   │
│ │ ⓘ Carhart (1997)                                   │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─ 2. 회귀 결과 표 ────────────────────────────────┐    │
│ │           | CAPM       | FF5                       │   │
│ │ α (annual)| +X.XX%     | +X.XX%                    │   │
│ │ β_MKT     | X.XX (***) | X.XX (***)                │   │
│ │ β_SMB     | -          | X.XX (**)                 │   │
│ │ β_HML     | -          | X.XX (*)                  │   │
│ │ β_RMW     | -          | X.XX                      │   │
│ │ β_CMA     | -          | X.XX                      │   │
│ │ R²        | XX%        | XX%                       │   │
│ │ p-value α | 0.XX       | 0.XX                      │   │
│ │ 95% CI α  | [X%, X%]   | [X%, X%]                  │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─ 3. FF5 Factor Exposure 막대 ─────────────────────┐   │
│ │ β_MKT   │██████ X.XX                                │   │
│ │ β_SMB   │██   X.XX (size: small ↑)                 │   │
│ │ β_HML   │█    X.XX (value: cheap ↑)                │   │
│ │ β_RMW   │██   X.XX (profitability: high ↑)         │   │
│ │ β_CMA   │█    X.XX (investment: conservative ↑)    │   │
│ └────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─ 4. Rolling Alpha 시계열 (FF5) ───────────────────┐   │
│ │  Alpha%                                            │   │
│ │  +5%┤    ╱─────╲                                   │   │
│ │  +3%┤  ╱╱       ╲                                  │   │
│ │  +1%┤╱           ╲╱╲                               │   │
│ │  -1%┤                ╲   ★HO 부진?                 │   │
│ │      2010  2014  2018  2022  2025                  │   │
│ └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- **재평가 트리거**: 구현 후 결과가 부정적일 시 → (B) 단순화 또는 (C) 제거 검토
- 학술 인용 (Jensen 1968, Fama-French 1993/2015, Carhart 1997) = 부록 학술 근거 일람 추가
- Factor 데이터 = `data/ff3_monthly.csv`, `data/ff5_monthly.csv` (이미 보유)
- 회귀 = `statsmodels.OLS` 활용
- 95% CI = `statsmodels` 결과의 `conf_int(alpha=0.05)` 활용
- Rolling Alpha = `pd.rolling.apply(lambda x: OLS(...).fit().params['Intercept'])` 활용

### 영역 7: 정규성 검정 (Jarque-Bera) — 확정

#### ★ 영역 7 의미 재구성 (사용자 지적 반영)

##### 사용자 핵심 지적 (★ 명확 기록)

**"우리는 수익률이 아닌 변동성을 예측함을 기억할 것."**

→ 영역 7 의 narrative 가 단순 "수익률 정규성 검정 → 위험 메트릭 가정" 이 아닌, **LSTM 변동성 예측의 학술적 정당화** 로 재정의되어야 함.

##### 수정된 영역 7 의미와 목적

**핵심 logic chain**:
1. **수익률이 정규분포라면** → 단순 통계 모델 (GARCH 등) 로 변동성 예측 충분
2. **수익률이 fat tail / 비대칭 분포 (정규분포 기각)** → 단순 모델 한계 → **비선형 모델 (LSTM) 필요**
3. **Jarque-Bera 가 정규분포 기각 → LSTM 변동성 예측의 학술적 정당화**

**우리 펀드 narrative 와의 관계**:
- 영역 5 (LSTM 변동성 예측 상세) — LSTM 구조 + 학습 절차
- 영역 7 (Jarque-Bera) — **왜 LSTM 인가** 학술 검증
- 영역 5 + 영역 7 = **LSTM 채택의 학술적 정당화 narrative**

#### 학술 토대

- Jarque & Bera (1980) "Efficient tests for normality, homoscedasticity and serial independence of regression residuals"
- Embrechts, Klüppelberg, Mikosch (1997) "Modelling Extremal Events" — fat tail 일반론
- Cont (2001) "Empirical properties of asset returns: stylized facts and statistical issues" — 금융 수익률의 fat tail (LSTM 정당화 근거)

#### Jarque-Bera 수식

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

- $S$ = Skewness, $K$ = Kurtosis, $n$ = 샘플 수
- $H_0$: 정규분포 / $H_1$: 정규분포 아님
- p-value < 0.05 → 정규분포 기각

#### 결정 항목 M7-1: 표시 형식

**결정**: (c) Card + 보조 시각화

**근거**:
1. **Card** = JB 통계량 + p-value 정확 노출
2. **보조 시각화** = Q-Q plot + 히스토그램 (직관)
3. (d) 추가 검정 포함은 narrative 단순화 ↓

#### 결정 항목 M7-2: 추가 정규성 검정

**결정**: (a) Jarque-Bera only

**근거**:
1. **단순화** — Performance S8-3 결정에서 Jarque-Bera 만 사용 결정
2. (b) Anderson-Darling / (c) Shapiro-Wilk 추가는 비슷한 결과 → 정보 중복
3. JB = 학술 표준 검정

#### 결정 항목 M7-3: 시각화

**결정**: (c) Q-Q plot + 히스토그램 둘 다

**근거**:
1. **Q-Q plot** = 정규분포 대비 deviation 시각 (학술 표준)
2. **히스토그램** = 분포 모양 직관
3. 두 시각화가 보완 (Q-Q = 정량 / 히스토그램 = 정성)

#### 결정 항목 M7-4: 비교

**결정**: (b) 펀드 vs SPY + 일별/월별 토글

**근거**:
1. **펀드 vs SPY** = 비교 narrative (둘 다 fat tail 인지)
2. **일별/월별 토글** = 시간 단위별 정규성 검증
3. 일별 N 더 큼 → 정규성 기각 더 명확

#### 결정 항목 M7-5: narrative

**결정**: (b) 의미 + 결과 + 함의 + **결과가 부정적일 경우에만 한계 추가 (동적)**

##### 동적 narrative logic

**Case A: 정규분포 기각 (p < 0.05) — fat tail 확인**

```
[narrative]
✅ Jarque-Bera 검정 결과: 정규분포 가설 기각 (p < 0.05)

[함의]
수익률이 fat tail / 비대칭 분포임이 통계적으로 입증되었습니다.
이는 LSTM 같은 비선형 모델로 변동성을 예측하는 것이 학술적으로
정당화됨을 의미합니다 (Cont 2001).

[관련 영역]
→ Risk Metrics 영역 8 (Hill estimator) 에서 fat tail 정량화
→ 영역 5 (LSTM 변동성 예측) 의 학술 정당화 보완
```

**Case B: 정규분포 채택 (p ≥ 0.05) — 정규분포에 가까움**

```
[narrative]
⚠ Jarque-Bera 검정 결과: 정규분포 가설 채택 (p ≥ 0.05)

[함의]
수익률이 정규분포에 가까움이 시사됩니다. 이는 단순 통계 모델
로도 변동성 예측이 충분할 수 있음을 의미하며, LSTM 의 부가가치
재검토가 필요합니다.

[자동 추가 — 영역 8 한계 narrative]
→ "단순 모델 대비 LSTM 의 우위 미입증" 한계 카드 추가
```

##### 동적 narrative 구현 logic

```python
if p_value_jb < 0.05:
    # Case A: fat tail 확인 → LSTM 정당화
    show_narrative("LSTM 학술 정당화 narrative")
    영역_8_한계_LSTM_가치_미입증 = False
else:
    # Case B: 정규분포 → LSTM 가치 재검토
    show_narrative("LSTM 가치 재검토 narrative")
    영역_8_한계_LSTM_가치_미입증 = True  # 영역 8 한계 카드 동적 추가
```

#### 시각화 예시 (확정 사항 + 동적 narrative)

```
[Factor 분석 (영역 6)]

┌─ 정규성 검정 (Jarque-Bera) ─────────────────────────────┐
│                                                         │
│ ┌─ ℹ️ 검정 의미 + 함의 (LSTM 학술 정당화) ───────────┐ │
│ │ Jarque-Bera 검정으로 수익률 분포의 정규성 검증.    │ │
│ │ 정규분포 기각 시 fat tail 확인 → LSTM 같은 비선형  │ │
│ │ 모델로 변동성 예측이 학술적으로 정당화됩니다.       │ │
│ │ (Cont 2001 — fat tail stylized fact)               │ │
│ └────────────────────────────────────────────────────┘ │
│                                                         │
│ [Tab: 월별 (Monthly) | 일별 (Daily)]                    │
│                                                         │
│ ┌─ Card ────────────────────────────────────────────┐  │
│ │ Series | JB     | p-value | Skew  | Kurtosis | 가설│ │
│ │ Fund   | XX.XX  | 0.XX    | -X.X  | X.X      | 기각│ │
│ │ SPY    | XX.XX  | 0.XX    | -X.X  | X.X      | 기각│ │
│ └─────────────────────────────────────────────────┘  │
│                                                         │
│ ┌─ Q-Q Plot + 히스토그램 ───────────────────────────┐ │
│ │ Q-Q plot (Fund)         Histogram + Normal line   │ │
│ │  ┄ ╱        ★            ▓▓▓ Fund                 │ │
│ │  ┄╱     ╱╱              ▓▓▓▓▓▓▓ ─── Normal       │ │
│ │  ╱   ╱╱                ▓▓▓▓▓▓▓▓▓▓                │ │
│ │ ╱ ╱╱   ★ deviation                                │ │
│ │ ★    = fat tail                                    │ │
│ └────────────────────────────────────────────────────┘ │
│                                                         │
│ ┌─ ✅ 동적 결론 narrative (Case A: 기각) ───────────┐ │
│ │ Jarque-Bera 결과 정규분포 기각 → fat tail 확인.   │ │
│ │ 단순 통계 모델 (GARCH 등) 로 변동성 예측 한계 있음.│ │
│ │ 본 펀드의 LSTM 변동성 예측 채택은 학술적 정당화.   │ │
│ │ ⓘ Cont (2001), Embrechts et al. (1997)            │ │
│ │                                                    │ │
│ │ → 영역 5 (LSTM 상세) 보완                          │ │
│ │ → Risk Metrics 영역 8 (Hill estimator) 와 연결    │ │
│ └────────────────────────────────────────────────────┘ │
│                                                         │
│ (Case B 결과 시: "LSTM 가치 재검토" narrative 표시 +   │
│  영역 8 한계 카드 자동 추가)                            │
└─────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- **재구성된 narrative**: 영역 7 = "왜 LSTM 인가" 학술 검증 (수익률 정규성 검정 → LSTM 정당화)
- 동적 narrative = 검정 결과에 따라 자동 변경 (Case A/B)
- 영역 8 한계 카드와 동적 연결 (Case B 시 카드 추가)
- 학술 인용 추가:
  - Jarque & Bera (1980)
  - **Cont (2001) "Empirical properties of asset returns: stylized facts and statistical issues"** — fat tail stylized fact (LSTM 정당화 핵심)
  - Embrechts et al. (1997) — fat tail 일반론

### 영역 8: 한계 + 향후 개선 (카드 + Expander) — 확정

#### 영역 8 통합 배경 (Context)

학술 정직성 강화 영역 — 펀드의 5가지 한계 차원 + 향후 개선 방향 + 학술 인용.

**3가지 한계 차원** (사용자 균형 옵션 B + 모델 구조 정확화 반영, Q-A1):

★ **이전 결정 (5개) → 3개 축소** — 균형 옵션 (B) 적용:

| 변경 | 이유 |
|---|---|
| ❌ 데이터 한계 (Walk-forward 미적용) | 사실 적용됨 (LSTM walk-forward + BL walk-forward) |
| ❌ 데이터 한계 (is_len/embargo sensitivity) | 시계열_Test 폴더에서 검증 완료 |
| ❌ 모델 한계 카드 | 균형 (B) — 학술 깊이 청중 부담 ↑ |
| ✓ HO 24m 부진 인정 | 핵심 narrative (감출 수 없음) |
| ✓ 향후 개선 방향 | 마케팅 친화 (Future Work 강조) |
| ✓ 실무 적용 제약 | 표준 disclosure |

**3가지 한계 차원**:
1. 🟧 **HO 24m 부진 인정** — SPY 대비 -12.9%p / Sector trade-off
2. 🟩 **향후 개선 방향** — Multi-factor / 실거래 시뮬레이션 / 추가 학술 검증
3. 🟥 **실무 적용 제약** — 가상 펀드 / 운용 규모 / Tax 미반영

**Selection Bias / Data Snooping (Q-B3)**: About 페이지 학술 부록으로 이동 (대시보드 본문에서는 제외)

**동적 추가** (영역 7 결과 Case B 시):
- "단순 모델 대비 LSTM 우위 미입증" 카드 추가 — "향후 개선 방향" 카드 안의 ablation study 항목으로 부드럽게 통합

#### 결정 항목 M8-1: 카드 디자인

**결정**: (c) 카테고리 색상 코딩

**근거**:
1. **5가지 차원 시각 구분** — 청중이 차원별 즉시 인지
2. (a) Holdings 일관은 시각적 차별화 ↓
3. (b) warning 톤만은 부정적 인상 ↑↑

**카테고리 색상 안**:
- 🟦 데이터 한계: Blue (`#3B82F6`)
- 🟪 모델 한계: Purple (`#8B5CF6`)
- 🟧 HO 부진: Orange (`#F59E0B`)
- 🟩 개선 방향: Green (`#10B981`)
- 🟥 실무 제약: Red (`#EF4444`)

#### 결정 항목 M8-2: Expander 내용

**결정**: (b) 학술 인용 + 텍스트

**근거**:
1. **학술 인용** = 한계 차원의 학술 토대 명시 (Lopez de Prado / Engle / Markowitz 등)
2. (a) 단순 텍스트는 학술 정직성 ↓
3. (c) 시각화는 영역 8 분량 ↑↑

**예상 인용** (차원별):
- 데이터 한계: Lopez de Prado (2018) "Advances in Financial Machine Learning"
- 모델 한계: Engle (2002) "DCC-GARCH"
- HO 부진: Markowitz (1952) (분산 trade-off)
- 개선 방향: Fama-French (2015) FF5
- 실무 제약: Frazzini, Israel, Moskowitz (2018) (Trading costs)

#### 결정 항목 M8-3: 동적 카드 추가

**결정**: (a) 동적 추가 채택

**근거**:
1. **영역 7 (Jarque-Bera) 동적 narrative 와 통합**
2. Case B (정규분포 채택) 시 → "LSTM 가치 미입증" 카드 자동 추가
3. 학술 정직성 ★★★ — 결과에 따른 동적 한계 인정

**동적 카드 텍스트**:
```
⚠ LSTM 가치 미입증 (영역 7 Jarque-Bera 결과)

정규분포 채택 → 단순 모델 대비 LSTM 우위 재검토 필요.
향후 ablation study (BL only vs BL+LSTM) 을 통해
LSTM 가치 정량 검증 권장.
```

#### 결정 항목 M8-4: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- Expander 펼침/접기 ✓ (M-1 결정)
- 학술 인용 link (Google Scholar / DOI) ✓
- 향후 개선 카드 → Backtesting 페이지 navigation ✓

#### 결정 항목 M8-5: 추가 표시

**결정**: (c) 학술 정직성 선언 박스

**근거**:
1. **펀드 신뢰성 narrative 강화** — 한계 인정 = 신뢰성 토대
2. (a) PDF 다운로드는 별도 작업 (구현 비용 ↑)
3. (b) Future Work 박스는 향후 개선 카드와 중복

**학술 정직성 선언 텍스트** (확정):
```
✅ 학술 정직성 선언

본 펀드는 학술 정직성을 위해 모든 한계를 명시합니다.
한계 인정은 신뢰성 강화의 토대이며, 향후 개선 방향을
통해 지속적 발전을 추구합니다.
```

#### 시각화 예시 (확정 사항 조합)

```
[정규성 검정 (영역 7)]

┌─ 한계 + 향후 개선 ───────────────────────────────────────┐
│                                                          │
│ ┌─ ✅ 학술 정직성 선언 ────────────────────────────┐    │
│ │ 본 펀드는 학술 정직성을 위해 모든 한계를 명시합니다.│   │
│ │ 한계 인정은 신뢰성 강화의 토대이며, 향후 개선 방향│   │
│ │ 을 통해 지속적 발전을 추구합니다.                  │    │
│ └────────────────────────────────────────────────────┘  │
│                                                          │
│ [5가지 카드 그리드 — 카테고리 색상 코딩]                  │
│ ┌──────────┬──────────┬──────────┬──────────┬──────────┐│
│ │ 🟦 데이터│ 🟪 모델  │ 🟧 HO    │ 🟩 개선  │ 🟥 실무  ││
│ │  한계    │  한계    │  부진    │  방향    │  제약    ││
│ │ (blue)   │ (purple) │ (orange) │ (green)  │ (red)    ││
│ └──────────┴──────────┴──────────┴──────────┴──────────┘│
│                                                          │
│ ▼ 🟦 데이터 한계 [Expander] [학술 인용]                  │
│   • 학습 168m / 검증 24m — 단일 OOS                      │
│   • Walk-forward 미적용                                  │
│   • 거래비용 20bp 가정 (slippage 미반영)                 │
│   ⓘ Lopez de Prado (2018) [Click: link]                 │
│                                                          │
│ ▼ 🟪 모델 한계 [Expander] [학술 인용]                    │
│   • LSTM 1차 변동성 (공분산 미예측)                      │
│   • BL CAPM prior                                        │
│   • Sector rotation 단기 신호 미반영                     │
│   ⓘ Engle (2002) — DCC-GARCH                            │
│                                                          │
│ ▼ 🟧 HO 24m 부진 인정 [Expander]                          │
│   • SPY 대비 -12.9%p                                     │
│   • IT under-weight (분산 trade-off)                     │
│   • Sector Watch 영역 8 narrative 일관                   │
│   ⓘ Markowitz (1952) — 평균-분산 이론                   │
│                                                          │
│ ▼ 🟩 향후 개선 방향 [Expander] [Backtesting nav]          │
│   • Multi-factor (Momentum + Value + Quality)            │
│   • Walk-forward validation                              │
│   • 실제 매매 시뮬레이션                                 │
│   ⓘ Fama-French (2015) FF5                              │
│   → [Backtesting 페이지로 이동]                          │
│                                                          │
│ ▼ 🟥 실무 적용 제약 [Expander]                            │
│   • 가상 펀드 (실제 운용 X)                              │
│   • 운용 규모 가정 (소형)                                │
│   • Tax / 유동성 제약 미반영                             │
│   ⓘ Frazzini, Israel, Moskowitz (2018) — Trading costs  │
│                                                          │
│ [동적 추가 카드 — 영역 7 Case B 시]                       │
│ ┌──────────────────────────────────────────────────┐    │
│ │ ⚠ LSTM 가치 미입증                                │    │
│ │ (영역 7 Jarque-Bera Case B 시 동적 표시)          │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- 카드 색상 = `st.markdown(unsafe_allow_html=True)` + CSS background-color
- Expander = `st.expander` (Streamlit 표준)
- 학술 인용 link = `st.markdown` 의 markdown link
- Backtesting navigation = `st.session_state.page = 'Backtesting'` + `st.rerun()`
- 동적 카드 = `if jarque_bera_p_value >= 0.05: show_dynamic_card()`

---

### Methodology 페이지 — 전체 확정 (영역 1~9)

#### 페이지 시각화 통합

```
┌────────────────────────────────────────────────────────────────┐
│ [영역 1: Header — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 2: Sub-header]                                            │
│ Methodology (방법론) — BL + LSTM + Factor + 한계 명시           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 3: Methodology Overview (Plotly Sankey)]                  │
│ 7-9 노드 4그룹 색상 + narrative 박스                            │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 4: Black-Litterman 상세 + 4-slot config]                  │
│ Equilibrium + View 통합 + 4-slot 표 (★ 본 펀드 강조)           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 5: LSTM 변동성 예측 상세]                                 │
│ Cell + Architecture + Input/Output 표 + 학습 곡선 + 4번 재학습  │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 6: Factor 분석 (CAPM + FF5)] — 조건부 (결과 확인 후 검토) │
│ Card + 회귀 표 + Factor Exposure 막대 + Rolling Alpha          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 7: 정규성 검정 (Jarque-Bera)] — LSTM 정당화               │
│ Card + Q-Q plot + 히스토그램 + 동적 narrative (Case A/B)        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 8: 한계 + 향후 개선 (카드 + Expander)]                    │
│ 5가지 차원 색상 카드 + 학술 인용 + 학술 정직성 선언 + 동적 추가 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 9: Footer — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘
```

#### Methodology 페이지 결정 결과 / 함의

- **8 영역 모두 확정** → 학술 정직성 강화
- **재구성된 narrative**:
  - 영역 7 (Jarque-Bera) = "왜 LSTM 인가" 학술 정당화
  - 영역 8 = 학술 정직성 선언 + 5가지 한계 + 동적 추가
- **재평가 트리거** (영역 6): Factor 분석 결과 확인 후 축소/삭제 검토
- **인터랙션 일관성 원칙**:
  - Sankey (영역 3) 노드 클릭 → 영역 4-7 navigation
  - 영역 8 한계 카드 → Backtesting 페이지 navigation
  - 영역 7 Jarque-Bera 결과 → 영역 8 한계 카드 동적 추가
- 학술 인용 추가 (10개 이상):
  - Markowitz (1952), Black-Litterman (1990, 1992), He-Litterman (1999), Idzorek (2005)
  - Hochreiter (1997), Gers (2000), Kim-Won (2018)
  - Jensen (1968), Fama-French (1993, 2015), Carhart (1997)
  - Jarque-Bera (1980), Cont (2001), Embrechts et al. (1997)
  - Lopez de Prado (2018), Engle (2002), Frazzini et al. (2018)

### Methodology 페이지 → Backtesting 페이지로 진행

---


---

[← 06_sector_watch.md](06_sector_watch.md) | [08_backtesting.md](08_backtesting.md) →
