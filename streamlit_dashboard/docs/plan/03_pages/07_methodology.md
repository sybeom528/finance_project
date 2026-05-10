# Methodology 페이지 — 와이어프레임

> **관련 decisionlog**: `07_methodology.md`
> **상태**: 확정 (★ LSTM walk-forward 명시 + 5→3 한계 축소 + 동적 narrative)
> **결정 수**: 8 영역 (메타 Method M-1~M-4 + 영역 1~9)

---

## 페이지 역할 정의

펀드 운용 방법론 설명 전담 페이지. **BL (Black-Litterman) + LSTM 변동성 예측 + 4-slot config** 의 학술/실무 토대 노출.

**vs 다른 페이지**:
- Performance / Risk / Holdings = **결과 위주**
- **Methodology = 과정 / 토대 위주**

---

## 페이지 영역 구조 (시선 흐름)

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트)
3. Methodology Overview         (BL+LSTM Plotly Sankey)
4. Black-Litterman 상세         (학술 설명 + 4-slot config + ★ 본 펀드 강조)
5. LSTM 변동성 예측 상세        (Cell + Architecture + Input/Output + 학습 곡선 + 4번 재학습 검증)
6. Factor 분석 (CAPM / FF5)     (alpha + exposures, 조건부 — 결과 확인 후 재평가)
7. 정규성 검정 (Jarque-Bera)    (★ LSTM 정당화 narrative + 동적 Case A/B)
8. 한계 + 향후 개선             (★ 3개 한계 카드 + Expander + 동적 추가)
9. Footer                       (Overview 동일)
```

---

## 영역별 와이어프레임

### 영역 1: Header — Overview 동일

→ `02_common.md` 의 `render_page_header()` 호출

---

### 영역 2: Sub-header

**결정사항** (M2-1):
- (a) Performance 패턴 동일

**텍스트 안**:
```
Methodology (방법론)
Black-Litterman + LSTM 변동성 예측 + Factor 분석 + 한계 명시.
사이드바에서 기간 + 비교 토글 가능.
```

→ `02_common.md` 의 `render_subheader()` 호출

---

### 영역 3: Methodology Overview (Plotly Sankey)

**결정사항** (Method M-4 + M3-1 ~ M3-5):
- 표시 형식: (c2) Plotly Sankey
- 노드 구성: (b) 표준 7-9 노드 (안: 9 노드)
- 노드 그룹화: (b) 4 그룹 (데이터 / BL / LSTM / Optimizer)
- 인터랙션: 모두 채택
- 추가 표시: (b) Sankey + 학술 narrative 박스
- 4-slot config 노출: (a) + (c) Sankey 통합 + 영역 4 자세히

**Sankey 노드 (9개)**:
1. Market Data
2. Returns Data
3. Sector / Market Cap
4. BL Prior (CAPM equilibrium)
5. LSTM Vol Predict
6. View / Confidence
7. BL Posterior
8. Optimizer (4-slot)
9. Portfolio Weights

**4그룹 색상** (`lib/colors.py` SANKEY_GROUP_COLORS):
- 데이터: Cobalt Blue `#3B82F6`
- BL: Green `#10B981`
- LSTM: Purple `#8B5CF6`
- Optimizer: Orange `#F59E0B`

**narrative 박스 텍스트**:
```
Adaptive VolControl Fund 는 Black-Litterman (1990) 의 균형
prior 와 LSTM (Hochreiter 1997) 변동성 예측을 결합한 적응형
자산배분 펀드입니다. Markowitz (1952) 평균-분산 이론에
기반하여 4-slot config 로 robust 검증을 수행합니다.
```

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] Plotly `go.Sankey` (9 노드, 4 그룹 색상)
- [ ] Hover tooltip = 노드 detail + 학술 인용
- [ ] 노드 클릭 → 영역 4-7 navigation (`st.session_state.scroll_to`)
- [ ] 학술 narrative 박스 (Markowitz, Black-Litterman, Hochreiter 인용)

**코드 snippet (Sankey)**:
```python
import plotly.graph_objects as go
from lib.colors import SANKEY_GROUP_COLORS

nodes = [
    "Market Data",      # 0 - data
    "Returns Data",     # 1 - data
    "Sector / Mcap",    # 2 - data
    "BL Prior (CAPM eq)",  # 3 - bl
    "LSTM Vol Predict", # 4 - lstm
    "View / Confidence",# 5 - lstm
    "BL Posterior",     # 6 - bl
    "Optimizer (4-slot)", # 7 - optimizer
    "Portfolio Weights" # 8 - optimizer
]
node_colors = [
    SANKEY_GROUP_COLORS["data"], SANKEY_GROUP_COLORS["data"], SANKEY_GROUP_COLORS["data"],
    SANKEY_GROUP_COLORS["bl"], SANKEY_GROUP_COLORS["lstm"], SANKEY_GROUP_COLORS["lstm"],
    SANKEY_GROUP_COLORS["bl"], SANKEY_GROUP_COLORS["optimizer"], SANKEY_GROUP_COLORS["optimizer"]
]

links = {
    "source": [0, 0, 1, 1, 2, 3, 4, 5, 6],
    "target": [1, 2, 3, 4, 3, 6, 6, 6, 7],
    "value":  [1, 1, 1, 1, 1, 1, 1, 1, 1]
}

fig = go.Figure(data=go.Sankey(
    node=dict(label=nodes, color=node_colors),
    link=links
))
```

---

### 영역 4: Black-Litterman 상세 + 4-slot config

**결정사항** (M4-1 ~ M4-6):
- 표시 형식: (b) 도식 + 수식
- 4-slot config 시각화: (e) 표 + 우리 펀드 강조 결합
- 학술 인용: (b) 표준 4개 (Black-Litterman 1990/1992, He-Litterman 1999, Idzorek 2005)
- 인터랙션: hover + click 인용 (시뮬레이션 보류)
- 추가 시각화: (a) Equilibrium prior 도식 + (b) View 통합 도식
- 우리 펀드 강조: (b) 시각화 highlight + (c) 선택 이유 narrative

**BL 핵심 수식**:
$$E[R] = \left[ (\tau\Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1} \left[ (\tau\Sigma)^{-1}\Pi + P^T \Omega^{-1} Q \right]$$

**4-slot config 표**:

| Slot | Options | Our Fund |
|---|---|---|
| prior | mat / mcap / rp | **mat** ★ |
| p_weight | eq / mcap | **eq** ★ |
| q_mode | eq / mcap / lam / raw | **eq** ★ |
| omega_mode | pap / he / lam / rms | **pap** ★ |

→ 본 펀드 config: **mat_eq_eq_(raw)_pap**

**선택 이유 narrative 박스**:
```
✅ 본 펀드 config: mat_eq_eq_(raw)_pap

별도 기준으로 선정된 Top 1 config 입니다. 4-slot 조합 64개
(3 × 2 × 4 × 4 + 일부 제외) 중 학술/실무 다중 메트릭 검증을
통해 선정되었습니다.

자세한 선정 절차는 Backtesting 페이지의 Regime 분석 + Sensitivity
Test 참조.
```

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] LaTeX 수식 = `st.latex` 또는 `st.markdown`
- [ ] 도식 (Equilibrium / View 통합) = matplotlib / Plotly 정적 이미지 또는 `st.image`
- [ ] 4-slot 표 = pandas DataFrame + Streamlit Styler (★ 마커 highlight)
- [ ] 학술 인용 link = Google Scholar / DOI link
- [ ] 선택 이유 narrative 박스 = `st.success` 또는 custom card

---

### 영역 5: LSTM 변동성 예측 상세 (★ Walk-forward 명시)

**결정사항** (M5-1 ~ M5-6):
- 표시 형식: (b) 도식 + 수식
- LSTM 구조 시각화: (c) Cell + Architecture 둘 다
- Input/Output 명시: (a) 표 형식
- 학습 절차: (c) 학습 곡선 + 4번 재학습 검증 보고서
- 학술 인용: (b) 표준 3개 (Hochreiter 1997, Gers 2000, Kim-Won 2018)
- 인터랙션: 모두 채택

### ★ Walk-forward 구조 (사용자 지적 반영 — 명확 기록)

**모델 구조 정확 파악** (`final/lstm_pipeline.py` V4_BEST_CONFIG):
- **is_len: 1250일** (~5년 일별 데이터로 학습)
- **oos_len: 21일** (~1개월 OOS 예측)
- **step: 21일** (월별 슬라이딩)
- **embargo: 63일** (~3개월 buffer)
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

**TEST / HOLD_OUT 의 정확한 의미** (`final/master_table.py` EVAL_PERIODS):
- **TEST 168m** (2010-01 ~ 2023-12) = walk-forward 결과의 in-sample evaluation (config selection 평가)
- **HOLD_OUT 24m** (2024-01 ~ 2025-12) = walk-forward 결과의 true out-of-sample (untouched)

→ **TEST/HOLD_OUT 은 학습/검증 분리가 아닌, walk-forward 결과의 평가 기간 분리**

**Input/Output 표**:

| Type | Item | Description |
|---|---|---|
| Input | Past 60d returns | 과거 60일 일별 수익률 |
| Input | Sector dummy | 11개 GICS 섹터 one-hot |
| Input | Market state | SPY rolling vol / VIX 등 |
| Output | σ_next | 다음 월 변동성 예측 |
| Downstream | → BL Ω | View confidence 입력 |

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] LSTM Cell 도식 = 표준 LSTM diagram (참조 이미지)
- [ ] Architecture = matplotlib Block diagram 또는 SVG
- [ ] 학습 곡선 = Plotly `go.Scatter` (training/val loss)
- [ ] 4번 재학습 보고서 = 이전 폐기된 노트북의 검증 보고서 참조
- [ ] 학술 인용 (Hochreiter 1997, Gers 2000, Kim-Won 2018)

---

### 영역 6: Factor 분석 (CAPM + FF5) — 조건부

### ★ Factor 분석 영역의 의미와 목적

**핵심 질문**: "펀드의 outperformance 가 진정한 운용 skill 때문인가, 아니면 단순 factor 노출 때문인가?"

**4가지 목적**:
1. **BL+LSTM 의 진정한 가치 검증** — Factor 통제 후 양의 alpha 가 나오는가?
2. **펀드의 factor tilt 명시** — β_MKT < 1 기대 / β_SMB / HML / RMW / CMA 노출 패턴
3. **HO 정당화 narrative 보완** — HO 24m 부진을 factor 노출로 분해
4. **학술 심사위원 어필** — Factor 분석 = 학술 펀드 평가의 필수 절차

### 영역 6 처리 결정 (조건부)

**결정**: (A) 영역 6 전체 유지 (조건부)

**조건**:
1. 일단 생성 — 학술 깊이 우선
2. **결과 확인 후 재평가**:
   - alpha 가 양수 + 통계적 유의 → narrative 강화 + 유지
   - alpha 음수 / 통계적 무의 → 영역 축소 (B) 또는 제거 (C) 검토
3. 재평가 시점: 구현 후 결과 확인

**결정사항** (M6-1 ~ M6-6):
- Factor 모델: (c) CAPM + FF5 둘 다
- 표시 형식: (c) 표 + 막대 둘 다
- Alpha 시각화: (c) Card + Rolling 둘 다
- 통계 보강: (a) R²+p-value + (b) 95% CI
- 비교: (b) 펀드 vs SPY
- 학술 인용: (b) 표준 4개 (Jensen 1968, Fama-French 1993/2015, Carhart 1997)

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] Factor 데이터 = `data/ff5_monthly.csv` (이미 보유)
- [ ] 회귀 = `statsmodels.OLS` 활용
- [ ] 95% CI = `statsmodels` 결과의 `conf_int(alpha=0.05)`
- [ ] Rolling Alpha = `pd.rolling.apply(lambda x: OLS(...).fit().params['Intercept'])`
- [ ] **재평가 트리거**: 구현 후 결과가 부정적일 시 → 단순화 또는 제거 검토

---

### 영역 7: 정규성 검정 (Jarque-Bera) — ★ LSTM 정당화

### ★ 영역 7 의미 재구성 (사용자 지적 반영)

**사용자 핵심 지적**: "우리는 수익률이 아닌 변동성을 예측함을 기억할 것."

→ 영역 7 의 narrative 가 단순 "수익률 정규성 검정 → 위험 메트릭 가정" 이 아닌, **LSTM 변동성 예측의 학술적 정당화** 로 재정의되어야 함.

**핵심 logic chain**:
1. 수익률이 정규분포라면 → 단순 통계 모델 (GARCH 등) 로 변동성 예측 충분
2. 수익률이 fat tail / 비대칭 분포 (정규분포 기각) → 단순 모델 한계 → **비선형 모델 (LSTM) 필요**
3. **Jarque-Bera 가 정규분포 기각 → LSTM 변동성 예측의 학술적 정당화**

**Jarque-Bera 수식**:
$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

- $H_0$: 정규분포 / $H_1$: 정규분포 아님
- p-value < 0.05 → 정규분포 기각

**결정사항** (M7-1 ~ M7-5):
- 표시 형식: (c) Card + 보조 시각화
- 추가 정규성 검정: (a) Jarque-Bera only
- 시각화: (c) Q-Q plot + 히스토그램 둘 다
- 비교: (b) 펀드 vs SPY + 일별/월별 토글
- narrative: (b) 의미 + 결과 + 함의 + **결과가 부정적일 경우에만 한계 추가 (동적)**

### 동적 narrative logic

**Case A: 정규분포 기각 (p < 0.05) — fat tail 확인**
```
✅ Jarque-Bera 검정 결과: 정규분포 가설 기각 (p < 0.05)

수익률이 fat tail / 비대칭 분포임이 통계적으로 입증되었습니다.
이는 LSTM 같은 비선형 모델로 변동성을 예측하는 것이 학술적으로
정당화됨을 의미합니다 (Cont 2001).
```

**Case B: 정규분포 채택 (p ≥ 0.05) — 정규분포에 가까움**
```
⚠ Jarque-Bera 검정 결과: 정규분포 가설 채택 (p ≥ 0.05)

수익률이 정규분포에 가까움이 시사됩니다. 이는 단순 통계 모델
로도 변동성 예측이 충분할 수 있음을 의미하며, LSTM 의 부가가치
재검토가 필요합니다.

→ 영역 8 한계 카드 자동 추가 ("LSTM 가치 미입증")
```

**구현 logic**:
```python
if p_value_jb < 0.05:
    show_narrative("LSTM 학술 정당화 narrative")
    영역_8_한계_LSTM_가치_미입증 = False
else:
    show_narrative("LSTM 가치 재검토 narrative")
    영역_8_한계_LSTM_가치_미입증 = True  # 영역 8 한계 카드 동적 추가
```

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] Jarque-Bera 검정 = `scipy.stats.jarque_bera()`
- [ ] Q-Q plot = `scipy.stats.probplot` + Plotly
- [ ] Histogram + Normal overlay = Plotly
- [ ] 동적 narrative = `if p_value < 0.05: ... else: ...`
- [ ] 영역 8 동적 카드 추가 (`st.session_state.lstm_value_unproven`)

---

### 영역 8: 한계 + 향후 개선 (★ 3개 한계 카드)

### ★ 균형 옵션 (B) Q-A1 적용 — 5개 → 3개 축소

**3가지 한계 차원** (사용자 균형 옵션 B + 모델 구조 정확화 반영, Q-A1):

| 변경 | 이유 |
|---|---|
| ❌ 데이터 한계 (Walk-forward 미적용) | **사실 적용됨 (LSTM walk-forward + BL walk-forward)** |
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

**동적 추가 (영역 7 결과 Case B 시)**:
- "단순 모델 대비 LSTM 우위 미입증" 카드 추가 — "향후 개선 방향" 카드 안의 ablation study 항목으로 부드럽게 통합

**결정사항** (M8-1 ~ M8-5):
- 카드 디자인: (c) 카테고리 색상 코딩
- Expander 내용: (b) 학술 인용 + 텍스트
- 동적 카드 추가: (a) 채택
- 인터랙션: 모두 채택
- 추가 표시: (c) **학술 정직성 선언 박스** (L-3 결정 — Methodology 영역 8 만)

**카테고리 색상** (`lib/colors.py` LIMITATION_COLORS):
- 🟧 HO 부진: Orange `#F59E0B`
- 🟩 개선 방향: Green `#10B981`
- 🟥 실무 제약: Red `#EF4444`

**학술 정직성 선언 텍스트** (확정):
```
✅ 학술 정직성 선언

본 펀드는 학술 정직성을 위해 모든 한계를 명시합니다.
한계 인정은 신뢰성 강화의 토대이며, 향후 개선 방향을
통해 지속적 발전을 추구합니다.
```

**시각화 예시 (3개 카드 + 동적 추가)**:

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
│ [3가지 카드 그리드 — 카테고리 색상 코딩]                  │
│ ┌──────────┬──────────┬──────────┐                      │
│ │ 🟧 HO    │ 🟩 개선  │ 🟥 실무  │                      │
│ │  부진    │  방향    │  제약    │                      │
│ │ (orange) │ (green)  │ (red)    │                      │
│ └──────────┴──────────┴──────────┘                      │
│                                                          │
│ ▼ 🟧 HO 24m 부진 인정 [Expander]                         │
│   • SPY 대비 -12.9%p                                     │
│   • IT under-weight (분산 trade-off)                     │
│   • Sector Watch 영역 8 narrative 일관                   │
│   ⓘ Markowitz (1952) — 평균-분산 이론                   │
│                                                          │
│ ▼ 🟩 향후 개선 방향 [Expander] [Backtesting nav]          │
│   • Multi-factor (Momentum + Value + Quality)            │
│   • Walk-forward validation 추가                         │
│   • 실제 매매 시뮬레이션                                 │
│   • (동적 추가 시) Ablation study (BL only vs BL+LSTM)   │
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
│ │ → 향후 ablation study 권장                         │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] 학술 정직성 선언 박스 = `st.success` (영역 8 상단)
- [ ] 3 카드 = `st.markdown(unsafe_allow_html=True)` + CSS background-color (LIMITATION_COLORS)
- [ ] Expander = `st.expander` (Streamlit 표준)
- [ ] 학술 인용 link = markdown link (Google Scholar / DOI)
- [ ] Backtesting navigation = `st.switch_page()` (개선 방향 카드)
- [ ] 동적 카드: `if st.session_state.lstm_value_unproven: show_dynamic_card()`

---

### 영역 9: Footer — Overview 동일

→ `02_common.md` 의 `render_footer()` 호출

---

## 페이지 데이터 의존성

- monthly_panel.csv (월별 fund/SPY return)
- daily_returns.pkl (영역 7 분포 검정 일별)
- ff5_monthly.csv (영역 6 Factor 회귀)
- 4번 재학습 검증 보고서 (참조 — 이전 폐기 노트북)

---

## 메트릭 (C-2 풀)

- Pool-6 Factor 분석: CAPM Alpha / FF5 Alpha / Factor Exposures (β_MKT, SMB, HML, RMW, CMA)
- 추가 (영역 7): Jarque-Bera test (학술 — Methodology 페이지 추가)
- 추가 (영역 5): LSTM 4-slot config + 4번 재학습 보고서

---

## 인터랙션 / 토글 적용

| 영역 | 사이드바 토글 영향 | Q-Zoom |
|---|---|---|
| 영역 3 (Sankey) | ✗ | 노드 클릭 → 영역 4-7 navigation |
| 영역 4 (BL 상세) | ✗ | 학술 인용 link click |
| 영역 5 (LSTM 상세) | ✗ | 학습 곡선 zoom |
| 영역 6 (Factor) | ✗ (조건부 영역) | ✗ |
| 영역 7 (Jarque-Bera) | Tab 전환 (월별/일별) | ✗ (동적 narrative) |
| 영역 8 (한계) | ✗ | Expander + 학술 인용 link + Backtesting navigation |

---

## 페이지 구현 우선순위

- **Phase 3 (검증, 1-2주)**: Methodology 페이지 (학술 정직성 강화)

---

## 결과 / 함의

- **재구성된 narrative**:
  - 영역 7 (Jarque-Bera) = "왜 LSTM 인가" 학술 정당화
  - 영역 8 = 학술 정직성 선언 + 3가지 한계 + 동적 추가
- **재평가 트리거** (영역 6): Factor 분석 결과 확인 후 축소/삭제 검토
- **인터랙션 일관성**:
  - Sankey 노드 클릭 → 영역 4-7 navigation
  - 영역 8 한계 카드 → Backtesting 페이지 navigation
  - 영역 7 Jarque-Bera 결과 → 영역 8 한계 카드 동적 추가
- 학술 인용 (10개 이상) — 00_README 학술 근거 일람 일괄 갱신

---

[← 06_sector_watch.md](06_sector_watch.md) | [08_backtesting.md →](08_backtesting.md)
