# Overview 페이지 — 와이어프레임

> **관련 decisionlog**: `02_overview.md`
> **상태**: 확정 (영역 1~6 모두)
> **결정 수**: 6 영역

---

## 페이지 역할 정의

랜딩 페이지 = URL 진입 직후 노출. **5초 안에 펀드 정체성 + 핵심 성과 전달, 다음 페이지로 navigation 유도**.

---

## 페이지 영역 구조 (시선 흐름)

```
1. Header           → 펀드명 + 슬로건 + 메타
2. Hero KPI         → 5개 핵심 메트릭 카드 (반응형 + sparkline)
3. 누적수익 곡선     → vs SPY 메인 차트 (이중 차트: 누적수익 + Drawdown)
4. 핵심 강점 카드    → 3개 차별화 포인트 (A + C + E)
5. Navigation cards → 다음 7개 페이지 연결
6. Footer           → Disclosure 짧은 버전 (3 column)
```

---

## 영역별 와이어프레임

### 영역 1: Header

**결정사항** (영역 1 결정 1-1 ~ 1-4):
- (a) 영문 강조 + 한글 부제 + 좌측 정렬
- 메타: 펀드 상태 ✓ / 벤치마크 ✓ / 데이터 기준일 ✓ / AUM ✗

**시각화 예시**:

```
┌─────────────────────────────────────────────────────────────┐
│  Adaptive VolControl Fund                  ● Active         │
│  어댑티브 볼컨트롤 펀드                       Benchmark: SPY  │
│  변동성 예측 기반 적응형 자산배분 —             Data: 2025-12 │
│  Volatility-Aware Adaptive Allocation                       │
└─────────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] `render_page_header()` 호출 (`02_common.md` 참조)
- [ ] 좌측 정렬 (col1=3, col2=1 비율)
- [ ] 펀드 상태 = "● Active (Simulated)" (학술 정직성)
- [ ] 한/영 병기 (A-3 결정)

---

### 영역 2: Hero KPI

**결정사항** (영역 2 결정 2-1 ~ 2-5 + 통합 결정 ①②③):
- 5개 KPI: **Cumulative Return / Net CAGR / Sortino / Volatility / Max Drawdown**
- 표시 기간: (d) **TEST + HO 별도** (학술 정직성 + HO 부진 회피 의심 제거)
- 카드 디자인: (c) 큰 숫자 + sparkline (FULL 192m, HO 다른 색)
- HO 부진 처리: (c) 토글로 전환 (단, **Hero 자체는 고정** — 통합 결정 ①)
- 레이아웃: (c) 반응형

**시각화 예시**:

```
┌──────────────────┬──────────────────┬──────────────────┐
│ Cumulative Return│ Net CAGR (TC 20bp)│ Sortino Ratio    │
│ TEST: +XXX.X%    │ TEST: +X.XX%     │ TEST: X.XX       │
│ HO:   +X.X%      │ HO:   +X.XX%     │ HO:   0.685      │
│ ▁▂▄▆█ sparkline  │ ▁▃▆█ sparkline   │ ▁▂▆█ sparkline   │
│ (FULL, HO red)   │ (FULL, HO red)   │ (FULL, HO red)   │
├──────────────────┼──────────────────┴──────────────────┤
│ Volatility       │ Max Drawdown     │                  │
│ TEST: XX.X%      │ TEST: -XX.X%     │                  │
│ HO:   10.3%      │ HO:   -8.3%      │                  │
│ ▆▂▃▄▆ sparkline  │ ▂▁▆█ sparkline   │                  │
└──────────────────┴──────────────────┘

레이아웃: 반응형 (큰 화면 5개 한 줄 / 작은 화면 3+2 분할)
Sparkline: FULL 192m, HO 24m 다른 색 (red dashed)
```

**구현 체크리스트**:
- [ ] `st.columns(5)` 반응형 (작은 화면 자동 3+2 분할)
- [ ] 각 카드 = TEST + HO 두 줄 (HO 빨간색 강조)
- [ ] Sparkline = FULL 192m (HO 영역 red dashed)
- [ ] Hero 카드는 사이드바 토글에 영향 받지 않음 (고정)
- [ ] 카드 5개 메트릭:
  - Cumulative Return
  - Net CAGR (TC 20bp 차감 후)
  - Sortino Ratio
  - Volatility
  - Max Drawdown

**코드 snippet (예시)**:
```python
import plotly.graph_objects as go

def hero_kpi_card(label, test_value, ho_value, sparkline_data):
    """Hero KPI 카드 (sparkline 포함)."""
    st.metric(label=label, value=f"TEST: {test_value}")
    st.caption(f"HO: {ho_value}")
    # Sparkline 추가
    fig = go.Figure(go.Scatter(y=sparkline_data, mode="lines", line=dict(width=1)))
    fig.update_layout(height=50, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
```

---

### 영역 3: 누적수익 곡선 (Cumulative Return Curve)

**결정사항** (영역 3 결정 3-1 ~ 3-7 + 추가 Q-A, Q-B):
- 표시 기간: (c) 항상 FULL + 구간 강조 (Regime 배경색)
- 비교 대상: (e) 토글 가능 (기본 SPY only)
  - **EW 구현 (Q-A)**: (A1) 우리 펀드 동일 universe EW (1/N)
  - **IVW 구현 (Q-B)**: (B1) Inverse Volatility Weighting (120일 변동성 역수 가중)
- Y축 스케일: (c) 토글 (기본 선형 / Log 전환 가능)
- Regime 시각화: (a) 배경색 + (c) annotation 라벨
- Drawdown 추가: (b) 이중 차트 (위: 누적수익 / 아래: drawdown 영역)
- 인터랙션: 모두 적용 (Hover / Zoom / Slider / Annotation)
- 차트 크기: (a) 전체 너비

**시각화 예시**:

```
┌──────────────────────────────────────────────────────────┐
│ Cumulative Return — Adaptive VolControl Fund vs SPY      │
│ Y축: [Linear ▼] (토글로 Log 전환)                        │
│ ┌─ R1 회복기 ─┬── R2 확장기 ──┬── R3 변동기 ──┬─ HO ─┐  │
│ │ (배경색 1)  │  (배경색 2)   │  (배경색 3)   │(4)   │   │
│ │             │               │               │      │   │
│ │     ╱─────╱─╲────╱──╲╱─╲──── Adaptive       │      │   │
│ │   ╱╱   ╱─╱╱        ╲╱   ╲   VolControl     │      │   │
│ │ ╱╱ ╱─╱╱             ╲    ╲╱╲                │      │   │
│ │╱─╱─╱                 ╲   ╱──── SPY          │      │   │
│ └─────────────────────────────────────────────┴──────┘   │
│ 2010    2012    2014    2016    2018    2020   2024     │
│ Annotation: ▼COVID-19 (2020-03)  ▼2022 Bear  ▼2024-12  │
│                                                          │
│ ┌─ Drawdown ──────────────────────────────────────────┐  │
│ │  ▔▔▔▔▔▔▔▔▔▔▔▁▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▁▔▔▔▔▔▔▔▔▔▔▔▔▔        │  │
│ │       ╲     ╱  ╲       ╲    ╲   ╲     ╲           │  │
│ │        ╲___╱    ╲___╱   ╲___╱    ╲___╱            │  │
│ └──────────────────────────────────────────────────────┘  │
│                                                          │
│ 비교 라인: [☑ SPY]  [☐ EW (펀드 universe 1/N)]           │
│           [☐ Naive Low-vol (IVW 120d)]                   │
│ 기간 슬라이더: 2010 ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2025 │
└──────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Plotly `make_subplots(rows=2, shared_xaxes=True)` (이중 차트)
- [ ] Regime 배경색 4개 + annotation 라벨 (`add_regime_backgrounds()` 헬퍼)
- [ ] 위기 annotation: COVID-19 / 2022 Bear / 2024 Sector Rotation (`add_event_annotations()`)
- [ ] 비교 라인 토글 (사이드바 SPY/EW/IVW 활성에 따라 동적 추가)
- [ ] EW baseline: `equal_weight_returns(universe, returns_df)` (`lib/data_loader.py`)
- [ ] IVW baseline: `ivw_returns(returns_df, window=120)` (Frazzini-Pedersen 2014)
- [ ] Y축 Log 토글
- [ ] Range slider (Plotly 기본)
- [ ] Q-Zoom: 시기 클릭 → 같은 페이지 expand

**코드 snippet (EW + IVW 산출)**:
```python
# lib/data_loader.py
@st.cache_data
def equal_weight_returns(universe_dict, returns_df):
    """EW baseline: 매월 우리 펀드 universe 의 1/N."""
    monthly_ew = []
    for date, tickers in universe_dict.items():
        N = len(tickers)
        weights = {t: 1/N for t in tickers}
        # 월별 EW 수익률 계산
        ...
    return pd.Series(monthly_ew)

@st.cache_data
def ivw_returns(returns_df, window=120):
    """IVW baseline: weight = (1/σ_i) / Σ(1/σ_j)."""
    # Frazzini & Pedersen (2014) "Betting Against Beta"
    sigma = returns_df.rolling(window).std()
    inv_sigma = 1 / sigma
    weights = inv_sigma.div(inv_sigma.sum(axis=1), axis=0)
    monthly_returns = (weights.shift(1) * returns_df).sum(axis=1).resample("M").apply(lambda x: (1+x).prod() - 1)
    return monthly_returns
```

---

### 영역 4: 핵심 강점 카드 (Differentiator Cards)

**결정사항** (영역 4 결정 4-1 ~ 4-5):
- 카드 개수: (a) 3개
- 카드 콘텐츠: **A + C + E** (LSTM + 3-Regime + Net Costs)
- 카드 디자인: (d) 아이콘 + 헤드라인 + 본문 + 숫자
- 추가 정보: (c) Hover + 클릭 (둘 다)
  - 카드 1 → Methodology 페이지
  - 카드 2 → Backtesting 페이지
  - 카드 3 → Performance 페이지
- 구현: 옵션 3 (`streamlit-card` 라이브러리)
- 레이아웃: (c) 반응형

**3 카드 구성**:

| 카드 | 헤드라인 | 본문 | 숫자 | 학술 근거 | navigation |
|---|---|---|---|---|---|
| **카드 1: A** | Volatility-Aware Allocation | LSTM 으로 미래 변동성 예측 → 적응형 비중 결정 | Volatility (HO 10.3%) | LSTM (Hochreiter 1997) | Methodology |
| **카드 2: C** | Validated Across Market Regimes | 회복/확장/변동기 + 24m OOS 검증 | Sortino IR | Regime stability | Backtesting |
| **카드 3: E** | Net of Conservative Costs | 20bp 거래비용 차감 후에도 양호한 위험조정 수익 | Net Sortino / Net CAGR | AQR Frazzini et al. (2018) | Performance |

**시각화 예시**:

```
┌────────────────────┬────────────────────┬────────────────────┐
│  [📊]              │  [✓]               │  [💎]              │
│                    │                    │                    │
│  Volatility-Aware  │  Validated Across  │  Net of            │
│  Allocation        │  Market Regimes    │  Conservative Costs│
│                    │                    │                    │
│  LSTM 으로 미래    │  회복/확장/변동기  │  20bp 거래비용     │
│  변동성 예측 →     │  + 24m OOS 검증   │  차감 후 양호      │
│  적응형 비중       │                    │                    │
│                    │                    │                    │
│  Vol: 10.3%        │  Sortino IR: X.XX  │  Net Sortino: X.XX │
│  (vs SPY 14.2%)    │                    │                    │
│                    │                    │                    │
│  [Methodology→]    │  [Backtesting→]    │  [Performance→]    │
└────────────────────┴────────────────────┴────────────────────┘
        ↑ Hover                ↑ Hover                ↑ Hover
   (학술 근거 tooltip)
```

**구현 체크리스트**:
- [ ] `streamlit-card` 라이브러리 활용 (J-1 결정)
- [ ] 3 카드 가로 정렬 (반응형 — 작은 화면 세로 분할)
- [ ] Hover tooltip = 학술 인용 / 추가 메트릭
- [ ] 카드 클릭 → `st.session_state.page = ...` + `st.switch_page()`
- [ ] 각 카드 = 아이콘 + 헤드라인 + 본문 + 숫자 + navigation 버튼

**코드 snippet (예시)**:
```python
from streamlit_card import card

def render_differentiator_cards():
    cards_data = [
        {
            "icon": "📊",
            "title_en": "Volatility-Aware Allocation",
            "title_ko": "변동성 인지 자산배분",
            "body": "LSTM 으로 미래 변동성 예측 → 적응형 비중 결정",
            "value": f"Vol: 10.3% (vs SPY 14.2%)",
            "citation": "Hochreiter & Schmidhuber (1997)",
            "navigate_to": "07_Methodology"
        },
        # ... 카드 2, 3
    ]

    cols = st.columns(3)
    for col, card_data in zip(cols, cards_data):
        with col:
            clicked = card(
                title=card_data["title_en"],
                text=[card_data["body"], card_data["value"]],
                # ...
            )
            if clicked:
                st.switch_page(f"pages/{card_data['navigate_to']}.py")
```

---

### 영역 5: Navigation Cards

**결정사항** (영역 5 결정 5-1 ~ 5-4):
- 표시 방식: (a) 카드 그리드 (7개 페이지 카드)
- 카드 구성: (a) 모두 동등 (영역 4 에서 이미 3개 강조)
- 카드 디자인: (b) 단순 디자인 (아이콘 + 페이지명 + 1줄 설명)
- 위치: (a) 포함 (영역 4 → 영역 5 → 영역 6)

**시각화 예시**:

```
[영역 5: Navigation Cards — 7개 단순 카드 그리드]
┌─────────┬─────────┬─────────┬─────────┐
│ [📈]    │ [⚠️]    │ [🏢]    │ [🌐]    │
│Performance│Risk     │Holdings │Sector   │
│성과 분석 │위험 지표│보유 종목│Watch    │
│   →     │   →     │   →     │   →     │
└─────────┴─────────┴─────────┴─────────┘
┌─────────┬─────────┬─────────┐
│ [🧪]    │ [✓]     │ [ℹ️]    │
│Method.  │Backtest │About    │
│방법론   │검증     │FAQ      │
│   →     │   →     │   →     │
└─────────┴─────────┴─────────┘
```

**구현 체크리스트**:
- [ ] 7 페이지 카드 (Investment Simulator 제외 — Sim 은 영역 4 강점 카드 흐름과 다름)
- [ ] 4 + 3 그리드 (작은 화면 자동 분할)
- [ ] 각 카드 = 아이콘 + 페이지명 (한/영) + 1줄 설명 + 화살표
- [ ] 카드 클릭 → `st.switch_page()` 페이지 이동

---

### 영역 6: Footer

**결정사항** (영역 6 결정 6-1 ~ 6-4):
- 콘텐츠: (c) Disclosure + 데이터 출처 + 메서드 요약
- 표시 형식: (b) 다단 (3 column)
- Disclosure 정직성: 권장대로 + **HO 24m 부진 언급** (E-3)
- Copyright: 모두 포함 (Last updated, Built with, GitHub link)

**시각화 예시**:

```
═══════════════════════════════════════════════════════════════════
┌─ Disclosure ───────┬─ Data Sources ─────┬─ Meta ─────────────────┐
│ ※ 본 결과는 백테스트│ Yahoo Finance      │ Last updated:          │
│   시뮬레이션이며   │ (Adj Close 기준)   │ 2026-05-10             │
│   실제 운용 성과를 │                    │                        │
│   보장하지 않습니다│ Fama-French Library│ Built with:            │
│                    │ (FF5 factors)      │ Streamlit + Plotly     │
│ 데이터 기간:       │                    │                        │
│ 2010-01 ~ 2025-12  │ FRED               │ © 2026 [팀명]          │
│ (TEST 평가 168m +  │ (Risk-free rate)   │                        │
│  HOLD_OUT 24m)     │                    │ [GitHub →]             │
│                    │ Methodology:       │                        │
│ ※ HOLD_OUT 24m 부진│ Black-Litterman +  │                        │
│   — Backtesting 참조│ LSTM 4-slot        │                        │
└────────────────────┴────────────────────┴────────────────────────┘
═══════════════════════════════════════════════════════════════════
```

**구현 체크리스트**:
- [ ] `render_footer()` 호출 (`02_common.md` 참조)
- [ ] 3 column 레이아웃
- [ ] HO 부진 언급 = E-3 결정 일관 적용
- [ ] GitHub link 포함

---

## 페이지 데이터 의존성

- **monthly_panel.csv** (rf, spy_ret, sector, log_mcap)
- **daily_returns.pkl** (sparkline 산출 + IVW baseline)
- **results/mat_eq_eq_raw_pap.pkl** (펀드 weights + returns)
- **universe.csv** (EW baseline universe)

→ 모두 `lib/data_loader.py` 캐시 활용

---

## 메트릭 (C-2 풀에서 picking)

- Hero KPI 5개 (Pool-1 + Pool-2 + Pool-3 + Pool-8):
  - Cumulative Return / Net CAGR / Sortino / Volatility / MDD
- 강점 카드 (영역 4) 메트릭:
  - 카드 1: Volatility (Pool-3)
  - 카드 2: Sortino IR (Pool-7)
  - 카드 3: Net Sortino / Net CAGR (Pool-8)

---

## 인터랙션 / 토글 적용

| 영역 | 사이드바 토글 영향 | Q-Zoom |
|---|---|---|
| 영역 1 (Header) | ✗ | ✗ |
| 영역 2 (Hero KPI) | ✗ **고정** (TEST + HO 별도) | ✗ |
| 영역 3 (누적수익 곡선) | ✓ 비교 토글 (SPY/EW/IVW) + 기간 (zoom) | ✓ 시기 클릭 → expand |
| 영역 4 (강점 카드) | ✗ | 카드 클릭 → 페이지 navigation |
| 영역 5 (Navigation) | ✗ | 카드 클릭 → 페이지 navigation |
| 영역 6 (Footer) | ✗ | ✗ |

---

## 페이지 구현 우선순위

- **Phase 1 (MVP)**: Overview 페이지 (모든 영역) — `04_implementation_steps.md` 참조
- 영역 4 (`streamlit-card`) 는 라이브러리 없을 시 옵션 2 (HTML + 별도 버튼) 으로 fallback

---

## 결과 / 함의

- 모든 페이지 상단에 동일 Header 패턴 적용 (전체 페이지 일관성)
- 사이드바 상단에도 동일 펀드명 텍스트 마크 표시 (B-5)
- Hero (5개 KPI) → 누적수익 (대형) 시선 흐름 자연스러움
- EW + IVW baseline 산출 → `lib/data_loader.py` 별도 함수 구현
- 학술 근거: Frazzini & Pedersen (2014) — IVW (00_README 학술 인용 일람)

---

[← 02_common.md](../02_common.md) | [02_simulator.md →](02_simulator.md)
