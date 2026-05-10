# 02. Common — lib/* + 사이드바 + 디자인 + Disclosure

> **관련 decisionlog**: `10_sidebar.md` (C-4) + `11_dl_sections.md` (G, H, I + F-6 Insight)
> **상태**: 확정
> **목적**: 모든 페이지가 공유하는 공통 컴포넌트 정의

---

## 1. 사이드바 (C-4 결정)

### 1.1 사이드바 구조 (6 그룹 + 2 토글)

C-4 결정 (5 그룹 → 6 그룹 갱신, F-6 Investment Simulator 추가):

```
┌──────────────────────────┐
│ Adaptive VolControl Fund │   ← 텍스트 마크 (B-5)
│ 어댑티브 볼컨트롤 펀드   │
│                          │
│ Benchmark: SPY           │   ← 메타 (C4-3)
│ Data: 2025-12            │
├──────────────────────────┤
│ ── 첫인상 ──             │
│ ◉ Overview               │
│                          │
│ ── 체험 ── ★ 신규         │   ← F-6 Investment Simulator
│ ◯ Investment Simulator   │
│                          │
│ ── 성과 ──               │
│ ◯ Performance            │
│ ◯ Risk Metrics           │
│                          │
│ ── 보유 ──               │
│ ◯ Holdings               │
│ ◯ Sector Watch           │
│                          │
│ ── 검증 ──               │
│ ◯ Methodology            │
│ ◯ Backtesting            │
│                          │
│ ── 메타 ──               │
│ ◯ About / FAQ            │
├──────────────────────────┤
│ 📅 기간 (Period):        │   ← C4-4 토글 1
│  ● FULL  ○ TEST  ○ HO    │
│                          │
│ 📊 비교 (Benchmark):     │   ← C4-4 토글 2
│  ☑ SPY                   │
│  ☐ EW (펀드 universe)    │
│  ☐ IVW (Naive Low-vol)   │
└──────────────────────────┘
```

### 1.2 사이드바 구현 (`app.py`)

```python
import streamlit as st
from lib.validators import startup_data_check
from lib.disclosure import init_session_state

# 페이지 설정 (Wide layout 권장)
st.set_page_config(
    page_title="Adaptive VolControl Fund",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Startup check (D-5)
startup_data_check()

# Session state 초기화 (토글 값 저장)
init_session_state()

# 사이드바 — 펀드명 + 메타 + 6 그룹 카테고리 헤더 + 페이지 navigation (C4-1 (c) + C4-2 (a))
# 결정 근거: decisionlog C4-1 = "(c) 카테고리 헤더 + 페이지 (Streamlit pagelink)"
#         → st.page_link 명시적 사용 + 그룹별 헤더 표시 (자동 multi-page 만으로는 불가)

with st.sidebar:
    # === 펀드명 + 메타 (C4-3) ===
    st.markdown("# Adaptive VolControl Fund")
    st.markdown("어댑티브 볼컨트롤 펀드")
    st.caption("Benchmark: SPY  |  Data: 2025-12")
    st.divider()

    # === 페이지 navigation — 6 그룹 + st.page_link (C4-1 c, C4-2 a) ===
    # 그룹 1: 첫인상
    st.markdown("##### ── 첫인상 ──")
    st.page_link("app.py", label="Overview", icon="📊")

    # 그룹 2: 체험 (★ Investment Simulator F-6)
    st.markdown("##### ── 체험 ──")
    st.page_link("pages/02_💵_Investment_Simulator.py",
                 label="Investment Simulator", icon="💵")

    # 그룹 3: 성과
    st.markdown("##### ── 성과 ──")
    st.page_link("pages/03_📈_Performance.py", label="Performance", icon="📈")
    st.page_link("pages/04_⚠️_Risk_Metrics.py", label="Risk Metrics", icon="⚠️")

    # 그룹 4: 보유
    st.markdown("##### ── 보유 ──")
    st.page_link("pages/05_🏢_Holdings.py", label="Holdings", icon="🏢")
    st.page_link("pages/06_🌐_Sector_Watch.py", label="Sector Watch", icon="🌐")

    # 그룹 5: 검증
    st.markdown("##### ── 검증 ──")
    st.page_link("pages/07_🧪_Methodology.py", label="Methodology", icon="🧪")
    st.page_link("pages/08_✓_Backtesting.py", label="Backtesting", icon="✓")

    # 그룹 6: 메타
    st.markdown("##### ── 메타 ──")
    st.page_link("pages/09_ℹ️_About.py", label="About / FAQ", icon="ℹ️")

    st.divider()

    # === 토글 1: 기간 (C4-4) ===
    st.subheader("📅 기간 (Period)")
    period = st.radio(
        "기간 선택",
        options=["FULL", "TEST", "HO"],
        index=0,  # 기본 FULL
        key="period",
        label_visibility="collapsed"
    )

    # === 토글 2: 비교 벤치마크 (C4-4) ===
    st.subheader("📊 비교 (Benchmark)")
    show_spy = st.checkbox("SPY", value=True, key="show_spy")
    show_ew = st.checkbox("EW (펀드 universe)", value=False, key="show_ew")
    show_ivw = st.checkbox("IVW (Naive Low-vol)", value=False, key="show_ivw")
```

**구현 주의사항** (안정성 — 사용자 요구사항 #2):
1. `st.page_link` 사용을 위해 Streamlit 1.30+ 필수 (`requirements.txt` 명시)
2. `pages/` 폴더 자동 multi-page 와 `st.page_link` 동시 활용 (Streamlit 자동 sidebar 비활성화 → `.streamlit/config.toml` 의 `client.showSidebarNavigation = false` 설정 필요)
3. icon 은 emoji string (별도 라이브러리 X)
4. 그룹 헤더 (`##### ── 첫인상 ──`) = `st.markdown` 사용 (커스텀 CSS 최소화)

### 1.3 토글 영향 범위

**기간 토글 (Period: FULL / TEST / HO)**:
- Overview Hero KPI (영역 2): ✗ **고정** (TEST + HO 별도 표시, 통합 결정 ①)
- Overview 누적수익 곡선 (영역 3): ✓ 토글 영향 (zoom)
- Performance / Risk / Holdings / Sector Watch / Methodology / Backtesting / Sim: ✓ 모두 영향

**비교 토글 (Benchmark: SPY / EW / IVW)**:
- 모든 페이지의 비교 차트 + KPI delta 동시 영향
- 활성 벤치마크에 따라 차트 라인 / 카드 컬럼 / 탭 동적 추가

### 1.4 Session state 표준

```python
# lib/disclosure.py 또는 lib/session.py
def init_session_state():
    """모든 페이지에서 사용하는 session state 초기화."""
    if "period" not in st.session_state:
        st.session_state.period = "FULL"
    if "show_spy" not in st.session_state:
        st.session_state.show_spy = True
    if "show_ew" not in st.session_state:
        st.session_state.show_ew = False
    if "show_ivw" not in st.session_state:
        st.session_state.show_ivw = False
```

---

## 2. lib/data_loader.py 명세 (D-3 캐싱)

### 2.1 핵심 함수

```python
"""
lib/data_loader.py
모든 데이터 로딩 함수 + Streamlit 캐싱 표준
"""
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = Path("streamlit_dashboard/data")

# === 핵심 데이터 ===

@st.cache_data
def load_monthly_panel() -> pd.DataFrame:
    """월별 패널 데이터 (rf, spy_ret, sector, log_mcap)."""
    return pd.read_csv(DATA_DIR / "monthly_panel.csv", parse_dates=["date"])

@st.cache_data
def load_daily_returns() -> pd.DataFrame:
    """일별 수익률 (822 ticker × 6099 영업일)."""
    with open(DATA_DIR / "daily_returns.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_ff5_monthly() -> pd.DataFrame:
    """Fama-French 5-factor (Methodology 영역 6)."""
    return pd.read_csv(DATA_DIR / "ff5_monthly.csv", parse_dates=["date"])

@st.cache_data
def load_universe() -> pd.DataFrame:
    """833 ticker + gics_sector."""
    return pd.read_csv(DATA_DIR / "universe.csv")

@st.cache_data
def load_ticker_company_map() -> pd.DataFrame:
    """yfinance 회사명 매핑 (D-2)."""
    return pd.read_csv(DATA_DIR / "ticker_company_map.csv")

# === 펀드 결과 ===

@st.cache_data
def load_fund_results(config_name: str = "mat_eq_eq_raw_pap") -> dict:
    """펀드 백테스트 결과 (weights, returns, dates)."""
    with open(DATA_DIR / "results" / f"{config_name}.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_other_config_results(config_name: str) -> dict:
    """다른 155 config (Backtesting 영역 6 Sensitivity Test 시)."""
    with open(f"final/data/results/{config_name}.pkl", "rb") as f:
        return pickle.load(f)

# === Baseline 산출 (Overview 영역 3) ===

@st.cache_data
def compute_equal_weight_returns(universe_dict: dict, daily_returns: pd.DataFrame) -> pd.Series:
    """EW baseline: weight = 1/N (펀드 universe 일치)."""
    # universe_dict[date] = list of tickers
    # 매월 EW 계산
    ...

@st.cache_data
def compute_ivw_returns(daily_returns: pd.DataFrame, window: int = 120) -> pd.Series:
    """IVW baseline: weight = (1/σ_i) / Σ(1/σ_j), 120일 변동성."""
    # σ_i = 120일 일별 수익률 표준편차
    # Frazzini-Pedersen (2014) "Betting Against Beta"
    ...
```

---

## 3. lib/colors.py 명세 (H-4 GICS 11개 섹터 색상)

### 3.1 Cobalt Blue 팔레트 (B-4)

```python
"""
lib/colors.py
색상 팔레트 dictionary (B-4 + H-4 결정)
"""

# === Primary 팔레트 (B-4) ===
COLORS = {
    "primary": "#3B82F6",       # Cobalt Blue
    "accent_green": "#10B981",  # Positive return
    "accent_red": "#EF4444",    # Drawdown / negative
    "background": "#0E1117",    # Streamlit 다크 기본
    "secondary_bg": "#1F2937",  # 카드 배경
    "text": "#FAFAFA",          # 본문 텍스트
    "text_muted": "#9CA3AF",    # 보조 텍스트
}

# === 벤치마크 라인 색상 (H-4) ===
BENCHMARK_COLORS = {
    "Fund": "#3B82F6",   # Cobalt Blue
    "SPY": "#6B7280",    # Gray
    "EW": "#10B981",     # Green
    "IVW": "#8B5CF6",    # Purple
}

# === Regime 배경색 (Overview 영역 3 일관) ===
REGIME_COLORS = {
    "R1": "#1F2937",  # Dark Gray (회복기)
    "R2": "#0E1117",  # Background (확장기, 기본)
    "R3": "#1F2937",  # Dark Gray (변동기)
    "HO": "#374151",  # Slightly Lighter Gray (홀드아웃)
}

# === GICS 11개 섹터 색상 (H-4) ===
SECTOR_COLORS = {
    "Information Technology": "#3B82F6",      # Blue
    "Health Care": "#10B981",                 # Green
    "Financials": "#F59E0B",                  # Amber
    "Consumer Discretionary": "#EC4899",      # Pink
    "Consumer Staples": "#8B5CF6",            # Purple
    "Industrials": "#06B6D4",                 # Cyan
    "Energy": "#EF4444",                      # Red
    "Materials": "#84CC16",                   # Lime
    "Communication Services": "#F97316",      # Orange
    "Real Estate": "#A855F7",                 # Violet
    "Utilities": "#14B8A6",                   # Teal
}

# === Methodology 영역 8 한계 카드 색상 ===
LIMITATION_COLORS = {
    "data": "#3B82F6",      # Blue
    "model": "#8B5CF6",     # Purple
    "ho_decline": "#F59E0B", # Orange
    "future_work": "#10B981", # Green
    "practical": "#EF4444",  # Red
}

# === Methodology Sankey 4그룹 색상 (M3-2) ===
SANKEY_GROUP_COLORS = {
    "data": "#3B82F6",      # Cobalt Blue
    "bl": "#10B981",        # Green
    "lstm": "#8B5CF6",      # Purple
    "optimizer": "#F59E0B", # Orange
}
```

---

## 4. lib/disclosure.py 명세 (E + I)

### 4.1 표준 Disclosure 텍스트

```python
"""
lib/disclosure.py
모든 페이지의 Disclosure 통일 텍스트 + Footer 렌더링
E-3 + I-2 + I-5 결정 통합
"""
import streamlit as st

# === Footer 통일 텍스트 (E-3 + I-2) ===
FOOTER_DISCLOSURE = """
※ 본 결과는 백테스트 시뮬레이션이며 실제 운용 성과를 보장하지 않습니다.
   데이터 기간: 2010-01 ~ 2025-12 (TEST 평가 168m + HOLD_OUT 24m)
※ HOLD_OUT 24m (2024-2025) 구간에서 SPY 대비 부진 (Net CAGR +X.XX%
   vs SPY +21.2%) — 자세한 분석은 Backtesting 페이지 참조
   자세한 Disclosure / Risk factors: About 페이지 참조
"""

# === Investment Simulator 상단 disclaimer (I-5) ===
SIMULATOR_DISCLAIMER = """
⚠️ 본 시뮬레이션은 가상의 백테스트 결과이며, 실제 투자권유 또는
   투자 자문을 목적으로 하지 않습니다. 과거의 성과는 미래의 수익을
   보장하지 않습니다.
"""

# === HO 표현 통일 (E-1) ===
HO_LABEL = "HOLD_OUT 24m (2024-2025)"

# === Footer 렌더링 함수 ===
def render_footer():
    """모든 페이지 하단에 통일 Footer 표시 (Overview 영역 6 / 다른 페이지 동일)."""
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("**Disclosure**")
        st.caption(FOOTER_DISCLOSURE)

    with col2:
        st.caption("**Data Sources**")
        st.caption("""
        - Yahoo Finance (Adj Close 기준)
        - Fama-French Library (FF5 factors)
        - FRED (Risk-free rate)
        - Methodology: Black-Litterman + LSTM 4-slot
        """)

    with col3:
        st.caption("**Meta**")
        st.caption("""
        - Last updated: 2026-05-10
        - Built with: Streamlit + Plotly
        - © 2026 [팀명]
        - [GitHub →]
        """)

# === Sim 페이지 disclaimer 박스 ===
def render_simulator_disclaimer():
    """Investment Simulator 영역 2 하단 + 영역 3 상단 사이 표시."""
    st.warning(SIMULATOR_DISCLAIMER)
```

---

## 5. lib/insight_generator.py 명세 (Sim 페이지 영역 6)

### 5.1 Insight 박스 구현 패턴 (정적 템플릿 + 카드 그리드)

```python
"""
lib/insight_generator.py
Investment Simulator 영역 6 Insight 박스 생성
F-6 Sim6-1 + Sim6-2 결정 (정적 템플릿 + 카드 그리드)
"""
import streamlit as st

def generate_insight_cards(sim_result: dict, benchmarks: dict, scenario: str):
    """
    Sim 결과 + 활성 벤치마크 + 시나리오 (Lump-sum / DCA / Goal) 에 따라
    조건부 카드 4-8개 생성.

    Args:
        sim_result: {"final_value", "total_profit", "cagr", "mdd", "total_invested", ...}
        benchmarks: {"SPY": pd.Series, "EW": ..., "IVW": ...} (활성 벤치마크만)
        scenario: "lump_sum" / "dca" / "goal"
    """
    cards = []

    # 1. 누적 수익 (항상)
    cards.append({
        "icon": "💰",
        "title": "누적 수익 — Total Profit",
        "value": f"${sim_result['total_invested']:,.0f} → ${sim_result['final_value']:,.0f}",
        "delta": f"+${sim_result['total_profit']:,.0f} (+{sim_result['total_profit']/sim_result['total_invested']*100:.1f}%)",
        "color": "green" if sim_result['total_profit'] > 0 else "red"
    })

    # 2. 연환산 CAGR (항상)
    cards.append({
        "icon": "📈",
        "title": "연환산 수익률 — CAGR",
        "value": f"+{sim_result['cagr']*100:.2f}% per year",
        "color": "green" if sim_result['cagr'] > 0 else "red"
    })

    # 3-5. vs Benchmark (활성 벤치마크만)
    for name in ["SPY", "EW", "IVW"]:
        if name in benchmarks:
            delta = sim_result['cagr'] - benchmarks[name]['cagr']
            cards.append({
                "icon": "📊",
                "title": f"vs {name}",
                "value": f"{name} 대비 {'+'if delta > 0 else ''}{delta*100:.2f}%",
                "color": "green" if delta > 0 else "red"
            })

    # 6. 최대 손실 / 회복 (항상)
    cards.append({
        "icon": "⚠️",
        "title": "최대 손실 / 회복 — Max Drawdown",
        "value": f"{sim_result['mdd']*100:.1f}%",
        "subtitle": f"COVID-19 회복: {sim_result.get('recovery_months', 'N/A')}개월",
        "color": "orange"
    })

    # 7. DCA 효과 (DCA Tab 활성 시만)
    if scenario == "dca":
        cards.append({
            "icon": "🔄",
            "title": "DCA 효과 — Dollar Cost Averaging",
            "value": f"매월 ${sim_result.get('dca_monthly', 0):,.0f} 분산 투자",
            "subtitle": f"일시 투자 대비 +${sim_result.get('dca_advantage', 0):,.0f}",
            "color": "blue"
        })

    # 8. Goal 달성 분석 (Goal Tab 활성 시만)
    if scenario == "goal":
        cards.append({
            "icon": "🎯",
            "title": "Goal 달성 분석 — Goal-based",
            "value": f"${sim_result.get('goal_amount', 0):,.0f} 목표",
            "subtitle": f"달성 시점: {sim_result.get('goal_achievement_date', '미달성')}",
            "color": "purple"
        })

    return cards


def render_insight_grid(cards: list):
    """카드 그리드 렌더링 (반응형 3-column)."""
    cols_per_row = 3
    for i in range(0, len(cards), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, card in enumerate(cards[i:i+cols_per_row]):
            with cols[j]:
                _render_card(card)


def _render_card(card: dict):
    """단일 카드 렌더링."""
    color_map = {
        "green": "#10B981",
        "red": "#EF4444",
        "blue": "#3B82F6",
        "orange": "#F59E0B",
        "purple": "#8B5CF6"
    }
    border_color = color_map.get(card.get("color", "blue"), "#3B82F6")

    st.markdown(f"""
    <div style="
        border: 2px solid {border_color};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
        background-color: #1F2937;
    ">
        <div style="font-size: 24px;">{card['icon']}</div>
        <div style="font-weight: bold; margin: 8px 0;">{card['title']}</div>
        <div style="font-size: 18px; color: {border_color};">{card['value']}</div>
        {f'<div style="font-size: 14px; color: #9CA3AF; margin-top: 4px;">{card.get("subtitle", "")}</div>' if card.get("subtitle") else ''}
    </div>
    """, unsafe_allow_html=True)
```

---

## 6. 디자인 (H 섹션)

### 6.1 H-1. 폰트 — Pretendard fallback chain (안정성 보강)

`.streamlit/config.toml` 외 추가 CSS 적용 — `app.py` 또는 별도 함수:

```python
def inject_custom_css():
    """Pretendard 폰트 fallback chain 주입 (H-1)."""
    st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    html, body, [class*="css"] {
        font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", -apple-system, sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)
```

**Fallback chain**: Pretendard (한국어 친화) → Noto Sans KR → Malgun Gothic → -apple-system → sans-serif

**근거 (H-1)**:
- Pretendard = 한국어 가독성 ★★★ (무료 웹 폰트)
- CDN 차단 시 자동 fallback → 기능 정상

### 6.2 H-3. 표준 CSS 최소화 원칙

**최소화 적용**:
1. 카드 / Insight 박스 / accent 색상만 custom CSS
2. Streamlit 자체 컴포넌트 우선 (`st.metric`, `st.expander`, `st.dataframe`)
3. inline style 우선 (`st.markdown` 내 `<div style="...">`)
4. 복잡한 layout = `st.columns` (CSS 직접 X)

### 6.3 H-5. `.streamlit/config.toml` 다크 테마

```toml
[theme]
base = "dark"
primaryColor = "#3B82F6"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1F2937"
textColor = "#FAFAFA"
font = "sans serif"
```

---

## 7. 인터랙션 (G 섹션)

### 7.1 G-1. Q-Zoom — 같은 페이지 expand 패턴

**구현**: `lib/interactions.py`

```python
"""
lib/interactions.py
Q-Zoom (클릭 → expand) 인터랙션 헬퍼
G-1 결정: 같은 페이지 expand 영역 (Modal X / 별도 페이지 X)
"""
import streamlit as st
import plotly.graph_objects as go

def render_zoomable_chart(fig: go.Figure, key: str, expand_func=None):
    """
    Plotly 차트 렌더 + 클릭 시 같은 페이지 expand 영역 표시.

    Args:
        fig: Plotly Figure (이미 Range slider, Hover tooltip 포함)
        key: Streamlit unique key
        expand_func: 클릭 시 호출될 expand 함수 (선택)
    """
    selected = st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select="rerun"  # Streamlit 1.27+ 필요
    )

    # 사용자 selection 처리
    if selected and selected.get("selection") and expand_func:
        with st.expander("📍 선택한 시점 상세", expanded=True):
            expand_func(selected["selection"])
```

### 7.2 Q-Zoom 영향 범위 (모든 페이지)

| 페이지 / 영역 | 적용 |
|---|---|
| Performance 영역 4 (Annual Returns) | 연도 클릭 → 월별 |
| Performance 영역 5 (Active Return) | 연도 클릭 → 월별 |
| Performance 영역 6 (Rolling Return) | 윈도우 클릭 → detail |
| Performance 영역 7 (Regime Heatmap) | Regime 클릭 → 메트릭 expand |
| Risk Metrics 영역 4 (Drawdown) | 시기 클릭 → detail |
| Holdings 영역 4 (Top N 표) | 종목 클릭 → detail |
| Holdings 영역 5 (Bubble/Treemap) | 종목 클릭 → detail |
| Holdings 영역 6 (변천사) | 시점 클릭 → detail |
| Holdings 영역 7 (Tornado) | 종목 클릭 → detail |
| Sector Watch 영역 4 (Treemap) | 섹터 클릭 → 종목 list |
| Sector Watch 영역 6 (Tilt) | 섹터 클릭 → detail |
| Sector Watch 영역 7 (Rotation) | 시점 클릭 → detail |
| Backtesting 영역 4 (Regime 자세) | Regime 행 클릭 → 영역 5 navigation |
| Backtesting 영역 5 (Sub-events) | Event 클릭 → 차트 expand |

### 7.3 G-2. Sim 페이지 토글 영향

Investment Simulator 페이지도 사이드바 토글 영향 (인터랙션 일관성 원칙).

### 7.4 G-3 ~ G-5. Streamlit 기본 활용

- **navigation**: 사이드바 page_link + Overview 영역 4/5 카드 클릭
- **모바일 반응형**: `st.columns` 자동 반응 (별도 CSS X)
- **키보드 / 접근성**: Streamlit 표준 (별도 ARIA X)

---

## 8. lib/tooltips.py 명세 (메트릭 정의 dictionary)

```python
"""
lib/tooltips.py
모든 페이지에서 사용하는 메트릭 정의 + Hover tooltip 표준
"""

METRIC_TOOLTIPS = {
    # Pool-1 수익성
    "Cumulative Return": "누적수익률. 시작 시점부터 현재까지 총 수익.",
    "CAGR": "연환산 복리 수익률 (Geometric Mean).",
    "Arithmetic Mean": "산술 평균 수익률 (월별 / 연환산).",
    "Net CAGR": "거래비용 차감 후 연환산 수익률 (One-way 20bp).",

    # Pool-2 위험조정수익
    "Sortino": "(R - Rf) / σ_downside. 하방위험만 페널티.",
    "Sharpe": "(R - Rf) / σ. 전체 변동성 페널티.",
    "Calmar": "CAGR / |MDD|. 최대낙폭 대비 수익.",
    "IR": "Information Ratio. 액티브 수익 / Tracking Error.",
    "M²": "Modigliani² (1997). Sharpe 의 % 환산.",

    # Pool-3 위험
    "Volatility": "변동성 (연환산 표준편차). 펀드 가격 변동의 강도. 낮을수록 안정적.",
    "MDD": "최대낙폭 — 고점 대비 최대 손실. 절대값이 작을수록 좋음.",
    "Downside Deviation": "하방 표준편차 (음수 수익률만).",
    "VaR 5%": "Historical VaR 5% — 5% 분위수의 손실.",
    "CVaR 5%": "Conditional VaR — VaR 초과 시 평균 손실 (Expected Shortfall).",
    "Beta": "시장 베타 — SPY 대비 시장 노출도. 1보다 작으면 시장 변동에 덜 민감.",
    "R²": "결정계수 — Beta 의 설명력. 100% 에 가까울수록 시장 변동으로 펀드 변동 설명 가능.",
    "Tracking Error": "추적오차 — 벤치마크 대비 액티브 운용 위험. 낮을수록 벤치마크 추적, 높을수록 액티브.",

    # Pool-4 운용 효율성
    "Number of Holdings": "보유 종목 수.",
    "Effective N": "유효 종목 수 = 1 / Σw² (분산도 척도). 높을수록 분산.",
    "Single Stock HHI": "개별 종목 집중도 = Σw² (Herfindahl-Hirschman Index). 낮을수록 분산.",
    "Sector HHI": "섹터 집중도 = Σ(섹터 weight)². 낮을수록 섹터 분산.",
    "Top Weights": "상위 N 종목의 비중 합계. 낮을수록 분산.",
    "Avg Turnover": "월평균 회전율. 낮을수록 안정 운용.",

    # Pool-5 시장 비교
    "Win Rate": "양수 수익률 월 비율.",
    "Up Capture": "시장 상승 시 펀드 상승 비율.",
    "Down Capture": "시장 하락 시 펀드 하락 비율.",
    "Active Return": "Fund - Benchmark 차이.",

    # Sector Watch
    "Avg |Tilt|": "섹터별 (펀드 weight - SPY weight) 절대값 평균. Active 운용 강도.",
    "Active Bets": "|Tilt| > 1% 인 섹터 수. 높을수록 액티브 운용.",
}

def get_tooltip(metric_name: str) -> str:
    """메트릭 이름 → tooltip 텍스트."""
    return METRIC_TOOLTIPS.get(metric_name, "정의 미정.")
```

---

## 9. lib/plot_helpers.py 명세 (Plotly 공통 헬퍼)

```python
"""
lib/plot_helpers.py
Plotly 공통 헬퍼 (Regime 배경색 / 한/영 병기 라벨 등)
"""
import plotly.graph_objects as go
from lib.colors import REGIME_COLORS

REGIME_PERIODS = [
    ("R1", "2010-01-01", "2012-06-30", "회복기"),
    ("R2", "2012-07-01", "2019-12-31", "확장기"),
    ("R3", "2020-01-01", "2023-12-31", "변동기"),
    ("HO", "2024-01-01", "2025-12-31", "홀드아웃"),
]

def add_regime_backgrounds(fig: go.Figure, with_labels: bool = True):
    """
    모든 시계열 차트에 Regime 4개 배경색 추가 (Overview 영역 3 패턴).
    """
    for regime_id, start, end, label in REGIME_PERIODS:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=REGIME_COLORS[regime_id],
            opacity=0.3,
            layer="below",
            line_width=0,
        )
        if with_labels:
            fig.add_annotation(
                x=start, y=1.0,
                yref="paper",
                text=f"{regime_id} {label}",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color="#9CA3AF"),
            )
    return fig


def add_event_annotations(fig: go.Figure):
    """위기 이벤트 annotation (COVID / 2022 Bear / 2024 IT Rotation)."""
    events = [
        ("2020-03-01", "▼ COVID-19"),
        ("2022-09-01", "▼ 2022 Bear"),
        ("2024-12-01", "▼ AI Rally / IT Rotation"),
    ]
    for date, label in events:
        fig.add_vline(
            x=date,
            line_dash="dot",
            line_color="#EF4444",
            line_width=1,
            annotation_text=label,
            annotation_position="top",
        )
    return fig


def bilingual_label(en: str, ko: str) -> str:
    """한/영 병기 라벨 (A-3 결정)."""
    return f"{en} ({ko})"
```

---

## 10. 페이지 헤더 / 서브헤더 표준

### 10.1 모든 페이지 동일 Header (Overview 영역 1 패턴)

```python
def render_page_header(page_name_en: str, page_name_ko: str):
    """모든 페이지 상단 Header (Overview 영역 1 일관 적용)."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"{page_name_en}")
        st.caption(f"{page_name_ko}")
    with col2:
        st.markdown("● **Active (Simulated)**")
        st.caption("Benchmark: S&P 500 (SPY)")
        st.caption("Data as of: 2025-12-31")
```

### 10.2 모든 페이지 동일 Sub-header (Performance 영역 2 패턴 일관)

```python
def render_subheader(title_en: str, title_ko: str, description: str):
    """페이지별 Sub-header 카드/배너 (S2-2 패턴)."""
    st.markdown(f"""
    <div style="
        background-color: #1F2937;
        border-left: 4px solid #3B82F6;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 16px;
    ">
        <div style="font-size: 18px; font-weight: bold; color: #FAFAFA;">
            ℹ️ {title_en} ({title_ko})
        </div>
        <div style="font-size: 14px; color: #9CA3AF; margin-top: 8px;">
            {description}
        </div>
    </div>
    """, unsafe_allow_html=True)
```

---

## 11. 다음 단계

→ `03_pages/`: 9 페이지 각각 와이어프레임 + 결정사항 + 시각화 + 구현 체크리스트

---

[← 01_setup.md](01_setup.md) | [03_pages/01_overview.md →](03_pages/01_overview.md)
