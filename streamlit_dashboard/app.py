"""
Adaptive VolControl Fund - 대시보드 메인 entry (Overview 페이지)

이 파일 = Overview 페이지 (사이드바 첫 page_link).
사용자는 사이드바에서 다른 페이지로 이동.

Overview 6 영역:
  1. Header           — 펀드명 + 메타 (page_helpers.render_page_header)
  2. Hero KPI         — 5 KPI 카드 (TEST/HO 별도, 사이드바 토글 영향 X)
  3. 누적수익 곡선     — 이중 차트 + Regime + 비교 라인 (SPY/EW/IVW 토글)
  4. 핵심 강점 카드    — 3 카드 (Methodology / Backtesting / Performance navigation)
  5. Navigation cards — 7 페이지 카드 그리드
  6. Footer           — Disclosure 통일 (disclosure.render_footer)

참조:
  - docs/plan/03_pages/01_overview.md
  - docs/decisionlog/02_overview.md
"""

import streamlit as st

from lib.data_loader import (
    compute_equal_weight_returns,
    compute_ivw_returns,
    load_fund_results,
    load_monthly_panel,
    load_sp500_membership,
)
from lib.disclosure import init_session_state, render_footer
from lib.overview_charts import (
    render_cumulative_chart,
    render_differentiator_cards,
    render_hero_kpi,
    render_navigation_cards,
)
from lib.page_helpers import inject_custom_css, render_page_header, render_subheader
from lib.validators import startup_data_check


# === Step 1: 페이지 설정 (반드시 가장 먼저 호출) ======================
st.set_page_config(
    page_title="Adaptive VolControl Fund",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# === Step 2: 폰트 주입 (Pretendard fallback chain) ====================
inject_custom_css()


# === Step 3: Startup check — 필수 데이터 파일 검증 (D-5) ==============
startup_data_check()


# === Step 4: Session state 초기화 (사이드바 토글) =====================
init_session_state()


# === Step 5: 사이드바 (C-4 6 그룹 + 2 토글) ===========================
with st.sidebar:
    # ── 펀드명 + 메타 (C4-3) ────────────────────────────────────────
    st.markdown("# Adaptive VolControl Fund")
    st.markdown("어댑티브 볼컨트롤 펀드")
    st.caption("Benchmark: SPY  |  Data: 2025-12")
    st.divider()

    # ── 6 그룹 페이지 navigation (C4-1 c, C4-2 a) ──────────────────
    # 자동 nav 비활성화 (.streamlit/config.toml) + st.page_link 명시적

    # 그룹 1: 개요
    st.markdown("##### ── 개요 ──")
    st.page_link("app.py", label="Overview", icon="📊")

    # 그룹 2: 체험 (★ Investment Simulator F-6)
    st.markdown("##### ── 체험 ──")
    st.page_link("pages/02_Investment_Simulator.py", label="Investment Simulator", icon="💵")

    # 그룹 3: 성과
    st.markdown("##### ── 성과 ──")
    st.page_link("pages/03_Performance.py", label="Performance", icon="📈")
    st.page_link("pages/04_Risk_Metrics.py", label="Risk Metrics", icon="⚠️")

    # 그룹 4: 보유
    st.markdown("##### ── 보유 ──")
    st.page_link("pages/05_Holdings.py", label="Holdings", icon="🏢")
    st.page_link("pages/06_Sector_Watch.py", label="Sector Watch", icon="🌐")

    # 그룹 5: 검증
    st.markdown("##### ── 검증 ──")
    st.page_link("pages/07_Methodology.py", label="Methodology", icon="🧪")
    st.page_link("pages/08_Backtesting.py", label="Backtesting", icon="✅")

    # 그룹 6: 메타
    st.markdown("##### ── 메타 ──")
    st.page_link("pages/09_About.py", label="About / FAQ", icon="ℹ️")

    st.divider()

    # ── 토글 1: 기간 (Period) — C4-4 ───────────────────────────────
    st.subheader("📅 기간 (Period)")
    st.radio(
        "기간 선택",
        options=["FULL", "TEST", "HO"],
        index=["FULL", "TEST", "HO"].index(st.session_state.period),
        key="period",
        label_visibility="collapsed",
    )

    # ── 토글 2: 비교 벤치마크 — C4-4 ────────────────────────────────
    st.subheader("📊 비교 (Benchmark)")
    st.checkbox("SPY", value=st.session_state.show_spy, key="show_spy")
    st.checkbox("EW (펀드 universe)", value=st.session_state.show_ew, key="show_ew")
    st.checkbox("IVW (Naive Low-vol)", value=st.session_state.show_ivw, key="show_ivw")


# === 데이터 로드 (캐시) ================================================
fund = load_fund_results("mat_eq_eq_raw_pap")
fund_ret = fund["ret"]               # Net 월별 수익률 (Series)
fund_gross = fund["gross_ret"]       # TC 차감 전
fund_spy = fund["spy_ret"]           # SPY 월별 수익률


# === 영역 1: Header ===================================================
render_page_header("Overview", "전체 개요")


# === Sub-header (페이지 핵심 메시지) ===================================
render_subheader(
    title_en="Adaptive Volatility Control Fund",
    title_ko="어댑티브 볼컨트롤 펀드",
    description=(
        "Black-Litterman + LSTM 기반 4-slot 적응적 변동성 제어 전략. "
        "TEST 168m 평가 + HOLD_OUT 24m (true OOS) 구조로 검증한 가상 펀드."
    ),
)


# === 영역 2: Hero KPI 5개 (TEST + HO 별도, 사이드바 토글 영향 X) ======
st.subheader("핵심 성과 지표")
st.caption("TEST 평가 168m / HOLD_OUT 24m 별도 표시 — 사이드바 토글에 영향 받지 않음 (학술 정직성)")
render_hero_kpi(fund_ret, fund_gross)
st.divider()


# === 영역 3: 누적수익 곡선 (이중 차트 + 비교 라인 토글) ===============
st.subheader("누적 수익률 — Cumulative Return")

# 사이드바 토글에 따라 비교 baseline 산출 (캐시됨, monthly_panel 기반 — 옵션 E)
ew_ret = None
ivw_ret = None
if st.session_state.show_ew or st.session_state.show_ivw:
    # monthly_panel 기반 EW/IVW (decisionlog Q-D 옵션 E)
    monthly_panel = load_monthly_panel()
    sp500_membership = load_sp500_membership()
    if st.session_state.show_ew:
        with st.spinner("EW baseline 산출 중 (최초 1회만, 약 5초)..."):
            ew_ret = compute_equal_weight_returns(monthly_panel, sp500_membership, fund_ret.index)
    if st.session_state.show_ivw:
        with st.spinner("IVW baseline 산출 중 (최초 1회만, 약 5초)..."):
            ivw_ret = compute_ivw_returns(monthly_panel, sp500_membership, fund_ret.index)

# Y축 Log 토글 + 차트 옵션
col_opt1, col_opt2 = st.columns([1, 5])
with col_opt1:
    show_log = st.toggle("Log scale", value=False, key="overview_log")

# SPY 표시는 사이드바 show_spy 에 따름 (False 면 빈 series)
spy_input = fund_spy if st.session_state.show_spy else fund_spy.iloc[:0]

render_cumulative_chart(
    fund_ret=fund_ret,
    spy_ret=spy_input,
    ew_ret=ew_ret,
    ivw_ret=ivw_ret,
    show_log=show_log,
    period=st.session_state.period,
)

st.divider()


# === 영역 4: 핵심 강점 카드 3개 =======================================
st.subheader("핵심 차별화 — Why this Fund")
render_differentiator_cards(fund_ret)
st.divider()


# === 영역 5: Navigation cards 7개 =====================================
st.subheader("페이지 둘러보기 — Explore")
render_navigation_cards()


# === 영역 6: Footer ===================================================
render_footer()
