"""
Adaptive VolControl Fund - 대시보드 메인 entry (Overview 페이지)

이 파일 = Overview 페이지 (사이드바 첫 page_link).
사용자는 사이드바에서 다른 페이지로 이동.

Overview 7 영역:
  1. Header               — 펀드명 + 메타 (page_helpers.render_page_header)
  2. Hero KPI             — 5 KPI 카드 (TEST/HO 별도, 사이드바 토글 영향 X)
  3. 누적수익 곡선         — 이중 차트 + Regime + 비교 라인 (SPY/EW/IVW 토글)
  4. 핵심 강점 카드        — 3 카드 (Risk Metrics / Performance navigation)
  5. Navigation cards     — 5 페이지 카드 (Methodology / Backtesting 통합 후)
  6. Methodology Overview — BL+LSTM Sankey (이전 Methodology 페이지에서 통합, 2026-05-11)
  7. Footer               — Disclosure 통일 (disclosure.render_footer)

통합 이력 (2026-05-11):
  - Methodology 페이지 삭제 → Sankey 만 영역 6 으로
  - Backtesting 페이지 삭제 → Regime 메트릭 + Sub-events 는 Risk Metrics 영역 5/6 으로

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
    render_methodology_sankey,
    render_navigation_cards,
)
from lib.page_helpers import inject_custom_css, render_page_header, render_sidebar, render_subheader
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
# 모든 페이지에서 동일한 사이드바를 그리도록 lib 함수 사용
render_sidebar()


# === 데이터 로드 (캐시) ================================================
fund = load_fund_results()  # default = mat_eq_eq_raw_pap (최종 Top 1)
fund_ret = fund["ret"]               # Net 월별 수익률 (Series)
fund_gross = fund["gross_ret"]       # TC 차감 전
fund_spy = fund["spy_ret"]           # SPY 월별 수익률

# 무위험 수익률 (월별) — final 정합성을 위해 panel.rf_1m 사용
# (Sortino / Sharpe / Beta / Alpha 계산 시 final/bl_functions:compute_metrics 와 일치)
_panel_for_rf = load_monthly_panel()
fund_rf = _panel_for_rf.groupby("date")["rf_1m"].first()


# === 영역 1: Header ===================================================
render_page_header("Overview", "전체 개요")


# === Sub-header (페이지 핵심 메시지) ===================================
render_subheader(
    title_en="Adaptive Volatility Control Fund",
    title_ko="변동성 인지 적응 펀드",
    description=(
        "변동성을 예측해 그에 맞춰 자산 배분을 조정하는 가상 펀드입니다. "
        "매 시점의 운용 결정 시 그 시점 이전 데이터만 사용하는 **Walk-forward 방식** 으로 "
        "전체 16년 (2010-2025) 운용. "
        "**TEST 168개월** (2010-2023) 에서는 시장보다 우수한 성과를 보였으며, "
        "**Hold Out 24개월** (2024-2025) AI 빅테크 집중 상승 시기에는 변동성 인지 배분 특성상 IT 섹터 비중을 낮게 유지해 시장을 일시 하회했습니다."
    ),
)


# === 영역 2: Hero KPI 5개 (TEST + HO 별도, 사이드바 토글 영향 X) ======
st.subheader("핵심 성과 지표")
st.caption("학습 기간 (14년) 과 검증 기간 (2년) 의 성과를 각각 표시합니다. 사이드바 기간 설정과 무관하게 고정됩니다.")
render_hero_kpi(fund_ret, fund_gross, rf=fund_rf)
st.divider()


# === 영역 3: 누적수익 곡선 (이중 차트 + 비교 라인 토글) ===============
st.subheader("누적 수익률 — Cumulative Return")
st.caption(
    "2010-2025년 (16년) 자산 누적 변화 (위) + 손실 폭 (아래). "
    "시장 국면 (회복기 / 확장기 / 변동기 / Hold Out) 배경색 + 주요 이벤트 (COVID, 2022 약세장 등) 표시. "
    "사이드바에서 비교 벤치마크 추가 + Y축 Linear/Log 변환 가능."
)

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
st.caption(
    "본 펀드의 3가지 핵심 가치 — **변동성 예측 기반 자산 배분** / "
    "**시장 국면별 일관된 검증** / **거래비용 차감 후 우수한 성과**."
)
render_differentiator_cards(fund_ret, rf=fund_rf)
st.divider()


# === 영역 5: 페이지 둘러보기 (Navigation Cards) =======================
st.subheader("페이지 둘러보기 — Explore")
st.caption(
    "5 페이지 detail 분석 — Performance / Risk Metrics / Holdings / Sector Watch / About. "
    "**Investment Simulator** 는 사이드바 \"체험\" 그룹에서 별도 접근."
)
render_navigation_cards()
st.divider()


# === 영역 6: Methodology Overview Sankey (이전 Methodology 페이지에서 통합) ===
st.subheader("Methodology Overview — BL+LSTM 흐름")
st.caption(
    "본 펀드의 운용 방법론 흐름도 — 데이터 입력부터 최종 포트폴리오 비중 산출까지의 4단계: "
    "**데이터 → Black-Litterman → LSTM 변동성 예측 → 최적화**."
)
render_methodology_sankey()


# === 영역 6: Footer ===================================================
render_footer()
