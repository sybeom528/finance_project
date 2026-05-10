"""
pages/03_Performance.py - Performance 페이지 (9 영역)

성과 분석 전담 페이지. Overview Hero KPI / 누적수익 곡선의 깊이 있는 버전.

9 영역:
  1. Header
  2. Sub-header (페이지 컨텍스트)
  3. Performance Summary KPI 5개 (CAGR / Sortino / Sharpe / IR / Active Return)
  4. Annual Returns 막대 (사이드바이 + 평균선)
  5. Active Return 분석 (위 막대 + 아래 Rolling 이중 축)
  6. Annualized Rolling Return (1y/3y/5y 다중 토글 + Regime 배경)
  7. Regime 메트릭 Heatmap (Tab 전환)
  8. 분포 통계 카드 (Skewness / Excess Kurtosis / Tail Ratio)
  9. Footer

모든 메트릭 = final/bl_functions + master_table 정확 재현 (decisionlog Q-E).

참조: docs/plan/03_pages/03_performance.md, decisionlog/03_performance.md
"""

import streamlit as st

from lib.data_loader import (
    compute_equal_weight_returns,
    compute_fund_daily_returns,
    compute_ivw_returns,
    load_daily_returns,
    load_fund_results,
    load_monthly_panel,
    load_sp500_membership,
)
from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import (
    inject_custom_css,
    render_page_header,
    render_sidebar,
    render_subheader,
)
from lib.performance_charts import (
    render_active_return_analysis,
    render_annual_returns,
    render_distribution_stats,
    render_performance_kpi,
    render_regime_heatmap,
    render_rolling_return,
)


# === 페이지 설정 ======================================================
inject_custom_css()
init_session_state()
render_sidebar()


# === 데이터 로드 ======================================================
fund = load_fund_results(st.session_state.get("config_name", "mat_eq_eq_raw_pap"))
fund_ret = fund["ret"]
fund_gross = fund["gross_ret"]
fund_spy = fund["spy_ret"]
fund_weights = fund["weights"]

panel = load_monthly_panel()
fund_rf = panel.groupby("date")["rf_1m"].first()


# === EW / IVW 산출 (사이드바 토글에 따라) =============================
ew_ret = None
ivw_ret = None
if st.session_state.get("show_ew", False) or st.session_state.get("show_ivw", False):
    sp500_membership = load_sp500_membership()
    if st.session_state.get("show_ew", False):
        with st.spinner("EW baseline 산출 중 (최초 1회만)..."):
            ew_ret = compute_equal_weight_returns(panel, sp500_membership, fund_ret.index)
    if st.session_state.get("show_ivw", False):
        with st.spinner("IVW baseline 산출 중 (최초 1회만)..."):
            ivw_ret = compute_ivw_returns(panel, sp500_membership, fund_ret.index)


# === 영역 1: Header ===================================================
render_page_header("Performance", "성과 분석")


# === 영역 2: Sub-header ===============================================
render_subheader(
    title_en="Performance Analysis",
    title_ko="성과 분석",
    description=(
        "펀드의 수익성, 시장 비교, regime 별 분해 분석. "
        "사이드바에서 기간 (FULL / TEST / HO) 선택 시 차트 자동 갱신. "
        "비교 벤치마크 (SPY / EW / IVW) 도 선택 가능."
    ),
)


# === 영역 3: Performance Summary KPI ==================================
period = st.session_state.get("period", "FULL")
st.subheader(f"핵심 성과 지표 — {period}")
st.caption("CAGR / Sortino / Sharpe / IR / Active Return — final/master_table 정합")
render_performance_kpi(fund_ret, fund_spy, ew_ret, ivw_ret, fund_rf, period)
st.divider()


# === 영역 4: Annual Returns 막대 ======================================
st.subheader("연간 수익률 — Annual Returns")
st.caption("연도별 사이드바이 막대 + 평균선 (Fund / 활성 벤치마크)")
render_annual_returns(fund_ret, fund_spy, ew_ret, ivw_ret, period)
st.divider()


# === 영역 5: Active Return 분석 =======================================
st.subheader("액티브 수익 분석 — Active Return Analysis")
col_w1, col_w2 = st.columns([1, 5])
with col_w1:
    rolling_window = st.selectbox(
        "Rolling 윈도우",
        options=[12, 36, 60],
        index=1,
        key="perf_rolling_window",
        format_func=lambda x: f"{x}m",
    )
render_active_return_analysis(
    fund_ret, fund_spy, ew_ret, ivw_ret, period, rolling_window
)
st.divider()


# === 영역 6: Annualized Rolling Return ================================
st.subheader("연환산 Rolling 수익률 — Annualized Rolling Return")
st.caption("1y / 3y / 5y 다중 윈도우 토글. 기간 토글 = FULL 일 때만 Regime 배경 표시.")

col_w = st.columns([1, 1, 1, 5])
with col_w[0]:
    show_1y = st.checkbox("1y", value=True, key="perf_w1y")
with col_w[1]:
    show_3y = st.checkbox("3y", value=True, key="perf_w3y")
with col_w[2]:
    show_5y = st.checkbox("5y", value=False, key="perf_w5y")

windows = []
if show_1y:
    windows.append(1)
if show_3y:
    windows.append(3)
if show_5y:
    windows.append(5)

render_rolling_return(fund_ret, fund_spy, ew_ret, ivw_ret, period, windows)
st.divider()


# === 영역 7: Regime 메트릭 Heatmap ====================================
st.subheader("Regime 메트릭 비교 — Regime Heatmap")
st.caption(
    "R1 회복 (2010-01 — 2012-06) / R2 확장 (2012-07 — 2019-12) / R3 변동 (2020-01 — 2024-12) / "
    "HO 홀드아웃 (2024-01 — 2025-12) / FULL (2010 — 2025) — final REGIMES 정의 그대로"
)
render_regime_heatmap(fund_ret, fund_spy, ew_ret, ivw_ret, fund_rf)
st.divider()


# === 영역 8: 분포 통계 카드 ===========================================
st.subheader("분포 통계 — Distribution Statistics")
st.caption(
    "Skewness (왜도) / Excess Kurtosis (초과첨도, fat tail) / Tail Ratio (꼬리 비율). "
    "일별 데이터는 fund.weights × daily_returns (final/bl_functions.compute_daily_slice 패턴: "
    "NaN < 10% threshold + fillna(0)) 로 산출."
)

# 일별 portfolio return 산출 (캐시됨)
fund_daily = None
spy_daily = None
try:
    daily_returns = load_daily_returns()
    with st.spinner("일별 portfolio return 산출 중 (최초 1회만)..."):
        fund_daily = compute_fund_daily_returns(fund_weights, daily_returns)
    if "SPY" in daily_returns.columns:
        spy_daily = daily_returns["SPY"].dropna()
except Exception as exc:
    st.warning(f"일별 데이터 로드 실패: {exc}")

render_distribution_stats(
    fund_ret_monthly=fund_ret,
    spy_ret_monthly=fund_spy,
    ew_ret_monthly=ew_ret,
    ivw_ret_monthly=ivw_ret,
    fund_ret_daily=fund_daily,
    spy_ret_daily=spy_daily,
    period=period,
)


# === 영역 9: Footer ===================================================
render_footer()
