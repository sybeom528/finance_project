"""
pages/04_Risk_Metrics.py - Risk Metrics 페이지 (9 영역)

위험 지표 전담 페이지. Performance 가 수익 위주라면 Risk Metrics 는 위험 위주.

9 영역:
  1. Header
  2. Sub-header
  3. Risk Summary KPI 5개 (Vol / MDD / Beta / R² / TE)
  4. Drawdown 시계열 + Recovery Time + Top 3 표
  5. VaR / CVaR 분포 (월별/일별 Tab + Fund vs SPY 오버레이 + 정규분포 비교)
  6. Volatility / Sortino / Beta / R² / TE Rolling 시계열 (Q-G 보강 — 5 메트릭 통합)
  7. Risk Metrics 종합 표 (~22 메트릭, 카테고리 expander, CSV 다운로드)
  8. Tail Risk 분석 (Hill estimator + Hill plot, 일별 only)
  9. Footer

모든 메트릭 = final/bl_functions.compute_metrics + 학술 표준 (decisionlog Q-E,Q-F,Q-G).

참조: docs/plan/03_pages/04_risk_metrics.md, decisionlog/04_risk_metrics.md
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
from lib.risk_charts import (
    render_drawdown_recovery,
    render_risk_kpi,
    render_risk_metrics_table,
    render_rolling_risk_metrics,
    render_tail_risk,
    render_var_cvar_distribution,
)


# === 페이지 설정 ======================================================
inject_custom_css()
init_session_state()
render_sidebar()


# === 데이터 로드 ======================================================
fund = load_fund_results("mat_eq_eq_raw_pap")
fund_ret = fund["ret"]
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
render_page_header("Risk Metrics", "위험 지표")


# === 영역 2: Sub-header ===============================================
render_subheader(
    title_en="Risk Metrics",
    title_ko="위험 지표",
    description=(
        "펀드의 위험 통제 / 시장 노출 / Tail Risk 분석. "
        "사이드바에서 기간 (FULL / TEST / HO) + 비교 벤치마크 (SPY / EW / IVW) 토글 가능. "
        "모든 메트릭은 final/bl_functions.compute_metrics 와 정확 정합 + 학술 표준."
    ),
)


period = st.session_state.get("period", "FULL")


# === 영역 3: Risk Summary KPI 5 ======================================
st.subheader(f"위험 KPI — {period}")
st.caption(
    "Volatility / MDD / Beta / R² / Tracking Error. "
    "위험 KPI 는 **낮을수록 좋음** (delta 음수 = green = 좋음)."
)
render_risk_kpi(fund_ret, fund_spy, ew_ret, ivw_ret, fund_rf, period)
st.divider()


# === 영역 4: Drawdown + Recovery Time =================================
st.subheader("Drawdown + Recovery Time")
st.caption(
    "DD 시계열 (Fund + SPY) + Top 3 DD 마커 + Recovery Time 표 + 평균 DD 라인. "
    "Top 3 = 가장 깊은 드로다운 3개 (depth 기준)."
)
render_drawdown_recovery(fund_ret, fund_spy, period)
st.divider()


# === 영역 5: VaR / CVaR 분포 ==========================================
st.subheader("VaR / CVaR 분포")
st.caption(
    "월별/일별 Tab + Fund vs SPY 오버레이 히스토그램 + 정규분포 비교 라인 + VaR/CVaR 임계선. "
    "VaR 5% / CVaR 5% (Historical method, pandas.quantile) — final compute_metrics.cvar_5 정합."
)

# 일별 portfolio (영역 5, 8 공유 — 한 번 산출)
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

render_var_cvar_distribution(
    fund_ret_monthly=fund_ret,
    spy_ret_monthly=fund_spy,
    fund_ret_daily=fund_daily,
    spy_ret_daily=spy_daily,
    period=period,
)
st.divider()


# === 영역 6: Rolling 5 메트릭 (Q-G 보강) ==============================
st.subheader("Rolling 위험 메트릭 — Volatility / Sortino / Beta / R² / TE")
st.caption(
    "5 메트릭 통합 rolling 시계열 (Q-G 결정 보강). Hero KPI sparkline 제거 보강. "
    "Window 토글 (12 / 36 / 60m, **36m 권장 — Sortino 분모 안정성**). "
    "Beta < 0 시 R² 함께 확인 (신뢰성 평가, R6-6)."
)
col_w = st.columns([1, 5])
with col_w[0]:
    rolling_window = st.selectbox(
        "Window", options=[12, 36, 60], index=1,  # 36m default
        key="risk_rolling_window", format_func=lambda x: f"{x}m",
    )

if rolling_window == 12:
    st.caption(
        "⚠️ **12m window 한계**: 일부 시점에서 음수 수익률 < 2 → Sortino 분모 산출 불가 (NaN 끊김), "
        "또는 음수 수익률 < 5 → 분모 매우 작아 spike 발생. 학술적으로 정확하지만 시각적 변동 큼. "
        "안정적 해석을 위해 **36m 사용 권장** (Sortino & Price 1994)."
    )
render_rolling_risk_metrics(fund_ret, fund_spy, fund_rf, period, rolling_window)
st.divider()


# === 영역 7: Risk Metrics 종합 표 ====================================
st.subheader(f"위험 메트릭 종합 표 — {period}")
st.caption(
    "~20 메트릭 카테고리 expander (Performance Returns / Risk-adj / Risk / Market Exposure "
    "/ Distribution). CSV 다운로드 가능. Final compute_metrics 정합 메트릭 + 학술 표준 메트릭."
)
render_risk_metrics_table(fund_ret, fund_spy, ew_ret, ivw_ret, fund_rf, period)
st.divider()


# === 영역 8: Tail Risk (Hill estimator) ===============================
st.subheader("Tail Risk 분석 — Hill Estimator")
st.caption(
    "분포 tail 두께 측정 (일별 only). Hill (1975) 학술 표준. "
    "ξ̂ > 0 = fat tail (극단 손실). 펀드 ξ̂ < SPY ξ̂ → tail risk 적음."
)
render_tail_risk(
    fund_ret_daily=fund_daily,
    spy_ret_daily=spy_daily,
)


# === 영역 9: Footer ===================================================
render_footer()
