"""
pages/03_Performance.py - Performance 페이지 (10 영역)

성과 분석 전담 페이지.

10 영역 (2026-05-11 영역 4 신규 추가):
  1. Header
  2. Sub-header (페이지 컨텍스트)
  3. Performance Summary KPI 5개 (CAGR / Sortino / Sharpe / IR / Active Return)
  4. (신규) 누적 수익률 — Performance Trend (단일 차트, Drawdown 제외)
  5. Annual Returns 막대 (사이드바이 + 평균선)
  6. Active Return 분석 (위 막대 + 아래 Rolling 이중 축)
  7. Annualized Rolling Return (1y/3y/5y 다중 토글 + Regime 배경)
  8. Regime 메트릭 Heatmap (Tab 전환)
  9. 분포 통계 카드 (Skewness / Excess Kurtosis / Tail Ratio — 일별 only)
  10. Footer

모든 메트릭 = final/bl_functions + master_table 정확 재현.

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
    render_cumulative_only,
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
fund = load_fund_results()  # default = mat_eq_eq_raw_pap (최종 Top 1)
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
        "펀드의 수익성을 시장 대비 + 시장 국면별로 분석합니다. "
        "사이드바에서 기간 / 비교 벤치마크 선택 시 차트가 자동 갱신됩니다."
    ),
)


# === 영역 3: Performance Summary KPI ==================================
period = st.session_state.get("period", "FULL")
st.subheader(f"핵심 성과 지표 — {period}")
st.caption(
    "**연평균 수익률 (CAGR) / 하방위험 조정 수익 (Sortino) / 위험 조정 수익 (Sharpe) / "
    "정보 비율 (IR) / 시장 초과 수익 (Active Return)** — 모두 **거래비용 차감 후** 기준 "
    "(편측 20bp = 1회 거래당 0.20%). "
    "CAGR 카드 하단에는 비용 차감 전 (Gross) 비교를 함께 표시합니다."
)
render_performance_kpi(
    fund_ret, fund_spy, ew_ret, ivw_ret, fund_rf, period,
    fund_gross_ret=fund_gross,
)
st.divider()


# === 영역 4: 누적 수익률 — Performance Trend (신규 — 2026-05-11) =======
st.subheader("누적 수익률 — Performance Trend")
st.caption(
    "선택한 기간 (사이드바) 동안의 자산 누적 변화. "
    "비교 벤치마크 (SPY / 균등가중 / 역변동성) 는 사이드바에서 토글하시면 함께 표시됩니다. "
    "**전체 (FULL) 일 때만** 시장 국면 배경 + 주요 이벤트가 표시됩니다."
)
col_log = st.columns([1, 5])
with col_log[0]:
    show_log_perf = st.toggle("Log scale", value=False, key="perf_log")
render_cumulative_only(fund_ret, fund_spy, ew_ret, ivw_ret, show_log_perf, period)
st.divider()


# === 영역 5: Annual Returns 막대 ======================================
st.subheader("연간 수익률 — Annual Returns")
st.caption("연도별 수익률 막대 + 평균 수익률 선 (펀드 vs 선택한 벤치마크)")
render_annual_returns(fund_ret, fund_spy, ew_ret, ivw_ret, period)
st.divider()


# === 영역 6: Active Return 분석 =======================================
st.subheader("액티브 수익 분석 — Active Return Analysis")
st.caption(
    "**액티브 수익 (Active Return)** = 펀드 수익률 − 벤치마크 수익률. "
    "양수 = 시장 초과 (펀드가 더 잘함), 음수 = 시장 열위. "
    "**Tracking Error** = 액티브 수익의 표준편차로, 펀드가 벤치마크를 얼마나 일관되게 추종하는지 측정 "
    "(낮을수록 추종형 운용, 높을수록 자유 운용 — 일반 active fund 는 보통 4-8%)."
)
render_active_return_analysis(fund_ret, fund_spy, ew_ret, ivw_ret, period)
st.divider()


# === 영역 7: Annualized Rolling Return ================================
st.subheader("연환산 Rolling 수익률 — Annualized Rolling Return")
st.caption("1년 / 3년 / 5년 기간 연환산 수익률 시계열. 사이드바 기간이 \"전체 (FULL)\" 일 때만 시장 국면 배경이 표시됩니다.")

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


# === 영역 8: Regime 메트릭 Heatmap ====================================
st.subheader("Regime 메트릭 비교 — Regime Heatmap")
st.caption(
    "시장 국면별 펀드 vs 벤치마크 메트릭 비교. "
    "회복기 (2010-2012) / 확장기 (2012-2019) / 변동기 (2020-2023) / "
    "Hold Out 24m (2024-2025) / 전체 (2010-2025)."
)
render_regime_heatmap(fund_ret, fund_spy, ew_ret, ivw_ret, fund_rf)
st.divider()


# === 영역 9: 분포 통계 카드 ===========================================
st.subheader("분포 통계 — Distribution Statistics")
st.caption(
    "수익률 분포의 **왜도** (비대칭성, 기준=0) / **첨도** (극단값 정도, 기준=0) / "
    "**꼬리 비율** (상승 vs 하락 꼬리, 기준=1). "
    "**일별 데이터** (~4,000 sample) 기준 — 월별 (192 sample) 은 중심극한정리 효과로 분포가 정규에 수렴해 의미가 약함. "
    "균형에 가까운 왜도 (✓ — SPY 음수 왜도 대비 우월), 높은 첨도 = fat tail (주식 일반 패턴), 꼬리 비율 > 1 = 상승 꼬리 우세 (✓). "
    "ℹ️ 일별 = 시장 데이터 기반 (펀드 backtest 의 월별 source 와 별도)."
)

with st.expander("ℹ️ 데이터 source 안내 (분포 통계)"):
    st.markdown(
        """
        **왜 일별 데이터인가?**
        - 분포 형상 (왜도/첨도/꼬리비율) 은 sample 수가 충분해야 신뢰 가능
        - 일별 (~4,000) vs 월별 (192): 월별은 중심극한정리 (CLT) 효과로 정규에 수렴 → 분포 특성 분석 의미 약함
        - 학술 표준: 분포 통계는 일별 데이터 사용 (Cont 2001 — *Empirical properties of asset returns*)

        **일별 데이터의 source**
        - 출처: yfinance Adjusted Close 의 일별 가격 변동 (`daily_returns.pkl`)
        - 원래 의도: LW (Ledoit-Wolf) 공분산 추정 / 변동성 분포 분석용

        **펀드 backtest 와의 관계**
        - 펀드의 누적 수익률 / KPI (CAGR, Sortino 등) = **월별 backtest 데이터** (`monthly_panel.fwd_ret_1m`) 기반
        - 본 영역의 분포 형상 = **일별 시장 데이터** 기반
        - 두 데이터는 다른 분석 목적으로 설계되어 누적 수익률 일치 X
        - 이는 **실무 펀드 분석의 표준 패턴**: 월별 NAV (official 성과) + 일별 risk metric (학술 표준)
        """
    )

# 일별 portfolio return 산출 (캐시됨)
fund_daily = None
spy_daily = None
try:
    daily_returns = load_daily_returns()
    with st.spinner("일별 portfolio return 산출 중 (최초 1회만)..."):
        fund_daily = compute_fund_daily_returns(fund_weights, daily_returns, _comp=fund["comp"], tc=0.002)
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


# === 영역 10: Footer ==================================================
render_footer()
