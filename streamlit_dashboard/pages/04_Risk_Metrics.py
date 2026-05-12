"""
pages/04_Risk_Metrics.py - Risk Metrics 페이지 (11 영역)

위험 지표 전담 페이지. Performance 가 수익 위주라면 Risk Metrics 는 위험 위주.

11 영역:
  1. Header
  2. Sub-header
  3. Risk Summary KPI 5개 (Vol / MDD / Beta / R² / TE)
  4. Drawdown 시계열 + Recovery Time + Top 3 표
  5. (이전) Regime 메트릭 자세한 비교 — Backtesting 페이지에서 통합 (2026-05-11)
  6. (이전) Sub-events 분석 — 4 위기 — Backtesting 페이지에서 통합
  7. VaR / CVaR 분포 (월별/일별 Tab + Fund vs SPY 오버레이 + 정규분포 비교)
  8. Volatility / Sortino / Beta / R² / TE Rolling 시계열 (Q-G 보강 — 5 메트릭 통합)
  9. Risk Metrics 종합 표 (~22 메트릭, 카테고리 expander, CSV 다운로드)
  10. Tail Risk 분석 (Hill estimator + Hill plot, 일별 only)
  11. Footer

모든 메트릭 = final/bl_functions.compute_metrics + 학술 표준 (decisionlog Q-E,Q-F,Q-G).

참조: docs/plan/03_pages/04_risk_metrics.md, decisionlog/04_risk_metrics.md
"""

import streamlit as st

from lib.backtesting_charts import (
    render_regime_detail_table,
    render_sub_events,
)
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
fund = load_fund_results()  # default = mat_eq_eq_raw_pap (최종 Top 1)
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
        "펀드의 위험 (변동성, 손실 폭, 시장 민감도) 을 분석합니다. "
        "사이드바에서 기간 / 비교 벤치마크 선택 가능."
    ),
)


period = st.session_state.get("period", "FULL")


# === 영역 3: Risk Summary KPI 5 ======================================
st.subheader(f"위험 KPI — {period}")
st.caption(
    "**변동성 / 최대 손실 폭 (MDD) / 시장 민감도 (β) / 시장 설명력 (R²) / 추적 오차**. "
    "위험 지표는 **낮을수록 좋습니다** — 시장 대비 차이가 음수 (녹색) 면 펀드가 더 안정적이라는 뜻입니다."
)
render_risk_kpi(fund_ret, fund_spy, ew_ret, ivw_ret, fund_rf, period)
st.divider()


# === 영역 4: Drawdown + Recovery Time =================================
st.subheader("Drawdown + Recovery Time")
st.caption(
    "펀드와 시장의 손실 폭 (drawdown) 시계열 + 가장 컸던 손실 3개 표시. "
    "**회복 시간** = 손실 후 다시 최고점에 도달하기까지 걸린 개월 수."
)
render_drawdown_recovery(fund_ret, fund_spy, period)
st.divider()


# === 영역 5: Regime 메트릭 자세한 비교 (Backtesting 페이지에서 통합) ===
st.subheader("Regime 메트릭 자세한 비교")
st.caption(
    "**시장 국면별 12개 메트릭 종합 비교**  \n"
    "- **R1 회복**: 2010년 1월 ~ 2012년 6월 (30개월, Post-GFC + EU 위기)  \n"
    "- **R2 확장**: 2012년 7월 ~ 2019년 12월 (90개월, 장기 Bull)  \n"
    "- **R3 변동**: 2020년 1월 ~ 2024년 12월 (60개월, COVID + 베어마켓 + AI 랠리)  \n"
    "- **Hold Out**: 2024년 1월 ~ 2025년 12월 (24개월)  \n"
    "- **전체**: 2010년 1월 ~ 2025년 12월 (192개월)  \n\n"
    "★ 최고 국면 / 🔴 최악 국면 강조. **전체 기간 고정** — 사이드바 기간 설정 영향 받지 않음."
)
render_regime_detail_table(fund_ret, fund_spy, fund_rf)
st.divider()


# === 영역 6: Sub-events 분석 — 4 위기 (Backtesting 페이지에서 통합) ====
st.subheader("Sub-events 분석 — 4 위기")
st.caption(
    "**4가지 시장 위기**: 2018년 4분기 급락 / COVID-19 / 2022년 인플레이션 약세장 / 2024년 섹터 로테이션. "
    "각 위기에서 펀드의 시장 대비 대응 분석. "
    "**전체 기간 고정** — 사이드바 기간 설정 영향 받지 않음."
)
render_sub_events(fund_ret, fund_spy, fund_rf)
st.divider()


# === 영역 7: VaR / CVaR 분포 ==========================================
st.subheader("VaR / CVaR 분포")
st.caption(
    "수익률 분포 + 펀드 vs 시장 비교 + 정규분포 곡선 + VaR/CVaR 표시 (**일별 데이터, ~4,000 영업일**). "
    "**VaR 5%** = 95% 확률로 손실이 이 값을 넘지 않음 (= 가장 나쁜 5% 시나리오의 경계). "
    "**CVaR 5%** = 최악 5% 시나리오의 평균 손실 (VaR 보다 보수적). "
    "월별 (192 sample) 은 5% 분위수가 단 ~10개로 통계 신뢰성 낮아 학술 표준 (일별) 만 표시합니다. "
    "ℹ️ Basel III 표준 = 일별 (시장 데이터 기반). 펀드 누적 수익률과는 별도 source."
)

with st.expander("ℹ️ 데이터 source 안내 (VaR / CVaR)"):
    st.markdown(
        """
        **왜 일별 데이터인가?**
        - Basel III, EU CRR 등 학술 / 규제 표준 = 일별 VaR / CVaR
        - 월별 (192 sample) 의 5% 분위수 = 단 9~10 개 점 → 통계 신뢰성 ↓
        - 일별 (~4,000) 의 5% 분위수 = 약 200 개 점 → 통계 신뢰성 ↑

        **일별 데이터의 source**
        - 출처: yfinance Adjusted Close 의 일별 가격 변동 (`daily_returns.pkl`)
        - 산식: `portfolio_daily_return = Σ weight_i × daily_ret_i`

        **펀드 backtest 와의 관계**
        - 펀드 CAGR / Sortino 등 (Performance KPI) = **월별 backtest 정답값** (`monthly_panel.fwd_ret_1m`)
        - 본 영역의 VaR / CVaR = **일별 시장 변동 분포** 기반
        - 두 데이터의 누적 수익률은 다를 수 있음
        - 이는 **실무 펀드 분석의 표준 패턴**: 월별 NAV (official) + 일별 risk metric (학술 표준)
        """
    )

# 일별 portfolio (영역 5, 8 공유 — 한 번 산출)
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

render_var_cvar_distribution(
    fund_ret_monthly=fund_ret,
    spy_ret_monthly=fund_spy,
    fund_ret_daily=fund_daily,
    spy_ret_daily=spy_daily,
    period=period,
)
st.divider()


# === 영역 8: Rolling 5 메트릭 (Q-G 보강) ==============================
st.subheader("Rolling 위험 메트릭 — Volatility / Sortino / Beta / R² / TE")
st.caption(
    "5개 메트릭 (변동성 / Sortino / β / R² / 추적오차) 시계열. "
    "윈도우 선택 (1년 / 3년 / 5년, **3년 권장** — 안정적 산출). "
    "**비교선**: 사이드바에서 SPY/균등가중/역변동성 토글 시 함께 표시 — "
    "**Vol/Sortino** (절대 메트릭) = 각 벤치마크의 자체 값 / "
    "**β/R²/TE** (SPY 기준 메트릭) = EW/IVW 의 SPY 대비 값 (펀드의 SPY 추적 특성과 직접 비교). "
    "β 가 음수일 때는 R² (시장 설명력) 함께 확인."
)
col_w = st.columns([1, 5])
with col_w[0]:
    rolling_window = st.selectbox(
        "Window", options=[12, 36, 60], index=1,  # 36m default
        key="risk_rolling_window", format_func=lambda x: f"{x}m",
    )

if rolling_window == 12:
    st.caption(
        "⚠️ **1년 윈도우 주의**: 기간이 짧아 일부 시점에서 Sortino 값이 끊기거나 급변할 수 있습니다. "
        "안정적 해석을 위해 **3년 사용 권장**."
    )
render_rolling_risk_metrics(fund_ret, fund_spy, ew_ret, ivw_ret, fund_rf, period, rolling_window)
st.divider()


# === 영역 9: Risk Metrics 종합 표 ====================================
st.subheader(f"위험 메트릭 종합 표 — {period}")
st.caption(
    "**월별 표준 메트릭 15개** 를 4개 카테고리로 (수익률 / 위험조정 / 위험 / 시장 노출) 정리. "
    "펼치기로 카테고리 상세 + CSV 다운로드 가능. "
    "ℹ️ **VaR / CVaR / 분포 통계 (Skewness / Kurtosis / Tail Ratio)** 는 학술 표준 (일별 데이터 필수) 이므로 "
    "**영역 7** (VaR / CVaR 분포), **영역 10** (Hill Tail) 에서 일별 기반으로 확인하실 수 있습니다."
)
render_risk_metrics_table(fund_ret, fund_spy, ew_ret, ivw_ret, fund_rf, period)
st.divider()


# === 영역 10: Tail Risk (Hill estimator) ==============================
st.subheader("Tail Risk 분석 — Hill Estimator")
st.caption(
    "수익률 분포의 꼬리 (극단값) 두께 측정 (일별 데이터). "
    "펀드의 꼬리 추정치가 시장보다 작으면 **극단적 손실 위험이 시장보다 적다** 는 의미입니다. "
    "ℹ️ Hill 은 일별 필수 (꼬리 sample 필요). 시장 데이터 기반 — 펀드 backtest 와 별도 source."
)

with st.expander("ℹ️ 데이터 source 안내 (Hill Estimator)"):
    st.markdown(
        """
        **왜 일별 데이터 필수인가?**
        - Hill estimator 는 분포 top-k 꼬리만 사용 (k ≈ 50~100)
        - 월별 (192 sample) 의 top 50 = 상위 26% → 꼬리 분석 의미 상실
        - 일별 (~4,000) 의 top 50 = 상위 1.3% → 진정한 꼬리 분석
        - **월별 데이터로는 산출 불가능** (sample 부족)

        **학술 근거**
        - Hill (1975) — *A simple general approach to inference about the tail of a distribution*
        - Embrechts, Klüppelberg, Mikosch (1997) — *Modelling Extremal Events* (EVT 표준 교과서)

        **일별 데이터의 source / 한계**
        - 출처: yfinance Adjusted Close 의 일별 변동 (`daily_returns.pkl`)
        - 펀드 backtest 의 monthly panel 과 **별도 source** (다른 학술 목적)
        - 꼬리 형상은 일별 시장 데이터 기반 → 펀드 backtest 의 실제 꼬리와 정확히 일치 X
        - 이는 **실무 펀드 분석의 표준 패턴**: 월별 NAV + 일별 EVT 분석
        """
    )

render_tail_risk(
    fund_ret_daily=fund_daily,
    spy_ret_daily=spy_daily,
)


# === 영역 11: Footer ==================================================
render_footer()
