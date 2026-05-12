"""
lib/risk_charts.py - Risk Metrics 페이지 6 영역 차트 함수

영역 3-8 함수화:
  - 영역 3: render_risk_kpi (Vol / MDD / Beta / R² / TE)
  - 영역 4: render_drawdown_recovery (DD 시계열 + Top 3 + Recovery 표)
  - 영역 5: render_var_cvar_distribution (월별/일별 Tab + 분포)
  - 영역 6: render_rolling_risk_metrics (Vol/Sortino/Beta/R²/TE rolling — Q-G 보강)
  - 영역 7: render_risk_metrics_table (~22 메트릭 카테고리 expander)
  - 영역 8: render_tail_risk (Hill estimator + plot)

모든 메트릭 = final/bl_functions.compute_metrics + 학술 표준 (decisionlog Q-E,Q-F,Q-G).

참조: docs/plan/03_pages/04_risk_metrics.md, decisionlog/04_risk_metrics.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

from lib import metric_calculators as mc
from lib.colors import BENCHMARK_COLORS, COLORS
from lib.data_loader import EVAL_PERIODS, filter_by_eval_period, filter_period
from lib.plot_helpers import add_event_annotations, add_regime_backgrounds


# ======================================================================
# 공통 유틸
# ======================================================================

def _format_pct(v: float, plus_sign: bool = False) -> str:
    if pd.isna(v):
        return "—"
    sign = "+" if plus_sign and v > 0 else ""
    return f"{sign}{v * 100:.2f}%"


def _format_ratio(v: float, decimals: int = 2) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.{decimals}f}"


def _delta_color_lower_is_better(delta: float) -> str:
    """위험 KPI 는 낮을수록 좋음 → delta < 0 이면 green."""
    if pd.isna(delta):
        return COLORS["text_muted"]
    return COLORS["accent_green"] if delta < 0 else COLORS["accent_red"]


def _delta_color_higher_is_better(delta: float) -> str:
    if pd.isna(delta):
        return COLORS["text_muted"]
    return COLORS["accent_green"] if delta > 0 else COLORS["accent_red"]


def _get_active_benchmarks(spy_ret, ew_ret, ivw_ret) -> dict[str, pd.Series]:
    out = {}
    if st.session_state.get("show_spy", True) and spy_ret is not None:
        out["SPY"] = spy_ret
    if st.session_state.get("show_ew", False) and ew_ret is not None:
        out["EW"] = ew_ret
    if st.session_state.get("show_ivw", False) and ivw_ret is not None:
        out["IVW"] = ivw_ret
    return out


def _period_to_eval_label(period: str) -> str:
    return {"FULL": "FULL", "TEST": "TEST", "HO": "HOLD_OUT"}.get(period, "FULL")


# ======================================================================
# 영역 3: Risk Summary KPI 5개
# ======================================================================

def render_risk_kpi(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    rf: pd.Series,
    period: str,
) -> None:
    """
    영역 3: Vol / MDD / Beta / R² / TE 5 카드.
    위험 KPI = 낮을수록 좋음 (delta 색상 반전: 음수=green).
    """
    eval_label = _period_to_eval_label(period)
    f = filter_by_eval_period(fund_ret, eval_label)
    benchmarks = _get_active_benchmarks(spy_ret, ew_ret, ivw_ret)

    # 펀드 메트릭
    fund_vol = mc.calc_volatility(f)
    fund_mdd = mc.calc_mdd(f)
    fund_beta = mc.calc_beta(f, filter_by_eval_period(spy_ret, eval_label), rf)
    fund_r2 = mc.calc_r_squared(f, filter_by_eval_period(spy_ret, eval_label))
    fund_te = mc.calc_tracking_error(f, filter_by_eval_period(spy_ret, eval_label))

    from lib.tooltips import get_tooltip

    # SPY = primary delta 기준, 추가 벤치마크는 caption 으로
    has_spy = "SPY" in benchmarks
    spy_bench = benchmarks.get("SPY")

    cols = st.columns(5)

    # Vol — 낮을수록 좋음 (delta_color="inverse")
    with cols[0]:
        spy_vol_delta = None
        if has_spy:
            b_vol_spy = mc.calc_volatility(filter_by_eval_period(spy_bench, eval_label))
            spy_vol_delta = fund_vol - b_vol_spy
        st.metric(
            label="Volatility",
            value=_format_pct(fund_vol),
            delta=(
                f"{_format_pct(spy_vol_delta, plus_sign=True)} vs SPY"
                if spy_vol_delta is not None else None
            ),
            delta_color="inverse",
            help=get_tooltip("Volatility") or "수익률의 변동성 (annualized)",
        )
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            b_vol = mc.calc_volatility(filter_by_eval_period(bench, eval_label))
            d = fund_vol - b_vol
            st.caption(f"vs {name} {_format_pct(d, plus_sign=True)}")

    # MDD — |MDD| 비교, 낮을수록 좋음
    with cols[1]:
        spy_mdd_delta = None
        if has_spy:
            b_mdd_spy = mc.calc_mdd(filter_by_eval_period(spy_bench, eval_label))
            spy_mdd_delta = abs(fund_mdd) - abs(b_mdd_spy)
        st.metric(
            label="MDD",
            value=_format_pct(fund_mdd),
            delta=(
                f"|Δ|{_format_pct(spy_mdd_delta, plus_sign=True)} vs SPY"
                if spy_mdd_delta is not None else None
            ),
            delta_color="inverse",
            help=get_tooltip("MDD") or "최대 낙폭 (Maximum Drawdown)",
        )
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            b_mdd = mc.calc_mdd(filter_by_eval_period(bench, eval_label))
            d = abs(fund_mdd) - abs(b_mdd)
            st.caption(f"vs {name} |Δ|{_format_pct(d, plus_sign=True)}")

    # Beta — 단순 값 (vs SPY) + caption 해설
    with cols[2]:
        st.metric(
            label="Beta",
            value=_format_ratio(fund_beta),
            delta="vs SPY",
            delta_color="off",
            help=get_tooltip("Beta") or "시장 민감도 (β=1 이면 시장과 동일)",
        )
        if not pd.isna(fund_beta):
            if fund_beta < 1:
                st.caption("시장 변동에 덜 민감 (β<1)")
            else:
                st.caption("시장과 비슷/더 민감 (β≥1)")

    # R² — 단순 값 + caption 해설
    with cols[3]:
        st.metric(
            label="R²",
            value=_format_pct(fund_r2),
            delta="vs SPY",
            delta_color="off",
            help=get_tooltip("R²") or "시장 설명력 (1.0 = 시장으로 완전 설명)",
        )
        if not pd.isna(fund_r2):
            st.caption(f"시장이 펀드 변동의 {fund_r2*100:.0f}%를 설명")
        else:
            st.caption("—")

    # TE — 단순 값 (vs SPY) + 추가 벤치마크 caption
    with cols[4]:
        st.metric(
            label="Tracking Error",
            value=_format_pct(fund_te),
            delta="vs SPY (annualized)",
            delta_color="off",
            help=get_tooltip("Tracking Error") or "벤치마크 대비 추적 오차",
        )
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            te_b = mc.calc_tracking_error(f, filter_by_eval_period(bench, eval_label))
            st.caption(f"vs {name} {_format_pct(te_b)}")


# ======================================================================
# 영역 4: Drawdown 시계열 + Recovery Time
# ======================================================================

def _drawdown_series(ret: pd.Series) -> pd.Series:
    """Drawdown 시계열 (cum / cum.cummax - 1)."""
    cum = (1 + ret.fillna(0)).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def render_drawdown_recovery(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    period: str,
) -> None:
    """
    영역 4: Drawdown 시계열 (Fund + SPY) + Top 3 마커 + Recovery 표.
    """
    eval_label = _period_to_eval_label(period)
    f_filt = filter_by_eval_period(fund_ret, eval_label)
    spy_filt = filter_by_eval_period(spy_ret, eval_label)

    fund_dd = _drawdown_series(f_filt) * 100
    spy_dd = _drawdown_series(spy_filt) * 100

    # Top 3 DD (펀드)
    top3 = mc.calc_top_n_drawdowns(f_filt, n=3)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fund_dd.index, y=fund_dd.values,
        name="Fund DD",
        line=dict(color=BENCHMARK_COLORS["Fund"], width=2),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.2)",
    ))

    fig.add_trace(go.Scatter(
        x=spy_dd.index, y=spy_dd.values,
        name="SPY DD",
        line=dict(color=BENCHMARK_COLORS["SPY"], width=1, dash="dot"),
    ))

    # 평균 DD 라인 (펀드)
    avg_dd = fund_dd[fund_dd < 0].mean() if (fund_dd < 0).any() else 0
    fig.add_hline(y=avg_dd, line_dash="dash", line_color=COLORS["text_muted"], line_width=1,
                  annotation_text=f"Avg DD {avg_dd:.1f}%", annotation_position="bottom right")

    # Top 3 마커
    for i, dd_event in enumerate(top3, 1):
        date = dd_event["trough_date"]
        depth = dd_event["depth"] * 100
        fig.add_annotation(
            x=date, y=depth,
            text=f"★ #{i}<br>{depth:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLORS["accent_red"],
            font=dict(color=COLORS["accent_red"], size=11),
            bgcolor="rgba(0,0,0,0.5)",
        )

    # Regime 배경 (FULL 만)
    if period == "FULL":
        add_regime_backgrounds(fig, with_labels=False)

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key="risk_drawdown")

    # Top 3 표
    st.markdown("##### Top 3 Drawdown 분석")
    rows = []
    for i, dd_event in enumerate(top3, 1):
        rec = (
            f"{dd_event['recovery_months']}m"
            if dd_event["recovery_months"] is not None
            else "(진행 중)"
        )
        rows.append({
            "Rank": f"#{i}",
            "Trough Date": dd_event["trough_date"].strftime("%Y-%m"),
            "Depth": f"{dd_event['depth']*100:.2f}%",
            "Duration (peak→trough)": f"{dd_event['duration']}m",
            "Recovery (trough→new peak)": rec,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 평균 Recovery
    avg_rec = mc.calc_avg_recovery_time(f_filt)
    if not pd.isna(avg_rec):
        st.caption(f"※ 평균 Recovery Time (완료된 모든 DD): **{avg_rec:.1f} 개월**")


# ======================================================================
# 영역 5: VaR / CVaR 분포
# ======================================================================

def render_var_cvar_distribution(
    fund_ret_monthly: pd.Series,    # noqa: ARG001 (시그니처 호환성)
    spy_ret_monthly: pd.Series,     # noqa: ARG001
    fund_ret_daily: pd.Series | None,
    spy_ret_daily: pd.Series | None,
    period: str,
) -> None:
    """
    영역 7: VaR / CVaR 분포 (히스토그램 + KDE + 임계선 + 정규분포 비교) — **일별 only** (2026-05-11 변경).

    이유:
      - 월별 (192 sample) 의 5% 분위수는 단 ~9.6개 → 통계적 신뢰성 매우 낮음
      - 학술/실무 VaR 표준 = 일별 (Basel III, J.P. Morgan RiskMetrics)
      - 분포 통계 (영역 9 Performance) 와 일관성 — 꼬리/극단 메트릭은 모두 일별
      - 사이드바 기간 토글 (TEST / Hold Out / FULL) 은 일별에서도 동일 작동

    monthly 매개변수는 시그니처 호환성 유지 (호출부 변경 회피).
    """
    eval_label = _period_to_eval_label(period)

    if fund_ret_daily is None or len(fund_ret_daily.dropna()) == 0:
        st.info("일별 데이터 미산출")
        return

    s_dt, e_dt = EVAL_PERIODS[eval_label]
    f_d = fund_ret_daily[(fund_ret_daily.index >= s_dt) & (fund_ret_daily.index <= e_dt)]
    s_d = (
        spy_ret_daily[(spy_ret_daily.index >= s_dt) & (spy_ret_daily.index <= e_dt)]
        if spy_ret_daily is not None else None
    )
    _render_distribution_chart(f_d, s_d, label_unit="day", key_suffix="daily")


def _render_distribution_chart(
    fund_ret: pd.Series, spy_ret: pd.Series | None, label_unit: str, key_suffix: str
) -> None:
    """단일 분포 차트 (Fund + SPY 오버레이 히스토그램 + 임계선)."""
    if fund_ret is None or len(fund_ret.dropna()) == 0:
        st.info("데이터 없음")
        return

    # 메트릭 산출
    f_var = mc.calc_var(fund_ret, alpha=0.05)
    f_cvar = mc.calc_cvar(fund_ret, alpha=0.05)

    fig = go.Figure()

    # Fund 히스토그램
    fig.add_trace(go.Histogram(
        x=fund_ret.dropna() * 100,
        name="Fund",
        marker_color=BENCHMARK_COLORS["Fund"],
        opacity=0.6,
        nbinsx=40,
        histnorm="probability density",
    ))

    # SPY 히스토그램
    if spy_ret is not None and len(spy_ret.dropna()) > 0 and st.session_state.get("show_spy", True):
        fig.add_trace(go.Histogram(
            x=spy_ret.dropna() * 100,
            name="SPY",
            marker_color=BENCHMARK_COLORS["SPY"],
            opacity=0.4,
            nbinsx=40,
            histnorm="probability density",
        ))

    # 정규분포 비교 라인 (Fund 기준)
    f_clean = fund_ret.dropna() * 100
    if len(f_clean) > 0:
        x_norm = np.linspace(f_clean.min(), f_clean.max(), 100)
        y_norm = scipy_stats.norm.pdf(x_norm, f_clean.mean(), f_clean.std())
        fig.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            name="Normal (Fund μ,σ)",
            line=dict(color=COLORS["accent_amber"], width=2, dash="dot"),
        ))

    # VaR / CVaR 수직선
    fig.add_vline(
        x=f_var * 100,
        line_dash="dash",
        line_color=COLORS["accent_red"],
        annotation_text=f"VaR 5% {f_var*100:.2f}%",
        annotation_position="top",
    )
    fig.add_vline(
        x=f_cvar * 100,
        line_dash="solid",
        line_color=COLORS["accent_red"],
        line_width=2,
        annotation_text=f"CVaR 5% {f_cvar*100:.2f}%",
        annotation_position="bottom",
    )

    # x축 동적 range — 0.5%/99.5% 분위수 기반 (99% 데이터 포함, 극단 outlier 제외)
    # 극단값까지 표시하면 중앙 분포가 압축되어 보이는 문제 해결.
    all_data = [f_clean]
    if spy_ret is not None and len(spy_ret.dropna()) > 0:
        all_data.append(spy_ret.dropna() * 100)
    combined = pd.concat(all_data)
    q_low = combined.quantile(0.005)
    q_high = combined.quantile(0.995)
    x_max = max(abs(q_low), abs(q_high)) * 1.15

    fig.update_layout(
        height=380,
        barmode="overlay",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(title=f"Return (%, {label_unit})", range=[-x_max, x_max]),
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"risk_distribution_{key_suffix}")

    # Statistics 보조 표
    rows = []
    rows.append({
        "Statistic": "Mean", "Fund": f"{f_clean.mean():.4f}%",
        "SPY": f"{(spy_ret.dropna()*100).mean():.4f}%" if spy_ret is not None and len(spy_ret.dropna()) > 0 else "—",
    })
    rows.append({
        "Statistic": "Std", "Fund": f"{f_clean.std():.4f}%",
        "SPY": f"{(spy_ret.dropna()*100).std():.4f}%" if spy_ret is not None and len(spy_ret.dropna()) > 0 else "—",
    })
    rows.append({
        "Statistic": "VaR 5%", "Fund": f"{f_var*100:.4f}%",
        "SPY": f"{mc.calc_var(spy_ret, 0.05)*100:.4f}%" if spy_ret is not None and len(spy_ret.dropna()) > 0 else "—",
    })
    rows.append({
        "Statistic": "CVaR 5%", "Fund": f"{f_cvar*100:.4f}%",
        "SPY": f"{mc.calc_cvar(spy_ret, 0.05)*100:.4f}%" if spy_ret is not None and len(spy_ret.dropna()) > 0 else "—",
    })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ======================================================================
# 영역 6: Rolling 5 메트릭 (Volatility / Sortino / Beta / R² / TE)
# ======================================================================

def render_rolling_risk_metrics(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    rf: pd.Series,
    period: str,
    window: int,
) -> None:
    """
    영역 8: 5 메트릭 통합 rolling 시계열.
    Volatility / Sortino / Beta / R² / Tracking Error.

    벤치마크 비교선 (2026-05-11 보강):
      - Vol / Sortino: 절대 메트릭 → SPY/EW/IVW 자체 rolling 값을 비교선으로
      - Beta / R² / TE: SPY 기준 메트릭 → EW/IVW 의 SPY 대비 rolling 값을 비교선으로
        (펀드의 SPY 추적 특성을 EW/IVW 와 직접 비교)
    """
    eval_label = _period_to_eval_label(period)
    active_benchmarks = _get_active_benchmarks(spy_ret, ew_ret, ivw_ret)

    # Fund 메트릭 (5개)
    rv = filter_by_eval_period(mc.calc_rolling_volatility(fund_ret, window), eval_label)
    rsor = filter_by_eval_period(mc.calc_rolling_sortino(fund_ret, rf, window), eval_label)
    rb = filter_by_eval_period(mc.calc_rolling_beta(fund_ret, spy_ret, rf, window), eval_label)
    rr2 = filter_by_eval_period(mc.calc_rolling_r_squared(fund_ret, spy_ret, window), eval_label)
    rte = filter_by_eval_period(mc.calc_rolling_tracking_error(fund_ret, spy_ret, window), eval_label)

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Volatility (annualized)",
            "Sortino (sub<0 std)",
            "Beta vs SPY",
            "R² vs SPY",
            "Tracking Error (annualized)",
        ),
    )

    # ── Fund 메인 라인 (5 메트릭) — 메트릭별 다른 색상 (시각 구분) ───
    fund_metric_colors = {
        "vol": BENCHMARK_COLORS["Fund"],   # Cobalt Blue
        "sortino": COLORS["accent_green"], # 초록
        "beta": COLORS["accent_amber"],    # 주황
        "r2": BENCHMARK_COLORS["IVW"],     # 보라/IVW 색
        "te": COLORS["accent_red"],        # 빨강
    }
    fig.add_trace(go.Scatter(x=rv.index, y=rv.values * 100, name="Fund Vol",
                             line=dict(color=fund_metric_colors["vol"], width=2),
                             legendgroup="vol", legendgrouptitle_text="Volatility"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=rsor.index, y=rsor.values, name="Fund Sortino",
                             line=dict(color=fund_metric_colors["sortino"], width=2),
                             legendgroup="sortino", legendgrouptitle_text="Sortino"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=rb.index, y=rb.values, name="Fund β vs SPY",
                             line=dict(color=fund_metric_colors["beta"], width=2),
                             legendgroup="beta", legendgrouptitle_text="Beta vs SPY"),
                  row=3, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color=COLORS["text_muted"], line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(x=rr2.index, y=rr2.values * 100, name="Fund R² vs SPY",
                             line=dict(color=fund_metric_colors["r2"], width=2),
                             legendgroup="r2", legendgrouptitle_text="R² vs SPY"),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=rte.index, y=rte.values * 100, name="Fund TE vs SPY",
                             line=dict(color=fund_metric_colors["te"], width=2),
                             legendgroup="te", legendgrouptitle_text="TE vs SPY"),
                  row=5, col=1)

    # ── 벤치마크 비교선 — 점선 + 벤치마크별 색상 (Fund 의 실선과 구분) ─
    # Fund (메트릭별 실선) vs 벤치마크 (벤치마크별 점선) 시각 위계 분리.
    dash_map = {"SPY": "dot", "EW": "dash", "IVW": "dashdot"}
    for name, bench in active_benchmarks.items():
        color = BENCHMARK_COLORS.get(name, COLORS["text_muted"])
        dash = dash_map.get(name, "dot")

        # Vol (모든 벤치마크 — 절대 메트릭)
        bv = filter_by_eval_period(mc.calc_rolling_volatility(bench, window), eval_label)
        fig.add_trace(go.Scatter(x=bv.index, y=bv.values * 100, name=f"{name} Vol",
                                 line=dict(color=color, width=1.5, dash=dash),
                                 legendgroup="vol"),
                      row=1, col=1)

        # Sortino (모든 벤치마크 — 절대 메트릭)
        bsor = filter_by_eval_period(mc.calc_rolling_sortino(bench, rf, window), eval_label)
        fig.add_trace(go.Scatter(x=bsor.index, y=bsor.values, name=f"{name} Sortino",
                                 line=dict(color=color, width=1.5, dash=dash),
                                 legendgroup="sortino"),
                      row=2, col=1)

        # Beta / R² / TE — SPY 자체는 의미 없음 (β vs SPY = 1, R² = 100%, TE = 0)
        # 따라서 EW / IVW 만 추가 (SPY 대비)
        if name == "SPY":
            continue

        bb = filter_by_eval_period(mc.calc_rolling_beta(bench, spy_ret, rf, window), eval_label)
        fig.add_trace(go.Scatter(x=bb.index, y=bb.values, name=f"{name} β vs SPY",
                                 line=dict(color=color, width=1.5, dash=dash),
                                 legendgroup="beta"),
                      row=3, col=1)

        br2 = filter_by_eval_period(mc.calc_rolling_r_squared(bench, spy_ret, window), eval_label)
        fig.add_trace(go.Scatter(x=br2.index, y=br2.values * 100, name=f"{name} R² vs SPY",
                                 line=dict(color=color, width=1.5, dash=dash),
                                 legendgroup="r2"),
                      row=4, col=1)

        bte = filter_by_eval_period(mc.calc_rolling_tracking_error(bench, spy_ret, window), eval_label)
        fig.add_trace(go.Scatter(x=bte.index, y=bte.values * 100, name=f"{name} TE vs SPY",
                                 line=dict(color=color, width=1.5, dash=dash),
                                 legendgroup="te"),
                      row=5, col=1)

    fig.update_layout(
        height=950,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"], size=11),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.02,
            groupclick="toggleitem",
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Vol (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sortino", row=2, col=1)
    fig.update_yaxes(title_text="Beta", row=3, col=1)
    fig.update_yaxes(title_text="R² (%)", row=4, col=1)
    fig.update_yaxes(title_text="TE (%)", row=5, col=1)

    st.plotly_chart(fig, use_container_width=True, key="risk_rolling_5metrics")


# ======================================================================
# 영역 7: Risk Metrics 종합 표 (~22 메트릭, 카테고리 expander)
# ======================================================================

_TABLE_CATEGORIES = {
    "Performance Returns": [
        ("Cumulative Return", "cum_ret"),
        ("CAGR", "cagr"),
        ("Arithmetic Mean", "arith_mean"),
    ],
    "Risk-adjusted Return": [
        ("Sharpe", "sharpe"),
        ("Sortino", "sortino"),
        ("Calmar", "calmar"),
        ("IR (vs SPY)", "ir"),
    ],
    "Risk": [
        ("Volatility", "vol"),
        ("MDD", "mdd"),
        ("Downside Dev", "down_dev"),
        # VaR/CVaR 는 일별 학술 표준 (Basel III) → 영역 7 에서 일별 기반 산출
    ],
    "Market Exposure": [
        ("Beta", "beta"),
        ("R²", "r2"),
        ("Correlation", "corr"),
        ("Tracking Error", "te"),
        ("Alpha (CAPM)", "alpha"),
    ],
    # Distribution (Skew/Kurt/Tail Ratio) 는 일별 학술 표준 (sample size 필수)
    # → 영역 7 (Performance 영역 9 와 정합) 에서 일별 기반 산출
}


def _calc_all_metrics_for_series(ret: pd.Series, mkt: pd.Series, rf: pd.Series) -> dict:
    """
    단일 시리즈 (Fund or 벤치마크) 의 종합 표 메트릭 산출 (월별 표준만).

    제외 메트릭 (학술 표준 = 일별 → 영역 7, 10 에서 별도 산출):
      - VaR 5%, CVaR 5% (Basel III 표준 = 일별)
      - Skewness, Excess Kurtosis, Tail Ratio (Cont 2001 — sample size 필요, 일별)
      - Hill Tail Index (Hill 1975 — 꼬리 sample 필요, 일별)
    """
    return {
        "cum_ret": mc.calc_cumulative_return(ret),
        "cagr": mc.calc_cagr(ret),
        "arith_mean": mc.calc_arithmetic_mean(ret),
        "sharpe": mc.calc_sharpe(ret, rf),
        "sortino": mc.calc_sortino(ret, rf),
        "calmar": mc.calc_calmar(ret),
        "ir": mc.calc_ir(ret, mkt) if mkt is not None else np.nan,
        "vol": mc.calc_volatility(ret),
        "mdd": mc.calc_mdd(ret),
        "down_dev": mc.calc_downside_deviation(ret),
        "beta": mc.calc_beta(ret, mkt, rf) if mkt is not None else np.nan,
        "r2": mc.calc_r_squared(ret, mkt) if mkt is not None else np.nan,
        "corr": mc.calc_correlation(ret, mkt) if mkt is not None else np.nan,
        "te": mc.calc_tracking_error(ret, mkt) if mkt is not None else np.nan,
        "alpha": mc.calc_alpha(ret, mkt, rf) if mkt is not None else np.nan,
    }


def _format_metric(key: str, v: float) -> str:
    """메트릭별 포맷."""
    if pd.isna(v):
        return "—"
    pct_keys = {"cum_ret", "cagr", "arith_mean", "vol", "mdd", "down_dev",
                "alpha", "te"}
    if key in pct_keys:
        return f"{v*100:+.2f}%"
    if key == "r2":
        return f"{v*100:.1f}%"
    return f"{v:+.3f}" if v < 0 else f"{v:.3f}"


def render_risk_metrics_table(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    rf: pd.Series,
    period: str,
) -> None:
    """
    영역 7: ~20 메트릭 종합 표 (카테고리 expander, CSV 다운로드).
    """
    eval_label = _period_to_eval_label(period)
    f = filter_by_eval_period(fund_ret, eval_label)
    s = filter_by_eval_period(spy_ret, eval_label)

    fund_metrics = _calc_all_metrics_for_series(f, s, rf)

    benchmarks = _get_active_benchmarks(spy_ret, ew_ret, ivw_ret)

    # 표 구조 (Fund + 활성 벤치마크 + Diff)
    bench_metrics = {
        name: _calc_all_metrics_for_series(filter_by_eval_period(b, eval_label),
                                           filter_by_eval_period(spy_ret, eval_label), rf)
        for name, b in benchmarks.items()
    }

    # 전체 표 생성 (CSV 다운로드용)
    rows_for_csv = []

    for cat, items in _TABLE_CATEGORIES.items():
        with st.expander(f"▶ {cat}", expanded=(cat == "Performance Returns")):
            display_rows = []
            for label, key in items:
                row = {"Metric": label, "Fund": _format_metric(key, fund_metrics[key])}
                csv_row = {"Category": cat, "Metric": label, "Fund": fund_metrics[key]}
                for bn, bm in bench_metrics.items():
                    row[bn] = _format_metric(key, bm[key])
                    csv_row[bn] = bm[key]
                    if not pd.isna(fund_metrics[key]) and not pd.isna(bm[key]):
                        diff = fund_metrics[key] - bm[key]
                        row[f"Δ {bn}"] = _format_metric(key, diff)
                        csv_row[f"Diff_{bn}"] = diff
                display_rows.append(row)
                rows_for_csv.append(csv_row)
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    # CSV 다운로드
    csv_df = pd.DataFrame(rows_for_csv)
    st.download_button(
        label="⬇ CSV 다운로드 (전체 메트릭 표)",
        data=csv_df.to_csv(index=False, float_format="%.6f").encode("utf-8-sig"),
        file_name=f"risk_metrics_{period}.csv",
        mime="text/csv",
    )


# ======================================================================
# 영역 8: Tail Risk (Hill estimator)
# ======================================================================

def render_tail_risk(
    fund_ret_daily: pd.Series | None,
    spy_ret_daily: pd.Series | None,
) -> None:
    """
    영역 8: Hill estimator + Hill plot (일별 only).
    학술: Hill (1975) — fat tail 측정.
    """
    if fund_ret_daily is None or len(fund_ret_daily.dropna()) < 50:
        st.warning("일별 데이터 미산출 또는 부족. Tail Risk 분석 불가.")
        return

    # 학술 설명 박스
    st.info(
        "**Hill estimator (Hill 1975)** — 분포 tail 의 두께 측정.\n\n"
        "ξ̂ > 0 = fat tail (극단 손실 자주 발생). 주식 일반 ξ̂ ≈ 0.2-0.4.\n"
        "펀드 ξ̂ 가 SPY 보다 낮으면 tail risk 가 적음."
    )

    fund_xi = mc.calc_hill_estimator(fund_ret_daily, side="loss")
    spy_xi = (
        mc.calc_hill_estimator(spy_ret_daily, side="loss")
        if spy_ret_daily is not None else np.nan
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f'<div style="border-left:3px solid {BENCHMARK_COLORS["Fund"]};padding:8px 12px;">'
            f'<div style="color:#9CA3AF;font-size:13px;">Fund ξ̂ (loss tail)</div>'
            f'<div style="font-size:24px;color:#FAFAFA;font-weight:700;">{fund_xi:.4f}</div>'
            f'<div style="font-size:11px;color:#9CA3AF;">{"Fat tail" if fund_xi > 0.1 else "Light tail"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div style="border-left:3px solid {BENCHMARK_COLORS["SPY"]};padding:8px 12px;">'
            f'<div style="color:#9CA3AF;font-size:13px;">SPY ξ̂ (loss tail)</div>'
            f'<div style="font-size:24px;color:#FAFAFA;font-weight:700;">{_format_ratio(spy_xi, 4)}</div>'
            f'<div style="font-size:11px;color:#9CA3AF;">'
            f'{("Fat tail" if spy_xi > 0.1 else "Light tail") if not pd.isna(spy_xi) else ""}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Hill Plot (k 별 ξ̂)
    st.markdown("##### Hill Plot — k 별 ξ̂ (plateau 영역 = 안정 추정)")
    fig = go.Figure()

    fund_hill = mc.hill_plot_data(fund_ret_daily, side="loss")
    if len(fund_hill) > 0:
        fig.add_trace(go.Scatter(
            x=fund_hill["k"], y=fund_hill["xi"],
            name="Fund",
            line=dict(color=BENCHMARK_COLORS["Fund"], width=2),
        ))

    if spy_ret_daily is not None:
        spy_hill = mc.hill_plot_data(spy_ret_daily, side="loss")
        if len(spy_hill) > 0:
            fig.add_trace(go.Scatter(
                x=spy_hill["k"], y=spy_hill["xi"],
                name="SPY",
                line=dict(color=BENCHMARK_COLORS["SPY"], width=1.5, dash="dot"),
            ))

    # Auto k = sqrt(n) 마커
    n_loss = len(fund_ret_daily[fund_ret_daily < 0])
    auto_k = int(np.sqrt(n_loss))
    fig.add_vline(
        x=auto_k,
        line_dash="dash",
        line_color=COLORS["accent_amber"],
        annotation_text=f"auto k=√n={auto_k}",
        annotation_position="top",
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        xaxis_title="k (tail order)",
        yaxis_title="ξ̂ (Hill estimator)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key="risk_hill_plot")

    # Footnote
    st.caption(
        "※ Hill, B.M. (1975). \"A simple general approach to inference about the tail "
        "of a distribution.\" Annals of Statistics, 3, 1163-1174. / "
        "Embrechts, Klüppelberg, Mikosch (1997) \"Modelling Extremal Events.\"\n\n"
        "※ **차트 해석**: 단조 증가 패턴은 **plateau 가 명확하지 않은 fat tail 분포** 의 "
        "일반적 특성 (Mandelbrot 1963 — 금융 시계열은 정확히 power-law 가 아닌 complex tail). "
        "auto k = √n 영역 (작은 k) 의 ξ̂ 가 안정 추정값."
    )
