"""
lib/performance_charts.py - Performance 페이지 6 영역 차트 함수

영역 3-8 모두 함수화:
  - 영역 3: render_performance_kpi (CAGR / Sortino / Sharpe / IR / Active Return)
  - 영역 4: render_annual_returns (사이드바이 막대 + 평균선 + Q-Zoom)
  - 영역 5: render_active_return_analysis (위 막대 + 아래 이중 축 Rolling)
  - 영역 6: render_rolling_return (1y/3y/5y 다중 + Regime 배경)
  - 영역 7: render_regime_heatmap (Tab 전환 + diverging colormap)
  - 영역 8: render_distribution_stats (일별/월별 + vs Tab + KDE mini)

모든 메트릭은 final/bl_functions + master_table 정확 재현 (decisionlog Q-E).

참조: docs/plan/03_pages/03_performance.md, decisionlog/03_performance.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import metric_calculators as mc
from lib.colors import BENCHMARK_COLORS, COLORS, REGIME_COLORS
from lib.data_loader import (
    EVAL_PERIODS,
    REGIMES,
    filter_by_eval_period,
    filter_by_regime,
    filter_period,
)
from lib.plot_helpers import add_event_annotations, add_regime_backgrounds


# ======================================================================
# 공통 유틸
# ======================================================================

def _format_pct(v: float, plus_sign: bool = False) -> str:
    if pd.isna(v):
        return "—"
    sign = "+" if plus_sign and v > 0 else ""
    return f"{sign}{v * 100:.2f}%"


def _format_ratio(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.2f}"


def _delta_color(delta: float) -> str:
    """delta 부호에 따른 색상."""
    if pd.isna(delta):
        return COLORS["text_muted"]
    return COLORS["accent_green"] if delta > 0 else COLORS["accent_red"]


def _get_active_benchmarks(spy_ret, ew_ret, ivw_ret) -> dict[str, pd.Series]:
    """사이드바 토글 활성 벤치마크 dict 반환 (None 제외)."""
    out = {}
    if st.session_state.get("show_spy", True) and spy_ret is not None:
        out["SPY"] = spy_ret
    if st.session_state.get("show_ew", False) and ew_ret is not None:
        out["EW"] = ew_ret
    if st.session_state.get("show_ivw", False) and ivw_ret is not None:
        out["IVW"] = ivw_ret
    return out


def _period_to_eval_label(period: str) -> str:
    """사이드바 period (FULL/TEST/HO) → EVAL_PERIODS 키."""
    return {"FULL": "FULL", "TEST": "TEST", "HO": "HOLD_OUT"}.get(period, "FULL")


# ======================================================================
# 영역 3: Performance Summary KPI 5개
# ======================================================================

def render_performance_kpi(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    rf: pd.Series,
    period: str,
    fund_gross_ret: pd.Series | None = None,
) -> None:
    """
    영역 3: KPI 5 카드 — CAGR / Sortino / Sharpe / IR / Active Return.

    final master_table 정합 (subperiod 함수 사용).
    벤치마크 delta 행: 활성 벤치마크 (사이드바 토글) 마다 추가.

    fund_gross_ret 전달 시 CAGR 카드에 Gross + TC 누적 추가 표시 (B-1).

    period: "FULL" / "TEST" / "HO"
    """
    eval_label = _period_to_eval_label(period)
    s, e = EVAL_PERIODS[eval_label]

    # 펀드 메트릭 (final 정합)
    cagr = mc.calc_cagr_subperiod(fund_ret, s, e)
    sortino = mc.calc_sortino_subperiod(fund_ret, rf, s, e)
    sharpe = mc.calc_sharpe_subperiod(fund_ret, rf, s, e)
    cagr_gross = (
        mc.calc_cagr_subperiod(fund_gross_ret, s, e)
        if fund_gross_ret is not None else None
    )

    # 벤치마크 활성 (delta 표시용)
    benchmarks = _get_active_benchmarks(spy_ret, ew_ret, ivw_ret)

    # IR / Active Return 은 SPY 기준 (1차 표시) — 다른 벤치마크는 카드 안 추가행
    ir_spy = mc.calc_ir(filter_by_eval_period(fund_ret, eval_label),
                        filter_by_eval_period(spy_ret, eval_label)) if spy_ret is not None else np.nan
    active_spy = mc.calc_active_return(filter_by_eval_period(fund_ret, eval_label),
                                       filter_by_eval_period(spy_ret, eval_label)) if spy_ret is not None else np.nan

    from lib.tooltips import get_tooltip

    def _label(text: str, tip_key: str) -> str:
        """카드 라벨 + tooltip ⓘ HTML."""
        tip = get_tooltip(tip_key) or ""
        if not tip:
            return f"**{text}**"
        return (
            f'<div style="font-weight:700;">{text}'
            f'<span title="{tip}" style="cursor:help;color:#9CA3AF;font-size:11px;'
            f'margin-left:6px;border:1px solid #374151;border-radius:50%;'
            f'width:14px;height:14px;display:inline-flex;align-items:center;'
            f'justify-content:center;">ⓘ</span></div>'
        )

    cols = st.columns(5)

    # 카드 1: CAGR (Net + Gross + TC 누적)
    with cols[0]:
        st.markdown(_label("CAGR", "Net CAGR"), unsafe_allow_html=True)
        st.markdown(f"## {_format_pct(cagr, plus_sign=True)}")
        for name, bench in benchmarks.items():
            bench_cagr = mc.calc_cagr_subperiod(bench, s, e)
            delta = cagr - bench_cagr
            color = _delta_color(delta)
            st.markdown(
                f'<div style="font-size:12px;color:{color};">'
                f'vs {name} {_format_pct(delta, plus_sign=True)}'
                f'</div>',
                unsafe_allow_html=True,
            )
        # B-1: Gross + TC 누적 (작은 회색 글씨)
        if cagr_gross is not None and not pd.isna(cagr_gross):
            tc_diff = cagr_gross - cagr
            st.markdown(
                f'<div style="font-size:11px;color:{COLORS["text_muted"]};margin-top:6px;'
                f'border-top:1px solid #374151;padding-top:4px;">'
                f'Gross {_format_pct(cagr_gross, plus_sign=True)}<br>'
                f'TC -{_format_pct(tc_diff)} (One-way 20bp)'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 카드 2: Sortino
    with cols[1]:
        st.markdown(_label("Sortino", "Sortino"), unsafe_allow_html=True)
        st.markdown(f"## {_format_ratio(sortino)}")
        for name, bench in benchmarks.items():
            bench_sortino = mc.calc_sortino_subperiod(bench, rf, s, e)
            delta = sortino - bench_sortino
            color = _delta_color(delta)
            st.markdown(
                f'<div style="font-size:12px;color:{color};">'
                f'vs {name} {("+" if delta > 0 else "")}{delta:.2f}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 카드 3: Sharpe
    with cols[2]:
        st.markdown(_label("Sharpe", "Sharpe"), unsafe_allow_html=True)
        st.markdown(f"## {_format_ratio(sharpe)}")
        for name, bench in benchmarks.items():
            bench_sharpe = mc.calc_sharpe_subperiod(bench, rf, s, e)
            delta = sharpe - bench_sharpe
            color = _delta_color(delta)
            st.markdown(
                f'<div style="font-size:12px;color:{color};">'
                f'vs {name} {("+" if delta > 0 else "")}{delta:.2f}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 카드 4: IR (active vs SPY 기준 — 다른 벤치마크는 추가행)
    with cols[3]:
        st.markdown(_label("IR", "IR"), unsafe_allow_html=True)
        st.markdown(f"## {_format_ratio(ir_spy)}")
        st.caption("vs SPY")
        # 다른 벤치마크 활성 시 추가 IR
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            ir_b = mc.calc_ir(filter_by_eval_period(fund_ret, eval_label),
                              filter_by_eval_period(bench, eval_label))
            st.markdown(
                f'<div style="font-size:12px;color:{COLORS["text_muted"]};">'
                f'vs {name} {_format_ratio(ir_b)}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 카드 5: Active Return (vs SPY 기준)
    with cols[4]:
        st.markdown(_label("Active Return", "Active Return"), unsafe_allow_html=True)
        st.markdown(f"## {_format_pct(active_spy, plus_sign=True)}")
        st.caption("vs SPY (annualized)")
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            active_b = mc.calc_active_return(filter_by_eval_period(fund_ret, eval_label),
                                             filter_by_eval_period(bench, eval_label))
            st.markdown(
                f'<div style="font-size:12px;color:{COLORS["text_muted"]};">'
                f'vs {name} {_format_pct(active_b, plus_sign=True)}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ======================================================================
# 영역 4: Annual Returns 막대
# ======================================================================

def _annual_returns(monthly_ret: pd.Series) -> pd.Series:
    """월별 → 연도별 컴파운딩 수익률."""
    if len(monthly_ret) == 0:
        return pd.Series(dtype=float)
    return (1 + monthly_ret.fillna(0)).resample("YE").prod() - 1


def render_annual_returns(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    period: str,
) -> None:
    """
    영역 4: Annual Returns 사이드바이 막대 + 평균선.
    """
    eval_label = _period_to_eval_label(period)
    f_filt = filter_by_eval_period(fund_ret, eval_label)
    fund_annual = _annual_returns(f_filt)

    fig = go.Figure()

    # Fund 막대
    fig.add_trace(go.Bar(
        x=fund_annual.index.year,
        y=fund_annual.values * 100,
        name="Fund",
        marker_color=BENCHMARK_COLORS["Fund"],
    ))
    # Fund 평균선
    fig.add_hline(
        y=fund_annual.mean() * 100,
        line_dash="dash",
        line_color=BENCHMARK_COLORS["Fund"],
        annotation_text=f"Avg Fund {fund_annual.mean()*100:+.1f}%",
        annotation_position="top right",
    )

    # 활성 벤치마크 막대
    benchmarks = _get_active_benchmarks(spy_ret, ew_ret, ivw_ret)
    for name, bench in benchmarks.items():
        b_filt = filter_by_eval_period(bench, eval_label)
        b_annual = _annual_returns(b_filt)
        fig.add_trace(go.Bar(
            x=b_annual.index.year,
            y=b_annual.values * 100,
            name=name,
            marker_color=BENCHMARK_COLORS.get(name, COLORS["text_muted"]),
        ))

    # 0% 기준선
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], line_width=1)

    fig.update_layout(
        height=400,
        barmode="group",
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Annual Return (%)",
        xaxis_title="Year",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key="perf_annual_returns")


# ======================================================================
# 영역 5: Active Return 분석 (위아래 두 차트)
# ======================================================================

def render_active_return_analysis(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    period: str,
    rolling_window: int,
) -> None:
    """
    영역 5: 위 = 연별 Active Return 막대, 아래 = Rolling Active Return + Tracking Error 이중 축.
    """
    eval_label = _period_to_eval_label(period)
    benchmarks = _get_active_benchmarks(spy_ret, ew_ret, ivw_ret)

    # ── 위: 연별 Active Return 막대 ─────────────────────────────
    st.markdown("##### 연별 Active Return (Fund - Benchmark)")
    fig_top = go.Figure()
    f_filt = filter_by_eval_period(fund_ret, eval_label)
    fund_annual = _annual_returns(f_filt)

    for name, bench in benchmarks.items():
        b_filt = filter_by_eval_period(bench, eval_label)
        b_annual = _annual_returns(b_filt)
        active_annual = (fund_annual - b_annual) * 100
        fig_top.add_trace(go.Bar(
            x=active_annual.index.year,
            y=active_annual.values,
            name=f"vs {name}",
            marker_color=[
                COLORS["accent_green"] if v > 0 else COLORS["accent_red"]
                for v in active_annual.values
            ] if name == "SPY" else None,
            marker={"color": BENCHMARK_COLORS.get(name, COLORS["text_muted"])} if name != "SPY" else None,
        ))

    fig_top.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], line_width=1)
    fig_top.update_layout(
        height=300,
        barmode="group",
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        yaxis_title="Active Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_top, use_container_width=True, key="perf_active_top")

    # ── 아래: Rolling Active + Tracking Error 이중 축 ─────────────
    st.markdown(f"##### Rolling Active Return + Tracking Error ({rolling_window}m window)")
    fig_bot = make_subplots(specs=[[{"secondary_y": True}]])

    for name, bench in benchmarks.items():
        # 펀드와 벤치마크 정렬
        df = pd.DataFrame({"f": fund_ret, "b": bench}).dropna()
        active = df["f"] - df["b"]
        # Annualized rolling active = 12 × rolling mean(active)
        rolling_active = active.rolling(rolling_window).mean() * 12 * 100
        # Annualized tracking error
        rolling_te = active.rolling(rolling_window).std() * np.sqrt(12) * 100

        # 기간 필터
        rolling_active = filter_by_eval_period(rolling_active, eval_label)
        rolling_te = filter_by_eval_period(rolling_te, eval_label)

        # 좌축: Active
        fig_bot.add_trace(
            go.Scatter(
                x=rolling_active.index, y=rolling_active.values,
                name=f"Active vs {name}",
                line=dict(color=BENCHMARK_COLORS.get(name, COLORS["text_muted"]), width=2),
            ),
            secondary_y=False,
        )
        # 우축: TE (점선)
        fig_bot.add_trace(
            go.Scatter(
                x=rolling_te.index, y=rolling_te.values,
                name=f"TE vs {name}",
                line=dict(color=BENCHMARK_COLORS.get(name, COLORS["text_muted"]),
                          width=1, dash="dot"),
            ),
            secondary_y=True,
        )

    fig_bot.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], line_width=1, secondary_y=False)
    fig_bot.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig_bot.update_yaxes(title_text="Active Return (%, ann.)", secondary_y=False)
    fig_bot.update_yaxes(title_text="Tracking Error (%, ann.)", secondary_y=True)

    st.plotly_chart(fig_bot, use_container_width=True, key="perf_active_bot")


# ======================================================================
# 영역 6: Annualized Rolling Return (1y / 3y / 5y)
# ======================================================================

def _rolling_annualized_return(monthly_ret: pd.Series, window_years: int) -> pd.Series:
    """월별 → window_years 년 annualized rolling return (월 단위 rolling)."""
    n = window_years * 12
    if len(monthly_ret) < n:
        return pd.Series(dtype=float)
    # rolling(n).apply(prod) 가 무거움 → log-return 합 후 exp
    log_r = np.log1p(monthly_ret.fillna(0))
    rolling_log_sum = log_r.rolling(n).sum()
    annualized = (np.exp(rolling_log_sum) ** (1 / window_years) - 1)
    return annualized


def render_rolling_return(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    period: str,
    windows: list[int],
) -> None:
    """
    영역 6: 1y/3y/5y 다중 윈도우 + 활성 벤치마크 + Regime 배경.

    windows: [1, 3, 5] 중 활성 윈도우 list
    """
    if not windows:
        st.info("최소 한 개의 rolling 윈도우를 선택하세요.")
        return

    eval_label = _period_to_eval_label(period)
    benchmarks = {"Fund": fund_ret}
    benchmarks.update(_get_active_benchmarks(spy_ret, ew_ret, ivw_ret))

    fig = go.Figure()

    # 활성 윈도우 × 활성 벤치마크 라인
    line_dash_map = {1: "solid", 3: "dash", 5: "dot"}
    for w in windows:
        for name, ret in benchmarks.items():
            rolling = _rolling_annualized_return(ret, window_years=w)
            rolling = filter_by_eval_period(rolling, eval_label)
            if len(rolling.dropna()) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values * 100,
                name=f"{name} {w}y",
                line=dict(
                    color=BENCHMARK_COLORS.get(name, COLORS["text_muted"]),
                    dash=line_dash_map.get(w, "solid"),
                    width=2 if w == 1 else 1.5,
                ),
            ))

    # 0% 기준선
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], line_width=1)

    # Regime 배경 (FULL 일 때만 의미)
    if period == "FULL":
        add_regime_backgrounds(fig, with_labels=True)

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Annualized Return (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key="perf_rolling_return")


# ======================================================================
# 영역 7: Regime 메트릭 Heatmap
# ======================================================================

def _calc_regime_metric(
    ret: pd.Series, rf: pd.Series, regime_label: str, metric: str
) -> float:
    """단일 (regime, 메트릭) 값 계산. final master_table 정합."""
    if regime_label in REGIMES:
        s, e = REGIMES[regime_label]
    elif regime_label == "HO":
        s, e = EVAL_PERIODS["HOLD_OUT"]
    elif regime_label == "FULL":
        s, e = EVAL_PERIODS["FULL"]
    else:
        return np.nan

    if metric == "CAGR":
        return mc.calc_cagr_subperiod(ret, s, e)
    if metric == "Sortino":
        return mc.calc_sortino_subperiod(ret, rf, s, e)
    if metric == "MDD":
        return mc.calc_mdd_subperiod(ret, s, e)
    return np.nan


def render_regime_heatmap(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    rf: pd.Series,
) -> None:
    """
    영역 7: Regime 5행 × 메트릭 4열 Heatmap.

    Tab 전환: Fund / vs SPY / vs EW / vs IVW (활성 벤치마크에 따라).
    """
    regime_rows = ["R1_회복", "R2_확장", "R3_변동", "HO", "FULL"]
    metrics = ["CAGR", "Sortino", "MDD", "Active"]

    benchmarks = _get_active_benchmarks(spy_ret, ew_ret, ivw_ret)

    # Tab 구성: Fund + 활성 벤치마크
    tab_labels = ["Fund"] + [f"vs {n}" for n in benchmarks.keys()]
    tabs = st.tabs(tab_labels)

    # Fund Tab
    with tabs[0]:
        _render_heatmap_for(fund_ret, None, rf, regime_rows, metrics, "Fund")

    # vs Benchmark Tabs
    for i, (name, bench) in enumerate(benchmarks.items(), start=1):
        with tabs[i]:
            _render_heatmap_for(fund_ret, bench, rf, regime_rows, metrics, f"vs {name}")


def _render_heatmap_for(
    fund_ret: pd.Series,
    bench: pd.Series | None,
    rf: pd.Series,
    regime_rows: list[str],
    metrics: list[str],
    label: str,
) -> None:
    """단일 Tab 의 Heatmap 렌더."""
    z = []
    text = []
    for regime in regime_rows:
        row = []
        text_row = []
        for metric in metrics:
            if metric == "Active":
                if bench is None:
                    row.append(np.nan)
                    text_row.append("—")
                    continue
                # Active = (Fund CAGR - Bench CAGR) for that regime
                f = _calc_regime_metric(fund_ret, rf, regime, "CAGR")
                b = _calc_regime_metric(bench, rf, regime, "CAGR")
                v = f - b if not (pd.isna(f) or pd.isna(b)) else np.nan
            elif bench is None:
                v = _calc_regime_metric(fund_ret, rf, regime, metric)
            else:
                # vs Bench Tab: Fund - Bench
                f = _calc_regime_metric(fund_ret, rf, regime, metric)
                b = _calc_regime_metric(bench, rf, regime, metric)
                v = f - b if not (pd.isna(f) or pd.isna(b)) else np.nan
            row.append(v)
            text_row.append(_format_cell(v, metric))
        z.append(row)
        text.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=metrics,
        y=regime_rows,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale="RdYlGn",
        zmid=0,
        showscale=True,
        hoverongaps=False,
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=80, r=20, t=20, b=40),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),  # R1 위, FULL 아래
    )
    st.plotly_chart(fig, use_container_width=True, key=f"perf_regime_heatmap_{label}")


def _format_cell(v: float, metric: str) -> str:
    """Heatmap 셀 텍스트 포맷."""
    if pd.isna(v):
        return "—"
    if metric == "Sortino":
        return f"{v:+.2f}" if v != 0 else "0.00"
    return f"{v*100:+.1f}%"


# ======================================================================
# 영역 8: 분포 통계 카드 (Skewness / Excess Kurtosis / Tail Ratio)
# ======================================================================

def _render_distribution_card(
    title: str,
    ko: str,
    fund_value: float,
    bench_dict: dict[str, float],
    series: pd.Series,
    key_suffix: str = "",
) -> None:
    """
    단일 분포 통계 카드 (값 + KDE mini chart).

    Args:
        key_suffix: plotly_chart key 충돌 방지용 (월별/일별 구분).
    """
    border_color = COLORS["primary"]

    # KDE mini chart (간단 히스토그램)
    fig = go.Figure()
    if len(series.dropna()) > 0:
        fig.add_trace(go.Histogram(
            x=series.dropna() * 100,
            nbinsx=30,
            marker_color=border_color,
            opacity=0.7,
            showlegend=False,
        ))
    fig.update_layout(
        height=120,
        margin=dict(l=10, r=10, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"], size=10),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor=COLORS["text_muted"]),
        yaxis=dict(showgrid=False, visible=False),
    )

    bench_lines = ""
    for name, v in bench_dict.items():
        bench_lines += (
            f'<div style="font-size:12px;color:{COLORS["text_muted"]};">'
            f'{name}: {_format_dist_value(v, title)}'
            f'</div>'
        )

    st.markdown(
        f'<div style="border:1px solid {border_color};border-radius:8px;'
        f'padding:14px;background-color:#1F2937;">'
        f'<div style="font-weight:bold;color:#FAFAFA;">{title}</div>'
        f'<div style="font-size:12px;color:#9CA3AF;margin-bottom:8px;">{ko}</div>'
        f'<div style="font-size:18px;color:{border_color};font-weight:bold;">'
        f'Fund: {_format_dist_value(fund_value, title)}'
        f'</div>'
        f'{bench_lines}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key=f"perf_dist_{title}_{key_suffix}")


def _format_dist_value(v: float, metric: str) -> str:
    if pd.isna(v):
        return "—"
    if metric == "Skewness" or metric == "Excess Kurtosis":
        return f"{v:+.3f}"
    return f"{v:.2f}"


def render_distribution_stats(
    fund_ret_monthly: pd.Series,
    spy_ret_monthly: pd.Series,
    ew_ret_monthly: pd.Series | None,
    ivw_ret_monthly: pd.Series | None,
    fund_ret_daily: pd.Series | None,
    spy_ret_daily: pd.Series | None,
    period: str,
) -> None:
    """
    영역 8: Skewness / Excess Kurtosis / Tail Ratio.
    일별 / 월별 Tab + vs SPY Tab.

    fund_ret_daily, spy_ret_daily: 일별 데이터 (없으면 일별 Tab 비활성).
    """
    eval_label = _period_to_eval_label(period)
    tab_monthly, tab_daily = st.tabs(["월별 (Monthly)", "일별 (Daily)"])

    # 월별
    with tab_monthly:
        f = filter_by_eval_period(fund_ret_monthly, eval_label)
        bench_dict_skew = {}
        bench_dict_kurt = {}
        bench_dict_tail = {}
        active_benchmarks = _get_active_benchmarks(spy_ret_monthly, ew_ret_monthly, ivw_ret_monthly)
        for name, bench in active_benchmarks.items():
            b = filter_by_eval_period(bench, eval_label)
            bench_dict_skew[name] = mc.calc_skewness(b)
            bench_dict_kurt[name] = mc.calc_excess_kurtosis(b)
            bench_dict_tail[name] = mc.calc_tail_ratio(b)

        cols = st.columns(3)
        with cols[0]:
            _render_distribution_card("Skewness", "왜도", mc.calc_skewness(f), bench_dict_skew, f, key_suffix="monthly")
        with cols[1]:
            _render_distribution_card("Excess Kurtosis", "초과첨도", mc.calc_excess_kurtosis(f), bench_dict_kurt, f, key_suffix="monthly")
        with cols[2]:
            _render_distribution_card("Tail Ratio", "꼬리 비율", mc.calc_tail_ratio(f), bench_dict_tail, f, key_suffix="monthly")

        st.caption(f"※ 월별 통계 = {len(f)} 개월")

    # 일별
    with tab_daily:
        if fund_ret_daily is None or len(fund_ret_daily.dropna()) == 0:
            st.info("일별 데이터 미산출.")
            return

        # 기간 필터 — winsorize 적용 없이 원본 그대로 (final 정합).
        # 분석 기간 (2010-2025) 내 SPY/fund_daily 모두 정상 범위 (-11.59% ~ +11.46%).
        # 진단 결과 SPY -41% 는 분석 기간 외 (2004-01-02 first-row 오류),
        # ^IRX 등 비정상 ticker 는 fund.weights 에 자동 제외됨.
        # final/bl_functions.py:compute_daily_slice 도 NaN threshold + fillna(0) 만,
        # winsorize 처리 없음 → 우리도 동일 정합 (학술 정직성).
        s, e = EVAL_PERIODS[eval_label]
        f_daily = fund_ret_daily[(fund_ret_daily.index >= s) & (fund_ret_daily.index <= e)]

        bench_dict_d_skew = {}
        bench_dict_d_kurt = {}
        bench_dict_d_tail = {}

        if spy_ret_daily is not None and st.session_state.get("show_spy", True):
            spy_d_filt = spy_ret_daily[(spy_ret_daily.index >= s) & (spy_ret_daily.index <= e)]
            bench_dict_d_skew["SPY"] = mc.calc_skewness(spy_d_filt)
            bench_dict_d_kurt["SPY"] = mc.calc_excess_kurtosis(spy_d_filt)
            bench_dict_d_tail["SPY"] = mc.calc_tail_ratio(spy_d_filt)

        cols = st.columns(3)
        with cols[0]:
            _render_distribution_card("Skewness", "왜도 (일별)", mc.calc_skewness(f_daily), bench_dict_d_skew, f_daily, key_suffix="daily")
        with cols[1]:
            _render_distribution_card("Excess Kurtosis", "초과첨도 (일별)", mc.calc_excess_kurtosis(f_daily), bench_dict_d_kurt, f_daily, key_suffix="daily")
        with cols[2]:
            _render_distribution_card("Tail Ratio", "꼬리 비율 (일별)", mc.calc_tail_ratio(f_daily), bench_dict_d_tail, f_daily, key_suffix="daily")

        st.caption(
            f"※ 일별 통계 = {len(f_daily)} 영업일 — fund.weights × daily_returns 산출 "
            f"(final/bl_functions.compute_daily_slice 패턴: NaN < 10% threshold + fillna(0)). "
            f"Excess Kurtosis 12-19 는 정상 시장 일별 통계 (fat tail) — Bollerslev (1986) GARCH 동기."
        )
