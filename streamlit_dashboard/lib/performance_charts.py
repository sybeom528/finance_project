"""
lib/performance_charts.py - Performance 페이지 7 영역 차트 함수

영역 3-9 모두 함수화 (2026-05-11 영역 4 신규 추가):
  - 영역 3: render_performance_kpi (CAGR / Sortino / Sharpe / IR / Active Return)
  - 영역 4: render_cumulative_only (★ 신규 — 누적 수익률 단일 차트)
  - 영역 5: render_annual_returns (사이드바이 막대 + 평균선)
  - 영역 6: render_active_return_analysis (위 막대 + 아래 이중 축 Rolling)
  - 영역 7: render_rolling_return (1y/3y/5y 다중 + Regime 배경)
  - 영역 8: render_regime_heatmap (Tab 전환 + diverging colormap)
  - 영역 9: render_distribution_stats (일별 only, 2026-05-11 단순화)

모든 메트릭은 final/bl_functions + master_table 정확 재현.

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

    # SPY = primary delta 기준 (활성 시), 추가 벤치마크는 caption 으로
    has_spy = "SPY" in benchmarks
    spy_bench = benchmarks.get("SPY")

    cols = st.columns(5)

    # 카드 1: CAGR (Net + Gross + TC 누적)
    with cols[0]:
        spy_cagr_delta = None
        if has_spy:
            bench_cagr_spy = mc.calc_cagr_subperiod(spy_bench, s, e)
            spy_cagr_delta = cagr - bench_cagr_spy
        st.metric(
            label="CAGR",
            value=_format_pct(cagr, plus_sign=True),
            delta=(
                f"{_format_pct(spy_cagr_delta, plus_sign=True)} vs SPY"
                if spy_cagr_delta is not None else None
            ),
            help=get_tooltip("Net CAGR") or "Net 연환산 수익률 (TC 차감 후)",
        )
        # 추가 벤치마크 (EW / IVW) delta
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            bench_cagr = mc.calc_cagr_subperiod(bench, s, e)
            d = cagr - bench_cagr
            st.caption(f"vs {name} {_format_pct(d, plus_sign=True)}")
        # B-1: Gross + TC 누적
        if cagr_gross is not None and not pd.isna(cagr_gross):
            tc_diff = cagr_gross - cagr
            st.caption(
                f"Gross {_format_pct(cagr_gross, plus_sign=True)} · "
                f"TC -{_format_pct(tc_diff)} (편측 20bp = 0.20%/거래)"
            )

    # 카드 2: Sortino
    with cols[1]:
        spy_sortino_delta = None
        if has_spy:
            bench_sortino_spy = mc.calc_sortino_subperiod(spy_bench, rf, s, e)
            spy_sortino_delta = sortino - bench_sortino_spy
        st.metric(
            label="Sortino",
            value=_format_ratio(sortino),
            delta=(
                f"{('+' if spy_sortino_delta > 0 else '')}{spy_sortino_delta:.2f} vs SPY"
                if spy_sortino_delta is not None else None
            ),
            help=get_tooltip("Sortino") or "하방위험 조정 수익률",
        )
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            bench_sortino = mc.calc_sortino_subperiod(bench, rf, s, e)
            d = sortino - bench_sortino
            st.caption(f"vs {name} {('+' if d > 0 else '')}{d:.2f}")

    # 카드 3: Sharpe
    with cols[2]:
        spy_sharpe_delta = None
        if has_spy:
            bench_sharpe_spy = mc.calc_sharpe_subperiod(spy_bench, rf, s, e)
            spy_sharpe_delta = sharpe - bench_sharpe_spy
        st.metric(
            label="Sharpe",
            value=_format_ratio(sharpe),
            delta=(
                f"{('+' if spy_sharpe_delta > 0 else '')}{spy_sharpe_delta:.2f} vs SPY"
                if spy_sharpe_delta is not None else None
            ),
            help=get_tooltip("Sharpe") or "위험 조정 수익률",
        )
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            bench_sharpe = mc.calc_sharpe_subperiod(bench, rf, s, e)
            d = sharpe - bench_sharpe
            st.caption(f"vs {name} {('+' if d > 0 else '')}{d:.2f}")

    # 카드 4: IR (vs SPY 기준)
    with cols[3]:
        st.metric(
            label="IR",
            value=_format_ratio(ir_spy),
            delta="vs SPY",
            delta_color="off",
            help=get_tooltip("IR") or "Information Ratio = Active Return / Tracking Error",
        )
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            ir_b = mc.calc_ir(
                filter_by_eval_period(fund_ret, eval_label),
                filter_by_eval_period(bench, eval_label),
            )
            st.caption(f"vs {name} {_format_ratio(ir_b)}")

    # 카드 5: Active Return (vs SPY 기준)
    with cols[4]:
        st.metric(
            label="Active Return",
            value=_format_pct(active_spy, plus_sign=True),
            delta="vs SPY (annualized)",
            delta_color="off",
            help=get_tooltip("Active Return") or "Active Return = Fund − Benchmark (annualized)",
        )
        for name, bench in benchmarks.items():
            if name == "SPY":
                continue
            active_b = mc.calc_active_return(
                filter_by_eval_period(fund_ret, eval_label),
                filter_by_eval_period(bench, eval_label),
            )
            st.caption(f"vs {name} {_format_pct(active_b, plus_sign=True)}")


# ======================================================================
# 영역 4: 누적 수익률 (신규 — 2026-05-11)
#   Overview 의 이중 차트 (누적 + Drawdown) 와 달리 누적만 표시.
#   Drawdown 은 Risk Metrics 영역 4 에 별도 (책임 분리).
# ======================================================================

def _annotate_final_return(fig: go.Figure, cum_series: pd.Series, color: str) -> None:
    """누적 수익률 시계열의 마지막 점에 최종 누적 수익률 % 라벨 추가."""
    if cum_series is None or len(cum_series) == 0:
        return
    final_pct = (cum_series.iloc[-1] - 1) * 100
    fig.add_annotation(
        x=cum_series.index[-1], y=cum_series.iloc[-1],
        text=f"<b>{final_pct:+.1f}%</b>",
        showarrow=False, xanchor="left", yanchor="middle", xshift=8,
        font=dict(color=color, size=12),
    )


def render_cumulative_only(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    show_log: bool,
    period: str,
) -> None:
    """
    Performance 영역 4: 단일 누적 수익률 차트 (Drawdown 제외).

    Overview 의 이중 차트와 다른 점:
      - Drawdown 없음 (Risk Metrics 페이지 영역 4 에 별도)
      - 사이드바 기간 토글 (TEST / Hold Out / FULL) 에 따라 줌인/줌아웃
      - Hero KPI 와 달리 period 영향 받음

    Args:
        fund_ret: 펀드 월별 수익률
        spy_ret: SPY 월별 수익률
        ew_ret / ivw_ret: 벤치마크 (None 가능)
        show_log: True 면 Y축 log scale
        period: "FULL" / "TEST" / "HO"
    """
    eval_label = _period_to_eval_label(period)

    # 기간 필터
    fund_filt = filter_by_eval_period(fund_ret, eval_label)
    spy_filt = filter_by_eval_period(spy_ret, eval_label) if spy_ret is not None else None
    ew_filt = filter_by_eval_period(ew_ret, eval_label) if ew_ret is not None else None
    ivw_filt = filter_by_eval_period(ivw_ret, eval_label) if ivw_ret is not None else None

    fig = go.Figure()

    # Fund 누적 (필수)
    fund_cum = (1 + fund_filt.fillna(0)).cumprod()
    fig.add_trace(go.Scatter(
        x=fund_cum.index, y=fund_cum.values,
        name="Fund (Adaptive VolControl)",
        line=dict(color=BENCHMARK_COLORS["Fund"], width=2.5),
    ))
    _annotate_final_return(fig, fund_cum, BENCHMARK_COLORS["Fund"])

    # SPY (사이드바 토글에 따라)
    if spy_filt is not None and len(spy_filt) > 0 and st.session_state.get("show_spy", True):
        spy_cum = (1 + spy_filt.fillna(0)).cumprod()
        fig.add_trace(go.Scatter(
            x=spy_cum.index, y=spy_cum.values,
            name="SPY",
            line=dict(color=BENCHMARK_COLORS["SPY"], width=1.5),
        ))
        _annotate_final_return(fig, spy_cum, BENCHMARK_COLORS["SPY"])

    # EW
    if ew_filt is not None and len(ew_filt) > 0:
        ew_cum = (1 + ew_filt.fillna(0)).cumprod()
        fig.add_trace(go.Scatter(
            x=ew_cum.index, y=ew_cum.values,
            name="균등가중 (EW)",
            line=dict(color=BENCHMARK_COLORS["EW"], width=1.5, dash="dash"),
        ))
        _annotate_final_return(fig, ew_cum, BENCHMARK_COLORS["EW"])

    # IVW
    if ivw_filt is not None and len(ivw_filt) > 0:
        ivw_cum = (1 + ivw_filt.fillna(0)).cumprod()
        fig.add_trace(go.Scatter(
            x=ivw_cum.index, y=ivw_cum.values,
            name="역변동성 (IVW)",
            line=dict(color=BENCHMARK_COLORS["IVW"], width=1.5, dash="dot"),
        ))
        _annotate_final_return(fig, ivw_cum, BENCHMARK_COLORS["IVW"])

    # Regime 배경 + Event annotation (FULL 일 때만 의미 있음)
    if period == "FULL":
        add_regime_backgrounds(fig, with_labels=True)
        add_event_annotations(fig)

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=70, t=40, b=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
        hovermode="x unified",
        yaxis=dict(
            title="누적 수익률 (시작 = 1.0)",
            type="log" if show_log else "linear",
        ),
    )

    st.plotly_chart(fig, use_container_width=True, key="performance_cumulative_only")


# ======================================================================
# 영역 5: Annual Returns 막대
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
# 영역 6: Active Return 분석 (위아래 두 차트)
# ======================================================================

def render_active_return_analysis(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    period: str,
) -> None:
    """
    영역 6: 위 = 연별 Active Return 막대, 아래 = Rolling Active Return + Tracking Error 이중 축.

    Rolling 윈도우 선택은 함수 내부 (Rolling 차트 직전) 에서 처리 — UX 인접성 향상 (2026-05-11).
    """
    eval_label = _period_to_eval_label(period)
    benchmarks = _get_active_benchmarks(spy_ret, ew_ret, ivw_ret)

    # ── 위: 연별 Active Return 막대 ─────────────────────────────
    st.markdown("##### 연별 Active Return (Fund − Benchmark)")
    st.caption(
        "연도별로 펀드가 벤치마크 대비 얼마나 더 (덜) 벌었는지 막대로 표시. "
        "**녹색** = 시장 초과 (양수), **적색** = 시장 열위 (음수). "
        "0% 기준선 위/아래로 한눈에 우월/열위 시기를 확인할 수 있습니다."
    )
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
    # Rolling 윈도우 선택 (Rolling 차트 직전 — UX 인접성)
    col_rw, _ = st.columns([1, 5])
    with col_rw:
        rolling_window = st.selectbox(
            "Rolling 윈도우",
            options=[12, 36, 60],
            index=1,
            key="perf_rolling_window",
            format_func=lambda x: f"{x}개월",
        )

    st.markdown(f"##### Rolling Active Return + Tracking Error ({rolling_window}개월 윈도우)")
    st.caption(
        f"최근 {rolling_window}개월 기준 시점별 연환산 값. "
        f"**실선 (좌측 Y축)** = Rolling Active Return = 평균적으로 시장을 얼마나 이겼는지. "
        f"**점선 (우측 Y축)** = Rolling Tracking Error = 시장과의 수익률 차이가 얼마나 변동했는지. "
        f"두 라인 비교: Active 가 양수 + Tracking Error 가 작으면 ✓ 안정적 시장 초과 / "
        f"Active 가 양수 + Tracking Error 가 크면 ⚠️ 불안정한 초과 수익."
    )
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
# 영역 7: Annualized Rolling Return (1y / 3y / 5y)
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
# 영역 8: Regime 메트릭 Heatmap
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
# 영역 9: 분포 통계 카드 (Skewness / Excess Kurtosis / Tail Ratio)
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
    # x축 동적 range — 0.5%/99.5% 분위수 기반 (극단 outlier 제외, 분포 중앙 강조)
    x_range = None
    if len(series.dropna()) > 0:
        s_pct = series.dropna() * 100
        fig.add_trace(go.Histogram(
            x=s_pct,
            nbinsx=30,
            marker_color=border_color,
            opacity=0.7,
            showlegend=False,
        ))
        q_low = s_pct.quantile(0.005)
        q_high = s_pct.quantile(0.995)
        x_max = max(abs(q_low), abs(q_high)) * 1.15
        x_range = [-x_max, x_max]
    fig.update_layout(
        height=120,
        margin=dict(l=10, r=10, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"], size=10),
        xaxis=dict(
            showgrid=False, zeroline=True, zerolinecolor=COLORS["text_muted"],
            range=x_range,
        ),
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
    fund_ret_monthly: pd.Series,  # noqa: ARG001 (시그니처 호환성)
    spy_ret_monthly: pd.Series,   # noqa: ARG001
    ew_ret_monthly: pd.Series | None,   # noqa: ARG001
    ivw_ret_monthly: pd.Series | None,  # noqa: ARG001
    fund_ret_daily: pd.Series | None,
    spy_ret_daily: pd.Series | None,
    period: str,
) -> None:
    """
    영역 8: Skewness / Excess Kurtosis / Tail Ratio — **일별 only** (2026-05-11 변경).

    이유: 월별 (192 sample) 은 중심극한정리 (CLT) 효과로 분포가 거의 정규에 수렴.
          분포 형태 (fat tail / 비대칭) 의 진짜 의미는 일별 (~4,000 sample) 에서 관찰 가능.

    fund_ret_daily, spy_ret_daily: 일별 데이터 (없으면 안내).
    monthly 매개변수는 시그니처 호환성 유지 (호출부 변경 회피).
    """
    eval_label = _period_to_eval_label(period)

    if fund_ret_daily is None or len(fund_ret_daily.dropna()) == 0:
        st.info("일별 데이터 미산출.")
        return

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
        _render_distribution_card("Skewness", "왜도", mc.calc_skewness(f_daily), bench_dict_d_skew, f_daily, key_suffix="daily")
    with cols[1]:
        _render_distribution_card("Excess Kurtosis", "초과첨도", mc.calc_excess_kurtosis(f_daily), bench_dict_d_kurt, f_daily, key_suffix="daily")
    with cols[2]:
        _render_distribution_card("Tail Ratio", "꼬리 비율", mc.calc_tail_ratio(f_daily), bench_dict_d_tail, f_daily, key_suffix="daily")

    st.caption(f"※ 일별 통계 = {len(f_daily)} 영업일 기준.")
