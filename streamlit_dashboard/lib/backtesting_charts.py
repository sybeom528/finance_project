"""
lib/backtesting_charts.py — Backtesting 페이지 4 영역 차트 함수

영역:
  3. render_backtest_kpi (5 KPI: Gap / Sensitivity / 4-slot / Recovery / Regime)
  4. render_regime_detail_table (12 메트릭 × 5 Regime + Sortino 막대)
  5. render_sub_events (4 위기 Timeline + 표)
  6. render_sensitivity_test (Top 10 + 신모델 강조 + Top 1-10 차이 막대)

설계 원칙 (초안 — 향후 구조 변경 용이):
  - 각 영역 독립 함수 (영역 추가/제거 시 페이지 entry 만 변경)
  - 메트릭 / 위기 정의는 metric_calculators.py 의 외부 dict (REGIME_METRICS / SUB_EVENTS)
  - 데이터 의존성: fund_pkl + 156 config (sensitivity 만)
  - 모든 메트릭 final 정합 또는 학술 출처 명시

참조: docs/plan/03_pages/08_backtesting.md, decisionlog/08_backtesting.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import metric_calculators as mc
from lib.colors import BENCHMARK_COLORS, COLORS
from lib.data_loader import list_available_configs, load_fund_results, load_monthly_panel
from lib.plot_helpers import add_event_annotations, add_regime_backgrounds


# ======================================================================
# 공통 유틸
# ======================================================================

def _fmt_pct(v, plus_sign=False, digits=2):
    if pd.isna(v):
        return "—"
    sign = "+" if plus_sign and v > 0 else ""
    return f"{sign}{v * 100:.{digits}f}%"


def _fmt_ratio(v, digits=3):
    if pd.isna(v):
        return "—"
    return f"{v:.{digits}f}"


def _color_by_threshold(val, good_thr, bad_thr, lower_is_better=False):
    """val 이 good/bad threshold 기준 어디 있는지 → 🟢/🟧/🔴."""
    if pd.isna(val):
        return "⚪"
    if lower_is_better:
        if val <= good_thr:
            return "🟢"
        elif val <= bad_thr:
            return "🟧"
        return "🔴"
    else:
        if val >= good_thr:
            return "🟢"
        elif val >= bad_thr:
            return "🟧"
        return "🔴"


# ======================================================================
# 156 config 메트릭 캐시 (영역 3 + 6)
# ======================================================================

@st.cache_data
def _compute_all_config_metrics() -> pd.DataFrame:
    """
    156 config × 핵심 메트릭 산출 (cache). 영역 3 / 6 공유.

    한 번 산출 후 streamlit session 동안 메모리 유지.
    """
    panel = load_monthly_panel()
    rf = panel.groupby("date")["rf_1m"].first()

    configs = list_available_configs()
    rows = []
    for cn in configs:
        try:
            d = load_fund_results(cn)
            ret = d["ret"]
            spy = d["spy_ret"]
            cfg = d.get("config", {})
            rows.append({
                "config": cn,
                "p_mode": cfg.get("p_mode"),
                "p_weight": cfg.get("p_weight"),
                "q_mode": cfg.get("q_mode"),
                "omega_mode": cfg.get("omega_mode"),
                "prior": cfg.get("prior"),
                "CAGR": mc.calc_cagr(ret),
                "Vol": mc.calc_volatility(ret),
                "MDD": mc.calc_mdd(ret),
                "Sharpe": mc.calc_sharpe(ret, rf),
                "Sortino": mc.calc_sortino(ret, rf),
                "Beta": mc.calc_beta(ret, spy, rf),
                "Sortino_TEST": mc.calc_sortino_subperiod(ret, rf, "2010-01-01", "2023-12-31"),
                "Sortino_HO": mc.calc_sortino_subperiod(ret, rf, "2024-01-01", "2025-12-31"),
            })
        except Exception:
            continue
    df = pd.DataFrame(rows).set_index("config")
    df["TEST_HO_Gap"] = df["Sortino_TEST"] - df["Sortino_HO"]
    return df


# ======================================================================
# 영역 3: Backtest Summary KPI 5개
# ======================================================================

def render_backtest_kpi(
    ret: pd.Series,
    spy: pd.Series,
    rf: pd.Series,
    current_config: str = "mat_eq_mcap_raw_rms",
) -> None:
    """
    5 KPI: TEST/HO Gap / Sensitivity Robustness / 4-slot Robust /
           Avg Recovery Time / Regime 일관성 Score.
    모든 메트릭 FULL 기준 (검증 메트릭 특성).
    """
    # 1. TEST/HO Gap
    gap = mc.calc_test_ho_gap(ret, rf)

    # 2/3. Sensitivity / Robustness — 156 config Sortino 분포
    config_df = _compute_all_config_metrics()
    sortino_mean = float(config_df["Sortino"].mean())
    sortino_std = float(config_df["Sortino"].std())
    robustness_score = sortino_mean / sortino_std if sortino_std > 0 else np.nan
    # 신모델 Top 10 안에 있는지 → "Sensitivity Robustness" (Top 1 일관성 근사)
    rank = (
        config_df["Sortino"].rank(ascending=False)[current_config]
        if current_config in config_df.index
        else np.nan
    )
    sensitivity_robust = (
        100 * (1 - (rank - 1) / max(1, len(config_df) - 1))
        if not pd.isna(rank)
        else np.nan
    )

    # 4. Avg Recovery Time (Sub-events)
    sub_df = mc.calc_sub_event_metrics(ret, spy, rf)
    avg_recovery = mc.calc_avg_recovery_time(sub_df)

    # 5. Regime 일관성 Score
    regime_consistency = mc.calc_regime_consistency_score(ret, rf)

    cols = st.columns(5)

    with cols[0]:
        emoji = _color_by_threshold(gap, 0.5, 1.0, lower_is_better=True)
        st.metric(
            label=f"{emoji} TEST/HO Gap",
            value=_fmt_ratio(gap),
            help="TEST Sortino - HO Sortino. 작을수록 학습편향 ↓ (robust)",
        )

    with cols[1]:
        emoji = _color_by_threshold(sensitivity_robust, 90, 50)
        st.metric(
            label=f"{emoji} Sensitivity Robust",
            value=f"Top {int(rank)}/{len(config_df)}" if not pd.isna(rank) else "—",
            delta=f"상위 {sensitivity_robust:.1f}%" if not pd.isna(sensitivity_robust) else None,
            delta_color="off",
            help="156 config 중 신모델 Sortino rank — 상위일수록 robust",
        )

    with cols[2]:
        emoji = _color_by_threshold(robustness_score, 5.0, 2.0)
        st.metric(
            label=f"{emoji} 4-slot Robustness",
            value=f"{robustness_score:.2f}",
            delta=f"mean {sortino_mean:.2f} / std {sortino_std:.2f}",
            delta_color="off",
            help="156 config Sortino mean/std (= 1/CV). 높을수록 결과 안정",
        )

    with cols[3]:
        emoji = _color_by_threshold(avg_recovery, 6, 12, lower_is_better=True)
        val = f"{avg_recovery:.1f}m" if not pd.isna(avg_recovery) else "—"
        st.metric(
            label=f"{emoji} Avg Recovery",
            value=val,
            help="4 Sub-events 평균 회복 개월 (위기 종료 후 신고가)",
        )

    with cols[4]:
        score = regime_consistency.get("score", np.nan)
        emoji = _color_by_threshold(score, 1.0, 0.5)
        st.metric(
            label=f"{emoji} Regime 일관성",
            value=_fmt_ratio(score, 2),
            delta=(
                f"R1/R2/R3/HO mean {regime_consistency.get('mean', np.nan):.2f}"
                if not pd.isna(regime_consistency.get("mean", np.nan)) else None
            ),
            delta_color="off",
            help="R1/R2/R3/HO Sortino mean/std. 높을수록 모든 시기 일관",
        )


# ======================================================================
# 영역 4: 156 config 누적 수익률 비교 (Robustness 시각)
# ======================================================================

@st.cache_data
def _compute_all_config_cumulative(start: str = "2010-01-31") -> pd.DataFrame:
    """
    156 config × 시점 의 누적 수익률 matrix (initial=1).
    cache 후 spaghetti / percentile 모두 활용.
    """
    configs = list_available_configs()
    series_dict: dict[str, pd.Series] = {}
    for cn in configs:
        try:
            d = load_fund_results(cn)
            ret = d["ret"].dropna()
            cum = (1 + ret).cumprod()
            series_dict[cn] = cum
        except Exception:
            continue
    df = pd.DataFrame(series_dict)
    return df


def render_cumulative_comparison(
    fund_ret: pd.Series,
    fund_spy: pd.Series,
    current_config: str = "mat_eq_mcap_raw_rms",
) -> None:
    """
    156 config 의 누적 수익률 비교 — spaghetti / percentile 토글.

    "Backtesting = 과거 시뮬레이션" 의 시각적 표현.
    156 회색 라인 (또는 percentile fill) + 신모델 cobalt 강조 + SPY 비교.
    """
    cols_t = st.columns([1.5, 1.5, 4])
    with cols_t[0]:
        view_mode = st.selectbox(
            "표시 방식",
            options=["156 라인 (spaghetti)", "Percentile (P5-P95 fill + median)"],
            index=0, key="bt_cum_view",
        )
    with cols_t[1]:
        log_scale = st.checkbox("Log scale", value=False, key="bt_cum_log")

    df_cum = _compute_all_config_cumulative()
    if len(df_cum) == 0:
        st.warning("156 config 누적 수익률 산출 불가.")
        return

    fig = go.Figure()

    if view_mode.startswith("156"):
        # Spaghetti — 156 회색 라인
        for cn in df_cum.columns:
            if cn == current_config:
                continue
            cum = df_cum[cn].dropna()
            fig.add_trace(go.Scatter(
                x=cum.index, y=cum.values,
                mode="lines", name=cn, showlegend=False,
                line=dict(color="rgba(156, 163, 175, 0.18)", width=0.6),
                hovertemplate=f"{cn}<br>%{{x|%Y-%m}}: %{{y:.2f}}x<extra></extra>",
            ))
    else:
        # Percentile fill (P5-P95) + median
        p5 = df_cum.quantile(0.05, axis=1)
        p95 = df_cum.quantile(0.95, axis=1)
        median = df_cum.median(axis=1)
        fig.add_trace(go.Scatter(
            x=p95.index, y=p95.values, mode="lines", name="P95",
            line=dict(color="rgba(156, 163, 175, 0)"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=p5.index, y=p5.values, mode="lines", name="P5-P95 (156 config)",
            line=dict(color="rgba(156, 163, 175, 0)"),
            fill="tonexty", fillcolor="rgba(156, 163, 175, 0.25)",
            hovertemplate="%{x|%Y-%m}<br>P5: %{y:.2f}x<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=median.index, y=median.values, mode="lines", name="156 median",
            line=dict(color=COLORS["text_muted"], width=1.5, dash="dash"),
            hovertemplate="%{x|%Y-%m}<br>median: %{y:.2f}x<extra></extra>",
        ))

    # 신모델 — Cobalt 강조 라인 (메인)
    if current_config in df_cum.columns:
        cum_main = df_cum[current_config].dropna()
        fig.add_trace(go.Scatter(
            x=cum_main.index, y=cum_main.values,
            mode="lines", name=f"★ {current_config} (Top 1)",
            line=dict(color=BENCHMARK_COLORS.get("Fund", COLORS["primary"]), width=2.5),
            hovertemplate=f"<b>{current_config}</b><br>%{{x|%Y-%m}}: %{{y:.2f}}x<extra></extra>",
        ))

    # SPY 비교
    spy_clean = fund_spy.dropna()
    if len(spy_clean) > 0:
        spy_cum = (1 + spy_clean).cumprod()
        fig.add_trace(go.Scatter(
            x=spy_cum.index, y=spy_cum.values,
            mode="lines", name="SPY (S&P 500)",
            line=dict(color=BENCHMARK_COLORS.get("SPY", COLORS["text"]), width=2, dash="dot"),
            hovertemplate="%{x|%Y-%m}<br>SPY: %{y:.2f}x<extra></extra>",
        ))

    fig = add_regime_backgrounds(fig, with_labels=False)
    fig = add_event_annotations(fig)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        xaxis_title="시점", yaxis_title="누적 수익률 (initial = 1)",
        yaxis=dict(type="log" if log_scale else "linear"),
        height=480, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # narrative 보강
    if current_config in df_cum.columns:
        latest_t = df_cum[current_config].dropna().index[-1]
        main_final = float(df_cum[current_config].dropna().iloc[-1])
        spy_final = float((1 + spy_clean).cumprod().iloc[-1]) if len(spy_clean) > 0 else None
        median_final = float(df_cum.median(axis=1).iloc[-1])
        msg = (
            f"📊 **신모델** {main_final:.2f}x → "
            f"**156 median** {median_final:.2f}x → "
        )
        if spy_final is not None:
            msg += f"**SPY** {spy_final:.2f}x"
        msg += f"  (시점: {latest_t.strftime('%Y-%m')})"
        st.caption(msg)


# ======================================================================
# 영역 5: Regime 메트릭 자세한 비교 (12 메트릭)
# ======================================================================

def render_regime_detail_table(
    ret: pd.Series,
    spy: pd.Series,
    rf: pd.Series,
) -> None:
    """12 메트릭 × 5 Regime 표 + Sortino 막대 + Best/Worst 강조."""
    df = mc.calc_regime_full_metrics(ret, spy, rf)
    if len(df) == 0:
        st.warning("Regime 데이터 산출 불가.")
        return

    # 표 표시 (포맷 적용)
    df_display = df.copy()
    pct_cols = ["CAGR", "MDD", "Volatility", "Active_Return", "Win_Rate", "VaR_5", "CVaR_5"]
    ratio_cols = ["Sortino", "Sharpe", "IR", "Calmar", "Beta"]

    st.dataframe(
        df_display.style.format(
            {c: "{:+.2%}" for c in pct_cols} | {c: "{:+.3f}" for c in ratio_cols}
        ),
        use_container_width=True,
        height=240,
    )

    # Best/Worst Regime (Sortino 기준)
    if "Sortino" in df.columns and df["Sortino"].notna().any():
        best = df["Sortino"].idxmax()
        worst = df["Sortino"].idxmin()
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"⭐ **Best Regime**: {best} (Sortino {df.loc[best, 'Sortino']:.2f})")
        with c2:
            st.error(f"🔴 **Worst Regime**: {worst} (Sortino {df.loc[worst, 'Sortino']:.2f})")

    # Sortino 막대
    fig = go.Figure(go.Bar(
        x=df.index, y=df["Sortino"],
        marker_color=[
            COLORS["accent_green"] if v == df["Sortino"].max()
            else (COLORS["accent_red"] if v == df["Sortino"].min() else COLORS["primary"])
            for v in df["Sortino"]
        ],
        text=[f"{v:.2f}" for v in df["Sortino"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Sortino: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        yaxis_title="Sortino",
        height=320, margin=dict(t=20, l=0, r=0, b=0),
        title=dict(text="Regime별 Sortino 비교", font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # CSV 다운로드
    csv = df.to_csv().encode("utf-8-sig")
    st.download_button(
        label="⬇ CSV 다운로드 (Regime × 12 메트릭)",
        data=csv,
        file_name="regime_metrics.csv",
        mime="text/csv",
        key="bt_regime_csv",
    )


# ======================================================================
# 영역 5: Sub-events 분석 (4 위기)
# ======================================================================

def render_sub_events(
    ret: pd.Series,
    spy: pd.Series,
    rf: pd.Series,
) -> None:
    """4 위기 Timeline + 표 + Best/Worst 강조."""
    df = mc.calc_sub_event_metrics(ret, spy, rf)
    if len(df) == 0:
        st.warning("Sub-events 데이터 산출 불가.")
        return

    # 표 표시
    df_display = df.copy()
    df_display["Recovery_months"] = df_display["Recovery_months"].apply(
        lambda v: f"{int(v)}m" if v is not None and not pd.isna(v) else "미회복"
    )
    st.dataframe(
        df_display.style.format(
            {"Fund": "{:+.2%}", "SPY": "{:+.2%}", "Active": "{:+.2%}", "MDD": "{:.2%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Best/Worst (Active Return 기준)
    if df["Active"].notna().any():
        best_idx = df["Active"].idxmax()
        worst_idx = df["Active"].idxmin()
        c1, c2 = st.columns(2)
        with c1:
            ev = df.loc[best_idx, "Event"]
            v = df.loc[best_idx, "Active"]
            st.success(f"⭐ **Best 방어**: {ev} (Active {v:+.2%})")
        with c2:
            ev = df.loc[worst_idx, "Event"]
            v = df.loc[worst_idx, "Active"]
            star = " ★" if "2024" in ev else ""
            st.error(f"🔴 **Worst**: {ev}{star} (Active {v:+.2%})")
            if star:
                st.caption("→ HO 정당화 narrative 핵심 (Sector Watch 영역 8 참조)")

    # Active Return 막대 (위기별)
    fig = go.Figure(go.Bar(
        x=df["Event"], y=df["Active"],
        marker_color=[
            COLORS["accent_green"] if v > 0 else COLORS["accent_red"]
            for v in df["Active"]
        ],
        text=[f"{v:+.2%}" for v in df["Active"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Active: %{y:+.3%}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        yaxis=dict(tickformat=".1%", title="Active Return (Fund − SPY)"),
        height=320, margin=dict(t=30, l=0, r=0, b=0),
        title=dict(text="위기별 Active Return — Fund vs SPY", font=dict(size=13)),
    )
    fig.add_hline(y=0, line_color=COLORS["text"], line_width=1)
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# 영역 6: Sensitivity Test (Top 10 + 신모델 강조)
# ======================================================================

def render_sensitivity_test(current_config: str = "mat_eq_mcap_raw_rms") -> None:
    """Top 10 config 표 + 신모델 강조 + Top 1-10 차이 막대."""
    df = _compute_all_config_metrics()
    if len(df) == 0:
        st.warning("156 config 데이터 부재.")
        return

    # Top N 토글
    cols_t = st.columns([1.5, 5])
    with cols_t[0]:
        top_n = st.selectbox(
            "Top N", options=[10, 20, 50], index=0, key="bt_sensitivity_top_n",
        )

    # Sortino 정렬 + Top N
    df_sorted = df.sort_values("Sortino", ascending=False).head(top_n).copy()
    df_sorted["Rank"] = range(1, len(df_sorted) + 1)

    # 신모델 marker 추가
    def _mark(idx: str) -> str:
        return f"{idx} ★" if idx == current_config else idx

    df_display = df_sorted.reset_index().rename(columns={"config": "Config"})
    df_display["Config"] = df_display["Config"].apply(_mark)
    df_display = df_display[["Rank", "Config", "Sortino", "CAGR", "MDD", "Sharpe", "p_weight", "q_mode", "omega_mode"]]

    st.dataframe(
        df_display.style.format(
            {"CAGR": "{:+.2%}", "MDD": "{:.2%}", "Sortino": "{:.3f}", "Sharpe": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    # Top N 차이 막대
    fig = go.Figure(go.Bar(
        x=df_sorted.index, y=df_sorted["Sortino"],
        marker_color=[
            COLORS["primary"] if c == current_config else COLORS["text_muted"]
            for c in df_sorted.index
        ],
        text=[f"{v:.3f}" for v in df_sorted["Sortino"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Sortino: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        xaxis_title="Config", yaxis_title="Sortino",
        height=360, margin=dict(t=30, l=0, r=0, b=80),
        title=dict(text=f"Top {top_n} config Sortino 비교 — 신모델 ★ 강조", font=dict(size=13)),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 신모델 위치 narrative
    if current_config in df.index:
        rank = int(df["Sortino"].rank(ascending=False)[current_config])
        sortino_val = df.loc[current_config, "Sortino"]
        top_diff = df_sorted["Sortino"].iloc[0] - df_sorted["Sortino"].iloc[-1]
        st.info(
            f"🎯 **신모델 mat_eq_mcap_raw_rms** Sortino **{sortino_val:.3f}** "
            f"= Rank **{rank} / {len(df)}** (상위 {(1 - (rank-1)/(len(df)-1))*100:.1f}%). "
            f"Top {top_n} 의 Sortino 차이 = **{top_diff:.3f}** → "
            f"**{'4-slot 변경에도 결과 안정 (robust)' if top_diff < 0.10 else '일부 모델 차이 큼 — Top 1 의존성'}**."
        )

    # CSV 다운로드
    csv = df_sorted.to_csv().encode("utf-8-sig")
    st.download_button(
        label=f"⬇ Top {top_n} CSV 다운로드",
        data=csv,
        file_name=f"sensitivity_top{top_n}.csv",
        mime="text/csv",
        key="bt_sensitivity_csv",
    )
