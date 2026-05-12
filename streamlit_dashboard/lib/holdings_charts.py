"""
lib/holdings_charts.py — Holdings 페이지 5 영역 차트 함수

영역 3-7 함수화:
  - 영역 3: render_holdings_kpi (Number/EffN/SingleHHI/SectorHHI/TopW/Turnover)
  - 영역 4: render_top_n_table (Top N + 7컬럼 + Weight 막대 + 섹터 색상)
  - 영역 5: render_market_cap_distribution (Bubble + Treemap 탭, 시점 슬라이더)
  - 영역 6: render_holdings_history (Multi-line 변천사 + Regime + 신규/제외 마커)
  - 영역 7: render_attribution_tornado (Top + Bottom Contributors + Sector 합계)

메트릭 = decisionlog/05_holdings.md Hold-Q1:
  - final 정합: comp.n_stocks / eff_n / top1_weight / top10_share / turnover
  - 학술 표준: Single/Sector HHI (Hirschman 1945) / Top 5 / Simple Contribution (Brinson 1986)

참조: docs/plan/03_pages/05_holdings.md, decisionlog/05_holdings.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import metric_calculators as mc
from lib.colors import BENCHMARK_COLORS, COLORS, SECTOR_COLORS
from lib.data_loader import EVAL_PERIODS
from lib.plot_helpers import add_event_annotations, add_regime_backgrounds


# ======================================================================
# 공통 유틸
# ======================================================================

def _fmt_pct(v: float, plus_sign: bool = False, digits: int = 2) -> str:
    if pd.isna(v):
        return "—"
    sign = "+" if plus_sign and v > 0 else ""
    return f"{sign}{v * 100:.{digits}f}%"


def _fmt_ratio(v: float, digits: int = 2) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.{digits}f}"


def _delta_color_lower_is_better(delta: float) -> str:
    if pd.isna(delta):
        return COLORS["text_muted"]
    return COLORS["accent_green"] if delta < 0 else COLORS["accent_red"]


def _filter_period_index(idx: pd.DatetimeIndex, period: str) -> pd.DatetimeIndex:
    """사이드바 토글 (FULL/TEST/HO) 기준 DatetimeIndex 필터."""
    if period == "TEST":
        s, e = EVAL_PERIODS["TEST"]
    elif period == "HO":
        s, e = EVAL_PERIODS["HOLD_OUT"]
    else:
        return idx
    return idx[(idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))]


def _build_ticker_to_sector(universe: pd.DataFrame) -> dict[str, str]:
    """universe DataFrame → dict[ticker, gics_sector]."""
    return dict(zip(universe["ticker"], universe["gics_sector"]))


def _spy_kpi_baseline(panel: pd.DataFrame, ticker_to_sector: dict) -> dict:
    """
    SPY 벤치마크 KPI 근사 — universe 의 모든 활성 종목 균등가중 가정.
    SPY 직접 weight 데이터 없음 → universe 균등 분산 = "전체 분산" 기준선.

    이 baseline 은 펀드 vs "완전 분산 시장" 비교용 (학술 정직 — 'SPY 503' 가정 X).
    """
    latest_date = panel["date"].max()
    rows = panel[panel["date"] == latest_date]
    universe_tickers = rows["ticker"].dropna().tolist()
    if not universe_tickers:
        return {"eff_n": np.nan, "single_hhi": np.nan, "sector_hhi": np.nan,
                "top1": np.nan, "top5": np.nan, "top10": np.nan}

    n = len(universe_tickers)
    # 균등가중 weight = 1/n
    eq_w = pd.Series(1 / n, index=universe_tickers)
    return {
        "eff_n": float(n),               # = 1 / Σ(1/n)² = n
        "single_hhi": float(1 / n),      # = Σ(1/n)² × n = 1/n
        "sector_hhi": mc.calc_sector_hhi(eq_w, ticker_to_sector),
        "top1": float(1 / n),
        "top5": float(5 / n),
        "top10": float(10 / n),
    }


# ======================================================================
# 영역 3: Holdings Summary KPI 6개
# ======================================================================

def render_holdings_kpi(
    weights: pd.DataFrame,
    comp: pd.DataFrame,
    universe: pd.DataFrame,
    panel: pd.DataFrame,
    period: str,
) -> None:
    """
    Holdings KPI 6개 — 사이드바 토글 기간 평균 (메인) + Latest snapshot (caption).

    KPI:
      1. Number of Holdings (final.comp.n_stocks)
      2. Effective N (final.comp.eff_n)
      3. Single Stock HHI (학술, Hirschman 1945)
      4. Sector HHI (학술, Hirschman 1945)
      5. Top Weights (T1/T5/T10) — final.top1/top10 + 학술 T5
      6. Avg Turnover (final.comp.turnover)

    표시 규칙 (2026-05-13 변경):
      - 메인 value = 기간 평균 (사이드바 토글에 반응)
      - caption = "최신 (YYYY-MM): ..." 형식 — 최신 시점 값 보조 표시
      - delta = 기간 평균 vs 균등가중 기준선

    SPY 비교 (4 KPI: EffN/SingleHHI/SectorHHI/TopW): universe 균등가중 기준선.
    """
    ticker_to_sector = _build_ticker_to_sector(universe)
    latest_date = weights.index.max()

    # Latest snapshot — caption 보조 표시용
    snap = mc.calc_holdings_kpi_snapshot(weights.loc[latest_date], ticker_to_sector)
    snap["turnover_latest"] = (
        float(comp.loc[latest_date, "turnover"])
        if (latest_date in comp.index and "turnover" in comp.columns)
        else np.nan
    )

    # 기간 평균 (사이드바 토글) — 메인 KPI value
    period_idx = _filter_period_index(weights.index, period)
    if len(period_idx) > 0:
        weights_p = weights.loc[period_idx]
        comp_p = comp.loc[comp.index.intersection(period_idx)]
        avg = mc.calc_holdings_kpi_period_avg(weights_p, ticker_to_sector, comp_p)
    else:
        avg = {}

    # SPY baseline (4 KPI)
    spy = _spy_kpi_baseline(panel, ticker_to_sector)

    # === 헤더 라벨 ===
    period_label_map = {"FULL": "FULL 192개월", "TEST": "TEST 168개월", "HO": "Hold Out 24개월"}
    latest_label = latest_date.strftime("%Y-%m")
    st.caption(
        f"**기간 평균**: {period_label_map.get(period, period)} (메인 표시)  ·  "
        f"**최신 시점**: {latest_label} (보조 표시)"
    )

    cols = st.columns(6)

    # --- KPI 1: Number of Holdings ---
    with cols[0]:
        n_avg = avg.get("n_stocks_avg", np.nan)
        st.metric(
            label="Number of Holdings",
            value=f"{n_avg:.1f}" if not pd.isna(n_avg) else "—",
            help="보유 종목 수 (활성 weight > 0). 메인 = 기간 평균, caption = 최신 시점.",
        )
        st.caption(f"최신 ({latest_label}): {snap['n_stocks']}")

    # --- KPI 2: Effective N ---
    with cols[1]:
        eff_avg = avg.get("eff_n_avg", np.nan)
        delta = (eff_avg - spy["eff_n"]) if not pd.isna(eff_avg) else np.nan
        st.metric(
            label="Effective N",
            value=_fmt_ratio(eff_avg, 1),
            delta=f"{delta:+.0f} vs 균등" if not pd.isna(delta) else None,
            delta_color="off",
            help="유효 종목 수 = 1/Σw². 높을수록 분산 (Hirschman 1945). 메인 = 기간 평균.",
        )
        st.caption(f"최신 ({latest_label}): {_fmt_ratio(snap['eff_n'], 1)}")

    # --- KPI 3: Single Stock HHI ---
    with cols[2]:
        shhi_avg = avg.get("single_hhi_avg", np.nan)
        delta = (shhi_avg - spy["single_hhi"]) if not pd.isna(shhi_avg) else np.nan
        st.metric(
            label="Single Stock HHI",
            value=_fmt_ratio(shhi_avg, 4),
            delta=f"{delta:+.4f} vs 균등" if not pd.isna(delta) else None,
            delta_color="inverse",  # 낮을수록 좋음
            help="개별 종목 집중도 = Σw². 낮을수록 분산. 메인 = 기간 평균.",
        )
        st.caption(f"최신 ({latest_label}): {_fmt_ratio(snap['single_hhi'], 4)}")

    # --- KPI 4: Sector HHI ---
    with cols[3]:
        sec_avg = avg.get("sector_hhi_avg", np.nan)
        delta = (sec_avg - spy["sector_hhi"]) if not pd.isna(sec_avg) else np.nan
        st.metric(
            label="Sector HHI",
            value=_fmt_ratio(sec_avg, 3),
            delta=f"{delta:+.3f} vs 균등" if not pd.isna(delta) else None,
            delta_color="inverse",
            help="섹터 집중도 = Σ(섹터 weight)². 낮을수록 섹터 분산. 메인 = 기간 평균.",
        )
        st.caption(f"최신 ({latest_label}): {_fmt_ratio(snap['sector_hhi'], 3)}")

    # --- KPI 5: Top 10 비중 (가장 비중 큰 10 종목의 weight 합) ---
    with cols[4]:
        top10_avg = avg.get("top10_avg", np.nan)
        top1_avg = avg.get("top1_avg", np.nan)
        top5_avg = avg.get("top5_avg", np.nan)
        delta_top = ((top10_avg - spy["top10"]) * 100) if not pd.isna(top10_avg) else np.nan
        st.metric(
            label="Top 10 비중",
            value=_fmt_pct(top10_avg, digits=1),
            delta=f"{delta_top:+.1f}%p vs 균등가중" if not pd.isna(delta_top) else None,
            delta_color="inverse",
            help=(
                "**Top 10 종목 weight 합계** — 가장 비중 큰 10개 종목이 portfolio 의 X% 차지. "
                "낮을수록 분산 운용, 높을수록 소수 종목 집중. 메인 = 기간 평균. "
                "**vs 균등가중** = 모든 종목을 1/N (= 1/펀드 universe 종목 수, 약 1/500) 씩 보유한 경우 대비 차이."
            ),
        )
        st.caption(
            f"Top 1 평균: {_fmt_pct(top1_avg, digits=2)} · "
            f"Top 5 평균: {_fmt_pct(top5_avg, digits=1)}"
        )

    # --- KPI 6: Avg Turnover ---
    with cols[5]:
        turnover_avg = avg.get("turnover_avg", np.nan)
        st.metric(
            label="평균 회전율",
            value=_fmt_pct(turnover_avg, digits=1),
            help="월별 회전율 = Σ|w_new - w_prev|. 낮을수록 안정 운용. 메인 = 기간 평균.",
        )
        st.caption(f"최신 ({latest_label}): {_fmt_pct(snap['turnover_latest'], digits=1)}")


# ======================================================================
# 영역 4: Top N Holdings 표 + 비중
# ======================================================================

def render_top_n_table(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    ticker_company_map: dict,
    snapshot_date: pd.Timestamp | None = None,
) -> None:
    """
    Top N Holdings 표 (선택 시점 snapshot, 7컬럼).

    컬럼:
      1. Rank / 2. Ticker (Company) / 3. Sector / 4. Weight (막대)
      5. Market Cap ($B) / 6. 12m Return / 7. ΔWeight (vs 전월)

    Args:
        snapshot_date: 표시할 시점 (None 이면 Latest = 마지막 월).
                       페이지 영역 4 의 시점 슬라이더 값 전달용 (2026-05-12 추가).
    """
    target_date = snapshot_date if snapshot_date is not None else weights.index.max()

    cols_top = st.columns([1.5, 6])
    with cols_top[0]:
        top_n_choice = st.selectbox(
            "Top N",
            options=[10, 20, 50, "All"],
            index=0,
            key="holdings_top_n",
            format_func=lambda x: f"Top {x}" if x != "All" else "All",
        )

    # 선택 시점 weight > 0 종목, 정렬 후 N 개
    w_t = weights.loc[target_date]
    w_t = w_t[w_t > 0].sort_values(ascending=False)
    if top_n_choice != "All":
        w_t = w_t.head(int(top_n_choice))

    tickers = w_t.index.tolist()
    ticker_to_sector = _build_ticker_to_sector(universe)

    # 데이터 산출 (선택 시점 기준)
    mcap = mc.get_market_cap_from_panel(panel, target_date, tickers) / 1e9  # $B
    ret_12m = mc.get_12m_return_from_panel(panel, target_date, tickers)
    delta_w = mc.calc_delta_weight(weights, target_date, tickers)
    holding_m = mc.calc_holding_period(weights, target_date, tickers)  # 최근 연속 보유 (개월)

    df = pd.DataFrame({
        "Rank": range(1, len(tickers) + 1),
        "Ticker": tickers,
        "Company": [ticker_company_map.get(t, t) for t in tickers],
        "Sector": [ticker_to_sector.get(t, "—") for t in tickers],
        "Weight": w_t.values,
        "Market Cap ($B)": mcap.values,
        "12m Return": ret_12m.values,
        "ΔWeight (vs 전월)": delta_w.values,
        "보유 (월)": holding_m.values,
    })

    # printf "%.Nf%%" 는 Python ":.N%" 와 달리 자동 ×100 하지 않음.
    # raw 비율 (0~1) 컬럼들 ×100 적용한 표시용 사본 생성.
    # 원본 df 는 CSV 다운로드 시 raw 값 유지 (계산용 일관성).
    df_display = df.copy()
    pct_cols = ["Weight", "12m Return", "ΔWeight (vs 전월)"]
    for col in pct_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col] * 100

    st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Weight": st.column_config.ProgressColumn(
                "Weight",
                format="%.2f%%",
                min_value=0,
                max_value=float(df_display["Weight"].max()) if len(df_display) > 0 else 100.0,
            ),
            "Market Cap ($B)": st.column_config.NumberColumn(
                "Mcap ($B)", format="%.1f"
            ),
            "12m Return": st.column_config.NumberColumn(
                "12개월 수익률", format="%+.2f%%",
                help="종목 자체의 12개월 가격 추세 (펀드 보유 기간 수익이 아님 — 종목 정보 참고용)",
            ),
            "ΔWeight (vs 전월)": st.column_config.NumberColumn(
                "ΔW (%p)", format="%+.3f"
            ),
            "보유 (월)": st.column_config.NumberColumn(
                "보유 (월)", format="%d",
                help="선택 시점부터 거꾸로 끊김 없이 연속 보유한 개월 수 (1 = 이번 달 새로 편입, 큰 값 = 오래 보유)",
            ),
        },
    )

    # CSV 다운로드 (선택 시점 명시) — raw 비율 (0~1) 그대로 다운로드
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇ CSV 다운로드",
        data=csv,
        file_name=f"holdings_top_{top_n_choice}_{target_date.strftime('%Y-%m')}.csv",
        mime="text/csv",
        key="holdings_top_n_csv",
    )


# ======================================================================
# 영역 5: 시가총액 분포 (Bubble + Treemap)
# ======================================================================

def render_market_cap_distribution(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    ticker_company_map: dict,
) -> None:
    """Bubble + Treemap 탭, 시점 슬라이더."""
    ticker_to_sector = _build_ticker_to_sector(universe)

    # 시점 슬라이더 (월 단위)
    dates = weights.index.tolist()
    date_idx = st.select_slider(
        "시점",
        options=list(range(len(dates))),
        value=len(dates) - 1,  # 기본 = Latest
        format_func=lambda i: dates[i].strftime("%Y-%m"),
        key="holdings_mcap_slider",
    )
    selected_date = dates[date_idx]

    # 해당 시점 데이터 산출
    w = weights.loc[selected_date]
    w = w[w > 0].sort_values(ascending=False)
    tickers = w.index.tolist()

    mcap = mc.get_market_cap_from_panel(panel, selected_date, tickers) / 1e9
    ret_12m = mc.get_12m_return_from_panel(panel, selected_date, tickers)

    df = pd.DataFrame({
        "Ticker": tickers,
        "Company": [ticker_company_map.get(t, t) for t in tickers],
        "Sector": [ticker_to_sector.get(t, "Unknown") for t in tickers],
        "Weight": w.values,
        "Mcap_B": mcap.values,
        "Return_12m": ret_12m.values,
    }).dropna(subset=["Mcap_B"])

    if len(df) == 0:
        st.warning("선택 시점에 시가총액 데이터가 없습니다.")
        return

    tabs = st.tabs(["Treemap", "Bubble Chart"])

    # === Treemap ===
    with tabs[0]:
        col_tm = st.columns([2, 5])
        with col_tm[0]:
            tm_mode = st.selectbox(
                "Treemap 차원",
                options=[
                    "면적=비중 / 색상=섹터",
                    "면적=비중 / 색상=12개월 수익률",
                    "면적=시가총액 / 색상=섹터",
                ],
                index=0,
                key="holdings_treemap_mode",
            )

        if tm_mode.startswith("면적=비중") and "섹터" in tm_mode:
            fig = px.treemap(
                df, path=["Sector", "Ticker"], values="Weight",
                color="Sector", color_discrete_map=SECTOR_COLORS,
                hover_data={"Company": True, "Weight": ":.3%", "Mcap_B": ":.1f"},
            )
        elif tm_mode.startswith("면적=비중"):
            fig = px.treemap(
                df, path=["Sector", "Ticker"], values="Weight",
                color="Return_12m",
                color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                hover_data={"Company": True, "Weight": ":.3%", "Return_12m": ":.2%"},
            )
        else:  # 시가총액
            fig = px.treemap(
                df, path=["Sector", "Ticker"], values="Mcap_B",
                color="Sector", color_discrete_map=SECTOR_COLORS,
                hover_data={"Company": True, "Mcap_B": ":.1f", "Weight": ":.3%"},
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
            font_color=COLORS["text"], margin=dict(t=20, l=0, r=0, b=0), height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Bubble Chart ===
    with tabs[1]:
        cols_b = st.columns(4)
        axis_options = ["Mcap_B", "Weight", "Return_12m"]
        axis_labels = {"Mcap_B": "시가총액 ($B)", "Weight": "비중", "Return_12m": "12개월 수익률"}

        with cols_b[0]:
            x_col = st.selectbox("X축", axis_options, index=0, key="hb_x", format_func=axis_labels.get)
        with cols_b[1]:
            y_col = st.selectbox("Y축", axis_options, index=1, key="hb_y", format_func=axis_labels.get)
        with cols_b[2]:
            size_col = st.selectbox("크기", axis_options, index=0, key="hb_size", format_func=axis_labels.get)
        with cols_b[3]:
            color_col = st.selectbox("색상", ["Sector", "Return_12m"], index=0, key="hb_color")

        # 사이즈 음수/0 방지 (절대값)
        df_b = df.copy()
        df_b["_size"] = df_b[size_col].abs().clip(lower=1e-6)

        scatter_kwargs = dict(
            x=x_col, y=y_col, size="_size",
            hover_name="Ticker",
            hover_data={"Company": True, "Sector": True, "_size": False,
                        "Weight": ":.3%", "Mcap_B": ":.1f", "Return_12m": ":.2%"},
            size_max=40,
        )
        if color_col == "Sector":
            scatter_kwargs["color"] = "Sector"
            scatter_kwargs["color_discrete_map"] = SECTOR_COLORS
        else:
            scatter_kwargs["color"] = "Return_12m"
            scatter_kwargs["color_continuous_scale"] = "RdYlGn"
            scatter_kwargs["color_continuous_midpoint"] = 0

        fig_b = px.scatter(df_b, **scatter_kwargs)
        fig_b.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
            font_color=COLORS["text"],
            xaxis_title=axis_labels[x_col], yaxis_title=axis_labels[y_col],
            height=520,
        )
        st.plotly_chart(fig_b, use_container_width=True)


# ======================================================================
# 영역 6: 보유 종목 변천사 (Multi-line)
# ======================================================================

def render_holdings_history(
    weights: pd.DataFrame,
    universe: pd.DataFrame,
) -> None:
    """
    섹터 단위 Multi-line 변천사: 11 GICS 섹터별 weight 합계 시계열.

    종목 단위 동적 변천은 영역 6.5 (render_top_n_share_timeseries) 에서 보완.
    본 영역은 섹터 narrative 에 집중 — 시간 단위 토글 + Regime 배경 + 이벤트 annotation.
    """
    ticker_to_sector = _build_ticker_to_sector(universe)

    cols_t = st.columns([1.5, 5])
    with cols_t[0]:
        time_mode = st.selectbox(
            "시간 단위", options=["월별", "분기별"], index=0, key="holdings_time_mode",
        )

    # 시간 단위 변환
    weights_resampled = weights.resample("QE").last() if time_mode == "분기별" else weights

    # 섹터별 weight 합계 시계열
    series_dict: dict[str, pd.Series] = {}
    for sector in sorted(set(ticker_to_sector.values())):
        sector_tickers = [t for t, s in ticker_to_sector.items() if s == sector]
        sector_tickers = [t for t in sector_tickers if t in weights_resampled.columns]
        if not sector_tickers:
            continue
        series_dict[sector] = weights_resampled[sector_tickers].sum(axis=1)

    # === Plotly figure ===
    fig = go.Figure()
    for label, series in series_dict.items():
        color = SECTOR_COLORS.get(label, None)
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name=label,
            line=dict(color=color) if color else dict(),
            hovertemplate=f"%{{x|%Y-%m}}<br>{label}: %{{y:.2%}}<extra></extra>",
        ))

    fig = add_regime_backgrounds(fig, with_labels=False)
    fig = add_event_annotations(fig)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        xaxis_title="시점", yaxis_title="섹터 weight 합",
        yaxis=dict(tickformat=".0%"),
        height=480, hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# 영역 6.5: 시점별 Top N 합 vs Others 시계열 (시점별 동적 — 영역 6 보완)
# ======================================================================

def render_top_n_share_timeseries(
    comp: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    """
    각 시점별 Top N 합 (시점별 동적 ticker) + Others 100%-stacked area 시계열.

    영역 6 vs 본 영역의 차이:
      - 영역 6 = 선택 시점 Top N ticker 의 전 기간 weight (정적 ticker, 동적 weight)
      - 본 영역 = 각 시점의 Top N 합 (동적 ticker, 시점별 합)
      → 펀드 집중도의 시점별 동적 추세 (final.comp.top10_share 직접 정합)

    100%-stacked area: Top N (아래) + Others (위) 누적 = 100%, 경계선이 Top N 합.

    Args:
        comp: pd.DataFrame (final 산출 — top1_weight, top10_share 활용)
        weights: pd.DataFrame (Top 5 / 20 직접 산출 시 fallback)
    """
    cols_t = st.columns([1.5, 5])
    with cols_t[0]:
        n = st.selectbox(
            "Top N",
            options=[1, 5, 10, 20],
            index=2,  # default 10
            key="holdings_topn_share",
            help="N 별 시점별 합 시계열 — 펀드 집중도 동적 추세",
        )

    # 시점별 Top N 합 산출 — final 정합 우선, 부재 시 직접 산출
    if n == 10 and "top10_share" in comp.columns:
        top_share = comp["top10_share"]  # final 직접 사용
        source_label = "final.comp.top10_share"
    elif n == 1 and "top1_weight" in comp.columns:
        top_share = comp["top1_weight"]
        source_label = "final.comp.top1_weight"
    else:
        # 직접 산출 (Top 5 / 20)
        top_share = pd.Series({
            t: float(weights.loc[t][weights.loc[t] > 0].nlargest(n).sum())
            for t in weights.index
        })
        source_label = f"직접 산출 — 각 시점 Top {n} weight 합"

    others = 1 - top_share

    # 시점별 Top N 종목 list + weight (hover 정보)
    hover_text_per_t: list[str] = []
    for t in weights.index:
        row = weights.loc[t]
        row = row[row > 0].sort_values(ascending=False).head(n)
        if len(row) == 0:
            hover_text_per_t.append("")
            continue
        lines = [f"  {i+1}. <b>{tk}</b>: {w * 100:.2f}%"
                 for i, (tk, w) in enumerate(row.items())]
        hover_text_per_t.append("<br>".join(lines))

    # 100%-stacked area — 두 면적 누적, 경계선이 Top N 합을 자연 표현
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=top_share.index, y=top_share.values,
        name=f"Top {n} 합",
        stackgroup="one",
        line=dict(color=COLORS["primary"], width=2),
        fillcolor="rgba(59, 130, 246, 0.55)",  # primary blue 반투명
        customdata=hover_text_per_t,
        hovertemplate=(
            f"<b>%{{x|%Y-%m}} — Top {n} 합: %{{y:.2%}}</b>"
            "<br>%{customdata}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=others.index, y=others.values,
        name="Others",
        stackgroup="one",
        line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
        fillcolor="rgba(156, 163, 175, 0.30)",  # text_muted 반투명
        hovertemplate="Others: %{y:.2%}<extra></extra>",
    ))

    fig = add_regime_backgrounds(fig, with_labels=False)
    fig = add_event_annotations(fig)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        xaxis_title="시점",
        yaxis_title="시점별 weight 누적 (Top N + Others = 100%)",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, l=0, r=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"데이터 출처: {source_label}")


# ======================================================================
# 영역 7: 종목별 기여도 분석 (Tornado Chart)
# ======================================================================

def render_attribution_tornado(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    ticker_company_map: dict,
    fund_ret: pd.Series,
    period: str,
) -> None:
    """
    Attribution Tornado — Simple (Brinson 1986) / Carino (1999) 토글.
      - Simple: Σ(w × R) — 단일 기간 선형 합 (multi-period 누적과 차이 큼)
      - Carino Smoothed: log smoothing → Σ = Fund 누적 수익률 정확 일치
    """
    cols_t = st.columns([1.5, 2, 3])
    with cols_t[0]:
        top_n = st.selectbox(
            "Top N", options=[10, 20, 50], index=0, key="holdings_attr_top_n",
        )
    with cols_t[1]:
        method = st.selectbox(
            "기여도 계산 방식",
            options=["단순 합 (월별 합산)", "복리 보정 (장기 일치)"],
            index=1,  # default = 복리 보정 (학술 표준)
            key="holdings_attr_method",
            help=(
                "**단순 합 (월별 합산)** = 매월 (weight × return) 을 단순히 더함. "
                "직관적이지만 다기간 누적 시 복리 효과를 반영하지 못해 펀드 전체 수익률과 차이. "
                "(Brinson 1986)\n\n"
                "**복리 보정 (장기 일치)** = log smoothing 으로 다기간 복리 효과 반영. "
                "전체 합계 = 펀드 누적 수익률과 정확히 일치. "
                "(Carino 1999, multi-period linking 학술 표준)"
            ),
        )
    is_carino = method.startswith("복리")

    # 기간 boundary
    period_label_map = {"FULL": "FULL 192개월", "TEST": "TEST 168개월", "HO": "Hold Out 24개월"}
    if period == "TEST":
        s, e = EVAL_PERIODS["TEST"]
    elif period == "HO":
        s, e = EVAL_PERIODS["HOLD_OUT"]
    else:
        s, e = EVAL_PERIODS["FULL"]

    period_start = pd.Timestamp(s)
    period_end = pd.Timestamp(e)

    # 종목별 contribution (선택 method 기준)
    if is_carino:
        contrib = mc.calc_carino_smoothed_contribution(weights, panel, period_start, period_end)
    else:
        contrib = mc.calc_simple_contribution(weights, panel, period_start, period_end)
    if len(contrib) == 0:
        st.warning("기여도 산출 불가 (기간 내 데이터 없음).")
        return

    # 펀드 누적 수익률 (기간) — 검증용
    fund_p = fund_ret[(fund_ret.index >= period_start) & (fund_ret.index <= period_end)]
    fund_cum = float((1 + fund_p).prod() - 1) if len(fund_p) > 0 else np.nan
    contrib_sum = float(contrib.sum())

    # 산출 portfolio R_t 의 누적 — Carino 의 정합 비교 기준
    panel_p_pivot = panel[(panel["date"] >= period_start) & (panel["date"] <= period_end)] \
        .pivot_table(index="date", columns="ticker", values="ret_1m", aggfunc="first")
    common_dates = weights.index.intersection(panel_p_pivot.index)
    common_dates = common_dates[(common_dates >= period_start) & (common_dates <= period_end)]
    if len(common_dates) > 0:
        common_tickers = weights.columns.intersection(panel_p_pivot.columns)
        contrib_t_check = (weights.loc[common_dates, common_tickers]
                           * panel_p_pivot.loc[common_dates, common_tickers]).fillna(0)
        calc_R_cum = float((1 + contrib_t_check.sum(axis=1)).prod() - 1)
    else:
        calc_R_cum = np.nan

    # Top N 양수 / Top N 음수
    pos = contrib[contrib > 0].head(top_n)
    neg = contrib[contrib < 0].tail(top_n).iloc[::-1]  # 최하위 부터

    ticker_to_sector = _build_ticker_to_sector(universe)

    # === Tornado: 두 컬럼 ===
    cols_chart = st.columns(2)

    with cols_chart[0]:
        st.markdown(f"**Top {top_n} Contributors (양수)**")
        if len(pos) == 0:
            st.info("양수 기여 종목 없음")
        else:
            df_pos = pd.DataFrame({
                "Ticker": pos.index,
                "Company": [ticker_company_map.get(t, t) for t in pos.index],
                "Sector": [ticker_to_sector.get(t, "—") for t in pos.index],
                "Contribution": pos.values,
            })
            fig = go.Figure(go.Bar(
                x=df_pos["Contribution"], y=df_pos["Ticker"], orientation="h",
                marker_color=COLORS["accent_green"],
                text=[f"{v:+.2%}" for v in df_pos["Contribution"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>" +
                              "Company: " + df_pos["Company"] + "<br>" +
                              "Sector: " + df_pos["Sector"] + "<br>" +
                              "Contribution: %{x:+.3%}<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
                font_color=COLORS["text"],
                xaxis=dict(tickformat=".1%"), yaxis=dict(autorange="reversed"),
                height=400, margin=dict(t=20, l=0, r=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with cols_chart[1]:
        st.markdown(f"**Bottom {top_n} Contributors (음수)**")
        if len(neg) == 0:
            st.info("음수 기여 종목 없음")
        else:
            df_neg = pd.DataFrame({
                "Ticker": neg.index,
                "Company": [ticker_company_map.get(t, t) for t in neg.index],
                "Sector": [ticker_to_sector.get(t, "—") for t in neg.index],
                "Contribution": neg.values,
            })
            fig = go.Figure(go.Bar(
                x=df_neg["Contribution"], y=df_neg["Ticker"], orientation="h",
                marker_color=COLORS["accent_red"],
                text=[f"{v:+.2%}" for v in df_neg["Contribution"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>" +
                              "Company: " + df_neg["Company"] + "<br>" +
                              "Sector: " + df_neg["Sector"] + "<br>" +
                              "Contribution: %{x:+.3%}<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
                font_color=COLORS["text"],
                xaxis=dict(tickformat=".1%"), yaxis=dict(autorange="reversed"),
                height=400, margin=dict(t=20, l=0, r=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # === Sector 합계 ===
    sector_contrib = (
        contrib.groupby([ticker_to_sector.get(t, "Unknown") for t in contrib.index])
        .sum()
        .sort_values(ascending=False)
    )
    st.markdown("**Sector 합계** (보조)")
    sector_cols = st.columns(min(len(sector_contrib), 6))
    for i, (sector, val) in enumerate(sector_contrib.head(6).items()):
        with sector_cols[i % len(sector_cols)]:
            color = COLORS["accent_green"] if val > 0 else COLORS["accent_red"]
            st.markdown(
                f"<div style='padding:6px;border-radius:4px;"
                f"background:{COLORS['secondary_bg']};text-align:center'>"
                f"<div style='font-size:0.8em;color:{COLORS['text_muted']}'>{sector}</div>"
                f"<div style='font-size:1.2em;color:{color};font-weight:bold'>{val:+.2%}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # === 검증 (method 별 다른 narrative) ===
    st.markdown("---")
    if is_carino:
        diff_calc = contrib_sum - calc_R_cum if not pd.isna(calc_R_cum) else np.nan
        st.caption(
            f"**검증** (기간: {period_label_map.get(period, period)}, **복리 보정 방식 — 장기 일치**)"
        )
        st.caption(
            f"  • Σ 종목별 기여도 = **{contrib_sum:+.2%}**"
        )
        st.caption(
            f"  • 산출 portfolio 누적 수익률 = **{calc_R_cum:+.2%}** "
            f"(차이: {diff_calc:+.6%} — **수학적 정확 일치**)"
        )
        st.caption(
            f"  • 참고 — Fund 실제 Net 누적 = {fund_cum:+.2%} "
            f"(산출 누적과의 차이는 panel 데이터 vs 펀드 내부 수익률 차이 + 거래비용 누적 — 기여도 분해 범위 외)"
        )
    else:
        diff_fund = contrib_sum - fund_cum if not pd.isna(fund_cum) else np.nan
        st.caption(
            f"**검증** (기간: {period_label_map.get(period, period)}, **단순 합 방식 — 월별 합산**): "
            f"Σ 종목별 기여도 = **{contrib_sum:+.2%}** vs "
            f"Fund 누적 수익률 = **{fund_cum:+.2%}**  "
            f"(차이: {diff_fund:+.2%} — 단순 합은 복리 효과를 반영하지 못해 장기 누적과 차이가 큼. "
            f"정확한 분해를 위해서는 위 토글에서 **복리 보정 (장기 일치)** 선택)"
        )

    # CSV 다운로드 (전체 contribution)
    method_suffix = "carino" if is_carino else "simple"
    csv_df = pd.DataFrame({
        "Ticker": contrib.index,
        "Company": [ticker_company_map.get(t, t) for t in contrib.index],
        "Sector": [ticker_to_sector.get(t, "—") for t in contrib.index],
        f"Contribution_{method_suffix}": contrib.values,
    })
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇ 전체 기여도 CSV",
        data=csv_bytes,
        file_name=f"holdings_attribution_{method_suffix}_{period}.csv",
        mime="text/csv",
        key="holdings_attr_csv",
    )
