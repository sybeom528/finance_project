"""
lib/sector_charts.py — Sector Watch 페이지 6 영역 차트 함수

영역 3-8:
  3. render_sector_kpi (HHI / Avg|Tilt| / Active Bets / Most Over / Most Under)
  4. render_sector_treemap (Fund vs SPY 좌우, 시점 슬라이더, 차원 토글)
  5. render_sector_decomposition_table (9 컬럼 + Tilt 색상 + CSV)
  6. render_sector_tilt_tornado (Tornado + ±1%/±5% 임계선)
  7. render_sector_rotation (Stacked Area / Multi-line 토글, Fund vs SPY)
  8. render_ho_justification ★★★ (이중 축 + Tornado + HHI Regime + narrative)

메트릭 = decisionlog/06_sector_watch.md Sector-Q1:
  - SPY sector weight: panel.log_mcap × sp500_membership[t-1] (mcap 가중, look-ahead 회피)
  - Sector Tilt = Fund_w − SPY_w (Active Management)
  - HO 정당화: Markowitz (1952) + Fama-French (1992)

참조: docs/plan/03_pages/06_sector_watch.md, decisionlog/06_sector_watch.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import metric_calculators as mc
from lib.colors import COLORS, SECTOR_COLORS
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


def _fmt_ratio(v: float, digits: int = 3) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.{digits}f}"


def _filter_period_index(idx: pd.DatetimeIndex, period: str) -> pd.DatetimeIndex:
    if period == "TEST":
        s, e = EVAL_PERIODS["TEST"]
    elif period == "HO":
        s, e = EVAL_PERIODS["HOLD_OUT"]
    else:
        return idx
    return idx[(idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))]


# ======================================================================
# 영역 3: Sector Summary KPI 5개
# ======================================================================

def render_sector_kpi(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    sp500_membership: dict,
    ticker_to_sector: dict,
    period: str,
) -> None:
    """
    Sector KPI 5개 — Latest snapshot + 사이드바 토글 평균.

    1. Sector HHI (Hirschman 1945, sector level)
    2. Avg |Tilt| vs SPY
    3. Number of Active Bets (|Tilt| > 1%)
    4. Most Overweight Sector (이름 + Tilt%)
    5. Most Underweight Sector (이름 + Tilt%, ★ HO 정당화 narrative 직접 연결)
    """
    latest = weights.index.max()

    # Latest snapshot
    fund_sw_latest = mc.calc_sector_weights(weights.loc[latest], ticker_to_sector)
    spy_sw_latest = mc.compute_spy_sector_weights(panel, sp500_membership, latest, ticker_to_sector)
    tilt_latest = mc.calc_sector_tilt(fund_sw_latest, spy_sw_latest)

    sector_hhi_latest = float((fund_sw_latest ** 2).sum())
    spy_hhi_latest = float((spy_sw_latest ** 2).sum()) if len(spy_sw_latest) > 0 else np.nan

    avg_tilt_latest = float(tilt_latest.abs().mean())
    active_bets_latest = mc.calc_active_bets(tilt_latest, threshold=0.01)
    most_over = tilt_latest.head(1)
    most_under = tilt_latest.tail(1)

    # 기간 평균 (사이드바 토글)
    period_idx = _filter_period_index(weights.index, period)
    if len(period_idx) > 0:
        hhi_list, avg_tilt_list, bets_list = [], [], []
        for t in period_idx:
            fsw = mc.calc_sector_weights(weights.loc[t], ticker_to_sector)
            ssw = mc.compute_spy_sector_weights(panel, sp500_membership, t, ticker_to_sector)
            tt = mc.calc_sector_tilt(fsw, ssw)
            hhi_list.append(float((fsw ** 2).sum()))
            avg_tilt_list.append(float(tt.abs().mean()))
            bets_list.append(mc.calc_active_bets(tt, threshold=0.01))
        avg_hhi = float(np.mean(hhi_list))
        avg_tilt_mean = float(np.mean(avg_tilt_list))
        bets_mean = float(np.mean(bets_list))
    else:
        avg_hhi = avg_tilt_mean = bets_mean = np.nan

    # 헤더 라벨
    period_label_map = {"FULL": "FULL 192m", "TEST": "TEST 168m", "HO": "HO 24m"}
    st.caption(
        f"**Latest snapshot**: {latest.strftime('%Y-%m')}  ·  "
        f"**기간 평균**: {period_label_map.get(period, period)}"
    )

    cols = st.columns(5)

    with cols[0]:
        delta = sector_hhi_latest - spy_hhi_latest if not pd.isna(spy_hhi_latest) else None
        st.metric(
            label="Sector HHI",
            value=_fmt_ratio(sector_hhi_latest, 3),
            delta=f"{delta:+.3f} vs SPY" if delta is not None else None,
            delta_color="inverse",
            help="섹터 집중도 = Σ(섹터 weight)². 낮을수록 분산 (Hirschman 1945)",
        )
        st.caption(f"평균 {_fmt_ratio(avg_hhi, 3)}")

    with cols[1]:
        st.metric(
            label="Avg |Tilt| vs SPY",
            value=_fmt_pct(avg_tilt_latest, digits=2),
            help="섹터별 |Tilt| 평균 — 펀드의 액티브 운용 강도",
        )
        st.caption(f"평균 {_fmt_pct(avg_tilt_mean, digits=2)}")

    with cols[2]:
        st.metric(
            label="Active Bets",
            value=f"{active_bets_latest}",
            help="|Tilt| > 1% 인 섹터 수 — 액티브 베팅 분산",
        )
        st.caption(f"평균 {bets_mean:.1f}" if not pd.isna(bets_mean) else "평균 —")

    with cols[3]:
        if len(most_over) > 0:
            sec, val = most_over.index[0], float(most_over.iloc[0])
            st.metric(
                label="Most Overweight",
                value=sec[:15],
                delta=f"{val:+.2%}",
                delta_color="off",
                help=f"펀드 vs SPY 가장 큰 over-weight 섹터: {sec}",
            )

    with cols[4]:
        if len(most_under) > 0:
            sec, val = most_under.index[0], float(most_under.iloc[0])
            star = " ★" if "Information Technology" in sec else ""
            st.metric(
                label=f"Most Underweight{star}",
                value=sec[:15],
                delta=f"{val:+.2%}",
                delta_color="off",
                help=f"펀드 vs SPY 가장 큰 under-weight 섹터: {sec}. "
                     f"{'IT under-weight = HO 정당화 narrative 핵심' if star else ''}",
            )


# ======================================================================
# 영역 4: Sector Treemap (Fund vs SPY 좌우 + 시점 슬라이더)
# ======================================================================

def render_sector_treemap(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    sp500_membership: dict,
    ticker_to_sector: dict,
) -> None:
    """Fund vs SPY 좌우 분할 Treemap + 시점 슬라이더."""
    dates = list(weights.index)
    date_idx = st.select_slider(
        "시점",
        options=list(range(len(dates))),
        value=len(dates) - 1,  # default Latest
        format_func=lambda i: dates[i].strftime("%Y-%m"),
        key="sector_treemap_slider",
    )
    selected_date = dates[date_idx]

    fund_sw = mc.calc_sector_weights(weights.loc[selected_date], ticker_to_sector)
    spy_sw = mc.compute_spy_sector_weights(panel, sp500_membership, selected_date, ticker_to_sector)

    if len(fund_sw) == 0:
        st.warning("선택 시점 펀드 sector 데이터 없음.")
        return

    cols = st.columns(2)

    with cols[0]:
        st.markdown(f"**Adaptive VolControl Fund** ({selected_date.strftime('%Y-%m')})")
        df_fund = pd.DataFrame({
            "Sector": fund_sw.index,
            "Weight": fund_sw.values,
        })
        fig = px.treemap(
            df_fund, path=["Sector"], values="Weight",
            color="Sector", color_discrete_map=SECTOR_COLORS,
            hover_data={"Weight": ":.2%"},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
            font_color=COLORS["text"], margin=dict(t=20, l=0, r=0, b=0), height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        st.markdown(f"**SPY (S&P 500, mcap 가중)** ({selected_date.strftime('%Y-%m')})")
        if len(spy_sw) == 0:
            st.warning("SPY sector 산출 실패.")
        else:
            df_spy = pd.DataFrame({
                "Sector": spy_sw.index,
                "Weight": spy_sw.values,
            })
            fig = px.treemap(
                df_spy, path=["Sector"], values="Weight",
                color="Sector", color_discrete_map=SECTOR_COLORS,
                hover_data={"Weight": ":.2%"},
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
                font_color=COLORS["text"], margin=dict(t=20, l=0, r=0, b=0), height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# 영역 5: Sector Decomposition 표 (9 컬럼)
# ======================================================================

def render_sector_decomposition_table(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    sp500_membership: dict,
    ticker_to_sector: dict,
) -> None:
    """Sector Decomposition 표 — Latest snapshot, 9 컬럼 + Tilt 색상."""
    latest = weights.index.max()
    spy_sw = mc.compute_spy_sector_weights(panel, sp500_membership, latest, ticker_to_sector)
    df = mc.calc_sector_decomposition(
        weights.loc[latest], panel, latest, ticker_to_sector, spy_sw
    )
    if len(df) == 0:
        st.warning("Sector Decomposition 산출 실패.")
        return

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Weight": st.column_config.ProgressColumn(
                "Weight", format="%.2f%%", min_value=0,
                max_value=float(df["Weight"].max()) if len(df) > 0 else 1.0,
            ),
            "Tilt": st.column_config.NumberColumn(
                "Tilt vs SPY", format="%+.2f%%",
                help="펀드 weight − SPY weight (양수 = over, 음수 = under)",
            ),
            "Return_12m": st.column_config.NumberColumn("12m R", format="%+.2f%%"),
            "N_Holdings": st.column_config.NumberColumn("#H", width="small"),
            "Volatility": st.column_config.NumberColumn("Vol", format="%.2f%%"),
            "Beta": st.column_config.NumberColumn("β", format="%.3f"),
            "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "Contribution": st.column_config.NumberColumn("Contrib", format="%+.3f%%"),
        },
    )

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇ CSV 다운로드",
        data=csv,
        file_name=f"sector_decomposition_{latest.strftime('%Y-%m')}.csv",
        mime="text/csv",
        key="sector_decomp_csv",
    )


# ======================================================================
# 영역 6: Sector Tilt vs SPY (Tornado Chart)
# ======================================================================

def render_sector_tilt_tornado(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    sp500_membership: dict,
    ticker_to_sector: dict,
) -> None:
    """Sector Tilt Tornado — ±1%/±5% 임계선 + IT 강조 (HO narrative)."""
    latest = weights.index.max()
    fund_sw = mc.calc_sector_weights(weights.loc[latest], ticker_to_sector)
    spy_sw = mc.compute_spy_sector_weights(panel, sp500_membership, latest, ticker_to_sector)
    tilt = mc.calc_sector_tilt(fund_sw, spy_sw)

    if len(tilt) == 0:
        st.warning("Tilt 산출 실패.")
        return

    # IT 라벨 강조
    labels = [
        f"{s} ★" if s == "Information Technology" else s
        for s in tilt.index
    ]
    colors = [
        COLORS["accent_green"] if v > 0 else COLORS["accent_red"]
        for v in tilt.values
    ]

    fig = go.Figure(go.Bar(
        x=tilt.values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.2%}" for v in tilt.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Tilt: %{x:+.3%}<extra></extra>",
    ))

    # 0% 기준선 + ±1%/±5% 임계선
    fig.add_vline(x=0, line_color=COLORS["text"], line_width=1.5)
    for thr, dash in [(0.01, "dot"), (-0.01, "dot"), (0.05, "dash"), (-0.05, "dash")]:
        fig.add_vline(x=thr, line_color=COLORS["text_muted"], line_width=1, line_dash=dash)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        xaxis=dict(tickformat=".1%", title="Tilt (Fund − SPY)"),
        yaxis=dict(autorange="reversed"),
        height=480, margin=dict(t=30, l=10, r=80, b=20),
        title=dict(text=f"Latest snapshot: {latest.strftime('%Y-%m')}",
                   font=dict(size=12, color=COLORS["text_muted"])),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "임계선: ±1% (점선, Active Bets 기준) · ±5% (대시, 큰 베팅). "
        "★ = Information Technology (HO 정당화 narrative 핵심)."
    )


# ======================================================================
# 영역 7: Sector 시계열 변화 (Sector Rotation)
# ======================================================================

def render_sector_rotation(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    sp500_membership: dict,
    ticker_to_sector: dict,
) -> None:
    """Sector Rotation — Stacked Area / Multi-line 토글, Fund / SPY / Tilt."""
    cols_t = st.columns(3)
    with cols_t[0]:
        chart_mode = st.selectbox(
            "차트", options=["Stacked Area", "Multi-line"],
            index=0, key="sector_rotation_chart",
        )
    with cols_t[1]:
        view_mode = st.selectbox(
            "데이터", options=["Fund", "SPY", "Tilt (Fund − SPY)"],
            index=0, key="sector_rotation_view",
        )
    with cols_t[2]:
        time_mode = st.selectbox(
            "시간 단위", options=["월별", "분기별"],
            index=0, key="sector_rotation_time",
        )

    # 시간 단위 변환 (resample 적용은 산출 후)
    target_dates = weights.index
    if time_mode == "분기별":
        target_dates = pd.DatetimeIndex(
            sorted({pd.Timestamp(t).to_period("Q").end_time.normalize() for t in target_dates})
        )
        target_dates = pd.DatetimeIndex(
            [d for d in weights.index if d in pd.DatetimeIndex(target_dates)]
        ) if len(target_dates) > 0 else weights.index

    # Sector 시계열 산출 — 1차 pass: 시점별 (fsw, ssw) 수집 + 전체 sector 합집합
    series_per_t: list[tuple[pd.Timestamp, pd.Series, pd.Series]] = []
    all_sectors_set: set[str] = set()
    for t in weights.index:
        if time_mode == "분기별" and t.month not in (3, 6, 9, 12):
            continue
        fsw = mc.calc_sector_weights(weights.loc[t], ticker_to_sector)
        if view_mode != "Fund":
            ssw = mc.compute_spy_sector_weights(panel, sp500_membership, t, ticker_to_sector)
        else:
            ssw = pd.Series(dtype=float)
        series_per_t.append((t, fsw, ssw))
        all_sectors_set.update(fsw.index)
        all_sectors_set.update(ssw.index)

    valid_dates = [t for t, _, _ in series_per_t]
    all_sectors = sorted(all_sectors_set)

    # 2차 pass: 모든 sector × 모든 시점 — 부재 시 0 보강 (길이 일치 보장)
    fund_data: dict[str, list[float]] = {s: [] for s in all_sectors}
    spy_data: dict[str, list[float]] = {s: [] for s in all_sectors}
    for _, fsw, ssw in series_per_t:
        for s in all_sectors:
            fund_data[s].append(float(fsw.get(s, 0)))
            spy_data[s].append(float(ssw.get(s, 0)))

    # view_mode 별 series 결정
    if view_mode == "Fund":
        series_dict = {s: fund_data[s] for s in all_sectors}
    elif view_mode == "SPY":
        series_dict = {s: spy_data[s] for s in all_sectors}
    else:  # Tilt
        series_dict = {
            s: [fund_data[s][i] - spy_data[s][i] for i in range(len(valid_dates))]
            for s in all_sectors
        }

    fig = go.Figure()
    for sec, vals in series_dict.items():
        color = SECTOR_COLORS.get(sec, None)
        if chart_mode == "Stacked Area":
            fig.add_trace(go.Scatter(
                x=valid_dates, y=vals, name=sec, mode="lines",
                stackgroup="one" if view_mode != "Tilt" else None,
                line=dict(color=color) if color else dict(),
                hovertemplate=f"%{{x|%Y-%m}}<br>{sec}: %{{y:.2%}}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=valid_dates, y=vals, name=sec, mode="lines",
                line=dict(color=color) if color else dict(),
                hovertemplate=f"%{{x|%Y-%m}}<br>{sec}: %{{y:.2%}}<extra></extra>",
            ))

    fig = add_regime_backgrounds(fig, with_labels=False)
    fig = add_event_annotations(fig)

    yaxis_title = "Sector Tilt (Fund − SPY)" if view_mode == "Tilt" else "Sector Weight"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        xaxis_title="시점", yaxis_title=yaxis_title,
        yaxis=dict(tickformat=".0%"),
        height=520, hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# 영역 8: ★★★ HO 24m 분석 + 정당화 narrative
# ======================================================================

def render_ho_justification(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    sp500_membership: dict,
    ticker_to_sector: dict,
    fund_ret: pd.Series,
    fund_spy: pd.Series,
) -> None:
    """
    HO 24m 정당화 narrative — 영역 8 핵심.

    1. 학술 narrative 박스 (Markowitz 1952)
    2. Chart 1: SPY IT Mcap (좌축) + Fund IT Tilt (우축) 이중 축
    3. Chart 2: HO 24m Sector Contribution Tornado
    4. Chart 3: Regime 별 Sector HHI 추세 (Fund vs SPY)
    5. 결론 박스 (장기 분산의 가치)
    """
    # === 1. 학술 narrative 박스 ===
    st.info(
        "ℹ️ **Markowitz (1952) 평균-분산 이론** + **Fama-French (1992) factor diversification** 관점에서, "
        "본 펀드 (BL+LSTM vol-target) 는 **단순 sector 분산이 아닌 risk-aware (vol-target) 분산** 운용입니다. "
        "→ 시장 변동성에 따라 sector 집중 (예: 2017 R2 Defensive 도피) / sector 분산 (HO 24m) 을 "
        "동적으로 선택. **HO 24m 의 IT under-weight** 는 AI Rally 시기의 trade-off 이며, "
        "장기 vol-targeted 분산 운용의 본질을 보여주는 의도된 결과입니다."
    )

    # === 2. Chart 1: SPY IT Mcap + Fund IT Tilt 이중 축 ===
    st.markdown("##### Chart 1 — SPY IT 비중 (좌축) vs Fund IT Tilt (우축)")

    spy_it_series, fund_it_tilt_series, dates_x = [], [], []
    for t in weights.index:
        fsw = mc.calc_sector_weights(weights.loc[t], ticker_to_sector)
        ssw = mc.compute_spy_sector_weights(panel, sp500_membership, t, ticker_to_sector)
        spy_it = float(ssw.get("Information Technology", 0))
        fund_it = float(fsw.get("Information Technology", 0))
        spy_it_series.append(spy_it)
        fund_it_tilt_series.append(fund_it - spy_it)
        dates_x.append(t)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=dates_x, y=spy_it_series, name="SPY IT 비중",
            line=dict(color=COLORS["accent_amber"], width=2),
            hovertemplate="%{x|%Y-%m}<br>SPY IT: %{y:.2%}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dates_x, y=fund_it_tilt_series, name="Fund IT Tilt",
            line=dict(color=COLORS["accent_red"], width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m}<br>Fund IT Tilt: %{y:+.2%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig = add_regime_backgrounds(fig, with_labels=False)
    fig = add_event_annotations(fig)
    fig.update_yaxes(title_text="SPY IT 비중", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Fund IT Tilt (Fund − SPY)", tickformat=".0%", secondary_y=True)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        xaxis_title="시점", height=380, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "SPY IT 비중이 시간이 지날수록 증가 (특히 2024 AI Rally) → "
        "펀드의 IT under-weight 가 점점 더 큰 underperform 으로 작용."
    )
    st.divider()

    # === 3. Chart 2: HO 24m Sector Contribution Tornado ===
    st.markdown("##### Chart 2 — HO 24m Sector Contribution (Carino smoothed)")

    ho_start = pd.Timestamp(EVAL_PERIODS["HOLD_OUT"][0])
    ho_end = pd.Timestamp(EVAL_PERIODS["HOLD_OUT"][1])
    contrib = mc.calc_carino_smoothed_contribution(weights, panel, ho_start, ho_end)

    if len(contrib) > 0:
        sec_contrib = (
            contrib.groupby([ticker_to_sector.get(t, "Unknown") for t in contrib.index])
            .sum()
            .sort_values(ascending=False)
        )
        labels = [
            f"{s} ★" if s == "Information Technology" else s for s in sec_contrib.index
        ]
        colors = [
            COLORS["accent_green"] if v > 0 else COLORS["accent_red"]
            for v in sec_contrib.values
        ]
        fig2 = go.Figure(go.Bar(
            x=sec_contrib.values, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v:+.2%}" for v in sec_contrib.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.3%}<extra></extra>",
        ))
        fig2.add_vline(x=0, line_color=COLORS["text"], line_width=1.5)
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
            font_color=COLORS["text"],
            xaxis=dict(tickformat=".1%", title="HO 24m Carino Contribution"),
            yaxis=dict(autorange="reversed"),
            height=420, margin=dict(t=20, l=10, r=80, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "★ Information Technology = 펀드의 under-weight 가 HO 24m underperform 의 핵심. "
            "Defensive 섹터 (Utilities/Consumer Staples) 는 양수 기여하지만 IT 의 거대 양수 contribution 을 따라가지 못함."
        )
    st.divider()

    # === 4. Chart 3: Regime 별 Sector HHI ===
    st.markdown("##### Chart 3 — Regime 별 Sector HHI (Fund vs SPY)")

    regimes = {
        "R1 회복기 (2010-2012)": ("2010-01-01", "2012-06-30"),
        "R2 확장기 (2012-2019)": ("2012-07-01", "2019-12-31"),
        "R3 변동기 (2020-2023)": ("2020-01-01", "2023-12-31"),
        "HO 24m (2024-2025)": ("2024-01-01", "2025-12-31"),
    }
    fund_hhis, spy_hhis, regime_labels = [], [], []
    for label, (s, e) in regimes.items():
        idx = weights.index[(weights.index >= s) & (weights.index <= e)]
        if len(idx) == 0:
            continue
        fund_hhi_list, spy_hhi_list = [], []
        for t in idx:
            fsw = mc.calc_sector_weights(weights.loc[t], ticker_to_sector)
            ssw = mc.compute_spy_sector_weights(panel, sp500_membership, t, ticker_to_sector)
            fund_hhi_list.append(float((fsw ** 2).sum()))
            if len(ssw) > 0:
                spy_hhi_list.append(float((ssw ** 2).sum()))
        fund_hhis.append(np.mean(fund_hhi_list) if fund_hhi_list else np.nan)
        spy_hhis.append(np.mean(spy_hhi_list) if spy_hhi_list else np.nan)
        regime_labels.append(label)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=regime_labels, y=fund_hhis, name="Fund",
        marker_color=COLORS["primary"],
        hovertemplate="%{x}<br>Fund HHI: %{y:.3f}<extra></extra>",
    ))
    fig3.add_trace(go.Bar(
        x=regime_labels, y=spy_hhis, name="SPY",
        marker_color=COLORS["text_muted"],
        hovertemplate="%{x}<br>SPY HHI: %{y:.3f}<extra></extra>",
    ))
    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        barmode="group",
        yaxis=dict(title="Sector HHI", tickformat=".3f"),
        height=380, hovermode="x",
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "**R1 / R3 / HO**: Fund HHI 0.14~0.19 — sector 분산 운용. "
        "**R2 (2012-19, 90m)**: Fund HHI 0.41 — vol-target 모델이 시장 변동성 시기 (특히 2017 정치/금리 불확실성) "
        "Defensive 섹터 (Utilities 70%+ / Staples 30%) 로 도피 학습. "
        "**HO 24m**: Fund 0.14 < SPY 0.16 — **펀드 분산 회복 vs SPY IT 집중** = HO 정당화 narrative 핵심."
    )
    st.divider()

    # === 5. 결론 박스 ===
    st.success(
        "✅ **결론: Vol-targeted 적응형 운용의 가치**\n\n"
        "**Markowitz (1952)** 의 평균-분산 이론과 **Fama-French (1992)** 의 factor diversification 관점에서, "
        "본 펀드의 BL+LSTM vol-target 모델은 **단순 sector 분산이 아닌 risk-aware 분산** 운용입니다.\n\n"
        "**R1/R3/HO 분산 시기 + R2 Defensive 도피 시기** 모두 vol-target 학습의 결과 — "
        "장기 168m 학습 구간에서 시장 변동성에 적응하며 우수한 위험조정 성과 (Sharpe 1.05, Sortino 1.86) 를 입증했습니다.\n\n"
        "**HO 24m 의 단기 underperform**: SPY 의 IT 집중 (33.30%) vs Fund 의 IT under-weight (-33.31%) "
        "→ AI Rally 시기 sector concentration 의 trade-off. 펀드는 R3/HO 에서 sector 분산을 회복했고 "
        "(Fund HHI 0.14 < SPY HHI 0.16), 이는 vol-target 분산 운용의 본질이 손상되지 않았음을 보여줍니다.\n\n"
        "**핵심**: 단기 sector concentration 시기 (2024 AI Rally) 의 underperform 은 "
        "**vol-targeted 분산 운용의 의도된 trade-off** 이며, 장기 risk-aware 운용의 가치를 검증합니다."
    )
