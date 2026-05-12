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
from lib.data_loader import EVAL_PERIODS, load_daily_returns
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


# === GICS 섹터 영문 → 한글 매핑 (KPI 카드 표시 전용, 2026-05-12) ===========
# 좁은 KPI 카드 공간에서 영문 섹터명 (Information Technology 등) 이 잘리는 문제 해결.
# KPI 3, 4, 5 의 sector name 표시 부분에만 적용.
# 다른 차트 (Treemap, Decomposition 표, Tilt Tornado, Rotation) 는 영문 GICS 표준 유지.
SECTOR_KO_MAP: dict[str, str] = {
    "Information Technology": "IT",
    "Health Care": "헬스케어",
    "Financials": "금융",
    "Consumer Discretionary": "임의소비재",
    "Industrials": "산업재",
    "Communication Services": "통신",
    "Consumer Staples": "필수소비재",
    "Energy": "에너지",
    "Utilities": "유틸리티",
    "Real Estate": "부동산",
    "Materials": "소재",
}


def _sector_ko(sector_name: str) -> str:
    """GICS 영문 섹터명 → 한글 매핑 (KPI 카드 표시용)."""
    return SECTOR_KO_MAP.get(sector_name, sector_name[:10])


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
    Sector KPI 5개 — 사이드바 토글 기간 평균 (메인) + Latest snapshot (caption).

    1. Sector HHI (Hirschman 1945, sector level)
    2. 섹터 비중 평균 차이 (vs SPY) = Avg |Tilt|
    3. 섹터 비중 최대 차이 (vs SPY) = Max |Tilt|
    4. Most Overweight Sector (기간 평균 Tilt 기준)
    5. Most Underweight Sector (기간 평균 Tilt 기준, HO 정당화 narrative 직접 연결)

    표시 규칙 (2026-05-13 변경):
      - 메인 value = 기간 평균 (사이드바 토글에 반응)
      - caption = "최신 (YYYY-MM): ..." 형식 — 최신 시점 값 보조 표시
      - KPI 4/5 의 Most Over/Under 섹터도 기간 평균 Tilt 기준으로 산출
    """
    latest = weights.index.max()

    # Latest snapshot — caption 보조 표시용
    fund_sw_latest = mc.calc_sector_weights(weights.loc[latest], ticker_to_sector)
    spy_sw_latest = mc.compute_spy_sector_weights(panel, sp500_membership, latest, ticker_to_sector)
    tilt_latest = mc.calc_sector_tilt(fund_sw_latest, spy_sw_latest)

    sector_hhi_latest = float((fund_sw_latest ** 2).sum())
    spy_hhi_latest = float((spy_sw_latest ** 2).sum()) if len(spy_sw_latest) > 0 else np.nan
    avg_tilt_latest = float(tilt_latest.abs().mean()) if len(tilt_latest) > 0 else np.nan
    if len(tilt_latest) > 0:
        max_tilt_latest = float(tilt_latest.abs().max())
        max_tilt_sector_latest = str(tilt_latest.abs().idxmax())
    else:
        max_tilt_latest = np.nan
        max_tilt_sector_latest = ""
    most_over_latest = tilt_latest.head(1) if len(tilt_latest) > 0 else pd.Series(dtype=float)
    most_under_latest = tilt_latest.tail(1) if len(tilt_latest) > 0 else pd.Series(dtype=float)

    # 기간 평균 (사이드바 토글) — 메인 KPI value
    period_idx = _filter_period_index(weights.index, period)
    tilt_frames: list[pd.Series] = []
    spy_hhi_list: list[float] = []
    if len(period_idx) > 0:
        hhi_list, avg_tilt_list, max_tilt_list = [], [], []
        for t in period_idx:
            fsw = mc.calc_sector_weights(weights.loc[t], ticker_to_sector)
            ssw = mc.compute_spy_sector_weights(panel, sp500_membership, t, ticker_to_sector)
            tt = mc.calc_sector_tilt(fsw, ssw)
            hhi_list.append(float((fsw ** 2).sum()))
            avg_tilt_list.append(float(tt.abs().mean()))
            if len(tt) > 0:
                max_tilt_list.append(float(tt.abs().max()))
                tilt_frames.append(tt)
            if len(ssw) > 0:
                spy_hhi_list.append(float((ssw ** 2).sum()))
        avg_hhi = float(np.mean(hhi_list)) if hhi_list else np.nan
        avg_tilt_mean = float(np.mean(avg_tilt_list)) if avg_tilt_list else np.nan
        max_tilt_mean = float(np.mean(max_tilt_list)) if max_tilt_list else np.nan
        spy_hhi_mean = float(np.mean(spy_hhi_list)) if spy_hhi_list else np.nan
    else:
        avg_hhi = avg_tilt_mean = max_tilt_mean = spy_hhi_mean = np.nan

    # 기간 평균 Tilt → 섹터별 평균 → Most Over/Under
    if tilt_frames:
        mean_tilt_by_sector = (
            pd.concat(tilt_frames, axis=1).mean(axis=1).sort_values(ascending=False)
        )
        most_over_avg = mean_tilt_by_sector.head(1)
        most_under_avg = mean_tilt_by_sector.tail(1)
    else:
        most_over_avg = pd.Series(dtype=float)
        most_under_avg = pd.Series(dtype=float)

    # 헤더 라벨
    period_label_map = {"FULL": "FULL 192개월", "TEST": "TEST 168개월", "HO": "Hold Out 24개월"}
    latest_label = latest.strftime("%Y-%m")
    st.caption(
        f"**기간 평균**: {period_label_map.get(period, period)} (메인 표시)  ·  "
        f"**최신 시점**: {latest_label} (보조 표시)"
    )

    cols = st.columns(5)

    with cols[0]:
        delta = (avg_hhi - spy_hhi_mean) if not (pd.isna(avg_hhi) or pd.isna(spy_hhi_mean)) else None
        st.metric(
            label="Sector HHI",
            value=_fmt_ratio(avg_hhi, 3),
            delta=f"{delta:+.3f} vs SPY" if delta is not None else None,
            delta_color="inverse",
            help="섹터 집중도 = Σ(섹터 weight)². 낮을수록 분산 (Hirschman 1945). 메인 = 기간 평균.",
        )
        st.caption(f"최신 ({latest_label}): {_fmt_ratio(sector_hhi_latest, 3)}")

    with cols[1]:
        st.metric(
            label="섹터 비중 평균 차이 (vs SPY)",
            value=_fmt_pct(avg_tilt_mean, digits=2),
            help=(
                "**11개 GICS 섹터의 |Fund 비중 − SPY 비중| 평균**. "
                "펀드가 SPY 대비 각 섹터를 얼마나 다르게 보유했는지의 평균 차이 (절대값 평균 — 방향 무관). "
                "예: 5%p = 각 섹터가 평균적으로 SPY 대비 ±5%p 차이. "
                "일반 active fund 4-8%p, passive ~0%p. 메인 = 기간 평균."
            ),
        )
        st.caption(f"최신 ({latest_label}): {_fmt_pct(avg_tilt_latest, digits=2)}")

    with cols[2]:
        # Max |Tilt| — 단일 섹터 최대 차이 (under/over 무관)
        st.metric(
            label="섹터 비중 최대 차이 평균 (vs SPY)",
            value=_fmt_pct(max_tilt_mean, digits=2),
            help=(
                "**각 시점의 최대 단일 섹터 비중 차이를 기간 평균** = mean( max |Fund 비중 − SPY 비중| ). "
                "방향 (over/under) 무관 — 절대값 기준. "
                "매월 가장 크게 벌어진 섹터 차이가 평균적으로 얼마였는지를 나타냅니다 "
                "(어느 섹터인지는 KPI 4 / 5 참조). "
                "KPI 2 (전체 섹터 평균 차이) 와 짝 — 전체 평균 vs 최대 섹터 평균. 메인 = 기간 평균."
            ),
        )
        st.caption(
            f"최신 ({latest_label}): {_sector_ko(max_tilt_sector_latest)} "
            f"{_fmt_pct(max_tilt_latest, digits=2)}"
        )

    with cols[3]:
        if len(most_over_avg) > 0:
            sec_avg, val_avg = most_over_avg.index[0], float(most_over_avg.iloc[0])
            st.metric(
                label="Most Overweight",
                value=_sector_ko(sec_avg),
                delta=f"{val_avg:+.2%}",
                delta_color="off",
                help=(
                    f"기간 평균 Tilt 기준 가장 큰 over-weight 섹터: {sec_avg}. "
                    "메인 = 기간 평균 Tilt 최대 섹터."
                ),
            )
            if len(most_over_latest) > 0:
                sec_l = str(most_over_latest.index[0])
                val_l = float(most_over_latest.iloc[0])
                st.caption(
                    f"최신 ({latest_label}): {_sector_ko(sec_l)} {val_l:+.2%}"
                )

    with cols[4]:
        if len(most_under_avg) > 0:
            sec_avg, val_avg = most_under_avg.index[0], float(most_under_avg.iloc[0])
            st.metric(
                label="Most Underweight",
                value=_sector_ko(sec_avg),
                delta=f"{val_avg:+.2%}",
                delta_color="off",
                help=(
                    f"기간 평균 Tilt 기준 가장 큰 under-weight 섹터: {sec_avg}. "
                    "메인 = 기간 평균 Tilt 최소 섹터."
                ),
            )
            if len(most_under_latest) > 0:
                sec_l = str(most_under_latest.index[0])
                val_l = float(most_under_latest.iloc[0])
                st.caption(
                    f"최신 ({latest_label}): {_sector_ko(sec_l)} {val_l:+.2%}"
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

    # printf style "%.2f%%" 는 Python ":.2%" 와 달리 자동 ×100 하지 않음.
    # 표시용 사본을 만들어 % 컬럼들에 ×100 적용 (원본 df 는 CSV 다운로드용으로 보존).
    df_display = df.copy()
    pct_cols = ["Weight", "Tilt", "Return_12m", "Volatility", "Contribution"]
    for col in pct_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col] * 100

    st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Weight": st.column_config.ProgressColumn(
                "Weight", format="%.2f%%", min_value=0,
                max_value=float(df_display["Weight"].max()) if len(df_display) > 0 else 100.0,
            ),
            "Tilt": st.column_config.NumberColumn(
                "Tilt vs SPY", format="%+.2f%%",
                help="펀드 weight − SPY weight (양수 = over, 음수 = under)",
            ),
            "Return_12m": st.column_config.NumberColumn("12개월 R", format="%+.2f%%"),
            "N_Holdings": st.column_config.NumberColumn("#H", width="small"),
            "Volatility": st.column_config.NumberColumn("Vol", format="%.2f%%"),
            "Beta": st.column_config.NumberColumn("β", format="%.3f"),
            "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "Contribution": st.column_config.NumberColumn("Contrib", format="%+.3f%%"),
        },
    )

    # CSV 는 raw 비율 (0~1) 그대로 다운로드 (계산용 일관성 유지)
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
        "임계선: ±1% (점선, 의미 있는 차이) · ±5% (대시, 큰 차이). "
        "★ = Information Technology (Hold Out 24개월 정당화 narrative 핵심)."
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
    Hold Out 24m 분석 — 영역 8 핵심.

    1. 학술 narrative 박스 (Markowitz 1952)
    2. Chart 1: SPY IT 비중 (좌축) + Fund IT Tilt (우축) 이중 축
    3. Chart 2: Hold Out 24m Sector Contribution Tornado
    4. Chart 3: Regime 별 Sector HHI 추세 (Fund vs SPY)
    5. 결론 박스 (장기 분산의 가치)
    """
    # === 1. 도입 narrative 박스 ===
    st.info(
        "ℹ️ 본 펀드 (**BL+LSTM 변동성 예측** 모델) 는 단순한 섹터 분산이 아니라, "
        "**시장 변동성에 따라 비중을 조정하는 적응형 분산 운용** 입니다. "
        "시장 상황에 따라 섹터 집중 (예: 2017 방어주 선호 시기) 과 섹터 분산 (Hold Out 24개월) 을 "
        "동적으로 선택합니다. **Hold Out 24개월 시기의 IT 축소 (under-weight)** 는 "
        "AI Rally 라는 단기 시장 쏠림에 따른 트레이드오프이며, "
        "장기 변동성 인지 운용의 일관성을 보여주는 의도된 결과입니다."
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
    st.markdown("##### Chart 2 — Hold Out 24개월 Sector Contribution (복리 보정)")

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
            xaxis=dict(tickformat=".1%", title="HO 24개월 Carino Contribution"),
            yaxis=dict(autorange="reversed"),
            height=420, margin=dict(t=20, l=10, r=80, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "★ Information Technology = 펀드의 under-weight 가 HO 24개월 underperform 의 핵심. "
            "Defensive 섹터 (Utilities/Consumer Staples) 는 양수 기여하지만 IT 의 거대 양수 contribution 을 따라가지 못함."
        )
    st.divider()

    # === 4. Chart 3: Regime 별 Sector HHI ===
    st.markdown("##### Chart 3 — Regime 별 Sector HHI (Fund vs SPY)")

    regimes = {
        "R1 회복기 (2010-2012)": ("2010-01-01", "2012-06-30"),
        "R2 확장기 (2012-2019)": ("2012-07-01", "2019-12-31"),
        "R3 변동기 (2020-2023)": ("2020-01-01", "2023-12-31"),
        "HO 24개월 (2024-2025)": ("2024-01-01", "2025-12-31"),
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
        "**R2 (2012-19, 90개월)**: Fund HHI 0.41 — vol-target 모델이 시장 변동성 시기 (특히 2017 정치/금리 불확실성) "
        "Defensive 섹터 (Utilities 70%+ / Staples 30%) 로 도피 학습. "
        "**HO 24개월**: Fund 0.14 < SPY 0.16 — **펀드 분산 회복 vs SPY IT 집중** = HO 정당화 narrative 핵심."
    )
    st.divider()

    # === 6. 결론 박스 ===
    st.success(
        "✅ **결론: 변동성 기반 적응형 운용의 가치**\n\n"
        "본 펀드의 **BL+LSTM 변동성 예측 모델** 은 단순한 섹터 분산이 아닌 "
        "**변동성을 인지한 적응형 분산 운용** 입니다.\n\n"
        "**회복기 / 변동기 / Hold Out 분산 시기** + **확장기 방어주 선호 시기** 모두 "
        "시장 변동성에 따른 적응 결과입니다. **TEST 168개월 (14년) walk-forward 운용 동안** "
        "시장 변화에 적응하며 **우수한 위험 조정 성과 (Sharpe 1.05, Sortino 1.86)** 를 입증했습니다.\n\n"
        "**Hold Out 24개월 (2024-2025) 의 단기 부진**: 시장의 IT 쏠림 (SPY IT 비중 33.30%) vs "
        "펀드의 IT 축소 (-33.31%p 차이) → AI Rally 라는 단기 섹터 쏠림 시기의 트레이드오프. "
        "펀드는 이 시기에도 **섹터 분산을 유지** (Fund 집중도 0.14 < SPY 0.16) 하여 "
        "운용 철학의 일관성을 잃지 않았습니다.\n\n"
        "**핵심**: 2024-2025 AI Rally 시기의 단기 부진은 "
        "**변동성 인지 분산 운용의 의도된 단기 비용** 이며, "
        "장기 위험 인지 운용의 가치를 입증합니다."
    )
