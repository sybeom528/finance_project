"""
lib/overview_charts.py - Overview 페이지 전용 차트 / 카드 헬퍼

Overview 6 영역 중 영역 2-5 시각 컴포넌트 생성:
  - 영역 2: render_hero_kpi (5 KPI 카드 + sparkline, TEST/HO 별도)
  - 영역 3: render_cumulative_chart (이중 차트 + Regime + 비교 라인)
  - 영역 4: render_differentiator_cards (3 강점 카드)
  - 영역 5: render_navigation_cards (7 페이지 카드 그리드)

참조: docs/plan/03_pages/01_overview.md, decisionlog/02_overview.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import metric_calculators as mc
from lib.colors import BENCHMARK_COLORS, COLORS
from lib.data_loader import HO_START, TEST_END, filter_period
from lib.plot_helpers import add_event_annotations, add_regime_backgrounds


# ======================================================================
# 영역 2: Hero KPI
# ======================================================================

def _format_pct(v: float, plus_sign: bool = False) -> str:
    """fraction → % 문자열 (NaN 대응)"""
    if pd.isna(v):
        return "—"
    sign = "+" if plus_sign and v > 0 else ""
    return f"{sign}{v * 100:.2f}%"


def _format_ratio(v: float) -> str:
    """ratio (Sharpe / Sortino 등) → 소수 2자리"""
    if pd.isna(v):
        return "—"
    return f"{v:.2f}"


def _calc_hero_metrics(ret: pd.Series, gross_ret: pd.Series) -> dict[str, dict]:
    """
    Hero KPI 5 메트릭 계산 (TEST + HO 별도).

    Returns:
        {
            "Cumulative Return": {"test": float, "ho": float, "label": str},
            "Net CAGR":          {...},
            "Sortino":           {...},
            "Volatility":        {...},
            "MDD":               {...},
        }
    """
    ret_test = filter_period(ret, "TEST")
    ret_ho = filter_period(ret, "HO")

    return {
        "Cumulative Return": {
            "test": float((1 + ret_test).prod() - 1) if len(ret_test) else np.nan,
            "ho": float((1 + ret_ho).prod() - 1) if len(ret_ho) else np.nan,
            "format": "pct",
        },
        "Net CAGR": {
            "test": mc.calc_cagr(ret_test),
            "ho": mc.calc_cagr(ret_ho),
            "format": "pct",
        },
        "Sortino": {
            "test": mc.calc_sortino(ret_test),
            "ho": mc.calc_sortino(ret_ho),
            "format": "ratio",
        },
        "Volatility": {
            "test": mc.calc_volatility(ret_test),
            "ho": mc.calc_volatility(ret_ho),
            "format": "pct",
        },
        "MDD": {
            "test": mc.calc_mdd(ret_test),
            "ho": mc.calc_mdd(ret_ho),
            "format": "pct",
        },
    }


def _make_sparkline(ret: pd.Series, color_test: str, color_ho: str) -> go.Figure:
    """
    누적 wealth 곡선 sparkline. TEST 영역은 기본색, HO 영역은 빨간 점선.
    높이 ~50px, 축/legend 없음.
    """
    if len(ret) == 0:
        return go.Figure()
    wealth = (1 + ret).cumprod()
    test_part = wealth[wealth.index <= TEST_END]
    ho_part = wealth[wealth.index >= HO_START]

    fig = go.Figure()
    if len(test_part) > 0:
        fig.add_trace(go.Scatter(
            x=test_part.index, y=test_part.values,
            mode="lines",
            line=dict(color=color_test, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))
    if len(ho_part) > 0:
        fig.add_trace(go.Scatter(
            x=ho_part.index, y=ho_part.values,
            mode="lines",
            line=dict(color=color_ho, width=1.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
    fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def render_hero_kpi(ret: pd.Series, gross_ret: pd.Series) -> None:
    """
    영역 2: Hero KPI 5 카드 그리드 (TEST + HO 별도, sparkline 포함).
    사이드바 토글 영향 받지 않음 (고정).
    """
    metrics = _calc_hero_metrics(ret, gross_ret)
    cols = st.columns(5)
    items = list(metrics.items())

    for col, (label, data) in zip(cols, items):
        with col:
            test_v = data["test"]
            ho_v = data["ho"]
            fmt = data["format"]
            fmt_fn = _format_pct if fmt == "pct" else _format_ratio
            plus = (label != "MDD")  # MDD 는 음수, +기호 부적절

            # 메인 라벨
            st.markdown(f"**{label}**")
            # TEST / HO 두 줄
            st.caption(f"TEST: {fmt_fn(test_v, plus) if fmt == 'pct' else fmt_fn(test_v)}")
            ho_color = COLORS["accent_red"] if (not pd.isna(ho_v) and ho_v < 0 and label != "MDD") else COLORS["text_muted"]
            st.markdown(
                f'<div style="color:{COLORS["accent_red"] if label != "Volatility" else COLORS["text_muted"]};font-size:13px;">'
                f"HO: {fmt_fn(ho_v, plus) if fmt == 'pct' else fmt_fn(ho_v)}"
                f"</div>",
                unsafe_allow_html=True,
            )
            # Sparkline (각 카드별 unique key 로 ID 충돌 회피)
            spark = _make_sparkline(ret, COLORS["primary"], COLORS["accent_red"])
            st.plotly_chart(
                spark,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"hero_sparkline_{label}",
            )


# ======================================================================
# 영역 3: 누적수익 곡선 (이중 차트)
# ======================================================================

def _to_cumulative(ret: pd.Series) -> pd.Series:
    """월별 수익률 → 누적 wealth (1.0 시작)"""
    return (1 + ret.fillna(0)).cumprod()


def _to_drawdown(ret: pd.Series) -> pd.Series:
    """월별 수익률 → drawdown 시계열 (음수, 0 시작)"""
    wealth = _to_cumulative(ret)
    peak = wealth.cummax()
    return (wealth - peak) / peak


def render_cumulative_chart(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    ew_ret: pd.Series | None,
    ivw_ret: pd.Series | None,
    show_log: bool,
    period: str,
) -> None:
    """
    영역 3: 이중 차트 (위=누적수익 / 아래=Drawdown).

    Args:
        fund_ret: 펀드 월별 수익률 (Series, DatetimeIndex)
        spy_ret: SPY 월별 수익률
        ew_ret: EW baseline (None 이면 표시 X)
        ivw_ret: IVW baseline (None 이면 표시 X)
        show_log: True 면 누적수익 차트 Y축 log
        period: "FULL" / "TEST" / "HO" — 기간 필터
    """
    # 사이드바 기간 토글 적용
    fund_ret = filter_period(fund_ret, period)
    spy_ret = filter_period(spy_ret, period)
    if ew_ret is not None:
        ew_ret = filter_period(ew_ret, period)
    if ivw_ret is not None:
        ivw_ret = filter_period(ivw_ret, period)

    # 이중 차트
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Cumulative Return", "Drawdown"),
    )

    # 1) 누적수익 (행 1)
    fund_cum = _to_cumulative(fund_ret)
    fig.add_trace(
        go.Scatter(
            x=fund_cum.index, y=fund_cum.values,
            name="Fund (Adaptive VolControl)",
            line=dict(color=BENCHMARK_COLORS["Fund"], width=2.5),
        ),
        row=1, col=1,
    )

    spy_cum = _to_cumulative(spy_ret)
    fig.add_trace(
        go.Scatter(
            x=spy_cum.index, y=spy_cum.values,
            name="SPY",
            line=dict(color=BENCHMARK_COLORS["SPY"], width=1.5),
        ),
        row=1, col=1,
    )

    if ew_ret is not None and len(ew_ret) > 0:
        ew_cum = _to_cumulative(ew_ret)
        fig.add_trace(
            go.Scatter(
                x=ew_cum.index, y=ew_cum.values,
                name="EW (펀드 universe)",
                line=dict(color=BENCHMARK_COLORS["EW"], width=1.5, dash="dash"),
            ),
            row=1, col=1,
        )

    if ivw_ret is not None and len(ivw_ret) > 0:
        ivw_cum = _to_cumulative(ivw_ret)
        fig.add_trace(
            go.Scatter(
                x=ivw_cum.index, y=ivw_cum.values,
                name="IVW (Naive Low-vol)",
                line=dict(color=BENCHMARK_COLORS["IVW"], width=1.5, dash="dot"),
            ),
            row=1, col=1,
        )

    # 2) Drawdown (행 2)
    fund_dd = _to_drawdown(fund_ret) * 100  # %
    fig.add_trace(
        go.Scatter(
            x=fund_dd.index, y=fund_dd.values,
            name="Fund DD",
            line=dict(color=COLORS["accent_red"], width=1),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.2)",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Regime 배경 + Event annotation (FULL 일 때만 의미 있음)
    if period == "FULL":
        add_regime_backgrounds(fig, with_labels=True)
        add_event_annotations(fig)

    # 레이아웃
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["secondary_bg"],
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis2=dict(rangeslider=dict(visible=True, thickness=0.05)),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Cumulative", row=1, col=1, type="log" if show_log else "linear")
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, key="overview_cumulative_chart")


# ======================================================================
# 영역 4: 핵심 강점 카드 (3개)
# ======================================================================

_DIFFERENTIATOR_CARDS = [
    {
        "icon": "📊",
        "title_en": "Volatility-Aware Allocation",
        "title_ko": "변동성 인지 자산배분",
        "body": "LSTM 으로 미래 변동성 예측 → 적응형 비중 결정",
        "value_template": "Volatility (HO): {ho_vol}",
        "metric_key": "vol_ho",
        "citation": "Hochreiter & Schmidhuber (1997)",
        "page": "pages/07_Methodology.py",
        "color": COLORS["primary"],
    },
    {
        "icon": "✅",
        "title_en": "Validated Across Market Regimes",
        "title_ko": "시장 국면별 검증 완료",
        "body": "회복기 / 확장기 / 변동기 + 24m HOLD_OUT 검증",
        "value_template": "Sortino (TEST): {sortino}",
        "metric_key": "sortino",
        "citation": "Walk-forward (is=1250d / oos=21d / embargo=63d)",
        "page": "pages/08_Backtesting.py",
        "color": COLORS["accent_green"],
    },
    {
        "icon": "💎",
        "title_en": "Net of Conservative Costs",
        "title_ko": "거래비용 차감 후 양호",
        "body": "20bp One-way 거래비용 차감 후에도 양호한 위험조정 수익",
        "value_template": "Net CAGR (TEST): {net_cagr}",
        "metric_key": "net_cagr",
        "citation": "Frazzini, Israel & Moskowitz (2018)",
        "page": "pages/03_Performance.py",
        "color": COLORS["accent_amber"],
    },
]


def _calc_card_values(ret: pd.Series) -> dict[str, str]:
    """강점 카드에 표시할 메트릭 값 계산."""
    ret_test = filter_period(ret, "TEST")
    ret_ho = filter_period(ret, "HO")
    return {
        "vol_ho": _format_pct(mc.calc_volatility(ret_ho)),
        "sortino": _format_ratio(mc.calc_sortino(ret_test)),
        "net_cagr": _format_pct(mc.calc_cagr(ret_test), plus_sign=True),
    }


def render_differentiator_cards(ret: pd.Series) -> None:
    """
    영역 4: 3 핵심 강점 카드 (Methodology / Backtesting / Performance 로 navigation).
    """
    values = _calc_card_values(ret)
    cols = st.columns(3)

    for col, card in zip(cols, _DIFFERENTIATOR_CARDS):
        with col:
            value_text = card["value_template"].format(
                ho_vol=values.get("vol_ho", "—"),
                sortino=values.get("sortino", "—"),
                net_cagr=values.get("net_cagr", "—"),
            )
            card_html = (
                f'<div style="border:2px solid {card["color"]};border-radius:8px;'
                f'padding:18px;margin-bottom:12px;background-color:#1F2937;min-height:240px;">'
                f'<div style="font-size:28px;">{card["icon"]}</div>'
                f'<div style="font-weight:bold;font-size:16px;margin:8px 0 4px 0;color:#FAFAFA;">{card["title_en"]}</div>'
                f'<div style="font-size:13px;color:#9CA3AF;margin-bottom:12px;">{card["title_ko"]}</div>'
                f'<div style="font-size:13px;color:#FAFAFA;line-height:1.5;margin-bottom:12px;">{card["body"]}</div>'
                f'<div style="font-size:15px;color:{card["color"]};font-weight:bold;margin-bottom:6px;">{value_text}</div>'
                f'<div style="font-size:11px;color:#9CA3AF;font-style:italic;">{card["citation"]}</div>'
                f"</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)
            if st.button(
                f"→ {card['page'].split('_', 1)[1].replace('.py', '').replace('_', ' ')}",
                key=f"diff_card_{card['metric_key']}",
                use_container_width=True,
            ):
                st.switch_page(card["page"])


# ======================================================================
# 영역 5: Navigation cards (7개)
# ======================================================================

_NAVIGATION_CARDS = [
    {"icon": "📈", "label_en": "Performance",         "label_ko": "성과 분석",     "desc": "액티브 운용 + 다중 벤치마크",   "page": "pages/03_Performance.py"},
    {"icon": "⚠️", "label_en": "Risk Metrics",        "label_ko": "위험 지표",     "desc": "위험 통제 + Tail Risk (Hill)", "page": "pages/04_Risk_Metrics.py"},
    {"icon": "🏢", "label_en": "Holdings",            "label_ko": "보유 종목",     "desc": "종목 detail + 기여도 분석",    "page": "pages/05_Holdings.py"},
    {"icon": "🌐", "label_en": "Sector Watch",        "label_ko": "섹터 분석",     "desc": "섹터 분산 + HO 정당화",        "page": "pages/06_Sector_Watch.py"},
    {"icon": "🧪", "label_en": "Methodology",         "label_ko": "방법론",        "desc": "BL + LSTM walk-forward",       "page": "pages/07_Methodology.py"},
    {"icon": "✅", "label_en": "Backtesting",         "label_ko": "검증",          "desc": "Regime / Sub-events 검증",     "page": "pages/08_Backtesting.py"},
    {"icon": "ℹ️", "label_en": "About / FAQ",         "label_ko": "소개 / FAQ",    "desc": "프로젝트 소개 / 자주 묻는 질문", "page": "pages/09_About.py"},
]


def render_navigation_cards() -> None:
    """
    영역 5: 7 페이지 navigation 카드 그리드 (4 + 3 분할).
    Investment Simulator 는 영역 4 와 흐름이 달라 제외 (사이드바 체험 그룹에서 접근).
    """
    # 4 + 3 분할
    rows = [_NAVIGATION_CARDS[:4], _NAVIGATION_CARDS[4:]]
    for row_cards in rows:
        cols = st.columns(len(row_cards))
        for col, nav in zip(cols, row_cards):
            with col:
                card_html = (
                    f'<div style="border:1px solid #374151;border-radius:6px;'
                    f'padding:14px;margin-bottom:8px;background-color:#1F2937;min-height:130px;">'
                    f'<div style="font-size:24px;">{nav["icon"]}</div>'
                    f'<div style="font-weight:bold;font-size:14px;margin:6px 0 2px 0;color:#FAFAFA;">{nav["label_en"]}</div>'
                    f'<div style="font-size:11px;color:#9CA3AF;margin-bottom:8px;">{nav["label_ko"]}</div>'
                    f'<div style="font-size:11px;color:#9CA3AF;line-height:1.4;">{nav["desc"]}</div>'
                    f"</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button(
                    f"열기 →",
                    key=f"nav_{nav['label_en']}",
                    use_container_width=True,
                ):
                    st.switch_page(nav["page"])
