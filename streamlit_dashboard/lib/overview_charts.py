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
from lib.data_loader import EVAL_PERIODS, HO_START, TEST_END, filter_period
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


def _calc_hero_metrics(
    ret: pd.Series,
    gross_ret: pd.Series,
    rf: pd.Series | None = None,
) -> dict[str, dict]:
    """
    Hero KPI 5 메트릭 (TEST + HO 별도) — final/master_table 과 정확 일치.

    각 메트릭은 final/master_table.py 의 sortino_TEST / cagr_TEST / mdd_TEST 등과
    동일하게 subperiod 함수로 계산 (EVAL_PERIODS 기반):
      - Sortino → calc_sortino_subperiod (final master_table _sortino_subperiod)
      - CAGR    → calc_cagr_subperiod    (final master_table _cagr_subperiod)
      - MDD     → calc_mdd_subperiod     (final master_table _mdd_subperiod)
      - Vol     → final 에 vol_subperiod 부재 → ret.std() * sqrt(12) 직접
      - CumRet  → final 에 cum_subperiod 부재 → (1+ret).prod() - 1 직접

    Args:
        ret: 펀드 월별 수익률 (Net), DatetimeIndex 필수
        gross_ret: TC 차감 전 (향후 확장 위해 받음, 현재 미사용)
        rf: 무위험 수익률 시리즈 (월별). None 이면 0 (final 정합 X).
    """
    rf_input = rf if rf is not None else 0.0
    test_s, test_e = EVAL_PERIODS["TEST"]
    ho_s, ho_e = EVAL_PERIODS["HOLD_OUT"]

    # Cum Return / Vol — final 에 subperiod 함수 부재 → filter_period 직접 사용
    ret_test = filter_period(ret, "TEST")
    ret_ho = filter_period(ret, "HO")

    return {
        "Cumulative Return": {
            "test": float((1 + ret_test).prod() - 1) if len(ret_test) else np.nan,
            "ho": float((1 + ret_ho).prod() - 1) if len(ret_ho) else np.nan,
            "format": "pct",
        },
        "Net CAGR": {
            "test": mc.calc_cagr_subperiod(ret, test_s, test_e),
            "ho": mc.calc_cagr_subperiod(ret, ho_s, ho_e),
            "format": "pct",
        },
        "Sortino": {
            "test": mc.calc_sortino_subperiod(ret, rf_input, test_s, test_e),
            "ho": mc.calc_sortino_subperiod(ret, rf_input, ho_s, ho_e),
            "format": "ratio",
        },
        "Volatility": {
            "test": mc.calc_volatility(ret_test),
            "ho": mc.calc_volatility(ret_ho),
            "format": "pct",
        },
        "MDD": {
            "test": mc.calc_mdd_subperiod(ret, test_s, test_e),
            "ho": mc.calc_mdd_subperiod(ret, ho_s, ho_e),
            "format": "pct",
        },
    }


def render_hero_kpi(
    ret: pd.Series,
    gross_ret: pd.Series,
    rf: pd.Series | None = None,
) -> None:
    """
    영역 2: Hero KPI 5 카드 그리드 (TEST + HO 별도).

    Sparkline 제거 (옵션 C, 2026-05-10 결정 Q-G):
    각 메트릭의 시간적 추이는 다른 페이지에서 명확히 표시:
      - Cumulative Return → Overview 영역 3 (이중 차트 위)
      - Net CAGR          → Performance 영역 6 (Rolling Return)
      - Sortino           → Risk Metrics (Sortino rolling 시계열)
      - Volatility        → Risk Metrics 영역 6 (Vol/Beta/R²/TE 시계열, Volatility 추가)
      - MDD               → Overview 영역 3 (이중 차트 아래) + Risk Metrics 영역 4

    사이드바 토글 영향 받지 않음 (고정 — 학술 정직성).

    Args:
        rf: 무위험 수익률 시리즈 (월별). None 이면 0 — Sortino 결과가 final 과
            정확히 일치하려면 panel.rf_1m 전달 필요.
    """
    from lib.tooltips import get_tooltip

    metrics = _calc_hero_metrics(ret, gross_ret, rf=rf)
    cols = st.columns(5)
    items = list(metrics.items())

    for col, (label, data) in zip(cols, items):
        with col:
            test_v = data["test"]
            ho_v = data["ho"]
            fmt = data["format"]
            fmt_fn = _format_pct if fmt == "pct" else _format_ratio
            plus = (label != "MDD")

            test_str = fmt_fn(test_v, plus) if fmt == "pct" else fmt_fn(test_v)
            ho_str = fmt_fn(ho_v, plus) if fmt == "pct" else fmt_fn(ho_v)

            # tooltip 텍스트 (메트릭 정의)
            tip = get_tooltip(label) or ""
            tip_html = (
                f'<span title="{tip}" style="cursor:help;color:#9CA3AF;font-size:11px;'
                f'margin-left:6px;border:1px solid #374151;border-radius:50%;'
                f'width:14px;height:14px;display:inline-flex;align-items:center;'
                f'justify-content:center;">ⓘ</span>'
                if tip else ""
            )

            # 단순 카드 — 큰 숫자 강조 + TEST / HO 두 줄 (sparkline 제거) + tooltip
            st.markdown(
                f'<div style="border-left:3px solid {COLORS["primary"]};'
                f'padding:8px 0 8px 12px;margin-bottom:4px;">'
                f'<div style="font-size:13px;color:#9CA3AF;font-weight:600;">'
                f'{label}{tip_html}'
                f'</div>'
                f'<div style="font-size:24px;color:#FAFAFA;font-weight:700;margin-top:4px;">'
                f'{test_str}'
                f'</div>'
                f'<div style="font-size:11px;color:#9CA3AF;margin-top:2px;">TEST</div>'
                f'<div style="font-size:14px;color:{COLORS["accent_red"]};font-weight:600;margin-top:8px;">'
                f'{ho_str}'
                f'</div>'
                f'<div style="font-size:11px;color:#9CA3AF;">HOLD_OUT</div>'
                f'</div>',
                unsafe_allow_html=True,
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


def _calc_card_values(ret: pd.Series, rf: pd.Series | None = None) -> dict[str, str]:
    """
    강점 카드 메트릭 — final master_table 정합 (subperiod 함수 사용).
    """
    rf_input = rf if rf is not None else 0.0
    test_s, test_e = EVAL_PERIODS["TEST"]
    ho_s, ho_e = EVAL_PERIODS["HOLD_OUT"]
    ret_ho = filter_period(ret, "HO")

    return {
        "vol_ho": _format_pct(mc.calc_volatility(ret_ho)),
        "sortino": _format_ratio(mc.calc_sortino_subperiod(ret, rf_input, test_s, test_e)),
        "net_cagr": _format_pct(mc.calc_cagr_subperiod(ret, test_s, test_e), plus_sign=True),
    }


def render_differentiator_cards(ret: pd.Series, rf: pd.Series | None = None) -> None:
    """
    영역 4: 3 핵심 강점 카드 (Methodology / Backtesting / Performance 로 navigation).

    Args:
        rf: 무위험 수익률 시리즈. None 이면 0 (final 정합 X).
    """
    values = _calc_card_values(ret, rf=rf)
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
