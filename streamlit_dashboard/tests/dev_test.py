"""
dev_test.py - lib/* 시각 컴포넌트 검증 페이지 (Phase 1.2 검증 방식 B)

Phase 1.2 의 자동 테스트 (test_lib.py) 에서 검증할 수 없었던 시각 컴포넌트
(색상 / 차트 / 카드 / Footer 등) 를 streamlit 환경에서 직접 시각 확인.

실행:
  streamlit run streamlit_dashboard/tests/dev_test.py

검증 항목:
  1. 색상 swatch (COLORS / BENCHMARK / REGIME / SECTOR)
  2. Tooltip 미리보기 (METRIC_TOOLTIPS dict 표)
  3. Plot helpers 차트 (Regime 4 + Event 3 — plotly 6.x 페이지 환경 동작 확인)
  4. Insight card 그리드 (시나리오별 4-8개)
  5. Footer / Disclaimer 미리보기
  6. Page header / Sub-header 미리보기

검증 완료 후 이 파일은 삭제 가능 (메인 앱 영향 없음).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# === lib import path 설정 (scripts/ 단독 실행 대응) ====================
SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SCRIPT_DIR.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


from lib.colors import (
    BENCHMARK_COLORS,
    COLORS,
    LIMITATION_COLORS,
    REGIME_COLORS,
    SANKEY_GROUP_COLORS,
    SECTOR_COLORS,
)
from lib.disclosure import (
    init_session_state,
    render_footer,
    render_simulator_disclaimer,
)
from lib.insight_generator import generate_insight_cards, render_insight_grid
from lib.page_helpers import (
    inject_custom_css,
    render_page_header,
    render_subheader,
)
from lib.plot_helpers import add_event_annotations, add_regime_backgrounds
from lib.tooltips import METRIC_TOOLTIPS


# === 페이지 설정 ======================================================
st.set_page_config(
    page_title="dev_test - lib 시각 검증",
    page_icon="🔧",
    layout="wide",
)
inject_custom_css()
init_session_state()


st.title("🔧 lib/* 시각 검증 페이지 (dev_test)")
st.caption(
    "Phase 1.2 검증 방식 B — 자동 테스트로 확인 불가능한 시각 컴포넌트를 streamlit 환경에서 직접 확인합니다. "
    "검증 완료 후 이 파일은 삭제 가능합니다."
)
st.divider()


# === 1. 색상 swatch ===================================================
st.header("1. 색상 swatch")
st.caption("color HEX 가 의도대로 표시되는지 확인 (Cobalt Blue / GICS 11 섹터 등)")


def _swatch(name: str, hex_color: str):
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;margin:4px 0;">
            <div style="width:40px;height:40px;background:{hex_color};
                        border:1px solid #374151;border-radius:4px;margin-right:12px;"></div>
            <div>
                <div style="color:#FAFAFA;font-weight:bold;">{name}</div>
                <div style="color:#9CA3AF;font-family:monospace;font-size:13px;">{hex_color}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_swatches(title: str, colors: dict[str, str], n_cols: int = 4):
    st.subheader(title)
    items = list(colors.items())
    for i in range(0, len(items), n_cols):
        cols = st.columns(n_cols)
        for j, (name, hex_color) in enumerate(items[i : i + n_cols]):
            with cols[j]:
                _swatch(name, hex_color)


_render_swatches("1.1 COLORS — Primary 팔레트", COLORS)
_render_swatches("1.2 BENCHMARK_COLORS", BENCHMARK_COLORS)
_render_swatches("1.3 REGIME_COLORS", REGIME_COLORS)
_render_swatches("1.4 SECTOR_COLORS — GICS 11 섹터", SECTOR_COLORS, n_cols=3)
_render_swatches("1.5 LIMITATION_COLORS", LIMITATION_COLORS)
_render_swatches("1.6 SANKEY_GROUP_COLORS", SANKEY_GROUP_COLORS)

st.divider()


# === 2. Tooltip 미리보기 ==============================================
st.header("2. METRIC_TOOLTIPS 미리보기")
st.caption(f"총 {len(METRIC_TOOLTIPS)} 메트릭 정의 — 각 메트릭이 페이지에서 사용될 때 hover 로 표시됨")

tooltip_df = pd.DataFrame(
    [{"Metric": k, "Definition": v} for k, v in METRIC_TOOLTIPS.items()]
)
st.dataframe(tooltip_df, use_container_width=True, height=400)

st.divider()


# === 3. Plot helpers 차트 (Regime 4 + Event 3) =======================
st.header("3. Plot helpers — Regime 배경 + Event annotation")
st.caption("실제 시계열 trace 위에 Regime 4 구간 + Event 3 점선이 정상 렌더되는지 확인")

# 더미 시계열 (2010-01 ~ 2025-12 월별)
dates = pd.date_range("2010-01-01", "2025-12-31", freq="MS")
np.random.seed(42)
returns = np.random.normal(0.008, 0.04, len(dates))
cumulative = (1 + pd.Series(returns)).cumprod()

fig = go.Figure()
fig.add_scatter(
    x=dates,
    y=cumulative,
    name="Dummy Fund",
    line=dict(color=COLORS["primary"], width=2),
)
fig = add_regime_backgrounds(fig, with_labels=True)
fig = add_event_annotations(fig)
fig.update_layout(
    title="Cumulative Return (Dummy) — Regime 4 + Event 3",
    xaxis_title="Date",
    yaxis_title="Cumulative",
    height=450,
    paper_bgcolor=COLORS["background"],
    plot_bgcolor=COLORS["secondary_bg"],
    font=dict(color=COLORS["text"]),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()


# === 4. Insight card 그리드 ===========================================
st.header("4. Insight card 그리드 (Investment Simulator)")
st.caption("시나리오별 카드 4-8개가 정상 렌더되는지 확인 (lump_sum / dca / goal)")

dummy_sim = {
    "total_invested": 10000.0,
    "final_value": 18500.0,
    "total_profit": 8500.0,
    "cagr": 0.0921,
    "mdd": -0.5384,
    "recovery_months": 14,
    "dca_monthly": 833.0,
    "dca_advantage": 1250.0,
    "goal_amount": 20000.0,
    "goal_achievement_date": "2025-08",
}
dummy_benchmarks = {
    "SPY": {"cagr": 0.0850},
    "EW": {"cagr": 0.0780},
    "IVW": {"cagr": 0.0720},
}

scenario = st.radio(
    "시나리오 선택",
    options=["lump_sum", "dca", "goal"],
    horizontal=True,
)
cards = generate_insight_cards(dummy_sim, dummy_benchmarks, scenario=scenario)
st.caption(f"생성된 카드 수: {len(cards)}")
render_insight_grid(cards)

st.divider()


# === 5. Page header / Sub-header / Footer / Disclaimer ===============
st.header("5. Page header / Sub-header / Disclaimer / Footer 미리보기")

st.subheader("5.1 render_page_header")
render_page_header("Performance", "성과 분석")

st.subheader("5.2 render_subheader")
render_subheader(
    title_en="Net CAGR",
    title_ko="순 연환산 수익률",
    description="거래비용 (One-way 20bp) 차감 후 연환산 복리 수익률. Frazzini et al. (2018) 경계값.",
)

st.subheader("5.3 render_simulator_disclaimer")
render_simulator_disclaimer()

st.subheader("5.4 render_footer (페이지 하단)")
st.caption("👇 페이지 맨 아래에 통일 Footer 가 표시됩니다.")


# === 검증 완료 안내 ===================================================
st.divider()
st.success(
    "✅ 모든 시각 컴포넌트가 정상 표시되면 Phase 1.2 검증 방식 B 완료입니다.\n\n"
    "이슈 발견 시: 해당 lib 모듈 수정 → dev_test 재실행으로 재검증."
)


# === Footer (마지막) ==================================================
render_footer()
