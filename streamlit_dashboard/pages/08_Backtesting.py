"""
pages/08_Backtesting.py — Backtesting 페이지 (7 영역, 초안)

검증 / 강건성 narrative 전담 페이지.

7 영역:
  1. Header
  2. Sub-header
  3. Backtest Summary KPI 5개 (TEST/HO Gap / Sensitivity Robust / 4-slot Robust /
                              Avg Recovery / Regime 일관성)
  4. Regime 메트릭 자세한 비교 (12 메트릭 × 5 Regime + Sortino 막대)
  5. Sub-events 분석 (4 위기 — 2018Q4 / COVID / 2022 Bear / 2024 IT Rotation)
  6. Sensitivity Test (156 config Top 10 + 신모델 강조)
  7. Footer

설계 원칙 (초안):
  - 영역별 독립 호출 (영역 추가/제거/변경 용이)
  - 메트릭 / 위기 정의는 metric_calculators.py 의 외부 dict (변경 용이)
  - 데이터: 우리 펀드 pkl + 156 config (이미 git 포함) + monthly_panel

참조: docs/plan/03_pages/08_backtesting.md, decisionlog/08_backtesting.md
"""

import streamlit as st

from lib.backtesting_charts import (
    render_backtest_kpi,
    render_regime_detail_table,
    render_sensitivity_test,
    render_sub_events,
)
from lib.data_loader import load_fund_results, load_monthly_panel
from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import (
    inject_custom_css,
    render_page_header,
    render_sidebar,
    render_subheader,
)


# === 페이지 설정 ======================================================
inject_custom_css()
init_session_state()
render_sidebar()


# === 데이터 로드 ======================================================
fund = load_fund_results()  # default = mat_eq_mcap_raw_rms
ret = fund["ret"]
spy = fund["spy_ret"]
config_name = fund["config"]["name"]

panel = load_monthly_panel()
rf = panel.groupby("date")["rf_1m"].first()


# === 영역 1: Header ===================================================
render_page_header("Backtesting", "백테스트 검증")


# === 영역 2: Sub-header ===============================================
render_subheader(
    title_en="Backtesting",
    title_ko="백테스트 검증",
    description=(
        "Regime / Sub-events / Sensitivity 분석 — **TEST 평가 168m + HOLD_OUT 24m walk-forward** 결과의 깊이 있는 검증. "
        "다른 페이지가 \"결과 위주\" 라면 본 페이지는 **검증 / 강건성 narrative 전담**. "
        "신모델 `mat_eq_mcap_raw_rms` 의 Robustness + Regime 일관성 + 위기 방어 + 156 config Top 위치 학술 검증."
    ),
)


# === 영역 3: Backtest Summary KPI 5개 =================================
st.subheader("Backtest Summary — 검증 KPI 5")
st.caption(
    "**TEST/HO Gap** (학습편향 검증) / **Sensitivity Robustness** (156 config Top 위치) / "
    "**4-slot Robustness** (156 config Sortino mean/std) / **Avg Recovery** (위기 평균 회복) / "
    "**Regime 일관성** (R1/R2/R3/HO Sortino 일관성). FULL 기준."
)
render_backtest_kpi(ret, spy, rf, current_config=config_name)
st.divider()


# === 영역 4: Regime 메트릭 자세한 비교 =================================
st.subheader("Regime 메트릭 자세한 비교")
st.caption(
    "12 메트릭 × 5 Regime (R1 회복 / R2 확장 / R3 변동 / HO 24m / FULL). "
    "Performance 페이지의 Regime Heatmap 보다 **자세한 버전** — Active Return / IR / Calmar / VaR-CVaR 등 추가. "
    "★ Best Regime / 🔴 Worst Regime 강조 + Sortino 막대 시각."
)
render_regime_detail_table(ret, spy, rf)
st.divider()


# === 영역 5: Sub-events 분석 (4 위기) =================================
st.subheader("Sub-events 분석 — 4 위기")
st.caption(
    "**2018 Q4 Sell-off / COVID-19 / 2022 Inflation Bear / 2024 Sector Rotation** ★. "
    "위기별 Fund vs SPY Active Return + MDD + Recovery Time. "
    "**2024-12 위기는 Sector Watch 페이지 영역 8 (HO 정당화 narrative) 와 직접 연결**."
)
render_sub_events(ret, spy, rf)
st.divider()


# === 영역 6: Sensitivity Test =========================================
st.subheader("Sensitivity Test — 156 config 비교")
st.caption(
    "**156 config 의 Sortino Top N + 신모델 ★ 강조**. "
    "Top N Sortino 차이 → 4-slot 변경에도 결과 안정 (robust) 검증. "
    "config 차원 (p_weight / q_mode / omega_mode) 도 표시 — 어떤 4-slot 조합이 우수한지 비교 가능."
)
render_sensitivity_test(current_config=config_name)
st.divider()


# === 영역 7: Footer ===================================================
render_footer()
