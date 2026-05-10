"""
pages/08_Backtesting.py — Backtesting (Robustness & Validation) 페이지 (8 영역)

학술 정직성 — 본 페이지는 백테스팅의 **검증 / 강건성 측면** 전담 (walk-forward
/ stress test 의 정확한 시뮬레이션 어려움 → 가능한 검증만 표시).

8 영역:
  1. Header
  2. Sub-header (narrative 재정의 — Robustness & Validation)
  3. Backtest Summary KPI 5
  4. ★ 156 config 누적 수익률 비교 (Robustness 시각 — 신규)
  5. Regime 메트릭 자세한 비교 (12 메트릭 × 5 Regime)
  6. Sub-events 분석 (4 위기)
  7. Sensitivity Test (156 config Top N)
  8. Footer

설계 원칙 (초안 — 향후 변경 용이):
  - 영역별 독립 호출 (영역 추가/제거/변경 용이)
  - 메트릭 / 위기 정의는 metric_calculators.py 의 외부 dict
  - 데이터: 우리 펀드 pkl + 156 config (이미 git 포함) + monthly_panel

참조: docs/plan/03_pages/08_backtesting.md, decisionlog/08_backtesting.md
"""

import streamlit as st

from lib.backtesting_charts import (
    render_backtest_kpi,
    render_cumulative_comparison,
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
render_page_header("Backtesting", "Robustness & Validation")


# === 영역 2: Sub-header (narrative 재정의) ============================
render_subheader(
    title_en="Backtesting — Robustness & Validation",
    title_ko="백테스트 검증 — 강건성 분석",
    description=(
        "본 페이지는 백테스팅의 **검증 / 강건성 측면** 전담. "
        "BL+LSTM 자체의 walk-forward 동작 + 누적 수익률은 **Methodology / Performance 페이지** 참조. "
        "여기서는 **156 config Robustness + Regime 일관성 + 위기 방어 + 학습편향 검증** 에 집중. "
        "학술 정직성 — Stress Test / Walk-forward 의 정확한 시뮬레이션 어려움으로 일부 의도적 제외 (Methodology 영역 8 한계 참조)."
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


# === 영역 4: 156 config 누적 수익률 비교 (★ 신규) =====================
st.subheader("156 config 누적 수익률 비교 — Robustness 시각")
st.caption(
    "**156 config 의 누적 수익률 분포** + 신모델 ★ 강조 + SPY 비교. "
    "Spaghetti (156 라인) 또는 Percentile (P5-P95 fill + median) 토글. "
    "\"Backtesting = 과거 시뮬레이션\" 의 직관 표현 — 모든 config 가 어떻게 진행되었는지 한눈에. "
    "**핵심 narrative**: 156 config 모두 SPY 수준 이상 + 신모델 Top 위치 = 모델 자체가 robust."
)
render_cumulative_comparison(ret, spy, current_config=config_name)
st.divider()


# === 영역 5: Regime 메트릭 자세한 비교 ================================
st.subheader("Regime 메트릭 자세한 비교")
st.caption(
    "12 메트릭 × 5 Regime (R1 회복 / R2 확장 / R3 변동 / HO 24m / FULL). "
    "Performance 페이지의 Regime Heatmap 보다 **자세한 버전** — Active Return / IR / Calmar / VaR-CVaR 등 추가. "
    "★ Best Regime / 🔴 Worst Regime 강조 + Sortino 막대 시각."
)
render_regime_detail_table(ret, spy, rf)
st.divider()


# === 영역 6: Sub-events 분석 (4 위기) =================================
st.subheader("Sub-events 분석 — 4 위기")
st.caption(
    "**2018 Q4 Sell-off / COVID-19 / 2022 Inflation Bear / 2024 Sector Rotation** ★. "
    "위기별 Fund vs SPY Active Return + MDD + Recovery Time. "
    "**2024-12 위기는 Sector Watch 페이지 영역 8 (HO 정당화 narrative) 와 직접 연결**."
)
render_sub_events(ret, spy, rf)
st.divider()


# === 영역 7: Sensitivity Test =========================================
st.subheader("Sensitivity Test — 156 config 비교")
st.caption(
    "**156 config 의 Sortino Top N + 신모델 ★ 강조**. "
    "Top N Sortino 차이 → 4-slot 변경에도 결과 안정 (robust) 검증. "
    "config 차원 (p_weight / q_mode / omega_mode) 도 표시 — 어떤 4-slot 조합이 우수한지 비교 가능."
)
render_sensitivity_test(current_config=config_name)
st.divider()


# === 영역 8: Footer ===================================================
render_footer()
