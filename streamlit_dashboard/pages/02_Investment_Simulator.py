"""
pages/02_Investment_Simulator.py — Investment Simulator 페이지 (7 영역)

★ F-section 핵심 페이지 — 5분 demo 의 1.5분 차지. 가상 투자자 친화 마케팅 핵심.

7 영역:
  1. Header
  2. Sub-header (친근 톤) + Sim disclaimer
  3. Input 영역 (Tab: Lump-sum / DCA / Goal-based + 다단 입력)
  4. Result KPI 카드 5개 (Final / Profit / CAGR / MDD / Invested)
  5. 누적 자산 곡선 (Fund + 사이드바 토글 SPY/EW/IVW + DCA 누적 + Regime + 이벤트)
  6. Insight 박스 (카드 그리드 4-8개 조건부)
  7. Footer

시뮬레이션 로직:
  - Lump-sum: cumprod(1+r) (학술 표준)
  - DCA: 매월 추가 매수 (Constantinides 1979)
  - Goal-based: closed-form 역산

참조: docs/plan/03_pages/02_simulator.md, decisionlog/11_dl_sections.md F-6
"""

import streamlit as st

from lib.data_loader import (
    compute_equal_weight_returns,
    compute_ivw_returns,
    load_fund_results,
    load_monthly_panel,
    load_sp500_membership,
)
from lib.disclosure import init_session_state, render_footer, render_simulator_disclaimer
from lib.insight_generator import generate_insight_cards, render_insight_grid
from lib.page_helpers import (
    inject_custom_css,
    render_page_header,
    render_sidebar,
    render_subheader,
)
from lib.simulator import (
    simulate_benchmark,
    simulate_dca,
    simulate_goal_based,
    simulate_lump_sum,
)
from lib.simulator_charts import (
    render_cumulative_curve,
    render_input_section,
    render_kpi_section,
)


# === 페이지 설정 ======================================================
inject_custom_css()
init_session_state()
render_sidebar()


# === 데이터 로드 ======================================================
fund = load_fund_results()  # default = mat_eq_mcap_raw_he
fund_ret = fund["ret"]
fund_spy = fund["spy_ret"]


# === 영역 1: Header ===================================================
render_page_header("Investment Simulator", "내 투자 시뮬레이션")


# === 영역 2: Sub-header (친근 톤) + Sim disclaimer =====================
render_subheader(
    title_en="Investment Simulator",
    title_ko="내 투자 시뮬레이션",
    description=(
        "**\"내가 이때 얼마를 투자했더라면?\"** 실제 수익을 시뮬레이션해 보세요. "
        "Lump-sum (일시 투자) / DCA (분산 투자) / Goal-based (목표 역산) 3가지 시나리오. "
        "사이드바에서 비교 벤치마크 (SPY / EW / IVW) 토글 가능."
    ),
)
render_simulator_disclaimer()


# === 영역 3: Input 영역 ================================================
st.subheader("시뮬레이션 입력")
sim_input = render_input_section()
st.divider()


# === 시뮬레이션 실행 (시나리오별) =====================================
scenario = sim_input["scenario"]
start_date = sim_input["start_date"]
end_date = sim_input["end_date"]

if scenario == "lump_sum":
    result = simulate_lump_sum(fund_ret, start_date, end_date, sim_input["initial_amount"])
    initial_amount_for_curve = sim_input["initial_amount"]
elif scenario == "dca":
    result = simulate_dca(
        fund_ret, start_date, end_date,
        sim_input["initial_amount"], sim_input["monthly_amount"],
    )
    initial_amount_for_curve = sim_input["initial_amount"] or 1.0
else:  # goal
    result = simulate_goal_based(fund_ret, start_date, end_date, sim_input["goal_amount"])
    initial_amount_for_curve = result.get("required_initial", 10_000)


# === 영역 4: Result KPI 카드 5개 ======================================
st.subheader("시뮬레이션 결과")
if scenario == "goal":
    req = result.get("required_initial")
    period_years = (end_date - start_date).days / 365.25
    st.info(
        f"🎯 **Goal 역산 결과** — 목표 \\${result['goal_amount']:,.0f} 달성을 위한 "
        f"필요 초기 투자금 = **\\${req:,.0f}**\n\n"
        f"이 금액으로 시작 시 {period_years:.1f}년 후 종료 시점에 정확히 "
        f"\\${result['goal_amount']:,.0f} 도달. "
        f"펀드의 누적 factor = **{result['goal_amount'] / req:.2f}배**."
    )
render_kpi_section(result)
st.divider()


# === 영역 5: 누적 자산 곡선 ===========================================
st.subheader("누적 자산 곡선")
st.caption(
    "Fund (Adaptive VolControl) 자산 변화 시계열. "
    "사이드바에서 비교 벤치마크 (SPY / EW / IVW) 토글 시 동일 시작 금액 normalize 라인 추가. "
    "DCA 시나리오 시 누적 투자금액 점선 표시. "
    "Regime 배경 (R1/R2/R3/HO) + COVID/2022 Bear/2024 AI Rally 이벤트 annotation. "
    "Y축 Linear/Log 토글 + 기간 슬라이더 가능."
)

# 활성 벤치마크만 산출 — Fund 와 동일 시나리오 (DCA 시 매월 추가 동일 적용)
bench_kwargs = dict(
    scenario=scenario,
    start_date=start_date,
    end_date=end_date,
    initial_amount=initial_amount_for_curve,
    monthly_amount=sim_input.get("monthly_amount", 0) if scenario == "dca" else 0,
)
benchmarks: dict = {}
if st.session_state.get("show_spy", True):
    benchmarks["SPY"] = simulate_benchmark(fund_spy, **bench_kwargs)
if st.session_state.get("show_ew", False) or st.session_state.get("show_ivw", False):
    panel = load_monthly_panel()
    sp_mem = load_sp500_membership()
    if st.session_state.get("show_ew", False):
        with st.spinner("EW baseline 산출 중..."):
            ew_ret = compute_equal_weight_returns(panel, sp_mem, fund_ret.index)
        benchmarks["EW"] = simulate_benchmark(ew_ret, **bench_kwargs)
    if st.session_state.get("show_ivw", False):
        with st.spinner("IVW baseline 산출 중..."):
            ivw_ret = compute_ivw_returns(panel, sp_mem, fund_ret.index)
        benchmarks["IVW"] = simulate_benchmark(ivw_ret, **bench_kwargs)

render_cumulative_curve(result, benchmarks, initial_amount_for_curve)
st.divider()


# === 영역 6: Insight 박스 (카드 그리드) ===============================
st.subheader("인사이트 — Insights")
st.caption(
    "시뮬레이션 결과 + 활성 벤치마크 + 시나리오별 조건부 카드 (4-8개). "
    "정적 템플릿 + 동적 값 (LLM 미사용) — 빠른 응답 + 일관 메시지."
)
cards = generate_insight_cards(result, benchmarks, scenario)
render_insight_grid(cards, cols_per_row=3)


# === 영역 7: Footer ===================================================
render_footer()
