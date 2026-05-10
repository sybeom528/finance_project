"""
lib/simulator_charts.py — Investment Simulator 페이지 영역 3, 4, 5 시각화

영역:
  3. render_input_section — Input Tab (Lump-sum / DCA / Goal) + 다단 입력
  4. render_kpi_section — KPI 5개 카드 (최종자산 / 총수익 / CAGR / MDD / 총투자)
  5. render_cumulative_curve — 누적 자산 곡선 (Fund + 사이드바 토글 SPY/EW/IVW)

영역 6 (Insight) 는 lib/insight_generator.py 활용.

참조: docs/plan/03_pages/02_simulator.md, decisionlog/11_dl_sections.md F-6
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import simulator as sim
from lib.colors import BENCHMARK_COLORS, COLORS
from lib.plot_helpers import add_event_annotations, add_regime_backgrounds


# ======================================================================
# 영역 3: Input 영역 (Tab 전환 + 다단 입력)
# ======================================================================

def render_input_section() -> dict:
    """
    Tab 전환 + 시점 입력 + 금액 입력. dict 반환:
      {
        "scenario": "lump_sum" / "dca" / "goal",
        "start_date": pd.Timestamp,
        "end_date": pd.Timestamp,
        "initial_amount": float,
        "monthly_amount": float | None,  # DCA only
        "goal_amount": float | None,     # Goal only
      }

    데이터 범위 (2010-01-31 ~ 2025-12-31) 안에서만 가능.
    """
    DATA_MIN = pd.Timestamp("2010-01-31")
    DATA_MAX = pd.Timestamp("2025-12-31")

    # === Number ↔ Slider 양방향 동기화 helper ===
    def _ensure_state(key: str, default: int) -> None:
        if key not in st.session_state:
            st.session_state[key] = default

    def _bind(num_key: str, slider_key: str, default: int):
        """number_input 과 slider 가 같은 logical 값 공유 — on_change callback 양방향."""
        _ensure_state(num_key, default)
        _ensure_state(slider_key, default)

        def _from_num():
            st.session_state[slider_key] = st.session_state[num_key]

        def _from_slider():
            st.session_state[num_key] = st.session_state[slider_key]

        return _from_num, _from_slider

    # === 시나리오 단일 selector (radio) — Tab 제거 (UI 단일화) ===
    scenario = st.radio(
        "시나리오 선택",
        options=["lump_sum", "dca", "goal"],
        format_func=lambda s: {
            "lump_sum": "💵 Lump-sum (일시 투자)",
            "dca": "🔄 DCA (분산 투자)",
            "goal": "🎯 Goal-based (목표 역산)",
        }[s],
        horizontal=True,
        key="sim_scenario",
    )

    # 시나리오별 caption
    if scenario == "lump_sum":
        st.caption("일시 투자 — 시작 시점에 한 번 투자하고 종료 시점까지 보유.")
    elif scenario == "dca":
        st.caption("Dollar Cost Averaging — 매월 일정 금액 추가 투자 (Constantinides 1979).")
    else:
        st.caption("Goal-based 역산 — 시작/종료 + 목표 금액 → 필요 초기 투자 금액.")

    # 공통: 시작 / 종료 시점
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "시작 시점", value=pd.Timestamp("2015-01-31").date(),
            min_value=DATA_MIN.date(), max_value=DATA_MAX.date(),
            key="sim_start_date",
        )
    with c2:
        end_date = st.date_input(
            "종료 시점", value=DATA_MAX.date(),
            min_value=DATA_MIN.date(), max_value=DATA_MAX.date(),
            key="sim_end_date",
        )

    sim_input: dict = {
        "scenario": scenario,
        "start_date": pd.Timestamp(start_date),
        "end_date": pd.Timestamp(end_date),
    }

    # 시나리오별 입력 필드 (조건부 렌더링)
    if scenario == "lump_sum":
        on_num, on_slider = _bind("sim_lump_initial", "sim_lump_initial_slider", 10_000)
        st.number_input(
            "초기 금액 ($)", min_value=100, max_value=1_000_000, step=1_000,
            key="sim_lump_initial", on_change=on_num,
        )
        st.slider(
            "초기 금액 슬라이더",
            min_value=100, max_value=1_000_000, step=1_000,
            key="sim_lump_initial_slider", on_change=on_slider,
            label_visibility="collapsed",
        )
        sim_input["initial_amount"] = float(st.session_state.sim_lump_initial)

    elif scenario == "dca":
        # 초기 금액
        on_num_di, on_slider_di = _bind("sim_dca_initial", "sim_dca_initial_slider", 10_000)
        st.number_input(
            "초기 금액 ($)", min_value=0, max_value=1_000_000, step=1_000,
            key="sim_dca_initial", on_change=on_num_di,
        )
        st.slider(
            "초기 금액 슬라이더",
            min_value=0, max_value=1_000_000, step=1_000,
            key="sim_dca_initial_slider", on_change=on_slider_di,
            label_visibility="collapsed",
        )
        # 매월 추가
        on_num_dm, on_slider_dm = _bind("sim_dca_monthly", "sim_dca_monthly_slider", 500)
        st.number_input(
            "매월 추가 ($)", min_value=0, max_value=100_000, step=100,
            key="sim_dca_monthly", on_change=on_num_dm,
        )
        st.slider(
            "매월 추가 슬라이더",
            min_value=0, max_value=100_000, step=100,
            key="sim_dca_monthly_slider", on_change=on_slider_dm,
            label_visibility="collapsed",
        )
        sim_input["initial_amount"] = float(st.session_state.sim_dca_initial)
        sim_input["monthly_amount"] = float(st.session_state.sim_dca_monthly)

    else:  # goal
        on_num_g, on_slider_g = _bind("sim_goal_amount", "sim_goal_amount_slider", 1_000_000)
        st.number_input(
            "목표 금액 ($)", min_value=10_000, max_value=10_000_000, step=10_000,
            key="sim_goal_amount", on_change=on_num_g,
        )
        st.slider(
            "목표 금액 슬라이더",
            min_value=10_000, max_value=10_000_000, step=10_000,
            key="sim_goal_amount_slider", on_change=on_slider_g,
            label_visibility="collapsed",
        )
        sim_input["goal_amount"] = float(st.session_state.sim_goal_amount)

    return sim_input


# ======================================================================
# 영역 4: Result KPI 카드 5개
# ======================================================================

def render_kpi_section(result: dict) -> None:
    """
    KPI 5개 카드: 최종자산 / 총수익 / CAGR / MDD / 총투자.
    """
    cols = st.columns(5)

    with cols[0]:
        st.metric(
            label="최종 자산 — Final Value",
            value=f"${result['final_value']:,.0f}",
            help="시뮬레이션 종료 시점의 평가액",
        )
    with cols[1]:
        profit = result["total_profit"]
        invested = result["total_invested"]
        pct = profit / invested * 100 if invested > 0 else 0
        st.metric(
            label="총 수익 — Total Profit",
            value=f"${profit:+,.0f}",
            delta=f"{pct:+.2f}%",
            delta_color="normal",
            help="최종 자산 − 총 투자 원금",
        )
    with cols[2]:
        st.metric(
            label="연환산 수익률 — CAGR",
            value=f"{result['cagr'] * 100:+.2f}%",
            help="(Final / Invested)^(12/N) − 1. DCA 시 단순 근사 (정확 값은 IRR).",
        )
    with cols[3]:
        st.metric(
            label="최대 낙폭 — Max Drawdown",
            value=f"{result['mdd'] * 100:.2f}%",
            help="시뮬레이션 기간 동안 최대 자산 손실 (peak → trough).",
        )
    with cols[4]:
        st.metric(
            label="총 투자 — Total Invested",
            value=f"${result['total_invested']:,.0f}",
            help="투자 원금 합계 (Lump-sum=초기금액 / DCA=초기+매월 합계).",
        )


# ======================================================================
# 영역 5: 누적 자산 곡선
# ======================================================================

def render_cumulative_curve(
    result: dict,
    benchmarks: dict,
    initial_amount: float,
) -> None:
    """
    누적 자산 곡선 — Fund (메인) + 사이드바 토글 활성 벤치마크 + DCA 누적 투자금액 (DCA 시).

    benchmarks: {"SPY": value_series, "EW": ..., "IVW": ...} (활성 토글만)
    initial_amount: 벤치마크 normalize 기준 (Fund value 와 동일 scale 유지)
    """
    fig = go.Figure()

    # Fund (메인 라인)
    value_series = result.get("value_series")
    if value_series is None or len(value_series) == 0:
        st.warning("시뮬레이션 데이터 없음 — 시점 / 금액을 확인해주세요.")
        return

    fig.add_trace(go.Scatter(
        x=value_series.index, y=value_series.values,
        name="Fund (Adaptive VolControl)",
        mode="lines",
        line=dict(color=BENCHMARK_COLORS["Fund"], width=2.5),
        hovertemplate="%{x|%Y-%m}<br><b>Fund</b>: $%{y:,.0f}<extra></extra>",
    ))

    # DCA 누적 투자금액 라인 (DCA 시나리오만)
    if result["scenario"] == "dca" and "invested_series" in result:
        inv = result["invested_series"]
        fig.add_trace(go.Scatter(
            x=inv.index, y=inv.values,
            name="DCA 누적 투자금액",
            mode="lines",
            line=dict(color=COLORS["text_muted"], width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m}<br>총 투자: $%{y:,.0f}<extra></extra>",
        ))

    # 벤치마크 라인 — simulate_benchmark 결과 (이미 dollar value, Fund 와 동일 시나리오)
    for name, bench_data in benchmarks.items():
        bench_value = bench_data.get("value_series")
        if bench_value is None or len(bench_value) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=bench_value.index, y=bench_value.values,
            name=name,
            mode="lines",
            line=dict(color=BENCHMARK_COLORS.get(name, COLORS["text_muted"]), width=1.5),
            hovertemplate="%{x|%Y-%m}<br>" + name + ": $%{y:,.0f}<extra></extra>",
        ))

    fig = add_regime_backgrounds(fig, with_labels=False)
    fig = add_event_annotations(fig)

    # Y축 토글 (Linear / Log)
    yaxis_type = st.radio(
        "Y축",
        options=["Linear", "Log"],
        index=0,
        horizontal=True,
        key="sim_curve_yaxis",
        label_visibility="collapsed",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        xaxis_title="시점", yaxis_title="자산 ($)",
        yaxis=dict(tickformat="$,.0f", type="log" if yaxis_type == "Log" else "linear"),
        height=480, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    st.plotly_chart(fig, use_container_width=True)
