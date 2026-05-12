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
# 헬퍼 함수 — 시점 변환 (월말 snap, 2026-05-12 신규)
# ======================================================================

def _snap_to_month_end(
    input_date,
    available_dates: pd.DatetimeIndex,
    direction: str = "forward",
) -> pd.Timestamp:
    """
    일별 날짜를 펀드 backtest 의 월말 시점으로 변환 (snap to month-end).

    펀드는 월말 리밸런싱 모델 (`monthly_panel.fwd_ret_1m` 산식) 이므로 backtest 결과
    가 월말 시점에만 존재. 사용자가 입력한 임의 일별 날짜를 산식 정합되는 월말로 변환.

    Parameters
    ----------
    input_date : pd.Timestamp | datetime.date
        사용자가 date_input 으로 입력한 일별 날짜
    available_dates : pd.DatetimeIndex
        펀드 backtest 가 존재하는 월말 시점 목록 (예: fund_ret.index)
    direction : {"forward", "backward"}
        - "forward": 입력일 이후 첫 월말 (시작 시점용)
        - "backward": 입력일 이전 마지막 월말 (종료 시점용)

    Returns
    -------
    pd.Timestamp
        조정된 월말 시점. 입력일이 가능 범위 밖이면 가장 근접한 경계 반환.
    """
    input_ts = pd.Timestamp(input_date)
    if direction == "forward":
        candidates = available_dates[available_dates >= input_ts]
        return candidates[0] if len(candidates) > 0 else available_dates[-1]
    else:  # backward
        candidates = available_dates[available_dates <= input_ts]
        return candidates[-1] if len(candidates) > 0 else available_dates[0]


# ======================================================================
# 영역 3: Input 영역 (Tab 전환 + 다단 입력)
# ======================================================================

def render_input_section(available_dates: pd.DatetimeIndex | None = None) -> dict:
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

    Parameters
    ----------
    available_dates : pd.DatetimeIndex | None (2026-05-12 신규)
        펀드 backtest 가 존재하는 월말 시점 목록 (예: fund_ret.index).
        - None: 시점 변환 안 함 (기존 동작, backwards-compatible)
        - 제공: 입력 일별 날짜를 가장 가까운 월말로 자동 변환 + 변환 결과 안내
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
            "dca": "🔄 DCA (적립식 투자)",
            "goal": "🎯 Goal-based (목표 역산)",
        }[s],
        horizontal=True,
        key="sim_scenario",
    )

    # 시나리오별 caption
    if scenario == "lump_sum":
        st.caption("일시 투자 — 시작 시점에 한 번 투자하고 종료 시점까지 보유.")
    elif scenario == "dca":
        st.caption("적립식 투자 (Dollar Cost Averaging) — 매월 일정 금액 추가 투자.")
    else:
        st.caption("Goal-based 역산 — 시작/종료 + 목표 금액 → 필요 초기 투자 금액.")

    # 공통: 시작 / 종료 시점
    st.caption(
        "**백테스팅 기간: 2010년 1월 ~ 2025년 12월 (총 192개월) 만 지원됩니다.** "
        "본 펀드는 매월 한 번씩 종목 비중을 재조정하므로 성과는 월말 단위로 측정되며, "
        "임의의 날짜를 입력하시면 가장 가까운 월말로 자동 변환됩니다."
    )
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "시작 시점", value=pd.Timestamp("2010-01-31").date(),
            min_value=DATA_MIN.date(), max_value=DATA_MAX.date(),
            key="sim_start_date",
        )
    with c2:
        end_date = st.date_input(
            "종료 시점", value=DATA_MAX.date(),
            min_value=DATA_MIN.date(), max_value=DATA_MAX.date(),
            key="sim_end_date",
        )

    # 일별 입력 → 월말 시점 변환 (펀드 backtest 산식 정합)
    if available_dates is not None and len(available_dates) > 0:
        start_snapped = _snap_to_month_end(start_date, available_dates, "forward")
        end_snapped = _snap_to_month_end(end_date, available_dates, "backward")
        messages = []
        if pd.Timestamp(start_date) != start_snapped:
            messages.append(f"시작: `{start_date}` → **{start_snapped.date()}**")
        if pd.Timestamp(end_date) != end_snapped:
            messages.append(f"종료: `{end_date}` → **{end_snapped.date()}**")
        if messages:
            st.info(
                "🔄 **입력하신 날짜가 가장 가까운 월말로 자동 변환되었습니다.** "
                "시작 시점은 입력일 이후 첫 월말로, 종료 시점은 입력일 이전 마지막 월말로 변환됩니다.\n\n"
                + "\n".join([f"- {m}" for m in messages])
            )
        start_date, end_date = start_snapped.date(), end_snapped.date()

    # 시점 산정 방식 안내 expander
    with st.expander("ℹ️ 시점 산정 방식 안내"):
        st.markdown(
            """
            **왜 월말 시점만 사용하나요?**
            - 본 펀드는 **매월 말일에 한 번씩** 종목 비중을 재조정합니다 (월별 리밸런싱 방식)
            - 따라서 펀드 성과 데이터는 매월 말일 기준으로 정확하게 측정되어 있습니다
            - 사용 가능한 기간: **2010년 1월 ~ 2025년 12월 (총 192개월)**

            **시점 변환 규칙**
            - **시작 시점**: 입력하신 날짜 이후의 첫 월말로 변환
              - 예시: 2015-01-15 입력 → 2015-01-31 로 시작
            - **종료 시점**: 입력하신 날짜 이전의 마지막 월말로 변환
              - 예시: 2025-12-15 입력 → 2025-11-30 로 종료
            - 정확히 월말 날짜를 입력하시면 그대로 사용됩니다

            **왜 일별 시점은 지원하지 않나요?**
            - 본 펀드의 성과는 월말 시점에만 실제로 측정되어 있습니다
            - 임의의 일별 시점에 대한 성과는 가정에 기반한 추정이 필요한데, 이는 실제 펀드 운용 방식과 차이가 발생할 수 있습니다
            - 본 시뮬레이션은 **실제로 측정된 시점만 사용**하여 정확한 결과를 보장합니다
            """
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
