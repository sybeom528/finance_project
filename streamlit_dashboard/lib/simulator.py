"""
lib/simulator.py — Investment Simulator 시뮬레이션 로직

시나리오 3종:
  - Lump-sum: 일시 투자 — value(t) = initial × ∏(1 + r(s))
  - DCA: 매월 추가 매수 — value(t) = value(t-1) × (1 + r(t)) + monthly
  - Goal-based: 목표 금액 → 필요 초기 금액 역산 (binary search)

KPI 5개:
  - final_value (최종 자산)
  - total_profit (= final - total_invested)
  - cagr (연환산 수익률 — DCA 는 (final/invested)^(12/N)-1 단순 근사)
  - mdd (최대 낙폭, value cumprod 기준)
  - total_invested (총 투자 원금)

학술 출처:
  - Lump-sum cumprod = 표준 누적 수익률
  - DCA = Dollar Cost Averaging (Constantinides 1979)
  - CAGR (DCA) = 단순 근사 — IRR (Internal Rate of Return) 이 정확하지만 narrative 우선

참조: docs/plan/03_pages/02_simulator.md, decisionlog/11_dl_sections.md F-6
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _calc_mdd_from_value(value_series: pd.Series) -> float:
    """value 시계열에서 최대 낙폭 (음수)."""
    if len(value_series) == 0:
        return 0.0
    peak = value_series.cummax()
    dd = (value_series / peak - 1).min()
    return float(dd) if not pd.isna(dd) else 0.0


def _calc_recovery_months(value_series: pd.Series) -> int | None:
    """
    MDD trough → 신고가 회복 개월 수. 미회복 시 None.
    """
    if len(value_series) < 2:
        return None
    peak = value_series.cummax()
    dd = value_series / peak - 1
    trough_idx = dd.idxmin()
    if pd.isna(trough_idx):
        return None
    pre_peak_value = peak.loc[trough_idx]
    after = value_series.loc[trough_idx:]
    recovered = after[after >= pre_peak_value]
    if len(recovered) == 0:
        return None
    recovery_t = recovered.index[0]
    months = (recovery_t.year - trough_idx.year) * 12 + (recovery_t.month - trough_idx.month)
    return int(months)


def simulate_lump_sum(
    fund_ret: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_amount: float,
) -> dict:
    """
    Lump-sum (일시 투자) 시뮬레이션.

    value(t) = initial × ∏_{s ≤ t} (1 + r(s))
    """
    period_ret = fund_ret[(fund_ret.index >= start_date) & (fund_ret.index <= end_date)].dropna()
    if len(period_ret) == 0:
        return {
            "scenario": "lump_sum",
            "total_invested": float(initial_amount),
            "final_value": float(initial_amount),
            "total_profit": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "recovery_months": None,
            "value_series": pd.Series(dtype=float),
        }

    cumulative = (1 + period_ret).cumprod()
    value_series = float(initial_amount) * cumulative

    final_value = float(value_series.iloc[-1])
    n_months = len(period_ret)
    cagr = (final_value / initial_amount) ** (12 / n_months) - 1 if initial_amount > 0 else 0.0

    return {
        "scenario": "lump_sum",
        "total_invested": float(initial_amount),
        "final_value": final_value,
        "total_profit": final_value - float(initial_amount),
        "cagr": float(cagr),
        "mdd": _calc_mdd_from_value(value_series),
        "recovery_months": _calc_recovery_months(value_series),
        "value_series": value_series,
        "n_months": n_months,
    }


def simulate_dca(
    fund_ret: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_amount: float,
    monthly_amount: float,
) -> dict:
    """
    DCA (Dollar Cost Averaging) 시뮬레이션.

    매월 말:
      value(t) = value(t-1) × (1 + r(t)) + monthly_amount
    초기 시점 (t=0):
      value(0) = initial_amount

    학술 출처: Constantinides (1979) "A note on the suboptimality of dollar-cost
    averaging as an investment policy" — DCA 는 일시 투자 대비 보통 underperform
    이지만 평균 매입가 안정 + risk averse 투자자 친화 narrative.
    """
    period_ret = fund_ret[(fund_ret.index >= start_date) & (fund_ret.index <= end_date)].dropna()
    if len(period_ret) == 0:
        return {
            "scenario": "dca",
            "total_invested": float(initial_amount),
            "final_value": float(initial_amount),
            "total_profit": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "recovery_months": None,
            "value_series": pd.Series(dtype=float),
            "invested_series": pd.Series(dtype=float),
            "dca_monthly": float(monthly_amount),
            "dca_advantage": None,
        }

    value = float(initial_amount)
    values: list[float] = []
    invested_cum: list[float] = []
    for i, r in enumerate(period_ret.values):
        # 월말 returns 후 추가 매수
        value = value * (1 + r) + float(monthly_amount)
        values.append(value)
        invested_cum.append(float(initial_amount) + float(monthly_amount) * (i + 1))

    value_series = pd.Series(values, index=period_ret.index)
    invested_series = pd.Series(invested_cum, index=period_ret.index)

    final_value = float(value_series.iloc[-1])
    total_invested = float(invested_series.iloc[-1])
    n_months = len(period_ret)
    cagr = (final_value / total_invested) ** (12 / n_months) - 1 if total_invested > 0 else 0.0

    # 일시 투자 대비 차이 (동일 총액 lump-sum 비교)
    lump_result = simulate_lump_sum(fund_ret, start_date, end_date, total_invested)
    dca_advantage = final_value - lump_result["final_value"]

    return {
        "scenario": "dca",
        "total_invested": total_invested,
        "final_value": final_value,
        "total_profit": final_value - total_invested,
        "cagr": float(cagr),
        "mdd": _calc_mdd_from_value(value_series),
        "recovery_months": _calc_recovery_months(value_series),
        "value_series": value_series,
        "invested_series": invested_series,
        "dca_monthly": float(monthly_amount),
        "dca_advantage": float(dca_advantage),
        "n_months": n_months,
    }


def simulate_goal_based(
    fund_ret: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    goal_amount: float,
) -> dict:
    """
    Goal-based 역산 — 주어진 시작/종료 + 목표 금액 → 필요 초기 금액.

    closed-form: required_initial = goal / ∏(1 + r) (lump-sum 가정)

    또한 누적 자산이 goal 에 도달하는 시점도 산출 (initial=$10K 기준).
    """
    period_ret = fund_ret[(fund_ret.index >= start_date) & (fund_ret.index <= end_date)].dropna()
    if len(period_ret) == 0:
        return {
            "scenario": "goal",
            "total_invested": 0.0,
            "final_value": 0.0,
            "total_profit": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "recovery_months": None,
            "value_series": pd.Series(dtype=float),
            "goal_amount": float(goal_amount),
            "required_initial": np.nan,
            "goal_achievement_date": None,
        }

    cumulative = (1 + period_ret).cumprod()
    cum_factor = float(cumulative.iloc[-1])
    required_initial = float(goal_amount) / cum_factor if cum_factor > 0 else np.nan

    # 시뮬레이션은 필요 초기 금액 기준 (역산 결과 동일)
    sim = simulate_lump_sum(fund_ret, start_date, end_date, required_initial)
    sim["scenario"] = "goal"
    sim["goal_amount"] = float(goal_amount)
    sim["required_initial"] = required_initial

    # 도달 시점 ($10K 기준 별도 산출 — 직관적 narrative)
    ref_value = 10_000.0 * cumulative
    reached = ref_value[ref_value >= float(goal_amount)]
    if len(reached) > 0:
        sim["goal_achievement_date"] = reached.index[0].strftime("%Y-%m")
    else:
        sim["goal_achievement_date"] = None
    return sim


def simulate_benchmark(
    bench_ret: pd.Series,
    scenario: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_amount: float = 10_000,
    monthly_amount: float = 0,
) -> dict:
    """
    벤치마크 (SPY/EW/IVW) 의 시나리오별 시뮬레이션 — Fund 와 동일 전략 적용.

    DCA 시나리오에서 Fund 만 매월 추가 매수하고 SPY 는 lump-sum 가정하면
    비대칭 비교 → 본 함수로 SPY 도 동일 DCA 적용 (정확한 비교).

    Goal 시나리오: 역산은 Fund 만 적용. SPY 는 동일 initial 로 lump-sum.

    Returns:
        dict {scenario, total_invested, final_value, cagr, value_series, ...}
    """
    if scenario == "dca":
        return simulate_dca(bench_ret, start_date, end_date, initial_amount, monthly_amount)
    # lump_sum / goal 모두 lump-sum 시뮬레이션 (initial = 비교 기준)
    return simulate_lump_sum(bench_ret, start_date, end_date, initial_amount)


# Backward compatibility alias (이전 코드 호출용)
def compute_benchmark_cagr(bench_ret, start_date, end_date):
    """[Deprecated] simulate_benchmark 사용 권장. lump-sum 가정 (initial=1) 으로 cumulative factor 반환."""
    period = bench_ret[(bench_ret.index >= start_date) & (bench_ret.index <= end_date)].dropna()
    if len(period) == 0:
        return {"cagr": 0.0, "value_series": pd.Series(dtype=float)}
    cumulative = (1 + period).cumprod()
    cagr = float(cumulative.iloc[-1] ** (12 / len(period)) - 1)
    return {"cagr": cagr, "value_series": cumulative, "n_months": len(period)}
