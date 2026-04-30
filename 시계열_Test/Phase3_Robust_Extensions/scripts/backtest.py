"""Phase 2 — BL 백테스트 엔진 (transaction_cost 인자화 + 상장폐지 처리).

결정 1 (PLAN.md §2): 거래비용 0 default, 인자화 (추후 sensitivity 분석 가능).
결정 8 (PLAN.md §2): 상장폐지 종목 비중을 남은 종목 시가총액 비중으로 비례 전이.

핵심 함수
---------
- backtest_strategy()        : 매월 가중치 시계열 → portfolio return + transaction cost
- handle_delisting()         : 상장폐지 종목 비중 비례 재배분
- compute_portfolio_metrics(): Sharpe / alpha / MDD / CumRet 계산

설계 원칙
---------
- 누수 방지: 매월 t 의 weight 는 t 이전 정보로만 결정
- 거래비용 인자화: 0 default → 후속 실험에서 0.0005, 0.001 등 sensitivity
- 상장폐지 안전망: 종목 NaN return 시 비중 0 처리 + 남은 종목 재배분
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# 백테스트 (transaction_cost 인자화)
# =============================================================================
def backtest_strategy(
    weights_history: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost: float = 0.0,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """매월 가중치 시계열을 받아 portfolio return 계산.

    Parameters
    ----------
    weights_history : pd.DataFrame (rebalance × n)
        BL 리밸런싱 시점별 자산 비중. 행 = 리밸런싱 시점, 열 = 종목.
    returns : pd.DataFrame (T × n)
        월별 수익률 (자산별). 행 = 월, 열 = 종목.
    transaction_cost : float, default 0.0
        매 리밸런싱 회전당 거래비용 비율.
        0 = 무비용 가정 (본 단계 default)
        0.0005 = 0.05% (실무 보수)
        0.001 = 0.1% (일반)
        0.002 = 0.2% (보수)
    rebalance_dates : pd.DatetimeIndex | None
        리밸런싱 시점. None 이면 weights_history.index 사용.

    Returns
    -------
    portfolio_returns : pd.Series
        월별 net portfolio return (거래비용 차감).

    Notes
    -----
    각 시점 t 에서:
        gross_return[t] = Σ_i (w[t-1, i] · r[t, i])  (전월 가중치 × 당월 수익률)
        cost[t] = Σ_i |w[t, i] - w[t-1, i]| · transaction_cost  (turnover)
        net_return[t] = gross_return[t] - cost[t]

    누수 방지:
        weight 는 시점 t 직전에 결정된 값 사용 (forward-looking 방지)
        return 은 시점 t 의 실제 시장 수익률
    """
    if rebalance_dates is None:
        rebalance_dates = weights_history.index

    portfolio_returns = []
    prev_w = None

    for date in rebalance_dates:
        if date not in weights_history.index:
            continue
        cur_w = weights_history.loc[date]

        # 거래비용 (turnover × tc)
        if prev_w is not None and transaction_cost > 0:
            # 동일 종목 사이 가중치 차이의 절대값 합 = turnover
            common_idx = cur_w.index.intersection(prev_w.index)
            new_only = cur_w.index.difference(prev_w.index)
            old_only = prev_w.index.difference(cur_w.index)

            turnover = (
                (cur_w.loc[common_idx] - prev_w.loc[common_idx]).abs().sum()
                + cur_w.loc[new_only].abs().sum()  # 신규 진입 종목 매입
                + prev_w.loc[old_only].abs().sum()  # 이탈 종목 매도
            )
            cost = float(turnover) * transaction_cost
        else:
            cost = 0.0

        # gross return (당월 수익률)
        if date in returns.index:
            ret_today = returns.loc[date]
            common = cur_w.index.intersection(ret_today.index)
            ret_today_aligned = ret_today.reindex(common).fillna(0)
            cur_w_aligned = cur_w.reindex(common).fillna(0)
            gross_ret = float((cur_w_aligned * ret_today_aligned).sum())
        else:
            gross_ret = 0.0

        net_ret = gross_ret - cost
        portfolio_returns.append({'date': date, 'return': net_ret})
        prev_w = cur_w

    out = pd.DataFrame(portfolio_returns).set_index('date')['return']
    out.name = 'portfolio_return'
    return out


# =============================================================================
# 상장폐지 처리 (비중 비례 전이)
# =============================================================================
def handle_delisting(
    weights: pd.Series,
    delisted: list,
    valid_tickers: list,
    mcaps: pd.Series,
) -> pd.Series:
    """상장폐지 종목 비중을 남은 종목 시가총액 비중으로 비례 전이.

    Parameters
    ----------
    weights : pd.Series (n,)
        현재 시점 portfolio 가중치.
    delisted : list of str
        본 시점에 상장폐지된 종목 list.
    valid_tickers : list of str
        본 시점 거래 가능한 종목 list.
    mcaps : pd.Series (n,)
        본 시점 시가총액 (전이 비례 계산용).

    Returns
    -------
    new_weights : pd.Series
        조정된 가중치. 폐지 종목 제거 + 비중 비례 재배분.

    Notes
    -----
    예시:
        BL 가중치 = {AAPL: 0.20, GE: 0.10, MSFT: 0.30, ...}
        GE 폐지 →
        새 가중치 = {AAPL: 0.20 + 0.10 × (mcap_AAPL / Σ_remaining_mcap),
                    MSFT: 0.30 + 0.10 × (mcap_MSFT / Σ_remaining_mcap), ...}

    엄격 가정:
        - 폐지 시점 가격 = 0 가정 (실제는 청산가 / 합병가 가능)
        - 본 baseline 은 보수적 가정 (Step 4 기본 진행)
    """
    new_weights = weights.copy()
    for d in delisted:
        if d not in new_weights.index:
            continue
        delisted_w = new_weights[d]
        if delisted_w == 0:
            new_weights = new_weights.drop(d)
            continue

        # 남은 종목 시총 합 (delisted 제외)
        remaining = [t for t in valid_tickers if t != d and t in new_weights.index]
        if not remaining:
            new_weights = new_weights.drop(d)
            continue

        remaining_mcaps = mcaps.reindex(remaining).fillna(0)
        total_mcap = remaining_mcaps.sum()
        if total_mcap <= 0:
            new_weights = new_weights.drop(d)
            continue

        # 비례 배분
        for t in remaining:
            new_weights[t] = new_weights.get(t, 0) + delisted_w * (remaining_mcaps[t] / total_mcap)

        new_weights = new_weights.drop(d)

    return new_weights


# =============================================================================
# Portfolio metrics — Sharpe / alpha / MDD / CumRet
# =============================================================================
def compute_portfolio_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    rf_returns: Optional[pd.Series] = None,
    periods_per_year: int = 12,
) -> dict:
    """Portfolio 핵심 메트릭 계산 (Sharpe, alpha, MDD, CumRet 등).

    Parameters
    ----------
    portfolio_returns : pd.Series
        월별 net portfolio return.
    benchmark_returns : pd.Series | None
        벤치마크 월별 return (alpha 계산용).
    rf_returns : pd.Series | None
        무위험 수익률 월별. None 이면 0 가정.
    periods_per_year : int, default 12
        연환산 계수 (월별 = 12).

    Returns
    -------
    dict with keys:
        - cum_return: 누적 수익률 (1+r 의 곱 - 1)
        - annualized_return: 연환산 평균
        - annualized_vol: 연환산 표준편차
        - sharpe: Sharpe ratio (excess return / vol)
        - alpha: 연환산 alpha (vs 벤치마크)
        - beta: 시장 베타 (벤치마크 대비)
        - max_drawdown: 최대 낙폭
        - calmar: 연환산 / |MDD|
    """
    if rf_returns is None:
        rf_aligned = pd.Series(0, index=portfolio_returns.index)
    else:
        rf_aligned = rf_returns.reindex(portfolio_returns.index).fillna(0)

    excess = portfolio_returns - rf_aligned

    # Cumulative return
    cum_return = float((1 + portfolio_returns).prod() - 1)

    # Annualized stats
    ann_return = float(portfolio_returns.mean() * periods_per_year)
    ann_vol = float(portfolio_returns.std() * np.sqrt(periods_per_year))
    sharpe = float(excess.mean() / portfolio_returns.std() * np.sqrt(periods_per_year)) if portfolio_returns.std() > 0 else np.nan

    # Max drawdown
    cum_curve = (1 + portfolio_returns).cumprod()
    rolling_max = cum_curve.cummax()
    drawdown = (cum_curve - rolling_max) / rolling_max
    mdd = float(drawdown.min())

    # Calmar
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else np.nan

    out = {
        'cum_return': cum_return,
        'annualized_return': ann_return,
        'annualized_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': mdd,
        'calmar': calmar,
        'n_periods': len(portfolio_returns),
    }

    # Alpha / beta (벤치마크 있을 때만)
    if benchmark_returns is not None:
        bench_aligned = benchmark_returns.reindex(portfolio_returns.index).dropna()
        port_aligned = portfolio_returns.reindex(bench_aligned.index)
        if len(bench_aligned) >= 12:
            bench_excess = bench_aligned - rf_aligned.reindex(bench_aligned.index).fillna(0)
            port_excess = port_aligned - rf_aligned.reindex(bench_aligned.index).fillna(0)
            # OLS: port_excess = α + β · bench_excess
            X = np.column_stack([np.ones(len(bench_excess)), bench_excess.values])
            coef, *_ = np.linalg.lstsq(X, port_excess.values, rcond=None)
            alpha_monthly, beta = coef
            out['alpha'] = float(alpha_monthly * periods_per_year)
            out['beta'] = float(beta)

    return out


# =============================================================================
# 누적 수익 곡선
# =============================================================================
def compute_cumulative_curve(
    portfolio_returns: pd.Series,
    initial_value: float = 1.0,
) -> pd.Series:
    """누적 수익 곡선 (시각화용)."""
    return initial_value * (1 + portfolio_returns).cumprod()


# =============================================================================
# Drawdown 시계열
# =============================================================================
def compute_drawdown_curve(portfolio_returns: pd.Series) -> pd.Series:
    """Drawdown 시계열 (peak-to-trough %)."""
    cum_curve = (1 + portfolio_returns).cumprod()
    rolling_max = cum_curve.cummax()
    return (cum_curve - rolling_max) / rolling_max
