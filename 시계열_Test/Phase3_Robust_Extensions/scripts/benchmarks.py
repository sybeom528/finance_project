"""Phase 2 — 벤치마크 portfolio 가중치 (SPY / 1/N / Mcap).

결정 6 (PLAN.md §2): SPY + 1/N + Mcap + BL 4종 비교.

벤치마크 종류
------------
1. **SPY**: S&P 500 ETF (시장 벤치마크)
2. **EqualWeight (1/N)**: 매월 universe 의 1/N 등가 (DeMiguel et al. 2009 강력 baseline)
3. **McapWeight**: 매월 universe 의 시총 가중 (S&P 500 인덱스 가중 방식과 동등)
4. **BL_trailing**: 서윤범 baseline (vol_21d 기반 P 행렬)
5. **BL_ml**: Phase 2 ensemble (ML 예측 기반 P 행렬) ⭐

본 모듈은 1, 2, 3 의 가중치 계산만 담당. 4, 5 는 black_litterman.py 사용.

핵심 함수
---------
- equal_weight_portfolio()  : 1/N 가중치
- mcap_weight_portfolio()   : 시총 가중치
- spy_returns()             : SPY 수익률 시계열 (벤치마크)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# 1/N 등가 (DeMiguel et al. 2009 baseline)
# =============================================================================
def equal_weight_portfolio(
    universe_tickers: list,
) -> pd.Series:
    """1/N 등가 portfolio 가중치.

    Parameters
    ----------
    universe_tickers : list of str
        해당 시점 universe 종목 list.

    Returns
    -------
    weights : pd.Series (n,)
        모든 종목에 1/n 균등 가중치.

    Notes
    -----
    DeMiguel, Garlappi, Uppal (2009) "Optimal Versus Naive Diversification"
    → 1/N 등가가 14가지 정교한 평균-분산 모델을 OOS 에서 모두 능가
    → 강력한 baseline. BL 이 이를 능가해야 의미 있음.
    """
    n = len(universe_tickers)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=universe_tickers, name='equal_weight')


# =============================================================================
# 시총 가중 (Mcap Weight)
# =============================================================================
def mcap_weight_portfolio(
    mcaps: pd.Series,
    universe_tickers: Optional[list] = None,
) -> pd.Series:
    """시가총액 가중 portfolio.

    Parameters
    ----------
    mcaps : pd.Series (n,)
        자산별 시가총액. NaN 자동 제외.
    universe_tickers : list of str | None
        universe 제한. None 이면 mcaps.index 전체 사용.

    Returns
    -------
    weights : pd.Series
        시총 비례 가중치. Σw = 1.

    Notes
    -----
    - S&P 500 인덱스 가중 방식과 동등 (Vanguard, BlackRock 표준)
    - SPY ETF 는 정확히 본 가중치를 따름 (단, S&P 500 전체 vs 본 50)
    - "대형주 = 안정주" 가정에서 합리적 baseline
    """
    if universe_tickers is not None:
        mcap_filtered = mcaps.reindex(universe_tickers).dropna()
    else:
        mcap_filtered = mcaps.dropna()

    if len(mcap_filtered) == 0 or mcap_filtered.sum() <= 0:
        return pd.Series(dtype=float)

    weights = mcap_filtered / mcap_filtered.sum()
    weights.name = 'mcap_weight'
    return weights


# =============================================================================
# SPY 수익률 (외부 시장 벤치마크)
# =============================================================================
def spy_returns(
    market_data: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    return_type: str = 'monthly',
) -> pd.Series:
    """SPY 수익률 시계열 (벤치마크).

    Parameters
    ----------
    market_data : pd.DataFrame
        market_data.csv (columns: SPY, VIX, TNX).
    rebalance_dates : pd.DatetimeIndex
        BL 리밸런싱 시점 (월말 등).
    return_type : 'monthly' | 'daily', default 'monthly'

    Returns
    -------
    spy_ret : pd.Series
        지정 시점별 SPY 수익률.

    Notes
    -----
    - return_type='monthly': pct_change() — 월별 단순 수익률
    - return_type='daily': log(P/P.shift(1)) — 일별 log return
    """
    if 'SPY' not in market_data.columns:
        raise ValueError('market_data 에 SPY 컬럼 없음')

    spy = market_data['SPY']

    if return_type == 'monthly':
        # 월말 가격 → 월별 단순 수익률
        spy_at_rebalance = spy.reindex(rebalance_dates, method='ffill')
        rets = spy_at_rebalance.pct_change()
    elif return_type == 'daily':
        rets = np.log(spy / spy.shift(1))
    else:
        raise ValueError(f'알 수 없는 return_type: {return_type}')

    rets.name = 'spy_return'
    return rets


# =============================================================================
# 통합 벤치마크 시뮬레이션
# =============================================================================
def simulate_benchmark(
    universe_history: pd.DataFrame,
    panel: pd.DataFrame,
    market_data: pd.DataFrame,
    benchmark: str,
    transaction_cost: float = 0.0,
) -> pd.Series:
    """매월 벤치마크 가중치 → portfolio return 시뮬레이션.

    Parameters
    ----------
    universe_history : pd.DataFrame
        universe_top50_history.csv (columns: oos_year, ticker, mcap_value, ...).
    panel : pd.DataFrame
        daily_panel.csv (date, ticker, ret_1m, mcap_value 등).
    market_data : pd.DataFrame
        market_data.csv.
    benchmark : str
        'spy' / 'equal' / 'mcap' 중 하나.
    transaction_cost : float, default 0.0

    Returns
    -------
    portfolio_returns : pd.Series
        월별 net return 시계열.

    Notes
    -----
    - 'spy': SPY 수익률 직접 사용 (가중치 산정 X)
    - 'equal' / 'mcap': 매월 universe 종목 가중치 → 월별 자산 수익률 가중평균
    """
    raise NotImplementedError(
        '본 함수는 04_BL_yearly_rebalance.ipynb 의 백테스트 루프 안에서 '
        'equal_weight_portfolio() / mcap_weight_portfolio() 를 직접 호출하여 구현. '
        '본 함수는 향후 통합 시뮬레이션 인터페이스 placeholder.'
    )
