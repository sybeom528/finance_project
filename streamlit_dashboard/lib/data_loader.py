"""
lib/data_loader.py - 데이터 로딩 + Streamlit 캐싱 (D-3)

모든 페이지가 공유하는 데이터 로더.
@st.cache_data 표준 적용 → Streamlit 세션 동안 1회만 실제 IO 발생.

경로:
  - 모듈 위치 기준 절대 경로 사용 (cwd 무관 동작)
  - 핵심 데이터: streamlit_dashboard/data/ (사본)
  - Sensitivity Test: final/results/ (원본 직접 참조, 156 config 중 필요 시)

참조: docs/plan/02_common.md 2절, docs/plan/01_setup.md 2.3절
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# === 경로 (모듈 위치 기준, cwd 무관) ===================================
DASHBOARD_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DASHBOARD_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
PROJECT_ROOT = DASHBOARD_DIR.parent
ORIGINAL_RESULTS_DIR = PROJECT_ROOT / "final" / "results"


# === 핵심 데이터 (대시보드 사본) =======================================

@st.cache_data
def load_monthly_panel() -> pd.DataFrame:
    """월별 패널 (date, ticker, rf, spy_ret, sector, log_mcap 등)."""
    return pd.read_csv(DATA_DIR / "monthly_panel.csv", parse_dates=["date"])


@st.cache_data
def load_daily_returns() -> pd.DataFrame:
    """일별 수익률 DataFrame (index=date, columns=ticker)."""
    with open(DATA_DIR / "daily_returns.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_ff5_monthly() -> pd.DataFrame:
    """Fama-French 5-factor 월별 (date, mkt_rf, smb, hml, rmw, cma, rf)."""
    return pd.read_csv(DATA_DIR / "ff5_monthly.csv", parse_dates=["date"])


@st.cache_data
def load_universe() -> pd.DataFrame:
    """Universe 정의 (ticker, gics_sector, 등)."""
    return pd.read_csv(DATA_DIR / "universe.csv")


@st.cache_data
def load_ticker_company_map() -> pd.DataFrame | None:
    """
    yfinance 회사명 매핑 (D-2). 파일 없으면 None.
    None 반환 시 호출 측에서 ticker 자체를 사용 (graceful degradation).
    """
    p = DATA_DIR / "ticker_company_map.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_sp500_membership() -> dict:
    """
    S&P500 시점별 편입 종목 (look-ahead 회피용 universe).

    Returns:
        dict[Timestamp, frozenset[str]] — 월말 시점 → 그 시점 편입 ticker set

    decisionlog/02_overview.md Q-C 결정 (사용자 지적 2026-05-10):
      EW/IVW 의 ret[t] 계산 시 universe = sp500_membership[t-1] (look-ahead 회피).
    """
    with open(DATA_DIR / "sp500_membership.pkl", "rb") as f:
        return pickle.load(f)


# === 펀드 결과 ========================================================

@st.cache_data
def load_fund_results(config_name: str = "mat_eq_eq_raw_pap") -> dict:
    """
    펀드 backtest 결과 (Top 1 = mat_eq_eq_raw_pap).

    pkl 구조 (예상):
      {"weights": pd.DataFrame, "returns": pd.Series, "dates": ..., ...}
    """
    with open(RESULTS_DIR / f"{config_name}.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_other_config_results(config_name: str) -> dict:
    """
    다른 155 config 결과 (Backtesting 영역 6 Sensitivity Test 시).
    원본 final/results/ 에서 직접 참조 (대시보드 사본에는 Top 1 만 존재).
    """
    with open(ORIGINAL_RESULTS_DIR / f"{config_name}.pkl", "rb") as f:
        return pickle.load(f)


# === Baseline 산출 (Overview 영역 3 비교 라인) ========================
# 모두 시점 t-1 sp500_membership 기준 (look-ahead bias 회피, decisionlog Q-C)


def _resolve_universe_at(t_prev: pd.Timestamp, sp500_membership: dict) -> list[str]:
    """
    t_prev 시점 (또는 가장 가까운 직전 시점) S&P500 편입 종목 list.
    sp500_membership 의 key 가 t_prev 와 정확히 일치하지 않으면 직전 가장 가까운 시점 사용.
    """
    sp_dates = sorted(sp500_membership.keys())
    prior = [d for d in sp_dates if d <= t_prev]
    if not prior:
        return []
    return list(sp500_membership[max(prior)])


@st.cache_data
def compute_equal_weight_returns(
    _monthly_panel: pd.DataFrame,
    _sp500_membership: dict,
    _fund_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    EW baseline — 시점 t-1 S&P500 universe 의 ret_1m 동일가중 평균 (옵션 E).

    학술 표준 (decisionlog Q-C + Q-D):
      1) 데이터: monthly_panel (daily_returns 결함 회피, fund.spy_ret 와 일관성 검증)
      2) Universe: sp500_membership[t-1] (look-ahead 회피)
      3) ret[t] = monthly_panel[date=t & ticker in universe].ret_1m.mean()
         (월별 단위라 daily drift 표현은 단순화 — Σ(w_i × r_i) 로 단순 평균)

    Args (인자명에 underscore = streamlit cache hashing 제외):
        _monthly_panel / _sp500_membership / _fund_dates: hash 불가 타입

    Returns:
        pd.Series (index=fund_dates, value=monthly EW return)
    """
    panel = _monthly_panel
    out: dict[pd.Timestamp, float] = {}
    fund_dates = pd.DatetimeIndex(_fund_dates)

    for i, t in enumerate(fund_dates):
        # rebalance 시점 = 직전 월말 (look-ahead 회피)
        if i > 0:
            t_prev = fund_dates[i - 1]
        else:
            t_prev = t - pd.offsets.MonthEnd(1)

        # t-1 시점 sp500 universe (또는 가장 가까운 직전)
        universe = _resolve_universe_at(t_prev, _sp500_membership)
        if not universe:
            continue
        universe_set = set(universe)

        # t 시점 active ticker 의 ret_1m 평균 (NaN 제외)
        rows = panel[(panel["date"] == t) & (panel["ticker"].isin(universe_set))]
        ret_1m = rows["ret_1m"].dropna()
        if len(ret_1m) == 0:
            continue

        out[t] = float(ret_1m.mean())

    return pd.Series(out, name="EW")


@st.cache_data
def compute_ivw_returns(
    _monthly_panel: pd.DataFrame,
    _sp500_membership: dict,
    _fund_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    IVW baseline (Naive Low-vol) — weight_i = (1/σ_i) / Σ(1/σ_j).
    Frazzini & Pedersen (2014) "Betting Against Beta".

    학술 표준 (decisionlog Q-B 정정 + Q-C + Q-D):
      1) 데이터: monthly_panel (vol_60d, ret_1m 사용)
      2) Universe: sp500_membership[t-1] (look-ahead 회피)
      3) σ_i = monthly_panel[date=t-1, ticker=i].vol_60d (t-1 시점 60일 변동성)
              → look-ahead 회피 (t 시점 vol 사용 X)
      4) weight_i = (1/σ_i) / Σ(1/σ_j)
      5) ret[t] = Σ(weight_i × ret_1m_i_at_t) / Σ(weight_i)  (t 시점 데이터 누락 ticker normalize)

    NOTE: 윈도우 60d 는 plan 정정 (원안 120d) — monthly_panel 부재로 정정.
          decisionlog Q-B 변경 이력 참조.
    """
    panel = _monthly_panel
    out: dict[pd.Timestamp, float] = {}
    fund_dates = pd.DatetimeIndex(_fund_dates)

    for i, t in enumerate(fund_dates):
        if i > 0:
            t_prev = fund_dates[i - 1]
        else:
            t_prev = t - pd.offsets.MonthEnd(1)

        universe = _resolve_universe_at(t_prev, _sp500_membership)
        if not universe:
            continue
        universe_set = set(universe)

        # t-1 시점 active ticker 의 vol_60d (look-ahead 회피)
        prev_rows = panel[(panel["date"] == t_prev) & (panel["ticker"].isin(universe_set))]
        prev_rows = prev_rows[(prev_rows["vol_60d"] > 0) & prev_rows["vol_60d"].notna()]
        if len(prev_rows) == 0:
            continue

        # weight = (1/σ) / Σ(1/σ)
        inv_vol = 1.0 / prev_rows["vol_60d"].values
        weights = pd.Series(inv_vol / inv_vol.sum(), index=prev_rows["ticker"].values)

        # t 시점 ret_1m (활성 ticker 만)
        curr_rows = panel[
            (panel["date"] == t) & (panel["ticker"].isin(weights.index))
        ][["ticker", "ret_1m"]].dropna()
        if len(curr_rows) == 0:
            continue

        # Σ(weight_i × ret_1m_i) / Σ(weight_i 의 valid ticker 부분 합)
        # → t 시점 데이터 누락 ticker 의 weight 만큼 누락 보정
        valid_weights = weights.loc[curr_rows["ticker"].values]
        weighted_ret = (curr_rows["ret_1m"].values * valid_weights.values).sum()
        weight_sum = valid_weights.sum()
        if weight_sum == 0:
            continue

        out[t] = float(weighted_ret / weight_sum)

    return pd.Series(out, name="IVW")


# === 기간 필터 (사이드바 토글 — FULL / TEST / HO) =====================

# TEST = 2010-01 ~ 2023-12 (168m), HO = 2024-01 ~ 2025-12 (24m)
TEST_END = pd.Timestamp("2023-12-31")
HO_START = pd.Timestamp("2024-01-01")


def filter_period(returns: pd.Series, period: str) -> pd.Series:
    """
    사이드바 기간 토글 (FULL / TEST / HO) 에 따라 returns 필터링.

    Args:
        returns: pd.Series with DatetimeIndex
        period: "FULL" / "TEST" / "HO"

    Returns:
        필터된 pd.Series (FULL 은 그대로 반환)
    """
    if period == "TEST":
        return returns[returns.index <= TEST_END]
    if period == "HO":
        return returns[returns.index >= HO_START]
    return returns  # "FULL"
