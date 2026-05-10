"""
lib/metric_calculators.py - 16개 메트릭 계산 함수

단일 책임 원칙: 1 함수 = 1 메트릭. 각 함수 NaN-safe (.dropna() 우선).
빈 입력 / 분모 0 / 음수 누적 시 np.nan 반환.

수익률 입력 단위:
  - returns: fraction (0.01 = 1%)
  - rf: fraction (월별 또는 연환산은 함수 내부에서 처리)

연환산 기본값 (periods_per_year):
  - 월별 데이터 = 12 (기본)
  - 일별 데이터 = 252

참조: docs/decisionlog/04_risk_metrics.md, docs/decisionlog/03_performance.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# === Pool-1 수익성 ===================================================

def calc_cagr(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    연환산 복리 수익률 (Compound Annual Growth Rate).

    공식: (Π(1 + r_i))^(1/n_years) - 1
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    n_years = len(r) / periods_per_year
    if n_years <= 0:
        return np.nan
    cumulative = (1 + r).prod()
    if cumulative <= 0:
        return np.nan
    return cumulative ** (1 / n_years) - 1


def calc_arithmetic_mean(returns: pd.Series, annualize: bool = True, periods_per_year: int = 12) -> float:
    """산술 평균 수익률 (단순 mean × periods_per_year)."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    mean = r.mean()
    return float(mean * periods_per_year if annualize else mean)


# === Pool-3 위험 (Sharpe / Sortino 분모로 사용되므로 먼저 정의) =====

def calc_volatility(returns: pd.Series, annualize: bool = True, periods_per_year: int = 12) -> float:
    """변동성 (표준편차). annualize=True 면 √periods_per_year 곱함."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    std = r.std()
    if pd.isna(std):
        return np.nan
    return float(std * np.sqrt(periods_per_year) if annualize else std)


def calc_mdd(returns: pd.Series) -> float:
    """
    Maximum Drawdown — 고점 대비 최대 손실 (음수, fraction).

    예: -0.5384 = -53.84% 최대 낙폭
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return float(dd.min())


def calc_downside_deviation(
    returns: pd.Series, mar: float = 0.0, annualize: bool = True, periods_per_year: int = 12
) -> float:
    """
    하방 표준편차 — Minimum Acceptable Return (MAR) 미달 분만 RMS.
    Sortino 의 분모로 사용.
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    diff = r - mar
    downside = diff[diff < 0]
    if len(downside) == 0:
        return 0.0  # 모든 수익률이 MAR 이상이면 하방 0
    dd = np.sqrt((downside ** 2).mean())
    return float(dd * np.sqrt(periods_per_year) if annualize else dd)


def calc_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical VaR — alpha 분위수의 손실 (음수가 일반적).

    예: alpha=0.05 → 최악 5% 손실 분위
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    return float(np.percentile(r, alpha * 100))


def calc_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Conditional VaR (Expected Shortfall) — VaR 이하 평균 손실.

    VaR 보다 더 보수적 (꼬리 위험 평균).
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    var_threshold = np.percentile(r, alpha * 100)
    tail = r[r <= var_threshold]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


# === Pool-2 위험조정 수익 ============================================

def calc_sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 12) -> float:
    """
    Sharpe Ratio = (R - Rf) / σ × √periods_per_year.

    rf 는 연환산 무위험 수익률 (예: 0.04 = 4%/year).
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    excess = r - rf / periods_per_year
    std = excess.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def calc_sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 12) -> float:
    """
    Sortino Ratio = (R - Rf) / σ_downside × √periods_per_year.

    Sharpe 와 다르게 하방 변동성만 페널티.
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    excess = r - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan  # 손실 없으면 정의 불가
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return np.nan
    return float((excess.mean() / dd) * np.sqrt(periods_per_year))


def calc_calmar(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Calmar Ratio = CAGR / |MDD|."""
    cagr = calc_cagr(returns, periods_per_year)
    mdd = calc_mdd(returns)
    if pd.isna(cagr) or pd.isna(mdd) or mdd == 0:
        return np.nan
    return float(cagr / abs(mdd))


def calc_ir(fund_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 12) -> float:
    """
    Information Ratio = mean(Fund - Bench) / std(Fund - Bench) × √periods_per_year.

    벤치마크 대비 액티브 운용의 효율성.
    """
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    if len(df) == 0:
        return np.nan
    active = df["f"] - df["b"]
    te = active.std()
    if pd.isna(te) or te == 0:
        return np.nan
    return float((active.mean() / te) * np.sqrt(periods_per_year))


# === Pool-5 시장 비교 ================================================

def calc_beta(fund_ret: pd.Series, mkt_ret: pd.Series) -> float:
    """
    Market Beta = Cov(Fund, Mkt) / Var(Mkt).

    < 1 = 시장보다 덜 민감, > 1 = 시장보다 더 민감.
    """
    df = pd.DataFrame({"f": fund_ret, "m": mkt_ret}).dropna()
    if len(df) < 2:
        return np.nan
    var_m = df["m"].var()
    if pd.isna(var_m) or var_m == 0:
        return np.nan
    cov = df["f"].cov(df["m"])
    return float(cov / var_m)


def calc_tracking_error(
    fund_ret: pd.Series, bench_ret: pd.Series, annualize: bool = True, periods_per_year: int = 12
) -> float:
    """추적오차 = std(Fund - Bench) × √periods_per_year."""
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    if len(df) == 0:
        return np.nan
    te = (df["f"] - df["b"]).std()
    if pd.isna(te):
        return np.nan
    return float(te * np.sqrt(periods_per_year) if annualize else te)


def calc_win_rate(returns: pd.Series) -> float:
    """양수 수익률 월 비율 (예: 0.65 = 65%)."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    return float((r > 0).mean())


def calc_up_capture(fund_ret: pd.Series, mkt_ret: pd.Series) -> float:
    """
    Up Capture = 시장 상승 월의 펀드 평균수익 / 시장 평균수익.

    100% 이상 = 시장보다 더 상승, 100% 미만 = 시장보다 덜 상승.
    """
    df = pd.DataFrame({"f": fund_ret, "m": mkt_ret}).dropna()
    up = df[df["m"] > 0]
    if len(up) == 0:
        return np.nan
    fund_mean = up["f"].mean()
    mkt_mean = up["m"].mean()
    if pd.isna(mkt_mean) or mkt_mean == 0:
        return np.nan
    return float(fund_mean / mkt_mean)


def calc_down_capture(fund_ret: pd.Series, mkt_ret: pd.Series) -> float:
    """
    Down Capture = 시장 하락 월의 펀드 평균수익 / 시장 평균수익.

    100% 미만 = 시장보다 덜 하락 (방어성 ↑).
    """
    df = pd.DataFrame({"f": fund_ret, "m": mkt_ret}).dropna()
    down = df[df["m"] < 0]
    if len(down) == 0:
        return np.nan
    fund_mean = down["f"].mean()
    mkt_mean = down["m"].mean()
    if pd.isna(mkt_mean) or mkt_mean == 0:
        return np.nan
    return float(fund_mean / mkt_mean)


# === Pool-4 운용 효율성 (집중도) =====================================

def calc_hhi(weights: pd.Series) -> float:
    """
    Herfindahl-Hirschman Index = Σw²

    weights 가 비중 (sum=1) 이라 가정. 0~1 범위.
    낮을수록 분산 (예: N개 동일가중 → 1/N).
    """
    w = pd.Series(weights).dropna()
    if len(w) == 0:
        return np.nan
    return float((w ** 2).sum())


def calc_effective_n(weights: pd.Series) -> float:
    """Effective N = 1 / HHI = 1 / Σw². 유효 종목 수 (분산 척도)."""
    hhi = calc_hhi(weights)
    if pd.isna(hhi) or hhi == 0:
        return np.nan
    return float(1 / hhi)
