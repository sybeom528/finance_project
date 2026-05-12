"""
lib/metric_calculators.py - 메트릭 계산 함수

★★★ Ground truth: final/bl_functions.py:compute_metrics + master_table.py 헬퍼 ★★★
모든 결과가 final 폴더의 펀드 backtest 결과와 정확히 일치하도록 구현.

핵심 일관성 원칙:
- ANN = sqrt(12) (월별 → 연환산)
- Sortino: excess.mean() * 12 / (excess[excess<0].std() * sqrt(12))
  → final 정의 그대로 (downside 의 일반 std, RMS 아님)
- Beta: cov(excess, mkt_excess) / var(mkt_excess) — excess 기준
- Alpha: (excess.mean() - β × mkt_excess.mean()) × 12
- CAGR: cum.iloc[-1] ** (12/n) - 1
- MDD: (cum - cum.cummax()) / cum.cummax() 의 min
- CVaR: ret[ret <= ret.quantile(α)].mean() (pandas quantile)
- Skewness: pd.Series.skew()
- Excess Kurtosis: pd.Series.kurt() (pandas default = fisher=True)

참조: final/bl_functions.py compute_metrics (line 446-548)
       final/master_table.py _{sharpe/sortino/cagr/mdd}_subperiod (line 127-164)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# === Annualization 상수 (final 과 동일) ===============================
ANN = np.sqrt(12)
PERIODS_PER_YEAR = 12


# ======================================================================
# Helper — rf 정규화 (Series / scalar 둘 다 처리)
# ======================================================================

def _normalize_rf(ret: pd.Series, rf) -> pd.Series:
    """
    rf 를 ret 의 인덱스에 맞춰 월별 시리즈로 변환.

    - Series: reindex + fillna(0) (final compute_metrics 패턴)
    - scalar: 연환산이라 가정 → 월별 (rf / 12) 로 broadcast
    """
    if isinstance(rf, pd.Series):
        return rf.reindex(ret.index).fillna(0)
    return pd.Series([rf / PERIODS_PER_YEAR] * len(ret), index=ret.index)


# ======================================================================
# Pool-1 수익성
# ======================================================================

def calc_cagr(ret: pd.Series, periods_per_year: int = 12) -> float:
    """
    CAGR — final compute_metrics 정의 정확 재현:
      cagr = cum.iloc[-1] ** (12/n_months) - 1.0
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    n_months = len(r)
    cum_last = float((1 + r).cumprod().iloc[-1])
    if n_months <= 0 or cum_last <= 0:
        return np.nan
    return float(cum_last ** (periods_per_year / n_months) - 1.0)


def calc_arithmetic_mean(ret: pd.Series, annualize: bool = True, periods_per_year: int = 12) -> float:
    """단순 평균 (월별 → 연환산 기본)."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    return float(r.mean() * periods_per_year) if annualize else float(r.mean())


def calc_cumulative_return(ret: pd.Series) -> float:
    """final compute_metrics 의 cum_ret = cum.iloc[-1] - 1."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).cumprod().iloc[-1] - 1)


# ======================================================================
# Pool-3 위험 (Sharpe / Sortino 분모로도 사용)
# ======================================================================

def calc_volatility(ret: pd.Series, annualize: bool = True, periods_per_year: int = 12) -> float:
    """final 의 vol = ret.std() * sqrt(12)."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    std = r.std()
    if pd.isna(std):
        return np.nan
    return float(std * np.sqrt(periods_per_year)) if annualize else float(std)


def calc_mdd(ret: pd.Series) -> float:
    """final 의 mdd = (cum - cum.cummax()) / cum.cummax() 의 min."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())


def calc_mdd_duration(ret: pd.Series) -> int:
    """
    final 의 mdd_duration: 연속 underwater (dd<0) 월 최대.
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return 0
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    underwater = (dd < 0).astype(int)
    max_dur, cur = 0, 0
    for v in underwater:
        cur = cur + 1 if v else 0
        if cur > max_dur:
            max_dur = cur
    return int(max_dur)


def calc_downside_deviation(
    ret: pd.Series, mar: float = 0.0, annualize: bool = True, periods_per_year: int = 12
) -> float:
    """
    final Sortino 의 분모와 동일 정의:
      downside = ret[ret < MAR]; downside.std() * sqrt(12)

    NOTE: pandas .std() 는 ddof=1 (sample std). final 도 동일.
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    downside = r[r < mar]
    if len(downside) <= 1:
        return np.nan
    dd_std = downside.std()
    if pd.isna(dd_std) or dd_std == 0:
        return 0.0
    return float(dd_std * np.sqrt(periods_per_year)) if annualize else float(dd_std)


def calc_var(ret: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR — pandas quantile (final 과 일관)."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    return float(r.quantile(alpha))


def calc_cvar(ret: pd.Series, alpha: float = 0.05) -> float:
    """
    final compute_metrics 의 cvar_5 정의 정확 재현:
      cutoff = ret.quantile(0.05); cvar = ret[ret <= cutoff].mean()
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    cutoff = r.quantile(alpha)
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


# ======================================================================
# Pool-2 위험조정 수익
# ======================================================================

def calc_sharpe(ret: pd.Series, rf=0.0, periods_per_year: int = 12) -> float:
    """
    final compute_metrics 정의:
      sharpe = excess.mean() / excess.std() * sqrt(12)
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    rf_a = _normalize_rf(r, rf)
    excess = r - rf_a
    if excess.std() == 0 or pd.isna(excess.std()):
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def calc_sortino(ret: pd.Series, rf=0.0, periods_per_year: int = 12) -> float:
    """
    final compute_metrics 정의 정확 재현:
      excess  = ret - rf_a
      downside = excess[excess < 0]
      down_std = downside.std() * sqrt(12)        # 일반 std (RMS 아님)
      sortino  = excess.mean() * 12 / down_std
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    rf_a = _normalize_rf(r, rf)
    excess = r - rf_a
    downside = excess[excess < 0]
    if len(downside) <= 1:
        return np.nan
    down_std = downside.std() * np.sqrt(periods_per_year)
    if down_std is None or down_std == 0 or pd.isna(down_std):
        return np.nan
    return float(excess.mean() * periods_per_year / down_std)


def calc_calmar(ret: pd.Series, periods_per_year: int = 12) -> float:
    """final 의 calmar = cagr / |mdd|."""
    cagr = calc_cagr(ret, periods_per_year)
    mdd = calc_mdd(ret)
    if pd.isna(cagr) or pd.isna(mdd) or mdd == 0:
        return np.nan
    return float(cagr / abs(mdd))


def calc_ir(fund_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 12) -> float:
    """
    Information Ratio = mean(Fund - Bench) / std(Fund - Bench) × sqrt(12).
    """
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    if len(df) == 0:
        return np.nan
    active = df["f"] - df["b"]
    te = active.std()
    if pd.isna(te) or te == 0:
        return np.nan
    return float(active.mean() / te * np.sqrt(periods_per_year))


# ======================================================================
# Pool-5 시장 비교 (Beta / Alpha — final 정확 재현)
# ======================================================================

def calc_beta(fund_ret: pd.Series, mkt_ret: pd.Series, rf=0.0, periods_per_year: int = 12) -> float:
    """
    final compute_metrics 정의 정확 재현:
      excess     = fund - rf
      mkt_excess = mkt - rf
      cov_mat    = np.cov(excess, mkt_excess)
      beta       = cov_mat[0,1] / cov_mat[1,1]   (var_m > 0 일 때)

    NOTE: final 은 "valid.sum() >= 12" 조건 (12개월 이상 정렬된 데이터 필요).
    """
    r = pd.Series(fund_ret)
    m = pd.Series(mkt_ret)
    rf_a = _normalize_rf(r, rf) if isinstance(rf, pd.Series) or rf != 0 else pd.Series(0.0, index=r.index)
    df = pd.DataFrame({"r": r, "m": m, "rf": rf_a}).dropna()
    if len(df) < 12:
        return np.nan
    excess = (df["r"] - df["rf"]).values
    mkt_excess = (df["m"] - df["rf"]).values
    cov_mat = np.cov(excess, mkt_excess)
    var_m = cov_mat[1, 1]
    if var_m <= 0 or pd.isna(var_m):
        return np.nan
    return float(cov_mat[0, 1] / var_m)


def calc_alpha(fund_ret: pd.Series, mkt_ret: pd.Series, rf=0.0, periods_per_year: int = 12) -> float:
    """
    final compute_metrics 정의 정확 재현:
      alpha = (excess.mean() - beta × mkt_excess.mean()) × 12
    """
    r = pd.Series(fund_ret)
    m = pd.Series(mkt_ret)
    rf_a = _normalize_rf(r, rf) if isinstance(rf, pd.Series) or rf != 0 else pd.Series(0.0, index=r.index)
    df = pd.DataFrame({"r": r, "m": m, "rf": rf_a}).dropna()
    if len(df) < 12:
        return np.nan
    excess = (df["r"] - df["rf"]).values
    mkt_excess = (df["m"] - df["rf"]).values
    cov_mat = np.cov(excess, mkt_excess)
    var_m = cov_mat[1, 1]
    if var_m <= 0 or pd.isna(var_m):
        return np.nan
    beta = cov_mat[0, 1] / var_m
    return float((excess.mean() - beta * mkt_excess.mean()) * periods_per_year)


def calc_tracking_error(
    fund_ret: pd.Series, bench_ret: pd.Series, annualize: bool = True, periods_per_year: int = 12
) -> float:
    """추적오차 = std(Fund - Bench) × sqrt(12)."""
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    if len(df) == 0:
        return np.nan
    te = (df["f"] - df["b"]).std()
    if pd.isna(te):
        return np.nan
    return float(te * np.sqrt(periods_per_year)) if annualize else float(te)


def calc_active_return(fund_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized Active Return = (mean_fund - mean_bench) × 12."""
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    if len(df) == 0:
        return np.nan
    return float((df["f"].mean() - df["b"].mean()) * periods_per_year)


def calc_win_rate(ret: pd.Series) -> float:
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    return float((r > 0).mean())


def calc_avg_win(ret: pd.Series) -> float:
    """final 의 avg_win = wins.mean() (월별)."""
    r = pd.Series(ret).dropna()
    wins = r[r > 0]
    if len(wins) == 0:
        return np.nan
    return float(wins.mean())


def calc_avg_loss(ret: pd.Series) -> float:
    """final 의 avg_loss = losses.mean() (월별, 음수)."""
    r = pd.Series(ret).dropna()
    losses = r[r < 0]
    if len(losses) == 0:
        return np.nan
    return float(losses.mean())


def calc_up_capture(fund_ret: pd.Series, mkt_ret: pd.Series) -> float:
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
    df = pd.DataFrame({"f": fund_ret, "m": mkt_ret}).dropna()
    down = df[df["m"] < 0]
    if len(down) == 0:
        return np.nan
    fund_mean = down["f"].mean()
    mkt_mean = down["m"].mean()
    if pd.isna(mkt_mean) or mkt_mean == 0:
        return np.nan
    return float(fund_mean / mkt_mean)


# ======================================================================
# Pool-9 분포 통계 (Performance 영역 8)
# ======================================================================

def calc_skewness(ret: pd.Series) -> float:
    """final 의 skewness = ret.skew() (pandas default)."""
    r = pd.Series(ret).dropna()
    if len(r) < 3:
        return np.nan
    return float(r.skew())


def calc_excess_kurtosis(ret: pd.Series) -> float:
    """
    Excess Kurtosis = pd.Series.kurt() (pandas default = Fisher's, 즉 -3 적용된 excess).
    정규분포 대비 첨도 — 양수 = fat tail.
    """
    r = pd.Series(ret).dropna()
    if len(r) < 4:
        return np.nan
    return float(r.kurt())


def calc_tail_ratio(ret: pd.Series, q_high: float = 0.95, q_low: float = 0.05) -> float:
    """
    Tail Ratio = q_high 분위 / |q_low 분위|.
    > 1 = 우측 꼬리 비대칭 (좋음).
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    high = r.quantile(q_high)
    low = abs(r.quantile(q_low))
    if low == 0 or pd.isna(low):
        return np.nan
    return float(high / low)


# ======================================================================
# Pool-4 운용 효율성 (집중도)
# ======================================================================

def calc_hhi(weights) -> float:
    """Herfindahl-Hirschman Index = Σw² (weights sum=1 가정)."""
    w = pd.Series(weights).dropna()
    if len(w) == 0:
        return np.nan
    return float((w ** 2).sum())


def calc_effective_n(weights) -> float:
    """Effective N = 1 / HHI."""
    hhi = calc_hhi(weights)
    if pd.isna(hhi) or hhi == 0:
        return np.nan
    return float(1 / hhi)


# ======================================================================
# Risk Metrics 페이지 전용 — final 정합 + 학술 표준 (Phase 2.1)
# ======================================================================

def calc_correlation(fund_ret: pd.Series, bench_ret: pd.Series) -> float:
    """Pearson correlation 펀드 vs 벤치마크 (학술 표준)."""
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    if len(df) < 2:
        return np.nan
    return float(df["f"].corr(df["b"]))


def calc_r_squared(fund_ret: pd.Series, bench_ret: pd.Series) -> float:
    """
    R² = corr(fund, bench)² (CAPM 회귀 결정계수와 동일).
    학술 표준 — Sharpe (1964) CAPM, Pearson² = OLS R² 동치.
    """
    corr = calc_correlation(fund_ret, bench_ret)
    if pd.isna(corr):
        return np.nan
    return float(corr ** 2)


def calc_top_n_drawdowns(returns: pd.Series, n: int = 3) -> list[dict]:
    """
    Top N drawdown 식별 + 각각 Duration + Recovery Time.

    Returns:
        list of dict (depth 큰 순):
        {
            "start_date": pd.Timestamp,         # peak 직후 시점
            "trough_date": pd.Timestamp,        # 최저점 시점
            "recovery_date": pd.Timestamp | None,  # 신고가 회복 (None = 진행 중)
            "depth": float,                      # 최대 손실 (음수)
            "duration": int,                     # peak → trough 개월 수
            "recovery_months": int | None,       # trough → 회복 개월 수
        }

    학술 표준 drawdown 분해 (Magdon-Ismail & Atiya 2004).
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return []

    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    # Drawdown periods 식별 (drawdown 시작 → 회복 또는 진행 중)
    in_dd = False
    dd_start = None
    dd_periods: list[dict] = []

    for i in range(len(dd)):
        is_dd = dd.iloc[i] < -1e-9  # 거의 0 무시
        if is_dd and not in_dd:
            in_dd = True
            dd_start = i
        elif not is_dd and in_dd:
            in_dd = False
            sub = dd.iloc[dd_start:i + 1]
            trough_offset = int(sub.values.argmin())
            trough_idx = dd_start + trough_offset
            dd_periods.append({
                "start_idx": max(dd_start - 1, 0),  # peak 시점
                "trough_idx": trough_idx,
                "recovery_idx": i,
                "depth": float(dd.iloc[trough_idx]),
                "duration": trough_idx - dd_start + 1,
                "recovery_months": i - trough_idx,
            })

    # 진행 중 DD (회복 안 됨)
    if in_dd and dd_start is not None:
        sub = dd.iloc[dd_start:]
        trough_offset = int(sub.values.argmin())
        trough_idx = dd_start + trough_offset
        dd_periods.append({
            "start_idx": max(dd_start - 1, 0),
            "trough_idx": trough_idx,
            "recovery_idx": None,
            "depth": float(dd.iloc[trough_idx]),
            "duration": trough_idx - dd_start + 1,
            "recovery_months": None,
        })

    # 가장 깊은 drawdown 순 정렬 (depth 가장 음수)
    dd_periods.sort(key=lambda x: x["depth"])
    top_n = dd_periods[:n]

    # 날짜 부착
    for p in top_n:
        p["start_date"] = r.index[p["start_idx"]]
        p["trough_date"] = r.index[p["trough_idx"]]
        p["recovery_date"] = r.index[p["recovery_idx"]] if p["recovery_idx"] is not None else None

    return top_n


def calc_avg_recovery_time(returns: pd.Series) -> float:
    """
    완료된 모든 drawdown 의 평균 recovery 개월 수.

    Recovery = trough 시점 → 신고가 회복 시점 (월 단위).
    진행 중 DD 는 제외.
    """
    top_dds = calc_top_n_drawdowns(returns, n=100)  # 모든 dd
    completed = [p["recovery_months"] for p in top_dds if p["recovery_months"] is not None]
    if not completed:
        return np.nan
    return float(np.mean(completed))


# ======================================================================
# Rolling 메트릭 (영역 6 — Volatility / Sortino / Beta / R² / TE)
# ======================================================================

def calc_rolling_volatility(returns: pd.Series, window: int = 36) -> pd.Series:
    """
    Rolling annualized volatility (window 월).
    학술 표준: rolling std × sqrt(12).
    """
    r = pd.Series(returns).dropna()
    return r.rolling(window).std() * np.sqrt(12)


def calc_rolling_sortino(
    returns: pd.Series, rf, window: int = 36, periods_per_year: int = 12
) -> pd.Series:
    """
    Rolling Sortino — final master_table _sortino_subperiod 정합 (sub<0 std 정의).
    """
    r = pd.Series(returns).dropna()
    rf_a = _normalize_rf(r, rf) if not isinstance(rf, (int, float)) else None
    out = []
    for i in range(window, len(r) + 1):
        sub = r.iloc[i - window:i]
        if isinstance(rf, (int, float)):
            rf_sub = pd.Series(rf / periods_per_year, index=sub.index)
        else:
            rf_sub = rf_a.reindex(sub.index).fillna(0)
        exc = sub - rf_sub
        down_series = sub[sub < 0]
        if len(down_series) <= 1:
            out.append((sub.index[-1], np.nan))
            continue
        down = down_series.std() * np.sqrt(periods_per_year)
        if down is None or down <= 0 or pd.isna(down):
            out.append((sub.index[-1], np.nan))
            continue
        out.append((sub.index[-1], float(exc.mean() * periods_per_year / down)))

    if not out:
        return pd.Series(dtype=float)
    dates, values = zip(*out)
    return pd.Series(values, index=pd.DatetimeIndex(dates))


def calc_rolling_beta(
    fund_ret: pd.Series, mkt_ret: pd.Series, rf, window: int = 36, periods_per_year: int = 12
) -> pd.Series:
    """
    Rolling Beta — final compute_metrics 정합 (excess 기준 cov/var).
    """
    df = pd.DataFrame({"f": fund_ret, "m": mkt_ret}).dropna()
    rf_a = _normalize_rf(df["f"], rf) if not isinstance(rf, (int, float)) else pd.Series(rf / periods_per_year, index=df.index)

    out = []
    for i in range(window, len(df) + 1):
        sub = df.iloc[i - window:i]
        sub_rf = rf_a.reindex(sub.index).fillna(0)
        excess = (sub["f"] - sub_rf).values
        mkt_excess = (sub["m"] - sub_rf).values
        cov_mat = np.cov(excess, mkt_excess)
        var_m = cov_mat[1, 1]
        if var_m <= 0 or pd.isna(var_m):
            out.append((sub.index[-1], np.nan))
            continue
        out.append((sub.index[-1], float(cov_mat[0, 1] / var_m)))

    if not out:
        return pd.Series(dtype=float)
    dates, values = zip(*out)
    return pd.Series(values, index=pd.DatetimeIndex(dates))


def calc_rolling_r_squared(
    fund_ret: pd.Series, mkt_ret: pd.Series, window: int = 36
) -> pd.Series:
    """Rolling R² = rolling correlation²."""
    df = pd.DataFrame({"f": fund_ret, "m": mkt_ret}).dropna()
    rolling_corr = df["f"].rolling(window).corr(df["m"])
    return rolling_corr ** 2


def calc_rolling_tracking_error(
    fund_ret: pd.Series, bench_ret: pd.Series, window: int = 36, periods_per_year: int = 12
) -> pd.Series:
    """Rolling Tracking Error — annualized."""
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    active = df["f"] - df["b"]
    return active.rolling(window).std() * np.sqrt(periods_per_year)


# ======================================================================
# 서브기간 헬퍼 (final master_table 정확 재현)
# ======================================================================

def calc_sharpe_subperiod(
    ret: pd.Series, rf, start: str, end: str, periods_per_year: int = 12
) -> float:
    """
    final master_table._sharpe_subperiod 정확 1:1 재현 (line 127-135).

    final 코드 그대로:
        sub = ret[(ret.index >= start) & (ret.index <= end)]
        rf_sub = rf.reindex(sub.index).fillna(0)
        exc = sub - rf_sub
        vol = sub.std() * sqrt(12)
        return exc.mean() * 12 / vol

    NaN 처리: final 은 dropna 안 하고 pandas default skipna=True 사용. 동일 적용.
    """
    r = pd.Series(ret)
    sub = r[(r.index >= start) & (r.index <= end)]
    if len(sub) < 6:
        return np.nan
    rf_sub = (
        pd.Series(rf).reindex(sub.index).fillna(0)
        if isinstance(rf, pd.Series)
        else pd.Series(rf / periods_per_year, index=sub.index)
    )
    exc = sub - rf_sub
    vol = sub.std() * np.sqrt(periods_per_year)
    if vol <= 0 or pd.isna(vol):
        return np.nan
    return float(exc.mean() * periods_per_year / vol)


def calc_sortino_subperiod(
    ret: pd.Series, rf, start: str, end: str, periods_per_year: int = 12
) -> float:
    """
    final master_table._sortino_subperiod 정확 재현 (line 138-146):
      down = sub[sub < 0].std() * sqrt(12)   # ret < 0 의 std (excess 아님!)
      return exc.mean() * 12 / down
    """
    r = pd.Series(ret)
    sub = r[(r.index >= start) & (r.index <= end)].dropna()
    if len(sub) < 6:
        return np.nan
    rf_sub = pd.Series(rf).reindex(sub.index).fillna(0) if isinstance(rf, pd.Series) else pd.Series(rf / periods_per_year, index=sub.index)
    exc = sub - rf_sub
    down_series = sub[sub < 0]
    if len(down_series) <= 1:
        return np.nan
    down = down_series.std() * np.sqrt(periods_per_year)
    if down is None or down <= 0 or pd.isna(down):
        return np.nan
    return float(exc.mean() * periods_per_year / down)


def calc_cagr_subperiod(ret: pd.Series, start: str, end: str, periods_per_year: int = 12) -> float:
    """
    final master_table._cagr_subperiod 정확 재현 (line 149-155) + NaN-safe.

    final 은 NaN 처리를 안 하지만 (1+NaN).cumprod() = NaN 전염 → CAGR 산출 실패.
    fund.ret 자체는 NaN 없어 final 결과와 동일. 벤치마크 (e.g., fund.spy_ret 의
    마지막 월 NaN) 산출 시에만 dropna 효과 — final 정합 유지.
    """
    r = pd.Series(ret)
    sub = r[(r.index >= start) & (r.index <= end)].dropna()
    if len(sub) < 6:
        return np.nan
    cum = (1 + sub).cumprod().iloc[-1]
    if cum <= 0 or pd.isna(cum):
        return np.nan
    return float(cum ** (periods_per_year / len(sub)) - 1)


def calc_mdd_subperiod(ret: pd.Series, start: str, end: str) -> float:
    """
    final master_table._mdd_subperiod 정확 재현 (line 158-164) + NaN-safe.

    cumprod NaN 전염 회피 위해 dropna 적용 (fund.ret 영향 X — NaN 없음).
    """
    r = pd.Series(ret)
    sub = r[(r.index >= start) & (r.index <= end)].dropna()
    if len(sub) < 6:
        return np.nan
    cum = (1 + sub).cumprod()
    return float((cum / cum.cummax() - 1).min())


# ======================================================================
# 통합 메트릭 dict (final compute_metrics 정확 재현)
# ======================================================================

def compute_all_metrics(
    ret: pd.Series,
    rf: pd.Series,
    label: str = "",
    mkt_ret: pd.Series | None = None,
) -> dict:
    """
    final/bl_functions.py:compute_metrics 와 동일한 인터페이스 + 결과.

    rf 필수 (월별 시리즈, fund_results 기간과 일치해야 함).
    mkt_ret None 이면 Beta/Alpha NaN.

    Returns: dict (label, sharpe, sortino, cagr, vol, mdd, calmar, cum_ret,
                   skewness, win_rate, avg_win, avg_loss, cvar_5, mdd_duration,
                   beta, alpha)

    NOTE: final 은 round() 적용. 우리는 raw float 반환 (호출 측에서 포맷팅).
    """
    rf_a = pd.Series(rf).reindex(ret.index).fillna(0)
    excess = ret - rf_a

    return {
        "label": label,
        "sharpe": calc_sharpe(ret, rf_a),
        "sortino": calc_sortino(ret, rf_a),
        "cagr": calc_cagr(ret),
        "vol": calc_volatility(ret),
        "mdd": calc_mdd(ret),
        "calmar": calc_calmar(ret),
        "cum_ret": calc_cumulative_return(ret),
        "skewness": calc_skewness(ret),
        "win_rate": calc_win_rate(ret),
        "avg_win": calc_avg_win(ret),
        "avg_loss": calc_avg_loss(ret),
        "cvar_5": calc_cvar(ret, alpha=0.05),
        "mdd_duration": calc_mdd_duration(ret),
        "beta": calc_beta(ret, mkt_ret, rf_a) if mkt_ret is not None else np.nan,
        "alpha": calc_alpha(ret, mkt_ret, rf_a) if mkt_ret is not None else np.nan,
    }


# ======================================================================
# Holdings 페이지 전용 — final 정합 + 학술 표준 (Phase 2.2)
# ======================================================================
# decisionlog 05_holdings.md Hold-Q1:
#   - final 정합 메트릭: comp DataFrame (n_stocks, eff_n, top1_weight,
#                       top10_share, turnover) — final/bl_functions 산출
#   - 학술 표준 메트릭: Single HHI / Sector HHI / Top 5 / Simple Contribution
#                     (Hirschman 1945, Brinson et al. 1986)


def calc_top_n_share(weights_row: pd.Series, n: int) -> float:
    """
    Top N 종목 weight 합계 (학술 표준).

    Args:
        weights_row: pd.Series (index=ticker, value=weight, 단일 시점)
        n: int (Top N 개수)

    Returns:
        float (sum of top N weights)
    """
    w = weights_row.dropna()
    w = w[w > 0]
    if len(w) == 0:
        return np.nan
    return float(w.nlargest(n).sum())


def calc_sector_weights(weights_row: pd.Series, ticker_to_sector: dict) -> pd.Series:
    """
    종목 weight → 섹터 weight 합계 (학술 표준).

    Args:
        weights_row: pd.Series (index=ticker, value=weight)
        ticker_to_sector: dict[ticker, sector] (universe 기반)

    Returns:
        pd.Series (index=sector, value=summed weight)
    """
    w = weights_row.dropna()
    w = w[w > 0]
    if len(w) == 0:
        return pd.Series(dtype=float)
    sector = w.index.to_series().map(ticker_to_sector).fillna("Unknown")
    return w.groupby(sector).sum()


def calc_sector_hhi(weights_row: pd.Series, ticker_to_sector: dict) -> float:
    """
    Sector HHI = Σ(섹터 weight)² (Hirschman 1945, sector level).
    """
    sector_w = calc_sector_weights(weights_row, ticker_to_sector)
    if len(sector_w) == 0:
        return np.nan
    return float((sector_w ** 2).sum())


def calc_holdings_kpi_snapshot(
    weights_row: pd.Series,
    ticker_to_sector: dict,
) -> dict:
    """
    단일 시점 (Latest snapshot) Holdings KPI 6개 — Hold-Q1 final 정합 + 학술 표준.

    n_stocks 정의 (final/_extend_bl_to_2025.py:251 정합):
      = active universe 내 모든 ticker (NaN 제외, weight=0 종목 포함)
      → 최적화 결과 weight=0 으로 산출된 ticker 도 active universe 의 일원

    Args:
        weights_row: pd.Series (index=ticker, value=weight, 단일 시점)
        ticker_to_sector: dict[ticker, sector]

    Returns:
        dict {n_stocks, eff_n, single_hhi, sector_hhi, top1, top5, top10}
    """
    w_active = weights_row.dropna()       # active universe (NaN 제외)
    w_pos = w_active[w_active > 0]        # 실제 보유 (weight > 0)

    return {
        "n_stocks": int(len(w_active)),   # final 정합 (valid_tix len)
        "eff_n": calc_effective_n(w_pos),
        "single_hhi": calc_hhi(w_pos),
        "sector_hhi": calc_sector_hhi(weights_row, ticker_to_sector),
        "top1": calc_top_n_share(weights_row, 1),
        "top5": calc_top_n_share(weights_row, 5),
        "top10": calc_top_n_share(weights_row, 10),
    }


def calc_holdings_kpi_period_avg(
    weights: pd.DataFrame,
    ticker_to_sector: dict,
    comp: pd.DataFrame | None = None,
) -> dict:
    """
    기간 평균 Holdings KPI — 사이드바 토글 (FULL/TEST/HO) 의 월별 평균.

    final 정합 우선:
      n_stocks_avg / eff_n_avg / turnover_avg = comp 직접 사용 (final master_table 정합)

    학술 산출:
      single_hhi_avg / sector_hhi_avg / top1_avg / top5_avg / top10_avg
      = 월별 weight → snapshot 산출 → 평균

    Args:
        weights: pd.DataFrame (index=date, columns=ticker)
        ticker_to_sector: dict
        comp: pd.DataFrame (final 산출, optional)

    Returns:
        dict (위 메트릭 평균)
    """
    monthly = []
    for t in weights.index:
        snap = calc_holdings_kpi_snapshot(weights.loc[t], ticker_to_sector)
        monthly.append(snap)

    df = pd.DataFrame(monthly, index=weights.index)

    # final 정합 (comp 우선) + 학술 산출 평균 통합
    out: dict = {}
    if comp is not None and len(comp) > 0:
        comp_filtered = comp.loc[comp.index.intersection(weights.index)]
        if "n_stocks" in comp_filtered.columns:
            out["n_stocks_avg"] = float(comp_filtered["n_stocks"].mean())
        if "eff_n" in comp_filtered.columns:
            out["eff_n_avg"] = float(comp_filtered["eff_n"].mean())
        if "turnover" in comp_filtered.columns:
            out["turnover_avg"] = float(comp_filtered["turnover"].mean())
    else:
        out["n_stocks_avg"] = float(df["n_stocks"].mean())
        out["eff_n_avg"] = float(df["eff_n"].mean())
        out["turnover_avg"] = np.nan  # comp 없으면 산출 불가

    # 학술 메트릭 (단순 평균)
    out["single_hhi_avg"] = float(df["single_hhi"].mean())
    out["sector_hhi_avg"] = float(df["sector_hhi"].mean())
    out["top1_avg"] = float(df["top1"].mean())
    out["top5_avg"] = float(df["top5"].mean())
    out["top10_avg"] = float(df["top10"].mean())

    return out


def calc_carino_smoothed_contribution(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.Series:
    """
    Carino (1999) Logarithmic Smoothing — 종목별 누적 기여도.
    Σ smoothed contribution = portfolio 누적 수익률 (정확 일치).

    학술 출처:
      - Carino, D. (1999) "Combining attribution effects over time"
        Journal of Performance Measurement.
      - Brinson Simple Attribution 의 multi-period linking 한계 해결 표준 방법.

    수식:
      1. per-period contribution: C_{i,t} = w_{i,t} × R_{i,t}
      2. portfolio per-period return: R_t = Σ_i C_{i,t}
      3. Carino factor: k_t = ln(1+R_t) / R_t   (R_t≠0; 0 시 lim → 1)
      4. 정규화: K = ln(1+R_cum) / R_cum, R_cum = ∏(1+R_t) - 1
      5. smoothed: C_i = Σ_t (k_t / K) × C_{i,t}

    검증: Σ smoothed C_i = R_cum (Brinson Σ 의 누적 환산)

    Returns:
        pd.Series (index=ticker, value=smoothed cumulative contribution, 내림차순)
    """
    weights_p = weights[(weights.index >= period_start) & (weights.index <= period_end)]
    if len(weights_p) == 0:
        return pd.Series(dtype=float)

    panel_p = panel[(panel["date"] >= period_start) & (panel["date"] <= period_end)]
    panel_pivot = panel_p.pivot_table(
        index="date", columns="ticker", values="ret_1m", aggfunc="first"
    )

    common_dates = weights_p.index.intersection(panel_pivot.index)
    if len(common_dates) == 0:
        return pd.Series(dtype=float)

    weights_a = weights_p.loc[common_dates]
    rets_a = panel_pivot.loc[common_dates]

    common_tickers = weights_a.columns.intersection(rets_a.columns)
    weights_a = weights_a[common_tickers].fillna(0)
    rets_a = rets_a[common_tickers].fillna(0)

    # per-period × per-ticker contribution
    contrib_t = weights_a * rets_a   # DataFrame (date × ticker)

    # portfolio per-period return (산출된 contribution 합 = 펀드 추정 R_t)
    R_t = contrib_t.sum(axis=1)

    # Carino factor per period — k_t = ln(1+R_t)/R_t, R_t=0 시 lim = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        k_t = np.where(R_t.abs() > 1e-12, np.log(1 + R_t) / R_t, 1.0)
    k_t = pd.Series(k_t, index=R_t.index)

    # 정규화 상수 — K = ln(1+R_cum) / R_cum
    R_cum = float((1 + R_t).prod() - 1)
    if abs(R_cum) < 1e-12:
        return pd.Series(0.0, index=common_tickers)
    K = float(np.log(1 + R_cum) / R_cum)

    # smoothed C_i = Σ_t (k_t / K) × C_{i,t}
    smoothed = contrib_t.mul(k_t / K, axis=0).sum(axis=0)

    return smoothed.sort_values(ascending=False)


def calc_simple_contribution(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.Series:
    """
    Simple Contribution = Σ_t (w_t × R_t) per ticker. Brinson et al. (1986).

    학술 출처:
      - Brinson, G. P., Hood, L. R., & Beebower, G. L. (1986).
        "Determinants of Portfolio Performance." Financial Analysts Journal.
      - Simple = w × R (allocation/selection 분해 안 함, 청중 친화)

    검증: Σ(per-ticker contribution) ≈ 펀드 누적 수익률 (선형 근사 한계 ε)

    Args:
        weights: pd.DataFrame (index=date, columns=ticker)
        panel: pd.DataFrame (date, ticker, ret_1m)
        period_start, period_end: 기간 boundary

    Returns:
        pd.Series (index=ticker, value=cumulative contribution, 정렬 X)
    """
    weights_p = weights[(weights.index >= period_start) & (weights.index <= period_end)]
    if len(weights_p) == 0:
        return pd.Series(dtype=float)

    panel_p = panel[(panel["date"] >= period_start) & (panel["date"] <= period_end)]
    panel_pivot = panel_p.pivot_table(
        index="date", columns="ticker", values="ret_1m", aggfunc="first"
    )

    # weights[t] 와 ret[t] 매칭 (forward alignment: weight at t-1 보유 → ret at t)
    common_dates = weights_p.index.intersection(panel_pivot.index)
    if len(common_dates) == 0:
        return pd.Series(dtype=float)

    weights_aligned = weights_p.loc[common_dates]
    rets_aligned = panel_pivot.loc[common_dates]

    # 공통 ticker
    common_tickers = weights_aligned.columns.intersection(rets_aligned.columns)
    weights_a = weights_aligned[common_tickers].fillna(0)
    rets_a = rets_aligned[common_tickers].fillna(0)

    # 종목별 누적 contribution = Σ_t (w_t × r_t)
    contribution = (weights_a * rets_a).sum(axis=0)

    return contribution.sort_values(ascending=False)


def calc_avg_turnover(comp: pd.DataFrame, start: str, end: str) -> float:
    """
    평균 회전율 (final 정합) — comp.turnover.mean() 기간 필터.
    final/master_table.py:227 정확 재현 (turnover_avg).

    NOTE: comp.turnover 첫 행 (2010-01) = 0 (rebalancing 시작점) 포함.
    """
    if "turnover" not in comp.columns:
        return np.nan
    sub = comp.loc[(comp.index >= start) & (comp.index <= end), "turnover"]
    if len(sub) == 0:
        return np.nan
    return float(sub.mean())


def get_market_cap_from_panel(
    panel: pd.DataFrame, date: pd.Timestamp, tickers: list[str]
) -> pd.Series:
    """
    monthly_panel.log_mcap → Market Cap ($) 환원. = exp(log_mcap).
    decisionlog D-1 정합 — final 데이터 그대로 사용.
    """
    rows = panel[(panel["date"] == date) & (panel["ticker"].isin(tickers))]
    if len(rows) == 0:
        return pd.Series(dtype=float)
    mcap = np.exp(rows.set_index("ticker")["log_mcap"])
    return mcap.reindex(tickers)


def get_12m_return_from_panel(
    panel: pd.DataFrame, end_date: pd.Timestamp, tickers: list[str]
) -> pd.Series:
    """
    종목별 최근 12m 누적 수익률 = ∏(1 + ret_1m) - 1 (12개월 rolling).
    결측 ticker / 결측 월은 NaN 으로 반환.
    """
    start = end_date - pd.DateOffset(months=12)
    sub = panel[
        (panel["date"] > start)
        & (panel["date"] <= end_date)
        & (panel["ticker"].isin(tickers))
    ]
    if len(sub) == 0:
        return pd.Series([np.nan] * len(tickers), index=tickers)

    pivoted = sub.pivot_table(
        index="date", columns="ticker", values="ret_1m", aggfunc="first"
    )
    cum_12m = (1 + pivoted).prod() - 1
    return cum_12m.reindex(tickers)


def calc_delta_weight(
    weights: pd.DataFrame, current_date: pd.Timestamp, tickers: list[str]
) -> pd.Series:
    """
    ΔWeight = w[current] - w[prev_month]. 결측 ticker = NaN.
    """
    if current_date not in weights.index:
        return pd.Series([np.nan] * len(tickers), index=tickers)

    pos = weights.index.get_loc(current_date)
    if pos == 0:
        # 첫 달은 ΔWeight 없음 → 모두 NaN
        return pd.Series([np.nan] * len(tickers), index=tickers)

    prev_date = weights.index[pos - 1]
    curr = weights.loc[current_date].reindex(tickers).fillna(0)
    prev = weights.loc[prev_date].reindex(tickers).fillna(0)
    return curr - prev


def calc_holding_period(
    weights: pd.DataFrame, current_date: pd.Timestamp, tickers: list[str]
) -> pd.Series:
    """
    최근 연속 보유 기간 (개월).
    current_date 부터 거꾸로 가며 weight > 0 인 연속 시점 카운트.
    중간에 weight = 0 또는 NaN 인 시점이 나오면 중단.

    Args:
        weights: 월별 weight DataFrame (index=date, columns=ticker)
        current_date: 기준 시점 (이 시점 포함하여 거꾸로 카운트)
        tickers: 대상 ticker 리스트

    Returns:
        pd.Series (index=tickers, value=연속 보유 개월 수)
        - current_date 에 보유 안한 ticker = 0
        - 첫 진입 ~ current_date 까지 연속 보유 = N (N = 진입 후 경과 개월)
    """
    if current_date not in weights.index:
        return pd.Series([0] * len(tickers), index=tickers, dtype=int)

    pos = weights.index.get_loc(current_date)
    dates_reversed = weights.index[: pos + 1][::-1]  # current_date 부터 거꾸로

    result = {}
    for tk in tickers:
        if tk not in weights.columns:
            result[tk] = 0
            continue
        count = 0
        for t in dates_reversed:
            w = weights.loc[t, tk]
            if pd.notna(w) and w > 0:
                count += 1
            else:
                break  # 끊김 → 카운트 종료
        result[tk] = count
    return pd.Series(result, dtype=int)


# ======================================================================
# Sector Watch 페이지 전용 (Phase 2.3)
# ======================================================================
# decisionlog 06_sector_watch.md Sector-Q1:
#   - SPY sector weight 산출: panel.log_mcap × sp500_membership[t-1] (look-ahead 회피)
#   - Sector Tilt = Fund_w − SPY_w (Active Management)
#   - Active Bets = (|Tilt| > 1%).sum()
#   - HO 정당화 narrative: Markowitz (1952) + Fama-French (1992)


def compute_spy_sector_weights(
    panel: pd.DataFrame,
    sp500_membership: dict,
    date: pd.Timestamp,
    ticker_to_sector: dict | None = None,
) -> pd.Series:
    """
    SPY sector weight (mcap 가중) — date 시점 sp500_membership 의 mcap 가중 합.

    학술 산출:
      1. members = sp500_membership[date 직전] (look-ahead 회피)
      2. mcap_i = exp(log_mcap_i)
      3. sector_weight_s = Σ_{i ∈ s} mcap_i / Σ_all mcap_i

    Args:
        panel: monthly_panel (date, ticker, log_mcap, gics_sector)
        sp500_membership: dict[Timestamp, frozenset[str]]
        date: 산출 시점
        ticker_to_sector: panel.gics_sector 외부 lookup (선택)

    Returns:
        pd.Series (index=sector, value=weight, sum=1)
    """
    sp_dates = sorted(sp500_membership.keys())
    prior = [d for d in sp_dates if d <= date - pd.offsets.MonthEnd(1)]
    if not prior:
        prior = [d for d in sp_dates if d <= date]
    if not prior:
        return pd.Series(dtype=float)

    members = list(sp500_membership[max(prior)])
    rows = panel[(panel["date"] == date) & (panel["ticker"].isin(members))].copy()
    rows = rows[rows["log_mcap"].notna()]
    if len(rows) == 0:
        return pd.Series(dtype=float)

    rows["_mcap"] = np.exp(rows["log_mcap"])
    if ticker_to_sector is not None:
        rows["_sector"] = rows["ticker"].map(ticker_to_sector).fillna("Unknown")
        sector_col = "_sector"
    else:
        sector_col = "gics_sector"

    total = rows["_mcap"].sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return rows.groupby(sector_col)["_mcap"].sum() / total


def calc_sector_weights_fund(
    weights_row: pd.Series, ticker_to_sector: dict
) -> pd.Series:
    """단일 시점 Fund sector weight (Holdings calc_sector_weights 와 동일)."""
    return calc_sector_weights(weights_row, ticker_to_sector)


def calc_sector_tilt(
    fund_sw: pd.Series, spy_sw: pd.Series
) -> pd.Series:
    """
    Sector Tilt = Fund_w − SPY_w (Active Management 표준).
    두 series 의 합집합 sector — 어느 쪽에만 있어도 0 보강 후 차이 산출.
    """
    all_sectors = sorted(set(fund_sw.index) | set(spy_sw.index))
    tilt = pd.Series(
        {s: float(fund_sw.get(s, 0)) - float(spy_sw.get(s, 0)) for s in all_sectors}
    )
    return tilt.sort_values(ascending=False)


def calc_active_bets(tilt: pd.Series, threshold: float = 0.01) -> int:
    """Active Bets = |Tilt| > threshold (default 1%) 의 sector 개수."""
    return int((tilt.abs() > threshold).sum())


def calc_sector_decomposition(
    weights_row: pd.Series,
    panel: pd.DataFrame,
    date: pd.Timestamp,
    ticker_to_sector: dict,
    spy_sw: pd.Series,
    fund_ret: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Sector level decomposition 표 — 9 컬럼 (영역 5 plan 정합).

    컬럼: sector, weight, tilt, return_12m, n_holdings, vol, beta, sharpe, contribution

    sector level 메트릭 산출 (가중 평균):
      - return_12m: Σ_i (w_i × ret_12m_i) / sector_weight (sector-conditional)
      - vol / beta: ticker level 의 가중 평균
      - sharpe: (sector return - rf) / sector vol
      - contribution: sector weight × sector return (단순)
    """
    w = weights_row.dropna()
    w = w[w > 0]
    if len(w) == 0:
        return pd.DataFrame()

    fund_sw = calc_sector_weights(weights_row, ticker_to_sector)

    # 종목별 12m return + vol / beta (panel 기반)
    ret_12m = get_12m_return_from_panel(panel, date, w.index.tolist())
    panel_d = panel[panel["date"] == date].set_index("ticker")
    vol_d = panel_d["vol_252d"].reindex(w.index)
    beta_d = panel_d["beta_252d"].reindex(w.index)

    # sector level groupby
    sectors = w.index.to_series().map(ticker_to_sector).fillna("Unknown")
    rows = []
    for sec, group in pd.DataFrame({
        "ticker": w.index, "weight": w.values,
        "ret_12m": ret_12m.values, "vol": vol_d.values, "beta": beta_d.values,
        "sector": sectors.values,
    }).groupby("sector"):
        sw = float(group["weight"].sum())
        if sw <= 0:
            continue
        # sector-conditional 가중 평균
        wts_norm = group["weight"] / sw
        sec_ret_12m = float((wts_norm * group["ret_12m"]).sum(skipna=True))
        sec_vol = float((wts_norm * group["vol"]).sum(skipna=True))
        sec_beta = float((wts_norm * group["beta"]).sum(skipna=True))
        # Sharpe (annualized) = sector return / sector vol (rf 무시 — 단순 근사)
        sec_sharpe = sec_ret_12m / sec_vol if sec_vol > 0 else np.nan
        # contribution = sector_weight × sector_return
        sec_contribution = sw * sec_ret_12m
        rows.append({
            "Sector": sec,
            "Weight": sw,
            "Tilt": sw - float(spy_sw.get(sec, 0)),
            "Return_12m": sec_ret_12m,
            "N_Holdings": int(len(group)),
            "Volatility": sec_vol,
            "Beta": sec_beta,
            "Sharpe": sec_sharpe,
            "Contribution": sec_contribution,
        })

    df = pd.DataFrame(rows).sort_values("Weight", ascending=False).reset_index(drop=True)
    return df


# ======================================================================
# Backtesting 페이지 전용 (Phase 3.2)
# ======================================================================
# decisionlog 08_backtesting.md BT-Q1:
#   - Regime 12 메트릭 (영역 4)
#   - 4 Sub-events (영역 5)
#   - 156 config Sensitivity (영역 6)
#   - 모든 메트릭 final/master_table 정합 또는 학술 표준 (출처 명시)


def calc_test_ho_gap(ret: pd.Series, rf: pd.Series) -> float:
    """
    TEST/HO Gap = TEST Sortino - HO Sortino.

    학술 의미:
      - Gap 클수록 (양수, > 0): 학습편향 (overfitting) ↑ — TEST 에서 잘하지만 HO 에서 못함
      - Gap 작을수록 (0 에 가까울수록): TEST/HO 일관 = robust
      - Gap 음수 (HO > TEST): 드문 케이스 — HO 가 TEST 보다 좋음

    검증 표준 (학술 출처):
      - Bailey & López de Prado (2014) "The Probability of Backtest Overfitting"
        — TEST/OOS gap 으로 학습편향 정량화
    """
    s_test = calc_sortino_subperiod(ret, rf, "2010-01-01", "2023-12-31")
    s_ho = calc_sortino_subperiod(ret, rf, "2024-01-01", "2025-12-31")
    if pd.isna(s_test) or pd.isna(s_ho):
        return np.nan
    return float(s_test - s_ho)


def calc_regime_consistency_score(ret: pd.Series, rf: pd.Series) -> dict:
    """
    Regime 일관성 = R1/R2/R3/HO Sortino mean / std.
    높을수록 모든 시기에서 일관된 성과 (안정).
    """
    regimes = {
        "R1": ("2010-01-01", "2012-06-30"),
        "R2": ("2012-07-01", "2019-12-31"),
        "R3": ("2020-01-01", "2023-12-31"),
        "HO": ("2024-01-01", "2025-12-31"),
    }
    sortinos = {label: calc_sortino_subperiod(ret, rf, s, e) for label, (s, e) in regimes.items()}
    valid = [v for v in sortinos.values() if not pd.isna(v)]
    if len(valid) < 2:
        return {"sortinos": sortinos, "mean": np.nan, "std": np.nan, "score": np.nan}
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1))
    score = mean / std if std > 0 else np.nan
    return {"sortinos": sortinos, "mean": mean, "std": std, "score": score}


# 영역 5 — Sub-events (4 위기) 정의 (변경 용이성 위해 외부 dict)
SUB_EVENTS: list[dict] = [
    {"name": "2018 Q4 Sell-off", "start": "2018-09-30", "end": "2018-12-31"},
    {"name": "COVID-19 Crash",    "start": "2020-02-29", "end": "2020-03-31"},
    {"name": "2022 Inflation Bear", "start": "2022-01-31", "end": "2022-10-31"},
    {"name": "2024-12 Sector Rotation", "start": "2024-11-30", "end": "2024-12-31"},
]


def calc_sub_event_metrics(ret: pd.Series, spy: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """
    4 Sub-events 별 Fund / SPY / Active Return / MDD / Recovery Time 산출.

    각 위기 기간 동안:
      - Fund 누적 수익률 / SPY 누적 수익률 / Active = Fund - SPY
      - Fund MDD 기간 내
      - Recovery Time = trough → 신고가 회복 (위기 종료 후 N 개월)
    """
    rows = []
    ret_clean = ret.dropna()
    full_cum = (1 + ret_clean).cumprod()  # 시작 시점부터 누적

    for ev in SUB_EVENTS:
        s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        sub_ret = ret[(ret.index >= s) & (ret.index <= e)].dropna()
        sub_spy = spy[(spy.index >= s) & (spy.index <= e)].dropna()
        if len(sub_ret) == 0:
            continue
        fund_cum = float((1 + sub_ret).prod() - 1)
        spy_cum = float((1 + sub_spy).prod() - 1) if len(sub_spy) > 0 else np.nan
        active = fund_cum - spy_cum if not pd.isna(spy_cum) else np.nan
        # MDD (위기 기간만)
        cum_in = (1 + sub_ret).cumprod()
        mdd = float((cum_in / cum_in.cummax() - 1).min())

        # Recovery — 위기 시작 직전 peak (전체 누적) → 위기 종료 후 신고가 회복
        before = full_cum[full_cum.index < s]
        if len(before) == 0:
            recovery = None
        else:
            peak_pre = float(before.iloc[-1])
            after_e = full_cum[full_cum.index > e]
            recovered = after_e[after_e >= peak_pre]
            if len(recovered) > 0:
                rec_t = recovered.index[0]
                # 위기 종료 시점부터 회복 시점까지의 개월 수
                recovery = (rec_t.year - e.year) * 12 + (rec_t.month - e.month)
                recovery = max(0, int(recovery))
            else:
                recovery = None

        rows.append({
            "Event": ev["name"],
            "Period": f"{s.strftime('%Y-%m')} ~ {e.strftime('%Y-%m')}",
            "Fund": fund_cum,
            "SPY": spy_cum,
            "Active": active,
            "MDD": mdd,
            "Recovery_months": recovery,
        })
    return pd.DataFrame(rows)


def calc_avg_recovery_from_subevents(sub_event_df: pd.DataFrame) -> float:
    """
    4 Sub-events DataFrame 의 평균 Recovery Time (개월). NaN 제외.

    Backtesting 페이지 영역 6 sub-events 용.
    Risk Metrics 페이지의 'Series 기반' 평균 recovery 와 분리
      → calc_avg_recovery_time (Series) vs calc_avg_recovery_from_subevents (DataFrame).
    """
    if "Recovery_months" not in sub_event_df.columns:
        return np.nan
    rec = sub_event_df["Recovery_months"].dropna()
    return float(rec.mean()) if len(rec) > 0 else np.nan


# 영역 4 — Regime 12 메트릭 정의 (변경 용이)
REGIME_METRICS: list[str] = [
    "CAGR", "Sortino", "Sharpe", "MDD", "Volatility",
    "Active_Return", "IR", "Win_Rate", "Calmar",
    "VaR_5", "CVaR_5", "Beta",
]

REGIME_PERIODS: dict[str, tuple[str, str]] = {
    "R1 회복": ("2010-01-01", "2012-06-30"),
    "R2 확장": ("2012-07-01", "2019-12-31"),
    "R3 변동": ("2020-01-01", "2023-12-31"),
    "HO 24개월": ("2024-01-01", "2025-12-31"),
    "FULL":   ("2010-01-01", "2025-12-31"),
}


def calc_regime_full_metrics(
    ret: pd.Series, spy: pd.Series, rf: pd.Series
) -> pd.DataFrame:
    """
    영역 4 — Regime × 12 메트릭 표.
    행: Regime / 컬럼: 메트릭. 모두 final 정합 함수 사용.
    """
    rows = []
    for label, (s, e) in REGIME_PERIODS.items():
        sub_ret = ret[(ret.index >= s) & (ret.index <= e)].dropna()
        sub_spy = spy[(spy.index >= s) & (spy.index <= e)].dropna()
        sub_rf = rf.reindex(sub_ret.index).fillna(0)
        if len(sub_ret) < 6:
            continue
        cum = (1 + sub_ret).cumprod()
        spy_cum = (1 + sub_spy).cumprod() if len(sub_spy) > 0 else None

        cagr_val = calc_cagr_subperiod(ret, s, e)
        sortino_val = calc_sortino_subperiod(ret, rf, s, e)
        sharpe_val = calc_sharpe_subperiod(ret, rf, s, e)
        mdd_val = calc_mdd_subperiod(ret, s, e)
        vol_val = float(sub_ret.std() * np.sqrt(12))
        active_ret = (
            float(cum.iloc[-1] - 1) - float(spy_cum.iloc[-1] - 1)
            if spy_cum is not None and len(spy_cum) > 0
            else np.nan
        )
        # IR = Active Return / Tracking Error
        if len(sub_spy) > 0:
            common = sub_ret.index.intersection(sub_spy.index)
            te = float((sub_ret.reindex(common) - sub_spy.reindex(common)).std() * np.sqrt(12))
            ir_val = (active_ret * (12 / len(sub_ret))) / te if te > 0 else np.nan
        else:
            ir_val = np.nan
        win_rate = calc_win_rate(sub_ret)
        calmar = (cagr_val / abs(mdd_val)) if (not pd.isna(mdd_val) and mdd_val != 0) else np.nan
        var_5 = float(sub_ret.quantile(0.05))
        cvar_5 = calc_cvar(sub_ret, alpha=0.05)
        beta_val = calc_beta(sub_ret, sub_spy, sub_rf) if len(sub_spy) > 0 else np.nan

        rows.append({
            "Regime": label,
            "CAGR": cagr_val, "Sortino": sortino_val, "Sharpe": sharpe_val,
            "MDD": mdd_val, "Volatility": vol_val,
            "Active_Return": active_ret, "IR": ir_val,
            "Win_Rate": win_rate, "Calmar": calmar,
            "VaR_5": var_5, "CVaR_5": cvar_5, "Beta": beta_val,
        })
    return pd.DataFrame(rows).set_index("Regime")
