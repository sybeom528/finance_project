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


def calc_hill_estimator(
    returns: pd.Series, k: int | None = None, side: str = "loss"
) -> float:
    """
    Hill (1975) tail index estimator.

    ξ̂_k = (1/k) Σ log(X_(i)) - log(X_(k+1))   (i=1..k)

    Args:
        returns: 일별 수익률 (월별도 가능하지만 sample size 부족)
        k: tail order statistic (None → auto sqrt(n))
        side: "loss" (음수 꼬리, 큰 손실) / "gain" (양수 꼬리)

    Returns:
        ξ̂ (tail index). > 0 = fat tail (극단값 자주). 주식 일반 0.2-0.4.

    학술: Hill, B.M. (1975) "A simple general approach to inference about
    the tail of a distribution." Annals of Statistics, 3, 1163-1174.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 50:
        return np.nan

    if side == "loss":
        values = -r[r < 0].values  # 큰 손실을 양수로
    else:
        values = r[r > 0].values

    if len(values) < 50:
        return np.nan

    # 양수만 (log 위해)
    values = values[values > 1e-12]
    if len(values) < 50:
        return np.nan

    sorted_values = np.sort(values)[::-1]  # 큰 값 먼저
    n = len(sorted_values)

    if k is None:
        k = int(np.sqrt(n))  # auto: sqrt(n) 표준
    if k >= n - 1 or k < 2:
        return np.nan

    log_vals = np.log(sorted_values[:k + 1])
    xi_hat = float(np.mean(log_vals[:k]) - log_vals[k])
    return xi_hat


def hill_plot_data(
    returns: pd.Series, side: str = "loss", k_min: int = 10, k_max: int | None = None
) -> pd.DataFrame:
    """
    Hill plot 데이터 (k 별 ξ̂) — plateau detection 시각화용.

    Returns:
        pd.DataFrame: columns=[k, xi]
    """
    r = pd.Series(returns).dropna()
    if len(r) < 50:
        return pd.DataFrame(columns=["k", "xi"])

    if side == "loss":
        values = -r[r < 0].values
    else:
        values = r[r > 0].values
    values = values[values > 1e-12]

    if len(values) < 50:
        return pd.DataFrame(columns=["k", "xi"])

    sorted_values = np.sort(values)[::-1]
    n = len(sorted_values)

    if k_max is None:
        k_max = min(n - 1, n // 2)

    rows = []
    for k in range(k_min, k_max):
        log_vals = np.log(sorted_values[:k + 1])
        xi = float(np.mean(log_vals[:k]) - log_vals[k])
        rows.append({"k": k, "xi": xi})

    return pd.DataFrame(rows)


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
    final master_table._sharpe_subperiod 정확 재현 (line 127-135).

    NaN 처리: final 은 sub.std() / exc.mean() 등이 pandas default skipna=True
              로 NaN 자동 무시 → 우리도 동일 동작 (명시적 dropna 불필요).
    """
    r = pd.Series(ret)
    sub_raw = r[(r.index >= start) & (r.index <= end)]
    sub = sub_raw.dropna()
    if len(sub) < 6:
        return np.nan
    rf_sub = pd.Series(rf).reindex(sub.index).fillna(0) if isinstance(rf, pd.Series) else pd.Series(rf / periods_per_year, index=sub.index)
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
