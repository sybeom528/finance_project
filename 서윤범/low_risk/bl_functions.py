"""
bl_functions.py — Black-Litterman 수식 함수 모음

순수 함수만 포함. 데이터 로딩/파일 I/O 없음.
각 함수는 BL 슬롯 하나(Sigma / Prior / P / Q / Omega / BL / TC / Metrics)를 담당.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import warnings

ANN = np.sqrt(12)

# ══════════════════════════════════════════════════════════════
# 1. SIGMA — 공분산 추정
# ══════════════════════════════════════════════════════════════

def compute_sigma(ret_matrix: pd.DataFrame, scale: float = 21) -> pd.DataFrame:
    """일별 수익률 행렬 → Ledoit-Wolf 공분산 × scale (월별 단위)."""
    lw = LedoitWolf().fit(ret_matrix.values)
    cov = lw.covariance_ * scale
    return pd.DataFrame(cov, index=ret_matrix.columns, columns=ret_matrix.columns)


def compute_daily_slice(
    pred_date: pd.Timestamp,
    universe: list,
    daily_ret: pd.DataFrame,
    train_window: int = 60,
    thresh_daily: float = 0.9,
) -> tuple[pd.DataFrame, list]:
    """
    pred_date 이전 train_window 개월 일별 수익률 슬라이스 추출.
    NaN이 thresh_daily 미만인 종목만 포함 (나머지 NaN은 0으로 채움).
    Returns: (유효 티커 daily_slice, valid_tickers)
    """
    start = pred_date - pd.DateOffset(months=train_window)
    cols = [t for t in universe if t in daily_ret.columns]
    window = daily_ret.loc[start:pred_date, cols]
    T = len(window)
    thresh = int(T * thresh_daily)
    valid = window.columns[window.notna().sum() >= thresh].tolist()
    return window[valid].fillna(0), valid


# ══════════════════════════════════════════════════════════════
# 2. PRIOR — 균형 기대수익률 π
# ══════════════════════════════════════════════════════════════

def compute_pi(
    Sigma: pd.DataFrame,
    w_mkt: pd.Series,
    spy_excess_ret: float, # 동적 lambda 구현을 위해 남겨 놓음. 2.5 고정이라 spy_excess_ret은 사용되진 않음.
    sigma2_mkt: float,
    lam_fixed: float = 2.5,
) -> tuple[pd.Series, float]:
    """CAPM 역산 π = λ·Σ·w_mkt. lam_fixed=None 이면 동적 계산."""
    if lam_fixed is not None:
        lam = float(lam_fixed)
    else:
        lam = float(np.clip(spy_excess_ret / sigma2_mkt if sigma2_mkt > 0 else 2.5, 0.5, 10.0))
    pi = lam * Sigma @ w_mkt
    return pi, lam


# ══════════════════════════════════════════════════════════════
# 3. P — 뷰 포트폴리오 행렬
# ══════════════════════════════════════════════════════════════

def build_P(
    vol_series: pd.Series,
    mcap_series: pd.Series,
    pct: float = 0.30,
    weighting: str = 'mcap',
) -> pd.Series:
    """
    변동성 정렬 P 행렬 (뷰 포트폴리오).
    weighting: 'mcap' | 'eq' | 'rp' | 'vol_mcap'
      - mcap     : 시총 비례 (하위/상위 30% 컷, 기본)
      - eq       : 동일가중 (하위/상위 30% 컷)
      - rp       : 전체 유니버스, Long 1/σ / Short σ  (순수 vol 신호, 대칭)
      - vol_mcap : 전체 유니버스, Long (1/σ)×mcap / Short σ×mcap  (vol × 시총, 대칭)
    """
    n_group = max(1, int(len(vol_series) * pct))
    sorted_idx = vol_series.sort_values().index
    low_risk  = sorted_idx[:n_group]
    high_risk = sorted_idx[-n_group:]

    P = pd.Series(0.0, index=vol_series.index)

    if weighting == 'mcap':
        low_m  = mcap_series[low_risk]
        high_m = mcap_series[high_risk]
        P[low_risk]  =  low_m  / low_m.sum()
        P[high_risk] = -high_m / high_m.sum()

    elif weighting == 'eq':
        P[low_risk]  =  1.0 / n_group
        P[high_risk] = -1.0 / n_group

    elif weighting == 'rp':
        # 전체 유니버스 사용 — 30% 컷 없음, 순수 vol 신호 (대칭)
        # 롱: 1/σ → 저변동일수록 더 많이 담김
        # 숏:   σ → 고변동일수록 더 강하게 숏
        # 이 부분은 재천님 구현 방식과 다른데, 원하시면 수정하거나 추가하셔도 좋습니다!
        inv_vol = (1.0 / vol_series).replace(0, np.nan).dropna()
        vol_all = vol_series.replace(0, np.nan).dropna()
        common  = inv_vol.index.intersection(vol_all.index)
        P[common] = inv_vol[common] / inv_vol[common].sum() \
                  - vol_all[common] / vol_all[common].sum()

    elif weighting == 'vol_mcap':
        # 전체 유니버스 사용 — 30% 컷 없음, vol × 시총 신호 (대칭)
        # 롱: (1/σ)×mcap → 저변동 대형주일수록 더 많이 담김
        # 숏:    σ ×mcap → 고변동 대형주일수록 더 강하게 숏
        inv_vol_mcap = (mcap_series / vol_series).replace(0, np.nan).dropna()
        vol_mcap     = (vol_series  * mcap_series).replace(0, np.nan).dropna()
        common       = inv_vol_mcap.index.intersection(vol_mcap.index)
        P[common] = inv_vol_mcap[common] / inv_vol_mcap[common].sum() \
                  - vol_mcap[common]     / vol_mcap[common].sum()

    else:
        raise ValueError(f"weighting={weighting!r} 미지원. 'mcap'/'eq'/'rp'/'vol_mcap' 중 선택")

    return P


# ══════════════════════════════════════════════════════════════
# 4. Q — 뷰 기대수익률
# ══════════════════════════════════════════════════════════════

def compute_Q_fixed(q_value: float) -> float:
    """고정 Q값 그대로 반환."""
    return float(q_value)


def compute_Q_ff3(
    P: pd.Series,
    ret_matrix: pd.DataFrame,
    ff3_train: pd.DataFrame,
    rf_train: pd.Series,
) -> float:
    """
    FF3 회귀로 각 종목 기대수익률 추정 → Q = P @ r_hat.
    ret_matrix: 훈련 구간 월별 수익률 (date × ticker)
    """
    view_tickers = P[P != 0].index.intersection(ret_matrix.columns).tolist()
    if not view_tickers:
        return 0.003

    ff3_aligned = ff3_train.reindex(ret_matrix.index).dropna()
    rf_aligned  = rf_train.reindex(ff3_aligned.index).fillna(0)
    n = len(ff3_aligned)
    if n < 24:
        return 0.003

    X      = np.column_stack([np.ones(n), ff3_aligned[['mkt_rf', 'smb', 'hml']].values])
    
    X_next = np.array([1.0] + ff3_aligned[['mkt_rf', 'smb', 'hml']].mean().tolist())
    rf_next = float(rf_train.iloc[-1]) if len(rf_train) > 0 else 0.0

    r_hat = pd.Series(0.0, index=ret_matrix.columns)
    for t in view_tickers:
        y = ret_matrix[t].reindex(ff3_aligned.index) - rf_aligned
        valid = y.notna()
        if valid.sum() < 12:
            continue
        coef = np.linalg.lstsq(X[valid], y[valid].values, rcond=None)[0]
        r_hat[t] = float(X_next @ coef) + rf_next

    return float(P.reindex(ret_matrix.columns).fillna(0) @ r_hat)


def compute_Q_spread(
    panel_train: pd.DataFrame,
    pct: float = 0.30,
    vol_col: str = 'vol_21d',
    ret_col: str = 'ret_1m',
) -> float:
    """
    훈련 구간 월별 저변동-고변동 수익률 스프레드 평균.
    look-ahead bias 없음 (훈련 구간 과거 데이터만 사용).
    """
    spreads = []
    for _, group in panel_train.groupby(level='date'):
        valid = group[[vol_col, ret_col]].dropna()
        if len(valid) < 20:
            continue
        n_g = max(1, int(len(valid) * pct))
        sorted_idx = valid[vol_col].sort_values().index
        low_r  = valid.loc[sorted_idx[:n_g],  ret_col].mean()
        high_r = valid.loc[sorted_idx[-n_g:], ret_col].mean()
        spreads.append(low_r - high_r)
    return float(np.nanmean(spreads)) if spreads else 0.003


def compute_Q_regime(
    spy_series_train: pd.Series,
    q_table: dict,
    lookback: int = 12,
) -> float:
    """
    최근 lookback 개월 SPY 변동성으로 레짐 분류 → Q 결정.
    q_table 예시: {'low_vol': 0.001, 'normal': 0.003, 'high_vol': 0.006}
    레짐 기준: 연환산 시장변동성 > 20% → high_vol / > 12% → normal / else → low_vol
    
    레짐은 간단하게만 구현. 추후 HMM으로 고도화 예정
    """
    recent = spy_series_train.iloc[-lookback:] if len(spy_series_train) >= lookback else spy_series_train
    ann_vol = recent.std() * ANN
    if ann_vol > 0.20:
        return float(q_table.get('high_vol', 0.006))
    elif ann_vol > 0.12:
        return float(q_table.get('normal',   0.003))
    else:
        return float(q_table.get('low_vol',  0.001))


# ══════════════════════════════════════════════════════════════
# 5. OMEGA — 뷰 불확실성
# ══════════════════════════════════════════════════════════════

def compute_omega_he(P: pd.Series, Sigma: pd.DataFrame, tau: float) -> float:
    """He-Litterman (1999) 표준: ω = τ·P·Σ·P^T."""
    p = P.values
    return max(float(tau * p @ Sigma.values @ p), 1e-8)


def compute_omega_scaled(
    P: pd.Series,
    Sigma: pd.DataFrame,
    tau: float,
    scale: float,
) -> float:
    """He-Litterman × scale 배수. scale < 1 → 뷰에 더 자신감 / scale > 1 → 덜 자신감."""
    return max(compute_omega_he(P, Sigma, tau) * scale, 1e-8)


def compute_omega_rmse(
    P: pd.Series,
    Sigma: pd.DataFrame,
    tau: float,
    pred_rmse: float,
    base_rmse: float = 0.39,
) -> float:
    """
    LSTM 예측 RMSE로 Omega 스케일링.
    RMSE가 base_rmse보다 높으면 Omega 증가 (뷰 신뢰도 감소).
    scale = (pred_rmse / base_rmse)^2

    이 함수의 경우, baseline을 없애는 것도 방법이고, 
    다른 방식으로 구현해도 됨(변경해도 된다는 의미)
    아니면 그냥 추가해도 되고
    """
    scale = (pred_rmse / base_rmse) ** 2 if base_rmse > 0 else 1.0
    return compute_omega_scaled(P, Sigma, tau, scale)


# ══════════════════════════════════════════════════════════════
# 6. BL 핵심 수식
# ══════════════════════════════════════════════════════════════

def black_litterman(
    pi: pd.Series,
    Sigma: pd.DataFrame,
    P: pd.Series,
    q: float,
    omega: float,
    tau: float,
) -> pd.Series:
    """
    Sherman-Woodbury 단순화 BL 사후 기대수익률.
    μ_BL = π + (τΣ·P^T) · (q - P·π) / (P·τΣ·P^T + ω)
    """
    p    = P.values
    tSig = tau * Sigma.values
    M    = float(p @ tSig @ p) + omega
    diff = q - float(p @ pi.values)
    return pd.Series(pi.values + tSig @ p * (diff / M), index=pi.index)


def optimize_portfolio(
    mu_BL: pd.Series,
    Sigma: pd.DataFrame,
    lam: float,
    max_weight: float = 0.10,
) -> pd.Series:
    """
    Markowitz MVO (long-only).
    min_w (λ/2)·w^T·Σ·w - w^T·μ_BL
    s.t. Σw=1, 0≤w≤max_weight
    """
    n   = len(mu_BL)
    mu  = mu_BL.values
    Sig = Sigma.values

    res = minimize(
        lambda w: 0.5 * lam * w @ Sig @ w - w @ mu,
        np.ones(n) / n,
        jac=lambda w: lam * Sig @ w - mu,
        method='SLSQP',
        bounds=[(0, max_weight)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
    )
    if res.success:
        return pd.Series(res.x, index=mu_BL.index)
    warnings.warn(f'optimize_portfolio 수렴 실패 (n={n}) → 1/N fallback')
    return pd.Series(np.ones(n) / n, index=mu_BL.index)


# ══════════════════════════════════════════════════════════════
# 7. 거래비용 (Phase 3 방식)
# ══════════════════════════════════════════════════════════════

def compute_turnover(w_new: pd.Series, w_prev: pd.Series) -> float:
    """
    |w_new - w_prev| 합계 (신규/이탈 종목 포함).
    결과는 0~2 범위 (완전 교체 시 2.0).

    재천님 코드 기반으로 만들어진 부분.
    """
    common   = w_new.index.intersection(w_prev.index)
    new_only = w_new.index.difference(w_prev.index)
    old_only = w_prev.index.difference(w_new.index)
    return (
        (w_new.loc[common] - w_prev.loc[common]).abs().sum()
        + w_new.loc[new_only].abs().sum()
        + w_prev.loc[old_only].abs().sum()
    )


def apply_tc(gross_ret: float, turnover: float, tc: float) -> float:
    """net_ret = gross_ret - turnover × tc."""
    return gross_ret - turnover * tc


# ══════════════════════════════════════════════════════════════
# 8. 성과 지표
# ══════════════════════════════════════════════════════════════

def compute_metrics(
    ret: pd.Series,
    rf: pd.Series,
    label: str = '',
    mkt_ret: pd.Series = None,
) -> dict:
    """
    포트폴리오 성과 지표 전체 계산.

    Parameters
    ----------
    ret     : 월별 순수익률 시리즈
    rf      : 무위험 수익률 시리즈 (월별)
    label   : 출력용 레이블
    mkt_ret : 시장(SPY) 월별 수익률 — None이면 Beta/Alpha/Treynor NaN 처리

    Returns
    -------
    dict: 아래 지표 포함
      기본  : sharpe, sortino, cagr, vol, mdd, calmar, cum_ret
      분포  : skewness, win_rate, avg_win, avg_loss, cvar_5
      MDD   : mdd_duration (월)
      시장  : beta, alpha, treynor  (mkt_ret 있을 때만 유효)
    """
    rf_a   = rf.reindex(ret.index).fillna(0)
    excess = ret - rf_a

    # ── 기본 ──────────────────────────────────────────────────
    sr     = excess.mean() / excess.std() * ANN if excess.std() > 0 else np.nan
    cagr   = ret.mean() * 12
    vol    = ret.std() * ANN
    cum    = (1 + ret).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    mdd    = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    # ── Sortino (하방 변동성 기준) ────────────────────────────
    downside     = excess[excess < 0]
    down_std     = downside.std() * ANN if len(downside) > 1 else np.nan
    sortino      = excess.mean() * 12 / down_std if (down_std and down_std > 0) else np.nan

    # ── 분포 ──────────────────────────────────────────────────
    skewness     = float(ret.skew())
    win_rate     = float((ret > 0).mean())
    wins         = ret[ret > 0]
    losses       = ret[ret < 0]
    avg_win      = float(wins.mean())  if len(wins)   > 0 else np.nan
    avg_loss     = float(losses.mean()) if len(losses) > 0 else np.nan

    # CVaR 5% — 최악 5% 구간 평균 손실
    cutoff       = ret.quantile(0.05)
    cvar_5       = float(ret[ret <= cutoff].mean())

    # ── MDD 지속 기간 (연속 underwater 월 최대) ───────────────
    underwater   = (dd < 0).astype(int)
    max_dur, cur = 0, 0
    for v in underwater:
        cur = cur + 1 if v else 0
        max_dur = max(max_dur, cur)
    mdd_duration = max_dur

    # ── 시장 대비 지표 (mkt_ret 있을 때만) ───────────────────
    if mkt_ret is not None:
        mkt_a     = mkt_ret.reindex(ret.index).fillna(0)
        mkt_exc   = mkt_a - rf_a
        cov_mat   = np.cov(excess.values, mkt_exc.values)
        beta      = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else np.nan
        alpha     = (excess.mean() - beta * mkt_exc.mean()) * 12 if not np.isnan(beta) else np.nan
        treynor   = excess.mean() * 12 / beta if (not np.isnan(beta) and beta != 0) else np.nan
    else:
        beta, alpha, treynor = np.nan, np.nan, np.nan

    return {
        'label'       : label,
        # 기본
        'sharpe'      : round(sr,           3),
        'sortino'     : round(sortino,       3),
        'cagr'        : round(cagr,          4),
        'vol'         : round(vol,           4),
        'mdd'         : round(mdd,           4),
        'calmar'      : round(calmar,        3),
        'cum_ret'     : round(float(cum.iloc[-1] - 1), 4),
        # 분포
        'skewness'    : round(skewness,      3),
        'win_rate'    : round(win_rate,      3),
        'avg_win'     : round(avg_win,       4),
        'avg_loss'    : round(avg_loss,      4),
        'cvar_5'      : round(cvar_5,        4),
        # MDD
        'mdd_duration': mdd_duration,
        # 시장 대비
        'beta'        : round(beta,          3) if not np.isnan(beta)    else np.nan,
        'alpha'       : round(alpha,         4) if not np.isnan(alpha)   else np.nan,
        'treynor'     : round(treynor,       3) if not np.isnan(treynor) else np.nan,
    }
