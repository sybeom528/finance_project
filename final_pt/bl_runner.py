"""
bl_runner.py — Black-Litterman walk-forward 실행 엔진

04_BL_Walkforward.ipynb 의 dispatcher + monthly_cache + walk_forward 를
재사용 가능한 모듈로 분리. 05b_Analyze.ipynb 에서 Q/PCT_GROUP sensitivity
sweep 을 in-notebook 으로 돌릴 때도 동일 모듈 import 해서 사용.

사용 예 (04 또는 05b_Analyze 안):
    from bl_runner import load_lstm_pred, build_monthly_cache, walk_forward

    # 1. LSTM 예측 로드
    lstm_state = load_lstm_pred(lstm_path, pred_dates)

    # 2. monthly_cache 빌드 (1회)
    cache = build_monthly_cache(
        panel=panel, daily_ret=daily_ret,
        pred_dates=pred_dates, all_dates=all_dates,
        spy_series=spy_series, rf_series=rf_series,
        train_window=60, thresh_daily=0.9,
    )

    # 3. walk_forward 실행 (N회 sensitivity sweep 가능)
    result = walk_forward(cfg, cache, pred_dates, lstm_state,
                          spy_series=spy_series, tau=0.1, pct_group=0.30)
"""
from __future__ import annotations
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from bl_functions import (
    compute_sigma, compute_daily_slice, compute_pi,
    build_P,
    compute_Q_fixed, compute_Q_lambda, compute_Q_inv_lambda,
    compute_Q_raw_lam, compute_Q_vol_spread,
    compute_omega_he,
    black_litterman, optimize_portfolio,
    compute_turnover, apply_tc,
)


# ════════════════════════════════════════════════════════════════
# SPY forward 21bd cumprod fallback
# ════════════════════════════════════════════════════════════════
# walk_forward 안의 r_spy 계산에서 next_date 가 None 이거나 spy_series 룩업이
# NaN 일 때 사용. 종목 fwd_ret_1m (forward 21영업일 cumprod) 과 단위 정합.
_BASE_DIR = Path(__file__).parent
_DAILY_RETURNS_PATH = _BASE_DIR / 'data' / 'daily_returns.pkl'


def _load_daily_spy_simple() -> pd.Series:
    with open(_DAILY_RETURNS_PATH, 'rb') as f:
        daily = pickle.load(f)
    return (np.exp(daily['SPY']) - 1).dropna()


_daily_spy_simple: Optional[pd.Series] = None


def _get_daily_spy_simple() -> pd.Series:
    global _daily_spy_simple
    if _daily_spy_simple is None:
        _daily_spy_simple = _load_daily_spy_simple()
    return _daily_spy_simple


def _compute_spy_fwd_21d(pred_date: pd.Timestamp,
                        daily_spy_simple: Optional[pd.Series] = None) -> float:
    """pred_date 다음 21영업일 SPY 누적수익. panel.fwd_ret_1m 과 정합."""
    if daily_spy_simple is None:
        daily_spy_simple = _get_daily_spy_simple()
    after = daily_spy_simple[daily_spy_simple.index > pred_date]
    if len(after) >= 21:
        return float((1 + after.iloc[:21]).prod() - 1)
    return np.nan


# ════════════════════════════════════════════════════════════════
# 1. LSTM 예측 로드
# ════════════════════════════════════════════════════════════════
def load_lstm_pred(
    lstm_path: str | Path,
    pred_dates: pd.DatetimeIndex,
) -> Dict:
    """
    LSTM 예측 csv 로드 → vol_pred 연환산 + 월별 피벗 + (date, ticker) 인덱스.

    Returns dict:
        - 'available': bool
        - 'preds': DataFrame indexed by (date, ticker)
        - 'monthly': DataFrame pivot (pred_date × ticker, value=vol_pred)
    """
    lstm_path = Path(lstm_path)
    if not lstm_path.exists():
        return {'available': False, 'preds': None, 'monthly': None}

    raw = pd.read_csv(lstm_path, parse_dates=['date'])
    # 단위: log-daily-RV → 연환산 vol (vol_21d 와 같은 단위)
    raw['vol_pred'] = np.exp(raw['y_pred_ensemble']) * np.sqrt(252)
    raw['abs_err']  = (raw['y_pred_ensemble'] - raw['y_true']).abs()

    preds = raw.set_index(['date', 'ticker'])

    # 월별 피벗: 종목·월 기준 마지막 예측값 → 시장 거래일 월말 매핑
    m = raw.copy()
    m['month'] = m['date'].dt.to_period('M')
    last = m.groupby(['ticker', 'month'])['vol_pred'].last().reset_index()
    month_map = {pd.Timestamp(d).to_period('M'): pd.Timestamp(d) for d in pred_dates}
    last['pred_date'] = last['month'].map(month_map)
    last = last.dropna(subset=['pred_date'])
    monthly = last.pivot_table(index='pred_date', columns='ticker', values='vol_pred')

    return {'available': True, 'preds': preds, 'monthly': monthly}


# ════════════════════════════════════════════════════════════════
# 2. Dispatcher 함수 (config → 분기)
# ════════════════════════════════════════════════════════════════
def get_vol_series(
    cfg: dict,
    month_df: pd.DataFrame,
    pred_date: pd.Timestamp,
    lstm_monthly: Optional[pd.DataFrame],
) -> pd.Series:
    """P 행렬에 쓸 변동성 시리즈 반환 — LSTM 예측 단일 옵션."""
    if lstm_monthly is None:
        raise RuntimeError(
            'LSTM 예측 데이터가 없습니다. '
            '03b_Volatility_Forecasting.ipynb 를 먼저 실행하세요.'
        )
    if pred_date not in lstm_monthly.index:
        # 예측 부재 월은 vol_21d 로 fallback (드물게 발생)
        return month_df['vol_21d']

    pred_slice = lstm_monthly.loc[pred_date].dropna()

    # 단위 가드: vol_21d 는 연환산. pred_slice 도 연환산이어야 함.
    if len(pred_slice) > 0 and pred_slice.median() < 0.05:
        raise ValueError(
            f'LSTM pred_slice가 일별 단위로 의심됩니다 '
            f'(median={pred_slice.median():.4f} < 0.05). '
            f'`× np.sqrt(252)` 누락되지 않았는지 vol_pred 산출 코드 확인 필요.'
        )

    vol = month_df['vol_21d'].copy()
    common = vol.index.intersection(pred_slice.index)
    if len(common) >= 5:
        vol[common] = pred_slice[common]
    return vol


def get_prior_weights(
    cfg: dict, valid_tix: list, mcap: pd.Series, vol: Optional[pd.Series] = None,
) -> pd.Series:
    """prior 시장가중치 w_mkt 반환."""
    prior = cfg.get('prior', 'capm_mcap')
    if prior == 'capm_mcap':
        return (mcap / mcap.sum()).reindex(valid_tix).fillna(0)
    elif prior == 'capm_eq':
        n = len(valid_tix)
        return pd.Series(1.0 / n, index=valid_tix)
    elif prior == 'capm_rp':
        if vol is None:
            raise ValueError('capm_rp requires vol parameter')
        inv_vol = 1.0 / vol.replace(0, np.nan).dropna()
        w = inv_vol / inv_vol.sum()
        return w.reindex(valid_tix).fillna(0)
    else:
        raise ValueError(f'prior={prior!r} 미지원')


def get_Q(
    cfg: dict, P: pd.Series, train_dates: pd.DatetimeIndex,
    pred_date: pd.Timestamp, lstm_monthly: Optional[pd.DataFrame],
    pct_group: float = 0.30,
    lam: float = 2.5, spy_excess: float = 0.0, sigma2_mkt: float = 0.001,
) -> float:
    """Q 값 반환."""
    mode = cfg.get('q_mode', 'fixed')

    if mode == 'fixed':
        return compute_Q_fixed(cfg.get('q_value', 0.003))

    elif mode == 'lambda':
        return compute_Q_lambda(lam, cfg.get('q_value', 0.003), cfg.get('lam_mean', 2.5))

    elif mode == 'inv_lambda':
        return compute_Q_inv_lambda(lam, cfg.get('q_value', 0.003), cfg.get('lam_mean', 2.5))

    elif mode == 'raw_lam':
        return compute_Q_raw_lam(spy_excess, sigma2_mkt,
                                  cfg.get('q_value', 0.003), cfg.get('lam_mean', 2.5))

    elif mode == 'vol_spread':
        # LSTM 예측 vol 격차 기반 동적 Q
        if lstm_monthly is None or pred_date not in lstm_monthly.index:
            return cfg.get('q_value', 0.003)
        vol_pred_curr = lstm_monthly.loc[pred_date].dropna()

        past_spreads = []
        for past_dt in train_dates:
            if past_dt not in lstm_monthly.index:
                continue
            vp = lstm_monthly.loc[past_dt].dropna()
            if len(vp) < 20:
                continue
            n_grp = max(1, int(len(vp) * cfg.get('pct_group', pct_group)))
            sorted_vp = vp.sort_values()
            low_m  = sorted_vp.iloc[:n_grp].mean()
            high_m = sorted_vp.iloc[-n_grp:].mean()
            past_spreads.append(high_m - low_m)
        if not past_spreads:
            return cfg.get('q_value', 0.003)
        spread_ref = float(np.median(past_spreads))

        return compute_Q_vol_spread(P, vol_pred_curr,
                                     cfg.get('q_value', 0.003), spread_ref)

    else:
        raise ValueError(f'q_mode={mode!r} 미지원')


def get_omega(cfg: dict, P: pd.Series, Sigma: pd.DataFrame, tau: float) -> float:
    """Omega 값 반환. ff3_paper 모드는 walk_forward 에서 직접 처리."""
    mode = cfg.get('omega_mode', 'he_litterman')
    if mode == 'he_litterman':
        return compute_omega_he(P, Sigma, tau)
    # 'ff3_paper' 는 walk_forward 안 inline 처리 (직전월 예측오차²)
    raise ValueError(f'omega_mode={mode!r} 미지원 (he_litterman 또는 ff3_paper 만 허용)')


# ════════════════════════════════════════════════════════════════
# 3. 월별 캐시 빌드 (모든 실험 공유)
# ════════════════════════════════════════════════════════════════
def build_monthly_cache(
    panel: pd.DataFrame,
    daily_ret: pd.DataFrame,
    pred_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
    spy_series: pd.Series,
    rf_series: pd.Series,
    train_window: int = 60,
    thresh_daily: float = 0.9,
    verbose: bool = True,
) -> Dict:
    """
    월별 공유 데이터 캐시 빌드 (Sigma, π 컴포넌트). 매 실험에서 동일하므로 1회만 계산.

    실험별로 달라지는 것: vol_series(p_mode), P(p_weight), Q, omega, w
    """
    cache: Dict = {}
    t0 = time.time()

    for i, pred_date in enumerate(pred_dates):
        if verbose and i % 24 == 0:
            elapsed = time.time() - t0
            pct = (i + 1) / len(pred_dates)
            eta = elapsed / pct * (1 - pct) if pct > 0.01 else 0
            print(f'  캐시 {pred_date.strftime("%Y-%m")} '
                  f'({i+1}/{len(pred_dates)}, {pct:.0%}) | '
                  f'경과 {elapsed/60:.1f}분 | ETA {eta/60:.1f}분', flush=True)
        try:
            month_df = panel.xs(pred_date, level='date').dropna(
                subset=['vol_21d', 'log_mcap', 'ret_1m'])
            if len(month_df) < 30:
                continue

            daily_slice, valid_tix = compute_daily_slice(
                pred_date, month_df.index.tolist(),
                daily_ret, train_window, thresh_daily)
            if len(valid_tix) < 20:
                continue

            Sigma = compute_sigma(daily_slice, scale=21)
            month_df = month_df.reindex(valid_tix)
            mcap = np.exp(month_df['log_mcap'])

            idx = all_dates.get_loc(pred_date)
            train_dates = all_dates[max(0, idx - train_window): idx]
            spy_s = spy_series.reindex(train_dates)
            rf_s  = rf_series.reindex(train_dates)

            next_date = all_dates[idx + 1] if idx + 1 < len(all_dates) else None

            cache[pred_date] = {
                'month_df'   : month_df,
                'valid_tix'  : valid_tix,
                'Sigma'      : Sigma,
                'mcap'       : mcap,
                'train_dates': train_dates,
                'spy_excess' : float((spy_s - rf_s).mean()),
                'sigma2_mkt' : float(spy_s.var()),
                'next_date'  : next_date,
            }
        except Exception as e:
            if verbose:
                print(f'  [캐시 에러] {pred_date.date()}: {e}')

    if verbose:
        elapsed_min = (time.time() - t0) / 60
        print(f'\n캐시 완료: {len(cache)}/{len(pred_dates)}개월 | {elapsed_min:.1f}분')
    return cache


# ════════════════════════════════════════════════════════════════
# 4. walk_forward — 한 실험 실행 (캐시에서 Σ 로드)
# ════════════════════════════════════════════════════════════════
def walk_forward(
    cfg: dict,
    monthly_cache: Dict,
    pred_dates: pd.DatetimeIndex,
    lstm_state: Dict,
    spy_series: pd.Series,
    tau: float = 0.1,
    pct_group: float = 0.30,
    verbose: bool = True,
) -> Dict:
    """
    한 슬롯의 walk-forward 백테스트 → result dict.

    Returns:
        'config', 'ret', 'gross_ret', 'spy_ret',
        'weights', 'comp', 'meta', 'errors'
    """
    name     = cfg.get('name', 'unnamed')
    tc       = cfg.get('tc', 0.002)   # BASELINE 과 일치 (20bp per-side)
    max_w    = cfg.get('max_weight', 0.10)
    p_weight = cfg.get('p_weight', 'mcap')
    lstm_monthly = lstm_state.get('monthly') if lstm_state else None

    ret_list, comp_list, meta_list = [], [], []
    spy_list, err_list = [], []
    weights_history = {}
    prev_w = None
    _op_q_prev = None   # ff3_paper omega 상태 (직전월 Q)
    _op_P_prev = None
    t0 = time.time()

    for i, pred_date in enumerate(pred_dates):
        if verbose and (i % 12 == 0 or i == len(pred_dates) - 1):
            elapsed = time.time() - t0
            pct = (i + 1) / len(pred_dates)
            eta = elapsed / pct * (1 - pct) if pct > 0.01 else 0
            print(f'  [{name}] {pred_date.strftime("%Y-%m")} '
                  f'({i+1}/{len(pred_dates)}, {pct:.0%}) | '
                  f'경과 {elapsed/60:.1f}분 | ETA {eta/60:.1f}분', flush=True)

        if pred_date not in monthly_cache:
            continue
        c = monthly_cache[pred_date]
        month_df    = c['month_df']
        valid_tix   = c['valid_tix']
        Sigma       = c['Sigma']
        mcap        = c['mcap']
        train_dates = c['train_dates']
        spy_excess  = c['spy_excess']
        sigma2_mkt  = c['sigma2_mkt']
        next_date   = c['next_date']

        try:
            w_mkt = get_prior_weights(cfg, valid_tix, mcap,
                                       vol=month_df['vol_21d'])
            pi, lam = compute_pi(Sigma, w_mkt, spy_excess, sigma2_mkt)

            vol_series = get_vol_series(cfg, month_df, pred_date, lstm_monthly)
            P = build_P(
                vol_series.reindex(valid_tix).fillna(vol_series.median()),
                mcap, pct=cfg.get('pct_group', pct_group), weighting=p_weight)

            lam_q = (float(np.clip(spy_excess / sigma2_mkt, 0.5, 10.0))
                     if sigma2_mkt > 1e-10 else 2.5)
            Q = get_Q(cfg, P, train_dates, pred_date, lstm_monthly,
                      pct_group=pct_group,
                      lam=lam_q, spy_excess=spy_excess, sigma2_mkt=sigma2_mkt)

            # ff3_paper omega: 직전월 예측오차² 적응형 Bayesian
            if cfg.get('omega_mode') == 'ff3_paper':
                if _op_q_prev is not None and _op_P_prev is not None:
                    actual_p = float(
                        month_df['ret_1m'].reindex(_op_P_prev.index).fillna(0)
                        @ _op_P_prev
                    )
                    omega = max((_op_q_prev - actual_p) ** 2, 1e-8)
                else:
                    omega = compute_omega_he(P, Sigma, tau)   # 첫달 폴백
                _op_q_prev = Q
                _op_P_prev = P.copy()
            else:
                omega = get_omega(cfg, P, Sigma, tau)

            mu_BL, sig_BL = black_litterman(pi, Sigma, P, Q, omega, tau)
            w = optimize_portfolio(mu_BL, Sigma.values, lam, max_w)

            actual_ret = month_df['fwd_ret_1m'].reindex(valid_tix).fillna(0)
            gross_ret  = float(w @ actual_ret)
            turnover   = compute_turnover(w, prev_w) if prev_w is not None else 0.0
            net_ret    = apply_tc(gross_ret, turnover, tc)
            # next_date 가 None 이거나 spy_series 룩업이 NaN 이면 21bd cumprod fallback.
            # walk-forward 가 분리 실행(TEST+HO)되면 각 구간 끝에서 next_date=None 이라
            # 이전 코드(`else np.nan`) 가 마지막 시점을 NaN 으로 박는 버그가 있었음.
            if next_date is not None:
                r_spy = float(spy_series.get(next_date, np.nan))
                if pd.isna(r_spy):
                    r_spy = _compute_spy_fwd_21d(pred_date)
            else:
                r_spy = _compute_spy_fwd_21d(pred_date)

            ret_list.append({'date': pred_date, 'ret': net_ret, 'gross_ret': gross_ret})
            spy_list.append({'date': pred_date, 'ret': r_spy})

            vol_col      = month_df['vol_21d'].reindex(valid_tix)
            n_g          = max(1, int(len(valid_tix) * cfg.get('pct_group', pct_group)))
            sv           = vol_col.sort_values()
            low_tix_all  = sv.index[:n_g].tolist()
            high_tix_all = sv.index[-n_g:].tolist()

            comp_list.append({
                'date'        : pred_date,
                'n_stocks'    : len(valid_tix),
                'eff_n'       : 1.0 / float((w**2).sum()) if (w**2).sum() > 0 else 0,
                'top10_share' : float(w.nlargest(10).sum()),
                'top1_weight' : float(w.max()),
                'top1_ticker' : w.idxmax(),
                'avg_vol'     : float((w * vol_col).sum()),
                'low_weight'  : float(w.reindex(low_tix_all).sum()),
                'high_weight' : float(w.reindex(high_tix_all).sum()),
                'turnover'    : turnover,
                'tc_cost'     : turnover * tc,
            })
            meta_list.append({'date': pred_date, 'Q': Q, 'lam': lam})
            weights_history[pred_date] = w
            prev_w = w

        except Exception as e:
            err_list.append({'date': pred_date, 'error': str(e)})
            if verbose and len(err_list) <= 3:
                print(f'  [{name}] 에러 {pred_date.date()}: {e}')

    total_min = (time.time() - t0) / 60
    ret_df  = pd.DataFrame(ret_list).set_index('date')  if ret_list  else pd.DataFrame()
    spy_df  = pd.DataFrame(spy_list).set_index('date')  if spy_list  else pd.DataFrame()
    comp_df = pd.DataFrame(comp_list).set_index('date') if comp_list else pd.DataFrame()
    meta_df = pd.DataFrame(meta_list).set_index('date') if meta_list else pd.DataFrame()
    wts_df  = pd.DataFrame(weights_history).T            if weights_history else pd.DataFrame()

    if verbose:
        print(f'  → [{name}] 완료: {len(ret_df)}개월 / {len(err_list)}개 에러 / {total_min:.1f}분')

    return {
        'config'   : cfg,
        'ret'      : ret_df['ret']       if 'ret'       in ret_df.columns else pd.Series(dtype=float),
        'gross_ret': ret_df['gross_ret'] if 'gross_ret' in ret_df.columns else pd.Series(dtype=float),
        'spy_ret'  : spy_df['ret']       if 'ret'       in spy_df.columns else pd.Series(dtype=float),
        'weights'  : wts_df,
        'comp'     : comp_df,
        'meta'     : meta_df,
        'errors'   : err_list,
    }
