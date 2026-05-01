"""
Phase 3-2 v2 §6 standalone runner (nbconvert 1h timeout 우회용).

02a_v2.ipynb Cell 22 (§6-1) + Cell 23 (§6-2) 의 BL 백테스트 walk-forward 루프를
별도 Python 프로세스로 실행. 결과는 Cell 23 이 기대하는 동일한 pickle 형식으로 저장
(`data/bl_weights_v2_sanity_check.pkl`).

이후 jupyter nbconvert 재실행 시 §6-2 셀이 캐시 hit 으로 즉시 통과.

사용:
    python scripts/_run_02a_v2_sec6.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# Force utf-8 stdout (Windows CP949 회피)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# Working directory: 노트북과 동일하게 시계열_Test/Phase3_Robust_Extensions/
SCRIPT_DIR = Path(__file__).resolve().parent
NB_DIR = SCRIPT_DIR.parent
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import warnings
import pickle

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR
from scripts.black_litterman import (
    compute_pi, build_P, compute_omega, black_litterman, optimize_portfolio,
    Q_FIXED, PCT_GROUP, DEFAULT_TAU, LAM_FIXED,
)
from scripts.covariance import estimate_covariance, DAYS_PER_MONTH
from scripts.universe import get_or_build_membership

t0 = time.time()
print('=' * 70)
print('  Phase 3-2 v2 §6 standalone runner (BL backtest walk-forward)')
print('=' * 70)

bootstrap()
print(f'NB_DIR: {NB_DIR}')
print(f'DATA_DIR: {DATA_DIR}')
print()

# ============================================================
# §3 학습 결과 로드 (ensemble_sw)
# ============================================================
ens_path = DATA_DIR / 'ensemble_predictions_stockwise.csv'
print(f'[로드] {ens_path.name} ({ens_path.stat().st_size / 1e6:.0f} MB)')
ensemble_sw = pd.read_csv(ens_path, parse_dates=['date'])
print(f'  ensemble_sw: {ensemble_sw.shape}, unique 종목: {ensemble_sw["ticker"].nunique()}')
print()

# ============================================================
# Cell 22 §6-1: 데이터 로드 + 사전 준비
# ============================================================
print('=' * 70)
print('  §6-1: 데이터 로드 + 사전 준비')
print('=' * 70)

# panel + market
panel = pd.read_csv(
    DATA_DIR / 'daily_panel.csv', parse_dates=['date'],
    usecols=['date', 'ticker', 'log_ret', 'vol_21d', 'mcap_value', 'rf_daily'],
)
print(f'  panel: {panel.shape}')

market = pd.read_csv(DATA_DIR / 'market_data.csv', index_col='date', parse_dates=True)
spy_daily = market['SPY'].pct_change().dropna()
spy_lr = np.log(1 + spy_daily)

# rf 일별
rf_daily = panel.groupby('date')['rf_daily'].mean()
rf_lr = rf_daily.reindex(spy_lr.index, method='ffill').fillna(0)

# 시장 risk premium (월별 환산)
spy_excess_monthly = float((spy_lr - rf_lr).mean() * DAYS_PER_MONTH)
spy_sigma2_monthly = float(spy_lr.var() * DAYS_PER_MONTH)
print(f'  spy_excess_monthly: {spy_excess_monthly:.6f} (월 {spy_excess_monthly*100:.2f}%)')
print(f'  spy_sigma2_monthly: {spy_sigma2_monthly:.6f}')

# 일별 ret pivot (Σ 추정용)
daily_lr = panel.pivot_table(index='date', columns='ticker', values='log_ret')
print(f'  daily_lr: {daily_lr.shape}')

# 학습 종목 (02a)
trained_tickers = set(ensemble_sw['ticker'].unique())
print(f'  학습 종목: {len(trained_tickers)}')

# Phase 3-2 (2026-04-30): OOS 2010-2024 + 2025 hold-out
OOS_START = '2010-01-01'
OOS_END = '2024-12-31'
HOLDOUT_START = '2025-01-01'
HOLDOUT_END = '2025-12-31'

reb_dates_all = market.groupby(market.index.to_period('M')).tail(1).index
oos_dates = reb_dates_all[(reb_dates_all >= OOS_START) & (reb_dates_all <= OOS_END)]
holdout_dates = reb_dates_all[(reb_dates_all >= HOLDOUT_START) & (reb_dates_all <= HOLDOUT_END)]
reb_dates = pd.DatetimeIndex(list(oos_dates) + list(holdout_dates))
month_to_eom = {pd.Timestamp(d).to_period('M'): pd.Timestamp(d) for d in reb_dates}
print(f'  리밸런싱 시점:')
print(f'    OOS      ({OOS_START}~{OOS_END}): {len(oos_dates)} 개월')
print(f'    Hold-out ({HOLDOUT_START}~{HOLDOUT_END}): {len(holdout_dates)} 개월')
print(f'    total: {len(reb_dates)} 개월')

# ML 예측 월별 피벗 (rebalance date 기준)
ens_copy = ensemble_sw.copy()
ens_copy['month'] = ens_copy['date'].dt.to_period('M')
ml_monthly = ens_copy.groupby(['ticker', 'month'])['y_pred_ensemble'].last().reset_index()
ml_monthly['rebalance_date'] = ml_monthly['month'].map(month_to_eom)
ml_monthly = ml_monthly.dropna(subset=['rebalance_date'])
ml_pred_pivot = ml_monthly.pivot_table(
    index='rebalance_date', columns='ticker', values='y_pred_ensemble'
)
print(f'  ml_pred_pivot: {ml_pred_pivot.shape}')

# Dynamic-Membership (Step 7)
membership = get_or_build_membership(
    start=pd.Timestamp('2008-12-01'),
    end=pd.Timestamp('2026-01-01'),
    cache_path=DATA_DIR / 'sp500_membership.pkl',
)
print(f'  membership: {len(membership)} 월말 시점')

# Phase 3-2 v2 경로
OUT_DIR_V2_SW = OUTPUTS_DIR / '02a_v2_stockwise'
OUT_DIR_V2_SW.mkdir(parents=True, exist_ok=True)
CACHE_PATH_V2 = DATA_DIR / 'bl_weights_v2_sanity_check.pkl'
METRICS_PKL_V2 = DATA_DIR / 'bl_metrics_v2_sanity_check.pkl'
print(f'  v2 BL 캐시: {CACHE_PATH_V2.name}')

t1 = time.time()
print(f'\n[ §6-1 완료, {t1-t0:.1f}s ]')
print()

# ============================================================
# Cell 23 §6-2: BL 백테스트 walk-forward (6 시나리오)
# ============================================================
print('=' * 70)
print('  §6-2: BL backtest walk-forward (6 시나리오 × 192 개월)')
print('=' * 70)

DAYS_IS = 1260
MIN_UNIVERSE = 30
MIN_VALID_TIX = 20
STALE_RATIO_THRESHOLD = 0.30

CACHE_PATH = CACHE_PATH_V2
FORCE_RECOMPUTE = True   # ⭐ standalone runner 는 항상 재계산 (목적 그대로)

# 6 시나리오 dict 초기화
WEIGHTINGS = ['mcap', 'eq', 'rp']
weights = {f'ml_sw_{w}': {} for w in WEIGHTINGS}
weights.update({f'trailing_{w}': {} for w in WEIGHTINGS})
n_skip = 0
n_sigma_fail = 0

# Progress 표시 (10시점마다 print)
N_TOTAL = len(reb_dates)
PROGRESS_EVERY = 10
print(f'  총 {N_TOTAL} 시점 처리 시작...')
print()

t_loop_start = time.time()

for idx, reb_date in enumerate(reb_dates):
    # ─── universe 결정 ───
    panel_at = panel[panel['date'] == reb_date].dropna(
        subset=['vol_21d', 'mcap_value', 'log_ret']
    )
    month_end_key = pd.Timestamp(reb_date.to_period('M').to_timestamp(how='end').normalize())
    members_at_date = membership.get(month_end_key, frozenset())
    is_window = daily_lr.loc[reb_date - pd.offsets.BDay(1260):reb_date]
    zero_ratio = (is_window == 0).mean()
    non_stale = set(zero_ratio[zero_ratio <= STALE_RATIO_THRESHOLD].index)
    avail = members_at_date & set(panel_at['ticker']) & trained_tickers & non_stale
    if len(avail) < MIN_UNIVERSE:
        n_skip += 1
        continue

    # ─── IS 데이터 + 60% threshold ───
    is_end = reb_date
    is_start = is_end - pd.offsets.BDay(DAYS_IS)
    is_data = daily_lr.loc[is_start:is_end, :]
    cols_in_data = [t for t in avail if t in is_data.columns]
    valid_tix = [t for t in cols_in_data
                  if is_data[t].notna().sum() >= int(DAYS_IS * 0.7)]
    if len(valid_tix) < MIN_VALID_TIX:
        n_skip += 1
        continue

    # ─── Σ 추정 ───
    try:
        Sigma = estimate_covariance(
            is_data[valid_tix].fillna(0),
            is_start=is_start,
            is_end=is_end,
        )
    except Exception as e:
        if n_sigma_fail < 3:
            print(f'  [{reb_date.date()}] Sigma 실패: {type(e).__name__}: {str(e)[:80]}')
        n_sigma_fail += 1
        continue

    panel_idx = panel_at.set_index('ticker')
    mcap = panel_idx['mcap_value'].reindex(Sigma.index).dropna()
    common = list(mcap.index)
    if len(common) < MIN_VALID_TIX:
        n_skip += 1
        continue

    Sigma_c = Sigma.loc[common, common]
    mcap_c = mcap[common]
    w_mkt = mcap_c / mcap_c.sum()

    pi, lam = compute_pi(Sigma_c, w_mkt, spy_excess_monthly, spy_sigma2_monthly,
                          lam_fixed=LAM_FIXED)

    # 시나리오 1: BL_ml_sw - 3 weighting
    if reb_date in ml_pred_pivot.index:
        vol_ml = ml_pred_pivot.loc[reb_date].reindex(common).dropna()
        vol_ml_actual = np.exp(vol_ml)
        if len(vol_ml_actual) >= 5:
            for w_method in WEIGHTINGS:
                P_ml = build_P(vol_ml_actual, mcap_c[vol_ml_actual.index],
                                pct=PCT_GROUP, weighting=w_method)
                P_ml = P_ml.reindex(common).fillna(0)
                omega_ml = compute_omega(P_ml, Sigma_c, DEFAULT_TAU)
                mu_ml = black_litterman(pi, Sigma_c, P_ml, q=Q_FIXED,
                                         omega=omega_ml, tau=DEFAULT_TAU)
                w_ml = optimize_portfolio(mu_ml, Sigma_c, lam)
                weights[f'ml_sw_{w_method}'][reb_date] = w_ml

    # 시나리오 2: BL_trailing - 3 weighting
    vol_t = panel_idx['vol_21d'].reindex(common).dropna()
    if len(vol_t) >= 5:
        for w_method in WEIGHTINGS:
            P_t = build_P(vol_t, mcap_c[vol_t.index],
                           pct=PCT_GROUP, weighting=w_method)
            P_t = P_t.reindex(common).fillna(0)
            omega_t = compute_omega(P_t, Sigma_c, DEFAULT_TAU)
            mu_t = black_litterman(pi, Sigma_c, P_t, q=Q_FIXED,
                                    omega=omega_t, tau=DEFAULT_TAU)
            w_t = optimize_portfolio(mu_t, Sigma_c, lam)
            weights[f'trailing_{w_method}'][reb_date] = w_t

    # 진행률 표시
    if (idx + 1) % PROGRESS_EVERY == 0 or (idx + 1) == N_TOTAL:
        elapsed = time.time() - t_loop_start
        per = elapsed / (idx + 1)
        eta = per * (N_TOTAL - idx - 1)
        print(f'  [{idx+1:3d}/{N_TOTAL}] {reb_date.date()} | '
              f'평균 {per:.1f}s/시점 | 경과 {elapsed:.0f}s | ETA {eta:.0f}s '
              f'({eta/60:.1f} min)', flush=True)

t_loop_end = time.time()
print()
print(f'[ BL 루프 완료, 총 {t_loop_end - t_loop_start:.1f}s ({(t_loop_end - t_loop_start)/60:.1f} min) ]')

# ============================================================
# 캐시 저장
# ============================================================
print()
print('=' * 70)
print('  캐시 저장')
print('=' * 70)

with open(CACHE_PATH, 'wb') as f:
    pickle.dump({
        'weights': weights,
        'n_skip': n_skip,
        'n_sigma_fail': n_sigma_fail,
    }, f)

print(f'  결과 (Phase 3-2 v2):')
for s_name, w_dict in weights.items():
    print(f'    BL_{s_name} 가중치: {len(w_dict)} 시점')
print(f'  Skip (universe 부족): {n_skip}')
print(f'  Σ 추정 실패: {n_sigma_fail}')
print(f'  💾 캐시 저장: {CACHE_PATH}')
print(f'    파일 크기: {CACHE_PATH.stat().st_size / 1e6:.2f} MB')

t_end = time.time()
print()
print('=' * 70)
print(f'  ALL DONE — total {t_end - t0:.1f}s ({(t_end - t0)/60:.1f} min)')
print('=' * 70)
