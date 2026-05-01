"""
05a_v2_weighting.ipynb 검증 스크립트.
환경 + 9 returns 로드 + Layer 2 + ML 효과 분해 시뮬레이션.
"""
import sys, io, time, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR
import scripts.diagnostics as diag

font_used = bootstrap()

OUT_DIR = OUTPUTS_DIR / '05a_v2_weighting'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'OUT_DIR: {OUT_DIR}')

OOS_START, OOS_END = '2010-01-01', '2024-12-31'
HOLD_START, HOLD_END = '2025-01-01', '2025-12-31'

# ─── 9 returns 로드 ───
t0 = time.time()
BL_DIR = OUTPUTS_DIR / '03_bl_backtest'
scenarios = [
    'BL_ml_sw_mcap', 'BL_ml_sw_eq', 'BL_ml_sw_rp',
    'BL_trailing_mcap', 'BL_trailing_eq', 'BL_trailing_rp',
    'BL_ml_cs', 'EqualWeight', 'McapWeight', 'SPY',
]
returns_dict = {}
for sc in scenarios:
    p = BL_DIR / f'returns_{sc}.csv'
    if p.exists():
        ret = pd.read_csv(p, index_col=0, parse_dates=True).squeeze()
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        returns_dict[sc] = ret

print(f'returns 로드: {len(returns_dict)} / {len(scenarios)} ({time.time()-t0:.1f}s)')
for sc, ret in returns_dict.items():
    print(f'  {sc}: {len(ret)} 개월, {ret.index.min().date()} ~ {ret.index.max().date()}')

# ─── spy_monthly ───
market = pd.read_csv(DATA_DIR / 'market_data.csv', index_col='date', parse_dates=True)
_reb_all = market.groupby(market.index.to_period('M')).tail(1).index
oos_dates = _reb_all[(_reb_all >= OOS_START) & (_reb_all <= OOS_END)]
holdout_dates = _reb_all[(_reb_all >= HOLD_START) & (_reb_all <= HOLD_END)]
all_reb_dates = pd.DatetimeIndex(list(oos_dates) + list(holdout_dates))
spy_monthly = market['SPY'].reindex(all_reb_dates, method='ffill').pct_change().dropna()
print(f'\nspy_monthly: {len(spy_monthly)} 개월')

# ─── §2 Layer 2 메트릭 ───
print()
print('─── §2 Layer 2 메트릭 (6 시나리오 + 3 baseline) ───')
KEY_METRICS = ['sharpe', 'cagr', 'ann_vol', 'mdd', 'capm_alpha',
               'capm_beta', 'information_ratio', 'sortino', 'hit_rate', 'cvar_5']

def calc_metrics(rets, name, period_start=None, period_end=None):
    if period_start and period_end:
        rets = rets.loc[period_start:period_end]
    if len(rets) == 0:
        return {k: np.nan for k in KEY_METRICS}
    m = diag.evaluate_portfolio_standalone(
        returns=rets, scenario_name=name,
        spy_returns=spy_monthly, rf_returns=None, weights_dict=None,
    )
    return {k: m.get(k, np.nan) for k in KEY_METRICS}

t1 = time.time()
all_scenarios = [
    'BL_ml_sw_mcap', 'BL_ml_sw_eq', 'BL_ml_sw_rp',
    'BL_trailing_mcap', 'BL_trailing_eq', 'BL_trailing_rp',
    'EqualWeight', 'McapWeight', 'SPY',
]
metrics_full = {}
metrics_oos = {}
metrics_hold = {}
for sc in all_scenarios:
    if sc not in returns_dict:
        continue
    rets = returns_dict[sc]
    metrics_full[sc] = calc_metrics(rets, sc)
    metrics_oos[sc] = calc_metrics(rets, sc + ' (OOS)', OOS_START, OOS_END)
    metrics_hold[sc] = calc_metrics(rets, sc + ' (Hold-out)', HOLD_START, HOLD_END)

print(f'  ({time.time()-t1:.1f}s 소요)')

mf_df = pd.DataFrame(metrics_full).T.round(3)
print()
print('=== 전체 (2010-2025) ===')
print(mf_df[['sharpe', 'cagr', 'mdd', 'capm_alpha']].to_string())

mo_df = pd.DataFrame(metrics_oos).T.round(3)
print()
print('=== OOS (2010-2024) ===')
print(mo_df[['sharpe', 'cagr', 'mdd', 'capm_alpha']].to_string())

mh_df = pd.DataFrame(metrics_hold).T.round(3)
print()
print('=== Hold-out (2025) ===')
print(mh_df[['sharpe', 'cagr', 'mdd', 'capm_alpha']].to_string())

# ─── §4 ML 효과 분해 ───
print()
print('─── §4 ML 효과 분해 (가중치별 robustness) ───')

ml_effect_full = {}
ml_effect_oos = {}
ml_effect_hold = {}
for w in ['mcap', 'eq', 'rp']:
    ml_sc = f'BL_ml_sw_{w}'
    tr_sc = f'BL_trailing_{w}'
    diffs_f = {k: metrics_full[ml_sc][k] - metrics_full[tr_sc][k] for k in KEY_METRICS}
    diffs_o = {k: metrics_oos[ml_sc][k] - metrics_oos[tr_sc][k] for k in KEY_METRICS}
    diffs_h = {k: metrics_hold[ml_sc][k] - metrics_hold[tr_sc][k] for k in KEY_METRICS}
    ml_effect_full[w] = diffs_f
    ml_effect_oos[w] = diffs_o
    ml_effect_hold[w] = diffs_h

mef_df = pd.DataFrame(ml_effect_full).round(3)
meo_df = pd.DataFrame(ml_effect_oos).round(3)
meh_df = pd.DataFrame(ml_effect_hold).round(3)

print()
print('=== ML 효과 (Full) ===')
print(mef_df.to_string())
print()
print('=== ML 효과 (OOS) ===')
print(meo_df.to_string())
print()
print('=== ML 효과 (Hold-out) ===')
print(meh_df.to_string())

# 학술적 함의
print()
print('=' * 75)
print('  ML 효과 robustness 자동 분석')
print('=' * 75)
sharpe_full = mef_df.loc['sharpe']
print(f'  Sharpe diff (Full):    mcap={sharpe_full["mcap"]:+.3f}, eq={sharpe_full["eq"]:+.3f}, rp={sharpe_full["rp"]:+.3f}')
sharpe_oos = meo_df.loc['sharpe']
print(f'  Sharpe diff (OOS):     mcap={sharpe_oos["mcap"]:+.3f}, eq={sharpe_oos["eq"]:+.3f}, rp={sharpe_oos["rp"]:+.3f}')

sign_consistent_full = all([sharpe_full[w] > 0 for w in ['mcap', 'eq', 'rp']]) or \
                       all([sharpe_full[w] < 0 for w in ['mcap', 'eq', 'rp']])
sign_consistent_oos = all([sharpe_oos[w] > 0 for w in ['mcap', 'eq', 'rp']]) or \
                      all([sharpe_oos[w] < 0 for w in ['mcap', 'eq', 'rp']])
print()
print(f'  부호 일관성:')
print(f'    Full: {"✅ 모든 가중치 동일 부호" if sign_consistent_full else "⚠️ 가중치별 부호 다름"}')
print(f'    OOS:  {"✅ 모든 가중치 동일 부호" if sign_consistent_oos else "⚠️ 가중치별 부호 다름"}')
print(f'  평균 Sharpe diff (Full): {sharpe_full.mean():+.3f}')
print(f'  평균 Sharpe diff (OOS):  {sharpe_oos.mean():+.3f}')

print()
print('=' * 75)
print(f'  ✅ 검증 완료 ({time.time()-t0:.1f}s)')
print('=' * 75)
