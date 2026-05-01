"""
05a_v2_lstm.ipynb 빠른 검증 스크립트.
환경 부트스트랩 + 데이터 로드 + 가벼운 분석 1-2개 시뮬레이션.
"""
import sys, io, time
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
font_used = bootstrap()

OUT_DIR = OUTPUTS_DIR / '05a_v2_lstm_diag'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'OUT_DIR: {OUT_DIR}')

# ─── 데이터 로드 ───
t0 = time.time()
ens_sw = pd.read_csv(DATA_DIR / 'ensemble_predictions_stockwise.csv', parse_dates=['date'])
print(f'ensemble_sw: {ens_sw.shape} ({time.time()-t0:.1f}s)')

# inf 제거
n_before = len(ens_sw)
inf_mask = np.isfinite(ens_sw['y_true'])
ens_sw = ens_sw[inf_mask].copy()
print(f'  inf 제거: {n_before - len(ens_sw)} 행')

# year-month + period
ens_sw['ym'] = ens_sw['date'].dt.to_period('M')

def assign_period(d):
    if d < pd.Timestamp('2015-01-01'): return 'P1 (2010-2014)'
    elif d < pd.Timestamp('2019-01-01'): return 'P2 (2015-2018)'
    elif d < pd.Timestamp('2021-01-01'): return 'P3 (2019-2020)'
    elif d < pd.Timestamp('2023-01-01'): return 'P4 (2021-2022)'
    else: return 'P5 (2023-2025)'
ens_sw['period'] = ens_sw['date'].apply(assign_period)
PERIOD_ORDER = ['P1 (2010-2014)', 'P2 (2015-2018)', 'P3 (2019-2020)',
                'P4 (2021-2022)', 'P5 (2023-2025)']

# loss diff
ens_sw['err_lstm'] = ens_sw['y_pred_lstm'] - ens_sw['y_true']
ens_sw['err_har'] = ens_sw['y_pred_har'] - ens_sw['y_true']
ens_sw['err_ens'] = ens_sw['y_pred_ensemble'] - ens_sw['y_true']
ens_sw['sq_lstm'] = ens_sw['err_lstm'] ** 2
ens_sw['sq_har'] = ens_sw['err_har'] ** 2
ens_sw['sq_ens'] = ens_sw['err_ens'] ** 2

# vix
vix = pd.read_csv(DATA_DIR / 'vix_daily.csv', parse_dates=['date']).set_index('date')['VIX']
print(f'vix: {len(vix)} days')

# ─── 분석 A: 월별 RMSE ───
print()
print('─── 분석 A: 월별 RMSE ───')
t1 = time.time()
monthly_rmse = ens_sw.groupby('ym').agg(
    rmse_lstm=('sq_lstm', lambda x: np.sqrt(x.mean())),
    rmse_har=('sq_har', lambda x: np.sqrt(x.mean())),
    rmse_ens=('sq_ens', lambda x: np.sqrt(x.mean())),
    n=('y_true', 'count'),
)
print(f'  monthly_rmse: {monthly_rmse.shape} ({time.time()-t1:.1f}s)')
print(monthly_rmse.head(3).round(4).to_string())

# ─── 분석 B: 종목 × 시기 RMSE ───
print()
print('─── 분석 B: 종목 × 시기 RMSE ───')
t1 = time.time()
ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']
stock_period_rmse = ens_oos.groupby(['ticker', 'period'])['sq_ens'].apply(
    lambda x: np.sqrt(x.mean())
).unstack('period').reindex(columns=PERIOD_ORDER)
print(f'  shape: {stock_period_rmse.shape} ({time.time()-t1:.1f}s)')
print(stock_period_rmse.head(3).round(4).to_string())

# ─── 분석 D: vol regime ───
print()
print('─── 분석 D: vol regime ───')
t1 = time.time()
def quintile_label(g):
    g = g.copy()
    g['vol_q'] = pd.qcut(g['y_true'], 5, labels=['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high'],
                         duplicates='drop')
    return g

ens_oos_q = ens_oos.groupby('ticker', group_keys=False).apply(quintile_label)
regime_rmse = ens_oos_q.groupby('vol_q', observed=True).agg(
    rmse_lstm=('sq_lstm', lambda x: np.sqrt(x.mean())),
    rmse_har=('sq_har', lambda x: np.sqrt(x.mean())),
    rmse_ens=('sq_ens', lambda x: np.sqrt(x.mean())),
    n=('y_true', 'count'),
)
print(f'  vol quintile별 RMSE ({time.time()-t1:.1f}s):')
print(regime_rmse.round(4).to_string())

# ─── 분석 E: VIX tier ───
print()
print('─── 분석 E: VIX tier ───')
t1 = time.time()
vix_oos = vix.loc['2010-01-01':'2025-12-31']
vix_t1, vix_t2 = vix_oos.quantile([0.33, 0.67]).values
print(f'  VIX tertile 기준: low<={vix_t1:.2f} <mid<= {vix_t2:.2f} <high')

ens_oos2 = ens_oos.copy()
def vix_tier(d):
    v = vix_oos.get(d, np.nan)
    if pd.isna(v): return 'Unknown'
    if v <= vix_t1: return 'Low'
    elif v <= vix_t2: return 'Mid'
    else: return 'High'

ens_oos2['vix_tier'] = ens_oos2['date'].apply(vix_tier)
ens_oos2 = ens_oos2[ens_oos2['vix_tier'] != 'Unknown']
vix_rmse = ens_oos2.groupby('vix_tier', observed=True).agg(
    rmse_lstm=('sq_lstm', lambda x: np.sqrt(x.mean())),
    rmse_har=('sq_har', lambda x: np.sqrt(x.mean())),
    rmse_ens=('sq_ens', lambda x: np.sqrt(x.mean())),
    n=('y_true', 'count'),
).reindex(['Low', 'Mid', 'High'])
print(f'  VIX tier별 RMSE ({time.time()-t1:.1f}s):')
print(vix_rmse.round(4).to_string())

# ─── 분석 F: DM test (sample) ───
print()
print('─── 분석 F: DM test (sample 10 ticker) ───')
t1 = time.time()
from scipy.stats import norm

def dm_test(loss_lstm, loss_har, h=21):
    d = loss_lstm - loss_har
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2 * h:
        return float('nan'), float('nan'), n
    d_mean = d.mean()
    gamma_0 = ((d - d_mean) ** 2).mean()
    s_var = gamma_0
    for k in range(1, h):
        if k >= n: break
        gamma_k = ((d[:-k] - d_mean) * (d[k:] - d_mean)).mean()
        s_var += 2 * (1 - k/h) * gamma_k
    if s_var <= 0:
        return float('nan'), float('nan'), n
    dm_stat = d_mean / np.sqrt(s_var / n)
    p = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p, n

sample_tickers = sorted(ens_oos['ticker'].unique())[:10]
for t in sample_tickers:
    g = ens_oos[ens_oos['ticker'] == t].sort_values('date')
    dm, p, n = dm_test(g['sq_lstm'].values, g['sq_har'].values, h=21)
    print(f'  {t:6s}: DM={dm:+.3f}, p={p:.4f}, n={n}')
print(f'  ({time.time()-t1:.1f}s for 10 tickers)')

print()
print('=' * 75)
print('  ✅ 핵심 분석 시뮬레이션 통과 — 노트북 실행 가능')
print('=' * 75)
print(f'\n전체 시간: {time.time()-t0:.1f}s')
