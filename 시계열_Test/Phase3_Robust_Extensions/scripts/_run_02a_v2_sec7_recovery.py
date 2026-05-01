"""
Phase 3-2 v2 §7-K. V자 반등 분석 standalone runner.

목적:
- BL_trailing_mcap 의 우위 (CAGR 14.34% / Sharpe 1.206) 가 대형주 V자 반등 덕분인지 검증
- BL_ml_sw_mcap 의 약점 (CAGR 12.84% / MDD -18.13%) 이 V자 반등을 못 따라간 결과인지 분석

분석:
A. Recovery 능력: drawdown trough 후 6개월 누적 수익률 (event 별, 시나리오 비교)
B. Top 10 holdings 비교: 충격 직전 (2020-01) vs 반등 후 (2020-05)
C. 대형주 비중 시계열: portfolio 가중치 중 universe top 50 mcap 종목 차지 비율
D. 변동성 예측 비교: 특정 event 직전 ML vs trailing 의 vol prediction 차이

결과: data/sec7_k_recovery_analysis.pkl
"""
from __future__ import annotations

import io
import sys
import time
import pickle
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
NB_DIR = SCRIPT_DIR.parent
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR

t0 = time.time()
print('=' * 70)
print('  Phase 3-2 v2 §7-K. V자 반등 분석 standalone runner')
print('=' * 70)
bootstrap()
print()

# ============================================================
# 0. 데이터 로드
# ============================================================
print('[로드] BL 가중치 + ensemble + panel + market')

with open(DATA_DIR / 'bl_weights_v2_sanity_check.pkl', 'rb') as f:
    bl_cache = pickle.load(f)
weights_v2 = bl_cache['weights']

ens = pd.read_csv(
    DATA_DIR / 'ensemble_predictions_stockwise.csv',
    parse_dates=['date'],
    usecols=['date', 'ticker', 'y_pred_ensemble'],
)
panel = pd.read_csv(
    DATA_DIR / 'daily_panel.csv',
    parse_dates=['date'],
    usecols=['date', 'ticker', 'log_ret', 'vol_21d', 'mcap_value'],
)
market = pd.read_csv(DATA_DIR / 'market_data.csv', index_col='date', parse_dates=True)
print(f'  panel: {panel.shape}, ensemble: {ens.shape}')

# 시점 정의 + reb_dates
OOS_START = '2010-01-01'; OOS_END = '2024-12-31'
HOLDOUT_START = '2025-01-01'; HOLDOUT_END = '2025-12-31'
reb_dates_all = market.groupby(market.index.to_period('M')).tail(1).index
oos_dates = reb_dates_all[(reb_dates_all >= OOS_START) & (reb_dates_all <= OOS_END)]
holdout_dates = reb_dates_all[(reb_dates_all >= HOLDOUT_START) & (reb_dates_all <= HOLDOUT_END)]
reb_dates = pd.DatetimeIndex(list(oos_dates) + list(holdout_dates))
month_to_eom = {pd.Timestamp(d).to_period('M'): pd.Timestamp(d) for d in reb_dates}

# 월별 수익률 + forward
def compute_monthly_returns_sw(panel_df, tickers, month_to_eom):
    sub = panel_df[panel_df['ticker'].isin(tickers)].set_index('date')
    sub['month'] = sub.index.to_period('M')
    monthly_lr = sub.groupby(['ticker', 'month'])['log_ret'].sum().reset_index()
    monthly_lr['date'] = monthly_lr['month'].map(month_to_eom)
    monthly_lr = monthly_lr.dropna(subset=['date'])
    monthly_lr['ret'] = np.exp(monthly_lr['log_ret']) - 1
    return monthly_lr.pivot_table(index='date', columns='ticker', values='ret')


def make_returns_manual(weights_dict, name, forward_rets):
    if not weights_dict:
        return pd.Series(dtype=float, name=name)
    rets, dates = [], []
    for reb_date in sorted(weights_dict.keys()):
        if reb_date not in forward_rets.index:
            continue
        w = weights_dict[reb_date]
        r_next = forward_rets.loc[reb_date]
        common_t = w.index.intersection(r_next.index)
        if len(common_t) == 0:
            continue
        rets.append(float((w.reindex(common_t).fillna(0) * r_next.reindex(common_t).fillna(0)).sum()))
        dates.append(reb_date)
    return pd.Series(rets, index=pd.DatetimeIndex(dates), name=name)


all_tickers = sorted(set().union(*[set(w.index) for d in weights_v2.values() for w in d.values()]))
monthly_rets = compute_monthly_returns_sw(panel, all_tickers, month_to_eom)
forward_rets = monthly_rets.shift(-1)

returns_v2 = {f'BL_{s}': make_returns_manual(w_dict, f'BL_{s}', forward_rets)
              for s, w_dict in weights_v2.items()}

# SPY
spy_daily = market['SPY'].pct_change().dropna()
spy_monthly = (1 + spy_daily).resample('ME').prod() - 1
spy_at_eom = spy_monthly.reindex(reb_dates, method='nearest')
returns_v2['SPY'] = spy_at_eom.shift(-1).dropna().rename('SPY')

# 공통 기간
common_idx = None
for r in returns_v2.values():
    common_idx = r.index if common_idx is None else common_idx.intersection(r.index)
returns_v2 = {n: r.reindex(common_idx).dropna() for n, r in returns_v2.items()}
print(f'  공통 기간: {common_idx[0].date()} ~ {common_idx[-1].date()} ({len(common_idx)} 개월)')

# mcap_pivot
mcap_pivot = panel.pivot_table(index='date', columns='ticker', values='mcap_value')
mcap_pivot_reb = mcap_pivot.reindex(reb_dates).ffill()
print()


# ============================================================
# A. Recovery 능력 (drawdown trough 후 6개월 누적 수익률)
# ============================================================
print('=' * 70)
print('  A. Recovery 능력 (drawdown trough 후 6개월 누적)')
print('=' * 70)


def identify_dd_events_with_recovery(returns, threshold=0.05, recovery_window_m=6):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1)

    events = []
    in_dd = False
    start = trough = None
    trough_dd = 0
    for date, val in dd.items():
        if not in_dd and val < -threshold:
            in_dd = True
            start = trough = date
            trough_dd = val
        elif in_dd:
            if val < trough_dd:
                trough = date
                trough_dd = val
            if val >= 0:
                # recovery 시점 + 6개월 누적
                trough_idx = list(returns.index).index(trough)
                end_idx = min(trough_idx + recovery_window_m, len(returns) - 1)
                recov_rets = returns.iloc[trough_idx:end_idx + 1]
                cum_6m = (1 + recov_rets).prod() - 1
                events.append({
                    'start': start.strftime('%Y-%m'),
                    'trough': trough.strftime('%Y-%m'),
                    'recovery': date.strftime('%Y-%m'),
                    'depth_%': trough_dd * 100,
                    'recov_6m_%': cum_6m * 100,
                    'recovery_m': (date - trough).days // 30,
                })
                in_dd = False
    return events


print()
print(f'{"시나리오":<25s} {"# events":>10s} {"avg recov_6m_%":>15s} {"max recov_6m_%":>15s}')
print('-' * 70)
recovery_results = {}
for name, r in returns_v2.items():
    events = identify_dd_events_with_recovery(r, threshold=0.05, recovery_window_m=6)
    recovery_results[name] = events
    if events:
        avg_recov = np.mean([e['recov_6m_%'] for e in events])
        max_recov = max([e['recov_6m_%'] for e in events])
        print(f'{name:<25s} {len(events):>10d} {avg_recov:>15.2f} {max_recov:>15.2f}')
    else:
        print(f'{name:<25s} {0:>10d}')

print()
print('  === 주요 V자 반등 event (2020-COVID 등): 6개월 누적 수익률 비교 ===')
target_events = [('2020-01', '2020-02')]  # COVID 충격 시작 / trough
for ev_start, ev_trough in target_events:
    print(f'\n  ⭐ Event: start={ev_start}, trough={ev_trough}')
    for name, events in recovery_results.items():
        for e in events:
            if e['start'].startswith(ev_start) or e['trough'].startswith(ev_trough):
                print(f'    {name:<25s} depth {e["depth_%"]:+.2f}% / recov_6m {e["recov_6m_%"]:+.2f}%')


# ============================================================
# B. Top 10 holdings 비교 (충격 직전 vs 반등 후)
# ============================================================
print()
print('=' * 70)
print('  B. Top 10 holdings 비교 (특정 시기, 두 mcap 시나리오)')
print('=' * 70)

KEY_DATES = {
    '2020-01': '2020-01-31',  # 충격 직전
    '2020-05': '2020-05-29',  # 반등 후
    '2022-08': '2022-08-31',  # 긴축기 trough
    '2023-12': '2023-12-29',  # 회복기
    '2025-04': '2025-04-30',  # Hold-out 트럼프 관세 충격 직후
}

holdings_comparison = {}
for label, date_str in KEY_DATES.items():
    # 가장 가까운 reb_date 찾기
    target = pd.Timestamp(date_str)
    # actual reb_date in weights
    avail_dates_ml = sorted(weights_v2['ml_sw_mcap'].keys())
    reb_d = min(avail_dates_ml, key=lambda d: abs((d - target).days))

    print(f'\n  --- {label} (실제 reb_date: {reb_d.date()}) ---')
    holdings_comparison[label] = {'reb_date': reb_d.strftime('%Y-%m-%d'), 'top10': {}}
    for s_name in ['ml_sw_mcap', 'trailing_mcap']:
        w = weights_v2[s_name].get(reb_d)
        if w is None:
            continue
        top10 = w.sort_values(ascending=False).head(10)
        holdings_comparison[label]['top10'][f'BL_{s_name}'] = {
            'tickers': top10.index.tolist(),
            'weights_%': (top10.values * 100).round(2).tolist(),
            'top10_share_%': float(top10.sum() * 100),
            'n_total_long': int((w > 0).sum()),
        }
        print(f'    [BL_{s_name}] top10 share: {top10.sum()*100:.1f}% / total long: {(w>0).sum()}')
        for t, wv in zip(top10.index, top10.values):
            print(f'      {t:6s} {wv*100:5.2f}%')


# ============================================================
# C. 대형주 비중 시계열 (top 50 mcap)
# ============================================================
print()
print('=' * 70)
print('  C. 대형주 비중 시계열 (universe top 50 mcap 차지 비율)')
print('=' * 70)


def compute_largecap_share(weights_dict, mcap_pivot_local, top_n=50):
    results = {}
    for date in sorted(weights_dict.keys()):
        if date not in mcap_pivot_local.index:
            continue
        mcap = mcap_pivot_local.loc[date].dropna()
        top_mcap_tickers = mcap.nlargest(top_n).index
        w = weights_dict[date]
        common = w.index.intersection(top_mcap_tickers)
        share = float(w.loc[common].sum()) if len(common) > 0 else 0.0
        results[date] = share
    return pd.Series(results)


largecap_share = {}
for s_name in ['ml_sw_mcap', 'trailing_mcap', 'ml_sw_eq', 'trailing_eq']:
    largecap_share[f'BL_{s_name}'] = compute_largecap_share(weights_v2[s_name], mcap_pivot_reb, top_n=50)

print()
print(f'{"시나리오":<25s} {"평균 top50 share":>20s} {"std":>10s} {"min":>10s} {"max":>10s}')
print('-' * 80)
for name, ts in largecap_share.items():
    print(f'{name:<25s} {ts.mean()*100:>18.2f}% {ts.std()*100:>9.2f}% {ts.min()*100:>9.2f}% {ts.max()*100:>9.2f}%')

# 시기별 평균
print()
print('  === 시기별 평균 대형주 (top 50) 비중 ===')
PERIODS_LC = {
    'Pre-COVID (2018-2019)': ('2018-01', '2019-12'),
    'COVID 회복 (2020)':       ('2020-01', '2020-12'),
    '긴축 (2021-2022)':        ('2021-01', '2022-12'),
    '회복·AI (2023-2024)':     ('2023-01', '2024-12'),
    'Hold-out (2025)':         ('2025-01', '2025-12'),
}
period_largecap = {}
for pname, (start, end) in PERIODS_LC.items():
    print(f'  [{pname}]')
    period_largecap[pname] = {}
    for name, ts in largecap_share.items():
        sub = ts.loc[start:end]
        avg = sub.mean() * 100 if len(sub) > 0 else np.nan
        period_largecap[pname][name] = avg
        print(f'    {name:<25s} top50 share avg: {avg:.2f}%')


# ============================================================
# D. 변동성 예측 비교 (특정 event 직전)
# ============================================================
print()
print('=' * 70)
print('  D. 변동성 예측 비교 (ML vs trailing, 특정 event)')
print('=' * 70)

# ml_pred_pivot 재구성
ens_copy = ens.copy()
ens_copy['month'] = ens_copy['date'].dt.to_period('M')
ml_monthly = ens_copy.groupby(['ticker', 'month'])['y_pred_ensemble'].last().reset_index()
ml_monthly['rebalance_date'] = ml_monthly['month'].map(month_to_eom)
ml_monthly = ml_monthly.dropna(subset=['rebalance_date'])
ml_pred_pivot = ml_monthly.pivot_table(index='rebalance_date', columns='ticker', values='y_pred_ensemble')

# 주요 event 시점에서 ML vs trailing 의 vol 분포 비교
EVENT_DATES = {
    '2020-01-31 (COVID 직전)': '2020-01-31',
    '2020-02-28 (COVID 충격)': '2020-02-28',
    '2020-04-30 (반등 시작)':   '2020-04-30',
    '2022-08-31 (긴축 trough)':  '2022-08-31',
    '2025-04-30 (관세 충격)':    '2025-04-30',
}

vol_comparison = {}
for label, date_str in EVENT_DATES.items():
    target = pd.Timestamp(date_str)
    avail = sorted(ml_pred_pivot.index.intersection(set(reb_dates)))
    if not avail:
        continue
    reb_d = min(avail, key=lambda d: abs((d - target).days))

    # ML pred (log → exp 변환 → 실제 σ scale)
    ml_pred = np.exp(ml_pred_pivot.loc[reb_d]).dropna()
    # Trailing vol_21d
    panel_at = panel[panel['date'] == reb_d].set_index('ticker')
    if 'vol_21d' not in panel_at.columns:
        continue
    tr_vol = panel_at['vol_21d'].dropna()
    common = ml_pred.index.intersection(tr_vol.index)
    if len(common) < 20:
        continue
    ml_v = ml_pred.loc[common]
    tr_v = tr_vol.loc[common]

    # rank correlation (Spearman) — 두 vol의 ranking 일치도
    from scipy import stats as sp_stats
    rank_corr = sp_stats.spearmanr(ml_v, tr_v).correlation
    pearson_corr = ml_v.corr(tr_v)

    # 양극단 30% 그룹 일치도 (Hit rate)
    n_group = max(1, int(len(common) * 0.30))
    ml_low = set(ml_v.nsmallest(n_group).index)
    ml_high = set(ml_v.nlargest(n_group).index)
    tr_low = set(tr_v.nsmallest(n_group).index)
    tr_high = set(tr_v.nlargest(n_group).index)
    low_overlap = len(ml_low & tr_low) / n_group * 100
    high_overlap = len(ml_high & tr_high) / n_group * 100

    vol_comparison[label] = {
        'reb_date': reb_d.strftime('%Y-%m-%d'),
        'n_common': len(common),
        'ml_vol_mean': float(ml_v.mean()),
        'tr_vol_mean': float(tr_v.mean()),
        'ml_vol_std':  float(ml_v.std()),
        'tr_vol_std':  float(tr_v.std()),
        'rank_correlation': float(rank_corr),
        'pearson_corr':     float(pearson_corr),
        'low_overlap_%':  float(low_overlap),
        'high_overlap_%': float(high_overlap),
    }

print()
print(f'{"Event":<35s} {"reb_date":<12s} {"rank_corr":>10s} {"low_overlap":>13s} {"high_overlap":>15s}')
print('-' * 100)
for label, d in vol_comparison.items():
    print(f'{label:<35s} {d["reb_date"]:<12s} {d["rank_correlation"]:>10.3f} '
          f'{d["low_overlap_%"]:>12.1f}% {d["high_overlap_%"]:>14.1f}%')

print()
print('  📌 해석:')
print('    - rank_correlation ~ 1: ML 과 trailing 의 vol ranking 거의 일치 (ML 의 추가 정보 적음)')
print('    - low_overlap, high_overlap < 70%: ML 과 trailing 이 다른 종목 선택 (ML 의 차별화 영향)')


# ============================================================
# 결과 저장
# ============================================================
print()
print('=' * 70)
print('  결과 저장')
print('=' * 70)

results = {
    'recovery_results': recovery_results,
    'holdings_comparison': holdings_comparison,
    'largecap_share_timeseries': {n: ts.to_dict() for n, ts in largecap_share.items()},
    'period_largecap': period_largecap,
    'vol_comparison': vol_comparison,
}
out_path = DATA_DIR / 'sec7_k_recovery_analysis.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f'  💾 저장: {out_path.name} ({out_path.stat().st_size/1e3:.1f} KB)')

t_end = time.time()
print()
print('=' * 70)
print(f'  ALL DONE — total {t_end - t0:.1f}s ({(t_end - t0)/60:.1f} min)')
print('=' * 70)
