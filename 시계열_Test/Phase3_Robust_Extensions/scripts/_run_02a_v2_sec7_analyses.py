"""
Phase 3-2 v2 §7 심층 분석 standalone runner.

Heavy 분석을 노트북 외부에서 사전 계산:
- §7-B: Bootstrap CI for Sharpe ratio (5000회 × 7 시나리오)
- §7-H: Sector exposure analysis (192 시점 × 6 시나리오 × 12 sectors)
- §7-I: Market Regime (VIX quantile) analysis

결과는 pkl 캐시로 저장 → 노트북 §7 셀들은 load + 출력 (~수 초).

사용:
    python scripts/_run_02a_v2_sec7_analyses.py
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
print('  Phase 3-2 v2 §7 심층 분석 standalone runner')
print('=' * 70)
bootstrap()
print()

# ============================================================================
# 0. 공통 데이터 로드 (BL 가중치 + 메트릭 캐시)
# ============================================================================
print('[로드] BL 가중치 + 메트릭 캐시')
with open(DATA_DIR / 'bl_weights_v2_sanity_check.pkl', 'rb') as f:
    bl_cache = pickle.load(f)
weights_v2 = bl_cache['weights']
print(f'  6 시나리오 가중치 dict: {list(weights_v2.keys())}')

with open(DATA_DIR / 'bl_metrics_v2_sanity_check.pkl', 'rb') as f:
    metrics_cache = pickle.load(f)
print(f'  메트릭 키: {list(metrics_cache.keys())}')
print()

# panel + market 로드
print('[로드] panel + market')
panel = pd.read_csv(
    DATA_DIR / 'daily_panel.csv', parse_dates=['date'],
    usecols=['date', 'ticker', 'log_ret', 'mcap_value'],
)
market = pd.read_csv(DATA_DIR / 'market_data.csv', index_col='date', parse_dates=True)
spy_daily = market['SPY'].pct_change().dropna()
print(f'  panel: {panel.shape}')
print(f'  market: {market.shape}')

# Phase 3-2 시점 정의
OOS_START = '2010-01-01'
OOS_END = '2024-12-31'
HOLDOUT_START = '2025-01-01'
HOLDOUT_END = '2025-12-31'

reb_dates_all = market.groupby(market.index.to_period('M')).tail(1).index
oos_dates = reb_dates_all[(reb_dates_all >= OOS_START) & (reb_dates_all <= OOS_END)]
holdout_dates = reb_dates_all[(reb_dates_all >= HOLDOUT_START) & (reb_dates_all <= HOLDOUT_END)]
reb_dates = pd.DatetimeIndex(list(oos_dates) + list(holdout_dates))
month_to_eom = {pd.Timestamp(d).to_period('M'): pd.Timestamp(d) for d in reb_dates}

# 종목별 월별 수익률 (Cell 24 와 동일)
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
    rets = []
    dates = []
    for reb_date in sorted(weights_dict.keys()):
        w = weights_dict[reb_date]
        if reb_date not in forward_rets.index:
            continue
        r_next = forward_rets.loc[reb_date]
        common_t = w.index.intersection(r_next.index)
        if len(common_t) == 0:
            continue
        w_c = w.reindex(common_t).fillna(0)
        r_c = r_next.reindex(common_t).fillna(0)
        port_ret = float((w_c * r_c).sum())
        rets.append(port_ret)
        dates.append(reb_date)
    if not rets:
        return pd.Series(dtype=float, name=name)
    return pd.Series(rets, index=pd.DatetimeIndex(dates), name=name)


# returns_v2 재구성
print('[재구성] 6 시나리오 returns_v2')
all_tickers = set()
for s_name, w_dict in weights_v2.items():
    for w in w_dict.values():
        all_tickers.update(w.index)
all_tickers = sorted(all_tickers)
monthly_rets = compute_monthly_returns_sw(panel, all_tickers, month_to_eom)
forward_rets = monthly_rets.shift(-1)

returns_v2 = {}
for s_name, w_dict in weights_v2.items():
    returns_v2[f'BL_{s_name}'] = make_returns_manual(w_dict, f'BL_{s_name}', forward_rets)

# SPY 월별 수익률
spy_monthly = (1 + spy_daily).resample('ME').prod() - 1
spy_monthly_at_eom = spy_monthly.reindex(reb_dates, method='nearest')
ret_spy = spy_monthly_at_eom.shift(-1).dropna().rename('SPY')

all_returns_fair = {**returns_v2, 'SPY': ret_spy}
common_idx = None
for r in all_returns_fair.values():
    common_idx = r.index if common_idx is None else common_idx.intersection(r.index)
all_returns_fair = {n: r.reindex(common_idx).dropna() for n, r in all_returns_fair.items()}
print(f'  공통 기간: {common_idx[0].date()} ~ {common_idx[-1].date()} ({len(common_idx)} 개월)')
print()

# ============================================================================
# §7-B. Bootstrap Confidence Intervals for Sharpe Ratios
# ============================================================================
print('=' * 70)
print('  §7-B. Bootstrap CI for Sharpe ratios (N=5000)')
print('=' * 70)
t_b = time.time()

N_BOOTSTRAP = 5000
np.random.seed(42)


def sharpe_ratio(returns, annual_factor=12):
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(annual_factor)


def bootstrap_sharpe_ci(returns, n_bootstrap=N_BOOTSTRAP, ci_lo=2.5, ci_hi=97.5):
    n = len(returns)
    if n < 5:
        return np.nan, np.nan, np.nan
    rets_arr = returns.values
    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(rets_arr, size=n, replace=True)
        boot_sharpes[i] = (sample.mean() / sample.std() * np.sqrt(12)
                           if sample.std() > 0 else np.nan)
    boot_sharpes = boot_sharpes[~np.isnan(boot_sharpes)]
    return (
        np.percentile(boot_sharpes, ci_lo),
        np.percentile(boot_sharpes, ci_hi),
        boot_sharpes.std(),
    )


# 전체 + OOS + Hold-out 구간 모두
periods = {
    'overall_191m': (None, None),
    'oos_180m': (OOS_START, OOS_END),
    'holdout_11m': (HOLDOUT_START, HOLDOUT_END),
}

bootstrap_results = {}
for period_name, (start, end) in periods.items():
    print(f'  [{period_name}]')
    period_results = {}
    for name, r in all_returns_fair.items():
        if start is not None:
            r_p = r.loc[start:end]
        else:
            r_p = r
        sr = sharpe_ratio(r_p)
        ci_lo, ci_hi, se = bootstrap_sharpe_ci(r_p)
        period_results[name] = {
            'sharpe': sr,
            'ci_lo_2.5': ci_lo,
            'ci_hi_97.5': ci_hi,
            'std_err': se,
            'n_months': len(r_p),
        }
        print(f'    {name:25s}: Sharpe={sr:.3f}, CI=[{ci_lo:.3f}, {ci_hi:.3f}], SE={se:.3f}')
    bootstrap_results[period_name] = period_results

# ML 효과 (BL_ml_sw - BL_trailing) bootstrap CI
print()
print('  [ML 효과 bootstrap CI: BL_ml_sw - BL_trailing]')
ml_effect_results = {}
for w_method in ['mcap', 'eq', 'rp']:
    for period_name, (start, end) in periods.items():
        ml_r = all_returns_fair[f'BL_ml_sw_{w_method}']
        tr_r = all_returns_fair[f'BL_trailing_{w_method}']
        if start is not None:
            ml_r = ml_r.loc[start:end]
            tr_r = tr_r.loc[start:end]
        # paired bootstrap
        n = len(ml_r)
        diffs = np.empty(N_BOOTSTRAP)
        ml_arr = ml_r.values
        tr_arr = tr_r.values
        for i in range(N_BOOTSTRAP):
            idx = np.random.choice(n, size=n, replace=True)
            ml_s = ml_arr[idx]
            tr_s = tr_arr[idx]
            ml_sr = ml_s.mean() / ml_s.std() * np.sqrt(12) if ml_s.std() > 0 else np.nan
            tr_sr = tr_s.mean() / tr_s.std() * np.sqrt(12) if tr_s.std() > 0 else np.nan
            diffs[i] = ml_sr - tr_sr
        diffs = diffs[~np.isnan(diffs)]
        observed_diff = sharpe_ratio(ml_r) - sharpe_ratio(tr_r)
        # two-sided p-value (proportion of |bootstrap| >= |observed| under null=0)
        # We use the bootstrap distribution itself - probability that the difference is on the wrong side
        p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        key = f'{w_method}_{period_name}'
        ml_effect_results[key] = {
            'observed_diff': observed_diff,
            'ci_lo_2.5': np.percentile(diffs, 2.5),
            'ci_hi_97.5': np.percentile(diffs, 97.5),
            'std_err': diffs.std(),
            'p_value': p_value,
            'n_months': n,
        }
        print(f'    {key:25s}: ΔSharpe={observed_diff:+.3f}, CI=[{np.percentile(diffs,2.5):+.3f}, {np.percentile(diffs,97.5):+.3f}], p={p_value:.4f}')

print(f'  [§7-B 완료, {time.time()-t_b:.1f}s]')
print()

# ============================================================================
# §7-H. Sector Exposure Analysis
# ============================================================================
print('=' * 70)
print('  §7-H. Sector Exposure Analysis (6 시나리오 × 12 sectors)')
print('=' * 70)
t_h = time.time()

with open(DATA_DIR / 'sector_map_combined.pkl', 'rb') as f:
    sector_map = pickle.load(f)
print(f'  sector_map: {len(sector_map)} tickers, {len(set(sector_map.values()))} unique sectors')

# 각 시나리오 × 시점 × sector 의 평균 exposure
sector_exposure = {}
for s_name, w_dict in weights_v2.items():
    sector_at_t = []
    for reb_date in sorted(w_dict.keys()):
        w = w_dict[reb_date]
        # ticker → sector mapping
        s_weight = pd.Series(0.0, index=sorted(set(sector_map.values())))
        for ticker, weight in w.items():
            sec = sector_map.get(ticker, 'Unknown')
            s_weight[sec] = s_weight.get(sec, 0.0) + weight
        s_weight.name = reb_date
        sector_at_t.append(s_weight)
    sec_df = pd.DataFrame(sector_at_t)
    sector_exposure[f'BL_{s_name}'] = {
        'mean_exposure': sec_df.mean().to_dict(),
        'std_exposure': sec_df.std().to_dict(),
        'time_series': sec_df,  # 시기별 분석용
    }
    # 평균 sector exposure 출력
    print(f'  [BL_{s_name}] 평균 sector exposure:')
    top5 = sec_df.mean().sort_values(ascending=False).head(5)
    for sec, val in top5.items():
        print(f'    {sec:30s}: {val*100:.2f}%')

# ML vs trailing의 sector tilt 차이 (mcap, eq, rp 각각)
print()
print('  [ML vs Trailing sector tilt 차이 (각 weighting)]')
sector_tilt = {}
for w_method in ['mcap', 'eq', 'rp']:
    ml_avg = pd.Series(sector_exposure[f'BL_ml_sw_{w_method}']['mean_exposure'])
    tr_avg = pd.Series(sector_exposure[f'BL_trailing_{w_method}']['mean_exposure'])
    diff = (ml_avg - tr_avg) * 100
    sector_tilt[w_method] = diff.to_dict()
    print(f'    [{w_method}] 가장 큰 ML-trailing 차이 (절대값 top 5):')
    for sec, val in diff.abs().sort_values(ascending=False).head(5).items():
        actual = diff[sec]
        print(f'      {sec:30s}: {actual:+.2f}%p')

print(f'  [§7-H 완료, {time.time()-t_h:.1f}s]')
print()

# ============================================================================
# §7-I. Market Regime Analysis (VIX quantile)
# ============================================================================
print('=' * 70)
print('  §7-I. Market Regime Analysis (VIX quantile)')
print('=' * 70)
t_i = time.time()

# VIX 데이터 다운로드 (yfinance)
vix_path = DATA_DIR / 'vix_daily.csv'
if vix_path.exists():
    print(f'  [캐시] VIX 일별 데이터 사용: {vix_path.name}')
    vix_df = pd.read_csv(vix_path, index_col='date', parse_dates=True)
    vix_close = vix_df['VIX']
else:
    print('  [다운로드] VIX 일별 데이터 (yfinance: ^VIX)')
    try:
        import yfinance as yf
        vix_yf = yf.download('^VIX', start='2009-01-01', end='2026-01-01', progress=False, auto_adjust=False)
        # Multi-index columns 처리
        if isinstance(vix_yf.columns, pd.MultiIndex):
            vix_close = vix_yf['Close']['^VIX']
        else:
            vix_close = vix_yf['Close']
        vix_close.name = 'VIX'
        vix_close.index.name = 'date'
        vix_close.to_frame('VIX').to_csv(vix_path)
        print(f'    저장: {vix_path.name} ({len(vix_close)} 거래일)')
    except Exception as e:
        print(f'  ⚠️ yfinance 실패: {type(e).__name__}: {e}')
        print('  → VIX regime 분석 skip')
        vix_close = None

if vix_close is not None and len(vix_close) > 0:
    # 월별 평균 VIX
    vix_monthly = vix_close.resample('ME').mean()
    vix_at_eom = vix_monthly.reindex(reb_dates, method='nearest').dropna()
    # forward (다음 달 reb_date 기준 매핑)
    vix_at_reb = vix_at_eom.shift(0)  # 현재 시점 VIX 기준

    # 분위수 (33%, 67%)
    q33, q67 = vix_at_reb.quantile([0.33, 0.67])
    print(f'  VIX 분위수: 33%={q33:.1f}, 67%={q67:.1f}')

    def regime_label(v):
        if pd.isna(v):
            return None
        if v <= q33:
            return 'Low_VIX (<{:.0f})'.format(q33)
        elif v <= q67:
            return 'Mid_VIX'
        else:
            return 'High_VIX (>{:.0f})'.format(q67)

    regime_map = vix_at_reb.apply(regime_label)
    n_regimes = regime_map.value_counts()
    print(f'  Regime 분포: {n_regimes.to_dict()}')

    # 각 시나리오 × regime 별 메트릭
    regime_metrics = {}
    for name, r in all_returns_fair.items():
        regime_metrics[name] = {}
        for regime in ['Low_VIX', 'Mid_VIX', 'High_VIX']:
            mask = regime_map.apply(lambda x: x is not None and x.startswith(regime))
            mask = mask.reindex(r.index, fill_value=False)
            r_regime = r[mask]
            if len(r_regime) >= 5:
                sr = sharpe_ratio(r_regime)
                ann_ret = r_regime.mean() * 12
                ann_vol = r_regime.std() * np.sqrt(12)
                regime_metrics[name][regime] = {
                    'sharpe': sr,
                    'ann_ret_%': ann_ret * 100,
                    'ann_vol_%': ann_vol * 100,
                    'n_months': len(r_regime),
                }
    print()
    print('  [Regime별 Sharpe 표]')
    print(f'    {"시나리오":<25s} {"Low_VIX":>10s} {"Mid_VIX":>10s} {"High_VIX":>10s}')
    for name in all_returns_fair.keys():
        m = regime_metrics.get(name, {})
        low = m.get('Low_VIX', {}).get('sharpe', np.nan)
        mid = m.get('Mid_VIX', {}).get('sharpe', np.nan)
        high = m.get('High_VIX', {}).get('sharpe', np.nan)
        print(f'    {name:<25s} {low:>10.3f} {mid:>10.3f} {high:>10.3f}')

    vix_regime_results = {
        'q33': float(q33),
        'q67': float(q67),
        'regime_metrics': regime_metrics,
        'regime_map': regime_map.to_dict(),
        'n_regimes': n_regimes.to_dict(),
    }
else:
    vix_regime_results = None

print(f'  [§7-I 완료, {time.time()-t_i:.1f}s]')
print()

# ============================================================================
# 결과 저장
# ============================================================================
print('=' * 70)
print('  결과 저장')
print('=' * 70)

results = {
    'bootstrap_results': bootstrap_results,
    'ml_effect_bootstrap': ml_effect_results,
    'sector_exposure': sector_exposure,
    'sector_tilt': sector_tilt,
    'vix_regime': vix_regime_results,
}

# time_series DataFrame 은 to_dict 으로 변환 (pkl 호환성 위해 그대로 둠)
out_path = DATA_DIR / 'sec7_v2_analyses.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f'  💾 저장: {out_path.name} ({out_path.stat().st_size/1e3:.1f} KB)')

t_end = time.time()
print()
print('=' * 70)
print(f'  ALL DONE — total {t_end - t0:.1f}s ({(t_end - t0)/60:.1f} min)')
print('=' * 70)
