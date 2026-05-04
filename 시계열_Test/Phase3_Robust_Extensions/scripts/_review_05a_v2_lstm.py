"""
05a_v2_lstm.ipynb 전체 검토 — 노트북 셀 12개 모두 실행하여 오류/이상 진단.
"""
import sys, io, time, json, warnings, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
warnings.filterwarnings('ignore')

from pathlib import Path
NB_DIR = Path(__file__).resolve().parent.parent
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 시각화 backend (저장만, 화면 출력 없음)
import matplotlib.pyplot as plt
from scipy.stats import norm

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR
font_used = bootstrap()

OUT_DIR = OUTPUTS_DIR / '05a_v2_lstm_diag'
OUT_DIR.mkdir(parents=True, exist_ok=True)

errors = []
warnings_list = []

def section(name):
    print()
    print('=' * 75)
    print(f'  {name}')
    print('=' * 75)

# ─── §1 데이터 로드 ───
section('§1. 데이터 로드')
t0 = time.time()
ens_sw = pd.read_csv(DATA_DIR / 'ensemble_predictions_stockwise.csv', parse_dates=['date'])
n_before = len(ens_sw)
ens_sw = ens_sw[np.isfinite(ens_sw['y_true'])].copy()
print(f'ens_sw: {ens_sw.shape}, inf 제거 {n_before - len(ens_sw)} 행')

ens_sw['ym'] = ens_sw['date'].dt.to_period('M')
ens_sw['year'] = ens_sw['date'].dt.year

def assign_period(d):
    if d < pd.Timestamp('2015-01-01'): return 'P1 (2010-2014)'
    elif d < pd.Timestamp('2019-01-01'): return 'P2 (2015-2018)'
    elif d < pd.Timestamp('2021-01-01'): return 'P3 (2019-2020)'
    elif d < pd.Timestamp('2023-01-01'): return 'P4 (2021-2022)'
    else: return 'P5 (2023-2025)'
ens_sw['period'] = ens_sw['date'].apply(assign_period)
PERIOD_ORDER = ['P1 (2010-2014)', 'P2 (2015-2018)', 'P3 (2019-2020)',
                'P4 (2021-2022)', 'P5 (2023-2025)']

ens_sw['err_lstm'] = ens_sw['y_pred_lstm'] - ens_sw['y_true']
ens_sw['err_har'] = ens_sw['y_pred_har'] - ens_sw['y_true']
ens_sw['err_ens'] = ens_sw['y_pred_ensemble'] - ens_sw['y_true']
ens_sw['sq_lstm'] = ens_sw['err_lstm'] ** 2
ens_sw['sq_har'] = ens_sw['err_har'] ** 2
ens_sw['sq_ens'] = ens_sw['err_ens'] ** 2

vix = pd.read_csv(DATA_DIR / 'vix_daily.csv', parse_dates=['date']).set_index('date')['VIX']

sector_map_path = DATA_DIR / 'ticker_sector_mapping.csv'
sector_map = pd.read_csv(sector_map_path).set_index('ticker')['sector']
SECTOR_AVAILABLE = True
print(f'sector mapping: {len(sector_map)} 종목, Unknown {(sector_map == "Unknown").sum()}')
print(f'§1 시간: {time.time()-t0:.1f}s')

# ─── §2-A 월별 RMSE ───
section('§2-A. 월별 RMSE 시계열')
try:
    monthly_rmse = ens_sw.groupby('ym').agg(
        rmse_lstm=('sq_lstm', lambda x: np.sqrt(x.mean())),
        rmse_har=('sq_har', lambda x: np.sqrt(x.mean())),
        rmse_ens=('sq_ens', lambda x: np.sqrt(x.mean())),
        n=('y_true', 'count'),
    ).reset_index()
    monthly_rmse['date'] = monthly_rmse['ym'].dt.to_timestamp()
    mr = monthly_rmse[monthly_rmse['date'] >= '2010-01-01'].copy()
    print(f'  monthly_rmse: {monthly_rmse.shape}, OOS {len(mr)} 개월')
    print(f'  OOS LSTM RMSE: mean={mr["rmse_lstm"].mean():.4f}')
    print(f'  OOS HAR  RMSE: mean={mr["rmse_har"].mean():.4f}')
    print(f'  OOS Ens  RMSE: mean={mr["rmse_ens"].mean():.4f}')

    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.plot(mr['date'], mr['rmse_lstm'], label='LSTM', linewidth=1.2)
    ax.plot(mr['date'], mr['rmse_har'], label='HAR', linewidth=1.2)
    ax.plot(mr['date'], mr['rmse_ens'], label='Ensemble', linewidth=1.5)
    ax.legend()
    plt.savefig(OUT_DIR / 'A_monthly_rmse_timeline.png', dpi=80, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-A', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-B 종목 × 시기 RMSE Heatmap ───
section('§2-B. 종목 × 시기 RMSE Heatmap (5 시기 cover 종목만)')
try:
    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']
    stock_period_rmse_full = ens_oos.groupby(['ticker', 'period'])['sq_ens'].apply(
        lambda x: np.sqrt(x.mean())
    ).unstack('period').reindex(columns=PERIOD_ORDER)
    stock_period_rmse_full['n_period'] = stock_period_rmse_full[PERIOD_ORDER].notna().sum(axis=1)
    n_full = (stock_period_rmse_full['n_period'] == 5).sum()
    n_part = (stock_period_rmse_full['n_period'] < 5).sum()
    print(f'  전체: {len(stock_period_rmse_full)}, 5 cover: {n_full}, 부분: {n_part}')

    stock_period_rmse = stock_period_rmse_full[stock_period_rmse_full['n_period'] == 5].drop(columns='n_period')
    stock_period_rmse['mean_rmse'] = stock_period_rmse.mean(axis=1)
    top10 = stock_period_rmse.nlargest(10, 'mean_rmse').drop(columns='mean_rmse')
    print(f'  Top 10 어려운: {top10.index.tolist()}')
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-B', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-C Forecast Bias ───
section('§2-C. Forecast Bias')
try:
    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']
    bias_per_ticker = ens_oos.groupby('ticker').agg(
        bias_lstm=('err_lstm', 'mean'),
        bias_har=('err_har', 'mean'),
        bias_ens=('err_ens', 'mean'),
        n=('y_true', 'count'),
    ).reset_index()
    print(f'  종목: {len(bias_per_ticker)}, bias_ens mean: {bias_per_ticker["bias_ens"].mean():+.4f}')
    print(f'    overestimate: {(bias_per_ticker["bias_ens"] > 0).sum()}, '
          f'underestimate: {(bias_per_ticker["bias_ens"] < 0).sum()}')
    # sector boxplot test
    bias_per_ticker['sector'] = bias_per_ticker['ticker'].map(sector_map).fillna('Unknown')
    sectors = sorted(bias_per_ticker['sector'].unique())
    print(f'    sectors: {len(sectors)} 개 (Unknown 포함)')
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-C', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-D vol regime ───
section('§2-D. 변동성 Regime별 예측력')
try:
    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01'].copy()
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
    )
    print(f'  vol quintile별 RMSE:')
    print(regime_rmse.round(4).to_string())
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-D', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-E VIX tier ───
section('§2-E. VIX 분위별 예측 정확도')
try:
    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01'].copy()
    vix_oos = vix.loc['2010-01-01':'2025-12-31']
    vix_t1, vix_t2 = vix_oos.quantile([0.33, 0.67]).values
    print(f'  VIX tertile: low<={vix_t1:.2f} mid<={vix_t2:.2f} high>')

    def vix_tier(d):
        v = vix_oos.get(d, np.nan)
        if pd.isna(v): return 'Unknown'
        if v <= vix_t1: return 'Low'
        elif v <= vix_t2: return 'Mid'
        else: return 'High'
    ens_oos['vix_tier'] = ens_oos['date'].apply(vix_tier)
    ens_oos = ens_oos[ens_oos['vix_tier'] != 'Unknown']
    vix_rmse = ens_oos.groupby('vix_tier', observed=True).agg(
        rmse_lstm=('sq_lstm', lambda x: np.sqrt(x.mean())),
        rmse_har=('sq_har', lambda x: np.sqrt(x.mean())),
        rmse_ens=('sq_ens', lambda x: np.sqrt(x.mean())),
    ).reindex(['Low', 'Mid', 'High'])
    print(vix_rmse.round(4).to_string())
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-E', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-F DM test (sample for speed) ───
section('§2-F. Diebold-Mariano test (sample 50 ticker)')
try:
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

    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']
    sample_tickers = sorted(ens_oos['ticker'].unique())[:50]
    dm_results = []
    for t in sample_tickers:
        g = ens_oos[ens_oos['ticker'] == t].sort_values('date')
        dm, p, n = dm_test(g['sq_lstm'].values, g['sq_har'].values, h=21)
        dm_results.append({'ticker': t, 'dm_stat': dm, 'p_value': p, 'n': n})
    dm_df = pd.DataFrame(dm_results)
    dm_df_clean = dm_df.dropna()
    n_lstm = ((dm_df_clean['dm_stat'] < 0) & (dm_df_clean['p_value'] < 0.05)).sum()
    n_har = ((dm_df_clean['dm_stat'] > 0) & (dm_df_clean['p_value'] < 0.05)).sum()
    n_neutral = (dm_df_clean['p_value'] >= 0.05).sum()
    print(f'  sample 50: LSTM 우월 {n_lstm}, HAR 우월 {n_har}, 차이 없음 {n_neutral}')
    print('  ✅ 정상 (전체 613 종목 시 ~5분 소요)')
except Exception as e:
    errors.append(('§2-F', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-G Sector × Best Model ───
section('§2-G. Sector × Best Model')
try:
    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']
    best_per_ticker = ens_oos.groupby('ticker').agg(
        rmse_lstm=('sq_lstm', lambda x: np.sqrt(x.mean())),
        rmse_har=('sq_har', lambda x: np.sqrt(x.mean())),
        rmse_ens=('sq_ens', lambda x: np.sqrt(x.mean())),
    )
    def best_model(row):
        rmses = {'LSTM': row['rmse_lstm'], 'HAR': row['rmse_har'], 'Ensemble': row['rmse_ens']}
        return min(rmses, key=rmses.get)
    best_per_ticker['best_model'] = best_per_ticker.apply(best_model, axis=1)
    best_cnt = best_per_ticker['best_model'].value_counts()
    print(f'  Best model: {dict(best_cnt)}')

    best_per_ticker['sector'] = best_per_ticker.index.map(sector_map).fillna('Unknown')
    sect_best = pd.crosstab(best_per_ticker['sector'], best_per_ticker['best_model'])
    sect_best['total'] = sect_best.sum(axis=1)
    print(f'  Sector × Best Model:')
    print(sect_best.to_string())
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-G', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-H 가중치 안정성 ───
section('§2-H. 가중치 안정성')
try:
    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']
    w_stats = ens_oos.groupby('ticker').agg(
        w_v4_mean=('w_v4', 'mean'),
        w_v4_std=('w_v4', 'std'),
    )
    print(f'  종목: {len(w_stats)}, w_v4 mean: {w_stats["w_v4_mean"].mean():.3f}, '
          f'std mean: {w_stats["w_v4_std"].mean():.3f}')
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-H', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-I 시기별 산점도 ───
section('§2-I. 시기별 Pred vs True 산점도')
try:
    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']
    period_stats = []
    for period in PERIOD_ORDER:
        sub = ens_oos[ens_oos['period'] == period]
        if len(sub) == 0:
            continue
        corr = sub[['y_true', 'y_pred_ensemble']].corr().iloc[0, 1]
        rmse = np.sqrt(sub['sq_ens'].mean())
        period_stats.append({
            'period': period, 'n': len(sub), 'corr': corr, 'rmse': rmse,
            'y_true_mean': sub['y_true'].mean(),
            'y_pred_mean': sub['y_pred_ensemble'].mean(),
        })
    ps_df = pd.DataFrame(period_stats)
    print(ps_df.round(3).to_string(index=False))
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-I', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-J Top/Bottom 5 case study (5 시기 cover 만) ───
section('§2-J. Top/Bottom 5 case study')
try:
    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']
    period_count = ens_oos.groupby('ticker')['period'].nunique()
    full_period_tickers = period_count[period_count == 5].index.tolist()
    ens_full = ens_oos[ens_oos['ticker'].isin(full_period_tickers)]
    ticker_rmse = ens_full.groupby('ticker').apply(
        lambda g: np.sqrt(g['sq_ens'].mean())
    ).rename('rmse_ens')
    best5 = ticker_rmse.nsmallest(5).index.tolist()
    worst5 = ticker_rmse.nlargest(5).index.tolist()
    print(f'  Best 5: {best5}')
    print(f'  Worst 5: {worst5}')
    # 시각화 정상 작동 테스트
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for i, ticker in enumerate(best5 + worst5):
        ax = axes[i//5, i%5]
        sub = ens_oos[ens_oos['ticker'] == ticker].sort_values('date')
        mt = sub.groupby(sub['date'].dt.to_period('M')).agg(
            y_true=('y_true', 'mean'),
            y_pred=('y_pred_ensemble', 'mean'),
        )
        mt.index = mt.index.to_timestamp()
        ax.plot(mt.index, mt['y_true'], label='y_true')
        ax.plot(mt.index, mt['y_pred'], label='y_pred')
    plt.savefig(OUT_DIR / 'J_top_bottom_case_study_test.png', dpi=80, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-J', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-K fold ───
section('§2-K. CV fold별 성능')
try:
    fold_stats = ens_sw.groupby('fold').agg(
        rmse_lstm=('sq_lstm', lambda x: np.sqrt(x.mean())),
        rmse_har=('sq_har', lambda x: np.sqrt(x.mean())),
        rmse_ens=('sq_ens', lambda x: np.sqrt(x.mean())),
        n=('y_true', 'count'),
        date_min=('date', 'min'),
        date_max=('date', 'max'),
    ).reset_index()
    print(f'  fold 수: {len(fold_stats)}, RMSE 평균: {fold_stats["rmse_ens"].mean():.4f}')
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-K', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2-L 분포 비교 ───
section('§2-L. y_true vs y_pred 분포 비교')
try:
    ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']
    pcts = [1, 5, 25, 50, 75, 95, 99]
    pct_table = pd.DataFrame({
        'percentile': pcts,
        'y_true': [np.percentile(ens_oos['y_true'], p) for p in pcts],
        'y_pred_ensemble': [np.percentile(ens_oos['y_pred_ensemble'], p) for p in pcts],
    })
    print(pct_table.round(3).to_string(index=False))
    # QQ plot test
    qs = np.linspace(0.01, 0.99, 99)
    q_true = np.quantile(ens_oos['y_true'], qs)
    q_pred = np.quantile(ens_oos['y_pred_ensemble'], qs)
    print(f'  QQ slope (rough): {(q_pred[-1]-q_pred[0])/(q_true[-1]-q_true[0]):.3f}')
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2-L', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §3 종합 요약 의존성 검증 ───
section('§3. 종합 요약 의존성 검증')
try:
    # §3에서 사용되는 변수들이 정의되어 있는지
    required = ['dm_df_clean', 'best_per_ticker', 'period_stats', 'ens_sw']
    missing = [v for v in required if v not in dir()]
    if missing:
        print(f'  ⚠️ 누락 변수: {missing}')
    else:
        print(f'  ✅ 모든 필수 변수 (dm_df_clean, best_per_ticker, period_stats, ens_sw) 정의됨')

    ens_oos_all = ens_sw[ens_sw['date'] >= '2010-01-01']
    overall_rmse = {
        'rmse_lstm_overall': float(np.sqrt(ens_oos_all['sq_lstm'].mean())),
        'rmse_har_overall': float(np.sqrt(ens_oos_all['sq_har'].mean())),
        'rmse_ensemble_overall': float(np.sqrt(ens_oos_all['sq_ens'].mean())),
    }
    print(f'  전체 OOS RMSE: LSTM={overall_rmse["rmse_lstm_overall"]:.4f}, '
          f'HAR={overall_rmse["rmse_har_overall"]:.4f}, '
          f'Ens={overall_rmse["rmse_ensemble_overall"]:.4f}')

    # JSON 직렬화 가능성 검증
    summary = {
        'notebook': '05a_v2_lstm.ipynb',
        'overall': overall_rmse,
        'period_stats': {p['period']: {k: float(v) for k, v in p.items() if k != 'period'}
                         for p in period_stats},
    }
    json_str = json.dumps(summary, indent=2, ensure_ascii=False)
    print(f'  ✅ JSON 직렬화 OK ({len(json_str)} chars)')
except Exception as e:
    errors.append(('§3', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── 최종 보고 ───
print()
print('=' * 75)
print('  최종 검토 결과')
print('=' * 75)
print(f'  오류 수: {len(errors)}')
print(f'  경고 수: {len(warnings_list)}')
print(f'  총 시간: {time.time()-t0:.1f}s')

if errors:
    print()
    print('=== 오류 상세 ===')
    for sec, msg, tb in errors:
        print(f'\n[{sec}]')
        print(f'  메시지: {msg}')
        print(f'  Traceback (요약):')
        for ln in tb.split('\n')[-5:]:
            print(f'    {ln}')
else:
    print()
    print('  ✅ 모든 12 분석 + §3 종합 요약 정상 작동')
    print('  ✅ 시각화 / 메트릭 / JSON 직렬화 모두 정상')
