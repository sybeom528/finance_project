"""
EWMA omega 8개 신규 실험 결과 분석.

- 시나리오 1 (lo, λ=0.825): 첫 12개월 trim, 168개월 분석
- 시나리오 2 (std, λ=0.94): 첫 36개월 trim, 144개월 분석
- baseline 도 같은 구간으로 trim 해 공정 비교
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bl_functions import compute_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

# ── 데이터 로드 (rf, spy) ─────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
panel    = pd.read_csv(DATA_DIR / 'monthly_panel.csv', parse_dates=['date'])
panel    = panel.set_index(['date', 'ticker'])
rf_series  = panel['rf_1m'].groupby(level='date').first()
spy_series = panel['spy_ret'].groupby(level='date').first()


def load_result(name):
    path = RESULTS_DIR / f'{name}.pkl'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def metrics_for_period(res, start, end, label):
    """주어진 기간으로 trim 후 metrics 계산."""
    if res is None:
        return None
    ret = res['ret'].loc[start:end]
    if len(ret) == 0:
        return None
    return compute_metrics(
        ret, rf_series.loc[start:end], label=label,
        mkt_ret=spy_series.loc[start:end])


def print_table(rows, columns):
    """간단한 텍스트 표 출력."""
    widths = {c: max(len(c), max(len(str(r.get(c, ''))) for r in rows)) for c in columns}
    header = ' | '.join(f'{c:<{widths[c]}}' for c in columns)
    print(header)
    print('-' * len(header))
    for r in rows:
        print(' | '.join(f'{str(r.get(c, "")):<{widths[c]}}' for c in columns))


def fmt(x, pct=False, dec=3):
    if x is None:
        return 'N/A'
    if isinstance(x, float) and np.isnan(x):
        return 'N/A'
    if pct:
        return f'{x*100:.{dec}f}%'
    return f'{x:.{dec}f}'


# ══════════════════════════════════════════════════════════════
# 1. 신규 8개 실험 결과 확인 — 전체 기간 (180개월)
# ══════════════════════════════════════════════════════════════
print('=' * 100)
print('1. EWMA 신규 8개 실험 — 전체 OOS 180개월 (2010-01 ~ 2024-12)')
print('=' * 100)

ewma_names = [
    'p_lstm_mcap_ewma_lo', 'p_lstm_eq_ewma_lo', 'p_lstm_rp_ewma_lo', 'p_lstm_vol_mcap_ewma_lo',
    'p_lstm_mcap_ewma_std', 'p_lstm_eq_ewma_std', 'p_lstm_rp_ewma_std', 'p_lstm_vol_mcap_ewma_std',
]

# 비교 대상: 동일 P 가중의 LSTM 기본 + RMSE 변형
compare_names = [
    'baseline', 'prior_eq',
    'p_lstm_mcap', 'p_lstm_eq', 'p_lstm_rp', 'p_lstm_vol_mcap',
    'p_lstm_mcap_omega_rmse', 'p_lstm_eq_omega_rmse', 'p_lstm_rp_omega_rmse', 'p_lstm_vol_mcap_omega_rmse',
    'p_lstm_mcap_omega_rmse_pt', 'p_lstm_eq_omega_rmse_pt', 'p_lstm_rp_omega_rmse_pt', 'p_lstm_vol_mcap_omega_rmse_pt',
]

all_names = ewma_names + compare_names
rows_full = []
for name in all_names:
    res = load_result(name)
    if res is None:
        rows_full.append({'name': name, 'note': 'pkl 없음'})
        continue
    m = metrics_for_period(res, '2010-01-01', '2024-12-31', name)
    if m is None:
        continue
    rows_full.append({
        'name': name,
        'Sharpe': fmt(m['sharpe']),
        'CAGR': fmt(m['cagr'], pct=True, dec=2),
        'Vol': fmt(m['vol'], pct=True, dec=2),
        'MDD': fmt(m['mdd'], pct=True, dec=2),
        'Sortino': fmt(m['sortino']),
        'Calmar': fmt(m['calmar']),
    })

print_table(rows_full, ['name', 'Sharpe', 'CAGR', 'Vol', 'MDD', 'Sortino', 'Calmar'])

# ══════════════════════════════════════════════════════════════
# 2. 시나리오 1 (lo, λ=0.825) — trim 12개월 후 168개월 분석
# ══════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('2. 시나리오 1 (λ=0.825) — trim 12개월: 2011-01 ~ 2024-12 (168개월)')
print('=' * 100)

rows_s1 = []
s1_compare = ['baseline', 'prior_eq', 'p_lstm_mcap', 'p_lstm_eq', 'p_lstm_rp', 'p_lstm_vol_mcap']
s1_targets = [n for n in ewma_names if 'ewma_lo' in n] + s1_compare
for name in s1_targets:
    res = load_result(name)
    if res is None:
        continue
    m = metrics_for_period(res, '2011-01-01', '2024-12-31', name)
    if m is None:
        continue
    rows_s1.append({
        'name': name,
        'Sharpe': fmt(m['sharpe']),
        'CAGR': fmt(m['cagr'], pct=True, dec=2),
        'Vol': fmt(m['vol'], pct=True, dec=2),
        'MDD': fmt(m['mdd'], pct=True, dec=2),
        'Sortino': fmt(m['sortino']),
    })

print_table(rows_s1, ['name', 'Sharpe', 'CAGR', 'Vol', 'MDD', 'Sortino'])

# ══════════════════════════════════════════════════════════════
# 3. 시나리오 2 (std, λ=0.94) — trim 36개월 후 144개월 분석
# ══════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('3. 시나리오 2 (λ=0.94) — trim 36개월: 2013-01 ~ 2024-12 (144개월)')
print('=' * 100)

rows_s2 = []
s2_compare = ['baseline', 'prior_eq', 'p_lstm_mcap', 'p_lstm_eq', 'p_lstm_rp', 'p_lstm_vol_mcap']
s2_targets = [n for n in ewma_names if 'ewma_std' in n] + s2_compare
for name in s2_targets:
    res = load_result(name)
    if res is None:
        continue
    m = metrics_for_period(res, '2013-01-01', '2024-12-31', name)
    if m is None:
        continue
    rows_s2.append({
        'name': name,
        'Sharpe': fmt(m['sharpe']),
        'CAGR': fmt(m['cagr'], pct=True, dec=2),
        'Vol': fmt(m['vol'], pct=True, dec=2),
        'MDD': fmt(m['mdd'], pct=True, dec=2),
        'Sortino': fmt(m['sortino']),
    })

print_table(rows_s2, ['name', 'Sharpe', 'CAGR', 'Vol', 'MDD', 'Sortino'])

# ══════════════════════════════════════════════════════════════
# 4. omega 시계열 진단 — 각 EWMA 실험의 omega 분포 (meta 의 omega 컬럼)
# ══════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('4. EWMA omega 시계열 진단 (실험별 omega 분포)')
print('=' * 100)

rows_omega = []
for name in ewma_names:
    res = load_result(name)
    if res is None:
        continue
    meta = res['meta']
    if 'omega' not in meta.columns:
        rows_omega.append({'name': name, 'note': 'omega 미기록 (재실행 필요)'})
        continue
    omega_series = meta['omega'].dropna()
    if len(omega_series) == 0:
        continue
    rows_omega.append({
        'name'         : name,
        'n'            : len(omega_series),
        'omega_mean'   : f'{omega_series.mean():.6f}',
        'omega_median' : f'{omega_series.median():.6f}',
        'omega_min'    : f'{omega_series.min():.6f}',
        'omega_max'    : f'{omega_series.max():.6f}',
        'omega_std'    : f'{omega_series.std():.6f}',
    })

if rows_omega:
    print_table(rows_omega, ['name', 'n', 'omega_mean', 'omega_median', 'omega_min', 'omega_max', 'omega_std'])

# ══════════════════════════════════════════════════════════════
# 5. 상위 5개 (Sharpe 순) — trim 적용 후
# ══════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('5. 시나리오 1·2 통합 Top 10 (Sharpe 순, 시나리오별 trim 적용)')
print('=' * 100)

# 시나리오 1 결과 + 시나리오 2 결과 합쳐서 Sharpe 정렬
combined = []
for name in ewma_names:
    res = load_result(name)
    if res is None:
        continue
    period_start = '2011-01-01' if 'ewma_lo' in name else '2013-01-01'
    m = metrics_for_period(res, period_start, '2024-12-31', name)
    if m is None:
        continue
    combined.append({
        'name': name,
        'period': '168M' if 'ewma_lo' in name else '144M',
        'Sharpe': m['sharpe'],
        'CAGR': m['cagr'],
        'MDD': m['mdd'],
    })

# baseline 도 두 구간으로 추가
for period_start, period_label in [('2011-01-01', '168M'), ('2013-01-01', '144M')]:
    res = load_result('baseline')
    if res is None:
        continue
    m = metrics_for_period(res, period_start, '2024-12-31', f'baseline ({period_label})')
    if m is None:
        continue
    combined.append({
        'name': f'baseline ({period_label})',
        'period': period_label,
        'Sharpe': m['sharpe'],
        'CAGR': m['cagr'],
        'MDD': m['mdd'],
    })

combined.sort(key=lambda x: x['Sharpe'], reverse=True)
rows_top = []
for i, r in enumerate(combined, 1):
    rows_top.append({
        'rank': i,
        'name': r['name'],
        'period': r['period'],
        'Sharpe': fmt(r['Sharpe']),
        'CAGR': fmt(r['CAGR'], pct=True, dec=2),
        'MDD': fmt(r['MDD'], pct=True, dec=2),
    })

print_table(rows_top, ['rank', 'name', 'period', 'Sharpe', 'CAGR', 'MDD'])

print()
print('=' * 100)
print('분석 완료')
print('=' * 100)
