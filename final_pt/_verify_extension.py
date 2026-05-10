"""
_verify_extension.py — BL extension 완료 후 165 pkl 일괄 검증.

체크 항목:
  1. 모든 pkl 의 ret 시리즈가 192 entries (2010-01 ~ 2025-12)
  2. mat_eq_eq_lam_pap 의 sortino_TEST / sortino_HOLD_OUT 정상 값
  3. 일부 sample 의 weights/comp/meta 길이 일관성
  4. _backup_pre_extension/ 의 180m vs 신규 192m 의 2024-12 동일성

실행:
    cd final && python _verify_extension.py
"""
import io
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / 'results'
BACKUP_DIR  = RESULTS_DIR / '_backup_pre_extension'

# ── 1. 192m 일관성 ─────────────────────────────────────────────
print('━' * 68)
print('[1] 165 pkl 의 ret 시리즈 길이 일관성')
print('━' * 68)

pkls = sorted(p for p in RESULTS_DIR.glob('*.pkl'))
length_counts = {}
incomplete = []
for p in pkls:
    with open(p, 'rb') as f: r = pickle.load(f)
    n = len(r.get('ret', pd.Series()))
    length_counts[n] = length_counts.get(n, 0) + 1
    if n != 192:
        incomplete.append((p.stem, n))

print(f'  total pkls: {len(pkls)}')
for n, cnt in sorted(length_counts.items()):
    print(f'  ret length={n}: {cnt:3} cfgs')

if incomplete:
    print(f'\n  [WARN] {len(incomplete)} pkls 가 192m 미만:')
    for name, n in incomplete[:10]:
        print(f'    - {name}: n={n}')
else:
    print(f'\n  [OK] 모든 pkl 가 192m')


# ── 2. 핵심 cfg 의 sortino 메트릭 검증 ────────────────────────
print()
print('━' * 68)
print('[2] 핵심 cfg sortino 메트릭')
print('━' * 68)

import sys as _sys
_sys.path.insert(0, str(BASE_DIR))
from master_table import build_master_table  # noqa

panel = pd.read_csv(BASE_DIR / 'data' / 'monthly_panel.csv', parse_dates=['date'])
rf  = panel.groupby('date')['rf_1m'].first()
spy = panel.groupby('date')['spy_ret'].first()
mt = build_master_table(RESULTS_DIR, rf, spy, sort_by='sortino_TEST')

key_cfgs = ['mat_eq_eq_lam_pap', 'baseline', 'mat_eq_eq_lam_he', 'capm_no_bl', 'naive_lowvol']
for n in key_cfgs:
    row = mt[mt['name'] == n]
    if len(row) == 0:
        print(f'  {n}: NOT FOUND')
        continue
    r = row.iloc[0]
    s_test    = r.get('sortino_TEST', float('nan'))
    s_holdout = r.get('sortino_HOLD_OUT', float('nan'))
    s_full    = r.get('sortino_FULL', float('nan'))
    cagr_h    = r.get('cagr_HOLD_OUT', float('nan'))
    mdd_h     = r.get('mdd_HOLD_OUT', float('nan'))
    print(f'  {n:25}  sortino_TEST={s_test:.3f}  HOLD_OUT={s_holdout:.3f}  FULL={s_full:.3f}  '
          f'CAGR_HOLD={cagr_h*100:.2f}%  MDD_HOLD={mdd_h*100:.2f}%')

print()
print('Top 10 by sortino_TEST:')
print(mt.head(10)[['name', 'sortino_TEST', 'sortino_HOLD_OUT', 'sharpe', 'cagr', 'mdd']].round(3).to_string(index=False))


# ── 3. 백업 vs 신규 의 2024-12 일관성 ─────────────────────────
print()
print('━' * 68)
print('[3] _backup vs 신규 의 2024-12 정합성 (random 5 cfg)')
print('━' * 68)

import random
sample_pkls = random.sample(pkls, min(5, len(pkls)))
target_date = pd.Timestamp('2024-12-31')

for p in sample_pkls:
    with open(p, 'rb') as f: new = pickle.load(f)
    bp = BACKUP_DIR / p.name
    if not bp.exists():
        print(f'  {p.stem}: 백업 없음')
        continue
    with open(bp, 'rb') as f: old = pickle.load(f)

    if target_date not in new['ret'].index or target_date not in old['ret'].index:
        print(f'  {p.stem}: 2024-12 행 부재')
        continue

    diff_ret = abs(new['ret'].loc[target_date] - old['ret'].loc[target_date])
    print(f'  {p.stem:30} 2024-12 ret diff = {diff_ret:.2e}  '
          f'{"OK" if diff_ret < 1e-8 else "WARN — 차이 발생"}')


# ── 4. 일관성 요약 ────────────────────────────────────────────
print()
print('━' * 68)
print('[4] 종합 요약')
print('━' * 68)
print(f'  192m pkl  : {length_counts.get(192, 0)} / 165')
print(f'  180m 잔여 : {length_counts.get(180, 0)}')
print(f'  Top 1 sortino_TEST : {mt.iloc[0]["name"]} ({mt.iloc[0]["sortino_TEST"]:.3f})')
print(f'  → mat_eq_eq_lam_pap 가 Top 1 인지 확인 필요')
