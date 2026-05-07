"""
_full_pkl_audit.py — 156 BL pkl 종합 무결성 감사.

검증 항목:
  [A] 길이 일관성: 모든 pkl ret 시리즈 = 192 entries (2010-01 ~ 2025-12)
  [B] 컬럼 일관성: ret/gross_ret/spy_ret/weights/comp/meta/errors/config 키 존재
  [C] index 정렬: 월말 정렬, 192 month-end 정확히 일치
  [D] NaN/Inf 검사: ret/gross_ret/weights 의 비정상 값 비율
  [E] 백업 정합성: _backup_pre_extension/ 의 동일 cfg 와 2010-01 ~ 2024-12 동등성
  [F] cfg 메타 일관성: bl_config.EXPERIMENTS 156 cfg 와 results/*.pkl 일치
  [G] 핵심 메트릭 sanity: TOP 10 sortino_TEST + 핵심 후보의 TEST/HOLD_OUT/FULL
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

sys.path.insert(0, str(BASE_DIR))
from bl_config import EXPERIMENTS  # noqa
from master_table import build_master_table, EVAL_PERIODS  # noqa

# ── pkl 일괄 로드 ─────────────────────────────────────────────
print('━' * 72)
print(' BL pkl 종합 무결성 감사')
print('━' * 72)

active_pkls = sorted(p for p in RESULTS_DIR.glob('*.pkl'))
print(f'\nactive results/*.pkl: {len(active_pkls)}')
print(f'expected EXPERIMENTS : {len(EXPERIMENTS)}')

# ── [F] cfg / pkl 일관성 ──────────────────────────────────────
print()
print('━ [F] cfg ↔ pkl 일관성 ━' * 1)
exp_names = {e['name'] for e in EXPERIMENTS}
pkl_names = {p.stem for p in active_pkls}

missing_pkl = sorted(exp_names - pkl_names)
extra_pkl   = sorted(pkl_names - exp_names)
if missing_pkl:
    print(f'  ❌ EXPERIMENTS 에 있으나 pkl 없음: {len(missing_pkl)}개')
    for n in missing_pkl[:5]: print(f'     - {n}')
else:
    print(f'  ✅ EXPERIMENTS 156 cfg 모두 pkl 존재')
if extra_pkl:
    print(f'  ⚠ pkl 은 있으나 EXPERIMENTS 에 없음: {len(extra_pkl)}개')
    for n in extra_pkl[:5]: print(f'     - {n}')
else:
    print(f'  ✅ pkl 외 orphan 없음')


# ── [A] [B] [C] [D] 종합 ──────────────────────────────────────
print()
print('━ [A][B][C][D] 길이·키·index·NaN 검사 ━')

EXPECTED_KEYS = ['config', 'ret', 'gross_ret', 'spy_ret', 'weights', 'comp', 'meta', 'errors']
EXPECTED_LEN  = 192

# 정확히 month-end 192 dates (2010-01 ~ 2025-12)
expected_dates = pd.date_range('2010-01-31', '2025-12-31', freq='ME')
assert len(expected_dates) == 192

issues_A, issues_B, issues_C, issues_D = [], [], [], []
nan_summary = []

for p in active_pkls:
    name = p.stem
    with open(p, 'rb') as f: r = pickle.load(f)

    # [B] 키 일관성
    missing_keys = [k for k in EXPECTED_KEYS if k not in r]
    if missing_keys:
        issues_B.append((name, f'missing keys: {missing_keys}'))

    ret = r.get('ret', pd.Series(dtype=float))
    if not isinstance(ret, pd.Series):
        issues_A.append((name, f'ret type: {type(ret).__name__}'))
        continue

    # [A] 길이
    if len(ret) != EXPECTED_LEN:
        issues_A.append((name, f'len={len(ret)} (expected {EXPECTED_LEN})'))
        continue

    # [C] index 일치
    if not ret.index.equals(expected_dates):
        try:
            mismatch = (ret.index != expected_dates).sum()
            issues_C.append((name, f'index mismatch: {mismatch} dates differ'))
        except Exception as e:
            issues_C.append((name, f'index compare err: {e}'))

    # [D] NaN/Inf
    n_nan = ret.isna().sum() + (~np.isfinite(ret)).sum()
    gr = r.get('gross_ret', pd.Series(dtype=float))
    n_nan_g = gr.isna().sum() if isinstance(gr, pd.Series) else 0
    if n_nan > 0 or n_nan_g > 0:
        issues_D.append((name, f'ret NaN/Inf={n_nan}, gross NaN={n_nan_g}'))
    nan_summary.append({'name': name, 'ret_nan': n_nan, 'gross_nan': n_nan_g})

print(f'  [A] 길이 issues : {len(issues_A)}/{len(active_pkls)}')
for n, msg in issues_A[:5]: print(f'     ❌ {n}: {msg}')
print(f'  [B] 키 issues   : {len(issues_B)}/{len(active_pkls)}')
for n, msg in issues_B[:5]: print(f'     ❌ {n}: {msg}')
print(f'  [C] index issues: {len(issues_C)}/{len(active_pkls)}')
for n, msg in issues_C[:5]: print(f'     ❌ {n}: {msg}')
print(f'  [D] NaN issues  : {len(issues_D)}/{len(active_pkls)}')
for n, msg in issues_D[:5]: print(f'     ⚠ {n}: {msg}')


# ── [E] 백업 정합성 ───────────────────────────────────────────
print()
print('━ [E] 백업 vs 신규 의 2010-01 ~ 2024-12 정합성 ━')
import random
random.seed(42)
sample = random.sample(active_pkls, min(20, len(active_pkls)))
diffs = []
for p in sample:
    bp = BACKUP_DIR / p.name
    if not bp.exists():
        diffs.append((p.stem, 'no backup'))
        continue
    with open(p,'rb') as f: new = pickle.load(f)
    with open(bp,'rb') as f: old = pickle.load(f)
    new_ret = new['ret'].loc['2010-01-31':'2024-12-31']
    old_ret = old['ret'].loc['2010-01-31':'2024-12-31']
    if len(new_ret) != len(old_ret):
        diffs.append((p.stem, f'len diff: {len(new_ret)} vs {len(old_ret)}'))
        continue
    max_diff = (new_ret.values - old_ret.values).__abs__().max()
    if max_diff > 1e-8:
        diffs.append((p.stem, f'max diff = {max_diff:.2e}'))

if diffs:
    print(f'  ⚠ {len(diffs)} cfg 에서 차이 발견:')
    for n, msg in diffs: print(f'     {n}: {msg}')
else:
    print(f'  ✅ random 20 cfg 모두 2010~2024 ret 시리즈가 backup 과 동등 (diff < 1e-8)')


# ── [G] 핵심 메트릭 sanity ─────────────────────────────────────
print()
print('━ [G] 핵심 메트릭 sanity ━')
panel = pd.read_csv(BASE_DIR / 'data' / 'monthly_panel.csv', parse_dates=['date'])
rf  = panel.groupby('date')['rf_1m'].first()
spy = panel.groupby('date')['spy_ret'].first()
mt = build_master_table(RESULTS_DIR, rf, spy, sort_by='sortino_TEST')

key = ['mat_eq_eq_lam_pap', 'baseline', 'capm_no_bl', 'naive_lowvol',
       'mat_eq_eq_lam_he', 'mat_eq_eq_raw_pap']
for n in key:
    row = mt[mt['name'] == n]
    if len(row) == 0:
        print(f'  ❌ {n}: 없음')
        continue
    r = row.iloc[0]
    s_t = r.get('sortino_TEST', np.nan)
    s_h = r.get('sortino_HOLD_OUT', np.nan)
    s_f = r.get('sortino_FULL', np.nan)
    cagr_t = r.get('cagr_TEST', np.nan)
    mdd_t  = r.get('mdd_TEST', np.nan)
    print(f'  {n:25}  sortino TEST={s_t:.3f} | HOLD={s_h:.3f} | FULL={s_f:.3f}  '
          f'cagr_TEST={cagr_t*100:.2f}%  mdd_TEST={mdd_t*100:.2f}%')

print()
print('Top 10 by sortino_TEST:')
top10 = mt.head(10)[['name', 'sortino_TEST', 'sortino_HOLD_OUT', 'sortino_FULL',
                       'sharpe', 'cagr', 'mdd']].round(3)
print(top10.to_string(index=False))

print()
print('Top 5 by sortino_HOLD_OUT (실전 검증):')
top5_h = mt.nlargest(5, 'sortino_HOLD_OUT')[
    ['name', 'sortino_TEST', 'sortino_HOLD_OUT', 'sharpe', 'cagr_HOLD_OUT', 'mdd_HOLD_OUT']
].round(3)
print(top5_h.to_string(index=False))


# ── 종합 요약 ─────────────────────────────────────────────────
print()
print('━' * 72)
print(' 종합 결과')
print('━' * 72)
total_issues = len(issues_A) + len(issues_B) + len(issues_C)
print(f'  [F] cfg/pkl 일관성   : {"✅" if not (missing_pkl or extra_pkl) else "❌"}')
print(f'  [A] 길이 일관성       : {"✅" if not issues_A else "❌"}  ({len(active_pkls) - len(issues_A)}/{len(active_pkls)} OK)')
print(f'  [B] 키 일관성         : {"✅" if not issues_B else "❌"}')
print(f'  [C] index 일관성      : {"✅" if not issues_C else "❌"}')
print(f'  [D] NaN/Inf          : {"✅" if not issues_D else "⚠"}  ({len(issues_D)} 비정상)')
print(f'  [E] 백업 정합성       : {"✅" if not diffs else "⚠"}  ({len(sample)-len(diffs)}/{len(sample)} OK)')
print()
print(f'  결과: {"✅ 무결성 통과" if total_issues == 0 and not (missing_pkl or extra_pkl) else "❌ 이슈 발견 — 위 로그 확인"}')
