"""
_extend_bl_to_2025.py — BL pkl 의 ret 시리즈 incremental 확장 / 재계산.

두 가지 모드:
  1. 기본 (--mode=extend) : 2024-12 까지인 pkl 을 2025-12 로 확장 (12 month append)
  2. 재계산 (--mode=recompute_dec_2025) : 이미 192m 인 pkl 의 마지막 12 시점만 재계산
     (LSTM 예측 (ensemble csv) 가 갱신된 경우 사용)

기존 prev_w, ff3_paper 의 _op_q_prev/_op_P_prev 상태를 정확히 재구성하여
full re-run 과 동등한 결과 보장.

전제 조건:
  - final/data/monthly_panel.csv 가 2025-12-31 까지 cover
  - final/results/*.pkl 156 개가 학습된 상태

소요 시간 예상:
  - monthly_cache build (192 months): ~10 분 (cache 미존재 시)
  - 156 cfgs × 12 months: ~15 분
  - 전체: ~25-30 분

실행:
    cd final && python _extend_bl_to_2025.py                     # 기본 (extend)
    cd final && python _extend_bl_to_2025.py --recompute-2025    # LSTM 갱신 후 재계산
"""
import io
import sys
import time
import pickle
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR    = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

NB_PATH       = BASE_DIR / '99_run.ipynb'
RESULTS_DIR   = BASE_DIR / 'results'
BACKUP_DIR    = RESULTS_DIR / '_backup_pre_extension'

PRED_END    = '2025-12-31'
PHASE0_TS   = pd.Timestamp(PRED_END)

# ── 모드 결정 ──────────────────────────────────────────────────
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--recompute-2025', action='store_true',
                     help='LSTM 갱신 후 192m pkl 의 2025년 12 시점 재계산 (truncate + extend)')
_args, _ = _parser.parse_known_args()
RECOMPUTE_2025 = _args.recompute_2025

if RECOMPUTE_2025:
    print('━' * 72)
    print(' MODE: recompute-2025 (LSTM 갱신 반영 — pkl 의 2025년 12 시점 재계산)')
    print('━' * 72)
else:
    print('━' * 72)
    print(' MODE: extend (180m → 192m, 12 month append)')
    print('━' * 72)

# ── 1. 99_run 노트북 cells 0~6 (실행 전 6개) 실행하여 dispatcher + walk_forward 로드 ──
print('[1/4] 99_run.ipynb cells 0-6 실행 (data load + dispatcher + monthly_cache build) ...')
nb = nbformat.read(NB_PATH, as_version=4)
NS = {'__name__': '__main__', '__file__': str(NB_PATH)}

# Switch CWD into final/ so relative paths in cells (data/, results/) work
import os
os.chdir(BASE_DIR)

CELL_RANGE_TO_EXEC = list(range(7))   # cell 0~6, skip cell 7 (TEAM_RANK parallel runner)
for i in CELL_RANGE_TO_EXEC:
    cell = nb.cells[i]
    if cell.cell_type != 'code':
        continue
    print(f'  → executing cell {i} ({len(cell.source)} chars)')
    exec(cell.source, NS)

# ── 2. 필요 객체 추출 ──────────────────────────────────────────
EXPERIMENTS       = NS['EXPERIMENTS']
pred_dates_full   = NS['pred_dates']
monthly_cache     = NS['monthly_cache']
get_vol_series    = NS['get_vol_series']
build_P           = NS['build_P']
PCT_GROUP         = NS['PCT_GROUP']
get_prior_weights = NS['get_prior_weights']
compute_pi        = NS['compute_pi']
get_Q             = NS['get_Q']
get_omega         = NS['get_omega']
black_litterman   = NS['black_litterman']
optimize_portfolio = NS['optimize_portfolio']
compute_omega_he  = NS['compute_omega_he']
compute_turnover  = NS['compute_turnover']
apply_tc          = NS['apply_tc']
spy_series        = NS['spy_series']
TAU               = NS['TAU']
all_dates         = NS['all_dates']

# ── 3. 새 시점 (2025-01 ~ 2025-12) 결정 ────────────────────────
new_dates = [d for d in pred_dates_full if d >= pd.Timestamp('2025-01-01') and d <= PHASE0_TS]
print(f'\n[2/4] 새 계산 대상: {len(new_dates)} months ({new_dates[0].date()} ~ {new_dates[-1].date()})')

# ── 4. cfg 별 incremental extend 함수 ─────────────────────────
def extend_one(cfg):
    name = cfg['name']
    out  = RESULTS_DIR / f'{name}.pkl'
    if not out.exists():
        return f'[SKIP] {name}: pkl 없음 (백업 폴더 확인 필요)'

    with open(out, 'rb') as f:
        existing = pickle.load(f)

    if not isinstance(existing.get('ret'), pd.Series) or len(existing['ret']) == 0:
        return f'[SKIP] {name}: 기존 ret 비어있음'

    last_date = existing['ret'].index[-1]

    if RECOMPUTE_2025:
        # LSTM 갱신 후 — 2025년 12 시점만 재계산. 기존 pkl 의 2024-12 까지는 보존.
        # 2024-12-31 시점에 대한 state (prev_w, _op_q_prev/P_prev) 재구성 후
        # 2025 시점들 재실행
        if last_date < pd.Timestamp('2025-12-31'):
            return f'[SKIP] {name}: 192m 미만 ({last_date.date()}) — 먼저 기본 extend 모드 실행'
        # truncate to 2024-12-31, then extend
        cutoff = pd.Timestamp('2024-12-31')
        for k in ['ret', 'gross_ret', 'spy_ret']:
            if isinstance(existing.get(k), pd.Series):
                existing[k] = existing[k][existing[k].index <= cutoff]
        for k in ['weights', 'comp', 'meta']:
            if isinstance(existing.get(k), pd.DataFrame):
                existing[k] = existing[k][existing[k].index <= cutoff]
        last_date = cutoff
    else:
        if last_date >= PHASE0_TS:
            return f'[SKIP] {name}: 이미 {last_date.date()} 까지 cover'

    p_weight   = cfg.get('p_weight', 'mcap')
    q_mode     = cfg.get('q_mode', 'fixed')
    omega_mode = cfg.get('omega_mode', 'he_litterman')
    is_naive   = (q_mode == 'none')
    is_capm    = (q_mode == 'capm')
    max_w      = cfg.get('max_weight', 0.10)
    tc         = cfg.get('tc', 0.001)

    # ── prev_w 재구성: 기존 weights 의 마지막 행 ─────────
    prev_w = None
    if isinstance(existing.get('weights'), pd.DataFrame) and len(existing['weights']) > 0:
        last_w = existing['weights'].iloc[-1].dropna()
        if len(last_w) > 0:
            prev_w = last_w

    # ── ff3_paper state 재구성: _op_q_prev (meta 에서) + _op_P_prev (build_P 재계산) ─
    _op_q_prev = None
    _op_P_prev = None
    if omega_mode == 'ff3_paper' and last_date in monthly_cache:
        c = monthly_cache[last_date]
        valid_tix = c['valid_tix']
        month_df  = c['month_df']
        mcap      = c['mcap']
        vol_series = get_vol_series(cfg, month_df, last_date)
        _op_P_prev = build_P(
            vol_series.reindex(valid_tix).fillna(vol_series.median()),
            mcap, pct=PCT_GROUP, weighting=p_weight)
        if last_date in existing['meta'].index:
            qv = existing['meta'].loc[last_date, 'Q']
            if pd.notna(qv):
                _op_q_prev = float(qv)

    # ── 12 시점 loop ─────────────────────────────────────
    new_ret_list, new_gross_list, new_spy_list = [], [], []
    new_comp_list, new_meta_list, new_err_list = [], [], []
    new_weights = {}

    for pred_date in new_dates:
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
            w_mkt = get_prior_weights(cfg, valid_tix, mcap, vol=month_df['vol_21d'])
            pi, lam = compute_pi(Sigma, w_mkt, spy_excess, sigma2_mkt)

            vol_series = get_vol_series(cfg, month_df, pred_date)
            P = build_P(
                vol_series.reindex(valid_tix).fillna(vol_series.median()),
                mcap, pct=PCT_GROUP, weighting=p_weight)

            if is_capm:
                w = optimize_portfolio(pi, Sigma.values, lam, max_w)
                mu_meta = None
            elif is_naive:
                n_g = max(1, int(len(valid_tix) * PCT_GROUP))
                vol_v = vol_series.reindex(valid_tix)
                low_tix = vol_v.sort_values().index[:n_g]
                if p_weight == 'mcap':
                    w_low = mcap.reindex(low_tix); w_low = w_low / w_low.sum()
                elif p_weight == 'rp':
                    inv_v = (1.0 / vol_v[low_tix]).replace(0, np.nan).dropna()
                    w_low = inv_v / inv_v.sum()
                else:
                    w_low = pd.Series(1.0 / len(low_tix), index=low_tix)
                w = w_low.reindex(valid_tix).fillna(0)
                w = w.clip(upper=max_w); w = w / w.sum()
                mu_meta = None
            else:
                lam_q = float(np.clip(spy_excess / sigma2_mkt, 0.5, 10.0)) if sigma2_mkt > 1e-10 else 2.5
                Q = get_Q(cfg, P, valid_tix, train_dates, pred_date, all_dates,
                          lam=lam_q, spy_excess=spy_excess, sigma2_mkt=sigma2_mkt)
                if omega_mode == 'ff3_paper':
                    if _op_q_prev is not None and _op_P_prev is not None:
                        actual_p = float(month_df['ret_1m'].reindex(_op_P_prev.index).fillna(0) @ _op_P_prev)
                        omega = max((_op_q_prev - actual_p) ** 2, 1e-8)
                    else:
                        omega = compute_omega_he(P, Sigma, TAU)
                    _op_q_prev = Q
                    _op_P_prev = P.copy()
                else:
                    omega = get_omega(cfg, P, Sigma, pred_date)
                mu_BL, sig_BL = black_litterman(pi, Sigma, P, Q, omega, TAU)
                w = optimize_portfolio(mu_BL, Sigma.values, lam, max_w)
                mu_meta = Q

            actual_ret = month_df['fwd_ret_1m'].reindex(valid_tix).fillna(0)
            gross_ret  = float(w @ actual_ret)
            turnover   = compute_turnover(w, prev_w) if prev_w is not None else 0.0
            net_ret    = apply_tc(gross_ret, turnover, tc)
            r_spy      = float(spy_series.get(next_date, np.nan)) if next_date else np.nan

            new_ret_list.append({'date': pred_date, 'ret': net_ret})
            new_gross_list.append({'date': pred_date, 'ret': gross_ret})
            new_spy_list.append({'date': pred_date, 'ret': r_spy})

            vol_col = month_df['vol_21d'].reindex(valid_tix)
            n_g = max(1, int(len(valid_tix) * PCT_GROUP))
            sv = vol_col.sort_values()
            low_tix_all  = sv.index[:n_g].tolist()
            high_tix_all = sv.index[-n_g:].tolist()

            new_comp_list.append({
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
            new_meta_list.append({'date': pred_date, 'Q': mu_meta, 'lam': lam})
            new_weights[pred_date] = w
            prev_w = w
        except Exception as e:
            new_err_list.append({'date': pred_date, 'error': str(e)})

    if not new_ret_list:
        return f'[SKIP] {name}: 새 시점 결과 없음 (cache miss?)'

    # ── 결과 concat + 저장 ───────────────────────────────
    new_ret   = pd.DataFrame(new_ret_list).set_index('date')['ret']
    new_gross = pd.DataFrame(new_gross_list).set_index('date')['ret']
    new_spy   = pd.DataFrame(new_spy_list).set_index('date')['ret']
    new_comp  = pd.DataFrame(new_comp_list).set_index('date')
    new_meta  = pd.DataFrame(new_meta_list).set_index('date')
    new_weights_df = pd.DataFrame(new_weights).T

    existing['ret']       = pd.concat([existing['ret'], new_ret])
    existing['gross_ret'] = pd.concat([existing['gross_ret'], new_gross])
    existing['spy_ret']   = pd.concat([existing['spy_ret'], new_spy])
    existing['weights']   = pd.concat([existing['weights'], new_weights_df])
    existing['comp']      = pd.concat([existing['comp'], new_comp])
    existing['meta']      = pd.concat([existing['meta'], new_meta])
    existing['errors'].extend(new_err_list)

    with open(out, 'wb') as f:
        pickle.dump(existing, f)

    return (f'[OK] {name}: +{len(new_ret_list)} months '
            f'→ total {len(existing["ret"])} '
            f'({existing["ret"].index[0].date()}~{existing["ret"].index[-1].date()})')


# ── 5. 모든 cfg 순회 ───────────────────────────────────────────
print(f'\n[3/4] {len(EXPERIMENTS)} cfgs × 12 months extension ...')
print('━' * 70)
t0 = time.time()
ok_cnt, skip_cnt = 0, 0
for i, cfg in enumerate(EXPERIMENTS):
    msg = extend_one(cfg)
    elapsed_min = (time.time() - t0) / 60
    is_ok = msg.startswith('[OK]')
    if is_ok: ok_cnt += 1
    else: skip_cnt += 1
    if (i + 1) % 10 == 0 or i < 3 or not is_ok:
        print(f'  [{i+1:3}/{len(EXPERIMENTS)} | {elapsed_min:5.1f}min | OK {ok_cnt}/SKIP {skip_cnt}] {msg}')

total_min = (time.time() - t0) / 60
print('━' * 70)
print(f'[4/4] 완료: OK {ok_cnt} / SKIP {skip_cnt} (총 {len(EXPERIMENTS)})')
print(f'      총 소요: {total_min:.1f} 분')
