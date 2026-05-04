"""
05a_v2_weighting.ipynb 전체 검토 — 노트북 §1-§7 모두 실행하여 오류/이상 진단.
"""
import sys, io, time, json, warnings, traceback, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
warnings.filterwarnings('ignore')

from pathlib import Path
NB_DIR = Path(__file__).resolve().parent.parent
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR
import scripts.diagnostics as diag

font_used = bootstrap()

OUT_DIR = OUTPUTS_DIR / '05a_v2_weighting'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'OUT_DIR: {OUT_DIR}')

OOS_START, OOS_END = '2010-01-01', '2024-12-31'
HOLD_START, HOLD_END = '2025-01-01', '2025-12-31'

errors = []
warnings_list = []

def section(name):
    print()
    print('=' * 75)
    print(f'  {name}')
    print('=' * 75)

# ─── §1 환경 + 9 returns 로드 + weights_dict ───
section('§1. 환경 + 9 returns + weights_dict 로드')
t0 = time.time()
try:
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
    print(f'  returns 로드: {len(returns_dict)} / {len(scenarios)}')

    SCENARIOS_6 = [
        'BL_ml_sw_mcap', 'BL_ml_sw_eq', 'BL_ml_sw_rp',
        'BL_trailing_mcap', 'BL_trailing_eq', 'BL_trailing_rp',
    ]

    market = pd.read_csv(DATA_DIR / 'market_data.csv', index_col='date', parse_dates=True)
    _reb_all = market.groupby(market.index.to_period('M')).tail(1).index
    oos_dates = _reb_all[(_reb_all >= OOS_START) & (_reb_all <= OOS_END)]
    holdout_dates = _reb_all[(_reb_all >= HOLD_START) & (_reb_all <= HOLD_END)]
    all_reb_dates = pd.DatetimeIndex(list(oos_dates) + list(holdout_dates))
    spy_monthly = market['SPY'].reindex(all_reb_dates, method='ffill').pct_change().dropna()
    print(f'  spy_monthly: {len(spy_monthly)} 개월')

    weights_path = DATA_DIR / 'scenario_weights_03_v2.pkl'
    with open(weights_path, 'rb') as f:
        weights_pkl = pickle.load(f)
    scenario_weights = weights_pkl.get('scenario_weights', {})
    print(f'  weights_dict: {len(scenario_weights)} 시나리오')
    print(f'  ✅ 정상 ({time.time()-t0:.1f}s)')
except Exception as e:
    errors.append(('§1', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §2 Layer 2 메트릭 ───
section('§2. Layer 2 — 6 시나리오 + 3 baseline 메트릭')
try:
    t1 = time.time()
    KEY_METRICS = ['sharpe', 'cagr', 'ann_vol', 'mdd', 'capm_alpha', 'capm_beta',
                   'information_ratio', 'sortino', 'hit_rate', 'cvar_5']

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

    all_scenarios = SCENARIOS_6 + ['EqualWeight', 'McapWeight', 'SPY']
    metrics_full, metrics_oos, metrics_hold = {}, {}, {}
    for sc in all_scenarios:
        if sc not in returns_dict:
            continue
        rets = returns_dict[sc]
        metrics_full[sc] = calc_metrics(rets, sc)
        metrics_oos[sc] = calc_metrics(rets, sc + ' (OOS)', OOS_START, OOS_END)
        metrics_hold[sc] = calc_metrics(rets, sc + ' (Hold-out)', HOLD_START, HOLD_END)

    mf_df = pd.DataFrame(metrics_full).T.round(3)
    mo_df = pd.DataFrame(metrics_oos).T.round(3)
    mh_df = pd.DataFrame(metrics_hold).T.round(3)

    # 메트릭 키 모두 존재하는지
    missing_keys_full = []
    for sc in all_scenarios:
        if sc in metrics_full:
            for k in KEY_METRICS:
                if k not in metrics_full[sc] or pd.isna(metrics_full[sc][k]):
                    missing_keys_full.append(f'{sc}.{k}')
    if missing_keys_full:
        print(f'  ⚠️ NaN 메트릭: {missing_keys_full[:5]}...')
        warnings_list.append(('§2', f'{len(missing_keys_full)} NaN metrics'))

    print(f'  계산: 9 시나리오 × 3 기간 = 27 메트릭 set ({time.time()-t1:.1f}s)')
    print(f'  Full 메트릭 sample (BL_ml_sw_mcap):')
    print(f'    sharpe={metrics_full["BL_ml_sw_mcap"]["sharpe"]:.3f}, '
          f'cagr={metrics_full["BL_ml_sw_mcap"]["cagr"]:.2f}%, '
          f'mdd={metrics_full["BL_ml_sw_mcap"]["mdd"]:.2f}%')
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§2', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §3 누적 수익 + Drawdown 시각화 ───
section('§3. 누적 수익 + Drawdown 시각화')
try:
    t1 = time.time()
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    plot_scenarios = SCENARIOS_6 + ['EqualWeight', 'McapWeight', 'SPY']

    cum_data = {}
    for sc in plot_scenarios:
        if sc not in returns_dict:
            continue
        cum = (1 + returns_dict[sc]).cumprod()
        cum_data[sc] = cum
        axes[0].plot(cum.index, cum.values, label=sc, linewidth=1.4, alpha=0.85)
        dd = (cum / cum.cummax() - 1) * 100
        axes[1].plot(dd.index, dd.values, linewidth=1.4, alpha=0.85)

    axes[0].set_yscale('log')
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / '3_cumret_drawdown_review.png'
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)

    # 누적 / drawdown 통계 출력
    print(f'  시각화 9 시나리오 OK ({time.time()-t1:.1f}s)')
    print(f'  {len(cum_data)} 시나리오 누적 시계열 생성:')
    for sc, cum in cum_data.items():
        final_val = cum.iloc[-1]
        max_dd = ((cum / cum.cummax() - 1) * 100).min()
        print(f'    {sc:20s}: 최종 {final_val:.2f}배, max DD {max_dd:.2f}%')
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§3', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §4 ML 효과 분해 ───
section('§4. ML 효과 분해 (가중치별 robustness)')
try:
    t1 = time.time()
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
    print(f'  ML 효과 (Full, Sharpe diff): mcap={mef_df.loc["sharpe", "mcap"]:+.3f}, '
          f'eq={mef_df.loc["sharpe", "eq"]:+.3f}, rp={mef_df.loc["sharpe", "rp"]:+.3f}')

    # 시각화
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    key_4 = ['sharpe', 'cagr', 'mdd', 'capm_alpha']
    for i, k in enumerate(key_4):
        ax = axes[i]
        full_vals = [ml_effect_full[w][k] for w in ['mcap', 'eq', 'rp']]
        oos_vals = [ml_effect_oos[w][k] for w in ['mcap', 'eq', 'rp']]
        hold_vals = [ml_effect_hold[w][k] for w in ['mcap', 'eq', 'rp']]
        x = np.arange(3)
        width = 0.25
        ax.bar(x - width, full_vals, width, label='Full')
        ax.bar(x, oos_vals, width, label='OOS')
        ax.bar(x + width, hold_vals, width, label='Hold-out')
        ax.set_xticks(x)
        ax.set_xticklabels(['mcap', 'eq', 'rp'])
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(k)
    plt.tight_layout()
    out_path = OUT_DIR / '4_ml_effect_review.png'
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)
    print(f'  ML 효과 분해 시각화 OK ({time.time()-t1:.1f}s)')
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§4', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §5 Layer 3 ML→BL 인과 (가중치별) ───
section('§5. Layer 3 — ML→BL 인과 (가중치별)')
try:
    t1 = time.time()
    ens_sw = pd.read_csv(DATA_DIR / 'ensemble_predictions_stockwise.csv', parse_dates=['date'])
    ens_sw = ens_sw[np.isfinite(ens_sw['y_true'])].copy()
    panel_diag = pd.read_csv(
        DATA_DIR / 'daily_panel.csv', parse_dates=['date'],
        usecols=['date', 'ticker', 'vol_21d'],
    )
    print(f'  ens_sw + panel 로드: {time.time()-t1:.1f}s')

    causality_results = {}
    for w in ['mcap', 'eq', 'rp']:
        sc = f'BL_ml_sw_{w}'
        if sc not in scenario_weights:
            continue
        wd = scenario_weights[sc]
        try:
            t2 = time.time()
            res = diag.evaluate_ml_to_bl_pipeline(
                pred_df=ens_sw,
                weights_dict=wd,
                panel=panel_diag,
                scenario_name=sc,
                pred_col='y_pred_ensemble',
                pct=0.30,
            )
            causality_results[w] = {
                'low_vol_hit_rate': res['low_vol_hit_rate'],
                'high_vol_hit_rate': res['high_vol_hit_rate'],
                'rank_consistency_mean': res['rank_consistency_timeline'].mean(),
                'p_matrix_turnover_mean': res['p_matrix_turnover'].mean(),
                'n_dates': len(res['rank_consistency_timeline']),
            }
            print(f'  {sc} 인과 분석: {time.time()-t2:.1f}s, '
                  f'low_hit={res["low_vol_hit_rate"]:.3f}, '
                  f'high_hit={res["high_vol_hit_rate"]:.3f}')
        except Exception as e:
            errors.append((f'§5/{sc}', str(e), traceback.format_exc()))
            print(f'  ❌ {sc} 오류: {e}')

    if causality_results:
        causality_df = pd.DataFrame(causality_results).T.round(4)
        print()
        print(causality_df.to_string())
    print('  ✅ 정상' if causality_results else '  ⚠️ 결과 없음')
except Exception as e:
    errors.append(('§5', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §6 Layer 4 시기별 ───
section('§6. Layer 4 — 시기별 분해')
try:
    t1 = time.time()
    PERIODS = {
        'P1 (2010-2014)': ('2010-01-01', '2014-12-31'),
        'P2 (2015-2018)': ('2015-01-01', '2018-12-31'),
        'P3 (2019-2020)': ('2019-01-01', '2020-12-31'),
        'P4 (2021-2022)': ('2021-01-01', '2022-12-31'),
        'P5 (2023-2025)': ('2023-01-01', '2025-12-31'),
    }
    period_sharpe = {}
    period_cagr = {}
    for sc in SCENARIOS_6 + ['EqualWeight', 'SPY']:
        if sc not in returns_dict:
            continue
        period_sharpe[sc] = {}
        period_cagr[sc] = {}
        rets = returns_dict[sc]
        for pname, (s, e) in PERIODS.items():
            sub = rets.loc[s:e]
            if len(sub) < 6:
                period_sharpe[sc][pname] = np.nan
                period_cagr[sc][pname] = np.nan
                continue
            m = diag.evaluate_portfolio_standalone(
                returns=sub, scenario_name=f'{sc}_{pname}',
                spy_returns=spy_monthly, rf_returns=None, weights_dict=None,
            )
            period_sharpe[sc][pname] = m.get('sharpe', np.nan)
            period_cagr[sc][pname] = m.get('cagr', np.nan)

    ps_df = pd.DataFrame(period_sharpe).T.round(3)
    pc_df = pd.DataFrame(period_cagr).T.round(3)
    print(f'  시기별 Sharpe ({time.time()-t1:.1f}s):')
    print(ps_df.to_string())

    # 시기별 ML 효과
    period_ml_effect = {}
    for w in ['mcap', 'eq', 'rp']:
        ml_sc = f'BL_ml_sw_{w}'
        tr_sc = f'BL_trailing_{w}'
        diffs = {}
        for pname in PERIODS:
            v_ml = period_sharpe[ml_sc].get(pname, np.nan)
            v_tr = period_sharpe[tr_sc].get(pname, np.nan)
            diffs[pname] = v_ml - v_tr if not (np.isnan(v_ml) or np.isnan(v_tr)) else np.nan
        period_ml_effect[w] = diffs
    pme_df = pd.DataFrame(period_ml_effect).T.round(3)
    print(f'\n  시기별 ML 효과:')
    print(pme_df.to_string())

    # 시각화 시뮬레이션
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, df, title in [(axes[0], ps_df, 'Sharpe'), (axes[1], pc_df, 'CAGR (%)')]:
        im = ax.imshow(df.values, aspect='auto', cmap='RdYlGn')
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=20, ha='right')
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out_path = OUT_DIR / '6_layer4_review.png'
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§6', str(e), traceback.format_exc()))
    print(f'  ❌ 오류: {e}')

# ─── §7 종합 요약 + JSON 저장 ───
section('§7. 종합 요약 + JSON 직렬화')
try:
    t1 = time.time()

    def to_json_safe(d):
        return {k: ({k2: float(v2) if isinstance(v2, (np.floating, np.integer)) else v2
                     for k2, v2 in v.items()} if isinstance(v, dict) else v)
                for k, v in d.items()}

    summary = {
        'notebook': '05a_v2_weighting.ipynb',
        'scenarios': SCENARIOS_6,
        'metrics_full': to_json_safe(metrics_full),
        'metrics_oos': to_json_safe(metrics_oos),
        'metrics_holdout': to_json_safe(metrics_hold),
        'ml_effect_full': {w: {k: float(v) for k, v in d.items()} for w, d in ml_effect_full.items()},
        'ml_effect_oos': {w: {k: float(v) for k, v in d.items()} for w, d in ml_effect_oos.items()},
        'ml_effect_holdout': {w: {k: float(v) for k, v in d.items()} for w, d in ml_effect_hold.items()},
    }
    json_str = json.dumps(summary, indent=2, ensure_ascii=False)
    print(f'  JSON 직렬화: {len(json_str)} chars ({time.time()-t1:.1f}s)')
    print('  ✅ 정상')
except Exception as e:
    errors.append(('§7', str(e), traceback.format_exc()))
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
        for ln in tb.split('\n')[-6:]:
            print(f'    {ln}')

if warnings_list:
    print()
    print('=== 경고 상세 ===')
    for sec, msg in warnings_list:
        print(f'  [{sec}] {msg}')

if not errors:
    print()
    print('  ✅ 모든 §1-§7 정상 작동')
    print('  ✅ Layer 2/3/4 메트릭 / 시각화 / JSON 직렬화 모두 정상')
