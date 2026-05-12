"""
analyze_plots.py — 05b_Analyze.ipynb 시각화 + 심층분석 함수.

master_table.build_master_table()로 만든 DataFrame을 입력받음.
원본 ret 시리즈가 필요한 함수는 results_dir에서 즉석 로드.

함수 (현재 활성, 2026-05-07)
----
1. plot_marginal_effects        : 슬롯별 분포 boxplot 5장 (J2)
2. plot_matrix_heatmap          : prior×pw vs q×Ω 매트릭스 + 행/열 평균 (J3)
3. plot_top_n_analysis          : Top-N equity / rolling sharpe / drawdown (J4)
4. crisis_comparison            : 위기구간 행동 비교 표 (J5)
5. benchmark_table              : 벤치마크 대비 IR / Δsharpe / Δmdd (J6)
6. plot_styled_regime_dashboard : Top N × 3 metric × 3-레짐 통합 대시보드 (K2, K2-H)

제거된 dead code (2026-05-07): plot_regime_heatmap, regime_winners_table,
styled_regime_table — K3/K5 셀 삭제 + dashboard로 대체되어 미사용.
"""
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ── 한글 폰트 (모듈 import 시 자동 적용) ───────────────────────────────
if platform.system() == 'Darwin':
    plt.rcParams['font.family']     = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['AppleGothic', 'Arial Unicode MS', 'DejaVu Sans']
elif platform.system() == 'Windows':
    plt.rcParams['font.family']     = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ── 위기구간 정의 ──────────────────────────────────────────────────────
CRISIS_PERIODS = {
    '2018Q4_변동성쇼크': ('2018-10-01', '2018-12-31'),
    '2020_COVID'      : ('2020-02-01', '2020-04-30'),
    '2022_인플레'      : ('2022-01-01', '2022-10-31'),
}

# ── 표준 벤치마크 ──────────────────────────────────────────────────────
# 비매트릭스 비교군 슬롯(baseline/capm_no_bl/naive_lowvol)은 2026-05-11 제거됨.
# 호출 시 benchmarks 인자로 매트릭스 슬롯명을 명시적으로 전달.
BENCHMARK_NAMES: list = []

# ── 3-레짐 라벨 (HMM n=3 기반, master_table.REGIMES 동기) ─────────────
REGIME_LABELS = ['R1_회복', 'R2_확장', 'R3_변동']


# ══════════════════════════════════════════════════════════════════════
# 헬퍼: 원본 ret 시리즈 로드
# ══════════════════════════════════════════════════════════════════════
def load_returns(names, results_dir) -> dict:
    """이름 리스트 → {name: ret 시리즈} dict."""
    results_dir = Path(results_dir)
    out = {}
    for n in names:
        p = results_dir / f'{n}.pkl'
        if not p.exists():
            print(f'[skip] {n}: pkl 없음')
            continue
        with open(p, 'rb') as f:
            res = pickle.load(f)
        ret = res.get('ret', pd.Series(dtype=float))
        if isinstance(ret, pd.Series) and len(ret.dropna()) > 0:
            out[n] = ret.dropna()
    return out


# ══════════════════════════════════════════════════════════════════════
# 1. 슬롯별 marginal effect (boxplot)
# ══════════════════════════════════════════════════════════════════════
def plot_marginal_effects(
    mt: pd.DataFrame,
    metric: str = 'sortino',   # 사용자 요청: Sortino 우선 (2026-05-07)
    slots: list = None,
    figsize=(13, 9),
    save_path=None,
):
    """
    각 슬롯값별 metric 분포를 boxplot으로 비교. 2×2 레이아웃 (§2.1 turnover boxplot 와 통일, 2026-05-12).

    slots: None이면 ['prior_s','pw_s','q_s','om_s'] 4개 (p_s = 'ls' 단일이라 제외, §2.1 와 동일 순서).
    """
    if slots is None:
        slots = ['prior_s', 'pw_s', 'q_s', 'om_s']

    # §2.1 와 동일한 짧은 라벨
    slot_label = {
        'prior_s': 'Prior',
        'p_s'    : 'P_mode',
        'pw_s'   : 'P_weight',
        'q_s'    : 'Q_mode',
        'om_s'   : 'Omega',
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for i, slot in enumerate(slots):
        ax = axes[i]
        # 그룹별 mean 기준 정렬 (descending — high metric = better)
        order = mt.groupby(slot)[metric].mean().sort_values(ascending=False).index.tolist()
        data  = [mt[mt[slot] == k][metric].dropna().values for k in order]
        counts = [len(d) for d in data]

        bp = ax.boxplot(data, labels=order, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red',
                                       markeredgecolor='red', markersize=6))
        for patch in bp['boxes']:
            patch.set_facecolor('#9ecae1')
            patch.set_alpha(0.7)

        ax.set_title(f'{slot_label.get(slot, slot)}  (n={counts})', fontsize=11)
        ax.set_ylabel(f'{metric} (월평균)' if metric == 'turnover' else metric)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(alpha=0.3, axis='y')

        # 전체 중앙값 가로선 + 레전드 (§2.1 와 동일)
        overall = mt[metric].median()
        ax.axhline(overall, color='gray', linestyle='--', alpha=0.5, lw=1,
                   label=f'전체 중앙값={overall:.3f}')
        ax.legend(fontsize=8, loc='upper right')

    # 남는 axes 숨김 (slots < 4 인 경우)
    for j in range(len(slots), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'슬롯별 {metric} 분포 (red ◆=평균, 회색선=전체 중앙값)',
                 fontsize=13, y=1.00)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


# ══════════════════════════════════════════════════════════════════════
# 2. 108-cell 매트릭스 히트맵
# ══════════════════════════════════════════════════════════════════════
def plot_matrix_heatmap(
    mt: pd.DataFrame,
    metric: str = 'sortino',   # 사용자 요청: Sortino 우선 (2026-05-07)
    only_lstm: bool = True,
    only_matrix_cells: bool = False,
    row_keys=('prior_s', 'pw_s'),
    col_keys=('q_s', 'om_s'),
    figsize=(18, 9),
    cmap='RdYlGn',
    annotate: bool = True,
    fmt: str = '.2f',
    save_path=None,
):
    """
    행=prior×p_weight, 열=q×Ω 매트릭스 히트맵.
    + 행 평균 / 열 평균 막대.

    only_matrix_cells=True면 정식 108셀(prior 3 × pw 3 × q 4 × Ω 3)만.
    """
    sub = mt.copy()
    if only_lstm:
        sub = sub[sub.p_s == 'ls']

    if only_matrix_cells:
        sub = sub[
            sub.prior_s.isin(['mcap','eq','rp']) &
            sub.pw_s.isin(['mcap','eq','rp']) &
            sub.q_s.isin(['fix','lam','raw','inv']) &
            sub.om_s.isin(['he','pap','rms'])
        ]

    sub = sub.copy()
    sub['_row'] = sub[list(row_keys)].agg('_'.join, axis=1)
    sub['_col'] = sub[list(col_keys)].agg('_'.join, axis=1)
    pv = sub.pivot_table(index='_row', columns='_col', values=metric, aggfunc='mean')

    # 정렬: 의미 있는 순서로
    row_order = [f'{p}_{w}' for p in ['mcap','eq','rp']
                            for w in ['mcap','eq','rp','volm']
                            if f'{p}_{w}' in pv.index]
    row_order += [r for r in pv.index if r not in row_order]
    pv = pv.reindex(row_order)

    col_order = [f'{q}_{o}' for q in ['fix','lam','raw','inv','vsp','ff3']
                            for o in ['he','pap','rms','sh','sd']
                            if f'{q}_{o}' in pv.columns]
    col_order += [c for c in pv.columns if c not in col_order]
    pv = pv[col_order]

    # 행/열 평균
    row_mean = pv.mean(axis=1)
    col_mean = pv.mean(axis=0)

    # ── plot: 메인 + 우측(행평균) + 하단(열평균) ─────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 2, width_ratios=[20, 1.5], height_ratios=[20, 1.5],
                           hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_row  = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_col  = fig.add_subplot(gs[1, 0], sharex=ax_main)

    vmin, vmax = np.nanmin(pv.values), np.nanmax(pv.values)
    im = ax_main.imshow(pv.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax_main.set_xticks(range(len(pv.columns)))
    ax_main.set_xticklabels(pv.columns, rotation=30, ha='right', fontsize=11)
    ax_main.set_yticks(range(len(pv.index)))
    ax_main.set_yticklabels(pv.index, fontsize=11)
    ax_main.set_xlabel('q_mode × omega', fontsize=12)
    ax_main.set_ylabel('prior × p_weight', fontsize=12)

    if annotate:
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if not np.isnan(v):
                    ax_main.text(j, i, format(v, fmt),
                                 ha='center', va='center', fontsize=8,
                                 color='black')

    # 우측: 행 평균
    ax_row.barh(range(len(row_mean)), row_mean.values, color='#3182bd', alpha=0.8)
    ax_row.set_xticks([])
    ax_row.set_yticks([])
    ax_row.set_xlabel('row avg', fontsize=8)
    for i, v in enumerate(row_mean.values):
        if not np.isnan(v):
            ax_row.text(v, i, f' {v:.2f}', va='center', fontsize=7)

    # 하단: 열 평균
    ax_col.bar(range(len(col_mean)), col_mean.values, color='#3182bd', alpha=0.8)
    ax_col.set_yticks([])
    ax_col.set_xticks([])
    ax_col.set_ylabel('col avg', fontsize=8)
    for j, v in enumerate(col_mean.values):
        if not np.isnan(v):
            ax_col.text(j, v, f'{v:.2f}', ha='center', fontsize=7,
                        rotation=90, va='bottom')

    # colorbar를 메인 위쪽 가로로 배치 (좌측에 두면 yticklabels 자리 침범)
    cbar = fig.colorbar(im, ax=ax_main, fraction=0.04, pad=0.04,
                        orientation='horizontal', location='top')
    cbar.set_label(metric, fontsize=10)

    suffix = ' [LSTM 전용]' if only_lstm else ''
    suffix += ' [108셀만]' if only_matrix_cells else ''
    fig.suptitle(f'매트릭스 히트맵: {metric}{suffix}\n'
                 f'행 = prior × p_weight (9 행), 열 = q_mode × omega (12 열)\n'
                 f'각 셀 = 해당 슬롯 조합의 평균 {metric}',
                 fontsize=12, y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig, pv


# ══════════════════════════════════════════════════════════════════════
# 3. Top-N 정밀 분석
# ══════════════════════════════════════════════════════════════════════
def plot_top_n_analysis(
    mt: pd.DataFrame,
    results_dir,
    spy_ret: pd.Series,
    rf: pd.Series,
    n: int = 5,
    metric: str = 'sortino',   # 사용자 요청: Sortino 우선 (2026-05-07)
    top_names: list = None,
    rolling_window: int = 12,
    figsize=(15, 10),
    save_path=None,
    title_suffix: str = '',
):
    """
    Top-N의 Equity Curve / Rolling Sharpe / Drawdown 3-panel figure.
    SPY도 함께 표시.

    Parameters
    ----------
    top_names : list[str], optional
        외부에서 미리 선정한 Top-N 후보 name 리스트 (rt 기반 sortino_ir 등 정렬에서 활용).
        지정 시 metric 기준 nlargest 무시.
    title_suffix : str
        그림 제목에 추가할 문구 (예: ' — sortino_ir 정렬').
    """
    if top_names is not None:
        top = mt[mt.name.isin(top_names)].copy()
    else:
        top = mt.nlargest(n, metric)
    names = top['name'].tolist()
    rets  = load_returns(names, results_dir)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ax_eq, ax_sh, ax_dd = axes

    colors = plt.cm.tab10(np.linspace(0, 1, len(rets) + 1))

    # ── Equity Curve (누적수익 1.0 시작) ────────────────────────
    for (name, ret), c in zip(rets.items(), colors):
        cum = (1 + ret).cumprod()
        canonical = mt[mt.name == name].iloc[0]['canonical']
        sr        = mt[mt.name == name].iloc[0]['sharpe']
        ax_eq.plot(cum.index, cum.values, label=f'{canonical} (Sh={sr:.2f})', lw=1.5, color=c)
    cum_spy = (1 + spy_ret).cumprod()
    ax_eq.plot(cum_spy.index, cum_spy.values, 'k--', lw=1.5, label='SPY', alpha=0.7)
    ax_eq.set_ylabel('Equity (1.0 시작)')
    ax_eq.set_title(f'Top-{len(rets)} 누적수익 곡선{title_suffix}', fontsize=12)
    ax_eq.legend(loc='upper left', fontsize=8)
    ax_eq.grid(True, alpha=0.3)
    ax_eq.set_yscale('log')

    # ── Rolling Sharpe (12-month) ───────────────────────────────
    for (name, ret), c in zip(rets.items(), colors):
        rf_a = rf.reindex(ret.index).fillna(0)
        exc  = ret - rf_a
        rs   = (exc.rolling(rolling_window).mean() * 12) / (ret.rolling(rolling_window).std() * np.sqrt(12))
        canonical = mt[mt.name == name].iloc[0]['canonical']
        ax_sh.plot(rs.index, rs.values, lw=1.2, label=canonical, color=c)
    ax_sh.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax_sh.set_ylabel(f'Rolling Sharpe ({rolling_window}m)')
    ax_sh.set_title(f'{rolling_window}개월 롤링 Sharpe', fontsize=12)
    ax_sh.grid(True, alpha=0.3)

    # ── Drawdown ─────────────────────────────────────────────────
    for (name, ret), c in zip(rets.items(), colors):
        cum = (1 + ret).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        canonical = mt[mt.name == name].iloc[0]['canonical']
        ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.25, color=c)
        ax_dd.plot(dd.index, dd.values, lw=1.0, color=c, label=canonical)
    ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_dd.set_ylabel('Drawdown')
    ax_dd.set_title('Drawdown 비교', fontsize=12)
    ax_dd.set_xlabel('Date')
    ax_dd.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


# ══════════════════════════════════════════════════════════════════════
# 4. 위기구간 행동 비교
# ══════════════════════════════════════════════════════════════════════
def crisis_comparison(
    mt: pd.DataFrame,
    results_dir,
    names: list = None,
    crisis_periods: dict = None,
    spy_ret: pd.Series = None,
    n_top: int = 5,
) -> pd.DataFrame:
    """
    위기구간별로 각 실험의 행동 (cum_ret, mdd_intraperiod, worst_month).

    names: None이면 mt에서 sharpe Top n_top.
    crisis_periods: None이면 CRISIS_PERIODS.
    """
    if crisis_periods is None:
        crisis_periods = CRISIS_PERIODS
    if names is None:
        names = mt.nlargest(n_top, 'sharpe')['name'].tolist()

    rets = load_returns(names, results_dir)
    if spy_ret is not None:
        rets['_SPY'] = spy_ret

    rows = []
    for label, (s, e) in crisis_periods.items():
        for name, ret in rets.items():
            sub = ret[(ret.index >= s) & (ret.index <= e)]
            if len(sub) == 0:
                continue
            cum         = (1 + sub).prod() - 1
            cum_path    = (1 + sub).cumprod()
            mdd_intra   = float(((cum_path / cum_path.cummax()) - 1).min())
            worst_month = float(sub.min())
            best_month  = float(sub.max())
            canonical   = (mt[mt.name == name].iloc[0]['canonical']
                           if name != '_SPY' and (mt.name == name).any() else 'SPY')
            rows.append({
                '구간'        : label,
                'name'        : name if name != '_SPY' else 'SPY',
                'canonical'   : canonical,
                '구간_누적수익': round(cum, 4),
                '구간_MDD'    : round(mdd_intra, 4),
                '최악월'      : round(worst_month, 4),
                '최고월'      : round(best_month, 4),
                '월수'        : len(sub),
            })

    df = pd.DataFrame(rows)
    return df.set_index(['구간', 'canonical'])


# ══════════════════════════════════════════════════════════════════════
# 5. 벤치마크 대비 IR 표
# ══════════════════════════════════════════════════════════════════════
def benchmark_table(
    mt: pd.DataFrame,
    results_dir,
    spy_ret: pd.Series,
    rf: pd.Series,
    candidate_names: list = None,
    benchmarks: list = None,
    n_top: int = 10,
) -> pd.DataFrame:
    """
    각 후보가 각 벤치마크 대비 어떤 부가가치를 내는지.
    Active return = ret - bench_ret (월별)
    IR = annualized active mean / annualized active std

    Parameters
    ----------
    candidate_names : None이면 mt sharpe Top n_top.
    benchmarks      : None이면 BENCHMARK_NAMES + 'SPY'.

    Returns
    -------
    DataFrame: index=candidate, columns=각 벤치마크별 (IR, Δsharpe, Δmdd)
    """
    if benchmarks is None:
        benchmarks = BENCHMARK_NAMES
    if candidate_names is None:
        candidate_names = mt.nlargest(n_top, 'sharpe')['name'].tolist()

    # 벤치마크 ret 로드
    bench_rets = load_returns(benchmarks, results_dir)
    bench_rets['SPY'] = spy_ret.dropna()

    cand_rets = load_returns(candidate_names, results_dir)

    rows = []
    for cname, cret in cand_rets.items():
        meta_row = mt[mt.name == cname].iloc[0]
        cand_canonical = meta_row['canonical']
        cand_sharpe    = meta_row['sharpe']
        cand_mdd       = meta_row['mdd']

        out = {'name': cname, 'canonical': cand_canonical, 'sharpe': cand_sharpe}

        for bname, bret in bench_rets.items():
            common = cret.index.intersection(bret.index)
            if len(common) < 12:
                out[f'{bname}_IR']      = np.nan
                out[f'{bname}_Δsharpe'] = np.nan
                continue
            active = cret.reindex(common) - bret.reindex(common)
            ir = (active.mean() * 12) / (active.std() * np.sqrt(12)) if active.std() > 0 else np.nan

            # 벤치마크 sharpe / mdd
            rf_a   = rf.reindex(common).fillna(0)
            bexc   = bret.reindex(common) - rf_a
            bvol   = bret.reindex(common).std() * np.sqrt(12)
            bsh    = (bexc.mean()*12) / bvol if bvol > 0 else np.nan
            bcum   = (1 + bret.reindex(common)).cumprod()
            bmdd   = float(((bcum / bcum.cummax()) - 1).min())

            out[f'{bname}_IR']      = round(ir, 3)
            out[f'{bname}_Δsharpe'] = round(cand_sharpe - bsh, 3)
            out[f'{bname}_Δmdd']    = round(cand_mdd - bmdd, 3)

        rows.append(out)

    return pd.DataFrame(rows).set_index('canonical')


# ══════════════════════════════════════════════════════════════════════
# 6. 레짐별 안정성 분석 — Top 20 × 3 metric 통합 대시보드
# ══════════════════════════════════════════════════════════════════════

def plot_styled_regime_dashboard(
    rt: pd.DataFrame,
    rank_by: str = 'sortino_ir',
    top_n: int = 20,
    regime_labels: list = None,
    save_path=None,
    winner_name: str = None,
    threshold_rank: int = None,
    threshold_label: str = None,
):
    """
    Top N × 3 metric (sortino/sharpe/mdd) × 3 레짐을 단일 matplotlib 그림으로.
    1 figure, 3 panel 병치.

    Parameters
    ----------
    winner_name : str | None
        자동 식별된 winner 의 pkl `name` (e.g., 'mat_eq_eq_raw_pap').
        주어지면 y축 라벨에서 매칭되는 행 끝에 ⭐ 표시.
    threshold_rank : int | None
        가로 구분선을 그을 rank (1-base). e.g., 13 이면 rank 13 과 rank 14 사이에 점선.
    threshold_label : str | None
        구분선 위쪽에 표시할 캡션.

    Returns matplotlib Figure (PNG로 저장 가능).
    """
    if regime_labels is None:
        regime_labels = REGIME_LABELS

    top = rt.nlargest(top_n, rank_by).reset_index(drop=True)
    canonicals = top['canonical'].tolist()
    names = top['name'].tolist() if 'name' in top.columns else [None] * top_n
    rank_labels = []
    for i, (nm, c) in enumerate(zip(names, canonicals)):
        # ★ (U+2605, AppleGothic / NanumGothic 등 한글 폰트에 글리프 있음).
        # ⭐ (U+2B50) 은 emoji 폰트 필요해서 matplotlib 한글 폰트에서 □ 로 깨짐.
        marker = ' ★' if (winner_name and nm == winner_name) else ''
        rank_labels.append(f'{i+1:>2}. {c}{marker}')

    # x축 라벨 + colorbar 분리 공간 확보를 위해 세로 여백 +3 (기존 +2)
    height = max(10, top_n * 0.45 + 3)
    fig, axes = plt.subplots(1, 3, figsize=(19, height), sharey=True,
                              gridspec_kw={'wspace': 0.06})

    for ax, metric in zip(axes, ['sortino', 'sharpe', 'mdd']):
        cols = [f'{metric}_{lbl}' for lbl in regime_labels]
        data = top[cols].values

        if metric == 'mdd':
            vmin = float(np.nanmin(data))
            vmax = 0.0
        else:
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(regime_labels)))
        # 회전각 30° + ha='right' → 한글 라벨 (R4_HOLD_OUT 등) 충돌 방지
        ax.set_xticklabels(regime_labels, rotation=30, ha='right', fontsize=10)
        ax.set_title(f'{metric.upper()}', fontsize=13, fontweight='bold', pad=8)

        # 셀 값 표시
        for i in range(top_n):
            for j in range(len(regime_labels)):
                v = data[i, j]
                if not np.isnan(v):
                    fmt = f'{v*100:.1f}%' if metric == 'mdd' else f'{v:.2f}'
                    ax.text(j, i, fmt, ha='center', va='center', fontsize=8)

        # colorbar (각 panel 하단) — pad 키워서 회전된 x축 라벨과 분리
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.18,
                            orientation='horizontal', location='bottom')
        cbar.ax.tick_params(labelsize=8)

    # y축 라벨 (첫 panel만, 순위 번호 포함)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(rank_labels, fontsize=10)

    # 가로 구분선 (threshold_rank) — 모든 panel 에 점선 + 첫 panel 좌측 캡션
    if threshold_rank is not None and 0 < threshold_rank < top_n:
        y_line = threshold_rank - 0.5
        for ax in axes:
            ax.axhline(y_line, color='black', linestyle='--', linewidth=1.6, alpha=0.75)
        if threshold_label:
            axes[-1].text(len(regime_labels) - 0.4, y_line, f' {threshold_label}',
                          ha='left', va='center', fontsize=10, color='black',
                          fontweight='bold')

    suptitle = (f'Top {top_n} × 3 레짐 통합 대시보드 (정렬: {rank_by})\n'
                f'좌→우: SORTINO / SHARPE / MDD  ·  녹=좋음 / 적=나쁨')
    if winner_name:
        suptitle += f'  ·  ★ = winner ({winner_name})'
    fig.suptitle(suptitle, fontsize=13, y=1.0)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig



