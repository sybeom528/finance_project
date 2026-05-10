"""
plot_helpers.py — Streamlit 대시보드용 시각화 함수.

기존 99_explore.ipynb 의 plot_single / plot_compare / plot_omega / plot_weights
함수를 .py 로 추출하고, plt.show() 대신 fig 를 반환 (st.pyplot 호환).

추가 (사용자 요청, 2026-05-07):
  - plot_compare 의 4 패널 → 6 패널 (rolling sortino + downside dev 추가)
"""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── 한글 폰트 (CLAUDE.md 표준) ─────────────────────────────────
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


# ══════════════════════════════════════════════════════════════
# 1. 단일 실험 6 panel 종합 (99_explore plot_single)
# ══════════════════════════════════════════════════════════════
def plot_single_panel(res: dict, rf: pd.Series, spy: pd.Series,
                      period: tuple[str, str], name: str,
                      figsize=(15, 10)) -> plt.Figure:
    """단일 실험 6 panel.

    1. 누적수익률 vs SPY (log scale)
    2. Drawdown
    3. 월별 수익률 분포
    4. 12m rolling Sharpe
    5. 포트폴리오 분산도 (eff_n + top10_share)
    6. Turnover 시계열
    """
    start, end = period
    ret = res['ret'].loc[start:end].dropna()
    if len(ret) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'[{name}] 기간 {period} 데이터 없음',
                ha='center', va='center', fontsize=14)
        return fig

    rf_a  = rf.reindex(ret.index).fillna(0)
    spy_a = res.get('spy_ret', spy).reindex(ret.index).fillna(0)

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'[{name}]  {ret.index[0].date()} ~ {ret.index[-1].date()} ({len(ret)} months)',
                 fontsize=13, fontweight='bold')

    # 1. 누적수익률
    ax = axes[0, 0]
    cum     = (1 + ret).cumprod()
    cum_spy = (1 + spy_a).cumprod()
    ax.plot(cum.index, cum.values, label=name, color='C0', linewidth=1.6)
    ax.plot(cum_spy.index, cum_spy.values, label='SPY', color='gray',
            linestyle='--', linewidth=1.2)
    ax.set_title('누적수익률 (1.0 시작, log scale)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    dd = (cum - cum.cummax()) / cum.cummax()
    ax.fill_between(dd.index, dd.values, 0, color='C3', alpha=0.4)
    ax.plot(dd.index, dd.values, color='C3', linewidth=1.0)
    ax.set_title(f'Drawdown (MDD = {dd.min()*100:.2f}%)')
    ax.grid(True, alpha=0.3)

    # 3. 월별 수익률 분포
    ax = axes[1, 0]
    ax.hist(ret.values * 100, bins=40, color='C0', alpha=0.7,
            edgecolor='black', linewidth=0.4)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(ret.mean() * 100, color='C3', linestyle='--', linewidth=1.0,
               label=f'평균 {ret.mean()*100:.2f}%')
    ax.set_title('월별 수익률 분포 (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. 12m rolling Sharpe
    ax = axes[1, 1]
    excess = ret - rf_a
    rs = excess.rolling(12).mean() / excess.rolling(12).std() * np.sqrt(12)
    ax.plot(rs.index, rs.values, color='C2', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axhline(rs.mean(), color='C3', linestyle='--', linewidth=0.8,
               label=f'평균 {rs.mean():.2f}')
    ax.set_title('12m rolling Sharpe')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. eff_n + top10
    ax = axes[2, 0]
    comp = res.get('comp', pd.DataFrame())
    if isinstance(comp, pd.DataFrame) and len(comp) > 0:
        comp_sub = comp.loc[start:end]
        if 'eff_n' in comp_sub.columns:
            ax.plot(comp_sub.index, comp_sub['eff_n'].values,
                    color='C0', linewidth=1.0, label='effective N')
            ax.set_ylabel('eff_n', color='C0')
            if 'top10_share' in comp_sub.columns:
                ax2 = ax.twinx()
                ax2.plot(comp_sub.index, comp_sub['top10_share'].values * 100,
                         color='C1', linewidth=1.0, label='top10 share')
                ax2.set_ylabel('top10 share (%)', color='C1')
            ax.set_title(f'분산도 (eff_n 평균 {comp_sub["eff_n"].mean():.0f})')
    ax.grid(True, alpha=0.3)

    # 6. turnover
    ax = axes[2, 1]
    if isinstance(comp, pd.DataFrame) and 'turnover' in comp.columns:
        comp_sub = comp.loc[start:end]
        ax.bar(comp_sub.index, comp_sub['turnover'].values,
               color='C0', alpha=0.6, width=20)
        avg_to = comp_sub['turnover'].mean()
        tc_cum = comp_sub['tc_cost'].sum() if 'tc_cost' in comp_sub.columns else 0
        ax.set_title(f'월별 turnover (평균 {avg_to:.2f}, tc 누적 {tc_cum*100:.2f}%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 2. 다중 실험 비교 6 panel (Sortino 강조, 사용자 요청 확장본)
# ══════════════════════════════════════════════════════════════
def plot_compare_panel(rets_dict: dict[str, pd.Series],
                       rf: pd.Series, spy: pd.Series,
                       period: tuple[str, str],
                       include_spy: bool = True,
                       figsize=(15, 13)) -> plt.Figure:
    """여러 실험 비교 6 panel — Sortino 강조 (2026-05-07 확장).

    1. 누적수익률 (log)
    2. Drawdown
    3. metrics bar (sortino / cagr×5 / |mdd|×5)
    4. 12m rolling Sortino  ← 신규 (사용자 요청)
    5. 12m rolling Sharpe   ← 보조
    6. Downside deviation timeline (12m rolling) ← 신규
    """
    if not rets_dict:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, '비교할 실험 없음', ha='center', va='center', fontsize=14)
        return fig

    start, end = period
    rets = {n: r.loc[start:end].dropna() for n, r in rets_dict.items()}
    rets = {n: r for n, r in rets.items() if len(r) >= 6}
    if not rets:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'기간 {period} 에 충분한 데이터 없음',
                ha='center', va='center', fontsize=14)
        return fig

    # 공통 인덱스
    idx_all = sorted(set().union(*[r.index for r in rets.values()]))
    rf_a  = rf.reindex(idx_all).fillna(0)
    spy_a = spy.reindex(idx_all).fillna(0)

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'실험 비교 ({len(rets)}개) | 기간 {period[0]} ~ {period[1]}',
                 fontsize=13, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rets), 10)))

    # 1. 누적수익률
    ax = axes[0, 0]
    for i, (n, r) in enumerate(rets.items()):
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, label=n, color=colors[i], linewidth=1.4)
    if include_spy and len(spy_a.dropna()) > 0:
        cum_spy = (1 + spy_a).cumprod()
        ax.plot(cum_spy.index, cum_spy.values, label='SPY', color='black',
                linestyle='--', linewidth=1.0, alpha=0.6)
    ax.set_title('누적수익률 (log)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    for i, (n, r) in enumerate(rets.items()):
        cum = (1 + r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax.plot(dd.index, dd.values, label=n, color=colors[i], linewidth=1.0)
    ax.set_title('Drawdown')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. metrics bar (Sortino / CAGR×5 / |MDD|×5)
    ax = axes[1, 0]
    rows = []
    for n, r in rets.items():
        rf_b = rf.reindex(r.index).fillna(0)
        exc  = r - rf_b
        ann  = exc.mean() * 12
        down = r[r < 0].std() * np.sqrt(12)
        sortino = float(ann / down) if (down and down > 0) else 0.0
        cum = (1 + r).cumprod()
        mdd = float((cum / cum.cummax() - 1).min())
        cagr = float(cum.iloc[-1] ** (12 / len(r)) - 1) if cum.iloc[-1] > 0 else 0.0
        rows.append({'name': n, 'sortino': sortino, 'cagr': cagr, 'mdd': mdd})
    mdf = pd.DataFrame(rows)
    x = np.arange(len(mdf))
    w = 0.27
    ax.bar(x - w, mdf['sortino'], w, label='Sortino',  color='C0')
    ax.bar(x,     mdf['cagr']*5,  w, label='CAGR×5',   color='C2')
    ax.bar(x + w, -mdf['mdd']*5,  w, label='|MDD|×5', color='C3')
    ax.set_xticks(x)
    ax.set_xticklabels(mdf['name'].values, rotation=45, ha='right', fontsize=8)
    ax.set_title('Sortino / CAGR×5 / |MDD|×5')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 12m rolling Sortino (신규)
    ax = axes[1, 1]
    for i, (n, r) in enumerate(rets.items()):
        rf_b = rf.reindex(r.index).fillna(0)
        exc  = r - rf_b
        # rolling sortino — exc.mean / downside_std (12m window)
        def _roll_sortino(window):
            mu = window.mean() * 12
            neg = window[window < 0]
            if len(neg) == 0:
                return np.nan
            ds = neg.std() * np.sqrt(12)
            return mu / ds if ds > 0 else np.nan
        roll_so = exc.rolling(12).apply(_roll_sortino, raw=False)
        ax.plot(roll_so.index, roll_so.values, label=n, color=colors[i], linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_title('12m rolling Sortino')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. 12m rolling Sharpe (보조)
    ax = axes[2, 0]
    for i, (n, r) in enumerate(rets.items()):
        rf_b = rf.reindex(r.index).fillna(0)
        exc  = r - rf_b
        rs = exc.rolling(12).mean() / exc.rolling(12).std() * np.sqrt(12)
        ax.plot(rs.index, rs.values, label=n, color=colors[i], linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_title('12m rolling Sharpe')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Downside deviation timeline (신규)
    ax = axes[2, 1]
    for i, (n, r) in enumerate(rets.items()):
        # 12m rolling 하방편차 (annualized): r 의 음수 sample 들의 std × √12
        def _down_dev(window):
            neg = window[window < 0]
            return neg.std() * np.sqrt(12) if len(neg) > 1 else np.nan
        dd_ts = r.rolling(12).apply(_down_dev, raw=False) * 100
        ax.plot(dd_ts.index, dd_ts.values, label=n, color=colors[i], linewidth=1.0)
    ax.set_title('12m rolling 하방편차 (%, annualized)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 3. EWMA omega 진단 (99_explore plot_omega)
# ══════════════════════════════════════════════════════════════
def plot_omega_panel(res: dict, name: str, period: tuple[str, str],
                     figsize=(15, 10)) -> plt.Figure:
    """EWMA 실험의 omega·view_e 진단 4 panel.

    EWMA 가 아닌 (meta 에 omega/view_e 컬럼 없는) 실험은 안내 문구.
    """
    meta = res.get('meta', pd.DataFrame())
    needed = ['omega', 'view_pred_ret', 'view_real_ret', 'view_e']
    if not isinstance(meta, pd.DataFrame) or any(c not in meta.columns for c in needed):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'[{name}] meta에 omega/view_e 컬럼 없음 (EWMA 실험만 지원)',
                ha='center', va='center', fontsize=12)
        return fig

    start, end = period
    meta = meta.loc[start:end]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'[{name}] EWMA 진단', fontsize=13, fontweight='bold')

    # 1. omega 시계열
    ax = axes[0, 0]
    omega = meta['omega'].dropna()
    ax.plot(omega.index, omega.values, color='C0', linewidth=1.0)
    ax.axhline(omega.mean(), color='C3', linestyle='--', linewidth=0.8,
               label=f'평균 {omega.mean():.5f}')
    ax.set_title('Ω 시계열')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. view_e
    ax = axes[0, 1]
    e = meta['view_e'].dropna()
    ax.plot(e.index, e.values * 100, color='C2', linewidth=0.9)
    ax.fill_between(e.index, e.values * 100, 0, alpha=0.2, color='C2')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_title(f'view_e (편향 {e.mean()*100:.4f}%, std {e.std()*100:.3f}%)')
    ax.grid(True, alpha=0.3)

    # 3. pred vs real 산점도
    ax = axes[1, 0]
    pred = meta['view_pred_ret'].dropna()
    real = meta['view_real_ret'].dropna()
    common = pred.index.intersection(real.index)
    pr, rr = pred.loc[common].values * 100, real.loc[common].values * 100
    if len(pr) > 0:
        ax.scatter(pr, rr, s=15, alpha=0.5, color='C0')
        lim = max(np.abs(np.concatenate([pr, rr])).max(), 0.5)
        ax.plot([-lim, lim], [-lim, lim], color='C3', linestyle='--', linewidth=0.8, label='y=x')
        if len(pr) > 5:
            beta, alpha = np.polyfit(pr, rr, 1)
            ax.plot([-lim, lim], [beta*-lim+alpha, beta*lim+alpha],
                    color='black', linewidth=0.8, label=f'fit β={beta:.3f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('view_pred_ret (%)'); ax.set_ylabel('view_real_ret (%)')
    ax.set_title('예측 vs 실현 (Mincer-Zarnowitz)')
    ax.grid(True, alpha=0.3)

    # 4. omega 분포
    ax = axes[1, 1]
    ax.hist(omega.values, bins=30, color='C0', alpha=0.7, edgecolor='black', linewidth=0.4)
    ax.axvline(omega.mean(), color='C3', linestyle='--', linewidth=1.0)
    ax.set_title('Ω 분포')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 4. 가중치 heatmap (99_explore plot_weights)
# ══════════════════════════════════════════════════════════════
def plot_weights_panel(res: dict, name: str, period: tuple[str, str],
                       n_top: int = 20, figsize=(15, 8)) -> plt.Figure:
    """월별 종목 가중치 heatmap (top n_top 평균 비중)."""
    weights = res.get('weights', pd.DataFrame())
    if not isinstance(weights, pd.DataFrame) or len(weights) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'[{name}] weights 없음', ha='center', va='center', fontsize=14)
        return fig
    start, end = period
    weights = weights.loc[start:end]
    if len(weights) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'[{name}] 기간 {period} weights 비어있음',
                ha='center', va='center', fontsize=14)
        return fig

    avg_w = weights.mean(axis=0).sort_values(ascending=False).head(n_top)
    top_tix = avg_w.index.tolist()
    w_top = weights[top_tix]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(w_top.T.values, aspect='auto', cmap='viridis',
                   extent=[mdates.date2num(w_top.index[0]),
                           mdates.date2num(w_top.index[-1]),
                           len(top_tix) - 0.5, -0.5])
    ax.set_yticks(range(len(top_tix)))
    ax.set_yticklabels(top_tix, fontsize=8)
    ax.xaxis_date()
    ax.set_title(f'[{name}] 평균 비중 상위 {n_top} 종목 가중치 시계열')
    fig.colorbar(im, ax=ax, label='weight')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 5. 파이프라인 다이어그램 (Home 페이지 hero)
# ══════════════════════════════════════════════════════════════
def render_pipeline_diagram_md() -> str:
    """7-Step 파이프라인 markdown — Home 페이지 hero."""
    return '''
```
1. Data Collection      ✅  monthly_panel + daily_returns + ensemble (2025-12 cover)
2. Preprocessing/EDA    ✅  02_LowRiskAnomaly.ipynb — quintile sort 검증
3. LSTM σ 예측           ✅  Phase 1.5 — V4_BEST + HAR-RV + Diebold-Pauly Ensemble
4. Black-Litterman       ✅  99_run.ipynb — 165 실험 walk_forward (192m)
5. MVO + 위험성향 매핑    ✅  99_analyze.ipynb K1~K7 + 3-레짐 안정성
6. 학술 통계 검증         ✅  04_Statistical_Validation.ipynb (η², Welch, KW, Heavy-tail)
7. Streamlit 대시보드    ✅  app/streamlit_app.py — 6 페이지 인터랙티브 시연
```
'''
