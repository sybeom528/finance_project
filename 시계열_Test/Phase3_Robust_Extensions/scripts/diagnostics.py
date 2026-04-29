"""Phase 3 — 평가·진단 모듈 (Layer 1~5).

신규 모듈 (2026-04-29). 모델·시나리오 단독 평가 + 시나리오 간 통계 검정.
05a/05b/05c 평가 노트북에서 호출하여 동일 형식 보장.

설계 원칙
---------
- 02a 학습 영향 0 (02a 가 import 하지 않음)
- 단일 책임 함수 (Layer 별 분리)
- 표준 출력 (DEFAULT_COLORS, METRIC_ORDER 통일)
- pd.DataFrame / dict 일관 반환

레이어
------
- Layer 1: 변동성 예측 진단 (RMSE, QLIKE, MZ, DM-test, ...)
- Layer 2: 포트폴리오 단독 (Sharpe, CAPM α, IR, Sortino, CVaR, ...)
- Layer 3: ML → BL 인과 추적 (low/high vol hit rate, P 안정성)
- Layer 4: 시기별 분해 (5 시기 × 모든 메트릭)
- Layer 5: 시나리오 간 통계 검정 (Jobson-Korkie, Memmel, MCS)

학술 근거
---------
- Patton (2011) "Volatility forecast comparison using imperfect volatility proxies" — QLIKE
- Mincer & Zarnowitz (1969) "Forecast evaluation" — MZ regression
- Diebold & Mariano (1995) "Comparing predictive accuracy" — DM test
- Jobson & Korkie (1981) "Performance hypothesis testing" — Sharpe diff
- Memmel (2003) "Performance hypothesis testing with the Sharpe ratio" — JK 보정
- Hansen, Lunde, Nason (2011) "The model confidence set" — MCS
- Sortino & Price (1994) — Sortino ratio
- Treynor & Black (1973) — Information ratio
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# 표준 헬퍼 (모든 노트북 동일 형식 보장)
# =============================================================================

DEFAULT_COLORS = {
    'BL_ml_sw': '#1f77b4',       # 파랑
    'BL_ml_cs': '#2ca02c',       # 초록
    'BL_trailing': '#d62728',    # 빨강
    'EqualWeight': '#ff7f0e',    # 주황
    'McapWeight': '#9467bd',     # 보라
    'SPY': '#8c564b',             # 갈색
    'LSTM': '#1f77b4',
    'HAR': '#d62728',
    'Ensemble': '#2ca02c',
}

# Metric 표시 순서 (모든 표 동일)
METRIC_ORDER_PORTFOLIO = [
    'sharpe', 'cagr', 'ann_vol', 'mdd',
    'capm_alpha', 'capm_beta', 'capm_t',
    'information_ratio', 'sortino', 'calmar',
    'hit_rate', 'skew', 'kurt',
    'cvar_5', 'var_5',
    'turnover', 'top10_concentration',
    'n_months',
]

METRIC_ORDER_PREDICTION = [
    'rmse', 'qlike', 'r2_train_mean',
    'mz_alpha', 'mz_beta', 'mz_r2',
    'pred_std_ratio', 'spearman',
    'dm_stat_vs_har', 'dm_pvalue_vs_har',
    'n_pairs',
]

# 5 시기 정의 (Layer 4)
PERIODS = {
    'GFC 회복 (09~11)': ('2009-01-01', '2011-12-31'),
    '정상 강세장 (12~19)': ('2012-01-01', '2019-12-31'),
    'COVID 충격 (20)': ('2020-01-01', '2020-12-31'),
    '긴축·전환 (21~22)': ('2021-01-01', '2022-12-31'),
    '회복·AI (23~25)': ('2023-01-01', '2025-12-31'),
}


def render_metrics_table(
    metrics_dict: Dict[str, dict],
    metric_order: Optional[List[str]] = None,
    sort_by: Optional[str] = 'sharpe',
    ascending: bool = False,
) -> pd.DataFrame:
    """표준 형식 메트릭 표.

    Parameters
    ----------
    metrics_dict : dict {scenario_name: {metric: value}}
    metric_order : list, default METRIC_ORDER_PORTFOLIO
    sort_by : str, sort key (None 이면 dict 순서)
    ascending : bool

    Returns
    -------
    pd.DataFrame (시나리오 × 메트릭, 표준 정렬)
    """
    df = pd.DataFrame(metrics_dict).T
    if metric_order:
        cols = [c for c in metric_order if c in df.columns]
        df = df[cols]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    return df


# =============================================================================
# 공통 메트릭 계산
# =============================================================================

def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE (NaN 제거 후)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _qlike_loss(y_true_logrv: np.ndarray, y_pred_logrv: np.ndarray) -> float:
    """QLIKE (Patton 2011) 비대칭 손실. log-RV 입력.

    QLIKE = mean(σ²_true / σ²_pred - log(σ²_true / σ²_pred) - 1)
    σ² = exp(2 · log_rv)
    """
    mask = ~(np.isnan(y_true_logrv) | np.isnan(y_pred_logrv))
    if mask.sum() == 0:
        return np.nan
    sigma2_true = np.exp(2 * y_true_logrv[mask])
    sigma2_pred = np.exp(2 * y_pred_logrv[mask])
    sigma2_pred = np.maximum(sigma2_pred, 1e-12)
    ratio = sigma2_true / sigma2_pred
    return float(np.mean(ratio - np.log(ratio) - 1))


# =============================================================================
# Layer 1 — 변동성 예측 진단
# =============================================================================

def evaluate_volatility_prediction(
    pred_df: pd.DataFrame,
    model_name: str = 'Model',
    pred_col: str = 'y_pred_ensemble',
    true_col: str = 'y_true',
    har_pred_col: Optional[str] = 'y_pred_har',
) -> Dict[str, object]:
    """⭐ Layer 1: 변동성 예측 진단 (모델 단독).

    Parameters
    ----------
    pred_df : pd.DataFrame (date, ticker, fold, y_pred_*, y_true 컬럼)
    model_name : str
    pred_col : 평가할 예측 컬럼
    true_col : 실제값 컬럼
    har_pred_col : HAR baseline 컬럼 (DM-test 용). None 이면 DM-test 생략.

    Returns
    -------
    dict {
        'overall': {rmse, qlike, r2_train_mean, mz_*, pred_std_ratio, spearman, dm_*},
        'by_ticker': pd.DataFrame (ticker × metric),
        'best_model': pd.Series (LSTM/HAR/Ensemble 분포),
        'weight_timeline': pd.DataFrame (date × {w_v4, w_har} 평균),
    }
    """
    df = pred_df.dropna(subset=[pred_col, true_col]).copy()
    if len(df) == 0:
        return {'overall': {}, 'by_ticker': pd.DataFrame(), 'best_model': pd.Series(), 'weight_timeline': pd.DataFrame()}

    y_true = df[true_col].values
    y_pred = df[pred_col].values

    # 1. RMSE
    rmse = _safe_rmse(y_true, y_pred)

    # 2. QLIKE
    qlike = _qlike_loss(y_true, y_pred)

    # 3. R²_train_mean (train 평균값 baseline)
    train_mean = np.nanmean(y_true)
    sse_model = np.sum((y_pred - y_true) ** 2)
    sse_baseline = np.sum((train_mean - y_true) ** 2)
    r2_train_mean = float(1 - sse_model / max(sse_baseline, 1e-12))

    # 4. MZ regression: y_true = α + β · y_pred
    try:
        slope, intercept, r_value, _, _ = stats.linregress(y_pred, y_true)
        mz_alpha = float(intercept)
        mz_beta = float(slope)
        mz_r2 = float(r_value ** 2)
    except Exception:
        mz_alpha = mz_beta = mz_r2 = np.nan

    # 5. pred_std_ratio (mean-collapse 진단)
    pred_std_ratio = float(np.std(y_pred) / max(np.std(y_true), 1e-12))

    # 6. Spearman rank correlation
    try:
        spearman, _ = stats.spearmanr(y_pred, y_true)
        spearman = float(spearman)
    except Exception:
        spearman = np.nan

    # 7. DM-test vs HAR (Diebold-Mariano)
    dm_stat = dm_pvalue = np.nan
    if har_pred_col and har_pred_col in df.columns:
        df_dm = df.dropna(subset=[har_pred_col])
        if len(df_dm) > 30:
            err_model = (df_dm[pred_col].values - df_dm[true_col].values) ** 2
            err_har = (df_dm[har_pred_col].values - df_dm[true_col].values) ** 2
            d = err_model - err_har
            try:
                dm_stat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
                dm_pvalue = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
            except Exception:
                pass

    overall = {
        'rmse': rmse,
        'qlike': qlike,
        'r2_train_mean': r2_train_mean,
        'mz_alpha': mz_alpha,
        'mz_beta': mz_beta,
        'mz_r2': mz_r2,
        'pred_std_ratio': pred_std_ratio,
        'spearman': spearman,
        'dm_stat_vs_har': dm_stat,
        'dm_pvalue_vs_har': dm_pvalue,
        'n_pairs': len(df),
    }

    # 종목별 RMSE/QLIKE
    by_ticker_rows = []
    for tk, g in df.groupby('ticker'):
        if len(g) < 5:
            continue
        by_ticker_rows.append({
            'ticker': tk,
            'rmse': _safe_rmse(g[true_col].values, g[pred_col].values),
            'qlike': _qlike_loss(g[true_col].values, g[pred_col].values),
            'spearman': stats.spearmanr(g[pred_col].values, g[true_col].values)[0]
                        if len(g) >= 10 else np.nan,
            'n': len(g),
        })
    by_ticker = pd.DataFrame(by_ticker_rows).set_index('ticker') if by_ticker_rows else pd.DataFrame()

    # Best model 분포 (LSTM/HAR/Ensemble RMSE 비교)
    best_model = pd.Series(dtype=int)
    if 'y_pred_lstm' in df.columns and 'y_pred_har' in df.columns:
        best_choices = []
        for tk, g in df.groupby('ticker'):
            if len(g) < 5:
                continue
            r_lstm = _safe_rmse(g[true_col].values, g['y_pred_lstm'].values)
            r_har = _safe_rmse(g[true_col].values, g['y_pred_har'].values)
            r_ens = _safe_rmse(g[true_col].values, g[pred_col].values) if pred_col != 'y_pred_lstm' else r_lstm
            best = min([('LSTM', r_lstm), ('HAR', r_har), ('Ensemble', r_ens)], key=lambda x: x[1])
            best_choices.append(best[0])
        if best_choices:
            best_model = pd.Series(best_choices).value_counts()

    # Performance weight 시계열 평균
    weight_timeline = pd.DataFrame()
    if 'w_v4' in df.columns and 'w_har' in df.columns:
        if 'date' in df.columns:
            df_w = df.copy()
            df_w['date'] = pd.to_datetime(df_w['date'])
            weight_timeline = df_w.groupby('date')[['w_v4', 'w_har']].mean()

    return {
        'overall': overall,
        'by_ticker': by_ticker,
        'best_model': best_model,
        'weight_timeline': weight_timeline,
        'model_name': model_name,
    }


def plot_prediction_diagnostic_panel(
    eval_result: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 10),
) -> plt.Figure:
    """Layer 1 시각화 — 6 panel."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    by_ticker = eval_result.get('by_ticker', pd.DataFrame())
    best_model = eval_result.get('best_model', pd.Series())
    weight_timeline = eval_result.get('weight_timeline', pd.DataFrame())
    overall = eval_result.get('overall', {})
    model_name = eval_result.get('model_name', 'Model')

    # 1. RMSE 분포 (히스토그램)
    ax = axes[0, 0]
    if len(by_ticker) > 0 and 'rmse' in by_ticker.columns:
        ax.hist(by_ticker['rmse'].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(overall.get('rmse', np.nan), color='red', linestyle='--', label=f'전체 평균: {overall.get("rmse", np.nan):.3f}')
        ax.set_title('종목별 RMSE 분포', fontsize=11)
        ax.set_xlabel('RMSE'); ax.legend(); ax.grid(alpha=0.3)

    # 2. QLIKE 분포 (boxplot)
    ax = axes[0, 1]
    if len(by_ticker) > 0 and 'qlike' in by_ticker.columns:
        ax.boxplot([by_ticker['qlike'].dropna()], labels=[model_name],
                   patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax.set_title(f'QLIKE 분포 (Patton 2011)\n전체: {overall.get("qlike", np.nan):.3f}', fontsize=11)
        ax.set_ylabel('QLIKE')
        ax.grid(alpha=0.3)

    # 3. MZ regression scatter (예측 vs 실제 — 종목별 평균)
    ax = axes[0, 2]
    mz_alpha = overall.get('mz_alpha', np.nan)
    mz_beta = overall.get('mz_beta', np.nan)
    mz_r2 = overall.get('mz_r2', np.nan)
    ax.text(0.05, 0.95, f'MZ Regression\nα = {mz_alpha:.4f}\nβ = {mz_beta:.4f}\nR² = {mz_r2:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Mincer-Zarnowitz Regression', fontsize=11)
    ax.axis('off')

    # 4. Calibration plot (decile 비교)
    ax = axes[1, 0]
    if len(by_ticker) > 0 and 'spearman' in by_ticker.columns:
        spearman_dist = by_ticker['spearman'].dropna()
        ax.hist(spearman_dist, bins=20, color='green', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.axvline(overall.get('spearman', np.nan), color='red', linestyle='--',
                   label=f'전체: {overall.get("spearman", np.nan):.3f}')
        ax.set_title('Spearman Rank 분포\n(BL P 행렬 입력 품질)', fontsize=11)
        ax.set_xlabel('Spearman ρ'); ax.legend(); ax.grid(alpha=0.3)

    # 5. Best model 분포
    ax = axes[1, 1]
    if len(best_model) > 0:
        colors = [DEFAULT_COLORS.get(m, 'gray') for m in best_model.index]
        best_model.plot(kind='bar', ax=ax, color=colors, alpha=0.8)
        ax.set_title('종목별 Best 모델 분포', fontsize=11)
        ax.set_xlabel('모델'); ax.set_ylabel('종목 수')
        ax.tick_params(axis='x', rotation=0)

    # 6. Performance weight 시계열
    ax = axes[1, 2]
    if len(weight_timeline) > 0:
        ax.plot(weight_timeline.index, weight_timeline['w_v4'], label='LSTM (v4)',
                color=DEFAULT_COLORS['LSTM'], linewidth=1.5)
        ax.plot(weight_timeline.index, weight_timeline['w_har'], label='HAR',
                color=DEFAULT_COLORS['HAR'], linewidth=1.5)
        ax.set_title('Performance Weight 시계열\n(Diebold-Pauly rolling)', fontsize=11)
        ax.set_ylabel('Weight'); ax.legend(); ax.grid(alpha=0.3)
        ax.axhline(0.5, color='black', linewidth=0.5, linestyle=':')

    plt.suptitle(f'{model_name} — Layer 1 변동성 예측 진단', fontsize=13, y=1.005)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


# =============================================================================
# Layer 2 — 포트폴리오 단독 평가
# =============================================================================

def evaluate_portfolio_standalone(
    returns: pd.Series,
    scenario_name: str = 'Scenario',
    spy_returns: Optional[pd.Series] = None,
    rf_returns: Optional[pd.Series] = None,
    weights_dict: Optional[Dict[pd.Timestamp, pd.Series]] = None,
    annual_factor: int = 12,
) -> Dict[str, float]:
    """⭐ Layer 2: 포트폴리오 성과 단독 평가.

    Parameters
    ----------
    returns : pd.Series (date index, monthly returns)
    scenario_name : str
    spy_returns : pd.Series, optional
        시장 수익률 (CAPM, IR 계산)
    rf_returns : pd.Series, optional
        무위험 수익률 (excess return 계산)
    weights_dict : dict, optional
        매월 가중치 (turnover, top10 계산)
    annual_factor : int, default 12 (월별 → 연환산)

    Returns
    -------
    dict {sharpe, cagr, ann_vol, mdd, capm_alpha, capm_beta, capm_t,
          information_ratio, sortino, calmar, hit_rate,
          skew, kurt, cvar_5, var_5,
          turnover, top10_concentration, n_months}
    """
    rets = returns.dropna()
    if len(rets) == 0:
        return {k: np.nan for k in METRIC_ORDER_PORTFOLIO}

    # 기본 메트릭
    ann_ret = float(rets.mean() * annual_factor)
    ann_vol = float(rets.std() * np.sqrt(annual_factor))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + rets).cumprod()
    mdd = float((cum / cum.cummax() - 1).min())
    n = len(rets)
    cagr = float(cum.iloc[-1] ** (annual_factor / n) - 1) if n > 0 else np.nan

    # CAPM α, β
    capm_alpha = capm_beta = capm_t = np.nan
    if spy_returns is not None:
        common_idx = rets.index.intersection(spy_returns.index)
        if len(common_idx) >= 12:
            r_p = rets.reindex(common_idx).values
            r_m = spy_returns.reindex(common_idx).values
            if rf_returns is not None:
                rf_r = rf_returns.reindex(common_idx).fillna(0).values
                r_p_excess = r_p - rf_r
                r_m_excess = r_m - rf_r
            else:
                r_p_excess = r_p
                r_m_excess = r_m
            try:
                slope, intercept, r_v, _, std_err = stats.linregress(r_m_excess, r_p_excess)
                capm_alpha = float(intercept * annual_factor)  # 연환산
                capm_beta = float(slope)
                capm_t = float(intercept / max(std_err, 1e-12))
            except Exception:
                pass

    # Information ratio (vs SPY)
    information_ratio = np.nan
    if spy_returns is not None:
        common_idx = rets.index.intersection(spy_returns.index)
        if len(common_idx) >= 12:
            excess = rets.reindex(common_idx) - spy_returns.reindex(common_idx)
            te = excess.std() * np.sqrt(annual_factor)  # tracking error
            information_ratio = float(excess.mean() * annual_factor / te) if te > 0 else np.nan

    # Sortino (downside 표준편차)
    downside = rets[rets < 0]
    downside_std = float(downside.std() * np.sqrt(annual_factor)) if len(downside) > 1 else np.nan
    sortino = ann_ret / downside_std if downside_std and downside_std > 0 else np.nan

    # Calmar = CAGR / |MDD|
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    # Hit rate (양수 수익률 비율)
    hit_rate = float((rets > 0).mean())

    # Skewness, Kurtosis
    skew = float(rets.skew())
    kurt = float(rets.kurtosis())

    # CVaR_5, VaR_5
    var_5 = float(rets.quantile(0.05))
    cvar_5 = float(rets[rets <= var_5].mean()) if (rets <= var_5).sum() > 0 else np.nan

    # Turnover, top-10 concentration (가중치 제공 시)
    turnover = top10 = np.nan
    if weights_dict and len(weights_dict) >= 2:
        dates_sorted = sorted(weights_dict.keys())
        turnovers = []
        top10_list = []
        for i in range(1, len(dates_sorted)):
            w_old = weights_dict[dates_sorted[i - 1]]
            w_new = weights_dict[dates_sorted[i]]
            common = w_old.index.union(w_new.index)
            w_o = w_old.reindex(common).fillna(0)
            w_n = w_new.reindex(common).fillna(0)
            turnovers.append(float((w_n - w_o).abs().sum()))
            top10_list.append(float(w_n.nlargest(10).sum()))
        turnover = float(np.mean(turnovers)) if turnovers else np.nan
        top10 = float(np.mean(top10_list)) if top10_list else np.nan

    return {
        'sharpe': sharpe,
        'cagr': cagr * 100,  # %
        'ann_vol': ann_vol * 100,
        'mdd': mdd * 100,
        'capm_alpha': capm_alpha * 100 if not np.isnan(capm_alpha) else np.nan,
        'capm_beta': capm_beta,
        'capm_t': capm_t,
        'information_ratio': information_ratio,
        'sortino': sortino,
        'calmar': calmar,
        'hit_rate': hit_rate * 100,
        'skew': skew,
        'kurt': kurt,
        'cvar_5': cvar_5 * 100,
        'var_5': var_5 * 100,
        'turnover': turnover,
        'top10_concentration': top10,
        'n_months': n,
    }


def plot_portfolio_diagnostic_panel(
    returns: pd.Series,
    scenario_name: str = 'Scenario',
    spy_returns: Optional[pd.Series] = None,
    weights_dict: Optional[Dict[pd.Timestamp, pd.Series]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 10),
) -> plt.Figure:
    """Layer 2 시각화 — 6 panel."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    rets = returns.dropna()
    color = DEFAULT_COLORS.get(scenario_name, 'steelblue')

    # 1. 누적 수익률
    ax = axes[0, 0]
    cum = (1 + rets).cumprod()
    ax.plot(cum.index, cum.values, color=color, linewidth=1.8, label=scenario_name)
    if spy_returns is not None:
        common_idx = rets.index.intersection(spy_returns.index)
        if len(common_idx) > 0:
            cum_spy = (1 + spy_returns.reindex(common_idx)).cumprod()
            ax.plot(cum_spy.index, cum_spy.values, color=DEFAULT_COLORS['SPY'],
                    linewidth=1.5, linestyle='--', label='SPY', alpha=0.7)
    ax.set_title('누적 수익률', fontsize=11)
    ax.set_ylabel('Cum Return (1=base)'); ax.legend(); ax.grid(alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    dd = (cum / cum.cummax() - 1) * 100
    ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.5)
    ax.set_title(f'Drawdown\nMDD: {dd.min():.2f}%', fontsize=11)
    ax.set_ylabel('Drawdown (%)'); ax.grid(alpha=0.3)

    # 3. Rolling Sharpe (12m)
    ax = axes[0, 2]
    if len(rets) >= 12:
        roll_sharpe = rets.rolling(12).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan
        )
        ax.plot(roll_sharpe.index, roll_sharpe.values, color=color, linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title('Rolling Sharpe (12m)', fontsize=11)
        ax.set_ylabel('Sharpe'); ax.grid(alpha=0.3)

    # 4. Monthly return histogram
    ax = axes[1, 0]
    ax.hist(rets.values * 100, bins=30, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(rets.mean() * 100, color='red', linestyle='--',
               label=f'평균: {rets.mean()*100:.2f}%')
    ax.set_title('월별 수익률 분포', fontsize=11)
    ax.set_xlabel('월별 수익률 (%)'); ax.legend(); ax.grid(alpha=0.3)

    # 5. Tail histogram (CVaR shaded)
    ax = axes[1, 1]
    var_5 = rets.quantile(0.05) * 100
    cvar_5 = rets[rets <= rets.quantile(0.05)].mean() * 100
    ax.hist(rets.values * 100, bins=30, color='lightgray', alpha=0.7, edgecolor='white')
    tail = rets[rets <= rets.quantile(0.05)] * 100
    ax.hist(tail.values, bins=10, color='red', alpha=0.8, label=f'5% tail (CVaR={cvar_5:.2f}%)')
    ax.axvline(var_5, color='darkred', linestyle='--', label=f'VaR_5: {var_5:.2f}%')
    ax.set_title('Tail Risk (5% 좌측)', fontsize=11)
    ax.set_xlabel('월별 수익률 (%)'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 6. Calendar heatmap (year × month)
    ax = axes[1, 2]
    rets_df = rets.to_frame('ret')
    rets_df['year'] = rets_df.index.year
    rets_df['month'] = rets_df.index.month
    pivot = rets_df.pivot_table(index='year', columns='month', values='ret') * 100
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f'{m}' for m in range(1, 13)], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title('Calendar Heatmap (% 월별)', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'{scenario_name} — Layer 2 포트폴리오 성과', fontsize=13, y=1.005)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


# =============================================================================
# Layer 3 — ML → BL 인과 추적
# =============================================================================

def evaluate_ml_to_bl_pipeline(
    pred_df: pd.DataFrame,
    weights_dict: Dict[pd.Timestamp, pd.Series],
    panel: pd.DataFrame,
    scenario_name: str = 'BL_ml',
    pred_col: str = 'y_pred_ensemble',
    pct: float = 0.30,
) -> Dict[str, object]:
    """⭐ Layer 3: ML 예측 → BL P 행렬 → 포트폴리오 성과 인과 추적.

    Parameters
    ----------
    pred_df : 예측 DataFrame (date, ticker, y_pred_*, y_true)
    weights_dict : {rebalance_date: weights pd.Series}
    panel : daily panel (vol_21d 실현값 비교용)
    scenario_name : str
    pred_col : 예측 컬럼
    pct : 양극단 비율 (0.30 = 상하위 30%)

    Returns
    -------
    dict {
        'low_vol_hit_rate': float (예측 하위 30% ∩ 실제 하위 30%),
        'high_vol_hit_rate': float,
        'rank_consistency_timeline': pd.Series (date × Spearman),
        'p_matrix_turnover': pd.Series (date × selection turnover),
    }
    """
    if len(weights_dict) < 2:
        return {'low_vol_hit_rate': np.nan, 'high_vol_hit_rate': np.nan,
                'rank_consistency_timeline': pd.Series(), 'p_matrix_turnover': pd.Series()}

    # 매월 hit rate + rank consistency
    hit_rates_low = []
    hit_rates_high = []
    rank_consistency = {}
    selection_low_prev = None
    selection_high_prev = None
    turnovers = {}

    pred_df_idx = pred_df.set_index(['date', 'ticker'])

    for reb_date in sorted(weights_dict.keys()):
        # 예측 vol (해당 시점)
        pred_at = pred_df[pred_df['date'] == reb_date].set_index('ticker')[pred_col].dropna()
        # 실제 vol_21d (해당 시점)
        actual_at = panel[panel['date'] == reb_date].set_index('ticker')['vol_21d'].dropna()

        common = pred_at.index.intersection(actual_at.index)
        if len(common) < 20:
            continue

        pred_c = pred_at.loc[common]
        actual_c = actual_at.loc[common]
        n = len(common)
        n_group = max(1, int(n * pct))

        # 예측 ranking
        pred_low = set(pred_c.nsmallest(n_group).index)
        pred_high = set(pred_c.nlargest(n_group).index)
        # 실제 ranking
        actual_low = set(actual_c.nsmallest(n_group).index)
        actual_high = set(actual_c.nlargest(n_group).index)

        # Hit rate
        hit_rates_low.append(len(pred_low & actual_low) / n_group)
        hit_rates_high.append(len(pred_high & actual_high) / n_group)

        # Rank consistency
        try:
            rho, _ = stats.spearmanr(pred_c.values, actual_c.values)
            rank_consistency[reb_date] = float(rho)
        except Exception:
            pass

        # Selection turnover (직전 대비)
        if selection_low_prev is not None:
            tov = (len(selection_low_prev ^ pred_low) + len(selection_high_prev ^ pred_high)) / (2 * n_group)
            turnovers[reb_date] = float(tov)
        selection_low_prev = pred_low
        selection_high_prev = pred_high

    return {
        'low_vol_hit_rate': float(np.mean(hit_rates_low)) if hit_rates_low else np.nan,
        'high_vol_hit_rate': float(np.mean(hit_rates_high)) if hit_rates_high else np.nan,
        'rank_consistency_timeline': pd.Series(rank_consistency).sort_index(),
        'p_matrix_turnover': pd.Series(turnovers).sort_index(),
        'hit_rates_low_timeline': pd.Series(hit_rates_low),
        'hit_rates_high_timeline': pd.Series(hit_rates_high),
    }


def plot_ml_bl_diagnostic_panel(
    causality_data: dict,
    scenario_name: str = 'BL_ml',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 5),
) -> plt.Figure:
    """Layer 3 시각화 — 3 panel."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Hit rate 시계열
    ax = axes[0]
    hr_low = causality_data.get('hit_rates_low_timeline', pd.Series())
    hr_high = causality_data.get('hit_rates_high_timeline', pd.Series())
    if len(hr_low) > 0:
        ax.plot(hr_low.values, label='Low vol hit rate', color='blue', linewidth=1.5)
        ax.plot(hr_high.values, label='High vol hit rate', color='red', linewidth=1.5)
        ax.axhline(0.30, color='black', linewidth=0.5, linestyle='--', label='Random (0.3)')
        ax.set_title(f'Hit Rate 시계열\n평균: low={causality_data.get("low_vol_hit_rate", np.nan):.3f}, '
                     f'high={causality_data.get("high_vol_hit_rate", np.nan):.3f}', fontsize=11)
        ax.set_ylabel('Hit rate'); ax.legend(); ax.grid(alpha=0.3)

    # 2. Rank consistency 시계열
    ax = axes[1]
    rc = causality_data.get('rank_consistency_timeline', pd.Series())
    if len(rc) > 0:
        ax.plot(rc.index, rc.values, color='steelblue', linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axhline(rc.mean(), color='red', linestyle='--', label=f'평균: {rc.mean():.3f}')
        ax.set_title('Rank Consistency 시계열\n(Spearman ρ pred vs actual)', fontsize=11)
        ax.set_ylabel('Spearman ρ'); ax.legend(); ax.grid(alpha=0.3)

    # 3. Selection turnover 시계열
    ax = axes[2]
    tov = causality_data.get('p_matrix_turnover', pd.Series())
    if len(tov) > 0:
        ax.plot(tov.index, tov.values, color='darkgreen', linewidth=1.5)
        ax.axhline(tov.mean(), color='red', linestyle='--', label=f'평균: {tov.mean():.3f}')
        ax.set_title('P 행렬 안정성\n(Selection turnover)', fontsize=11)
        ax.set_ylabel('Turnover ratio'); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle(f'{scenario_name} — Layer 3 ML → BL 인과 추적', fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


# =============================================================================
# Layer 4 — 시기별 분해
# =============================================================================

def evaluate_by_period(
    returns_dict: Dict[str, pd.Series],
    periods: Optional[Dict[str, Tuple[str, str]]] = None,
    spy_returns: Optional[pd.Series] = None,
    rf_returns: Optional[pd.Series] = None,
    weights_dict_per_scenario: Optional[Dict[str, Dict]] = None,
    annual_factor: int = 12,
) -> pd.DataFrame:
    """⭐ Layer 4: 시기별 메트릭 분해.

    Parameters
    ----------
    returns_dict : {scenario: returns Series}
    periods : 시기 dict (default 5 시기: GFC 회복 / 강세장 / COVID / 긴축 / 회복)
    spy_returns, rf_returns : Layer 2 인자

    Returns
    -------
    pd.DataFrame
        MultiIndex (period, metric) × scenario
    """
    if periods is None:
        periods = PERIODS

    rows = []
    for period_name, (start, end) in periods.items():
        for scenario, rets in returns_dict.items():
            sub = rets.loc[start:end].dropna()
            if len(sub) < 6:
                continue
            spy_sub = spy_returns.loc[start:end].dropna() if spy_returns is not None else None
            rf_sub = rf_returns.loc[start:end].dropna() if rf_returns is not None else None
            weights_sub = None
            if weights_dict_per_scenario and scenario in weights_dict_per_scenario:
                weights_sub = {d: w for d, w in weights_dict_per_scenario[scenario].items()
                                if pd.Timestamp(start) <= d <= pd.Timestamp(end)}

            metrics = evaluate_portfolio_standalone(
                sub, scenario_name=scenario,
                spy_returns=spy_sub, rf_returns=rf_sub,
                weights_dict=weights_sub, annual_factor=annual_factor,
            )
            for k, v in metrics.items():
                rows.append({'period': period_name, 'metric': k, 'scenario': scenario, 'value': v})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.pivot_table(index=['period', 'metric'], columns='scenario', values='value')


def plot_period_decomposition(
    period_metrics: pd.DataFrame,
    metrics_to_plot: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 8),
) -> plt.Figure:
    """Layer 4 시각화 — 시기별 grouped bar (주요 메트릭)."""
    if metrics_to_plot is None:
        metrics_to_plot = ['sharpe', 'cagr', 'mdd', 'sortino']

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        try:
            sub = period_metrics.xs(metric, level='metric')
        except KeyError:
            continue

        x = np.arange(len(sub.index))
        width = 0.8 / len(sub.columns) if len(sub.columns) > 0 else 0.8
        for j, scenario in enumerate(sub.columns):
            offset = (j - len(sub.columns) / 2 + 0.5) * width
            color = DEFAULT_COLORS.get(scenario, None)
            ax.bar(x + offset, sub[scenario].values, width, label=scenario,
                   color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=20, ha='right', fontsize=8)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=7, loc='best')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

    # Hide extra axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    plt.suptitle('Layer 4 — 시기별 분해', fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


# =============================================================================
# Layer 5 — 시나리오 간 통계 검정
# =============================================================================

def jobson_korkie_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    annual_factor: int = 12,
) -> Dict[str, float]:
    """Jobson-Korkie (1981) Sharpe ratio difference test.

    H0: Sharpe_a = Sharpe_b
    """
    common = returns_a.index.intersection(returns_b.index)
    a = returns_a.reindex(common).dropna()
    b = returns_b.reindex(common).dropna()
    common = a.index.intersection(b.index)
    a = a.reindex(common)
    b = b.reindex(common)
    n = len(a)
    if n < 24:
        return {'jk_stat': np.nan, 'jk_pvalue': np.nan,
                'sharpe_a': np.nan, 'sharpe_b': np.nan, 'sharpe_diff': np.nan, 'n': n}

    mu_a = a.mean()
    mu_b = b.mean()
    sig_a = a.std()
    sig_b = b.std()
    sig_ab = a.cov(b)

    sr_a = mu_a / sig_a * np.sqrt(annual_factor) if sig_a > 0 else np.nan
    sr_b = mu_b / sig_b * np.sqrt(annual_factor) if sig_b > 0 else np.nan
    sr_diff = sr_a - sr_b

    # JK 분산 추정
    theta = (1 / n) * (
        2 * sig_a ** 2 * sig_b ** 2 - 2 * sig_a * sig_b * sig_ab
        + 0.5 * (mu_a ** 2 * sig_b ** 2 + mu_b ** 2 * sig_a ** 2)
        - mu_a * mu_b * sig_ab ** 2 / (sig_a * sig_b)
    )
    if theta <= 0:
        return {'jk_stat': np.nan, 'jk_pvalue': np.nan,
                'sharpe_a': float(sr_a), 'sharpe_b': float(sr_b),
                'sharpe_diff': float(sr_diff), 'n': n}

    se_diff = np.sqrt(theta) / (sig_a * sig_b) * np.sqrt(annual_factor)
    jk_stat = (mu_a / sig_a - mu_b / sig_b) * np.sqrt(annual_factor) / se_diff
    pvalue = 2 * (1 - stats.norm.cdf(abs(jk_stat)))

    return {
        'jk_stat': float(jk_stat),
        'jk_pvalue': float(pvalue),
        'sharpe_a': float(sr_a),
        'sharpe_b': float(sr_b),
        'sharpe_diff': float(sr_diff),
        'n': int(n),
    }


def memmel_correction(
    returns_a: pd.Series,
    returns_b: pd.Series,
    annual_factor: int = 12,
) -> Dict[str, float]:
    """Memmel (2003) Jobson-Korkie 보정.

    Memmel 은 JK 의 분산 추정 편의를 보정한 버전.
    """
    common = returns_a.index.intersection(returns_b.index)
    a = returns_a.reindex(common).dropna()
    b = returns_b.reindex(common).dropna()
    common = a.index.intersection(b.index)
    a = a.reindex(common); b = b.reindex(common)
    n = len(a)
    if n < 24:
        return {'memmel_stat': np.nan, 'memmel_pvalue': np.nan, 'n': n}

    mu_a = a.mean(); mu_b = b.mean()
    sig_a = a.std(); sig_b = b.std()
    sig_ab = a.cov(b)
    sr_a = mu_a / sig_a; sr_b = mu_b / sig_b

    # Memmel 보정
    var_diff = (1 / n) * (
        2 - 2 * sig_ab / (sig_a * sig_b)
        + 0.5 * (sr_a ** 2 + sr_b ** 2) - sr_a * sr_b * (sig_ab / (sig_a * sig_b)) ** 2
    )
    if var_diff <= 0:
        return {'memmel_stat': np.nan, 'memmel_pvalue': np.nan, 'n': n}

    stat = (sr_a - sr_b) / np.sqrt(var_diff)
    pvalue = 2 * (1 - stats.norm.cdf(abs(stat)))

    return {
        'memmel_stat': float(stat * np.sqrt(annual_factor)),
        'memmel_pvalue': float(pvalue),
        'n': int(n),
    }


def dm_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
) -> Dict[str, float]:
    """Diebold-Mariano test.

    H0: 두 예측 모델의 손실 평균이 같음.
    """
    e_a = np.asarray(errors_a)
    e_b = np.asarray(errors_b)
    mask = ~(np.isnan(e_a) | np.isnan(e_b))
    e_a = e_a[mask]; e_b = e_b[mask]
    n = len(e_a)
    if n < 30:
        return {'dm_stat': np.nan, 'dm_pvalue': np.nan, 'n': n}

    d = e_a - e_b
    d_mean = d.mean()

    # Newey-West 분산 (h=1 이면 단순)
    if h == 1:
        var_d = d.var(ddof=1) / n
    else:
        autocov = []
        for k in range(1, h):
            ck = np.cov(d[:-k], d[k:])[0, 1]
            autocov.append(ck)
        var_d = (d.var(ddof=1) + 2 * sum(autocov)) / n

    if var_d <= 0:
        return {'dm_stat': np.nan, 'dm_pvalue': np.nan, 'n': n}

    stat = d_mean / np.sqrt(var_d)
    pvalue = 2 * (1 - stats.norm.cdf(abs(stat)))

    return {
        'dm_stat': float(stat),
        'dm_pvalue': float(pvalue),
        'n': int(n),
    }


def hansen_mcs(
    losses_dict: Dict[str, np.ndarray],
    alpha: float = 0.05,
    n_boot: int = 5000,
    block_size: int = 3,
    seed: int = 42,
) -> Dict[str, object]:
    """Hansen, Lunde, Nason (2011) Model Confidence Set.

    여러 모델 중 통계적으로 'best' 와 구분 안 되는 모델 set 반환.

    Parameters
    ----------
    losses_dict : {model_name: loss array (e.g., squared errors)}
    alpha : 유의 수준
    n_boot : Bootstrap iter
    block_size : Block bootstrap
    seed : random seed

    Returns
    -------
    dict {
        'mcs_set': list (alpha 수준에서 best 와 동등한 모델),
        'eliminated_order': list (제거 순서),
        'pvalues': dict (모델별 MCS p-value),
    }
    """
    np.random.seed(seed)
    models = list(losses_dict.keys())
    losses = {m: np.asarray(losses_dict[m]) for m in models}
    # 공통 길이
    n_min = min(len(l) for l in losses.values())
    losses = {m: l[:n_min] for m, l in losses.items()}

    eliminated_order = []
    mcs_pvalues = {}
    current_models = list(models)

    while len(current_models) > 1:
        # 평균 손실 차이 행렬
        L = np.column_stack([losses[m] for m in current_models])
        L_mean = L.mean(axis=0)
        # Range statistic: max(L_i - L_j)
        d_ij = L_mean[:, None] - L_mean[None, :]
        max_diff = np.abs(d_ij).max()

        # Bootstrap
        boot_stats = []
        for _ in range(n_boot):
            n_blocks = int(np.ceil(n_min / block_size))
            starts = np.random.randint(0, n_min - block_size + 1, size=n_blocks)
            sample_idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n_min]
            L_boot = L[sample_idx]
            L_boot_mean = L_boot.mean(axis=0) - L_mean
            d_boot = L_boot_mean[:, None] - L_boot_mean[None, :]
            boot_stats.append(np.abs(d_boot).max())

        pvalue = float(np.mean(np.array(boot_stats) >= max_diff))
        mcs_pvalues['_'.join(sorted(current_models))] = pvalue

        if pvalue >= alpha:
            break

        # 가장 큰 평균 손실 모델 제거
        worst_idx = int(np.argmax(L_mean))
        eliminated = current_models.pop(worst_idx)
        eliminated_order.append(eliminated)

    return {
        'mcs_set': current_models,
        'eliminated_order': eliminated_order,
        'pvalues': mcs_pvalues,
    }


def plot_significance_matrix(
    p_matrix: pd.DataFrame,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """시나리오 쌍 유의성 heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    mask = p_matrix.values < alpha
    im = ax.imshow(p_matrix.values, cmap='RdYlGn_r', vmin=0, vmax=alpha * 4)

    for i in range(p_matrix.shape[0]):
        for j in range(p_matrix.shape[1]):
            v = p_matrix.iloc[i, j]
            if not np.isnan(v):
                color = 'white' if mask[i, j] else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', color=color, fontsize=8)

    ax.set_xticks(range(p_matrix.shape[1]))
    ax.set_yticks(range(p_matrix.shape[0]))
    ax.set_xticklabels(p_matrix.columns, rotation=30, ha='right')
    ax.set_yticklabels(p_matrix.index)
    ax.set_title(f'Pairwise p-values (α={alpha} 빨강=유의)', fontsize=11)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


# =============================================================================
# 종합 보고 헬퍼
# =============================================================================

def render_diagnostic_summary(
    model_name: str,
    layer_results: dict,
) -> str:
    """단일 모델·시나리오 종합 진단 markdown.

    layer_results : {'layer1': {...}, 'layer2': {...}, 'layer3': {...}, 'layer4': pd.DataFrame}
    """
    lines = [f'# {model_name} — 종합 진단 보고서\n']

    # Layer 1
    if 'layer1' in layer_results:
        l1 = layer_results['layer1'].get('overall', {})
        lines.append('## Layer 1 — 변동성 예측 진단\n')
        lines.append(f'- RMSE: {l1.get("rmse", np.nan):.4f}')
        lines.append(f'- QLIKE: {l1.get("qlike", np.nan):.4f}')
        lines.append(f'- R²_train_mean: {l1.get("r2_train_mean", np.nan):.4f}')
        lines.append(f'- MZ: α={l1.get("mz_alpha", np.nan):.4f}, β={l1.get("mz_beta", np.nan):.4f}, R²={l1.get("mz_r2", np.nan):.4f}')
        lines.append(f'- pred_std_ratio: {l1.get("pred_std_ratio", np.nan):.3f} (mean-collapse 진단)')
        lines.append(f'- Spearman: {l1.get("spearman", np.nan):.3f}')
        if not np.isnan(l1.get('dm_stat_vs_har', np.nan)):
            lines.append(f'- DM-test vs HAR: stat={l1.get("dm_stat_vs_har", np.nan):.3f}, p={l1.get("dm_pvalue_vs_har", np.nan):.4f}')
        lines.append('')

    # Layer 2
    if 'layer2' in layer_results:
        l2 = layer_results['layer2']
        lines.append('## Layer 2 — 포트폴리오 단독 성과\n')
        lines.append(f'- Sharpe: {l2.get("sharpe", np.nan):.3f}')
        lines.append(f'- CAGR: {l2.get("cagr", np.nan):.2f}%')
        lines.append(f'- MDD: {l2.get("mdd", np.nan):.2f}%')
        lines.append(f'- Sortino: {l2.get("sortino", np.nan):.3f}')
        lines.append(f'- Calmar: {l2.get("calmar", np.nan):.3f}')
        if not np.isnan(l2.get('capm_alpha', np.nan)):
            lines.append(f'- CAPM α: {l2.get("capm_alpha", np.nan):.2f}% (β={l2.get("capm_beta", np.nan):.3f}, t={l2.get("capm_t", np.nan):.2f})')
        if not np.isnan(l2.get('information_ratio', np.nan)):
            lines.append(f'- Information ratio: {l2.get("information_ratio", np.nan):.3f}')
        lines.append(f'- Hit rate: {l2.get("hit_rate", np.nan):.1f}%')
        lines.append(f'- CVaR_5: {l2.get("cvar_5", np.nan):.2f}%')
        if not np.isnan(l2.get('turnover', np.nan)):
            lines.append(f'- Turnover: {l2.get("turnover", np.nan):.3f}')
            lines.append(f'- Top-10 concentration: {l2.get("top10_concentration", np.nan):.3f}')
        lines.append('')

    # Layer 3
    if 'layer3' in layer_results:
        l3 = layer_results['layer3']
        lines.append('## Layer 3 — ML → BL 인과 추적\n')
        lines.append(f'- Low vol hit rate: {l3.get("low_vol_hit_rate", np.nan):.3f} (random=0.30)')
        lines.append(f'- High vol hit rate: {l3.get("high_vol_hit_rate", np.nan):.3f}')
        rc = l3.get('rank_consistency_timeline', pd.Series())
        if len(rc) > 0:
            lines.append(f'- Rank consistency 평균: {rc.mean():.3f}')
        tov = l3.get('p_matrix_turnover', pd.Series())
        if len(tov) > 0:
            lines.append(f'- P 행렬 turnover 평균: {tov.mean():.3f}')
        lines.append('')

    return '\n'.join(lines)
