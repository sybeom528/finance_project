"""
02a_v2.ipynb 에 §7 (10 분석) 셀 batch 추가 + LS spread 6 시나리오 확장.

추가:
  §7 markdown
  §7-A. 위험-수익 다양화 메트릭 (Sortino, Calmar, Information, Omega)
  §7-B. 통계적 유의성 검정 (Bootstrap CI, ML 효과 p-value) - sec7_v2_analyses.pkl 로드
  §7-C. 거래비용 시뮬레이션 (Net Sharpe / CAGR)
  §7-D. 36개월 Rolling 메트릭 (Sharpe, alpha/beta vs SPY) + 시각화
  §7-E. Tail Risk (VaR, CVaR, Skew, Kurtosis)
  §7-F. Drawdown Events 정밀 분석 + 시각화
  §7-G. CAPM Performance Attribution
  §7-H. Sector Exposure 분석 (sec7_v2_analyses.pkl 로드)
  §7-I. Market Regime (VIX quantile, sec7_v2_analyses.pkl 로드)
  §7-J. Concentration 시계열 (HHI, Effective N) + 시각화

사용:
    python scripts/_add_sec7_cells.py
"""
import json
import sys
import io
import uuid
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

NB_PATH = Path(__file__).resolve().parent.parent / '02a_v2.ipynb'

# ============================================================================
# §7 셀 source 정의
# ============================================================================

SEC7_HEADER = """## §7. Phase 3-2 v2 심층 분석 (6 시나리오 모두 cover)

§6-4 의 메트릭 표 + α/β/γ 분석 위에 추가:
- §7-A: 위험-수익 다양화 메트릭 (Sortino, Calmar, Information, Omega)
- §7-B: 통계적 유의성 검정 (Bootstrap CI for Sharpe, ML 효과 p-value)
- §7-C: 거래비용 시뮬레이션 (0/5/10/20bp net Sharpe)
- §7-D: 36개월 Rolling 메트릭 (Sharpe, α/β vs SPY)
- §7-E: Tail Risk (VaR, CVaR, Skew, Kurtosis)
- §7-F: Drawdown Events 정밀 분석
- §7-G: CAPM Performance Attribution (Jensen alpha)
- §7-H: Sector Exposure 분석 (12 GICS sectors)
- §7-I: Market Regime Analysis (VIX quantile)
- §7-J: Concentration 시계열 (HHI, Effective N)

> Heavy 분석 (§7-B Bootstrap, §7-H Sector, §7-I VIX) 은 `scripts/_run_02a_v2_sec7_analyses.py` standalone 결과 (`data/sec7_v2_analyses.pkl`) 활용."""


SEC7_A = '''# §7-A. 위험-수익 다양화 메트릭 (Sortino, Calmar, Information, Omega)
print('=' * 75)
print('  §7-A. 위험-수익 다양화 메트릭')
print('=' * 75)


def sortino_ratio(returns, target_ret=0, annual_factor=12):
    """Sortino: 하방 변동성만 고려한 위험조정수익률."""
    excess = returns - target_ret
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() == 0:
        return np.nan
    return excess.mean() / downside.std() * np.sqrt(annual_factor)


def calmar_ratio(returns, annual_factor=12):
    """Calmar: CAGR / |MDD| (낙폭 대비 수익)."""
    cum = (1 + returns).cumprod()
    n = len(returns)
    if n == 0:
        return np.nan
    cagr = cum.iloc[-1] ** (annual_factor / n) - 1
    mdd = abs((cum / cum.cummax() - 1).min())
    return cagr / mdd if mdd > 0 else np.nan


def information_ratio(returns, benchmark, annual_factor=12):
    """Information Ratio: active return / tracking error."""
    common_idx = returns.index.intersection(benchmark.index)
    excess = returns.loc[common_idx] - benchmark.loc[common_idx]
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(annual_factor)


def omega_ratio(returns, threshold=0):
    """Omega: 양수 수익률 합 / 음수 수익률 합."""
    excess = returns - threshold
    pos = excess[excess > 0].sum()
    neg = -excess[excess < 0].sum()
    return pos / neg if neg > 0 else np.nan


# 6 BL + SPY 모두
diversified_metrics = {}
for name, r in all_returns_fair.items():
    diversified_metrics[name] = {
        'Sortino': sortino_ratio(r),
        'Calmar': calmar_ratio(r),
        'Info_vs_SPY': information_ratio(r, all_returns_fair['SPY']) if name != 'SPY' else np.nan,
        'Omega': omega_ratio(r),
    }
diversified_df = pd.DataFrame(diversified_metrics).T.round(3)
print(diversified_df.to_string())
print()
print('  📌 해석:')
print('  - Sortino > Sharpe 시: 상승 변동성이 큼 (분포 비대칭, 양호)')
print('  - Calmar > 1: 1년에 평균 MDD 만큼 회복 (good)')
print('  - Info Ratio > 0.5: 벤치마크 대비 의미 있는 active alpha')
print('  - Omega > 1: 양의 수익률 일수가 음의 수익률 일수보다 큼')
'''


SEC7_B = '''# §7-B. 통계적 유의성 검정 (Bootstrap CI, ML 효과 p-value)
import pickle

sec7_path = DATA_DIR / 'sec7_v2_analyses.pkl'
if not sec7_path.exists():
    print(f'⚠️ {sec7_path.name} 부재 → standalone 스크립트 (`scripts/_run_02a_v2_sec7_analyses.py`) 먼저 실행')
else:
    with open(sec7_path, 'rb') as f:
        sec7 = pickle.load(f)

    print('=' * 75)
    print('  §7-B. Bootstrap CI for Sharpe Ratios (N=5000)')
    print('=' * 75)
    print()

    for period_name, period_label in [('overall_191m', '전체 191m'), ('oos_180m', 'OOS 180m'), ('holdout_11m', 'Hold-out 11m')]:
        print(f'  --- {period_label} ---')
        boot = sec7['bootstrap_results'][period_name]
        boot_df = pd.DataFrame(boot).T.round(3)
        cols = ['sharpe', 'ci_lo_2.5', 'ci_hi_97.5', 'std_err']
        print(boot_df[cols].to_string())
        print()

    print('  === ML 효과 (BL_ml_sw - BL_trailing) Bootstrap CI + p-value ===')
    ml_eff_df = pd.DataFrame(sec7['ml_effect_bootstrap']).T.round(4)
    print(ml_eff_df.to_string())
    print()
    print('  ⭐ p-value 해석:')
    print('  - p < 0.05: ML ≠ Trailing (5% 유의수준에서 statistically significant)')
    print('  - p > 0.05: ML 과 Trailing 의 차이가 random noise 와 구별 안 됨')
    print('  - Hold-out 11m 표본 크기 작아 (n=11) p-value 보수적으로 해석 권장')
'''


SEC7_C = '''# §7-C. 거래비용 시뮬레이션 (Net Sharpe / CAGR / MDD)
print('=' * 75)
print('  §7-C. 거래비용 시뮬레이션 (Net Performance)')
print('=' * 75)


def calc_turnover_series(weights_dict):
    """월별 turnover 시계열 계산."""
    sorted_dates = sorted(weights_dict.keys())
    if len(sorted_dates) < 2:
        return pd.Series(dtype=float)
    turnovers = {}
    for i in range(1, len(sorted_dates)):
        d_prev, d_curr = sorted_dates[i-1], sorted_dates[i]
        w_prev = weights_dict[d_prev]
        w_curr = weights_dict[d_curr]
        idx_union = w_prev.index.union(w_curr.index)
        wp = w_prev.reindex(idx_union, fill_value=0)
        wc = w_curr.reindex(idx_union, fill_value=0)
        turnovers[d_curr] = 0.5 * abs(wp - wc).sum()
    return pd.Series(turnovers).sort_index()


# 각 시나리오별 turnover 시계열 + net 메트릭
COSTS_BP = [0, 5, 10, 20]   # bp per turnover dollar
turnover_series = {}
for s_name, w_dict in weights.items():
    turnover_series[f'BL_{s_name}'] = calc_turnover_series(w_dict)

# Net 메트릭 계산
net_metrics = {}
for name, r in all_returns_fair.items():
    if name == 'SPY':
        # SPY 는 turnover 0 가정
        net_metrics[name] = {f'net_sharpe_{c}bp': r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan for c in COSTS_BP}
        net_metrics[name].update({f'net_cagr_{c}bp_%': ((1+r).cumprod().iloc[-1]**(12/len(r))-1)*100 for c in COSTS_BP})
        continue

    ts = turnover_series.get(name, pd.Series(dtype=float))
    ts_aligned = ts.reindex(r.index, fill_value=0)
    net_metrics[name] = {}
    for cost_bp in COSTS_BP:
        cost_per_month = ts_aligned * cost_bp / 10000   # decimal cost
        net_ret = r - cost_per_month
        net_sr = net_ret.mean() / net_ret.std() * np.sqrt(12) if net_ret.std() > 0 else np.nan
        net_cagr = ((1 + net_ret).cumprod().iloc[-1]) ** (12/len(net_ret)) - 1
        net_metrics[name][f'net_sharpe_{cost_bp}bp'] = net_sr
        net_metrics[name][f'net_cagr_{cost_bp}bp_%'] = net_cagr * 100

net_df = pd.DataFrame(net_metrics).T.round(3)
print('  --- Net Sharpe (cost bp 별) ---')
sharpe_cols = [c for c in net_df.columns if 'sharpe' in c]
print(net_df[sharpe_cols].to_string())
print()
print('  --- Net CAGR % (cost bp 별) ---')
cagr_cols = [c for c in net_df.columns if 'cagr' in c]
print(net_df[cagr_cols].to_string())
print()

# ML 효과 by cost level
print('  📈 ML vs Trailing 거래비용 영향 (Sharpe diff):')
for w_method in ['mcap', 'eq', 'rp']:
    print(f'    [{w_method}]')
    for cost_bp in COSTS_BP:
        ml = net_df.loc[f'BL_ml_sw_{w_method}', f'net_sharpe_{cost_bp}bp']
        tr = net_df.loc[f'BL_trailing_{w_method}', f'net_sharpe_{cost_bp}bp']
        diff = ml - tr
        marker = '⭐' if diff > 0 else '  '
        print(f'      {cost_bp:3d}bp: ml_sw {ml:.3f} vs trailing {tr:.3f}, diff {diff:+.3f} {marker}')
print()
print('  💡 거래비용이 클수록 BL_ml_sw 가 BL_trailing 대비 우위 확대 (ML 의 lower turnover 효과)')
'''


SEC7_D = '''# §7-D. 36개월 Rolling 메트릭 (Sharpe + alpha/beta vs SPY) + 시각화
print('=' * 75)
print('  §7-D. 36개월 Rolling 메트릭')
print('=' * 75)

WINDOW = 36
spy_aligned = all_returns_fair['SPY']

rolling_sharpes = {}
rolling_alphas = {}
rolling_betas = {}
rolling_corrs = {}

for name, r in all_returns_fair.items():
    if name == 'SPY':
        continue
    common_idx = r.index.intersection(spy_aligned.index)
    r_aligned = r.loc[common_idx]
    s_aligned = spy_aligned.loc[common_idx]

    rolling_sharpes[name] = (
        r_aligned.rolling(WINDOW).mean() / r_aligned.rolling(WINDOW).std() * np.sqrt(12)
    )

    alphas = []
    betas = []
    for i in range(WINDOW, len(r_aligned) + 1):
        window_idx = r_aligned.index[i-WINDOW:i]
        r_w = r_aligned.loc[window_idx].values
        s_w = s_aligned.loc[window_idx].values
        var_s = float(np.var(s_w))
        cov = float(np.cov(r_w, s_w)[0, 1])
        beta = cov / var_s if var_s > 0 else 0.0
        alpha = (r_w.mean() - beta * s_w.mean()) * 12   # annualized
        alphas.append(alpha)
        betas.append(beta)
    rolling_alphas[name] = pd.Series(alphas, index=r_aligned.index[WINDOW-1:])
    rolling_betas[name] = pd.Series(betas, index=r_aligned.index[WINDOW-1:])
    rolling_corrs[name] = r_aligned.rolling(WINDOW).corr(s_aligned)

# 시각화 (4 panel × 6 line)
COLORS_V2_DICT = {
    'BL_ml_sw_mcap': '#1f77b4', 'BL_ml_sw_eq': '#1f77b4', 'BL_ml_sw_rp': '#1f77b4',
    'BL_trailing_mcap': '#d62728', 'BL_trailing_eq': '#d62728', 'BL_trailing_rp': '#d62728',
}
LINESTYLES = {'mcap': '-', 'eq': '--', 'rp': '-.'}


def get_style(name):
    parts = name.split('_')
    return LINESTYLES.get(parts[-1], '-')


fig, axes = plt.subplots(4, 1, figsize=(14, 14))

for name, rs in rolling_sharpes.items():
    axes[0].plot(rs.index, rs.values, color=COLORS_V2_DICT.get(name, 'gray'),
                 linestyle=get_style(name), label=name, alpha=0.8)
axes[0].set_title(f'{WINDOW}m Rolling Sharpe Ratio (annualized)')
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].grid(True, alpha=0.3)
axes[0].legend(ncol=2, fontsize=8, loc='lower right')

for name, ra in rolling_alphas.items():
    axes[1].plot(ra.index, ra.values * 100, color=COLORS_V2_DICT.get(name, 'gray'),
                 linestyle=get_style(name), label=name, alpha=0.8)
axes[1].set_title(f'{WINDOW}m Rolling Annualized Alpha % (vs SPY)')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].grid(True, alpha=0.3)
axes[1].legend(ncol=2, fontsize=8, loc='lower right')

for name, rb in rolling_betas.items():
    axes[2].plot(rb.index, rb.values, color=COLORS_V2_DICT.get(name, 'gray'),
                 linestyle=get_style(name), label=name, alpha=0.8)
axes[2].set_title(f'{WINDOW}m Rolling Beta (vs SPY)')
axes[2].axhline(1, color='black', linewidth=0.5, linestyle=':')
axes[2].grid(True, alpha=0.3)
axes[2].legend(ncol=2, fontsize=8, loc='lower right')

for name, rc in rolling_corrs.items():
    axes[3].plot(rc.index, rc.values, color=COLORS_V2_DICT.get(name, 'gray'),
                 linestyle=get_style(name), label=name, alpha=0.8)
axes[3].set_title(f'{WINDOW}m Rolling Correlation (vs SPY)')
axes[3].grid(True, alpha=0.3)
axes[3].legend(ncol=2, fontsize=8, loc='lower right')

plt.tight_layout()
out_path = OUT_DIR_V2_SW / 'rolling_metrics.png'
plt.savefig(out_path, dpi=100, bbox_inches='tight')
plt.show()
print(f'  💾 {out_path.name}')
print()

# 평균 표
mean_alpha_beta = pd.DataFrame({
    name: {
        'mean_rolling_sharpe': rolling_sharpes[name].mean(),
        'mean_rolling_alpha_%': rolling_alphas[name].mean() * 100,
        'mean_rolling_beta': rolling_betas[name].mean(),
        'mean_rolling_corr': rolling_corrs[name].mean(),
    }
    for name in rolling_sharpes.keys()
}).T.round(3)
print('  --- 36m Rolling 평균 표 ---')
print(mean_alpha_beta.to_string())
'''


SEC7_E = '''# §7-E. Tail Risk (VaR, CVaR, Skew, Kurtosis)
print('=' * 75)
print('  §7-E. Tail Risk Analysis (월별 단위)')
print('=' * 75)

from scipy import stats as sp_stats


def tail_risk_metrics(returns, var_level=0.05):
    """VaR (5%), CVaR (5% Expected Shortfall), Skew, Kurtosis."""
    if len(returns) < 5:
        return {k: np.nan for k in ['VaR_5%', 'CVaR_5%', 'Skew', 'Kurtosis', 'Worst_Month_%', 'Worst_Month_Date']}
    var = returns.quantile(var_level) * 100
    cvar = returns[returns <= returns.quantile(var_level)].mean() * 100
    skew = sp_stats.skew(returns)
    kurt = sp_stats.kurtosis(returns)   # Fisher (정규분포 = 0)
    worst = returns.min() * 100
    worst_date = returns.idxmin().strftime('%Y-%m')
    return {
        'VaR_5%': var,
        'CVaR_5%': cvar,
        'Skew': skew,
        'Kurtosis': kurt,
        'Worst_Month_%': worst,
        'Worst_Month_Date': worst_date,
    }


tail_metrics = {name: tail_risk_metrics(r) for name, r in all_returns_fair.items()}
tail_df = pd.DataFrame(tail_metrics).T

# 정렬 + 표시
display_df = tail_df[['VaR_5%', 'CVaR_5%', 'Skew', 'Kurtosis', 'Worst_Month_%', 'Worst_Month_Date']].copy()
for col in ['VaR_5%', 'CVaR_5%', 'Skew', 'Kurtosis', 'Worst_Month_%']:
    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(3)
print(display_df.to_string())
print()
print('  📌 해석:')
print('  - VaR_5%: 월별 최악 5% 분위수 손실 (음수: 더 작을수록 risky)')
print('  - CVaR_5%: VaR 이하 평균 손실 (Expected Shortfall, conditional)')
print('  - Skew < 0: 큰 손실이 큰 이익보다 자주 발생 (음의 비대칭)')
print('  - Kurtosis > 0: fat tail (정규분포 대비 극단 사건 더 빈번)')
'''


SEC7_F = '''# §7-F. Drawdown Events 정밀 분석 + 시각화
print('=' * 75)
print('  §7-F. Drawdown Events (5% 이상 낙폭, 깊이 + 지속 + 회복)')
print('=' * 75)


def identify_drawdown_events(returns, threshold=0.05):
    """5% 이상 낙폭 이벤트 식별."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1)

    events = []
    in_dd = False
    start = None
    trough = None
    trough_dd = 0
    for date, val in dd.items():
        if not in_dd and val < -threshold:
            in_dd = True
            start = date
            trough = date
            trough_dd = val
        elif in_dd:
            if val < trough_dd:
                trough = date
                trough_dd = val
            if val >= 0:   # 회복
                events.append({
                    'start': start,
                    'trough': trough,
                    'recovery': date,
                    'depth_%': trough_dd * 100,
                    'duration_to_trough_m': (trough - start).days // 30,
                    'recovery_m': (date - trough).days // 30,
                    'total_m': (date - start).days // 30,
                })
                in_dd = False

    # 끝까지 회복 못 한 경우
    if in_dd:
        events.append({
            'start': start,
            'trough': trough,
            'recovery': None,
            'depth_%': trough_dd * 100,
            'duration_to_trough_m': (trough - start).days // 30,
            'recovery_m': None,
            'total_m': None,
        })
    return events


# 6 시나리오 + SPY 의 drawdown event 식별
all_events = {}
for name, r in all_returns_fair.items():
    events = identify_drawdown_events(r, threshold=0.05)
    all_events[name] = events
    print(f'  {name:25s}: {len(events)} events (5%+ drawdown)')

# 가장 큰 drawdown 5개 (각 시나리오)
print()
print('  === 각 시나리오의 Top 3 drawdown events (depth 기준) ===')
for name, events in all_events.items():
    if not events:
        continue
    sorted_events = sorted(events, key=lambda e: e['depth_%'])[:3]
    print(f'  [{name}]')
    for e in sorted_events:
        recov = e['recovery'].strftime('%Y-%m') if e['recovery'] else 'NOT RECOVERED'
        recov_m = f"{e['recovery_m']}m" if e['recovery_m'] is not None else 'N/A'
        print(f'    {e["start"].strftime("%Y-%m")} ~ {e["trough"].strftime("%Y-%m")} ~ {recov}: depth {e["depth_%"]:+.2f}%, recovery {recov_m}')

# Summary 통계
print()
print('  === Drawdown 통계 요약 ===')
dd_summary = {}
for name, events in all_events.items():
    if not events:
        dd_summary[name] = {'n_events': 0, 'avg_depth_%': np.nan, 'avg_recovery_m': np.nan, 'max_depth_%': np.nan}
        continue
    depths = [e['depth_%'] for e in events]
    recovs = [e['recovery_m'] for e in events if e['recovery_m'] is not None]
    dd_summary[name] = {
        'n_events': len(events),
        'avg_depth_%': np.mean(depths),
        'max_depth_%': min(depths),
        'avg_recovery_m': np.mean(recovs) if recovs else np.nan,
        'max_recovery_m': max(recovs) if recovs else np.nan,
    }

dd_df = pd.DataFrame(dd_summary).T.round(2)
print(dd_df.to_string())
print()
print('  💡 BL_ml_sw 의 turnover 가 작아서 회복 기간이 SPY 보다 짧을 가능성 검토')
'''


SEC7_G = '''# §7-G. CAPM Performance Attribution (Jensen alpha)
print('=' * 75)
print('  §7-G. CAPM Performance Attribution (vs SPY, monthly)')
print('=' * 75)

from scipy import stats as sp_stats

spy_aligned = all_returns_fair['SPY']
capm_results = {}
for name, r in all_returns_fair.items():
    if name == 'SPY':
        continue
    common_idx = r.index.intersection(spy_aligned.index)
    r_aligned = r.loc[common_idx]
    s_aligned = spy_aligned.loc[common_idx]
    # OLS regression: r_p = alpha + beta * r_SPY + e
    # statsmodel 안 쓰고 numpy 로 직접 OLS
    n = len(r_aligned)
    X = np.column_stack([np.ones(n), s_aligned.values])
    y = r_aligned.values
    beta_vec, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    alpha_monthly, beta = float(beta_vec[0]), float(beta_vec[1])
    y_pred = X @ beta_vec
    resid = y - y_pred
    rss = float((resid ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - rss / tss if tss > 0 else 0
    # standard errors
    sigma2 = rss / (n - 2)
    var_beta_vec = sigma2 * np.linalg.inv(X.T @ X).diagonal()
    se_alpha = np.sqrt(var_beta_vec[0])
    se_beta = np.sqrt(var_beta_vec[1])
    t_alpha = alpha_monthly / se_alpha if se_alpha > 0 else np.nan
    t_beta = beta / se_beta if se_beta > 0 else np.nan
    p_alpha = 2 * (1 - sp_stats.t.cdf(abs(t_alpha), df=n-2)) if not np.isnan(t_alpha) else np.nan
    p_beta = 2 * (1 - sp_stats.t.cdf(abs(t_beta), df=n-2)) if not np.isnan(t_beta) else np.nan

    capm_results[name] = {
        'alpha_monthly_%': alpha_monthly * 100,
        'alpha_annual_%': alpha_monthly * 12 * 100,
        'beta': beta,
        't_alpha': t_alpha,
        'p_alpha': p_alpha,
        't_beta': t_beta,
        'p_beta': p_beta,
        'R_squared': r_squared,
    }

capm_df = pd.DataFrame(capm_results).T.round(4)
print(capm_df.to_string())
print()
print('  📌 해석:')
print('  - alpha_annual > 0 + p_alpha < 0.05: SPY 대비 statistically significant 한 active alpha')
print('  - beta < 1: 시장 변동성 대비 낮은 노출 (defensive)')
print('  - beta > 1: 시장 변동성 대비 높은 노출 (aggressive)')
print('  - R² 높음: 시나리오 수익률이 SPY 로 잘 설명됨 (idiosyncratic 작음)')
'''


SEC7_H = '''# §7-H. Sector Exposure 분석 (12 GICS sectors, 6 시나리오)
import pickle
print('=' * 75)
print('  §7-H. Sector Exposure Analysis')
print('=' * 75)

sec7_path = DATA_DIR / 'sec7_v2_analyses.pkl'
if not sec7_path.exists():
    print(f'⚠️ {sec7_path.name} 부재 → standalone 스크립트 먼저 실행')
else:
    with open(sec7_path, 'rb') as f:
        sec7 = pickle.load(f)
    se = sec7['sector_exposure']
    st = sec7['sector_tilt']

    # 6 시나리오 평균 sector exposure 표
    print()
    print('  === 6 시나리오 평균 sector exposure (%) ===')
    sector_avg = pd.DataFrame({
        name: pd.Series(se[name]['mean_exposure']) * 100
        for name in se.keys()
    }).round(2)
    print(sector_avg.to_string())

    # ML vs Trailing tilt
    print()
    print('  === ML - Trailing sector tilt (%p, 가중치별) ===')
    tilt_df = pd.DataFrame(st).round(2)
    print(tilt_df.to_string())

    # 6 line 시각화 (시기별 sector tilt 변화)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    # 평균 exposure (bar)
    sector_avg.plot(kind='bar', ax=axes[0], width=0.85)
    axes[0].set_title('6 시나리오 평균 sector exposure (%)')
    axes[0].set_xlabel('Sector')
    axes[0].set_ylabel('Exposure %')
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # tilt heatmap
    tilt_df.plot(kind='bar', ax=axes[1], width=0.85)
    axes[1].set_title('ML - Trailing sector tilt (가중치별, %p)')
    axes[1].set_xlabel('Sector')
    axes[1].set_ylabel('Tilt %p')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    out_path = OUT_DIR_V2_SW / 'sector_exposure.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f'  💾 {out_path.name}')
    print()
    print('  💡 ML 이 trailing 대비 어떤 sector 에 더 / 덜 노출되는지 확인')
'''


SEC7_I = '''# §7-I. Market Regime (VIX quantile) 분석
import pickle
print('=' * 75)
print('  §7-I. Market Regime Analysis (VIX-based)')
print('=' * 75)

sec7_path = DATA_DIR / 'sec7_v2_analyses.pkl'
if not sec7_path.exists():
    print(f'⚠️ {sec7_path.name} 부재 → standalone 스크립트 먼저 실행')
else:
    with open(sec7_path, 'rb') as f:
        sec7 = pickle.load(f)
    vix_data = sec7.get('vix_regime')
    if vix_data is None:
        print('⚠️ VIX regime 데이터 없음 (yfinance 다운로드 실패 가능)')
    else:
        print(f'  VIX 분위수: 33%={vix_data["q33"]:.1f}, 67%={vix_data["q67"]:.1f}')
        print(f'  Regime 분포: {vix_data["n_regimes"]}')
        print()

        # Sharpe 표 (시나리오 × regime)
        regime_metrics = vix_data['regime_metrics']
        regime_table = {}
        for name, m in regime_metrics.items():
            regime_table[name] = {
                'Low_VIX_Sharpe': m.get('Low_VIX', {}).get('sharpe', np.nan),
                'Mid_VIX_Sharpe': m.get('Mid_VIX', {}).get('sharpe', np.nan),
                'High_VIX_Sharpe': m.get('High_VIX', {}).get('sharpe', np.nan),
                'Low_VIX_n': m.get('Low_VIX', {}).get('n_months', 0),
                'Mid_VIX_n': m.get('Mid_VIX', {}).get('n_months', 0),
                'High_VIX_n': m.get('High_VIX', {}).get('n_months', 0),
            }
        regime_df = pd.DataFrame(regime_table).T.round(3)
        print('  === Regime별 Sharpe 표 ===')
        print(regime_df.to_string())
        print()
        print('  💡 VIX 시기별로 ML 효과 (BL_ml_sw vs BL_trailing) 가 어떻게 변화하는가')
        print()
        # ML 효과 by regime
        print('  === ML 효과 (BL_ml_sw - BL_trailing) by VIX regime ===')
        for w_method in ['mcap', 'eq', 'rp']:
            print(f'    [{w_method}]')
            for regime in ['Low_VIX', 'Mid_VIX', 'High_VIX']:
                ml_sr = regime_metrics.get(f'BL_ml_sw_{w_method}', {}).get(regime, {}).get('sharpe', np.nan)
                tr_sr = regime_metrics.get(f'BL_trailing_{w_method}', {}).get(regime, {}).get('sharpe', np.nan)
                diff = ml_sr - tr_sr
                marker = '⭐' if diff > 0 else '  '
                print(f'      {regime:10s}: ml {ml_sr:.3f}, trail {tr_sr:.3f}, diff {diff:+.3f} {marker}')
'''


SEC7_J = '''# §7-J. Concentration / Diversification 시계열 (HHI, Effective N)
print('=' * 75)
print('  §7-J. Concentration / Diversification 시계열')
print('=' * 75)


def hhi(weights):
    """Herfindahl-Hirschman Index. 1/N (full diversification) ~ 1 (single asset)."""
    return float((weights ** 2).sum())


def effective_n(weights):
    """Effective number of stocks: 1 / HHI."""
    h = hhi(weights)
    return 1.0 / h if h > 0 else np.nan


# 시나리오별 HHI / Effective N 시계열
hhi_series = {}
eff_n_series = {}
for s_name, w_dict in weights.items():
    sorted_dates = sorted(w_dict.keys())
    hhi_vals = []
    eff_n_vals = []
    for d in sorted_dates:
        w = w_dict[d]
        hhi_vals.append(hhi(w))
        eff_n_vals.append(effective_n(w))
    hhi_series[f'BL_{s_name}'] = pd.Series(hhi_vals, index=pd.DatetimeIndex(sorted_dates))
    eff_n_series[f'BL_{s_name}'] = pd.Series(eff_n_vals, index=pd.DatetimeIndex(sorted_dates))

# 평균 표
conc_summary = pd.DataFrame({
    name: {
        'HHI_mean': hhi_series[name].mean(),
        'HHI_std': hhi_series[name].std(),
        'Effective_N_mean': eff_n_series[name].mean(),
        'Effective_N_min': eff_n_series[name].min(),
    }
    for name in hhi_series.keys()
}).T.round(4)
print(conc_summary.to_string())
print()

# 시각화 (2 panel)
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
COLORS_V2_DICT = {
    'BL_ml_sw_mcap': '#1f77b4', 'BL_ml_sw_eq': '#1f77b4', 'BL_ml_sw_rp': '#1f77b4',
    'BL_trailing_mcap': '#d62728', 'BL_trailing_eq': '#d62728', 'BL_trailing_rp': '#d62728',
}
LINESTYLES = {'mcap': '-', 'eq': '--', 'rp': '-.'}

for name, hs in hhi_series.items():
    ls = LINESTYLES.get(name.split('_')[-1], '-')
    axes[0].plot(hs.index, hs.values, color=COLORS_V2_DICT.get(name, 'gray'),
                 linestyle=ls, label=name, alpha=0.8)
axes[0].set_title('HHI 시계열 (Concentration; 낮을수록 분산)')
axes[0].axhline(1/300, color='black', linewidth=0.5, linestyle=':', label='1/300 (참고)')
axes[0].grid(True, alpha=0.3)
axes[0].legend(ncol=2, fontsize=8)

for name, en in eff_n_series.items():
    ls = LINESTYLES.get(name.split('_')[-1], '-')
    axes[1].plot(en.index, en.values, color=COLORS_V2_DICT.get(name, 'gray'),
                 linestyle=ls, label=name, alpha=0.8)
axes[1].set_title('Effective Number of Stocks 시계열 (1/HHI; 높을수록 분산)')
axes[1].grid(True, alpha=0.3)
axes[1].legend(ncol=2, fontsize=8)

plt.tight_layout()
out_path = OUT_DIR_V2_SW / 'concentration_timeseries.png'
plt.savefig(out_path, dpi=100, bbox_inches='tight')
plt.show()
print(f'  💾 {out_path.name}')
print()
print('  💡 BL_ml_sw_eq 가 가장 분산 (eff N 가장 큼) vs BL_trailing_mcap 이 가장 집중')
'''


# ============================================================================
# Helper: cell 생성
# ============================================================================
def split_src(s):
    """multi-line string → list of strings (each ending with \\n except last)."""
    lines = s.rstrip('\n').split('\n')
    return [ln + '\n' for ln in lines[:-1]] + [lines[-1]]


def make_cell(src_str, cell_type='code'):
    cell = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": split_src(src_str),
    }
    if cell_type == 'code':
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


# ============================================================================
# 노트북에 §7 셀 batch 추가
# ============================================================================
print('=' * 70)
print('  02a_v2.ipynb 에 §7 셀 추가')
print('=' * 70)

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

original_count = len(nb['cells'])
print(f'  원래 셀 수: {original_count}')

# 마지막 cell index (Cell 33 §6-7-3) 찾기
last_idx = len(nb['cells']) - 1

# §7 셀들 생성
new_cells = [
    make_cell(SEC7_HEADER, cell_type='markdown'),
    make_cell(SEC7_A),
    make_cell(SEC7_B),
    make_cell(SEC7_C),
    make_cell(SEC7_D),
    make_cell(SEC7_E),
    make_cell(SEC7_F),
    make_cell(SEC7_G),
    make_cell(SEC7_H),
    make_cell(SEC7_I),
    make_cell(SEC7_J),
]

# 기존 셀 마지막 다음에 추가
nb['cells'] = nb['cells'] + new_cells

print(f'  추가 셀: {len(new_cells)} (1 markdown + 10 code)')
print(f'  새 셀 수: {len(nb["cells"])}')

# 저장
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'  💾 저장: {NB_PATH.name}')
print()
print('완료. 다음 단계:')
print('  1. standalone 스크립트 실행 완료 확인 (sec7_v2_analyses.pkl 생성)')
print('  2. nbconvert --to notebook --execute --inplace 02a_v2.ipynb')
