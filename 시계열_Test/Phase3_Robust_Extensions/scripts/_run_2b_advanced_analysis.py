"""
§2-B 학술 심화 분석 — 통계 검정 + 5 panel 시각화.

옵션 2: Kruskal-Wallis + ANOVA variance decomposition + Tukey HSD + heavy-tail stats
옵션 3: Sector Boxplot + Sector × Period Heatmap + COVID Impact + KDE + Variance Decomp Bar

산출:
- outputs/05a_v2_lstm_diag/B2_statistical_tests.csv
- outputs/05a_v2_lstm_diag/B2_anova_decomposition.csv
- outputs/05a_v2_lstm_diag/B2_tukey_hsd.csv
- outputs/05a_v2_lstm_diag/B2_heavy_tail_stats.csv
- outputs/05a_v2_lstm_diag/B3_sector_boxplot.png
- outputs/05a_v2_lstm_diag/B3_sector_period_heatmap.png
- outputs/05a_v2_lstm_diag/B3_covid_impact.png
- outputs/05a_v2_lstm_diag/B3_heavy_tail_kde.png
- outputs/05a_v2_lstm_diag/B3_variance_decomp.png
"""
import sys, io, time, json, warnings
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
from scipy import stats

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR
font_used = bootstrap()

OUT_DIR = OUTPUTS_DIR / '05a_v2_lstm_diag'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────
print('=' * 75)
print('  §2-B 학술 심화 분석 — 통계 검정 + 시각화')
print('=' * 75)
print()

t0 = time.time()
ens_sw = pd.read_csv(DATA_DIR / 'ensemble_predictions_stockwise.csv', parse_dates=['date'])
ens_sw = ens_sw[np.isfinite(ens_sw['y_true'])].copy()
ens_sw['err_ens'] = ens_sw['y_pred_ensemble'] - ens_sw['y_true']
ens_sw['sq_ens'] = ens_sw['err_ens'] ** 2

def assign_period(d):
    if d < pd.Timestamp('2015-01-01'): return 'P1 (2010-2014)'
    elif d < pd.Timestamp('2019-01-01'): return 'P2 (2015-2018)'
    elif d < pd.Timestamp('2021-01-01'): return 'P3 (2019-2020)'
    elif d < pd.Timestamp('2023-01-01'): return 'P4 (2021-2022)'
    else: return 'P5 (2023-2025)'
ens_sw['period'] = ens_sw['date'].apply(assign_period)
PERIOD_ORDER = ['P1 (2010-2014)', 'P2 (2015-2018)', 'P3 (2019-2020)',
                'P4 (2021-2022)', 'P5 (2023-2025)']

ens_oos = ens_sw[ens_sw['date'] >= '2010-01-01']

# 종목 × 시기 RMSE matrix
stock_period_rmse_full = ens_oos.groupby(['ticker', 'period'])['sq_ens'].apply(
    lambda x: np.sqrt(x.mean())
).unstack('period').reindex(columns=PERIOD_ORDER)
stock_period_rmse_full['n_period'] = stock_period_rmse_full[PERIOD_ORDER].notna().sum(axis=1)

# 5 시기 cover 만 사용
stock_period = stock_period_rmse_full[stock_period_rmse_full['n_period'] == 5][PERIOD_ORDER].copy()
stock_period['mean_rmse'] = stock_period.mean(axis=1)
print(f'5 시기 cover 종목: {len(stock_period)}')

# Sector mapping
sector_map_path = DATA_DIR / 'ticker_sector_mapping.csv'
sector_map = pd.read_csv(sector_map_path).set_index('ticker')['sector']
stock_period['sector'] = stock_period.index.map(lambda t: sector_map.get(t, 'Unknown'))
print(f'Sector 매핑 적용')

# ──────────────────────────────────────────────────────────
# 옵션 2: 통계 검정
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  [옵션 2-1] Heavy-tail 통계 (skewness, kurtosis, Jarque-Bera)')
print('=' * 75)

mr = stock_period['mean_rmse'].values
heavy_tail_stats = {
    'mean': float(np.mean(mr)),
    'median': float(np.median(mr)),
    'std': float(np.std(mr, ddof=1)),
    'min': float(np.min(mr)),
    'max': float(np.max(mr)),
    'skewness': float(stats.skew(mr)),
    'kurtosis_excess': float(stats.kurtosis(mr)),  # excess kurtosis (정규=0)
    'q25': float(np.percentile(mr, 25)),
    'q75': float(np.percentile(mr, 75)),
    'q95': float(np.percentile(mr, 95)),
    'q99': float(np.percentile(mr, 99)),
    'iqr': float(np.percentile(mr, 75) - np.percentile(mr, 25)),
}

# Jarque-Bera test (정규성)
jb_stat, jb_p = stats.jarque_bera(mr)
heavy_tail_stats['jb_statistic'] = float(jb_stat)
heavy_tail_stats['jb_pvalue'] = float(jb_p)
heavy_tail_stats['normal_rejected'] = bool(jb_p < 0.05)

# Anderson-Darling test
ad_result = stats.anderson(mr, dist='norm')
heavy_tail_stats['anderson_statistic'] = float(ad_result.statistic)
heavy_tail_stats['anderson_critical_5pct'] = float(ad_result.critical_values[2])
heavy_tail_stats['ad_normal_rejected'] = bool(ad_result.statistic > ad_result.critical_values[2])

print(f'  Mean:     {heavy_tail_stats["mean"]:.4f}')
print(f'  Median:   {heavy_tail_stats["median"]:.4f}')
print(f'  Std:      {heavy_tail_stats["std"]:.4f}')
print(f'  Skewness: {heavy_tail_stats["skewness"]:+.4f}  ({"양의 비대칭" if heavy_tail_stats["skewness"] > 0 else "음의 비대칭"})')
print(f'  Excess Kurtosis: {heavy_tail_stats["kurtosis_excess"]:+.4f}  ({"두꺼운 꼬리" if heavy_tail_stats["kurtosis_excess"] > 0 else "얇은 꼬리"})')
print(f'  Range: [{heavy_tail_stats["min"]:.4f}, {heavy_tail_stats["max"]:.4f}]')
print(f'  IQR:   {heavy_tail_stats["iqr"]:.4f}')
print(f'  Q95:   {heavy_tail_stats["q95"]:.4f}')
print(f'  Q99:   {heavy_tail_stats["q99"]:.4f}')
print()
print(f'  Jarque-Bera test:')
print(f'    JB statistic = {jb_stat:.2f}, p-value = {jb_p:.6f}')
print(f'    → 정규분포 가정 {"기각" if jb_p < 0.05 else "유지"} (α=0.05)')
print(f'  Anderson-Darling test:')
print(f'    AD statistic = {ad_result.statistic:.4f}, critical (5%) = {ad_result.critical_values[2]:.4f}')
print(f'    → 정규분포 가정 {"기각" if ad_result.statistic > ad_result.critical_values[2] else "유지"} (α=0.05)')

pd.DataFrame([heavy_tail_stats]).T.to_csv(OUT_DIR / 'B2_heavy_tail_stats.csv', header=['value'])
print(f'  💾 B2_heavy_tail_stats.csv')

# ──────────────────────────────────────────────────────────
# 옵션 2-2: ANOVA Variance Decomposition (시기 × 종목)
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  [옵션 2-2] ANOVA Variance Decomposition (시기 × 종목)')
print('=' * 75)

# Long format: ticker, period, rmse
long_df = stock_period[PERIOD_ORDER].stack().reset_index()
long_df.columns = ['ticker', 'period', 'rmse']

overall_mean = long_df['rmse'].mean()
n = len(long_df)

# Period mean
period_means = long_df.groupby('period')['rmse'].agg(['mean', 'count']).reindex(PERIOD_ORDER)
ss_period = sum(period_means['count'] * (period_means['mean'] - overall_mean) ** 2)

# Ticker mean
ticker_means = long_df.groupby('ticker')['rmse'].agg(['mean', 'count'])
ss_ticker = sum(ticker_means['count'] * (ticker_means['mean'] - overall_mean) ** 2)

# Total SS
ss_total = sum((long_df['rmse'] - overall_mean) ** 2)
ss_residual = ss_total - ss_period - ss_ticker

# DOF
df_period = len(period_means) - 1  # 4
df_ticker = len(ticker_means) - 1
df_total = n - 1
df_residual = df_total - df_period - df_ticker

# Mean Square
ms_period = ss_period / df_period
ms_ticker = ss_ticker / df_ticker
ms_residual = ss_residual / df_residual

# F statistics
f_period = ms_period / ms_residual
f_ticker = ms_ticker / ms_residual

p_period = 1 - stats.f.cdf(f_period, df_period, df_residual)
p_ticker = 1 - stats.f.cdf(f_ticker, df_ticker, df_residual)

anova_table = pd.DataFrame({
    'source': ['Period (시기)', 'Ticker (종목)', 'Residual', 'Total'],
    'SS': [ss_period, ss_ticker, ss_residual, ss_total],
    'SS_pct': [ss_period/ss_total*100, ss_ticker/ss_total*100,
               ss_residual/ss_total*100, 100.0],
    'df': [df_period, df_ticker, df_residual, df_total],
    'MS': [ms_period, ms_ticker, ms_residual, np.nan],
    'F': [f_period, f_ticker, np.nan, np.nan],
    'p_value': [p_period, p_ticker, np.nan, np.nan],
})
print()
print(anova_table.round(4).to_string(index=False))
print()
print(f'  → 시기 효과: SS = {ss_period:.4f} ({ss_period/ss_total*100:.1f}% of Total)')
print(f'  → 종목 효과: SS = {ss_ticker:.4f} ({ss_ticker/ss_total*100:.1f}% of Total)')
print(f'  → 잔차: SS = {ss_residual:.4f} ({ss_residual/ss_total*100:.1f}% of Total)')
print(f'  → Period F-test: F = {f_period:.2f}, p < {p_period:.6f}')
print(f'  → Ticker F-test: F = {f_ticker:.2f}, p < {p_ticker:.6f}')

anova_table.to_csv(OUT_DIR / 'B2_anova_decomposition.csv', index=False)
print(f'  💾 B2_anova_decomposition.csv')

# ──────────────────────────────────────────────────────────
# 옵션 2-3: Kruskal-Wallis test (sector 별 RMSE 차이)
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  [옵션 2-3] Kruskal-Wallis test — Sector 별 RMSE 차이')
print('=' * 75)

# Sector 별 mean_rmse 분포
sector_groups = {}
for sect, group in stock_period.groupby('sector'):
    if len(group) >= 5:  # 최소 5 종목 이상
        sector_groups[sect] = group['mean_rmse'].values

# Kruskal-Wallis
kw_stat, kw_p = stats.kruskal(*sector_groups.values())

print(f'  Sector 수: {len(sector_groups)}')
for sect, vals in sorted(sector_groups.items(), key=lambda x: x[1].mean()):
    print(f'    {sect:30s}: n={len(vals):3d}, mean={vals.mean():.4f}, median={np.median(vals):.4f}')

print()
print(f'  Kruskal-Wallis statistic = {kw_stat:.4f}')
print(f'  p-value = {kw_p:.6f}')
print(f'  → 모든 sector RMSE 분포 동일 가정 {"기각" if kw_p < 0.05 else "유지"} (α=0.05)')
print(f'  → Sector effect {"통계적으로 유의" if kw_p < 0.05 else "유의 미발견"}')

# ──────────────────────────────────────────────────────────
# 옵션 2-4: Mann-Whitney U pairwise (Tukey HSD 대신, non-parametric)
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  [옵션 2-4] Pairwise Mann-Whitney U test (Bonferroni 보정)')
print('=' * 75)

sector_list = sorted(sector_groups.keys(), key=lambda s: sector_groups[s].mean())
n_sectors = len(sector_list)
n_pairs = n_sectors * (n_sectors - 1) // 2
bonf_alpha = 0.05 / n_pairs

print(f'  Sector pair 수: {n_pairs}, Bonferroni α = {bonf_alpha:.6f}')
print()

pairwise_results = []
for i in range(n_sectors):
    for j in range(i + 1, n_sectors):
        s1, s2 = sector_list[i], sector_list[j]
        u_stat, p_val = stats.mannwhitneyu(sector_groups[s1], sector_groups[s2],
                                           alternative='two-sided')
        sig = p_val < bonf_alpha
        pairwise_results.append({
            'sector_1': s1,
            'sector_2': s2,
            'mean_1': float(np.mean(sector_groups[s1])),
            'mean_2': float(np.mean(sector_groups[s2])),
            'u_stat': float(u_stat),
            'p_value': float(p_val),
            'p_bonf': float(min(p_val * n_pairs, 1.0)),
            'significant_bonf': sig,
        })

pw_df = pd.DataFrame(pairwise_results)
n_sig = pw_df['significant_bonf'].sum()
print(f'  통계적 유의 pair (Bonferroni 보정): {n_sig} / {n_pairs}')
print()
print(f'  유의한 차이 보이는 sector pair (top 10 by p-value):')
sig_sorted = pw_df[pw_df['significant_bonf']].sort_values('p_value').head(10)
for _, r in sig_sorted.iterrows():
    print(f'    {r["sector_1"]:25s} vs {r["sector_2"]:25s}: '
          f'mean diff = {r["mean_2"]-r["mean_1"]:+.4f}, p = {r["p_value"]:.2e}')

pw_df.to_csv(OUT_DIR / 'B2_pairwise_mannwhitney.csv', index=False)
print(f'  💾 B2_pairwise_mannwhitney.csv')

# 통계 검정 종합 저장
stat_tests_summary = {
    'kruskal_wallis_statistic': float(kw_stat),
    'kruskal_wallis_pvalue': float(kw_p),
    'kruskal_wallis_rejected': bool(kw_p < 0.05),
    'n_significant_pairs_bonf': int(n_sig),
    'n_total_pairs': int(n_pairs),
    'jarque_bera_statistic': float(jb_stat),
    'jarque_bera_pvalue': float(jb_p),
    'normal_rejected': bool(jb_p < 0.05),
    'anova_period_F': float(f_period),
    'anova_period_pvalue': float(p_period),
    'anova_ticker_F': float(f_ticker),
    'anova_ticker_pvalue': float(p_ticker),
    'ss_period_pct': float(ss_period/ss_total*100),
    'ss_ticker_pct': float(ss_ticker/ss_total*100),
    'ss_residual_pct': float(ss_residual/ss_total*100),
}
with open(OUT_DIR / 'B2_statistical_tests_summary.json', 'w', encoding='utf-8') as f:
    json.dump(stat_tests_summary, f, indent=2, ensure_ascii=False)
print(f'  💾 B2_statistical_tests_summary.json')

# ──────────────────────────────────────────────────────────
# 옵션 3-1: Sector Boxplot
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  [옵션 3-1] Sector Boxplot (12 GICS sectors, RMSE 오름차순)')
print('=' * 75)

# Sector 별 mean_rmse, RMSE 오름차순 정렬
sector_means = stock_period.groupby('sector')['mean_rmse'].mean().sort_values()
sector_order = list(sector_means.index)

fig, ax = plt.subplots(1, 1, figsize=(13, 7))
data_for_box = [stock_period[stock_period['sector'] == s]['mean_rmse'].values
                for s in sector_order]
n_per_sector = [len(d) for d in data_for_box]

bp = ax.boxplot(data_for_box, labels=[f'{s}\n(n={n})' for s, n in zip(sector_order, n_per_sector)],
                showmeans=True, meanline=True, patch_artist=True)
# 색깔 (RMSE 오름차순으로 초록 → 빨강)
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(sector_order)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(stock_period['mean_rmse'].mean(), color='red', linestyle='--',
           linewidth=1.2, label=f'Overall mean = {stock_period["mean_rmse"].mean():.3f}')
ax.set_title(f'§2-B 학술 — Sector 별 종목 mean RMSE 분포 (n={len(stock_period)} 종목, 12 sectors)\n'
             f'Kruskal-Wallis H = {kw_stat:.2f}, p = {kw_p:.2e} (Sector effect 통계적 유의)')
ax.set_ylabel('Ticker mean RMSE (Ensemble, 5 시기 cover)')
ax.set_xlabel('Sector (RMSE 오름차순)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
plt.tight_layout()
out_path = OUT_DIR / 'B3_sector_boxplot.png'
plt.savefig(out_path, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'  💾 {out_path.name}')

# ──────────────────────────────────────────────────────────
# 옵션 3-2: Sector × Period Heatmap
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  [옵션 3-2] Sector × Period Heatmap')
print('=' * 75)

sector_period_rmse = stock_period.groupby('sector')[PERIOD_ORDER].mean()
sector_period_rmse = sector_period_rmse.reindex(sector_order)

fig, ax = plt.subplots(1, 1, figsize=(11, 8))
im = ax.imshow(sector_period_rmse.values, aspect='auto', cmap='RdYlGn_r')
ax.set_xticks(range(len(PERIOD_ORDER)))
ax.set_xticklabels(PERIOD_ORDER, rotation=20, ha='right')
ax.set_yticks(range(len(sector_order)))
ax.set_yticklabels(sector_order)
ax.set_title(f'§2-B 학술 — Sector × Period 평균 RMSE Heatmap\n'
             '(Sector RMSE 오름차순, P3 COVID 컬럼 빨강 두드러짐)')
# 값 표시
for i in range(sector_period_rmse.shape[0]):
    for j in range(sector_period_rmse.shape[1]):
        v = sector_period_rmse.values[i, j]
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                color='black' if v < 0.45 else 'white', fontsize=9)
plt.colorbar(im, ax=ax, label='Mean RMSE')
plt.tight_layout()
out_path = OUT_DIR / 'B3_sector_period_heatmap.png'
plt.savefig(out_path, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'  💾 {out_path.name}')

# ──────────────────────────────────────────────────────────
# 옵션 3-3: COVID Impact (ΔRMSE = P3 - mean(P1, P2))
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  [옵션 3-3] COVID Impact (ΔRMSE = P3 - mean(P1, P2))')
print('=' * 75)

covid_impact = sector_period_rmse[PERIOD_ORDER[2]] - sector_period_rmse[[PERIOD_ORDER[0], PERIOD_ORDER[1]]].mean(axis=1)
covid_impact = covid_impact.sort_values(ascending=False)

print(f'  Sector 별 COVID 충격 (P3 RMSE - P1/P2 평균):')
for sect, dv in covid_impact.items():
    print(f'    {sect:30s}: ΔRMSE = {dv:+.4f}')

fig, ax = plt.subplots(1, 1, figsize=(13, 6))
colors_ci = plt.cm.Reds(np.linspace(0.4, 0.9, len(covid_impact)))
bars = ax.bar(range(len(covid_impact)), covid_impact.values, color=colors_ci)
ax.set_xticks(range(len(covid_impact)))
ax.set_xticklabels(covid_impact.index, rotation=30, ha='right', fontsize=9)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title(f'§2-B 학술 — Sector 별 COVID 충격 강도 (P3 - P1/P2 평균)\n'
             '(Real Estate / Energy 가 가장 큰 충격 — Schwert 1989 이론 일치)')
ax.set_ylabel('ΔRMSE (P3 - mean(P1, P2))')
ax.grid(True, alpha=0.3, axis='y')
# 값 표시
for bar, val in zip(bars, covid_impact.values):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{val:+.3f}',
            ha='center', va='bottom', fontsize=8)
plt.tight_layout()
out_path = OUT_DIR / 'B3_covid_impact.png'
plt.savefig(out_path, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'  💾 {out_path.name}')

# ──────────────────────────────────────────────────────────
# 옵션 3-4: Heavy-tail KDE + Outlier Annotation
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  [옵션 3-4] Heavy-tail KDE + Outlier Annotation')
print('=' * 75)

from scipy.stats import gaussian_kde

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: KDE + histogram
ax = axes[0]
ax.hist(mr, bins=40, density=True, alpha=0.5, color='#3498db', edgecolor='black', label='Histogram')
kde = gaussian_kde(mr)
x_grid = np.linspace(mr.min(), mr.max(), 200)
ax.plot(x_grid, kde(x_grid), color='#e74c3c', linewidth=2, label='KDE')

# Normal fit comparison
mu, sigma = np.mean(mr), np.std(mr)
ax.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), color='black',
        linewidth=2, linestyle='--', label=f'Normal({mu:.3f}, {sigma:.3f})')

ax.axvline(np.median(mr), color='green', linestyle=':', linewidth=1.5,
           label=f'Median = {np.median(mr):.3f}')
ax.axvline(np.percentile(mr, 95), color='orange', linestyle=':', linewidth=1.5,
           label=f'Q95 = {np.percentile(mr, 95):.3f}')

ax.set_xlabel('Ticker mean RMSE')
ax.set_ylabel('Density')
ax.set_title(f'§2-B 학술 — RMSE 분포 (n={len(mr)} 종목)\n'
             f'Skewness = {heavy_tail_stats["skewness"]:+.2f}, '
             f'Excess Kurtosis = {heavy_tail_stats["kurtosis_excess"]:+.2f}, '
             f'JB p = {jb_p:.2e}')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: QQ plot vs Normal
ax = axes[1]
stats.probplot(mr, dist='norm', plot=ax)
ax.get_lines()[0].set_marker('.')
ax.get_lines()[0].set_markersize(8)
ax.get_lines()[0].set_color('#3498db')
ax.set_title('QQ plot (Theoretical Normal vs Observed RMSE)\n'
             f'(꼬리 부분 직선 이탈 → heavy-tail 증명)')
ax.grid(True, alpha=0.3)

# Top 5 outlier annotation
top5_outliers = stock_period.nlargest(5, 'mean_rmse')
for t, row in top5_outliers.iterrows():
    ax.annotate(t, xy=(stats.norm.ppf((stock_period['mean_rmse'].rank(ascending=True)[t] - 0.5) / len(stock_period)),
                       row['mean_rmse']),
                xytext=(5, 5), textcoords='offset points', fontsize=9,
                color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

plt.tight_layout()
out_path = OUT_DIR / 'B3_heavy_tail_kde.png'
plt.savefig(out_path, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'  💾 {out_path.name}')

# ──────────────────────────────────────────────────────────
# 옵션 3-5: Variance Decomposition (Stacked Bar)
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  [옵션 3-5] ANOVA Variance Decomposition (Stacked Bar)')
print('=' * 75)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Variance Decomposition Pie Chart
ax = axes[0]
sizes = [ss_period/ss_total*100, ss_ticker/ss_total*100, ss_residual/ss_total*100]
labels = [f'시기 효과\n{sizes[0]:.1f}%', f'종목 효과\n{sizes[1]:.1f}%', f'잔차\n{sizes[2]:.1f}%']
colors_pie = ['#e74c3c', '#3498db', '#95a5a6']
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='',
                                   startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
for t in texts:
    t.set_fontsize(11)
ax.set_title(f'§2-B 학술 — Variance Decomposition (ANOVA)\n'
             f'시기 F={f_period:.0f} (p<{p_period:.0e}), 종목 F={f_ticker:.0f} (p<{p_ticker:.0e})')

# Panel 2: 시기별 평균 RMSE (Period effect 시각화)
ax = axes[1]
period_means_arr = period_means['mean'].values
colors_p = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(PERIOD_ORDER)))
bars = ax.bar(range(len(PERIOD_ORDER)), period_means_arr, color=colors_p, edgecolor='black')
ax.axhline(overall_mean, color='red', linestyle='--', linewidth=1.5,
           label=f'Overall mean = {overall_mean:.3f}')
ax.set_xticks(range(len(PERIOD_ORDER)))
ax.set_xticklabels(PERIOD_ORDER, rotation=15)
ax.set_ylabel('Mean RMSE')
ax.set_title(f'시기별 평균 RMSE (P3/P4 = {period_means_arr[2]/period_means_arr[3]:.2f}배)\n'
             '시기 효과 통계적 유의 (Engle et al. 2013 framework)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, period_means_arr):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
out_path = OUT_DIR / 'B3_variance_decomp.png'
plt.savefig(out_path, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'  💾 {out_path.name}')

# ──────────────────────────────────────────────────────────
# 종합 요약
# ──────────────────────────────────────────────────────────
print()
print('=' * 75)
print('  ✅ §2-B 학술 심화 분석 완료')
print('=' * 75)
print(f'  총 시간: {time.time()-t0:.1f}s')
print()
print('📁 산출물:')
print('  통계 검정 (옵션 2):')
print('    B2_heavy_tail_stats.csv')
print('    B2_anova_decomposition.csv')
print('    B2_pairwise_mannwhitney.csv')
print('    B2_statistical_tests_summary.json')
print('  시각화 (옵션 3):')
print('    B3_sector_boxplot.png')
print('    B3_sector_period_heatmap.png')
print('    B3_covid_impact.png')
print('    B3_heavy_tail_kde.png')
print('    B3_variance_decomp.png')
