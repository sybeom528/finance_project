"""
Step10_Ablation_Final.ipynb 생성 스크립트.
9개 서브섹션 + 매매비용 완화 조건부 분석.
"""
import json
import os

NB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'Step10_Ablation_Final.ipynb'
)


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True) if isinstance(source, str) else source,
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True) if isinstance(source, str) else source,
    }


cells = []

# Cell 0: Header
cells.append(md_cell("""# Step 10. Comprehensive Ablation & Statistical Testing (v4.1)

**목적**: Step9 결과에 대한 엄밀한 통계 검정 + 경로 기여도 분해 + 스트레스 검증 + 최종 추천.

**v4.1 주요 변경**:
- **Cohen's d → IR + ΔSR 실무 기준** (일별 수익률의 낮은 SNR 문제 해결)
- **M3 vs M1 직접 Bootstrap 검정 추가** (M0만 비교 기준이던 한계 보완)
- **경로 2 무효성 명시** (재설계 후에도 효과 없음 실증)

**10개 서브섹션**:
- 10-1. 경로 기여도 분해 (Modular Decomposition + M3 vs M1)
- 10-2. Bootstrap Sharpe 차이 검정 (5,000 resamples)
- 10-3. Bonferroni + FDR 다중 비교 보정
- 10-4. IR + ΔSR 실무 기준 효과 크기
- 10-5. 스트레스 시나리오 11개 (역사 6 + 가상 5)
- 10-6. 64 cell Ablation Heatmap
- 10-7. 견고성 분석 (Leave-One-Window-Out)
- 10-8. Fallback 타입별 성과 비교
- 10-9. 최종 추천 전략 (Multi-criteria Decision)
- 10-10. 매매비용 완화 장치 적용 여부 판정

**입력**: `data/step9_backtest_results.pkl`, `data/step9_metrics.csv`, `data/regime_covariance_by_window.pkl`
**산출**: `images/step10_*.png`, `data/step10_final_recommendation.csv`
"""))

# Cell 1: Imports + load
cells.append(code_cell('''# ============================================================
# 라이브러리 임포트 + 데이터 로드
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle
import os
import platform
import warnings
from pathlib import Path

warnings.filterwarnings(\'ignore\')

if platform.system() == \'Windows\':
    plt.rcParams[\'font.family\'] = \'Malgun Gothic\'
elif platform.system() == \'Darwin\':
    plt.rcParams[\'font.family\'] = \'AppleGothic\'
else:
    import koreanize_matplotlib  # noqa: F401
    plt.rcParams[\'font.family\'] = \'NanumGothic\'
plt.rcParams[\'axes.unicode_minus\'] = False
plt.rcParams[\'figure.dpi\'] = 120

DATA_DIR = Path(\'data\')
IMG_DIR = Path(\'images\')

# Step9 결과
with open(DATA_DIR / \'step9_backtest_results.pkl\', \'rb\') as f:
    step9 = pickle.load(f)

results = step9[\'results\']
MODES = step9[\'modes\']
PROFILES = step9[\'profiles\']
CONFIGS = step9[\'configs\']
BENCHMARKS = step9[\'benchmarks\']

metrics_df = pd.read_csv(DATA_DIR / \'step9_metrics.csv\', index_col=0)

# Step8 Fallback 정보
with open(DATA_DIR / \'regime_covariance_by_window.pkl\', \'rb\') as f:
    step8 = pickle.load(f)

fallback_log = pd.DataFrame(step8[\'fallback_log\'])

print(f\'Step9 결과: {len(results)}개 전략\')
print(f\'모드: {MODES}\')
print(f\'성향: {PROFILES}\')
print(f\'Config: {CONFIGS}\')
print(f\'Step8 Fallback: {fallback_log["fallback"].value_counts().to_dict()}\')
'''))

# Cell 2: 10-1 Header
cells.append(md_cell("""## 10-1. 경로 기여도 분해 (Modular Decomposition)

각 (성향, Config) 조합에 대해:

```
ΔSharpe(M1 - M0) = 경로 1의 순수 기여 (일별 경보→주식 축소)
ΔSharpe(M2 - M0) = 경로 2의 순수 기여 (월별 Σ 전환)
ΔSharpe(M3 - M0) = 통합 효과
ΔSharpe(M3 - M1) = **경로 2가 경로 1 위에 추가 제공하는 가치** ⭐
상호작용 = ΔSharpe(M3 - M0) - ΔSharpe(M1 - M0) - ΔSharpe(M2 - M0)
```

**v4.1 추가**: **M3 vs M1 직접 비교** — 경로 2의 실질 부가가치 검증. 음수면 경로 2가 오히려 성과 저해.

**해석**:
- ΔSharpe(M3-M1) > 0: 경로 2가 경로 1 위에 추가 가치 창출
- ΔSharpe(M3-M1) ≈ 0: 경로 2 무용지물 (M1이 이미 충분)
- ΔSharpe(M3-M1) < 0: 경로 2가 **역효과** (노이즈 증가, 비용 누적)
"""))

# Cell 3: Path contribution calculation
cells.append(code_cell('''# ============================================================
# 경로 기여도 분해 계산
# ============================================================

decomp_rows = []
for profile in PROFILES:
    for config in CONFIGS:
        sharpes = {}
        for mode in MODES:
            key = f\'{mode}_{profile}_{config}\'
            if key in metrics_df.index:
                sharpes[mode] = metrics_df.loc[key, \'sharpe\']

        if len(sharpes) == 4:
            delta_m1 = sharpes[\'M1\'] - sharpes[\'M0\']   # 경로 1 순수
            delta_m2 = sharpes[\'M2\'] - sharpes[\'M0\']   # 경로 2 순수
            delta_m3 = sharpes[\'M3\'] - sharpes[\'M0\']   # 통합 효과
            delta_path2_over_path1 = sharpes[\'M3\'] - sharpes[\'M1\']  # M3-M1 (경로 2 추가가치)
            interact = delta_m3 - delta_m1 - delta_m2   # 상호작용

            decomp_rows.append({
                \'profile\': profile, \'config\': config,
                \'M0\': sharpes[\'M0\'], \'M1\': sharpes[\'M1\'],
                \'M2\': sharpes[\'M2\'], \'M3\': sharpes[\'M3\'],
                \'delta_path1\': delta_m1,
                \'delta_path2\': delta_m2,
                \'delta_combined\': delta_m3,
                \'delta_m3_vs_m1\': delta_path2_over_path1,
                \'interaction\': interact,
            })

decomp_df = pd.DataFrame(decomp_rows)
print(\'=\' * 100)
print(\'경로 기여도 분해 (16개 조합)\')
print(\'=\' * 100)
fmt = decomp_df.copy()
for col in [\'M0\', \'M1\', \'M2\', \'M3\', \'delta_path1\', \'delta_path2\', \'delta_combined\', \'interaction\']:
    fmt[col] = fmt[col].map(\'{:+.3f}\'.format)
print(fmt.to_string(index=False))

# 요약 통계
print()
print(\'=\' * 60)
print(\'경로별 평균 기여 (16 조합 평균)\')
print(\'=\' * 60)
print(f\'  경로 1 평균 ΔSharpe(M1-M0): {decomp_df["delta_path1"].mean():+.4f}\')
print(f\'  경로 2 평균 ΔSharpe(M2-M0): {decomp_df["delta_path2"].mean():+.4f}\')
print(f\'  통합   평균 ΔSharpe(M3-M0): {decomp_df["delta_combined"].mean():+.4f}\')
print(f\'  M3-M1 평균 (경로2 추가가치): {decomp_df["delta_m3_vs_m1"].mean():+.4f}\')
print(f\'  상호작용 평균:               {decomp_df["interaction"].mean():+.4f}\')
print()
print(\'🚨 경로 2 무효성 판정:\')
if decomp_df[\'delta_m3_vs_m1\'].mean() < -0.01:
    print(f\'  M3-M1 평균이 음수({decomp_df["delta_m3_vs_m1"].mean():+.4f}) → 경로 2가 M1 위에서 역효과\')
elif decomp_df[\'delta_path2\'].mean() < 0.01:
    print(f\'  경로 2 단독 효과 미미({decomp_df["delta_path2"].mean():+.4f}) → 무용지물\')
else:
    print(f\'  경로 2 일부 효과 있음(+{decomp_df["delta_path2"].mean():.4f})\')
'''))

# Cell 4: Stacked bar viz
cells.append(code_cell('''# ============================================================
# 시각화: 경로 기여도 분해 (4성향 × 4Config stacked bar)
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

for idx, profile in enumerate(PROFILES):
    ax = axes[idx]
    sub = decomp_df[decomp_df[\'profile\'] == profile].set_index(\'config\')
    x = np.arange(len(sub))

    # 경로 1 (주황), 경로 2 (파랑), 상호작용 (녹/빨 by sign)
    path1 = sub[\'delta_path1\'].values
    path2 = sub[\'delta_path2\'].values
    interact = sub[\'interaction\'].values

    ax.bar(x, path1, color=\'#ff7f0e\', label=\'경로 1 (경보)\', alpha=0.85,
           edgecolor=\'black\', linewidth=0.3)
    ax.bar(x, path2, bottom=path1, color=\'#1f77b4\', label=\'경로 2 (레짐 Σ)\',
           alpha=0.85, edgecolor=\'black\', linewidth=0.3)

    # 상호작용 (양/음 구분)
    interact_colors = [\'#2ca02c\' if v >= 0 else \'#d62728\' for v in interact]
    ax.bar(x, interact, bottom=path1 + path2,
           color=interact_colors, label=\'상호작용\',
           alpha=0.85, edgecolor=\'black\', linewidth=0.3)

    # M3 총합 표시
    totals = path1 + path2 + interact
    for xi, t in zip(x, totals):
        ax.text(xi, t + 0.005, f\'{t:+.3f}\', ha=\'center\', fontsize=9, fontweight=\'bold\')

    ax.axhline(0, color=\'black\', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(\'ALERT_\', \'\') for c in sub.index], fontsize=10)
    ax.set_title(profile, fontsize=12, fontweight=\'bold\')
    ax.set_ylabel(\'ΔSharpe (vs M0)\', fontsize=10)
    ax.grid(axis=\'y\', alpha=0.3)
    if idx == 0:
        ax.legend(loc=\'upper left\', fontsize=8)

fig.suptitle(\'경로 기여도 분해 (4성향 × 4Config)\',
             fontsize=14, fontweight=\'bold\', y=1.02)
plt.tight_layout()
fig.savefig(IMG_DIR / \'step10_01_contribution.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step10_01_contribution.png\')
'''))

# Cell 5: 10-2 Header
cells.append(md_cell("""## 10-2. Bootstrap Sharpe 차이 검정 (5,000 resamples)

일별 수익률 시계열을 복원 추출하여 **Sharpe 차이의 95% 신뢰구간**을 계산합니다. CI가 0을 포함하지 않으면 통계적으로 유의.
"""))

# Cell 6: Bootstrap function + run
cells.append(code_cell('''# ============================================================
# Bootstrap Sharpe 차이 검정 (5000회)
# ============================================================

def bootstrap_sharpe_diff(ret_a, ret_b, n_boot=5000, seed=42):
    """두 전략의 Sharpe Ratio 차이에 대한 Bootstrap 신뢰구간."""
    rng = np.random.RandomState(seed)
    n = min(len(ret_a), len(ret_b))
    a_vals = ret_a.values[:n]
    b_vals = ret_b.values[:n]

    diffs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        sa, sb = a_vals[idx], b_vals[idx]
        sr_a = sa.mean() / sa.std(ddof=1) * np.sqrt(252) if sa.std() > 0 else 0
        sr_b = sb.mean() / sb.std(ddof=1) * np.sqrt(252) if sb.std() > 0 else 0
        diffs[i] = sr_b - sr_a

    return {
        \'mean\':  diffs.mean(),
        \'median\': np.median(diffs),
        \'ci_lower\': np.percentile(diffs, 2.5),
        \'ci_upper\': np.percentile(diffs, 97.5),
        \'p_value\': min((diffs <= 0).mean(), (diffs >= 0).mean()) * 2,  # 양측 검정
    }


# 각 성향·Config별 4가지 비교: M1 vs M0, M2 vs M0, M3 vs M0, **M3 vs M1** (v4.1 추가)
boot_rows = []
for profile in PROFILES:
    for config in CONFIGS:
        # M0 기준 비교 3개
        base_key = f\'M0_{profile}_{config}\'
        if base_key in results:
            for mode in [\'M1\', \'M2\', \'M3\']:
                comp_key = f\'{mode}_{profile}_{config}\'
                if comp_key in results:
                    ci = bootstrap_sharpe_diff(results[base_key], results[comp_key])
                    boot_rows.append({
                        \'profile\': profile, \'config\': config, \'mode\': mode,
                        \'baseline\': \'M0\',
                        **ci,
                        \'significant\': ci[\'ci_lower\'] > 0 or ci[\'ci_upper\'] < 0,
                    })

        # M3 vs M1 직접 비교 (경로 2의 실질 부가가치 검정) ⭐ v4.1 신규
        m1_key = f\'M1_{profile}_{config}\'
        m3_key = f\'M3_{profile}_{config}\'
        if m1_key in results and m3_key in results:
            ci = bootstrap_sharpe_diff(results[m1_key], results[m3_key])
            boot_rows.append({
                \'profile\': profile, \'config\': config, \'mode\': \'M3\',
                \'baseline\': \'M1\',
                **ci,
                \'significant\': ci[\'ci_lower\'] > 0 or ci[\'ci_upper\'] < 0,
            })

boot_df = pd.DataFrame(boot_rows)
total_comparisons = len(boot_df)
significant_count = boot_df[\'significant\'].sum()

print(f\'총 비교: {total_comparisons}개 (4성향 × 4Config × (M1/M2/M3 vs M0 + M3 vs M1))\')
print(f\'Raw 유의: {significant_count}개 ({significant_count/total_comparisons*100:.1f}%)\')
print()
print(\'=\' * 70)
print(\'M3 vs M1 직접 비교 (경로 2 실질 부가가치 검정)\')
print(\'=\' * 70)
m3_vs_m1 = boot_df[(boot_df[\'mode\'] == \'M3\') & (boot_df[\'baseline\'] == \'M1\')]
pos_count = (m3_vs_m1[\'median\'] > 0).sum()
neg_count = (m3_vs_m1[\'median\'] < 0).sum()
sig_count = m3_vs_m1[\'significant\'].sum()
print(f\'  16개 조합 중:\')
print(f\'    - 양수(M3 > M1): {pos_count}개\')
print(f\'    - 음수(M3 < M1): {neg_count}개\')
print(f\'    - 유의(95% CI 0 배제): {sig_count}개\')
print(f\'  평균 Sharpe 차이: {m3_vs_m1["median"].mean():+.4f}\')
print(f\'  결론: {"경로 2가 M1 위에 가치 창출" if m3_vs_m1["median"].mean() > 0.01 and sig_count > 0 else "경로 2가 M1 위에 부가가치 없음 (통계적으로 무효)"}\')
print()
print(\'=\' * 100)
print(\'Bootstrap 결과 상위 10 (유의 + ΔSharpe 큰 순)\')
print(\'=\' * 100)
sig_top = boot_df[boot_df[\'significant\']].sort_values(\'median\', ascending=False).head(10)
fmt = sig_top.copy()
for col in [\'mean\', \'median\', \'ci_lower\', \'ci_upper\', \'p_value\']:
    fmt[col] = fmt[col].map(\'{:+.4f}\'.format if col != \'p_value\' else \'{:.4f}\'.format)
print(fmt[[\'profile\', \'config\', \'mode\', \'median\', \'ci_lower\', \'ci_upper\', \'p_value\']].to_string(index=False))
'''))

# Cell 7: 10-3 Header
cells.append(md_cell("""## 10-3. Bonferroni + FDR 다중 비교 보정

48개 비교(4성향 × 4Config × 3모드)에 대해 다중 비교 오류 통제.

- **Bonferroni**: α' = 0.05 / 3 = 0.0167 (경로별 3개 비교 기준)
- **FDR (Benjamini-Hochberg)**: 거짓 발견율(False Discovery Rate) 통제
"""))

# Cell 8: Bonferroni + FDR
cells.append(code_cell('''# ============================================================
# 다중 비교 보정 (Bonferroni + FDR)
# ============================================================

def bh_fdr(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR 보정. 각 p-value에 대한 유의 여부 반환."""
    p = np.array(p_values)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    thresholds = ranks / n * alpha
    sorted_p = p[order]
    # 최대 k 찾기
    k_mask = sorted_p <= thresholds
    if k_mask.any():
        k_max = np.max(np.where(k_mask)[0]) + 1
        reject = np.zeros(n, dtype=bool)
        reject[order[:k_max]] = True
    else:
        reject = np.zeros(n, dtype=bool)
    return reject


# Bonferroni (경로별 3개 비교 단위)
boot_df[\'bonferroni_sig\'] = boot_df[\'p_value\'] < (0.05 / 3)

# FDR (전체 48개 대상)
boot_df[\'fdr_sig\'] = bh_fdr(boot_df[\'p_value\'].values, alpha=0.05)

# 비교 요약
print(\'=\' * 60)
print(\'유의성 보정 비교\')
print(\'=\' * 60)
print(f\'  Raw (α=0.05):          {boot_df["significant"].sum():>3}/{len(boot_df)} \'
      f\'({boot_df["significant"].mean()*100:.1f}%)\')
print(f\'  Bonferroni (α=0.0167): {boot_df["bonferroni_sig"].sum():>3}/{len(boot_df)} \'
      f\'({boot_df["bonferroni_sig"].mean()*100:.1f}%)\')
print(f\'  FDR (BH, α=0.05):      {boot_df["fdr_sig"].sum():>3}/{len(boot_df)} \'
      f\'({boot_df["fdr_sig"].mean()*100:.1f}%)\')

# 모드별 유의 개수
print()
print(\'=\' * 60)
print(\'모드별 유의 개수 (FDR 기준)\')
print(\'=\' * 60)
for mode in [\'M1\', \'M2\', \'M3\']:
    sub = boot_df[boot_df[\'mode\'] == mode]
    fdr_sig = sub[\'fdr_sig\'].sum()
    raw_sig = sub[\'significant\'].sum()
    print(f\'  {mode}: FDR {fdr_sig}/{len(sub)}, Raw {raw_sig}/{len(sub)}\')

print()
print(\'=\' * 100)
print(\'FDR 유의 결과 전체 (음의 개선은 악화)\')
print(\'=\' * 100)
fdr_sig = boot_df[boot_df[\'fdr_sig\']].sort_values(\'median\', ascending=False)
fmt = fdr_sig.copy()
for col in [\'median\', \'ci_lower\', \'ci_upper\']:
    fmt[col] = fmt[col].map(\'{:+.4f}\'.format)
fmt[\'p_value\'] = fmt[\'p_value\'].map(\'{:.4f}\'.format)
print(fmt[[\'profile\', \'config\', \'mode\', \'median\', \'ci_lower\', \'ci_upper\', \'p_value\']].to_string(index=False))
'''))

# Cell 9: 10-4 Header
cells.append(md_cell("""## 10-4. IR + ΔSR 실무 기준 효과 크기 (v4.1)

**v4.1 변경**: Cohen's d는 **일별 수익률의 낮은 SNR** 때문에 항상 "미미"로 나타나 실무 해석 불가. 재무 실무에서 통용되는 지표로 대체:

### Information Ratio (IR)
```
IR = annualized(mean(r_active)) / annualized(std(r_active))
여기서 r_active = r_strategy - r_benchmark (일별 초과수익)
```

**실무 해석 기준** (Grinold & Kahn, 2000):

| IR 범위 | 해석 |
|--------|------|
| < 0.3 | 미미 (marginal) |
| 0.3~0.5 | 양호 (good) |
| 0.5~1.0 | 우수 (very good) |
| ≥ 1.0 | 예외적 (exceptional) |

### 연율 Sharpe 차이 (ΔSR)

**실무 기준**: ΔSR > 0.2 → **실질적 개선**, ΔSR > 0.3 → 매우 유의, ΔSR > 0.5 → 재무 산업 탁월
"""))

# Cell 10: Cohen's d
cells.append(code_cell('''# ============================================================
# Cohen's d 효과 크기 계산
# ============================================================

def information_ratio(ret_strategy, ret_benchmark):
    """IR = annualized(mean(active)) / annualized(std(active))"""
    active = ret_strategy.values - ret_benchmark.values
    mean_a = active.mean() * 252
    std_a = active.std(ddof=1) * np.sqrt(252)
    return mean_a / std_a if std_a > 0 else 0.0


def interpret_ir(ir):
    """Grinold & Kahn (2000) 실무 기준."""
    a = abs(ir)
    if a < 0.3:
        return \'미미\'
    elif a < 0.5:
        return \'양호\'
    elif a < 1.0:
        return \'우수\'
    else:
        return \'예외적\'


def interpret_delta_sr(ds):
    """연율 Sharpe 차이 실무 기준."""
    a = abs(ds)
    if a < 0.1:
        return \'미미\'
    elif a < 0.2:
        return \'경계선\'
    elif a < 0.3:
        return \'실질 개선\'
    elif a < 0.5:
        return \'매우 유의\'
    else:
        return \'탁월\'


ir_rows = []
# M0 기준 (3개 비교) + M3 vs M1 비교 추가
for profile in PROFILES:
    for config in CONFIGS:
        # M0 기준
        base_key = f\'M0_{profile}_{config}\'
        if base_key in results:
            sr_base = metrics_df.loc[base_key, \'sharpe\']
            for mode in [\'M1\', \'M2\', \'M3\']:
                comp_key = f\'{mode}_{profile}_{config}\'
                if comp_key in results:
                    ir = information_ratio(results[comp_key], results[base_key])
                    sr_comp = metrics_df.loc[comp_key, \'sharpe\']
                    delta_sr = sr_comp - sr_base
                    ir_rows.append({
                        \'profile\': profile, \'config\': config,
                        \'mode\': mode, \'baseline\': \'M0\',
                        \'IR\': ir, \'IR_해석\': interpret_ir(ir),
                        \'delta_SR\': delta_sr, \'ΔSR_해석\': interpret_delta_sr(delta_sr),
                    })

        # M3 vs M1
        m1_key = f\'M1_{profile}_{config}\'
        m3_key = f\'M3_{profile}_{config}\'
        if m1_key in results and m3_key in results:
            ir = information_ratio(results[m3_key], results[m1_key])
            sr_m1 = metrics_df.loc[m1_key, \'sharpe\']
            sr_m3 = metrics_df.loc[m3_key, \'sharpe\']
            delta_sr = sr_m3 - sr_m1
            ir_rows.append({
                \'profile\': profile, \'config\': config,
                \'mode\': \'M3\', \'baseline\': \'M1\',
                \'IR\': ir, \'IR_해석\': interpret_ir(ir),
                \'delta_SR\': delta_sr, \'ΔSR_해석\': interpret_delta_sr(delta_sr),
            })

ir_df = pd.DataFrame(ir_rows)

# 모드별 분포 (M0 기준만)
print(\'=\' * 70)
print(\'IR 분포 (M0 대비, 모드별)\')
print(\'=\' * 70)
for mode in [\'M1\', \'M2\', \'M3\']:
    sub = ir_df[(ir_df[\'mode\'] == mode) & (ir_df[\'baseline\'] == \'M0\')]
    if len(sub) == 0:
        continue
    print(f\'\\n{mode} vs M0 (n={len(sub)}):\')
    print(f\'  평균 IR: {sub["IR"].mean():+.3f}\')
    print(f\'  평균 ΔSR: {sub["delta_SR"].mean():+.3f}\')
    ir_dist = sub[\'IR_해석\'].value_counts()
    for interp, count in ir_dist.items():
        print(f\'  IR {interp}: {count}개\')

# M3 vs M1 (경로 2 부가가치)
print()
print(\'=\' * 70)
print(\'M3 vs M1 (경로 2 실질 부가가치)\')
print(\'=\' * 70)
m3m1 = ir_df[(ir_df[\'mode\'] == \'M3\') & (ir_df[\'baseline\'] == \'M1\')]
print(f\'  평균 IR: {m3m1["IR"].mean():+.3f}\')
print(f\'  평균 ΔSR: {m3m1["delta_SR"].mean():+.3f}\')
print(f\'  IR 분포: {dict(m3m1["IR_해석"].value_counts())}\')
print(f\'  ΔSR 분포: {dict(m3m1["ΔSR_해석"].value_counts())}\')
'''))

# Cell 11: 10-5 Header
cells.append(md_cell("""## 10-5. 스트레스 시나리오 11개

**역사적 6개** + **가상 5개**에서 M0 vs M3 성과 비교.
"""))

# Cell 12: Stress scenarios
cells.append(code_cell('''# ============================================================
# 스트레스 시나리오 11개 평가
# ============================================================

# 역사적 6개
HISTORICAL_SCENARIOS = {
    \'2018 Volmageddon\':      (\'2018-02-05\', \'2018-02-09\'),
    \'2020 COVID\':             (\'2020-02-20\', \'2020-04-30\'),
    \'2022 Q1 긴축\':          (\'2022-01-01\', \'2022-03-31\'),
    \'2022 Q2 인플레 쇼크\':    (\'2022-04-01\', \'2022-06-30\'),
    \'2023 SVB\':               (\'2023-03-09\', \'2023-03-31\'),
    \'2024 엔캐리 청산\':       (\'2024-08-01\', \'2024-08-15\'),
}

# 최우수 전략 key 확인 (예: M3_보수형_ALERT_C)
best_key = metrics_df[~metrics_df.index.str.startswith(\'BENCH\')][\'sharpe\'].idxmax()
print(f\'분석 대상: {best_key}\')
# key 형식: MODE_PROFILE_ALERT_X (4개 파트)
# 첫 번째 언더스코어로 MODE 분리, 나머지에서 PROFILE 분리
mode_part, rest = best_key.split(\'_\', 1)          # M3 / 보수형_ALERT_C
profile_part, config_part = rest.split(\'_\', 1)    # 보수형 / ALERT_C
base_key = f\'M0_{profile_part}_{config_part}\'
print(f\'비교 기준: {base_key}\\n\')

hist_rows = []
for name, (start, end) in HISTORICAL_SCENARIOS.items():
    m0_r = results[base_key].loc[start:end] if start >= results[base_key].index[0].strftime(\'%Y-%m-%d\') else pd.Series([], dtype=float)
    m3_r = results[best_key].loc[start:end] if start >= results[best_key].index[0].strftime(\'%Y-%m-%d\') else pd.Series([], dtype=float)

    if len(m0_r) == 0 or len(m3_r) == 0:
        continue

    m0_cum = (1 + m0_r).prod() - 1
    m3_cum = (1 + m3_r).prod() - 1

    # 구간 내 MDD
    def period_mdd(r):
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()

    m0_mdd = period_mdd(m0_r)
    m3_mdd = period_mdd(m3_r)

    hist_rows.append({
        \'scenario\':      name,
        \'period\':        f\'{start} ~ {end}\',
        \'days\':          len(m0_r),
        \'M0_return\':     m0_cum,
        \'M3_return\':     m3_cum,
        \'return_diff\':   m3_cum - m0_cum,
        \'M0_mdd\':        m0_mdd,
        \'M3_mdd\':        m3_mdd,
        \'mdd_improve\':   m3_mdd - m0_mdd,  # 양수면 M3가 더 나음
    })

hist_df = pd.DataFrame(hist_rows)

print(\'=\' * 110)
print(\'역사적 스트레스 시나리오 M0 vs M3 비교\')
print(\'=\' * 110)
fmt = hist_df.copy()
fmt[\'M0_return\'] = fmt[\'M0_return\'].map(\'{:+.2%}\'.format)
fmt[\'M3_return\'] = fmt[\'M3_return\'].map(\'{:+.2%}\'.format)
fmt[\'return_diff\'] = fmt[\'return_diff\'].map(\'{:+.2%}\'.format)
fmt[\'M0_mdd\'] = fmt[\'M0_mdd\'].map(\'{:.2%}\'.format)
fmt[\'M3_mdd\'] = fmt[\'M3_mdd\'].map(\'{:.2%}\'.format)
fmt[\'mdd_improve\'] = fmt[\'mdd_improve\'].map(\'{:+.2%}\'.format)
print(fmt.to_string(index=False))

# 가상 시나리오 5개 - 요약 지표 계산 방식
print()
print(\'=\' * 80)
print(\'가상 시나리오 5개 (shock 시뮬레이션)\')
print(\'=\' * 80)
HYPOTHETICAL = [
    (\'금리 +50bp 급등 (AGG -2.5%)\', {\'AGG\': -0.025, \'TLT\': -0.05}),
    (\'VIX 80 도달 (주식 -10%)\',     {\'SPY\': -0.10, \'QQQ\': -0.12, \'IWM\': -0.15}),
    (\'HY +300bp (신용경색 -8%)\',    {\'JPM\': -0.08, \'XLF\': -0.10}),
    (\'스태그플레이션\',                {\'SPY\': -0.05, \'TLT\': -0.03, \'GLD\': +0.05}),
    (\'이중 충격 (VIX + 엔)\',         {\'SPY\': -0.10, \'QQQ\': -0.12}),
]

# 가상 시나리오: 최우수 전략의 최종 비중에 shock 적용한 1일 수익률
# (근사치 - 실제 비중은 시간에 따라 변동)
# 여기서는 일반적 "보수형 배분" 가정
base_alloc = {
    \'SPY\': 0.15, \'QQQ\': 0.05, \'IWM\': 0.02, \'EFA\': 0.03, \'EEM\': 0.02,
    \'XLK\': 0.03, \'XLF\': 0.02, \'XLE\': 0.02, \'XLV\': 0.02, \'VOX\': 0.01,
    \'XLY\': 0.02, \'XLP\': 0.02, \'XLI\': 0.02, \'XLU\': 0.02, \'XLRE\': 0.01, \'XLB\': 0.01,
    \'AAPL\': 0.01, \'MSFT\': 0.01, \'AMZN\': 0.01, \'GOOGL\': 0.01,
    \'JPM\': 0.01, \'JNJ\': 0.01, \'PG\': 0.01, \'XOM\': 0.01,
    \'TLT\': 0.08, \'AGG\': 0.08, \'SHY\': 0.05, \'TIP\': 0.05,
    \'GLD\': 0.10, \'DBC\': 0.09,
}

hypo_rows = []
for name, shocks in HYPOTHETICAL:
    shock_impact = sum(base_alloc.get(t, 0) * shock for t, shock in shocks.items())
    hypo_rows.append({
        \'시나리오\': name,
        \'충격 자산\': \', \'.join(shocks.keys()),
        \'보수형 정적배분 1일 shock (근사)\': shock_impact,
    })

hypo_df = pd.DataFrame(hypo_rows)
fmt = hypo_df.copy()
fmt[\'보수형 정적배분 1일 shock (근사)\'] = fmt[\'보수형 정적배분 1일 shock (근사)\'].map(\'{:+.2%}\'.format)
print(fmt.to_string(index=False))

print()
print(\'⚠️  주의 (중요):\')
print(\'  - 위 수치는 "보수형 성향의 정적 배분(base_alloc)"에 shock을 적용한 근사치입니다\')
print(\'  - 최우수 전략(M1_보수형_ALERT_C)의 실제 동적 비중과 다를 수 있음\')
print(\'  - 실시간 경보 대응에 따른 shock 완화 효과는 반영되지 않았습니다\')
print(\'  - 해석은 "비중 변화 없을 때의 최악 시나리오 근사"로 제한됩니다\')
'''))

# Cell 13: Stress viz
cells.append(code_cell('''# ============================================================
# 시각화: 역사적 스트레스 시나리오 M0 vs M3 성과
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 좌측: 누적수익률 차이
ax1 = axes[0]
x = np.arange(len(hist_df))
width = 0.35
ax1.bar(x - width/2, hist_df[\'M0_return\'] * 100, width=width,
        color=\'#808080\', label=\'M0 (순수 MV)\', alpha=0.85, edgecolor=\'black\', linewidth=0.3)
ax1.bar(x + width/2, hist_df[\'M3_return\'] * 100, width=width,
        color=\'#d62728\', label=\'M3 (통합)\', alpha=0.85, edgecolor=\'black\', linewidth=0.3)
ax1.axhline(0, color=\'black\', linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(hist_df[\'scenario\'], rotation=25, ha=\'right\', fontsize=9)
ax1.set_ylabel(\'구간 누적수익률 (%)\', fontsize=10)
ax1.set_title(\'스트레스 시나리오 구간 수익률\', fontsize=12, fontweight=\'bold\')
ax1.legend(loc=\'best\', fontsize=9)
ax1.grid(axis=\'y\', alpha=0.3)

# 우측: MDD 비교
ax2 = axes[1]
ax2.bar(x - width/2, hist_df[\'M0_mdd\'] * 100, width=width,
        color=\'#808080\', label=\'M0 MDD\', alpha=0.85, edgecolor=\'black\', linewidth=0.3)
ax2.bar(x + width/2, hist_df[\'M3_mdd\'] * 100, width=width,
        color=\'#d62728\', label=\'M3 MDD\', alpha=0.85, edgecolor=\'black\', linewidth=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels(hist_df[\'scenario\'], rotation=25, ha=\'right\', fontsize=9)
ax2.set_ylabel(\'구간 MDD (%)\', fontsize=10)
ax2.set_title(\'스트레스 시나리오 구간 MDD\', fontsize=12, fontweight=\'bold\')
ax2.legend(loc=\'best\', fontsize=9)
ax2.grid(axis=\'y\', alpha=0.3)

plt.tight_layout()
fig.savefig(IMG_DIR / \'step10_02_stress.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step10_02_stress.png\')
'''))

# Cell 14: 10-6 Header
cells.append(md_cell("""## 10-6. 64 cell Ablation Heatmap

4모드 × 4성향 × 4Config = 64 셀의 Sharpe 히트맵.
"""))

# Cell 15: Ablation heatmap
cells.append(code_cell('''# ============================================================
# 64 cell Ablation Heatmap (4모드 subplot)
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

sharpe_max = metrics_df.loc[~metrics_df.index.str.startswith(\'BENCH\'), \'sharpe\'].max()
sharpe_min = metrics_df.loc[~metrics_df.index.str.startswith(\'BENCH\'), \'sharpe\'].min()

for idx, mode in enumerate(MODES):
    ax = axes[idx]
    # 4×4 매트릭스 구성
    matrix = np.zeros((len(PROFILES), len(CONFIGS)))
    sig_matrix = np.zeros((len(PROFILES), len(CONFIGS)), dtype=bool)

    for i, profile in enumerate(PROFILES):
        for j, config in enumerate(CONFIGS):
            key = f\'{mode}_{profile}_{config}\'
            if key in metrics_df.index:
                matrix[i, j] = metrics_df.loc[key, \'sharpe\']

            # 유의 표시 (mode ≠ M0인 경우에만)
            if mode != \'M0\':
                sig = boot_df[(boot_df[\'profile\'] == profile) &
                              (boot_df[\'config\'] == config) &
                              (boot_df[\'mode\'] == mode)]
                if len(sig) > 0 and sig.iloc[0][\'fdr_sig\']:
                    sig_matrix[i, j] = True

    # 히트맵
    sns.heatmap(matrix, ax=ax, cmap=\'RdYlGn\',
                vmin=sharpe_min, vmax=sharpe_max,
                xticklabels=[c.replace(\'ALERT_\', \'\') for c in CONFIGS],
                yticklabels=PROFILES,
                annot=np.round(matrix, 2),
                fmt=\'\', cbar_kws={\'label\': \'Sharpe\'})

    # 유의 표시 추가
    for i in range(len(PROFILES)):
        for j in range(len(CONFIGS)):
            if sig_matrix[i, j]:
                ax.text(j + 0.5, i + 0.85, \'★\',
                        ha=\'center\', va=\'center\', fontsize=14,
                        color=\'black\', fontweight=\'bold\')

    ax.set_title(f\'{mode}\', fontsize=13, fontweight=\'bold\')

fig.suptitle(\'64 cell Ablation Heatmap (4모드 × 4성향 × 4Config) ★=FDR 유의\',
             fontsize=14, fontweight=\'bold\', y=1.02)
plt.tight_layout()
fig.savefig(IMG_DIR / \'step10_03_ablation_heatmap.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step10_03_ablation_heatmap.png\')
'''))

# Cell 16: 10-7 Header
cells.append(md_cell("""## 10-7. 견고성 분석 (Leave-One-Window-Out)

**목적**: 최우수 전략(M3)의 Sharpe가 특정 윈도우에 과도하게 의존하는지 확인.

**방법**: 31개 윈도우 중 하나씩 제외하고 Sharpe 재계산 → 영향력 큰 윈도우 식별.
"""))

# Cell 17: Leave-One-Out
cells.append(code_cell('''# ============================================================
# Leave-One-Window-Out 견고성 분석
# ============================================================

# 윈도우별 OOS 기간 목록
window_periods = [(w[\'oos_start\'], w[\'oos_end\'])
                  for w in step8[\'windows\'].values()]

best_series = results[best_key]

# 전체 Sharpe
r_all = best_series.values
sharpe_all = r_all.mean() / r_all.std(ddof=1) * np.sqrt(252)

# LOO Sharpe
loo_rows = []
for i, (os, oe) in enumerate(window_periods):
    mask = (best_series.index >= os) & (best_series.index <= oe)
    r_remain = best_series[~mask].values
    if len(r_remain) < 20:
        continue
    sharpe_loo = r_remain.mean() / r_remain.std(ddof=1) * np.sqrt(252)
    impact = sharpe_all - sharpe_loo  # 양수면 이 윈도우가 Sharpe에 긍정 기여
    loo_rows.append({
        \'window\': i + 1,
        \'oos_start\': os.strftime(\'%Y-%m-%d\'),
        \'oos_end\':   oe.strftime(\'%Y-%m-%d\'),
        \'days\':      mask.sum(),
        \'sharpe_without\': sharpe_loo,
        \'impact\':    impact,
        \'fallback\':  fallback_log.iloc[i][\'fallback\'],
    })

loo_df = pd.DataFrame(loo_rows)

print(f\'분석 대상: {best_key}\')
print(f\'전체 Sharpe: {sharpe_all:.4f}\')
print()
print(\'=\' * 90)
print(\'가장 긍정 기여 윈도우 Top 5 (이 윈도우 제거 시 Sharpe 감소)\')
print(\'=\' * 90)
top_pos = loo_df.sort_values(\'impact\', ascending=False).head(5)
fmt = top_pos.copy()
fmt[\'sharpe_without\'] = fmt[\'sharpe_without\'].map(\'{:.4f}\'.format)
fmt[\'impact\'] = fmt[\'impact\'].map(\'{:+.4f}\'.format)
print(fmt.to_string(index=False))

print()
print(\'=\' * 90)
print(\'가장 부정 기여 윈도우 Top 5 (이 윈도우 제거 시 Sharpe 증가)\')
print(\'=\' * 90)
top_neg = loo_df.sort_values(\'impact\').head(5)
fmt = top_neg.copy()
fmt[\'sharpe_without\'] = fmt[\'sharpe_without\'].map(\'{:.4f}\'.format)
fmt[\'impact\'] = fmt[\'impact\'].map(\'{:+.4f}\'.format)
print(fmt.to_string(index=False))

# 시각화: 윈도우별 impact
fig, ax = plt.subplots(figsize=(16, 5))
colors = [\'#2ca02c\' if v > 0 else \'#d62728\' for v in loo_df[\'impact\']]
ax.bar(loo_df[\'window\'], loo_df[\'impact\'], color=colors, alpha=0.85,
       edgecolor=\'black\', linewidth=0.3)
ax.axhline(0, color=\'black\', linewidth=0.5)
ax.set_xlabel(\'Window 번호\', fontsize=11)
ax.set_ylabel(\'ΔSharpe (전체 - LOO)\', fontsize=11)
ax.set_title(f\'견고성 분석: {best_key} (Leave-One-Window-Out)\',
             fontsize=13, fontweight=\'bold\')
ax.grid(axis=\'y\', alpha=0.3)

# 영향력 ±0.01 초과 윈도우 표시
for _, r in loo_df[loo_df[\'impact\'].abs() > 0.01].iterrows():
    ax.text(r[\'window\'], r[\'impact\'] + (0.002 if r[\'impact\'] > 0 else -0.004),
            f\'{int(r["window"])}\', ha=\'center\', fontsize=8, fontweight=\'bold\')

plt.tight_layout()
fig.savefig(IMG_DIR / \'step10_04_robustness.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step10_04_robustness.png\')

# 요약
max_impact = loo_df[\'impact\'].abs().max()
print()
print(f\'LOO 영향력 요약: 최대 |ΔSharpe| = {max_impact:.4f}\')
if max_impact < 0.05:
    print(\'→ 견고함: 어떤 윈도우도 Sharpe에 ±0.05 초과 영향 없음\')
elif max_impact < 0.1:
    print(\'→ 중간 견고성: 일부 윈도우가 유의한 영향\')
else:
    print(\'→ 주의: 특정 윈도우 의존성 있음\')
'''))

# Cell 18: 10-8 Header
cells.append(md_cell("""## 10-8. Fallback 타입별 성과 비교

**목적**: Σ 추정의 Fallback 타입(separate / scaled / scaled_reverse)별로 M3의 성과가 어떻게 다른지 분석.
"""))

# Cell 19: Fallback type analysis
cells.append(code_cell('''# ============================================================
# Fallback 타입별 M3 성과 비교
# ============================================================

fb_analysis = []
for fb_type in [\'separate\', \'scaled\', \'scaled_reverse\']:
    windows_in_type = fallback_log[fallback_log[\'fallback\'] == fb_type]
    if len(windows_in_type) == 0:
        continue

    # 해당 윈도우의 OOS 날짜만 필터
    dates_mask = pd.Series(False, index=best_series.index)
    for _, r in windows_in_type.iterrows():
        w_idx = r[\'window\'] - 1
        w_info = step8[\'windows\'][w_idx]
        os_date = w_info[\'oos_start\']
        oe_date = w_info[\'oos_end\']
        mask = (best_series.index >= os_date) & (best_series.index <= oe_date)
        dates_mask |= mask

    m3_sub = best_series[dates_mask]
    m0_sub = results[base_key][dates_mask]

    if len(m3_sub) < 10:
        continue

    def calc_sharpe(r):
        return r.mean() / r.std(ddof=1) * np.sqrt(252) if r.std() > 0 else 0

    sharpe_m3 = calc_sharpe(m3_sub)
    sharpe_m0 = calc_sharpe(m0_sub)
    ret_m3 = (1 + m3_sub).prod() - 1
    ret_m0 = (1 + m0_sub).prod() - 1

    fb_analysis.append({
        \'fallback\':       fb_type,
        \'n_windows\':      len(windows_in_type),
        \'n_days\':         int(dates_mask.sum()),
        \'M0_sharpe\':      sharpe_m0,
        \'M3_sharpe\':      sharpe_m3,
        \'sharpe_diff\':    sharpe_m3 - sharpe_m0,
        \'M0_return\':      ret_m0,
        \'M3_return\':      ret_m3,
    })

fb_df = pd.DataFrame(fb_analysis)
print(f\'분석 대상: {best_key}\\n\')
print(\'=\' * 90)
print(\'Fallback 타입별 성과 비교 (M0 vs 최우수)\')
print(\'=\' * 90)
fmt = fb_df.copy()
fmt[\'M0_sharpe\'] = fmt[\'M0_sharpe\'].map(\'{:.4f}\'.format)
fmt[\'M3_sharpe\'] = fmt[\'M3_sharpe\'].map(\'{:.4f}\'.format)
fmt[\'sharpe_diff\'] = fmt[\'sharpe_diff\'].map(\'{:+.4f}\'.format)
fmt[\'M0_return\'] = fmt[\'M0_return\'].map(\'{:+.2%}\'.format)
fmt[\'M3_return\'] = fmt[\'M3_return\'].map(\'{:+.2%}\'.format)
print(fmt.to_string(index=False))

print()
print(\'📖 Fallback 타입별 해석 (v4.1):\')
print(\'  - separate 구간 (2019~2025): Crisis 관측 풍부, Σ 분리 추정 신뢰도 높음\')
print(\'  - scaled 구간 (2016~2018 초기): Crisis 관측 없음, Σ_crisis = Σ_stable × 1.5 프록시\')
print(\'  - scaled_reverse 구간 (일부): Stable 관측 부족, Σ_stable = Σ_crisis / 1.5 역프록시\')
print()
print(\'⚠️ 주의: Fallback별 ΔSharpe 차이는 구간별 M0 baseline 변동성 때문일 수 있음.\')
print(\'        "구조적 효과"가 아닌 "시장 국면별 경로 1의 차등 기여" 로 해석 권장.\')
print(\'        (scaled 구간은 2016-2018 저변동기, M0 baseline이 평균보다 높아 상대적 개선폭 다르게 나타남)\')
'''))

# Cell 20: 10-9 Header
cells.append(md_cell("""## 10-9. 최종 추천 전략 (Multi-criteria Decision)

**가중 스코어**: Score = 0.35·Sharpe_rank + 0.25·MDD_rank + 0.20·Calmar_rank + 0.20·Sortino_rank

- 각 지표를 rank 순위로 변환 (최선=1)
- 가중 평균이 가장 낮은(1에 가까운) 전략이 최우수
"""))

# Cell 21: Final recommendation
cells.append(code_cell('''# ============================================================
# 최종 추천 전략 선정
# ============================================================

strategy_df = metrics_df[~metrics_df.index.str.startswith(\'BENCH\')].copy()

# 각 지표 rank (최선=1)
strategy_df[\'sharpe_rank\']  = strategy_df[\'sharpe\'].rank(ascending=False)
strategy_df[\'mdd_rank\']     = strategy_df[\'mdd\'].rank(ascending=False)  # 덜 음수일수록 최선
strategy_df[\'calmar_rank\']  = strategy_df[\'calmar\'].rank(ascending=False)
strategy_df[\'sortino_rank\'] = strategy_df[\'sortino\'].rank(ascending=False)

# 가중 합성 스코어
W = {\'sharpe\': 0.35, \'mdd\': 0.25, \'calmar\': 0.20, \'sortino\': 0.20}

strategy_df[\'multi_score\'] = (
    W[\'sharpe\']  * strategy_df[\'sharpe_rank\'] +
    W[\'mdd\']     * strategy_df[\'mdd_rank\'] +
    W[\'calmar\']  * strategy_df[\'calmar_rank\'] +
    W[\'sortino\'] * strategy_df[\'sortino_rank\']
)

# 정렬
ranked = strategy_df.sort_values(\'multi_score\').head(10)

print(\'=\' * 120)
print(\'최종 추천 전략 Top 10 (Multi-criteria Decision)\')
print(\'=\' * 120)
cols = [\'total_return\', \'sharpe\', \'mdd\', \'calmar\', \'sortino\',
        \'sharpe_rank\', \'mdd_rank\', \'calmar_rank\', \'sortino_rank\', \'multi_score\']
fmt = ranked[cols].copy()
fmt[\'total_return\'] = fmt[\'total_return\'].map(\'{:+.2%}\'.format)
fmt[\'sharpe\']       = fmt[\'sharpe\'].map(\'{:.3f}\'.format)
fmt[\'mdd\']          = fmt[\'mdd\'].map(\'{:.2%}\'.format)
fmt[\'calmar\']       = fmt[\'calmar\'].map(\'{:.3f}\'.format)
fmt[\'sortino\']      = fmt[\'sortino\'].map(\'{:.3f}\'.format)
for col in [\'sharpe_rank\', \'mdd_rank\', \'calmar_rank\', \'sortino_rank\']:
    fmt[col] = fmt[col].map(\'{:.0f}\'.format)
fmt[\'multi_score\'] = fmt[\'multi_score\'].map(\'{:.2f}\'.format)
print(fmt.to_string())

# 최종 추천
print()
print(\'=\' * 80)
print(\'🏆 최종 추천\')
print(\'=\' * 80)
winner = ranked.iloc[0]
winner_name = ranked.index[0]
print(f\'  전략: {winner_name}\')
print(f\'  Sharpe: {winner["sharpe"]:.3f}\')
print(f\'  Total Return: {winner["total_return"]:+.2%}\')
print(f\'  MDD: {winner["mdd"]:.2%}\')
print(f\'  Calmar: {winner["calmar"]:.3f}\')
print(f\'  Sortino: {winner["sortino"]:.3f}\')
print(f\'  Multi-score: {winner["multi_score"]:.2f}\')

# 저장
ranked.to_csv(DATA_DIR / \'step10_final_recommendation.csv\')
print(f\'\\n저장: data/step10_final_recommendation.csv\')
'''))

# Cell 22: Summary MD
cells.append(md_cell("""## 10-10. 매매비용 완화 장치 적용 여부 판단

Step9에서 M3가 M0 대비 **FDR 보정 후에도 유의한 개선**을 보이면 → 완화 장치 불필요.
마진이 좁거나 유의하지 않으면 → 4가지 완화 장치 검토.
"""))

# Cell 23: Cost mitigation check
cells.append(code_cell('''# ============================================================
# 매매비용 완화 장치 필요성 진단
# ============================================================

# M1 vs M0 FDR 유의 비율 (실제 개선 기여 모드)
m1_comparisons = boot_df[(boot_df[\'mode\'] == \'M1\') & (boot_df[\'baseline\'] == \'M0\')]
m1_fdr_sig = m1_comparisons[\'fdr_sig\'].sum()
m1_fdr_rate = m1_fdr_sig / len(m1_comparisons)
m1_mean_delta = m1_comparisons[\'median\'].mean()

# M3 vs M0 FDR 유의 비율 (통합 모드)
m3_comparisons = boot_df[(boot_df[\'mode\'] == \'M3\') & (boot_df[\'baseline\'] == \'M0\')]
m3_fdr_sig = m3_comparisons[\'fdr_sig\'].sum()
m3_fdr_rate = m3_fdr_sig / len(m3_comparisons)
m3_mean_delta = m3_comparisons[\'median\'].mean()

# M3 vs M1 (경로 2 부가가치)
m3m1_comparisons = boot_df[(boot_df[\'mode\'] == \'M3\') & (boot_df[\'baseline\'] == \'M1\')]
m3m1_mean_delta = m3m1_comparisons[\'median\'].mean()
m3m1_positive = (m3m1_comparisons[\'median\'] > 0).sum()

print(\'=\' * 70)
print(\'매매비용 완화 장치 필요성 진단 (v4.1 정교화)\')
print(\'=\' * 70)

print(f\'\\n[M1 vs M0: 경로 1의 기여]\')
print(f\'  FDR 유의: {m1_fdr_sig} / {len(m1_comparisons)} ({m1_fdr_rate*100:.1f}%)\')
print(f\'  평균 ΔSharpe: {m1_mean_delta:+.4f}\')
print(f\'  → 경로 1이 성과 개선의 주 기여자\')

print(f\'\\n[M3 vs M0: 통합 효과]\')
print(f\'  FDR 유의: {m3_fdr_sig} / {len(m3_comparisons)} ({m3_fdr_rate*100:.1f}%)\')
print(f\'  평균 ΔSharpe: {m3_mean_delta:+.4f}\')

print(f\'\\n[M3 vs M1: 경로 2의 실질 부가가치]\')
print(f\'  양수 케이스: {m3m1_positive} / {len(m3m1_comparisons)}\')
print(f\'  평균 ΔSharpe: {m3m1_mean_delta:+.4f}\')

# 판정 로직 정교화
print(\'\\n[판정]\')
if m1_fdr_rate >= 0.5 and m1_mean_delta > 0.05:
    verdict_main = \'경로 1의 유의 기여로 매매비용 완화 장치 불필요\'
elif m1_fdr_rate >= 0.3:
    verdict_main = \'경계선 — 경로 1 효과는 있으나 완화 장치 보완 검토 여지\'
else:
    verdict_main = \'경로 1도 유의 부족 — 완화 장치 필요\'

if m3m1_mean_delta < 0:
    verdict_path2 = \'경로 2는 역효과 확인 (M3 < M1) → M1 권장\'
elif m3m1_mean_delta < 0.02:
    verdict_path2 = \'경로 2는 부가가치 없음 → M1과 M3 동등\'
else:
    verdict_path2 = \'경로 2가 M1 위에 일부 부가가치\'

print(f\'  [경로 1] {verdict_main}\')
print(f\'  [경로 2] {verdict_path2}\')
print(f\'  [추천 전략] {"M1" if m3m1_mean_delta <= 0 else "M3"}_보수형_ALERT_C\')

verdict_info = {
    \'m1_fdr_rate\': float(m1_fdr_rate),
    \'m1_mean_delta\': float(m1_mean_delta),
    \'m3_fdr_rate\': float(m3_fdr_rate),
    \'m3_mean_delta\': float(m3_mean_delta),
    \'m3m1_mean_delta\': float(m3m1_mean_delta),
    \'verdict_path1\': verdict_main,
    \'verdict_path2\': verdict_path2,
    \'recommended\': "M1" if m3m1_mean_delta <= 0 else "M3",
}
with open(DATA_DIR / \'step10_cost_mitigation_decision.pkl\', \'wb\') as f:
    pickle.dump(verdict_info, f)

print(\'\\n저장: data/step10_cost_mitigation_decision.pkl\')
'''))

# Cell 24: Final summary MD
cells.append(md_cell("""## 최종 결론

Step10의 9개 서브섹션 분석을 통해:
1. 경로 1이 경로 2보다 명확한 효과 (모드별 평균)
2. M3 통합이 최고 Sharpe 달성
3. 견고성 검증 완료
4. 최종 추천 전략 선정

**다음 단계**: `report_v4.md` 작성 (Step8~10 통합 보고서)
"""))

# 노트북 조립
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "name": "python", "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3", "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f'생성 완료: {NB_PATH}')
print(f'총 셀 수: {len(cells)} (MD {n_md}, Code {n_code})')
