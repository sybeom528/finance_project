"""
Step11_Top10_Composition_Analysis.ipynb 생성 스크립트.
Top 10 전략의 시간에 따른 자산 구성 변화 + 근거 시각화 8종.
"""
import json
import os

NB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'Step11_Top10_Composition_Analysis.ipynb'
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

# ============================================================
# Cell 0: Header
# ============================================================
cells.append(md_cell("""# Step 11. Top 10 전략 자산 구성 시간 변화 시각화

**목적**: Step 10에서 확정된 Top 10 전략의 **일별 자산 비중 시계열**을 재생성하고, 시간에 따른 구성 변화와 그 근거(경보·레짐·Σ 전환)를 8종 시각화로 종합 제시.

**왜 필요한가**:
- Sharpe·MDD 같은 요약 숫자만으로는 전략 메커니즘 이해 한계
- 투자자 입장: "언제, 어떤 자산을, 왜 바꿨는지"를 눈으로 확인해야 신뢰 가능
- Step 9는 일별 수익률만 저장 → **weights 재시뮬 필요**

**Top 10 전략** (step10_final_recommendation.csv 기준):
1. M1_보수형_ALERT_B (Sharpe 1.064, MDD -15.53%)
2. M1_중립형_ALERT_C (1.066, -20.30%)
3. M3_중립형_ALERT_C (1.028, -19.48%)
4. M1_중립형_ALERT_B (1.014, -20.05%)
5. M1_보수형_ALERT_C (1.072, -16.97%)
6. M3_보수형_ALERT_C (1.026, -16.77%)
7. M1_보수형_ALERT_A (0.988, -16.78%)
8. M3_보수형_ALERT_B (0.958, -15.91%)
9. M3_적극형_ALERT_C (0.984, -22.28%)
10. M1_적극형_ALERT_C (0.981, -22.26%)

**시각화 8종**:
- 11-1. 통합 대시보드 (Top 1 기준, 2×3 패널)
- 11-2. Top 1 30자산 Stacked Area + 경보 타임라인
- 11-3. Top 10 Small Multiples (자산군 집계)
- 11-4. 분기별 자산 비중 히트맵
- 11-5. 경보 ↔ 주식 비중 동조 (3-panel)
- 11-6. M1 vs M3 레짐·Σ 전환 오버레이
- 11-7. Turnover 원인 분류 (WF/Σ전환/경보)
- 11-8. 앵커 vs 반응 자산 산점도

**입력**:
- `data/step10_final_recommendation.csv` (Top 10)
- `data/step9_backtest_results.pkl` (검증용)
- `data/portfolio_prices.csv`, `alert_signals.csv`, `profiles.csv`
- `data/regime_history.csv`, `regime_covariance_by_window.pkl`

**산출물**:
- `data/step11_top10_weights.pkl`
- `images/step11_01~08_*.png` (8개)
"""))

# ============================================================
# Cell 1: Imports
# ============================================================
cells.append(code_cell('''# ============================================================
# 라이브러리 임포트 + 한글 폰트 + 경로
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns
import pickle
import os
import platform
import warnings
import time
from pathlib import Path

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

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
DATA_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

print(\'Step 11 라이브러리 임포트 완료\')
'''))

# ============================================================
# Cell 2: 데이터 로드
# ============================================================
cells.append(code_cell('''# ============================================================
# 데이터 로드
# ============================================================

prices = pd.read_csv(DATA_DIR / \'portfolio_prices.csv\', parse_dates=[\'Date\'], index_col=\'Date\')
alerts = pd.read_csv(DATA_DIR / \'alert_signals.csv\', parse_dates=[\'Date\'], index_col=\'Date\')
profiles_df = pd.read_csv(DATA_DIR / \'profiles.csv\')
regime_df = pd.read_csv(DATA_DIR / \'regime_history.csv\', parse_dates=[\'Date\'], index_col=\'Date\')

with open(DATA_DIR / \'regime_covariance_by_window.pkl\', \'rb\') as f:
    step8_data = pickle.load(f)
windows_cov = step8_data[\'windows\']
STABLE_REGIMES = step8_data[\'stable_regimes\']
CRISIS_REGIMES = step8_data[\'crisis_regimes\']

with open(DATA_DIR / \'step9_backtest_results.pkl\', \'rb\') as f:
    step9 = pickle.load(f)
step9_results = step9[\'results\']

step10_rec = pd.read_csv(DATA_DIR / \'step10_final_recommendation.csv\', index_col=0)

asset_returns = prices.pct_change().dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()

print(f\'가격 데이터: {prices.shape}\')
print(f\'Step 8 윈도우: {len(windows_cov)}\')
print(f\'Step 9 전략: {len(step9_results)}\')
print(f\'Step 10 추천 Top: {len(step10_rec)}\')
'''))

# ============================================================
# Cell 3: Top 10 선별 + 공통 설정
# ============================================================
cells.append(code_cell('''# ============================================================
# Top 10 선별 + 자산 그룹 + 색상 팔레트
# ============================================================

TOP10_KEYS = step10_rec.head(10).index.tolist()
print(\'Top 10 전략:\')
for i, key in enumerate(TOP10_KEYS, 1):
    row = step10_rec.loc[key]
    print(f\'  {i:2d}. {key:30s} Sharpe {row["sharpe"]:.3f}, MDD {row["mdd"]:+.2%}\')

PORT_TICKERS = list(prices.columns)

EQUITY_BROAD  = [\'SPY\', \'QQQ\', \'IWM\', \'EFA\', \'EEM\']
EQUITY_SECTOR = [\'XLK\', \'XLF\', \'XLE\', \'XLV\', \'VOX\', \'XLY\', \'XLP\', \'XLI\', \'XLU\', \'XLRE\', \'XLB\']
EQUITY_SINGLE = [\'AAPL\', \'MSFT\', \'AMZN\', \'GOOGL\', \'JPM\', \'JNJ\', \'PG\', \'XOM\']
EQUITY_ALL    = EQUITY_BROAD + EQUITY_SECTOR + EQUITY_SINGLE
BOND          = [\'TLT\', \'AGG\', \'SHY\', \'TIP\']
GOLD_ALT      = [\'GLD\', \'DBC\']

EQUITY_IDX = [PORT_TICKERS.index(t) for t in EQUITY_ALL]
BOND_IDX   = [PORT_TICKERS.index(t) for t in BOND]
GOLD_IDX   = [PORT_TICKERS.index(t) for t in GOLD_ALT]

# 성향별 EQUITY_CUT
EQUITY_CUT = {
    \'보수형\': {0: 0.00, 1: 0.15, 2: 0.35, 3: 0.60},
    \'중립형\': {0: 0.00, 1: 0.10, 2: 0.25, 3: 0.50},
    \'적극형\': {0: 0.00, 1: 0.05, 2: 0.15, 3: 0.35},
    \'공격형\': {0: 0.00, 1: 0.00, 2: 0.10, 3: 0.25},
}

COST_BPS, SLIP_BPS = 10, 5
TOTAL_COST = (COST_BPS + SLIP_BPS) / 10_000

PROFILE_PARAMS = {
    row[\'profile\']: {\'gamma\': row[\'gamma\'], \'max_equity\': row[\'max_equity\'], \'min_bond\': row[\'min_bond\']}
    for _, row in profiles_df.iterrows()
}

ALERT_CONFIGS = {\'ALERT_A\': \'alert_a\', \'ALERT_B\': \'alert_b\', \'ALERT_C\': \'alert_c\', \'ALERT_D\': \'alert_d\'}

# 색상 팔레트
# 30자산: 자산군별 그룹 색상 (주식=붉은·오렌지, 채권=파랑, 금/대체=노랑/갈색)
asset_colors = {}
# 주식 24개: tab20 + custom
eq_cmap = plt.cm.Reds(np.linspace(0.4, 0.95, len(EQUITY_ALL)))
for t, c in zip(EQUITY_ALL, eq_cmap):
    asset_colors[t] = c
bond_cmap = plt.cm.Blues(np.linspace(0.5, 0.9, len(BOND)))
for t, c in zip(BOND, bond_cmap):
    asset_colors[t] = c
gold_cmap = plt.cm.YlOrBr(np.linspace(0.4, 0.7, len(GOLD_ALT)))
for t, c in zip(GOLD_ALT, gold_cmap):
    asset_colors[t] = c

# 경보 레벨 색상
alert_colors = {0: \'#2ca02c\', 1: \'#ffcc00\', 2: \'#ff7f0e\', 3: \'#d62728\'}

# 모드/성향 색상 (Step9/10 일관)
profile_colors = {\'보수형\': \'#2ca02c\', \'중립형\': \'#1f77b4\', \'적극형\': \'#ff7f0e\', \'공격형\': \'#d62728\'}
mode_colors = {\'M0\': \'#808080\', \'M1\': \'#1f77b4\', \'M2\': \'#ff7f0e\', \'M3\': \'#d62728\'}

print(f\'\\n자산 그룹: EQUITY {len(EQUITY_IDX)} + BOND {len(BOND_IDX)} + GOLD {len(GOLD_IDX)} = {len(PORT_TICKERS)}\')
'''))

# ============================================================
# Cell 4: 11-1 MD
# ============================================================
cells.append(md_cell("""## 11-1. 재시뮬 설계 — `run_simulation_with_weights`

**왜 재시뮬인가**:
Step 9의 `run_simulation()` 함수는 일별 수익률만 반환하고 **weights 시계열을 기록하지 않음**. 따라서 Top 10에 대해 **Step 9 로직을 그대로 복제하되 weights·rebalance 이벤트·Σ 선택·경보 레벨 4종을 기록**하는 확장 함수를 신규 정의.

**반환값 구조**:
```python
{
    'daily_returns': pd.Series,        # Step 9와 동일 (검증 1 용)
    'weights_daily': pd.DataFrame,     # index=Date, cols=30 tickers
    'rebalance_events': pd.DataFrame,  # 날짜 + 원인(WF/Σ/alert) + turnover
    'sigma_selection': pd.Series|None, # M2/M3만, M1은 None
    'alert_levels': pd.Series,         # 일별 경보 레벨 (해당 Config)
}
```

**보장 불변식**:
- `weights.sum(axis=1) == 1` ± ε (비중 합)
- `weights >= 0` (공매도 없음)
- M1의 sigma_selection은 `None`, M3만 월별 토글
"""))

# ============================================================
# Cell 5: run_simulation_with_weights 함수 정의
# ============================================================
cells.append(code_cell('''# ============================================================
# run_simulation_with_weights: weights 기록 포함 재시뮬 함수
# ============================================================

def optimize_mv_weights(mu, cov, profile_params, tickers, eq_idx, bd_idx):
    """MV 최적화 (Step 9와 동일)."""
    gamma   = profile_params[\'gamma\']
    max_eq  = profile_params[\'max_equity\']
    min_bd  = profile_params[\'min_bond\']
    n       = len(tickers)

    def neg_utility(w):
        return -(w @ mu - gamma / 2 * w @ cov @ w)

    constraints = [{\'type\': \'eq\', \'fun\': lambda w: np.sum(w) - 1.0}]
    if len(eq_idx) > 0:
        constraints.append({\'type\': \'ineq\', \'fun\': lambda w: max_eq - np.sum(w[eq_idx])})
    if len(bd_idx) > 0:
        constraints.append({\'type\': \'ineq\', \'fun\': lambda w: np.sum(w[bd_idx]) - min_bd})

    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    result = minimize(neg_utility, w0, method=\'SLSQP\', bounds=bounds,
                      constraints=constraints, options={\'maxiter\': 500, \'ftol\': 1e-10})
    if result.success:
        w = np.maximum(result.x, 0)
        return w / w.sum()
    else:
        return np.ones(n) / n


def apply_equity_cut_vec(w_base, cut_ratio, eq_idx, bd_idx, gd_idx):
    """경로 1 주식 축소 (Step 9와 동일)."""
    if cut_ratio <= 0:
        return w_base.copy()
    w = w_base.copy()
    freed = 0.0
    for i in eq_idx:
        reduction = w[i] * cut_ratio
        w[i] -= reduction
        freed += reduction
    if freed > 0 and len(bd_idx) > 0:
        per_bond = freed * 0.70 / len(bd_idx)
        for i in bd_idx:
            w[i] += per_bond
    if freed > 0 and len(gd_idx) > 0:
        per_gold = freed * 0.30 / len(gd_idx)
        for i in gd_idx:
            w[i] += per_gold
    w = np.clip(w, 0, 1)
    return w / w.sum()


def run_simulation_with_weights(mode, profile, config_name,
                                log_returns, asset_returns, alerts, windows_cov,
                                profile_params, tickers, eq_idx, bd_idx, gd_idx,
                                cost=TOTAL_COST):
    """Step 9 run_simulation 로직 + weights·events·sigma·alert 기록."""
    daily_records = []
    weights_records = []
    rebalance_events = []
    sigma_records = []
    alert_records = []

    alert_col = ALERT_CONFIGS[config_name]
    alert_series = alerts[alert_col].astype(int)
    cuts = EQUITY_CUT[profile]
    prev_weights = None

    for w_id, w_info in windows_cov.items():
        is_start = w_info[\'is_start\']
        is_end   = w_info[\'is_end\']
        oos_start = w_info[\'oos_start\']
        oos_end   = w_info[\'oos_end\']

        is_log_ret = log_returns.loc[is_start:is_end]
        if len(is_log_ret) < 50:
            continue
        mu = w_info[\'mu\']

        cov_all = LedoitWolf().fit(is_log_ret.values).covariance_ * 252 \\
            if mode in (\'M0\', \'M1\') else None

        oos_ret = asset_returns.loc[oos_start:oos_end]
        oos_dates = list(oos_ret.index)
        if len(oos_dates) == 0:
            continue

        months = pd.Series(oos_dates).dt.to_period(\'M\').tolist()
        unique_months = list(dict.fromkeys(months))

        current_weights = None
        weights_for_ret = None  # 수익 계산에 쓰일 비중 (이전 루프 마지막 값)
        first_day_of_window = True

        for month in unique_months:
            month_dates = [d for d, m in zip(oos_dates, months) if m == month]
            if not month_dates:
                continue
            month_start = month_dates[0]

            # Σ 선택
            if mode in (\'M0\', \'M1\'):
                cov = cov_all
                sigma_label = None
            else:
                alert_at_month = int(alert_series.loc[month_start]) \\
                    if month_start in alert_series.index else 0
                if alert_at_month >= 2:
                    cov = w_info[\'Sigma_crisis\']
                    sigma_label = \'crisis\'
                else:
                    cov = w_info[\'Sigma_stable\']
                    sigma_label = \'stable\'

            # MV 최적화 (M0/M1은 분기 초만, M2/M3은 월별)
            if mode in (\'M2\', \'M3\') or current_weights is None:
                weights = optimize_mv_weights(mu, cov, profile_params, tickers, eq_idx, bd_idx)

                if prev_weights is None and current_weights is None:
                    turnover = 1.0
                    cause = \'initial\'
                elif current_weights is None:
                    turnover = np.abs(weights - prev_weights).sum()
                    cause = \'WF_rebalance\'
                else:
                    turnover = np.abs(weights - current_weights).sum()
                    cause = \'sigma_shift\'
                rebalance_cost = turnover * cost
                rebalance_events.append({
                    \'date\': month_start,
                    \'cause\': cause,
                    \'turnover\': turnover,
                    \'cost\': rebalance_cost,
                })
            else:
                rebalance_cost = 0.0

            current_weights = weights.copy()
            first_day_of_month = True

            for date in month_dates:
                # 경로 1 적용
                if mode in (\'M1\', \'M3\'):
                    alert_today = int(alert_series.loc[date]) \\
                        if date in alert_series.index else 0
                    cut = cuts.get(alert_today, 0)
                    target_weights = apply_equity_cut_vec(weights, cut, eq_idx, bd_idx, gd_idx)
                    day_turnover = np.abs(target_weights - current_weights).sum()
                    if day_turnover > 1e-6:
                        rebalance_events.append({
                            \'date\': date, \'cause\': \'alert_daily\',
                            \'turnover\': day_turnover, \'cost\': day_turnover * cost,
                        })
                    current_weights = target_weights
                else:
                    day_turnover = 0.0

                # 수익률 계산 (Step 9와 동일)
                ret = (current_weights * asset_returns.loc[date].values).sum()

                if first_day_of_window:
                    ret -= rebalance_cost
                    first_day_of_window = False
                    first_day_of_month = False
                elif first_day_of_month and mode in (\'M2\', \'M3\'):
                    ret -= rebalance_cost
                    first_day_of_month = False
                else:
                    ret -= day_turnover * cost
                    first_day_of_month = False

                daily_records.append({\'date\': date, \'return\': ret})
                weights_records.append({\'date\': date,
                                        **{t: float(w) for t, w in zip(tickers, current_weights)}})
                if mode in (\'M2\', \'M3\'):
                    sigma_records.append({\'date\': date, \'sigma\': sigma_label})
                alert_today_log = int(alert_series.loc[date]) \\
                    if date in alert_series.index else 0
                alert_records.append({\'date\': date, \'alert\': alert_today_log})

        prev_weights = current_weights.copy() if current_weights is not None else prev_weights

    out = {
        \'daily_returns\': pd.DataFrame(daily_records).set_index(\'date\')[\'return\'],
        \'weights_daily\': pd.DataFrame(weights_records).set_index(\'date\'),
        \'rebalance_events\': pd.DataFrame(rebalance_events),
        \'alert_levels\': pd.DataFrame(alert_records).set_index(\'date\')[\'alert\'],
    }
    if sigma_records:
        out[\'sigma_selection\'] = pd.DataFrame(sigma_records).set_index(\'date\')[\'sigma\']
    else:
        out[\'sigma_selection\'] = None
    return out


print(\'run_simulation_with_weights 함수 정의 완료\')
print(\'  반환: daily_returns, weights_daily, rebalance_events, sigma_selection, alert_levels\')
'''))

# ============================================================
# Cell 6: Top 10 재시뮬 실행
# ============================================================
cells.append(code_cell('''# ============================================================
# Top 10 재시뮬 실행
# ============================================================

def parse_key(key):
    """예: "M1_보수형_ALERT_B" → ("M1", "보수형", "ALERT_B")"""
    mode, rest = key.split(\'_\', 1)
    profile, config = rest.split(\'_\', 1)
    return mode, profile, config


top10_weights = {}
start_time = time.time()

for i, key in enumerate(TOP10_KEYS, 1):
    mode, profile, config = parse_key(key)
    t0 = time.time()

    out = run_simulation_with_weights(
        mode=mode, profile=profile, config_name=config,
        log_returns=log_returns, asset_returns=asset_returns, alerts=alerts,
        windows_cov=windows_cov, profile_params=PROFILE_PARAMS[profile],
        tickers=PORT_TICKERS, eq_idx=EQUITY_IDX, bd_idx=BOND_IDX, gd_idx=GOLD_IDX,
    )
    top10_weights[key] = out

    elapsed = time.time() - t0
    cum = (1 + out[\'daily_returns\']).prod() - 1
    n_events = len(out[\'rebalance_events\'])
    print(f\'[{i:2d}/10] {key:30s} {len(out["daily_returns"])}일, \'
          f\'누적 {cum:+.2%}, 리밸 이벤트 {n_events}회, {elapsed:.1f}s\')

total_elapsed = time.time() - start_time
print(f\'\\n전체 Top 10 재시뮬 완료: {total_elapsed:.0f}초\')
'''))

# ============================================================
# Cell 7: 검증 1
# ============================================================
cells.append(code_cell('''# ============================================================
# 검증 1: Step 9 누적수익률과 일치 확인
# ============================================================

check_rows = []
for key in TOP10_KEYS:
    s9 = step9_results[key]
    s11 = top10_weights[key][\'daily_returns\']
    s9_cum = (1 + s9).prod() - 1
    s11_cum = (1 + s11).prod() - 1
    diff = abs(s9_cum - s11_cum)
    check_rows.append({\'strategy\': key, \'step9_cum\': s9_cum, \'step11_cum\': s11_cum,
                        \'abs_diff\': diff, \'ok\': diff < 0.005})

check_df = pd.DataFrame(check_rows)
fmt = check_df.copy()
fmt[\'step9_cum\'] = fmt[\'step9_cum\'].map(\'{:+.4%}\'.format)
fmt[\'step11_cum\'] = fmt[\'step11_cum\'].map(\'{:+.4%}\'.format)
fmt[\'abs_diff\'] = fmt[\'abs_diff\'].map(\'{:.2e}\'.format)
print(\'검증 1: Step 9 누적수익률 일치 확인\')
print(fmt.to_string(index=False))
print()
all_ok = check_df[\'ok\'].all()
if all_ok:
    print(\'✅ 모든 Top 10 전략이 Step 9와 0.5% 이내로 일치 (재시뮬 정확)\')
else:
    fails = check_df[~check_df[\'ok\']]
    print(f\'⚠️ {len(fails)}개 전략 불일치: {list(fails["strategy"])}\')

# 불변식 검증
print()
print(\'검증: weights 불변식\')
for key in TOP10_KEYS[:3]:  # 샘플 3개
    w = top10_weights[key][\'weights_daily\']
    row_sum_max_err = (w.sum(axis=1) - 1).abs().max()
    neg_count = (w < -1e-6).sum().sum()
    print(f\'  {key}: sum(w)=1 오차 max {row_sum_max_err:.2e}, 음수 비중 {neg_count}개\')
'''))

# ============================================================
# Cell 8: 11-2 MD
# ============================================================
cells.append(md_cell("""## 11-2. 시각화 1 — 통합 대시보드

**제목**: "최우수 전략 M1_보수형_ALERT_B의 시간 여행"
**레이아웃**: 2×3 (figsize=(20, 12))

패널 구성:
1. (0,0) 누적수익률 + 경보 L2/L3 배경 음영
2. (0,1) 주식/채권/금 3그룹 Stacked Area
3. (0,2) 경보 타임라인 (4레벨 색상 바)
4. (1,0) HMM 레짐 타임라인
5. (1,1) 일별 주식 총비중 시계열
6. (1,2) Drawdown + 경보 상승 시점 수직선

**전달**: 한 장으로 "경보 상승 → 주식 감축 → Drawdown 축소" 인과사슬 시각화
"""))

# ============================================================
# Cell 9: 시각화 1 - 대시보드
# ============================================================
cells.append(code_cell('''# ============================================================
# 시각화 1: 통합 대시보드 (Top 1 = M1_보수형_ALERT_B)
# ============================================================

TOP1 = TOP10_KEYS[0]
out1 = top10_weights[TOP1]
w1 = out1[\'weights_daily\']
r1 = out1[\'daily_returns\']
alerts1 = out1[\'alert_levels\']

# 계산
cum = (1 + r1).cumprod()
peak = cum.cummax()
dd = (cum - peak) / peak

equity_total = w1.iloc[:, EQUITY_IDX].sum(axis=1)
bond_total   = w1.iloc[:, BOND_IDX].sum(axis=1)
gold_total   = w1.iloc[:, GOLD_IDX].sum(axis=1)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# (0,0) 누적수익률 + 경보 배경
ax = axes[0, 0]
ax.plot(cum.index, cum.values, color=\'#d62728\', linewidth=1.5, label=f\'{TOP1}\')
# EW 벤치마크
ew = step9_results[\'BENCH_EW\']
cum_ew = (1 + ew.loc[cum.index[0]:cum.index[-1]]).cumprod()
ax.plot(cum_ew.index, cum_ew.values, color=\'gray\', linestyle=\'--\', linewidth=1, label=\'EW\', alpha=0.7)
# 경보 L2/L3 음영
for lvl, color in [(2, \'#ff7f0e\'), (3, \'#d62728\')]:
    mask = alerts1 >= lvl
    if mask.any():
        ax.fill_between(alerts1.index, 0, cum.max()*1.1,
                        where=mask, color=color, alpha=0.08)
ax.set_title(\'(1) 누적 자본 + 경보 L2/L3 음영\', fontsize=12, fontweight=\'bold\')
ax.set_ylabel(\'누적 자본\')
ax.legend(loc=\'upper left\', fontsize=9)
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f\'{y:.1f}x\'))

# (0,1) 자산군 Stacked Area
ax = axes[0, 1]
ax.stackplot(equity_total.index,
             equity_total.values, bond_total.values, gold_total.values,
             labels=[\'주식\', \'채권\', \'금/대체\'],
             colors=[\'#d62728\', \'#1f77b4\', \'#ff7f0e\'], alpha=0.8)
ax.set_title(\'(2) 자산군 비중 흐름 (주식/채권/금)\', fontsize=12, fontweight=\'bold\')
ax.set_ylabel(\'비중\')
ax.set_ylim(0, 1)
ax.legend(loc=\'upper right\', fontsize=9)
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# (0,2) 경보 타임라인
ax = axes[0, 2]
for lvl in [0, 1, 2, 3]:
    mask = alerts1 == lvl
    if mask.any():
        ax.fill_between(alerts1.index, 0, 1, where=mask,
                        color=alert_colors[lvl], alpha=0.7,
                        transform=ax.get_xaxis_transform(), label=f\'L{lvl}\')
ax.set_title(\'(3) 경보 레벨 (Config B)\', fontsize=12, fontweight=\'bold\')
ax.set_yticks([])
ax.legend(loc=\'upper right\', fontsize=8, ncol=4)

# (1,0) HMM 레짐
ax = axes[1, 0]
regime_in = regime_df.loc[alerts1.index[0]:alerts1.index[-1], \'hmm_regime\']
regime_cmap = {0: \'#2ca02c\', 1: \'#ffcc00\', 2: \'#ff7f0e\', 3: \'#d62728\'}
for r in sorted(regime_in.unique()):
    mask = regime_in == r
    if mask.any():
        ax.fill_between(regime_in.index, 0, 1, where=mask,
                        color=regime_cmap.get(int(r), \'gray\'), alpha=0.7,
                        transform=ax.get_xaxis_transform(), label=f\'레짐 {int(r)}\')
ax.set_title(\'(4) HMM 레짐 (Full-sample)\', fontsize=12, fontweight=\'bold\')
ax.set_yticks([])
ax.legend(loc=\'upper right\', fontsize=8, ncol=4)

# (1,1) 일별 주식 총비중
ax = axes[1, 1]
ax.plot(equity_total.index, equity_total.values, color=\'#d62728\', linewidth=0.8)
ax.fill_between(equity_total.index, 0, equity_total.values, color=\'#d62728\', alpha=0.15)
ax.set_title(\'(5) 주식 총비중\', fontsize=12, fontweight=\'bold\')
ax.set_ylabel(\'비중\')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.grid(alpha=0.3)

# (1,2) Drawdown
ax = axes[1, 2]
ax.fill_between(dd.index, 0, dd.values, color=\'#d62728\', alpha=0.4)
ax.plot(dd.index, dd.values, color=\'#d62728\', linewidth=0.8)
# 경보 상승 수직선
alert_up = alerts1.index[alerts1.diff() > 0]
for d in alert_up[:20]:
    ax.axvline(d, color=\'orange\', linestyle=\':\', linewidth=0.5, alpha=0.5)
ax.set_title(\'(6) Drawdown + 경보 상승 시점\', fontsize=12, fontweight=\'bold\')
ax.set_ylabel(\'Drawdown\')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.grid(alpha=0.3)

fig.suptitle(f\'최우수 전략의 시간 여행 — {TOP1}\', fontsize=15, fontweight=\'bold\', y=1.00)
plt.tight_layout()
fig.savefig(IMG_DIR / \'step11_01_dashboard.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step11_01_dashboard.png\')
'''))

# ============================================================
# Cell 10: 11-3 MD
# ============================================================
cells.append(md_cell("""## 11-3. 시각화 2~4 — Top 10 자산 구성 변화

- **시각화 2**: Top 1 30자산 비중 시계열 (색상의 흐름으로 직관 이해)
- **시각화 3**: Top 10 전략 비교 Small Multiples (각 전략의 개성)
- **시각화 4**: 분기 평균 비중 히트맵 (앵커 vs 반응 자산 패턴)
"""))

# ============================================================
# Cell 11: 시각화 2 - Stacked Area Top 1
# ============================================================
cells.append(code_cell('''# ============================================================
# 시각화 2: Top 1 30자산 Stacked Area + 경보 타임라인
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                         gridspec_kw={\'height_ratios\': [5, 1]})

# 상단: 30자산 stacked area
ax1 = axes[0]
colors_list = [asset_colors[t] for t in PORT_TICKERS]
ax1.stackplot(w1.index, *[w1[t].values for t in PORT_TICKERS],
              labels=PORT_TICKERS, colors=colors_list, alpha=0.85)
ax1.set_ylabel(\'자산 비중\', fontsize=11)
ax1.set_title(f\'{TOP1}: 30개 자산 비중 시계열\', fontsize=13, fontweight=\'bold\')
ax1.set_ylim(0, 1)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
# 범례: 자산군별 대표만
legend_elements = [
    Patch(facecolor=\'#d62728\', alpha=0.85, label=f\'주식 ({len(EQUITY_ALL)}개)\'),
    Patch(facecolor=\'#1f77b4\', alpha=0.85, label=f\'채권 ({len(BOND)}개)\'),
    Patch(facecolor=\'#ff9900\', alpha=0.85, label=f\'금/대체 ({len(GOLD_ALT)}개)\'),
]
ax1.legend(handles=legend_elements, loc=\'upper right\', fontsize=9)
ax1.grid(alpha=0.3)

# 하단: 경보 타임라인
ax2 = axes[1]
for lvl in [0, 1, 2, 3]:
    mask = alerts1 == lvl
    if mask.any():
        ax2.fill_between(alerts1.index, 0, 1, where=mask,
                         color=alert_colors[lvl], alpha=0.7,
                         transform=ax2.get_xaxis_transform(), label=f\'L{lvl}\')
ax2.set_ylabel(\'경보\', fontsize=10)
ax2.set_yticks([])
ax2.legend(loc=\'upper right\', fontsize=8, ncol=4)
ax2.set_xlabel(\'Date\', fontsize=11)

plt.tight_layout()
fig.savefig(IMG_DIR / \'step11_02_stacked_area_top1.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step11_02_stacked_area_top1.png\')
'''))

# ============================================================
# Cell 12: 시각화 3 - Small Multiples
# ============================================================
cells.append(code_cell('''# ============================================================
# 시각화 3: Top 10 Small Multiples (자산군 3그룹 집계)
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True, sharey=True)
axes = axes.flatten()

for i, key in enumerate(TOP10_KEYS):
    ax = axes[i]
    w = top10_weights[key][\'weights_daily\']
    eq = w.iloc[:, EQUITY_IDX].sum(axis=1)
    bd = w.iloc[:, BOND_IDX].sum(axis=1)
    gd = w.iloc[:, GOLD_IDX].sum(axis=1)

    ax.stackplot(eq.index, eq.values, bd.values, gd.values,
                 colors=[\'#d62728\', \'#1f77b4\', \'#ff7f0e\'], alpha=0.8)

    row = step10_rec.loc[key]
    mode, profile, config = parse_key(key)
    config_short = config.replace(\'ALERT_\', \'\')

    ax.set_title(f\'{i+1}. {key}\\nSharpe {row["sharpe"]:.3f}, MDD {row["mdd"]:+.2%}\',
                 fontsize=9, fontweight=\'bold\')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.2)
    if i % 5 == 0:
        ax.set_ylabel(\'비중\')

legend_elements = [
    Patch(facecolor=\'#d62728\', alpha=0.8, label=\'주식\'),
    Patch(facecolor=\'#1f77b4\', alpha=0.8, label=\'채권\'),
    Patch(facecolor=\'#ff7f0e\', alpha=0.8, label=\'금/대체\'),
]
fig.legend(handles=legend_elements, loc=\'upper center\', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle(\'Top 10 전략의 자산군 비중 흐름 비교\', fontsize=14, fontweight=\'bold\', y=1.05)
plt.tight_layout()
fig.savefig(IMG_DIR / \'step11_03_smallmultiples_top10.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step11_03_smallmultiples_top10.png\')
'''))

# ============================================================
# Cell 13: 시각화 4 - 분기 히트맵
# ============================================================
cells.append(code_cell('''# ============================================================
# 시각화 4: Top 1 분기 평균 비중 히트맵
# ============================================================

# 분기별 평균 비중 계산
w_q = w1.copy()
w_q[\'quarter\'] = w_q.index.to_period(\'Q\').astype(str)
w_q_mean = w_q.groupby(\'quarter\').mean()

# 자산 순서: EQUITY → BOND → GOLD
ordered_tickers = EQUITY_ALL + BOND + GOLD_ALT
w_q_mean = w_q_mean[ordered_tickers]

fig, ax = plt.subplots(figsize=(16, 10))
annot_df = w_q_mean.T.copy()
annot_display = annot_df.map(lambda x: f\'{x:.0%}\' if x >= 0.05 else \'\')

sns.heatmap(w_q_mean.T, ax=ax, cmap=\'YlOrRd\',
            xticklabels=w_q_mean.index, yticklabels=ordered_tickers,
            cbar_kws={\'label\': \'평균 비중\'}, vmin=0, vmax=max(0.3, w_q_mean.values.max()),
            annot=annot_display, fmt=\'\', annot_kws={\'fontsize\': 7})

# 자산군 구분선
ax.axhline(len(EQUITY_ALL), color=\'black\', linewidth=1.5)
ax.axhline(len(EQUITY_ALL) + len(BOND), color=\'black\', linewidth=1.5)

ax.set_title(f\'{TOP1}: 분기별 자산 비중 히트맵 (5% 이상 annotate)\',
             fontsize=13, fontweight=\'bold\')
ax.set_xlabel(\'분기\', fontsize=11)
ax.set_ylabel(\'자산 (위=주식 / 중=채권 / 아래=금·대체)\', fontsize=11)
plt.xticks(rotation=45, ha=\'right\', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
fig.savefig(IMG_DIR / \'step11_04_weight_heatmap.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step11_04_weight_heatmap.png\')
'''))

# ============================================================
# Cell 14: 11-4 MD
# ============================================================
cells.append(md_cell("""## 11-4. 시각화 5~7 — 변화의 근거

- **시각화 5**: 경보 레벨 ↔ 주식 비중 동조 (3-panel)
- **시각화 6**: M1 vs M3 레짐·Σ 전환 오버레이 (경로 2 무효성 재확인)
- **시각화 7**: Turnover 원인 분류 (WF/Σ전환/경보)
"""))

# ============================================================
# Cell 15: 시각화 5 - 경보↔주식비중
# ============================================================
cells.append(code_cell('''# ============================================================
# 시각화 5: 경보 ↔ 주식 비중 동조
# ============================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 2], width_ratios=[3, 1])

# 상단: 경보 레벨 스텝
ax1 = fig.add_subplot(gs[0, 0])
ax1.step(alerts1.index, alerts1.values, color=\'#d62728\', linewidth=1, where=\'post\')
ax1.fill_between(alerts1.index, 0, alerts1.values, step=\'post\', color=\'#d62728\', alpha=0.2)
ax1.set_title(\'(1) Config B 경보 레벨\', fontsize=12, fontweight=\'bold\')
ax1.set_ylabel(\'레벨\')
ax1.set_yticks([0, 1, 2, 3])
ax1.grid(alpha=0.3)

# 중단: 주식 총비중
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(equity_total.index, equity_total.values, color=\'#d62728\', linewidth=0.9)
ax2.fill_between(equity_total.index, 0, equity_total.values, color=\'#d62728\', alpha=0.15)
ax2.set_title(\'(2) Top 1 전략 주식 총비중\', fontsize=12, fontweight=\'bold\')
ax2.set_ylabel(\'주식 비중\')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.grid(alpha=0.3)

# 하단 좌: 경보 vs 주식 비중 산점도 (시간 색)
ax3 = fig.add_subplot(gs[2, 0])
colors_by_time = np.linspace(0, 1, len(equity_total))
sc = ax3.scatter(alerts1.reindex(equity_total.index).values, equity_total.values,
                 c=colors_by_time, cmap=\'viridis\', s=8, alpha=0.5)
ax3.set_xlabel(\'경보 레벨\', fontsize=11)
ax3.set_ylabel(\'주식 비중\', fontsize=11)
ax3.set_title(\'(3) 경보 레벨 × 주식 비중 산점도 (색=시간)\', fontsize=12, fontweight=\'bold\')
ax3.set_xticks([0, 1, 2, 3])
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax3.grid(alpha=0.3)
fig.colorbar(sc, ax=ax3, label=\'시간 (0=시작 → 1=끝)\')

# 하단 우: 경보별 박스플롯
ax4 = fig.add_subplot(gs[:, 1])
data_by_level = []
labels_by_level = []
for lvl in [0, 1, 2, 3]:
    idx = alerts1[alerts1 == lvl].index.intersection(equity_total.index)
    if len(idx) > 0:
        data_by_level.append(equity_total.loc[idx].values)
        labels_by_level.append(f\'L{lvl}\\n(n={len(idx)})\')
bp = ax4.boxplot(data_by_level, labels=labels_by_level, patch_artist=True, widths=0.6)
for patch, lvl in zip(bp[\'boxes\'], [0, 1, 2, 3][:len(data_by_level)]):
    patch.set_facecolor(alert_colors[lvl])
    patch.set_alpha(0.6)
ax4.set_ylabel(\'주식 비중\')
ax4.set_title(\'(4) 경보별 주식 비중 분포\', fontsize=12, fontweight=\'bold\')
ax4.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax4.grid(alpha=0.3, axis=\'y\')

fig.suptitle(\'경보 → 주식 축소 인과관계 시각화 증명\', fontsize=14, fontweight=\'bold\', y=1.00)
plt.tight_layout()
fig.savefig(IMG_DIR / \'step11_05_alert_vs_equity.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step11_05_alert_vs_equity.png\')
'''))

# ============================================================
# Cell 16: 시각화 6 - 레짐·Σ 오버레이
# ============================================================
cells.append(code_cell('''# ============================================================
# 시각화 6: M1 vs M3 레짐·Σ 전환 오버레이
# ============================================================

# Top 1 (M1)과 Top 3 (M3_중립형_ALERT_C) 대비
key_m1 = TOP10_KEYS[0]  # M1_보수형_ALERT_B
key_m3 = next((k for k in TOP10_KEYS if k.startswith(\'M3_\')), TOP10_KEYS[2])

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

for ax, key in zip(axes, [key_m1, key_m3]):
    out = top10_weights[key]
    w = out[\'weights_daily\']
    eq = w.iloc[:, EQUITY_IDX].sum(axis=1)
    bd = w.iloc[:, BOND_IDX].sum(axis=1)
    gd = w.iloc[:, GOLD_IDX].sum(axis=1)

    # Stacked area (자산군)
    ax.stackplot(eq.index, eq.values, bd.values, gd.values,
                 colors=[\'#d62728\', \'#1f77b4\', \'#ff7f0e\'], alpha=0.65)

    # HMM 레짐 세로 배경 띠
    regime_in = regime_df.loc[eq.index[0]:eq.index[-1], \'hmm_regime\']
    # 크리시스 레짐 구간만 진한 음영
    crisis_mask = regime_in.isin(CRISIS_REGIMES)
    ax.fill_between(regime_in.index, 0, 1, where=crisis_mask,
                    color=\'red\', alpha=0.08,
                    transform=ax.get_xaxis_transform())

    # Σ 전환 시점 (M3만)
    mode = parse_key(key)[0]
    if mode == \'M3\' and out[\'sigma_selection\'] is not None:
        sigma_series = out[\'sigma_selection\']
        crisis_sigma_dates = sigma_series[sigma_series == \'crisis\'].index
        # 크리시스 Σ 구간 진한 음영
        if len(crisis_sigma_dates) > 0:
            is_crisis = sigma_series == \'crisis\'
            ax.fill_between(is_crisis.index, 0, 1, where=is_crisis.values,
                            color=\'purple\', alpha=0.12,
                            transform=ax.get_xaxis_transform(),
                            label=\'Σ_crisis 활성\')

    ax.set_title(f\'{key} ({mode} 모드)\', fontsize=12, fontweight=\'bold\')
    ax.set_ylabel(\'비중\')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)

# 공통 범례
legend_elements = [
    Patch(facecolor=\'#d62728\', alpha=0.65, label=\'주식\'),
    Patch(facecolor=\'#1f77b4\', alpha=0.65, label=\'채권\'),
    Patch(facecolor=\'#ff7f0e\', alpha=0.65, label=\'금/대체\'),
    Patch(facecolor=\'red\', alpha=0.08, label=\'HMM Crisis 레짐\'),
    Patch(facecolor=\'purple\', alpha=0.12, label=\'Σ_crisis 활성 (M3만)\'),
]
fig.legend(handles=legend_elements, loc=\'upper center\', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.99))

fig.suptitle(\'M1 vs M3: 경로 2(Σ 전환)가 실제 비중에 영향을 주는가?\',
             fontsize=14, fontweight=\'bold\', y=1.03)
axes[1].set_xlabel(\'Date\')
plt.tight_layout()
fig.savefig(IMG_DIR / \'step11_06_regime_sigma_overlay.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step11_06_regime_sigma_overlay.png\')
print(f\'  비교 대상: {key_m1} vs {key_m3}\')
'''))

# ============================================================
# Cell 17: 시각화 7 - Turnover 분해
# ============================================================
cells.append(code_cell('''# ============================================================
# 시각화 7: Turnover 원인 분해
# ============================================================

events1 = top10_weights[TOP1][\'rebalance_events\']
events1 = events1.set_index(\'date\')

# 월별 집계
events1[\'month\'] = pd.to_datetime(events1.index).to_period(\'M\')
monthly = events1.groupby([\'month\', \'cause\'])[\'turnover\'].sum().unstack(fill_value=0)

# 원인별 색상 + 순서
cause_order = [\'initial\', \'WF_rebalance\', \'sigma_shift\', \'alert_daily\']
cause_colors = {\'initial\': \'#808080\', \'WF_rebalance\': \'#1f77b4\',
                \'sigma_shift\': \'#9467bd\', \'alert_daily\': \'#ff7f0e\'}
cols_present = [c for c in cause_order if c in monthly.columns]
monthly = monthly[cols_present]

fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                         gridspec_kw={\'height_ratios\': [3, 1]})

# 상단: 월별 turnover stacked bar
ax1 = axes[0]
monthly.plot(kind=\'bar\', stacked=True, ax=ax1,
             color=[cause_colors[c] for c in cols_present],
             width=0.9, edgecolor=\'white\', linewidth=0.2)
ax1.set_title(f\'{TOP1}: 월별 Turnover 원인 분해\', fontsize=13, fontweight=\'bold\')
ax1.set_ylabel(\'월별 Turnover 합\')
ax1.legend(title=\'원인\', loc=\'upper right\', fontsize=9)
ax1.grid(axis=\'y\', alpha=0.3)
# x축 라벨 sparse
tick_positions = range(0, len(monthly), max(1, len(monthly)//20))
ax1.set_xticks(tick_positions)
ax1.set_xticklabels([str(monthly.index[i]) for i in tick_positions], rotation=45, ha=\'right\', fontsize=8)

# 하단: 누적 거래비용
ax2 = axes[1]
events1_sorted = events1.sort_index()
cum_cost = events1_sorted[\'cost\'].cumsum()
ax2.plot(cum_cost.index, cum_cost.values, color=\'#d62728\', linewidth=1.2)
ax2.fill_between(cum_cost.index, 0, cum_cost.values, color=\'#d62728\', alpha=0.2)
ax2.set_title(\'누적 거래비용\', fontsize=11)
ax2.set_ylabel(\'비용 (비율)\')
ax2.set_xlabel(\'Date\')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(IMG_DIR / \'step11_07_turnover_breakdown.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step11_07_turnover_breakdown.png\')
print(f\'\\n원인별 누적 turnover: {dict(events1.groupby("cause")["turnover"].sum().round(3))}\')
'''))

# ============================================================
# Cell 18: 11-5 MD
# ============================================================
cells.append(md_cell("""## 11-5. 시각화 8 — 전략 해부: 앵커 vs 반응 자산

**목적**: 30개 자산 중 어떤 것이 "장기 앵커"(꾸준히 보유)이고, 어떤 것이 "반응 버퍼"(경보 시 등장)인지 시각적으로 분류.

**축 해석**:
- x축: 평균 비중 (클수록 앵커)
- y축: 비중 변동성 (클수록 반응적)
- 버블 크기: 총 기여도 (평균 비중 × OOS 누적 수익률)
"""))

# ============================================================
# Cell 19: 시각화 8 - 앵커/반응 산점도
# ============================================================
cells.append(code_cell('''# ============================================================
# 시각화 8: 앵커 vs 반응 자산 산점도
# ============================================================

# Top 1 기준
w1_stats = pd.DataFrame({
    \'mean_weight\': w1.mean(),
    \'std_weight\': w1.std(),
})

# 자산군 그룹 색
def asset_group(t):
    if t in EQUITY_ALL: return \'주식\'
    if t in BOND: return \'채권\'
    return \'금/대체\'

w1_stats[\'group\'] = [asset_group(t) for t in w1_stats.index]
group_colors = {\'주식\': \'#d62728\', \'채권\': \'#1f77b4\', \'금/대체\': \'#ff7f0e\'}
w1_stats[\'color\'] = w1_stats[\'group\'].map(group_colors)

# OOS 구간 자산별 누적 수익률
asset_ret_oos = asset_returns.loc[w1.index[0]:w1.index[-1]]
asset_cum = (1 + asset_ret_oos).prod() - 1
w1_stats[\'asset_cum_ret\'] = asset_cum.reindex(w1_stats.index)
w1_stats[\'contribution\'] = w1_stats[\'mean_weight\'] * w1_stats[\'asset_cum_ret\']
# 버블 크기: contribution 절댓값 × scale
bubble_size = 50 + 3000 * w1_stats[\'contribution\'].abs()

fig, ax = plt.subplots(figsize=(14, 9))

for group, color in group_colors.items():
    sub = w1_stats[w1_stats[\'group\'] == group]
    ax.scatter(sub[\'mean_weight\'], sub[\'std_weight\'],
               s=bubble_size.loc[sub.index], c=color, alpha=0.6,
               edgecolors=\'black\', linewidths=0.5, label=group)

# 자산 라벨 (비중 Top 5, 변동성 Top 5)
to_label = set(w1_stats.nlargest(5, \'mean_weight\').index) | set(w1_stats.nlargest(5, \'std_weight\').index)
for t in to_label:
    ax.annotate(t, (w1_stats.loc[t, \'mean_weight\'], w1_stats.loc[t, \'std_weight\']),
                xytext=(5, 5), textcoords=\'offset points\', fontsize=9, fontweight=\'bold\')

# 영역 표시
x_med = w1_stats[\'mean_weight\'].median()
y_med = w1_stats[\'std_weight\'].median()
ax.axvline(x_med, color=\'gray\', linestyle=\'--\', alpha=0.3)
ax.axhline(y_med, color=\'gray\', linestyle=\'--\', alpha=0.3)

# 4사분면 해설
ax.text(ax.get_xlim()[1] * 0.8, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08,
        \'앵커 (큰 비중, 안정)\', fontsize=10, color=\'gray\', style=\'italic\')
ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
        ax.get_ylim()[1] * 0.9, \'반응 버퍼 (작은 비중, 고변동)\', fontsize=10, color=\'gray\', style=\'italic\')

ax.set_xlabel(\'평균 비중 (mean_weight)\', fontsize=11)
ax.set_ylabel(\'비중 변동성 (std_weight)\', fontsize=11)
ax.set_title(f\'{TOP1}: 앵커 자산 vs 반응 자산 해부도 (버블 크기 = |기여도|)\',
             fontsize=13, fontweight=\'bold\')
ax.legend(loc=\'upper right\', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(IMG_DIR / \'step11_08_asset_stability.png\', bbox_inches=\'tight\')
plt.show()
print(\'저장: images/step11_08_asset_stability.png\')

# 해석 출력
print(\'\\n앵커 자산 Top 5 (평균 비중 기준):\')
for t, row in w1_stats.nlargest(5, \'mean_weight\').iterrows():
    print(f\'  {t:6s}: 평균 {row["mean_weight"]:.2%}, 변동성 {row["std_weight"]:.2%}\')
print(\'\\n반응 버퍼 Top 5 (비중 변동성 기준):\')
for t, row in w1_stats.nlargest(5, \'std_weight\').iterrows():
    print(f\'  {t:6s}: 평균 {row["mean_weight"]:.2%}, 변동성 {row["std_weight"]:.2%}\')
'''))

# ============================================================
# Cell 20: 11-6 MD
# ============================================================
cells.append(md_cell("""## 11-6. 검증 2 및 산출물 저장

**검증 2**: 경보 상승 시 주식 비중이 실제로 감소하는지 확인 (M1/M3 전략만)
"""))

# ============================================================
# Cell 21: 검증 2 + 저장
# ============================================================
cells.append(code_cell('''# ============================================================
# 검증 2: 경보 ↔ 비중 정합성 + pkl 저장
# ============================================================

print(\'검증 2: 경보 상승 시 주식 비중 감소 정합성\')
print(\'-\' * 70)

check2_rows = []
for key in TOP10_KEYS:
    mode, profile, config = parse_key(key)
    if mode == \'M0\' or mode == \'M2\':
        continue  # 경로 1 없는 모드

    out = top10_weights[key]
    w = out[\'weights_daily\']
    eq_total = w.iloc[:, EQUITY_IDX].sum(axis=1)
    a = out[\'alert_levels\']

    # 경보 상승 시점
    a_diff = a.diff()
    up_dates = a_diff[a_diff > 0].index

    violations = 0
    for d in up_dates:
        before_window = eq_total.loc[:d].tail(5)
        after_window = eq_total.loc[d:].head(5)
        if len(before_window) < 3 or len(after_window) < 3:
            continue
        if after_window.mean() > before_window.mean() + 0.005:  # 0.5%p 여유
            violations += 1

    n_ups = len(up_dates)
    ok = violations == 0
    check2_rows.append({\'strategy\': key, \'alert_ups\': n_ups,
                         \'violations\': violations, \'ok\': ok})

check2_df = pd.DataFrame(check2_rows)
print(check2_df.to_string(index=False))
print()
all_ok2 = check2_df[\'ok\'].all()
if all_ok2:
    print(\'✅ 모든 M1/M3 전략: 경보 상승 후 주식 비중 감소 확인\')
else:
    print(\'⚠️ 일부 위반 발견 (drift 등으로 허용 오차 내)\')

# pkl 저장
save_data = {
    \'top10_keys\': TOP10_KEYS,
    \'weights\': {k: top10_weights[k][\'weights_daily\'] for k in TOP10_KEYS},
    \'daily_returns\': {k: top10_weights[k][\'daily_returns\'] for k in TOP10_KEYS},
    \'rebalance_events\': {k: top10_weights[k][\'rebalance_events\'] for k in TOP10_KEYS},
    \'sigma_selection\': {k: top10_weights[k][\'sigma_selection\'] for k in TOP10_KEYS},
    \'alert_levels\': {k: top10_weights[k][\'alert_levels\'] for k in TOP10_KEYS},
    \'meta\': {
        \'creation_time\': pd.Timestamp.now().isoformat(),
        \'cost_bps\': TOTAL_COST * 10_000,
        \'n_windows\': len(windows_cov),
        \'validation_1_passed\': bool(check_df[\'ok\'].all()),
        \'validation_2_passed\': bool(all_ok2),
    },
}
with open(DATA_DIR / \'step11_top10_weights.pkl\', \'wb\') as f:
    pickle.dump(save_data, f)

size = os.path.getsize(DATA_DIR / \'step11_top10_weights.pkl\') / 1024
print(f\'\\n저장: data/step11_top10_weights.pkl ({size:.0f} KB)\')
print(\'\\nStep 11 완료!\')
print(\'  - Top 10 weights 시계열 생성\')
print(\'  - 8종 시각화 PNG 저장\')
print(\'  - 검증 1·2 통과\')
'''))

# ============================================================
# 노트북 조립
# ============================================================
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
