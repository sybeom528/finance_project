"""
03_v2.py — Phase 3 Step 3 BL 백테스트 (03_v2.ipynb 에러 수정 실행본)

수정 사항 (2026-05-01):
  Fix 1: estimate_covariance(sub) → daily_to_monthly(compute_sigma_daily(sub))
          (estimate_covariance 는 is_start, is_end 필수 인자 — 이미 슬라이싱된 배열에 직접 사용 불가)
  Fix 2: FORCE_RECOMPUTE = True (기존 빈 캐시 0-weights 덮어쓰기)
  Fix 3: base_scenario = 'BL_ml_sw_mcap' (9 시나리오 키 일관)
  Fix 4: COLORS dict 9 시나리오 전체 포함
  Fix 5: §7 키 참조 수정 (BL_trailing → BL_trailing_mcap, BL_ml_sw → BL_ml_sw_mcap)
"""
import sys
import time
import warnings
import json
import pickle

import matplotlib
matplotlib.use('Agg')          # GUI 없이 파일 저장 (백그라운드 실행 필수)
warnings.filterwarnings('ignore')

from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    try:
        import koreanize_matplotlib  # noqa
    except ImportError:
        pass
plt.rcParams['axes.unicode_minus'] = False

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR
from scripts.black_litterman import (
    compute_pi, build_P, compute_omega, black_litterman, optimize_portfolio,
    Q_FIXED, PCT_GROUP, DEFAULT_TAU, LAM_FIXED,
)
# Fix 1: compute_sigma_daily + daily_to_monthly 추가 (estimate_covariance 대체)
from scripts.covariance import (
    compute_sigma_daily, daily_to_monthly, diagnose_sigma, DAYS_PER_MONTH,
)
from scripts.backtest import backtest_strategy
from scripts.benchmarks import equal_weight_portfolio, mcap_weight_portfolio, spy_returns

bootstrap(verbose=False)

OUT_DIR = OUTPUTS_DIR / '03_bl_backtest'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'BL hyperparameters (서윤범 99 일관):')
print(f'  Q_FIXED={Q_FIXED}, PCT_GROUP={PCT_GROUP}')
print(f'  DEFAULT_TAU={DEFAULT_TAU}, LAM_FIXED={LAM_FIXED}')


# ============================================================
# §2. 데이터 로드
# ============================================================
print('\n=== §2. 데이터 로드 ===')

universe = pd.read_csv(
    DATA_DIR / 'universe_full_history.csv', parse_dates=['cutoff_date']
)
print(f'universe: {universe.shape}, {universe["ticker"].nunique()} unique 종목')

panel = pd.read_csv(
    DATA_DIR / 'daily_panel.csv', parse_dates=['date'],
    usecols=['date', 'ticker', 'log_ret', 'vol_21d', 'mcap_value', 'log_mcap',
             'spy_close', 'rf_daily', 'vix'],
)
panel['date'] = pd.to_datetime(panel['date'])
print(f'panel: {panel.shape}, {panel["ticker"].nunique()} 종목')

ens_sw_path = DATA_DIR / 'ensemble_predictions_stockwise.csv'
assert ens_sw_path.exists(), f'02a 결과 없음: {ens_sw_path}'
ens_sw = pd.read_csv(ens_sw_path, parse_dates=['date'])
print(f'ensemble_sw: {ens_sw.shape} ({ens_sw["ticker"].nunique()} 학습 종목)')

ens_cs_path = DATA_DIR / 'ensemble_predictions_crosssec.csv'
assert ens_cs_path.exists(), f'02b 결과 없음: {ens_cs_path}'
ens_cs = pd.read_csv(ens_cs_path, parse_dates=['date'])
print(f'ensemble_cs: {ens_cs.shape} ({ens_cs["ticker"].nunique()} 학습 종목)')

market = pd.read_csv(DATA_DIR / 'market_data.csv', index_col='date', parse_dates=True)
print(f'market: {market.shape}')

trained_tickers_sw = set(ens_sw['ticker'].unique())
trained_tickers_cs = set(ens_cs['ticker'].unique())
trained_tickers = trained_tickers_sw   # 02a 기준 (서윤범 일관)
print(f'\n학습된 종목 (BL universe 제한용):')
print(f'  02a stockwise: {len(trained_tickers_sw)}')
print(f'  02b crosssec:  {len(trained_tickers_cs)}')
print(f'  교집합:         {len(trained_tickers_sw & trained_tickers_cs)}')

from scripts.universe import get_or_build_membership
membership = get_or_build_membership(
    start=pd.Timestamp('2008-12-01'),
    end=pd.Timestamp('2026-01-01'),
    cache_path=DATA_DIR / 'sp500_membership.pkl',
)
print(f'\nmembership (Dynamic-Membership): {len(membership)} 월말 시점')


# ============================================================
# §3. 헬퍼 함수
# ============================================================
print('\n=== §3. 헬퍼 함수 설정 ===')

daily_lr = panel.pivot_table(index='date', columns='ticker', values='log_ret')
print(f'daily_lr: {daily_lr.shape}')

market_lastday_per_month = market.groupby(market.index.to_period('M')).tail(1)
rebalance_dates_all_raw = market_lastday_per_month.index

OOS_START = '2010-01-01'
OOS_END = '2024-12-31'
HOLDOUT_START = '2025-01-01'
HOLDOUT_END = '2025-12-31'
oos_dates = rebalance_dates_all_raw[
    (rebalance_dates_all_raw >= OOS_START) & (rebalance_dates_all_raw <= OOS_END)
]
holdout_dates = rebalance_dates_all_raw[
    (rebalance_dates_all_raw >= HOLDOUT_START) & (rebalance_dates_all_raw <= HOLDOUT_END)
]
rebalance_dates = pd.DatetimeIndex(list(oos_dates) + list(holdout_dates))
print(f'rebalance_dates:')
print(f'  OOS      ({OOS_START}~{OOS_END}): {len(oos_dates)} 개월')
print(f'  Hold-out ({HOLDOUT_START}~{HOLDOUT_END}): {len(holdout_dates)} 개월')
print(f'  total: {len(rebalance_dates)} 개월')

# Period → market 월말 매핑 (Issue #1 수정의 핵심)
month_to_market_eom = {
    pd.Timestamp(d).to_period('M'): pd.Timestamp(d) for d in rebalance_dates
}

OUT_DIR_V2_03 = OUTPUTS_DIR / '03_v2_bl_backtest'
OUT_DIR_V2_03.mkdir(parents=True, exist_ok=True)
CACHE_PATH_V2_03 = DATA_DIR / 'scenario_weights_03_v2.pkl'
print(f'\nv2 산출 폴더: {OUT_DIR_V2_03.name}')
print(f'v2 BL 캐시:   {CACHE_PATH_V2_03.name}')


def compute_monthly_returns(panel_df, tickers, start_date, end_date, month_to_eom=None):
    """종목별 월별 단순 수익률 (Issue #1B: market eom index 사용)."""
    sub = panel_df[
        panel_df['ticker'].isin(tickers) &
        (panel_df['date'] >= start_date) & (panel_df['date'] <= end_date)
    ].set_index('date')
    sub['month'] = sub.index.to_period('M')
    monthly_lr = sub.groupby(['ticker', 'month'])['log_ret'].sum().reset_index()
    if month_to_eom is not None:
        monthly_lr['date'] = monthly_lr['month'].map(month_to_eom)
        monthly_lr = monthly_lr.dropna(subset=['date'])
    else:
        monthly_lr['date'] = monthly_lr['month'].dt.to_timestamp(how='end').dt.normalize()
    monthly_lr['ret'] = np.exp(monthly_lr['log_ret']) - 1
    return monthly_lr.pivot_table(index='date', columns='ticker', values='ret')


def get_mcap(panel_df, date, tickers):
    """특정 시점의 종목별 시가총액 취득."""
    sub = panel_df[(panel_df['date'] <= date) & panel_df['ticker'].isin(tickers)]
    return sub.sort_values(['ticker', 'date']).groupby('ticker').last()['mcap_value'].dropna()


# SPY 월별 수익률 + rf (Issue #2 수정: spy_excess = spy - rf)
spy_prices = market['SPY']
spy_daily = spy_prices.pct_change().dropna()
rf_daily = panel.groupby('date')['rf_daily'].mean()
spy_lr = np.log(1 + spy_daily)
rf_lr = rf_daily.reindex(spy_lr.index, method='ffill').fillna(0)
spy_excess_monthly = float((spy_lr - rf_lr).mean() * DAYS_PER_MONTH)
spy_sigma2_monthly = float(spy_lr.var() * DAYS_PER_MONTH)
print(f'spy_excess_monthly: {spy_excess_monthly:.6f}')
print(f'spy_sigma2_monthly: {spy_sigma2_monthly:.6f}')
print(f'implied lambda: {spy_excess_monthly / spy_sigma2_monthly:.3f}')


def get_monthly_pred(ens_df, pred_col='y_pred_ensemble'):
    """ensemble 예측 DataFrame → 월별 피벗 (rebalance 시점 기준)."""
    ens_copy = ens_df.copy()
    ens_copy['month'] = ens_copy['date'].dt.to_period('M')
    monthly = ens_copy.groupby(['ticker', 'month'])[pred_col].last().reset_index()
    monthly['rebalance_date'] = monthly['month'].map(month_to_market_eom)
    monthly = monthly.dropna(subset=['rebalance_date'])
    return monthly.pivot_table(index='rebalance_date', columns='ticker', values=pred_col)


sw_pred_col = 'y_pred_ensemble'
cs_pred_col = 'y_pred_ensemble' if 'y_pred_ensemble' in ens_cs.columns else 'y_pred_lstm_cs'

monthly_pred_sw = get_monthly_pred(ens_sw, sw_pred_col)
monthly_pred_cs = get_monthly_pred(ens_cs, cs_pred_col)
print(f'monthly_pred_sw: {monthly_pred_sw.shape}')
print(f'monthly_pred_cs: {monthly_pred_cs.shape}')


# ============================================================
# §4. BL 9 시나리오 백테스트
# ============================================================
print('\n=== §4. BL 9 시나리오 백테스트 ===')

TRANSACTION_COST = 0.0   # 거래비용 0 default (캐시 분기 전에 정의)
CACHE_PATH = CACHE_PATH_V2_03
FORCE_RECOMPUTE = True   # Fix 2: 기존 빈 캐시(0-weights) 덮어쓰기

if CACHE_PATH.exists() and not FORCE_RECOMPUTE:
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    scenario_weights = cache['scenario_weights']
    diagnostics = cache['diagnostics']
    print(f'캐시 사용: {CACHE_PATH.name}')
    for s, w in scenario_weights.items():
        print(f'  {s}: {len(w)} 리밸런싱 시점')
else:
    if FORCE_RECOMPUTE:
        print('FORCE_RECOMPUTE=True → 캐시 무시 + 재계산')

    DAYS_IS = 1260             # IS lookback 5년 (서윤범 60 monthly ≈ 1260 daily)
    MIN_UNIVERSE = 30          # 매월 최소 universe 크기 (서윤범 일관)
    MIN_VALID_TIX = 20         # IS 데이터 가용 종목 최소
    STALE_RATIO_THRESHOLD = 0.30   # stale price 필터 threshold

    WEIGHTINGS = ['mcap', 'eq', 'rp']
    scenario_weights = {f'BL_ml_sw_{w}': {} for w in WEIGHTINGS}
    scenario_weights['BL_ml_cs'] = {}
    scenario_weights.update({f'BL_trailing_{w}': {} for w in WEIGHTINGS})
    scenario_weights['EqualWeight'] = {}
    scenario_weights['McapWeight'] = {}

    diagnostics = {
        'monthly_universe_size': {},
        'sigma_psd': {},
        'sigma_condnum_log10': {},
        'slsqp_success': {},
        'monthly_skip_reason': {},
    }

    print(f'리밸런싱 시점 수: {len(rebalance_dates)}')
    print(f'9 시나리오: BL_ml_sw x3 + BL_ml_cs + BL_trailing x3 + EW + Mcap')
    print(f'DAYS_IS={DAYS_IS}, MIN_UNIVERSE={MIN_UNIVERSE}, MIN_VALID_TIX={MIN_VALID_TIX}')
    print('백테스트 시작...')
    t0 = time.time()

    for i, reb_date in enumerate(rebalance_dates):
        # 매월 universe: sp500_member_at_t ∩ panel ∩ 학습 613 ∩ non-stale
        panel_at_date = panel[panel['date'] == reb_date].dropna(
            subset=['vol_21d', 'log_mcap', 'log_ret']
        )
        panel_tickers_at_date = set(panel_at_date['ticker'])

        # Dynamic-Membership: reb_date(거래일 월말) → calendar 월말로 멤버십 lookup
        month_end_key = pd.Timestamp(
            reb_date.to_period('M').to_timestamp(how='end').normalize()
        )
        members_at_date = membership.get(month_end_key, frozenset())

        # stale price 필터: IS 1260일 안에서 zero ratio > 30% 종목 제외
        is_window = daily_lr.loc[reb_date - pd.offsets.BDay(1260):reb_date]
        zero_ratio = (is_window == 0).mean()
        non_stale = set(zero_ratio[zero_ratio <= STALE_RATIO_THRESHOLD].index)

        available_tickers = members_at_date & panel_tickers_at_date & trained_tickers & non_stale
        if len(available_tickers) < MIN_UNIVERSE:
            diagnostics['monthly_skip_reason'][reb_date] = (
                f'member+panel+trained+non_stale<{MIN_UNIVERSE} ({len(available_tickers)})'
            )
            continue

        tickers = sorted(available_tickers)

        # IS 기간 데이터
        is_end = reb_date
        is_start = is_end - pd.offsets.BDay(DAYS_IS)
        is_data = daily_lr.loc[is_start:is_end, :]

        # IS 데이터 70% 이상 가용 종목 필터
        avail_tickers = [
            t for t in tickers if t in is_data.columns
            and is_data[t].notna().sum() >= int(DAYS_IS * 0.7)
        ]
        if len(avail_tickers) < MIN_VALID_TIX:
            diagnostics['monthly_skip_reason'][reb_date] = (
                f'valid_tix<{MIN_VALID_TIX} ({len(avail_tickers)})'
            )
            continue

        diagnostics['monthly_universe_size'][reb_date] = len(avail_tickers)

        # ─── Fix 1: estimate_covariance() → compute_sigma_daily() + daily_to_monthly() ───
        # estimate_covariance(returns_daily, is_start, is_end) 는 is_start/is_end 필수
        # → 이미 슬라이싱된 배열에 직접 사용 불가 → 분리 호출로 우회
        try:
            sub = is_data[avail_tickers].dropna(how='any')
            Sigma = daily_to_monthly(compute_sigma_daily(sub))
        except Exception as e:
            diagnostics['monthly_skip_reason'][reb_date] = f'Sigma fail: {str(e)[:50]}'
            continue

        # Sigma 진단 (PSD, condition number)
        try:
            sig_info = diagnose_sigma(Sigma)
            diagnostics['sigma_psd'][reb_date] = sig_info['is_psd']
            diagnostics['sigma_condnum_log10'][reb_date] = sig_info['condition_number_log10']
        except Exception:
            pass

        # mcap + BL 공통 입력
        mcap = get_mcap(panel, reb_date, avail_tickers)
        common_tickers = list(Sigma.index.intersection(mcap.index))
        if len(common_tickers) < MIN_VALID_TIX:
            continue

        Sigma_c = Sigma.loc[common_tickers, common_tickers]
        mcap_c = mcap[common_tickers]
        w_mkt = mcap_c / mcap_c.sum()

        # 서윤범 99 일관: lam_fixed=2.5
        pi, lam = compute_pi(
            Sigma_c, w_mkt, spy_excess_monthly, spy_sigma2_monthly,
            lam_fixed=LAM_FIXED,
        )

        # BL_ml_sw - 3 weighting (mcap / eq / rp)
        if reb_date in monthly_pred_sw.index:
            vol_sw = monthly_pred_sw.loc[reb_date].reindex(common_tickers)
            vol_sw_actual = np.exp(vol_sw).fillna(vol_sw.median())
            if vol_sw_actual.notna().sum() >= 5:
                for w_method in WEIGHTINGS:
                    valid_sw = vol_sw_actual.dropna()
                    P_sw = build_P(valid_sw, mcap_c[valid_sw.index],
                                   pct=PCT_GROUP, weighting=w_method)
                    P_sw = P_sw.reindex(common_tickers).fillna(0)
                    omega_sw = compute_omega(P_sw, Sigma_c, DEFAULT_TAU)
                    mu_bl_sw = black_litterman(pi, Sigma_c, P_sw, q=Q_FIXED,
                                               omega=omega_sw, tau=DEFAULT_TAU)
                    w_sw = optimize_portfolio(mu_bl_sw, Sigma_c, lam)
                    scenario_weights[f'BL_ml_sw_{w_method}'][reb_date] = w_sw

        # BL_ml_cs - mcap only (단일 가중치)
        if reb_date in monthly_pred_cs.index:
            vol_cs = monthly_pred_cs.loc[reb_date].reindex(common_tickers)
            vol_cs_actual = np.exp(vol_cs).fillna(vol_cs.median())
            if vol_cs_actual.notna().sum() >= 5:
                valid_cs = vol_cs_actual.dropna()
                P_cs = build_P(valid_cs, mcap_c[valid_cs.index],
                               pct=PCT_GROUP, weighting='mcap')
                P_cs = P_cs.reindex(common_tickers).fillna(0)
                omega_cs = compute_omega(P_cs, Sigma_c, DEFAULT_TAU)
                mu_bl_cs = black_litterman(pi, Sigma_c, P_cs, q=Q_FIXED,
                                           omega=omega_cs, tau=DEFAULT_TAU)
                w_cs = optimize_portfolio(mu_bl_cs, Sigma_c, lam)
                scenario_weights['BL_ml_cs'][reb_date] = w_cs

        # BL_trailing - 3 weighting (mcap / eq / rp)
        vol_trailing = panel[
            (panel['date'] == reb_date) & (panel['ticker'].isin(common_tickers))
        ].set_index('ticker')['vol_21d']
        if vol_trailing.notna().sum() >= 5:
            vol_t = vol_trailing.reindex(common_tickers).fillna(vol_trailing.median())
            for w_method in WEIGHTINGS:
                P_t = build_P(vol_t, mcap_c, pct=PCT_GROUP, weighting=w_method)
                omega_t = compute_omega(P_t, Sigma_c, DEFAULT_TAU)
                mu_bl_t = black_litterman(pi, Sigma_c, P_t, q=Q_FIXED,
                                          omega=omega_t, tau=DEFAULT_TAU)
                w_t = optimize_portfolio(mu_bl_t, Sigma_c, lam)
                scenario_weights[f'BL_trailing_{w_method}'][reb_date] = w_t

        # EqualWeight: 1/N 균등
        ew_w = equal_weight_portfolio(avail_tickers)
        scenario_weights['EqualWeight'][reb_date] = ew_w

        # McapWeight: 시총 가중
        mw_w = mcap_weight_portfolio(mcap, avail_tickers)
        if len(mw_w) > 0:
            scenario_weights['McapWeight'][reb_date] = mw_w

        if (i + 1) % 24 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(rebalance_dates)}] {reb_date.date()}: {elapsed:.0f}s')

    print(f'\n백테스트 완료: {time.time() - t0:.1f}초')
    for s, w in scenario_weights.items():
        print(f'  {s}: {len(w)} 리밸런싱 시점')

    # 캐시 저장
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'scenario_weights': scenario_weights, 'diagnostics': diagnostics}, f)
    print(f'캐시 저장: {CACHE_PATH.name}')


# ============================================================
# §4-B. 월별 수익률 + portfolio returns
# ============================================================
print('\n=== §4-B. 월별 수익률 + 포트폴리오 수익률 ===')

all_tickers_union = universe['ticker'].unique().tolist()

monthly_rets = compute_monthly_returns(
    panel, all_tickers_union,
    start_date='2009-01-01',
    end_date='2025-12-31',
    month_to_eom=month_to_market_eom,
)
print(f'monthly_rets: {monthly_rets.shape}')

forward_rets = monthly_rets.shift(-1)

portfolio_returns = {}
for s_name, weights_dict in scenario_weights.items():
    if len(weights_dict) == 0:
        print(f'{s_name}: 가중치 없음 — skip')
        continue
    weights_df = pd.DataFrame(weights_dict).T.fillna(0)
    port_ret = backtest_strategy(
        weights_history=weights_df,
        returns=forward_rets,
        transaction_cost=TRANSACTION_COST,
    )
    portfolio_returns[s_name] = port_ret.dropna()
    print(f'{s_name}: {len(port_ret.dropna())} 개월')

spy_ret = spy_returns(market, rebalance_dates, return_type='monthly')
portfolio_returns['SPY'] = spy_ret.dropna()
print(f'SPY: {len(spy_ret.dropna())} 개월')

print('\n=== Portfolio returns 요약 ===')
for s, r in portfolio_returns.items():
    if len(r) > 0:
        print(f'  {s}: {len(r)} 개월, 평균 {r.mean()*100:.2f}%/월')


# ============================================================
# §5. 진단 통계 + Fair 비교 메트릭
# ============================================================
print('\n=== §5. 진단 통계 ===')
print('=' * 60)

us_sizes = pd.Series(diagnostics['monthly_universe_size'])
print(f'\n[매월 universe 크기]')
print(f'  rebalance_dates 총: {len(rebalance_dates)} 개월')
print(f'  유효: {len(us_sizes)} 개월')
if len(us_sizes) > 0:
    print(
        f'  size 분포: min={us_sizes.min()}, '
        f'p25={us_sizes.quantile(0.25):.0f}, '
        f'median={us_sizes.median():.0f}, '
        f'p75={us_sizes.quantile(0.75):.0f}, '
        f'max={us_sizes.max()}'
    )

psd_series = pd.Series(diagnostics['sigma_psd'])
print(f'\n[Sigma PSD]')
if len(psd_series) > 0:
    print(f'  PSD 만족: {int(psd_series.sum())}/{len(psd_series)} ({psd_series.mean()*100:.1f}%)')
else:
    print(f'  PSD 데이터 없음')

condnum = pd.Series(diagnostics['sigma_condnum_log10']).dropna()
if len(condnum) > 0:
    print(f'\n[Sigma condition number (log10)]')
    print(f'  분포: min={condnum.min():.1f}, median={condnum.median():.1f}, max={condnum.max():.1f}')
    print(
        f'  수치 안정 (log10 < 12): '
        f'{(condnum < 12).sum()}/{len(condnum)} ({(condnum < 12).mean()*100:.1f}%)'
    )

if diagnostics['monthly_skip_reason']:
    skip_df = pd.Series(diagnostics['monthly_skip_reason']).value_counts()
    print(f'\n[Skip 사유 ({len(diagnostics["monthly_skip_reason"])} 개월)]')
    print(skip_df.head(5).to_string())

print(f'\n[시나리오별 가중치 등록]')
for s, w in scenario_weights.items():
    print(f'  {s}: {len(w)}/{len(rebalance_dates)} 개월')

# Fix 3: base_scenario 키 수정 (9 시나리오 키 일관)
if 'BL_ml_sw_mcap' in portfolio_returns:
    base_scenario = 'BL_ml_sw_mcap'
elif 'BL_trailing_mcap' in portfolio_returns:
    base_scenario = 'BL_trailing_mcap'
else:
    non_spy = [k for k in portfolio_returns if k != 'SPY']
    base_scenario = non_spy[0] if non_spy else 'SPY'

common_dates = portfolio_returns[base_scenario].dropna().index
print(f'\nFair 비교 기준: {base_scenario}')
print(
    f'공통 기간: {common_dates[0].date()} ~ {common_dates[-1].date()} '
    f'({len(common_dates)} 개월)'
)

fair_returns = {s: r.reindex(common_dates).dropna() for s, r in portfolio_returns.items()}


def compute_metrics(rets, annual_factor=12):
    """Sharpe (rf=0 raw), annual return, MDD, CAGR."""
    if len(rets) == 0:
        return {k: np.nan for k in ['sharpe_raw', 'annual_ret', 'ann_vol', 'mdd', 'cagr', 'n_months']}
    ann_ret = rets.mean() * annual_factor
    ann_vol = rets.std() * np.sqrt(annual_factor)
    sharpe_raw = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + rets).cumprod()
    mdd = ((cum / cum.cummax()) - 1).min()
    n = len(rets)
    cagr = (cum.iloc[-1] ** (annual_factor / n) - 1) if n > 0 else np.nan
    return {
        'sharpe_raw': sharpe_raw,
        'annual_ret': ann_ret * 100,
        'ann_vol': ann_vol * 100,
        'mdd': mdd * 100,
        'cagr': cagr * 100,
        'n_months': n,
    }


metrics_fair = {s: compute_metrics(r) for s, r in fair_returns.items()}
metrics_df = pd.DataFrame(metrics_fair).T.sort_values('sharpe_raw', ascending=False)
print('\n=== 9 시나리오 Fair 비교 ===')
print(metrics_df.to_string())

# Fix 5: §7 키 참조 수정 (BL_trailing → BL_trailing_mcap)
print()
print('=' * 60)
print('  서윤범 99_baseline 재현 검증')
print('=' * 60)
print()
print('서윤범 99 reported: Sharpe=1.065/1.157, CAGR=15.00%, MDD=-11.80%')
print()

bl_trailing_key = 'BL_trailing_mcap' if 'BL_trailing_mcap' in metrics_df.index else None
if bl_trailing_key:
    tr = metrics_df.loc[bl_trailing_key]
    print(f'Phase 3 {bl_trailing_key} (TAU={DEFAULT_TAU}, LAM={LAM_FIXED}):')
    print(f'  Sharpe (raw): {tr["sharpe_raw"]:.3f}')
    print(f'  CAGR        : {tr["cagr"]:.2f}%')
    print(f'  MDD         : {tr["mdd"]:.2f}%')
    print(f'  Annual Vol  : {tr["ann_vol"]:.2f}%')
    print()
    sharpe_target = 1.157
    sharpe_diff_pct = (tr['sharpe_raw'] - sharpe_target) / sharpe_target * 100
    print(f'재현 검증:')
    print(f'  Sharpe 차이: {sharpe_diff_pct:+.2f}% (목표 +-5%)')
    if abs(sharpe_diff_pct) < 5:
        print('  [PASS] 서윤범 99 재현 성공')
    elif abs(sharpe_diff_pct) < 10:
        print('  [BORDERLINE] +-10% 이내 — universe 차이 영향 추정')
    else:
        print('  [FAIL] 재현 차이 큼 → universe 정의/hyperparameter 추적 필요')
else:
    print('BL_trailing_mcap 없음 — 재현 검증 생략')

print()
print('=' * 60)
print('  ML 통합 효과 정량화')
print('=' * 60)

bl_sw_key = 'BL_ml_sw_mcap'
bl_cs_key = 'BL_ml_cs'
bl_tr_key = 'BL_trailing_mcap'

if bl_sw_key in metrics_df.index and bl_tr_key in metrics_df.index:
    bl_sw_sharpe = metrics_df.loc[bl_sw_key, 'sharpe_raw']
    bl_tr_sharpe = metrics_df.loc[bl_tr_key, 'sharpe_raw']
    bl_cs_sharpe = (
        metrics_df.loc[bl_cs_key, 'sharpe_raw']
        if bl_cs_key in metrics_df.index else float('nan')
    )
    mcap_sharpe = (
        metrics_df.loc['McapWeight', 'sharpe_raw']
        if 'McapWeight' in metrics_df.index else float('nan')
    )
    spy_sharpe = (
        metrics_df.loc['SPY', 'sharpe_raw']
        if 'SPY' in metrics_df.index else float('nan')
    )

    print(f'\nSharpe 비교:')
    print(f'  {bl_sw_key:<20}: {bl_sw_sharpe:.3f}')
    print(f'  {bl_cs_key:<20}: {bl_cs_sharpe:.3f}')
    print(f'  {bl_tr_key:<20}: {bl_tr_sharpe:.3f}  (서윤범 99 재현)')
    print(f'  {"McapWeight":<20}: {mcap_sharpe:.3f}')
    print(f'  {"SPY":<20}: {spy_sharpe:.3f}')

    print(f'\nML 통합 효과:')
    print(f'  {bl_sw_key} - {bl_tr_key} = {bl_sw_sharpe - bl_tr_sharpe:+.3f}')
    if not np.isnan(bl_cs_sharpe):
        print(f'  {bl_cs_key} - {bl_tr_key} = {bl_cs_sharpe - bl_tr_sharpe:+.3f}')

    sw_better = bl_sw_sharpe > bl_tr_sharpe
    cs_better = (not np.isnan(bl_cs_sharpe)) and (bl_cs_sharpe > bl_tr_sharpe)
    if sw_better and cs_better:
        print('  [확인] ML 통합 효과: Stockwise + Cross-sec 모두 BL_trailing 능가')
    elif sw_better:
        print('  [부분] Stockwise 만 우위 (Cross-sec 미달)')
    elif cs_better:
        print('  [부분] Cross-sec 만 우위 (Stockwise 미달)')
    else:
        print('  [미확인] ML 통합 효과 없음 또는 역효과')
else:
    print('BL 시나리오 결과 없음 — ML 효과 정량화 생략')


# ============================================================
# §6. 시각화 (누적수익 / Drawdown / Rolling Sharpe)
# ============================================================
print('\n=== §6. 시각화 ===')

# Fix 4: COLORS dict 9 시나리오 전체 포함
COLORS = {
    'BL_ml_sw_mcap':    '#1f77b4',
    'BL_ml_sw_eq':      '#4a90d9',
    'BL_ml_sw_rp':      '#87ceeb',
    'BL_ml_cs':         '#2ca02c',
    'BL_trailing_mcap': '#d62728',
    'BL_trailing_eq':   '#e87070',
    'BL_trailing_rp':   '#ffa0a0',
    'EqualWeight':      '#ff7f0e',
    'McapWeight':       '#9467bd',
    'SPY':              '#8c564b',
}

# 주요 시나리오만 표시 (가독성: mcap 기준 각 1개 + EW + Mcap + SPY)
display_scenarios = [
    'BL_ml_sw_mcap', 'BL_ml_cs', 'BL_trailing_mcap',
    'EqualWeight', 'McapWeight', 'SPY',
]
display_rets = {
    s: r for s, r in fair_returns.items()
    if s in display_scenarios and len(r) > 0
}

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. 누적 수익률
ax = axes[0]
for s, r in display_rets.items():
    cum = (1 + r).cumprod()
    ax.plot(cum.index, cum.values, label=s, color=COLORS.get(s), linewidth=1.8)
ax.set_title('누적 수익률 (Fair 비교, 동일 기간)', fontsize=12)
ax.set_ylabel('누적 수익 (1=기준)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

# 2. Drawdown
ax = axes[1]
for s, r in display_rets.items():
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax()) - 1
    ax.plot(dd.index, dd.values * 100, label=s, color=COLORS.get(s), linewidth=1.5)
ax.set_title('Drawdown (%)', fontsize=12)
ax.set_ylabel('Drawdown (%)')
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=0.3)

# 3. Rolling Sharpe (12개월)
ax = axes[2]
for s, r in display_rets.items():
    if len(r) >= 12:
        roll_sharpe = r.rolling(12).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan
        )
        ax.plot(roll_sharpe.index, roll_sharpe.values, label=s,
                color=COLORS.get(s), linewidth=1.5)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Rolling Sharpe (12개월)', fontsize=12)
ax.set_ylabel('Rolling Sharpe')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
save_path = OUT_DIR / 'backtest_comparison.png'
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f'backtest_comparison.png 저장: {save_path}')
plt.close()

# 시기별 Sharpe 분해
periods = {
    'GFC 회복 (2009~2011)': ('2009-01-01', '2011-12-31'),
    '정상 강세장 (2012~2019)': ('2012-01-01', '2019-12-31'),
    'COVID+AI (2020~2025)': ('2020-01-01', '2025-12-31'),
}

print('\n=== 시기별 Sharpe 분해 ===')
for period_name, (start, end) in periods.items():
    pm = {}
    for s, r in fair_returns.items():
        sub = r.loc[start:end].dropna()
        if len(sub) >= 6:
            ann_ret = sub.mean() * 12
            ann_vol = sub.std() * np.sqrt(12)
            pm[s] = ann_ret / ann_vol if ann_vol > 0 else np.nan
    print(f'\n{period_name}:')
    for s, v in sorted(pm.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -99):
        print(f'  {s:<25}: {v:.3f}')


# ============================================================
# §7. 결과 저장
# ============================================================
print('\n=== §7. 결과 저장 ===')

metrics_df.to_csv(OUT_DIR / 'metrics_fair.csv')
print(f'metrics_fair.csv 저장')

metrics_json = {
    k: {kk: (float(vv) if not np.isnan(vv) else None) for kk, vv in v.items()}
    for k, v in metrics_fair.items()
}
with open(OUT_DIR / 'metrics_fair.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_json, f, ensure_ascii=False, indent=2)
print(f'metrics_fair.json 저장')

for s, r in fair_returns.items():
    fname = f'returns_{s}.csv'
    r.to_csv(OUT_DIR / fname)
print(f'returns_*.csv 저장 ({len(fair_returns)} 파일)')

print()
print('=' * 60)
print('  Phase 3 Step 3 완료')
print('=' * 60)
print(f'  출력 경로: {OUT_DIR}')
print('  다음 단계: 04_compare_stockwise_vs_cross.ipynb')
