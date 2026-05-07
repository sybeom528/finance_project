"""
_rebuild_panel_2025.py — monthly_panel.csv 를 2025-12-31 까지 확장.

기존 prices_raw.pkl (2004-01 ~ 2026-03) + sp500_membership.pkl + shares_outstanding.pkl
+ universe.csv 를 사용해 yfinance 재호출 없이 panel 만 재구축.

01_DataCollection.ipynb 의 panel-building 셀과 정확히 동일한 로직 (compute_features 포함).
변경점은 PANEL_END='2025-12-31' 만.

실행:
    cd final && python _rebuild_panel_2025.py
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'

PANEL_START = '2005-01-01'
PANEL_END   = '2025-12-31'   # ← 기존 2024-12-31 에서 확장

PANEL_PATH        = DATA_DIR / 'monthly_panel.csv'
BACKUP_PATH       = DATA_DIR / 'monthly_panel_2024_backup.csv'
PRICES_PATH       = DATA_DIR / 'prices_raw.pkl'
MEMBERSHIP_PATH   = DATA_DIR / 'sp500_membership.pkl'
SHARES_PATH       = DATA_DIR / 'shares_outstanding.pkl'
UNIVERSE_PATH     = DATA_DIR / 'universe.csv'

ANN = np.sqrt(252)

# ── 1. 입력 데이터 로드 ────────────────────────────────────────
print('[1/5] 입력 데이터 로드 ...')
prices_raw       = pd.read_pickle(PRICES_PATH)
sp500_membership = pickle.load(open(MEMBERSHIP_PATH, 'rb'))
shares_map       = pickle.load(open(SHARES_PATH, 'rb'))
df_universe      = pd.read_csv(UNIVERSE_PATH)
print(f'  prices_raw shape={prices_raw.shape}, '
      f'date={prices_raw.index.min().date()}~{prices_raw.index.max().date()}')
print(f'  membership months={len(sp500_membership)}')
print(f'  universe={len(df_universe)} tickers')

# Close 추출
if isinstance(prices_raw.columns, pd.MultiIndex):
    close = prices_raw['Close'].copy()
else:
    close = prices_raw.copy()
close = close[~close.index.duplicated(keep='last')]

# ── 2. SPY + RF 일별 시리즈 (compute_features 입력) ──────────
spy_close = close['SPY']
spy_lr    = np.log(spy_close / spy_close.shift(1))

rf_annual = close['^IRX'].ffill().bfill() / 100
rf_daily  = (1 + rf_annual) ** (1/252) - 1

# ── 3. compute_features 정의 (01 노트북과 동일) ──────────────
def compute_features(ticker: str, ac: pd.Series) -> pd.DataFrame:
    ac = ac[~ac.index.duplicated(keep='last')]
    df = pd.DataFrame(index=ac.index)

    lr  = np.log(ac / ac.shift(1))
    exc = lr - rf_daily.reindex(ac.index)
    mkt = spy_lr.reindex(ac.index)

    df['vol_21d']  = lr.rolling(21).std() * ANN
    df['vol_60d']  = lr.rolling(60).std() * ANN
    df['vol_252d'] = lr.rolling(252).std() * ANN

    cov_em          = exc.rolling(252).cov(mkt)
    var_m           = mkt.rolling(252).var()
    df['beta_252d'] = cov_em / var_m

    df['close'] = ac

    if ticker in shares_map:
        shares_ts = shares_map[ticker]
        shares_ts = shares_ts[~shares_ts.index.duplicated(keep='last')]
        if shares_ts.index.tz is not None:
            shares_ts.index = shares_ts.index.tz_localize(None)
        shares_ts = shares_ts.reindex(ac.index).ffill().bfill()
        df['log_mcap'] = np.log((ac * shares_ts).clip(lower=1))
    else:
        df['log_mcap'] = np.log(ac.clip(lower=1e-6))

    r1 = 1 + ac.pct_change()
    df['fwd_ret_1m'] = r1.shift(-1).rolling(21).apply(np.prod, raw=True).shift(-20) - 1

    df['ticker'] = ticker
    return df


# ── 4. 패널 빌드 ───────────────────────────────────────────────
print(f'[2/5] 종목별 피처 계산 (PANEL_END={PANEL_END}) ...')

sector_map       = dict(zip(df_universe['ticker'], df_universe['gics_sector']))
membership_dates = sorted(sp500_membership.keys())

def get_members_at(date: pd.Timestamp) -> frozenset:
    idx = pd.Series(membership_dates).searchsorted(date, side='right') - 1
    if idx < 0:
        return frozenset()
    return sp500_membership[membership_dates[idx]]

valid_tickers = [t for t in df_universe['ticker'].tolist() if t in close.columns]
print(f'  valid_tickers in close: {len(valid_tickers)}')

monthly_panels = []
for i, ticker in enumerate(valid_tickers):
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{len(valid_tickers)} 완료')
    ac = close[ticker].dropna()
    if len(ac) < 252:
        continue
    df_t = compute_features(ticker, ac)
    df_m = df_t.resample('ME').last()
    df_m['ticker']      = ticker
    df_m['gics_sector'] = sector_map.get(ticker, 'Unknown')
    df_m = df_m[(df_m.index >= PANEL_START) & (df_m.index <= PANEL_END)]
    monthly_panels.append(df_m)

monthly_df = pd.concat(monthly_panels)
monthly_df.index.name = 'date'
monthly_df = monthly_df.reset_index().set_index(['date', 'ticker'])
monthly_df = monthly_df.sort_index()
print(f'  before survivorship filter: {monthly_df.shape}')

# ── 5. 생존편향 필터 ───────────────────────────────────────────
print('[3/5] 생존편향 필터 (시점별 S&P500 멤버십) ...')
mask = [(ticker in get_members_at(date))
        for (date, ticker), _ in monthly_df.iterrows()]
monthly_df = monthly_df[mask]
print(f'  after filter: {monthly_df.shape[0]}rows '
      f'({monthly_df.index.get_level_values("ticker").nunique()} tickers)')

# ── 6. SPY 월별 수익률 ──────────────────────────────────────────
print('[4/5] SPY / RF / ret_1m 결합 ...')
spy_monthly = spy_close.resample('ME').last().pct_change().rename('spy_ret')
monthly_df  = monthly_df.join(spy_monthly, on='date')

# ── 7. RF 월별 환산 ─────────────────────────────────────────────
rf_monthly = rf_daily.resample('ME').apply(lambda x: (1+x).prod()-1).rename('rf_1m')
monthly_df = monthly_df.join(rf_monthly, on='date')

# ── 8. ret_1m: 전체 가격 히스토리로 계산 ────────────────────────
monthly_close_full = close[valid_tickers].resample('ME').last()
monthly_ret        = monthly_close_full.pct_change()
monthly_ret        = monthly_ret.stack(future_stack=True).rename('ret_1m')
monthly_ret.index.names = ['date', 'ticker']
monthly_df = monthly_df.join(monthly_ret)

# ── 9. 저장 ────────────────────────────────────────────────────
print('[5/5] 저장 ...')
monthly_df.to_csv(PANEL_PATH)

dmin = monthly_df.index.get_level_values('date').min()
dmax = monthly_df.index.get_level_values('date').max()
print('━' * 56)
print(f'monthly_panel.csv 저장 완료')
print(f'  shape   : {monthly_df.shape}')
print(f'  date    : {dmin.date()} ~ {dmax.date()}')
print(f'  tickers : {monthly_df.index.get_level_values("ticker").nunique()}')
print(f'  cols    : {list(monthly_df.columns)}')
print(f'  path    : {PANEL_PATH}')
