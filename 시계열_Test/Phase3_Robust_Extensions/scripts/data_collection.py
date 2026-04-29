"""Phase 2 — Data Collection (일별 패널 + 시장 데이터 + 보조 데이터).

서윤범 [01_DataCollection.ipynb](../../서윤범/low_risk/01_DataCollection.ipynb) 의
compute_features 패턴 + Phase 1.5 의 target_logrv 정의 통합.

핵심 함수
---------
- get_universe_tickers()         : universe_top50_history.csv 의 unique ticker list
- download_universe_ohlcv()      : 종목별 일별 OHLCV (auto_adjust=True)
- download_market_data()         : SPY, ^VIX, ^TNX 일별
- download_risk_free()           : FRED DGS3MO 또는 ^IRX → 일별 변환
- download_fama_french()         : K-French 3팩터 자동 다운로드 (서윤범 패턴)
- compute_panel_features()       : 종목별 lr, vol, beta, log_mcap, target_logrv
- build_daily_panel()            : 통합 일별 패널

설계 원칙
---------
- 변동성 단위: **일별 std (Phase 1.5 와 통일)** — 연환산 X
- 분할 보정 시총: scripts.universe.get_adjusted_shares_at() 사용
- Phase 1.5 타깃: target_logrv = log(rolling(21).std()).shift(-21)
- 캐싱: 모든 다운로드 pickle/csv 캐시
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .universe import (
    DUAL_LISTED_MAP,
    _strip_tz,
    get_adjusted_shares_at,
)


# =============================================================================
# Universe ticker 추출
# =============================================================================
def get_universe_tickers(universe_csv_path: Path) -> list[str]:
    """universe_top50_history.csv 의 unique ticker list."""
    df = pd.read_csv(universe_csv_path)
    return sorted(df['ticker'].unique().tolist())


# =============================================================================
# OHLCV 일별 다운로드
# =============================================================================
def download_universe_ohlcv(
    tickers: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    chunk_size: int = 30,
) -> dict[str, pd.DataFrame]:
    """종목별 일별 OHLCV 수집 (개별 CSV 캐시).

    Parameters
    ----------
    tickers : 대상 종목 list
    start, end : 다운로드 기간
    cache_dir : 종목별 CSV 저장 디렉토리 (data/prices_daily/)
    chunk_size : yfinance batch 크기

    Returns
    -------
    dict[str, pd.DataFrame]
        ticker → DataFrame (Open, High, Low, Close, Volume), tz-naive

    Notes
    -----
    auto_adjust=True 적용 (분할·배당 보정된 가격).
    이미 캐시된 종목은 skip → 부족분만 다운로드.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    tickers_list = list(tickers)
    cached_files = {p.stem: p for p in cache_dir.glob('*.csv')}
    new_tickers = [t for t in tickers_list if t not in cached_files]

    if new_tickers:
        print(f'  [data] OHLCV 다운로드: {len(new_tickers)} 신규 종목 ({len(cached_files)} 캐시)')
        for i in range(0, len(new_tickers), chunk_size):
            batch = new_tickers[i:i + chunk_size]
            try:
                data = yf.download(batch, start=start, end=end,
                                  auto_adjust=True, progress=False, group_by='ticker')
                # 단일 종목인 경우 처리
                if len(batch) == 1:
                    df = data.copy()
                    df.index = _strip_tz(df.index)
                    df.to_csv(cache_dir / f'{batch[0]}.csv')
                else:
                    for tk in batch:
                        if tk not in data.columns.get_level_values(0):
                            print(f'    {tk}: 다운로드 데이터 없음')
                            continue
                        df = data[tk].dropna(how='all').copy()
                        if len(df) > 0:
                            df.index = _strip_tz(df.index)
                            df.to_csv(cache_dir / f'{tk}.csv')
            except Exception as e:
                print(f'    chunk {i//chunk_size + 1} 실패: {e}')
            time.sleep(0.5)
        print(f'  [data] OHLCV 저장 완료')

    # 모든 종목 로드
    out = {}
    for tk in tickers_list:
        path = cache_dir / f'{tk}.csv'
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = _strip_tz(df.index)
            out[tk] = df
    return out


# =============================================================================
# 시장 데이터 (SPY, VIX, ^TNX)
# =============================================================================
MARKET_TICKERS = ['SPY', '^VIX', '^TNX']


def download_market_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path,
) -> pd.DataFrame:
    """시장 데이터 일별 (SPY 가격, VIX, ^TNX) 수집.

    Returns
    -------
    pd.DataFrame
        date 인덱스 + columns: SPY, VIX, TNX (Adj Close)
    """
    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col='date', parse_dates=True)
        df.index = _strip_tz(df.index)
        return df

    print(f'  [data] 시장 데이터 다운로드: {MARKET_TICKERS}')
    data = yf.download(MARKET_TICKERS, start=start, end=end,
                      auto_adjust=True, progress=False)['Close']
    data.index = _strip_tz(data.index)
    data.columns = [c.replace('^', '') for c in data.columns]
    data.index.name = 'date'

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path)
    print(f'  [data] 시장 데이터 저장: {cache_path.name}')
    return data


# =============================================================================
# Fama-French 3팩터 (월별)
# =============================================================================
FF3_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'


def download_fama_french(cache_path: Path) -> pd.DataFrame:
    """Fama-French 3팩터 (월별) — K-French 라이브러리 자동 다운로드.

    서윤범 99_baseline 패턴 재사용.

    Returns
    -------
    pd.DataFrame
        date (월말) 인덱스 + columns: Mkt-RF, SMB, HML, RF (모두 % 단위 → 소수)
    """
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col='date', parse_dates=True)

    print(f'  [data] Fama-French 3팩터 다운로드: K-French')
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(FF3_URL, headers=headers, timeout=60)
    resp.raise_for_status()

    # ZIP 안의 CSV 파싱
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        raw = zf.read(csv_name).decode('utf-8', errors='ignore')

    # 첫 번째 데이터 블록 (월별) 만 추출
    lines = raw.split('\n')
    # 첫 데이터 행 찾기
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        # 6자리 숫자 (YYYYMM) 으로 시작하는 첫 행
        if start_idx is None and len(s) >= 6 and s[:6].isdigit():
            start_idx = i
        # 종료: 빈 행 또는 다른 형식
        elif start_idx is not None:
            if not s or not s[:6].isdigit():
                end_idx = i
                break
    if start_idx is None:
        raise RuntimeError('FF3 CSV 파싱 실패 — 데이터 행 미발견')
    if end_idx is None:
        end_idx = len(lines)

    # 헤더 추출 (start_idx 직전 비-빈 행)
    header_idx = start_idx - 1
    while header_idx > 0 and not lines[header_idx].strip():
        header_idx -= 1
    header_line = lines[header_idx].strip()

    csv_text = header_line + '\n' + '\n'.join(lines[start_idx:end_idx])
    df = pd.read_csv(io.StringIO(csv_text))

    # 첫 컬럼명을 'date' 로 통일
    df = df.rename(columns={df.columns[0]: 'yyyymm'})
    df['yyyymm'] = df['yyyymm'].astype(str).str.strip()
    df['date'] = pd.to_datetime(df['yyyymm'], format='%Y%m') + pd.offsets.MonthEnd(0)
    df = df.drop(columns='yyyymm').set_index('date')

    # % → 소수
    df = df.astype(float) / 100.0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f'  [data] FF3 저장: {cache_path.name} (행={len(df)})')
    return df


# =============================================================================
# 무위험 수익률 (FRED DGS3MO 또는 ^IRX)
# =============================================================================
def download_risk_free(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path,
) -> pd.Series:
    """무위험 수익률 (일별).

    1차 시도: yfinance ^IRX (3개월 미국채, 연율 %)
    실패 시: FRED DGS3MO (pandas-datareader)

    Returns
    -------
    pd.Series
        date 인덱스, value = 일별 무위험 수익률 (소수, 일 단위)
    """
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col='date', parse_dates=True)['rf_daily']

    print(f'  [data] 무위험 수익률 다운로드 (^IRX)')
    try:
        irx = yf.download('^IRX', start=start, end=end,
                         auto_adjust=False, progress=False)['Close']
        if isinstance(irx, pd.DataFrame):
            irx = irx.iloc[:, 0]
        irx.index = _strip_tz(irx.index)
        # ^IRX 는 연율 % → 일별 소수 (252영업일 가정)
        rf_annual = irx / 100.0
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        rf_daily.name = 'rf_daily'
    except Exception as e:
        print(f'    ^IRX 실패: {e} → FRED DGS3MO 시도')
        from pandas_datareader import data as pdr
        dgs = pdr.DataReader('DGS3MO', 'fred', start, end)['DGS3MO']
        rf_annual = dgs / 100.0
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        rf_daily.name = 'rf_daily'

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rf_daily.to_frame().to_csv(cache_path, index_label='date')
    print(f'  [data] 무위험 수익률 저장: {cache_path.name}')
    return rf_daily


# =============================================================================
# 종목별 피처 계산
# =============================================================================
def compute_panel_features(
    ticker: str,
    ohlcv: pd.DataFrame,
    spy_ret: pd.Series,
    rf_daily: pd.Series,
    shares_map: dict,
    splits_map: dict,
) -> pd.DataFrame:
    """단일 종목 일별 피처 계산.

    Parameters
    ----------
    ticker : 종목 코드
    ohlcv : 일별 OHLCV (Open, High, Low, Close, Volume)
    spy_ret : SPY 일별 log return (시장)
    rf_daily : 무위험 수익률 (일별)
    shares_map : raw 발행주식수 시계열 dict
    splits_map : 분할 이력 dict

    Returns
    -------
    pd.DataFrame
        date 인덱스 + columns:
            close, log_ret, vol_21d, vol_60d, vol_252d, beta_252d,
            mcap_value, log_mcap, target_logrv (Phase 1.5 와 통일)

    Notes
    -----
    - log_ret = log(close[t] / close[t-1])
    - vol_*d = log_ret.rolling(*).std()  (일별 단위, Phase 1.5 와 동일)
    - target_logrv = log(rolling(21).std()).shift(-21)  (Phase 1.5 타깃)
    - log_mcap = log(close × adjusted_shares)
    """
    if ohlcv is None or len(ohlcv) == 0:
        return pd.DataFrame()

    close = ohlcv['Close'].dropna()
    close = close[~close.index.duplicated(keep='last')]

    df = pd.DataFrame(index=close.index)
    df['close'] = close

    # 1. 로그 수익률
    lr = np.log(close / close.shift(1))
    df['log_ret'] = lr

    # 2. 실현 변동성 (일별 단위 std, Phase 1.5 와 동일)
    df['vol_21d'] = lr.rolling(21).std()
    df['vol_60d'] = lr.rolling(60).std()
    df['vol_252d'] = lr.rolling(252).std()

    # 3. CAPM 베타 (vs SPY, 252일 rolling)
    excess = lr - rf_daily.reindex(close.index).fillna(0)
    spy = spy_ret.reindex(close.index)
    cov = excess.rolling(252).cov(spy)
    var = spy.rolling(252).var()
    df['beta_252d'] = cov / var

    # 4. 분할 보정 시총
    mcap_values = []
    for d in close.index:
        adj_shares = get_adjusted_shares_at(ticker, d, shares_map, splits_map)
        if pd.notna(adj_shares) and adj_shares > 0:
            mcap_values.append(close.loc[d] * adj_shares)
        else:
            mcap_values.append(np.nan)
    df['mcap_value'] = mcap_values
    df['log_mcap'] = np.log(df['mcap_value'].clip(lower=1))

    # 5. Phase 1.5 타깃 — 21일 forward log-RV
    rv_21 = lr.rolling(21).std()
    df['target_logrv'] = np.log(rv_21).shift(-21)

    # 6. 이중상장 secondary 면 빈 DataFrame 반환 (universe 에서 어차피 제외)
    if ticker in DUAL_LISTED_MAP:
        return pd.DataFrame()

    return df


# =============================================================================
# 통합 일별 패널
# =============================================================================
def build_daily_panel(
    universe_csv: Path,
    cache_dir: Path,
    sector_map: dict | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """전체 universe 종목 통합 일별 패널.

    Returns
    -------
    pd.DataFrame
        long format: (date, ticker) → features

    Steps
    -----
    1. universe ticker list 로드
    2. OHLCV 다운로드 (캐시 활용)
    3. 시장 데이터 (SPY, VIX, TNX) 다운로드
    4. 무위험 수익률 다운로드
    5. 종목별 피처 계산
    6. long format 통합
    """
    panel_path = cache_dir / 'daily_panel.csv'
    if panel_path.exists() and not overwrite:
        print(f'  [data] 캐시된 daily_panel 로드: {panel_path}')
        df = pd.read_csv(panel_path, parse_dates=['date'])
        return df

    # 1. universe ticker (이미 GOOG 등 secondary 제외된 unique list)
    tickers = get_universe_tickers(universe_csv)
    print(f'  [data] universe ticker: {len(tickers)} 종목')

    # 2. 시점 범위 (universe 의 가장 빠른 cutoff - 7년 ~ 가장 늦은 cutoff + 1년)
    universe_df = pd.read_csv(universe_csv, parse_dates=['cutoff_date'])
    earliest_cutoff = universe_df['cutoff_date'].min()
    start = earliest_cutoff - pd.DateOffset(years=7)
    end = pd.Timestamp(f'{universe_df["oos_year"].max() + 1}-01-01')
    print(f'  [data] 데이터 범위: {start.date()} ~ {end.date()}')

    # 3. OHLCV 다운로드
    ohlcv_dir = cache_dir / 'prices_daily'
    ohlcv_map = download_universe_ohlcv(tickers, start, end, ohlcv_dir)

    # 4. 시장 데이터
    market_path = cache_dir / 'market_data.csv'
    market = download_market_data(start, end, market_path)
    spy_ret = np.log(market['SPY'] / market['SPY'].shift(1))

    # 5. 무위험 수익률
    rf_path = cache_dir / 'risk_free.csv'
    rf_daily = download_risk_free(start, end, rf_path)

    # 6. 발행주식수 + 분할 이력 로드 (Step 1 캐시)
    import pickle
    with open(cache_dir / 'shares_outstanding.pkl', 'rb') as f:
        shares_map = pickle.load(f)
    with open(cache_dir / 'splits_history.pkl', 'rb') as f:
        splits_map = pickle.load(f)

    # 7. 종목별 피처 계산 + 통합
    print(f'  [data] 종목별 피처 계산 중...')
    rows = []
    for i, ticker in enumerate(tickers):
        if ticker not in ohlcv_map:
            print(f'    {ticker}: OHLCV 없음 → skip')
            continue
        df_t = compute_panel_features(
            ticker, ohlcv_map[ticker], spy_ret, rf_daily, shares_map, splits_map
        )
        if len(df_t) == 0:
            continue
        df_t['ticker'] = ticker
        if sector_map and ticker in sector_map:
            df_t['gics_sector'] = sector_map[ticker]
        else:
            df_t['gics_sector'] = 'Unknown'
        rows.append(df_t.reset_index().rename(columns={'index': 'date', 'Date': 'date'}))
        if (i + 1) % 20 == 0:
            print(f'    진행: {i + 1}/{len(tickers)}')

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(['date', 'ticker']).reset_index(drop=True)

    # 8. 시장 데이터 broadcast
    panel = panel.merge(market[['SPY', 'VIX', 'TNX']], left_on='date', right_index=True, how='left')
    panel = panel.rename(columns={'SPY': 'spy_close', 'VIX': 'vix', 'TNX': 'tnx'})
    panel['spy_log_ret'] = panel.groupby('ticker')['spy_close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    # rf_daily broadcast
    rf_lookup = rf_daily.to_frame('rf_daily').reset_index()
    rf_lookup.columns = ['date', 'rf_daily']
    panel = panel.merge(rf_lookup, on='date', how='left')

    # 9. 저장
    panel.to_csv(panel_path, index=False)
    print(f'  [data] daily_panel 저장: {panel_path}')
    print(f'  [data] shape: {panel.shape}, 종목 수: {panel["ticker"].nunique()}')

    return panel


# =============================================================================
# 빠른 panel 빌드 (prices_close_universe.pkl 활용, yfinance 재다운로드 X)
# =============================================================================
def build_daily_panel_fast(
    universe_csv: Path,
    cache_dir: Path,
    sector_map: dict | None = None,
    overwrite: bool = True,
    out_name: str = 'daily_panel.csv',
    min_data_days: int = 252,
) -> pd.DataFrame:
    """⭐ 전체 S&P 500 (~624) 빠른 panel 빌드.

    `prices_close_universe.pkl` 의 Close 데이터 직접 사용 → yfinance OHLCV 재다운로드 불필요.
    `compute_panel_features` 가 Close 만 사용하므로 OHLCV 다운로드는 불필요한 작업.

    Parameters
    ----------
    universe_csv : Path
        universe_full_history.csv 등 (전체 멤버 list).
    cache_dir : Path
        캐시 디렉토리. prices_close_universe.pkl, market_data.csv,
        risk_free.csv, shares_outstanding.pkl, splits_history.pkl 필요.
    sector_map : dict, optional
        ticker → gics_sector. None 시 'Unknown'.
    overwrite : bool, default True
        기존 daily_panel.csv 무시하고 재빌드.
    out_name : str, default 'daily_panel.csv'
    min_data_days : int, default 252 (서윤범 일관)
        최소 거래일 수. 미달 시 skip.

    Returns
    -------
    pd.DataFrame
        long format daily panel. shape: (sum_dates × n_valid_tickers, 17).
    """
    import pickle

    panel_path = cache_dir / out_name
    if panel_path.exists() and not overwrite:
        print(f'  [data-fast] 캐시된 daily_panel 로드: {panel_path}')
        return pd.read_csv(panel_path, parse_dates=['date'])

    # 1. universe ticker
    universe_df = pd.read_csv(universe_csv)
    tickers = sorted(universe_df['ticker'].unique().tolist())
    print(f'  [data-fast] universe: {len(tickers)} 종목')

    # 2. close 캐시 로드
    close_cache_path = cache_dir / 'prices_close_universe.pkl'
    if not close_cache_path.exists():
        raise FileNotFoundError(
            f'{close_cache_path} 없음. 먼저 build_universe_history() 실행 필요.'
        )
    close_df = pd.read_pickle(close_cache_path)
    print(f'  [data-fast] close 캐시: {close_df.shape} (rows × tickers)')

    # 3. 시장 데이터
    market_path = cache_dir / 'market_data.csv'
    if not market_path.exists():
        raise FileNotFoundError(f'{market_path} 없음')
    market = pd.read_csv(market_path, parse_dates=['date'], index_col='date')
    spy_ret = np.log(market['SPY'] / market['SPY'].shift(1))

    # 4. 무위험 수익률
    rf_path = cache_dir / 'risk_free.csv'
    if not rf_path.exists():
        raise FileNotFoundError(f'{rf_path} 없음')
    rf_daily = pd.read_csv(rf_path, parse_dates=['date'], index_col='date').iloc[:, 0]

    # 5. 발행주식수 + 분할
    with open(cache_dir / 'shares_outstanding.pkl', 'rb') as f:
        shares_map = pickle.load(f)
    with open(cache_dir / 'splits_history.pkl', 'rb') as f:
        splits_map = pickle.load(f)

    # 6. 종목별 피처 계산 (tqdm progress)
    print(f'  [data-fast] 종목별 피처 계산 중 (close 캐시 직접 활용)...')
    try:
        from tqdm.auto import tqdm
        ticker_iter = tqdm(tickers, desc='Panel features', ncols=100)
    except ImportError:
        ticker_iter = tickers

    rows = []
    skipped_no_close = 0
    skipped_short = 0
    skipped_dual = 0
    for ticker in ticker_iter:
        if ticker not in close_df.columns:
            skipped_no_close += 1
            continue
        if ticker in DUAL_LISTED_MAP:
            skipped_dual += 1
            continue
        close = close_df[ticker].dropna()
        if len(close) < min_data_days:
            skipped_short += 1
            continue

        # OHLCV 인터페이스 호환: 'Close' 컬럼만 있는 DataFrame
        ohlcv = pd.DataFrame({'Close': close})
        df_t = compute_panel_features(
            ticker, ohlcv, spy_ret, rf_daily, shares_map, splits_map
        )
        if len(df_t) == 0:
            continue
        df_t['ticker'] = ticker
        if sector_map and ticker in sector_map:
            df_t['gics_sector'] = sector_map[ticker]
        else:
            df_t['gics_sector'] = 'Unknown'
        rows.append(df_t.reset_index().rename(columns={'index': 'date', 'Date': 'date'}))

    print(f'  [data-fast] skip 통계: no_close={skipped_no_close}, '
          f'<{min_data_days}일={skipped_short}, dual={skipped_dual}')
    print(f'  [data-fast] 유효 종목: {len(rows)}')

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(['date', 'ticker']).reset_index(drop=True)

    # 7. 시장 데이터 broadcast
    panel = panel.merge(market[['SPY', 'VIX', 'TNX']],
                        left_on='date', right_index=True, how='left')
    panel = panel.rename(columns={'SPY': 'spy_close', 'VIX': 'vix', 'TNX': 'tnx'})
    panel['spy_log_ret'] = panel.groupby('ticker')['spy_close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    rf_lookup = rf_daily.to_frame('rf_daily').reset_index()
    rf_lookup.columns = ['date', 'rf_daily']
    panel = panel.merge(rf_lookup, on='date', how='left')

    # 8. 저장
    panel.to_csv(panel_path, index=False)
    print(f'  [data-fast] daily_panel 저장: {panel_path}')
    print(f'  [data-fast] shape: {panel.shape}, 종목 수: {panel["ticker"].nunique()}')

    return panel
