"""Phase 2 — Universe Construction (시총 상위 50 매년 산정).

Pyo & Lee (2018, PBFJ 51) "Low-Risk Anomaly + BL" 의 미국 시장 적응.
서윤범 [01_DataCollection.ipynb](../../서윤범/low_risk/01_DataCollection.ipynb) 의
S&P 500 멤버십 + 발행주식수 수집 로직을 재사용.

핵심 함수
---------
- fetch_sp500_membership_history()  : Wikipedia 시점별 S&P 500 멤버십
- fetch_shares_outstanding()        : yfinance 종목별 발행주식수 시계열
- compute_mcap_at_cutoff()          : 특정 시점 시총 계산
- validate_data_coverage()          : 데이터 가용성 검증 (≥ min_days 영업일)
- get_universe_top50_with_fallback(): 매년 시총 상위 50 + 부족 시 51위 이하 대체
- build_universe_history()          : 6 연도 통합 universe 산출

사용 예시
---------
from scripts.universe import build_universe_history
from scripts.setup import DATA_DIR

history_df = build_universe_history(
    years=range(2020, 2026),      # OOS 연도들
    n_top=50,                      # 시총 상위 N
    max_candidates=80,             # fallback 시 검토할 후보 수
    min_data_days=1732,            # 가용성 검증 임계 (≈ 7.7년)
    cache_dir=DATA_DIR,            # 캐시 디렉토리
)

설계 원칙
---------
- look-ahead 차단: cutoff = (year-1)년 12월 마지막 거래일
- 캐싱: Wikipedia 멤버십, 발행주식수, 종가 모두 pickle 캐시
- 부족 종목 자동 대체: max_candidates 까지 검토 후 50 종목 채움
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


# =============================================================================
# Wikipedia S&P 500 멤버십
# =============================================================================
WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'


def fetch_sp500_tables() -> list:
    """Wikipedia S&P 500 페이지에서 wikitable 2개 (현재 + 변경 이력) 반환.

    Notes
    -----
    Wikipedia 는 User-Agent 헤더 없는 요청에 403 Forbidden 응답.
    표준 브라우저 User-Agent 를 명시.
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'lxml')
    return soup.find_all('table', {'class': 'wikitable'})


def parse_current_sp500(table) -> pd.DataFrame:
    """Wikipedia 첫 번째 wikitable: 현재 S&P 500 list."""
    df = pd.read_html(str(table))[0]
    df = df.rename(columns={'Symbol': 'ticker', 'GICS Sector': 'gics_sector'})
    df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)  # BRK.B → BRK-B (yfinance 표기)
    return df[['ticker', 'gics_sector']].copy()


def parse_changes(table) -> pd.DataFrame:
    """Wikipedia 두 번째 wikitable: 추가/삭제 이력."""
    df = pd.read_html(str(table))[0]
    # 멀티 헤더 평탄화
    df.columns = [' '.join(c).strip() if isinstance(c, tuple) else c for c in df.columns]

    # 컬럼명 표준화 — 위키 페이지의 헤더 변경에 강건하게 대응
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = next((v for k, v in cols_lower.items() if 'date' in k), df.columns[0])
    add_col = next((v for k, v in cols_lower.items() if 'added' in k and 'ticker' in k), None)
    rem_col = next((v for k, v in cols_lower.items() if 'removed' in k and 'ticker' in k), None)

    out = pd.DataFrame({
        'date': pd.to_datetime(df[date_col], errors='coerce'),
        'added': df[add_col].astype(str).str.replace('.', '-', regex=False) if add_col else '',
        'removed': df[rem_col].astype(str).str.replace('.', '-', regex=False) if rem_col else '',
    }).dropna(subset=['date'])
    return out


def build_membership_history(
    df_current: pd.DataFrame,
    df_changes: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """월별 S&P 500 멤버십 히스토리 구성.

    역방향 재구성: 현재 멤버에서 시작 → 변경 이력 역적용 → 과거 시점 멤버.

    Parameters
    ----------
    df_current : 현재 S&P 500 (ticker, sector)
    df_changes : 추가/삭제 이력 (date, added, removed)
    start, end : 히스토리 범위

    Returns
    -------
    dict[pd.Timestamp, frozenset]
        월말 시점 → 멤버 set
    """
    current_set = set(df_current['ticker'])
    df_changes = df_changes.sort_values('date', ascending=False)

    membership = {}
    months = pd.date_range(start, end, freq='ME')  # 월말

    for target in reversed(months):
        # target 시점의 멤버: 현재에서 (target+1 이후) 변경 사항을 역적용
        snapshot = set(current_set)
        for _, row in df_changes.iterrows():
            change_date = row['date']
            if change_date <= target:
                break
            # change_date > target 이므로 역적용:
            # added 종목 제거, removed 종목 추가
            if row['added'] and row['added'] != 'nan':
                snapshot.discard(row['added'])
            if row['removed'] and row['removed'] != 'nan':
                snapshot.add(row['removed'])
        membership[target] = frozenset(snapshot)

    return membership


def get_or_build_membership(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path,
) -> dict:
    """멤버십 캐시 로드 또는 신규 빌드."""
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f'  [universe] Wikipedia S&P 500 멤버십 조회 중...')
    tables = fetch_sp500_tables()
    df_current = parse_current_sp500(tables[0])
    df_changes = parse_changes(tables[1]) if len(tables) >= 2 else pd.DataFrame(columns=['date', 'added', 'removed'])

    membership = build_membership_history(df_current, df_changes, start, end)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(membership, f)
    print(f'  [universe] 멤버십 빌드 완료 ({len(membership)} 개월) → {cache_path.name}')

    return membership


def get_members_at(date: pd.Timestamp, membership: dict) -> frozenset:
    """특정 날짜의 멤버십 (가장 가까운 과거 월말 매칭)."""
    keys = sorted(membership.keys())
    valid_keys = [k for k in keys if k <= date]
    if not valid_keys:
        return frozenset()
    return membership[valid_keys[-1]]


# =============================================================================
# 발행주식수 (yfinance)
# =============================================================================
def _strip_tz(ts_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """timezone-aware 인덱스 → timezone-naive (UTC 변환 후 tz 제거).

    yfinance 데이터는 종종 'America/New_York' tz-aware 로 반환되어
    tz-naive Timestamp 와 비교 시 TypeError 발생. 일관된 비교를 위해
    모든 시점 정보를 tz-naive 로 통일.
    """
    if ts_index.tz is not None:
        return ts_index.tz_localize(None)
    return ts_index


def fetch_shares_outstanding(
    tickers: Iterable[str],
    start: pd.Timestamp,
    cache_path: Path,
    sleep_per_ticker: float = 0.1,
) -> dict:
    """종목별 발행주식수 시계열 수집 (캐시 활용). tz-naive 정규화."""
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            existing = pickle.load(f)
        # 캐시 로드 시 timezone 정규화 (빈 Series RangeIndex 안전 처리)
        for tk, ts in existing.items():
            if (ts is not None
                and isinstance(ts.index, pd.DatetimeIndex)
                and ts.index.tz is not None):
                ts.index = ts.index.tz_localize(None)
    else:
        existing = {}

    new_tickers = [t for t in tickers if t not in existing]
    if not new_tickers:
        return existing

    print(f'  [universe] 발행주식수 수집: {len(new_tickers)}개 신규 종목')

    for i, ticker in enumerate(new_tickers):
        try:
            ts = yf.Ticker(ticker).get_shares_full(start=start)
            if ts is not None and len(ts) > 0:
                ts.index = _strip_tz(ts.index)
                existing[ticker] = ts
        except Exception as e:
            print(f'    {ticker}: 수집 실패 ({e})')
        time.sleep(sleep_per_ticker)
        if (i + 1) % 50 == 0:
            print(f'    진행: {i + 1}/{len(new_tickers)} (성공 누적: {len(existing)})')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(existing, f)
    print(f'  [universe] 발행주식수 저장 완료 ({len(existing)} 종목) → {cache_path.name}')

    return existing


def get_shares_at(
    ticker: str,
    date: pd.Timestamp,
    shares_map: dict,
    forward_fallback: bool = False,
) -> float:
    """특정 시점 raw 발행주식수 (가장 가까운 과거 일자 매칭). tz 안전.

    Parameters
    ----------
    forward_fallback : bool, default False
        True 이면 date 이전 데이터가 없을 때 가장 이른 미래 데이터를 proxy 로 사용.
        yfinance 가 2009 이전 shares 데이터를 보유하지 않는 경우 사용.
        ranking 목적에만 사용 (거래 판단 X).

    Notes
    -----
    이 함수는 **raw** (분할 미보정) 발행주식수를 반환한다.
    Adj Close (분할 보정된 가격) 와 함께 사용 시 시총이 왜곡된다.
    분할 보정된 시총은 `get_adjusted_shares_at()` 사용.
    """
    if ticker not in shares_map:
        return np.nan
    ts = shares_map[ticker]
    if ts is None or len(ts) == 0:
        return np.nan
    if ts.index.tz is not None:
        ts = ts.copy()
        ts.index = ts.index.tz_localize(None)
    ts_filtered = ts[ts.index <= date]
    if len(ts_filtered) == 0:
        if forward_fallback and len(ts) > 0:
            # yfinance 가 date 이전 데이터가 없는 경우 (예: 2008-12-31 cutoff).
            # 가장 이른 보유 데이터를 proxy 로 사용 (ranking 목적).
            return float(ts.iloc[0])
        return np.nan
    return float(ts_filtered.iloc[-1])


# =============================================================================
# 분할 이력 (yfinance.splits) + 분할 보정된 발행주식수
# =============================================================================
def fetch_splits(
    tickers: Iterable[str],
    cache_path: Path,
    sleep_per_ticker: float = 0.05,
) -> dict:
    """종목별 분할 이력 (yfinance Ticker.splits) 수집 (캐시 활용).

    Returns
    -------
    dict[str, pd.Series]
        ticker → splits Series (index=split date, value=ratio)
        4:1 split → ratio=4.0
        1:8 reverse split → ratio=0.125

    Notes
    -----
    yfinance 의 splits 는 tz-aware (America/New_York) → tz-naive 정규화.
    """
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            existing = pickle.load(f)
        # 캐시 로드 시 timezone 정규화 (빈 Series 의 RangeIndex 안전 처리)
        for tk, sp in existing.items():
            if (sp is not None
                and isinstance(sp.index, pd.DatetimeIndex)
                and sp.index.tz is not None):
                sp.index = sp.index.tz_localize(None)
    else:
        existing = {}

    new_tickers = [t for t in tickers if t not in existing]
    if not new_tickers:
        return existing

    print(f'  [universe] 분할 이력 수집: {len(new_tickers)}개 신규 종목')
    for i, ticker in enumerate(new_tickers):
        try:
            splits = yf.Ticker(ticker).splits
            if splits is not None and len(splits) > 0:
                splits.index = _strip_tz(splits.index)
                existing[ticker] = splits
            else:
                existing[ticker] = pd.Series(dtype=float)  # 분할 없음 (빈 Series)
        except Exception as e:
            print(f'    {ticker}: 분할 수집 실패 ({e})')
            existing[ticker] = pd.Series(dtype=float)
        time.sleep(sleep_per_ticker)
        if (i + 1) % 100 == 0:
            print(f'    진행: {i + 1}/{len(new_tickers)}')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(existing, f)
    print(f'  [universe] 분할 이력 저장: {len(existing)} 종목 → {cache_path.name}')

    return existing


def get_adjusted_shares_at(
    ticker: str,
    date: pd.Timestamp,
    shares_map: dict,
    splits_map: dict,
) -> float:
    """분할 보정된 발행주식수 (date 시점 기준).

    Adj Close (분할 보정된 가격) 와 일관된 시총 계산을 위해
    raw 발행주식수에 date 이후 발생한 모든 분할 ratio 의 누적곱을 적용.

    수식
    ----
    adjusted_shares = raw_shares × ∏ (future split ratios)

    예시 (GE 2019-12-31)
    ---------------------
    raw_shares = 8.73e9 (87억)
    future splits: 2021-08 (0.125) × 2023-01 (1.281) × 2024-04 (1.253) = 0.2007
    adjusted_shares = 87억 × 0.2007 = 17.52억
    시총 = 17.52억 × $54.02 (Adj Close) = $94.6B  ≈ 실제 ~$97B ✅

    예시 (AAPL 2019-12-31)
    -----------------------
    raw_shares ≈ 4.4e9
    future splits: 2020-08 (4:1, ratio=4.0)
    adjusted_shares = 4.4억 × 4 = 17.6억
    시총 = 17.6억 × Adj Close ≈ 정확
    ※ 그러나 yfinance 는 분할 후 raw 값을 갱신하므로 시점 따라 다름.
       시점 t 에 4.4억 (분할 전) 또는 17.6억 (분할 후) 둘 다 가능.
       이 함수는 어느 쪽이든 "현재 시점 기준 보정" 을 일관 적용.
    """
    # date 이전 데이터가 없으면 forward fallback (가장 이른 보유 데이터 proxy 사용)
    ts = shares_map.get(ticker)
    if ts is not None and isinstance(getattr(ts, 'index', None), pd.DatetimeIndex):
        if ts.index.tz is not None:
            ts = ts.copy()
            ts.index = ts.index.tz_localize(None)
        has_history = len(ts[ts.index <= date]) > 0
    else:
        has_history = True  # get_shares_at 이 NaN 반환하면 아래서 잡힘

    raw_shares = get_shares_at(ticker, date, shares_map, forward_fallback=True)
    if pd.isna(raw_shares):
        return np.nan

    splits = splits_map.get(ticker)
    if splits is None or len(splits) == 0:
        return raw_shares

    # tz 안전 (DatetimeIndex 일 때만 tz 처리, RangeIndex 빈 Series 방어)
    if isinstance(splits.index, pd.DatetimeIndex) and splits.index.tz is not None:
        splits = splits.copy()
        splits.index = splits.index.tz_localize(None)
    elif not isinstance(splits.index, pd.DatetimeIndex):
        # 비정상 인덱스 → split 미적용 (raw 그대로)
        return raw_shares

    if not has_history and ts is not None and len(ts) > 0:
        # Fallback 케이스: raw_shares 는 ts.iloc[0] (fallback_date) 시점 데이터.
        # fallback_date 이전에 발생한 분할은 이미 ts.iloc[0] 에 반영돼 있으므로
        # fallback_date 이후의 분할만 적용 (이중 반영 방지).
        fallback_date = ts.index[0]
        future_splits = splits[splits.index > fallback_date]
    else:
        future_splits = splits[splits.index > date]

    if len(future_splits) == 0:
        return raw_shares

    cum_ratio = float(future_splits.prod())
    return raw_shares * cum_ratio


# =============================================================================
# 종가 (yfinance)
# =============================================================================
def fetch_close_prices(
    tickers: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path,
) -> pd.DataFrame:
    """종목별 일별 종가 (Adj Close) 수집 (캐시 활용)."""
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        cached.index = _strip_tz(cached.index)
        existing_tickers = set(cached.columns)
        new_tickers = [t for t in tickers if t not in existing_tickers]
        if not new_tickers:
            return cached
        # 부족분만 추가 다운로드
        print(f'  [universe] 종가 수집: {len(new_tickers)}개 신규 종목')
        new_data = yf.download(new_tickers, start=start, end=end,
                              auto_adjust=True, progress=False)['Close']
        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame(new_tickers[0])
        new_data.index = _strip_tz(new_data.index)
        merged = pd.concat([cached, new_data], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged.to_pickle(cache_path)
        return merged

    tickers_list = list(tickers)
    print(f'  [universe] 종가 수집: {len(tickers_list)}개 종목 (전체)')
    data = yf.download(tickers_list, start=start, end=end,
                      auto_adjust=True, progress=False)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers_list[0])
    data.index = _strip_tz(data.index)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(cache_path)
    return data


# =============================================================================
# 이중상장 (Dual-Listed) 종목 통합 + 시총 산정
# =============================================================================
# 동일 회사가 두 개 이상 클래스 주식으로 상장된 경우 시총 합산 + 단일 대표 유지.
# 이유: BL 공분산 다중공선성 방지 (상관계수 ≈ 1.0) + Alphabet 단일 회사 정확 반영.
#
# 매핑: secondary → primary (시총 합산 후 secondary 제거, primary 만 유지)
DUAL_LISTED_MAP = {
    'GOOG': 'GOOGL',   # Alphabet Inc. — Class C (의결권 0) → Class A (의결권 1)
    # 향후 추가 가능: 'BRK-A': 'BRK-B' (가격 차이 매우 큼, 별도 검토)
    #                'FOX': 'FOXA', 'NWS': 'NWSA' (Top 50 외)
}


def consolidate_dual_listings(mcaps: pd.Series) -> pd.Series:
    """이중상장 종목 통합: secondary 제거, primary 시총 유지.

    Parameters
    ----------
    mcaps : pd.Series
        ticker → mcap (NaN 가능)

    Returns
    -------
    pd.Series
        secondary ticker 가 제거되고 primary 만 유지된 결과.

    Notes
    -----
    yfinance `get_shares_full()` 의 함정:
    GOOGL 과 GOOG 두 ticker 모두 Alphabet **전체** 발행주식수 (Class A + C) 를 반환.
    예: 2019-12-31 GOOGL raw = 688M, GOOG raw = 691M (거의 동일).
    각각의 시총 ≈ \$915B (Alphabet 전체). 따라서 합산 시 2배 과대평가.

    올바른 처리: secondary drop 만 수행. primary 시총은 이미 Alphabet 전체.

    Examples
    --------
    >>> mcaps = pd.Series({'GOOGL': 915e9, 'GOOG': 916e9, 'AAPL': 1258e9})
    >>> result = consolidate_dual_listings(mcaps)
    >>> result['GOOGL']  # 합산 X — primary 그대로
    915000000000.0
    >>> 'GOOG' in result.index
    False
    """
    result = mcaps.copy()
    for secondary, primary in DUAL_LISTED_MAP.items():
        if secondary in result.index:
            sec_val = result[secondary]
            if primary in result.index:
                pri_val = result[primary]
                # primary 가 NaN 이고 secondary 만 유효한 경우 → secondary 값을 primary 로 이전
                if pd.isna(pri_val) and pd.notna(sec_val):
                    result[primary] = sec_val
                # 그 외: primary 시총 그대로 유지 (yfinance 가 이미 전체 시총 반환)
            else:
                # primary 가 mcaps 에 없는 경우 → secondary 를 primary 로 rename
                result[primary] = sec_val
            # secondary 제거
            result = result.drop(secondary)
    return result


def compute_mcap_at_cutoff(
    tickers: Iterable[str],
    cutoff: pd.Timestamp,
    prices: pd.DataFrame,
    shares_map: dict,
    splits_map: dict | None = None,
    consolidate_dual: bool = True,
) -> pd.Series:
    """cutoff 시점 분할 보정 시총 = Adj Close × adjusted shares.

    Parameters
    ----------
    tickers : 대상 종목 list
    cutoff : 산정 시점 (예: 2023-12-31)
    prices : 일별 종가 (auto_adjust=True 의 Adj Close)
    shares_map : raw 발행주식수 시계열 dict
    splits_map : 분할 이력 dict. None 이면 raw shares 그대로 사용 (이전 버전 호환).
    consolidate_dual : True 이면 GOOG/GOOGL 등 이중상장 종목 시총 합산.

    Returns
    -------
    pd.Series
        ticker → mcap (NaN 은 데이터 부족 종목)

    Notes
    -----
    Adj Close 와 일관된 시총 = Adj Close × adjusted shares.
    splits_map 제공 시 raw shares 에 미래 분할 누적 ratio 곱하여 보정.
    consolidate_dual=True (기본값) 시 DUAL_LISTED_MAP 적용.
    """
    # cutoff 직전 영업일 (cutoff 가 휴장일인 경우 대비)
    available_dates = prices.index[prices.index <= cutoff]
    if len(available_dates) == 0:
        return pd.Series(dtype=float)
    actual_cutoff = available_dates[-1]

    mcaps = {}
    for ticker in tickers:
        if ticker not in prices.columns:
            mcaps[ticker] = np.nan
            continue
        price = prices.loc[actual_cutoff, ticker]
        if splits_map is not None:
            shares = get_adjusted_shares_at(ticker, actual_cutoff, shares_map, splits_map)
        else:
            shares = get_shares_at(ticker, actual_cutoff, shares_map)
        if pd.notna(price) and pd.notna(shares) and shares > 0:
            mcaps[ticker] = float(price) * shares
        else:
            mcaps[ticker] = np.nan

    mcaps_series = pd.Series(mcaps, name='mcap')

    # 이중상장 통합 (GOOG → GOOGL 등)
    if consolidate_dual:
        mcaps_series = consolidate_dual_listings(mcaps_series)

    return mcaps_series


def validate_data_coverage(
    ticker: str,
    oos_year: int,
    prices: pd.DataFrame,
    min_days: int = 1732,
) -> bool:
    """종목의 데이터 가용성 검증.

    OOS 연도 기준 [oos_year-7, oos_year+1) 범위 내 영업일 수가
    min_days 이상이어야 학습·평가 가능.
    """
    if ticker not in prices.columns:
        return False
    start = pd.Timestamp(f'{oos_year - 7}-01-01')
    end = pd.Timestamp(f'{oos_year + 1}-01-01')
    series = prices[ticker].loc[start:end].dropna()
    return len(series) >= min_days


# =============================================================================
# Universe 매년 산정
# =============================================================================
def get_universe_top50_with_fallback(
    oos_year: int,
    prices: pd.DataFrame,
    shares_map: dict,
    membership: dict,
    splits_map: dict | None = None,
    n_top: int = 50,
    max_candidates: int = 80,
    min_data_days: int = 1732,
) -> pd.DataFrame:
    """OOS 연도 t 의 시총 상위 50 + fallback.

    1) cutoff = (year-1)년 12월 마지막 거래일
    2) 그 시점 S&P 500 멤버 중 분할 보정 시총 상위 max_candidates
    3) 가용성 검증 통과 종목만 → 정확히 n_top 개 채움

    Returns
    -------
    pd.DataFrame
        columns: oos_year, ticker, mcap_rank, mcap_value, cutoff_date
    """
    cutoff = pd.Timestamp(f'{oos_year - 1}-12-31')
    members = list(get_members_at(cutoff, membership))

    if len(members) == 0:
        raise RuntimeError(f'{oos_year}: 멤버십 비어있음')

    # 분할 보정 시총 상위 max_candidates
    mcaps = compute_mcap_at_cutoff(members, cutoff, prices, shares_map, splits_map)

    # ⭐ 안전망: 어떤 import 캐시 상태든 secondary ticker 강제 제거
    # (compute_mcap_at_cutoff 가 옛 버전이라 consolidate_dual_listings 적용 안 됐어도 OK)
    for _secondary in DUAL_LISTED_MAP:
        if _secondary in mcaps.index:
            mcaps = mcaps.drop(_secondary)

    mcaps = mcaps.dropna().sort_values(ascending=False)
    candidates = mcaps.head(max_candidates).index.tolist()

    # 가용성 검증 → n_top 채움
    valid = []
    for ticker in candidates:
        if validate_data_coverage(ticker, oos_year, prices, min_data_days):
            valid.append(ticker)
            if len(valid) == n_top:
                break

    if len(valid) < n_top:
        print(f'  [universe] WARN {oos_year}: {len(valid)}/{n_top} 종목만 확보 (fallback 부족)')

    rows = []
    for rank, ticker in enumerate(valid, 1):
        rows.append({
            'oos_year': oos_year,
            'ticker': ticker,
            'mcap_rank': rank,
            'mcap_value': float(mcaps[ticker]),
            'cutoff_date': cutoff,
        })

    return pd.DataFrame(rows)


def build_universe_history(
    years: Iterable[int],
    cache_dir: Path,
    n_top: int = 50,
    max_candidates: int = 80,
    min_data_days: int = 1732,
) -> pd.DataFrame:
    """6 OOS 연도 통합 universe 산출.

    Returns
    -------
    pd.DataFrame
        columns: oos_year, ticker, mcap_rank, mcap_value, cutoff_date, is_new
        - is_new: 직전 연도 universe 에 없던 신규 편입 종목 (True/False)
    """
    years_list = sorted(set(years))
    earliest_year = min(years_list)
    latest_year = max(years_list)

    # 1) 멤버십 추적 범위 (서윤범 일관: OOS 시작년도 ~ 종료년도)
    # ⭐ 이전엔 earliest_year - 7 (= 2002) 으로 설정해 2002~2008 멤버까지 포함 (831 종목).
    # 서윤범 baseline 과 fair 비교 위해 OOS 기간 (2009~2025) 멤버만 추적 → 624 종목 예상.
    membership_start = pd.Timestamp(f'{earliest_year}-01-01')
    membership_end = pd.Timestamp(f'{latest_year + 1}-01-01')
    membership_cache = cache_dir / 'sp500_membership.pkl'
    membership = get_or_build_membership(membership_start, membership_end, membership_cache)

    # 2) 모든 멤버 union (캐시할 종목 목록)
    all_members = set()
    for member_set in membership.values():
        all_members.update(member_set)
    print(f'  [universe] 전체 unique 멤버: {len(all_members)} 종목 (멤버십 추적: {membership_start.date()} ~ {membership_end.date()})')

    # 3) 종가 캐시 (전체 종목)
    # ⭐ 가격 데이터는 IS 학습용으로 -7년 버퍼 유지 (LSTM IS=1250일 ≈ 5년 + 안전 버퍼)
    data_start = pd.Timestamp(f'{earliest_year - 7}-01-01')
    history_end = membership_end
    prices_cache = cache_dir / 'prices_close_universe.pkl'
    prices = fetch_close_prices(
        all_members, data_start, history_end, prices_cache,
    )
    print(f'  [universe] 종가 데이터: {prices.shape} (행={len(prices)}, 종목={prices.shape[1]})')

    # 4) 발행주식수 캐시 (전체 종목)
    shares_cache = cache_dir / 'shares_outstanding.pkl'
    shares_map = fetch_shares_outstanding(
        all_members, data_start, shares_cache,
    )
    print(f'  [universe] 발행주식수 데이터: {len(shares_map)} 종목')

    # 5) 분할 이력 캐시 (전체 종목, 시총 보정용)
    splits_cache = cache_dir / 'splits_history.pkl'
    splits_map = fetch_splits(all_members, splits_cache)
    print(f'  [universe] 분할 이력 데이터: {len(splits_map)} 종목')

    # 6) 매년 universe 산정 (분할 보정 시총 사용)
    all_universes = []
    for year in years_list:
        df = get_universe_top50_with_fallback(
            year, prices, shares_map, membership, splits_map=splits_map,
            n_top=n_top, max_candidates=max_candidates, min_data_days=min_data_days,
        )
        all_universes.append(df)
        if len(df) > 0:
            print(f'  [universe] {year}: {len(df)} 종목 산정 완료 (cutoff={df["cutoff_date"].iloc[0].date()})')
        else:
            print(f'  [universe] {year}: ⚠️ 유효 종목 0개 — shares 캐시 또는 데이터 부족 확인 필요')

    history = pd.concat(all_universes, ignore_index=True)

    # ⭐ 안전망 2: 최종 history 에 secondary ticker 가 있으면 강제 제거
    # (정상 흐름에선 발생 안 하지만 import 캐시 corner case 대비)
    secondary_in_history = history['ticker'].isin(list(DUAL_LISTED_MAP.keys()))
    if secondary_in_history.any():
        n_removed = secondary_in_history.sum()
        print(f'  [universe] WARN: 최종 history 에 secondary ticker {n_removed}행 발견 → 강제 제거')
        history = history[~secondary_in_history].reset_index(drop=True)

    # 6) 신규 편입 (is_new) 표시
    history['is_new'] = False
    prev_universe = set()
    for year in years_list:
        cur_universe = set(history.loc[history['oos_year'] == year, 'ticker'])
        new_tickers = cur_universe - prev_universe
        history.loc[
            (history['oos_year'] == year) & (history['ticker'].isin(new_tickers)),
            'is_new'
        ] = True
        prev_universe = cur_universe

    return history
