"""Phase 3 — Universe 확장 (2009~2025) + Panel 데이터 확장.

Phase 2 의 universe.py (years=2020~2025) 를 Phase 3 의 17 년 (2009~2025) 으로 확장.
서윤범의 99_baseline (2009-2025, 17 년) 와 fair 비교를 위해 OOS 2009 시작 표준.

핵심 차이
---------
[Phase 2]
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    cutoff_dates = [2019-12-31, ..., 2024-12-31]
    매년 50 종목 → unique 약 74 종목
    panel 시작: 2012-12-31

[Phase 3]
    years = [2009, 2010, ..., 2025]    ⭐ 서윤범 OOS 2009 시작과 일치
    cutoff_dates = [2008-12-31, ..., 2024-12-31]
    매년 50 종목 → unique 약 130-200 종목 (예상)
    panel 시작: 2003-12-31    ⭐ 5 년 IS 보장 (OOS 2009 - 5년 = 2004)

학술적 의미
---------
- 서윤범 BL TOP_50 (Sharpe 1.065, 2009-2025) 와 fair 비교
- Pyo & Lee (2018) KOSPI 결과 (+19%) 와 fair 비교
- 다양한 시기 (2009 GFC 회복, 2011 유럽 위기, 2015 차이나, COVID, AI) 평균
- 본 Phase 2 의 sampling bias 한계 극복

⚠️ 중요: panel 데이터도 함께 확장 필요
---------------------------------------
universe 만 확장으로는 부족 — daily_panel.csv 도 2003-12-31 ~ 2025-12-31 로 재구성 필요.
extend_panel_to_2009() 함수 사용.

사용 예시
---------
from scripts.universe_extended import extend_panel_to_2009, extend_universe
from scripts.setup import DATA_DIR

# Step 1: panel 확장 (yfinance 30-60 분)
panel_extended = extend_panel_to_2009(
    panel_start='2003-12-31',
    panel_end='2025-12-31',
    cache_dir=DATA_DIR,
)

# Step 2: universe 확장 (캐시 활용)
history_df = extend_universe(
    start_year=2009,
    end_year=2025,
    n_top=50,
    cache_dir=DATA_DIR,
)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .universe import build_universe_history, get_or_build_membership, DUAL_LISTED_MAP


def build_full_universe_for_panel(
    start_year: int = 2009,
    end_year: int = 2025,
    cache_dir: Optional[Path] = None,
    out_name: str = 'universe_full_history.csv',
    verbose: bool = True,
) -> pd.DataFrame:
    """⭐ 전체 S&P 500 멤버 universe (top-50 필터 X) — Phase 3 최종.

    Membership cache 에서 OOS 기간 (start_year ~ end_year) 의 모든 unique 멤버를
    추출하여 flat CSV 로 저장. 서윤범 baseline 의 624 종목과 일관.

    Parameters
    ----------
    start_year : int, default 2009
    end_year : int, default 2025
    cache_dir : Path
        membership cache (sp500_membership.pkl) 디렉토리.
    out_name : str
        저장할 CSV 파일명.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        cols: oos_year, ticker, mcap_rank, mcap_value, cutoff_date, is_new
        - 모든 종목이 같은 oos_year, cutoff_date 사용 (panel 빌드 용)
        - 실제 portfolio 시점별 멤버십은 membership cache 직접 조회

    Notes
    -----
    - top-50 필터 미적용 → 624 종목 (서윤범과 일관)
    - oos_year = start_year, cutoff_date = (start_year-1)-12-31
      → build_daily_panel 의 시점 범위가 earliest_cutoff - 7년 ~ end_year+1 로 결정됨
      → 결과: panel 데이터 2002-01-01 ~ 2026-01-01 (IS 5년 학습 가능)
    - DUAL_LISTED_MAP secondary (GOOG 등) 자동 제외
    """
    if cache_dir is None:
        from .setup import DATA_DIR
        cache_dir = DATA_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    membership_start = pd.Timestamp(f'{start_year}-01-01')
    membership_end = pd.Timestamp(f'{end_year + 1}-01-01')
    membership_cache = cache_dir / 'sp500_membership.pkl'
    membership = get_or_build_membership(membership_start, membership_end, membership_cache)

    # 모든 멤버 union
    all_members = set()
    for member_set in membership.values():
        all_members.update(member_set)

    # secondary 종목 (GOOG 등) 제외
    n_before = len(all_members)
    all_members = all_members - set(DUAL_LISTED_MAP.keys())
    if verbose and n_before != len(all_members):
        removed = sorted(set(DUAL_LISTED_MAP.keys()) & set(membership.values().__iter__().__next__() | set()))
        print(f'  [universe-full] DUAL_LISTED secondary 제외: {n_before} → {len(all_members)}')

    # Flat CSV 생성
    cutoff = pd.Timestamp(f'{start_year - 1}-12-31')
    rows = []
    for i, ticker in enumerate(sorted(all_members)):
        rows.append({
            'oos_year': start_year,
            'ticker': ticker,
            'mcap_rank': i + 1,
            'mcap_value': 0.0,
            'cutoff_date': cutoff,
            'is_new': False,
        })

    df = pd.DataFrame(rows)

    if verbose:
        print('=' * 60)
        print(f'  [universe-full] {start_year}~{end_year} 전체 멤버 universe')
        print('=' * 60)
        print(f'  unique 종목 수: {len(df)}')
        print(f'  멤버십 추적 기간: {membership_start.date()} ~ {membership_end.date()}')
        print(f'  cutoff_date (panel 시점 결정용): {cutoff.date()}')

    if cache_dir is not None:
        out_path = cache_dir / out_name
        df.to_csv(out_path, index=False)
        if verbose:
            print(f'  저장: {out_path}')

    return df


def extend_universe(
    start_year: int = 2009,
    end_year: int = 2025,
    n_top: int = 50,
    max_candidates: Optional[int] = None,
    min_data_days: int = 1260,
    cache_dir: Optional[Path] = None,
    out_name: str = 'universe_top50_history_extended.csv',
    verbose: bool = True,
) -> pd.DataFrame:
    """Universe 를 지정 기간 (start_year ~ end_year) 으로 확장.

    Parameters
    ----------
    start_year : int, default 2010
        OOS 시작 연도.
    end_year : int, default 2025
        OOS 종료 연도.
    n_top : int, default 50
        매년 시총 상위 N 종목.
    max_candidates : int, optional
        Fallback 시 검토할 후보 수. None 시 n_top * 1.6 (Phase 2 일관).
    min_data_days : int, default 1260 (5 년)
        데이터 가용성 검증 임계 (영업일).
    cache_dir : Path, optional
        Wikipedia 멤버십, shares 등 캐시 디렉토리.
    out_name : str, default 'universe_top50_history_extended.csv'
        결과 csv 파일명.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        universe history (cols: oos_year, ticker, mcap_value, ...).
        Phase 2 의 universe_top50_history.csv 와 동일 schema.

    Notes
    -----
    - Wikipedia 멤버십 + 발행주식수는 캐시되므로 첫 실행만 시간 소요.
    - 본 함수는 Phase 2 의 build_universe_history 를 wrapping.
    - cutoff_date = (year - 1)년 12월 마지막 거래일 → look-ahead 차단.

    예상 시간
    ---------
    - 첫 실행 (캐시 X): 30-60 분 (Wikipedia + yfinance API call)
    - 캐시 후: 1-2 분
    """
    if max_candidates is None:
        max_candidates = int(n_top * 1.6)    # Phase 2 일관

    years = list(range(start_year, end_year + 1))

    if verbose:
        print('=' * 60)
        print(f'  Universe 확장: {start_year}~{end_year} ({len(years)} 년)')
        print(f'  매년 top {n_top} → unique 종목 {n_top}*1.5~2 예상')
        print('=' * 60)

    history_df = build_universe_history(
        years=years,
        n_top=n_top,
        max_candidates=max_candidates,
        min_data_days=min_data_days,
        cache_dir=cache_dir,
    )

    # 저장
    if cache_dir is not None:
        out_path = cache_dir / out_name
        history_df.to_csv(out_path, index=False)
        if verbose:
            print(f'\n  저장: {out_path}')

    # 검증 출력
    if verbose:
        print()
        print('=' * 60)
        print('  Universe 확장 결과')
        print('=' * 60)
        print(f'  전체 행 수: {len(history_df)}')
        print(f'  Unique 종목 수: {history_df["ticker"].nunique()}')
        print()
        print('  연도별 종목 수:')
        for year in years:
            n = (history_df['oos_year'] == year).sum()
            print(f'    {year}: {n} 종목')

    return history_df


def diagnose_universe_coverage(
    universe_df: pd.DataFrame,
    panel_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """확장된 universe 의 데이터 가용성 진단.

    Parameters
    ----------
    universe_df : pd.DataFrame
        extend_universe() 결과.
    panel_csv : Path, optional
        daily_panel.csv 경로. 제공 시 panel 기준 가용성 검증.

    Returns
    -------
    pd.DataFrame
        종목별 가용성 정보 (cols: ticker, n_years_in_universe, first_year, last_year).
    """
    coverage = universe_df.groupby('ticker').agg(
        n_years_in_universe=('oos_year', 'count'),
        first_year=('oos_year', 'min'),
        last_year=('oos_year', 'max'),
    ).reset_index()

    coverage['span_years'] = coverage['last_year'] - coverage['first_year'] + 1

    if panel_csv is not None:
        panel = pd.read_csv(panel_csv, parse_dates=['date'], usecols=['date', 'ticker'])
        panel_dates = panel.groupby('ticker').agg(
            first_panel_date=('date', 'min'),
            last_panel_date=('date', 'max'),
            n_panel_days=('date', 'count'),
        ).reset_index()
        coverage = coverage.merge(panel_dates, on='ticker', how='left')

    coverage = coverage.sort_values('n_years_in_universe', ascending=False).reset_index(drop=True)
    return coverage


def extend_panel_to_2009(
    panel_start: str = '2003-12-31',
    panel_end: str = '2025-12-31',
    cache_dir: Optional[Path] = None,
    universe_df: Optional[pd.DataFrame] = None,
    universe_file: str = 'universe_top50_history_extended.csv',
    out_name: str = 'daily_panel_extended.csv',
    verbose: bool = True,
) -> pd.DataFrame:
    """Panel 데이터를 2009 OOS 시작용으로 확장.

    Phase 2 의 daily_panel.csv (2012-12 ~ 2025-12) 를
    Phase 3 의 17 년 OOS 환경 (2009 ~ 2025) 에 맞게 재구성.

    Parameters
    ----------
    panel_start : str, default '2003-12-31'
        Panel 시작 일자. OOS 2009 시작 - IS 5 년 = 2004 → 2003-12-31 권장.
    panel_end : str, default '2025-12-31'
    cache_dir : Path, optional
        캐시 + 결과 저장 디렉토리.
    universe_df : pd.DataFrame, optional
        확장된 universe (extend_universe 결과). None 시 자동 호출.
    out_name : str, default 'daily_panel_extended.csv'

    Returns
    -------
    pd.DataFrame
        확장된 panel (Phase 2 의 daily_panel.csv 와 동일 schema).

    Notes
    -----
    - yfinance 다운로드 (30-60 분 첫 실행).
    - Phase 2 의 캐시 (universe.csv, shares_outstanding.pkl, sp500_membership.pkl) 재사용.
    - 누락 종목 (예: 2003 시점 미상장) 은 데이터 시작점부터 자동 처리.

    Wraps
    -----
    Phase 2 의 scripts/data_collection.py 의 build_daily_panel 함수.
    build_daily_panel 은 universe_csv 의 earliest cutoff 에서 자동으로 7 년 전 데이터 다운로드.

    Notes
    -----
    실제로는 build_daily_panel 의 시점 범위 자동 결정:
        start = earliest_cutoff - 7년
        end = max oos_year + 1년

    예시 (2009 OOS 시작):
        earliest_cutoff = 2008-12-31
        start = 2001-12-31 (자동)
        end = 2026-01-01 (자동)
    → panel_start, panel_end 인자는 참고용, 실제 사용 X.
    """
    from .data_collection import build_daily_panel

    if cache_dir is None:
        from .setup import DATA_DIR
        cache_dir = DATA_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print('=' * 60)
        print(f'  Panel 확장 (Phase 3, 2009 OOS 시작)')
        print(f'  요청 기간 (참고): {panel_start} ~ {panel_end}')
        print(f'  실제 기간: build_daily_panel 자동 결정 (earliest cutoff - 7년)')
        print('=' * 60)

    # universe 가용 확인 (universe_file 인자로 파일명 변경 가능)
    universe_path = cache_dir / universe_file
    if universe_df is None:
        if not universe_path.exists():
            raise FileNotFoundError(
                f'{universe_file} 없음. '
                f'먼저 extend_universe() 또는 build_full_universe_for_panel() 호출 필요.'
            )
    else:
        # universe_df 가 직접 전달된 경우 csv 저장
        if not universe_path.exists():
            universe_df.to_csv(universe_path, index=False)

    if verbose:
        if universe_df is None:
            universe_df = pd.read_csv(universe_path, parse_dates=['cutoff_date'])
        print(f'  universe: {universe_df["ticker"].nunique()} unique 종목')
        print(f'  cutoff range: {universe_df["cutoff_date"].min().date()} ~ {universe_df["cutoff_date"].max().date()}')

    # build_daily_panel 호출 (Phase 2 시그니처 준수)
    if verbose:
        print(f'  daily_panel 빌드 중... (yfinance API call, 첫 실행 30-60 분)')

    panel_df = build_daily_panel(
        universe_csv=universe_path,
        cache_dir=cache_dir,
        sector_map=None,
        overwrite=False,    # 캐시 활용
    )

    # ⭐ 캐시는 'daily_panel.csv' 로 저장됨. 본 Phase 3 의 out_name 으로 복사.
    if out_name != 'daily_panel.csv':
        default_path = cache_dir / 'daily_panel.csv'
        out_path = cache_dir / out_name
        if default_path.exists() and default_path != out_path:
            import shutil
            shutil.copy(default_path, out_path)
            if verbose:
                print(f'  복사: {default_path.name} → {out_name}')
    else:
        out_path = cache_dir / out_name

    if verbose:
        print(f'\n  저장: {out_path}')
        print(f'  shape: {panel_df.shape}')
        print(f'  기간: {panel_df["date"].min()} ~ {panel_df["date"].max()}')
        print(f'  unique tickers: {panel_df["ticker"].nunique()}')

    return panel_df


def diagnose_panel_coverage(
    panel_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    target_oos_start: str = '2009-01-01',
    is_len_days: int = 1260,
) -> pd.DataFrame:
    """Panel 의 종목별 데이터 가용성 진단.

    OOS start 시점에서 IS 5 년 검증 통과 가능한 종목 식별.

    Parameters
    ----------
    panel_df : pd.DataFrame
        daily_panel (long format).
    universe_df : pd.DataFrame
        universe history.
    target_oos_start : str, default '2009-01-01'
        OOS 시작 시점. 본 시점에서 IS 5 년 가용 검증.
    is_len_days : int, default 1260
        IS 길이 (영업일).

    Returns
    -------
    pd.DataFrame
        종목별 가용성 (cols: ticker, n_panel_days, first_panel_date,
        is_5y_ok, oos_2009_ok).
    """
    target_oos_start_ts = pd.Timestamp(target_oos_start)
    is_start_estimate = target_oos_start_ts - pd.Timedelta(days=int(is_len_days * 365.25 / 252))

    coverage_list = []
    for ticker in universe_df['ticker'].unique():
        panel_t = panel_df[panel_df['ticker'] == ticker]
        if len(panel_t) == 0:
            coverage_list.append({
                'ticker': ticker,
                'n_panel_days': 0,
                'first_panel_date': None,
                'is_5y_ok': False,
                'oos_2009_ok': False,
            })
            continue

        first_date = pd.Timestamp(panel_t['date'].min())
        coverage_list.append({
            'ticker': ticker,
            'n_panel_days': len(panel_t),
            'first_panel_date': first_date,
            'is_5y_ok': first_date <= is_start_estimate,
            'oos_2009_ok': first_date <= target_oos_start_ts - pd.Timedelta(days=int(is_len_days * 365.25 / 252)),
        })

    coverage = pd.DataFrame(coverage_list)
    return coverage


def split_universe_by_period(
    universe_df: pd.DataFrame,
    pre_start: int = 2009,
    pre_end: int = 2019,
    post_start: int = 2020,
    post_end: int = 2025,
) -> tuple:
    """Universe 를 두 시기로 분할 (검증용).

    Phase 2 (2018-2025) vs Phase 3 추가 (2010-2017) 분리.

    Returns
    -------
    (pre_df, post_df) : tuple of pd.DataFrame
        pre_df: pre_start ~ pre_end (Phase 3 신규)
        post_df: post_start ~ post_end (Phase 2 와 동일)
    """
    pre_df = universe_df[
        (universe_df['oos_year'] >= pre_start) & (universe_df['oos_year'] <= pre_end)
    ].copy()
    post_df = universe_df[
        (universe_df['oos_year'] >= post_start) & (universe_df['oos_year'] <= post_end)
    ].copy()
    return pre_df, post_df
