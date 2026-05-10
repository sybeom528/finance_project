"""
build_ticker_company_map.py — yfinance 로 ticker → 회사명 매핑 1회 수집

목적:
  Holdings / Sector Watch 페이지에서 ticker 옆에 회사명을 표시하기 위한
  매핑 CSV (ticker, company_name) 를 1회 생성.

  yfinance API 호출은 Rate limit 이 있으므로 매번 호출하지 않고
  결과를 CSV 캐시로 저장 (D-2 결정).

산출물:
  streamlit_dashboard/data/ticker_company_map.csv
    - 컬럼: ticker, company_name
    - 매핑 실패 시 company_name = ticker (fallback)

실행 방법:
  python streamlit_dashboard/scripts/build_ticker_company_map.py

  - 1회만 실행하면 됨 (CSV 가 git 에 포함되어 다른 환경에서 재실행 불필요)
  - 강제 갱신 원할 시 기존 CSV 삭제 후 재실행

실행 시간 예상:
  ~833 ticker × ~0.5초 (rate limit) = 약 7~10분
  (개별 호출 실패는 ticker 자체로 fallback 하고 계속 진행)

참조: docs/plan/01_setup.md 2.2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd


# 스크립트 위치 기준 경로 (cwd 무관)
SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

# 입력: universe.csv (대시보드 사본 우선, 없으면 final/ 원본)
LOCAL_UNIVERSE = DASHBOARD_DIR / "data" / "universe.csv"
ORIGINAL_UNIVERSE = PROJECT_ROOT / "final" / "data" / "universe.csv"

# 출력: ticker_company_map.csv
OUTPUT_PATH = DASHBOARD_DIR / "data" / "ticker_company_map.csv"

# yfinance rate limit 회피용 sleep (초). 너무 빠르면 차단됨.
SLEEP_BETWEEN_CALLS = 0.3

# 진행 로그 출력 빈도 (N 개마다 1회)
PROGRESS_INTERVAL = 50


def get_universe_path() -> Path:
    """대시보드 사본이 있으면 우선 사용 (copy_data.py 실행 후), 없으면 원본 직접 참조"""
    if LOCAL_UNIVERSE.exists():
        return LOCAL_UNIVERSE
    if ORIGINAL_UNIVERSE.exists():
        print(f"  [INFO] 대시보드 사본이 없어 원본 사용: {ORIGINAL_UNIVERSE}")
        print(f"         (copy_data.py 를 먼저 실행하면 사본을 사용합니다)")
        return ORIGINAL_UNIVERSE
    raise FileNotFoundError(
        f"universe.csv 를 찾을 수 없습니다.\n"
        f"  확인한 경로:\n  - {LOCAL_UNIVERSE}\n  - {ORIGINAL_UNIVERSE}"
    )


def load_existing_mapping() -> dict[str, str]:
    """기존 매핑 CSV 가 있으면 dict 로 로드 (재실행 시 이미 매핑된 ticker skip)"""
    if not OUTPUT_PATH.exists():
        return {}
    df = pd.read_csv(OUTPUT_PATH)
    return dict(zip(df["ticker"], df["company_name"]))


def fetch_company_name(ticker: str) -> str:
    """
    단일 ticker → 회사명. 실패 시 ticker 자체 반환 (fallback).

    yfinance.Ticker(t).info 는 dict 반환:
      - "longName": "Apple Inc." (선호)
      - "shortName": "Apple Inc." (대안)
      - 둘 다 없으면 ticker 자체
    """
    try:
        # 지연 import — yfinance 미설치 시 스크립트 자체는 import 가능하게
        import yfinance as yf

        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception as e:
        print(f"    [WARN] {ticker}: {type(e).__name__} - {str(e)[:60]}")
        return ticker  # fallback


def save_mapping(mapping: dict[str, str]) -> None:
    """dict → DataFrame → CSV (정렬된 ticker 순)"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        sorted(mapping.items()),
        columns=["ticker", "company_name"],
    )
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  [OK]   저장: {OUTPUT_PATH} ({len(df)} rows)")


def main() -> int:
    print("=" * 72)
    print("Adaptive VolControl Fund - ticker -> 회사명 매핑 수집")
    print("=" * 72)

    universe_path = get_universe_path()
    universe_df = pd.read_csv(universe_path)

    # universe.csv 의 ticker 컬럼명 추측 (ticker 또는 Ticker)
    ticker_col = "ticker" if "ticker" in universe_df.columns else "Ticker"
    tickers = sorted(universe_df[ticker_col].dropna().unique().tolist())

    existing = load_existing_mapping()
    print(f"  대상 ticker 수:    {len(tickers)}")
    print(f"  기존 매핑 (skip):   {len(existing)}")
    print(f"  신규 수집 대상:    {len(tickers) - len(set(tickers) & set(existing))}")
    print(f"  예상 시간:         약 {(len(tickers) - len(existing)) * SLEEP_BETWEEN_CALLS / 60:.1f}분")
    print()

    mapping = dict(existing)  # 기존 결과 유지

    for idx, ticker in enumerate(tickers, start=1):
        if ticker in mapping:
            continue  # 이미 수집된 ticker 는 skip

        company_name = fetch_company_name(ticker)
        mapping[ticker] = company_name

        if idx % PROGRESS_INTERVAL == 0:
            print(f"  진행: {idx}/{len(tickers)} - 최근: {ticker} -> {company_name[:40]}")
            # 중간 저장 (크래시 대비)
            save_mapping(mapping)

        time.sleep(SLEEP_BETWEEN_CALLS)

    save_mapping(mapping)
    print()
    print("=" * 72)
    print(f"  완료: 총 {len(mapping)} 매핑 저장")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
