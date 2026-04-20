"""
02_data_collection.ipynb 빠른 검증 스크립트
  - Section 0: 함수 정의 및 import
  - Section 1: 가격 수집 — 3개 샘플 티커만
  - Section 2: FF5 + MOM 실제 다운로드
  - Section 3: 벤치마크 — 2개 샘플
  - Section 4: 요약 로그
"""
import io
import os
import pickle
import platform
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

# Windows cp949 출력 인코딩 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── 경로 설정 ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data"
CACHE_DIR   = DATA_DIR / "cache"
BENCH_DIR   = DATA_DIR / "benchmarks"
TEST_DIR    = DATA_DIR / "_test_cache"  # 테스트 전용 캐시 (종료 시 삭제)

for d in [DATA_DIR, CACHE_DIR, BENCH_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PRICE_START = "2010-01-01"
PRICE_END   = "2025-12-31"

FF_SAVE_PATH = DATA_DIR / "ff_factors.csv"


# ── 함수 정의 (노트북 cell-2 동일) ───────────────────────────────────────
def setup_korean_font():
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif platform.system() == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        try:
            import koreanize_matplotlib  # noqa: F401
        except ImportError:
            pass
    plt.rcParams["axes.unicode_minus"] = False


setup_korean_font()


def safe_download_price(
    ticker: str,
    start: str = PRICE_START,
    end: str = PRICE_END,
    max_retry: int = 5,
) -> Optional[pd.DataFrame]:
    for attempt in range(max_retry + 1):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
            )
            if df.empty:
                return None
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = "date"
            return df
        except Exception:
            if attempt < max_retry:
                time.sleep(2 ** attempt)
    return None


def download_ff_zip(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = zf.namelist()[0]
        raw = zf.read(csv_name).decode("utf-8", errors="ignore")

    lines = raw.splitlines()
    start_idx = next(
        i for i, ln in enumerate(lines)
        if re.match(r"^\s*\d{8}\s*,", ln)
    )
    end_idx = next(
        (i for i in range(start_idx, len(lines))
         if not re.match(r"^\s*\d{8}\s*,", lines[i])),
        len(lines)
    )
    data_block = "\n".join(lines[start_idx - 1 : end_idx])

    df = pd.read_csv(io.StringIO(data_block))
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(
        df[date_col].astype(int).astype(str), format="%Y%m%d"
    )
    df = df.rename(columns={date_col: "date"}).set_index("date")
    return df.astype(float) / 100.0


def download_benchmark(
    symbol: str,
    start: str = PRICE_START,
    end: str = PRICE_END,
) -> Optional[pd.DataFrame]:
    return safe_download_price(symbol, start, end, max_retry=3)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 0: 함수 정의 완료
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 0: 함수 정의 + 경로 설정")
print("=" * 60)
print(f"PROJECT_DIR : {PROJECT_DIR}")
print(f"DATA_DIR    : {DATA_DIR}")
print("SECTION 0 PASS [OK]")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: 가격 수집 (샘플 3개 — AAPL, MSFT, JPM)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 1: 가격 수집 (샘플 3 티커)")
print("=" * 60)

SAMPLE_TICKERS = ["AAPL", "MSFT", "JPM"]

for ticker in SAMPLE_TICKERS:
    test_cache = TEST_DIR / f"{ticker}.pkl"
    if test_cache.exists():
        with open(test_cache, "rb") as f:
            df_p = pickle.load(f)
        print(f"  {ticker}: 기존 테스트캐시 로드 ({len(df_p)}행)")
    else:
        df_p = safe_download_price(ticker, max_retry=3)
        assert df_p is not None and not df_p.empty, f"{ticker} 가격 수집 실패"
        with open(test_cache, "wb") as f:
            pickle.dump(df_p, f)
        print(f"  {ticker}: {len(df_p)}행, "
              f"{df_p.index[0].date()} ~ {df_p.index[-1].date()}")
        time.sleep(0.3)

    # 기본 품질 검증
    assert len(df_p) >= 1000, f"{ticker} 행 수 이상: {len(df_p)}"
    expected_cols_lower = {"close", "open", "high", "low", "volume"}
    actual_lower = {c.lower() for c in df_p.columns}
    missing = expected_cols_lower - actual_lower
    assert not missing, f"{ticker} 컬럼 누락: {missing}"

print("SECTION 1 PASS [OK]")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: FF5 + MOM 팩터 수집
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: FF5 + MOM 팩터 수집 (실제 다운로드)")
print("=" * 60)

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

TEST_FF_PATH = DATA_DIR / "ff_factors_TEST.csv"

print("  FF5 다운로드 중...")
df_ff5 = download_ff_zip(FF5_URL)
ff5_rename = {
    "Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
    "RMW": "rmw", "CMA": "cma", "RF": "rf",
}
df_ff5.columns = [c.strip() for c in df_ff5.columns]
df_ff5 = df_ff5.rename(columns={k: v for k, v in ff5_rename.items() if k in df_ff5.columns})
print(f"  FF5: {len(df_ff5)}행, 컬럼={df_ff5.columns.tolist()}")

print("  MOM 다운로드 중...")
df_mom = download_ff_zip(MOM_URL)
df_mom.columns = [c.strip() for c in df_mom.columns]
if "Mom" in df_mom.columns:
    df_mom = df_mom.rename(columns={"Mom": "mom_factor"})
elif "MOM" in df_mom.columns:
    df_mom = df_mom.rename(columns={"MOM": "mom_factor"})
elif len(df_mom.columns) >= 1:
    df_mom = df_mom.rename(columns={df_mom.columns[0]: "mom_factor"})
print(f"  MOM: {len(df_mom)}행, 컬럼={df_mom.columns.tolist()}")

df_ff = df_ff5.join(df_mom[["mom_factor"]], how="inner")
df_ff = df_ff[
    (df_ff.index >= pd.Timestamp(PRICE_START))
    & (df_ff.index <= pd.Timestamp(PRICE_END))
]

# 품질 검증
expected_ff_cols = {"mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom_factor"}
missing_ff = expected_ff_cols - set(df_ff.columns)
assert not missing_ff, f"FF 팩터 컬럼 누락: {missing_ff}"
assert len(df_ff) >= 3000, f"FF 팩터 행 수 이상: {len(df_ff)}"
assert df_ff["mkt_rf"].abs().max() < 0.2, "mkt_rf 값 범위 이상 (±20% 초과)"

# 테스트 CSV 저장 후 검증
df_ff.to_csv(TEST_FF_PATH, encoding="utf-8-sig")
loaded_ff = pd.read_csv(TEST_FF_PATH, index_col="date")
assert len(loaded_ff) == len(df_ff), "FF CSV 저장/로드 행 수 불일치"
TEST_FF_PATH.unlink()
print(f"  병합 결과: {len(df_ff)}행, 기간={df_ff.index[0].date()} ~ {df_ff.index[-1].date()}")
print("SECTION 2 PASS [OK]")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: 벤치마크 수집 (샘플 — ^GSPC, SPY)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: 벤치마크 수집 (샘플 2개)")
print("=" * 60)

BENCH_SAMPLE = {"^GSPC": "S&P 500 지수", "SPY": "SPDR S&P500 ETF"}

for symbol, desc in BENCH_SAMPLE.items():
    save_name = symbol.replace("^", "") + "_TEST.csv"
    save_path = BENCH_DIR / save_name

    df_bench = download_benchmark(symbol)
    assert df_bench is not None and not df_bench.empty, f"{symbol} 다운로드 실패"
    assert len(df_bench) >= 1000, f"{symbol} 행 수 이상: {len(df_bench)}"

    df_bench.to_csv(save_path, encoding="utf-8-sig")
    loaded_bench = pd.read_csv(save_path, index_col="date")
    assert len(loaded_bench) == len(df_bench), f"{symbol} CSV 저장/로드 행 수 불일치"
    save_path.unlink()  # 테스트 파일 삭제

    print(f"  {symbol:6s} ({desc}): {len(df_bench)}행, "
          f"{df_bench.index[0].date()} ~ {df_bench.index[-1].date()}")
    time.sleep(0.3)

print("SECTION 3 PASS [OK]")

# ══════════════════════════════════════════════════════════════════════════
# 테스트 캐시 정리
# ══════════════════════════════════════════════════════════════════════════
import shutil
shutil.rmtree(TEST_DIR, ignore_errors=True)

print("\n" + "=" * 60)
print("ALL SECTIONS PASS — 02_data_collection.ipynb 검증 완료")
print("=" * 60)
