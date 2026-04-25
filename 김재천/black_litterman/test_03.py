"""
03_feature_build.ipynb 빠른 검증 스크립트
  - Section 0: 함수 정의
  - Section 1: 2개 샘플 티커로 build_panel 실행 + CSV 구조 검증
  - compute_returns / compute_momentum / compute_volatility 단위 검증
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
PANELS_DIR  = DATA_DIR / "panels"
TEST_DIR    = DATA_DIR / "_test03"

for d in [DATA_DIR, CACHE_DIR, PANELS_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PRICE_START = "2010-01-01"
PRICE_END   = "2025-12-31"


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


def compute_returns(adj_close: pd.Series) -> pd.DataFrame:
    log_ret    = np.log(adj_close / adj_close.shift(1))
    simple_ret = adj_close.pct_change()
    return pd.DataFrame({
        "log_return_1d":    log_ret,
        "simple_return_1d": simple_ret,
    })


def compute_momentum(simple_ret: pd.Series) -> pd.DataFrame:
    r = 1 + simple_ret
    prod_252 = r.rolling(252).apply(np.prod, raw=True)
    prod_21  = r.rolling(21).apply(np.prod, raw=True)
    return pd.DataFrame({
        "mom_1m":          r.rolling(21).apply(np.prod, raw=True)  - 1,
        "mom_3m":          r.rolling(63).apply(np.prod, raw=True)  - 1,
        "mom_6m":          r.rolling(126).apply(np.prod, raw=True) - 1,
        "mom_12m":         prod_252 - 1,
        "mom_12m_skip_1m": prod_252 / prod_21 - 1,
    })


def compute_volatility(log_ret: pd.Series) -> pd.DataFrame:
    ann = np.sqrt(252)
    return pd.DataFrame({
        "vol_20d_ann":  log_ret.rolling(20).std()  * ann,
        "vol_60d_ann":  log_ret.rolling(60).std()  * ann,
        "vol_252d_ann": log_ret.rolling(252).std() * ann,
    })


def build_panel(ticker, gics_sector, price_df, ff_df, shares=None):
    df = price_df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_", regex=False)
    )
    rename = {
        "adj_close": "adj_close",
        "adjclose":  "adj_close",
        "stock_splits": "split_ratio",
    }
    df = df.rename(columns=rename)

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    if "split_ratio" not in df.columns:
        df["split_ratio"] = 0.0
    if "dividends" not in df.columns:
        df["dividends"] = 0.0

    ret = compute_returns(df["adj_close"])
    df["market_cap"] = (df["adj_close"] * shares) if shares else np.nan

    ff_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom_factor"]
    existing_ff = [c for c in ff_cols if c in ff_df.columns]
    df = df.join(ff_df[existing_ff], how="left")

    mom = compute_momentum(ret["simple_return_1d"])
    vol = compute_volatility(ret["log_return_1d"])

    df = df.join(ret, how="left")
    df = df.join(mom, how="left")
    df = df.join(vol, how="left")

    df["ticker"]      = ticker
    df["gics_sector"] = gics_sector

    ordered_cols = [
        "ticker", "gics_sector",
        "open", "high", "low", "close", "adj_close", "volume",
        "dividends", "split_ratio",
        "log_return_1d", "simple_return_1d", "market_cap",
        "mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom_factor",
        "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12m_skip_1m",
        "vol_20d_ann", "vol_60d_ann", "vol_252d_ann",
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[ordered_cols]


# FF 팩터 다운로드 (test_02에서 동일 함수 검증됨)
def download_ff_zip(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = zf.namelist()[0]
        raw = zf.read(csv_name).decode("utf-8", errors="ignore")
    lines = raw.splitlines()
    start_idx = next(i for i, ln in enumerate(lines) if re.match(r"^\s*\d{8}\s*,", ln))
    end_idx = next(
        (i for i in range(start_idx, len(lines)) if not re.match(r"^\s*\d{8}\s*,", lines[i])),
        len(lines)
    )
    data_block = "\n".join(lines[start_idx - 1 : end_idx])
    df = pd.read_csv(io.StringIO(data_block))
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col].astype(int).astype(str), format="%Y%m%d")
    df = df.rename(columns={date_col: "date"}).set_index("date")
    return df.astype(float) / 100.0


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 0: 함수 정의 완료")
print("=" * 60)
print("SECTION 0 PASS [OK]")


# ══════════════════════════════════════════════════════════════════════════
# 사전 준비: FF 팩터 + 2개 티커 가격 준비
# ══════════════════════════════════════════════════════════════════════════
print("\n[사전 준비] 테스트 데이터 준비 중...")

# FF 팩터 로드 (이미 있으면 재사용, 없으면 다운로드)
ff_main_path = DATA_DIR / "ff_factors.csv"
ff_test_path = TEST_DIR / "ff_factors.csv"

if ff_main_path.exists():
    df_ff = pd.read_csv(ff_main_path, index_col="date", parse_dates=True)
    print(f"  FF 팩터: {ff_main_path.name} 로드 ({len(df_ff)}행)")
elif ff_test_path.exists():
    df_ff = pd.read_csv(ff_test_path, index_col="date", parse_dates=True)
    print(f"  FF 팩터: 테스트 캐시 로드 ({len(df_ff)}행)")
else:
    FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
    print("  FF5 다운로드 중...")
    df_ff5 = download_ff_zip(FF5_URL)
    df_ff5.columns = [c.strip() for c in df_ff5.columns]
    df_ff5 = df_ff5.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
                                      "RMW": "rmw", "CMA": "cma", "RF": "rf"})
    print("  MOM 다운로드 중...")
    df_mom = download_ff_zip(MOM_URL)
    df_mom.columns = [c.strip() for c in df_mom.columns]
    if "Mom" in df_mom.columns:
        df_mom = df_mom.rename(columns={"Mom": "mom_factor"})
    elif "MOM" in df_mom.columns:
        df_mom = df_mom.rename(columns={"MOM": "mom_factor"})
    elif len(df_mom.columns) >= 1:
        df_mom = df_mom.rename(columns={df_mom.columns[0]: "mom_factor"})
    df_ff = df_ff5.join(df_mom[["mom_factor"]], how="inner")
    df_ff = df_ff[(df_ff.index >= pd.Timestamp(PRICE_START)) & (df_ff.index <= pd.Timestamp(PRICE_END))]
    df_ff.to_csv(ff_test_path, encoding="utf-8-sig")
    print(f"  FF 팩터: 다운로드 완료 ({len(df_ff)}행)")

# 2개 샘플 티커 가격 준비 (AAPL, MSFT)
SAMPLE_TICKERS = [("AAPL", "Information Technology"), ("MSFT", "Information Technology")]
sample_prices = {}

for ticker, sector in SAMPLE_TICKERS:
    # 실제 캐시 우선, 없으면 다운로드
    cache_path = CACHE_DIR / f"{ticker}.pkl"
    test_cache = TEST_DIR / f"{ticker}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            df_p = pickle.load(f)
        print(f"  {ticker}: 실제 캐시 로드 ({len(df_p)}행)")
    elif test_cache.exists():
        with open(test_cache, "rb") as f:
            df_p = pickle.load(f)
        print(f"  {ticker}: 테스트 캐시 로드 ({len(df_p)}행)")
    else:
        print(f"  {ticker}: 다운로드 중...")
        tk = yf.Ticker(ticker)
        df_p = tk.history(start=PRICE_START, end=PRICE_END, auto_adjust=False, actions=True)
        if df_p.index.tz is not None:
            df_p.index = df_p.index.tz_localize(None)
        df_p.index.name = "date"
        with open(test_cache, "wb") as f:
            pickle.dump(df_p, f)
        print(f"  {ticker}: 다운로드 완료 ({len(df_p)}행)")
        time.sleep(0.5)

    sample_prices[ticker] = df_p


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: compute 함수 단위 검증
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 1-A: compute 함수 단위 검증")
print("=" * 60)

df_test_price = sample_prices["AAPL"].copy()
df_test_price.columns = df_test_price.columns.str.strip().str.lower().str.replace(" ", "_")
if "adj_close" not in df_test_price.columns and "adjclose" in df_test_price.columns:
    df_test_price = df_test_price.rename(columns={"adjclose": "adj_close"})
adj = df_test_price["adj_close"]

ret_df = compute_returns(adj)
assert set(ret_df.columns) == {"log_return_1d", "simple_return_1d"}, "수익률 컬럼 오류"
assert len(ret_df) == len(adj), "수익률 길이 불일치"
# 첫 행은 NaN이어야 함
assert ret_df["log_return_1d"].iloc[0] is np.nan or pd.isna(ret_df["log_return_1d"].iloc[0])
print("  compute_returns: PASS")

mom_df = compute_momentum(ret_df["simple_return_1d"])
assert set(mom_df.columns) == {"mom_1m","mom_3m","mom_6m","mom_12m","mom_12m_skip_1m"}, "모멘텀 컬럼 오류"
# 앞부분은 NaN이어야 함 (252일 미만)
assert mom_df["mom_12m"].iloc[:251].isna().all(), "mom_12m 앞부분 NaN이어야 함"
# 뒷부분 (252일 이후)은 유효한 값이어야 함
valid_mom = mom_df["mom_12m"].dropna()
assert len(valid_mom) > 0, "mom_12m 유효 값 없음"
assert valid_mom.abs().max() < 10, f"mom_12m 값 범위 이상: {valid_mom.abs().max():.2f}"
print("  compute_momentum: PASS")

vol_df = compute_volatility(ret_df["log_return_1d"])
assert set(vol_df.columns) == {"vol_20d_ann","vol_60d_ann","vol_252d_ann"}, "변동성 컬럼 오류"
valid_vol = vol_df["vol_20d_ann"].dropna()
assert len(valid_vol) > 0, "vol_20d_ann 유효 값 없음"
assert (valid_vol > 0).all(), "변동성 음수 있음"
assert valid_vol.max() < 5.0, f"연환산 변동성 500% 초과: {valid_vol.max():.2f}"
print("  compute_volatility: PASS")

print("SECTION 1-A PASS [OK]")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1-B: build_panel + CSV 구조 검증
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 1-B: build_panel + CSV 구조 검증")
print("=" * 60)

EXPECTED_COLS = [
    "ticker", "gics_sector",
    "open", "high", "low", "close", "adj_close", "volume",
    "dividends", "split_ratio",
    "log_return_1d", "simple_return_1d", "market_cap",
    "mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom_factor",
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12m_skip_1m",
    "vol_20d_ann", "vol_60d_ann", "vol_252d_ann",
]

for ticker, sector in SAMPLE_TICKERS:
    df_price = sample_prices[ticker]

    # 주식수 조회 (오류 무시)
    shares = None
    try:
        info = yf.Ticker(ticker).info
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if shares:
            shares = float(shares)
    except Exception:
        pass

    # build_panel 실행
    panel = build_panel(ticker, sector, df_price, df_ff, shares)

    # 구조 검증
    assert list(panel.columns) == EXPECTED_COLS, (
        f"{ticker}: 컬럼 불일치\n  기대: {EXPECTED_COLS}\n  실제: {list(panel.columns)}"
    )
    assert len(panel) >= 1000, f"{ticker}: 행 수 부족 ({len(panel)})"
    assert (panel["ticker"] == ticker).all(), f"{ticker}: ticker 컬럼 오류"
    assert (panel["gics_sector"] == sector).all(), f"{ticker}: gics_sector 오류"

    # NaN 없어야 하는 컬럼 검증 (가격 컬럼)
    price_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in price_cols:
        nan_count = panel[col].isna().sum()
        assert nan_count < len(panel) * 0.05, f"{ticker}.{col}: NaN {nan_count}/{len(panel)}"

    # CSV 저장 + 로드 검증
    test_csv = TEST_DIR / f"{ticker}_panel_test.csv"
    panel.to_csv(test_csv, index=True, encoding="utf-8-sig")
    loaded = pd.read_csv(test_csv, index_col="date", parse_dates=True)
    assert len(loaded) == len(panel), f"{ticker}: CSV 저장/로드 행 수 불일치"
    assert set(loaded.columns) == set(EXPECTED_COLS), f"{ticker}: 로드 후 컬럼 불일치"
    test_csv.unlink()

    ff_nan = panel[["mkt_rf", "smb"]].isna().sum().sum()
    print(f"  {ticker}: {len(panel)}행, "
          f"FF NaN={ff_nan}, "
          f"mom_12m NaN={panel['mom_12m'].isna().sum()}, "
          f"market_cap NaN={panel['market_cap'].isna().all()}")

print("SECTION 1-B PASS [OK]")

# 테스트 데이터 정리 (ff_factors.csv 는 재사용 가능하므로 유지)
import shutil
# TEST_DIR 내 pkl 파일만 정리 (CSV는 남겨둠)
for pkl_file in TEST_DIR.glob("*.pkl"):
    pkl_file.unlink()

print("\n" + "=" * 60)
print("ALL SECTIONS PASS — 03_feature_build.ipynb 검증 완료")
print("=" * 60)
