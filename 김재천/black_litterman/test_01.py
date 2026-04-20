"""
01_universe_selection.ipynb 빠른 검증 스크립트
  - Section 0~2 코드를 그대로 실행
  - Section 3(시가총액 추정)은 섹터당 3개 샘플 티커만 처리 (속도 우선)
  - Section 4~5 전체 실행 (n=3 샘플 기반)
  - 에러 없이 완료되고 최종 CSV 가 생성되면 PASS
"""
import io
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Optional

# Windows cp949 터미널에서 한글·특수문자 출력 깨짐 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # 테스트 환경에서 화면 없이 실행
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_DATE = pd.Timestamp("2016-01-04")

GICS_11_SECTORS = [
    "Energy", "Materials", "Industrials",
    "Consumer Discretionary", "Consumer Staples",
    "Health Care", "Financials",
    "Information Technology", "Communication Services",
    "Utilities", "Real Estate",
]


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


# ── 함수 정의 (노트북 셀 [03] 동일) ──────────────────────────────────────────
def resolve_gics_sector(wiki_sector: str) -> str:
    s = str(wiki_sector).strip()
    aliases = {
        "Telecommunication Services": "Communication Services",
        "Healthcare": "Health Care",
        "Information Tech.": "Information Technology",
    }
    return aliases.get(s, s)


def fetch_sp500_snapshot() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    raw = tables[0].copy()
    raw.columns = [c.strip() for c in raw.columns]
    col_map = {}
    for c in raw.columns:
        cl = c.lower()
        if cl in ("symbol", "ticker"):
            col_map[c] = "ticker"
        elif cl in ("security", "company"):
            col_map[c] = "company_name"
        elif cl.startswith("gics sector"):
            col_map[c] = "gics_sector"
        elif cl.startswith("gics sub"):
            col_map[c] = "gics_sub_industry"
    df = raw.rename(columns=col_map)
    df = df[["ticker", "company_name", "gics_sector", "gics_sub_industry"]].copy()
    df["gics_sector"] = df["gics_sector"].map(resolve_gics_sector)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
    return df.reset_index(drop=True)


# ── 함수 정의 (노트북 셀 [04] — fetch_spy_holdings) ──────────────────────────
def fetch_spy_holdings() -> tuple:
    url = (
        "https://www.ssga.com/us/en/intermediary/etfs/library-content/"
        "products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.ssga.com/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        df_raw = pd.read_excel(io.BytesIO(r.content), header=None)
        header_row = None
        for idx, row in df_raw.iterrows():
            vals_lower = [
                str(v).lower().strip() for v in row.values if pd.notna(v)
            ]
            if any(v in ("ticker", "symbol") for v in vals_lower):
                header_row = idx
                break
        if header_row is None:
            return None, f"헤더 행 탐지 실패: {df_raw.iloc[:6].values.tolist()}"
        df_raw.columns = [str(c).strip() for c in df_raw.iloc[header_row]]
        df = df_raw.iloc[header_row + 1:].reset_index(drop=True).dropna(how="all")
        ticker_col = next(
            (c for c in df.columns if c.lower() in ("ticker", "symbol")), None
        )
        sector_col = next(
            (c for c in df.columns if "sector" in c.lower()), None
        )
        if not ticker_col or not sector_col:
            return None, f"필수 컬럼 없음: {df.columns.tolist()}"
        result = (
            df[[ticker_col, sector_col]]
            .rename(columns={ticker_col: "ticker", sector_col: "spy_sector"})
            .copy()
        )
        result["ticker"] = result["ticker"].astype(str).str.strip()
        result = result[
            result["ticker"].str.match(r"^[A-Z]", na=False)
        ].reset_index(drop=True)
        return result, None
    except Exception as e:
        return None, str(e)


# ── 함수 정의 (노트북 셀 [05] — estimate_historical_market_cap) ──────────────
def estimate_historical_market_cap(
    ticker: str,
    as_of_date: pd.Timestamp = ANCHOR_DATE,
    max_retry: int = 2,
) -> Optional[dict]:
    last_err = None
    for attempt in range(max_retry + 1):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(
                start=(as_of_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                end=(as_of_date + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                actions=False,
            )
            if hist.empty:
                return None
            hist_idx = hist.index.tz_localize(None) if hist.index.tz else hist.index
            hist = hist.copy()
            hist.index = hist_idx
            mask = hist.index >= as_of_date
            if not mask.any():
                return None
            close_idx = hist.index[mask][0]
            close_price = float(hist.loc[close_idx, "Close"])

            shares = None
            method = None
            try:
                sf = tk.get_shares_full(
                    start=(as_of_date - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                    end=(as_of_date + pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                )
                if sf is not None and len(sf) > 0:
                    sf_idx = sf.index.tz_localize(None) if sf.index.tz else sf.index
                    sf = pd.Series(sf.values, index=sf_idx)
                    nearest = sf.iloc[(sf.index - as_of_date).to_series().abs().argmin()]
                    shares = float(nearest)
                    method = "get_shares_full"
            except Exception:
                pass

            if shares is None or shares <= 0:
                info = tk.info or {}
                current_shares = info.get("sharesOutstanding") or info.get(
                    "impliedSharesOutstanding"
                )
                if not current_shares:
                    return {
                        "mcap": None, "shares": None, "close": close_price,
                        "close_date": close_idx.strftime("%Y-%m-%d"),
                        "method": None, "error": "no_shares_info",
                    }
                splits = tk.splits
                if splits is not None and len(splits) > 0:
                    sp_idx = splits.index.tz_localize(None) if splits.index.tz else splits.index
                    splits = pd.Series(splits.values, index=sp_idx)
                    post = splits[splits.index >= as_of_date]
                    cum_ratio = float(post.prod()) if len(post) > 0 else 1.0
                else:
                    cum_ratio = 1.0
                shares = float(current_shares) / cum_ratio
                method = "split_reversal"

            return {
                "mcap": close_price * shares,
                "shares": shares,
                "close": close_price,
                "close_date": close_idx.strftime("%Y-%m-%d"),
                "method": method,
                "error": None,
            }
        except Exception as e:
            last_err = str(e)
            if attempt < max_retry:
                time.sleep(2**attempt)
            else:
                return {
                    "mcap": None, "shares": None, "close": None,
                    "close_date": None, "method": None, "error": last_err,
                }


# ── 함수 정의 (노트북 셀 [06] — top_n_by_sector) ─────────────────────────────
def top_n_by_sector(universe_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    df = universe_df.dropna(subset=["mcap_estimate"]).copy()
    df = df.sort_values(
        ["gics_sector", "mcap_estimate", "ticker"],
        ascending=[True, False, True],
    )
    df["sector_rank"] = df.groupby("gics_sector").cumcount() + 1
    return df[df["sector_rank"] <= n].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Wikipedia 로드
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 1: Wikipedia S&P 500 로드")
print("="*60)
df_wiki = fetch_sp500_snapshot()
print(f"종목 수: {len(df_wiki)}")
print("섹터 분포:")
print(df_wiki["gics_sector"].value_counts().to_string())

unexpected = set(df_wiki["gics_sector"].unique()) - set(GICS_11_SECTORS)
missing    = set(GICS_11_SECTORS) - set(df_wiki["gics_sector"].unique())
assert len(unexpected) == 0, f"비표준 섹터 발견: {unexpected}"
assert len(missing)    == 0, f"누락 섹터: {missing}"
assert len(df_wiki) >= 490, f"종목 수 이상: {len(df_wiki)}"
print("SECTION 1 PASS [OK]")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1-1: SPY 교차검증
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 1-1: SPY 교차검증")
print("="*60)
spy_df, spy_err = fetch_spy_holdings()
if spy_err:
    print(f"[경고] SPY 다운로드 실패: {spy_err}")
    print("→ Wikipedia 단독으로 계속 진행합니다.")
else:
    wiki_tickers = set(df_wiki["ticker"])
    spy_tickers  = set(spy_df["ticker"])
    common = wiki_tickers & spy_tickers
    only_wiki = sorted(wiki_tickers - spy_tickers)
    only_spy  = sorted(spy_tickers - wiki_tickers)

    print(f"Wikipedia : {len(wiki_tickers)}개")
    print(f"SPY       : {len(spy_tickers)}개")
    print(f"교집합    : {len(common)}개  ({len(common)/max(len(wiki_tickers),1):.1%})")
    print(f"Wikipedia에만: {only_wiki}")
    print(f"SPY에만(최대20): {only_spy[:20]}")

    merged_cv = df_wiki.merge(spy_df, on="ticker", how="inner")
    spy_unique_sectors = sorted(merged_cv["spy_sector"].dropna().unique())
    print(f"SPY 섹터 고유값(최대 15개): {spy_unique_sectors[:15]}")

    # SPY 파일 섹터가 유효한 경우에만 섹터 일치율 검증
    spy_sectors_valid = [s for s in spy_unique_sectors if s.strip() not in ("", "-", "nan")]
    if spy_sectors_valid:
        merged_cv["sector_match"] = (
            merged_cv["gics_sector"].str.strip().str.lower()
            == merged_cv["spy_sector"].str.strip().str.lower()
        )
        match_rate = merged_cv["sector_match"].mean()
        print(f"섹터 일치율: {match_rate:.1%} ({merged_cv['sector_match'].sum()} / {len(merged_cv)})")
        mismatch = merged_cv[~merged_cv["sector_match"]][
            ["ticker", "company_name", "gics_sector", "spy_sector"]
        ]
        if len(mismatch) > 0:
            print(f"불일치 ({len(mismatch)}건) (ticker 목록만):")
            print(mismatch["ticker"].tolist())
        else:
            print("섹터 모두 일치 [OK]")
        assert match_rate >= 0.90, f"섹터 일치율 90% 미만: {match_rate:.1%}"
    else:
        print("[경고] SPY 섹터 정보 없음('-') — 섹터 일치율 검증 건너뜀 (교집합 커버리지만 검증)")

    assert len(common) / max(len(wiki_tickers), 1) >= 0.90, "교집합 90% 미만 — 데이터 불일치"
    print("SECTION 1-1 PASS [OK]")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: 티커 리네임 확인
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 2: 티커 리네임 확인")
print("="*60)
TICKER_RENAME_HISTORY = {
    "FB": "META", "VIAC": "PARA", "FISV": "FI", "SQ": "XYZ", "TWTR": None,
}
for old, new in TICKER_RENAME_HISTORY.items():
    if new and new in set(df_wiki["ticker"]):
        print(f"  확인: {old} → {new} (현재 S&P 500 포함)")
    elif new:
        print(f"  참고: {old} → {new} (현재 S&P 500 미포함)")
    else:
        print(f"  참고: {old} 비상장됨")
print("SECTION 2 PASS [OK]")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: 시가총액 추정 — 샘플 모드 (섹터당 3개, 33 티커)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 3: 시가총액 추정 (샘플 검증 — 섹터당 3개)")
print("="*60)

# 섹터당 3개씩 샘플 선정 (알파벳 앞쪽 티커 우선)
sample_tickers = (
    df_wiki.groupby("gics_sector")
    .apply(lambda g: g.sort_values("ticker").head(3))
    .reset_index(drop=True)
)
print(f"샘플 종목 수: {len(sample_tickers)} ({sample_tickers['gics_sector'].nunique()} 섹터)")

mcap_cache_path = DATA_DIR / "mcap_estimates_2016_01.pkl"
if mcap_cache_path.exists():
    with open(mcap_cache_path, "rb") as f:
        mcap_cache = pickle.load(f)
    print(f"기존 캐시 {len(mcap_cache)}건 로드")
else:
    mcap_cache = {}

todo = [t for t in sample_tickers["ticker"].tolist() if t not in mcap_cache]
print(f"추가 추정 필요: {len(todo)}개")

for i, ticker in enumerate(todo, 1):
    result = estimate_historical_market_cap(ticker, ANCHOR_DATE)
    mcap_cache[ticker] = result
    status = "OK" if (result and result.get("mcap")) else "FAIL"
    method = result.get("method", "?") if result else "none"
    print(f"  [{i:02d}/{len(todo)}] {ticker:8s} → {status} ({method})")
    time.sleep(0.2)

with open(mcap_cache_path, "wb") as f:
    pickle.dump(mcap_cache, f)
print(f"캐시 저장: {mcap_cache_path}")

# 결과 집계
records = []
for ticker, res in mcap_cache.items():
    if ticker not in set(sample_tickers["ticker"]):
        continue
    if res is None:
        records.append({"ticker": ticker, "mcap_estimate": np.nan,
                        "shares_estimate": np.nan, "close_2016_01": np.nan,
                        "close_date": None, "method": None, "error": "no_price"})
    else:
        records.append({"ticker": ticker,
                        "mcap_estimate": res.get("mcap"),
                        "shares_estimate": res.get("shares"),
                        "close_2016_01": res.get("close"),
                        "close_date": res.get("close_date"),
                        "method": res.get("method"),
                        "error": res.get("error")})

df_mcap = pd.DataFrame(records)
df_full = sample_tickers.merge(df_mcap, on="ticker", how="left")

n_ok = df_full["mcap_estimate"].notna().sum()
print(f"추정 성공: {n_ok} / {len(df_full)}")
print(df_full[["ticker", "gics_sector", "mcap_estimate", "method"]].to_string(index=False))

assert n_ok >= len(df_full) * 0.7, f"추정 성공률 70% 미만: {n_ok}/{len(df_full)}"
print("SECTION 3 PASS [OK]")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: 섹터별 Top N 선정 (샘플이므로 n=3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 4: 섹터별 Top 3 선정 (샘플 검증)")
print("="*60)
universe_sample = top_n_by_sector(df_full, n=3)
print(f"선정 결과: {len(universe_sample)}개")
print(universe_sample[["sector_rank","ticker","gics_sector","mcap_estimate"]].to_string(index=False))
assert len(universe_sample) > 0, "선정 결과 없음"
print("SECTION 4 PASS [OK]")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CSV 저장
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 5: CSV 저장")
print("="*60)
out_test = universe_sample[[
    "ticker", "gics_sector", "company_name", "sector_rank",
    "mcap_estimate", "shares_estimate", "close_2016_01", "close_date", "method"
]].copy()
out_test = out_test.rename(columns={"mcap_estimate": "mcap_estimate_2016_01"})
test_path = DATA_DIR / "universe_TEST.csv"
out_test.to_csv(test_path, index=False, encoding="utf-8-sig")
assert test_path.exists(), "CSV 파일 생성 실패"
loaded = pd.read_csv(test_path)
assert len(loaded) == len(out_test), "저장/로드 행 수 불일치"
expected_cols = {"ticker","gics_sector","company_name","sector_rank",
                 "mcap_estimate_2016_01","shares_estimate","close_2016_01","close_date","method"}
assert expected_cols.issubset(set(loaded.columns)), f"컬럼 누락: {expected_cols - set(loaded.columns)}"
# 테스트 파일 삭제
test_path.unlink()
print(f"CSV 구조 검증 완료 (테스트 파일 삭제)")
print("SECTION 5 PASS [OK]")

print("\n" + "="*60)
print("ALL SECTIONS PASS — 01_universe_selection.ipynb 검증 완료")
print("="*60)
