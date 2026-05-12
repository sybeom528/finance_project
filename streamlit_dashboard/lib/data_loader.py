"""
lib/data_loader.py - 데이터 로딩 + Streamlit 캐싱 (D-3)

모든 페이지가 공유하는 데이터 로더.
@st.cache_data 표준 적용 → Streamlit 세션 동안 1회만 실제 IO 발생.

경로:
  - 모듈 위치 기준 절대 경로 사용 (cwd 무관 동작)
  - 핵심 데이터: streamlit_dashboard/data/ (사본)
  - Sensitivity Test: final/results/ (원본 직접 참조, 156 config 중 필요 시)

참조: docs/plan/02_common.md 2절, docs/plan/01_setup.md 2.3절
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# === 경로 (모듈 위치 기준, cwd 무관) ===================================
DASHBOARD_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DASHBOARD_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
PROJECT_ROOT = DASHBOARD_DIR.parent
ORIGINAL_RESULTS_DIR = PROJECT_ROOT / "final" / "results"


# === GICS 11 섹터 정합화 (sector name normalization) ==================
# universe.csv / monthly_panel.csv 모두 yfinance 와 GICS 표준이 혼재.
# 두 source 모두 GICS 11 표준으로 통일 (load 시점 자동 적용).
#
# GICS 11 표준 (2018 GICS revision 기준):
#   Information Technology / Health Care / Financials / Consumer Discretionary
#   / Industrials / Communication Services / Consumer Staples / Energy
#   / Utilities / Real Estate / Materials
#
# 대표 mismatch (yfinance / 옛 GICS) → GICS 11 표준 매핑:
GICS_SECTOR_NORMALIZATION: dict[str, str] = {
    # yfinance 형식 → GICS 표준
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Consumer Defensive": "Consumer Staples",
    "Consumer Cyclical": "Consumer Discretionary",
    "Financial Services": "Financials",
    "Basic Materials": "Materials",
    # 그 외 (이미 GICS 표준이거나 Unknown — passthrough)
}


def normalize_gics_sector(name) -> str:
    """단일 sector 명을 GICS 11 표준으로 정합화. 매핑 부재 시 원본 반환."""
    if not isinstance(name, str):
        return "Unknown"
    return GICS_SECTOR_NORMALIZATION.get(name, name)


# === 핵심 데이터 (대시보드 사본) =======================================

@st.cache_data
def load_monthly_panel() -> pd.DataFrame:
    """
    월별 패널 (date, ticker, rf, spy_ret, sector, log_mcap 등).

    gics_sector 컬럼은 load 시점에 GICS 11 표준으로 자동 정합화
    (yfinance 형식 Healthcare/Technology/Consumer Defensive 등 → 표준명).
    """
    df = pd.read_csv(DATA_DIR / "monthly_panel.csv", parse_dates=["date"])
    if "gics_sector" in df.columns:
        df["gics_sector"] = df["gics_sector"].map(normalize_gics_sector).fillna("Unknown")
    return df


@st.cache_data
def load_daily_returns() -> pd.DataFrame:
    """일별 수익률 DataFrame (index=date, columns=ticker)."""
    with open(DATA_DIR / "daily_returns.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_ff5_monthly() -> pd.DataFrame:
    """Fama-French 5-factor 월별 (date, mkt_rf, smb, hml, rmw, cma, rf)."""
    return pd.read_csv(DATA_DIR / "ff5_monthly.csv", parse_dates=["date"])


@st.cache_data
def load_universe() -> pd.DataFrame:
    """
    Universe 정의 (ticker, gics_sector).

    gics_sector 컬럼은 load 시점에 GICS 11 표준으로 자동 정합화.
    """
    df = pd.read_csv(DATA_DIR / "universe.csv")
    if "gics_sector" in df.columns:
        df["gics_sector"] = df["gics_sector"].map(normalize_gics_sector).fillna("Unknown")
    return df


@st.cache_data
def load_ticker_company_map() -> pd.DataFrame | None:
    """
    yfinance 회사명 매핑 (D-2). 파일 없으면 None.
    None 반환 시 호출 측에서 ticker 자체를 사용 (graceful degradation).

    후처리 (D-2 보완 결정 2026-05-10):
      yfinance.info 가 longName/shortName 을 반환하지 않고 CIK 등 숫자만
      반환하는 케이스 (인수합병된 옛 종목 — GR, MOLX, TLAB, TWX 등) 자동 보정.
      → company_name 이 숫자만이거나 빈값/NaN 이면 ticker 자체로 덮어쓰기.
      이 후처리는 매핑 CSV 가 어떤 환경에서 생성되었든 동일 결과 보장.
    """
    p = DATA_DIR / "ticker_company_map.csv"
    if not p.exists():
        return None

    df = pd.read_csv(p)

    # 후처리: 잘못된 매핑을 ticker 로 fallback
    def _is_invalid(name) -> bool:
        if not isinstance(name, str):
            return True  # NaN / None
        s = name.strip()
        if s == "":
            return True
        if s.isdigit():  # 숫자만 (CIK 등)
            return True
        return False

    invalid_mask = df["company_name"].apply(_is_invalid)
    df.loc[invalid_mask, "company_name"] = df.loc[invalid_mask, "ticker"]

    return df


@st.cache_data
def get_ticker_sector_map() -> dict[str, str]:
    """
    ticker → GICS 11 표준 sector dict.

    lookup 우선순위 (정확도 ↑):
      1. monthly_panel 의 ticker 별 최신 시점 gics_sector (실제 GICS)
      2. universe.csv 의 gics_sector (panel 부재 시 fallback)
      3. "Unknown" (둘 다 부재)

    universe 의 "Unknown" 205 ticker 도 panel 에 있으면 정확 sector 산출 가능.
    """
    panel = load_monthly_panel()
    universe = load_universe()

    # 1) panel — 각 ticker 의 latest 시점 gics_sector (이미 normalize 됨)
    panel_latest = (
        panel.dropna(subset=["gics_sector"])
        .sort_values("date")
        .groupby("ticker")["gics_sector"]
        .last()
        .to_dict()
    )

    # 2) universe — panel 에 없는 ticker 보강
    universe_map = dict(zip(universe["ticker"], universe["gics_sector"]))

    # 통합 (panel 우선)
    out: dict[str, str] = {}
    all_tickers = set(panel_latest) | set(universe_map)
    for t in all_tickers:
        sector = panel_latest.get(t) or universe_map.get(t) or "Unknown"
        out[t] = sector
    return out


@st.cache_data
def get_ticker_company_dict() -> dict[str, str]:
    """
    ticker → company_name dict. 매핑 파일 부재 시 빈 dict.
    호출 측 사용 패턴: `name = mapping.get(ticker, ticker)` (graceful degradation).

    `load_ticker_company_map()` 의 후처리 (숫자/빈값 → ticker fallback) 가 적용된 결과.
    """
    df = load_ticker_company_map()
    if df is None:
        return {}
    return dict(zip(df["ticker"], df["company_name"]))


@st.cache_data
def load_sp500_membership() -> dict:
    """
    S&P500 시점별 편입 종목 (look-ahead 회피용 universe).

    Returns:
        dict[Timestamp, frozenset[str]] — 월말 시점 → 그 시점 편입 ticker set

    decisionlog/02_overview.md Q-C 결정 (사용자 지적 2026-05-10):
      EW/IVW 의 ret[t] 계산 시 universe = sp500_membership[t-1] (look-ahead 회피).
    """
    with open(DATA_DIR / "sp500_membership.pkl", "rb") as f:
        return pickle.load(f)


# === 펀드 결과 ========================================================

def _fill_spy_ret_nan(spy_ret: pd.Series) -> tuple[pd.Series, list[pd.Timestamp]]:
    """
    spy_ret 의 NaN 시점을 일별 SPY 데이터로 보강.

    원본 산식 (`monthly_panel.fwd_ret_1m`):
        spy_ret[t] = (1 + simple_daily_spy[t+1 .. t+21]).prod() - 1
        (다음 21 영업일 산술 cumprod, panel 산식과 일치)

    NaN 발생 원인: monthly_panel 생성 시점에 다음 21 영업일 데이터가
    충분히 없었을 가능성. 현재 daily_returns.pkl 가 더 확장되어 있어 보강 가능.

    보강 방식:
      1) daily_returns.pkl 의 SPY 로그 수익률 → 산술 변환 (`exp(log) - 1`)
      2) NaN 시점 t 에 대해 다음 21 영업일 cumprod 산출
      3) 21 영업일 데이터 부족 시 → NaN 유지

    Args:
        spy_ret: forward-looking 인덱스 (월말) 의 SPY 월별 수익률 Series

    Returns:
        (filled_spy_ret, filled_dates): 보강된 Series + 실제 채워진 날짜 list
    """
    nan_mask = spy_ret.isna()
    if not nan_mask.any():
        return spy_ret, []

    # daily SPY 로드 (별도 — cache 효율을 위해 직접 pickle.load)
    daily_path = DATA_DIR / "daily_returns.pkl"
    if not daily_path.exists():
        return spy_ret, []
    with open(daily_path, "rb") as f:
        daily = pickle.load(f)
    if "SPY" not in daily.columns:
        return spy_ret, []

    # log → simple 변환 (panel 산식과 정합)
    spy_daily_simple = (np.exp(daily["SPY"]) - 1).dropna()

    filled = spy_ret.copy()
    filled_dates = []
    for t in spy_ret[nan_mask].index:
        after = spy_daily_simple[spy_daily_simple.index > t]
        if len(after) >= 21:
            next_21 = after.iloc[:21]
            cum = float((1 + next_21).prod() - 1)
            filled.loc[t] = cum
            filled_dates.append(t)

    return filled, filled_dates


@st.cache_data
def load_fund_results(
    config_name: str = "mat_eq_eq_raw_pap",
    tc_override: float | None = 0.002,
    fill_spy_nan: bool = True,
) -> dict:
    """
    펀드 backtest 결과 (최종 Top 1 = mat_eq_eq_raw_pap, 2026-05-11 최종 변경).

    config 차원: prior=capm_eq / p_weight=eq / q_mode=raw_lam / omega_mode=ff3_paper

    대시보드 사본 (Top 1) 우선, 없으면 final 원본 fallback.

    ── TC override (2026-05-12, 안전망 역할로 유지) ──────────────────────
    [도입 시점 narrative]
    원래 pkl 은 tc=0.001 (편측 10bp/side) 로 산출되었으나, 펀드의 의도된
    보수적 거래비용 가정은 편측 20bp (tc=0.002) 였습니다. 차후 의도가 변경
    되었으나 원본 backtest 는 재실행하지 않고, **대시보드 사용 시점에 net
    재산출** 하는 방식으로 의도값을 반영하도록 도입.

    [후속 변경 (2026-05-12, commit a1e7122)]
    팀원이 final 측에서 pkl 매트릭스 자체를 tc=0.002 로 통일 push. 이후
    pkl 의 `config["tc"]` 가 이미 0.002 이므로 본 함수의 `needs_recompute`
    체크가 False 가 되어 **재산출 skip → pkl.ret 그대로 사용**.
    본 로직은 **안전망**으로 유지 (다른 tc 값 검증 / 향후 의도 재변경 대응).

    재산출 산식 (needs_recompute = True 일 때만 동작):
        net_ret = gross_ret − turnover × tc_override
        (turnover 는 two-way Σ|Δw| ∈ [0, 2], tc 는 편측 rate)

    tc_override 파라미터 (현재 환경 = pkl tc=0.002 기준):
      - 0.002 (기본) — pkl 원본과 동일 → 재산출 skip (idempotent)
      - 0.001        — 편측 10bp 로 재산출 (검증/실험용)
      - None         — pkl 원본 그대로 사용 (디버깅용)

    재산출 시 `config["tc"]` 는 새 값으로 갱신되고, 원본 tc 는
    `config["tc_original_in_pkl"]` 로 보존됩니다.

    ── SPY NaN 보강 (2026-05-12) ──────────────────────────────────────────
    `monthly_panel.spy_ret` 는 `fwd_ret_1m` 패턴 (다음 21 영업일 cumprod) 으로
    산출되며, panel 생성 시점에 다음 21 영업일 데이터가 부족하면 NaN 발생.
    현재 daily_returns.pkl 가 충분히 확장되어 있어, NaN 시점의 SPY 를 일별 데이터로
    보강 가능 (panel 산식 정합 유지).

    fill_spy_nan 파라미터:
      - True (기본): NaN 인 SPY 시점을 일별 데이터로 보강
      - False: pkl 원본 그대로 (NaN 유지)

    보강 후 `config["spy_filled_dates"]` 에 채워진 날짜 list 저장.

    pkl 구조: {"weights": DataFrame, "ret": Series (재산출), "gross_ret": Series,
              "spy_ret": Series (NaN 보강), "comp": DataFrame, "config": dict (갱신), ...}
    """
    local = RESULTS_DIR / f"{config_name}.pkl"
    if local.exists():
        with open(local, "rb") as f:
            d = pickle.load(f)
    else:
        original = ORIGINAL_RESULTS_DIR / f"{config_name}.pkl"
        if not original.exists():
            raise FileNotFoundError(
                f"Config '{config_name}' pkl 을 찾을 수 없습니다.\n"
                f"  확인한 경로:\n  - {local}\n  - {original}"
            )
        with open(original, "rb") as f:
            d = pickle.load(f)

    # TC override — 의도된 보수적 가정 (편측 20bp) 으로 net 재산출
    if tc_override is not None:
        original_tc = d.get("config", {}).get("tc")
        # 원본과 다를 때만 재산출 (소수점 비교는 ε 허용)
        needs_recompute = (
            original_tc is None
            or abs(float(tc_override) - float(original_tc)) > 1e-9
        )
        if needs_recompute and "gross_ret" in d and "comp" in d:
            comp = d["comp"]
            if "turnover" in comp.columns:
                turnover = comp["turnover"]
                # net = gross - turnover × tc (apply_tc 산식과 동일)
                d["ret"] = d["gross_ret"] - turnover * float(tc_override)
                # config 갱신 — 사용자 표시 / 검증 일관성
                cfg = dict(d.get("config", {}))
                cfg["tc_original_in_pkl"] = original_tc
                cfg["tc"] = float(tc_override)
                d["config"] = cfg

    # SPY NaN 보강 (panel.spy_ret 의 boundary NaN → 일별 SPY 로 채움)
    if fill_spy_nan and "spy_ret" in d:
        filled_spy, filled_dates = _fill_spy_ret_nan(d["spy_ret"])
        d["spy_ret"] = filled_spy
        if filled_dates:
            # config 에 보강 기록 (사용자 추적용)
            cfg = dict(d.get("config", {}))
            cfg["spy_filled_dates"] = [t.strftime("%Y-%m-%d") for t in filled_dates]
            d["config"] = cfg

    return d


@st.cache_data
def load_other_config_results(config_name: str) -> dict:
    """
    다른 155 config 결과 — Backtesting 영역 6 / model-comparison 등.
    load_fund_results 와 동일한 fallback (사본 우선 → final 원본).
    """
    return load_fund_results(config_name)


@st.cache_data
def list_available_configs() -> list[str]:
    """
    streamlit_dashboard/data/results/ 의 모든 config pkl 이름 list.
    부재 시 final/results/ fallback. Backtesting 페이지 영역 6 sensitivity 사용.
    """
    if RESULTS_DIR.exists():
        names = sorted(p.stem for p in RESULTS_DIR.glob("*.pkl") if p.is_file())
        if names:
            return names
    if ORIGINAL_RESULTS_DIR.exists():
        return sorted(p.stem for p in ORIGINAL_RESULTS_DIR.glob("*.pkl") if p.is_file())
    return ["mat_eq_eq_raw_pap"]  # fallback


# === Baseline 산출 (Overview 영역 3 비교 라인) ========================
# 모두 시점 t-1 sp500_membership 기준 (look-ahead bias 회피, decisionlog Q-C)


def _resolve_universe_at(t_prev: pd.Timestamp, sp500_membership: dict) -> list[str]:
    """
    t_prev 시점 (또는 가장 가까운 직전 시점) S&P500 편입 종목 list.
    sp500_membership 의 key 가 t_prev 와 정확히 일치하지 않으면 직전 가장 가까운 시점 사용.
    """
    sp_dates = sorted(sp500_membership.keys())
    prior = [d for d in sp_dates if d <= t_prev]
    if not prior:
        return []
    return list(sp500_membership[max(prior)])


@st.cache_data
def compute_equal_weight_returns(
    _monthly_panel: pd.DataFrame,
    _sp500_membership: dict,
    _fund_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    EW baseline — 시점 t-1 S&P500 universe 의 ret_1m 동일가중 평균 (옵션 E).

    학술 표준 (decisionlog Q-C + Q-D):
      1) 데이터: monthly_panel (daily_returns 결함 회피, fund.spy_ret 와 일관성 검증)
      2) Universe: sp500_membership[t-1] (look-ahead 회피)
      3) ret[t] = monthly_panel[date=t & ticker in universe].ret_1m.mean()
         (월별 단위라 daily drift 표현은 단순화 — Σ(w_i × r_i) 로 단순 평균)

    Args (인자명에 underscore = streamlit cache hashing 제외):
        _monthly_panel / _sp500_membership / _fund_dates: hash 불가 타입

    Returns:
        pd.Series (index=fund_dates, value=monthly EW return)
    """
    panel = _monthly_panel
    out: dict[pd.Timestamp, float] = {}
    fund_dates = pd.DatetimeIndex(_fund_dates)

    for i, t in enumerate(fund_dates):
        # rebalance 시점 = 직전 월말 (look-ahead 회피)
        if i > 0:
            t_prev = fund_dates[i - 1]
        else:
            t_prev = t - pd.offsets.MonthEnd(1)

        # t-1 시점 sp500 universe (또는 가장 가까운 직전)
        universe = _resolve_universe_at(t_prev, _sp500_membership)
        if not universe:
            continue
        universe_set = set(universe)

        # t 시점 active ticker 의 ret_1m 평균 (NaN 제외)
        rows = panel[(panel["date"] == t) & (panel["ticker"].isin(universe_set))]
        ret_1m = rows["ret_1m"].dropna()
        if len(ret_1m) == 0:
            continue

        out[t] = float(ret_1m.mean())

    return pd.Series(out, name="EW")


@st.cache_data
def compute_ivw_returns(
    _monthly_panel: pd.DataFrame,
    _sp500_membership: dict,
    _fund_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    IVW baseline (Naive Low-vol) — weight_i = (1/σ_i) / Σ(1/σ_j).
    Frazzini & Pedersen (2014) "Betting Against Beta".

    학술 표준 (decisionlog Q-B 정정 + Q-C + Q-D):
      1) 데이터: monthly_panel (vol_60d, ret_1m 사용)
      2) Universe: sp500_membership[t-1] (look-ahead 회피)
      3) σ_i = monthly_panel[date=t-1, ticker=i].vol_60d (t-1 시점 60일 변동성)
              → look-ahead 회피 (t 시점 vol 사용 X)
      4) weight_i = (1/σ_i) / Σ(1/σ_j)
      5) ret[t] = Σ(weight_i × ret_1m_i_at_t) / Σ(weight_i)  (t 시점 데이터 누락 ticker normalize)

    NOTE: 윈도우 60d 는 plan 정정 (원안 120d) — monthly_panel 부재로 정정.
          decisionlog Q-B 변경 이력 참조.
    """
    panel = _monthly_panel
    out: dict[pd.Timestamp, float] = {}
    fund_dates = pd.DatetimeIndex(_fund_dates)

    for i, t in enumerate(fund_dates):
        if i > 0:
            t_prev = fund_dates[i - 1]
        else:
            t_prev = t - pd.offsets.MonthEnd(1)

        universe = _resolve_universe_at(t_prev, _sp500_membership)
        if not universe:
            continue
        universe_set = set(universe)

        # t-1 시점 active ticker 의 vol_60d (look-ahead 회피)
        prev_rows = panel[(panel["date"] == t_prev) & (panel["ticker"].isin(universe_set))]
        prev_rows = prev_rows[(prev_rows["vol_60d"] > 0) & prev_rows["vol_60d"].notna()]
        if len(prev_rows) == 0:
            continue

        # weight = (1/σ) / Σ(1/σ)
        inv_vol = 1.0 / prev_rows["vol_60d"].values
        weights = pd.Series(inv_vol / inv_vol.sum(), index=prev_rows["ticker"].values)

        # t 시점 ret_1m (활성 ticker 만)
        curr_rows = panel[
            (panel["date"] == t) & (panel["ticker"].isin(weights.index))
        ][["ticker", "ret_1m"]].dropna()
        if len(curr_rows) == 0:
            continue

        # Σ(weight_i × ret_1m_i) / Σ(weight_i 의 valid ticker 부분 합)
        # → t 시점 데이터 누락 ticker 의 weight 만큼 누락 보정
        valid_weights = weights.loc[curr_rows["ticker"].values]
        weighted_ret = (curr_rows["ret_1m"].values * valid_weights.values).sum()
        weight_sum = valid_weights.sum()
        if weight_sum == 0:
            continue

        out[t] = float(weighted_ret / weight_sum)

    return pd.Series(out, name="IVW")


# === 기간 / Regime 정의 (final/master_table.py 정확 재현) ===========

# EVAL_PERIODS — final master_table.py:120-124 그대로
EVAL_PERIODS = {
    "TEST":     ("2010-01-01", "2023-12-31"),  # 168m
    "HOLD_OUT": ("2024-01-01", "2025-12-31"),  # 24m
    "FULL":     ("2010-01-01", "2025-12-31"),  # 192m
}

# REGIMES — final master_table.py:109-114 그대로 (HMM n=3 구조전환점)
# NOTE: HOLD_OUT 은 별도 (EVAL_PERIODS), Regime 은 시장 국면 정의
REGIMES = {
    "R1_회복": ("2010-01-01", "2012-06-30"),  # Post-GFC + EU위기 (30m)
    "R2_확장": ("2012-07-01", "2019-12-31"),  # 장기 Bull (90m)
    "R3_변동": ("2020-01-01", "2024-12-31"),  # COVID + 22 베어 + AI랠리 (60m)
}

# 사이드바 토글 호환 (기존 코드 영향 X)
TEST_END = pd.Timestamp(EVAL_PERIODS["TEST"][1])
HO_START = pd.Timestamp(EVAL_PERIODS["HOLD_OUT"][0])


def filter_period(returns: pd.Series, period: str) -> pd.Series:
    """
    사이드바 기간 토글 (FULL / TEST / HO) 에 따라 returns 필터링.

    Args:
        returns: pd.Series with DatetimeIndex
        period: "FULL" / "TEST" / "HO"

    Returns:
        필터된 pd.Series (FULL 은 그대로 반환)
    """
    if period == "TEST":
        return returns[returns.index <= TEST_END]
    if period == "HO":
        return returns[returns.index >= HO_START]
    return returns  # "FULL"


def filter_by_eval_period(returns: pd.Series, eval_label: str) -> pd.Series:
    """EVAL_PERIODS dict 기준 필터 (TEST / HOLD_OUT / FULL)."""
    start, end = EVAL_PERIODS[eval_label]
    return returns[(returns.index >= start) & (returns.index <= end)]


def filter_by_regime(returns: pd.Series, regime_label: str) -> pd.Series:
    """REGIMES dict 기준 필터 (R1_회복 / R2_확장 / R3_변동)."""
    start, end = REGIMES[regime_label]
    return returns[(returns.index >= start) & (returns.index <= end)]


# === 일별 portfolio return 산출 (Performance 영역 8 분포 통계) ========

@st.cache_data
def compute_fund_daily_returns(
    _fund_weights: pd.DataFrame,
    _daily_returns: pd.DataFrame,
    _comp: pd.DataFrame | None = None,
    tc: float = 0.002,
    nan_threshold: float = 0.9,
) -> pd.Series:
    """
    펀드 일별 portfolio return 산출 (drift 반영 + TC 차감, 펀드 실제 운용 정합).

    매월 t 시점 fund.weights 결정 → 다음 월 (t, t+1] 기간 일별 portfolio return:
      산식 (drift 반영, 2026-05-12 정정):
        - 시점 t (월말) 에 자산 1.0 → 종목 i 에 w_i 투자
        - 시점 t+d 의 종목 i 가치 = w_i × Π_d'=1..d (1+r_i,d')   ← 자연 drift
        - 시점 t+d 의 portfolio 가치 = Σ_i [w_i × Π_d' (1+r_i,d')]
        - 일별 수익률 = portfolio(d) / portfolio(d-1) - 1

      TC 처리 (2026-05-12 추가):
        - _comp + tc 제공 시 시점 t (월말 매매 시점) 에 turnover × tc 만큼 차감
        - 펀드 net 분포 산출 (펀드 자체 평가 의도)
        - _comp=None 시 gross 분포 (TC 미반영)

    산식 변경 이력:
      1) 이전: port_d = Σ_i (w_i × log_ret_i) — log return weighted sum (학술 부정확)
      2) → simple return weighted sum (학술 표준)
      3) → drift 반영 (펀드 실제 운용 정합)
      4) → TC 차감 추가 (펀드 자체 평가 — 분포 분석이 시장이 아닌 펀드 net 기준)

    데이터 결함 처리 (final compute_daily_slice 패턴):
      1) NaN 비율 < (1-nan_threshold) ticker 만 active universe 유지
      2) 남은 NaN 은 fillna(0) 처리 (그 일자 0% 수익)

    Args:
        _fund_weights: pd.DataFrame (월별 index, ticker columns, sum=1)
        _daily_returns: pd.DataFrame (일별 index, ticker columns, log return)
        _comp: pd.DataFrame | None (pkl['comp'], turnover 컬럼 사용). None 이면 TC 미반영 (gross).
        tc: float (편측 거래비용, 기본 0.002 = 20bp)
        nan_threshold: float (기본 0.9)

    Returns:
        pd.Series (일별 portfolio return, DatetimeIndex). _comp 제공 시 net, 미제공 시 gross.
    """
    daily_returns = _daily_returns
    weights = _fund_weights
    comp = _comp

    # 월별 weight 의 인덱스 정렬
    monthly_dates = pd.DatetimeIndex(weights.index).sort_values()

    parts: list[pd.Series] = []
    for i, t_decision in enumerate(monthly_dates):
        # 보유 기간: (t_decision, t_next]
        t_next = monthly_dates[i + 1] if i + 1 < len(monthly_dates) else daily_returns.index.max()
        mask = (daily_returns.index > t_decision) & (daily_returns.index <= t_next)
        period = daily_returns.loc[mask]
        if len(period) == 0:
            continue

        # active ticker = weight > 0
        active_w = weights.loc[t_decision]
        active_tickers = active_w[active_w > 0].index.tolist()
        active_in_data = [tk for tk in active_tickers if tk in period.columns]
        if not active_in_data:
            continue

        # final compute_daily_slice 결함처리:
        #   NaN 비율 < (1-nan_threshold) ticker 만 유지
        period_active = period[active_in_data]
        T = len(period_active)
        thresh = int(T * nan_threshold)
        valid_tickers = period_active.columns[period_active.notna().sum() >= thresh].tolist()
        if not valid_tickers:
            continue

        # 남은 NaN 은 0 처리 (final 패턴)
        period_clean = period_active[valid_tickers].fillna(0)

        # log return → simple return 변환
        # daily_returns.pkl 은 np.log(close/close.shift(1)) 산출본 (log return)
        period_simple = np.exp(period_clean) - 1

        # weight 정규화 (active ∩ valid 만으로 sum=1 재조정)
        w_valid = active_w[valid_tickers]
        w_sum = w_valid.sum()
        if w_sum <= 0:
            continue
        w_norm = w_valid / w_sum

        # 일별 portfolio return (drift 반영 + TC 차감, 2026-05-12 정정):
        #   - 시점 t (월말) 에 자산 1.0, 종목 i 에 w_i 투자
        #   - 시점 t+d 의 종목 i 가치 = w_i × Π_d'=1..d (1+r_i,d')  ← 자연 drift
        #   - 시점 t (월말 매매) 에 turnover × tc 만큼 차감 (TC 반영, _comp 제공 시)
        cum_factor = (1 + period_simple).cumprod()       # 종목별 누적 (drift 추적)
        port_value = cum_factor.dot(w_norm)              # 일별 portfolio 가치 (시작 1.0)

        # TC 차감 (월말 매매 시점, 첫 일자에 반영)
        # 첫 일자의 net return = tc_factor × port_value[0] - 1
        # 그 후 일자: gross 와 동일 비율 (= port_value 의 pct_change)
        if comp is not None and "turnover" in comp.columns and t_decision in comp.index:
            turnover = comp.loc[t_decision, "turnover"]
            tc_factor = (1 - turnover * tc) if (turnover and not pd.isna(turnover)) else 1.0
        else:
            tc_factor = 1.0  # TC 미반영 (gross)

        port_d = port_value / port_value.shift(1) - 1    # 일별 수익률 (둘째 일자부터)
        port_d.iloc[0] = tc_factor * port_value.iloc[0] - 1  # 첫 일자: TC 반영
        parts.append(port_d)

    if not parts:
        return pd.Series(dtype=float)

    return pd.concat(parts).sort_index()

