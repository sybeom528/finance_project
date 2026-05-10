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

@st.cache_data
def load_fund_results(config_name: str = "mat_eq_mcap_raw_rms") -> dict:
    """
    펀드 backtest 결과 (최종 Top 1 = mat_eq_mcap_raw_rms).

    config 차원: prior=capm_eq / p_weight=mcap / q_mode=raw_lam / omega_mode=rmse
    선정 근거: turnover 안정성 (~0.43, mat_eq_eq_raw_pap 의 0.99 대비 절반)
              + Sharpe 1.05 / MDD -13.9% — 운용 안정성 + 학술 narrative 균형

    대시보드 사본 (Top 1) 우선, 없으면 final 원본 fallback.

    pkl 구조: {"weights": DataFrame, "ret": Series, "spy_ret": Series,
              "comp": DataFrame, "config": dict, "meta": DataFrame, ...}
    """
    local = RESULTS_DIR / f"{config_name}.pkl"
    if local.exists():
        with open(local, "rb") as f:
            return pickle.load(f)

    original = ORIGINAL_RESULTS_DIR / f"{config_name}.pkl"
    if original.exists():
        with open(original, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(
        f"Config '{config_name}' pkl 을 찾을 수 없습니다.\n"
        f"  확인한 경로:\n  - {local}\n  - {original}"
    )


@st.cache_data
def load_other_config_results(config_name: str) -> dict:
    """
    다른 155 config 결과 (Backtesting 영역 6 Sensitivity Test 시).
    원본 final/results/ 에서 직접 참조 (대시보드 사본에는 Top 1 만 존재).
    """
    with open(ORIGINAL_RESULTS_DIR / f"{config_name}.pkl", "rb") as f:
        return pickle.load(f)


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
    nan_threshold: float = 0.9,
) -> pd.Series:
    """
    펀드 일별 portfolio return 산출 — final/bl_functions.py:compute_daily_slice 결함처리 차용.

    매월 t 시점 fund.weights 결정 → 다음 월 (t, t+1] 기간 일별 portfolio return:
      port_d = Σ_i (weight_i × daily_ret_i)

    데이터 결함 처리 (final compute_daily_slice 패턴):
      1) NaN 비율 < (1-nan_threshold) ticker 만 active universe 유지
         (예: nan_threshold=0.9 → NaN 10% 초과 ticker 자동 배제)
      2) 남은 NaN 은 fillna(0) 처리 (그 일자 0% 수익)

    이는 펀드 backtest 와 동일 처리 → 일관성 보장.

    Args:
        _fund_weights: pd.DataFrame (월별 index, ticker columns, sum=1)
        _daily_returns: pd.DataFrame (일별 index, ticker columns)
        nan_threshold: float (final compute_daily_slice thresh_daily 와 동일, 기본 0.9)

    Returns:
        pd.Series (일별 portfolio return, DatetimeIndex)

    NOTE: 펀드 backtest 자체는 forward-looking weights (t 시점 결정 → t+1 보유) 패턴.
          여기서는 fund.weights[t] = t 시점 보유 weight 가정 (= t-1 결정).
          호출 측에서 주의 — t 시점 weight 로 t 까지의 일별 수익률 산출은 look-ahead.
          따라서 weight[t]를 (t, t+1] 보유 기간에 적용 (shift 1).
    """
    daily_returns = _daily_returns
    weights = _fund_weights

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

        # weight 정규화 (active ∩ valid 만으로 sum=1 재조정)
        w_valid = active_w[valid_tickers]
        w_sum = w_valid.sum()
        if w_sum <= 0:
            continue
        w_norm = w_valid / w_sum

        # 일별 portfolio return = Σ(weight_i × daily_ret_i)
        port_d = period_clean.dot(w_norm)
        parts.append(port_d)

    if not parts:
        return pd.Series(dtype=float)

    return pd.concat(parts).sort_index()
