"""
pages/06_Sector_Watch.py — Sector Watch 페이지 (9 영역)

섹터 단위 비중 / 분석 / HO 정당화 narrative 전담 페이지.

9 영역:
  1. Header
  2. Sub-header (HO narrative 명시)
  3. Sector Summary KPI 5개 (HHI / Avg|Tilt| / Active Bets / Most Over / Most Under)
  4. Sector Treemap (Fund vs SPY 좌우, 시점 슬라이더)
  5. Sector Decomposition 표 (9 컬럼)
  6. Sector Tilt vs SPY (Tornado, ±1%/±5% 임계선)
  7. Sector 시계열 변화 (Stacked Area / Multi-line, Fund / SPY / Tilt)
  8. ★★★ HO 24m 정당화 narrative (Markowitz 1952 + Fama-French 1992)
  9. Footer

메트릭 = decisionlog/06_sector_watch.md Sector-Q1:
  - Sector HHI = Hirschman (1945) sector level (Holdings 영역 3 동일)
  - SPY sector weight = panel.log_mcap × sp500_membership[t-1] (mcap 가중)
  - Sector Tilt = Fund_w − SPY_w (Active Management)
  - HO 정당화 = Markowitz (1952) + Fama-French (1992)

참조: docs/plan/03_pages/06_sector_watch.md, decisionlog/06_sector_watch.md
"""

import streamlit as st

from lib.data_loader import (
    get_ticker_sector_map,
    load_fund_results,
    load_monthly_panel,
    load_sp500_membership,
)
from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import (
    inject_custom_css,
    render_page_header,
    render_sidebar,
    render_subheader,
)
from lib.sector_charts import (
    render_ho_justification,
    render_sector_decomposition_table,
    render_sector_kpi,
    render_sector_rotation,
    render_sector_tilt_tornado,
    render_sector_treemap,
)


# === 페이지 설정 ======================================================
inject_custom_css()
init_session_state()
render_sidebar()


# === 데이터 로드 ======================================================
fund = load_fund_results()  # default = mat_eq_mcap_raw_rms (최종 Top 1)
weights = fund["weights"]
fund_ret = fund["ret"]
fund_spy = fund["spy_ret"]

panel = load_monthly_panel()
sp500_membership = load_sp500_membership()
ticker_to_sector = get_ticker_sector_map()


# === 영역 1: Header ===================================================
render_page_header("Sector Watch", "섹터 분석")


# === 영역 2: Sub-header (HO narrative 명시) ===========================
render_subheader(
    title_en="Sector Watch",
    title_ko="섹터 분석",
    description=(
        "섹터 비중 / 분산 / SPY 비교 분석. "
        "**HO 24m (2024-2025) sector rotation 영향과 펀드의 sector 분산 운용의 양면성** 분석 포함. "
        "사이드바에서 기간 (FULL / TEST / HO) 토글 가능. "
        "SPY sector weight = panel.log_mcap × sp500_membership[t-1] (mcap 가중, 학술 정확)."
    ),
)


period = st.session_state.get("period", "FULL")


# === 영역 3: Sector Summary KPI 5개 ===================================
st.subheader(f"섹터 KPI — {period}")
st.caption(
    "Sector HHI / Avg |Tilt| / Active Bets / Most Overweight / **Most Underweight** (★ HO narrative 핵심). "
    "Latest snapshot = 최신 월 (2025-12), 기간 평균 = 사이드바 토글 월별 평균. "
    "HHI 만 vs SPY 비교 (학술적으로 의미 있는 비교)."
)
render_sector_kpi(weights, panel, sp500_membership, ticker_to_sector, period)
st.divider()


# === 영역 4: Sector Treemap ============================================
st.subheader("Sector Treemap — Fund vs SPY")
st.caption(
    "좌측 Fund / 우측 SPY (S&P500 mcap 가중) 동일 시점 비교. "
    "시점 슬라이더로 월별 시점 선택 가능 (192 시점). "
    "섹터 색상 = GICS 11 표준 (`SECTOR_COLORS`)."
)
render_sector_treemap(weights, panel, sp500_membership, ticker_to_sector)
st.divider()


# === 영역 5: Sector Decomposition 표 ===================================
st.subheader("Sector Decomposition — Latest snapshot")
st.caption(
    "9 컬럼: Sector / Weight / Tilt vs SPY / 12m Return / # Holdings / Volatility / Beta / Sharpe / Contribution. "
    "sector level 메트릭 = ticker level 가중 평균 (sector 내 weight 정규화). "
    "컬럼 헤더 클릭 → 정렬, CSV 다운로드 가능."
)
render_sector_decomposition_table(weights, panel, sp500_membership, ticker_to_sector)
st.divider()


# === 영역 6: Sector Tilt vs SPY (Tornado) =============================
st.subheader("Sector Tilt vs SPY — Active Bets")
st.caption(
    "Tornado Chart (Most Over → Most Under). "
    "0% 기준선 + ±1% 임계선 (Active Bets) + ±5% 임계선 (큰 베팅). "
    "★ Information Technology = HO 정당화 narrative 핵심 (펀드 under-weight)."
)
render_sector_tilt_tornado(weights, panel, sp500_membership, ticker_to_sector)
st.divider()


# === 영역 7: Sector 시계열 변화 (Sector Rotation) =====================
st.subheader("Sector Rotation — 시계열 변화")
st.caption(
    "차트 토글 (Stacked Area / Multi-line) + 데이터 토글 (Fund / SPY / Tilt) + 시간 단위 (월별 / 분기별). "
    "Regime 배경색 (R1/R2/R3/HO) + COVID/2022 Bear/2024 AI Rally 이벤트 annotation 포함."
)
render_sector_rotation(weights, panel, sp500_membership, ticker_to_sector)
st.divider()


# === 영역 8: ★★★ HO 24m 정당화 narrative ===============================
st.subheader("HO 24m 분석 + 정당화 narrative ★")
st.caption(
    "**핵심 영역** — 펀드의 HO 24m underperform 의 학술적 정당화. "
    "1) SPY IT 비중 vs Fund IT Tilt 이중 축 — IT 시장 집중 + 펀드 under-weight 추세. "
    "2) HO Sector Carino Contribution — IT under-weight 의 직접적 영향. "
    "3) Regime 별 Sector HHI — 펀드는 일관된 분산, SPY 는 HO 에 집중 ↑. "
    "4) 학술 결론 (Markowitz 1952 + Fama-French 1992)."
)
render_ho_justification(weights, panel, sp500_membership, ticker_to_sector, fund_ret, fund_spy)


# === 영역 9: Footer ===================================================
st.caption(
    "※ SPY sector weight 산출: panel.log_mcap × sp500_membership[t-1] (mcap 가중, look-ahead 회피). "
    "GICS sector 매핑: panel.gics_sector 우선 + universe.csv fallback (load 시점 자동 정합화 적용)."
)
render_footer()
