"""
pages/05_Holdings.py — Holdings 페이지 (8 영역)

보유 종목 / 비중 / 시가총액 분포 / 변천사 / 기여도 분석 전담 페이지.

8 영역:
  1. Header
  2. Sub-header
  3. Holdings Summary KPI 6개 (Number / Eff N / Single HHI / Sector HHI
                              / Top Weights / Avg Turnover)
  4. Top N Holdings 표 (10/20/50/All + 7컬럼 + Weight 막대 + 섹터 색상)
  5. 시가총액 분포 (Bubble + Treemap 탭, 시점 슬라이더)
  6. 보유 종목 변천사 (Multi-line + 섹터/Top N 토글 + Regime + 마커)
  7. 종목별 기여도 분석 (Tornado: Top + Bottom + Sector 합계 + 검증)
  8. Footer (+ yfinance footnote)

메트릭 = decisionlog/05_holdings.md Hold-Q1:
  - final 정합: comp.n_stocks / eff_n / top1_weight / top10_share / turnover
  - 학술 표준: Single/Sector HHI (Hirschman 1945) / Simple Contribution (Brinson 1986)

참조: docs/plan/03_pages/05_holdings.md, decisionlog/05_holdings.md
"""

import streamlit as st

from lib.data_loader import (
    get_ticker_company_dict,
    load_fund_results,
    load_monthly_panel,
    load_universe,
)
from lib.disclosure import init_session_state, render_footer
from lib.holdings_charts import (
    render_attribution_tornado,
    render_holdings_history,
    render_holdings_kpi,
    render_market_cap_distribution,
    render_top_n_table,
)
from lib.page_helpers import (
    inject_custom_css,
    render_page_header,
    render_sidebar,
    render_subheader,
)


# === 페이지 설정 ======================================================
inject_custom_css()
init_session_state()
render_sidebar()


# === 데이터 로드 ======================================================
fund = load_fund_results(st.session_state.get("config_name", "mat_eq_eq_raw_pap"))
weights = fund["weights"]
comp = fund["comp"]
fund_ret = fund["ret"]

panel = load_monthly_panel()
universe = load_universe()
ticker_company_map = get_ticker_company_dict()


# === 영역 1: Header ===================================================
render_page_header("Holdings", "보유 종목")


# === 영역 2: Sub-header ===============================================
render_subheader(
    title_en="Holdings",
    title_ko="보유 종목",
    description=(
        "보유 종목 / 비중 / 시가총액 분포 / 변천사 분석. "
        "사이드바에서 기간 (FULL / TEST / HO) 토글 가능. "
        "Holdings 메트릭 = final/bl_functions 산출 (n_stocks / eff_n / turnover) "
        "+ 학술 표준 (Hirschman HHI 1945 / Brinson Simple Contribution 1986)."
    ),
)


period = st.session_state.get("period", "FULL")


# === 영역 3: Holdings Summary KPI 6개 =================================
st.subheader(f"보유 종목 KPI — {period}")
st.caption(
    "Number / Effective N / Single HHI / Sector HHI / Top Weights / Latest Turnover. "
    "Latest snapshot = 최신 월 (2025-12), 기간 평균 = 사이드바 토글 월별 평균. "
    "vs 균등 = 펀드 universe 균등가중 기준선 (학술 정직 — SPY 503 가정 X)."
)
render_holdings_kpi(weights, comp, universe, panel, period)
st.divider()


# === 영역 4: Top N Holdings 표 ========================================
st.subheader("Top N Holdings — Latest snapshot")
st.caption(
    "현재 보유 종목 Top N (비중 내림차순). "
    "컬럼: Rank / Ticker (Company) / Sector / Weight / Market Cap / 12m Return / ΔWeight. "
    "ΔWeight = 전월 대비 비중 변화. 컬럼 헤더 클릭 → 정렬, CSV 다운로드 가능."
)
render_top_n_table(weights, panel, universe, ticker_company_map)
st.divider()


# === 영역 5: 시가총액 분포 (Bubble + Treemap) =========================
st.subheader("시가총액 분포 — Bubble + Treemap")
st.caption(
    "시점 슬라이더로 월별 시점 선택 가능 (192 시점). "
    "Treemap = 비중/시가총액 면적 + 섹터/수익률 색상 토글. "
    "Bubble = X/Y/크기/색상 4축 자유 조합 (시가총액-비중 관계 등 자유 탐색)."
)
render_market_cap_distribution(weights, panel, universe, ticker_company_map)
st.divider()


# === 영역 6: 보유 종목 변천사 =========================================
st.subheader("보유 종목 변천사 — Multi-line")
st.caption(
    "시간에 따른 섹터/종목 비중 변화. 그룹화 토글 — "
    "**섹터** (11 GICS 합계) / **Top N (선택 시점)** = 슬라이더 시점 Top N ticker 의 전 기간 시계열 "
    "(슬라이더 변경 시 라인/legend 자체가 변경 — 시점별 변천 narrative 직접 탐색). "
    "Others = 1 − 선택 시점 Top N 합 (정확한 보완). 슬라이더 시점에 점선 마커 표시. "
    "시간 단위 토글 (월별 192 / 분기별 64). "
    "Regime 배경색 (R1/R2/R3/HO) + COVID/2022 Bear 등 이벤트 annotation 포함."
)
render_holdings_history(weights, universe)
st.divider()


# === 영역 7: 종목별 기여도 분석 (Tornado) =============================
st.subheader(f"종목별 기여도 분석 — {period}")
st.caption(
    "Simple Contribution = Σ(w × R) per ticker (Brinson et al. 1986). "
    "Top N 양수 + Top N 음수 (학술 정직 — Bottom contributors 도 노출). "
    "Sector 합계 = Sector Watch 페이지로 navigation. "
    "검증: Σ Contribution ≈ Fund 누적 수익률 (선형 근사 한계 ε)."
)
render_attribution_tornado(weights, panel, universe, ticker_company_map, fund_ret, period)


# === 영역 8: Footer ===================================================
st.caption(
    "※ 회사명 / 시가총액 / 섹터 매핑: yfinance 기반 "
    "(`data/ticker_company_map.csv` 캐시, 인수합병 옛 종목 4건은 ticker 자체로 fallback)."
)
render_footer()
