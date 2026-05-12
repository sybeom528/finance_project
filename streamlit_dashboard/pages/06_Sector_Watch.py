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
fund = load_fund_results()  # default = mat_eq_eq_raw_pap (최종 Top 1)
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
        "펀드의 섹터 구성과 시장 (S&P 500) 대비 차이를 분석합니다. "
        "**2024-2025년 섹터 변화가 펀드 성과에 미친 영향** 도 함께 확인 가능. "
        "사이드바에서 기간 선택 가능."
    ),
)


period = st.session_state.get("period", "FULL")


# === 영역 3: Sector Summary KPI 5개 ===================================
st.subheader(f"섹터 KPI — {period}")
st.caption(
    "**섹터 집중도 / 섹터 비중 평균 차이 / 섹터 비중 최대 차이 / 최대 over-weight 섹터 / 최대 under-weight 섹터**. "
    "시장 (SPY) 대비 어떤 섹터에 더/덜 투자했는지 비교합니다."
)
render_sector_kpi(weights, panel, sp500_membership, ticker_to_sector, period)
st.divider()


# === 영역 4: Sector Treemap ============================================
st.subheader("Sector Treemap — Fund vs SPY")
st.caption(
    "좌측은 펀드, 우측은 시장 (S&P 500 시가총액 가중) 의 섹터 구성을 같은 시점에서 비교. "
    "시점 슬라이더로 원하는 월 선택 가능."
)
render_sector_treemap(weights, panel, sp500_membership, ticker_to_sector)
st.divider()


# === 영역 5: Sector Decomposition 표 ===================================
st.subheader("Sector Decomposition — Latest snapshot")
st.caption(
    "**섹터 / 비중 / 시장 대비 차이 / 12개월 수익률 / 보유 종목 수 / 변동성 / β / Sharpe / 기여도**. "
    "섹터 단위 값 = 섹터 내 종목들의 가중 평균. "
    "컬럼 클릭 정렬 + CSV 다운로드 가능."
)
render_sector_decomposition_table(weights, panel, sp500_membership, ticker_to_sector)
st.divider()


# === 영역 6: Sector Tilt vs SPY (Tornado) =============================
st.subheader("Sector Tilt vs SPY — 섹터 비중 차이")
st.caption(
    "시장 대비 펀드의 섹터 비중 차이 (over-weight ↔ under-weight). "
    "0% 기준선 + ±1% 임계 (의미 있는 차이) + ±5% 임계 (큰 차이). "
    "**IT 섹터의 under-weight 가 2024-2025년 부진의 핵심 원인**."
)
render_sector_tilt_tornado(weights, panel, sp500_membership, ticker_to_sector)
st.divider()


# === 영역 7: Sector 시계열 변화 (Sector Rotation) =====================
st.subheader("Sector Rotation — 시계열 변화")
st.caption(
    "섹터 비중 변화 시계열 — "
    "차트 유형 (누적 면적 / 다중 라인) + 데이터 (펀드 / 시장 / 차이) + 시간 단위 (월별 / 분기별) 선택. "
    "시장 국면 배경 + 주요 이벤트 표시."
)
render_sector_rotation(weights, panel, sp500_membership, ticker_to_sector)
st.divider()


# === 영역 8: Hold Out 24m 분석 ========================================
st.subheader("Hold Out 24개월 분석")
st.caption(
    "**Hold Out 24개월 (2024-2025) 부진 원인 분석**: "
    "1) 시장 IT 집중도가 급격히 상승, 반면 펀드는 IT under-weight 유지 → 시장 대비 부진. "
    "2) **IT 섹터 평균 변동성이 시장 평균보다 일관되게 높음** → LSTM 변동성 인지 운용의 IT under-weight 근거. "
    "3) Hold Out 기간 섹터별 기여도 분석. "
    "4) 시장 국면별 집중도 비교 — 펀드는 일관된 분산, 시장은 Hold Out 기간 IT 집중. "
    "5) 학술적 결론 — 분산 투자의 단기 trade-off."
)
render_ho_justification(weights, panel, sp500_membership, ticker_to_sector, fund_ret, fund_spy)


# === 영역 9: Footer ===================================================
st.caption(
    "※ 시장 (SPY) 섹터 비중은 S&P 500 종목들의 시가총액 가중으로 산출됩니다. "
    "섹터 분류는 GICS 표준 11분류를 따릅니다."
)
render_footer()
