"""
pages/05_Holdings.py — Holdings 페이지 (9 영역)

보유 종목 / 비중 / 시가총액 분포 / 변천사 / 집중도 / 기여도 분석 전담 페이지.

9 영역:
  1. Header
  2. Sub-header
  3. Holdings Summary KPI 6개 (Number / Eff N / Single HHI / Sector HHI
                              / Top Weights / Avg Turnover)
  4. Top N Holdings 표 (10/20/50/All + 7컬럼 + Weight 막대 + 섹터 색상)
  5. 시가총액 분포 (Bubble + Treemap 탭, 시점 슬라이더)
  6. 섹터 변천사 (11 GICS Multi-line + Regime + 이벤트 annotation)
  7. 시점별 Top N 합 vs Others (집중도 동적 추세 — 100% stacked area + hover Top N)
  8. 종목별 기여도 분석 (Tornado: Top + Bottom + Sector 합계 + 검증)
  9. Footer (+ yfinance footnote)

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
    render_top_n_share_timeseries,
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
fund = load_fund_results()  # default = mat_eq_eq_raw_pap (최종 Top 1)
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
        "펀드의 보유 종목 / 비중 / 시가총액 분포 / 시점별 변천을 분석합니다. "
        "사이드바에서 기간 선택 가능."
    ),
)


period = st.session_state.get("period", "FULL")


# === 영역 3: Holdings Summary KPI 6개 =================================
st.subheader(f"보유 종목 KPI — {period}")
st.caption(
    "**보유 종목 수 / 유효 종목 수 / 종목 집중도 / 섹터 집중도 / 상위 비중 / 회전율**. "
    "메인 = **사이드바 기간 평균** (토글에 즉시 반응) / caption = **최신 시점** 값 (보조). "
    "비교 기준은 균등 가중 (모든 종목 동일 비중) 기준선."
)
render_holdings_kpi(weights, comp, universe, panel, period)
st.divider()


# === 영역 4: Top N Holdings 표 ========================================
st.subheader("Top N Holdings — 선택 시점 snapshot")
st.caption(
    "선택 시점의 보유 종목 Top N (비중 큰 순). "
    "시점 슬라이더로 원하는 월 선택 가능 (기본 = 최신 월). "
    "컬럼: 순위 / 티커 (회사명) / 섹터 / 비중 / 시가총액 / 12개월 수익률 / 전월 대비 비중 변화 / 보유 개월. "
    "**12개월 수익률** = 종목 자체 가격 추세 (펀드 보유 기간 수익이 아님, 종목 정보 참고용). "
    "**보유 (월)** = 선택 시점부터 거꾸로 끊김 없이 연속 보유한 개월 수. "
    "컬럼 클릭 정렬 + CSV 다운로드 가능."
)
# 시점 슬라이더 (영역 5 와 동일 패턴)
available_dates_top = sorted(weights.index)
top_snapshot_date = st.select_slider(
    "시점 선택 (Top N)",
    options=available_dates_top,
    value=available_dates_top[-1],  # default = Latest
    format_func=lambda d: d.strftime("%Y-%m"),
    key="holdings_top_n_date",
)
render_top_n_table(weights, panel, universe, ticker_company_map, snapshot_date=top_snapshot_date)
st.divider()


# === 영역 5: 시가총액 분포 (Bubble + Treemap) =========================
st.subheader("시가총액 분포 — Bubble + Treemap")
st.caption(
    "시점 슬라이더로 원하는 월 선택. "
    "**Treemap** = 비중 / 시가총액 면적 + 색상 변경. "
    "**Bubble** = X축 / Y축 / 크기 / 색상 자유 조합 (시가총액-비중 관계 등 다각도 탐색)."
)
render_market_cap_distribution(weights, panel, universe, ticker_company_map)
st.divider()


# === 영역 6: 보유 종목 변천사 =========================================
st.subheader("섹터 변천사 — Multi-line")
st.caption(
    "11개 GICS 섹터별 비중 합계 변화 — 섹터 단위 펀드 구성 변천. "
    "시간 단위 (월별 / 분기별) + 시장 국면 배경 + 주요 이벤트 표시."
)
render_holdings_history(weights, universe)
st.divider()


# === 영역 7: 시점별 Top N 합 vs Others (집중도 동적 추세) ============
st.subheader("시점별 Top N 합 vs Others — 집중도 동적 추세")
st.caption(
    "각 시점에서 상위 N 종목이 차지하는 비중 변화 — 종목 단위 변천 분석. "
    "누적 면적 (상위 N 파란색 + 나머지 회색). "
    "**차트 위에 마우스를 올리면 그 시점의 상위 종목 목록과 비중을 확인** 할 수 있습니다."
)
render_top_n_share_timeseries(comp, weights)
st.divider()


# === 영역 8: 종목별 기여도 분석 (Tornado) =============================
st.subheader(f"종목별 기여도 분석 — {period}")
st.caption(
    "**기여도 계산 방식 선택**: "
    "**단순 합 (월별 합산)** — 직관적이지만 장기 누적과 차이 / "
    "**복리 보정 (장기 일치)** — 다기간 복리 효과 반영, 합계가 펀드 전체 수익률과 정확히 일치 (학술 표준). "
    "수익 기여 상위 + 손실 기여 상위 함께 표시."
)
render_attribution_tornado(weights, panel, universe, ticker_company_map, fund_ret, period)


# === 영역 9: Footer ===================================================
st.caption(
    "※ 회사명 / 시가총액 / 섹터 정보는 Yahoo Finance 출처입니다 "
    "(M&A 등으로 정보가 없는 일부 옛 종목은 티커로 표시)."
)
render_footer()
