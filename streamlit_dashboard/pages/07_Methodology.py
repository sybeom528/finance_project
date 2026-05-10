"""
pages/07_Methodology.py — Methodology 페이지 (8 영역, 초안)

펀드 운용 방법론 설명 전담 — BL + LSTM + 4-slot config 의 학술/실무 토대.

8 영역:
  1. Header
  2. Sub-header
  3. Methodology Overview (BL+LSTM Plotly Sankey)
  4. Black-Litterman 상세 + 4-slot config
  5. LSTM 변동성 예측 상세 (★ Walk-forward 명시)
  6. Factor 분석 (CAPM + FF5)
  7. 한계 + 향후 개선 (3 한계 카드)
  8. Footer

설계 원칙 (초안 — 향후 변경 용이):
  - 영역별 독립 함수 (영역 추가/제거 시 페이지 entry 만 변경)
  - 학술 인용 / 한계 정의 외부 dict
  - 데이터: ff5_monthly.csv + 펀드 ret

참조: docs/plan/03_pages/07_methodology.md, docs/decisionlog/07_methodology.md
"""

import streamlit as st

from lib.data_loader import load_ff5_monthly, load_fund_results
from lib.disclosure import init_session_state, render_footer
from lib.methodology_charts import (
    render_bl_detail,
    render_factor_analysis,
    render_limitations,
    render_lstm_detail,
    render_methodology_sankey,
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
fund = load_fund_results()  # default = mat_eq_mcap_raw_he
fund_ret = fund["ret"]
config_name = fund["config"]["name"]

ff5 = load_ff5_monthly()


# === 영역 1: Header ===================================================
render_page_header("Methodology", "방법론")


# === 영역 2: Sub-header ===============================================
render_subheader(
    title_en="Methodology",
    title_ko="방법론",
    description=(
        "**Black-Litterman + LSTM 변동성 예측 + 4-slot config + Factor 분석 + 한계 명시**. "
        "다른 페이지가 \"결과 위주\" 라면 본 페이지는 **과정 / 토대 위주**. "
        "BL+LSTM 의 학술/실무 정당화 + walk-forward 구조 + LSTM 정당화 narrative + 학술 정직성."
    ),
)


# === 영역 3: Methodology Overview (Sankey) ============================
st.subheader("Methodology Overview — BL+LSTM 흐름")
st.caption(
    "**9 노드 / 4 그룹** Plotly Sankey: 데이터 (3 노드) → BL Prior → LSTM Vol → View/Confidence "
    "→ BL Posterior → Optimizer → Portfolio Weights. "
    "각 그룹 색상 = `lib/colors.py:SANKEY_GROUP_COLORS`."
)
render_methodology_sankey()
st.divider()


# === 영역 4: Black-Litterman 상세 =====================================
st.subheader("Black-Litterman 상세 + 4-slot Config")
st.caption(
    "**BL 수식** (Bayesian Update of CAPM equilibrium with LSTM-informed view). "
    "**4-slot config**: prior / p_weight / q_mode / omega_mode — 본 펀드 ★ 강조. "
    "학술 인용: Black-Litterman (1990, 1992), He-Litterman (1999), Idzorek (2005)."
)
render_bl_detail(current_config=config_name)
st.divider()


# === 영역 5: LSTM 변동성 예측 상세 (Walk-forward) =====================
st.subheader("LSTM 변동성 예측 상세 — ★ Walk-forward")
st.caption(
    "**Walk-forward 구조** (Lopez de Prado 2018): is_len 1250d / oos_len 21d / step 21d / "
    "embargo 63d / seq_len 63d. 615 종목 × ~120 fold + HAR-RV 앙상블. "
    "TEST/HOLD_OUT 의 정확한 의미 명시 + Input/Output 표 + LSTM Cell 수식. "
    "학술 인용: Hochreiter & Schmidhuber (1997), Gers et al. (2000), Kim & Won (2018)."
)
render_lstm_detail()
st.divider()


# === 영역 6: Factor 분석 (CAPM + FF5) =================================
st.subheader("Factor 분석 — CAPM + FF5")
st.caption(
    "**핵심 질문**: \"펀드의 outperformance 가 진정한 운용 skill 인가, 단순 factor 노출인가?\". "
    "Factor 통제 후 alpha + factor exposure (β_MKT/SMB/HML/RMW/CMA) 분석. "
    "통계적 유의 (***/**/*) + 95% CI + R² 보강. "
    "학술 인용: Jensen (1968), Fama-French (1993, 2015), Carhart (1997)."
)
render_factor_analysis(fund_ret, ff5)
st.divider()


# === 영역 7: 한계 + 향후 개선 =========================================
st.subheader("한계 + 향후 개선")
st.caption(
    "**학술 정직성 선언** + **3 한계 카드** (균형 옵션 B): 🟧 HO 24m 부진 / 🟩 향후 개선 / 🟥 실무 제약. "
    "각 카드 자세한 내용은 Expander."
)
render_limitations()


# === 영역 8: Footer ===================================================
render_footer()
