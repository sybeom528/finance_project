"""
pages/07_Methodology.py - Methodology (방법론)

Phase 3 에서 구현 예정. 8 영역 (Hero / Sankey 4-slot / 4 Stage Tab /
LSTM walk-forward / Factor / 학술 근거 / 한계 카드 / Footer).

참조: docs/plan/03_pages/07_methodology.md
"""

import streamlit as st

from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import inject_custom_css, render_page_header

inject_custom_css()
init_session_state()

render_page_header("Methodology", "방법론")
st.info("🚧 이 페이지는 Phase 3 에서 구현 예정입니다 (8 영역).")

render_footer()
