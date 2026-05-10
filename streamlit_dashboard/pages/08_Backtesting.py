"""
pages/08_Backtesting.py - Backtesting (검증)

Phase 3 에서 구현 예정. 7 영역 (Hero / Regime / Sub-events / TC Sensitivity /
Stationarity / Sensitivity 156 config / Footer — 균형 옵션 (B) 적용).

참조: docs/plan/03_pages/08_backtesting.md
"""

import streamlit as st

from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import inject_custom_css, render_page_header

inject_custom_css()
init_session_state()

render_page_header("Backtesting", "검증")
st.info("🚧 이 페이지는 Phase 3 에서 구현 예정입니다 (7 영역).")

render_footer()
