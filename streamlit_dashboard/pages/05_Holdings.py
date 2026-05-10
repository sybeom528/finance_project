"""
pages/05_Holdings.py - Holdings (보유 종목)

Phase 2 에서 구현 예정. 8 영역 (KPI / Top N 분포 / Bubble / Treemap /
변천사 / Tornado / Effective N / Footer).

참조: docs/plan/03_pages/05_holdings.md
"""

import streamlit as st

from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import inject_custom_css, render_page_header

inject_custom_css()
init_session_state()

render_page_header("Holdings", "보유 종목")
st.info("🚧 이 페이지는 Phase 2 에서 구현 예정입니다 (8 영역).")

render_footer()
