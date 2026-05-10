"""
pages/02_Investment_Simulator.py - Investment Simulator (F-6)

Phase 2 에서 구현 예정.
사용자가 직접 기간 + 금액 입력 → 가상 백테스트 시뮬레이션 → Insight 카드 6개.

참조: docs/plan/03_pages/02_simulator.md
"""

import streamlit as st

from lib.disclosure import init_session_state, render_footer, render_simulator_disclaimer
from lib.page_helpers import inject_custom_css, render_page_header

inject_custom_css()
init_session_state()

render_page_header("Investment Simulator", "투자 시뮬레이터")
render_simulator_disclaimer()

st.info("🚧 이 페이지는 Phase 2 에서 구현 예정입니다.")

render_footer()
