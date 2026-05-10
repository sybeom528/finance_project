"""
pages/06_Sector_Watch.py - Sector Watch (섹터 분석)

Phase 2 에서 구현 예정. 8 영역 (KPI / SPY 비교 / Tilt / Treemap / Active Bets /
Tilt 변천사 / Rotation / Footer + HO 정당화 narrative).

참조: docs/plan/03_pages/06_sector_watch.md
"""

import streamlit as st

from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import inject_custom_css, render_page_header

inject_custom_css()
init_session_state()

render_page_header("Sector Watch", "섹터 분석")
st.info("🚧 이 페이지는 Phase 2 에서 구현 예정입니다 (8 영역).")

render_footer()
