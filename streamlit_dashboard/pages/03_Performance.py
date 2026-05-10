"""
pages/03_Performance.py - Performance (성과 분석)

Phase 1.5 에서 구현 예정. 9 영역 (KPI 11개 / 누적수익 / Annual / Active Return /
Rolling Return / Regime Heatmap / Multi-Benchmark / Footer 등).

참조: docs/plan/03_pages/03_performance.md
"""

import streamlit as st

from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import inject_custom_css, render_page_header

inject_custom_css()
init_session_state()

render_page_header("Performance", "성과 분석")
st.info("🚧 이 페이지는 Phase 1.5 에서 구현 예정입니다 (9 영역).")

render_footer()
