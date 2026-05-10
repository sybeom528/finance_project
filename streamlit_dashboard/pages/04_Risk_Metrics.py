"""
pages/04_Risk_Metrics.py - Risk Metrics (위험 지표)

Phase 2 에서 구현 예정. 8 영역 (KPI / Distribution / Downside / Drawdown /
Tail Risk / Stability / Correlation / Footer).

참조: docs/plan/03_pages/04_risk_metrics.md
"""

import streamlit as st

from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import inject_custom_css, render_page_header

inject_custom_css()
init_session_state()

render_page_header("Risk Metrics", "위험 지표")
st.info("🚧 이 페이지는 Phase 2 에서 구현 예정입니다 (8 영역).")

render_footer()
