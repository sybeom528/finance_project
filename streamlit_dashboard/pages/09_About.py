"""
pages/09_About.py - About / FAQ

영역별 자세한 결정은 구현 후 팀 상의 (M-1 c 결정).
현재는 메타 결정만 완료.

참조: docs/plan/03_pages/09_about.md, docs/decisionlog/09_about.md
"""

import streamlit as st

from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import inject_custom_css, render_page_header

inject_custom_css()
init_session_state()

render_page_header("About / FAQ", "소개 / 자주 묻는 질문")
st.info(
    "🚧 이 페이지의 영역별 자세한 결정은 다른 페이지 구현 완료 후 팀과 상의하여 진행 예정입니다."
)

render_footer()
