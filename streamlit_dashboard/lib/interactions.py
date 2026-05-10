"""
lib/interactions.py - Q-Zoom 인터랙션 (G-1)

같은 페이지 expand 패턴 — Modal X / 별도 페이지 X.
사용자가 차트 일부를 클릭하면 같은 페이지 하단에 expand 영역이 펼쳐짐.

Streamlit 1.30+ on_select="rerun" 활용.

참조: docs/plan/02_common.md 7절, decisionlog/11_dl_sections.md G-1
"""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st
import plotly.graph_objects as go


def render_zoomable_chart(
    fig: go.Figure,
    key: str,
    expand_func: Optional[Callable] = None,
    expand_title: str = "📍 선택한 시점 상세",
):
    """
    Plotly 차트 렌더 + 클릭 selection 시 expand 영역 표시.

    Args:
        fig: Plotly Figure (이미 trace + range slider + hover 포함)
        key: Streamlit unique key (페이지 내 충돌 방지)
        expand_func: 클릭 시 호출 (selection 객체를 인자로 받음). None 이면 expand 비활성.
        expand_title: expand 영역 제목

    Returns:
        Streamlit plotly_chart 의 selection 결과 (호출 측에서 추가 처리 가능)

    호출 예시:
        def show_detail(selection):
            st.write(f"선택된 데이터: {selection['points']}")
        render_zoomable_chart(fig, key="perf_annual", expand_func=show_detail)
    """
    selected = st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select="rerun",  # Streamlit 1.30+
    )

    # 사용자 selection 처리
    if expand_func is not None and selected and selected.get("selection"):
        sel = selected["selection"]
        # selection 의 points / box 등 비어있지 않은 경우만 expand
        if sel.get("points") or sel.get("box") or sel.get("lasso"):
            with st.expander(expand_title, expanded=True):
                expand_func(sel)

    return selected
