"""
lib/page_helpers.py - 페이지 헤더 + 서브헤더 + CSS 주입

모든 페이지의 상단부 일관성 보장.
참조: docs/plan/02_common.md 10절, 6.1절 (H-1 Pretendard 폰트)
"""

import streamlit as st


# === Pretendard 폰트 fallback chain (H-1) =============================
# CDN 차단 시 자동으로 다음 폰트로 fallback → 기능 정상 유지
_PRETENDARD_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"] {
    font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic",
                 -apple-system, BlinkMacSystemFont, sans-serif !important;
}
</style>
"""


def inject_custom_css() -> None:
    """
    Pretendard 폰트 fallback chain 주입 (H-1).
    app.py 진입 시 1회 호출.
    """
    st.markdown(_PRETENDARD_CSS, unsafe_allow_html=True)


def render_page_header(page_name_en: str, page_name_ko: str) -> None:
    """
    모든 페이지 상단 Header (Overview 영역 1 일관 적용).

    Args:
        page_name_en: 영문 페이지명 (예: "Performance")
        page_name_ko: 한글 페이지명 (예: "성과 분석")

    좌측 = 페이지명 (영/한 병기, A-3 결정)
    우측 = 펀드 메타 (Active 표시 / 벤치마크 / 데이터 시점)
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(page_name_en)
        st.caption(page_name_ko)
    with col2:
        st.markdown("● **Active (Simulated)**")
        st.caption("Benchmark: S&P 500 (SPY)")
        st.caption("Data as of: 2025-12-31")


def render_subheader(title_en: str, title_ko: str, description: str) -> None:
    """
    페이지별 Sub-header 카드 (Performance 영역 2 패턴).

    Cobalt Blue 좌측 테두리 + 어두운 카드 배경.
    페이지의 핵심 메시지 / 영역 설명을 표시.
    """
    st.markdown(
        f"""
        <div style="
            background-color: #1F2937;
            border-left: 4px solid #3B82F6;
            padding: 16px;
            border-radius: 4px;
            margin-bottom: 16px;
        ">
            <div style="font-size: 18px; font-weight: bold; color: #FAFAFA;">
                ℹ️ {title_en} ({title_ko})
            </div>
            <div style="font-size: 14px; color: #9CA3AF; margin-top: 8px;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
