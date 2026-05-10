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

    description 에서 자주 쓰는 markdown 자동 변환:
      - **bold**     → <strong>bold</strong>
      - `code`       → <code>code</code>
    (HTML 박스 안에 markdown 직접 안 됨 → 변환 필요)
    """
    import re

    desc_html = description
    # **bold** → <strong>
    desc_html = re.sub(r"\*\*(.+?)\*\*", r"<strong style='color:#FAFAFA;'>\1</strong>", desc_html)
    # `code` → <code>
    desc_html = re.sub(
        r"`([^`]+?)`",
        r"<code style='background:#374151;padding:1px 5px;border-radius:3px;color:#FAFAFA;'>\1</code>",
        desc_html,
    )

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
                {desc_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """
    모든 페이지에서 동일한 사이드바 렌더링 (C-4 6 그룹 + 2 토글).

    Streamlit multi-page 에서 각 페이지마다 자체 사이드바를 그려야 함
    (`config.toml: showSidebarNavigation = false` 설정 시 자동 nav 비활성).

    호출 위치: 각 페이지 진입 시 inject_custom_css() / init_session_state() 다음.
    """
    with st.sidebar:
        # ── 펀드명 + 메타 (C4-3) ──
        st.markdown("# Adaptive VolControl Fund")
        st.markdown("어댑티브 볼컨트롤 펀드")
        st.caption("Benchmark: SPY  |  Data: 2025-12")
        st.divider()

        # ── 6 그룹 페이지 navigation (C4-1 c, C4-2 a) ──
        # 그룹 1: 개요
        st.markdown("##### ── 개요 ──")
        st.page_link("app.py", label="Overview", icon="📊")

        # 그룹 2: 체험 (★ Investment Simulator F-6)
        st.markdown("##### ── 체험 ──")
        st.page_link("pages/02_Investment_Simulator.py", label="Investment Simulator", icon="💵")

        # 그룹 3: 성과
        st.markdown("##### ── 성과 ──")
        st.page_link("pages/03_Performance.py", label="Performance", icon="📈")
        st.page_link("pages/04_Risk_Metrics.py", label="Risk Metrics", icon="⚠️")

        # 그룹 4: 보유
        st.markdown("##### ── 보유 ──")
        st.page_link("pages/05_Holdings.py", label="Holdings", icon="🏢")
        st.page_link("pages/06_Sector_Watch.py", label="Sector Watch", icon="🌐")

        # 그룹 5: 검증
        st.markdown("##### ── 검증 ──")
        st.page_link("pages/07_Methodology.py", label="Methodology", icon="🧪")
        st.page_link("pages/08_Backtesting.py", label="Backtesting", icon="✅")

        # 그룹 6: 메타
        st.markdown("##### ── 메타 ──")
        st.page_link("pages/09_About.py", label="About / FAQ", icon="ℹ️")

        st.divider()

        # ── 토글 1: 기간 (Period) — C4-4 ──
        st.subheader("📅 기간 (Period)")
        st.radio(
            "기간 선택",
            options=["FULL", "TEST", "HO"],
            index=["FULL", "TEST", "HO"].index(st.session_state.get("period", "FULL")),
            key="period",
            label_visibility="collapsed",
        )

        # ── 토글 2: 비교 벤치마크 — C4-4 ──
        st.subheader("📊 비교 (Benchmark)")
        st.checkbox("SPY", value=st.session_state.get("show_spy", True), key="show_spy")
        st.checkbox("EW (펀드 universe)", value=st.session_state.get("show_ew", False), key="show_ew")
        st.checkbox("IVW (Naive Low-vol)", value=st.session_state.get("show_ivw", False), key="show_ivw")
