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

        # ── 🧪 model-comparison branch 전용: 펀드 모델 선택 ──
        _render_model_selector()
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


def _render_model_selector() -> None:
    """
    🧪 model-comparison branch 전용 — 사이드바 펀드 모델 선택 UI.

    구성:
      1. selectbox (검색 가능, 156 config 중 선택)
      2. expander "📊 156 config 메트릭 비교 표"

    페이지 간 선택 유지를 위한 설계 (streamlit multipage quirk 회피):
      - logical key   = `st.session_state.config_name` (페이지 entry 들이 읽음)
      - widget key    = `model_selector_widget` (selectbox 자체)
      - 변경 시점에 logical ← widget 명시 동기화
      → page_link 이동으로 widget instance 가 다시 그려져도 logical key 유지
    """
    from lib.data_loader import list_available_configs, load_all_config_metrics

    st.markdown("#### 🧪 펀드 모델 (실험)")
    st.caption("`model-comparison` branch 에서만 표시 — main 은 mat_eq_eq_raw_pap 단일 모델")

    configs = list_available_configs()

    # logical key 보장
    if "config_name" not in st.session_state or st.session_state.config_name not in configs:
        st.session_state.config_name = (
            "mat_eq_eq_raw_pap" if "mat_eq_eq_raw_pap" in configs else configs[0]
        )
    current = st.session_state.config_name

    # selectbox — widget key 분리 + 변경 시 logical key 동기화 (callback)
    def _sync_logical_from_widget() -> None:
        st.session_state.config_name = st.session_state.model_selector_widget

    st.selectbox(
        "모델 선택",
        options=configs,
        index=configs.index(current),
        key="model_selector_widget",
        on_change=_sync_logical_from_widget,
        help="텍스트 입력으로 검색 가능 (예: 'lstm', 'mat_eq', 'capm_mcap')",
    )

    # 디버그 표시 — 현재 활성 모델 (페이지 이동해도 유지되는지 확인용)
    st.caption(f"활성 모델: `{st.session_state.config_name}`")

    # 메트릭 비교 표 (expander)
    with st.expander("📊 156 config 메트릭 비교 표", expanded=False):
        try:
            df = load_all_config_metrics()
            # 검색 필터
            search = st.text_input(
                "검색 (config 이름 contains)",
                value="",
                key="config_search",
                placeholder="예: lstm, capm, mcap, raw",
            )
            if search:
                df = df[df.index.str.contains(search, case=False, na=False)]

            # 표시
            st.dataframe(
                df.style.format({
                    "CAGR": "{:.2%}", "Vol": "{:.2%}", "MDD": "{:.2%}",
                    "Sharpe": "{:.3f}", "Sortino": "{:.3f}", "Beta": "{:.3f}",
                }),
                use_container_width=True,
                height=300,
            )
            st.caption(
                f"총 {len(df)} config (FULL 192m). "
                f"컬럼 클릭 → 정렬, 행 더블클릭 → ticker 복사 후 위 selectbox 에 붙여넣기."
            )
        except Exception as e:
            st.warning(f"메트릭 표 로드 실패: {e}")
