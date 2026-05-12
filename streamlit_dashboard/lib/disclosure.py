"""
lib/disclosure.py - Footer / Disclaimer / Session state 표준

E-3 + I-2 + I-5 결정 통합:
  - 모든 페이지 하단 통일 Footer (Disclosure / Data Sources / Meta)
  - Investment Simulator 상단 disclaimer 박스
  - Session state 초기화 (사이드바 토글 값 보존)

참조: docs/decisionlog/11_dl_sections.md E/I 섹션, plan/02_common.md 4절, 1.4절
"""

import streamlit as st


# === 표준 Footer 텍스트 (E-3 + I-2) ===================================
FOOTER_DISCLOSURE = """
※ 본 결과는 백테스트 시뮬레이션이며 실제 운용 성과를 보장하지 않습니다.
   데이터 기간: 2010-01 ~ 2025-12 (TEST 평가 168개월 + Hold Out 24개월)
"""


# === Investment Simulator 상단 disclaimer (I-5) =======================
SIMULATOR_DISCLAIMER = """
⚠️ 본 시뮬레이션은 가상의 백테스트 결과이며, 실제 투자 권유 또는
   투자 자문을 목적으로 하지 않습니다. 과거의 성과는 미래의 수익을
   보장하지 않습니다.
"""


# === HO 라벨 통일 (E-1) ================================================
# 모든 페이지에서 Hold Out 구간 표기를 동일하게 유지 (사용자 표시용)
# 내부 EVAL_PERIODS dict key 는 여전히 "HOLD_OUT" (호환성 보존)
HO_LABEL = "Hold Out 24개월 (2024-2025)"


# === Session state 초기화 ==============================================
def init_session_state() -> None:
    """
    모든 페이지에서 공유하는 session state 초기화.
    사이드바 토글 (기간 / 벤치마크) 의 기본값 설정.

    app.py 진입 시 1회 호출. 이미 값이 있으면 덮어쓰지 않음.
    """
    if "period" not in st.session_state:
        st.session_state.period = "TEST"
    if "show_spy" not in st.session_state:
        st.session_state.show_spy = True
    if "show_ew" not in st.session_state:
        st.session_state.show_ew = False
    if "show_ivw" not in st.session_state:
        st.session_state.show_ivw = False


# === Footer 렌더링 =====================================================
def render_footer() -> None:
    """
    모든 페이지 하단에 통일 Footer 표시 (3 컬럼: Disclosure / Sources / Meta).
    Overview 영역 6 + 다른 페이지 동일 패턴.
    """
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("**Disclosure**")
        st.caption(FOOTER_DISCLOSURE)

    with col2:
        st.caption("**Data Sources**")
        st.caption(
            "- Yahoo Finance (Adj Close)\n"
            "- Fama-French Library (FF5 factors)\n"
            "- FRED (Risk-free rate)\n"
            "- Methodology: Black-Litterman + LSTM 4-slot"
        )

    with col3:
        st.caption("**Meta**")
        st.caption(
            "- Last updated: 2026-05-12\n"
            "- Adaptive VolControl Fund (가상 펀드)\n"
            "- 부트캠프 최종 프로젝트"
        )


# === Sim 페이지 disclaimer 박스 ========================================
def render_simulator_disclaimer() -> None:
    """Investment Simulator 영역 2 하단 / 영역 3 상단 사이에 표시."""
    st.warning(SIMULATOR_DISCLAIMER)
