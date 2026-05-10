"""
streamlit_app.py — Low-Risk BL 대시보드 메인 진입점.

실행:
    cd final && streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys

import streamlit as st

# Final/ 경로를 sys.path 에 추가 (bl_functions / master_table import 용)
APP_DIR = Path(__file__).resolve().parent
FINAL_DIR = APP_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from app.lib.data_loader import is_data_ready  # noqa: E402
from app.lib.narrative import HOME_INTRO, TOP_1_NAME  # noqa: E402


st.set_page_config(
    page_title='Low-Risk BL Dashboard',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'About': '''
**Low-Risk Anomaly + LSTM σ + Black-Litterman 통합 시연**

S&P 500 종목의 저변동성 anomaly + LSTM 변동성 예측 + BL 단일 view
프레임워크로 위험성향별 펀드 후보를 탐색하는 quant 프로젝트.

추천 Top 1: `mat_eq_eq_lam_pap` (사용자 지정)

소스: feature/streamlit-dashboard branch
        '''
    },
)


# ── theme.css 적용 (있으면) ─────────────────────────────────
_css_path = APP_DIR / 'assets' / 'theme.css'
if _css_path.exists():
    st.markdown(f'<style>{_css_path.read_text(encoding="utf-8")}</style>',
                unsafe_allow_html=True)


# ── 데이터 readiness 점검 ────────────────────────────────────
ready, msg = is_data_ready()
if not ready:
    st.error(f'⚠ 대시보드 실행 불가: {msg}')
    st.stop()


# ── Home 본문 ───────────────────────────────────────────────
st.markdown(HOME_INTRO)

st.info(
    f'👈 좌측 사이드바에서 페이지를 선택하세요. '
    f'**Overview** 페이지에서 Top 1 후보 (`{TOP_1_NAME}`) 의 KPI 와 '
    f'TEST vs HOLD-OUT 정합성을 먼저 확인하시는 것을 추천드립니다.'
)

with st.expander('📋 페이지 안내', expanded=True):
    st.markdown('''
| 페이지 | 역할 | 주 사용자 |
|---|---|---|
| **📊 Overview** | KPI hero + Top 1 추천 + 누적수익률 | 발표 첫 슬라이드 |
| **🔍 Experiment Explorer** | 165 실험 표 + 단일 실험 6 패널 | 자유 탐색 |
| **⚖️ Compare** | 다중 실험 비교 6 패널 (Sortino 강조) | 슬롯 효과 검증 |
| **🗺️ Matrix Heatmap** | 135-cell 매트릭스 히트맵 | 슬롯 trade-off |
| **🎯 Risk Profile** | 보수/중립/공격 슬라이더 + Top 3 | 발표 어필 |
| **📈 Volatility & Stats** | LSTM 예측 + 학술 통계 요약 | 검증 backstop |
''')

st.divider()

st.caption(f'📂 BASE_DIR : `{FINAL_DIR}`  ·  data ready: ✅')
