"""
Final Project — 인터랙티브 Streamlit 앱 (홈)
실행: streamlit run app.py
"""
import streamlit as st
from utils.theme import apply_custom_css, render_theme_selector, get_current_theme
from utils.data_loader import load_final_recommendation, load_metrics


# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title='Final Project — 포트폴리오 시뮬레이터',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ============================================================
# 사이드바 (테마 + 네비게이션)
# ============================================================

render_theme_selector()
apply_custom_css()

st.sidebar.markdown('---')
st.sidebar.markdown('### 📂 페이지')
st.sidebar.info("""
- 🏠 **홈** (현재)
- 📊 Overview
- 🏆 Top 10 Strategies
- 📈 Composition
- ⚠️ Alerts
- 🌪️ Crisis Cases
- 🎮 Simulator
- 📚 Learn
""")

st.sidebar.markdown('---')
st.sidebar.markdown('### ℹ️ 정보')
st.sidebar.markdown("""
- **프로젝트**: Final Project
- **기간**: 2016~2025 (10년)
- **자산**: 30종
- **전략**: 64개 시뮬
""")


# ============================================================
# 본문 — 헤더
# ============================================================

st.title('🎯 Final Project — 대안데이터 기반 포트폴리오 시뮬레이터')
st.caption('대안데이터(VIX·HY 등)로 매일 경보를 발동해 주식 비중을 자동 조절하는 전략 · 한눈에 탐색')

st.markdown('---')


# ============================================================
# KPI Cards
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label='Sharpe Ratio',
        value='1.064',
        delta='+29% vs EW',
        help='위험 대비 수익률. 1.0 이상 = 우수'
    )

with col2:
    st.metric(
        label='최대 낙폭 (MDD)',
        value='-15.53%',
        delta='SPY의 절반',
        delta_color='inverse',
        help='최악 손실 구간'
    )

with col3:
    st.metric(
        label='8년 누적수익',
        value='+151%',
        delta='연 +12.24%',
        help='2018-01 ~ 2025-12 OOS'
    )

with col4:
    st.metric(
        label='연율 변동성',
        value='11.41%',
        delta='SPY 65% 수준',
        delta_color='inverse',
        help='표준편차 × √252'
    )


# ============================================================
# 핵심 요약
# ============================================================

st.markdown('---')
st.markdown('## 🏆 최우수 전략: **M1_보수형_ALERT_B**')

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 📌 한 문장 요약
    > **"VIX와 VIX Contango 2개 지표로 매일 경보를 판정하여,
    > 위기 시 주식 비중을 15~60% 감축하고 채권·금으로 이동하는 보수형 포트폴리오"**

    ### 🎛️ 구성
    - **성향**: 보수형 (γ=8, max_equity 43%, min_bond 31%)
    - **모드**: M1 (경로 1만, 경로 2는 실증 무효)
    - **Config**: B (VIX + VIX Contango)

    ### 💡 5대 핵심 발견
    1. **경보 시스템의 실전 가치** — L0 주식 40% → L3 주식 16%
    2. **경로 2(Σ 전환)의 무효성** — 두 차례 재설계 후에도 역효과
    3. **단순성의 가치** — 2변수(Config B)가 7변수(C)보다 우수
    4. **앵커-반응 이원 구조** — 채권·금 = 앵커, 주식 = 반응
    5. **위기 시 2~3배 방어력** — COVID SPY -34% vs 우리 -7%
    """)

with col2:
    st.markdown("""
    ### 🚦 경보 규칙
    | 레벨 | VIX | 주식 감축 |
    |:---:|:---:|:---:|
    | 🟢 L0 | < 20 | 유지 |
    | 🟡 L1 | 20-28 | **15%** |
    | 🟠 L2 | 28-35 | **35%** |
    | 🔴 L3 | ≥ 35 | **60%** |

    *Contango < 0이면 +1 레벨*

    ### ⏱️ 운영 시간
    - 매일 10분 (VIX 체크)
    - 경보 발동 30분
    - 분기 1시간 리밸런싱
    """)


# ============================================================
# 탐색 가이드
# ============================================================

st.markdown('---')
st.markdown('## 🧭 이 앱 탐색 가이드')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 👤 비전문가 투자자
    1. 📊 **Overview** — 파이프라인 이해
    2. 🏆 **Top 10** — 추천 전략 비교
    3. 🌪️ **Crisis** — 위기 대응 사례
    4. 🎮 **Simulator** — 나만의 시뮬
    """)

with col2:
    st.markdown("""
    ### 💻 퀀트 엔지니어
    1. 📊 **Overview** — 전체 구조
    2. 📈 **Composition** — 비중 변화
    3. ⚠️ **Alerts** — 경보 시스템
    4. 📚 **Learn** — 상세 문서
    """)

with col3:
    st.markdown("""
    ### 📢 발표·공유
    1. 🏠 **Home** (현재)
    2. 🏆 **Top 10** — 핵심 성과
    3. 🌪️ **Crisis** — 스토리텔링
    4. 📚 **Learn** — FAQ
    """)


# ============================================================
# 최근 업데이트
# ============================================================

st.markdown('---')

with st.expander('🆕 Final Project 주요 업데이트 (2026-04-17)'):
    st.markdown("""
    ### 주요 변경
    - **Step 11 추가**: Top 10 전략 자산구성 시간 변화 시각화 8종
    - **보고서 추가**: `report_final.md` 전체 11단계 통합
    - **해설 문서**: `docs/Step1~11_해설.md` (11개 비전문가 해설)
    - **Quick Reference**: 13종 빠른 참조 자료
    - **Streamlit 앱**: 본 앱 (8 페이지, 3 테마)

    ### 이전 대비 차이
    - 초기 (Sharpe 1.473, EW baseline) → 최종 (Sharpe 1.064, MV baseline)
    - 경로 2 실증 무효성 확인
    - Cohen's d → IR/ΔSR 교체
    """)


# ============================================================
# Footer
# ============================================================

st.markdown('---')
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
        🎯 Final Project · 대안데이터 기반 포트폴리오 시뮬레이터 · 김재천 (2026-04-17)<br>
        📂 전체 자료: <code>Guide/</code> | 📘 해설: <code>docs/</code> | 📋 요약: <code>quick_reference/</code>
    </div>
    """,
    unsafe_allow_html=True
)
