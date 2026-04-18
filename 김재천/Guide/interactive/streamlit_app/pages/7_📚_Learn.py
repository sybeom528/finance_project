"""페이지 8: Learn — 용어 사전 + FAQ + 추천 학습 경로."""
import streamlit as st
from utils.theme import render_theme_selector, apply_custom_css, get_current_theme


st.set_page_config(page_title='Learn', page_icon='📚', layout='wide')
render_theme_selector()
apply_custom_css()
theme = get_current_theme()

st.title('📚 Learn — 용어 사전·FAQ·학습 경로')

st.markdown('---')


# ============================================================
# 탭 구성
# ============================================================

tab1, tab2, tab3 = st.tabs(['📖 용어 사전', '❓ FAQ', '🗺️ 학습 경로'])


# ============================================================
# 탭 1: 용어 사전
# ============================================================

with tab1:
    st.markdown('## 📖 용어 사전')
    st.caption('검색 + 카테고리 필터로 50+ 용어 탐색')

    GLOSSARY = {
        'Sharpe Ratio': {
            'category': '성과 지표',
            'definition': '연율수익률 / 연율변동성. 위험 대비 수익률의 표준 측정치.',
            'formula': 'Sharpe = E[R] / σ[R] × √252',
            'example': '우리 전략 1.064 (헤지펀드 평균 0.5~0.8)',
        },
        'MDD (Maximum Drawdown)': {
            'category': '성과 지표',
            'definition': '최고점 대비 최대 낙폭. 투자자가 겪은 최악의 손실.',
            'formula': 'MDD = min((cum - peak) / peak)',
            'example': '우리 전략 -15.53%, SPY -33.7%',
        },
        'VIX': {
            'category': '시장 지표',
            'definition': 'S&P 500 옵션 기반 30일 내재변동성. "공포지수".',
            'formula': 'CBOE 공식 계산',
            'example': '평상시 13~18, 2020 COVID 82',
        },
        'VIX Contango': {
            'category': '시장 지표',
            'definition': 'VIX3M - VIX. 양수면 정상, 음수면 백워데이션(공포).',
            'formula': 'Contango = VIX3M - VIX',
            'example': 'Contango < 0 → 단기 공포 > 장기 공포',
        },
        'Walk-Forward': {
            'category': '방법론',
            'definition': '과거 IS 데이터로 학습, 미래 OOS로 검증, 슬라이드 반복.',
            'formula': 'IS 24m → OOS 3m → slide 3m',
            'example': '우리 프로젝트: 31 윈도우',
        },
        'MV 최적화': {
            'category': '방법론',
            'definition': 'Markowitz 평균-분산 최적화. 수익↑ 위험↓ 균형.',
            'formula': 'max w\'μ - (γ/2) w\'Σw',
            'example': 'Step 3, 4, 9에서 사용',
        },
        'HMM (Hidden Markov Model)': {
            'category': '방법론',
            'definition': '관측 가능한 지표로 숨은 시장 상태 추정.',
            'formula': 'P(state | observations)',
            'example': 'BIC=5239로 4레짐 최적',
        },
        'Ledoit-Wolf 수축': {
            'category': '방법론',
            'definition': '공분산 추정 안정화. 표본 + 구조화 블렌딩.',
            'formula': 'Σ_hat = α × Σ_shrink + (1-α) × Σ_sample',
            'example': 'Step 4, 8, 9에서 사용',
        },
        '경로 1 (Path 1)': {
            'category': 'Final Project 핵심',
            'definition': '매일 경보 레벨 체크 → 주식 비중 즉시 축소.',
            'formula': 'cut = EQUITY_CUT[profile][alert_level]',
            'example': '효과적 (M1 Sharpe +0.17)',
        },
        '경로 2 (Path 2)': {
            'category': 'Final Project 핵심',
            'definition': 'HMM 레짐 → 공분산 전환 → MV 재최적화.',
            'formula': 'Σ = Σ_crisis if alert ≥ 2 else Σ_stable',
            'example': '두 차례 재설계 후에도 무효',
        },
        'M0 / M1 / M2 / M3': {
            'category': 'Final Project 핵심',
            'definition': '4개 모드: baseline / 경로1만 / 경로2만 / 통합.',
            'formula': '각 4성향 × 4Config = 16 조합',
            'example': 'M1이 최우수, M3 < M1',
        },
        'Information Ratio (IR)': {
            'category': '통계',
            'definition': '초과수익 / 추적오차 (연율화).',
            'formula': 'IR = mean(active) / std(active) × √252',
            'example': 'Grinold-Kahn: 0.3~0.5 양호, 1.0+ 예외적',
        },
        'Bootstrap': {
            'category': '통계',
            'definition': '복원추출 기반 분포 추정 (비모수적).',
            'formula': 'n_boot 번 resample → CI 산출',
            'example': '5,000회로 95% CI 계산',
        },
        'FDR (Benjamini-Hochberg)': {
            'category': '통계',
            'definition': 'False Discovery Rate 통제. Bonferroni보다 덜 보수적.',
            'formula': 'rank/n × α 임계',
            'example': '48개 비교 중 유의율 확인',
        },
    }

    # 검색 + 필터
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input('🔍 용어 검색', placeholder='예: Sharpe, VIX, HMM...')
    with col2:
        categories = sorted(set(v['category'] for v in GLOSSARY.values()))
        cat_filter = st.selectbox('카테고리', options=['전체'] + categories)

    filtered = GLOSSARY
    if search:
        s_lower = search.lower()
        filtered = {k: v for k, v in filtered.items()
                    if s_lower in k.lower() or s_lower in v['definition'].lower()}
    if cat_filter != '전체':
        filtered = {k: v for k, v in filtered.items() if v['category'] == cat_filter}

    st.markdown(f'### 결과: {len(filtered)}개')

    for term, info in filtered.items():
        with st.expander(f'📖 **{term}** ({info["category"]})'):
            st.markdown(f"**정의**: {info['definition']}")
            st.code(info['formula'], language='python')
            st.markdown(f"**예시**: {info['example']}")

    if not filtered:
        st.info('검색 결과가 없습니다. 다른 키워드나 카테고리를 시도해 보세요.')


# ============================================================
# 탭 2: FAQ
# ============================================================

with tab2:
    st.markdown('## ❓ 자주 묻는 질문')

    faqs = [
        ('Sharpe 1.064는 좋은 수치인가요?',
         '**상당히 우수합니다.** 헤지펀드 평균은 0.5~0.8, S&P 500 장기는 약 0.5입니다. Sharpe ≥ 1.0은 상위 수준입니다.'),
        ('왜 SPY보다 Total 수익이 낮은데 추천하나요?',
         'SPY +180% / 우리 +151%이지만, **변동성이 65%**밖에 안 됩니다. 같은 위험당 수익률(Sharpe)은 **1.06 vs 0.76**으로 우리가 40% 우수합니다.'),
        ('경로 2가 왜 실패했나요?',
         '3가지 원인: (1) Σ 차이가 MV 비중에 미약하게 반영, (2) 월별 재최적화 비용 > 이론적 이득, (3) 경로 1과 정보 중복. 자세히는 `quick_reference/05_path1_vs_path2.md` 참조.'),
        ('실전 운용에 얼마나 시간 걸리나요?',
         '평균 **하루 10분** (VIX 체크), 경보 발동 시 30분, 분기 리밸런싱 1시간. 월 약 6.6시간 예상.'),
        ('최소 자본은 얼마인가요?',
         '권장 **$30,000 (약 4천만원)** — 30자산 분산 + 거래비용 효율 고려. 이상적 $100,000+.'),
        ('경보가 울렸을 때 무엇을 해야 하나요?',
         'Config B 기준: L1 → 주식 15% 감축, L2 → 35%, L3 → 60%. 감축분은 채권 70% + 금 30%로 재배분. 자세히는 `quick_reference/06_decision_tree.md`.'),
        ('향후 이 성과가 지속될까요?',
         '**불확실합니다.** 2018~2025는 COVID·긴축 등 특이 기간. 구조적 방어력은 유지될 가능성 높으나 실전 운용 전 소액 테스트 권장.'),
        ('다른 시장(한국, 일본)에도 적용 가능한가요?',
         '**미검증**. v5에서 글로벌 확장 예정. 현재는 미국 시장에만 적용.'),
    ]

    for q, a in faqs:
        with st.expander(f'❓ {q}'):
            st.markdown(a)


# ============================================================
# 탭 3: 학습 경로
# ============================================================

with tab3:
    st.markdown('## 🗺️ 독자별 학습 경로')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 👤 비전문가 투자자

        **30분 빠른 이해**:
        1. 이 앱 🏠 Home
        2. `quick_reference/01_executive_one_pager.md`
        3. `quick_reference/09_timeline_narrative.md`

        **1~2시간 심화**:
        4. `quick_reference/02_investor_summary_card.md`
        5. `quick_reference/08_crisis_case_studies.md`
        6. `quick_reference/10_day_in_life.md`
        7. `quick_reference/13_operating_checklist.md`

        **실전 준비**:
        8. 🎮 이 앱 Simulator 페이지
        9. 소액 테스트 (1~3개월)
        """)

    with col2:
        st.markdown("""
        ### 💻 퀀트 엔지니어

        **기술 개요**:
        1. `report_final.md`
        2. `decision_log.md` + `decision_log_v31.md`
        3. 이 앱 📊 Overview

        **구현 상세**:
        4. Step 1~11 노트북 순차 실행
        5. `docs/Step8~10_해설.md` (확장 핵심)
        6. 코드 리뷰: `_build_step*.py`

        **확장**:
        7. `quick_reference/07_data_erd.md`
        8. v5 후보 (report_final Section 7)
        """)

    with col3:
        st.markdown("""
        ### 🔬 연구자

        **방법론 심화**:
        1. `stats_model.md`
        2. `docs/Step10_해설.md` — 통계 검정
        3. `decision_log_v31.md` Section 13 — 경로 2 실패 분석

        **학술 참조**:
        - Markowitz (1952) — MV 이론
        - Ledoit & Wolf (2004) — 수축
        - Benjamini & Hochberg (1995) — FDR
        - Grinold & Kahn (2000) — IR

        **Negative Result**:
        - 경로 2 실증 무효성 문서화
        - Cohen's d 재무 부적합성
        """)


# ============================================================
# 참고
# ============================================================

st.markdown('---')

with st.expander('📂 파일 지도'):
    st.markdown("""
    ```
    Guide/
    ├── 📓 노트북 (11개)
    │   └── Step1~11_*.ipynb
    ├── 📊 보고서 (3종)
    │   ├── report_v3.md
    │   ├── report_v4.md
    │   └── report_final.md
    ├── 📋 설계 기록 (3종)
    │   ├── decision_log.md
    │   ├── decision_log_v31.md
    │   └── stats_model.md
    ├── 📘 docs/ (11개 해설)
    │   └── Step1~11_해설.md
    ├── 📋 quick_reference/ (13종)
    │   ├── 01_executive_one_pager.md
    │   ├── 02_investor_summary_card.md
    │   ├── ... (total 13)
    │   └── 13_operating_checklist.md
    ├── 💾 data/ (CSV + PKL)
    ├── 🖼️ images/ (35+ PNG)
    └── 🎮 interactive/
        ├── dashboard.html
        └── streamlit_app/ (현재 앱)
    ```
    """)
