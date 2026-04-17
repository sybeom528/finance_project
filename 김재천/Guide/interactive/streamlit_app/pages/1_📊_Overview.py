"""페이지 2: Overview — 프로젝트 전체 개요."""
import streamlit as st
import plotly.graph_objects as go
from utils.theme import render_theme_selector, apply_custom_css, get_current_theme
from utils.data_loader import load_metrics, load_final_recommendation


st.set_page_config(page_title='Overview', page_icon='📊', layout='wide')
render_theme_selector()
apply_custom_css()
theme = get_current_theme()

st.title('📊 Overview — 프로젝트 개요')
st.caption('11단계 파이프라인과 주요 성과를 한눈에')

st.markdown('---')


# ============================================================
# 1. Executive Summary
# ============================================================

st.markdown('## 🎯 한 문장 요약')
st.info(
    """
    **"대안데이터(VIX · VIX Contango · HY 스프레드 등)로 매일 경보를 발동해
    주식 비중을 자동 조절하면, Sharpe 1.064, MDD -15.5%, 8년 +151% 성과.
    복잡한 레짐 기반 공분산 전환(경로 2)은 실증 무효."**
    """
)


# ============================================================
# 2. 11단계 파이프라인
# ============================================================

st.markdown('## 🗺️ 11단계 파이프라인')

phases = [
    ('Phase 1: 데이터 준비', ['Step 1\n데이터 수집', 'Step 2\n전처리+Granger'],
     theme['primary']),
    ('Phase 2: 최적화 기반', ['Step 3\n최적화 개념', 'Step 4\nWF 백테스트', 'Step 5\n리스크 분석'],
     '#7E57C2'),
    ('Phase 3: 경보 시스템', ['Step 6\nHMM+경보', 'Step 7\nv3 Ablation'],
     theme['warning']),
    ('Phase 4: 심화 확장', ['Step 8\n레짐 Σ', 'Step 9\n64 시뮬', 'Step 10\n통계 검정'],
     theme['success']),
    ('Phase 5: 시각화', ['Step 11\nTop 10 해부'],
     '#EC407A'),
]

for phase_name, steps, color in phases:
    st.markdown(f'### {phase_name}')
    cols = st.columns(len(steps))
    for col, step_label in zip(cols, steps):
        step_num = step_label.split('\n')[0]
        step_desc = step_label.split('\n')[1]
        col.markdown(f"""
        <div style='background:{color};color:white;padding:15px;border-radius:8px;
             text-align:center;margin-bottom:10px;'>
            <strong>{step_num}</strong><br>
            <small>{step_desc}</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown('---')


# ============================================================
# 3. v3 vs 최종 변화표
# ============================================================

st.markdown('## 🔄 v3 → 최종 주요 변화')

changes = {
    '항목': ['파이프라인 Step', 'Baseline', '경로 2 구현', 'Ablation 규모',
              '통계 검정', '최우수 Sharpe', '문서화'],
    'v3': ['7개', 'Equal Weight 1/30', '❌ 미구현', '16 조합',
            'Bootstrap 95% CI', '1.473 (EW base)', 'report_v3.md'],
    '최종': ['**11개**', '**MV 최적화**', '**✅ 구현 (무효 확인)**', '**64 조합**',
              '**+Bonferroni +FDR +IR**', '**1.064 (MV base)**',
              'report_final + 해설 11개'],
}

import pandas as pd
st.table(pd.DataFrame(changes))

st.markdown('---')


# ============================================================
# 4. 5대 핵심 발견
# ============================================================

st.markdown('## 💡 5대 핵심 발견')

findings = [
    ('①', '경보 시스템의 실전 가치',
     'L0 주식 40% → L3 주식 16%의 체계적 감축 패턴 확인',
     theme['success']),
    ('②', '경로 2(Σ 전환)의 실증 무효',
     '두 차례 재설계 후에도 M3 < M1, 역효과 발견',
     theme['danger']),
    ('③', '단순성의 가치 (Occam\'s Razor)',
     'VIX + Contango 2변수(Config B)가 7지표 복합보다 우수',
     theme['primary']),
    ('④', '앵커-반응 이원 구조',
     '채권·금 = 안정 앵커, 주식 = 경보 반응 버퍼',
     '#00897B'),
    ('⑤', '위기 시 2~3배 방어력',
     'COVID SPY -34% vs 보수형 -7% (약 5배 방어)',
     '#FFB300'),
]

for num, title, desc, color in findings:
    st.markdown(f"""
    <div style='background:{theme['secondary_bg']};padding:15px;border-radius:8px;
         border-left:4px solid {color};margin-bottom:10px;'>
        <span style='color:{color};font-size:20px;font-weight:bold;'>{num}</span>
        <strong style='margin-left:10px;color:{theme['text']};'>{title}</strong><br>
        <small style='color:{theme['muted']};margin-left:32px;'>{desc}</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown('---')


# ============================================================
# 5. 벤치마크 비교 (인터랙티브 차트)
# ============================================================

st.markdown('## 📈 벤치마크 비교')

metrics = load_metrics()
final_rec = load_final_recommendation()

strategies = ['M1_보수형_ALERT_B', 'BENCH_EW', 'BENCH_SPY', 'BENCH_60_40']
labels = ['우리 전략\n(M1_보수형_B)', 'EW 1/30', 'SPY 100%', '60/40']
sharpes = [metrics.loc[s, 'sharpe'] for s in strategies]
colors = [theme['success'], theme['muted'], theme['muted'], theme['muted']]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=labels, y=sharpes,
    marker_color=colors,
    text=[f'{s:.3f}' for s in sharpes],
    textposition='outside',
))
fig.add_hline(y=1.0, line_dash='dash', line_color=theme['warning'],
              annotation_text='탁월 기준 1.0', annotation_position='right')
fig.update_layout(
    title='Sharpe Ratio 비교',
    yaxis_title='Sharpe Ratio',
    height=400,
    template=theme['plotly_template'] or 'plotly',
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 6. 관련 자료
# ============================================================

st.markdown('---')
st.markdown('## 📚 상세 자료')

col1, col2, col3 = st.columns(3)
col1.markdown("""
### 📖 보고서
- `report_final.md` — 통합 보고서
- `report_v3.md` — Step 1~7
- `report_v4.md` — Step 8~10
""")
col2.markdown("""
### 📋 빠른 참조
- `quick_reference/01` — One-Pager
- `quick_reference/04` — 플로우차트
- `quick_reference/12` — FAQ
""")
col3.markdown("""
### 📘 심화
- `docs/Step1~11_해설.md`
- `decision_log.md`
- `decision_log_v31.md`
""")
