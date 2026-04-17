"""페이지 5: Alerts — 4 Config 경보 시스템 비교."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.theme import render_theme_selector, apply_custom_css, get_current_theme
from utils.data_loader import load_alerts, ALERT_COLORS


st.set_page_config(page_title='Alerts', page_icon='⚠️', layout='wide')
render_theme_selector()
apply_custom_css()
theme = get_current_theme()

st.title('⚠️ Alerts — 경보 시스템 비교')
st.caption('4가지 Config (A/B/C/D)의 경보 시계열과 정밀도')

st.markdown('---')

alerts = load_alerts()


# ============================================================
# Config 설명
# ============================================================

st.markdown('## 📋 4 Config 비교')

config_info = pd.DataFrame({
    'Config': ['A', 'B', 'C', 'D'],
    '정의': [
        'VIX 단독 (20/28/35)',
        'A + VIX Contango < 0 → +1',
        '7지표 복합 + 롤링 분위',
        'C + 5일 디바운스',
    ],
    '특징': [
        '가장 단순',
        '2변수, 실측 우수',
        '정교, 복잡',
        'False alarm 감소',
    ],
    '정밀도 (5일 예측)': ['62%', '**70%**', '68%', '**72%**'],
    '실측 성과': ['⭐⭐⭐', '⭐⭐⭐⭐⭐ (최우수)', '⭐⭐⭐⭐', '⭐⭐⭐'],
})
st.dataframe(config_info, use_container_width=True, hide_index=True)


# ============================================================
# 기간 선택
# ============================================================

st.markdown('---')
st.markdown('## 📊 경보 타임라인')

min_date = alerts.index.min().to_pydatetime().date()
max_date = alerts.index.max().to_pydatetime().date()

col1, col2 = st.columns([3, 1])
with col1:
    date_range = st.date_input(
        '기간 선택',
        value=(pd.Timestamp('2018-01-01').date(), pd.Timestamp('2025-12-31').date()),
        min_value=min_date, max_value=max_date,
    )
with col2:
    show_configs = st.multiselect(
        '표시할 Config',
        options=['ALERT_A', 'ALERT_B', 'ALERT_C', 'ALERT_D'],
        default=['ALERT_A', 'ALERT_B', 'ALERT_C', 'ALERT_D'],
    )

if len(date_range) == 2:
    start, end = date_range
    alerts_f = alerts.loc[str(start):str(end)]
else:
    alerts_f = alerts


# ============================================================
# 경보 시계열 비교 차트
# ============================================================

fig = go.Figure()
config_col_map = {'ALERT_A': 'alert_a', 'ALERT_B': 'alert_b',
                   'ALERT_C': 'alert_c', 'ALERT_D': 'alert_d'}
config_colors_map = {'ALERT_A': '#808080', 'ALERT_B': theme['primary'],
                      'ALERT_C': theme['warning'], 'ALERT_D': theme['success']}

for config in show_configs:
    col = config_col_map[config]
    if col in alerts_f.columns:
        fig.add_trace(go.Scatter(
            x=alerts_f.index, y=alerts_f[col].astype(int).values,
            name=config, mode='lines',
            line=dict(color=config_colors_map[config], width=1, shape='hv'),
        ))

fig.update_layout(
    title='Config별 경보 레벨 시계열',
    yaxis=dict(
        tickmode='array', tickvals=[0, 1, 2, 3],
        ticktext=['L0 정상', 'L1 주의', 'L2 경계', 'L3 위기']
    ),
    hovermode='x unified', height=500,
    template=theme['plotly_template'] or 'plotly',
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Config별 경보 분포
# ============================================================

st.markdown('## 📊 Config별 경보 분포')

cols = st.columns(4)
for i, config in enumerate(['ALERT_A', 'ALERT_B', 'ALERT_C', 'ALERT_D']):
    with cols[i]:
        col = config_col_map[config]
        if col in alerts_f.columns:
            counts = alerts_f[col].astype(int).value_counts().sort_index()
            total = counts.sum()

            fig_donut = go.Figure(data=[go.Pie(
                labels=[f'L{lvl}' for lvl in counts.index],
                values=counts.values,
                marker_colors=[ALERT_COLORS[lvl] for lvl in counts.index],
                hole=0.5,
                textinfo='label+percent',
            )])
            fig_donut.update_layout(
                title=f'{config}',
                height=300,
                showlegend=False,
                template=theme['plotly_template'] or 'plotly',
                annotations=[dict(text=f'{total}일', x=0.5, y=0.5,
                                   font_size=14, showarrow=False)],
            )
            st.plotly_chart(fig_donut, use_container_width=True)


# ============================================================
# VIX 및 관련 지표
# ============================================================

st.markdown('---')
st.markdown('## 📈 경보 생성에 쓰인 지표')

indicator_cols = ['VIX_level', 'VIX_contango', 'HY_spread', 'yield_curve']
available = [c for c in indicator_cols if c in alerts_f.columns]

if available:
    selected_ind = st.selectbox('지표 선택', options=available)

    fig_ind = go.Figure()
    fig_ind.add_trace(go.Scatter(
        x=alerts_f.index, y=alerts_f[selected_ind].values,
        mode='lines', line=dict(color=theme['primary'], width=1),
        name=selected_ind,
    ))

    # 주요 임계값
    if selected_ind == 'VIX_level':
        for thr, label, color in [(20, 'L1', 'yellow'), (28, 'L2', 'orange'), (35, 'L3', 'red')]:
            fig_ind.add_hline(y=thr, line_dash='dash', line_color=color,
                               annotation_text=label, annotation_position='right')
    elif selected_ind == 'VIX_contango':
        fig_ind.add_hline(y=0, line_dash='dash', line_color='red',
                           annotation_text='백워데이션 경계', annotation_position='right')

    fig_ind.update_layout(
        title=f'{selected_ind} 시계열',
        xaxis_title='날짜', yaxis_title=selected_ind,
        height=400, template=theme['plotly_template'] or 'plotly',
    )
    st.plotly_chart(fig_ind, use_container_width=True)


# ============================================================
# 경보 변동 통계
# ============================================================

st.markdown('---')
st.markdown('## 🔄 경보 변동 통계')

col1, col2 = st.columns(2)

with col1:
    st.markdown('### Config별 레벨 변경 횟수')
    change_stats = {}
    for config in ['ALERT_A', 'ALERT_B', 'ALERT_C', 'ALERT_D']:
        col = config_col_map[config]
        if col in alerts_f.columns:
            changes = (alerts_f[col].astype(int).diff() != 0).sum()
            change_stats[config] = changes

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=list(change_stats.keys()), y=list(change_stats.values()),
        marker_color=[config_colors_map[c] for c in change_stats.keys()],
        text=list(change_stats.values()), textposition='outside',
    ))
    fig_bar.update_layout(
        height=350, template=theme['plotly_template'] or 'plotly',
        yaxis_title='변경 횟수', showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown('### 경보 유지 기간 (평균)')
    st.info("""
    Config D는 5일 디바운스로 변경 빈도가 낮지만 유지 기간이 김.
    Config A는 변동 많으나 짧게 유지.

    **실전 추천**: Config B (변동 적당 + 신뢰 높음)
    """)


# ============================================================
# 팁
# ============================================================

st.markdown('---')
with st.expander('💡 해석 팁'):
    st.markdown("""
    ### Config B가 왜 최우수?
    - **VIX + Contango 2변수**가 단순하면서도 효과적
    - VIX 수준 + 기간구조 변화 = 시장 공포의 **강도**와 **긴급성** 모두 포착
    - Config C(7지표)보다 **과적합 위험 낮음**

    ### Config D의 역설
    - 디바운스로 false alarm 감소 → **단기 위기 놓침**
    - Step 10 Bootstrap: Config D는 Config A보다 오히려 악화 (보수형/중립형 기준)

    ### 언제 어떤 Config?
    - **보수형·중립형** → Config B 추천
    - **적극형** → Config C (더 많은 경보 수용)
    - 실전 신규 투자자 → **Config B** 먼저 익히기
    """)
