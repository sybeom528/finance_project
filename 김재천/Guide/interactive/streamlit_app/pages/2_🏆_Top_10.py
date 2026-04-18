"""페이지 3: Top 10 Strategies — 최우수 전략 상세 비교."""
import streamlit as st
import plotly.graph_objects as go
from utils.theme import render_theme_selector, apply_custom_css, get_current_theme
from utils.data_loader import (
    load_final_recommendation, load_step9_results, load_metrics, parse_key
)


st.set_page_config(page_title='Top 10', page_icon='🏆', layout='wide')
render_theme_selector()
apply_custom_css()
theme = get_current_theme()

st.title('🏆 Top 10 Strategies — 최우수 전략 상세')
st.caption('Multi-criteria Decision 기반 최상위 10 전략 탐색')

st.markdown('---')


# 데이터 로드
final_rec = load_final_recommendation()
step9 = load_step9_results()
results = step9['results']
metrics = load_metrics()

TOP10 = final_rec.head(10).index.tolist()


# ============================================================
# 1. Top 10 순위표
# ============================================================

st.markdown('## 📋 Top 10 순위표')

display_df = final_rec.head(10)[
    ['total_return', 'ann_return', 'sharpe', 'mdd', 'sortino', 'calmar', 'multi_score']
].copy()
display_df.columns = ['누적수익', '연율수익', 'Sharpe', 'MDD', 'Sortino', 'Calmar', 'Multi-Score']
display_df.index.name = '전략'
# 포맷
display_df['누적수익'] = display_df['누적수익'].map('{:+.2%}'.format)
display_df['연율수익'] = display_df['연율수익'].map('{:+.2%}'.format)
display_df['Sharpe'] = display_df['Sharpe'].map('{:.3f}'.format)
display_df['MDD'] = display_df['MDD'].map('{:.2%}'.format)
display_df['Sortino'] = display_df['Sortino'].map('{:.3f}'.format)
display_df['Calmar'] = display_df['Calmar'].map('{:.3f}'.format)
display_df['Multi-Score'] = display_df['Multi-Score'].map('{:.2f}'.format)

st.dataframe(display_df, use_container_width=True)


# ============================================================
# 2. 전략 구성 분석
# ============================================================

st.markdown('## 🎛️ Top 10 구성 분석')

col1, col2, col3 = st.columns(3)

modes_count = {'M0': 0, 'M1': 0, 'M2': 0, 'M3': 0}
profiles_count = {'보수형': 0, '중립형': 0, '적극형': 0, '공격형': 0}
configs_count = {'ALERT_A': 0, 'ALERT_B': 0, 'ALERT_C': 0, 'ALERT_D': 0}

for key in TOP10:
    m, p, c = parse_key(key)
    modes_count[m] += 1
    profiles_count[p] += 1
    configs_count[c] += 1

with col1:
    st.markdown('### 모드별')
    for mode, count in modes_count.items():
        pct = count / 10 * 100
        color = {'M0': theme['muted'], 'M1': theme['primary'],
                  'M2': theme['warning'], 'M3': theme['danger']}.get(mode, 'gray')
        st.markdown(f"""
        <div class='metric-card' style='border-left-color:{color};'>
            <div class='label'>{mode}</div>
            <div class='value' style='color:{color};'>{count}개 ({pct:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('### 성향별')
    for profile, count in profiles_count.items():
        if count == 0:
            continue
        st.markdown(f"""
        <div class='metric-card'>
            <div class='label'>{profile}</div>
            <div class='value'>{count}개</div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown('### Config별')
    for config, count in configs_count.items():
        if count == 0:
            continue
        st.markdown(f"""
        <div class='metric-card'>
            <div class='label'>{config}</div>
            <div class='value'>{count}개</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 3. 전략 선택 및 상세 비교
# ============================================================

st.markdown('---')
st.markdown('## 🔍 선택 전략 상세')

selected = st.multiselect(
    '비교할 전략 선택 (최대 5개)',
    options=TOP10,
    default=TOP10[:3],
    max_selections=5,
)

if selected:
    # 누적수익률 비교 차트
    fig = go.Figure()
    for key in selected:
        cum = (1 + results[key]).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name=key,
            mode='lines', line=dict(width=1.5)
        ))
    # 벤치마크 오버레이
    for bench in ['BENCH_SPY', 'BENCH_EW']:
        cum = (1 + results[bench]).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name=bench.replace('BENCH_', ''),
            mode='lines', line=dict(width=1, dash='dash')
        ))
    fig.update_layout(
        title=f'누적수익률 비교 ({len(selected)}개 전략)',
        xaxis_title='날짜', yaxis_title='누적 자본',
        hovermode='x unified', height=500,
        template=theme['plotly_template'] or 'plotly',
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown 비교
    fig2 = go.Figure()
    for key in selected:
        cum = (1 + results[key]).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak * 100
        fig2.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=key,
            mode='lines', line=dict(width=1)
        ))
    fig2.update_layout(
        title='Drawdown 비교',
        xaxis_title='날짜', yaxis_title='Drawdown (%)',
        hovermode='x unified', height=400,
        template=theme['plotly_template'] or 'plotly',
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 상세 지표 테이블
    st.markdown('### 📊 상세 지표 비교')
    compare_df = metrics.loc[selected][
        ['total_return', 'ann_return', 'ann_vol', 'sharpe', 'sortino', 'mdd', 'calmar']
    ].copy()
    compare_df.columns = ['누적수익', '연율수익', '연율변동성', 'Sharpe', 'Sortino', 'MDD', 'Calmar']
    for col in ['누적수익', '연율수익', '연율변동성', 'MDD']:
        compare_df[col] = compare_df[col].map('{:.2%}'.format)
    for col in ['Sharpe', 'Sortino', 'Calmar']:
        compare_df[col] = compare_df[col].map('{:.3f}'.format)
    st.dataframe(compare_df, use_container_width=True)


# ============================================================
# 4. Sharpe × MDD 버블 차트
# ============================================================

st.markdown('---')
st.markdown('## 🎯 Sharpe × MDD 산점도')
st.caption('버블 크기 = 총수익률 · 파랑 = M1 · 빨강 = M3')

sharpe = [final_rec.loc[k, 'sharpe'] for k in TOP10]
mdd = [final_rec.loc[k, 'mdd'] * 100 for k in TOP10]
total = [final_rec.loc[k, 'total_return'] * 100 for k in TOP10]
colors_by_mode = [theme['primary'] if 'M1_' in k else theme['danger'] for k in TOP10]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=mdd, y=sharpe, mode='markers+text',
    text=[f'#{i+1}' for i in range(10)], textposition='top center',
    marker=dict(size=[max(15, t/4) for t in total], color=colors_by_mode,
                line=dict(color='white', width=2), opacity=0.75),
    hovertext=[f'{k}<br>Sharpe: {s:.3f}<br>MDD: {m:.2f}%<br>Total: {t:.1f}%'
               for k, s, m, t in zip(TOP10, sharpe, mdd, total)],
    hoverinfo='text',
))
fig.update_layout(
    title='Top 10 전략 Sharpe × MDD',
    xaxis_title='MDD (%)', yaxis_title='Sharpe Ratio',
    height=500, template=theme['plotly_template'] or 'plotly',
)
st.plotly_chart(fig, use_container_width=True)
