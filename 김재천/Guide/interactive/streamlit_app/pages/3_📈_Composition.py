"""페이지 4: Composition — 선택 전략의 자산 비중 시간 변화."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.theme import render_theme_selector, apply_custom_css, get_current_theme
from utils.data_loader import (
    load_step11_weights, load_final_recommendation,
    EQUITY, BOND, GOLD, ALERT_COLORS, parse_key
)


st.set_page_config(page_title='Composition', page_icon='📈', layout='wide')
render_theme_selector()
apply_custom_css()
theme = get_current_theme()

st.title('📈 Composition — 자산 비중 시간 변화')
st.caption('선택한 Top 10 전략의 30개 자산 비중이 시간에 따라 어떻게 바뀌었는지')

st.markdown('---')


# 데이터 로드
step11 = load_step11_weights()
final_rec = load_final_recommendation()
TOP10 = step11['top10_keys']


# ============================================================
# 사이드바 필터
# ============================================================

st.sidebar.markdown('### 🔍 필터')
selected_key = st.sidebar.selectbox(
    '전략 선택',
    options=TOP10,
    index=0,
    format_func=lambda k: f'#{TOP10.index(k)+1} {k}',
)

# 기간
w = step11['weights'][selected_key]
min_date = w.index.min().to_pydatetime().date()
max_date = w.index.max().to_pydatetime().date()

date_range = st.sidebar.date_input(
    '기간 선택',
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

group_view = st.sidebar.radio(
    '비중 표시 방식',
    options=['자산군 집계 (3그룹)', '30개 자산 개별'],
    index=0,
)


# ============================================================
# 선택 전략 정보 카드
# ============================================================

mode, profile, config = parse_key(selected_key)
row = final_rec.loc[selected_key]

col1, col2, col3, col4 = st.columns(4)
col1.metric('Sharpe', f'{row["sharpe"]:.3f}')
col2.metric('MDD', f'{row["mdd"]:.2%}')
col3.metric('누적수익', f'{row["total_return"]:+.2%}')
col4.metric('Multi-Score', f'{row["multi_score"]:.2f}')

st.markdown(f"**모드**: {mode} | **성향**: {profile} | **Config**: {config}")


# ============================================================
# 필터 적용
# ============================================================

if len(date_range) == 2:
    start, end = date_range
    w_filtered = w.loc[str(start):str(end)]
    alerts = step11['alert_levels'][selected_key].loc[str(start):str(end)]
else:
    w_filtered = w
    alerts = step11['alert_levels'][selected_key]


# ============================================================
# 비중 시계열 차트
# ============================================================

st.markdown('---')
st.markdown('## 📊 비중 시계열')

if group_view == '자산군 집계 (3그룹)':
    eq = w_filtered[EQUITY].sum(axis=1)
    bd = w_filtered[BOND].sum(axis=1)
    gd = w_filtered[GOLD].sum(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, name='주식',
        mode='lines', fill='tozeroy',
        line=dict(width=0.5), fillcolor='rgba(214,39,40,0.7)'
    ))
    fig.add_trace(go.Scatter(
        x=eq.index, y=(eq + bd).values, name='채권',
        mode='lines', fill='tonexty',
        line=dict(width=0.5), fillcolor='rgba(31,119,180,0.7)'
    ))
    fig.add_trace(go.Scatter(
        x=eq.index, y=(eq + bd + gd).values, name='금/대체',
        mode='lines', fill='tonexty',
        line=dict(width=0.5), fillcolor='rgba(255,127,14,0.7)'
    ))
    fig.update_layout(
        title=f'{selected_key} — 자산군 비중 시계열',
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        hovermode='x unified', height=500,
        template=theme['plotly_template'] or 'plotly',
    )
else:
    fig = go.Figure()
    # 30자산 개별 stacked
    cumulative = pd.Series(0, index=w_filtered.index)
    # 색상: 주식 붉은, 채권 파란, 금 노란
    for asset in EQUITY:
        cumulative_new = cumulative + w_filtered[asset]
        fig.add_trace(go.Scatter(
            x=w_filtered.index, y=cumulative_new.values,
            name=asset, mode='lines',
            fill='tonexty' if asset != EQUITY[0] else 'tozeroy',
            line=dict(width=0),
            stackgroup='one',
            hovertemplate=f'{asset}: %{{y:.1%}}<extra></extra>',
            fillcolor='rgba(214,39,40,0.3)',
        ))
        cumulative = cumulative_new
    for asset in BOND:
        cumulative_new = cumulative + w_filtered[asset]
        fig.add_trace(go.Scatter(
            x=w_filtered.index, y=cumulative_new.values,
            name=asset, mode='lines',
            fill='tonexty', line=dict(width=0),
            stackgroup='one',
            fillcolor='rgba(31,119,180,0.6)',
        ))
        cumulative = cumulative_new
    for asset in GOLD:
        cumulative_new = cumulative + w_filtered[asset]
        fig.add_trace(go.Scatter(
            x=w_filtered.index, y=cumulative_new.values,
            name=asset, mode='lines',
            fill='tonexty', line=dict(width=0),
            stackgroup='one',
            fillcolor='rgba(255,127,14,0.7)',
        ))
        cumulative = cumulative_new
    fig.update_layout(
        title=f'{selected_key} — 30자산 개별 비중',
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        hovermode='x unified', height=600,
        template=theme['plotly_template'] or 'plotly',
        showlegend=True,
    )

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 경보 오버레이
# ============================================================

st.markdown('## 🚦 경보 레벨 시계열')

fig_alert = go.Figure()
fig_alert.add_trace(go.Scatter(
    x=alerts.index, y=alerts.values, mode='lines',
    line=dict(color=theme['danger'], width=1, shape='hv'),
    fill='tozeroy', fillcolor='rgba(244,67,54,0.2)',
    name='경보 레벨',
))
fig_alert.update_layout(
    title=f'{selected_key} — Config {config.replace("ALERT_", "")} 경보',
    yaxis=dict(
        tickmode='array',
        tickvals=[0, 1, 2, 3],
        ticktext=['L0 정상', 'L1 주의', 'L2 경계', 'L3 위기']
    ),
    hovermode='x unified', height=300,
    template=theme['plotly_template'] or 'plotly',
)
st.plotly_chart(fig_alert, use_container_width=True)


# ============================================================
# 경보별 주식 비중 박스플롯
# ============================================================

st.markdown('## 📦 경보별 주식 비중 분포')

eq_total = w_filtered[EQUITY].sum(axis=1)
alert_in = alerts.reindex(eq_total.index)

fig_box = go.Figure()
for lvl in [0, 1, 2, 3]:
    vals = eq_total[alert_in == lvl].values
    if len(vals) > 0:
        fig_box.add_trace(go.Box(
            y=vals * 100, name=f'L{lvl} (n={len(vals)})',
            marker_color=ALERT_COLORS[lvl],
            boxmean='sd',
        ))
fig_box.update_layout(
    title='경보 레벨별 주식 총비중 분포',
    yaxis_title='주식 비중 (%)',
    height=400,
    template=theme['plotly_template'] or 'plotly',
)
st.plotly_chart(fig_box, use_container_width=True)

st.info(
    '💡 **관찰**: 경보 레벨이 높을수록 주식 비중이 체계적으로 낮아짐 → '
    '경로 1이 실제로 작동함을 시각적으로 증명'
)


# ============================================================
# 주요 통계
# ============================================================

st.markdown('---')
st.markdown('## 📊 주요 통계')

col1, col2, col3 = st.columns(3)

# 앵커 자산 Top 5
w_stats = pd.DataFrame({
    'mean_weight': w_filtered.mean(),
    'std_weight': w_filtered.std(),
})
anchors = w_stats.nlargest(5, 'mean_weight')

with col1:
    st.markdown('### 🔒 앵커 자산 Top 5')
    for asset, row in anchors.iterrows():
        st.markdown(f"**{asset}**: 평균 {row['mean_weight']:.1%}")

with col2:
    st.markdown('### 🌊 반응 자산 Top 5')
    reactives = w_stats.nlargest(5, 'std_weight')
    for asset, row in reactives.iterrows():
        st.markdown(f"**{asset}**: 변동 σ {row['std_weight']:.1%}")

with col3:
    st.markdown('### 🚦 경보 분포')
    for lvl in [0, 1, 2, 3]:
        count = (alert_in == lvl).sum()
        pct = count / len(alert_in) * 100
        st.markdown(f"**L{lvl}**: {count}일 ({pct:.1f}%)")
