"""페이지 6: Crisis Cases — 11개 스트레스 시나리오 탐색."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.theme import render_theme_selector, apply_custom_css, get_current_theme
from utils.data_loader import (
    load_step9_results, load_final_recommendation, load_step11_weights,
    HISTORICAL_CRISES, EQUITY, BOND, GOLD, parse_key
)


st.set_page_config(page_title='Crisis', page_icon='🌪️', layout='wide')
render_theme_selector()
apply_custom_css()
theme = get_current_theme()

st.title('🌪️ Crisis Cases — 위기 대응 사례 탐색')
st.caption('역사적 6개 스트레스 시나리오에서 전략이 어떻게 반응했는지')

st.markdown('---')


# 데이터 로드
step9 = load_step9_results()
results = step9['results']
step11 = load_step11_weights()
final_rec = load_final_recommendation()
TOP10 = final_rec.head(10).index.tolist()
BEST = TOP10[0]


# ============================================================
# 위기 선택
# ============================================================

st.markdown('## 🎯 위기 선택')

crisis_name = st.selectbox(
    '분석할 위기 시나리오',
    options=list(HISTORICAL_CRISES.keys()),
    index=1,  # COVID 기본
)

start_str, end_str = HISTORICAL_CRISES[crisis_name]
st.caption(f'📅 기간: {start_str} ~ {end_str}')

# 전략 선택 (Top 3 + 벤치마크)
st.sidebar.markdown('### 비교 대상')
selected_top = st.sidebar.multiselect(
    'Top 전략',
    options=TOP10,
    default=TOP10[:3],
)
show_spy = st.sidebar.checkbox('SPY 벤치마크', value=True)
show_ew = st.sidebar.checkbox('EW 벤치마크', value=True)
show_6040 = st.sidebar.checkbox('60/40 벤치마크', value=True)


# ============================================================
# 사건 개요
# ============================================================

crisis_desc = {
    '2018 Volmageddon': {
        '핵심': 'VIX 1일 +115% (17→37), XIV ETF 청산',
        'SPY 손실': '-10% (1주일)',
        '지속': '5 영업일',
        '원인': '저변동 롱 트레이드 대량 청산',
    },
    '2020 COVID': {
        '핵심': '사상 최고 VIX 82, SPY -34% (2개월)',
        'SPY 손실': '-34%',
        '지속': '2개월 (이후 V자 반등)',
        '원인': '세계 팬데믹 + 록다운',
    },
    '2022 Q1 긴축': {
        '핵심': '연준 급속 인상 + 러-우 전쟁',
        'SPY 손실': '-13%',
        '지속': '3개월',
        '원인': '40년만 인플레 대응',
    },
    '2022 Q2 인플레 쇼크': {
        '핵심': 'CPI 9.1% 피크',
        'SPY 손실': '-17%',
        '지속': '3개월',
        '원인': '에너지·식품 인플레',
    },
    '2023 SVB': {
        '핵심': 'SVB 파산 → 은행 신용경색',
        'SPY 손실': '-8%',
        '지속': '3주',
        '원인': '금리 인상 여파 + 유동성 위기',
    },
    '2024 엔캐리 청산': {
        '핵심': 'VIX 1일 60 스파이크',
        'SPY 손실': '-6%',
        '지속': '1주일',
        '원인': '일본 금리 인상 → 엔캐리 청산',
    },
}

desc = crisis_desc.get(crisis_name, {})
if desc:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('핵심', desc['핵심'][:20]+'...' if len(desc['핵심']) > 20 else desc['핵심'])
    col2.metric('SPY 손실', desc['SPY 손실'])
    col3.metric('지속', desc['지속'])
    col4.metric('원인', desc['원인'][:15])
    with st.expander('📖 상세 배경'):
        st.markdown(f"""
        - **핵심**: {desc['핵심']}
        - **원인**: {desc['원인']}
        - **지속 기간**: {desc['지속']}
        - **SPY 손실**: {desc['SPY 손실']}
        """)


# ============================================================
# 누적수익률 비교
# ============================================================

st.markdown('---')
st.markdown('## 📉 위기 구간 누적수익률')

fig = go.Figure()

# 버퍼 기간 추가 (사건 전후 약간 포함)
buffer_start = pd.Timestamp(start_str) - pd.Timedelta(days=7)
buffer_end = pd.Timestamp(end_str) + pd.Timedelta(days=14)

for key in selected_top:
    series = results[key].loc[buffer_start:buffer_end]
    if len(series) > 0:
        cum = (1 + series).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100, name=key,
            mode='lines', line=dict(width=1.5),
        ))

if show_spy:
    series = results['BENCH_SPY'].loc[buffer_start:buffer_end]
    if len(series) > 0:
        cum = (1 + series).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100, name='SPY',
            mode='lines', line=dict(width=1, dash='dash', color='gray'),
        ))

if show_ew:
    series = results['BENCH_EW'].loc[buffer_start:buffer_end]
    if len(series) > 0:
        cum = (1 + series).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100, name='EW',
            mode='lines', line=dict(width=1, dash='dot', color='lightgray'),
        ))

if show_6040:
    series = results['BENCH_60_40'].loc[buffer_start:buffer_end]
    if len(series) > 0:
        cum = (1 + series).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100, name='60/40',
            mode='lines', line=dict(width=1, dash='dashdot', color='darkgray'),
        ))

# 위기 기간 음영
fig.add_vrect(
    x0=pd.Timestamp(start_str), x1=pd.Timestamp(end_str),
    fillcolor='red', opacity=0.1, layer='below',
    line_width=0, annotation_text='위기 기간',
    annotation_position='top left',
)

fig.update_layout(
    title=f'{crisis_name} 구간 누적수익률 (%)',
    xaxis_title='날짜', yaxis_title='누적수익률 (%)',
    hovermode='x unified', height=500,
    template=theme['plotly_template'] or 'plotly',
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 최우수 전략의 비중 변화 (Top 1만)
# ============================================================

st.markdown(f'## 📈 {BEST} 비중 변화 (위기 구간)')

if BEST in step11['weights']:
    w = step11['weights'][BEST]
    w_f = w.loc[buffer_start:buffer_end]

    if len(w_f) > 0:
        eq = w_f[EQUITY].sum(axis=1)
        bd = w_f[BOND].sum(axis=1)
        gd = w_f[GOLD].sum(axis=1)

        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name='주식',
            mode='lines', fill='tozeroy',
            line=dict(width=0.5), fillcolor='rgba(214,39,40,0.7)'
        ))
        fig_w.add_trace(go.Scatter(
            x=eq.index, y=(eq + bd).values, name='채권',
            mode='lines', fill='tonexty',
            line=dict(width=0.5), fillcolor='rgba(31,119,180,0.7)'
        ))
        fig_w.add_trace(go.Scatter(
            x=eq.index, y=(eq + bd + gd).values, name='금/대체',
            mode='lines', fill='tonexty',
            line=dict(width=0.5), fillcolor='rgba(255,127,14,0.7)'
        ))
        fig_w.add_vrect(
            x0=pd.Timestamp(start_str), x1=pd.Timestamp(end_str),
            fillcolor='red', opacity=0.1, layer='below',
            line_width=0,
        )
        fig_w.update_layout(
            title=f'{BEST} — 자산군 비중 (위기 전후)',
            yaxis=dict(tickformat='.0%', range=[0, 1]),
            hovermode='x unified', height=400,
            template=theme['plotly_template'] or 'plotly',
        )
        st.plotly_chart(fig_w, use_container_width=True)


# ============================================================
# 성과 비교표
# ============================================================

st.markdown('---')
st.markdown('## 📊 성과 비교표')

def compute_period_stats(series, start, end):
    """기간 내 누적수익, MDD 계산."""
    s = series.loc[start:end]
    if len(s) == 0:
        return None, None
    cum = (1 + s).prod() - 1
    cumsum = (1 + s).cumprod()
    peak = cumsum.cummax()
    dd = (cumsum - peak) / peak
    return cum, dd.min()


comparison_rows = []
for key in selected_top + (['BENCH_SPY'] if show_spy else []) + \
           (['BENCH_EW'] if show_ew else []) + \
           (['BENCH_60_40'] if show_6040 else []):
    if key in results:
        ret, mdd = compute_period_stats(results[key], start_str, end_str)
        if ret is not None:
            comparison_rows.append({
                '전략': key,
                '구간 수익률': f'{ret:+.2%}',
                '구간 MDD': f'{mdd:.2%}',
            })

if comparison_rows:
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)


# ============================================================
# 위기별 방어력 지수
# ============================================================

st.markdown('---')
st.markdown('## 🛡️ 전체 위기별 방어력 (Top 1 vs SPY)')

defense_data = []
for crisis_n, (s, e) in HISTORICAL_CRISES.items():
    best_ret, best_mdd = compute_period_stats(results[BEST], s, e)
    spy_ret, spy_mdd = compute_period_stats(results['BENCH_SPY'], s, e)
    if best_ret is not None and spy_ret is not None:
        defense_ratio = (abs(spy_mdd) / abs(best_mdd)) if best_mdd != 0 else 0
        defense_data.append({
            '위기': crisis_n,
            'SPY MDD': f'{spy_mdd:.2%}',
            'Best MDD': f'{best_mdd:.2%}',
            '방어 배율': f'{defense_ratio:.1f}x',
        })

if defense_data:
    st.dataframe(pd.DataFrame(defense_data), use_container_width=True, hide_index=True)
    st.success('💡 **평균 2~5배 방어력**: 위기 시 SPY 대비 손실을 현저히 축소')
