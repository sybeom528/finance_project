"""페이지 7: Simulator — 사용자 파라미터로 전략 탐색."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.theme import render_theme_selector, apply_custom_css, get_current_theme
from utils.data_loader import (
    load_step9_results, load_metrics, load_step11_weights,
    EQUITY, BOND, GOLD, parse_key
)


st.set_page_config(page_title='Simulator', page_icon='🎮', layout='wide')
render_theme_selector()
apply_custom_css()
theme = get_current_theme()

st.title('🎮 Interactive Simulator — 나만의 전략 탐색')
st.caption('성향·모드·Config를 선택하여 64개 조합 중 하나의 성과 확인')

st.markdown('---')


# 데이터 로드
step9 = load_step9_results()
results = step9['results']
metrics = load_metrics()
step11 = load_step11_weights()


# ============================================================
# 파라미터 선택 (사이드바)
# ============================================================

st.sidebar.markdown('### 🎛️ 파라미터')

profile = st.sidebar.selectbox(
    '성향 (γ 기반)',
    options=['보수형', '중립형', '적극형', '공격형'],
    index=0,
    help='γ: 8=보수, 4=중립, 2=적극, 1=공격'
)

mode = st.sidebar.radio(
    '모드',
    options=['M0', 'M1', 'M2', 'M3'],
    index=1,
    help='M0: 순수 MV, M1: +경로1, M2: +경로2, M3: 통합',
)

config = st.sidebar.radio(
    '경보 Config',
    options=['ALERT_A', 'ALERT_B', 'ALERT_C', 'ALERT_D'],
    index=1,
    help='A: VIX만, B: +Contango, C: 7지표, D: +디바운스',
)

cost_bps = st.sidebar.slider(
    '편도 거래비용 (bps)',
    min_value=0, max_value=50, value=15, step=5,
    help='기본 15bps. 낮을수록 성과 유리',
)

initial_capital = st.sidebar.number_input(
    '시작 자본 (USD)',
    min_value=10000, max_value=10000000, value=100000, step=10000,
)


# ============================================================
# 전략 키 구성 및 조회
# ============================================================

key = f'{mode}_{profile}_{config}'

if key not in results:
    st.error(f'❌ {key} 전략을 찾을 수 없습니다.')
    st.stop()

daily_ret = results[key]

# 거래비용 조정 (근사): 기본 15bps 대비 delta × turnover 추정
# 정확한 재시뮬 대신 근사치 사용
if cost_bps != 15 and key in step11['rebalance_events']:
    events = step11['rebalance_events'].get(key)
    if events is not None and len(events) > 0:
        # 간단 근사: 추가 비용만큼 선형 차감
        # 실제로는 재시뮬 필요하나 Streamlit 성능상 근사
        total_turnover = events['turnover'].sum() if 'turnover' in events.columns else 1.0
        n_years = len(daily_ret) / 252
        delta_cost_annual = (cost_bps - 15) / 10000 * total_turnover / n_years
        daily_adj = delta_cost_annual / 252
        daily_ret = daily_ret - daily_adj


# ============================================================
# 결과 표시
# ============================================================

st.markdown(f'## 📊 {key} — 시뮬레이션 결과')

# KPI 계산
cum = (1 + daily_ret).cumprod()
total_ret = cum.iloc[-1] - 1
ann_ret = daily_ret.mean() * 252
ann_vol = daily_ret.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
peak = cum.cummax()
dd = (cum - peak) / peak
mdd = dd.min()
calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
final_value = initial_capital * (1 + total_ret)


col1, col2, col3, col4 = st.columns(4)
col1.metric('Sharpe', f'{sharpe:.3f}')
col2.metric('MDD', f'{mdd:.2%}')
col3.metric('누적수익', f'{total_ret:+.2%}')
col4.metric('Calmar', f'{calmar:.3f}')

st.markdown(f'### 💰 최종 자본: **${final_value:,.0f}** (시작 ${initial_capital:,.0f})')


# ============================================================
# 누적수익률 차트
# ============================================================

st.markdown('---')
st.markdown('### 📈 누적수익률 시계열')

compare_bench = st.checkbox('벤치마크 오버레이', value=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cum.index, y=cum.values * initial_capital,
    name=key, mode='lines',
    line=dict(color=theme['primary'], width=2),
))

if compare_bench:
    for bench, color, style in [('BENCH_SPY', 'gray', 'dash'),
                                  ('BENCH_EW', 'lightgray', 'dot')]:
        if bench in results:
            bench_cum = (1 + results[bench]).cumprod()
            fig.add_trace(go.Scatter(
                x=bench_cum.index, y=bench_cum.values * initial_capital,
                name=bench.replace('BENCH_', ''), mode='lines',
                line=dict(color=color, width=1, dash=style),
            ))

fig.update_layout(
    title=f'누적 자본 (시작 ${initial_capital:,.0f})',
    yaxis_title='자본 (USD)',
    hovermode='x unified', height=500,
    template=theme['plotly_template'] or 'plotly',
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Drawdown
# ============================================================

st.markdown('### 📉 Drawdown 시계열')

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=dd.index, y=dd.values * 100, name='Drawdown',
    mode='lines', line=dict(width=1, color=theme['danger']),
    fill='tozeroy', fillcolor='rgba(244,67,54,0.3)',
))
fig_dd.update_layout(
    title=f'Drawdown (MDD: {mdd:.2%})',
    yaxis_title='Drawdown (%)',
    height=300, template=theme['plotly_template'] or 'plotly',
)
st.plotly_chart(fig_dd, use_container_width=True)


# ============================================================
# Top 10 대비 순위
# ============================================================

st.markdown('---')
st.markdown('### 🏆 Top 10 내 순위')

if key in metrics.index:
    strategy_only = metrics[~metrics.index.str.startswith('BENCH')].sort_values('sharpe', ascending=False)
    rank = strategy_only.index.get_loc(key) + 1
    total = len(strategy_only)
    st.info(f'현재 전략은 **64개 전략 중 {rank}위** (Sharpe 기준, {total}개 전체)')

    if rank <= 10:
        st.success(f'🏆 Top 10 내 순위 - 추천 전략!')
    elif rank <= 32:
        st.warning(f'🥈 상위 50% 내 - 괜찮은 수준')
    else:
        st.error(f'🔻 하위 50% - 다른 조합 고려 권장')


# ============================================================
# 파라미터 영향 설명
# ============================================================

st.markdown('---')
with st.expander('💡 파라미터 영향 이해하기'):
    st.markdown("""
    ### 성향 (γ)
    - **보수형 (γ=8)**: 주식 상한 43%, 채권 하한 31% → 안정
    - **공격형 (γ=1)**: 주식 상한 90% → 고수익·고위험

    ### 모드
    - **M0**: 경보 무시 (순수 MV)
    - **M1**: 경로 1만 (매일 경보 체크) ⭐ **추천**
    - **M2**: 경로 2만 (월별 Σ 전환, 효과 없음)
    - **M3**: 통합 (M1보다 효과 작음)

    ### Config
    - **A**: VIX만 (단순)
    - **B**: VIX + Contango ⭐ **추천**
    - **C**: 7지표 복합 (정교하나 복잡)
    - **D**: C + 디바운스 (변동 최소, 반응 느림)

    ### 거래비용
    - **0 bps**: 이상적 (불가능)
    - **15 bps**: 기관 수준 (기본값)
    - **30 bps**: 소매 투자자
    - **50 bps**: 고비용 환경 (효과 축소)
    """)


# ============================================================
# 제한 사항
# ============================================================

st.markdown('---')
st.caption('⚠️ **근사치 안내**: 거래비용 변경 시 재시뮬이 아닌 선형 근사. '
           '정확한 시뮬레이션은 노트북 `Step9_Integrated_Backtest.ipynb` 참조.')
