"""
HTML Plotly Standalone 대시보드 생성.
`interactive/dashboard.html` — 파일 하나로 모든 핵심 시각화 탐색 가능.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

GUIDE_DIR = Path(__file__).parent
DATA_DIR = GUIDE_DIR / 'data'
OUT_PATH = GUIDE_DIR / 'interactive' / 'dashboard.html'
OUT_PATH.parent.mkdir(exist_ok=True)

# 데이터 로드
metrics = pd.read_csv(DATA_DIR / 'step9_metrics.csv', index_col=0)
final_rec = pd.read_csv(DATA_DIR / 'step10_final_recommendation.csv', index_col=0)

with open(DATA_DIR / 'step9_backtest_results.pkl', 'rb') as f:
    step9 = pickle.load(f)
results = step9['results']

with open(DATA_DIR / 'step11_top10_weights.pkl', 'rb') as f:
    step11 = pickle.load(f)

alerts = pd.read_csv(DATA_DIR / 'alert_signals.csv',
                      parse_dates=['Date'], index_col='Date')

TOP10 = final_rec.head(10).index.tolist()
BEST = TOP10[0]  # M1_보수형_ALERT_B

# 자산 그룹
EQUITY = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'XLK', 'XLF', 'XLE', 'XLV', 'VOX',
          'XLY', 'XLP', 'XLI', 'XLU', 'XLRE', 'XLB',
          'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'JNJ', 'PG', 'XOM']
BOND = ['TLT', 'AGG', 'SHY', 'TIP']
GOLD = ['GLD', 'DBC']


def build_html():
    """HTML 대시보드 HTML 문자열 생성."""

    # === Figure 1: 누적수익률 Top 5 + 벤치마크 ===
    fig1 = go.Figure()
    for i, key in enumerate(TOP10[:5]):
        cum = (1 + results[key]).cumprod()
        fig1.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name=f'#{i+1} {key}',
            line=dict(width=1.5), mode='lines'
        ))
    for bench, color in [('BENCH_SPY', '#888888'), ('BENCH_EW', '#BBBBBB'),
                          ('BENCH_60_40', '#CCCCCC')]:
        cum = (1 + results[bench]).cumprod()
        fig1.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name=bench.replace('BENCH_', ''),
            line=dict(width=1.2, dash='dash'), mode='lines'
        ))
    fig1.update_layout(
        title='Top 5 전략 vs 벤치마크 누적수익률',
        xaxis_title='날짜', yaxis_title='누적 자본 (1=시작)',
        hovermode='x unified', height=500,
        legend=dict(orientation='h', y=-0.2)
    )

    # === Figure 2: Top 10 Sharpe + MDD 버블 ===
    fig2 = go.Figure()
    sharpe = [final_rec.loc[k, 'sharpe'] for k in TOP10]
    mdd = [final_rec.loc[k, 'mdd'] * 100 for k in TOP10]
    total = [final_rec.loc[k, 'total_return'] * 100 for k in TOP10]
    colors_by_mode = ['#1f77b4' if 'M1_' in k else '#d62728' for k in TOP10]
    fig2.add_trace(go.Scatter(
        x=mdd, y=sharpe, mode='markers+text',
        text=[f'#{i+1}' for i in range(10)],
        textposition='top center',
        marker=dict(size=[max(15, t/4) for t in total],
                    color=colors_by_mode,
                    line=dict(color='white', width=2), opacity=0.7),
        hovertext=[f'{k}<br>Sharpe: {s:.3f}<br>MDD: {m:.2f}%<br>Total: {t:.1f}%'
                   for k, s, m, t in zip(TOP10, sharpe, mdd, total)],
        hoverinfo='text',
    ))
    fig2.update_layout(
        title='Top 10 전략: Sharpe × MDD (버블 크기 = 총수익)',
        xaxis_title='MDD (%)', yaxis_title='Sharpe Ratio',
        height=500,
        annotations=[
            dict(x=-16, y=1.07, text='🏆', showarrow=False, font=dict(size=30)),
        ]
    )

    # === Figure 3: Best 전략 자산군 비중 (Stacked Area) ===
    w = step11['weights'][BEST]
    eq_total = w[EQUITY].sum(axis=1)
    bd_total = w[BOND].sum(axis=1)
    gd_total = w[GOLD].sum(axis=1)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=eq_total.index, y=eq_total.values, name='주식',
        mode='lines', fill='tozeroy',
        line=dict(width=0.5), fillcolor='rgba(214,39,40,0.7)'
    ))
    fig3.add_trace(go.Scatter(
        x=eq_total.index, y=(eq_total + bd_total).values, name='채권',
        mode='lines', fill='tonexty',
        line=dict(width=0.5), fillcolor='rgba(31,119,180,0.7)'
    ))
    fig3.add_trace(go.Scatter(
        x=eq_total.index, y=(eq_total + bd_total + gd_total).values, name='금/대체',
        mode='lines', fill='tonexty',
        line=dict(width=0.5), fillcolor='rgba(255,127,14,0.7)'
    ))
    fig3.update_layout(
        title=f'{BEST}: 자산군 비중 시계열',
        xaxis_title='날짜', yaxis_title='비중',
        hovermode='x unified', height=500,
        yaxis=dict(tickformat='.0%', range=[0, 1])
    )

    # === Figure 4: 경보 레벨 타임라인 ===
    alert_b = step11['alert_levels'][BEST]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=alert_b.index, y=alert_b.values, mode='lines',
        line=dict(color='red', width=1, shape='hv'),
        fill='tozeroy', fillcolor='rgba(214,39,40,0.2)',
        name='경보 레벨'
    ))
    fig4.update_layout(
        title=f'{BEST}: Config B 경보 레벨 시계열',
        xaxis_title='날짜', yaxis_title='경보 레벨',
        yaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3],
                    ticktext=['L0 정상', 'L1 주의', 'L2 경계', 'L3 위기']),
        hovermode='x unified', height=300
    )

    # === Figure 5: 모드별 평균 Sharpe ===
    modes = ['M0', 'M1', 'M2', 'M3']
    mode_avg = []
    for mode in modes:
        vals = []
        for p in ['보수형', '중립형', '적극형', '공격형']:
            for c in ['ALERT_A', 'ALERT_B', 'ALERT_C', 'ALERT_D']:
                key = f'{mode}_{p}_{c}'
                if key in metrics.index:
                    vals.append(metrics.loc[key, 'sharpe'])
        mode_avg.append(np.mean(vals))
    fig5 = go.Figure()
    colors = ['#808080', '#1f77b4', '#ff7f0e', '#d62728']
    fig5.add_trace(go.Bar(
        x=modes, y=mode_avg, marker_color=colors,
        text=[f'{v:.3f}' for v in mode_avg],
        textposition='outside',
        hovertemplate='%{x}: Sharpe %{y:.3f}<extra></extra>'
    ))
    # 벤치마크 기준선
    for name, val, color in [('EW', 0.82, 'gray'), ('SPY', 0.76, 'black')]:
        fig5.add_hline(y=val, line_dash='dash', line_color=color,
                        annotation_text=name, annotation_position='right')
    fig5.update_layout(
        title='4개 모드별 평균 Sharpe (16 조합 평균)',
        xaxis_title='모드', yaxis_title='평균 Sharpe Ratio',
        height=400, showlegend=False
    )

    # === Figure 6: Drawdown ===
    fig6 = go.Figure()
    for i, key in enumerate(TOP10[:3]):
        cum = (1 + results[key]).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        fig6.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100, name=f'#{i+1} {key}',
            mode='lines', fill='tozeroy' if i == 0 else None,
            line=dict(width=1)
        ))
    cum_spy = (1 + results['BENCH_SPY']).cumprod()
    peak_spy = cum_spy.cummax()
    dd_spy = (cum_spy - peak_spy) / peak_spy
    fig6.add_trace(go.Scatter(
        x=dd_spy.index, y=dd_spy.values * 100, name='SPY',
        mode='lines', line=dict(width=1, dash='dash', color='gray')
    ))
    fig6.update_layout(
        title='Top 3 전략 Drawdown (vs SPY)',
        xaxis_title='날짜', yaxis_title='Drawdown (%)',
        hovermode='x unified', height=400
    )

    # === HTML 조립 ===
    figs = [fig1, fig2, fig3, fig4, fig5, fig6]
    graph_html = ''
    for i, fig in enumerate(figs):
        graph_html += pio.to_html(fig, full_html=False, include_plotlyjs=('cdn' if i == 0 else False))

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Final Project — HTML 대시보드</title>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 0;
               padding: 20px; background: #F5F5F5; color: #212121; }}
        .header {{ background: #1976D2; color: white; padding: 25px;
                   border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 28px; }}
        .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
        .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr);
                    gap: 15px; margin-bottom: 20px; }}
        .kpi-card {{ background: white; padding: 20px; border-radius: 8px;
                     text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .kpi-card .label {{ font-size: 12px; color: #616161; }}
        .kpi-card .value {{ font-size: 28px; font-weight: bold; margin: 5px 0; }}
        .kpi-card .detail {{ font-size: 11px; color: #616161; }}
        .section {{ background: white; padding: 20px; border-radius: 8px;
                    margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .footer {{ text-align: center; color: #888; font-size: 12px;
                   padding: 20px; }}
        h2 {{ color: #1976D2; border-bottom: 2px solid #1976D2; padding-bottom: 5px; }}
    </style>
</head>
<body>

<div class="header">
    <h1>🎯 Final Project — 대안데이터 기반 포트폴리오 시뮬레이터</h1>
    <p>최우수 전략 <strong>{BEST}</strong> | Sharpe 1.064 | MDD -15.53% | 8년 +151%</p>
</div>

<div class="kpi-row">
    <div class="kpi-card">
        <div class="label">Sharpe Ratio</div>
        <div class="value" style="color:#4CAF50">1.064</div>
        <div class="detail">+29% vs EW</div>
    </div>
    <div class="kpi-card">
        <div class="label">최대 낙폭</div>
        <div class="value" style="color:#FF9800">-15.53%</div>
        <div class="detail">SPY의 절반</div>
    </div>
    <div class="kpi-card">
        <div class="label">8년 누적수익</div>
        <div class="value" style="color:#1976D2">+151%</div>
        <div class="detail">연 +12.2%</div>
    </div>
    <div class="kpi-card">
        <div class="label">연율 변동성</div>
        <div class="value" style="color:#607D8B">11.41%</div>
        <div class="detail">SPY 65%</div>
    </div>
</div>

<div class="section">
    <h2>📊 누적수익률 Top 5</h2>
    {pio.to_html(fig1, full_html=False, include_plotlyjs='cdn')}
</div>

<div class="section">
    <h2>🏆 Top 10 Sharpe × MDD 버블</h2>
    <p><em>버블 클수록 총수익률 큼 · 파랑 = M1 (경로 1만) · 빨강 = M3 (통합)</em></p>
    {pio.to_html(fig2, full_html=False, include_plotlyjs=False)}
</div>

<div class="section">
    <h2>📈 최우수 전략 자산군 비중 변화</h2>
    {pio.to_html(fig3, full_html=False, include_plotlyjs=False)}
</div>

<div class="section">
    <h2>🚦 경보 레벨 시계열</h2>
    {pio.to_html(fig4, full_html=False, include_plotlyjs=False)}
</div>

<div class="section">
    <h2>🎛️ 4개 모드별 평균 Sharpe (경로 2 무효성 확인)</h2>
    <p><em>M1 (경로 1만) &gt; M3 (통합) → 복잡한 경로 2 추가가 오히려 악화</em></p>
    {pio.to_html(fig5, full_html=False, include_plotlyjs=False)}
</div>

<div class="section">
    <h2>📉 Drawdown 비교</h2>
    {pio.to_html(fig6, full_html=False, include_plotlyjs=False)}
</div>

<div class="footer">
    <p>🗓️ 2026-04-17 · Final Project.1 · 김재천</p>
    <p>📂 상세 자료: Guide/ | 🎮 인터랙티브: Streamlit 앱 실행 (interactive/streamlit_app/)</p>
</div>

</body>
</html>
"""

    return html


def main():
    print('HTML 대시보드 생성 중...')
    html = build_html()
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'저장: {OUT_PATH}')
    print(f'크기: {OUT_PATH.stat().st_size / 1024:.0f} KB')
    print(f'브라우저에서 열기: file:///{OUT_PATH.as_posix()}')


if __name__ == '__main__':
    main()
