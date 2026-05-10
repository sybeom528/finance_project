"""
1_📊_Overview.py — KPI hero + Top 1 추천 + 누적수익률.

발표 첫 슬라이드: 핵심 결과를 클릭 없이도 보이게.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parent.parent
FINAL_DIR = APP_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from app.lib.data_loader import (
    load_all_results, load_rf_spy, get_master_table, get_period_metrics, TOP_1_NAME,
)
from app.lib.narrative import TOP_1_NARRATIVE
from app.lib.plot_helpers import render_pipeline_diagram_md


st.set_page_config(page_title='Overview · Low-Risk BL', page_icon='📊', layout='wide')

st.title('📊 Overview — 핵심 결과 + Top 1 추천')

# ── 사이드바 ─────────────────────────────────────────────────
period = st.sidebar.radio(
    '평가 기간',
    ['TEST', 'HOLD_OUT', 'FULL'],
    index=0,
    help='TEST: 2010-01~2023-12 (168m, 학습)  ·  HOLD_OUT: 2024-01~2025-12 (24m, 실전)  ·  FULL: 192m 전체'
)

# ── 데이터 로드 ───────────────────────────────────────────────
results = load_all_results()
rf, spy = load_rf_spy()
mt = get_master_table(sort_by=f'sortino_{period}' if period != 'FULL' else 'sortino')


# ══════════════════════════════════════════════════════════════
# Hero — Top 1 KPI
# ══════════════════════════════════════════════════════════════
st.markdown(TOP_1_NARRATIVE)

if TOP_1_NAME not in results:
    st.warning(f'추천 Top 1 모델 `{TOP_1_NAME}` 미존재 — pkl 확인 필요')
else:
    m_test     = get_period_metrics(TOP_1_NAME, 'TEST')
    m_holdout  = get_period_metrics(TOP_1_NAME, 'HOLD_OUT')

    st.subheader(f'🥇 `{TOP_1_NAME}` — TEST vs HOLD-OUT 정합성')
    cA, cB = st.columns(2)
    with cA:
        st.markdown('### TEST (168m, 학습/calibration)')
        if m_test:
            cA1, cA2, cA3, cA4 = st.columns(4)
            cA1.metric('Sortino', f'{m_test["sortino"]:.3f}')
            cA2.metric('Sharpe',  f'{m_test["sharpe"]:.3f}')
            cA3.metric('CAGR',    f'{m_test["cagr"]*100:.2f}%')
            cA4.metric('MDD',     f'{m_test["mdd"]*100:.2f}%')
        else:
            st.info('TEST 메트릭 데이터 부족')
    with cB:
        st.markdown('### HOLD-OUT (24m, 실전 검증)')
        if m_holdout:
            cB1, cB2, cB3, cB4 = st.columns(4)
            cB1.metric('Sortino', f'{m_holdout["sortino"]:.3f}',
                       delta=f'{m_holdout["sortino"]-m_test["sortino"]:+.3f}' if m_test else None)
            cB2.metric('Sharpe',  f'{m_holdout["sharpe"]:.3f}',
                       delta=f'{m_holdout["sharpe"]-m_test["sharpe"]:+.3f}' if m_test else None)
            cB3.metric('CAGR',    f'{m_holdout["cagr"]*100:.2f}%')
            cB4.metric('MDD',     f'{m_holdout["mdd"]*100:.2f}%')
        else:
            st.info('HOLD-OUT 메트릭 데이터 부족 (BL 재학습 진행 중일 수 있음)')


st.divider()


# ══════════════════════════════════════════════════════════════
# Top 5 vs SPY vs baseline 누적수익률
# ══════════════════════════════════════════════════════════════
st.subheader(f'📈 누적수익률 — Top 5 (Sortino_{period}) + SPY + baseline')

sortino_col = f'sortino_{period}' if f'sortino_{period}' in mt.columns else 'sortino'
top5_names  = mt.nlargest(5, sortino_col)['name'].tolist()
# 순서 보존 dedup — TOP_1 우선, baseline 끝, Top 5 사이에 끼움
_must_include = [TOP_1_NAME, 'baseline']
plot_names = []
for n in _must_include + top5_names:
    if n not in plot_names and n in results:
        plot_names.append(n)

# 기간 결정
period_map = {
    'TEST'    : ('2010-01-31', '2023-12-31'),
    'HOLD_OUT': ('2024-01-31', '2025-12-31'),
    'FULL'    : ('2010-01-31', '2025-12-31'),
}
start, end = period_map[period]

fig, ax = plt.subplots(figsize=(14, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(plot_names) + 1))

for i, n in enumerate(plot_names):   # 순서 보존 (TOP_1 우선)
    if n not in results:
        continue
    ret = results[n]['ret']
    sub = ret.loc[start:end].dropna()
    if len(sub) < 6:
        continue
    cum = (1 + sub).cumprod()
    lw = 2.0 if n == TOP_1_NAME else 1.2
    ls = '-' if n != 'baseline' else '--'
    ax.plot(cum.index, cum.values, label=n, color=colors[i], linewidth=lw, linestyle=ls)

# SPY
spy_sub = spy.loc[start:end].fillna(0)
if len(spy_sub) > 0:
    cum_spy = (1 + spy_sub).cumprod()
    ax.plot(cum_spy.index, cum_spy.values, label='SPY',
            color='black', linestyle='--', linewidth=1.4, alpha=0.7)

ax.set_title(f'기간: {start} ~ {end}  ({period})')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.divider()


# ══════════════════════════════════════════════════════════════
# 165 실험 중 SPY 압도 비율
# ══════════════════════════════════════════════════════════════
st.subheader('🎲 165 실험 중 SPY 압도 비율')

# SPY 의 sortino (period 기준)
spy_ret_sub = spy.loc[start:end].dropna()
rf_sub      = rf.reindex(spy_ret_sub.index).fillna(0)
spy_exc     = spy_ret_sub - rf_sub
spy_down    = spy_ret_sub[spy_ret_sub < 0].std() * np.sqrt(12)
spy_sortino = float(spy_exc.mean() * 12 / spy_down) if spy_down > 0 else float('nan')

if sortino_col in mt.columns:
    n_total = mt[sortino_col].notna().sum()
    n_beat  = (mt[sortino_col] > spy_sortino).sum()
    pct = (n_beat / n_total * 100) if n_total > 0 else 0
    cK1, cK2, cK3 = st.columns(3)
    cK1.metric('SPY Sortino (참조)', f'{spy_sortino:.3f}')
    cK2.metric('SPY 압도 실험 수', f'{n_beat} / {n_total}')
    cK3.metric('압도 비율', f'{pct:.1f}%')


st.divider()


# ══════════════════════════════════════════════════════════════
# 파이프라인
# ══════════════════════════════════════════════════════════════
st.subheader('🔧 7-Step 파이프라인')
st.markdown(render_pipeline_diagram_md())
