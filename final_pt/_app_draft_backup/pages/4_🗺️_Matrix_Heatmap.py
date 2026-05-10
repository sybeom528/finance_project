"""
4_🗺️_Matrix_Heatmap.py — 매트릭스 3 prior × 3 p_weight × 5 q_mode × 3 ω_mode = 135 cells.

Sortino default + 행/열 평균.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR   = Path(__file__).resolve().parent.parent
FINAL_DIR = APP_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from app.lib.data_loader import get_master_table
from app.lib.narrative import MATRIX_INTRO


st.set_page_config(page_title='Matrix Heatmap', page_icon='🗺️', layout='wide')

st.markdown(MATRIX_INTRO)


period = st.sidebar.radio('평가 기간', ['TEST', 'HOLD_OUT', 'FULL'], index=0)

# Get master_table 먼저 — 가용 컬럼 목록 확인용
_mt_check = get_master_table(sort_by='sortino')

# 메트릭 옵션 — mt 에 실제로 존재하는 컬럼만
_all_metric_options = {
    f'sortino_{period}'  : f'Sortino ({period})',
    f'sharpe_{period}'   : f'Sharpe ({period})',
    f'cagr_{period}'     : f'CAGR ({period})',
    f'mdd_{period}'      : f'MDD ({period})',
    'sortino'            : 'Sortino (overall)',
    'sortino_ir'         : 'Sortino IR (3-레짐 안정성)',
    'sharpe'             : 'Sharpe (overall)',
}
metric_options = {k: v for k, v in _all_metric_options.items() if k in _mt_check.columns}

if not metric_options:
    st.error('master_table 에 표시할 metric 컬럼이 없습니다. 데이터 상태 확인 필요.')
    st.stop()

metric_label = st.sidebar.radio('메트릭', list(metric_options.values()),
                                 index=0, help='Sortino 우선 (사용자 요청). 컬럼이 mt 에 있는 것만 노출됨.')
metric_col = [k for k, v in metric_options.items() if v == metric_label][0]

# Re-sort with chosen metric (mdd 는 작을수록 좋아 sortino 로 우회 정렬)
sort_metric = metric_col if metric_col != f'mdd_{period}' else f'sortino_{period}'
sort_metric = sort_metric if sort_metric in _mt_check.columns else 'sortino'
mt = get_master_table(sort_by=sort_metric)

# LSTM 매트릭스만 필터
mat_only = mt[mt['p_s'] == 'ls'].copy() if 'p_s' in mt.columns else mt.copy()

if metric_col not in mat_only.columns:
    st.error(f'메트릭 컬럼 {metric_col} 없음. master_table 빌드 확인 필요.')
    st.stop()


# ── 행 = prior×pw, 열 = q×Ω ────────────────────────────────
# 매트릭스 배치 (사용자 요청 default)
row_keys = ['prior_s', 'pw_s']
col_keys = ['q_s', 'om_s']

mat_only['row_label'] = mat_only[row_keys[0]].astype(str) + '×' + mat_only[row_keys[1]].astype(str)
mat_only['col_label'] = mat_only[col_keys[0]].astype(str) + '×' + mat_only[col_keys[1]].astype(str)

pivot = mat_only.pivot_table(index='row_label', columns='col_label',
                              values=metric_col, aggfunc='first')

# 행/열 평균 추가
pivot_with_avg = pivot.copy()
pivot_with_avg.loc[:, '⌐ 행평균'] = pivot.mean(axis=1)
pivot_with_avg.loc['⌐ 열평균', :] = pivot.mean(axis=0)


st.subheader(f'🌡️ Heatmap — {metric_label}  ·  {len(pivot)} × {len(pivot.columns)} cells')

# Matplotlib heatmap
fig, ax = plt.subplots(figsize=(13, 8))
data = pivot_with_avg.values
cmap = 'RdYlGn' if 'mdd' not in metric_col else 'RdYlGn_r'
im = ax.imshow(data, aspect='auto', cmap=cmap)

ax.set_xticks(range(len(pivot_with_avg.columns)))
ax.set_xticklabels(pivot_with_avg.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(pivot_with_avg.index)))
ax.set_yticklabels(pivot_with_avg.index, fontsize=9)

# 셀 위에 값 표시
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        if not np.isnan(v):
            color = 'white' if abs(v) > np.nanmean(np.abs(data)) * 1.2 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=color, fontsize=7)

# 행/열 평균 분리선
ax.axhline(len(pivot.index) - 0.5, color='black', linewidth=1.5)
ax.axvline(len(pivot.columns) - 0.5, color='black', linewidth=1.5)

ax.set_title(f'행 = {row_keys[0]}×{row_keys[1]}  ·  열 = {col_keys[0]}×{col_keys[1]}',
             fontsize=11)
fig.colorbar(im, ax=ax, label=metric_label)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.divider()


# ── Top / Bottom 셀 ─────────────────────────────────────────
st.subheader('🏆 Top / Bottom Cells')
c1, c2 = st.columns(2)
with c1:
    st.markdown('### Top 10')
    top = mat_only.nlargest(10, metric_col)[
        ['name', 'canonical', metric_col]].round(3)
    st.dataframe(top, use_container_width=True, hide_index=True)
with c2:
    st.markdown('### Bottom 10')
    bot = mat_only.nsmallest(10, metric_col)[
        ['name', 'canonical', metric_col]].round(3)
    st.dataframe(bot, use_container_width=True, hide_index=True)
