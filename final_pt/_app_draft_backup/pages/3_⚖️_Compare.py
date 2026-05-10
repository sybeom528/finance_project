"""
3_⚖️_Compare.py — 다중 실험 비교 6 panel (Sortino 강조).

99_explore.ipynb plot_compare 의 6 패널 확장본 (rolling sortino + 하방편차 추가).
"""
from pathlib import Path
import sys

import streamlit as st

APP_DIR   = Path(__file__).resolve().parent.parent
FINAL_DIR = APP_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from app.lib.data_loader import load_all_results, load_rf_spy, get_master_table, TOP_1_NAME
from app.lib.narrative import COMPARE_INTRO
from app.lib.plot_helpers import plot_compare_panel


st.set_page_config(page_title='Compare · Sortino', page_icon='⚖️', layout='wide')

st.markdown(COMPARE_INTRO)


# ── 사이드바: 기간 + preset ──────────────────────────────────
period = st.sidebar.radio('평가 기간', ['TEST', 'HOLD_OUT', 'FULL'], index=0)
include_spy = st.sidebar.checkbox('SPY 포함', value=True)

period_map = {
    'TEST'    : ('2010-01-31', '2023-12-31'),
    'HOLD_OUT': ('2024-01-31', '2025-12-31'),
    'FULL'    : ('2010-01-31', '2025-12-31'),
}

results = load_all_results()
rf, spy = load_rf_spy()
mt = get_master_table(sort_by=f'sortino_{period}' if period != 'FULL' else 'sortino')

# ── Preset 버튼 ────────────────────────────────────────────
sortino_col = f'sortino_{period}' if f'sortino_{period}' in mt.columns else 'sortino'
top5_names = mt.nlargest(5, sortino_col)['name'].tolist()

st.sidebar.markdown('### Preset 버튼')
preset = st.sidebar.radio('Preset', [
    '— 직접 선택 —',
    '① mat_eq_eq_lam_pap vs Top 5 vs baseline',
    '② BL 미사용 비교군 (capm_no_bl, naive_lowvol)',
    '③ Q 민감도 (baseline_q* 4종)',
    '④ Prior 효과 (mcap vs eq vs rp)',
], index=1)

# 기본값 = Preset ①
if preset.startswith('① mat_eq_eq_lam_pap'):
    default_picks = sorted(set([TOP_1_NAME] + top5_names + ['baseline']))
elif preset.startswith('② BL 미사용'):
    default_picks = ['baseline', 'capm_no_bl', 'naive_lowvol']
elif preset.startswith('③ Q 민감도'):
    default_picks = ['baseline', 'baseline_q55', 'baseline_q64', 'baseline_q70']
elif preset.startswith('④ Prior 효과'):
    default_picks = ['baseline', 'prior_eq', 'prior_rp']
else:
    default_picks = [TOP_1_NAME, 'baseline']

default_picks = [n for n in default_picks if n in results]


# ── 멀티셀렉트 ────────────────────────────────────────────────
all_names = sorted(results.keys())
selected = st.multiselect('비교할 실험 선택 (2~10개 추천)', all_names,
                          default=default_picks)

if not selected:
    st.warning('실험을 1개 이상 선택하세요.')
    st.stop()


# ── 메트릭 표 ────────────────────────────────────────────────
sub = mt[mt['name'].isin(selected)].copy()
_sortino_period_col = f'sortino_{period}'
display_cols = ['name', 'canonical',
                'sortino', _sortino_period_col,
                'sortino_ir',
                'sharpe', 'cagr', 'mdd', 'beta', 'alpha']
# 컬럼 존재 여부 + 중복 제거 (순서 보존)
seen = set()
display_cols = [c for c in display_cols
                if c in sub.columns and not (c in seen or seen.add(c))]
st.dataframe(sub[display_cols].round(3), use_container_width=True, hide_index=True)


# ── 6 패널 차트 ───────────────────────────────────────────────
st.subheader(f'📊 6 패널 비교 — 기간: {period_map[period][0]} ~ {period_map[period][1]} ({period})')

rets_dict = {n: results[n]['ret'] for n in selected if n in results}
fig = plot_compare_panel(rets_dict, rf, spy, period_map[period],
                          include_spy=include_spy)
st.pyplot(fig)
import matplotlib.pyplot as _plt; _plt.close(fig)
