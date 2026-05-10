"""
2_🔍_Experiment_Explorer.py — 165 실험 마스터 테이블 + 단일 실험 6 패널.

99_explore.ipynb 의 metrics_summary + plot_single GUI 화.
"""
from pathlib import Path
import sys

import streamlit as st

APP_DIR   = Path(__file__).resolve().parent.parent
FINAL_DIR = APP_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from app.lib.data_loader import (
    load_all_results, load_rf_spy, get_master_table, list_experiment_names, TOP_1_NAME,
)
from app.lib.narrative import EXPLORER_INTRO, SLOT_TOOLTIPS
from app.lib.plot_helpers import plot_single_panel


st.set_page_config(page_title='Experiment Explorer', page_icon='🔍', layout='wide')

st.markdown(EXPLORER_INTRO)


# ── 사이드바 필터 ───────────────────────────────────────────
period = st.sidebar.radio('평가 기간', ['TEST', 'HOLD_OUT', 'FULL'], index=0,
                          help='TEST 168m · HOLD-OUT 24m · FULL 192m')
sort_col = f'sortino_{period}' if period != 'FULL' else 'sortino'

mt = get_master_table(sort_by=sort_col)
results = load_all_results()
rf, spy = load_rf_spy()

st.sidebar.markdown('### 슬롯 필터')
prior_opts = sorted(mt['prior_s'].dropna().unique().tolist()) if 'prior_s' in mt.columns else []
pw_opts    = sorted(mt['pw_s'].dropna().unique().tolist())    if 'pw_s'    in mt.columns else []
q_opts     = sorted(mt['q_s'].dropna().unique().tolist())     if 'q_s'     in mt.columns else []
om_opts    = sorted(mt['om_s'].dropna().unique().tolist())    if 'om_s'    in mt.columns else []

f_prior = st.sidebar.multiselect('prior',    prior_opts, default=prior_opts, help=SLOT_TOOLTIPS['prior'])
f_pw    = st.sidebar.multiselect('p_weight', pw_opts,    default=pw_opts,    help=SLOT_TOOLTIPS['p_weight'])
f_q     = st.sidebar.multiselect('q_mode',   q_opts,     default=q_opts,     help=SLOT_TOOLTIPS['q_mode'])
f_om    = st.sidebar.multiselect('omega',    om_opts,    default=om_opts,    help=SLOT_TOOLTIPS['omega_mode'])

search_text = st.sidebar.text_input('이름 검색', '', placeholder='예: lam_pap, mat_eq, baseline')


# ── 필터 적용 ───────────────────────────────────────────────
mt_f = mt.copy()
if f_prior and 'prior_s' in mt_f.columns: mt_f = mt_f[mt_f['prior_s'].isin(f_prior)]
if f_pw    and 'pw_s'    in mt_f.columns: mt_f = mt_f[mt_f['pw_s'].isin(f_pw)]
if f_q     and 'q_s'     in mt_f.columns: mt_f = mt_f[mt_f['q_s'].isin(f_q)]
if f_om    and 'om_s'    in mt_f.columns: mt_f = mt_f[mt_f['om_s'].isin(f_om)]
if search_text:
    mt_f = mt_f[mt_f['name'].str.contains(search_text, case=False, na=False)]

st.caption(f'필터 후 {len(mt_f)} 실험 / 전체 {len(mt)}  ·  default 정렬: **{sort_col}**')


# ── 마스터 테이블 ───────────────────────────────────────────
_sortino_period_col = f'sortino_{period}'
_sharpe_period_col  = f'sharpe_{period}'
display_cols = ['name', 'canonical', 'prior_s', 'pw_s', 'q_s', 'om_s',
                'sortino', _sortino_period_col,
                'sharpe',  _sharpe_period_col,
                'sortino_ir',
                'cagr', 'mdd', 'beta', 'alpha']
# 컬럼 존재 여부 + 중복 제거 (순서 보존)
seen = set()
display_cols = [c for c in display_cols
                if c in mt_f.columns and not (c in seen or seen.add(c))]

st.dataframe(
    mt_f[display_cols].round(3),
    use_container_width=True,
    height=420,
    hide_index=True,
)


# ── 단일 실험 선택 ───────────────────────────────────────────
st.divider()
st.subheader('🔬 단일 실험 6 패널 진단')

names_options = mt_f['name'].tolist()
if not names_options:
    st.warning('필터 결과가 비어있습니다. 사이드바의 슬롯 필터를 다시 확인하세요.')
    st.stop()

default_idx = 0
if TOP_1_NAME in names_options:
    default_idx = names_options.index(TOP_1_NAME)

selected = st.selectbox('실험 선택 (필터된 목록 기준)', names_options, index=default_idx)

if selected and selected in results:
    period_map = {
        'TEST'    : ('2010-01-31', '2023-12-31'),
        'HOLD_OUT': ('2024-01-31', '2025-12-31'),
        'FULL'    : ('2010-01-31', '2025-12-31'),
    }
    fig = plot_single_panel(results[selected], rf, spy,
                            period_map[period], selected)
    st.pyplot(fig)
    import matplotlib.pyplot as _plt; _plt.close(fig)

    cfg = results[selected].get('config', {})
    with st.expander(f'⚙ Config — {selected}', expanded=False):
        cfg_show = {k: v for k, v in cfg.items() if k != 'lstm_pred_path'}
        st.json(cfg_show)
