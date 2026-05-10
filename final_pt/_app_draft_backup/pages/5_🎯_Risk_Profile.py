"""
5_🎯_Risk_Profile.py — 위험성향 슬라이더 + Top 3 추천.

mat_eq_eq_lam_pap 보수형 1순위 강제 (사용자 지정).
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

from app.lib.data_loader import (
    load_all_results, load_rf_spy, get_master_table, get_period_metrics,
)
from app.lib.narrative import RISK_PROFILE_INTRO, RISK_TIER_DESC, TOP_1_NAME
from app.lib.recommendations import (
    recommend_top3, map_risk_score_to_tier, reasoning_for_recommendation,
)


st.set_page_config(page_title='Risk Profile', page_icon='🎯', layout='wide')

st.markdown(RISK_PROFILE_INTRO)


# ── 사이드바 ────────────────────────────────────────────────
risk_score = st.sidebar.slider('위험허용도', 0, 100, 20,
                                 help='0=초보수 ~ 100=공격')
horizon = st.sidebar.selectbox('투자 horizon', ['3년', '5년', '10년'], index=1)
period = st.sidebar.radio('평가 기간', ['TEST', 'HOLD_OUT', 'FULL'], index=0)

tier = map_risk_score_to_tier(risk_score)
tier_emoji = {'conservative': '🛡️', 'balanced': '⚖️', 'aggressive': '🚀'}[tier]
tier_label = {'conservative': '보수형', 'balanced': '중립형', 'aggressive': '공격형'}[tier]


# ── 데이터 로드 ───────────────────────────────────────────────
results = load_all_results()
rf, spy = load_rf_spy()
mt = get_master_table(sort_by=f'sortino_{period}' if period != 'FULL' else 'sortino')

# ── 위험성향 카드 ───────────────────────────────────────────
st.markdown(f'## {tier_emoji} 추천 결과 — **{tier_label}** ({risk_score}/100)')
st.markdown(RISK_TIER_DESC[tier])

top3 = recommend_top3(mt, risk_score, period)
if not top3:
    st.warning('추천할 후보 없음')
    st.stop()


# ── Top 3 카드 (3 column) ────────────────────────────────────
st.divider()
st.subheader(f'🏆 Top 3 후보 (정렬: sortino_{period})')

period_map = {
    'TEST'    : ('2010-01-31', '2023-12-31'),
    'HOLD_OUT': ('2024-01-31', '2025-12-31'),
    'FULL'    : ('2010-01-31', '2025-12-31'),
}
start, end = period_map[period]

cols = st.columns(3)
for col_idx, name in enumerate(top3):
    with cols[col_idx]:
        rank_emoji = ['🥇', '🥈', '🥉'][col_idx]
        st.markdown(f'### {rank_emoji} `{name}`')

        m = get_period_metrics(name, period)
        if m:
            st.metric('Sortino', f'{m["sortino"]:.3f}')
            c11, c12 = st.columns(2)
            c11.metric('CAGR', f'{m["cagr"]*100:.2f}%')
            c12.metric('MDD',  f'{m["mdd"]*100:.2f}%')
            st.metric('Sharpe', f'{m["sharpe"]:.3f}')

        # mini 누적수익 chart
        if name in results:
            ret = results[name]['ret'].loc[start:end].dropna()
            if len(ret) >= 6:
                cum = (1 + ret).cumprod()
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.plot(cum.index, cum.values, color='C0', linewidth=1.4)
                ax.set_yscale('log')
                ax.set_title('누적수익률', fontsize=9)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)   # 메모리 누수 방지

        # 추천 근거
        reason = reasoning_for_recommendation(mt, name, risk_score, period)
        with st.expander('📌 추천 근거', expanded=(col_idx == 0)):
            st.markdown(reason)


st.divider()


# ── 위험성향 비교 ─────────────────────────────────────────────
st.subheader('🔄 위험성향 3 tier 비교')

tiers = [
    ('🛡️ 보수형 (score=20)', 20, 'conservative'),
    ('⚖️ 중립형 (score=50)', 50, 'balanced'),
    ('🚀 공격형 (score=85)', 85, 'aggressive'),
]

comparison_rows = []
for label, score, tier_name in tiers:
    picks = recommend_top3(mt, score, period)
    comparison_rows.append({
        'tier': label,
        'top1': picks[0] if len(picks) > 0 else 'N/A',
        'top2': picks[1] if len(picks) > 1 else 'N/A',
        'top3': picks[2] if len(picks) > 2 else 'N/A',
    })

st.dataframe(pd.DataFrame(comparison_rows),
             use_container_width=True, hide_index=True)


# ── 발표 메시지 ──────────────────────────────────────────────
st.divider()
st.info(f'''
💡 **발표용 핵심 메시지**:

본 시스템은 165 BL 실험을 sortino 기반으로 정렬하여 **위험성향 슬라이더 단일
입력만으로 자동 추천** 합니다. 보수형(score<30) 사용자에게는
사용자 지정 Top 1 후보 **`{TOP_1_NAME}`** 가 1순위로 노출되며, 추가 후보는
MDD-안정성 + sortino_ir 우선으로 선정됩니다.

**왜 Sortino 인가?** Sharpe 가 변동성(상승+하락) 전체를 패널티하는 반면,
Sortino 는 **하방 손실만 패널티** 하므로 보수적·안정 지향 투자자의 선호와
일치합니다.
''')
