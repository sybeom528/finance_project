"""
6_📈_Volatility_Stats.py — Phase 1.5 LSTM σ + 04 학술 통계 요약 카드.

RMSE Option A 보조 표기 (학술 표준 + 친숙한 % 보조표기) 적용.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR   = Path(__file__).resolve().parent.parent
FINAL_DIR = APP_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from app.lib.data_loader import load_ensemble_predictions, OUTPUTS_DIR, FINAL_DIR as _FD  # noqa
from app.lib.narrative import VOLATILITY_INTRO

# timeseries_lib 의 헬퍼 사용
import timeseries_lib as tlib  # noqa: E402


st.set_page_config(page_title='Volatility & Stats', page_icon='📈', layout='wide')

st.markdown(VOLATILITY_INTRO)


# ══════════════════════════════════════════════════════════════
# §A. LSTM 변동성 예측 요약
# ══════════════════════════════════════════════════════════════
st.subheader('§A. LSTM 변동성 예측 요약 (Phase 1.5)')

ens = load_ensemble_predictions(usecols=['date', 'ticker', 'y_true', 'y_pred_ensemble',
                                          'y_pred_lstm', 'y_pred_har'])

if len(ens) == 0:
    st.warning('ensemble_predictions_stockwise.csv 로드 실패')
else:
    # NaN 필터링
    ens_v = ens.dropna(subset=['y_true', 'y_pred_ensemble'])
    ens_v = ens_v[np.isfinite(ens_v['y_true']) & np.isfinite(ens_v['y_pred_ensemble'])]

    # 전체 RMSE 보조 표기
    summary_all = tlib.rmse_with_pct_summary(
        ens_v['y_true'].values, ens_v['y_pred_ensemble'].values)

    cA1, cA2, cA3, cA4 = st.columns(4)
    cA1.metric('log-RMSE (학술 표준)', f'{summary_all["rmse_log"]:.4f}',
               help='Patton 2011 표준. log(σ_RV) 공간에서의 RMSE.')
    cA2.metric('평균 σ (보조 표기)', f'{summary_all["mean_sigma_pct"]:.2f}%/일',
               help='exp(y_pred).mean() × 100. 친숙한 % 표기.')
    cA3.metric('상대오차 (Taylor)', f'±{summary_all["rel_error_approx"]:.2f}%',
               help='log RMSE × 100 — 1차 근사. 정확한 metric 은 log-RMSE.')
    cA4.metric('유효 sample 수', f'{summary_all["n"]:,}')

    st.caption(
        f'ℹ️ **Option A** — log RMSE 본값을 의사결정 기준으로 사용 (학술 표준 보존). '
        f'σ %/일 표기는 발표용 보조. log RMSE 와 % 공간 RMSE 는 다른 metric 입니다.'
    )

    # 추가: 주요 종목별 RMSE
    st.markdown('### 📊 7 종목 RMSE 비교 (ensemble vs LSTM vs HAR)')
    target_tickers = ['SPY', 'QQQ', 'DIA', 'EEM', 'XLF', 'GOOGL', 'WMT']
    avail = [t for t in target_tickers if t in ens_v['ticker'].unique()]
    rows = []
    for t in avail:
        sub = ens_v[ens_v['ticker'] == t]
        if len(sub) == 0:
            continue
        for col, label in [('y_pred_ensemble', 'Ensemble'),
                            ('y_pred_lstm', 'LSTM v4'),
                            ('y_pred_har', 'HAR-RV')]:
            if col not in sub.columns:
                continue
            r = tlib.rmse(sub['y_true'].values, sub[col].values)
            rows.append({'ticker': t, 'model': label, 'log_rmse': r,
                          'mean_sigma_pct': float(np.exp(sub[col]).mean() * 100)})
    if rows:
        df_t = pd.DataFrame(rows)
        st.dataframe(df_t.round(4), use_container_width=True, hide_index=True)


st.divider()


# ══════════════════════════════════════════════════════════════
# §B. 학술 통계 검증 요약 (Phase 3-2)
# ══════════════════════════════════════════════════════════════
st.subheader('§B. 학술 통계 검증 요약 (04_Statistical_Validation.ipynb)')

st.markdown('''
**4 학술 명제**:

1. 시기 효과 systematic — η²_period = **0.450** (LARGE) ⭐⭐⭐
2. 종목 difficulty Heavy-Tailed — Skew +1.30, Excess Kurt +4.71 ⭐⭐
3. Sector 효과 통계 유의 — KW H=70.55, ε²=0.121, **14/14 LARGE pair** ⭐⭐
4. COVID 충격 sector-specific — Utilities/Real Estate/Energy +0.20~0.17 ⭐
''')

cB1, cB2, cB3, cB4 = st.columns(4)
cB1.metric('η² (Period)', '0.450', help='Cohen 1988: >0.14 = LARGE effect')
cB2.metric('η² (Ticker)', '0.194')
cB3.metric('Welch F', '420.59', help='이분산 robust ANOVA, p<1e-16')
cB4.metric('Pairwise LARGE', '14/14', help='Bonferroni α=0.05/66, |Cohen d|≥0.8')

st.caption(
    '⚠️ 위 통계는 log-RV 공간에서 계산되었습니다 (학술 표준). '
    '04_Statistical_Validation.ipynb 의 §1.3-B 셀에 발표용 % 보조 표기가 추가됨.'
)


st.divider()


# ── B3 시각화 PNG (있는 경우) ─────────────────────────────────
st.markdown('### 🎨 B3 5 패널 (분포·시기·섹터·COVID·Heavy-tail)')

stats_dir = OUTPUTS_DIR / '04_statistics'
b3_files = [
    ('B3_sector_boxplot.png', 'Sector Boxplot (Kruskal-Wallis)'),
    ('B3_sector_period_heatmap.png', 'Sector × Period Heatmap'),
    ('B3_covid_impact.png', 'COVID Impact Bar (Schwert 1989)'),
    ('B3_heavy_tail_kde.png', 'Heavy-tail KDE + QQ'),
    ('B3_variance_decomp.png', 'Variance Decomposition'),
]

found_any = False
for fname, title in b3_files:
    p = stats_dir / fname
    if p.exists():
        found_any = True
        st.image(str(p), caption=title, use_container_width=True)

if not found_any:
    st.info(
        f'📂 {stats_dir} 의 B3*.png 8 장 자동 탐지 — 04 노트북 실행 후 자동 노출됩니다.\n\n'
        '재현: `cd final && jupyter nbconvert --to notebook --execute --inplace 04_Statistical_Validation.ipynb`'
    )


# ── 재현 노트북 ───────────────────────────────────────────────
st.divider()
st.markdown(f'''
### 📓 재현 노트북

| 단계 | 노트북 | 설명 |
|---|---|---|
| Phase 1.5 LSTM | `final/03_Volatility_Forecasting.ipynb` | 모델 구축 + cache hit ~5분 |
| Phase 3-2 통계 | `final/04_Statistical_Validation.ipynb` | ANOVA/Welch/KW/Heavy-tail ~3분 |

본 페이지의 모든 KPI 와 차트는 위 두 노트북의 산출물입니다.
log-RMSE 본값은 학술 표준을 보존하고, % 보조 표기로 발표 친숙도를 더했습니다.
''')
