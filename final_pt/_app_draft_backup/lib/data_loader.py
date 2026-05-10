"""
data_loader.py — Streamlit 대시보드용 캐시된 데이터 로더.

핵심 책임:
  - 165 BL pkl 일괄 로드 (cache_resource, 첫 로드 ~5초)
  - master_table TEST/HOLD-OUT/FULL 기간별 빌드 (cache_data, 기간 toggle 시 1회 재계산)
  - monthly_panel / spy / rf 시리즈 (cache_data)
  - LSTM ensemble 예측 (cache_data, 큰 csv → 한 번만 로드)

기존 final/ 의 .py 모듈 (bl_functions, master_table) 을 import 하여 재사용.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

import sys
FINAL_DIR = Path(__file__).resolve().parent.parent.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from master_table import build_master_table, EVAL_PERIODS  # noqa: E402

DATA_DIR    = FINAL_DIR / 'data'
RESULTS_DIR = FINAL_DIR / 'results'
PHASE3_DIR  = FINAL_DIR / 'phase3(data_outputs)'
OUTPUTS_DIR = FINAL_DIR / 'outputs'

# 발표용 추천 Top 1 (사용자 지정, 2026-05-07)
TOP_1_NAME = 'mat_eq_eq_lam_pap'


# ══════════════════════════════════════════════════════════════
# 1. 데이터 로더 (cached)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner='165 BL 결과 pkl 로드 중...')
def load_all_results() -> dict:
    """results/*.pkl 일괄 로드. _backup_pre_extension/ 폴더는 제외.

    Returns
    -------
    dict[name, dict]
      {실험명: {ret, gross_ret, spy_ret, weights, comp, meta, errors, config}}
    """
    if not RESULTS_DIR.exists():
        st.error(f'results/ 폴더 없음: {RESULTS_DIR}')
        return {}
    out = {}
    for pkl in sorted(RESULTS_DIR.glob('*.pkl')):
        try:
            with open(pkl, 'rb') as f:
                out[pkl.stem] = pickle.load(f)
        except Exception as e:
            st.warning(f'{pkl.name} 로드 실패: {e}')
    return out


@st.cache_data(show_spinner='monthly_panel 로드 중...')
def load_panel() -> pd.DataFrame:
    """monthly_panel.csv 로드 + (date, ticker) MultiIndex."""
    p = DATA_DIR / 'monthly_panel.csv'
    if not p.exists():
        st.error(f'monthly_panel.csv 없음: {p}')
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=['date'])
    return df.set_index(['date', 'ticker'])


@st.cache_data
def load_rf_spy() -> tuple[pd.Series, pd.Series]:
    """월별 무위험수익률 + SPY 수익률 시리즈."""
    panel = load_panel()
    if len(panel) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    rf  = panel['rf_1m'].groupby(level='date').first()
    spy = panel['spy_ret'].groupby(level='date').first()
    return rf, spy


@st.cache_data(show_spinner='LSTM ensemble 예측 로드 중...')
def load_ensemble_predictions(usecols: list = None) -> pd.DataFrame:
    """LSTM ensemble 예측 csv (~2.4M rows)."""
    p = PHASE3_DIR / 'data' / 'ensemble_predictions_stockwise.csv'
    if not p.exists():
        st.error(f'ensemble_predictions_stockwise.csv 없음: {p}')
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=['date'], usecols=usecols)


# ══════════════════════════════════════════════════════════════
# 2. master_table 캐시 (기간별)
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner='master_table 빌드 중...')
def get_master_table(sort_by: str = 'sortino_TEST') -> pd.DataFrame:
    """전체 165 실험의 master table.

    sortino_TEST/HOLD_OUT/FULL + sharpe_*/cagr_*/mdd_* 컬럼 포함.

    Parameters
    ----------
    sort_by : str
      default sort 키. 'sortino_TEST' (학습기간 정렬), 'sortino_HOLD_OUT' (실전),
      'sortino' (overall), 'sharpe' 등. 모든 컬럼명 가능.
    """
    rf, spy = load_rf_spy()
    return build_master_table(RESULTS_DIR, rf, spy, sort_by=sort_by)


def get_period_metrics(name: str, period: str = 'TEST') -> dict:
    """단일 실험의 단일 기간 metric dict.

    Parameters
    ----------
    name : 실험명 (예: 'mat_eq_eq_lam_pap')
    period : 'TEST' | 'HOLD_OUT' | 'FULL'
    """
    if period not in EVAL_PERIODS:
        raise ValueError(f'period={period!r} not in {list(EVAL_PERIODS)}')
    s, e = EVAL_PERIODS[period]
    results = load_all_results()
    if name not in results:
        return {}
    res = results[name]
    ret = res.get('ret', pd.Series(dtype=float))
    if not isinstance(ret, pd.Series) or len(ret) == 0:
        return {}
    sub = ret[(ret.index >= s) & (ret.index <= e)]
    if len(sub) < 6:
        return {}
    rf, spy = load_rf_spy()
    rf_sub  = rf.reindex(sub.index).fillna(0)
    spy_sub = res.get('spy_ret', spy).reindex(sub.index).fillna(0)
    exc     = sub - rf_sub
    vol     = sub.std() * (12 ** 0.5)
    sharpe  = float(exc.mean() * 12 / vol) if vol > 0 else float('nan')
    down    = sub[sub < 0].std() * (12 ** 0.5)
    sortino = float(exc.mean() * 12 / down) if (down and down > 0) else float('nan')
    cum     = (1 + sub).cumprod()
    cagr    = float(cum.iloc[-1] ** (12 / len(sub)) - 1) if cum.iloc[-1] > 0 else float('nan')
    mdd     = float((cum / cum.cummax() - 1).min())
    return {
        'sharpe' : sharpe,
        'sortino': sortino,
        'cagr'   : cagr,
        'mdd'    : mdd,
        'vol'    : float(vol),
        'n'      : len(sub),
        'period' : period,
        'start'  : sub.index.min(),
        'end'    : sub.index.max(),
    }


# ══════════════════════════════════════════════════════════════
# 3. 헬퍼
# ══════════════════════════════════════════════════════════════

def list_experiment_names() -> list[str]:
    """results/ 의 모든 실험명 (정렬)."""
    return sorted([p.stem for p in RESULTS_DIR.glob('*.pkl')])


def is_data_ready() -> tuple[bool, str]:
    """대시보드 실행 가능 여부 점검 (UX 친화적 에러 메시지용)."""
    if not (DATA_DIR / 'monthly_panel.csv').exists():
        return False, 'final/data/monthly_panel.csv 없음. 01_DataCollection 또는 _rebuild_panel_2025.py 실행 필요.'
    if len(list(RESULTS_DIR.glob('*.pkl'))) < 100:
        return False, f'results/*.pkl 부족 ({len(list(RESULTS_DIR.glob("*.pkl")))}개). 99_run.ipynb 실행 필요.'
    return True, 'OK'
