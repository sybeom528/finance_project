"""
recommendations.py — 위험성향별 Top N 매핑 로직.

mat_eq_eq_lam_pap 을 보수형 1 순위로 강제 노출 (사용자 지정).
"""
from __future__ import annotations

import pandas as pd

from .narrative import TOP_1_NAME


def map_risk_score_to_tier(risk_score: int) -> str:
    """0~100 슬라이더 → 'conservative' / 'balanced' / 'aggressive'."""
    if risk_score < 30:
        return 'conservative'
    elif risk_score < 70:
        return 'balanced'
    else:
        return 'aggressive'


def recommend_top3(mt: pd.DataFrame, risk_score: int,
                   period: str = 'TEST') -> list[str]:
    """위험성향 슬라이더 점수 → Top 3 추천 실험명.

    Parameters
    ----------
    mt : pd.DataFrame
      build_master_table() 결과 DataFrame. sortino, sortino_<period>, sortino_ir,
      mdd_<period>, alpha 등 컬럼이 있어야 함.
    risk_score : int
      0~100 슬라이더 값.
    period : str
      'TEST' | 'HOLD_OUT' | 'FULL' — sortino metric 기간 선택.

    Returns
    -------
    list[str] — 실험명 3개 (Top 1 의 가장 적합한 모델 우선)
    """
    tier = map_risk_score_to_tier(risk_score)

    sortino_col = f'sortino_{period}' if f'sortino_{period}' in mt.columns else 'sortino'
    mdd_col     = f'mdd_{period}'     if f'mdd_{period}'     in mt.columns else 'mdd'

    if tier == 'conservative':
        # 사용자 지정 mat_eq_eq_lam_pap 우선 + MDD 보수 + sortino_ir 안정
        base = []
        if TOP_1_NAME in mt['name'].values:
            base = [TOP_1_NAME]

        # 추가 후보 — MDD 가 -0.18 이하 + sortino_ir 상위 (안정성)
        rest_pool = mt[(mt[mdd_col] > -0.18) & (mt['name'] != TOP_1_NAME)] \
            if mdd_col in mt.columns else mt[mt['name'] != TOP_1_NAME]
        sort_key = 'sortino_ir' if 'sortino_ir' in rest_pool.columns else sortino_col
        rest = rest_pool.nlargest(2, sort_key)['name'].tolist()
        out = base + rest

        # 3개 미만이면 mt 전체에서 추가로 채움 (MDD 필터 완화)
        if len(out) < 3:
            backup = mt[~mt['name'].isin(out)].nlargest(3 - len(out), sortino_col)['name'].tolist()
            out = out + backup
        return out[:3]

    elif tier == 'balanced':
        # sortino_ir 우선 (3-레짐 안정성)
        sort_key = 'sortino_ir' if 'sortino_ir' in mt.columns else sortino_col
        return mt.nlargest(3, sort_key)['name'].tolist()

    else:  # aggressive
        # absolute sortino + alpha (Jensen)
        sort_key = sortino_col
        top_pool = mt.nlargest(10, sort_key)
        if 'alpha' in top_pool.columns:
            top_pool = top_pool.sort_values('alpha', ascending=False)
        return top_pool.head(3)['name'].tolist()


def reasoning_for_recommendation(mt: pd.DataFrame, name: str,
                                 risk_score: int,
                                 period: str = 'TEST') -> str:
    """단일 추천 실험명에 대한 "왜?" markdown narrative."""
    tier = map_risk_score_to_tier(risk_score)
    row = mt[mt['name'] == name]
    if len(row) == 0:
        return f'{name} 정보 없음'
    r = row.iloc[0]

    sortino_col = f'sortino_{period}' if f'sortino_{period}' in r else 'sortino'
    mdd_col     = f'mdd_{period}'     if f'mdd_{period}'     in r else 'mdd'

    so   = float(r[sortino_col]) if pd.notna(r[sortino_col]) else float('nan')
    sh   = float(r['sharpe']) if 'sharpe' in r and pd.notna(r['sharpe']) else float('nan')
    mdd  = float(r[mdd_col]) if pd.notna(r[mdd_col]) else float('nan')
    cagr = float(r['cagr']) if pd.notna(r['cagr']) else float('nan')

    canonical = r.get('canonical', name)
    sortino_ir_val = r.get('sortino_ir', float('nan'))
    sortino_ir_str = f'{float(sortino_ir_val):.2f}' if pd.notna(sortino_ir_val) else 'N/A'

    tier_emoji = {'conservative': '🛡️', 'balanced': '⚖️', 'aggressive': '🚀'}[tier]
    tier_label = {'conservative': '보수형', 'balanced': '중립형', 'aggressive': '공격형'}[tier]

    return f'''
{tier_emoji} **{tier_label}** 추천 — `{name}` (canonical: `{canonical}`)

| 메트릭 | 값 |
|---|---|
| Sortino ({period}) | **{so:.3f}** |
| Sharpe | {sh:.3f} |
| CAGR | {cagr*100:.2f}% |
| MDD ({period}) | {mdd*100:.2f}% |
| Sortino IR (3-레짐) | {sortino_ir_str} |
'''
