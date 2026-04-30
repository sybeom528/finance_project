"""1회용 빌드 스크립트 — 05a_eval_stockwise.ipynb 생성.

Phase 3 Step 5a — 02a (Stockwise) + BL_ml_sw 단독 평가 (Layer 1~4).

scripts/diagnostics.py 의 함수를 호출하여 동일 형식 보장.

평가 항목 (모두 단독, 비교 X):
  Layer 1: 변동성 예측 진단 (RMSE, QLIKE, MZ, DM-test, ...)
  Layer 2: 포트폴리오 단독 (Sharpe, CAPM α, IR, Sortino, CVaR, ...)
  Layer 3: ML → BL 인과 (low/high vol hit rate, P 안정성)
  Layer 4: 시기별 분해 (5 시기 × 모든 메트릭)
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

NB = nbf.v4.new_notebook()
NB.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
}

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))


# ============================================================================
# 헤더
# ============================================================================
md("""# Phase 3 Step 5a — 02a Stockwise 단독 평가 (`05a_eval_stockwise.ipynb`)

> **목적**: 02a 종목별 ensemble (Stockwise LSTM + HAR) 의 변동성 예측 + BL_ml_sw 포트폴리오 성과를
>          **다른 모델·시나리오와 비교 없이 단독으로** 평가한다.
>          비교는 별도 노트북 (`05c_eval_compare.ipynb`) 에서 수행.

## 평가 4 레이어

| § | Layer | 내용 |
|---|---|---|
| §2 | **Layer 1** | 변동성 예측 진단 (RMSE/QLIKE/MZ/DM-test/Best model) |
| §3 | **Layer 2** | BL_ml_sw 포트폴리오 단독 (Sharpe/CAPM α/IR/Sortino/CVaR/turnover) |
| §4 | **Layer 3** | ML → BL 인과 추적 (low/high vol hit rate, P 행렬 안정성) |
| §5 | **Layer 4** | 시기별 분해 (5 시기 × 모든 메트릭) |
| §6 | 종합 요약 | render_diagnostic_summary 자동 생성 |

## 평가 모듈 (`scripts/diagnostics.py`)
모든 평가 함수는 모듈에 정의됨 → 호출만 수행 → **05b/05c 와 동일 형식 보장**.

## 사전 조건
- ✅ `data/ensemble_predictions_stockwise.csv` (02a 결과)
- ✅ `data/daily_panel.csv`
- ✅ `outputs/03_bl_backtest/returns_BL_ml_sw.csv` (03 결과)
- 🔵 02a 학습 진행 중에도 본 노트북 코드 작성·검증 가능 (실행은 결과 가용 후)
""")


# ============================================================================
# §1 환경
# ============================================================================
md("""## §1. 환경 부트스트랩 + 결과 로드""")

code("""%load_ext autoreload
%autoreload 2

import sys, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

NB_DIR = Path.cwd()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR
import scripts.diagnostics as diag

font_used = bootstrap()

OUT_DIR = OUTPUTS_DIR / '05a_eval_stockwise'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'OUT_DIR: {OUT_DIR}')""")

code("""# 02a 결과 로드
ens_sw_path = DATA_DIR / 'ensemble_predictions_stockwise.csv'
assert ens_sw_path.exists(), f'02a 결과 없음: {ens_sw_path}'
ens_sw = pd.read_csv(ens_sw_path, parse_dates=['date'])
print(f'ensemble_sw: {ens_sw.shape}')
print(f'  unique 종목: {ens_sw["ticker"].nunique()}')
print(f'  날짜 범위: {ens_sw["date"].min().date()} ~ {ens_sw["date"].max().date()}')
print(f'  컬럼: {list(ens_sw.columns)}')""")

code("""# Daily panel (vol_21d 비교용)
panel = pd.read_csv(
    DATA_DIR / 'daily_panel.csv', parse_dates=['date'],
    usecols=['date', 'ticker', 'vol_21d', 'log_ret', 'log_mcap', 'spy_close'],
)
print(f'panel: {panel.shape}')

# Market data
market = pd.read_csv(DATA_DIR / 'market_data.csv', index_col='date', parse_dates=True)
spy_daily = market['SPY'].pct_change().dropna()
spy_monthly = (1 + spy_daily).resample('ME').prod() - 1
print(f'spy_monthly: {len(spy_monthly)} 개월')""")

code("""# BL_ml_sw 포트폴리오 결과 로드
bl_returns_path = OUTPUTS_DIR / '03_bl_backtest' / 'returns_BL_ml_sw.csv'

if bl_returns_path.exists():
    bl_ml_sw_returns = pd.read_csv(bl_returns_path, index_col=0, parse_dates=True).squeeze()
    print(f'BL_ml_sw returns: {len(bl_ml_sw_returns)} 개월')
    print(f'  기간: {bl_ml_sw_returns.index[0].date()} ~ {bl_ml_sw_returns.index[-1].date()}')
else:
    print(f'⚠️ BL_ml_sw 결과 없음: {bl_returns_path}')
    print('  → 03 노트북 실행 후 본 노트북 §3~§6 재실행 필요')
    bl_ml_sw_returns = None""")


# ============================================================================
# §2 Layer 1
# ============================================================================
md("""## §2. Layer 1 — 02a 변동성 예측 단독 진단

`evaluate_volatility_prediction()` 호출:
- RMSE, QLIKE (Patton 2011), R²_train_mean
- Mincer-Zarnowitz regression (편향 진단)
- pred_std_ratio (mean-collapse)
- Spearman rank (BL P 행렬 입력 품질)
- DM-test vs HAR (학술 표준)
- Best model 분포 (LSTM vs HAR vs Ensemble)
""")

code("""# Layer 1 호출
result_sw = diag.evaluate_volatility_prediction(
    pred_df=ens_sw,
    model_name='02a Stockwise Ensemble',
    pred_col='y_pred_ensemble',
    true_col='y_true',
    har_pred_col='y_pred_har',
)

# 전체 메트릭 출력
print('=== Layer 1 전체 메트릭 ===')
overall = result_sw['overall']
for k in diag.METRIC_ORDER_PREDICTION:
    if k in overall:
        v = overall[k]
        print(f'  {k:25s} = {v:.4f}' if not np.isnan(v) else f'  {k:25s} = NaN')""")

code("""# 종목별 RMSE 통계 + Best 모델
print('=== 종목별 RMSE 통계 ===')
print(result_sw['by_ticker'][['rmse', 'qlike', 'spearman']].describe())

print('\\n=== Best 모델 분포 ===')
print(result_sw['best_model'])""")

code("""# Layer 1 시각화 (6 panel)
fig = diag.plot_prediction_diagnostic_panel(
    result_sw,
    save_path=OUT_DIR / 'layer1_prediction_diagnostic.png',
)
plt.show()""")


# ============================================================================
# §3 Layer 2
# ============================================================================
md("""## §3. Layer 2 — BL_ml_sw 포트폴리오 단독 진단

`evaluate_portfolio_standalone()` 호출:
- Sharpe, CAGR, MDD, ann_vol
- CAPM α/β/t-stat (시장 위험 조정)
- Information ratio (vs SPY)
- Sortino (downside 위험)
- Calmar (CAGR/|MDD|)
- Hit rate, skew, kurt
- CVaR_5, VaR_5 (tail risk)
- Turnover, top-10 concentration (가중치 제공 시)
""")

code("""if bl_ml_sw_returns is not None:
    metrics_l2 = diag.evaluate_portfolio_standalone(
        returns=bl_ml_sw_returns,
        scenario_name='BL_ml_sw',
        spy_returns=spy_monthly,
        rf_returns=None,
        weights_dict=None,    # weights_dict 가 03 에서 별도 저장되어 있다면 여기 추가
    )

    print('=== Layer 2 전체 메트릭 ===')
    for k in diag.METRIC_ORDER_PORTFOLIO:
        if k in metrics_l2:
            v = metrics_l2[k]
            unit = '%' if k in ('cagr', 'ann_vol', 'mdd', 'capm_alpha', 'cvar_5', 'var_5', 'hit_rate') else ''
            if isinstance(v, (int, np.integer)):
                print(f'  {k:25s} = {v}')
            elif np.isnan(v):
                print(f'  {k:25s} = NaN')
            else:
                print(f'  {k:25s} = {v:.3f}{unit}')
else:
    metrics_l2 = None
    print('⚠️ BL_ml_sw returns 없음 → Layer 2 skip')""")

code("""# Layer 2 시각화 (6 panel)
if bl_ml_sw_returns is not None:
    fig = diag.plot_portfolio_diagnostic_panel(
        returns=bl_ml_sw_returns,
        scenario_name='BL_ml_sw',
        spy_returns=spy_monthly,
        save_path=OUT_DIR / 'layer2_portfolio_diagnostic.png',
    )
    plt.show()""")


# ============================================================================
# §4 Layer 3
# ============================================================================
md("""## §4. Layer 3 — ML → BL 인과 추적

`evaluate_ml_to_bl_pipeline()` 호출:
- low_vol_hit_rate: 예측 하위 30% ∩ 실제 하위 30% (BL Long 정확도)
- high_vol_hit_rate: 예측 상위 30% ∩ 실제 상위 30% (BL Short 정확도)
- rank_consistency: 매월 Spearman 시계열
- p_matrix_turnover: P 행렬 선택 종목 안정성
""")

code("""# Layer 3 호출 — 예측 vs 실제 vol 시점별 매칭
# weights_dict 가 03 에서 별도 저장되지 않은 경우 dummy 로 대체
weights_dict_dummy = {pd.Timestamp(d): pd.Series(dtype=float)
                       for d in ens_sw['date'].unique()}

causality_sw = diag.evaluate_ml_to_bl_pipeline(
    pred_df=ens_sw,
    weights_dict=weights_dict_dummy,
    panel=panel,
    scenario_name='BL_ml_sw',
    pred_col='y_pred_ensemble',
    pct=0.30,
)

print('=== Layer 3 메트릭 ===')
print(f'  low_vol_hit_rate:  {causality_sw["low_vol_hit_rate"]:.3f} (random=0.30)')
print(f'  high_vol_hit_rate: {causality_sw["high_vol_hit_rate"]:.3f} (random=0.30)')
rc = causality_sw['rank_consistency_timeline']
print(f'  rank_consistency 평균: {rc.mean():.3f} (n={len(rc)} 시점)')
tov = causality_sw['p_matrix_turnover']
print(f'  P 행렬 turnover 평균: {tov.mean():.3f} (n={len(tov)} 시점)')""")

code("""# Layer 3 시각화 (3 panel)
fig = diag.plot_ml_bl_diagnostic_panel(
    causality_sw,
    scenario_name='BL_ml_sw',
    save_path=OUT_DIR / 'layer3_ml_to_bl_causality.png',
)
plt.show()""")


# ============================================================================
# §5 Layer 4
# ============================================================================
md("""## §5. Layer 4 — 시기별 분해 (5 시기)

| 시기 | 기간 | 시장 환경 |
|---|---|---|
| GFC 회복 | 2009~2011 | 강력한 경기 회복 |
| 정상 강세장 | 2012~2019 | 10년 강세장 |
| COVID 충격 | 2020 | 팬데믹 + 급반등 |
| 긴축·전환 | 2021~2022 | 인플레·금리 급등 |
| 회복·AI | 2023~2025 | AI 붐, 양극화 |
""")

code("""# Layer 4 호출 — BL_ml_sw 시기별 분해
if bl_ml_sw_returns is not None:
    period_metrics_sw = diag.evaluate_by_period(
        returns_dict={'BL_ml_sw': bl_ml_sw_returns},
        periods=diag.PERIODS,
        spy_returns=spy_monthly,
    )

    print('=== Layer 4 시기별 메트릭 (BL_ml_sw) ===')
    # 주요 메트릭만 표시
    key_metrics = ['sharpe', 'cagr', 'mdd', 'sortino', 'capm_alpha', 'hit_rate']
    if not period_metrics_sw.empty:
        # period × metric × scenario → 보기 쉽게 reshape
        period_table = period_metrics_sw['BL_ml_sw'].unstack('metric')
        if not period_table.empty:
            display_cols = [c for c in key_metrics if c in period_table.columns]
            print(period_table[display_cols].round(3).to_string())
else:
    period_metrics_sw = None
    print('⚠️ BL_ml_sw returns 없음 → Layer 4 skip')""")

code("""# Layer 4 시각화
if bl_ml_sw_returns is not None and period_metrics_sw is not None and not period_metrics_sw.empty:
    fig = diag.plot_period_decomposition(
        period_metrics=period_metrics_sw,
        metrics_to_plot=['sharpe', 'cagr', 'mdd', 'sortino'],
        save_path=OUT_DIR / 'layer4_period_decomposition.png',
    )
    plt.show()""")


# ============================================================================
# §6 종합 요약
# ============================================================================
md("""## §6. 종합 요약 (render_diagnostic_summary)

위 4 레이어 결과를 markdown 으로 통합 → 보고서 자동 저장.
""")

code("""# 종합 요약 markdown 생성
layers_for_summary = {'layer1': result_sw}
if metrics_l2 is not None:
    layers_for_summary['layer2'] = metrics_l2
layers_for_summary['layer3'] = causality_sw

summary_md = diag.render_diagnostic_summary(
    model_name='02a Stockwise (BL_ml_sw)',
    layer_results=layers_for_summary,
)

print(summary_md)

# markdown 저장
summary_path = OUT_DIR / 'eval_summary_stockwise.md'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_md)
print(f'\\n✅ 저장: {summary_path}')""")

code("""# 메트릭 JSON 저장 (05c 비교에서 재사용)
import json

eval_results = {
    'model_name': '02a Stockwise',
    'layer1_overall': {k: (None if isinstance(v, float) and np.isnan(v) else v)
                        for k, v in result_sw['overall'].items()},
    'layer2_metrics': {} if metrics_l2 is None else {
        k: (None if isinstance(v, float) and np.isnan(v) else (
            int(v) if isinstance(v, (np.integer, int)) else float(v)
        ))
        for k, v in metrics_l2.items()
    },
    'layer3_metrics': {
        'low_vol_hit_rate': float(causality_sw['low_vol_hit_rate']),
        'high_vol_hit_rate': float(causality_sw['high_vol_hit_rate']),
        'rank_consistency_mean': float(causality_sw['rank_consistency_timeline'].mean())
                                  if len(causality_sw['rank_consistency_timeline']) > 0 else None,
        'p_matrix_turnover_mean': float(causality_sw['p_matrix_turnover'].mean())
                                   if len(causality_sw['p_matrix_turnover']) > 0 else None,
    },
}

json_path = OUT_DIR / 'eval_metrics_stockwise.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)
print(f'✅ 저장: {json_path}')

print('\\n=== Phase 3 Step 5a 완료 ===')
print('다음 단계: 05b_eval_crosssec.ipynb (02b + BL_ml_cs 단독)')""")


# ============================================================================
# 저장
# ============================================================================
NB.cells = cells

OUT_PATH = Path(__file__).parent / '05a_eval_stockwise.ipynb'
nbf.write(NB, str(OUT_PATH))
print(f'✅ 저장 완료: {OUT_PATH}')
