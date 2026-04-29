"""1회용 빌드 스크립트 — 02a_phase15_stockwise_extended.ipynb 생성.

Phase 3 Step 2a — 종목별 (Stock-wise) LSTM 학습 (17 년 확장).

Phase 1.5 v8 Performance-Weighted Ensemble (LSTM v4 + HAR-RV) 을
8-way 병렬 (RTX 4090 24GB) 로 전체 universe 종목에 확장 적용.
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
md("""# Phase 3 Step 2a — 종목별 LSTM 학습 (`02a_phase15_stockwise_extended.ipynb`)

> **목적**: Phase 1.5 v8 Performance-Weighted Ensemble 을 17 년 (2009~2025) 확장 universe 전체에
>          8-way 병렬 (RTX 4090 24GB) 로 학습·예측.

## 모델 구성

- **LSTM v4** (3ch_vix: rv_d, rv_w, rv_m + vix_log, IS=1250, embargo=63)
- **HAR-RV** (1d, 5d, 22d 이동평균 선형 회귀)
- **Ensemble** = Performance-Weighted (Diebold-Pauly rolling)

## 실행 전 확인

1. `01_universe_extended.ipynb` 완료 (universe + panel 준비)
2. RTX 4090 24GB GPU 가용
3. CUDA + PyTorch 정상 (검증 셀 §2)

## 산출물

| 파일 | 내용 |
|---|---|
| `data/ensemble_predictions_stockwise.csv` | 종목별 예측 (date, ticker, y_pred_lstm, y_pred_har, y_pred_ensemble, y_true) |

## 예상 시간

- **8-way 병렬 (RTX 4090)**: 약 1~2 시간
- 종목 수 × fold 수 / 8 병렬

## 셀 구성

| § | 내용 |
|---|---|
| §1 | 환경 + 경로 |
| §2 | GPU 환경 확인 |
| §3 | 8-way 병렬 학습 실행 |
| §4 | 결과 검증 (RMSE 분포, best model 분포) |
| §5 | Phase 1.5 v8 결과 비교 (74 종목 6 년 vs 확장) |
""")


# ============================================================================
# §1 환경
# ============================================================================
md("""## §1. 환경 부트스트랩""")

code("""%load_ext autoreload
%autoreload 2

import sys, time
from pathlib import Path

NB_DIR = Path.cwd()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR

font_used = bootstrap()

OUT_DIR = OUTPUTS_DIR / '02a_stockwise'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'OUT_DIR: {OUT_DIR}')""")


# ============================================================================
# §2 GPU 환경
# ============================================================================
md("""## §2. GPU 환경 확인""")

code("""import torch

print('=== GPU 환경 ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_mem = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated(0)) / 1e9
    print(f'Total VRAM: {total_mem:.1f} GB')
    print(f'Free  VRAM: {free_mem:.1f} GB')
    assert free_mem >= 3.0, f'VRAM 부족: {free_mem:.1f} GB < 3 GB'
    print('✅ GPU 조건 충족')
else:
    print('⚠️ CPU 모드 (매우 느림)')""")


# ============================================================================
# §3 8-way 병렬 학습
# ============================================================================
md("""## §3. 8-way 병렬 학습 (V4_BEST_CONFIG)

`run_ensemble_for_universe_parallel()`:
- `n_workers=8` → RTX 4090 24GB 최적화
- GPU 메모리 부족 시 `n_workers=4` 로 감소
- 학습 시간: 약 1~2 시간

> ⚠️ 배경 실행 권장 (jupyter nbconvert --execute)
""")

code("""from scripts.volatility_ensemble import run_ensemble_for_universe_parallel, V4_BEST_CONFIG

print('=== 하이퍼파라미터 ===')
for k, v in V4_BEST_CONFIG.items():
    print(f'  {k}: {v}')""")

code("""# 데이터 확인
panel_path = DATA_DIR / 'daily_panel.csv'
universe_path = DATA_DIR / 'universe_top50_history_extended.csv'

assert panel_path.exists(), f'panel 없음: {panel_path}'
assert universe_path.exists(), f'universe 없음: {universe_path}'

panel_info = pd.read_csv(panel_path, usecols=['date', 'ticker'], parse_dates=['date'])
uni_info = pd.read_csv(universe_path)
print(f'panel: {len(panel_info):,} 행, {panel_info["ticker"].nunique()} 종목')
print(f'universe: {uni_info["ticker"].nunique()} unique 종목, {uni_info["oos_year"].nunique()} 연도')""")

code("""# ⭐ 8-way 병렬 학습 실행
# GPU OOM 발생 시: n_workers=4, batch_size=32 으로 감소
t0 = time.time()

ensemble_sw = run_ensemble_for_universe_parallel(
    panel_csv=panel_path,
    universe_csv=universe_path,
    out_dir=DATA_DIR,
    config=V4_BEST_CONFIG,
    n_workers=8,               # RTX 4090 24GB
    out_name='ensemble_predictions_stockwise.csv',
    verbose=True,
)

elapsed = time.time() - t0
print(f'\\n⏱️ 총 소요 시간: {elapsed/60:.1f} 분')
print(f'결과: {ensemble_sw.shape}')
ensemble_sw.head()""")


# ============================================================================
# §4 결과 검증
# ============================================================================
md("""## §4. 결과 검증""")

code("""# 4-1. 기본 통계
print('=== 산출 CSV 기본 통계 ===')
print(f'행 수: {len(ensemble_sw):,}')
print(f'unique 종목: {ensemble_sw["ticker"].nunique()}')
print(f'unique fold: {ensemble_sw["fold"].nunique()}')
print(f'date 범위: {ensemble_sw["date"].min().date()} ~ {ensemble_sw["date"].max().date()}')
print()
print('컬럼별 NaN 수:')
print(ensemble_sw[['y_pred_lstm', 'y_pred_har', 'y_pred_ensemble', 'y_true']].isna().sum())""")

code("""# 4-2. 종목별 RMSE 계산
def rmse(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))

rmse_by_ticker = ensemble_sw.groupby('ticker').apply(
    lambda df: pd.Series({
        'rmse_lstm': rmse(df['y_true'].values, df['y_pred_lstm'].values),
        'rmse_har': rmse(df['y_true'].values, df['y_pred_har'].values),
        'rmse_ensemble': rmse(df['y_true'].values, df['y_pred_ensemble'].values),
    })
).reset_index()

print('=== 종목별 RMSE 통계 ===')
print(rmse_by_ticker[['rmse_lstm', 'rmse_har', 'rmse_ensemble']].describe())""")

code("""# 4-3. Best 모델 분포
best_model = rmse_by_ticker.apply(
    lambda row: 'lstm' if row['rmse_lstm'] == min(row['rmse_lstm'], row['rmse_har'], row['rmse_ensemble'])
               else ('har' if row['rmse_har'] == min(row['rmse_lstm'], row['rmse_har'], row['rmse_ensemble'])
                    else 'ensemble'),
    axis=1
)
print('=== Best 모델 분포 ===')
print(best_model.value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE 분포
axes[0].hist(rmse_by_ticker['rmse_ensemble'], bins=30, alpha=0.7, label='Ensemble', color='steelblue')
axes[0].hist(rmse_by_ticker['rmse_har'], bins=30, alpha=0.6, label='HAR-RV', color='darkorange')
axes[0].hist(rmse_by_ticker['rmse_lstm'], bins=30, alpha=0.5, label='LSTM', color='green')
axes[0].set_title('종목별 RMSE 분포 (02a 종목별 학습)', fontsize=12)
axes[0].set_xlabel('RMSE')
axes[0].legend()

# Best 모델 분포
best_model.value_counts().plot(kind='bar', ax=axes[1], color=['steelblue', 'darkorange', 'green'])
axes[1].set_title('Best 모델 분포', fontsize=12)
axes[1].set_xlabel('모델')
axes[1].set_ylabel('종목 수')

plt.tight_layout()
plt.savefig(OUT_DIR / 'rmse_distribution.png', dpi=100, bbox_inches='tight')
plt.show()""")


# ============================================================================
# §5 Phase 1.5 비교
# ============================================================================
md("""## §5. Phase 1.5 v8 결과 비교

Phase 1.5 v8 (74 종목, 6 년 2020~2025) vs Phase 3 (확장 universe, 17 년 2009~2025).
""")

code("""from scripts.setup import PHASE2_DIR

phase2_ens_path = PHASE2_DIR / 'data' / 'ensemble_predictions_top50.csv'
if phase2_ens_path.exists():
    phase2_ens = pd.read_csv(phase2_ens_path, parse_dates=['date'])

    # Phase 2 와 공통 종목
    common_tickers = set(ensemble_sw['ticker'].unique()) & set(phase2_ens['ticker'].unique())
    print(f'공통 종목 수: {len(common_tickers)}')

    # Phase 2 vs Phase 3 RMSE 비교 (공통 종목 + 2021 이후 공통 기간)
    sw_sub = ensemble_sw[
        (ensemble_sw['ticker'].isin(common_tickers)) &
        (ensemble_sw['date'] >= '2021-01-01')
    ]
    p2_sub = phase2_ens[
        (phase2_ens['ticker'].isin(common_tickers)) &
        (phase2_ens['date'] >= '2021-01-01')
    ]

    sw_rmse = sw_sub.groupby('ticker').apply(
        lambda df: rmse(df['y_true'].values, df['y_pred_ensemble'].values)
    ).mean()
    p2_rmse = p2_sub.groupby('ticker').apply(
        lambda df: rmse(df['y_true'].values, df['y_pred_ensemble'].values)
    ).mean()

    print(f'Phase 2 Ensemble RMSE (2021+): {p2_rmse:.4f}')
    print(f'Phase 3 Stockwise RMSE (2021+): {sw_rmse:.4f}')
    print(f'차이 (Phase 3 - Phase 2): {sw_rmse - p2_rmse:+.4f}')
else:
    print(f'Phase 2 ensemble 파일 없음: {phase2_ens_path}')""")

code("""# 최종 요약
print('=== Phase 3 Step 2a 완료 ===')
print(f'산출물: {DATA_DIR / "ensemble_predictions_stockwise.csv"}')
print(f'종목 수: {ensemble_sw["ticker"].nunique()}')
print(f'기간: {ensemble_sw["date"].min().date()} ~ {ensemble_sw["date"].max().date()}')
print()
print('다음 단계:')
print('  02b_phase15_cross_sectional.ipynb   (Cross-Sectional 학습)')
print('  03_BL_backtest_extended.ipynb        (BL 백테스트 — 02a + 02b 완료 후)')""")


# ============================================================================
# 저장
# ============================================================================
NB.cells = cells

OUT_PATH = Path(__file__).parent / '02a_phase15_stockwise_extended.ipynb'
nbf.write(NB, str(OUT_PATH))
print(f'✅ 저장 완료: {OUT_PATH}')
