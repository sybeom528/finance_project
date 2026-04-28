"""1회용 빌드 스크립트 — 03_phase15_ensemble_top50.ipynb 생성.

Phase 2 Step 3 — Phase 1.5 v8 ensemble 을 74 universe 종목으로 확장.

핵심:
- Phase 1.5 v4 best (3ch_vix, IS=1250) + HAR-RV → Performance-Weighted Ensemble
- 결정 4: 신규 종목만 첫 fold reset, 기존은 history 유지
- 결정 7: OOS=21, embargo=63, step=21 (Phase 1.5 일관)

산출물:
  data/fold_predictions_lstm_har.csv     (74 종목 × ~80 fold 모두의 OOS 예측)
  data/ensemble_predictions_top50.csv    ⭐ Performance-Weighted ensemble 결과
  outputs/03_ensemble/...                 시각화

실행 시간:
- PoC (5 종목): ~30분 (CPU) / ~5분 (GPU)
- 전체 (74 종목): ~24-48시간 (CPU) / ~5-10시간 (GPU)
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
md("""# Phase 2 Step 3 — Phase 1.5 Ensemble 74 종목 확장 (`03_phase15_ensemble_top50.ipynb`)

> **목적**: Phase 1.5 v8 Performance-Weighted Ensemble 을 universe 74 종목 전체에 학습.

## 구성

- **LSTM v4 best (3ch_vix)**: input_size=4 (rv_d, rv_w, rv_m, vix_log), hidden=32, dropout=0.3
- **HAR-RV**: 3 변수 OLS (Corsi 2009, log domain)
- **Performance-Weighted Ensemble**: 이전 fold OOS RMSE 역수 비율 (Diebold-Pauly 1987)

## Walk-Forward 파라미터 (Phase 1.5 v4 일관)

```
IS = 1,250 영업일 (~5년)
purge = 21 (forward target window)
embargo = 63
OOS = 21
step = 21
seq_len = 63
```

## 결정 4 — 신규 종목 reset

```
신규 편입 종목: 첫 fold 0.5/0.5 warmup
기존 종목: 이전 모든 fold history 유지 (compute_performance_weights 가 자동 처리)
```

## ⚠️ 실행 시간

| 모드 | 종목 수 | 시간 (CPU) | 시간 (GPU) |
|---|---|---|---|
| **PoC** | 5 종목 | ~30분 | ~5분 |
| **확장** | 30 종목 | ~3-5시간 | ~30분-1시간 |
| **전체** | 74 종목 | ~24-48시간 | ~5-10시간 |

→ 본 노트북은 **PoC 모드 (5 종목)** 를 default 로 함. 코드 검증 후 사용자가 본 학습 별도 실행.

## 셀 구성

| § | 내용 |
|---|---|
| §1 | 환경 + GPU 확인 + autoreload |
| §2 | universe + daily_panel 로드 |
| §3 | 학습 모드 선택 (PoC / 확장 / 전체) |
| §4 | 종목별 walk-forward 학습 + HAR (시간 병목) |
| §5 | Performance-Weighted ensemble 가중치 계산 |
| §6 | 결과 검증 (RMSE 분포, fold 별 가중치) |
| §7 | 시각화 |
""")


# ============================================================================
# §1 환경
# ============================================================================
md("""## §1. 환경 부트스트랩 + GPU 확인""")

code("""# Jupyter 모듈 자동 리로드
%load_ext autoreload
%autoreload 2

import sys, time
from pathlib import Path

NB_DIR = Path.cwd()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from scripts.setup import bootstrap, BASE_DIR, DATA_DIR, OUTPUTS_DIR
from scripts.volatility_ensemble import (
    V4_BEST_CONFIG,
    run_walkforward_for_ticker,
    compute_performance_weights,
    run_ensemble_for_universe,
)

font_used = bootstrap()

OUT_DIR = OUTPUTS_DIR / '03_ensemble'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU 확인
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\\nDevice: {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')""")


# ============================================================================
# §2 데이터 로드
# ============================================================================
md("""## §2. Universe + Daily Panel 로드""")

code("""universe_csv = DATA_DIR / 'universe_top50_history.csv'
panel_csv = DATA_DIR / 'daily_panel.csv'

universe_df = pd.read_csv(universe_csv, parse_dates=['cutoff_date'])
panel = pd.read_csv(panel_csv, parse_dates=['date'])

print(f'universe: {universe_df.shape} (unique {universe_df[\"ticker\"].nunique()} 종목)')
print(f'panel   : {panel.shape}')
print(f'\\npanel 컬럼: {list(panel.columns)}')
print(f'panel 종목 수: {panel[\"ticker\"].nunique()}')""")


# ============================================================================
# §3 모드 선택
# ============================================================================
md("""## §3. 학습 모드 선택

⭐ **본 노트북은 default 로 PoC 모드** (5 종목, ~30분 CPU / ~5분 GPU).
전체 학습은 본 노트북 검증 후 사용자가 별도 실행 권고 (`MODE = 'full'` 변경).""")

code("""# ⭐ 모드 변경 가능 (사용자 결정: 옵션 A = full 74 종목)
MODE = 'full'   # 'poc' (5 종목) / 'core' (30 종목) / 'full' (74 종목, 본 단계 default)

if MODE == 'poc':
    TICKERS_SUBSET = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B']  # 5 안정 대형주 (코드 검증용)
    print(f'PoC 모드: {len(TICKERS_SUBSET)} 종목')
elif MODE == 'core':
    # super-stable 종목 (6 OOS 연도 모두 등장한 33 종목 중 상위 30)
    cnt = universe_df.groupby('ticker').size().sort_values(ascending=False)
    TICKERS_SUBSET = cnt[cnt == universe_df['oos_year'].nunique()].index.tolist()[:30]
    print(f'Core 모드: {len(TICKERS_SUBSET)} 종목 (super-stable)')
else:
    TICKERS_SUBSET = None
    print(f'Full 모드: 전체 universe (74 종목, 옵션 A) ⭐')

print(f'\\n예상 시간 ({DEVICE}):')
n = len(TICKERS_SUBSET) if TICKERS_SUBSET else universe_df['ticker'].nunique()
sec_per_fold = 5 if DEVICE == 'cuda' else 30
n_folds_avg = 80
total_min = n * n_folds_avg * sec_per_fold / 60
print(f'  {n} 종목 × ~{n_folds_avg} fold × {sec_per_fold}s/fold ≈ {total_min:.0f} 분 ({total_min/60:.1f} 시간)')""")


# ============================================================================
# §4 학습 실행
# ============================================================================
md("""## §4. 종목별 walk-forward 학습 (시간 병목 ⚠️)

LSTM v4 + HAR-RV 매 fold 학습. 진행 출력 확인.""")

code("""print('=' * 70)
print(f'학습 시작: MODE={MODE}, device={DEVICE}')
print('=' * 70)

t0 = time.time()
ensemble_df = run_ensemble_for_universe(
    panel_csv=panel_csv,
    universe_csv=universe_csv,
    out_dir=DATA_DIR,
    config=V4_BEST_CONFIG,
    device=DEVICE,
    tickers_subset=TICKERS_SUBSET,
    verbose=True,
)
elapsed = time.time() - t0
print(f'\\n총 소요: {elapsed/60:.1f} 분')""")


# ============================================================================
# §5 검증
# ============================================================================
md("""## §5. 결과 검증""")

code("""# 5-1. shape 검증
print(f'ensemble_df shape: {ensemble_df.shape}')
print(f'학습 종목 수: {ensemble_df[\"ticker\"].nunique()}')
print(f'fold 수 (종목별 평균): {ensemble_df.groupby(\"ticker\")[\"fold\"].nunique().mean():.0f}')

# 5-2. RMSE 검증 (종목별 LSTM v4, HAR, Ensemble)
print(f'\\n=== 종목별 OOS RMSE 비교 ===')
metrics = []
for ticker, df_t in ensemble_df.groupby('ticker'):
    valid = df_t.dropna(subset=['y_true', 'y_pred_lstm', 'y_pred_har', 'y_pred_ensemble'])
    if len(valid) == 0:
        continue
    rmse_lstm = np.sqrt(((valid['y_pred_lstm'] - valid['y_true']) ** 2).mean())
    rmse_har = np.sqrt(((valid['y_pred_har'] - valid['y_true']) ** 2).mean())
    rmse_ens = np.sqrt(((valid['y_pred_ensemble'] - valid['y_true']) ** 2).mean())
    metrics.append({
        'ticker': ticker,
        'n_obs': len(valid),
        'rmse_lstm': rmse_lstm,
        'rmse_har': rmse_har,
        'rmse_ensemble': rmse_ens,
        'best': min(['lstm', 'har', 'ensemble'], key=lambda x: {'lstm': rmse_lstm, 'har': rmse_har, 'ensemble': rmse_ens}[x])
    })
metrics_df = pd.DataFrame(metrics)
print(metrics_df.round(4).to_string())

print(f'\\n=== best 모델 분포 ===')
print(metrics_df['best'].value_counts())""")


# ============================================================================
# §6 가중치 진화
# ============================================================================
md("""## §6. Performance-Weighted 가중치 진화""")

code("""# 종목별 fold 별 w_v4 가중치 (시간 동적 적응 시각화)
weights_summary = ensemble_df.groupby(['ticker', 'fold']).agg(
    w_v4=('w_v4', 'first'),
    w_har=('w_har', 'first'),
    date=('date', 'first'),
).reset_index()

print(f'가중치 진화 (종목별 fold 별 w_v4):')
print(weights_summary.head(20).to_string())""")


# ============================================================================
# §7 시각화
# ============================================================================
md("""## §7. 시각화""")

code("""# 7-1. 종목별 RMSE 비교 막대그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE 비교
metrics_sorted = metrics_df.sort_values('rmse_ensemble')
x = np.arange(len(metrics_sorted))
w = 0.25
axes[0].bar(x - w, metrics_sorted['rmse_lstm'], w, label='LSTM v4', color='steelblue')
axes[0].bar(x, metrics_sorted['rmse_har'], w, label='HAR-RV', color='coral')
axes[0].bar(x + w, metrics_sorted['rmse_ensemble'], w, label='Ensemble', color='green')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_sorted['ticker'], rotation=45, ha='right')
axes[0].set_ylabel('RMSE')
axes[0].set_title('종목별 OOS RMSE 비교')
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')

# best 모델 분포
ax = axes[1]
best_counts = metrics_df['best'].value_counts()
colors = {'lstm': 'steelblue', 'har': 'coral', 'ensemble': 'green'}
ax.bar(best_counts.index, best_counts.values, color=[colors[b] for b in best_counts.index])
ax.set_ylabel('종목 수')
ax.set_title('best 모델 분포')
for i, v in enumerate(best_counts.values):
    ax.text(i, v + 0.1, f'{v}', ha='center', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT_DIR / 'rmse_comparison.png', dpi=120, bbox_inches='tight')
plt.show()""")

code("""# 7-2. 가중치 진화 (시간 축)
fig, ax = plt.subplots(figsize=(14, 5))
for ticker in weights_summary['ticker'].unique():
    df_t = weights_summary[weights_summary['ticker'] == ticker].sort_values('fold')
    ax.plot(df_t['fold'], df_t['w_v4'], marker='o', alpha=0.6, label=ticker)
ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='warmup (0.5)')
ax.set_xlabel('Fold')
ax.set_ylabel('w_v4 (LSTM 가중치)')
ax.set_title('Performance-Weighted 가중치 진화 (종목별)')
ax.legend(loc='upper right', ncol=2, fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUT_DIR / 'weights_evolution.png', dpi=120, bbox_inches='tight')
plt.show()""")


md("""## 다음 단계

→ **Step 4**: `04_BL_yearly_rebalance.ipynb` — `ensemble_predictions_top50.csv` 의 변동성 예측을 BL P 행렬에 투입.

⚠️ **본 PoC 결과 (5 종목) 가 합리적이면, MODE='full' 으로 본 학습 (74 종목 ~5-10시간 GPU) 권고**.
""")


# ============================================================================
# 노트북 저장
# ============================================================================
NB.cells = cells
NB_PATH = Path(__file__).parent / '03_phase15_ensemble_top50.ipynb'
nbf.write(NB, str(NB_PATH))

n_md = sum(1 for c in cells if c.cell_type == 'markdown')
n_code = sum(1 for c in cells if c.cell_type == 'code')
print(f'노트북 생성 완료: {NB_PATH}')
print(f'  셀 수: {len(cells)} (markdown {n_md} + code {n_code})')
