"""1회용 빌드 스크립트 — 02_v4_final_evaluation.ipynb 생성.

Phase 1.5 v4 best 조합의 90 fold 재학습 + 종합 분석.

목적:
  Optuna v4 가 발견한 best (3ch_vix/IS=1250/emb=63 / RMSE 0.3107) 의
  단일 조합을 재학습하여 다음을 산출:
    - fold_predictions (y_true, y_pred 시계열) — §04 같은 자체 진단 가능
    - 5종 메트릭 + MZ regression + 관문 2, 3 재판정
    - DM 검정 (vs HAR / EWMA / Naive / Train-Mean / LSTM v1)
    - 잔차 진단 (Jarque-Bera, Durbin-Watson, Breusch-Pagan)
    - 체제별 RMSE (안정/COVID/긴축/회복/AI붐)
    - Phase 1.5 PASS 조건 종합 재판정 (관문 1, 2, 3)

Best 조합 (Optuna v4):
  input_channels = '3ch_vix'   (HAR + VIX, input_size=4)
  is_len         = 1250         (5년)
  embargo        = 63
  hidden=32, dropout=0.3, lr=1e-3, weight_decay=1e-3, seed=42

사용법 (사용자 GPU 환경):
    python _build_02_v4_final_eval_nb.py
    Jupyter 에서 02_v4_final_evaluation.ipynb 열고 Run All

저장:
  results/lstm_v4_final/{SPY,QQQ}_metrics.json
  results/comparison_report_v4.md (자동 생성)
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
md("""# Phase 1.5 v4 Final — Best 조합 종합 분석 (`02_v4_final_evaluation.ipynb`)

> **목적**: Optuna v4 의 best 조합 (3ch_vix / IS=1250 / emb=63 / RMSE 0.3107) 단일 조합을
> 재학습하여 fold_predictions + 모든 메트릭 + 자체 진단 + Phase 1.5 PASS 조건 재판정.

## v4 best 조합

```python
input_channels = '3ch_vix'   # rv_d + rv_w + rv_m + vix_log (input_size=4)
is_len         = 1250         # 5년 IS
embargo        = 63
hidden=32, dropout=0.3, lr=1e-3, weight_decay=1e-3, seed=42, batch=32
```

## 본 노트북의 역할 (Optuna v4 와의 차이)

| Optuna v4 (`02_v4_lstm_optuna.ipynb`) | **이 노트북** |
|---|---|
| 12 trials × 평균 RMSE 만 산출 | **best 1 조합 × fold 별 모든 메트릭** |
| fold_predictions 미저장 | **y_true, y_pred 시계열 모두 저장** |
| 자체 진단 X | **§04 같은 6 진단 항목** |
| 관문 1 만 판정 | **관문 1, 2, 3 모두 판정** |

## 본 노트북 셀 구성

| § | 내용 |
|---|---|
| §1 | 환경 + GPU + 데이터 로드 (VIX 포함) |
| §2 | v4 best 조합 90 fold 재학습 (fold_predictions 저장) |
| §3 | 종합 메트릭 (5종 + MZ regression) |
| §4 | 관문 1, 2, 3 재판정 |
| §5 | Diebold-Mariano 검정 (vs HAR/EWMA/Naive/Train-Mean/LSTM v1) |
| §6 | 자체 진단 (잔차 + 체제별 RMSE) |
| §7 | metrics.json 저장 + comparison_report_v4.md 생성 |
| §8 | Phase 1.5 최종 결론 (PASS/FAIL 갱신) |

## 사용자 GPU 환경 가이드

### 예상 시간
- IS=1250, 55 fold, 2 ticker: **~2~3분**
- 추가 분석 + 시각화: ~30초

### 산출물
```
results/lstm_v4_final/
├── SPY_metrics.json
└── QQQ_metrics.json

results/comparison_report_v4.md       (자동 생성)
```
""")


# ============================================================================
# §1 환경
# ============================================================================
md("""## §1. 환경 + GPU + 데이터 로드 (VIX 포함)
""")

code("""import sys
import json
import time
from pathlib import Path

NB_DIR = Path.cwd()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

from scripts.setup import bootstrap, BASE_DIR, RAW_DATA_DIR, RESULTS_DIR
font_used = bootstrap()

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
""")

code("""# 데이터 로드 (VIX 포함, v4 와 동일)
import pandas as pd
import numpy as np
from scripts.targets_volatility import build_daily_target_logrv_21d

ANALYSIS_START = '2016-01-01'
ANALYSIS_END   = '2025-12-31'
WINDOW = 21
SEQ_LEN = 63
HAR_W_WINDOW = 5
HAR_M_WINDOW = 22


def load_ticker_full(ticker: str) -> pd.DataFrame:
    csv_path = RAW_DATA_DIR / f'{ticker}.csv'
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    df['log_ret'] = np.log(df['Adj Close']).diff()
    return df


# VIX 캐시 로드 (v4 가 이미 다운로드)
vix_path = RAW_DATA_DIR / 'VIX.csv'
df_vix_full = pd.read_csv(vix_path, index_col=0, parse_dates=True).sort_index()
vix_col = 'Close' if 'Close' in df_vix_full.columns else df_vix_full.columns[0]
df_vix_full = df_vix_full[[vix_col]].rename(columns={vix_col: 'VIX'})
print(f'VIX 캐시 로드: {df_vix_full.index[0].date()} ~ {df_vix_full.index[-1].date()}, n={len(df_vix_full)}')

# SPY, QQQ + VIX 정렬 + 입력 채널 사전 준비
analysis_dict = {}
for tk in ('SPY', 'QQQ'):
    raw = load_ticker_full(tk)
    df = raw.loc[ANALYSIS_START:ANALYSIS_END].copy()
    df = df.join(df_vix_full, how='left')
    df['VIX'] = df['VIX'].ffill()
    # 3ch_vix 입력 (v4 best)
    df['rv_d'] = df['log_ret'].abs()
    df['rv_w'] = (df['log_ret'] ** 2).rolling(HAR_W_WINDOW).mean().pow(0.5)
    df['rv_m'] = (df['log_ret'] ** 2).rolling(HAR_M_WINDOW).mean().pow(0.5)
    df['vix_log'] = np.log(df['VIX'])
    # rv_trailing (Naive baseline 용)
    df['rv_trailing'] = df['log_ret'].rolling(WINDOW).std(ddof=1)
    # 타깃
    df['target_logrv'] = build_daily_target_logrv_21d(df['Adj Close'], window=WINDOW)
    analysis_dict[tk] = df
    print(f'{tk}: n={len(df)}, 유효 target={int(df[\"target_logrv\"].notna().sum())}')

N_VALID = min(int(analysis_dict[tk]['target_logrv'].notna().sum()) for tk in ('SPY', 'QQQ'))
print(f'\\nN_VALID = {N_VALID}')
""")


# ============================================================================
# §2 v4 best 90 fold 재학습
# ============================================================================
md("""## §2. v4 best 조합 90 fold 재학습 (fold_predictions 저장)
""")

code("""# v4 best hyperparameter
INPUT_CHANNELS = '3ch_vix'
IS_LEN = 1250
EMBARGO = 63
HIDDEN = 32
NUM_LAYERS = 1
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5
LR_PATIENCE = 3
LOSS_TYPE = 'mse'
BATCH_SIZE = 32
VAL_RATIO = 0.2
SEED = 42

PURGE = 21
OOS_LEN = 21
STEP = 21
INPUT_SIZE = 4   # 3ch_vix


from torch.utils.data import TensorDataset, DataLoader
from scripts.dataset import build_fold_datasets, walk_forward_folds
from scripts.models import LSTMRegressor
from scripts.train import train_one_fold


def build_input_3ch_vix(df: pd.DataFrame):
    series = df['rv_d'].fillna(0.0).values
    extra = np.column_stack([
        df['rv_w'].fillna(0.0).values,
        df['rv_m'].fillna(0.0).values,
        df['vix_log'].fillna(method='ffill').fillna(0.0).values,
    ])
    return series, extra


def build_train_val_loaders(train_ds, val_ratio=VAL_RATIO, batch_size=BATCH_SIZE):
    n = len(train_ds)
    n_val = max(1, int(n * val_ratio))
    n_tr = n - n_val
    train_sub = TensorDataset(train_ds.X[:n_tr], train_ds.y[:n_tr])
    val_sub   = TensorDataset(train_ds.X[n_tr:], train_ds.y[n_tr:])
    tr_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(val_sub, batch_size=max(1, n_val), shuffle=False)
    return tr_loader, va_loader


def run_v4_best_for_ticker(ticker: str, df: pd.DataFrame, log_every: int = 10):
    \"\"\"v4 best 조합으로 단일 ticker 의 모든 fold 학습 + fold_predictions 저장.\"\"\"
    series, extra = build_input_3ch_vix(df)
    target = df['target_logrv'].values

    folds = walk_forward_folds(
        n=N_VALID, is_len=IS_LEN, purge=PURGE, emb=EMBARGO,
        oos_len=OOS_LEN, step=STEP,
    )
    n_folds = len(folds)
    print(f'  [{ticker}] device={device}, n_folds={n_folds}')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)

    fold_out = []
    t_start = time.time()
    for k, (tr_idx, te_idx) in enumerate(folds):
        tr_ds_k, te_ds_k, _ = build_fold_datasets(
            series=series, train_idx=tr_idx, test_idx=te_idx,
            seq_len=SEQ_LEN, extra_features=extra, target_series=target,
        )
        tr_loader, va_loader = build_train_val_loaders(tr_ds_k)
        model_k = LSTMRegressor(
            input_size=INPUT_SIZE, hidden_size=HIDDEN,
            num_layers=NUM_LAYERS, dropout=DROPOUT, batch_first=True,
        )
        result = train_one_fold(
            model_k, tr_loader, va_loader,
            max_epochs=MAX_EPOCHS, early_stop_patience=PATIENCE,
            lr=LR, weight_decay=WEIGHT_DECAY, lr_patience=LR_PATIENCE,
            loss_type=LOSS_TYPE, device=device, verbose=False,
        )
        # best 로 train/val/test 모두 예측
        model_k.load_state_dict(result['best_state_dict'])
        model_k.to(device).eval()
        with torch.no_grad():
            y_pred_oos = model_k(te_ds_k.X.to(device)).cpu().numpy().flatten()
            n_tr_used = len(tr_loader.dataset)
            y_pred_train = model_k(tr_ds_k.X[:n_tr_used].to(device)).cpu().numpy().flatten()
            y_pred_val   = model_k(tr_ds_k.X[n_tr_used:].to(device)).cpu().numpy().flatten()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        fold_out.append({
            'fold': k,
            'train_idx_first': int(tr_idx[0]), 'train_idx_last': int(tr_idx[-1]),
            'test_idx_first':  int(te_idx[0]), 'test_idx_last':  int(te_idx[-1]),
            'y_true_test':  te_ds_k.y.numpy().astype(float).tolist(),
            'y_pred_test':  y_pred_oos.astype(float).tolist(),
            'y_true_train': tr_ds_k.y[:n_tr_used].numpy().astype(float).tolist(),
            'y_pred_train': y_pred_train.astype(float).tolist(),
            'y_true_val':   tr_ds_k.y[n_tr_used:].numpy().astype(float).tolist(),
            'y_pred_val':   y_pred_val.astype(float).tolist(),
            'best_epoch': int(result['best_epoch']),
            'best_val_loss': float(result['best_val_loss']),
            'stopped_early': bool(result['stopped_early']),
            'history': {kk: [float(v) for v in vs] for kk, vs in result['history'].items()},
        })
        if (k + 1) % log_every == 0 or k == n_folds - 1:
            elapsed = time.time() - t_start
            avg = elapsed / (k + 1)
            eta = avg * (n_folds - k - 1)
            print(f'    fold {k+1}/{n_folds} best_ep={result[\"best_epoch\"]:>2} '
                  f'val_loss={result[\"best_val_loss\"]:.4f} | '
                  f'elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m')
    return fold_out


print('=' * 70)
print('v4 best 90 fold 재학습 시작')
print('=' * 70)
spy_folds = run_v4_best_for_ticker('SPY', analysis_dict['SPY'])
print()
qqq_folds = run_v4_best_for_ticker('QQQ', analysis_dict['QQQ'])
print()
print(f'학습 완료: SPY {len(spy_folds)} fold, QQQ {len(qqq_folds)} fold')
""")


# ============================================================================
# §3 종합 메트릭
# ============================================================================
md("""## §3. 종합 메트릭 (5종 + MZ regression)
""")

code("""from scripts.metrics_volatility import (
    rmse, mae, qlike, r2_train_mean, mz_regression, pred_std_ratio,
    summarize_folds_volatility,
)


def compute_per_fold_metrics(folds_out):
    per_fold = []
    for fo in folds_out:
        y_true = np.array(fo['y_true_test'])
        y_pred = np.array(fo['y_pred_test'])
        y_train_full = np.array(fo['y_true_train'] + fo['y_true_val'])
        mz = mz_regression(y_true, y_pred)
        per_fold.append({
            'rmse': rmse(y_true, y_pred),
            'mae': mae(y_true, y_pred),
            'qlike': qlike(y_true, y_pred),
            'r2_train_mean': r2_train_mean(y_true, y_pred, y_train_full),
            'pred_std_ratio': pred_std_ratio(y_true, y_pred),
            'mz_alpha': mz['alpha'], 'mz_beta': mz['beta'], 'mz_r2': mz['r2'],
            'best_epoch': fo['best_epoch'],
            'best_val_loss': fo['best_val_loss'],
        })
    return per_fold


per_fold_spy = compute_per_fold_metrics(spy_folds)
per_fold_qqq = compute_per_fold_metrics(qqq_folds)
summary_spy  = summarize_folds_volatility(per_fold_spy)
summary_qqq  = summarize_folds_volatility(per_fold_qqq)

# 종합 메트릭 표
print('=' * 80)
print('Phase 1.5 v4 best 종합 메트릭 (90 fold mean ± std)')
print('=' * 80)
print(f'  {\"metric\":<20} {\"SPY mean\":>12} {\"SPY std\":>10} {\"QQQ mean\":>12} {\"QQQ std\":>10}')
print('  ' + '-' * 76)
for m in ['rmse', 'mae', 'qlike', 'r2_train_mean', 'pred_std_ratio',
          'mz_alpha', 'mz_beta', 'mz_r2', 'best_epoch']:
    s = summary_spy.get(m, {})
    q = summary_qqq.get(m, {})
    print(f'  {m:<20} {s.get(\"mean\", float(\"nan\")):>+12.4f} {s.get(\"std\", float(\"nan\")):>10.4f} '
          f'{q.get(\"mean\", float(\"nan\")):>+12.4f} {q.get(\"std\", float(\"nan\")):>10.4f}')
""")


# ============================================================================
# §4 관문 1, 2, 3 재판정
# ============================================================================
md("""## §4. Phase 1.5 PASS 조건 — 관문 1, 2, 3 재판정
""")

code("""# §03 결과의 HAR baseline (사전 계산)
HAR_BASELINE = {
    'spy': {'rmse': 0.3646, 'qlike': 0.7796, 'r2_train_mean': -0.5280, 'pred_std_ratio': 0.8968},
    'qqq': {'rmse': 0.3308, 'qlike': 0.5083, 'r2_train_mean': -0.2618, 'pred_std_ratio': 0.9191},
}
HAR_AVG_RMSE = (HAR_BASELINE['spy']['rmse'] + HAR_BASELINE['qqq']['rmse']) / 2

print('=' * 80)
print('Phase 1.5 PASS 조건 재판정 (v4 best)')
print('=' * 80)

results_gates = {}
for tk in ('SPY', 'QQQ'):
    s = summary_spy if tk == 'SPY' else summary_qqq
    h = HAR_BASELINE[tk.lower()]
    rmse_mean = s['rmse']['mean']
    r2_mean   = s['r2_train_mean']['mean']
    psr_mean  = s['pred_std_ratio']['mean']
    gate1 = rmse_mean < h['rmse']
    gate2 = r2_mean > 0
    gate3 = psr_mean > 0.5
    n_pass = int(gate1) + int(gate2) + int(gate3)
    results_gates[tk] = {'gate1': gate1, 'gate2': gate2, 'gate3': gate3, 'n_pass': n_pass,
                         'rmse': rmse_mean, 'r2_train_mean': r2_mean, 'pred_std_ratio': psr_mean}
    print(f'\\n  [{tk}]')
    print(f'    관문 1 (LSTM RMSE {rmse_mean:.4f} < HAR {h[\"rmse\"]:.4f})  : '
          f'{\"PASS\" if gate1 else \"FAIL\"}')
    print(f'    관문 2 (r2_train_mean {r2_mean:>+.4f} > 0)                : '
          f'{\"PASS\" if gate2 else \"FAIL\"}')
    print(f'    관문 3 (pred_std_ratio {psr_mean:.4f} > 0.5)              : '
          f'{\"PASS\" if gate3 else \"FAIL\"}')
    print(f'    종합: {n_pass}/3 → {\"PASS\" if n_pass == 3 else f\"FAIL ({n_pass}/3)\"}')
""")


# ============================================================================
# §5 Diebold-Mariano 검정
# ============================================================================
md("""## §5. Diebold-Mariano 검정 — v4 best vs HAR/EWMA/Naive/Train-Mean/LSTM v1
""")

code("""# 베이스라인 + LSTM v1 fold predictions 재계산
from scripts.baselines_volatility import (
    fit_har_rv, predict_ewma, predict_naive, predict_train_mean,
)
from scipy import stats


# v4 best 와 동일 fold 구조
folds_v4 = walk_forward_folds(
    n=N_VALID, is_len=IS_LEN, purge=PURGE, emb=EMBARGO,
    oos_len=OOS_LEN, step=STEP,
)
print(f'fold 수 (v4 best 기준): {len(folds_v4)}')


# 각 fold 의 베이스라인 예측 생성 (DM 검정용)
def build_baseline_residuals(ticker: str, df: pd.DataFrame, folds, lstm_folds):
    \"\"\"각 fold 의 5종 모델 잔차 (y_true - y_pred) 시계열 모음.\"\"\"
    log_ret = df['log_ret']
    target = df['target_logrv']
    rv_trailing = df['rv_trailing']

    res_lstm_v4, res_har, res_ewma, res_naive, res_tm = [], [], [], [], []
    for k, (tr_idx, te_idx) in enumerate(folds):
        y_true = target.values[te_idx]
        # LSTM v4 best (이미 학습한 결과)
        y_pred_v4 = np.array(lstm_folds[k]['y_pred_test'])
        # HAR
        y_pred_har, _ = fit_har_rv(log_ret, tr_idx, te_idx, horizon=WINDOW)
        # EWMA
        y_pred_ewma = predict_ewma(log_ret, tr_idx, te_idx, horizon=WINDOW, lam=0.94)
        # Naive
        y_pred_naive = predict_naive(rv_trailing, tr_idx, te_idx)
        # Train-Mean
        y_pred_tm = predict_train_mean(target, tr_idx, te_idx)

        res_lstm_v4.append(y_true - y_pred_v4)
        res_har.append(y_true - y_pred_har)
        res_ewma.append(y_true - y_pred_ewma)
        res_naive.append(y_true - y_pred_naive)
        res_tm.append(y_true - y_pred_tm)

    return {
        'lstm_v4':    np.concatenate(res_lstm_v4),
        'har':        np.concatenate(res_har),
        'ewma':       np.concatenate(res_ewma),
        'naive':      np.concatenate(res_naive),
        'train_mean': np.concatenate(res_tm),
    }


# DM 검정 함수 (§04 와 동일)
def diebold_mariano(e1, e2, h=1):
    L1 = e1 ** 2
    L2 = e2 ** 2
    d = L1 - L2
    n = len(d)
    d_bar = d.mean()
    var_d = d.var(ddof=1)
    DM = d_bar / np.sqrt(var_d / n) if var_d > 0 else float('nan')
    p = 2 * (1 - stats.norm.cdf(abs(DM))) if not np.isnan(DM) else float('nan')
    return {'DM': float(DM), 'p_value': float(p), 'n': int(n)}


# LSTM v1 (1ch/IS=504/emb=63) fold predictions 로드
v1_metrics = {}
V1_DIR = RESULTS_DIR / 'volatility_lstm'
for tk in ('SPY', 'QQQ'):
    with open(V1_DIR / f'{tk}_metrics.json', 'r', encoding='utf-8') as f:
        v1_metrics[tk] = json.load(f)


# 주의: LSTM v1 은 IS=504 fold 구조 → v4 (IS=1250) 와 fold 다름
# DM 검정은 동일 시간대만 가능 → v4 fold 의 test_idx 와 v1 의 test_idx 매칭하여 추출
def extract_v1_residuals_aligned_to_v4(ticker, v1_data, v4_folds):
    \"\"\"v4 fold 의 test_idx 에 해당하는 v1 fold 예측을 추출.\"\"\"
    # v1 의 모든 fold_predictions 를 (test_idx → y_pred) dict 로
    v1_predictions = {}
    for fp in v1_data['fold_predictions']:
        for i, pred in enumerate(fp['y_pred_test']):
            te_idx = fp['test_idx_first'] + i
            v1_predictions[te_idx] = (fp['y_true_test'][i], pred)
    # v4 fold 와 매칭
    res_v1_aligned = []
    for tr_idx, te_idx in v4_folds:
        for t in te_idx:
            if t in v1_predictions:
                yt, yp = v1_predictions[t]
                res_v1_aligned.append(yt - yp)
            else:
                res_v1_aligned.append(np.nan)
    return np.array(res_v1_aligned)


# DM 검정 실행
dm_results = {}
for tk in ('SPY', 'QQQ'):
    df = analysis_dict[tk]
    lstm_folds = spy_folds if tk == 'SPY' else qqq_folds
    res = build_baseline_residuals(tk, df, folds_v4, lstm_folds)
    res_v1_aligned = extract_v1_residuals_aligned_to_v4(tk, v1_metrics[tk], folds_v4)
    res['lstm_v1'] = res_v1_aligned

    dm_for_tk = {}
    base = res['lstm_v4']
    for model_name in ['har', 'ewma', 'naive', 'train_mean', 'lstm_v1']:
        e2 = res[model_name]
        # NaN 처리 (lstm_v1 매칭 누락 시)
        valid = np.isfinite(base) & np.isfinite(e2)
        if valid.sum() < 30:
            dm_for_tk[model_name] = {'DM': float('nan'), 'p_value': float('nan'), 'n': int(valid.sum())}
        else:
            dm_for_tk[model_name] = diebold_mariano(base[valid], e2[valid])
    dm_results[tk] = dm_for_tk


# 출력
print('=' * 90)
print('Diebold-Mariano 검정 — v4 best (모델 1) vs 비교 모델 (모델 2)')
print('  DM < 0 면 v4 best 우위, DM > 0 면 비교 모델 우위, |DM| > 1.96 이면 5% 유의')
print('=' * 90)
for tk in ('SPY', 'QQQ'):
    print(f'\\n[{tk}]')
    print(f'  {\"비교 모델\":<14} {\"DM\":>10} {\"p-value\":>14} {\"5% 유의?\":>10} {\"우위\":>14}')
    print('  ' + '-' * 70)
    for m, r in dm_results[tk].items():
        sig = '✓' if (not np.isnan(r['p_value'])) and r['p_value'] < 0.05 else ' '
        if np.isnan(r['DM']):
            winner = 'N/A'
        elif r['DM'] < 0:
            winner = 'LSTM v4'
        else:
            winner = m
        dm_str = f'{r[\"DM\"]:>+10.3f}' if not np.isnan(r['DM']) else '       N/A'
        p_str = f'{r[\"p_value\"]:>14.4e}' if not np.isnan(r['p_value']) else '          N/A'
        print(f'  {m:<14} {dm_str} {p_str} {sig:>10} {winner:>14}')
""")


# ============================================================================
# §6 자체 진단
# ============================================================================
md("""## §6. 자체 진단 — 잔차 진단 + 체제별 RMSE
""")

code("""from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm


# 전체 OOS 잔차 시계열 (90 fold 시간순 연결)
def build_full_residual(folds_out):
    res = []
    for fo in folds_out:
        y_true = np.array(fo['y_true_test'])
        y_pred = np.array(fo['y_pred_test'])
        res.append(y_true - y_pred)
    return np.concatenate(res)


residual_dict = {
    'SPY': build_full_residual(spy_folds),
    'QQQ': build_full_residual(qqq_folds),
}

print('=' * 80)
print('잔차 진단 — Jarque-Bera (정규성), Durbin-Watson (자기상관), Breusch-Pagan (이분산)')
print('=' * 80)
diagnostic_results = {}
for tk in ('SPY', 'QQQ'):
    res = residual_dict[tk]
    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(res)
    # Durbin-Watson
    dw = durbin_watson(res)
    # Breusch-Pagan
    exog = sm.add_constant(np.arange(len(res)).astype(float))
    bp_stat, bp_p, _, _ = het_breuschpagan(res, exog)
    diagnostic_results[tk] = {
        'jb_stat': float(jb_stat), 'jb_p': float(jb_p),
        'dw': float(dw), 'bp_p': float(bp_p),
        'res_mean': float(res.mean()), 'res_std': float(res.std(ddof=1)),
        'skew': float(stats.skew(res)), 'kurt': float(stats.kurtosis(res)),
    }
    print(f'\\n[{tk}] n={len(res)}, mean={res.mean():+.4f}, std={res.std(ddof=1):.4f}')
    print(f'  Jarque-Bera : stat={jb_stat:.2f}, p={jb_p:.4e}  '
          f'({\"비정규\" if jb_p < 0.05 else \"정규 가정 가능\"}, skew={stats.skew(res):+.3f}, kurt={stats.kurtosis(res):+.3f})')
    print(f'  Durbin-Watson: {dw:.4f}  '
          f'({\"양의 자기상관 (잔차 정보 잔존)\" if dw < 1.5 else \"음의 자기상관\" if dw > 2.5 else \"자기상관 없음 (정상)\"})')
    print(f'  Breusch-Pagan: stat={bp_stat:.2f}, p={bp_p:.4e}  '
          f'({\"이분산\" if bp_p < 0.05 else \"등분산\"})')


# 체제별 RMSE
REGIMES = {
    'stable_1':   ('2016-01-01', '2019-12-31'),
    'covid':      ('2020-01-01', '2020-06-30'),
    'recovery':   ('2020-07-01', '2021-12-31'),
    'tightening': ('2022-01-01', '2022-12-31'),
    'ai_boom':    ('2023-01-01', '2025-12-31'),
}


def assign_regime(dates):
    out = pd.Series(index=dates, dtype=object)
    for r, (s, e) in REGIMES.items():
        out.loc[(dates >= s) & (dates <= e)] = r
    return out


def build_full_oos_with_regime(ticker, folds_out, df):
    y_true_all, y_pred_all, regimes_all = [], [], []
    for fo in folds_out:
        y_true_all.append(np.array(fo['y_true_test']))
        y_pred_all.append(np.array(fo['y_pred_test']))
        te_first = fo['test_idx_first']
        te_last  = fo['test_idx_last'] + 1
        dates = df.index[te_first:te_last]
        regimes_all.append(assign_regime(dates).values)
    return (np.concatenate(y_true_all), np.concatenate(y_pred_all),
            np.concatenate(regimes_all).astype(object))


print()
print('=' * 90)
print('체제별 v4 best LSTM 성능 (RMSE / MAE / n)')
print('=' * 90)
regime_metrics = {}
for tk in ('SPY', 'QQQ'):
    folds_out = spy_folds if tk == 'SPY' else qqq_folds
    y_true, y_pred, regs = build_full_oos_with_regime(tk, folds_out, analysis_dict[tk])
    print(f'\\n[{tk}]')
    print(f'  {\"체제\":<14} {\"기간\":<25} {\"n\":>6} {\"RMSE\":>10} {\"MAE\":>10}')
    print('  ' + '-' * 70)
    rec = {}
    for regime, (start, end) in REGIMES.items():
        mask = regs == regime
        if mask.sum() == 0:
            continue
        rmse_r = float(np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean()))
        mae_r  = float(np.abs(y_true[mask] - y_pred[mask]).mean())
        rec[regime] = {'n': int(mask.sum()), 'rmse': rmse_r, 'mae': mae_r}
        print(f'  {regime:<14} {start} ~ {end} {mask.sum():>6} {rmse_r:>10.4f} {mae_r:>10.4f}')
    regime_metrics[tk] = rec
""")


# ============================================================================
# §7 metrics.json + comparison_report_v4.md
# ============================================================================
md("""## §7. metrics.json 저장 + comparison_report_v4.md 자동 생성
""")

code("""# metrics.json (§02 형식과 동일)
OUT_DIR = RESULTS_DIR / 'lstm_v4_final'
OUT_DIR.mkdir(parents=True, exist_ok=True)

hyperparams = {
    'seed': SEED,
    'analysis_period': [ANALYSIS_START, ANALYSIS_END],
    'walk_forward': {
        'IS': IS_LEN, 'purge': PURGE, 'embargo': EMBARGO,
        'OOS': OOS_LEN, 'step': STEP, 'n_folds': len(folds_v4),
    },
    'model': {
        'class': 'LSTMRegressor',
        'input_size': INPUT_SIZE, 'hidden_size': HIDDEN,
        'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
        'batch_first': True,
    },
    'training': {
        'loss_type': LOSS_TYPE, 'max_epochs': MAX_EPOCHS,
        'early_stop_patience': PATIENCE, 'lr': LR,
        'weight_decay': WEIGHT_DECAY, 'lr_patience': LR_PATIENCE,
        'val_ratio': VAL_RATIO, 'batch_size': BATCH_SIZE,
    },
    'seq_len': SEQ_LEN,
    'input_feature': '3ch_vix_har',
    'input_channels': ['rv_d', 'rv_w', 'rv_m', 'vix_log'],
    'target': 'log_realized_volatility_21d',
    'version': 'v4_best',
}

for tk, fo_list, pf_list, summary in [
    ('SPY', spy_folds, per_fold_spy, summary_spy),
    ('QQQ', qqq_folds, per_fold_qqq, summary_qqq),
]:
    out = {
        'ticker': tk,
        'hyperparams': hyperparams,
        'summary': summary,
        'per_fold': pf_list,
        'fold_predictions': fo_list,
        'gate_results': results_gates[tk],
        'dm_results': dm_results[tk],
        'diagnostic_results': diagnostic_results[tk],
        'regime_metrics': regime_metrics[tk],
    }
    out_path = OUT_DIR / f'{tk}_metrics.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'저장: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)')


# comparison_report_v4.md 자동 생성
report_path = RESULTS_DIR / 'comparison_report_v4.md'
lines = []
lines.append('# Phase 1.5 v4 Final — 종합 비교 보고서')
lines.append('')
lines.append('> **v4 best 조합 (3ch_vix / IS=1250 / emb=63) 90 fold 재학습 결과**')
lines.append(f'> 분석 기간: {ANALYSIS_START} ~ {ANALYSIS_END}')
lines.append(f'> Walk-Forward: IS={IS_LEN} / Purge={PURGE} / Embargo={EMBARGO} / OOS={OOS_LEN} / Step={STEP}')
lines.append(f'> fold 수: {len(folds_v4)}')
lines.append('')

lines.append('## 1. v4 best 종합 메트릭')
lines.append('')
lines.append('| metric | SPY | QQQ |')
lines.append('|---|---|---|')
for m in ['rmse', 'mae', 'qlike', 'r2_train_mean', 'pred_std_ratio',
          'mz_alpha', 'mz_beta', 'mz_r2', 'best_epoch']:
    s = summary_spy[m]
    q = summary_qqq[m]
    lines.append(f'| {m} | {s[\"mean\"]:+.4f} ± {s[\"std\"]:.3f} | '
                 f'{q[\"mean\"]:+.4f} ± {q[\"std\"]:.3f} |')
lines.append('')

lines.append('## 2. Phase 1.5 PASS 조건 종합 재판정')
lines.append('')
lines.append('| 관문 | SPY | QQQ |')
lines.append('|---|---|---|')
for tk in ('SPY', 'QQQ'):
    pass
g_spy = results_gates['SPY']
g_qqq = results_gates['QQQ']
lines.append(f'| 1 (RMSE < HAR) | {g_spy[\"rmse\"]:.4f} ({\"PASS\" if g_spy[\"gate1\"] else \"FAIL\"}) '
             f'| {g_qqq[\"rmse\"]:.4f} ({\"PASS\" if g_qqq[\"gate1\"] else \"FAIL\"}) |')
lines.append(f'| 2 (r2_train_mean > 0) | {g_spy[\"r2_train_mean\"]:+.4f} '
             f'({\"PASS\" if g_spy[\"gate2\"] else \"FAIL\"}) '
             f'| {g_qqq[\"r2_train_mean\"]:+.4f} ({\"PASS\" if g_qqq[\"gate2\"] else \"FAIL\"}) |')
lines.append(f'| 3 (pred_std_ratio > 0.5) | {g_spy[\"pred_std_ratio\"]:.4f} '
             f'({\"PASS\" if g_spy[\"gate3\"] else \"FAIL\"}) '
             f'| {g_qqq[\"pred_std_ratio\"]:.4f} ({\"PASS\" if g_qqq[\"gate3\"] else \"FAIL\"}) |')
lines.append(f'| **종합** | **{g_spy[\"n_pass\"]}/3** | **{g_qqq[\"n_pass\"]}/3** |')
lines.append('')

lines.append('## 3. Diebold-Mariano 검정 (v4 best vs 비교 모델)')
lines.append('')
lines.append('| 비교 모델 | SPY DM | SPY p | QQQ DM | QQQ p | 우위 |')
lines.append('|---|---|---|---|---|---|')
for m in ['har', 'ewma', 'naive', 'train_mean', 'lstm_v1']:
    s_dm = dm_results['SPY'][m]
    q_dm = dm_results['QQQ'][m]
    s_str = f'{s_dm[\"DM\"]:+.2f}' if not np.isnan(s_dm['DM']) else 'N/A'
    q_str = f'{q_dm[\"DM\"]:+.2f}' if not np.isnan(q_dm['DM']) else 'N/A'
    s_p = f'{s_dm[\"p_value\"]:.3e}' if not np.isnan(s_dm['p_value']) else 'N/A'
    q_p = f'{q_dm[\"p_value\"]:.3e}' if not np.isnan(q_dm['p_value']) else 'N/A'
    win = 'LSTM v4' if (not np.isnan(s_dm['DM']) and s_dm['DM'] < 0) else m
    lines.append(f'| {m} | {s_str} | {s_p} | {q_str} | {q_p} | {win} |')
lines.append('')

lines.append('## 4. 자체 진단')
lines.append('')
for tk in ('SPY', 'QQQ'):
    d = diagnostic_results[tk]
    lines.append(f'### {tk}')
    lines.append('')
    lines.append('| 검정 | 값 | 판정 |')
    lines.append('|---|---|---|')
    lines.append(f'| 잔차 mean | {d[\"res_mean\"]:+.4f} | 0 근방이면 무편향 |')
    lines.append(f'| 잔차 std | {d[\"res_std\"]:.4f} | RMSE 와 비슷 |')
    lines.append(f'| 왜도 | {d[\"skew\"]:+.3f} | 0 근방이면 정규 |')
    lines.append(f'| 첨도 | {d[\"kurt\"]:+.3f} | 0 근방이면 정규 |')
    lines.append(f'| Jarque-Bera p | {d[\"jb_p\"]:.4e} | < 0.05 면 비정규 |')
    lines.append(f'| Durbin-Watson | {d[\"dw\"]:.4f} | 2 근방 정상 |')
    lines.append(f'| Breusch-Pagan p | {d[\"bp_p\"]:.4e} | < 0.05 면 이분산 |')
    lines.append('')

lines.append('## 5. 체제별 RMSE')
lines.append('')
for tk in ('SPY', 'QQQ'):
    rec = regime_metrics[tk]
    lines.append(f'### {tk}')
    lines.append('')
    lines.append('| 체제 | 기간 | n | RMSE | MAE |')
    lines.append('|---|---|---|---|---|')
    for regime, (s, e) in REGIMES.items():
        if regime in rec:
            r = rec[regime]
            lines.append(f'| {regime} | {s} ~ {e} | {r[\"n\"]} | {r[\"rmse\"]:.4f} | {r[\"mae\"]:.4f} |')
    lines.append('')

lines.append('## 6. 최종 결론')
lines.append('')
total_pass = g_spy['n_pass'] + g_qqq['n_pass']
if total_pass == 6:
    lines.append('**Phase 1.5 PASS — v4 best (3ch_vix / IS=1250 / emb=63) 가 6개 관문 모두 충족**')
    lines.append('')
    lines.append('**"변동성 예측이 가능한가?" → YES, LSTM 으로도 가능 (HAR 능가).**')
elif total_pass >= 4:
    lines.append(f'**Phase 1.5 부분 PASS — {total_pass}/6 관문 PASS**')
    lines.append('')
    lines.append('관문 1 (HAR 능가) 은 입증, 일부 관문 (2 또는 3) 은 추가 검증 필요.')
else:
    lines.append(f'**Phase 1.5 결론 갱신 필요 — {total_pass}/6 관문 PASS**')
lines.append('')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\\n'.join(lines))
print(f'저장: {report_path} ({report_path.stat().st_size / 1024:.1f} KB)')
""")


# ============================================================================
# §8 결론
# ============================================================================
md("""## §8. Phase 1.5 최종 결론 — PASS/FAIL 요약
""")

code("""print('=' * 90)
print('Phase 1.5 v4 Final — 최종 결론')
print('=' * 90)

print()
print('단일 질문: \"변동성 예측이 가능한가?\"')
print()
print('=' * 90)
print('PASS 조건 종합 (90 fold 평균)')
print('=' * 90)

total_pass = 0
for tk in ('SPY', 'QQQ'):
    g = results_gates[tk]
    print(f'\\n  [{tk}]')
    print(f'    관문 1 (RMSE < HAR)         : {\"PASS\" if g[\"gate1\"] else \"FAIL\"}  '
          f'(LSTM {g[\"rmse\"]:.4f} vs HAR {HAR_BASELINE[tk.lower()][\"rmse\"]:.4f})')
    print(f'    관문 2 (r2_train_mean > 0)  : {\"PASS\" if g[\"gate2\"] else \"FAIL\"}  '
          f'({g[\"r2_train_mean\"]:>+.4f})')
    print(f'    관문 3 (pred_std_ratio>0.5) : {\"PASS\" if g[\"gate3\"] else \"FAIL\"}  '
          f'({g[\"pred_std_ratio\"]:.4f})')
    print(f'    종합: {g[\"n_pass\"]}/3')
    total_pass += g['n_pass']

print()
print('=' * 90)
print('Phase 1.5 최종 답변')
print('=' * 90)
print(f'\\n  PASS 종합: {total_pass}/6')
if total_pass == 6:
    print('  → 변동성 예측 가능, LSTM v4 best 가 HAR-RV 능가 (3 관문 모두 PASS)')
elif total_pass >= 4:
    print('  → 변동성 예측 가능, LSTM v4 best 가 HAR 능가 (관문 1 PASS), 일부 약점 존재')
elif total_pass >= 2:
    print('  → 부분 가능 — LSTM 의 일부 측면 우위, 일부는 미달')
else:
    print('  → LSTM 부적합 (이전 결론 유지)')

# DM 검정 종합
print()
print('  DM 검정 (v4 best 의 통계적 우위)')
for tk in ('SPY', 'QQQ'):
    print(f'    [{tk}] vs HAR: DM={dm_results[tk][\"har\"][\"DM\"]:+.2f}, '
          f'p={dm_results[tk][\"har\"][\"p_value\"]:.3e}')

print()
print('산출물:')
print(f'  results/lstm_v4_final/SPY_metrics.json')
print(f'  results/lstm_v4_final/QQQ_metrics.json')
print(f'  results/comparison_report_v4.md')
print()
print('=' * 90)
""")


# ============================================================================
# 노트북 저장
# ============================================================================
NB.cells = cells
OUT_PATH = Path(__file__).resolve().parent / '02_v4_final_evaluation.ipynb'
nbf.write(NB, str(OUT_PATH))
print(f'노트북 빌드 완료: {OUT_PATH}')
print(f'총 셀 수: {len(cells)}')
