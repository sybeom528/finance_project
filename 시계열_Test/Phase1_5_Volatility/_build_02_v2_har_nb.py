"""1회용 빌드 스크립트 — 02_v2_volatility_lstm_har3ch.ipynb 생성.

Phase 1.5 v2 분기 — HAR 3 채널 입력 검증 (변수 부족 가설 a 검증).

v1 (`02_volatility_lstm.ipynb`) 와의 차이 — 단 1가지:
- 입력: log_ret² (1ch) → HAR 3채널 [|log_ret|, RV_w, RV_m] (Corsi 2009 표준)
- input_size: 1 → 3
- 결과 폴더: volatility_lstm/ → volatility_lstm_har3ch/

불변 (통제 실험 원칙):
- Walk-Forward (IS=504, embargo=63, OOS=21, step=21, fold=90)
- 모델 (LSTM hidden=32, layers=1, dropout=0.3)
- 손실 (MSE), 옵티마 (AdamW lr=1e-3 wd=1e-3)
- seed=42, 분석기간 동일

사용법:
    python _build_02_v2_har_nb.py

저장: 같은 폴더의 ``02_v2_volatility_lstm_har3ch.ipynb`` (덮어쓰기).
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
md("""# Phase 1.5 v2 — 변동성 LSTM HAR 3채널 입력 (`02_v2_volatility_lstm_har3ch.ipynb`)

> **목적**: v1 의 mean-collapse FAIL 원인이 "**변수 부족**" 가설을 검증하기 위해
> HAR 3채널 입력으로 동일한 학습 환경 재실험 (통제 실험).

## v1 → v2 단 1가지 변경

| 항목 | v1 | **v2** |
|---|---|---|
| 입력 채널 | `log_ret²` (1ch) | **HAR 3채널 `[|log_ret|, RV_w, RV_m]`** |
| `input_size` | 1 | **3** |
| 결과 폴더 | `volatility_lstm/` | **`volatility_lstm_har3ch/`** |
| Walk-Forward / 모델 / 손실 / 옵티마 / seed | — | **v1 동일** |

## v2 입력 정의 (Corsi 2009 학술 표준 일간 적응)

```python
log_ret = log(adj_close).diff()
RV_d[t] = |log_ret[t]|                           # 채널 1: 1일 변동성
RV_w[t] = sqrt(mean(log_ret²[t-4 : t+1]))         # 채널 2: 5일 평균 변동성
RV_m[t] = sqrt(mean(log_ret²[t-21 : t+1]))        # 채널 3: 22일 평균 변동성
```

## 본 노트북 셀 구성

| § | 내용 |
|---|---|
| §1 | 환경 부트스트랩 |
| §2 | 데이터 로드 + log_ret · HAR 3채널 · target 생성 |
| §3 | 타깃 누수 검증 |
| §4 | SequenceDataset 점검 (3채널 확인) |
| §5 | Walk-Forward fold 생성 (90 fold, v1 동일) |
| §6 | LSTMRegressor smoke test (input_size=3) |
| §7 | run_all_folds helper (extra_features 지원) |
| §8 | **90 fold × 2 ticker 학습** (실행 시간 v1 과 유사) |
| §9 | 평가 + 진단 6종 (§9.A~F) |
| §10 | metrics.json 저장 + v1 vs v2 비교 안내 |

## v2 결과 → 분기 결정

```
v2 PASS    | 변수 부족 가설 a 확정 → §03 베이스라인 비교 진행
v2 FAIL    | LSTM 자체 한계 또는 CV 구조 영향 ↑ → 추가 실험 검토
v2 borderline | §03 비교로 HAR-RV 와 비교 후 종합 판정
```
""")

# ============================================================================
# §1 환경
# ============================================================================
md("""## §1. 환경 부트스트랩

`scripts/setup.py` 의 `bootstrap()` 한 줄로 환경 일괄 설정 (v1 동일).
""")
code("""import sys
from pathlib import Path
NB_DIR = Path.cwd()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

from scripts.setup import bootstrap, BASE_DIR, RAW_DATA_DIR, RESULTS_DIR
font_used = bootstrap()
print(f"BASE_DIR     = {BASE_DIR}")
print(f"RAW_DATA_DIR = {RAW_DATA_DIR}")
print(f"RESULTS_DIR  = {RESULTS_DIR}")
""")

# ============================================================================
# §2 데이터 로드 + HAR 3채널 + 타깃
# ============================================================================
md("""## §2. 데이터 로드 + HAR 3채널 입력 + target 생성

**v1 변경점**: 입력을 `log_ret²` 1ch 에서 HAR 3채널로 확장.

- **데이터**: Phase 1 raw_data 사본 (SPY/QQQ.csv) 재사용
- **분석 기간**: 2016-01-01 ~ 2025-12-31 (10년, 2,514 영업일)
- **입력 시계열**: HAR 3채널 (Corsi 2009 표준)
  - `RV_d[t] = |log_ret[t]|` — 1일 변동성 (instantaneous)
  - `RV_w[t] = sqrt(mean(log_ret²[t-4:t+1]))` — 5일 평균 변동성
  - `RV_m[t] = sqrt(mean(log_ret²[t-21:t+1]))` — 22일 평균 변동성
- **타깃 시계열**: `build_daily_target_logrv_21d(adj_close)` (v1 동일)
""")
code("""import pandas as pd
import numpy as np
from scripts.targets_volatility import build_daily_target_logrv_21d

ANALYSIS_START = '2016-01-01'
ANALYSIS_END   = '2025-12-31'
WINDOW = 21          # forward window (target)
SEQ_LEN = 63          # PLAN 확정값 (v1 동일)

# HAR 채널 정의 (Corsi 2009 학술 표준)
HAR_W_WINDOW = 5      # weekly
HAR_M_WINDOW = 22     # monthly


def load_ticker(ticker: str) -> pd.DataFrame:
    csv_path = RAW_DATA_DIR / f'{ticker}.csv'
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    df['log_ret'] = np.log(df['Adj Close']).diff()
    return df


raw_dict = {'SPY': load_ticker('SPY'), 'QQQ': load_ticker('QQQ')}

# 분석 기간 절단 + HAR 3채널 + 타깃 생성
analysis_dict = {}
for tk, raw in raw_dict.items():
    df = raw.loc[ANALYSIS_START:ANALYSIS_END].copy()

    # HAR 3채널 입력
    df['rv_d'] = df['log_ret'].abs()                                                 # 채널 1
    df['rv_w'] = (df['log_ret'] ** 2).rolling(HAR_W_WINDOW).mean().pow(0.5)          # 채널 2
    df['rv_m'] = (df['log_ret'] ** 2).rolling(HAR_M_WINDOW).mean().pow(0.5)          # 채널 3

    # 타깃 (v1 동일)
    df['target_logrv'] = build_daily_target_logrv_21d(df['Adj Close'], window=WINDOW)

    analysis_dict[tk] = df
    n = len(df)
    n_valid_target = int(df['target_logrv'].notna().sum())
    n_valid_rv_m   = int(df['rv_m'].notna().sum())
    print(f'{tk}: n={n}, 유효 타깃={n_valid_target} (마지막 {WINDOW}행 NaN), '
          f'유효 RV_m={n_valid_rv_m} (앞 {HAR_M_WINDOW - 1}행 NaN)')

# HAR 채널 분포 확인 (sanity check)
print()
print('=' * 70)
print('HAR 3채널 통계 (분석 기간 전체, NaN 제외)')
print('=' * 70)
for tk in ('SPY', 'QQQ'):
    df = analysis_dict[tk]
    print(f'\\n{tk}:')
    print(df[['rv_d', 'rv_w', 'rv_m']].describe().round(6))
""")

# ============================================================================
# §3 누수 검증 (v1 과 동일 — 타깃은 v1과 같음)
# ============================================================================
md("""## §3. 타깃 누수 검증 (v1 동일)

타깃 정의는 v1과 같으므로 동일한 검증 통과 예상. assert 3건 + 5행 표.
""")
code("""from scripts.targets_volatility import verify_no_leakage_logrv

for tk in ('SPY', 'QQQ'):
    print('=' * 70)
    print(f'{tk} Log-RV 누수 검증 (v1 와 결과 동일 예상)')
    print('=' * 70)
    df = analysis_dict[tk]
    verify_no_leakage_logrv(
        df['Adj Close'], df['target_logrv'],
        n_checks=3, window=WINDOW, seed=42, ddof=1,
    )
    print()
""")

# ============================================================================
# §4 SequenceDataset 점검 (3채널)
# ============================================================================
md("""## §4. SequenceDataset 점검 — 3채널 입력 확인

`build_fold_datasets(series=RV_d, extra_features=[RV_w, RV_m])` 로 호출 →
내부 `column_stack` 으로 자동 (T, 3) 결합.

**주요 sanity check**:
- `train_ds.X.shape = (441, 63, 3)` 확인 (마지막 차원 3)
- `scaler.mean_.shape = (3,)`, `scale_.shape = (3,)`
""")
code("""from scripts.dataset import build_fold_datasets, walk_forward_folds

# 유효 타깃 수 (v1 과 동일)
N_VALID = min(int(analysis_dict[tk]['target_logrv'].notna().sum()) for tk in ('SPY', 'QQQ'))
print(f'유효 타깃 수 (min over SPY/QQQ): {N_VALID}')

# series (RV_d) + extra_features (RV_w, RV_m) 분리 준비
# - log_ret 첫 행 NaN, RV_w 앞 4행 NaN, RV_m 앞 21행 NaN → 모두 0 으로 채움
# - fold 슬라이싱 (start=0) 시 NaN 위치는 train scaler fit 에 들어가지만,
#   타깃 NaN 은 fold 가 자동 회피하므로 입력 NaN 은 0 채움이 안전
series_dict = {}              # RV_d (1D)
extra_features_dict = {}      # [RV_w, RV_m] (T, 2)
target_dict = {}
for tk in ('SPY', 'QQQ'):
    df = analysis_dict[tk]
    series_dict[tk] = df['rv_d'].fillna(0.0).values                                  # (T,)
    extra_features_dict[tk] = np.column_stack([
        df['rv_w'].fillna(0.0).values,
        df['rv_m'].fillna(0.0).values,
    ])                                                                                # (T, 2)
    target_dict[tk] = df['target_logrv'].values                                       # (T,)
    print(f'{tk}: series.shape={series_dict[tk].shape}, '
          f'extra_features.shape={extra_features_dict[tk].shape}')

# Smoke: fold 0 으로 dataset 생성
sample_folds = walk_forward_folds(
    n=N_VALID, is_len=504, purge=21, emb=63, oos_len=21, step=21,
)
print(f'\\n생성된 fold 수: {len(sample_folds)}')
print(f'첫 fold: train_idx={sample_folds[0][0][:3]}...{sample_folds[0][0][-3:]}, '
      f'test_idx={sample_folds[0][1][:3]}...{sample_folds[0][1][-3:]}')

train_ds_smoke, test_ds_smoke, scaler_smoke = build_fold_datasets(
    series=series_dict['SPY'],
    train_idx=sample_folds[0][0],
    test_idx=sample_folds[0][1],
    seq_len=SEQ_LEN,
    extra_features=extra_features_dict['SPY'],
    target_series=target_dict['SPY'],
)
print(f'\\nSmoke (SPY fold 0):')
print(f'  train X.shape={train_ds_smoke.X.shape} (마지막 차원 3 확인)')
print(f'  test  X.shape={test_ds_smoke.X.shape}  (마지막 차원 3 확인)')
print(f'  scaler.mean_  = {scaler_smoke.mean_}  (3개 채널)')
print(f'  scaler.scale_ = {scaler_smoke.scale_} (3개 채널)')

# 가드: 마지막 차원이 3 인지 확인
assert train_ds_smoke.X.shape[-1] == 3, f'채널 수 불일치: {train_ds_smoke.X.shape}'
assert test_ds_smoke.X.shape[-1]  == 3, f'채널 수 불일치: {test_ds_smoke.X.shape}'
print('\\n[OK] 3채널 입력 구성 확인 완료')
""")

# ============================================================================
# §5 Walk-Forward fold (v1 동일)
# ============================================================================
md("""## §5. Walk-Forward fold 생성 — v1 과 동일 (90 fold)

PLAN §7-6 (정밀도 우선):
- IS=504 / Purge=21 / Embargo=63 / OOS=21 / Step=21 → **90 fold**
- 훈련 샘플 fold 당 441 (Phase 1 의 134 대비 3.3배)
""")
code("""IS_LEN = 504
PURGE = 21
EMBARGO = 63
OOS_LEN = 21
STEP = 21

folds_list = walk_forward_folds(
    n=N_VALID, is_len=IS_LEN, purge=PURGE, emb=EMBARGO, oos_len=OOS_LEN, step=STEP,
)
N_FOLDS = len(folds_list)
print(f'생성된 fold 수: {N_FOLDS} (예상 90)')
print(f'fold 당 훈련 샘플: IS({IS_LEN}) - seq_len({SEQ_LEN}) = {IS_LEN - SEQ_LEN}')
print(f'fold 당 OOS 샘플: {OOS_LEN}')
print(f'전체 OOS 샘플 수: {N_FOLDS * OOS_LEN}')
""")

# ============================================================================
# §6 LSTMRegressor smoke (input_size=3)
# ============================================================================
md("""## §6. LSTMRegressor smoke test — input_size=3 (v2 핵심 변경)

v1 대비 input_size 만 1 → 3 으로 변경. hidden/layers/dropout 동일.

**파라미터 수 변화**: input_size 가 1 → 3 으로 늘면 LSTM 의 input-to-hidden weight (4 × hidden × input_size)
가 4×32×1=128 → 4×32×3=384 (+256). 따라서 총 파라미터 4,513 → 4,769 예상.
""")
code("""import torch
from scripts.models import LSTMRegressor

HIDDEN = 32
NUM_LAYERS = 1
DROPOUT = 0.3
INPUT_SIZE = 3   # v2 변경: 1 → 3 (HAR 3채널)

model_smoke = LSTMRegressor(
    input_size=INPUT_SIZE, hidden_size=HIDDEN,
    num_layers=NUM_LAYERS, dropout=DROPOUT, batch_first=True,
)
n_params = sum(p.numel() for p in model_smoke.parameters() if p.requires_grad)
print(f'LSTMRegressor: input_size={INPUT_SIZE}, hidden={HIDDEN}, '
      f'num_layers={NUM_LAYERS}, dropout={DROPOUT}')
print(f'학습 가능 파라미터 수: {n_params:,} (v1 의 4,513 + 256 = 4,769 예상)')

# Smoke forward pass
xb_smoke = train_ds_smoke.X[:4]
yb_smoke_pred = model_smoke(xb_smoke)
print(f'Smoke forward: input shape={tuple(xb_smoke.shape)}, output shape={tuple(yb_smoke_pred.shape)}')
""")

# ============================================================================
# §7 run_all_folds helper (extra_features 지원)
# ============================================================================
md("""## §7. run_all_folds helper — `extra_features` 지원 추가

v1 대비 변경:
- `extra_features` 인자 추가 → `build_fold_datasets` 호출 시 전달
- `input_size=3` 기본값

기타 동일: loss_type='mse', weight_decay=1e-3, max_epochs=30, early_stop_patience=5, lr_patience=3.
""")
code("""import time
from torch.utils.data import TensorDataset, DataLoader
from scripts.train import train_one_fold, get_device

# 학습 하이퍼파라미터 (v1 동일)
MAX_EPOCHS   = 30
PATIENCE     = 5
LR_PATIENCE  = 3
LR           = 1e-3
WEIGHT_DECAY = 1e-3
LOSS_TYPE    = 'mse'
VAL_RATIO    = 0.2


def build_train_val_loaders(train_ds, val_ratio=VAL_RATIO, batch_size=32):
    \"\"\"IS 훈련을 시간순 80/20 으로 재분할 (val 은 뒤쪽 20% — 최신 구간).\"\"\"
    n = len(train_ds)
    n_val = max(1, int(n * val_ratio))
    n_tr = n - n_val
    train_sub = TensorDataset(train_ds.X[:n_tr], train_ds.y[:n_tr])
    val_sub   = TensorDataset(train_ds.X[n_tr:], train_ds.y[n_tr:])
    tr_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(val_sub, batch_size=max(1, n_val), shuffle=False)
    return tr_loader, va_loader


def run_all_folds(
    ticker: str,
    series: np.ndarray,
    extra_features: np.ndarray,    # v2 추가
    target: np.ndarray,
    folds: list,
    seq_len: int,
    *,
    input_size: int = 3,           # v2 기본값 1 → 3
    hidden: int = HIDDEN, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT,
    max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
    lr: float = LR, weight_decay: float = WEIGHT_DECAY,
    lr_patience: int = LR_PATIENCE,
    loss_type: str = LOSS_TYPE,
    log_every: int = 15, device=None,
):
    \"\"\"단일 ticker 의 90 fold 학습·예측 수집 (HAR 3채널).\"\"\"
    device = get_device('auto') if device is None else torch.device(device)
    print(f'  [{ticker}] device: {device}, n_folds={len(folds)}')
    fold_out = []
    t_start = time.time()
    for k, (tr_idx, te_idx) in enumerate(folds):
        tr_ds_k, te_ds_k, _ = build_fold_datasets(
            series=series, train_idx=tr_idx, test_idx=te_idx,
            seq_len=seq_len,
            extra_features=extra_features,    # v2 추가
            target_series=target,
        )
        tr_loader, va_loader = build_train_val_loaders(tr_ds_k, val_ratio=VAL_RATIO)
        model_k = LSTMRegressor(
            input_size=input_size, hidden_size=hidden, num_layers=num_layers,
            dropout=dropout, batch_first=True,
        )
        result = train_one_fold(
            model_k, tr_loader, va_loader,
            max_epochs=max_epochs, early_stop_patience=patience,
            lr=lr, weight_decay=weight_decay,
            lr_patience=lr_patience,
            loss_type=loss_type,
            device=device, verbose=False,
        )
        # best 로 train(IS 80%) / val(IS 20%) / test(OOS) 모두 예측
        model_k.load_state_dict(result['best_state_dict'])
        model_k.to(device).eval()
        with torch.no_grad():
            y_pred_oos = model_k(te_ds_k.X.to(device)).cpu().numpy()
            n_tr_used  = len(tr_loader.dataset)
            y_pred_train = model_k(tr_ds_k.X[:n_tr_used].to(device)).cpu().numpy()
            y_pred_val   = model_k(tr_ds_k.X[n_tr_used:].to(device)).cpu().numpy()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        fold_out.append({
            'fold': k,
            'train_idx_first': int(tr_idx[0]), 'train_idx_last': int(tr_idx[-1]),
            'test_idx_first': int(te_idx[0]),  'test_idx_last': int(te_idx[-1]),
            'y_true_test':  te_ds_k.y.numpy().astype(float).tolist(),
            'y_pred_test':  y_pred_oos.flatten().astype(float).tolist(),
            'y_true_train': tr_ds_k.y[:n_tr_used].numpy().astype(float).tolist(),
            'y_pred_train': y_pred_train.flatten().astype(float).tolist(),
            'y_true_val':   tr_ds_k.y[n_tr_used:].numpy().astype(float).tolist(),
            'y_pred_val':   y_pred_val.flatten().astype(float).tolist(),
            'best_epoch':   int(result['best_epoch']),
            'best_val_loss': float(result['best_val_loss']),
            'stopped_early': bool(result['stopped_early']),
            'history':      {k: [float(v) for v in vs] for k, vs in result['history'].items()},
        })
        if (k + 1) % log_every == 0 or k == len(folds) - 1:
            elapsed = time.time() - t_start
            avg = elapsed / (k + 1)
            eta = avg * (len(folds) - k - 1)
            print(f'    [{ticker}] fold {k+1}/{len(folds)} '
                  f'best_ep={result["best_epoch"]:>2}, val_loss={result["best_val_loss"]:.4f} '
                  f'| elapsed={elapsed/60:.1f}m, ETA={eta/60:.1f}m')
    return fold_out
""")

# ============================================================================
# §8 학습 — SPY, QQQ 90 fold 각각 (HAR 3ch)
# ============================================================================
md("""## §8. SPY · QQQ 90 fold 학습 ⏳ (HAR 3채널)

⚠️ **실행 시간**: v1 과 유사 (~1.6분 합계). input_size 만 늘었으므로 학습 시간 큰 차이 없음.

학습 과정 모니터링: 15 fold 마다 진행률·ETA 출력.
""")
code("""print('=' * 70)
print('SPY 학습 시작 (HAR 3채널 입력)')
print('=' * 70)
spy_folds_out = run_all_folds(
    ticker='SPY',
    series=series_dict['SPY'],
    extra_features=extra_features_dict['SPY'],
    target=target_dict['SPY'],
    folds=folds_list, seq_len=SEQ_LEN,
)
print()
print('=' * 70)
print('QQQ 학습 시작 (HAR 3채널 입력)')
print('=' * 70)
qqq_folds_out = run_all_folds(
    ticker='QQQ',
    series=series_dict['QQQ'],
    extra_features=extra_features_dict['QQQ'],
    target=target_dict['QQQ'],
    folds=folds_list, seq_len=SEQ_LEN,
)
print()
print(f'학습 완료: SPY {len(spy_folds_out)} fold, QQQ {len(qqq_folds_out)} fold')
""")

# ============================================================================
# §9 평가 — 메트릭 산출
# ============================================================================
md("""## §9. 평가 — 메트릭 산출 + Train-Mean baseline (v1 동일 메트릭)

### Phase 1.5 1차 메트릭 (PLAN §6 PASS 조건 평가용)
- **rmse on Log-RV** — 베이스라인 비교용
- **qlike** — Patton(2011) 비대칭 손실
- **r2_train_mean** — train 평균 baseline 대비 개선 (관문 2)
- **pred_std_ratio** — mean-collapse 진단 (관문 3)
- **mz_regression** — 편향 진단

v1 대비 비교는 §10 의 출력에서 확인.
""")
code("""from scripts.metrics_volatility import (
    rmse, qlike, r2_train_mean, mz_regression, pred_std_ratio,
    summarize_folds_volatility,
)


def compute_per_fold_metrics(folds_out, target_full):
    \"\"\"각 fold 의 메트릭 산출.\"\"\"
    per_fold = []
    for fo in folds_out:
        y_true = np.array(fo['y_true_test'])
        y_pred = np.array(fo['y_pred_test'])
        y_train_full = np.array(fo['y_true_train'] + fo['y_true_val'])
        mz = mz_regression(y_true, y_pred)
        per_fold.append({
            'rmse': rmse(y_true, y_pred),
            'qlike': qlike(y_true, y_pred),
            'r2_train_mean': r2_train_mean(y_true, y_pred, y_train_full),
            'pred_std_ratio': pred_std_ratio(y_true, y_pred),
            'mz_alpha': mz['alpha'], 'mz_beta': mz['beta'], 'mz_r2': mz['r2'],
            'best_epoch': fo['best_epoch'],
            'best_val_loss': fo['best_val_loss'],
        })
    return per_fold


per_fold_spy = compute_per_fold_metrics(spy_folds_out, target_dict['SPY'])
per_fold_qqq = compute_per_fold_metrics(qqq_folds_out, target_dict['QQQ'])

summary_spy = summarize_folds_volatility(per_fold_spy)
summary_qqq = summarize_folds_volatility(per_fold_qqq)

# 핵심 메트릭 요약 표
print('=' * 80)
print('Phase 1.5 v2 LSTM 종합 메트릭 (HAR 3ch, 90 fold mean ± std)')
print('=' * 80)
print(f"{'metric':<20} {'SPY mean':>12} {'SPY std':>10} {'QQQ mean':>12} {'QQQ std':>10}")
print('-' * 80)
for m in ['rmse', 'qlike', 'r2_train_mean', 'pred_std_ratio', 'mz_alpha', 'mz_beta', 'mz_r2', 'best_epoch']:
    s = summary_spy.get(m, {})
    q = summary_qqq.get(m, {})
    print(f"{m:<20} {s.get('mean', float('nan')):>12.4f} {s.get('std', float('nan')):>10.4f} "
          f"{q.get('mean', float('nan')):>12.4f} {q.get('std', float('nan')):>10.4f}")
""")

# ============================================================================
# §9.A 학습곡선 갤러리 (v1 동일)
# ============================================================================
md("""### §9.A 학습곡선 갤러리 — fold 0·25·50·75·89 × SPY/QQQ × train/val loss

v1 대비 학습 안정성 변화 진단.
""")
code("""import matplotlib.pyplot as plt


def plot_learning_curves(folds_out_dict, sample_folds_idx=(0, 25, 50, 75, 89)):
    n_folds_max = len(next(iter(folds_out_dict.values())))
    selected = [k for k in sample_folds_idx if k < n_folds_max]
    fig, axes = plt.subplots(len(folds_out_dict), len(selected),
                              figsize=(3.0 * len(selected), 2.8 * len(folds_out_dict)),
                              squeeze=False)
    for r, (tk, fo_list) in enumerate(folds_out_dict.items()):
        for c, k in enumerate(selected):
            hist = fo_list[k]['history']
            ax = axes[r, c]
            ax.plot(hist['train_loss'], label='train', lw=1.0)
            ax.plot(hist['val_loss'],   label='val',   lw=1.0)
            ax.set_title(f'{tk} v2 — fold {k}')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_learning_curves({'SPY': spy_folds_out, 'QQQ': qqq_folds_out})
""")

# ============================================================================
# §9.B best_epoch 분포 (v1 동일)
# ============================================================================
md("""### §9.B best_epoch 분포 — EarlyStopping 패턴 진단

`best_epoch == 1` 비율이 높으면 학습 자체가 거의 안 됐음을 의미.
""")
code("""def plot_best_epoch_hist(folds_out_dict):
    fig, axes = plt.subplots(1, len(folds_out_dict), figsize=(11, 3.5))
    for ax, (tk, fo_list) in zip(axes, folds_out_dict.items()):
        bes = [fo['best_epoch'] for fo in fo_list]
        ax.hist(bes, bins=range(1, max(bes)+2), color='steelblue', edgecolor='white')
        ratio_1 = sum(1 for e in bes if e == 1) / len(bes)
        ax.set_title(f'{tk} v2 best_epoch 분포 (best=1 비율: {ratio_1:.1%})')
        ax.set_xlabel('best_epoch')
        ax.set_ylabel('fold count')
    plt.tight_layout()
    plt.show()


plot_best_epoch_hist({'SPY': spy_folds_out, 'QQQ': qqq_folds_out})
""")

# ============================================================================
# §9.C 예측 분포 sanity (mean-collapse 진단)
# ============================================================================
md("""### §9.C 예측 분포 sanity (PLAN §6 관문 3 — pred_std_ratio > 0.5)

**v1 결과**: SPY=0.35, QQQ=0.35 (모두 FAIL)
**v2 기대**: HAR 3채널이 변수 부족 가설 a 의 핵심 검증 → ratio 가 0.5 이상으로 회복되는지 확인.
""")
code("""def plot_pred_distribution(folds_out_dict):
    fig, axes = plt.subplots(1, len(folds_out_dict), figsize=(11, 4))
    for ax, (tk, fo_list) in zip(axes, folds_out_dict.items()):
        all_true = np.concatenate([fo['y_true_test'] for fo in fo_list])
        all_pred = np.concatenate([fo['y_pred_test'] for fo in fo_list])
        std_t = float(all_true.std())
        std_p = float(all_pred.std())
        ratio = std_p / std_t if std_t > 0 else float('nan')
        ax.hist(all_true, bins=40, alpha=0.5, label='true', color='steelblue', density=True)
        ax.hist(all_pred, bins=40, alpha=0.5, label='pred', color='darkorange', density=True)
        ax.set_title(f'{tk} v2 — pred_std/true_std = {ratio:.3f} '
                     f'({"PASS" if ratio > 0.5 else "FAIL — mean-collapse"})')
        ax.set_xlabel('log(RV)')
        ax.legend()
    plt.tight_layout()
    plt.show()


plot_pred_distribution({'SPY': spy_folds_out, 'QQQ': qqq_folds_out})
""")

# ============================================================================
# §9.D 잔차 시계열 + MZ 산점도
# ============================================================================
md("""### §9.D 잔차 시계열 + MZ 산점도

- 잔차 (y_pred - y_true) 시간 추이로 체제 편향 검출
- MZ 산점도: y_true vs y_pred + 1:1 선 + OLS 적합선

**v1 결과**: SPY β=-0.68, QQQ β=-0.31 (음의 상관 → 편향 강함)
**v2 기대**: β 가 양수 영역으로 회복되는지.
""")
code("""def plot_residual_and_mz(folds_out_dict):
    fig, axes = plt.subplots(2, len(folds_out_dict), figsize=(11, 8))
    for c, (tk, fo_list) in enumerate(folds_out_dict.items()):
        all_true = np.concatenate([fo['y_true_test'] for fo in fo_list])
        all_pred = np.concatenate([fo['y_pred_test'] for fo in fo_list])
        residual = all_pred - all_true
        axes[0, c].plot(residual, lw=0.5, color='steelblue')
        axes[0, c].axhline(0, color='red', lw=0.7)
        axes[0, c].set_title(f'{tk} v2 — 잔차 (pred − true) 시계열')
        axes[0, c].set_xlabel('OOS sample order')
        axes[0, c].set_ylabel('residual')
        mz = mz_regression(all_true, all_pred)
        axes[1, c].scatter(all_pred, all_true, s=6, alpha=0.4, color='steelblue')
        x_line = np.linspace(all_pred.min(), all_pred.max(), 100)
        axes[1, c].plot(x_line, x_line, 'r--', lw=1, label='1:1')
        axes[1, c].plot(x_line, mz['alpha'] + mz['beta'] * x_line, 'g-', lw=1,
                         label=f"α={mz['alpha']:.3f}, β={mz['beta']:.3f}, R²={mz['r2']:.3f}")
        axes[1, c].set_title(f'{tk} v2 — MZ 산점도 (y_true vs y_pred)')
        axes[1, c].set_xlabel('y_pred')
        axes[1, c].set_ylabel('y_true')
        axes[1, c].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


plot_residual_and_mz({'SPY': spy_folds_out, 'QQQ': qqq_folds_out})
""")

# ============================================================================
# §9.E 박스플롯
# ============================================================================
md("""### §9.E 박스플롯 — fold 별 메트릭 분포

체제 변화 (COVID, 긴축 등) 에 의한 fold-level outlier 진단.
""")
code("""def plot_metric_boxplots(per_fold_dict):
    metrics = ['rmse', 'qlike', 'r2_train_mean', 'pred_std_ratio']
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        data = []
        labels = []
        for tk, pf in per_fold_dict.items():
            vals = [d[m] for d in pf if not np.isnan(d[m])]
            data.append(vals)
            labels.append(tk)
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(f'{m} (v2)')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_metric_boxplots({'SPY': per_fold_spy, 'QQQ': per_fold_qqq})
""")

# ============================================================================
# §9.F train/val/test 갭
# ============================================================================
md("""### §9.F train/val/test 갭 — 과적합 정량화

각 fold 의 train/val/test 에서 동일 메트릭(RMSE) 차이가 크면 과적합.

**v1 결과**: gap_test_minus_train SPY=-0.0448, QQQ=-0.0360 (test 가 train 보다 작음 — 약간 비전형)
""")
code("""def compute_train_val_test_gap(folds_out):
    rows = []
    for fo in folds_out:
        y_tr = np.array(fo['y_true_train']); p_tr = np.array(fo['y_pred_train'])
        y_va = np.array(fo['y_true_val']);   p_va = np.array(fo['y_pred_val'])
        y_te = np.array(fo['y_true_test']);  p_te = np.array(fo['y_pred_test'])
        rows.append({
            'fold': fo['fold'],
            'rmse_train': rmse(y_tr, p_tr),
            'rmse_val':   rmse(y_va, p_va),
            'rmse_test':  rmse(y_te, p_te),
        })
    return pd.DataFrame(rows)


gap_spy = compute_train_val_test_gap(spy_folds_out)
gap_qqq = compute_train_val_test_gap(qqq_folds_out)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (tk, gap) in zip(axes, [('SPY', gap_spy), ('QQQ', gap_qqq)]):
    ax.plot(gap['fold'], gap['rmse_train'], label='train', lw=1)
    ax.plot(gap['fold'], gap['rmse_val'],   label='val',   lw=1)
    ax.plot(gap['fold'], gap['rmse_test'],  label='test',  lw=1)
    ax.set_title(f'{tk} v2 — RMSE train/val/test 갭')
    ax.set_xlabel('fold')
    ax.set_ylabel('rmse')
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print('\\nRMSE 갭 평균 (test - train, 양수일수록 과적합 신호):')
for tk, gap in [('SPY', gap_spy), ('QQQ', gap_qqq)]:
    diff = (gap['rmse_test'] - gap['rmse_train']).mean()
    print(f'  {tk}: {diff:.4f}')
""")

# ============================================================================
# §10 결론 + metrics.json 저장 + v1 vs v2 비교
# ============================================================================
md("""## §10. 결론 + metrics.json 저장 + v1 vs v2 비교

`results/volatility_lstm_har3ch/{SPY,QQQ}_metrics.json` 에 직렬화 (v2 격리 보존).

추가로 **v1 vs v2 비교 표** 출력 — 변수 부족 가설 a 의 직접 검증 결과.
""")
code("""import json

OUT_DIR = RESULTS_DIR / 'volatility_lstm_har3ch'   # v2 결과 폴더 (v1 보존)
OUT_DIR.mkdir(parents=True, exist_ok=True)

hyperparams = {
    'seed': 42,
    'analysis_period': [ANALYSIS_START, ANALYSIS_END],
    'walk_forward': {
        'IS': IS_LEN, 'purge': PURGE, 'embargo': EMBARGO,
        'OOS': OOS_LEN, 'step': STEP, 'n_folds': N_FOLDS,
    },
    'model': {
        'class': 'LSTMRegressor',
        'input_size': INPUT_SIZE, 'hidden_size': HIDDEN,
        'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
        'batch_first': True,
    },
    'training': {
        'loss_type': LOSS_TYPE,
        'max_epochs': MAX_EPOCHS, 'early_stop_patience': PATIENCE,
        'lr': LR, 'weight_decay': WEIGHT_DECAY, 'lr_patience': LR_PATIENCE,
        'val_ratio': VAL_RATIO,
    },
    'seq_len': SEQ_LEN,
    'input_feature': 'har_3channel',           # v2 변경 (v1: log_ret_squared)
    'input_channels': ['rv_d', 'rv_w', 'rv_m'],  # v2 추가
    'har_w_window': HAR_W_WINDOW,
    'har_m_window': HAR_M_WINDOW,
    'target': 'log_realized_volatility_21d',
    'version': 'v2_har3ch',
}

for tk, fo_list, pf_list, summary in [
    ('SPY', spy_folds_out, per_fold_spy, summary_spy),
    ('QQQ', qqq_folds_out, per_fold_qqq, summary_qqq),
]:
    out = {
        'ticker': tk,
        'hyperparams': hyperparams,
        'summary': summary,
        'per_fold': pf_list,
        'fold_predictions': fo_list,
    }
    out_path = OUT_DIR / f'{tk}_metrics.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'저장: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)')

# ====================================================================
# v1 vs v2 비교 표 — 변수 부족 가설 a 의 직접 검증
# ====================================================================
print()
print('=' * 80)
print('v1 vs v2 비교 — 변수 부족 가설 a 직접 검증')
print('=' * 80)

V1_DIR = RESULTS_DIR / 'volatility_lstm'
v1_summary = {}
for tk in ('SPY', 'QQQ'):
    v1_path = V1_DIR / f'{tk}_metrics.json'
    if v1_path.exists():
        with open(v1_path, 'r', encoding='utf-8') as f:
            v1_summary[tk] = json.load(f)['summary']
    else:
        v1_summary[tk] = None
        print(f'[경고] v1 결과 없음: {v1_path}')

if all(v1_summary.values()):
    metrics_to_compare = ['rmse', 'qlike', 'r2_train_mean', 'pred_std_ratio',
                          'mz_alpha', 'mz_beta', 'mz_r2', 'best_epoch']
    for tk in ('SPY', 'QQQ'):
        v1_s = v1_summary[tk]
        v2_s = summary_spy if tk == 'SPY' else summary_qqq
        print(f'\\n[{tk}]')
        print(f"  {'metric':<20} {'v1 mean':>12} {'v2 mean':>12} {'Δ (v2-v1)':>14}")
        print('  ' + '-' * 62)
        for m in metrics_to_compare:
            v1_v = v1_s.get(m, {}).get('mean', float('nan'))
            v2_v = v2_s.get(m, {}).get('mean', float('nan'))
            delta = v2_v - v1_v
            print(f"  {m:<20} {v1_v:>12.4f} {v2_v:>12.4f} {delta:>14.4f}")

    # 관문 판정 요약
    print()
    print('=' * 80)
    print('Phase 1.5 PASS 조건 점검 (v2)')
    print('=' * 80)
    for tk in ('SPY', 'QQQ'):
        v2_s = summary_spy if tk == 'SPY' else summary_qqq
        r2 = v2_s.get('r2_train_mean', {}).get('mean', float('nan'))
        psr = v2_s.get('pred_std_ratio', {}).get('mean', float('nan'))
        gate2 = 'PASS' if r2 > 0 else 'FAIL'
        gate3 = 'PASS' if psr > 0.5 else 'FAIL'
        print(f'\\n  [{tk}]')
        print(f'    관문 2 (r2_train_mean > 0)   : {r2:>8.4f}  → {gate2}')
        print(f'    관문 3 (pred_std_ratio > 0.5): {psr:>8.4f}  → {gate3}')
        print(f'    관문 1 (LSTM RMSE < HAR-RV)  : §03 베이스라인 비교에서 판정')

print()
print('=' * 80)
print('Phase 1.5 v2 LSTM 실행 완료')
print()
print('다음 단계 안내:')
print('  - v2 PASS    | 변수 부족 가설 a 확정 → §03 베이스라인 비교 진행')
print('  - v2 FAIL    | LSTM 자체 한계 가능성 ↑ → 추가 실험 검토')
print('  - v2 borderline | §03 비교로 HAR-RV 와 비교 후 종합 판정')
print('=' * 80)
""")

# ============================================================================
# 노트북 저장
# ============================================================================
NB.cells = cells
OUT_PATH = Path(__file__).resolve().parent / '02_v2_volatility_lstm_har3ch.ipynb'
nbf.write(NB, str(OUT_PATH))
print(f'노트북 빌드 완료: {OUT_PATH}')
print(f'총 셀 수: {len(cells)}')
