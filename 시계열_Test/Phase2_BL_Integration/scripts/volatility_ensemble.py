"""Phase 2 — Volatility Ensemble (Phase 1.5 v8 Performance-Weighted) 74 종목 확장.

Phase 1.5 v4 best (3ch_vix / IS=1250 / embargo=63) + HAR-RV 의
Performance-Weighted Ensemble 을 본 단계 universe 74 종목에 적용.

핵심 결정사항 (PLAN.md §2):
- 결정 4: 신규 편입 종목만 첫 fold 0.5/0.5 reset, 기존 종목은 history 유지
- 결정 7: OOS=21, IS=1250, embargo=63, step=21 (Phase 1.5 일관)

핵심 함수
---------
- build_v4_inputs(panel_ticker)         : rv_d, rv_w, rv_m, vix_log 입력 준비
- run_lstm_v4_fold(...)                  : 단일 fold LSTM v4 학습 + 예측
- run_har_fold(log_ret, train_idx, ...)  : HAR-RV 단일 fold 적합 + 예측
- run_walkforward_for_ticker(...)        : 종목 한 개의 walk-forward 전체 실행
- compute_performance_weights(...)       : Performance-Weighted 가중치 (rolling)
- run_ensemble_for_universe(...)         : 전체 universe 종목 처리 + 신규 reset

사용 예시
---------
from scripts.volatility_ensemble import run_ensemble_for_universe
results = run_ensemble_for_universe(
    panel_csv=DATA_DIR / 'daily_panel.csv',
    universe_csv=DATA_DIR / 'universe_top50_history.csv',
    out_dir=DATA_DIR,
    is_len=1250,
    seq_len=63,
    embargo=63,
    oos_len=21,
    step=21,
    device='auto',
    tickers_subset=None,    # None → 전체 universe, 또는 ['AAPL', ...] subset
)
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Phase 1.5 모듈 직접 로드 (Phase 2 의 'scripts' 와 이름 충돌 회피)
from .setup import PHASE15_DIR


def _load_phase15_module(name: str, filename: str):
    """Phase 1.5 의 scripts/ 모듈을 importlib 로 직접 로드.

    Phase 2 의 'scripts' 패키지와 이름 충돌을 회피하기 위해
    'phase15_<name>' 으로 alias 하여 sys.modules 에 등록.
    """
    path = PHASE15_DIR / 'scripts' / filename
    if not path.exists():
        raise FileNotFoundError(f'Phase 1.5 모듈 없음: {path}')
    alias = f'phase15_{name}'
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# Phase 1.5 의 핵심 함수·클래스 alias
_dataset = _load_phase15_module('dataset', 'dataset.py')
_models = _load_phase15_module('models', 'models.py')
_train = _load_phase15_module('train', 'train.py')
_baselines = _load_phase15_module('baselines_volatility', 'baselines_volatility.py')

build_fold_datasets = _dataset.build_fold_datasets
walk_forward_folds = _dataset.walk_forward_folds
LSTMRegressor = _models.LSTMRegressor
train_one_fold = _train.train_one_fold
fit_har_rv = _baselines.fit_har_rv


# =============================================================================
# Phase 1.5 v4 best 하이퍼파라미터
# =============================================================================
V4_BEST_CONFIG = {
    'input_channels': '3ch_vix',
    'hidden_size': 32,
    'num_layers': 1,
    'dropout': 0.3,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'loss_type': 'mse',
    'huber_delta': 0.01,    # MSE 사용 시 무관
    'max_epochs': 50,
    'early_stop_patience': 10,
    'lr_patience': 5,
    'lr_factor': 0.5,
    'batch_size': 64,
    'is_len': 1250,
    'seq_len': 63,
    'embargo': 63,
    'oos_len': 21,
    'step': 21,
    'window': 21,           # forward target window
    'har_w': 5,
    'har_m': 22,
}


# =============================================================================
# 입력 빌드 (4ch_vix)
# =============================================================================
def build_v4_inputs(
    panel_ticker: pd.DataFrame,
    har_w: int = 5,
    har_m: int = 22,
) -> dict:
    """단일 종목 panel 에서 Phase 1.5 v4 best (3ch_vix) 입력 준비.

    Parameters
    ----------
    panel_ticker : pd.DataFrame
        daily_panel.csv 의 ticker 별 슬라이스. columns: log_ret, vix, target_logrv 등.

    Returns
    -------
    dict
        {
            'series': np.ndarray (rv_d, 1차원),
            'extra': np.ndarray (n × 3 — rv_w, rv_m, vix_log),
            'target': np.ndarray (target_logrv),
            'log_ret': pd.Series (HAR-RV 용),
            'index': pd.DatetimeIndex,
            'input_size': 4,
        }
    """
    df = panel_ticker.copy().sort_values('date').reset_index(drop=True)

    log_ret = df['log_ret'].values

    # rv_d, rv_w, rv_m (Phase 1.5 v4 정의)
    rv_d = pd.Series(log_ret).abs().fillna(0.0).values
    log_ret_sq = pd.Series(log_ret) ** 2
    rv_w = log_ret_sq.rolling(har_w).mean().pow(0.5).fillna(0.0).values
    rv_m = log_ret_sq.rolling(har_m).mean().pow(0.5).fillna(0.0).values

    # VIX log (pandas FutureWarning 회피: ffill/bfill 직접 호출)
    vix = df['vix'].ffill().bfill().fillna(20.0)
    vix_log = np.log(vix.clip(lower=1e-6)).values

    # target
    target = df['target_logrv'].values

    extra = np.column_stack([rv_w, rv_m, vix_log])

    return {
        'series': rv_d,
        'extra': extra,
        'target': target,
        'log_ret': pd.Series(log_ret, index=df['date'].values),
        'index': pd.to_datetime(df['date'].values),
        'input_size': 4,
        'date': df['date'].values,
    }


# =============================================================================
# 단일 fold LSTM v4 학습 + 예측
# =============================================================================
def run_lstm_v4_fold(
    inputs: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: dict = V4_BEST_CONFIG,
    device: str = 'auto',
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """단일 fold LSTM v4 학습 + OOS 예측.

    Returns
    -------
    y_pred : np.ndarray (len(test_idx),)
    info : dict (학습 정보)
    """
    series = inputs['series']
    extra = inputs['extra']
    target = inputs['target']

    # NaN target 제거된 인덱스만 사용
    train_idx_valid = train_idx[~np.isnan(target[train_idx])]
    test_idx_valid = test_idx[~np.isnan(target[test_idx])]

    if len(train_idx_valid) < config['seq_len'] + 50:
        return np.full(len(test_idx), np.nan), {'status': 'insufficient_train'}
    if len(test_idx_valid) == 0:
        return np.full(len(test_idx), np.nan), {'status': 'no_valid_test'}

    # Dataset 빌드
    train_ds, test_ds, scaler = build_fold_datasets(
        series=series,
        train_idx=train_idx_valid,
        test_idx=test_idx_valid,
        seq_len=config['seq_len'],
        extra_features=extra,
        target_series=target,
    )

    if len(train_ds) == 0 or len(test_ds) == 0:
        return np.full(len(test_idx), np.nan), {'status': 'empty_dataset'}

    # Train / Val 분할 (마지막 10% val)
    n_train = len(train_ds)
    n_val = max(int(n_train * 0.1), 5)
    n_train_only = n_train - n_val
    train_only_ds = torch.utils.data.Subset(train_ds, list(range(n_train_only)))
    val_ds = torch.utils.data.Subset(train_ds, list(range(n_train_only, n_train)))

    train_loader = DataLoader(train_only_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    # 모델
    model = LSTMRegressor(
        input_size=config['input_size'] if 'input_size' in config else inputs['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        batch_first=True,
    )

    # 학습
    info = train_one_fold(
        model, train_loader, val_loader,
        max_epochs=config['max_epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        huber_delta=config['huber_delta'],
        loss_type=config['loss_type'],
        early_stop_patience=config['early_stop_patience'],
        lr_patience=config['lr_patience'],
        lr_factor=config['lr_factor'],
        device=device,
        verbose=verbose,
        log_every=10,
    )

    # OOS 예측
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    model.eval()
    device_t = next(model.parameters()).device
    y_preds = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device_t)
            y_pred = model(X).cpu().numpy().flatten()
            y_preds.append(y_pred)
    y_pred_test = np.concatenate(y_preds) if y_preds else np.array([])

    # test_idx_valid 위치에 맞춰 결과 매핑 (전체 test_idx 길이 유지)
    y_pred_full = np.full(len(test_idx), np.nan)
    valid_mask = ~np.isnan(target[test_idx])
    if y_pred_test.size == valid_mask.sum():
        y_pred_full[valid_mask] = y_pred_test

    return y_pred_full, info


# =============================================================================
# 종목 한 개의 walk-forward 전체 실행
# =============================================================================
def run_walkforward_for_ticker(
    ticker: str,
    panel_ticker: pd.DataFrame,
    config: dict = V4_BEST_CONFIG,
    device: str = 'auto',
    verbose: bool = False,
) -> pd.DataFrame:
    """종목 1개의 walk-forward 전체 실행.

    Returns
    -------
    pd.DataFrame
        long format: (date, ticker, fold, y_true, y_pred_lstm, y_pred_har)
    """
    inputs = build_v4_inputs(panel_ticker, config['har_w'], config['har_m'])
    n = len(inputs['series'])

    folds = walk_forward_folds(
        n=n,
        is_len=config['is_len'],
        purge=config['window'],
        emb=config['embargo'],
        oos_len=config['oos_len'],
        step=config['step'],
    )

    target = inputs['target']
    log_ret = inputs['log_ret']
    dates = inputs['date']

    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        # LSTM v4
        y_pred_lstm, info = run_lstm_v4_fold(inputs, train_idx, test_idx, config, device, verbose=False)

        # HAR-RV
        try:
            y_pred_har, _ = fit_har_rv(
                log_ret=log_ret,
                train_idx=train_idx,
                test_idx=test_idx,
                horizon=config['window'],
            )
        except Exception:
            y_pred_har = np.full(len(test_idx), np.nan)

        for i, idx in enumerate(test_idx):
            rows.append({
                'date': dates[idx],
                'ticker': ticker,
                'fold': k,
                'y_true': target[idx] if idx < len(target) else np.nan,
                'y_pred_lstm': y_pred_lstm[i] if i < len(y_pred_lstm) else np.nan,
                'y_pred_har': y_pred_har[i] if i < len(y_pred_har) else np.nan,
            })

        if verbose and (k + 1) % 10 == 0:
            print(f'    [{ticker}] fold {k+1}/{len(folds)} 완료')

    return pd.DataFrame(rows)


# =============================================================================
# Performance-Weighted Ensemble (결정 4 — 신규 종목 reset)
# =============================================================================
def compute_performance_weights(
    fold_results: pd.DataFrame,
    initial_weights: Optional[dict] = None,
) -> pd.DataFrame:
    """Performance-Weighted ensemble 가중치 + 예측 계산.

    Parameters
    ----------
    fold_results : pd.DataFrame
        columns: date, ticker, fold, y_true, y_pred_lstm, y_pred_har
    initial_weights : dict | None
        {'w_v4': float, 'w_har': float} 첫 fold warmup. None 시 0.5/0.5.

    Returns
    -------
    pd.DataFrame
        + columns: w_v4, w_har, y_pred_ensemble

    Notes
    -----
    공식 (Phase 1.5 v8 Performance-Weighted, Diebold-Pauly 1987):
        w_v4[k]  = (1/RMSE_v4[k-1]) / (1/RMSE_v4[k-1] + 1/RMSE_har[k-1])
        w_har[k] = 1 - w_v4[k]
        y_pred_ensemble[k] = w_v4[k] · y_pred_lstm[k] + w_har[k] · y_pred_har[k]

    첫 fold (k=0): initial_weights 사용 (default 0.5/0.5).
    """
    df = fold_results.sort_values(['ticker', 'fold', 'date']).copy()

    if initial_weights is None:
        initial_weights = {'w_v4': 0.5, 'w_har': 0.5}

    out_rows = []
    for ticker, ticker_df in df.groupby('ticker'):
        ticker_df = ticker_df.sort_values(['fold', 'date']).copy()
        folds_unique = sorted(ticker_df['fold'].unique())

        # fold 별 RMSE 계산 (이전 fold 의 OOS RMSE 사용)
        prev_rmse_v4 = None
        prev_rmse_har = None
        cur_w_v4 = initial_weights['w_v4']
        cur_w_har = initial_weights['w_har']

        for k in folds_unique:
            mask = ticker_df['fold'] == k
            fold_rows = ticker_df[mask].copy()

            # 가중치 결정 (이전 fold 결과 기반)
            if prev_rmse_v4 is not None and prev_rmse_har is not None:
                inv_v4 = 1.0 / max(prev_rmse_v4, 1e-6)
                inv_har = 1.0 / max(prev_rmse_har, 1e-6)
                cur_w_v4 = inv_v4 / (inv_v4 + inv_har)
                cur_w_har = 1 - cur_w_v4
            # else: 첫 fold 또는 결측 → initial_weights 유지

            fold_rows['w_v4'] = cur_w_v4
            fold_rows['w_har'] = cur_w_har
            fold_rows['y_pred_ensemble'] = (
                cur_w_v4 * fold_rows['y_pred_lstm']
                + cur_w_har * fold_rows['y_pred_har']
            )

            # 본 fold OOS RMSE (다음 fold 가중치용)
            valid = fold_rows.dropna(subset=['y_true', 'y_pred_lstm', 'y_pred_har'])
            if len(valid) > 0:
                err_v4 = (valid['y_pred_lstm'] - valid['y_true']).values
                err_har = (valid['y_pred_har'] - valid['y_true']).values
                prev_rmse_v4 = float(np.sqrt(np.mean(err_v4 ** 2)))
                prev_rmse_har = float(np.sqrt(np.mean(err_har ** 2)))

            out_rows.append(fold_rows)

    return pd.concat(out_rows, ignore_index=True)


# =============================================================================
# 전체 universe 처리 (결정 4 — 신규 reset, 기존 history 유지)
# =============================================================================
def run_ensemble_for_universe(
    panel_csv: Path,
    universe_csv: Path,
    out_dir: Path,
    config: dict = V4_BEST_CONFIG,
    device: str = 'auto',
    tickers_subset: Optional[list] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """전체 universe 74 종목 walk-forward + Performance ensemble.

    종목별 가중치 history 처리:
    - 신규 편입 종목: 첫 fold 0.5/0.5 (warmup)
    - 기존 종목: 이전 모든 fold 의 history 가 자연스럽게 누적됨 (compute_performance_weights 가 처리)

    Parameters
    ----------
    panel_csv : daily_panel.csv 경로
    universe_csv : universe_top50_history.csv 경로
    out_dir : 결과 저장 디렉토리
    config : V4_BEST_CONFIG (기본)
    device : 'auto' / 'cuda' / 'cpu'
    tickers_subset : None 시 전체 universe. list 제공 시 subset 만 학습 (PoC)
    verbose : 진행 출력

    Returns
    -------
    pd.DataFrame
        ensemble_predictions_top50.csv 의 내용
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_csv, parse_dates=['date'])
    universe_df = pd.read_csv(universe_csv, parse_dates=['cutoff_date'])

    # 종목 선정
    all_tickers = sorted(universe_df['ticker'].unique().tolist())
    if tickers_subset is not None:
        tickers = [t for t in all_tickers if t in tickers_subset]
        print(f'  [ensemble] subset 모드: {len(tickers)}/{len(all_tickers)} 종목 학습')
    else:
        tickers = all_tickers
        print(f'  [ensemble] 전체 모드: {len(tickers)} 종목 학습')

    # 종목별 walk-forward (시간 병목)
    fold_results_all = []
    t_start = time.time()
    for i, ticker in enumerate(tickers):
        t_ticker = time.time()
        panel_t = panel[panel['ticker'] == ticker].copy()
        if len(panel_t) < config['is_len'] + config['seq_len'] + config['oos_len']:
            print(f'    [{ticker}] 데이터 부족 → skip')
            continue
        try:
            df_t = run_walkforward_for_ticker(ticker, panel_t, config, device, verbose=False)
            fold_results_all.append(df_t)
            elapsed = time.time() - t_ticker
            n_folds = df_t['fold'].nunique()
            print(f'    [{i+1}/{len(tickers)}] {ticker}: {n_folds} fold ({elapsed:.0f}s)')
        except Exception as e:
            print(f'    [{ticker}] 학습 실패: {e}')

        # 중간 저장 (안전망)
        if (i + 1) % 10 == 0:
            partial = pd.concat(fold_results_all, ignore_index=True)
            partial.to_csv(out_dir / 'ensemble_predictions_partial.csv', index=False)

    fold_results = pd.concat(fold_results_all, ignore_index=True)
    fold_results.to_csv(out_dir / 'fold_predictions_lstm_har.csv', index=False)
    print(f'  [ensemble] LSTM + HAR fold predictions 저장')

    # Performance ensemble 가중치 계산 (결정 4 — 신규 reset 자동 처리)
    print(f'  [ensemble] Performance-Weighted 가중치 계산 중...')
    ensemble_df = compute_performance_weights(fold_results)
    ensemble_df.to_csv(out_dir / 'ensemble_predictions_top50.csv', index=False)

    elapsed_total = time.time() - t_start
    print(f'  [ensemble] 전체 완료: {len(tickers)} 종목, 총 {elapsed_total/60:.1f}분')
    print(f'  [ensemble] 저장: {out_dir / "ensemble_predictions_top50.csv"}')

    return ensemble_df
