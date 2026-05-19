"""
lstm_pipeline.py — LSTM (4 채널: rv_d/rv_w/rv_m/vix_log) + HAR-RV Performance Ensemble
                    high-level orchestration.

final_pt/timeseries_lib.py 의 low-level 함수 (LSTMRegressor / build_fold_inputs /
train_one_fold / fit_har_rv / walk_forward_folds / diebold_pauly_weights) 를 사용하여
종목 단위 walk-forward 부터 universe 일괄 8-way 병렬 학습까지 완결합니다.

03b_Volatility_Forecasting.ipynb 에서 import 하여 사용:
    >>> from lstm_pipeline import (
    ...     V4_BEST_CONFIG, build_daily_panel,
    ...     prepare_incremental_state, run_ensemble_for_universe_parallel,
    ... )

User-facing API
---------------
- V4_BEST_CONFIG (dict)                : 03a Optuna best (input_channels='3ch_vix',
                                          is_len=1250, embargo=63, ...)
- build_daily_panel()                  : final_pt/data 의 daily_returns + macro(vix)
                                          → long-format daily_panel (date, ticker,
                                          log_ret, vix, target_logrv)
- prepare_incremental_state()          : fold csv 진단 + 자동 truncate
- run_ensemble_for_universe_parallel() : universe × 8-way 병렬 학습 (incremental 지원)

내부 helper (run_ensemble_for_universe_parallel 가 호출)
-------------------------------------------------------
- build_v4_inputs(), run_lstm_v4_fold(), run_walkforward_for_ticker(),
  diagnose_fold_csv(), compute_performance_weights(),
  auto_n_workers(), _train_ticker_worker()

설계 — final_pt 단독 완결
-------------------------
모든 입력 데이터는 final_pt/data + final_pt/data/03b_lstm/data 만 사용.
fold csv 가 final_pt/phase3 에 .csv 또는 .csv.gz 형태로 있으면 자동 활용.
외부 폴더 의존성 없음.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

_THIS_DIR = Path(__file__).parent.parent

from .timeseries_lib import (   # noqa: E402
    LSTMRegressor,
    build_fold_inputs,
    train_one_fold,
    walk_forward_folds,
    fit_har_rv,
    diebold_pauly_weights,
    rmse as _rmse,
)


# =============================================================================
# LSTM 채택 하이퍼파라미터 (03a_LSTM_Optuna_GridSearch.ipynb §2 Optuna best)
# =============================================================================
V4_BEST_CONFIG = {
    'input_channels'      : '3ch_vix',
    'hidden_size'         : 32,
    'num_layers'          : 1,
    'dropout'             : 0.3,
    'lr'                  : 1e-3,
    'weight_decay'        : 1e-3,
    'loss_type'           : 'mse',
    'max_epochs'          : 50,
    'early_stop_patience' : 10,
    'batch_size'          : 64,
    'is_len'              : 1250,
    'seq_len'             : 63,
    'embargo'             : 63,
    'oos_len'             : 21,
    'step'                : 21,
    'har_w'               : 5,
    'har_m'               : 22,
    'input_size'          : 4,   # 4 channels: rv_d, rv_w, rv_m, vix_log
}


# =============================================================================
# 1. Daily panel 빌드 (final_pt/data 의 daily_returns + macro(vix) → long format)
# =============================================================================
def build_daily_panel(
    daily_returns_path: str | Path = None,
    macro_path: str | Path = None,
    universe_path: str | Path = None,
    out_path: Optional[str | Path] = None,
    horizon: int = 21,
    max_date: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """final_pt/data 의 daily 데이터로 LSTM 학습용 long format daily_panel 빌드.

    출력 컬럼: date, ticker, log_ret, vix, target_logrv

    Parameters
    ----------
    daily_returns_path : str | Path
        final_pt/data/daily_returns.pkl (wide format: date × ticker)
    macro_path : str | Path
        final_pt/data/macro_daily.csv ('vix' 컬럼 포함)
    universe_path : str | Path
        final_pt/data/universe.csv (LSTM 학습 universe)
    out_path : str | Path | None
        결과 저장 경로 (None 시 저장 안 함, DataFrame 만 반환)
    horizon : int, default 21
        target_logrv 의 forward 영업일 수
    max_date : str | None, default None
        'YYYY-MM-DD' 형태로 panel 의 끝점 cap. None 시 daily_returns 의 모든 데이터 사용.
        BL 분석 기간 + forward 21d buffer 만큼만 cap 하면 학습 시간을 줄일 수 있음
        (예: 분석 끝이 2025-12-31 이면 max_date='2026-02-01' 정도). 03b 노트북은
        max_date='2026-04-30' 로 설정하여 BL 의 fold 222~226 까지 cover.
    verbose : bool

    Notes
    -----
    생존편향 (시점별 S&P500 멤버십) 필터는 본 함수에서 적용하지 않음 — universe 전체 종목으로
    LSTM 학습. 시점별 멤버십 strict 필터는 BL 단계 (01_DataCollection.ipynb 의 monthly_panel
    빌드) 에서 적용되며, BL walk-forward 입력에는 시점별 멤버 종목만 포함됨.

    Returns
    -------
    pd.DataFrame
        long format with [date, ticker, log_ret, vix, target_logrv]
    """
    base = _THIS_DIR / 'data'
    daily_returns_path = Path(daily_returns_path or base / 'daily_returns.pkl')
    macro_path         = Path(macro_path         or base / 'macro_daily.csv')
    universe_path      = Path(universe_path      or base / 'universe.csv')

    if verbose:
        print('━' * 70)
        print(' build_daily_panel — final_pt/data → long format LSTM 입력')
        print('━' * 70)

    # 1. daily_returns (wide → long)
    if verbose: print(f'  [1/4] daily_returns 로드 ...')
    with open(daily_returns_path, 'rb') as f:
        dr = pickle.load(f)
    if not isinstance(dr, pd.DataFrame):
        raise ValueError(f'daily_returns 가 DataFrame 이 아님: {type(dr)}')

    # max_date 적용 — panel 의 끝점 cap (학습 시간 단축)
    if max_date is not None:
        max_ts = pd.Timestamp(max_date)
        dr = dr[dr.index <= max_ts]
        if verbose:
            print(f'    max_date={max_date} 적용 — panel 끝점 cap')

    # long format
    long = dr.stack().reset_index()
    long.columns = ['date', 'ticker', 'log_ret']
    long['date'] = pd.to_datetime(long['date'])
    long = long.dropna(subset=['log_ret'])
    if verbose:
        print(f'    long format: {len(long):,} rows ({long.ticker.nunique()} tickers)')

    # 2. universe 필터 (시점별 strict 멤버십은 BL 단계에서 처리)
    if verbose: print(f'  [2/4] universe 필터 ...')
    universe = pd.read_csv(universe_path)
    uni_set = set(universe['ticker'].unique())
    long = long[long['ticker'].isin(uni_set)]
    if verbose:
        print(f'    universe 필터 후: {len(long):,} rows ({long.ticker.nunique()} tickers)')
        print(f'    (시점별 S&P500 멤버십 strict 필터는 BL 단계 monthly_panel 에서 적용)')

    # 3. macro 의 vix 추가 (date 기준 broadcast)
    if verbose: print(f'  [3/4] VIX broadcast ...')
    macro = pd.read_csv(macro_path, parse_dates=['date'])
    if 'vix' not in macro.columns:
        raise ValueError(f"macro_daily.csv 에 'vix' 컬럼 없음: {list(macro.columns)}")
    macro_vix = macro[['date', 'vix']].copy()
    long = long.merge(macro_vix, on='date', how='left')
    long['vix'] = long['vix'].ffill().bfill().fillna(20.0)

    # 4. target_logrv = log(max(rolling(21).std(), eps)).shift(-horizon) 종목별
    #    eps guard: 거래정지 종목의 std=0 → log(0)=-inf 방지 (timeseries_lib.build_log_rv_target 와 정합)
    EPS_RV = 1e-12
    if verbose: print(f'  [4/4] target_logrv = log(rolling({horizon}).std()).shift(-{horizon}) ...')
    long = long.sort_values(['ticker', 'date']).reset_index(drop=True)
    long['target_logrv'] = long.groupby('ticker', group_keys=False)['log_ret'].transform(
        lambda lr: np.log(np.maximum(lr.rolling(horizon).std(), EPS_RV)).shift(-horizon)
    )

    if verbose:
        tg_finite = long[np.isfinite(long.target_logrv)]
        print()
        print(f'  최종: {len(long):,} rows, {long.ticker.nunique()} tickers')
        print(f'  date range: {long.date.min().date()} ~ {long.date.max().date()}')
        print(f'  target_logrv finite: {len(tg_finite):,} ({len(tg_finite)/len(long)*100:.1f}%)')
        print(f'  target finite max date: {tg_finite.date.max().date()}')

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        long.to_csv(out_path, index=False)
        if verbose:
            print(f'  ✅ 저장: {out_path}')

    return long


# =============================================================================
# 2. 종목별 3ch_vix 입력 빌드
# =============================================================================
def build_v4_inputs(panel_ticker: pd.DataFrame,
                    har_w: int = 5,
                    har_m: int = 22) -> dict:
    """단일 종목 panel → (3ch_vix) 입력.

    출력 채널 (input_size=4):
      [0] series  : rv_d (|log_ret|)
      [1] extra_0 : rv_w (rolling(5).std())
      [2] extra_1 : rv_m (rolling(22).std())
      [3] extra_2 : vix_log (log(VIX))

    target: target_logrv = log(rolling(21).std()).shift(-21)

    Returns
    -------
    dict
        {
            'features' : np.ndarray (n × 4) — 시계열_lib build_fold_inputs 입력용
            'target'   : np.ndarray (n,)
            'log_ret'  : pd.Series (HAR-RV 용, date index)
            'date'     : np.ndarray (datetime64)
            'input_size': 4,
        }
    """
    df = panel_ticker.copy().sort_values('date').reset_index(drop=True)

    log_ret = df['log_ret'].values
    rv_d = pd.Series(log_ret).abs().fillna(0.0).values
    log_ret_sq = pd.Series(log_ret) ** 2
    rv_w = log_ret_sq.rolling(har_w).mean().pow(0.5).fillna(0.0).values
    rv_m = log_ret_sq.rolling(har_m).mean().pow(0.5).fillna(0.0).values

    vix = df['vix'].ffill().bfill().fillna(20.0)
    vix_log = np.log(vix.clip(lower=1e-6)).values

    features = np.column_stack([rv_d, rv_w, rv_m, vix_log])
    target = df['target_logrv'].values

    return {
        'features'  : features,
        'target'    : target,
        'log_ret'   : pd.Series(log_ret, index=df['date'].values),
        'date'      : df['date'].values,
        'input_size': 4,
    }


# =============================================================================
# 3. 단일 fold LSTM v4 학습 + OOS 예측
# =============================================================================
def run_lstm_v4_fold(inputs: dict,
                     train_idx: np.ndarray,
                     test_idx: np.ndarray,
                     config: dict = V4_BEST_CONFIG,
                     device: str = 'auto',
                     verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """단일 fold LSTM 학습 + OOS 예측 (timeseries_lib 인터페이스).

    Returns
    -------
    y_pred : np.ndarray (len(test_idx),) — NaN 으로 invalid 위치 채움
    info   : dict
    """
    features = inputs['features']
    target = inputs['target']

    # NaN target 제거된 인덱스
    train_idx_valid = train_idx[~np.isnan(target[train_idx])]
    test_idx_valid = test_idx[~np.isnan(target[test_idx])]

    if len(train_idx_valid) < config['seq_len'] + 50:
        return np.full(len(test_idx), np.nan), {'status': 'insufficient_train'}
    if len(test_idx_valid) == 0:
        return np.full(len(test_idx), np.nan), {'status': 'no_valid_test'}

    # build fold inputs (X_tr, y_tr, X_te, y_te)
    fold = {'train_idx': train_idx_valid, 'test_idx': test_idx_valid}
    X_tr, y_tr, X_te, y_te = build_fold_inputs(
        features=features, target=target, fold=fold, seq_len=config['seq_len']
    )

    # validation split: 마지막 10%
    n_train = len(X_tr)
    n_val = max(int(n_train * 0.1), 5)
    if n_train <= n_val:
        return np.full(len(test_idx), np.nan), {'status': 'insufficient_after_seq'}
    X_train_only, X_val = X_tr[:-n_val], X_tr[-n_val:]
    y_train_only, y_val = y_tr[:-n_val], y_tr[-n_val:]

    # 학습 (timeseries_lib.train_one_fold 시그니처)
    model, info = train_one_fold(
        X_tr=X_train_only, y_tr=y_train_only,
        X_val=X_val, y_val=y_val,
        input_size=config['input_size'],
        hidden=config['hidden_size'],
        dropout=config['dropout'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        max_epochs=config['max_epochs'],
        batch_size=config['batch_size'],
        patience=config['early_stop_patience'],
        device=device,
    )

    # OOS 예측
    model.eval()
    device_t = next(model.parameters()).device
    with torch.no_grad():
        y_pred_test = model(X_te.to(device_t)).cpu().numpy()

    # test_idx 길이 맞춤 (NaN 위치 fill)
    y_pred_full = np.full(len(test_idx), np.nan)
    valid_mask = ~np.isnan(target[test_idx])
    valid_indices_in_test_idx = np.where(valid_mask)[0]

    # build_fold_inputs 가 seq_len 미만 idx 를 skip 하므로 매핑 careful
    # 실제 예측된 sample 수를 valid_mask 의 처음 seq_len-1 missing 만큼 줄여야 할 수 있음
    n_predicted = len(y_pred_test)
    n_valid = valid_mask.sum()
    if n_predicted == n_valid:
        y_pred_full[valid_mask] = y_pred_test
    elif n_predicted < n_valid:
        # build_fold_inputs 내부 seq_len 필터로 일부 누락 — 마지막 n_predicted 만 매핑
        valid_pos = valid_indices_in_test_idx[-n_predicted:]
        y_pred_full[valid_pos] = y_pred_test
    # else: 예측이 더 많으면 비정상, NaN 유지

    return y_pred_full, info


# =============================================================================
# 4. 종목 한 개의 walk-forward 전체 실행
# =============================================================================
def run_walkforward_for_ticker(ticker: str,
                                panel_ticker: pd.DataFrame,
                                config: dict = V4_BEST_CONFIG,
                                device: str = 'auto',
                                verbose: bool = False,
                                existing_folds: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """종목 1개의 walk-forward 전체.

    Returns
    -------
    pd.DataFrame
        long format: (date, ticker, fold, y_true, y_pred_lstm, y_pred_har)
        incremental 모드 시 기존 fold (start_k 미만) + 신규 fold merge
    """
    inputs = build_v4_inputs(panel_ticker, config['har_w'], config['har_m'])
    n = len(inputs['features'])

    # forward y_true buffer — 마지막 oos_len 일은 target NaN 이므로 fold 생성 시 제외
    oos_len_buffer = config.get('oos_len', 21)
    min_required = config['is_len'] + config['seq_len'] + config['oos_len']
    n_safe = max(n - oos_len_buffer, min_required)

    folds = walk_forward_folds(
        n_obs=n_safe,
        is_len=config['is_len'],
        purge=oos_len_buffer,
        embargo=config['embargo'],
        oos=config['oos_len'],     # ⭐ timeseries_lib 시그니처 = 'oos' (lstm_pipeline 의 'oos_len' → 매핑)
        step=config['step'],
    )

    target = inputs['target']
    log_ret = inputs['log_ret']
    dates = inputs['date']

    # incremental: 기존 max fold 부터 재학습
    start_k = 0
    if existing_folds is not None and len(existing_folds) > 0:
        start_k = int(existing_folds['fold'].max())
        if verbose:
            print(f'    [{ticker}] incremental: fold {start_k} 부터 학습')

    rows = []
    for k, fold in enumerate(folds):
        if k < start_k:
            continue

        train_idx = np.array(fold['train_idx'])
        test_idx = np.array(fold['test_idx'])

        # LSTM v4
        y_pred_lstm, _ = run_lstm_v4_fold(
            inputs, train_idx, test_idx, config, device, verbose=False
        )

        # HAR-RV
        try:
            y_pred_har, _ = fit_har_rv(
                log_ret=log_ret,
                train_idx=train_idx,
                test_idx=test_idx,
                horizon=config.get('oos_len', 21),
            )
        except Exception:
            y_pred_har = np.full(len(test_idx), np.nan)

        for i, idx in enumerate(test_idx):
            rows.append({
                'date'        : dates[idx],
                'ticker'      : ticker,
                'fold'        : k,
                'y_true'      : target[idx] if idx < len(target) else np.nan,
                'y_pred_lstm' : y_pred_lstm[i] if i < len(y_pred_lstm) else np.nan,
                'y_pred_har'  : y_pred_har[i] if i < len(y_pred_har) else np.nan,
            })

        if verbose and (k + 1) % 10 == 0:
            print(f'    [{ticker}] fold {k+1}/{len(folds)} 완료')

    EXPECTED_COLS = ['date', 'ticker', 'fold', 'y_true', 'y_pred_lstm', 'y_pred_har']
    new_df = pd.DataFrame(rows, columns=EXPECTED_COLS) if rows else pd.DataFrame(columns=EXPECTED_COLS)

    if existing_folds is not None and len(existing_folds) > 0:
        kept = existing_folds[existing_folds['fold'] < start_k].copy()
        if 'date' in kept.columns and not pd.api.types.is_datetime64_any_dtype(kept['date']):
            kept['date'] = pd.to_datetime(kept['date'])
        return pd.concat([kept, new_df], ignore_index=True)

    return new_df


# =============================================================================
# 4-B. 캐시 점검 + 자동 truncate (incremental 학습 진입 준비)
# =============================================================================
def diagnose_fold_csv(fold_path: str | Path,
                      lstm_finite_threshold: float = 0.95,
                      verbose: bool = True) -> dict:
    """fold_predictions_stockwise.csv 진단 — LSTM 결손 fold 자동 식별.

    Returns
    -------
    dict
        {
            'exists'              : bool,
            'shape'               : tuple,
            'fold_min'            : int,
            'fold_max'            : int,
            'first_deficient_fold': int | None  — LSTM finite < threshold 인 첫 fold,
            'lstm_max_date'       : pd.Timestamp | None,
            'recommendation'      : str — 'use_as_is' | 'truncate_at_X' | 'rebuild',
        }
    """
    fold_path = Path(fold_path)
    if not fold_path.exists():
        if verbose:
            print(f'  [diagnose] {fold_path.name} 부재 → recommendation: rebuild')
        return {
            'exists': False, 'shape': None, 'fold_min': None, 'fold_max': None,
            'first_deficient_fold': None, 'lstm_max_date': None,
            'recommendation': 'rebuild',
        }

    if verbose:
        print(f'  [diagnose] 로드: {fold_path}')
    fold = pd.read_csv(fold_path, parse_dates=['date'])

    fold_min = int(fold['fold'].min())
    fold_max = int(fold['fold'].max())

    # 첫 결손 fold
    first_def = None
    for f in sorted(fold['fold'].unique()):
        sub = fold[fold['fold'] == f]
        if len(sub) == 0: continue
        n_lstm = int(np.isfinite(sub['y_pred_lstm']).sum())
        if n_lstm / len(sub) < lstm_finite_threshold:
            first_def = int(f)
            break

    # LSTM finite max date
    lstm_max = fold[np.isfinite(fold['y_pred_lstm'])]['date'].max()

    # 권장 사항
    if first_def is None:
        rec = 'use_as_is'
    elif first_def <= fold_min + 1:
        rec = 'rebuild'  # 너무 일찍부터 결손 — 처음부터 재학습 권장
    else:
        rec = f'truncate_at_{first_def - 1}'

    if verbose:
        print(f'  [diagnose] shape: {fold.shape}, fold {fold_min}~{fold_max}')
        print(f'  [diagnose] LSTM 마지막 finite: {lstm_max.date() if pd.notna(lstm_max) else "N/A"}')
        if first_def is not None:
            print(f'  [diagnose] 첫 결손 fold: {first_def} (이전 정상 fold ≤ {first_def - 1})')
        print(f'  [diagnose] 권장: {rec}')

    return {
        'exists': True, 'shape': fold.shape, 'fold_min': fold_min, 'fold_max': fold_max,
        'first_deficient_fold': first_def, 'lstm_max_date': lstm_max,
        'recommendation': rec,
    }


def prepare_incremental_state(out_dir: str | Path,
                              fold_name: str = 'fold_predictions_stockwise.csv',
                              fallback_search_paths: Optional[list] = None,
                              auto_truncate: bool = True,
                              backup: bool = True,
                              verbose: bool = True) -> dict:
    """캐시 점검 + 자동 truncate 로 incremental 학습 진입 상태 준비.

    동작:
      1. out_dir 에 fold csv 가 있는지 확인 (사용자 지정 fallback 가능)
      2. fold csv 진단 — 첫 LSTM 결손 fold 식별
      3. auto_truncate=True 면 결손 fold 직전까지 truncate (자동 백업)
      4. 결과 dict 반환 (학습 시 incremental 인자 결정용)

    Parameters
    ----------
    out_dir : final_pt/data/03b_lstm/data 등
    fallback_search_paths : list of Path | None, default None
        fold csv 가 out_dir 에 없을 때 검색할 외부 경로 후보.
        None 시 외부 fallback 없음 — final_pt/phase3 의 .csv.gz 자동 해제 후 사용.
        명시 시 (예: [Path('...')]) 그 경로에서 1회 복사. final 단독 동작이
        기본 — 외부 폴더 의존성 0.
    auto_truncate : bool
        True 면 결손 fold 자동 truncate. False 면 진단만.
    backup : bool
        truncate 전 .bak_pre_retrain 백업

    Returns
    -------
    dict
        {
            'incremental_ok'       : bool — incremental 학습 진입 가능 여부,
            'fold_path'            : Path,
            'truncated_at'         : int | None — truncate 한 fold (없으면 None),
            'first_deficient_fold' : int | None,
            'recommendation'       : str,
        }
    """
    import shutil
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / fold_name

    if verbose:
        print('━' * 70)
        print(' prepare_incremental_state — 캐시 점검 + 자동 truncate')
        print('━' * 70)

    # 1. fold csv 부재 시 — final_pt/phase3 안의 .csv.gz 자동 압축 해제 시도
    # 정책 (final_pt 단독): 외부 폴더 fallback 없음. final_pt/phase3 안에서만 검색.
    #   - fold_path (.csv) 가 있으면 그대로 사용
    #   - .csv 부재 + .csv.gz 존재 → 자동 압축 해제
    #   - 모두 부재 → full 학습 (start_k=0, 모든 종목)
    #   - fallback_search_paths 명시 시 그 경로에서 1회 복사 (사용자 명시 옵션)
    if not fold_path.exists():
        # .csv.gz 자동 압축 해제
        gz_path = fold_path.with_suffix('.csv.gz')
        if gz_path.exists():
            import gzip
            if verbose:
                print(f'  [extract] {gz_path.name} 자동 압축 해제 → {fold_path.name}')
                print(f'           ({gz_path.stat().st_size/1e6:.1f} MB → ...)')
            with gzip.open(gz_path, 'rb') as fin, open(fold_path, 'wb') as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)
            if verbose:
                print(f'           ✅ 추출 완료 ({fold_path.stat().st_size/1e6:.1f} MB)')

        # 사용자 지정 fallback (default None — 외부 의존성 0)
        elif fallback_search_paths is not None:
            copied = False
            for src in fallback_search_paths:
                src = Path(src)
                if src.exists():
                    if verbose:
                        print(f'  [copy] 사용자 지정 fallback 발견 → final 로 1회 복사')
                        print(f'         from: {src}')
                        print(f'           to: {fold_path}')
                    shutil.copy2(src, fold_path)
                    copied = True
                    break
            if not copied:
                if verbose:
                    print(f'  [check] fold csv 부재 (final, fallback 모두) → full 학습 필요')
                return {
                    'incremental_ok': False, 'fold_path': fold_path,
                    'truncated_at': None, 'first_deficient_fold': None,
                    'recommendation': 'rebuild',
                }
        else:
            if verbose:
                print(f'  [check] fold csv 부재 → full 학습 필요 (모든 종목 fold 0 부터)')
            return {
                'incremental_ok': False, 'fold_path': fold_path,
                'truncated_at': None, 'first_deficient_fold': None,
                'recommendation': 'rebuild',
            }

    # 2. 진단
    diag = diagnose_fold_csv(fold_path, verbose=verbose)

    # 3. auto_truncate
    truncated_at = None
    if auto_truncate and diag['recommendation'].startswith('truncate_at_'):
        cutoff_fold = int(diag['recommendation'].split('_')[-1])
        if verbose:
            print(f'  [truncate] fold ≤ {cutoff_fold} 보존, 결손 fold 제거')

        if backup:
            bak = fold_path.with_suffix('.csv.bak_pre_retrain')
            if not bak.exists():
                shutil.copy2(fold_path, bak)
                if verbose:
                    print(f'  [backup] {bak.name} 생성')

        fold = pd.read_csv(fold_path, parse_dates=['date'])
        kept = fold[fold['fold'] <= cutoff_fold].copy()
        kept.to_csv(fold_path, index=False)
        truncated_at = cutoff_fold
        if verbose:
            print(f'  [truncate] {len(fold) - len(kept):,} rows 제거 → {len(kept):,} rows 보존')

    incremental_ok = diag['recommendation'] != 'rebuild'

    if verbose:
        print()
        print(f'  ✅ incremental_ok = {incremental_ok}, truncated_at = {truncated_at}')

    return {
        'incremental_ok': incremental_ok,
        'fold_path': fold_path,
        'truncated_at': truncated_at,
        'first_deficient_fold': diag['first_deficient_fold'],
        'recommendation': diag['recommendation'],
    }


# =============================================================================
# 5. Performance-Weighted Ensemble (Diebold-Pauly 1987)
# =============================================================================
def compute_performance_weights(fold_results: pd.DataFrame,
                                  initial_weights: Optional[dict] = None) -> pd.DataFrame:
    """Performance-Weighted ensemble 가중치 + 예측 계산.

    공식 (Diebold-Pauly 1987):
        w_v4[k]  = (1/RMSE_v4[k-1]) / (1/RMSE_v4[k-1] + 1/RMSE_har[k-1])
        w_har[k] = 1 - w_v4[k]
        y_pred_ensemble[k] = w_v4[k] · y_pred_lstm[k] + w_har[k] · y_pred_har[k]

    첫 fold (k=0): initial_weights (default 0.5/0.5) 사용.

    Returns
    -------
    pd.DataFrame
        + columns: w_v4, w_har, y_pred_ensemble
    """
    df = fold_results.sort_values(['ticker', 'fold', 'date']).copy()

    # non-finite 행 제거 (예측 폭주 종목)
    finite_mask = (
        np.isfinite(df['y_pred_lstm'])
        & np.isfinite(df['y_pred_har'])
    )
    n_dropped = (~finite_mask).sum()
    if n_dropped > 0:
        dropped_tickers = df.loc[~finite_mask, 'ticker'].unique()
        print(f'  [compute_performance_weights] {n_dropped} 행 제거 '
              f'(non-finite). 영향 종목: {len(dropped_tickers)}')
    df = df[finite_mask].reset_index(drop=True)

    if initial_weights is None:
        initial_weights = {'w_v4': 0.5, 'w_har': 0.5}

    out_rows = []
    grouped = df.groupby('ticker')

    try:
        from tqdm.auto import tqdm
        ticker_iter = tqdm(grouped, total=df['ticker'].nunique(),
                            desc='Performance ensemble', ncols=100)
    except ImportError:
        ticker_iter = grouped

    for ticker, ticker_df in ticker_iter:
        ticker_df = ticker_df.sort_values(['fold', 'date']).copy()
        folds_unique = sorted(ticker_df['fold'].unique())

        prev_rmse_v4 = None
        prev_rmse_har = None
        cur_w_v4 = initial_weights['w_v4']
        cur_w_har = initial_weights['w_har']

        for k in folds_unique:
            mask = ticker_df['fold'] == k
            fold_rows = ticker_df[mask].copy()

            if prev_rmse_v4 is not None and prev_rmse_har is not None:
                rmse_v4_safe = (
                    prev_rmse_v4 if (np.isfinite(prev_rmse_v4) and prev_rmse_v4 > 0)
                    else 1e6
                )
                rmse_har_safe = (
                    prev_rmse_har if (np.isfinite(prev_rmse_har) and prev_rmse_har > 0)
                    else 1e6
                )
                inv_v4 = 1.0 / max(rmse_v4_safe, 1e-6)
                inv_har = 1.0 / max(rmse_har_safe, 1e-6)
                denom = inv_v4 + inv_har
                if denom > 0:
                    cur_w_v4 = inv_v4 / denom
                    cur_w_har = 1 - cur_w_v4

            fold_rows['w_v4'] = cur_w_v4
            fold_rows['w_har'] = cur_w_har
            fold_rows['y_pred_ensemble'] = (
                cur_w_v4 * fold_rows['y_pred_lstm']
                + cur_w_har * fold_rows['y_pred_har']
            )

            valid = fold_rows.dropna(subset=['y_true', 'y_pred_lstm', 'y_pred_har'])
            if len(valid) > 0:
                err_v4 = (valid['y_pred_lstm'] - valid['y_true']).values
                err_har = (valid['y_pred_har'] - valid['y_true']).values
                prev_rmse_v4 = float(np.sqrt(np.mean(err_v4 ** 2)))
                prev_rmse_har = float(np.sqrt(np.mean(err_har ** 2)))

            out_rows.append(fold_rows)

    return pd.concat(out_rows, ignore_index=True)


# =============================================================================
# 6. 8-way 병렬 universe 학습
# =============================================================================
def _train_ticker_worker(args: tuple) -> tuple:
    """ProcessPoolExecutor worker — 단일 종목 walk-forward.

    Spawn 호환을 위해 모듈 최상위 정의.

    재현성: worker 시작 시 setup_seeds(42) 호출 — main process seed 미전파 대응.
    """
    if len(args) == 5:
        ticker, panel_t, config, device, gpu_id = args
        existing_folds_t = None
    elif len(args) == 6:
        ticker, panel_t, config, device, gpu_id, existing_folds_t = args
    else:
        return (None, None, f'Invalid args length: {len(args)}')

    # ⭐ worker 의 seed 고정 (multiprocessing spawn 후 새 process 라 main 의 seed 미전파)
    try:
        from timeseries_lib import setup_seeds
        setup_seeds(42)
    except Exception:
        pass   # seed 못 잡으면 stochastic 변동만 약간 — 학습은 진행

    try:
        if device == 'cuda' or (isinstance(device, str) and device.startswith('cuda')):
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
                effective_device = f'cuda:{gpu_id}'
            else:
                effective_device = 'cpu'
        else:
            effective_device = device

        df_t = run_walkforward_for_ticker(
            ticker=ticker,
            panel_ticker=panel_t,
            config=config,
            device=effective_device,
            verbose=False,
            existing_folds=existing_folds_t,
        )
        return (ticker, df_t, None)
    except Exception as e:
        return (ticker, None, f'{type(e).__name__}: {str(e)[:200]}')


def auto_n_workers(device: str = 'cuda', verbose: bool = True) -> int:
    """GPU 메모리에 따라 안전한 n_workers 자동 결정 (OOM 회피).

    기준: RTX 4090 24GB → 8 workers, 16GB → 4, 12GB → 4, 8GB → 2, CPU → 4
    """
    if device == 'cpu' or not torch.cuda.is_available():
        return 4   # CPU 는 코어 수 의존, 안전한 default
    try:
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem_gb >= 22:
            n = 8
        elif gpu_mem_gb >= 14:
            n = 4
        elif gpu_mem_gb >= 10:
            n = 4
        elif gpu_mem_gb >= 6:
            n = 2
        else:
            n = 1
        if verbose:
            print(f'  [auto_n_workers] GPU memory {gpu_mem_gb:.1f} GB → n_workers={n}')
        return n
    except Exception:
        return 4   # GPU 정보 못 얻으면 보수적


def run_ensemble_for_universe_parallel(
    panel: pd.DataFrame,
    universe_tickers: list,
    out_dir: Path,
    config: dict = V4_BEST_CONFIG,
    device: str = 'cuda',
    n_workers: Optional[int] = None,
    out_name: str = 'ensemble_predictions_stockwise.csv',
    fold_name: str = 'fold_predictions_stockwise.csv',
    incremental: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """전체 universe 종목 walk-forward 8-way 병렬 학습.

    예상 시간 (RTX 4090 24GB):
      - 615 종목 × 17년 학습 → ~8 시간 (full)
      - incremental (마지막 + 신규 fold 만) → ~30-90분

    Parameters
    ----------
    panel : pd.DataFrame
        long format daily_panel (date, ticker, log_ret, vix, target_logrv)
    universe_tickers : list
        학습 대상 종목 리스트
    out_dir : Path
        결과 저장 디렉토리
    incremental : bool
        True 시 fold_predictions_stockwise.csv 의 max fold 부터 재학습
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    fold_path = out_dir / fold_name

    # n_workers 자동 결정 (None 시 GPU 메모리 기반)
    if n_workers is None:
        n_workers = auto_n_workers(device, verbose=verbose)

    # incremental 모드 — 기존 fold predictions 로드
    # ⭐ Universe 정책 (사용자 결정, 2026-05-08): final 의 universe 기준
    #   - target = final panel 의 종목
    #   - fold csv 에 학습 history 가 있는 종목 → incremental (max fold 부터)
    #   - fold csv 에 없지만 panel 에 있는 종목 → full 학습 (start_k=0)
    #   - panel 에 없는 종목 (학습 불가) → drop
    # 즉, 과거 학습 시 다른 universe 였던 종목은 자동으로 final universe 에서 제외됨
    # (panel 에 없으므로 args_list 에 안 들어감)
    existing_fold_all = None
    if incremental:
        if not fold_path.exists():
            if verbose:
                print(f'  ⚠ incremental=True 이지만 {fold_path.name} 부재 → 전체 학습으로 fallback')
            incremental = False
        else:
            existing_fold_all = pd.read_csv(fold_path, parse_dates=['date'])
            if verbose:
                print(f'  ⚡ Incremental 모드: {len(existing_fold_all):,} rows, '
                      f'max fold {int(existing_fold_all["fold"].max())}')

            # final universe 기준 진단 (정보 표시만, 종목 list 는 그대로 유지)
            fold_tickers = set(existing_fold_all['ticker'].unique())
            panel_tickers = set(panel['ticker'].unique())
            target_set = set(universe_tickers)

            in_panel_target = panel_tickers & target_set
            with_history = fold_tickers & in_panel_target          # incremental 학습
            without_history = in_panel_target - fold_tickers       # full 학습 (fold csv 에 없음)
            fold_only = (fold_tickers & target_set) - panel_tickers  # 학습 불가 (panel 부재)

            if verbose:
                print(f'  ⚙ final universe 기준 학습 분류:')
                print(f'    target universe            : {len(target_set)}')
                print(f'    panel ∩ target             : {len(in_panel_target)}')
                print(f'    그 중 fold history 있음    : {len(with_history)}  → incremental (max fold 부터)')
                print(f'    그 중 fold history 없음    : {len(without_history)}  → full 학습 (start_k=0)')
                print(f'    fold 에만 있음 (panel 부재): {len(fold_only)}  → 학습 skip (자동)')
                if without_history and len(without_history) <= 30:
                    print(f'    full 학습 대상 종목: {sorted(without_history)[:30]}')

            # universe_tickers 는 panel ∩ target 으로 재설정 (panel 에 없는 종목 자동 제외)
            universe_tickers = sorted(in_panel_target)

    # ticker 별 args 준비
    args_list = []
    for ticker in universe_tickers:
        panel_t = panel[panel['ticker'] == ticker].copy()
        if len(panel_t) < config['is_len'] + config['seq_len'] + config['oos_len']:
            continue
        if incremental and existing_fold_all is not None:
            existing_folds_t = existing_fold_all[existing_fold_all['ticker'] == ticker].copy()
            args_list.append((ticker, panel_t, config, device, 0, existing_folds_t))
        else:
            args_list.append((ticker, panel_t, config, device, 0))

    if verbose:
        print(f'  학습 대상: {len(args_list)}/{len(universe_tickers)} 종목  '
              f'({n_workers}-way 병렬)')

    t_start = time.time()
    fold_results_all = []
    errors = []

    ctx = mp.get_context('spawn')
    try:
        from tqdm.auto import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        future_to_ticker = {
            executor.submit(_train_ticker_worker, args): args[0] for args in args_list
        }

        if use_tqdm and verbose:
            pbar = tqdm(total=len(args_list), desc='Training tickers', ncols=100)
        else:
            pbar = None

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result_ticker, df_t, err = future.result()
                if err is None and df_t is not None:
                    fold_results_all.append(df_t)
                else:
                    errors.append((result_ticker or ticker, err))
            except Exception as e:
                errors.append((ticker, f'{type(e).__name__}: {str(e)[:200]}'))
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    elapsed = (time.time() - t_start) / 60
    if verbose:
        print(f'  ⏱ 학습 완료: {elapsed:.1f}분 / 성공 {len(fold_results_all)} / 에러 {len(errors)}')
        if errors:
            print(f'  에러 종목 (최대 5):')
            for tk, err in errors[:5]:
                print(f'    - {tk}: {err}')

    if not fold_results_all:
        return pd.DataFrame()

    # fold predictions 통합 + 저장
    fold_df = pd.concat(fold_results_all, ignore_index=True)
    fold_df.to_csv(fold_path, index=False)
    if verbose:
        print(f'  ✅ fold_predictions: {fold_path.name} ({len(fold_df):,} rows)')

    # ensemble 빌드
    if verbose: print(f'  ensemble 가중치 계산 ...')
    ensemble_df = compute_performance_weights(fold_df)
    ensemble_df.to_csv(out_path, index=False)
    if verbose:
        print(f'  ✅ ensemble_predictions: {out_path.name} ({len(ensemble_df):,} rows)')

    return ensemble_df
