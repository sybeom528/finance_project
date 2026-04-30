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
    path = PHASE15_DIR / "scripts" / filename
    if not path.exists():
        raise FileNotFoundError(f"Phase 1.5 모듈 없음: {path}")
    alias = f"phase15_{name}"
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# Phase 1.5 의 핵심 함수·클래스 alias
_dataset = _load_phase15_module("dataset", "dataset.py")
_models = _load_phase15_module("models", "models.py")
_train = _load_phase15_module("train", "train.py")
_baselines = _load_phase15_module("baselines_volatility", "baselines_volatility.py")

build_fold_datasets = _dataset.build_fold_datasets
walk_forward_folds = _dataset.walk_forward_folds
LSTMRegressor = _models.LSTMRegressor
train_one_fold = _train.train_one_fold
fit_har_rv = _baselines.fit_har_rv


# =============================================================================
# Phase 1.5 v4 best 하이퍼파라미터
# =============================================================================
V4_BEST_CONFIG = {
    "input_channels": "3ch_vix",
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "loss_type": "mse",
    "huber_delta": 0.01,  # MSE 사용 시 무관
    "max_epochs": 50,
    "early_stop_patience": 10,
    "lr_patience": 5,
    "lr_factor": 0.5,
    "batch_size": 64,
    "is_len": 1250,
    "seq_len": 63,
    "embargo": 63,
    "oos_len": 21,
    "step": 21,
    "window": 21,  # forward target window
    "har_w": 5,
    "har_m": 22,
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
    df = panel_ticker.copy().sort_values("date").reset_index(drop=True)

    log_ret = df["log_ret"].values

    # rv_d, rv_w, rv_m (Phase 1.5 v4 정의)
    rv_d = pd.Series(log_ret).abs().fillna(0.0).values
    log_ret_sq = pd.Series(log_ret) ** 2
    rv_w = log_ret_sq.rolling(har_w).mean().pow(0.5).fillna(0.0).values
    rv_m = log_ret_sq.rolling(har_m).mean().pow(0.5).fillna(0.0).values

    # VIX log (pandas FutureWarning 회피: ffill/bfill 직접 호출)
    vix = df["vix"].ffill().bfill().fillna(20.0)
    vix_log = np.log(vix.clip(lower=1e-6)).values

    # target
    target = df["target_logrv"].values

    extra = np.column_stack([rv_w, rv_m, vix_log])

    return {
        "series": rv_d,
        "extra": extra,
        "target": target,
        "log_ret": pd.Series(log_ret, index=df["date"].values),
        "index": pd.to_datetime(df["date"].values),
        "input_size": 4,
        "date": df["date"].values,
    }


# =============================================================================
# 단일 fold LSTM v4 학습 + 예측
# =============================================================================
def run_lstm_v4_fold(
    inputs: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: dict = V4_BEST_CONFIG,
    device: str = "auto",
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """단일 fold LSTM v4 학습 + OOS 예측.

    Returns
    -------
    y_pred : np.ndarray (len(test_idx),)
    info : dict (학습 정보)
    """
    series = inputs["series"]
    extra = inputs["extra"]
    target = inputs["target"]

    # NaN target 제거된 인덱스만 사용
    train_idx_valid = train_idx[~np.isnan(target[train_idx])]
    test_idx_valid = test_idx[~np.isnan(target[test_idx])]

    if len(train_idx_valid) < config["seq_len"] + 50:
        return np.full(len(test_idx), np.nan), {"status": "insufficient_train"}
    if len(test_idx_valid) == 0:
        return np.full(len(test_idx), np.nan), {"status": "no_valid_test"}

    # Dataset 빌드
    train_ds, test_ds, scaler = build_fold_datasets(
        series=series,
        train_idx=train_idx_valid,
        test_idx=test_idx_valid,
        seq_len=config["seq_len"],
        extra_features=extra,
        target_series=target,
    )

    if len(train_ds) == 0 or len(test_ds) == 0:
        return np.full(len(test_idx), np.nan), {"status": "empty_dataset"}

    # Train / Val 분할 (마지막 10% val)
    n_train = len(train_ds)
    n_val = max(int(n_train * 0.1), 5)
    n_train_only = n_train - n_val
    train_only_ds = torch.utils.data.Subset(train_ds, list(range(n_train_only)))
    val_ds = torch.utils.data.Subset(train_ds, list(range(n_train_only, n_train)))

    train_loader = DataLoader(
        train_only_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    # 모델
    model = LSTMRegressor(
        input_size=(
            config["input_size"] if "input_size" in config else inputs["input_size"]
        ),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        batch_first=True,
    )

    # 학습
    info = train_one_fold(
        model,
        train_loader,
        val_loader,
        max_epochs=config["max_epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        huber_delta=config["huber_delta"],
        loss_type=config["loss_type"],
        early_stop_patience=config["early_stop_patience"],
        lr_patience=config["lr_patience"],
        lr_factor=config["lr_factor"],
        device=device,
        verbose=verbose,
        log_every=10,
    )

    # OOS 예측
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
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
    device: str = "auto",
    verbose: bool = False,
) -> pd.DataFrame:
    """종목 1개의 walk-forward 전체 실행.

    Returns
    -------
    pd.DataFrame
        long format: (date, ticker, fold, y_true, y_pred_lstm, y_pred_har)
    """
    inputs = build_v4_inputs(panel_ticker, config["har_w"], config["har_m"])
    n = len(inputs["series"])

    folds = walk_forward_folds(
        n=n,
        is_len=config["is_len"],
        purge=config["window"],
        emb=config["embargo"],
        oos_len=config["oos_len"],
        step=config["step"],
    )

    target = inputs["target"]
    log_ret = inputs["log_ret"]
    dates = inputs["date"]

    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        # LSTM v4
        y_pred_lstm, info = run_lstm_v4_fold(
            inputs, train_idx, test_idx, config, device, verbose=False
        )

        # HAR-RV
        try:
            y_pred_har, _ = fit_har_rv(
                log_ret=log_ret,
                train_idx=train_idx,
                test_idx=test_idx,
                horizon=config["window"],
            )
        except Exception:
            y_pred_har = np.full(len(test_idx), np.nan)

        for i, idx in enumerate(test_idx):
            rows.append(
                {
                    "date": dates[idx],
                    "ticker": ticker,
                    "fold": k,
                    "y_true": target[idx] if idx < len(target) else np.nan,
                    "y_pred_lstm": y_pred_lstm[i] if i < len(y_pred_lstm) else np.nan,
                    "y_pred_har": y_pred_har[i] if i < len(y_pred_har) else np.nan,
                }
            )

        if verbose and (k + 1) % 10 == 0:
            print(f"    [{ticker}] fold {k+1}/{len(folds)} 완료")

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
    df = fold_results.sort_values(["ticker", "fold", "date"]).copy()

    # ⭐ non-finite 행 제거 (target 또는 예측 폭주 종목 — 폐상장 등 stale price)
    n_before = len(df)
    finite_mask = (
        np.isfinite(df["y_true"])
        & np.isfinite(df["y_pred_lstm"])
        & np.isfinite(df["y_pred_har"])
    )
    n_dropped = (~finite_mask).sum()
    if n_dropped > 0:
        dropped_tickers = df.loc[~finite_mask, "ticker"].unique()
        print(
            f"  [compute_performance_weights] {n_dropped}/{n_before} 행 제거 "
            f"(non-finite y_true/pred). 영향 종목: {sorted(dropped_tickers)}"
        )
    df = df[finite_mask].reset_index(drop=True)

    if initial_weights is None:
        initial_weights = {"w_v4": 0.5, "w_har": 0.5}

    out_rows = []
    grouped = df.groupby("ticker")
    try:
        from tqdm.auto import tqdm

        ticker_iter = tqdm(
            grouped,
            total=df["ticker"].nunique(),
            desc="Performance ensemble",
            ncols=100,
        )
    except ImportError:
        ticker_iter = grouped

    for ticker, ticker_df in ticker_iter:
        ticker_df = ticker_df.sort_values(["fold", "date"]).copy()
        folds_unique = sorted(ticker_df["fold"].unique())

        # fold 별 RMSE 계산 (이전 fold 의 OOS RMSE 사용)
        prev_rmse_v4 = None
        prev_rmse_har = None
        cur_w_v4 = initial_weights["w_v4"]
        cur_w_har = initial_weights["w_har"]

        for k in folds_unique:
            mask = ticker_df["fold"] == k
            fold_rows = ticker_df[mask].copy()

            # 가중치 결정 (이전 fold 결과 기반)
            if prev_rmse_v4 is not None and prev_rmse_har is not None:
                # ⭐ inf/NaN 방어 — fold 의 LSTM/HAR 예측이 폭주한 경우
                rmse_v4_safe = (
                    prev_rmse_v4
                    if (np.isfinite(prev_rmse_v4) and prev_rmse_v4 > 0)
                    else 1e6
                )
                rmse_har_safe = (
                    prev_rmse_har
                    if (np.isfinite(prev_rmse_har) and prev_rmse_har > 0)
                    else 1e6
                )
                inv_v4 = 1.0 / max(rmse_v4_safe, 1e-6)
                inv_har = 1.0 / max(rmse_har_safe, 1e-6)
                denom = inv_v4 + inv_har
                if denom > 0:
                    cur_w_v4 = inv_v4 / denom
                    cur_w_har = 1 - cur_w_v4
                # else: 직전 가중치 유지

            fold_rows["w_v4"] = cur_w_v4
            fold_rows["w_har"] = cur_w_har
            fold_rows["y_pred_ensemble"] = (
                cur_w_v4 * fold_rows["y_pred_lstm"]
                + cur_w_har * fold_rows["y_pred_har"]
            )

            # 본 fold OOS RMSE (다음 fold 가중치용)
            valid = fold_rows.dropna(subset=["y_true", "y_pred_lstm", "y_pred_har"])
            if len(valid) > 0:
                err_v4 = (valid["y_pred_lstm"] - valid["y_true"]).values
                err_har = (valid["y_pred_har"] - valid["y_true"]).values
                prev_rmse_v4 = float(np.sqrt(np.mean(err_v4**2)))
                prev_rmse_har = float(np.sqrt(np.mean(err_har**2)))

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
    device: str = "auto",
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

    panel = pd.read_csv(panel_csv, parse_dates=["date"])
    universe_df = pd.read_csv(universe_csv, parse_dates=["cutoff_date"])

    # 종목 선정
    all_tickers = sorted(universe_df["ticker"].unique().tolist())
    if tickers_subset is not None:
        tickers = [t for t in all_tickers if t in tickers_subset]
        print(f"  [ensemble] subset 모드: {len(tickers)}/{len(all_tickers)} 종목 학습")
    else:
        tickers = all_tickers
        print(f"  [ensemble] 전체 모드: {len(tickers)} 종목 학습")

    # 종목별 walk-forward (시간 병목)
    fold_results_all = []
    t_start = time.time()
    for i, ticker in enumerate(tickers):
        t_ticker = time.time()
        panel_t = panel[panel["ticker"] == ticker].copy()
        if len(panel_t) < config["is_len"] + config["seq_len"] + config["oos_len"]:
            print(f"    [{ticker}] 데이터 부족 → skip")
            continue
        try:
            df_t = run_walkforward_for_ticker(
                ticker, panel_t, config, device, verbose=False
            )
            fold_results_all.append(df_t)
            elapsed = time.time() - t_ticker
            n_folds = df_t["fold"].nunique()
            print(
                f"    [{i+1}/{len(tickers)}] {ticker}: {n_folds} fold ({elapsed:.0f}s)"
            )
        except Exception as e:
            print(f"    [{ticker}] 학습 실패: {e}")

        # 중간 저장 (안전망)
        if (i + 1) % 10 == 0:
            partial = pd.concat(fold_results_all, ignore_index=True)
            partial.to_csv(out_dir / "ensemble_predictions_partial.csv", index=False)

    fold_results = pd.concat(fold_results_all, ignore_index=True)
    fold_results.to_csv(out_dir / "fold_predictions_lstm_har.csv", index=False)
    print(f"  [ensemble] LSTM + HAR fold predictions 저장")

    # Performance ensemble 가중치 계산 (결정 4 — 신규 reset 자동 처리)
    print(f"  [ensemble] Performance-Weighted 가중치 계산 중...")
    ensemble_df = compute_performance_weights(fold_results)
    ensemble_df.to_csv(out_dir / "ensemble_predictions_top50.csv", index=False)

    elapsed_total = time.time() - t_start
    print(f"  [ensemble] 전체 완료: {len(tickers)} 종목, 총 {elapsed_total/60:.1f}분")
    print(f'  [ensemble] 저장: {out_dir / "ensemble_predictions_top50.csv"}')

    return ensemble_df


# =============================================================================
# Phase 3 — 병렬 학습 (RTX 4090 24GB 활용, 8-way 병렬)
# =============================================================================
def _train_ticker_worker(args: tuple) -> tuple:
    """ProcessPoolExecutor 의 worker 함수.

    하나의 종목에 대한 walk-forward 학습 수행.
    Spawn 방식 호환을 위해 모듈 최상위에 정의.

    Parameters
    ----------
    args : tuple
        (ticker, panel_t, config, device, gpu_id) 의 5-tuple.

    Returns
    -------
    tuple
        (ticker, fold_results_df, error_msg) — error_msg None 시 성공.
    """
    ticker, panel_t, config, device, gpu_id = args
    try:
        # ⭐ GPU device 명시적 설정 (multiprocessing 안전)
        if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                # 각 worker 시작 시 cache 정리 (메모리 누수 방지)
                torch.cuda.empty_cache()
                effective_device = f"cuda:{gpu_id}"
            else:
                effective_device = "cpu"
        else:
            effective_device = device

        df_t = run_walkforward_for_ticker(
            ticker=ticker,
            panel_ticker=panel_t,
            config=config,
            device=effective_device,
            verbose=False,
        )
        return (ticker, df_t, None)
    except Exception as e:
        # ⭐ traceback 단축 (메모리 절약)
        return (ticker, None, f"{type(e).__name__}: {str(e)[:200]}")


def run_ensemble_for_universe_parallel(
    panel_csv: Path,
    universe_csv: Path,
    out_dir: Path,
    config: dict = V4_BEST_CONFIG,
    device: str = "cuda",
    n_workers: int = 8,
    tickers_subset: Optional[list] = None,
    out_name: str = "ensemble_predictions_stockwise.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Phase 3 — 종목별 walk-forward 8-way 병렬 학습.

    Phase 2 의 run_ensemble_for_universe 와 동일 결과지만, 8-way 병렬로 학습 시간 단축.

    핵심 변경사항
    -----------
    - 종목별 학습 자체는 동일 (LSTMRegressor, HAR-RV, Performance ensemble)
    - 학습 단위 (종목당 walk-forward 전체) 를 ProcessPoolExecutor 로 병렬화
    - spawn context 명시 → Windows + Jupyter 호환

    학습 시간 추정 (RTX 4090 24GB)
    ------------------------------
    - 순차 (현재 Phase 2): 74 종목 × 6 년 = 약 2.3 시간
    - 8-way 병렬 (본 함수): 약 17 분 (8 배 단축)
    - 624 종목 × 17 년: 8-way 병렬 약 8 시간

    Parameters
    ----------
    panel_csv : Path
        daily_panel.csv 경로.
    universe_csv : Path
        universe history csv 경로.
    out_dir : Path
        결과 저장 디렉토리.
    config : dict
        V4_BEST_CONFIG (기본).
    device : str
        'cuda' / 'cpu' / 'auto'.
    n_workers : int, default 8
        동시 학습 worker 수. RTX 4090 24GB 기준 8-12 권장.
    tickers_subset : list of str, optional
        None 시 전체 universe. list 제공 시 subset 만 학습 (검증용).
    out_name : str, default 'ensemble_predictions_stockwise.csv'
        결과 csv 파일명.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        ensemble predictions (long format).
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_csv, parse_dates=["date"])
    universe_df = pd.read_csv(universe_csv, parse_dates=["cutoff_date"])

    # 종목 선정
    all_tickers = sorted(universe_df["ticker"].unique().tolist())
    if tickers_subset is not None:
        tickers = [t for t in all_tickers if t in tickers_subset]
        if verbose:
            print(f"  [parallel] subset 모드: {len(tickers)}/{len(all_tickers)} 종목")
    else:
        tickers = all_tickers
        if verbose:
            print(f"  [parallel] 전체 모드: {len(tickers)} 종목")

    # 종목별 args 준비 (gpu_id 추가 — 단일 GPU 환경에서 모두 0)
    args_list = []
    for i, ticker in enumerate(tickers):
        panel_t = panel[panel["ticker"] == ticker].copy()
        if len(panel_t) < config["is_len"] + config["seq_len"] + config["oos_len"]:
            if verbose:
                print(f"    [{ticker}] 데이터 부족 ({len(panel_t)}) → skip")
            continue
        # ⭐ gpu_id 0 (단일 GPU 환경). multi-GPU 환경 시 i % n_gpus 등으로 분배
        args_list.append((ticker, panel_t, config, device, 0))

    if verbose:
        print(
            f"  [parallel] {len(args_list)} 종목 × {n_workers}-way 병렬 학습 시작",
            flush=True,
        )
        print(f"  [parallel] device={device}, GPU 활용 80-90% 예상", flush=True)

    t_start = time.time()
    fold_results_all = []
    errors = []

    # ⭐ ProcessPoolExecutor + spawn context (Jupyter + Windows 호환)
    # submit + as_completed 로 완료 순서대로 결과 받기 (진정한 imap_unordered 효과)
    ctx = mp.get_context("spawn")
    from concurrent.futures import as_completed

    # ⭐ tqdm.auto: notebook 에선 진행 막대, CMD 에선 텍스트 진행률
    try:
        from tqdm.auto import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        future_to_ticker = {
            executor.submit(_train_ticker_worker, args): args[0] for args in args_list
        }

        # tqdm 으로 진행률 표시 (notebook + CMD 모두 작동)
        if use_tqdm and verbose:
            pbar = tqdm(
                total=len(args_list),
                desc="Training tickers",
                ncols=100,
                mininterval=0.5,
            )
        else:
            pbar = None

        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker_submitted = future_to_ticker[future]
            try:
                ticker, df_t, error = future.result()
            except Exception as e:
                errors.append((ticker_submitted, f"Future error: {e}"))
                if verbose:
                    print(
                        f"    [{i+1}/{len(args_list)}] {ticker_submitted} future 에러",
                        flush=True,
                    )
                if pbar is not None:
                    pbar.update(1)
                continue

            elapsed = time.time() - t_start
            avg_per_ticker = elapsed / (i + 1)
            eta = avg_per_ticker * (len(args_list) - i - 1)

            if error is not None:
                errors.append((ticker, error))
                if verbose:
                    print(
                        f"    [{i+1}/{len(args_list)}] {ticker} 실패: {error[:80]} "
                        f"(elapsed={elapsed/60:.1f}분, ETA={eta/60:.0f}분)",
                        flush=True,
                    )
            else:
                fold_results_all.append(df_t)
                n_folds = df_t["fold"].nunique() if df_t is not None else 0
                if verbose:
                    # CMD 에서도 잘 보이도록 한 줄 출력 + flush
                    print(
                        f"    [{i+1}/{len(args_list)}] {ticker}: {n_folds} fold "
                        f"(elapsed={elapsed/60:.1f}분, ETA={eta/60:.0f}분)",
                        flush=True,
                    )

            if pbar is not None:
                pbar.set_postfix(
                    {
                        "last": ticker[:6],
                        "elapsed": f"{elapsed/60:.1f}분",
                        "ETA": f"{eta/60:.0f}분",
                    }
                )
                pbar.update(1)

            # 중간 저장 (매 20 종목)
            if (i + 1) % 20 == 0 and len(fold_results_all) > 0:
                partial_df = pd.concat(fold_results_all, ignore_index=True)
                partial_df.to_csv(
                    out_dir / "ensemble_predictions_partial.csv", index=False
                )
                if verbose:
                    print(
                        f"    [중간 저장] {len(fold_results_all)} 종목 "
                        f"→ ensemble_predictions_partial.csv",
                        flush=True,
                    )

        if pbar is not None:
            pbar.close()

    # 결과 합치기
    if not fold_results_all:
        raise RuntimeError("모든 종목 학습 실패")

    fold_results = pd.concat(fold_results_all, ignore_index=True)
    fold_results.to_csv(out_dir / "fold_predictions_stockwise.csv", index=False)
    if verbose:
        print(f"  [parallel] LSTM + HAR fold predictions 저장", flush=True)

    # Performance ensemble 가중치 계산
    if verbose:
        print(f"  [parallel] Performance-Weighted 가중치 계산 중...", flush=True)
    ensemble_df = compute_performance_weights(fold_results)
    ensemble_df.to_csv(out_dir / out_name, index=False)

    elapsed_total = time.time() - t_start
    if verbose:
        print(
            f"  [parallel] 전체 완료: {len(args_list)} 종목, 총 {elapsed_total/60:.1f}분",
            flush=True,
        )
        print(f"  [parallel] 저장: {out_dir / out_name}", flush=True)
        if errors:
            print(f"  [parallel] 학습 실패 종목: {len(errors)}", flush=True)
            for t, e in errors[:5]:
                print(f"    {t}: {e[:100]}", flush=True)

    return ensemble_df


# =============================================================================
# Phase 3 — Cross-Sectional LSTM 학습
# =============================================================================
def build_cs_inputs(
    panel: pd.DataFrame,
    tickers: list,
    har_w: int = 5,
    har_m: int = 22,
    align_to_common_dates: bool = True,
) -> dict:
    """Cross-Sectional 학습용 input 빌드.

    Phase 1.5 v8 의 build_v4_inputs (종목별) 의 cross-sectional 확장.

    Parameters
    ----------
    panel : pd.DataFrame
        daily_panel.csv (long format: date, ticker, log_ret, vix, target_logrv).
    tickers : list of str
        학습 대상 종목 list.
    align_to_common_dates : bool, default True
        ⭐ C4+Mj5 수정 (2026-04-29):
        True  → 모든 종목을 panel 전체 날짜 축에 정렬 (IPO 이전/이후 = NaN).
                 동일 position = 동일 market date 보장.
                 _build_cs_dataset_for_fold 의 NaN 필터링이 자동 처리.
        False → 종목별 자체 날짜 축 유지 (구 동작 — 디버깅/역호환용).

    Returns
    -------
    dict
        {
            'series': dict {ticker: rv_d array (T,)},
            'extra': dict {ticker: (T, 3) — rv_w, rv_m, vix_log},
            'target': dict {ticker: target array (T,)},
            'date': dict {ticker: pd.DatetimeIndex},  ← align=True 시 모든 종목 동일
            'ticker_to_id': dict {ticker: int} — embedding 용,
            'n_tickers': int,
            'input_size': 4,
            'common_dates': pd.DatetimeIndex | None,  ← align=True 시 공통 날짜 축
        }

    Notes
    -----
    - align_to_common_dates=True 시 IPO 이전 구간은 NaN 으로 채워짐.
      _build_cs_dataset_for_fold 에서 NaN seq window 는 자동 제외됨.
    - run_ensemble_cross_sectional 은 common_dates 를 walk_forward_folds 의
      n 인자로 사용 → 모든 종목 동일 fold boundary 보장.
    """
    # ─── 공통 날짜 축 (C4+Mj5 수정) ───
    if align_to_common_dates:
        common_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    else:
        common_dates = None

    inputs = {
        "series": {},
        "extra": {},
        "target": {},
        "date": {},
        "ticker_to_id": {t: i for i, t in enumerate(sorted(tickers))},
        "n_tickers": len(tickers),
        "input_size": 4,
        "common_dates": common_dates,
    }

    for ticker in tickers:
        # 종목 실제 데이터 (정렬 전)
        panel_t_raw = panel[panel["ticker"] == ticker].copy().sort_values("date")
        n_actual = len(panel_t_raw)

        if n_actual < 30:
            # 실제 데이터가 30일 미만인 종목 제외
            continue

        if common_dates is not None:
            # ⭐ 공통 날짜 축에 정렬 — IPO 이전 / 상장폐지 이후 = NaN
            panel_t = (
                panel_t_raw.set_index("date")
                .reindex(common_dates)
                .reset_index()
                .rename(columns={"index": "date"})
            )
            # 'date' 컬럼이 이미 common_dates 로 세팅되어 있음
            # (set_index('date') 후 reindex → reset_index 시 date 컬럼 복원)
            if "date" not in panel_t.columns:
                panel_t = panel_t.rename(columns={panel_t.columns[0]: "date"})
        else:
            panel_t = panel_t_raw.reset_index(drop=True)

        log_ret = panel_t["log_ret"].values.astype(float)

        # ⭐ NaN 보존 — IPO 이전 구간은 NaN 으로 유지
        #    _build_cs_dataset_for_fold 의 seq NaN 검증이 자동 처리
        rv_d = np.abs(log_ret)  # (T,)  NaN preserved
        log_ret_sq = pd.Series(log_ret) ** 2
        rv_w = log_ret_sq.rolling(har_w).mean().pow(0.5).values  # NaN preserved
        rv_m = log_ret_sq.rolling(har_m).mean().pow(0.5).values  # NaN preserved

        # VIX: NaN 은 ffill/bfill/20.0 으로 보간 (VIX 는 항상 있어야 함)
        vix_col = (
            panel_t["vix"]
            if "vix" in panel_t.columns
            else pd.Series([20.0] * len(panel_t), index=panel_t.index)
        )
        vix = vix_col.ffill().bfill().fillna(20.0)
        vix_log = np.log(vix.clip(lower=1e-6)).values

        target_col = (
            panel_t["target_logrv"].values.astype(float)
            if "target_logrv" in panel_t.columns
            else np.full(len(panel_t), np.nan)
        )
        extra = np.column_stack([rv_w, rv_m, vix_log])  # (T, 3)

        inputs["series"][ticker] = rv_d
        inputs["extra"][ticker] = extra
        inputs["target"][ticker] = target_col
        inputs["date"][ticker] = (
            common_dates  # 모든 종목 동일 날짜 축
            if common_dates is not None
            else pd.to_datetime(panel_t["date"].values)
        )

    return inputs


class CrossSectionalDataset(torch.utils.data.Dataset):
    """Cross-Sectional 학습용 PyTorch Dataset.

    각 sample = (한 종목의 한 시점의 sequence, ticker_id, target).

    Attributes
    ----------
    samples : list of (ticker, idx) tuples
        valid (종목, 시점) 조합.
    inputs : dict (build_cs_inputs 결과)
    seq_len : int
    """

    def __init__(self, samples_list, inputs_dict, seq_len, scalers=None):
        """
        Parameters
        ----------
        samples_list : list of (ticker, position) tuples
            position = data 의 인덱스 (시퀀스 마지막 위치).
        inputs_dict : dict
            build_cs_inputs 결과.
        seq_len : int
        scalers : dict {ticker: StandardScaler}, optional
            종목별 scaler. None 시 unit-scale.
        """
        self.samples = samples_list
        self.inputs = inputs_dict
        self.seq_len = seq_len
        self.scalers = scalers
        self.ticker_to_id = inputs_dict["ticker_to_id"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, pos = self.samples[idx]
        series = self.inputs["series"][ticker]  # (T,)
        extra = self.inputs["extra"][ticker]  # (T, 3)
        target = self.inputs["target"][ticker]  # (T,)

        # Sequence: [pos - seq_len, pos)
        x_series = series[pos - self.seq_len : pos]
        x_extra = extra[pos - self.seq_len : pos]
        x = np.column_stack([x_series, x_extra])  # (seq_len, 4)

        # Scale (종목별 scaler, 없으면 unit)
        if self.scalers is not None and ticker in self.scalers:
            x = self.scalers[ticker].transform(x)

        y = target[pos]
        ticker_id = self.ticker_to_id[ticker]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(ticker_id, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


def _build_cs_dataset_for_fold(
    cs_inputs: dict,
    train_idx_per_ticker: dict,
    test_idx_per_ticker: dict,
    seq_len: int,
):
    """Fold 의 (train, test) Cross-Sectional Dataset 생성.

    각 종목의 train/test 인덱스를 받아서:
    - StandardScaler 종목별 fit (train 만)
    - train sample, test sample 추출
    - Dataset 생성

    Returns
    -------
    train_ds, test_ds : CrossSectionalDataset
    scalers : dict {ticker: StandardScaler}
    """
    from sklearn.preprocessing import StandardScaler

    scalers = {}
    train_samples = []
    test_samples = []

    for ticker in cs_inputs["series"].keys():
        if ticker not in train_idx_per_ticker:
            continue

        train_idx = train_idx_per_ticker[ticker]
        test_idx = test_idx_per_ticker.get(ticker, np.array([]))

        # NaN target 제거
        target = cs_inputs["target"][ticker]
        series = cs_inputs["series"][ticker]
        extra = cs_inputs["extra"][ticker]

        train_idx_valid = train_idx[~np.isnan(target[train_idx])]
        test_idx_valid = (
            test_idx[~np.isnan(target[test_idx])] if len(test_idx) > 0 else test_idx
        )

        if len(train_idx_valid) < seq_len + 30:
            continue

        # StandardScaler fit (train 만)
        train_data = np.column_stack([series[train_idx_valid], extra[train_idx_valid]])
        scaler = StandardScaler()
        scaler.fit(train_data)
        scalers[ticker] = scaler

        # ⭐ Sample 추출: 각 valid 위치에서 직전 seq_len 의 NaN 검증
        # (train_idx_valid 가 NaN target 제거 후 불연속일 수 있으므로, 위치별 검증)
        n_total = len(series)
        for pos in train_idx_valid:
            pos_int = int(pos)
            if pos_int < seq_len:
                continue
            # seq window 의 NaN 체크
            seq_x = series[pos_int - seq_len : pos_int]
            seq_extra = extra[pos_int - seq_len : pos_int]
            if np.any(np.isnan(seq_x)) or np.any(np.isnan(seq_extra)):
                continue
            train_samples.append((ticker, pos_int))

        # test: 각 test_idx 위치에서 직전 seq_len 의 NaN 검증
        for pos in test_idx_valid:
            pos_int = int(pos)
            if pos_int < seq_len:
                continue
            seq_x = series[pos_int - seq_len : pos_int]
            seq_extra = extra[pos_int - seq_len : pos_int]
            if np.any(np.isnan(seq_x)) or np.any(np.isnan(seq_extra)):
                continue
            test_samples.append((ticker, pos_int))

    train_ds = CrossSectionalDataset(train_samples, cs_inputs, seq_len, scalers)
    test_ds = CrossSectionalDataset(test_samples, cs_inputs, seq_len, scalers)

    return train_ds, test_ds, scalers


def run_ensemble_cross_sectional(
    panel_csv: Path,
    universe_csv: Path,
    out_dir: Path,
    config: Optional[dict] = None,
    device: str = "cuda",
    tickers_subset: Optional[list] = None,
    out_name: str = "ensemble_predictions_crosssec.csv",
    use_har: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Phase 3 — Cross-Sectional LSTM walk-forward 학습 + HAR ensemble.

    종목별 학습 (Phase 1.5 v8) → Cross-sectional 학습 (모든 종목 공유 LSTM)

    핵심 차이
    --------
    - 단일 LSTM 모델, 모든 종목 공유 (ticker embedding 으로 차별화)
    - 학습 sample: 441 × 종목수/fold (74 배 ↑)
    - HAR 베이스라인은 종목별 (Phase 1.5 와 동일)
    - Performance ensemble 결합 (Phase 1.5 의 Diebold-Pauly 일관)

    학술 근거
    ---------
    - Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via ML"
    - Chen, Pelger, Zhu (2024) "Deep Learning in Asset Pricing"
    - Diebold & Pauly (1987) Performance-Weighted ensemble

    Parameters
    ----------
    panel_csv : Path
    universe_csv : Path
    out_dir : Path
    config : dict, default CS_V4_BEST_CONFIG
    device : str
    tickers_subset : list, optional
    out_name : str
    use_har : bool, default True
        HAR 베이스라인 결합 여부. False 시 LSTM CS only.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        long format: date, ticker, fold, y_true, y_pred_lstm_cs, y_pred_har,
        w_lstm_cs, w_har, y_pred_ensemble.
    """
    from .models_cs import CrossSectionalLSTMRegressor, CS_V4_BEST_CONFIG
    from torch.utils.data import DataLoader

    if config is None:
        config = CS_V4_BEST_CONFIG.copy()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── 데이터 로드 ───
    panel = pd.read_csv(panel_csv, parse_dates=["date"])
    universe_df = pd.read_csv(universe_csv, parse_dates=["cutoff_date"])

    all_tickers = sorted(universe_df["ticker"].unique().tolist())
    if tickers_subset is not None:
        tickers = [t for t in all_tickers if t in tickers_subset]
    else:
        tickers = all_tickers

    if verbose:
        print(
            f"  [cross-sec] {len(tickers)} 종목 cross-sectional 학습 시작", flush=True
        )
        print(
            f'  [cross-sec] device={device}, hidden={config["hidden_size"]}, layers={config["num_layers"]}',
            flush=True,
        )
        print(f"  [cross-sec] HAR 결합: {use_har}", flush=True)

    # ─── Cross-sectional inputs 빌드 ───
    cs_inputs = build_cs_inputs(panel, tickers, config["har_w"], config["har_m"])

    if verbose:
        print(
            f'  [cross-sec] inputs: {len(cs_inputs["series"])} 종목 valid', flush=True
        )

    # ─── 공통 walk-forward fold 생성 ───
    # C4+Mj5 수정: build_cs_inputs 에서 align_to_common_dates=True (default) 를 사용하면
    # 모든 종목이 같은 common_dates 길이를 가짐 → common_length 일관성 보장
    ticker_lengths = {t: len(cs_inputs["series"][t]) for t in cs_inputs["series"]}

    if cs_inputs.get("common_dates") is not None:
        # ⭐ align_to_common_dates=True: 모든 종목 길이가 동일해야 함
        common_length = len(cs_inputs["common_dates"])
        n_unique_lengths = len(set(ticker_lengths.values()))
        if n_unique_lengths > 1:
            # 미정렬 종목이 섞인 경우 경고 (오류는 아님 — min 으로 fallback)
            if verbose:
                print(
                    f"  [cross-sec] ⚠️ 종목별 length 불일치 ({n_unique_lengths} 종류) → min 사용"
                )
            common_length = min(ticker_lengths.values())
    else:
        # align_to_common_dates=False: 구 동작 유지 (가장 짧은 종목 기준)
        common_length = min(ticker_lengths.values())

    if verbose:
        print(
            f"  [cross-sec] 종목별 length: min={min(ticker_lengths.values())}, max={max(ticker_lengths.values())}",
            flush=True,
        )
        print(f"  [cross-sec] common_length (fold 기준) = {common_length}", flush=True)

    # walk_forward_folds 호출 (Phase 1.5 의 dataset.py)
    folds = walk_forward_folds(
        n=common_length,
        is_len=config["is_len"],
        purge=config["window"],
        emb=config["embargo"],
        oos_len=config["oos_len"],
        step=config["step"],
    )

    if verbose:
        print(f"  [cross-sec] {len(folds)} fold 생성", flush=True)

    # ─── 매 fold 학습 + 예측 ───
    fold_results = []
    t_start = time.time()

    # device 설정
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch_device = torch.device("cuda:0")
    else:
        torch_device = torch.device("cpu")

    # ⭐ tqdm.auto: notebook + CMD 모두 작동
    try:
        from tqdm.auto import tqdm

        fold_iter = (
            tqdm(enumerate(folds), total=len(folds), desc="CS fold", ncols=100)
            if verbose
            else enumerate(folds)
        )
    except ImportError:
        fold_iter = enumerate(folds)

    for k, (train_idx_common, test_idx_common) in fold_iter:
        # ⭐ Seed 고정 (재현성)
        torch.manual_seed(42 + k)
        np.random.seed(42 + k)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + k)

        # 종목별 train/test idx (공통 length 기준 → 모든 종목 공통 fold)
        train_idx_per_ticker = {t: train_idx_common for t in cs_inputs["series"]}
        test_idx_per_ticker = {t: test_idx_common for t in cs_inputs["series"]}

        train_ds, test_ds, scalers = _build_cs_dataset_for_fold(
            cs_inputs, train_idx_per_ticker, test_idx_per_ticker, config["seq_len"]
        )

        if len(train_ds) < 100 or len(test_ds) == 0:
            if verbose:
                print(
                    f"    [fold {k}] sample 부족 (train {len(train_ds)}, test {len(test_ds)}) → skip"
                )
            continue

        # Train/Val split (마지막 10% val)
        n_train = len(train_ds)
        n_val = max(int(n_train * 0.1), 100)
        n_train_only = n_train - n_val
        train_only_ds = torch.utils.data.Subset(train_ds, list(range(n_train_only)))
        val_ds = torch.utils.data.Subset(train_ds, list(range(n_train_only, n_train)))

        # ⭐ DataLoader 최적화 (num_workers=0 권장 — Jupyter + Windows 호환)
        # num_workers > 0 시 multiprocessing fork 충돌 가능
        train_loader = DataLoader(
            train_only_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=(torch_device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=(torch_device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=(torch_device.type == "cuda"),
        )

        # 모델
        model = CrossSectionalLSTMRegressor(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            n_tickers=cs_inputs["n_tickers"],
            embedding_dim=config["embedding_dim"],
            output_dropout=config.get("output_dropout", 0.2),
        ).to(torch_device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["lr_factor"],
            patience=config["lr_patience"],
        )

        # 학습 루프 (early stopping)
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(config["max_epochs"]):
            model.train()
            train_loss_sum = 0.0
            for x, ticker_ids, y in train_loader:
                x, ticker_ids, y = (
                    x.to(torch_device),
                    ticker_ids.to(torch_device),
                    y.to(torch_device),
                )
                optimizer.zero_grad()
                y_pred = model(x, ticker_ids)
                loss = criterion(y_pred, y)
                loss.backward()
                # ⭐ Gradient clipping (LSTM 학습 안정)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item() * len(y)

            # Validation
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for x, ticker_ids, y in val_loader:
                    x, ticker_ids, y = (
                        x.to(torch_device),
                        ticker_ids.to(torch_device),
                        y.to(torch_device),
                    )
                    y_pred = model(x, ticker_ids)
                    val_loss_sum += criterion(y_pred, y).item() * len(y)

            val_loss_avg = val_loss_sum / max(n_val, 1)
            scheduler.step(val_loss_avg)

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_state = {k_p: v.clone() for k_p, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["early_stop_patience"]:
                    break

        # Best 모델 복원
        if best_state is not None:
            model.load_state_dict(best_state)

        # OOS 예측
        model.eval()
        preds_lstm_cs = []
        truths = []
        meta_list = []
        with torch.no_grad():
            for batch_idx, (x, ticker_ids, y) in enumerate(test_loader):
                x, ticker_ids = x.to(torch_device), ticker_ids.to(torch_device)
                y_pred = model(x, ticker_ids).cpu().numpy()
                preds_lstm_cs.extend(y_pred.tolist())
                truths.extend(y.numpy().tolist())

        # Test sample 의 (ticker, pos) 추출
        test_samples = test_ds.samples
        for (ticker, pos), y_pred_val, y_true_val in zip(
            test_samples, preds_lstm_cs, truths
        ):
            date = cs_inputs["date"][ticker][pos]
            fold_results.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": k,
                    "y_true": y_true_val,
                    "y_pred_lstm_cs": y_pred_val,
                }
            )

        if verbose:
            elapsed = time.time() - t_start
            avg = elapsed / (k + 1)
            eta = avg * (len(folds) - k - 1)
            print(
                f"    [fold {k+1}/{len(folds)}] epoch~{epoch+1}, "
                f"val={best_val_loss:.4f}, train/test={len(train_ds)}/{len(test_ds)}, "
                f"elapsed={elapsed/60:.1f}분, ETA={eta/60:.0f}분",
                flush=True,
            )

    fold_df = pd.DataFrame(fold_results)

    # ─── HAR 결합 (use_har=True 시) ───
    if use_har:
        if verbose:
            print(f"  [cross-sec] HAR 종목별 적합 + 결합...", flush=True)

        har_predictions = []
        ticker_keys = list(cs_inputs["series"].keys())
        try:
            from tqdm.auto import tqdm

            ticker_iter = (
                tqdm(ticker_keys, desc="HAR fit", ncols=100) if verbose else ticker_keys
            )
        except ImportError:
            ticker_iter = ticker_keys

        for ticker in ticker_iter:
            # log_ret series 재구성
            panel_t = (
                panel[panel["ticker"] == ticker]
                .sort_values("date")
                .reset_index(drop=True)
            )
            log_ret = pd.Series(panel_t["log_ret"].values, index=panel_t["date"].values)

            for k, (train_idx_common, test_idx_common) in enumerate(folds):
                try:
                    y_pred_har, _ = fit_har_rv(
                        log_ret=log_ret,
                        train_idx=train_idx_common,
                        test_idx=test_idx_common,
                        horizon=config["window"],
                    )
                    for i, idx in enumerate(test_idx_common):
                        if i < len(y_pred_har):
                            har_predictions.append(
                                {
                                    "date": cs_inputs["date"][ticker][idx],
                                    "ticker": ticker,
                                    "fold": k,
                                    "y_pred_har": y_pred_har[i],
                                }
                            )
                except Exception:
                    continue

        har_df = pd.DataFrame(har_predictions)

        # Merge
        merged_df = fold_df.merge(har_df, on=["date", "ticker", "fold"], how="left")

        # Performance ensemble 가중치 계산
        # LSTM_CS 와 HAR 의 RMSE 기반 (이전 fold)
        from collections import defaultdict

        ticker_prev_rmse = defaultdict(lambda: {"lstm_cs": None, "har": None})

        out_rows = []
        for ticker, ticker_df in merged_df.groupby("ticker"):
            ticker_df = ticker_df.sort_values(["fold", "date"]).copy()
            for k_fold, fold_rows in ticker_df.groupby("fold"):
                fold_rows = fold_rows.copy()
                prev_rmse_lstm = ticker_prev_rmse[ticker]["lstm_cs"]
                prev_rmse_har = ticker_prev_rmse[ticker]["har"]

                if prev_rmse_lstm is not None and prev_rmse_har is not None:
                    inv_lstm = 1.0 / max(prev_rmse_lstm, 1e-6)
                    inv_har = 1.0 / max(prev_rmse_har, 1e-6)
                    w_lstm_cs = inv_lstm / (inv_lstm + inv_har)
                    w_har = 1 - w_lstm_cs
                else:
                    w_lstm_cs, w_har = 0.5, 0.5

                fold_rows["w_lstm_cs"] = w_lstm_cs
                fold_rows["w_har"] = w_har
                fold_rows["y_pred_ensemble"] = w_lstm_cs * fold_rows[
                    "y_pred_lstm_cs"
                ] + w_har * fold_rows["y_pred_har"].fillna(fold_rows["y_pred_lstm_cs"])

                # 본 fold 의 RMSE 산출 → 다음 fold 가중치
                valid = fold_rows.dropna(
                    subset=["y_true", "y_pred_lstm_cs", "y_pred_har"]
                )
                if len(valid) > 0:
                    err_lstm = (valid["y_pred_lstm_cs"] - valid["y_true"]).values
                    err_har = (valid["y_pred_har"] - valid["y_true"]).values
                    ticker_prev_rmse[ticker]["lstm_cs"] = float(
                        np.sqrt(np.mean(err_lstm**2))
                    )
                    ticker_prev_rmse[ticker]["har"] = float(
                        np.sqrt(np.mean(err_har**2))
                    )

                out_rows.append(fold_rows)

        ensemble_df = pd.concat(out_rows, ignore_index=True)
    else:
        # HAR 결합 X → LSTM CS 만
        ensemble_df = fold_df.copy()
        ensemble_df["y_pred_har"] = np.nan
        ensemble_df["w_lstm_cs"] = 1.0
        ensemble_df["w_har"] = 0.0
        ensemble_df["y_pred_ensemble"] = ensemble_df["y_pred_lstm_cs"]

    # 저장
    ensemble_df.to_csv(out_dir / out_name, index=False)
    if verbose:
        elapsed_total = time.time() - t_start
        print(
            f"  [cross-sec] 전체 완료: {elapsed_total/60:.1f}분, {len(ensemble_df)} 행"
        )
        print(f"  [cross-sec] 저장: {out_dir / out_name}")

    return ensemble_df
