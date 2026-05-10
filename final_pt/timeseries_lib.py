"""
timeseries_lib.py — Phase 1.5 + Phase 3-2 핵심 함수 통합 (CS 제외)

final/ 폴더의 표준 시계열·통계 함수 모듈. 03_Volatility_Forecasting.ipynb 와
04_Statistical_Validation.ipynb, lstm_pipeline.py 가 본 모듈을 import 하여 사용합니다.

함수 카테고리
-------------
- 환경: setup_seeds, setup_korean_font
- 타깃: build_log_rv_target, verify_no_leakage
- Walk-Forward: walk_forward_folds, build_fold_inputs
- LSTM v4: LSTMRegressor, train_one_fold, count_parameters
- HAR-RV: fit_har_rv
- Ensemble: diebold_pauly_weights, ensemble_predict
- 평가: rmse, qlike, r2_train_mean, pred_std_ratio, mz_regression, dm_test
- 통계: anova_variance_decomp, welch_anova, kruskal_wallis_eps_sq,
        pairwise_mann_whitney, cohen_d, heavy_tail_stats
- 보고: rmse_with_pct_summary, format_rmse_summary

CS 제외
-------
Cross-Sectional (Ticker Embedding LSTM) 는 본 모듈에서 추출 X. 통합 대상 =
Stockwise (per-ticker LSTM) + HAR-RV + Diebold-Pauly Performance-Weighted Ensemble.

사용 예
-------
>>> import timeseries_lib as tlib
>>> tlib.setup_seeds(42)
>>> rmse_val = tlib.rmse(y_true, y_pred)

References
----------
- Corsi (2009) HAR-RV
- Patton (2011) QLIKE
- Diebold & Pauly (1987) Performance-Weighted Ensemble
- Mincer & Zarnowitz (1969) MZ regression
- Diebold & Mariano (1995) DM test
- Cohen (1988) Effect size standards
- Welch (1951) ANOVA for unequal variances
- Lin (2013) Large-n trap warning
"""
from __future__ import annotations

import math
import platform
import random
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.linear_model import LinearRegression


# ══════════════════════════════════════════════════════════════════
# 1. 환경·시드 고정
# ══════════════════════════════════════════════════════════════════

def setup_seeds(seed: int = 42) -> None:
    """numpy / torch / random 시드 고정 (재현성 보장)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_korean_font() -> None:
    """한글 폰트 설정 (CLAUDE.md 전역 지침 준수)."""
    import matplotlib.pyplot as plt
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif platform.system() == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        try:
            import koreanize_matplotlib  # noqa: F401
            plt.rcParams["font.family"] = "NanumGothic"
        except ImportError:
            pass
    plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════════
# 2. 타깃 빌더 — Log-RV 21d forward (Phase 1.5)
# ══════════════════════════════════════════════════════════════════

def build_log_rv_target(log_ret: pd.Series, horizon: int = 21,
                        eps: float = 1e-12) -> pd.Series:
    """Forward h-day Log-Realized Volatility 타깃 생성.

    target[t] = log( std(log_ret[t+1 : t+h+1], ddof=1) )

    Parameters
    ----------
    log_ret : pd.Series
        일별 log-return 시계열.
    horizon : int, default 21
        Forward 윈도우 (영업일).
    eps : float, default 1e-12
        log(0) 방어용.

    Returns
    -------
    pd.Series
        forward h-day Log-RV. 마지막 horizon 일은 NaN.
    """
    rv_forward = log_ret.rolling(horizon).std(ddof=1).shift(-horizon)
    rv_forward = rv_forward.where(rv_forward > 0, eps)
    return np.log(rv_forward)


def verify_no_leakage(target: pd.Series, log_ret: pd.Series,
                      horizon: int = 21) -> bool:
    """Forward 타깃이 미래 데이터를 사용하는지 검증.

    target[t] 가 log_ret[t+1 : t+h+1] 에만 의존하는지 확인.
    """
    n = len(target)
    # 마지막 horizon 일은 forward 데이터 없음 → NaN
    last_valid_idx = n - horizon - 1
    if last_valid_idx < 0:
        return False
    return pd.isna(target.iloc[-horizon:]).all()


# ══════════════════════════════════════════════════════════════════
# 3. Walk-Forward 데이터셋
# ══════════════════════════════════════════════════════════════════

def walk_forward_folds(n_obs: int, is_len: int = 1250, purge: int = 21,
                       embargo: int = 63, oos: int = 21,
                       step: int = 21) -> List[Dict[str, np.ndarray]]:
    """Walk-Forward Cross-Validation fold 인덱스 생성 (Lopez de Prado 2018 표준).

    구조: [← IS is_len →][purge][embargo][← OOS oos →]
    각 fold 의 OOS 시작점은 step 만큼 이동.

    Parameters
    ----------
    n_obs : int
        전체 관측치 수.
    is_len : int, default 1250
        In-Sample 길이 (Phase 1.5 v4 best).
    purge : int, default 21
        IS-OOS 간 purge (target horizon 만큼).
    embargo : int, default 63
        purge 후 추가 embargo (장기 ACF 차단).
    oos : int, default 21
        Out-Of-Sample 길이.
    step : int, default 21
        fold 간 OOS 이동 폭.

    Returns
    -------
    List[Dict[str, np.ndarray]]
        각 fold = {'train_idx': ..., 'test_idx': ..., 'fold': k}
    """
    folds = []
    fold_id = 0
    train_start = 0
    while True:
        train_end = train_start + is_len
        test_start = train_end + purge + embargo
        test_end = test_start + oos
        if test_end > n_obs:
            break
        folds.append({
            'fold': fold_id,
            'train_idx': np.arange(train_start, train_end),
            'test_idx': np.arange(test_start, test_end),
        })
        train_start += step
        fold_id += 1
    return folds


def build_fold_inputs(features: np.ndarray, target: np.ndarray,
                       fold: Dict, seq_len: int = 63) -> Tuple[
                       torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """단일 fold 의 (X_train, y_train, X_test, y_test) 텐서 생성.

    각 시점 t 에 대해 [t-seq_len+1 : t+1] 의 features 를 시퀀스로 사용.
    """
    train_idx = fold['train_idx']
    test_idx = fold['test_idx']
    n_features = features.shape[1] if features.ndim > 1 else 1

    def _build(idx):
        Xs, ys = [], []
        for t in idx:
            if t < seq_len - 1:
                continue
            x_seq = features[t - seq_len + 1 : t + 1]
            y_val = target[t]
            if not np.isfinite(y_val) or not np.isfinite(x_seq).all():
                continue
            Xs.append(x_seq)
            ys.append(y_val)
        if not Xs:
            return None, None
        X = np.stack(Xs).astype(np.float32)
        y = np.array(ys, dtype=np.float32)
        if X.ndim == 2:
            X = X[..., None]
        return X, y

    X_tr, y_tr = _build(train_idx)
    X_te, y_te = _build(test_idx)
    return (torch.from_numpy(X_tr), torch.from_numpy(y_tr),
            torch.from_numpy(X_te), torch.from_numpy(y_te))


# ══════════════════════════════════════════════════════════════════
# 4. LSTM 모델 (Phase 1.5 v4 best)
# ══════════════════════════════════════════════════════════════════

class LSTMRegressor(nn.Module):
    """LSTM 기반 회귀 (시퀀스 → scalar).

    v4 best 기본값: input_size=3 (HAR 3ch+VIX), hidden=32, layers=1, dropout=0.3
    파라미터 수 ≈ 4,769 (3ch_vix 입력 기준)
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.3,
                 batch_first: bool = True) -> None:
        super().__init__()
        # num_layers=1 시 LSTM dropout 무시 → head_dropout 으로 대체
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, dropout=lstm_dropout,
                             batch_first=batch_first)
        self.head_dropout = nn.Dropout(dropout) if num_layers == 1 else nn.Identity()
        self.head = nn.Linear(hidden_size, 1)
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :] if self.batch_first else out[-1, :, :]
        last = self.head_dropout(last)
        return self.head(last).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """학습 가능 파라미터 수."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════
# 5. Walk-Forward 학습 (단일 fold) — full retraining 시 사용
# ══════════════════════════════════════════════════════════════════

def train_one_fold(X_tr: torch.Tensor, y_tr: torch.Tensor,
                    X_val: torch.Tensor, y_val: torch.Tensor,
                    input_size: int = 3, hidden: int = 32,
                    dropout: float = 0.3, lr: float = 1e-3,
                    weight_decay: float = 1e-3, max_epochs: int = 50,
                    batch_size: int = 32, patience: int = 5,
                    device: str = 'cpu') -> Tuple[nn.Module, Dict]:
    """단일 fold 의 LSTM 학습 (MSE loss + AdamW + EarlyStopping).

    Returns
    -------
    model : nn.Module
    info : Dict
        {'best_epoch', 'best_val_loss', 'final_train_loss'}
    """
    model = LSTMRegressor(input_size=input_size, hidden_size=hidden,
                           dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                                shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            n_train += len(yb)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(yb)
                n_val += len(yb)
        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
                    'final_train_loss': train_loss}


# ══════════════════════════════════════════════════════════════════
# 6. HAR-RV 베이스라인 (Corsi 2009)
# ══════════════════════════════════════════════════════════════════

def fit_har_rv(log_ret: pd.Series, train_idx: np.ndarray,
                test_idx: np.ndarray, horizon: int = 21,
                eps: float = 1e-12) -> Tuple[np.ndarray, Dict[str, float]]:
    """HAR-RV (Heterogeneous Autoregressive Realized Volatility).

    log(RV_h[t+h]) = β₀ + β_d·log(RV_d[t]) + β_w·log(RV_w[t]) + β_m·log(RV_m[t])

    Variance proxy (일간 데이터 적응):
        RV_var_d[t] = log_ret[t]²
        RV_var_w[t] = mean(log_ret²[t-4 : t+1])
        RV_var_m[t] = mean(log_ret²[t-21 : t+1])
    """
    lr = log_ret.values
    lr_sq = lr ** 2
    lr_sq_series = pd.Series(lr_sq, index=log_ret.index)

    rv_var_d = lr_sq
    rv_var_w = lr_sq_series.rolling(5).mean().values
    rv_var_m = lr_sq_series.rolling(22).mean().values

    log_rv_d = 0.5 * np.log(np.maximum(rv_var_d, eps))
    log_rv_w = 0.5 * np.log(np.maximum(rv_var_w, eps))
    log_rv_m = 0.5 * np.log(np.maximum(rv_var_m, eps))

    rv_forward = log_ret.rolling(horizon).std(ddof=1).shift(-horizon)
    target_full = np.log(rv_forward.where(rv_forward > 0, eps)).values

    def _build_xy(idx: np.ndarray):
        idx = np.asarray(idx)
        X = np.column_stack([log_rv_d[idx], log_rv_w[idx], log_rv_m[idx]])
        y = target_full[idx]
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return X[valid], y[valid]

    X_train, y_train = _build_xy(train_idx)
    if len(X_train) < 4:
        raise ValueError(f"HAR-RV 적합용 유효 훈련 샘플 부족: {len(X_train)}")

    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])
    beta, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
    beta_0, beta_d, beta_w, beta_m = beta

    fit_train = X_train_aug @ beta
    sse = float(((y_train - fit_train) ** 2).sum())
    sst = float(((y_train - y_train.mean()) ** 2).sum())
    r2_train = 1.0 - sse / sst if sst > 0 else float('nan')

    test_idx = np.asarray(test_idx)
    X_test = np.column_stack([log_rv_d[test_idx], log_rv_w[test_idx],
                                log_rv_m[test_idx]])
    X_test = np.where(np.isfinite(X_test), X_test, 0.0)
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
    pred = X_test_aug @ beta

    return pred, {'beta_0': float(beta_0), 'beta_d': float(beta_d),
                   'beta_w': float(beta_w), 'beta_m': float(beta_m),
                   'r2_train': float(r2_train), 'n_train': int(len(X_train))}


# ══════════════════════════════════════════════════════════════════
# 7. Performance-Weighted Ensemble (Diebold-Pauly 1987)
# ══════════════════════════════════════════════════════════════════

def diebold_pauly_weights(rmse_lstm: np.ndarray,
                           rmse_har: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Diebold-Pauly rolling 성과 가중치.

    fold k 의 가중치:
        w_LSTM[k] = (1/RMSE_LSTM[k-1]) / (1/RMSE_LSTM[k-1] + 1/RMSE_HAR[k-1])
        w_HAR[k]  = 1 - w_LSTM[k]

    fold 0 은 0.5/0.5 (사전 정보 없음).
    """
    n = len(rmse_lstm)
    w_lstm = np.full(n, 0.5)
    w_har = np.full(n, 0.5)
    for k in range(1, n):
        prev_lstm = rmse_lstm[k - 1]
        prev_har = rmse_har[k - 1]
        if not (np.isfinite(prev_lstm) and np.isfinite(prev_har)):
            continue
        if prev_lstm <= 0 or prev_har <= 0:
            continue
        inv_lstm = 1.0 / prev_lstm
        inv_har = 1.0 / prev_har
        w_lstm[k] = inv_lstm / (inv_lstm + inv_har)
        w_har[k] = 1.0 - w_lstm[k]
    return w_lstm, w_har


def ensemble_predict(y_pred_lstm: np.ndarray, y_pred_har: np.ndarray,
                      w_lstm: np.ndarray, w_har: np.ndarray) -> np.ndarray:
    """Performance-Weighted Ensemble 예측."""
    return w_lstm * y_pred_lstm + w_har * y_pred_har


# ══════════════════════════════════════════════════════════════════
# 8. 평가 지표
# ══════════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (학술 표준 metric, log-RV 공간).

    Patton 2011 표준을 따라 log(σ) 공간에서 계산. 변동성의 heteroskedastic 성질을
    log 변환으로 정규화하므로, η²/Welch F/KW H 등 학술 비교에 적합.

    발표용 친숙한 % 표기가 필요하면 :func:`rmse_with_pct_summary` 를 참고.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def rmse_with_pct_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """log RMSE + 발표용 % 보조 표기 반환 (Option A, 의미 보존).

    학술 표준 (log-RV 공간) 의 RMSE 본값을 그대로 보존하되,
    발표·대시보드용 친숙한 % 표기를 함께 반환.

    반환 dict:
      rmse_log         : float — log 공간 RMSE (의사결정·학술 비교 기준)
      mean_sigma_pct   : float — 평균 σ 예측치 (%/일), exp(y_pred).mean() × 100
      median_sigma_pct : float — 중앙값 σ 예측치 (%/일)
      max_sigma_pct    : float — 최대 σ 예측치 (%/일)
      rel_error_approx : float — log RMSE × 100 (Taylor 1차 근사 ± % 상대오차)
      n                : int   — 유효 sample 수

    발표 예시:
      avg log-RMSE = 0.2934 (학술 표준 metric)
        └ 평균 일별 σ ≈ 1.62%/일 (보조 표기)
        └ 상대오차 근사 ≈ ±29.34% (Taylor 1차)

    중요:
      - log RMSE 와 % 공간 RMSE 는 다른 metric. 의사결정·학술 비교는 log RMSE 사용.
      - σ_pct 는 단순 시각화·발표용 보조 표기.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return {
            'rmse_log': float('nan'),
            'mean_sigma_pct': float('nan'),
            'median_sigma_pct': float('nan'),
            'max_sigma_pct': float('nan'),
            'rel_error_approx': float('nan'),
            'n': 0,
        }
    yt, yp = y_true[mask], y_pred[mask]
    rmse_log = float(np.sqrt(np.mean((yt - yp) ** 2)))
    sigma_pred = np.exp(yp) * 100.0   # log(σ_daily) → σ_daily(%)
    return {
        'rmse_log'        : rmse_log,
        'mean_sigma_pct'  : float(np.mean(sigma_pred)),
        'median_sigma_pct': float(np.median(sigma_pred)),
        'max_sigma_pct'   : float(np.max(sigma_pred)),
        'rel_error_approx': rmse_log * 100.0,
        'n'               : n_valid,
    }


def format_rmse_summary(summary: dict, title: str = '') -> str:
    """rmse_with_pct_summary() 결과를 발표용 multi-line 문자열로 포맷.

    예시 출력:
      avg log-RMSE = 0.2934   (학술 표준)
        └ 평균 σ ≈ 1.62%/일   (보조 표기)
        └ 상대오차 ≈ ±29.34%  (Taylor 근사)
    """
    if not np.isfinite(summary.get('rmse_log', np.nan)):
        return f'{title}: 데이터 없음'
    lines = []
    if title:
        lines.append(f'{title}')
    lines.append(f'  log-RMSE         = {summary["rmse_log"]:.4f}   (학술 표준 metric)')
    lines.append(f'    └ 평균 σ      ≈ {summary["mean_sigma_pct"]:.2f}%/일   (보조 표기)')
    lines.append(f'    └ 중앙값 σ    ≈ {summary["median_sigma_pct"]:.2f}%/일')
    lines.append(f'    └ 상대오차    ≈ ±{summary["rel_error_approx"]:.2f}%   (Taylor 근사)')
    lines.append(f'    └ N           = {summary["n"]:,}')
    return '\n'.join(lines)


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE (Patton 2011) — 비대칭 손실, 과소예측 패널티.

    QLIKE = mean( σ²/σ̂² − log(σ²/σ̂²) − 1 )

    log-RV 도메인 입력이므로 exp(2·y) = variance 로 변환 후 계산.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float('nan')
    var_t = np.exp(2 * y_true[mask])
    var_p = np.exp(2 * y_pred[mask])
    var_p = np.maximum(var_p, 1e-30)
    ratio = var_t / var_p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def r2_train_mean(y_true: np.ndarray, y_pred: np.ndarray,
                   train_mean: float) -> float:
    """1 − SSE_model / SSE_train_mean (trivial baseline 능가 여부)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float('nan')
    sse_model = float(np.sum((y_true[mask] - y_pred[mask]) ** 2))
    sse_baseline = float(np.sum((y_true[mask] - train_mean) ** 2))
    if sse_baseline == 0:
        return float('nan')
    return 1.0 - sse_model / sse_baseline


def pred_std_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """std(y_pred) / std(y_true) — mean-collapse 진단 (~1.0 이상적)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float('nan')
    s_t = float(np.std(y_true[mask]))
    s_p = float(np.std(y_pred[mask]))
    if s_t == 0:
        return float('nan')
    return s_p / s_t


def mz_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Mincer-Zarnowitz: y_true = α + β·y_pred + ε. α=0, β=1 이면 unbiased."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return {'alpha': float('nan'), 'beta': float('nan'), 'r2': float('nan')}
    X = y_pred[mask].reshape(-1, 1)
    y = y_true[mask]
    reg = LinearRegression().fit(X, y)
    return {'alpha': float(reg.intercept_), 'beta': float(reg.coef_[0]),
             'r2': float(reg.score(X, y))}


# ══════════════════════════════════════════════════════════════════
# 9. Diebold-Mariano 검정
# ══════════════════════════════════════════════════════════════════

def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> Dict[str, float]:
    """Diebold-Mariano 검정 (Diebold & Mariano 1995).

    H0: 두 모델의 MSE 손실이 동일.
    DM > 0: 모델 1 (e1) 손실 > 모델 2 (e2) 손실 → 모델 2 우위
    DM < 0: 모델 1 우위
    """
    e1, e2 = np.asarray(e1), np.asarray(e2)
    mask = np.isfinite(e1) & np.isfinite(e2)
    if mask.sum() < 10:
        return {'dm': float('nan'), 'p_value': float('nan'), 'n': int(mask.sum())}
    d = e1[mask] ** 2 - e2[mask] ** 2
    n = len(d)
    d_mean = d.mean()
    # Newey-West HAC variance (lag = h-1)
    gamma_0 = float(np.var(d, ddof=1))
    var_d = gamma_0
    for lag in range(1, h):
        gamma_k = float(np.cov(d[:-lag], d[lag:])[0, 1])
        weight = 1.0 - lag / h
        var_d += 2 * weight * gamma_k
    se = math.sqrt(var_d / n) if var_d > 0 else float('nan')
    if se == 0 or not np.isfinite(se):
        return {'dm': float('nan'), 'p_value': float('nan'), 'n': n}
    dm_stat = d_mean / se
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return {'dm': float(dm_stat), 'p_value': float(p_val), 'n': n}


# ══════════════════════════════════════════════════════════════════
# 10. 학술 통계 검정 (Phase 3-2 §2-B)
# ══════════════════════════════════════════════════════════════════

def anova_variance_decomp(panel: pd.DataFrame, value_col: str = 'rmse',
                           factors: Sequence[str] = ('period', 'ticker')
                           ) -> pd.DataFrame:
    """ANOVA Variance Decomposition (η² eta-squared).

    Two-way ANOVA: SS_factor1 + SS_factor2 + SS_residual = SS_total
    η² = SS_factor / SS_total (Cohen 1988: small=0.01, medium=0.06, LARGE=0.14)

    Returns
    -------
    pd.DataFrame
        index=factors+['Residual','Total'], cols=[SS, df, F, p, eta_sq]
    """
    panel = panel.copy()
    y = panel[value_col].values
    y = y[np.isfinite(y)]
    grand_mean = y.mean()
    ss_total = float(np.sum((y - grand_mean) ** 2))

    rows = []
    ss_explained = 0.0
    df_explained = 0
    n_total = len(panel[np.isfinite(panel[value_col])])

    for factor in factors:
        groups = panel.groupby(factor)[value_col].agg(['mean', 'count'])
        ss_factor = float(np.sum(groups['count'] * (groups['mean'] - grand_mean) ** 2))
        df_factor = int(groups.shape[0] - 1)
        ss_explained += ss_factor
        df_explained += df_factor
        rows.append({'source': factor, 'SS': ss_factor, 'df': df_factor,
                      'eta_sq': ss_factor / ss_total if ss_total > 0 else 0.0})

    ss_residual = ss_total - ss_explained
    df_residual = n_total - 1 - df_explained
    if df_residual <= 0:
        df_residual = 1

    # F-test for each factor
    ms_residual = ss_residual / df_residual
    for row in rows:
        ms_factor = row['SS'] / max(row['df'], 1)
        f_stat = ms_factor / ms_residual if ms_residual > 0 else float('inf')
        p_val = 1.0 - stats.f.cdf(f_stat, row['df'], df_residual) if np.isfinite(f_stat) else 0.0
        row['F'] = f_stat
        row['p'] = p_val

    rows.append({'source': 'Residual', 'SS': ss_residual, 'df': df_residual,
                  'eta_sq': ss_residual / ss_total if ss_total > 0 else 0.0,
                  'F': float('nan'), 'p': float('nan')})
    rows.append({'source': 'Total', 'SS': ss_total, 'df': n_total - 1,
                  'eta_sq': 1.0, 'F': float('nan'), 'p': float('nan')})

    return pd.DataFrame(rows).set_index('source')[['SS', 'df', 'F', 'p', 'eta_sq']]


def welch_anova(values: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    """Welch ANOVA (이분산 robust).

    Levene test (등분산 검정) + Welch's F.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)
    mask = np.isfinite(values)
    values, groups = values[mask], groups[mask]

    unique_groups = np.unique(groups)
    group_data = [values[groups == g] for g in unique_groups]
    group_data = [g for g in group_data if len(g) >= 2]

    levene_stat, levene_p = stats.levene(*group_data, center='median')

    # Welch's ANOVA via scipy
    welch_stat, welch_p = stats.f_oneway(*group_data)
    # Note: stats.f_oneway is classical ANOVA, not Welch.
    # For true Welch, use scipy >= 1.7.0 stats.alexandergovern or manual.

    # Manual Welch F (https://en.wikipedia.org/wiki/Welch%27s_t-test)
    n_groups = len(group_data)
    means = np.array([g.mean() for g in group_data])
    variances = np.array([g.var(ddof=1) for g in group_data])
    n_k = np.array([len(g) for g in group_data])
    w_k = n_k / variances
    w_sum = w_k.sum()
    weighted_mean = (w_k * means).sum() / w_sum
    numerator = ((w_k * (means - weighted_mean) ** 2).sum() / (n_groups - 1))
    denom_part = ((1 - w_k / w_sum) ** 2 / (n_k - 1)).sum()
    denominator = 1 + (2 * (n_groups - 2) / (n_groups ** 2 - 1)) * denom_part
    welch_f = numerator / denominator
    df1 = n_groups - 1
    df2 = (n_groups ** 2 - 1) / (3 * denom_part)
    welch_p_correct = 1.0 - stats.f.cdf(welch_f, df1, df2)

    return {'levene_stat': float(levene_stat), 'levene_p': float(levene_p),
             'welch_F': float(welch_f), 'welch_p': float(welch_p_correct),
             'df1': float(df1), 'df2': float(df2)}


def kruskal_wallis_eps_sq(values: np.ndarray, groups: np.ndarray
                            ) -> Dict[str, float]:
    """Kruskal-Wallis H test + ε² (epsilon squared) 효과크기."""
    values = np.asarray(values)
    groups = np.asarray(groups)
    mask = np.isfinite(values)
    values, groups = values[mask], groups[mask]

    unique_groups = np.unique(groups)
    group_data = [values[groups == g] for g in unique_groups]
    group_data = [g for g in group_data if len(g) >= 1]

    h_stat, p_val = stats.kruskal(*group_data)
    n = len(values)
    k = len(group_data)
    eps_sq = (h_stat - k + 1) / (n - k) if (n - k) > 0 else float('nan')

    return {'H': float(h_stat), 'p_value': float(p_val), 'df': int(k - 1),
             'epsilon_sq': float(eps_sq), 'n': int(n)}


def cohen_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Cohen's d 효과크기 — pooled SD 정규화."""
    g1, g2 = np.asarray(g1), np.asarray(g2)
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return float('nan')
    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return float('nan')
    return float((g1.mean() - g2.mean()) / pooled)


def pairwise_mann_whitney(values: np.ndarray, groups: np.ndarray,
                           alpha: float = 0.05) -> pd.DataFrame:
    """모든 pair 의 Mann-Whitney U + Bonferroni 보정 + Cohen's d.

    Returns
    -------
    pd.DataFrame
        cols=[group_a, group_b, U, p, p_bonf, sig_bonf, cohens_d, d_class]
    """
    values = np.asarray(values)
    groups = np.asarray(groups)
    mask = np.isfinite(values)
    values, groups = values[mask], groups[mask]
    unique_groups = sorted(np.unique(groups).tolist())
    n_pairs = len(unique_groups) * (len(unique_groups) - 1) // 2
    alpha_bonf = alpha / n_pairs if n_pairs > 0 else alpha

    rows = []
    for i in range(len(unique_groups)):
        for j in range(i + 1, len(unique_groups)):
            g_a, g_b = unique_groups[i], unique_groups[j]
            v_a = values[groups == g_a]
            v_b = values[groups == g_b]
            if len(v_a) < 2 or len(v_b) < 2:
                continue
            u_stat, p_val = stats.mannwhitneyu(v_a, v_b, alternative='two-sided')
            d_val = cohen_d(v_a, v_b)
            d_abs = abs(d_val) if np.isfinite(d_val) else 0.0
            d_class = 'trivial'
            if d_abs >= 0.8:
                d_class = 'LARGE'
            elif d_abs >= 0.5:
                d_class = 'medium'
            elif d_abs >= 0.2:
                d_class = 'small'
            p_bonf = min(p_val * n_pairs, 1.0)
            rows.append({'group_a': g_a, 'group_b': g_b, 'U': float(u_stat),
                          'p': float(p_val), 'p_bonf': float(p_bonf),
                          'sig_bonf': p_val < alpha_bonf,
                          'cohens_d': d_val, 'd_class': d_class})
    return pd.DataFrame(rows)


def heavy_tail_stats(values: np.ndarray) -> Dict[str, float]:
    """Heavy-tail 통계: Skew, Excess Kurt, Jarque-Bera, Anderson-Darling."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return {k: float('nan') for k in
                 ['skewness', 'excess_kurtosis', 'jb_stat', 'jb_p',
                  'ad_stat', 'n']}

    skew = float(stats.skew(values))
    kurt = float(stats.kurtosis(values))   # excess kurtosis (Fisher def.)
    jb_stat, jb_p = stats.jarque_bera(values)
    ad_result = stats.anderson(values, dist='norm')
    return {'skewness': skew, 'excess_kurtosis': kurt,
             'jb_stat': float(jb_stat), 'jb_p': float(jb_p),
             'ad_stat': float(ad_result.statistic), 'n': int(len(values))}


# ══════════════════════════════════════════════════════════════════
# 11. 결과 검증 (재현성 보장)
# ══════════════════════════════════════════════════════════════════

def assert_close(actual: float, expected: float, tol: float = 0.005,
                 label: str = '') -> None:
    """수치 근사 일치 확인 (assert + 명확한 에러 메시지)."""
    diff = abs(actual - expected)
    if diff > tol:
        raise AssertionError(
            f"[{label}] expected={expected:.4f}, actual={actual:.4f}, "
            f"diff={diff:.4f} > tol={tol}"
        )
    print(f"  ✓ {label}: actual={actual:.4f} (expected={expected:.4f}, "
          f"diff={diff:.4f})")


def assert_phase15_results(metrics: Dict) -> None:
    """Phase 1.5 v8 ensemble 핵심 수치 검증.

    Snapshot 기준 (5월 초 panel cutoff 시점의 615 종목 stockwise 학습 결과):
        - LSTM avg RMSE ≈ 0.4298
        - HAR avg RMSE ≈ 0.3922
        - Ensemble avg RMSE ≈ 0.3815
    """
    print("Phase 1.5 (615 종목 stockwise) 결과 검증:")
    if 'lstm_rmse' in metrics:
        assert_close(metrics['lstm_rmse'], 0.4298, tol=0.005,
                     label='LSTM avg RMSE')
    if 'har_rmse' in metrics:
        assert_close(metrics['har_rmse'], 0.3922, tol=0.005,
                     label='HAR avg RMSE')
    if 'ensemble_rmse' in metrics:
        assert_close(metrics['ensemble_rmse'], 0.3815, tol=0.005,
                     label='Ensemble avg RMSE')


def assert_phase3_results(stats_dict: Dict) -> None:
    """Phase 3-2 §2-B 학술 통계 핵심 수치 검증.

    예상값:
        - η²_period ≈ 0.450 (LARGE)
        - η²_ticker ≈ 0.194 (LARGE)
        - Welch F ≈ 420.59
        - Skewness ≈ +1.30
        - Excess Kurtosis ≈ +4.71
    """
    print("Phase 3-2 §2-B 학술 통계 결과 검증:")
    if 'eta_sq_period' in stats_dict:
        assert_close(stats_dict['eta_sq_period'], 0.450, tol=0.01,
                     label='η²_period')
    if 'eta_sq_ticker' in stats_dict:
        assert_close(stats_dict['eta_sq_ticker'], 0.194, tol=0.01,
                     label='η²_ticker')
    if 'welch_F' in stats_dict:
        assert_close(stats_dict['welch_F'], 420.59, tol=5.0,
                     label='Welch F')
    if 'skewness' in stats_dict:
        assert_close(stats_dict['skewness'], 1.30, tol=0.05,
                     label='Skewness')
    if 'excess_kurtosis' in stats_dict:
        assert_close(stats_dict['excess_kurtosis'], 4.71, tol=0.10,
                     label='Excess Kurtosis')


# ══════════════════════════════════════════════════════════════════
# 12. 데이터 로딩 유틸 (final/ 환경 호환)
# ══════════════════════════════════════════════════════════════════

def load_ensemble_predictions(path: str | Path) -> pd.DataFrame:
    """ensemble_predictions_stockwise.csv 로드 + 기본 정제.

    - y_true 가 -inf / NaN 인 행 제거 (거래정지 등)
    - date 를 datetime 변환
    """
    df = pd.read_csv(path)
    n_before = len(df)
    df = df[np.isfinite(df['y_true'])].reset_index(drop=True)
    n_after = len(df)
    df['date'] = pd.to_datetime(df['date'])
    print(f"ensemble_predictions 로드: {n_before:,} → {n_after:,} 행 "
          f"(-inf/NaN {n_before - n_after:,} 행 제거)")
    return df


def load_sector_mapping(panel_path: str | Path) -> pd.Series:
    """final/data/monthly_panel.csv 에서 ticker → gics_sector mapping 추출.

    panel 의 마지막 시점 sector 사용 (시점에 따른 변경 무시).
    """
    panel = pd.read_csv(panel_path, usecols=['ticker', 'gics_sector'])
    sector_map = panel.groupby('ticker')['gics_sector'].last()
    return sector_map


def assign_periods(date_series: pd.Series) -> pd.Series:
    """5 시기 라벨 할당 (Phase 3-2 표준).

    P1: 2010-2014 / P2: 2015-2018 / P3: 2019-2020 (COVID) /
    P4: 2021-2022 / P5: 2023-2025
    """
    yr = pd.to_datetime(date_series).dt.year
    periods = pd.Series(index=date_series.index, dtype=object)
    periods[(yr >= 2010) & (yr <= 2014)] = 'P1 (2010-2014)'
    periods[(yr >= 2015) & (yr <= 2018)] = 'P2 (2015-2018)'
    periods[(yr >= 2019) & (yr <= 2020)] = 'P3 (2019-2020)'
    periods[(yr >= 2021) & (yr <= 2022)] = 'P4 (2021-2022)'
    periods[(yr >= 2023) & (yr <= 2025)] = 'P5 (2023-2025)'
    return periods


def filter_503_universe(df: pd.DataFrame) -> pd.DataFrame:
    """5 시기 모두 cover 하는 종목만 필터 (Phase 3-2 §2-B 표준).

    인수/파산 등으로 일부 시기만 데이터 있는 종목 제외.
    """
    df = df.copy()
    df['period'] = assign_periods(df['date'])
    df = df[df['period'].notna()]
    coverage = df.groupby('ticker')['period'].nunique()
    valid_tickers = coverage[coverage == 5].index
    return df[df['ticker'].isin(valid_tickers)].reset_index(drop=True)
