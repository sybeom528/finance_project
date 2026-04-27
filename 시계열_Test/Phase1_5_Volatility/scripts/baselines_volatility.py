"""Phase 1.5 — 변동성 예측 베이스라인 모듈 (HAR-RV / EWMA / Naive).

본 모듈은 LSTM 모델과 비교할 학술·산업 표준 베이스라인을 제공한다.
모든 베이스라인은 ``log(RV)`` 도메인에서 작동하여 LSTM 과 직접 비교 가능하다.

공개 인터페이스
--------------
fit_har_rv(rv_trailing, train_idx, test_idx, horizon=21)  → Tuple[np.ndarray, Dict]  Corsi (2009)
predict_ewma(log_ret, train_idx, test_idx, horizon=21,    → np.ndarray              RiskMetrics
              lam=0.94)
predict_naive(rv_trailing, train_idx, test_idx)           → np.ndarray              직전 RV 유지
predict_train_mean(target, train_idx, test_idx)            → np.ndarray              train 평균

설계 원칙 — 누수 방지
--------------------
모든 함수는 ``train_idx`` 한정으로 적합·계산 후 ``test_idx`` 에서 예측.
``test_idx`` 의 데이터를 적합·계산에 절대 사용하지 않음.

References
----------
Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
*Journal of Financial Econometrics, 7*(2), 174-196.

J.P. Morgan & Reuters. (1996). RiskMetrics — Technical Document (4th ed.).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HAR-RV (Corsi 2009)
# ---------------------------------------------------------------------------
def fit_har_rv(
    log_ret: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    horizon: int = 21,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """HAR-RV (Heterogeneous Autoregressive Realized Volatility) — Corsi (2009).

    모델 (log-std domain — Phase 1.5 타깃과 일관)
    -------------------------------------------
    ``log(RV_h[t+h]) = β₀ + β_d · log(RV_d[t]) + β_w · log(RV_w[t]) + β_m · log(RV_m[t])``

    Variance proxy 정의 (일간 데이터 적응 — 일중 데이터 미보유)
        RV_var_d[t] = log_ret[t]²                              (1일 variance proxy)
        RV_var_w[t] = mean(log_ret²[t-4 : t+1])                (5일 평균 variance)
        RV_var_m[t] = mean(log_ret²[t-21 : t+1])               (22일 평균 variance)

    Std-domain 변환 (target 도메인 일치):
        RV_d[t] = sqrt(RV_var_d[t]) = |log_ret[t]|
        RV_w[t] = sqrt(RV_var_w[t])
        RV_m[t] = sqrt(RV_var_m[t])

    Target 정의 (build_daily_target_logrv_21d 와 동일):
        log(RV_h[t+h]) = log( std(log_ret[t+1 : t+h+1], ddof=1) )

    이론적 근거 — Heterogeneous Market Hypothesis (Müller et al. 1997)
        "시장 참가자가 단기·주간·월간 시야로 변동성을 본다" 는 가설을 OLS 로 단순화.
        단순한 선형 회귀이지만 변동성 예측의 사실상 학술 표준 베이스라인.

    누수 방지
    ---------
    - HAR features (RV_d, RV_w, RV_m) 는 trailing window 만 사용.
    - β 추정은 ``train_idx`` 한정으로만 — ``test_idx`` 데이터 절대 미사용.
    - ``RV_m`` 계산을 위해 train_idx 시작 22일 이전의 log_ret 필요 → 입력으로
      전체 시계열 (warmup 포함) 을 받아 외부 슬라이싱.

    Parameters
    ----------
    log_ret : pd.Series
        일별 log-return 시계열 (NaN 첫 행 가능). 길이 ≥ ``test_idx[-1] + horizon + 1``.
    train_idx : np.ndarray
        훈련 위치 인덱스 (정수 배열).
    test_idx : np.ndarray
        테스트 위치 인덱스.
    horizon : int, default 21
        Forward 예측 horizon (영업일).
    eps : float, default 1e-12
        ``log(0)`` 방어용 최소 양수.

    Returns
    -------
    pred : np.ndarray, shape (len(test_idx),)
        ``test_idx`` 각 시점에 대한 ``log(RV_h[t+horizon])`` 예측.
    coefs : Dict[str, float]
        ``{'beta_0', 'beta_d', 'beta_w', 'beta_m', 'r2_train', 'n_train'}``.

    References
    ----------
    Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
    *Journal of Financial Econometrics, 7*(2), 174-196.
    """
    lr = log_ret.values
    lr_sq = lr ** 2                                                  # variance proxy
    lr_sq_series = pd.Series(lr_sq, index=log_ret.index)

    # HAR features (variance domain) — 학술 표준 1·5·22일
    rv_var_d = lr_sq                                                 # 1일
    rv_var_w = lr_sq_series.rolling(5).mean().values                  # 5일 평균
    rv_var_m = lr_sq_series.rolling(22).mean().values                 # 22일 평균

    # Std-domain 으로 변환 (log(std) = 0.5 * log(variance))
    log_rv_d = 0.5 * np.log(np.maximum(rv_var_d, eps))
    log_rv_w = 0.5 * np.log(np.maximum(rv_var_w, eps))
    log_rv_m = 0.5 * np.log(np.maximum(rv_var_m, eps))

    # Forward target: log(std(log_ret[t+1 : t+horizon+1])) = build_daily_target_logrv_21d 와 동일
    rv_forward = log_ret.rolling(horizon).std(ddof=1).shift(-horizon)
    target_full = np.log(rv_forward).values                           # 누수: forward shift

    def _build_xy(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """idx 에 해당하는 (X, y) 페어 추출. NaN 행 제외."""
        idx = np.asarray(idx)
        X = np.column_stack([log_rv_d[idx], log_rv_w[idx], log_rv_m[idx]])
        y = target_full[idx]
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return X[valid], y[valid]

    X_train, y_train = _build_xy(train_idx)
    if len(X_train) < 4:
        raise ValueError(
            f"HAR-RV 적합용 유효 훈련 샘플 부족: {len(X_train)} (3 features + 절편 = 최소 4 필요)"
        )

    # OLS 적합 (절편 포함)
    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])
    beta, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
    beta_0, beta_d, beta_w, beta_m = beta

    # train R²
    fit_train = X_train_aug @ beta
    sse = float(((y_train - fit_train) ** 2).sum())
    sst = float(((y_train - y_train.mean()) ** 2).sum())
    r2_train = 1.0 - sse / sst if sst > 0 else float('nan')

    # Test 예측 — train_idx 미사용
    test_idx = np.asarray(test_idx)
    X_test = np.column_stack([log_rv_d[test_idx], log_rv_w[test_idx], log_rv_m[test_idx]])
    X_test = np.where(np.isfinite(X_test), X_test, 0.0)
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
    pred = X_test_aug @ beta

    coefs = {
        'beta_0': float(beta_0),
        'beta_d': float(beta_d),
        'beta_w': float(beta_w),
        'beta_m': float(beta_m),
        'r2_train': float(r2_train),
        'n_train': int(len(X_train)),
    }
    return pred, coefs


# ---------------------------------------------------------------------------
# EWMA (RiskMetrics 1996)
# ---------------------------------------------------------------------------
def predict_ewma(
    log_ret: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    horizon: int = 21,
    lam: float = 0.94,
) -> np.ndarray:
    """EWMA — Exponentially Weighted Moving Average (RiskMetrics, λ=0.94).

    재귀
    ----
    ``σ²[t] = λ · σ²[t-1] + (1-λ) · log_ret[t-1]²``

    h-step ahead 예측
    -----------------
    EWMA 의 1-step ahead variance forecast 를 horizon 동안 동일 유지 가정 (학술 관행).
    ``forecast[t] = sqrt(σ²[t])`` 를 모든 시점 t+1 ~ t+horizon 에 broadcast →
    horizon 평균 RV 예측은 ``sqrt(σ²[t])`` 자체와 동일.

    log-RV 변환 후 반환 — Phase 1.5 타깃 도메인과 일치.

    누수 방지
    ---------
    - 재귀 초기값: ``train_idx`` 의 첫 시점 ``log_ret²`` 평균 (간단 추정)
    - test 시점 t 의 σ²[t] 는 t-1 까지의 log_ret 만으로 계산되므로 누수 없음.
    - 단, EWMA 재귀는 train+test 전체 시계열을 통과하나 test 의 ``log_ret``
      자체는 t 시점 이후 (target) 가 아닌 t 시점까지의 데이터만 사용 → 안전.

    Parameters
    ----------
    log_ret : pd.Series
        일별 log-return 시계열 (NaN 첫 행 가능).
    train_idx, test_idx : np.ndarray
        정수 위치 인덱스.
    horizon : int, default 21
        Forward 예측 horizon — 본 함수에서는 단순화 (constant variance projection).
    lam : float, default 0.94
        EWMA 감쇠 계수 (RiskMetrics 표준).

    Returns
    -------
    np.ndarray, shape (len(test_idx),)
        ``log(sqrt(σ²[t]))`` = ``0.5 * log(σ²[t])`` — log-RV 도메인 예측.

    Notes
    -----
    표준 RiskMetrics 는 1-step ahead 만 직접 정의. h-step ahead 는 variance
    persistence 가정 (mean-reversion 무시) 에 따라 동일 σ² 유지. 본 함수는
    이를 채택. h 가 매우 길면 약간 낙관적이나 21일 horizon 에서는 합리적.
    """
    lr_arr = log_ret.values
    n = len(lr_arr)

    # 초기값: train 의 첫 시작 위치의 log_ret² 평균 (warmup 추정)
    # train_idx[0] 부터 시작하여 그 이전 데이터는 사용 X (누수 방지)
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    train_start = int(train_idx[0])
    train_end = int(train_idx[-1])

    # train 구간의 log_ret² 평균을 재귀 초기 σ²
    train_lr2 = lr_arr[train_start : train_end + 1] ** 2
    train_lr2 = train_lr2[np.isfinite(train_lr2)]
    if len(train_lr2) == 0:
        raise ValueError("EWMA 초기값 계산 실패: train 구간 모든 log_ret 이 NaN.")
    sigma2_init = float(train_lr2.mean())

    # 재귀: train 시작부터 test 끝까지 σ²[t] 시계열 생성
    test_end = int(test_idx[-1])
    sigma2 = np.full(n, np.nan, dtype=float)
    sigma2[train_start] = sigma2_init
    for t in range(train_start + 1, test_end + 1):
        prev_lr = lr_arr[t - 1]
        if not np.isfinite(prev_lr):
            sigma2[t] = sigma2[t - 1]                                # 누수 X — NaN 시 직전 σ² 유지
            continue
        sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * (prev_lr ** 2)

    # log-RV 도메인으로 변환: log(std) = 0.5 * log(variance)
    sigma2_test = sigma2[test_idx]
    sigma2_test = np.maximum(sigma2_test, 1e-30)                     # log(0) 방어
    log_rv_pred = 0.5 * np.log(sigma2_test)
    return log_rv_pred


# ---------------------------------------------------------------------------
# Naive — 직전 trailing RV 유지
# ---------------------------------------------------------------------------
def predict_naive(
    rv_trailing: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Naive baseline — 시점 t 의 trailing RV 를 forward window 예측으로 사용.

    예측
    ----
    ``log_rv_pred[t] = log( rv_trailing[t] )`` for each t in test_idx.

    의미
    ----
    "오늘까지의 21일 RV 가 내일~내달까지도 그대로 유지" — 변동성 강한 자기상관
    (lag 1 ACF ≈ 0.99) 의 단순 직접 활용.

    LSTM 이 이걸 못 이기면 모델이 의미 없음 (가장 무지성 baseline).

    Parameters
    ----------
    rv_trailing : pd.Series
        Trailing 21일 std (= 본 프로젝트 RV[t]).
    train_idx : np.ndarray
        훈련 인덱스 — 본 함수에서는 사용하지 않음 (인터페이스 통일을 위해 유지).
    test_idx : np.ndarray
        테스트 인덱스.

    Returns
    -------
    np.ndarray, shape (len(test_idx),)
    """
    _ = train_idx  # 인터페이스 통일을 위해 유지, 실제 사용 X
    test_idx = np.asarray(test_idx)
    rv_test = rv_trailing.values[test_idx]
    rv_test = np.maximum(rv_test, 1e-30)                             # log(0) 방어
    return np.log(rv_test)


# ---------------------------------------------------------------------------
# Train-Mean (편의용 — metrics 의 baseline_metrics_volatility 와 별개로 직접 호출 가능)
# ---------------------------------------------------------------------------
def predict_train_mean(
    target: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Train 평균 baseline — test 모든 시점에 train 의 target 평균 broadcast.

    Parameters
    ----------
    target : pd.Series
        Phase 1.5 의 forward log-RV 타깃 (build_daily_target_logrv_21d 결과).
    train_idx, test_idx : np.ndarray

    Returns
    -------
    np.ndarray, shape (len(test_idx),)
    """
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    train_target = target.values[train_idx]
    train_target = train_target[np.isfinite(train_target)]
    if len(train_target) == 0:
        raise ValueError("train_mean 계산 실패: train 구간 모든 target 이 NaN.")
    mean_val = float(train_target.mean())
    return np.full(len(test_idx), mean_val, dtype=float)
