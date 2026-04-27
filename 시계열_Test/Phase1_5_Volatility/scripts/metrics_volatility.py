"""Phase 1.5 — 변동성 예측 평가 지표 모듈.

본 모듈은 Phase 1 의 ``scripts/metrics.py`` 와 같은 역할을 하나, 변동성 예측에
특화된 지표를 제공한다 (Hit Rate / R²_OOS 폐기, RMSE/QLIKE/R²_train_mean/MZ 채택).

공개 인터페이스
--------------
rmse(y_true, y_pred)                                  → float
mae(y_true, y_pred)                                   → float
qlike(y_true_logrv, y_pred_logrv)                     → float   Patton (2011) 비대칭 손실
r2_train_mean(y_test, y_pred, y_train)                → float   train 평균 baseline 대비 개선
mz_regression(y_true, y_pred)                         → Dict[str, float]  Mincer-Zarnowitz
pred_std_ratio(y_true, y_pred)                        → float   mean-collapse 진단
baseline_metrics_volatility(y_test, y_train, **preds) → Dict[name, Dict[metric, value]]
summarize_folds_volatility(per_fold_metrics)          → Dict[metric, Dict[stat, value]]

PASS 조건 (PLAN §6 — 변동성 예측이 가능한가?)
--------------------------------------------
다음 3개 모두 충족 시 PASS:
1. ``LSTM RMSE < HAR-RV RMSE`` (105 fold 평균 기준)
2. ``r2_train_mean > 0`` (train 평균 baseline 능가)
3. ``pred_std_ratio > 0.5`` (mean-collapse 회피)

폐기된 지표 (Phase 1 vs Phase 1.5)
----------------------------------
- Hit Rate : 변동성은 항상 양수 → ``sign()`` 항상 +1 → trivially 1.0
- R²_OOS (zero baseline) : 변동성=0 비현실, 분모 ``sum(y²)`` 인공 증가 → R² 거짓 1 근접

References
----------
Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility
proxies. *Journal of Econometrics, 160*(1), 246-256.

Mincer, J., & Zarnowitz, V. (1969). The Evaluation of Economic Forecasts. NBER.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 단일 메트릭
# ---------------------------------------------------------------------------
def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(((yt - yp) ** 2).mean()))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.abs(yt - yp).mean())


def qlike(
    y_true_logrv: Sequence[float], y_pred_logrv: Sequence[float]
) -> float:
    """QLIKE — Quasi-Likelihood (Patton 2011, 변동성 예측 학술 표준 비대칭 손실).

    공식 (variance domain 으로 변환 후 계산)
    ----
    ``σ²_true = exp(2 · log_rv_true)``  (log(std) → variance)
    ``σ²_pred = exp(2 · log_rv_pred)``
    ``QLIKE  = mean( σ²_true/σ²_pred  -  log(σ²_true/σ²_pred)  -  1 )``

    핵심 특징
    --------
    - **비대칭 손실** : under-prediction (과소예측) 을 over-prediction 보다 크게 처벌.
      위험 관리 관점에서 변동성 과소평가가 더 위험 (실제는 더 출렁이는데 안전 판단).
    - 이상적 예측 (σ²_true ≡ σ²_pred) 시 QLIKE = 0.
    - 작을수록 좋음.

    Parameters
    ----------
    y_true_logrv, y_pred_logrv : array-like
        log(RV) 도메인의 실제·예측 값 (= log(std), variance 가 아님).

    Returns
    -------
    float
        QLIKE 값. 모든 입력이 finite 일 때만 유효.

    Notes
    -----
    σ²_pred = 0 (= log_rv → -∞) 일 시 division by zero 발생. 본 프로젝트는
    log-RV 직접 예측이므로 exp 변환 후 0 진입 매우 드물지만 방어용으로 1e-30 로 clip.

    References
    ----------
    Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility
    proxies. *Journal of Econometrics, 160*(1), 246-256.
    """
    yt = np.asarray(y_true_logrv, dtype=float)
    yp = np.asarray(y_pred_logrv, dtype=float)
    # log(std) → variance: σ² = (std)² = exp(2*log(std)) = exp(2*log_rv)
    var_true = np.exp(2.0 * yt)
    var_pred = np.exp(2.0 * yp)
    var_pred = np.maximum(var_pred, 1e-30)               # 누수 X — division 방어
    ratio = var_true / var_pred
    return float((ratio - np.log(ratio) - 1.0).mean())


def r2_train_mean(
    y_test: Sequence[float],
    y_pred: Sequence[float],
    y_train: Sequence[float],
) -> float:
    """Train 평균 baseline 대비 개선 R².

    공식
    ----
    ``r2_train_mean = 1 - SSE_model / SSE_train_mean``
    where::
        SSE_model      = sum( (y_pred - y_test)² )
        SSE_train_mean = sum( (mean(y_train) - y_test)² )

    의미
    ----
    "train 평균값으로만 예측하는 trivial baseline" 대비 모델이 얼마나 개선했는가.

    Returns
    -------
    float
        - ``> 0`` : 모델이 train 평균보다 우위 (PLAN §6 관문 2 PASS)
        - ``< 0`` : 모델이 trivial baseline 보다 못함 → 학습 자체 실패 신호
        - ``SSE_train_mean == 0`` (y_test 모두 동일) : ``nan``

    왜 zero baseline 이 아닌가
    --------------------------
    변동성=0 은 비현실적, 분모 ``sum(y²)`` 가 인공적으로 큰 값 → R² 거짓 1 근접.
    Phase 1 의 R²_OOS 함정 (윈도우 겹침으로 분모 부풀림) 회피.
    """
    yt = np.asarray(y_test, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ytr = np.asarray(y_train, dtype=float)
    if len(ytr) == 0:
        raise ValueError("y_train 이 비어 있습니다.")
    train_mean = float(ytr.mean())
    sse_model = float(((yt - yp) ** 2).sum())
    sse_train_mean = float(((yt - train_mean) ** 2).sum())
    if sse_train_mean == 0:
        return float('nan')
    return 1.0 - sse_model / sse_train_mean


def mz_regression(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> Dict[str, float]:
    """Mincer-Zarnowitz 회귀 — 예측 unbiasedness 검정 (편향 진단 도구).

    회귀
    ----
    ``y_true[i] = α + β · y_pred[i] + ε[i]``  (OLS 적합)

    이상적 unbiased forecast: α=0 AND β=1.
    - ``α ≠ 0`` : 시스템적 편향 (모델이 일관되게 과/저예측)
    - ``β < 1`` : 모델이 변동성 변화폭을 과대 예측 (실제는 덜 출렁임)
    - ``β > 1`` : 모델이 변동성 변화폭을 과소 예측 (실제는 더 출렁임)
    - ``r2 << 1`` : 예측-실제 상관 약함

    Returns
    -------
    Dict[str, float]
        ``{'alpha', 'beta', 'r2'}``. 표본 수 부족 또는 분산 0 일 시 ``nan``.

    Notes
    -----
    학술적으로 α=0, β=1 의 동시 검정 (Wald test) 도 가능하나 본 모듈은 추정값
    제공만 한다. 통계적 유의성 검정은 노트북에서 별도 수행.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) < 2:
        return {'alpha': float('nan'), 'beta': float('nan'), 'r2': float('nan')}
    var_yp = float(yp.var(ddof=0))
    if var_yp == 0:
        return {'alpha': float('nan'), 'beta': float('nan'), 'r2': float('nan')}
    cov_yp_yt = float(((yp - yp.mean()) * (yt - yt.mean())).mean())
    beta = cov_yp_yt / var_yp
    alpha = float(yt.mean() - beta * yp.mean())
    pred_fit = alpha + beta * yp
    sse = float(((yt - pred_fit) ** 2).sum())
    sst = float(((yt - yt.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float('nan')
    return {'alpha': float(alpha), 'beta': float(beta), 'r2': float(r2)}


def pred_std_ratio(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> float:
    """예측 분산 / 실제 분산 비율 — mean-collapse 진단.

    공식
    ----
    ``ratio = std(y_pred) / std(y_true)``

    PLAN §6 관문 3
    --------------
    ``> 0.5`` : 예측이 실제 변동성 일부 포착 (PASS)
    ``< 0.5`` : mean-collapse — 모델이 평균값 근처만 출력 (학습 실패 흔한 패턴)
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    std_yt = float(yt.std(ddof=0))
    std_yp = float(yp.std(ddof=0))
    if std_yt == 0:
        return float('nan')
    return std_yp / std_yt


# ---------------------------------------------------------------------------
# Baseline 비교 표
# ---------------------------------------------------------------------------
def baseline_metrics_volatility(
    y_test: Sequence[float],
    y_train: Sequence[float],
    *,
    naive_pred: Optional[Sequence[float]] = None,
    har_pred: Optional[Sequence[float]] = None,
    ewma_pred: Optional[Sequence[float]] = None,
    train_mean_pred: Optional[Sequence[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """4종 baseline 의 메트릭을 일괄 산출 (PLAN §5).

    Baseline
    --------
    - ``train_mean`` : ``y_train`` 평균 (자동 생성, 항상 포함)
    - ``naive``      : ``y_train`` 의 마지막 값을 모든 test 시점에 broadcast
                        (변동성 강한 자기상관 활용 단순 baseline)
    - ``har``        : 외부에서 적합한 HAR-RV 예측값 (Corsi 2009)
    - ``ewma``       : 외부에서 적합한 EWMA 예측값 (RiskMetrics, λ=0.94)

    Parameters
    ----------
    y_test : array-like
        실제 test 값 (log-RV 도메인).
    y_train : array-like
        훈련 실제 값 (train_mean 계산용).
    naive_pred, har_pred, ewma_pred : array-like, optional
        외부에서 ``baselines_volatility.py`` 로 생성된 예측. None 이면 해당 baseline 제외.
    train_mean_pred : array-like, optional
        외부 주입용 (테스트 편의). None 이면 ``y_train.mean()`` 자동 생성.

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``{baseline_name: {'rmse', 'mae', 'qlike', 'r2_train_mean', 'pred_std_ratio'}}``.
    """
    yt = np.asarray(y_test, dtype=float)
    ytr = np.asarray(y_train, dtype=float)
    if len(ytr) == 0:
        raise ValueError("y_train 이 비어 있습니다.")

    if train_mean_pred is None:
        y_tm = np.full_like(yt, fill_value=float(ytr.mean()))
    else:
        y_tm = np.asarray(train_mean_pred, dtype=float)

    baselines: Dict[str, np.ndarray] = {'train_mean': y_tm}
    if naive_pred is not None:
        baselines['naive'] = np.asarray(naive_pred, dtype=float)
    if har_pred is not None:
        baselines['har'] = np.asarray(har_pred, dtype=float)
    if ewma_pred is not None:
        baselines['ewma'] = np.asarray(ewma_pred, dtype=float)

    out: Dict[str, Dict[str, float]] = {}
    for name, yp in baselines.items():
        out[name] = {
            'rmse': rmse(yt, yp),
            'mae': mae(yt, yp),
            'qlike': qlike(yt, yp),
            'r2_train_mean': r2_train_mean(yt, yp, ytr),
            'pred_std_ratio': pred_std_ratio(yt, yp),
        }
    return out


# ---------------------------------------------------------------------------
# Fold 통합 요약
# ---------------------------------------------------------------------------
def summarize_folds_volatility(
    per_fold_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """fold 별 메트릭 list 의 통계 요약 (mean / std / min / max / n).

    Phase 1 의 ``summarize_folds`` 와 구조 동일 (재구현 — 격리 보존).

    Parameters
    ----------
    per_fold_metrics : List[Dict[str, float]]
        예: ``[{'rmse': 0.30, 'qlike': 0.05, ...}, ...]``

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``{metric_name: {'mean', 'std', 'min', 'max', 'n'}}``.
        ``std`` 는 표본표준편차 (ddof=1). 유효 fold ≤ 1 이면 0.
    """
    if not per_fold_metrics:
        return {}
    keys: set = set()
    for d in per_fold_metrics:
        keys.update(d.keys())

    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        raw = [d.get(k, float('nan')) for d in per_fold_metrics]
        arr = np.array(
            [v for v in raw if v is not None and not np.isnan(v)],
            dtype=float,
        )
        if len(arr) == 0:
            out[k] = {
                'mean': float('nan'), 'std': float('nan'),
                'min': float('nan'), 'max': float('nan'), 'n': 0,
            }
        else:
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            out[k] = {
                'mean': float(arr.mean()),
                'std': std,
                'min': float(arr.min()),
                'max': float(arr.max()),
                'n': int(len(arr)),
            }
    return out
