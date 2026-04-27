"""Phase 1 (GRU) — 평가 지표 모듈.

공개 인터페이스
--------------
hit_rate(y_true, y_pred, exclude_zero=True)         → float   부호 적중률 (관문 > 0.55)
r2_oos(y_true, y_pred)                              → float   Campbell & Thompson 2008 (관문 > 0)
r2_standard(y_true, y_pred)                         → float   sklearn r2_score 동일
mae(y_true, y_pred)                                 → float
rmse(y_true, y_pred)                                → float
baseline_metrics(y_test, y_train)                   → Dict[str, Dict[str, float]]
summarize_folds(per_fold_metrics)                   → Dict[str, Dict[str, float]]

Phase 1 관문 (PLAN.md)
----------------------
- ``hit_rate > 0.55`` AND ``r2_oos > 0`` 둘 다 충족 시 PASS → Phase 2 진행 권고

해석 주의 (학습자료_주의사항.md §6.5)
------------------------------------
통계적 유의성 ≠ 경제적 의미. 본 모듈의 모든 메트릭은 통계적 측정치이며,
실제 거래 가능성·거래비용·슬리피지는 별도 평가 필요.

R²_OOS 가 음수이면 모델이 ``0 예측 baseline`` 보다도 못함을 의미하므로
즉시 보고하고 모델·평가 코드를 재검토할 것.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 단일 메트릭
# ---------------------------------------------------------------------------
def hit_rate(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    exclude_zero: bool = True,
) -> float:
    """부호 적중률 (방향 일치 비율).

    Parameters
    ----------
    y_true, y_pred : array-like
        실제·예측 값.
    exclude_zero : bool, default True
        True 면 ``y_true == 0`` 또는 ``y_pred == 0`` 인 샘플 제외 (방향 정의 모호 회피).
        False 면 ``np.sign`` 기준 비교 (0 vs ±1 은 불일치).

    Returns
    -------
    float
        방향 일치 비율 (0.0 ~ 1.0). 비교 가능 샘플이 없으면 ``nan``.
        Phase 1 관문: > 0.55.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if exclude_zero:
        mask = (yt != 0) & (yp != 0)
        yt = yt[mask]
        yp = yp[mask]
    if len(yt) == 0:
        return float('nan')
    return float((np.sign(yt) == np.sign(yp)).mean())


def r2_oos(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """Out-of-Sample R² (Campbell & Thompson 2008).

    공식
    ----
    ``r2_oos = 1 - sum((y - y_hat)**2) / sum(y**2)``

    분모가 ``sum(y**2)`` 이므로 '0 예측 baseline' 대비 개선 정도를 의미.
    표준 R² 와 달리 ``y`` 의 평균을 빼지 않는다 (수익률 평균은 0 에 가까움 가정).

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    float
        ``> 0`` 이면 0 예측보다 개선. ``< 0`` 이면 0 예측보다 못함 (즉시 재검토).
        ``sum(y**2) == 0`` 이면 ``nan``.
        Phase 1 관문: > 0.

    References
    ----------
    Campbell, J. Y., & Thompson, S. B. (2008). Predicting Excess Stock Returns
    Out of Sample: Can Anything Beat the Historical Average? *Review of Financial
    Studies, 21*(4), 1509-1531.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sse = float(((yt - yp) ** 2).sum())
    sum_y2 = float((yt ** 2).sum())
    if sum_y2 == 0:
        return float('nan')
    return 1.0 - sse / sum_y2


def r2_standard(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """표준 R² (sklearn ``r2_score`` 와 동일).

    공식
    ----
    ``r2 = 1 - sum((y - y_hat)**2) / sum((y - mean(y))**2)``

    '평균 예측 baseline' 대비 개선 정도. 분모가 분산이므로 r2_oos 와 다른 척도.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sse = float(((yt - yp) ** 2).sum())
    sst = float(((yt - yt.mean()) ** 2).sum())
    if sst == 0:
        return float('nan')
    return 1.0 - sse / sst


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.abs(yt - yp).mean())


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(((yt - yp) ** 2).mean()))


# ---------------------------------------------------------------------------
# Baseline 비교 표
# ---------------------------------------------------------------------------
def baseline_metrics(
    y_test: Sequence[float],
    y_train: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """3가지 baseline 의 메트릭을 일괄 산출한다.

    Baseline
    --------
    - ``zero``       : 항상 0 예측 (R²_OOS 의 정의상 baseline)
    - ``previous``   : 직전 시점 값 (random walk 가정. y_test[0] 의 직전은 y_train[-1])
    - ``train_mean`` : y_train 평균 (역사적 평균 baseline)

    각 baseline 에 대해 ``hit_rate, r2_oos, r2_standard, mae, rmse`` 를 계산한다.
    LSTM 결과와 직접 비교하여 모델의 우위 여부를 판단한다.

    Parameters
    ----------
    y_test : array-like
        테스트 실제 값.
    y_train : array-like
        훈련 실제 값 (previous/train_mean baseline 계산용).

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``{'zero': {...}, 'previous': {...}, 'train_mean': {...}}`` 형태.
        각 inner dict 키: ``hit_rate, r2_oos, r2_standard, mae, rmse``.

    Notes
    -----
    'zero' baseline 의 ``hit_rate`` 는 정의상 nan (예측이 항상 0 → 방향 정보 없음).
    'train_mean' baseline 의 ``hit_rate`` 도 train mean 이 0 에 가까우면 nan 가능.
    """
    yt = np.asarray(y_test, dtype=float)
    ytr = np.asarray(y_train, dtype=float)
    if len(ytr) == 0:
        raise ValueError('y_train 이 비어 있습니다.')

    y_zero = np.zeros_like(yt)
    y_mean = np.full_like(yt, fill_value=float(ytr.mean()))
    # previous: 첫 샘플은 y_train 의 마지막 값으로 시작, 이후는 y_test[i-1]
    y_prev = np.concatenate([[float(ytr[-1])], yt[:-1]])

    baselines = {
        'zero': y_zero,
        'previous': y_prev,
        'train_mean': y_mean,
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, yp in baselines.items():
        out[name] = {
            'hit_rate': hit_rate(yt, yp),
            'r2_oos': r2_oos(yt, yp),
            'r2_standard': r2_standard(yt, yp),
            'mae': mae(yt, yp),
            'rmse': rmse(yt, yp),
        }
    return out


# ---------------------------------------------------------------------------
# Fold 통합 요약
# ---------------------------------------------------------------------------
def summarize_folds(
    per_fold_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """fold 별 메트릭 list 를 통계 요약 (mean / std / min / max / n).

    Parameters
    ----------
    per_fold_metrics : List[Dict[str, float]]
        예: ``[{'hit_rate': 0.52, 'r2_oos': -0.01, ...}, ...]``

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``{metric_name: {'mean': ..., 'std': ..., 'min': ..., 'max': ..., 'n': ...}}``.
        NaN 인 fold 값은 통계에서 제외 (집계 후 ``n`` 으로 유효 fold 수 표시).
        ``std`` 는 표본표준편차 (ddof=1). 유효 fold 1개 이하면 0.
    """
    if not per_fold_metrics:
        return {}
    # 모든 fold 의 키 합집합 (일부 fold 에 없는 키도 포함 가능)
    keys: set = set()
    for d in per_fold_metrics:
        keys.update(d.keys())

    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        raw = [d.get(k, float('nan')) for d in per_fold_metrics]
        arr = np.array([v for v in raw if v is not None and not np.isnan(v)],
                       dtype=float)
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
