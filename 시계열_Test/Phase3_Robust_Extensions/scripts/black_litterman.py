"""Phase 2 — Black-Litterman 모델 함수 (서윤범 99_baseline 추출).

서윤범 [`99_baseline.ipynb`](../../서윤범/low_risk/99_baseline.ipynb) 의 5 함수를
**변경 없이** 그대로 추출하여 모듈화 (Phase 2 의 BL 백테스트 에서 import 가능하도록).

원본 출처
---------
서윤범/low_risk/99_baseline.ipynb (라인 174-249)

핵심 함수 5 종
-------------
- compute_pi()         : CAPM 역산으로 사전 균형수익률 (π) 계산
- build_P()            : 변동성 정렬 → 양극단 30% long/short P 행렬 (시총 가중)
- compute_omega()      : He-Litterman (1999) 표준 공식 (τ · P · Σ · P^T)
- black_litterman()    : 단일 view 단순화 공식으로 μ_BL 계산
- optimize_portfolio() : Markowitz 평균-분산 최적화 (long-only, Σw=1)

설계 원칙
---------
- 서윤범 baseline 의 수학적 정합성 보존 (relative view: row_sum(P)=0)
- BL 응용에 필요한 최소 함수만 포함 (Q 추정 함수는 별도 — Q_FIXED=0.003 사용 결정 5)
- 모든 인자/리턴 타입을 명시 (type hints)

사용 예시
---------
from scripts.black_litterman import compute_pi, build_P, compute_omega, black_litterman, optimize_portfolio

pi, lam = compute_pi(Sigma, w_mkt, spy_excess_ret, sigma2_mkt)
P = build_P(vol_pred_series, mcap_series, pct=0.30)
omega = compute_omega(P, Sigma, tau)
mu_BL = black_litterman(pi, Sigma, P, q=Q_FIXED, omega=omega, tau=tau)
weights = optimize_portfolio(mu_BL, Sigma, lam)
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =============================================================================
# 1. CAPM 역산 — 사전 균형수익률 (π)
# =============================================================================
def compute_pi(
    Sigma: pd.DataFrame,
    w_mkt: pd.Series,
    spy_excess_ret: float,
    sigma2_mkt: float,
    lam_fixed: Optional[float] = 2.5,
) -> Tuple[pd.Series, float]:
    """CAPM 역산으로 사전 균형수익률 (π) 와 위험회피 계수 (λ) 계산.

    Parameters
    ----------
    Sigma : pd.DataFrame (n × n)
        자산 공분산 행렬 (월별 단위 — 본 단계는 일별 ret × 21 환산).
    w_mkt : pd.Series (n,)
        시장 포트폴리오 가중치 (시총 비례).
    spy_excess_ret : float
        SPY 월별 평균 초과수익 (시장 risk premium).
    sigma2_mkt : float
        SPY 월별 분산 (시장 분산).
    lam_fixed : float | None, default 2.5
        ⭐ 위험회피 계수 결정 모드:
        - float (default 2.5): 고정 사용 (서윤범 99_baseline 일관, He-Litterman 1999)
        - None: 시장 데이터로 동적 계산 (Phase 2 옵션, clip [0.5, 10.0])

    Returns
    -------
    pi : pd.Series (n,)
        사전 균형수익률 벡터.
    lam : float
        위험회피 계수.

    Notes
    -----
    - He-Litterman (1999) 표준 공식: π = λ · Σ · w_mkt
    - 서윤범 99_baseline 은 LAM_FIXED=2.5 사용 → Phase 3 default 동일하게 변경.
    - 동적 모드 (lam_fixed=None) 는 Phase 2 호환용 옵션.
    """
    if lam_fixed is not None:
        lam = float(lam_fixed)
    else:
        lam = spy_excess_ret / sigma2_mkt if sigma2_mkt > 0 else 2.5
        lam = float(np.clip(lam, 0.5, 10.0))
    pi = lam * Sigma @ w_mkt
    return pi, lam


# =============================================================================
# 2. P 행렬 — 변동성 정렬 → 양극단 30% long/short
# =============================================================================
def build_P(
    vol_series: pd.Series,
    mcap_series: pd.Series,
    pct: float = 0.30,
) -> pd.Series:
    """변동성 정렬 기반 양극단 30% long/short P 행렬 (시총 가중).

    Parameters
    ----------
    vol_series : pd.Series (n,)
        자산별 변동성 (예측 또는 실현).
        ⭐ 본 Phase 2 에서는 ML ensemble 의 예측 변동성 사용.
    mcap_series : pd.Series (n,)
        자산별 시가총액 (시총 가중치 계산용).
    pct : float, default 0.30
        양극단 비율 (Pyo & Lee 2018 일관 = 30%).

    Returns
    -------
    P : pd.Series (n,)
        BL view 포트폴리오. 저위험 그룹 양수 (long), 고위험 그룹 음수 (short).
        relative view: P.sum() ≈ 0 (수학적 검증 — Step 3 코드 Y 에서 1.34e-16 확인).

    Notes
    -----
    - 저위험 (long): 변동성 하위 30% 종목, 시총 비례 양수
    - 고위험 (short): 변동성 상위 30% 종목, 시총 비례 음수
    - 중간 40%: P=0 (view 영향 없음)
    - 시총 가중: BL P 행렬의 표준 (Pyo & Lee 2018, He-Litterman 2002 등)
    """
    n_group = max(1, int(len(vol_series) * pct))
    sorted_idx = vol_series.sort_values().index
    low_risk = sorted_idx[:n_group]
    high_risk = sorted_idx[-n_group:]

    P = pd.Series(0.0, index=vol_series.index)
    low_m = mcap_series[low_risk]
    high_m = mcap_series[high_risk]
    P[low_risk] = low_m / low_m.sum()
    P[high_risk] = -high_m / high_m.sum()
    return P


# =============================================================================
# 3. Ω — view 불확실성 (He-Litterman 표준)
# =============================================================================
def compute_omega(
    P: pd.Series,
    Sigma: pd.DataFrame,
    tau: float,
) -> float:
    """He-Litterman (1999) 표준 공식: Ω = τ · P · Σ · P^T (단일 view 스칼라).

    Parameters
    ----------
    P : pd.Series (n,)
        view 포트폴리오 (build_P 결과).
    Sigma : pd.DataFrame (n × n)
        공분산 행렬 (월별 단위).
    tau : float
        BL 사전 분포 신뢰도 스케일.

    Returns
    -------
    omega : float
        view 불확실성 (단일 view → 스칼라).
        하한 1e-8 보장 (수치 안정성).

    Notes
    -----
    - 단일 view (k=1) 환경 → 스칼라 반환
    - τ 가 작을수록 view 신뢰도 ↑ (사후 분포가 view 에 가깝게)
    - 본 baseline 의 핵심 단순화: 추가 학습 없이 시장 정보만으로 Ω 결정
    """
    p = P.values
    omega = float(tau * p @ Sigma.values @ p)
    return max(omega, 1e-8)


# =============================================================================
# 4. Black-Litterman 결합 — 단일 view 단순화 공식
# =============================================================================
def black_litterman(
    pi: pd.Series,
    Sigma: pd.DataFrame,
    P: pd.Series,
    q: float,
    omega: float,
    tau: float,
) -> pd.Series:
    """Sherman-Woodbury-Morrison 단순화 공식으로 사후 기대수익률 (μ_BL) 계산.

    Parameters
    ----------
    pi : pd.Series (n,)
        사전 균형수익률 (compute_pi 결과).
    Sigma : pd.DataFrame (n × n)
        공분산 행렬.
    P : pd.Series (n,)
        view 포트폴리오.
    q : float
        view 수익률 (본 baseline 은 Q_FIXED = 0.003 = 월 0.3%).
    omega : float
        view 불확실성 (compute_omega 결과).
    tau : float
        BL 사전 분포 스케일.

    Returns
    -------
    mu_BL : pd.Series (n,)
        사후 기대수익률.

    Notes
    -----
    수학 공식 (단일 view k=1):
        μ_BL = π + (τΣ · P^T) · (q - P·π) / (P · τΣ · P^T + Ω)
                                              ↑ M (분모)
                                      ↑ diff
               ↑ pi             ↑ adjust

    - 일반 BL 의 역행렬 2회 → 단일 view 단순화로 역행렬 0회 (수치 안정)
    - q - P·π = "view 와 사전 균형 의 차이" (BL 조정량의 방향 결정)
    - M = "view 의 변동성 + view 의 불확실성" (조정량의 크기 정규화)
    """
    p = P.values
    pi_v = pi.values
    tSig = tau * Sigma.values
    M = float(p @ tSig @ p) + omega
    diff = q - float(p @ pi_v)
    adjust = tSig @ p * (diff / M)
    return pd.Series(pi_v + adjust, index=pi.index)


# =============================================================================
# 5. Markowitz 평균-분산 최적화 (long-only)
# =============================================================================
def optimize_portfolio(
    mu_BL: pd.Series,
    Sigma: pd.DataFrame,
    lam: float,
) -> pd.Series:
    """Markowitz 평균-분산 최적화 — BL 사후 기대수익률로 자산 비중 결정.

    Parameters
    ----------
    mu_BL : pd.Series (n,)
        BL 사후 기대수익률 (black_litterman 결과).
    Sigma : pd.DataFrame (n × n)
        공분산 행렬.
    lam : float
        위험회피 계수 (compute_pi 의 lam 와 동일).

    Returns
    -------
    weights : pd.Series (n,)
        최적 자산 비중. Σw=1, 0 ≤ w ≤ 1 (long-only).

    Notes
    -----
    수학 공식:
        min_w  (λ/2) · w^T·Σ·w  -  w^T·μ_BL
        s.t.   Σ w_i = 1
               0 ≤ w_i ≤ 1   (long-only)

    - 목적함수: 위험-수익 trade-off (Markowitz 1952)
    - jac: 분석 gradient → 수렴 가속
    - SLSQP: 등식·부등식 제약 최적화에 표준 (scipy.optimize)
    - 수렴 실패 시 fallback: 1/N 등배 (DeMiguel et al. 2009 강력 baseline)
    """
    n = len(mu_BL)
    mu = mu_BL.values
    Sig = Sigma.values

    def obj(w):
        return 0.5 * lam * w @ Sig @ w - w @ mu

    def jac(w):
        return lam * Sig @ w - mu

    res = minimize(
        obj, np.ones(n) / n, jac=jac, method='SLSQP',
        bounds=[(0, 1)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
    )
    if res.success:
        w = res.x
    else:
        # ⭐ Phase 3: silent 1/N fallback → 명시적 경고 출력
        warnings.warn(
            f'optimize_portfolio: SLSQP 수렴 실패 (n={n}, '
            f'message={res.message[:80]}) → 1/N fallback 적용',
            RuntimeWarning,
        )
        w = np.ones(n) / n
    return pd.Series(w, index=mu_BL.index)


# =============================================================================
# 6. 본 baseline 의 결정사항 (Phase 3 — 서윤범 99 일관)
# =============================================================================
# Q_FIXED = 0.003       # 월 0.3% = 연 3.6% (BAB factor 보수 추정 — Pyo & Lee 일관)
# DEFAULT_TAU = 0.1     # 서윤범 99_baseline 일관 (Phase 2 의 0.05 → 0.1 변경)
# PCT_GROUP = 0.30      # 양극단 30% (Pyo & Lee 2018 표준)
# LAM_FIXED = 2.5       # 서윤범 99_baseline 일관 (compute_pi 의 lam_fixed default)

Q_FIXED = 0.003
PCT_GROUP = 0.30
DEFAULT_TAU = 0.1
LAM_FIXED = 2.5
