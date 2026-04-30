"""Phase 2 — 공분산 행렬 추정 (일별 ret + LedoitWolf shrinkage + ×21 월별 환산).

결정 5 (PLAN.md §2): 일별 ret 으로 Σ_daily 추정 (T/N=25.2 안정) → ×21 월별 환산.

이론적 근거
-----------
i.i.d. 가정 (E[r_d] ≈ 0, Cov(r_d_t, r_d_s) = 0 for t ≠ s) 하에서:
    Σ_monthly = 21 × Σ_daily

수식:
    Var(r_m) = Var(Σ_t r_d_t) = 21 × Var(r_d)
    Cov(r_m_X, r_m_Y) = 21 × Cov(r_d_X, r_d_Y)

→ Phase 1.5 OOS = 21 영업일 = 1개월 → 모든 BL 입력 (Σ, Q, π) 월별 단위 통일.

핵심 함수
---------
- compute_sigma_daily()  : 일별 ret + LedoitWolf shrinkage
- daily_to_monthly()     : Σ_monthly = Σ_daily × 21
- estimate_covariance()  : 통합 함수 (일별 입력 → 월별 Σ 출력)

학술 근거
---------
- Ledoit & Wolf (2004) "A well-conditioned estimator for large-dimensional covariance matrices"
- Markowitz (1952) 평균-분산 최적화의 핵심 입력
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


# =============================================================================
# 일별 공분산 추정 (LedoitWolf shrinkage)
# =============================================================================
def compute_sigma_daily(
    returns_daily: pd.DataFrame,
    use_shrinkage: bool = True,
) -> pd.DataFrame:
    """일별 수익률 → 일별 공분산 행렬 (LedoitWolf shrinkage 적용 옵션).

    Parameters
    ----------
    returns_daily : pd.DataFrame (T × n)
        일별 log return. 행 = 영업일, 열 = 종목.
        ⚠️ NaN 자동 제거 (시작 시점 차이 등).
    use_shrinkage : bool, default True
        LedoitWolf shrinkage 적용 여부.
        True: 안정성 ↑ (대형주 단위 공분산 추정에 표준).
        False: 표본 공분산 (T/N > 10 일 때 정확).

    Returns
    -------
    Sigma_daily : pd.DataFrame (n × n)
        일별 단위 공분산 행렬.

    Notes
    -----
    Ledoit-Wolf shrinkage:
        Σ_LW = δ · F + (1-δ) · S
        F = 단위 분산 행렬 (target)
        S = 표본 공분산
        δ = optimal shrinkage intensity (자동 결정)

    안정성 비교:
        T/N > 10: 표본 공분산도 안정 (use_shrinkage=False 가능)
        T/N < 5 : LedoitWolf 필수
        본 환경: T=1260 일 (5년), N=50 종목 → T/N=25.2 → 둘 다 안정
    """
    # NaN 제거 (모든 종목 공통 거래일만 유지)
    rets_clean = returns_daily.dropna(how='any')
    if len(rets_clean) < 30:
        raise ValueError(f'returns_daily 의 valid 행 수 부족: {len(rets_clean)} < 30')

    if use_shrinkage:
        lw = LedoitWolf().fit(rets_clean.values)
        Sigma_daily = pd.DataFrame(
            lw.covariance_,
            index=rets_clean.columns,
            columns=rets_clean.columns,
        )
    else:
        Sigma_daily = rets_clean.cov()

    return Sigma_daily


# =============================================================================
# 일별 → 월별 환산
# =============================================================================
DAYS_PER_MONTH = 21  # 영업일 기준


def daily_to_monthly(
    Sigma_daily: pd.DataFrame,
    days_per_month: int = DAYS_PER_MONTH,
) -> pd.DataFrame:
    """일별 공분산 → 월별 공분산 (단순 곱 환산).

    Parameters
    ----------
    Sigma_daily : pd.DataFrame (n × n)
        일별 단위 공분산 행렬.
    days_per_month : int, default 21
        월별 환산 영업일 수 (Phase 1.5 OOS = 21 일관).

    Returns
    -------
    Sigma_monthly : pd.DataFrame (n × n)
        월별 단위 공분산 행렬.

    Notes
    -----
    수학 (i.i.d. 근사):
        r_monthly = Σ_{i=1}^{21} r_daily_i
        Var(r_monthly) = 21 × Var(r_daily)  (자기상관 0 가정)
        Cov(r_monthly_X, r_monthly_Y) = 21 × Cov(r_daily_X, r_daily_Y)

    가정 위반 영향 (S&P 500 일별 lag-1 ACF ~0.02-0.05):
        ~5% 분산 과소추정 (수용 가능)
        LedoitWolf shrinkage 가 추가로 완화
    """
    return Sigma_daily * days_per_month


# =============================================================================
# 통합 함수 — 일별 ret → 월별 Σ
# =============================================================================
def estimate_covariance(
    returns_daily: pd.DataFrame,
    is_start: pd.Timestamp,
    is_end: pd.Timestamp,
    use_shrinkage: bool = True,
    days_per_month: int = DAYS_PER_MONTH,
) -> pd.DataFrame:
    """전체 파이프라인: IS 슬라이싱 → 일별 Σ + LedoitWolf → ×21 월별 환산.

    Parameters
    ----------
    returns_daily : pd.DataFrame (T × n)
        전체 일별 수익률 (Step 2 daily_panel.csv 의 log_ret 활용).
    is_start, is_end : pd.Timestamp
        IS (in-sample) 기간 — Step 4 백테스트 의 매월 시점에 따라 변동.
        ⚠️ OOS 데이터 누수 방지 (is_end 까지만 사용).
    use_shrinkage : bool, default True
    days_per_month : int, default 21

    Returns
    -------
    Sigma_monthly : pd.DataFrame (n × n)
        월별 단위 공분산 행렬 (BL 입력).

    Notes
    -----
    누수 방지 핵심:
        매월 BL 리밸런싱 시점 t 에서:
            is_end = t (또는 t-1, OOS 직전)
            is_start = is_end - 5 years (or 1260 days)
            → IS 기간만 슬라이싱 → 미래 정보 없음

    예시:
        2020-01 BL → IS = 2015-01 ~ 2019-12 (5년)
        2020-02 BL → IS = 2015-02 ~ 2020-01 (1개월 슬라이딩)
    """
    rets_is = returns_daily.loc[is_start:is_end]
    Sigma_daily = compute_sigma_daily(rets_is, use_shrinkage=use_shrinkage)
    Sigma_monthly = daily_to_monthly(Sigma_daily, days_per_month=days_per_month)
    return Sigma_monthly


# =============================================================================
# 진단 — Σ 의 수학적 정합성
# =============================================================================
def diagnose_sigma(Sigma: pd.DataFrame) -> dict:
    """Σ 의 수학적 정합성 진단 (PSD, condition number, 고유값 분포).

    Parameters
    ----------
    Sigma : pd.DataFrame (n × n)

    Returns
    -------
    dict
        - is_symmetric: 대칭 여부
        - is_psd: positive semi-definite 여부
        - condition_number: 조건수 (log10)
        - min_eigenvalue: 최소 고유값
        - max_eigenvalue: 최대 고유값
        - eigenvalue_ratio: max / min (안정성 지표)

    사용 예시
    ---------
    info = diagnose_sigma(Sigma_monthly)
    if not info['is_psd']:
        print('⚠️ Σ 가 PSD 아님 — LedoitWolf shrinkage 적용 권고')
    """
    Sig = Sigma.values
    eigvals = np.linalg.eigvalsh(Sig)
    return {
        'is_symmetric': bool(np.allclose(Sig, Sig.T)),
        'is_psd': bool((eigvals > -1e-10).all()),
        'condition_number_log10': float(np.log10(np.abs(eigvals[-1] / eigvals[0]))),
        'min_eigenvalue': float(eigvals[0]),
        'max_eigenvalue': float(eigvals[-1]),
        'eigenvalue_ratio': float(eigvals[-1] / max(eigvals[0], 1e-10)),
    }
