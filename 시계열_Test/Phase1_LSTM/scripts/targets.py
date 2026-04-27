"""Phase 1 — 타깃 시계열 생성 및 누수 검증 유틸리티.

공개 인터페이스
--------------
build_daily_target(adj_close, horizon)                  → pd.Series  horizon일 누적 forward log-return
build_daily_target_21d(adj_close)                       → pd.Series  설정 A 타깃 (horizon=21)
build_daily_target_14d(adj_close)                       → pd.Series  설정 A-14 타깃 (horizon=14)
build_monthly_target_1m(adj_close)                      → pd.Series  설정 B 타깃
verify_no_leakage(log_ret, target, n_checks, seed,
                  horizon)                              → None       assert + 육안 표
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_daily_target(adj_close: pd.Series, horizon: int = 21) -> pd.Series:
    """horizon일 누적 forward log-return 타깃을 생성한다.

    공식
    ----
    log_ret[t] = log(adj_close[t]) - log(adj_close[t-1])
    target[t]  = log_ret[t+1] + ... + log_ret[t+horizon]
               = log_ret.rolling(horizon).sum().shift(-horizon)[t]

    Parameters
    ----------
    adj_close : pd.Series
        수정 종가 시계열 (DatetimeIndex).
    horizon : int, default=21
        예측 horizon (영업일). Purge 기간과 동일하게 설정해야 누수 없음.

    Returns
    -------
    pd.Series
        target[t] = horizon일 누적 forward log-return.
        NaN: 첫 1행 (log_ret diff) + 마지막 horizon행 (forward 기간 부족).
    """
    log_ret = np.log(adj_close).diff()
    target = log_ret.rolling(horizon).sum().shift(-horizon)
    return target


def build_daily_target_21d(adj_close: pd.Series) -> pd.Series:
    """21일 누적 forward log-return 타깃을 생성한다.

    공식
    ----
    log_ret[t] = log(adj_close[t]) - log(adj_close[t-1])
    target[t]  = log_ret[t+1] + log_ret[t+2] + ... + log_ret[t+21]
               = log_ret.rolling(21).sum().shift(-21)[t]

    누수 주석
    ---------
    - ``diff()``        : 현재·직전 값만 사용, 미래 참조 없음 — 안전
    - ``.rolling(21)``  : trailing (과거 21일 합) — 안전
    - ``.shift(-21)``   : t 시점 값을 21일 앞으로 당김 → t 시점 target = 미래 21일 합
                          이것이 예측 목표이므로 정의상 미래값을 포함하지만 누수가 아님.
                          단, 모델 입력이 반드시 t 이전 데이터여야 함
                          (§6 Walk-Forward Purge+Embargo 로 보장).

    Parameters
    ----------
    adj_close : pd.Series
        수정 종가 시계열 (DatetimeIndex).

    Returns
    -------
    pd.Series
        target[t] = 21일 누적 forward log-return.
        NaN: 첫 1행 (log_ret diff 계산) + 마지막 21행 (forward 기간 부족).
    """
    log_ret = np.log(adj_close).diff()                      # 누수: trailing diff
    target = log_ret.rolling(21).sum().shift(-21)            # 누수: forward 21일 합 (예측 목표)
    return target


def build_daily_target_14d(adj_close: pd.Series) -> pd.Series:
    """14일 누적 forward log-return 타깃을 생성한다 (설정 A-14 용).

    ``build_daily_target(adj_close, horizon=14)`` 의 편의 래퍼.
    NaN: 첫 1행 + 마지막 14행.
    """
    return build_daily_target(adj_close, horizon=14)


def build_monthly_target_1m(adj_close: pd.Series) -> pd.Series:
    """1개월 후 log-return 타깃을 생성한다 (설정 B 용).

    공식
    ----
    monthly_close[m] = adj_close.resample('ME').last()[m]   # 월말 종가
    log_ret_m[m]     = log(monthly_close[m]) - log(monthly_close[m-1])
    target[m]        = log_ret_m[m+1]   (= log_ret_m.shift(-1)[m])

    즉 위치 m (월말) 의 타깃은 m → m+1 한 달 동안의 log-return.

    누수 주석
    ---------
    - ``resample('ME').last()`` : 각 월의 마지막 거래일 종가만 사용 — 미래 참조 없음.
    - ``diff()``                : 현재·직전 월만 사용 — 안전.
    - ``.shift(-1)``            : 위치 m 의 값을 m-1 로 당김
                                  → 위치 m 의 타깃은 m → m+1 수익률이 되어 예측 목표 형성.

    Parameters
    ----------
    adj_close : pd.Series
        수정 종가 시계열 (DatetimeIndex 필수).

    Returns
    -------
    pd.Series
        월말 인덱스 (resample 'ME'). 마지막 1행 NaN (다음 달 데이터 부족).

    Notes
    -----
    설정 B (월별 1개월 예측) 의 정식 타깃 빌더.
    설정 A (일별) 와 달리 누적이 아닌 단일 기간 수익률이므로 purge/embargo 는
    월 단위 1개월로 축소되어 적용된다 (PLAN.md 설정 B 파라미터 참고).
    """
    # 누수: 월말 종가 기준 — look-ahead 없음
    monthly_close = adj_close.resample('ME').last()
    log_ret_monthly = np.log(monthly_close).diff()           # 첫 1행 NaN
    # 누수: shift(-1) 로 위치 m 에 (m → m+1) 다음 달 수익률 배치 (예측 목표)
    target = log_ret_monthly.shift(-1)
    return target


def verify_no_leakage(
    log_ret: pd.Series,
    target: pd.Series,
    n_checks: int = 3,
    seed: int = 42,
    horizon: int = 21,
) -> None:
    """타깃 시계열의 데이터 누수를 2단계로 검증한다.

    검증 1 — Assert 단위 테스트
        ``n_checks`` 개 무작위 시점 t 에 대해
        ``target[t] == log_ret[t+1:t+1+horizon].sum()`` 을 assert.

    검증 2 — 육안 확인 표
        첫 5개 유효 행의 (날짜, log_ret, target, 직접계산, 일치 여부) 출력.

    Parameters
    ----------
    log_ret : pd.Series
        일별 log-return (분석 기간 내, NaN 없음).
    target : pd.Series
        build_daily_target() 반환값.
    n_checks : int
        무작위 검증 시점 수.
    seed : int
        재현성 시드.
    horizon : int, default=21
        타깃 생성 시 사용한 horizon (일). build_daily_target의 horizon과 일치해야 함.

    Raises
    ------
    AssertionError
        검증 불통과 시 즉시 중단 — 누수 의심 상황.
    ValueError
        유효 인덱스 수가 n_checks 보다 적을 때.
    """
    rng = np.random.default_rng(seed)

    # 유효 인덱스: NaN 없고 forward horizon일이 확보된 위치
    valid_pos = [
        i for i in range(len(target))
        if (not np.isnan(target.iloc[i])) and (i + horizon < len(log_ret))
    ]
    if len(valid_pos) < n_checks:
        raise ValueError(
            f"유효 인덱스 {len(valid_pos)}개 < n_checks {n_checks}. "
            "데이터 기간을 확인하십시오."
        )

    chosen = sorted(rng.choice(valid_pos, size=n_checks, replace=False))

    print(f"=== 누수 검증 1 — Assert 단위 테스트 (horizon={horizon}) ===")
    for pos in chosen:
        t = target.index[pos]
        expected = log_ret.iloc[pos + 1 : pos + 1 + horizon].sum()
        actual = float(target.iloc[pos])
        diff = abs(actual - expected)
        status = "PASS" if diff < 1e-10 else "FAIL"
        print(
            f"  [{status}] {str(t.date())}  "
            f"target={actual:.6f}  직접계산={expected:.6f}  Δ={diff:.2e}"
        )
        assert diff < 1e-10, (
            f"[누수 검증 FAIL] t={t}, target={actual:.6f}, expected={expected:.6f}"
        )
    print()

    print("=== 누수 검증 2 — 육안 확인 표 (첫 5개 유효 행) ===")
    first5 = valid_pos[:5]
    print(f"  {'날짜':>12}  {'log_ret':>10}  {'target':>10}  {'직접계산':>10}  {'일치':>4}")
    print("  " + "-" * 54)
    for pos in first5:
        t = target.index[pos]
        lr = float(log_ret.iloc[pos])
        tgt = float(target.iloc[pos])
        direct = log_ret.iloc[pos + 1 : pos + 1 + horizon].sum()
        match = "O" if abs(tgt - direct) < 1e-10 else "X"
        print(
            f"  {str(t.date()):>12}  {lr:>10.6f}  "
            f"{tgt:>10.6f}  {direct:>10.6f}  {match:>4}"
        )
    print()
    print("[OK] 누수 검증 완료 — 모든 체크포인트 PASS")
