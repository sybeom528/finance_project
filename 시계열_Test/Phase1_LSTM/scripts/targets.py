"""Phase 1 — 타깃 시계열 생성 및 누수 검증 유틸리티.

공개 인터페이스
--------------
build_daily_target_21d(adj_close)                       → pd.Series
verify_no_leakage(log_ret, target, n_checks, seed)      → None  (assert 기반)
build_leaky_target_for_test(adj_close)                  → pd.Series  (인공 누수용)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def verify_no_leakage(
    log_ret: pd.Series,
    target: pd.Series,
    n_checks: int = 3,
    seed: int = 42,
) -> None:
    """타깃 시계열의 데이터 누수를 2단계로 검증한다.

    검증 1 — Assert 단위 테스트
        ``n_checks`` 개 무작위 시점 t 에 대해
        ``target[t] == log_ret[t+1:t+22].sum()`` 을 assert.

    검증 2 — 육안 확인 표
        첫 5개 유효 행의 (날짜, log_ret, target, 직접계산, 일치 여부) 출력.

    Parameters
    ----------
    log_ret : pd.Series
        일별 log-return (분석 기간 내, NaN 없음).
    target : pd.Series
        build_daily_target_21d() 반환값.
    n_checks : int
        무작위 검증 시점 수.
    seed : int
        재현성 시드.

    Raises
    ------
    AssertionError
        검증 불통과 시 즉시 중단 — 누수 의심 상황.
    ValueError
        유효 인덱스 수가 n_checks 보다 적을 때.
    """
    rng = np.random.default_rng(seed)

    # 유효 인덱스: NaN 없고 forward 21일이 확보된 위치
    valid_pos = [
        i for i in range(len(target))
        if (not np.isnan(target.iloc[i])) and (i + 21 < len(log_ret))
    ]
    if len(valid_pos) < n_checks:
        raise ValueError(
            f"유효 인덱스 {len(valid_pos)}개 < n_checks {n_checks}. "
            "데이터 기간을 확인하십시오."
        )

    chosen = sorted(rng.choice(valid_pos, size=n_checks, replace=False))

    print("=== 누수 검증 1 — Assert 단위 테스트 ===")
    for pos in chosen:
        t = target.index[pos]
        expected = log_ret.iloc[pos + 1 : pos + 22].sum()
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
        direct = log_ret.iloc[pos + 1 : pos + 22].sum()
        match = "O" if abs(tgt - direct) < 1e-10 else "X"
        print(
            f"  {str(t.date()):>12}  {lr:>10.6f}  "
            f"{tgt:>10.6f}  {direct:>10.6f}  {match:>4}"
        )
    print()
    print("[OK] 누수 검증 완료 — 모든 체크포인트 PASS")


def build_leaky_target_for_test(adj_close: pd.Series) -> pd.Series:
    """인공 누수 타깃을 생성한다 — §4 대조 실험 전용.

    당일 log_return 자체를 타깃으로 사용한다.
    입력 시퀀스의 마지막 값과 타깃이 동일하므로 완전한 미래 누수 상황이다.
    이 타깃으로 학습한 모델의 R² 가 0.9 를 넘으면 평가 파이프라인이 정상임을 확인한다.
    R² 가 낮으면 평가 코드 자체에 버그가 있음을 의미한다.

    Parameters
    ----------
    adj_close : pd.Series
        수정 종가 시계열.

    Returns
    -------
    pd.Series
        target_leaky[t] = log_return[t]  (당일 수익률 — 입력과 동일한 값)
    """
    log_ret = np.log(adj_close).diff()   # 누수: 의도적 누수 — 실험용
    return log_ret
