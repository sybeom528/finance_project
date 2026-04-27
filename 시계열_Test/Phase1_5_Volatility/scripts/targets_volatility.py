"""Phase 1.5 — 변동성(Log-RV) 타깃 시계열 생성 및 누수 검증 유틸리티.

본 모듈은 Phase 1 의 ``scripts/targets.py`` 와 같은 역할을 하나, 타깃 정의가
다르므로 별도 파일로 분리한다. Phase 1 의 누적 수익률 타깃과 격리.

공개 인터페이스
--------------
build_daily_target_logrv_21d(adj_close, window=21)        → pd.Series  Phase 1.5 타깃
verify_no_leakage_logrv(adj_close, target,                 → None       assert + 육안 표
                         n_checks=3, window=21,
                         seed=42, ddof=1)

타깃 정의 (PLAN §2-2)
--------------------
``target[t] = log( std( log_ret[t+1 : t+window+1], ddof=1 ) )``

핵심 설계
---------
- ``log`` 변환 후 분포가 거의 정규에 근접 (skew 3.6 → 0.5, kurt 20 → 0.7) → MSE loss 정합
- ``exp(pred)`` 로 역변환 시 자동으로 양수 보장 (변동성 음수 불가 제약 충족)
- Corsi (2009) HAR-RV 계열 학술 문헌과 동일 도메인

누수 함정 5종 (PLAN §9)
----------------------
1. ``shift(-window)`` 부호 누락 → 미래 참조 (assert 검증)
2. ``np.log(0)`` 또는 ``np.log(NaN)`` → -inf/NaN 전파
3. ``ddof`` 불일치 (pandas 기본 1, numpy 기본 0) → 명시 통일
4. ``rolling`` 위치 (log 전후) → ``log(rolling().std())`` 패턴 통일
5. 마지막 window 행 NaN → ``dropna()`` 후 비교
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_daily_target_logrv_21d(
    adj_close: pd.Series, window: int = 21
) -> pd.Series:
    """21일 forward log-realized-volatility 타깃.

    공식
    ----
    ``log_ret[t] = log(adj_close[t]) - log(adj_close[t-1])``
    ``rv_trailing[t] = std(log_ret[t-window+1 : t+1], ddof=1)``  (pandas 기본)
    ``target[t] = log(rv_trailing[t+window])``  ( ``shift(-window)`` 후 log )

    즉 시점 t 의 타깃은 ``log(std(log_ret[t+1 : t+window+1]))`` 로
    "향후 ``window`` 영업일 동안 시장이 얼마나 출렁일 것인가" 의 로그 변환.

    누수 주석
    ---------
    - ``np.log(adj_close).diff()`` : 현재·직전 가격만 사용 — 안전.
    - ``rolling(window).std()`` : trailing window — 안전.
    - ``np.log(rv)`` : 변환 — 미래 참조 없음.
    - ``.shift(-window)`` : ``window`` 만큼 인덱스를 앞으로 당김 →
      시점 t 의 값이 t+``window`` 의 trailing RV 가 됨 = 미래 ``window`` 일 RV (예측 목표).
      Walk-Forward Purge+Embargo 로 학습 시 누수 차단 (PLAN §7-6).

    Parameters
    ----------
    adj_close : pd.Series
        수정 종가 시계열 (DatetimeIndex).
    window : int, default 21
        rolling 윈도우 길이 (영업일).

    Returns
    -------
    pd.Series
        ``target[t] = log( std(log_ret[t+1 : t+window+1], ddof=1) )``.
        NaN: 마지막 ``window`` 행 (forward 기간 부족) — 정상 동작.

    See Also
    --------
    verify_no_leakage_logrv : 본 함수 결과의 누수 검증.

    References
    ----------
    Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003).
    Modeling and forecasting realized volatility. *Econometrica, 71*(2), 579-625.
    """
    log_ret = np.log(adj_close).diff()                  # 누수: 현재·직전만 사용
    rv_trailing = log_ret.rolling(window).std(ddof=1)    # 누수: trailing window
    # 누수: forward shift → 시점 t target 이 미래 window 일 RV 가 되어 예측 목표 형성
    target = np.log(rv_trailing).shift(-window)
    return target


def verify_no_leakage_logrv(
    adj_close: pd.Series,
    target: pd.Series,
    n_checks: int = 3,
    window: int = 21,
    seed: int = 42,
    ddof: int = 1,
) -> None:
    """Log-RV 타깃 누수 2단계 검증 (Phase 1 verify_no_leakage 패턴 동일).

    검증 1 — Assert 단위 테스트
        ``n_checks`` 개 무작위 시점 t 에 대해
        ``target[t] == log(std(log_ret[t+1 : t+window+1], ddof=1))`` 을 assert.

    검증 2 — 육안 확인 표
        첫 5개 유효 행의 (날짜, log_ret, target, 직접계산, 일치 여부) 출력.

    Parameters
    ----------
    adj_close : pd.Series
        수정 종가 시계열 (DatetimeIndex).
    target : pd.Series
        ``build_daily_target_logrv_21d(adj_close, window)`` 반환값.
    n_checks : int, default 3
        무작위 검증 시점 수.
    window : int, default 21
        타깃의 forward window 길이.
    seed : int, default 42
        재현성 시드.
    ddof : int, default 1
        std 계산 자유도. pandas rolling.std 기본 1 과 일치 통일.

    Raises
    ------
    AssertionError
        검증 불통과 시 즉시 중단 — 누수 의심 상황.
    ValueError
        유효 인덱스 수가 ``n_checks`` 보다 적을 때.
    """
    log_ret = np.log(adj_close).diff()
    rng = np.random.default_rng(seed)

    # 유효 인덱스: target 이 NaN 아니고 forward window 일이 확보된 위치
    valid_pos = [
        i for i in range(len(target))
        if (not np.isnan(target.iloc[i])) and (i + window < len(log_ret))
    ]
    if len(valid_pos) < n_checks:
        raise ValueError(
            f"유효 인덱스 {len(valid_pos)}개 < n_checks {n_checks}. "
            "데이터 기간을 확인하십시오."
        )

    chosen = sorted(rng.choice(valid_pos, size=n_checks, replace=False))

    print("=== Log-RV 누수 검증 1 — Assert 단위 테스트 ===")
    for pos in chosen:
        t = target.index[pos]
        future_lr = log_ret.iloc[pos + 1 : pos + 1 + window]
        # 누수: future_lr 은 시점 t 의 미래 — 검증용으로만 직접 계산하여 expected 도출
        expected = float(np.log(future_lr.std(ddof=ddof)))
        actual = float(target.iloc[pos])
        diff = abs(actual - expected)
        status = "PASS" if diff < 1e-10 else "FAIL"
        print(
            f"  [{status}] {str(t.date())}  "
            f"target={actual:.6f}  직접계산={expected:.6f}  Δ={diff:.2e}"
        )
        assert diff < 1e-10, (
            f"[Log-RV 누수 검증 FAIL] t={t}, target={actual:.6f}, expected={expected:.6f}"
        )
    print()

    print("=== Log-RV 누수 검증 2 — 육안 확인 표 (첫 5개 유효 행) ===")
    first5 = valid_pos[:5]
    print(f"  {'날짜':>12}  {'log_ret':>10}  {'target':>10}  {'직접계산':>10}  {'일치':>4}")
    print("  " + "-" * 54)
    for pos in first5:
        t = target.index[pos]
        lr = float(log_ret.iloc[pos]) if not np.isnan(log_ret.iloc[pos]) else float('nan')
        tgt = float(target.iloc[pos])
        future_lr = log_ret.iloc[pos + 1 : pos + 1 + window]
        direct = float(np.log(future_lr.std(ddof=ddof)))
        match = "O" if abs(tgt - direct) < 1e-10 else "X"
        lr_str = f"{lr:>10.6f}" if not np.isnan(lr) else f"{'NaN':>10}"
        print(
            f"  {str(t.date()):>12}  {lr_str}  "
            f"{tgt:>10.6f}  {direct:>10.6f}  {match:>4}"
        )
    print()
    print("[OK] Log-RV 누수 검증 완료 — 모든 체크포인트 PASS")
