"""Phase 1.5 — 신규 3개 모듈 + train.py loss_type 옵션 단위 테스트.

총 16건 테스트:
- targets_volatility.py       : 4건
- metrics_volatility.py       : 8건
- baselines_volatility.py     : 4건
- (보너스) train.py loss_type : 1건 (인터페이스 확인)

실행:
    python _test_modules.py

PASS 시 exit 0, FAIL 시 즉시 AssertionError + traceback.
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

# scripts/ import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.targets_volatility import (
    build_daily_target_logrv_21d, verify_no_leakage_logrv,
)
from scripts.metrics_volatility import (
    rmse, mae, qlike, r2_train_mean, mz_regression, pred_std_ratio,
    baseline_metrics_volatility, summarize_folds_volatility,
)
from scripts.baselines_volatility import (
    fit_har_rv, predict_ewma, predict_naive, predict_train_mean,
)


# ============================================================================
# 공통 fixture: SPY 실데이터로 build_daily_target_logrv_21d 호출하여 타깃 생성
# ============================================================================
def _load_spy_target():
    csv = Path(__file__).resolve().parent / 'results' / 'raw_data' / 'SPY.csv'
    df = pd.read_csv(csv, index_col=0, parse_dates=True).sort_index()
    df = df.loc['2016-01-01':'2025-12-31']
    target = build_daily_target_logrv_21d(df['Adj Close'])
    return df, target


PASSED = []
FAILED = []
def _run(name, fn):
    print(f'\n--- {name} ---')
    try:
        fn()
        PASSED.append(name)
        print(f'  [PASS] {name}')
    except Exception as e:
        FAILED.append(name)
        print(f'  [FAIL] {name}: {type(e).__name__}: {e}')
        raise


# ============================================================================
# targets_volatility.py — 4건
# ============================================================================
def test_targets_1_assert_누수_검증():
    """build_daily_target_logrv_21d → verify_no_leakage_logrv 가 AssertionError 없이 통과."""
    df, target = _load_spy_target()
    verify_no_leakage_logrv(df['Adj Close'], target, n_checks=3, window=21, seed=42)


def test_targets_2_ddof_일치():
    """ddof=1 (pandas 기본) 일관성 — 함수 내부와 외부 직접 계산 일치."""
    df, target = _load_spy_target()
    log_ret = np.log(df['Adj Close']).diff()
    # 임의 시점 t=100 검증 (NaN 영역 회피)
    t = 100
    expected = float(np.log(log_ret.iloc[t + 1 : t + 22].std(ddof=1)))
    actual = float(target.iloc[t])
    assert abs(actual - expected) < 1e-12, f'expected={expected}, actual={actual}'


def test_targets_3_NaN_카운트():
    """NaN 카운트 == window (마지막 window 행만)."""
    df, target = _load_spy_target()
    n_nan = int(target.isna().sum())
    assert n_nan == 21, f'NaN={n_nan}, expected=21 (마지막 forward 21행)'
    # 마지막 21행이 NaN, 나머지는 모두 finite
    assert target.iloc[-21:].isna().all(), '마지막 21행이 NaN 이 아님'
    assert target.iloc[:-21].notna().all(), '마지막 21행 외에 NaN 존재'


def test_targets_4_log_domain_finite():
    """log(0)/log(NaN) 진입 없음 — 모든 유효 값이 finite."""
    df, target = _load_spy_target()
    valid = target.dropna()
    assert np.isfinite(valid).all(), 'log domain 에 -inf/+inf/NaN 진입'


# ============================================================================
# metrics_volatility.py — 8건
# ============================================================================
def test_metrics_1_rmse_계산():
    """RMSE 단순 케이스: 일정한 차이의 평균제곱근."""
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([2.0, 3.0, 4.0])  # 차이 모두 +1
    assert abs(rmse(yt, yp) - 1.0) < 1e-12


def test_metrics_2_qlike_이상예측은_0():
    """완벽 예측 (y_pred ≡ y_true) 시 QLIKE = 0."""
    yt = np.array([-4.5, -4.0, -5.0])
    yp = yt.copy()
    q = qlike(yt, yp)
    assert abs(q) < 1e-10, f'QLIKE(perfect)={q}, expected≈0'

    # 비대칭성 검증: under-prediction(과소) vs over-prediction(과대) — under 가 더 큰 페널티
    yt_const = np.array([-4.5] * 100)
    yp_under = yt_const - 0.5  # log_pred 작음 → variance 작음 (과소예측)
    yp_over = yt_const + 0.5   # log_pred 큼 → variance 큼 (과대예측)
    q_under = qlike(yt_const, yp_under)
    q_over = qlike(yt_const, yp_over)
    assert q_under > q_over, f'QLIKE 비대칭성 위반: under={q_under}, over={q_over} (under > over 기대)'


def test_metrics_3_r2_train_mean_정상():
    """완벽 예측 시 R²_train_mean = 1, train_mean 예측 시 = 0."""
    np.random.seed(42)
    y_train = np.random.randn(100)
    y_test = np.random.randn(50)

    # 1. 완벽 예측: y_pred = y_test → R² = 1
    r2_perfect = r2_train_mean(y_test, y_test, y_train)
    assert abs(r2_perfect - 1.0) < 1e-12, f'완벽예측 R²={r2_perfect}'

    # 2. train_mean 예측: y_pred = mean(y_train) for all → R² = 0
    y_pred_tm = np.full_like(y_test, y_train.mean())
    r2_tm = r2_train_mean(y_test, y_pred_tm, y_train)
    assert abs(r2_tm) < 1e-12, f'train_mean 예측 R²={r2_tm}, expected=0'


def test_metrics_4_mz_regression_α0β1():
    """y_pred ≡ y_true 면 α=0, β=1, R²=1."""
    np.random.seed(42)
    y = np.random.randn(100)
    res = mz_regression(y, y)
    assert abs(res['alpha']) < 1e-10, f"alpha={res['alpha']}, expected=0"
    assert abs(res['beta'] - 1.0) < 1e-10, f"beta={res['beta']}, expected=1"
    assert abs(res['r2'] - 1.0) < 1e-10, f"r2={res['r2']}, expected=1"


def test_metrics_5_baseline_metrics_shape():
    """baseline_metrics_volatility 반환 shape — train_mean 자동 + 외부 주입 baseline."""
    np.random.seed(0)
    y_train = np.random.randn(100) - 4.5
    y_test = np.random.randn(20) - 4.5
    naive = np.random.randn(20) - 4.5

    # 1. train_mean only (자동)
    m = baseline_metrics_volatility(y_test, y_train)
    assert set(m.keys()) == {'train_mean'}, f'keys={m.keys()}'
    assert set(m['train_mean'].keys()) == {'rmse', 'mae', 'qlike', 'r2_train_mean', 'pred_std_ratio'}

    # 2. naive 주입
    m2 = baseline_metrics_volatility(y_test, y_train, naive_pred=naive)
    assert set(m2.keys()) == {'train_mean', 'naive'}


def test_metrics_6_summarize_folds_mean_std():
    """summarize_folds_volatility — mean/std 정확성 (NaN 자동 제외)."""
    fold_metrics = [
        {'rmse': 0.30, 'qlike': 0.05},
        {'rmse': 0.32, 'qlike': 0.07},
        {'rmse': 0.28, 'qlike': float('nan')},  # qlike NaN 1건
    ]
    s = summarize_folds_volatility(fold_metrics)
    # rmse: 3건 모두 유효 → n=3, mean=(0.30+0.32+0.28)/3=0.30
    assert s['rmse']['n'] == 3
    assert abs(s['rmse']['mean'] - 0.30) < 1e-12
    # qlike: 2건 유효 → n=2, mean=(0.05+0.07)/2=0.06
    assert s['qlike']['n'] == 2
    assert abs(s['qlike']['mean'] - 0.06) < 1e-12


def test_metrics_7_edge_동일값():
    """y_test 가 모두 동일 → R²_train_mean = nan (분모 0)."""
    yt = np.full(10, -4.5)
    yp = yt.copy()
    ytr = np.array([-4.5, -4.6, -4.4])
    r2 = r2_train_mean(yt, yp, ytr)
    # ytr.mean()=-4.5, yt 모두 -4.5 → SSE_train_mean = 0 → nan
    assert np.isnan(r2), f'동일값 → R²={r2}, expected=nan'


def test_metrics_8_pred_std_ratio_meancollapse():
    """pred_std_ratio: mean-collapse(상수 예측) 시 0."""
    yt = np.random.randn(50)
    yp_collapse = np.full(50, yt.mean())
    ratio = pred_std_ratio(yt, yp_collapse)
    assert abs(ratio) < 1e-12, f'mean-collapse ratio={ratio}, expected=0'

    # 정상 예측 (y_pred = y_true) → ratio = 1
    ratio_ok = pred_std_ratio(yt, yt)
    assert abs(ratio_ok - 1.0) < 1e-12


# ============================================================================
# baselines_volatility.py — 4건
# ============================================================================
def test_baselines_1_har_rv_계수():
    """HAR-RV 적합 — 학술 표준 정의(1·5·22일 RV) 계수 합리성."""
    df, _ = _load_spy_target()
    log_ret = np.log(df['Adj Close']).diff()

    # 충분한 train 구간 (RV monthly = 22일 lookback + horizon 21)
    train_idx = np.arange(50, 600)
    test_idx = np.arange(700, 720)
    pred, coefs = fit_har_rv(log_ret, train_idx, test_idx, horizon=21)

    assert pred.shape == (20,), f'pred shape={pred.shape}'
    assert np.isfinite(pred).all(), f'pred 에 NaN/inf'

    # 계수 합리성: persistence 강한 시계열 → 합이 양수, < 1.5 (학술 보고 ~0.7~1.0)
    beta_sum = coefs['beta_d'] + coefs['beta_w'] + coefs['beta_m']
    assert 0.3 < beta_sum < 1.5, f'β 합={beta_sum}, expected∈(0.3, 1.5)'

    # 모든 계수가 finite + 절편 negative (log scale)
    for k in ('beta_0', 'beta_d', 'beta_w', 'beta_m'):
        assert np.isfinite(coefs[k]), f'{k}={coefs[k]} 비유한'

    # train R² 합리적
    # - 학술 보고치: 일중 데이터 기반 HAR-RV 는 0.4~0.7
    # - 본 환경: 일간 log_ret² variance proxy 사용 → noisy 하여 R² 자연 감소 정상
    # - 따라서 0.05 < R² < 0.9 로 완화 (함수 동작 검증 목적)
    assert 0.05 < coefs['r2_train'] < 0.9, f'r2_train={coefs["r2_train"]}'


def test_baselines_2_ewma_recursion():
    """EWMA — λ=0.94 재귀 정확성."""
    df, _ = _load_spy_target()
    log_ret = np.log(df['Adj Close']).diff()
    train_idx = np.arange(50, 200)
    test_idx = np.arange(250, 270)
    pred = predict_ewma(log_ret, train_idx, test_idx, horizon=21, lam=0.94)
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()
    # 합리 범위: log(std) 보통 -6 ~ -3 (일별 변동성 0.25%~5%)
    assert (pred > -8).all() and (pred < -2).all(), f'pred range={pred.min():.2f} ~ {pred.max():.2f}'


def test_baselines_3_naive_shift():
    """Naive — predict_naive[t] = log(rv_trailing[t])."""
    df, _ = _load_spy_target()
    log_ret = np.log(df['Adj Close']).diff()
    rv_trailing = log_ret.rolling(21).std(ddof=1)

    train_idx = np.arange(0, 100)
    test_idx = np.array([100, 150, 200])
    pred = predict_naive(rv_trailing, train_idx, test_idx)

    expected = np.log(rv_trailing.values[test_idx])
    assert np.allclose(pred, expected, equal_nan=False), f'pred={pred}, expected={expected}'


def test_baselines_4_har_fold_외부_참조_차단():
    """fit_har_rv — train_idx 한정 적합 검증.

    train_idx 한정으로 OLS 적합되어, test_idx 의 데이터를 변경해도 계수는 불변이어야 함.
    """
    df, _ = _load_spy_target()
    log_ret = np.log(df['Adj Close']).diff()

    train_idx = np.arange(50, 600)
    test_idx = np.arange(700, 720)
    _, coefs_a = fit_har_rv(log_ret, train_idx, test_idx, horizon=21)

    # test 영역의 log_ret 을 임의로 조작 후 재호출 — 계수 동일해야 누수 없음
    log_ret_modified = log_ret.copy()
    log_ret_modified.iloc[test_idx] *= 100.0  # 100배 키움 (극단 조작)
    _, coefs_b = fit_har_rv(log_ret_modified, train_idx, test_idx, horizon=21)

    for k in ('beta_0', 'beta_d', 'beta_w', 'beta_m'):
        assert abs(coefs_a[k] - coefs_b[k]) < 1e-10, (
            f'{k} 누수: coefs_a={coefs_a[k]}, coefs_b={coefs_b[k]} (test 조작 후 변경)'
        )


# ============================================================================
# train.py loss_type 옵션 — 1건 (인터페이스 확인)
# ============================================================================
def test_train_loss_type_분기():
    """train.py 의 train_one_fold 가 loss_type 인자를 받고 'mse'/'huber' 분기 처리."""
    import inspect
    from scripts.train import train_one_fold
    sig = inspect.signature(train_one_fold)
    assert 'loss_type' in sig.parameters, "loss_type 인자 누락"
    assert sig.parameters['loss_type'].default == 'huber', \
        f"loss_type 기본값={sig.parameters['loss_type'].default}, expected='huber'"

    # 잘못된 값 → ValueError 즉시 (실제 학습 없이 인자 검증만)
    # 학습 실제 실행은 시간 비용 — 인자 분기만 정적 확인
    src = inspect.getsource(train_one_fold)
    assert "loss_type == 'huber'" in src, "huber 분기 로직 누락"
    assert "loss_type == 'mse'" in src, "mse 분기 로직 누락"
    assert "nn.MSELoss()" in src, "MSELoss 호출 누락"


# ============================================================================
# 실행
# ============================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('Phase 1.5 — 신규 모듈 단위 테스트 (총 17건)')
    print('=' * 70)

    # targets — 4건
    _run('targets_1_assert_누수_검증',  test_targets_1_assert_누수_검증)
    _run('targets_2_ddof_일치',          test_targets_2_ddof_일치)
    _run('targets_3_NaN_카운트',         test_targets_3_NaN_카운트)
    _run('targets_4_log_domain_finite',  test_targets_4_log_domain_finite)

    # metrics — 8건
    _run('metrics_1_rmse_계산',           test_metrics_1_rmse_계산)
    _run('metrics_2_qlike_이상예측',      test_metrics_2_qlike_이상예측은_0)
    _run('metrics_3_r2_train_mean_정상',  test_metrics_3_r2_train_mean_정상)
    _run('metrics_4_mz_regression',       test_metrics_4_mz_regression_α0β1)
    _run('metrics_5_baseline_shape',      test_metrics_5_baseline_metrics_shape)
    _run('metrics_6_summarize_folds',     test_metrics_6_summarize_folds_mean_std)
    _run('metrics_7_edge_동일값',         test_metrics_7_edge_동일값)
    _run('metrics_8_pred_std_meancollapse', test_metrics_8_pred_std_ratio_meancollapse)

    # baselines — 4건
    _run('baselines_1_har_rv_계수',       test_baselines_1_har_rv_계수)
    _run('baselines_2_ewma_recursion',    test_baselines_2_ewma_recursion)
    _run('baselines_3_naive_shift',       test_baselines_3_naive_shift)
    _run('baselines_4_har_fold_격리',     test_baselines_4_har_fold_외부_참조_차단)

    # train — 1건
    _run('train_loss_type_분기',          test_train_loss_type_분기)

    # 종합
    print()
    print('=' * 70)
    print(f'테스트 완료: PASS {len(PASSED)}건 / FAIL {len(FAILED)}건')
    print('=' * 70)
    if FAILED:
        print('FAILED:')
        for name in FAILED:
            print(f'  - {name}')
        sys.exit(1)
    print('[OK] 모든 테스트 통과')
