"""
패널 버전 Step3 핵심 로직 단위 테스트
  - build_panel(), make_cs_quintile_labels() 검증
  - train_xgb() 패널 데이터 학습 검증
  - Walk-Forward 2 window 전체 흐름 검증
  (TabPFN v2 는 토큰 필요 → 스킵)
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── 경로 ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("C:/Users/gorhk/최종 프로젝트/finance_project/김재천/black_litterman")
DATA_DIR = BASE_DIR / "data"

IT_TICKERS    = ["MSFT", "INTC", "ORCL", "AAPL", "CSCO",
                 "IBM", "QCOM", "TXN", "CRM", "ADBE"]
FEATURE_COLS  = [
    "log_return_1d", "simple_return_1d",
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12m_skip_1m",
    "vol_20d_ann", "vol_60d_ann", "vol_252d_ann",
    "mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom_factor",
]
TARGET_COL   = "fwd_ret_21d"
DATA_START   = pd.Timestamp("2020-12-01")
DATA_END     = pd.Timestamp("2025-12-31")
IS_DAYS      = 252
EMBARGO_DAYS = 21
OOS_DAYS     = 21
STEP_SIZE    = 21
N_CLASSES    = 5
RANDOM_STATE = 42


# ─── 함수 (노트북 cell-functions 와 동일) ─────────────────────────────────────

def load_ticker_csv(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / "panels" / f"{ticker}.csv"
    df   = pd.read_csv(path, index_col="date", parse_dates=True)
    df   = df[(df.index >= DATA_START) & (df.index <= DATA_END)].copy()
    df.sort_index(inplace=True)
    df[TARGET_COL] = df["adj_close"].shift(-OOS_DAYS) / df["adj_close"] - 1
    return df


def make_wf_windows(
    dates, is_days=IS_DAYS, embargo=EMBARGO_DAYS, oos_days=OOS_DAYS, step=STEP_SIZE
):
    windows, n, i = [], len(dates), 0
    while True:
        is_end    = i + is_days
        oos_start = is_end + embargo
        oos_end   = oos_start + oos_days
        if oos_end > n:
            break
        purge = is_end - oos_days
        is_w  = dates[i:purge]
        oos_w = dates[oos_start:oos_end]
        if len(is_w) > 0 and len(oos_w) > 0:
            windows.append((is_w, oos_w))
        i += step
    return windows


def build_panel(ticker_data, dates, require_target=True):
    frames = []
    for ticker, df in ticker_data.items():
        sub = df.loc[df.index.isin(dates)].copy()
        if require_target:
            sub = sub.dropna(subset=FEATURE_COLS + [TARGET_COL])
        sub["ticker"] = ticker
        frames.append(sub)
    return pd.concat(frames).sort_index()


def make_cs_quintile_labels(df_is):
    def _quintile_per_date(group):
        if len(group) < N_CLASSES:
            return pd.Series(0, index=group.index)
        return pd.qcut(
            group[TARGET_COL].rank(method="first"),
            q=N_CLASSES, labels=False
        ).astype(int)

    labels_s = (
        df_is.groupby(df_is.index, group_keys=False)
             .apply(_quintile_per_date)
    )
    labels_s = labels_s.reindex(df_is.index).fillna(0).astype(int)
    r_bar    = {
        k: float(df_is.loc[labels_s == k, TARGET_COL].mean())
        for k in range(N_CLASSES)
    }
    return labels_s.values, r_bar


def compute_Q_Omega(proba, r_bar, n_classes=N_CLASSES):
    r_arr = np.array([r_bar.get(k, 0.0) for k in range(n_classes)])
    Q     = proba @ r_arr
    Omega = np.sum(proba * (r_arr[np.newaxis, :] - Q[:, np.newaxis]) ** 2, axis=1)
    return Q, Omega


def _xgb_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators",   50, 200),
        "max_depth"        : trial.suggest_int("max_depth",       3,  6),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample",     0.6, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective"        : "multi:softprob",
        "num_class"        : N_CLASSES,
        "eval_metric"      : "mlogloss",
        "tree_method"      : "hist",
        "random_state"     : RANDOM_STATE,
        "n_jobs"           : -1,
        "verbosity"        : 0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    proba_val = model.predict_proba(X_val)
    return log_loss(y_val, proba_val, labels=list(range(N_CLASSES)))


def train_xgb(X_is, y_is, X_oos, window_id=0, n_trials=3):
    n_is  = len(X_is)
    split = int(n_is * 0.8)
    X_tr, X_val = X_is[:split], X_is[split:]
    y_tr, y_val = y_is[:split], y_is[split:]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE + window_id)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: _xgb_objective(trial, X_tr, y_tr, X_val, y_val),
        n_trials=n_trials, show_progress_bar=False,
    )
    best_params = study.best_params
    best_params.update({
        "objective"   : "multi:softprob",
        "num_class"   : N_CLASSES,
        "eval_metric" : "mlogloss",
        "tree_method" : "hist",
        "random_state": RANDOM_STATE,
        "n_jobs"      : -1,
        "verbosity"   : 0,
    })
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_is, y_is, verbose=False)
    return final_model.predict_proba(X_oos), best_params


# ─── 테스트 ──────────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0

def ok(msg):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  [PASS] {msg}")

def fail(msg):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  [FAIL] {msg}")


print("=" * 60)
print("TEST: 패널 Step3 핵심 로직 검증")
print("=" * 60)

# ─── Section 0: 데이터 로드 ──────────────────────────────────────────────────
print("\n[Section 0] 데이터 로드")
ticker_data = {}
for t in IT_TICKERS:
    ticker_data[t] = load_ticker_csv(t)
ok(f"10종목 CSV 로드 완료, 예시 {IT_TICKERS[0]}: {len(ticker_data[IT_TICKERS[0]])}행")

all_dates = list(ticker_data.values())[0].index
windows   = make_wf_windows(all_dates)
ok(f"Walk-forward 윈도우: {len(windows)}개")

# ─── Section 1: build_panel 검증 ─────────────────────────────────────────────
print("\n[Section 1] build_panel 검증")
is_w0, oos_w0 = windows[0]
df_is_p  = build_panel(ticker_data, is_w0,  require_target=True)
df_oos_p = build_panel(ticker_data, oos_w0, require_target=False)

if len(df_is_p) >= 252 * 5:
    ok(f"IS 패널 크기 충분: {len(df_is_p)}행 (기대 ≥ {252*5})")
else:
    fail(f"IS 패널 크기 부족: {len(df_is_p)}행")

if "ticker" in df_is_p.columns:
    ok("IS 패널에 'ticker' 컬럼 존재")
else:
    fail("IS 패널에 'ticker' 컬럼 없음")

n_tickers_in_panel = df_is_p["ticker"].nunique()
if n_tickers_in_panel == len(IT_TICKERS):
    ok(f"IS 패널에 10종목 모두 존재: {n_tickers_in_panel}개")
else:
    fail(f"IS 패널 종목 수 불일치: {n_tickers_in_panel} ≠ {len(IT_TICKERS)}")

if df_is_p[TARGET_COL].isna().sum() == 0:
    ok("IS 패널 target NaN 없음 (require_target=True 정상 작동)")
else:
    fail(f"IS 패널 target NaN 존재: {df_is_p[TARGET_COL].isna().sum()}개")

print(f"    IS 패널: {len(df_is_p)}행  |  OOS 패널: {len(df_oos_p)}행")

# ─── Section 2: make_cs_quintile_labels 검증 ─────────────────────────────────
print("\n[Section 2] make_cs_quintile_labels 검증")
y_is_cls, r_bar = make_cs_quintile_labels(df_is_p)

if len(y_is_cls) == len(df_is_p):
    ok(f"레이블 길이 일치: {len(y_is_cls)}")
else:
    fail(f"레이블 길이 불일치: {len(y_is_cls)} ≠ {len(df_is_p)}")

unique_cls = np.unique(y_is_cls)
if set(unique_cls) <= set(range(N_CLASSES)):
    ok(f"레이블 범위 정상: {sorted(unique_cls)}")
else:
    fail(f"레이블 범위 이상: {sorted(unique_cls)}")

if len(r_bar) == N_CLASSES:
    ok(f"r_bar 키 수 정상: {N_CLASSES}개")
    for k in range(N_CLASSES):
        print(f"    분위 {k}: r_bar={r_bar[k]:.5f}")
else:
    fail(f"r_bar 키 수 이상: {len(r_bar)}")

# 분위별 행 수 균등성 확인 (~20% per class)
_, counts = np.unique(y_is_cls, return_counts=True)
ratio = counts / len(y_is_cls)
if all(0.10 < r < 0.35 for r in ratio):
    ok(f"분위 분포 균등: {[f'{r:.2%}' for r in ratio]}")
else:
    fail(f"분위 분포 편중: {[f'{r:.2%}' for r in ratio]}")

# ─── Section 3: OOS 크로스 섹션 5분위 레이블 ─────────────────────────────────
print("\n[Section 3] OOS 크로스 섹션 5분위 레이블 검증")
y_oos_ret = df_oos_p[TARGET_COL].values
y_oos_cls = np.zeros(len(df_oos_p), dtype=int)

df_oos_pos         = df_oos_p.copy()
df_oos_pos["_pos"] = np.arange(len(df_oos_p))

for _, grp in df_oos_pos.groupby(df_oos_pos.index):
    valid_mask = grp[TARGET_COL].notna()
    if valid_mask.sum() < N_CLASSES:
        continue
    try:
        cls_grp = pd.qcut(
            grp.loc[valid_mask, TARGET_COL].rank(method="first"),
            q=N_CLASSES, labels=False,
        ).astype(int)
        y_oos_cls[grp.loc[valid_mask, "_pos"].values] = cls_grp.values
    except Exception:
        pass

if set(np.unique(y_oos_cls)) <= set(range(N_CLASSES)):
    ok(f"OOS 레이블 범위 정상: {sorted(np.unique(y_oos_cls))}")
else:
    fail(f"OOS 레이블 범위 이상")

# ─── Section 4: train_xgb 패널 데이터 학습 ───────────────────────────────────
print("\n[Section 4] train_xgb 패널 학습 (n_trials=3, 빠른 검증)")
X_is  = df_is_p[FEATURE_COLS].values.astype(float)
X_oos = df_oos_p[FEATURE_COLS].fillna(0).values.astype(float)

try:
    proba_xgb, best_params = train_xgb(X_is, y_is_cls, X_oos, window_id=0, n_trials=3)

    if proba_xgb.shape == (len(df_oos_p), N_CLASSES):
        ok(f"XGB proba shape 정상: {proba_xgb.shape}")
    else:
        fail(f"XGB proba shape 이상: {proba_xgb.shape}")

    if np.allclose(proba_xgb.sum(axis=1), 1.0, atol=1e-5):
        ok("XGB 확률 합 = 1.0 (정상)")
    else:
        fail("XGB 확률 합 ≠ 1.0")

    Q_xgb, Om_xgb = compute_Q_Omega(proba_xgb, r_bar)
    ok(f"Q_xgb 범위: [{Q_xgb.min():.5f}, {Q_xgb.max():.5f}]")
    ok(f"Omega_xgb 범위: [{Om_xgb.min():.7f}, {Om_xgb.max():.7f}]")

    valid_oos = ~np.isnan(y_oos_ret)
    if valid_oos.sum() > 0:
        ll = log_loss(y_oos_cls[valid_oos], proba_xgb[valid_oos],
                      labels=list(range(N_CLASSES)))
        ok(f"OOS LogLoss 계산 정상: {ll:.4f}")
    else:
        ok("OOS NaN만 있음 → LogLoss 스킵 (정상)")

except Exception as e:
    fail(f"train_xgb 오류: {e}")
    import traceback
    traceback.print_exc()

# ─── Section 5: 2 window 전체 루프 검증 ──────────────────────────────────────
print("\n[Section 5] Walk-Forward 2 window 전체 루프 검증")
import time
records_test = []
t0 = time.time()

for w_idx in range(2):
    is_dates, oos_dates = windows[w_idx]

    df_is  = build_panel(ticker_data, is_dates,  require_target=True)
    df_oos = build_panel(ticker_data, oos_dates, require_target=False)

    X_is  = df_is[FEATURE_COLS].values.astype(float)
    X_oos = df_oos[FEATURE_COLS].fillna(0).values.astype(float)

    y_is_cls, r_bar = make_cs_quintile_labels(df_is)

    # OOS 레이블
    y_oos_ret = df_oos[TARGET_COL].values
    y_oos_cls = np.zeros(len(df_oos), dtype=int)
    df_oos_pos         = df_oos.copy()
    df_oos_pos["_pos"] = np.arange(len(df_oos))
    for _, grp in df_oos_pos.groupby(df_oos_pos.index):
        valid_mask = grp[TARGET_COL].notna()
        if valid_mask.sum() < N_CLASSES:
            continue
        try:
            cls_grp = pd.qcut(
                grp.loc[valid_mask, TARGET_COL].rank(method="first"),
                q=N_CLASSES, labels=False,
            ).astype(int)
            y_oos_cls[grp.loc[valid_mask, "_pos"].values] = cls_grp.values
        except Exception:
            pass

    valid_oos = ~np.isnan(y_oos_ret)

    proba_xgb, _ = train_xgb(X_is, y_is_cls, X_oos, window_id=w_idx, n_trials=3)
    Q_xgb, Om_xgb = compute_Q_Omega(proba_xgb, r_bar)
    pred_cls_xgb   = np.argmax(proba_xgb, axis=1)
    ll_xgb = (
        log_loss(y_oos_cls[valid_oos], proba_xgb[valid_oos],
                 labels=list(range(N_CLASSES)))
        if valid_oos.sum() > 0 else np.nan
    )

    for j, (idx, row) in enumerate(df_oos.iterrows()):
        records_test.append({
            "window_id" : w_idx,
            "date"      : idx,
            "ticker"    : row["ticker"],
            "model"     : "xgb",
            "Q"         : float(Q_xgb[j]),
            "Omega"     : float(Om_xgb[j]),
            "pred_cls"  : int(pred_cls_xgb[j]),
            "logloss"   : ll_xgb,
            "actual_ret": float(y_oos_ret[j]),
            "true_cls"  : int(y_oos_cls[j]),
        })
    print(f"  Window {w_idx+1}/2: IS={len(df_is)}행 OOS={len(df_oos)}행 LogLoss={ll_xgb:.4f}")

elapsed = time.time() - t0
ok(f"2 window 완료: {len(records_test)}개 기록, {elapsed:.1f}초")

df_res_test = pd.DataFrame(records_test)
if df_res_test["ticker"].nunique() == len(IT_TICKERS):
    ok(f"records에 10종목 모두 포함: {sorted(df_res_test['ticker'].unique())}")
else:
    fail(f"종목 수 불일치: {df_res_test['ticker'].nunique()}")

# ─── 최종 결과 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"테스트 완료: PASS={PASS_COUNT}, FAIL={FAIL_COUNT}")
if FAIL_COUNT == 0:
    print("ALL SECTIONS PASS [OK] — 패널 로직 이상 없음")
else:
    print(f"FAIL {FAIL_COUNT}개 존재 → 수정 필요")
print("=" * 60)
