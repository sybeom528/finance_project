"""
Step3 검증 스크립트 — 2 windows, 3 tickers, 3 optuna trials
실제 노트북 코드와 동일한 로직을 소규모로 실행하여 오류 유무 확인
"""
import os, sys, warnings, time
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows cp949 출력 인코딩 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, log_loss

# ─── 경로 설정 ───────────────────────────────────────────────────────────────
def _find_base_dir() -> Path:
    cwd = Path(os.getcwd()).resolve()
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "data" / "panels").is_dir():
            return candidate
    return Path("C:/Users/gorhk/최종 프로젝트/finance_project/김재천/black_litterman")

BASE_DIR     = _find_base_dir()
NOTEBOOK_DIR = BASE_DIR / "Test"
DATA_DIR     = BASE_DIR / "data"
OUT_DIR      = NOTEBOOK_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR : {BASE_DIR}")
print(f"DATA_DIR : {DATA_DIR}")

# ─── 파라미터 (테스트용 축소) ─────────────────────────────────────────────
IT_TICKERS    = ["MSFT", "INTC", "ORCL"]   # 3개만
FEATURE_COLS  = [
    "log_return_1d", "simple_return_1d",
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12m_skip_1m",
    "vol_20d_ann", "vol_60d_ann", "vol_252d_ann",
    "mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom_factor",
]
TARGET_COL    = "fwd_ret_21d"
DATA_START    = pd.Timestamp("2020-12-01")
DATA_END      = pd.Timestamp("2025-12-31")
IS_DAYS       = 252
EMBARGO_DAYS  = 21
OOS_DAYS      = 21
STEP_SIZE     = 21
N_CLASSES     = 5
N_OPTUNA_TRIALS = 3      # 테스트용: 3 trials
RANDOM_STATE  = 42
MAX_WINDOWS   = 2        # 테스트용: 2 windows만


# ─── 함수 정의 ────────────────────────────────────────────────────────────────

def load_ticker_csv(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / "panels" / f"{ticker}.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df[(df.index >= DATA_START) & (df.index <= DATA_END)].copy()
    df.sort_index(inplace=True)
    df[TARGET_COL] = df["adj_close"].shift(-OOS_DAYS) / df["adj_close"] - 1
    return df


def make_wf_windows(dates, is_days=IS_DAYS, embargo=EMBARGO_DAYS,
                    oos_days=OOS_DAYS, step=STEP_SIZE):
    windows, n, i = [], len(dates), 0
    while True:
        is_end    = i + is_days
        oos_start = is_end + embargo
        oos_end   = oos_start + oos_days
        if oos_end > n:
            break
        purge = is_end - oos_days
        windows.append((dates[i:purge], dates[oos_start:oos_end]))
        i += step
    return windows


def make_quintile_labels(y_is):
    _, bins = pd.qcut(y_is.dropna(), q=N_CLASSES, retbins=True, duplicates="drop")
    bins[0], bins[-1] = -np.inf, np.inf
    labels = pd.cut(y_is, bins=bins, labels=False).fillna(0).astype(int)
    r_bar  = {k: float(y_is[labels == k].mean()) for k in range(N_CLASSES)}
    return labels.values, bins, r_bar


def apply_bins(y, bins):
    return pd.cut(y, bins=bins, labels=False).fillna(0).astype(int).values


def compute_Q_Omega(proba, r_bar, n_classes=N_CLASSES):
    r_arr = np.array([r_bar.get(k, 0.0) for k in range(n_classes)])
    Q     = proba @ r_arr
    Omega = np.sum(proba * (r_arr[np.newaxis, :] - Q[:, np.newaxis]) ** 2, axis=1)
    return Q, Omega


def _xgb_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators",   100, 300),
        "max_depth"        : trial.suggest_int("max_depth",       3,   5),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample",     0.6, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 5),
        "reg_lambda"       : trial.suggest_float("reg_lambda",    0.1, 5.0, log=True),
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
    # labels 명시 — 검증 분할에 일부 분위 없을 수 있음
    return log_loss(y_val, proba_val, labels=list(range(N_CLASSES)))


def train_xgb(X_is, y_is, X_oos, window_id=0, n_trials=N_OPTUNA_TRIALS):
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
        "objective"    : "multi:softprob",
        "num_class"    : N_CLASSES,
        "eval_metric"  : "mlogloss",
        "tree_method"  : "hist",
        "random_state" : RANDOM_STATE,
        "n_jobs"       : -1,
        "verbosity"    : 0,
    })

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_is, y_is, verbose=False)
    proba = final_model.predict_proba(X_oos)
    return proba, best_params


def train_tabpfn(X_is, y_is, X_oos):
    # v0.1.9 API: N_ensemble_configurations, overwrite_warning
    clf = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
    clf.fit(X_is, y_is, overwrite_warning=True)
    proba = clf.predict_proba(X_oos)
    # 클래스 순서 보장
    classes = clf.classes_
    if not np.array_equal(classes, np.arange(N_CLASSES)):
        order = np.argsort(classes)
        proba = proba[:, order]
    return proba


# ─── 섹션 0 완료 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 0 PASS [OK] - 함수 정의 완료")
print("=" * 60)

# ─── 섹션 1: 데이터 로드 확인 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1: 데이터 로드 확인")
print("=" * 60)

ticker_data = {}
for ticker in IT_TICKERS:
    df = load_ticker_csv(ticker)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    assert not missing, f"{ticker} 피처 누락: {missing}"
    ticker_data[ticker] = df
    print(f"  {ticker}: {len(df)}행, target 유효={df[TARGET_COL].notna().sum()}")

all_dates = list(ticker_data.values())[0].index
windows   = make_wf_windows(all_dates)
print(f"  총 windows: {len(windows)}개  (테스트: {MAX_WINDOWS}개만 실행)")
print(f"  첫 OOS: {windows[0][1][0].date()} ~ {windows[0][1][-1].date()}")
print("SECTION 1 PASS [OK]")

# ─── 섹션 2: Walk-Forward 미니 테스트 ────────────────────────────────────────
print("\n" + "=" * 60)
print(f"SECTION 2: Walk-Forward (windows {MAX_WINDOWS}개, tickers {len(IT_TICKERS)}개)")
print("=" * 60)

records = []
t0      = time.time()

for w_idx, (is_dates, oos_dates) in enumerate(windows[:MAX_WINDOWS]):
    print(f"\n[Window {w_idx+1}/{MAX_WINDOWS}] IS ~{is_dates[-1].date()} | OOS {oos_dates[0].date()}~{oos_dates[-1].date()}")

    for ticker in IT_TICKERS:
        df_t   = ticker_data[ticker]
        df_is  = df_t.loc[is_dates].dropna(subset=FEATURE_COLS + [TARGET_COL])
        df_oos = df_t.loc[df_t.index.isin(oos_dates)]

        if len(df_is) < 50 or len(df_oos) == 0:
            print(f"  {ticker}: 데이터 부족, 스킵")
            continue

        X_is   = df_is[FEATURE_COLS].values.astype(float)
        y_is   = df_is[TARGET_COL]
        X_oos  = df_oos[FEATURE_COLS].fillna(0).values.astype(float)
        y_oos  = df_oos[TARGET_COL].values

        try:
            y_is_cls, bins, r_bar = make_quintile_labels(y_is)
        except ValueError as e:
            print(f"  {ticker}: 분위 생성 실패({e}), 스킵")
            continue

        y_oos_cls = apply_bins(df_oos[TARGET_COL], bins)

        # XGBoost
        try:
            proba_xgb, _ = train_xgb(X_is, y_is_cls, X_oos, window_id=w_idx)
            Q_xgb, Om_xgb = compute_Q_Omega(proba_xgb, r_bar)
            ll_xgb = log_loss(y_oos_cls, proba_xgb, labels=list(range(N_CLASSES)))
            pred_cls_xgb = np.argmax(proba_xgb, axis=1)
            print(f"  {ticker} XGB   : ll={ll_xgb:.4f}, Q mean={Q_xgb.mean():.4f}")
        except Exception as e:
            print(f"  {ticker} XGB 오류: {e}")
            Q_xgb = np.zeros(len(X_oos)); Om_xgb = np.ones(len(X_oos))*1e-4
            ll_xgb = np.nan; pred_cls_xgb = np.zeros(len(X_oos), dtype=int)

        # TabPFN
        try:
            proba_tab = train_tabpfn(X_is, y_is_cls, X_oos)
            Q_tab, Om_tab = compute_Q_Omega(proba_tab, r_bar)
            ll_tab = log_loss(y_oos_cls, proba_tab, labels=list(range(N_CLASSES)))
            pred_cls_tab = np.argmax(proba_tab, axis=1)
            print(f"  {ticker} TabPFN: ll={ll_tab:.4f}, Q mean={Q_tab.mean():.4f}")
        except Exception as e:
            print(f"  {ticker} TabPFN 오류: {e}")
            Q_tab = np.zeros(len(X_oos)); Om_tab = np.ones(len(X_oos))*1e-4
            ll_tab = np.nan; pred_cls_tab = np.zeros(len(X_oos), dtype=int)

        for j, oos_date in enumerate(df_oos.index):
            base = {"window_id": w_idx, "date": oos_date, "ticker": ticker,
                    "actual_ret": y_oos[j] if j < len(y_oos) else np.nan,
                    "true_cls":   y_oos_cls[j] if j < len(y_oos_cls) else -1}
            if j < len(Q_xgb):
                records.append({**base, "model": "xgb",    "Q": Q_xgb[j],
                                 "Omega": Om_xgb[j], "pred_cls": pred_cls_xgb[j], "logloss": ll_xgb})
            if j < len(Q_tab):
                records.append({**base, "model": "tabpfn", "Q": Q_tab[j],
                                 "Omega": Om_tab[j], "pred_cls": pred_cls_tab[j], "logloss": ll_tab})

print(f"\n소요 시간: {time.time()-t0:.1f}초")
print(f"기록 수: {len(records)}개")
assert len(records) > 0, "기록이 없습니다!"

df_res = pd.DataFrame(records)
print("\n[샘플 결과]")
print(df_res[["ticker","model","Q","Omega","logloss"]].groupby(["ticker","model"]).mean().round(4))
print("\nSECTION 2 PASS [OK]")

# ─── 섹션 3: 성능 지표 확인 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: 성능 지표 샘플")
print("=" * 60)

df_valid = df_res.dropna(subset=["actual_ret"])
for (ticker, model), grp in df_valid.groupby(["ticker","model"]):
    q_pred   = grp["Q"].values
    cls_pred = grp["pred_cls"].values
    cls_true = grp["true_cls"].values
    ret_true = grp["actual_ret"].values
    valid    = ~np.isnan(ret_true)
    if valid.sum() == 0:
        continue
    acc  = accuracy_score(cls_true[valid], cls_pred[valid])
    hit  = np.mean(np.sign(q_pred[valid]) == np.sign(ret_true[valid]))
    ic,_ = spearmanr(q_pred[valid], ret_true[valid])
    print(f"  {ticker:5s} {model:7s}: acc={acc:.3f}, hit={hit:.3f}, IC={ic:.3f}")

print("\nALL SECTIONS PASS [OK]")
