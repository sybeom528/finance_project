"""Black-Litterman 저위험 프리미엄 포트폴리오 — 공통 유틸리티

low_risk/ 파이프라인의 노트북 06~10, 98, 99에서 공유하는 함수 모음.

구성
----
1. 핵심 BL 함수
   - compute_sigma          : Ledoit-Wolf 축소 공분산
   - compute_pi             : CAPM 균형수익률 + 위험회피계수 λ
   - build_P                : 저위험 long / 고위험 short view 행렬
   - compute_omega          : view 불확실성 τ·P·Σ·Pᵀ
   - black_litterman        : posterior μ_BL (Sherman-Woodbury)
   - optimize_portfolio     : long-only mean-variance (SLSQP)

2. 성과 평가
   - performance            : 연환산 수익/변동성/Sharpe/Calmar/MDD/누적수익률

3. Q 추정 (BL view 강도)
   - compute_Q_hist         : 학습 기간 P 포트폴리오 평균 실현수익률
   - compute_Q_momentum     : 최근 N개월 P 포트폴리오 평균 실현수익률
   - compute_Q_lambda       : clipped λ-스케일 Q (위험회피계수 기반, 항상 양수)
   - compute_Q_raw_lambda   : raw λ-스케일 Q (음수 가능 → max(0,·)로 자연 게이팅)
   - compute_Q_pi_ratio     : CAPM 스프레드 비율 기반 Q
   - compute_Q_vol_spread   : 예측 vol 격차 기반 Q 스케일러 (파이프라인 일관성)
   - compute_Q_ff3          : FF3 회귀로 추정한 P 포트폴리오 기대수익률

4. Fama-French 3-factor
   - download_ff3           : Tuck 서버에서 월별 FF3 다운로드 (단위: 소수)

5. 시각화 헬퍼
   - drawdown               : 누적 drawdown 시계열
   - rolling_sharpe         : 롤링 Sharpe (디폴트 12개월)
   - shade_high_vol         : 고변동 시기 음영
"""
from __future__ import annotations

import io
import re
import warnings
import zipfile

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

ANN = np.sqrt(12.0)  # 월간 → 연환산 변환 상수

__all__ = [
    'ANN',
    # core BL
    'compute_sigma', 'compute_pi', 'build_P',
    'compute_omega', 'black_litterman', 'optimize_portfolio',
    # performance
    'performance',
    # Q methods
    'compute_Q_hist', 'compute_Q_momentum', 'compute_Q_lambda',
    'compute_Q_raw_lambda', 'compute_Q_pi_ratio', 'compute_Q_vol_spread',
    'compute_Q_ff3',
    # FF3
    'download_ff3',
    # plotting helpers
    'drawdown', 'rolling_sharpe', 'shade_high_vol',
]


# ──────────────────────────────────────────────────────────────────────
# 1. 핵심 BL 함수
# ──────────────────────────────────────────────────────────────────────
def compute_sigma(ret_matrix: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf 축소 공분산 추정 (월간 returns DataFrame 입력)."""
    lw = LedoitWolf().fit(ret_matrix.values)
    return pd.DataFrame(lw.covariance_,
                        index=ret_matrix.columns,
                        columns=ret_matrix.columns)


def compute_pi(Sigma: pd.DataFrame,
               w_mkt: pd.Series,
               spy_excess_ret: float,
               sigma2_mkt: float) -> tuple[pd.Series, float]:
    """CAPM 역최적화: π = λ·Σ·w_mkt, 위험회피계수 λ = E[r_m]/σ²_m.

    λ는 [0.5, 10.0]으로 클리핑(극단값 방지). 시장 분산이 0 이하이면 λ=2.5.
    """
    lam = spy_excess_ret / sigma2_mkt if sigma2_mkt > 0 else 2.5
    lam = float(np.clip(lam, 0.5, 10.0))
    return lam * Sigma @ w_mkt, lam


def build_P(vol_series: pd.Series,
            mcap_series: pd.Series,
            pct: float = 0.30) -> pd.Series:
    """저위험 long / 고위험 short — 시가총액 가중 P 벡터.

    vol_series 하위 pct = +mcap weight, 상위 pct = -mcap weight.
    각 그룹 내 합이 ±1이 되도록 정규화.
    """
    n_group    = max(1, int(len(vol_series) * pct))
    sorted_idx = vol_series.sort_values().index
    low_risk   = sorted_idx[:n_group]
    high_risk  = sorted_idx[-n_group:]
    P = pd.Series(0.0, index=vol_series.index)
    low_m  = mcap_series[low_risk]
    high_m = mcap_series[high_risk]
    P[low_risk]  =  low_m  / low_m.sum()
    P[high_risk] = -high_m / high_m.sum()
    return P


def compute_omega(P: pd.Series, Sigma: pd.DataFrame, tau: float) -> float:
    """View 불확실성 Ω = τ·P·Σ·Pᵀ (단일 view 스칼라). 1e-8 미만은 클리핑."""
    p = P.values
    return max(float(tau * p @ Sigma.values @ p), 1e-8)


def black_litterman(pi: pd.Series,
                    Sigma: pd.DataFrame,
                    P: pd.Series,
                    q: float,
                    omega: float,
                    tau: float) -> pd.Series:
    """단일 view BL posterior — Sherman-Woodbury 닫힌 해.

    μ_BL = π + (τΣPᵀ) · (q − P·π) / (P·τΣ·Pᵀ + Ω)
    """
    p    = P.values
    pi_v = pi.values
    tSig = tau * Sigma.values
    M      = float(p @ tSig @ p) + omega
    diff   = q - float(p @ pi_v)
    adjust = tSig @ p * (diff / M)
    return pd.Series(pi_v + adjust, index=pi.index)


def optimize_portfolio(mu: pd.Series,
                       Sigma: pd.DataFrame,
                       lam: float) -> pd.Series:
    """Long-only mean-variance: max  μᵀw − ½·λ·wᵀΣw  s.t. Σw=1, w∈[0,1]."""
    n   = len(mu)
    mu_ = mu.values
    Sig = Sigma.values

    def obj(w): return 0.5 * lam * w @ Sig @ w - w @ mu_
    def jac(w): return lam * Sig @ w - mu_

    res = minimize(
        obj, np.ones(n) / n, jac=jac, method='SLSQP',
        bounds=[(0, 1)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
    )
    if not res.success:
        warnings.warn(f'optimize_portfolio 수렴 실패 → 1/N 대체: {res.message}')
    w = res.x if res.success else np.ones(n) / n
    return pd.Series(w, index=mu.index)


# ──────────────────────────────────────────────────────────────────────
# 2. 성과 평가
# ──────────────────────────────────────────────────────────────────────
def performance(ret: pd.Series,
                rf: pd.Series | float,
                label: str,
                verbose: bool = False) -> dict:
    """월간 수익률 시계열의 표준 성과 지표.

    rf는 Series(월간 무위험 수익률) 또는 스칼라(0 등) 모두 허용.
    반환 dict 키: label, ann_ret, ann_vol, sharpe, calmar, cum_ret, mdd.
    """
    if isinstance(rf, pd.Series):
        rf_aligned = rf.reindex(ret.index).fillna(0)
    else:
        rf_aligned = pd.Series(float(rf), index=ret.index)

    excess  = ret - rf_aligned
    ann_ret = ret.mean() * 12
    ann_vol = ret.std() * ANN
    sharpe  = (excess.mean() / excess.std() * ANN
               if excess.std() > 0 else np.nan)
    cum     = (1 + ret).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    cum_ret = cum.iloc[-1] - 1
    calmar  = ann_ret / abs(mdd) if mdd != 0 else np.nan

    if verbose:
        print(f'[{label}]')
        print(f'  연환산 수익률: {ann_ret:.2%}')
        print(f'  연환산 변동성: {ann_vol:.2%}')
        print(f'  Sharpe Ratio:  {sharpe:.3f}')
        print(f'  Calmar Ratio:  {calmar:.3f}')
        print(f'  누적 수익률:   {cum_ret:.2%}')
        print(f'  MDD:           {mdd:.2%}')

    return {'label': label,
            'ann_ret': ann_ret, 'ann_vol': ann_vol,
            'sharpe':  sharpe,  'calmar':  calmar,
            'cum_ret': cum_ret, 'mdd':     mdd}


# ──────────────────────────────────────────────────────────────────────
# 3. Q 추정 함수군
# ──────────────────────────────────────────────────────────────────────
def compute_Q_hist(P: pd.Series, ret_matrix: pd.DataFrame) -> float:
    """학습 기간 전체 P 포트폴리오 평균 실현수익률 (저위험 − 고위험)."""
    p_vec = P.reindex(ret_matrix.columns).fillna(0).values
    return float((ret_matrix.values @ p_vec).mean())


def compute_Q_momentum(P: pd.Series,
                       ret_matrix: pd.DataFrame,
                       window: int = 12) -> float:
    """최근 window개월 P 포트폴리오 평균 실현수익률 (단기 모멘텀)."""
    n = min(window, len(ret_matrix))
    if n < 3:
        return compute_Q_hist(P, ret_matrix)
    p_vec  = P.reindex(ret_matrix.columns).fillna(0).values
    recent = ret_matrix.iloc[-n:]
    return float((recent.values @ p_vec).mean())


def compute_Q_lambda(lam: float,
                     q_base: float,
                     lam_mean: float = 2.5) -> float:
    """Q = q_base × clip(λ_t / λ_mean, 0.1, 3.0).

    λ↑(risk-off) → Q 확대 (저위험 프리미엄 강화 베팅).
    λ↓(risk-on)  → Q 축소.
    """
    return float(q_base * np.clip(lam / lam_mean, 0.1, 3.0))


def compute_Q_raw_lambda(lam_raw: float,
                         q_base: float,
                         lam_mean: float = 2.5) -> float:
    """Q = max(0, q_base × (λ_raw / λ_mean)) — raw λ 부호 기반 자연 게이팅.

    `compute_pi`가 반환하는 lam은 [0.5, 10.0]으로 clip되어 있으므로 여기서는
    clip 이전의 raw λ = spy_excess_ret / σ²_mkt 를 그대로 받아 사용한다.

    raw λ < 0 (SPY 하락 국면) → max(0,...) = 0 → BL 뷰 자동 차단.
    raw λ > 0 (정상/회복 국면) → λ 크기에 비례한 자연 스케일.

    Regime3 하드스탑과 유사한 효과를 명시적 레짐 분류 없이 달성.
    """
    return float(max(0.0, q_base * (lam_raw / lam_mean)))


def compute_Q_vol_spread(P: pd.Series,
                         vol_pred: pd.Series,
                         q_base: float,
                         spread_ref: float) -> float:
    """예측 vol 격차(고위험 그룹 평균 vol − 저위험 그룹 평균 vol) 기반 Q 스케일러.

    vol_spread = mean(vol_pred[P<0]) − mean(vol_pred[P>0])  (항상 양수)
    Q = q_base × clip(vol_spread / spread_ref, 0.1, 3.0)

    spread_ref 는 walk-forward에서 expanding median으로 누적 (look-ahead bias 없음).
    파이프라인 일관성: 같은 vol_pred 가 P 분류에도, Q 강도 결정에도 사용됨.

    주의: 위기 시 vol_spread가 오히려 확대 → Q 증가 → 역방향 위험.
    레짐 하드스탑(고변동성 → Q=0)을 결합하면 이 위험이 차단된다.
    """
    low_tix  = P[P > 0].index
    high_tix = P[P < 0].index
    vol_low  = vol_pred.reindex(low_tix).mean()
    vol_high = vol_pred.reindex(high_tix).mean()
    if pd.isna(vol_low) or pd.isna(vol_high):
        return q_base
    spread = float(vol_high - vol_low)
    if spread_ref <= 1e-8 or spread <= 0:
        return q_base
    return float(q_base * np.clip(spread / spread_ref, 0.1, 3.0))


def compute_Q_pi_ratio(P: pd.Series,
                       pi: pd.Series,
                       q_base: float,
                       spread_ref: float) -> float:
    """Q = q_base × clip(|P·π| / spread_ref, 0.1, 3.0).

    CAPM이 저위험-고위험을 강하게 차별할수록(|P·π| 큼) Q를 확대.
    """
    spread = abs(float(P.values @ pi.values))
    if spread_ref <= 1e-8:
        return q_base
    return float(q_base * np.clip(spread / spread_ref, 0.1, 3.0))


def compute_Q_ff3(P: pd.Series,
                  ret_matrix: pd.DataFrame,
                  ff3_train: pd.DataFrame,
                  rf_train: pd.Series) -> float:
    """FF3 회귀로 추정한 P 포트폴리오 기대수익률.

    각 view ticker별로 (mkt_rf, smb, hml)로 회귀 → 최근 12개월 factor 평균을
    next-month forecast로 사용 → P로 결합.
    """
    view_tickers = [t for t in P[P != 0].index if t in ret_matrix.columns]
    if not view_tickers:
        return 0.0
    ff3_aligned = ff3_train.reindex(ret_matrix.index).dropna()
    rf_aligned  = rf_train.reindex(ff3_aligned.index).fillna(0)
    n = len(ff3_aligned)
    if n < 24:
        return 0.0
    X      = np.column_stack([np.ones(n), ff3_aligned[['mkt_rf', 'smb', 'hml']].values])
    X_next = np.array([1.0] + ff3_aligned[['mkt_rf', 'smb', 'hml']].tail(12).mean().tolist())
    rf_next = float(rf_train.iloc[-1]) if len(rf_train) > 0 else 0.0
    Y    = (ret_matrix[view_tickers].reindex(ff3_aligned.index).fillna(0).values
            - rf_aligned.values.reshape(-1, 1))
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    r_hat_view = X_next @ coef + rf_next
    r_hat = pd.Series(0.0, index=ret_matrix.columns)
    for j, t in enumerate(view_tickers):
        r_hat[t] = r_hat_view[j]
    return float(P.reindex(ret_matrix.columns).fillna(0) @ r_hat)


# ──────────────────────────────────────────────────────────────────────
# 4. Fama-French 3-factor
# ──────────────────────────────────────────────────────────────────────
def download_ff3() -> pd.DataFrame:
    """Tuck 서버에서 FF3 월별 데이터 다운로드.

    반환 컬럼: mkt_rf, smb, hml, rf (모두 소수, 월말 인덱스).
    """
    url = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
           'F-F_Research_Data_Factors_CSV.zip')
    resp = requests.get(url, timeout=60)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(zf.namelist()[0]).decode('utf-8', errors='ignore')
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines) if re.match(r'^\s*\d{6}\s*,', l))
    end   = next((i for i in range(start, len(lines))
                  if not re.match(r'^\s*\d{6}\s*,', lines[i])), len(lines))
    df = pd.read_csv(io.StringIO('\n'.join(lines[start - 1:end])))
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = (pd.to_datetime(df[date_col].astype(str), format='%Y%m')
                    + pd.offsets.MonthEnd(0))
    df = df.rename(columns={date_col: 'date', 'Mkt-RF': 'mkt_rf',
                            'SMB': 'smb', 'HML': 'hml', 'RF': 'rf'})
    return df.set_index('date').astype(float) / 100.0


# ──────────────────────────────────────────────────────────────────────
# 5. 시각화 헬퍼
# ──────────────────────────────────────────────────────────────────────
def drawdown(ret: pd.Series) -> pd.Series:
    """월간 수익률 시계열의 drawdown 시계열 ((cum − peak)/peak)."""
    cum = (1 + ret).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def rolling_sharpe(ret: pd.Series,
                   rf: pd.Series | float,
                   window: int = 12) -> pd.Series:
    """롤링 Sharpe (디폴트 12개월, 연환산)."""
    if isinstance(rf, pd.Series):
        rf_aligned = rf.reindex(ret.index).fillna(0)
    else:
        rf_aligned = pd.Series(float(rf), index=ret.index)
    exc = ret - rf_aligned
    return exc.rolling(window).mean() / exc.rolling(window).std() * ANN


def shade_high_vol(ax, dates, color: str = '#FFCDD2', alpha: float = 0.3) -> None:
    """ax에 고변동 구간(연속된 dates)을 음영 처리."""
    if len(dates) == 0:
        return
    in_block = False
    start = None
    all_d = list(pd.Series(dates).sort_values())
    for i, d in enumerate(all_d):
        if not in_block:
            start = d
            in_block = True
        if i == len(all_d) - 1 or (all_d[i + 1] - d).days > 60:
            ax.axvspan(start, d, alpha=alpha, color=color, linewidth=0)
            in_block = False
