"""
lib/methodology_charts.py — Methodology 페이지 5 영역 함수 (초안)

영역:
  3. render_methodology_sankey (BL+LSTM Plotly Sankey, 9 노드 4 그룹)
  4. render_bl_detail (수식 + 4-slot config 표 + 본 펀드 강조)
  5. render_lstm_detail (Walk-forward 구조 + Architecture + Input/Output 표)
  6. render_factor_analysis (CAPM + FF5 회귀)
  7. render_limitations (3 한계 카드 + 학술 정직성 선언)

설계 원칙 (초안 — 향후 변경 용이):
  - 영역별 독립 함수
  - 학술 인용 / 한계 정의 외부 dict (변경 용이)
  - 각 영역 데이터 의존성 명시

참조: docs/plan/03_pages/07_methodology.md, docs/decisionlog/07_methodology.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats

from lib.colors import COLORS, LIMITATION_COLORS, SANKEY_GROUP_COLORS


# ======================================================================
# 영역 3: Methodology Overview (Plotly Sankey)
# ======================================================================

# Sankey 9 노드 — 4 그룹 (변경 용이)
SANKEY_NODES = [
    # 그룹 1: 데이터
    {"label": "Market Data", "group": "data"},
    {"label": "Returns Data", "group": "data"},
    {"label": "Sector / Mcap", "group": "data"},
    # 그룹 2: BL prior
    {"label": "BL Prior\n(CAPM equilibrium)", "group": "bl"},
    # 그룹 3: LSTM
    {"label": "LSTM Vol\nPredict", "group": "lstm"},
    {"label": "View / Confidence\n(P, Q, Ω)", "group": "lstm"},
    # 그룹 4: BL posterior + Optimizer
    {"label": "BL Posterior\nE[R]", "group": "bl"},
    {"label": "Optimizer\n(4-slot config)", "group": "optimizer"},
    {"label": "Portfolio Weights", "group": "optimizer"},
]

# Sankey 링크 (source → target) — 흐름
SANKEY_LINKS = [
    (0, 3),   # Market Data → BL Prior
    (1, 3),   # Returns → BL Prior
    (2, 3),   # Sector → BL Prior
    (1, 4),   # Returns → LSTM
    (4, 5),   # LSTM → View/Confidence
    (3, 6),   # BL Prior → BL Posterior
    (5, 6),   # View → BL Posterior
    (6, 7),   # BL Posterior → Optimizer
    (7, 8),   # Optimizer → Weights
]


def render_methodology_sankey() -> None:
    """Methodology 전체 흐름 Sankey 다이어그램 — 9 노드 4 그룹."""
    node_labels = [n["label"] for n in SANKEY_NODES]
    node_colors = [SANKEY_GROUP_COLORS.get(n["group"], COLORS["text_muted"]) for n in SANKEY_NODES]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=20, line=dict(color="black", width=0.5),
            label=node_labels, color=node_colors,
        ),
        link=dict(
            source=[s for s, _ in SANKEY_LINKS],
            target=[t for _, t in SANKEY_LINKS],
            value=[1] * len(SANKEY_LINKS),
            color="rgba(156, 163, 175, 0.3)",
        ),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"], height=420,
        margin=dict(t=10, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4 그룹 색상 범례
    st.markdown(
        f'<div style="display:flex;gap:16px;font-size:12px;color:{COLORS["text_muted"]};margin-top:6px;">'
        f'<span><span style="display:inline-block;width:10px;height:10px;background:{SANKEY_GROUP_COLORS["data"]};margin-right:4px;"></span>데이터</span>'
        f'<span><span style="display:inline-block;width:10px;height:10px;background:{SANKEY_GROUP_COLORS["bl"]};margin-right:4px;"></span>Black-Litterman</span>'
        f'<span><span style="display:inline-block;width:10px;height:10px;background:{SANKEY_GROUP_COLORS["lstm"]};margin-right:4px;"></span>LSTM</span>'
        f'<span><span style="display:inline-block;width:10px;height:10px;background:{SANKEY_GROUP_COLORS["optimizer"]};margin-right:4px;"></span>Optimizer</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ======================================================================
# 영역 4: Black-Litterman 상세 + 4-slot config
# ======================================================================

def render_bl_detail(current_config: str = "mat_eq_mcap_raw_he") -> None:
    """BL 수식 + 4-slot config 표 + 본 펀드 강조."""
    # === 1. Equilibrium Prior (CAPM) ===
    st.markdown("##### 1. Equilibrium Prior (CAPM)")
    st.latex(r"\Pi = \delta \, \Sigma \, w_{market}")
    st.caption(
        "δ = risk aversion / Σ = covariance / w_market = market cap weights. "
        "**Black & Litterman (1990, 1992)** — 시장 균형 기반 사전 평균."
    )

    # === 2. View 통합 (Bayesian Update) ===
    st.markdown("##### 2. View 통합 — Bayesian Update")
    st.latex(
        r"E[R] = \left[(\tau\Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1} "
        r"\left[(\tau\Sigma)^{-1}\Pi + P^T \Omega^{-1} Q\right]"
    )
    st.caption(
        "τ = uncertainty / P = view matrix / Q = view returns / **Ω = view confidence (LSTM σ̂ 활용)**. "
        "**He & Litterman (1999)**, **Idzorek (2005)**."
    )

    # === 3. 4-slot config 표 + 본 펀드 강조 ===
    st.markdown("##### 3. 4-slot Configuration — 본 펀드 ★ 강조")

    # current_config 차원 분해 (mat_{prior}_{p_weight}_{q_mode}_{omega})
    parts = current_config.split("_")
    cur_prior = parts[1] if len(parts) > 1 else "?"
    cur_p_weight = parts[2] if len(parts) > 2 else "?"
    cur_q_mode = parts[3] if len(parts) > 3 else "?"
    cur_omega = parts[4] if len(parts) > 4 else "?"

    config_table = pd.DataFrame({
        "Slot": ["prior", "p_weight", "q_mode", "omega_mode"],
        "Options": [
            "capm_eq / capm_mcap / capm_rp",
            "eq / mcap / rp / vol_mcap",
            "raw_lam / lambda / fixed / capm / ff3_paper / inv_lambda / vol_spread / none",
            "ff3_paper (pap) / he_litterman (he) / rmse (rms)",
        ],
        "본 펀드 ★": [
            f"capm_{cur_prior} ★",
            f"{cur_p_weight} ★",
            f"{cur_q_mode} ★",
            f"{cur_omega} ★",
        ],
    })
    st.dataframe(config_table, hide_index=True, use_container_width=True)

    st.success(
        f"✅ **본 펀드 config**: `{current_config}`\n\n"
        f"별도 기준으로 선정된 Top 1 config (Sortino Rank 14/159, Top 8.8%). "
        f"4-slot 조합 중 학술/실무 다중 메트릭 검증 결과. "
        f"자세한 sensitivity 분석은 **Backtesting 페이지** 영역 7 참조."
    )


# ======================================================================
# 영역 5: LSTM 변동성 예측 상세 (Walk-forward)
# ======================================================================

# Walk-forward 파라미터 (외부 dict — 변경 용이)
LSTM_WALKFWD_CONFIG = {
    "is_len": 1250,    # ~5년 일별
    "oos_len": 21,     # ~1개월 OOS
    "step": 21,        # 월별 슬라이딩
    "embargo": 63,     # ~3개월 buffer
    "seq_len": 63,     # sequence length
    "n_stocks": 615,
    "n_folds_approx": 120,
}


def render_lstm_detail() -> None:
    """LSTM 구조 + Walk-forward 명시 + Input/Output 표."""
    # === 1. Walk-forward 구조 (★ 사용자 강조) ===
    st.markdown("##### 1. ★ Walk-forward 구조 (Lopez de Prado 2018 표준)")

    cfg = LSTM_WALKFWD_CONFIG
    cols = st.columns(5)
    with cols[0]:
        st.metric("is_len", f"{cfg['is_len']}d", help=f"~{cfg['is_len']/250:.1f}년 일별 학습")
    with cols[1]:
        st.metric("oos_len", f"{cfg['oos_len']}d", help="~1개월 OOS 예측")
    with cols[2]:
        st.metric("step", f"{cfg['step']}d", help="월별 슬라이딩")
    with cols[3]:
        st.metric("embargo", f"{cfg['embargo']}d", help="~3개월 buffer (학습/예측 분리)")
    with cols[4]:
        st.metric("seq_len", f"{cfg['seq_len']}d", help="LSTM 입력 sequence length")

    st.caption(
        f"각 시점별 학습 → embargo → OOS 예측 → 21d 슬라이딩 후 재학습. "
        f"**{cfg['n_stocks']} 종목 × ~{cfg['n_folds_approx']} fold = walk-forward 학습 + HAR-RV 앙상블**. "
        f"Lopez de Prado (2018) *Advances in Financial Machine Learning* 표준."
    )

    # Walk-forward flow 다이어그램 (텍스트 기반)
    st.code(
        """[t=2010-01]
  ├─ 학습: 1250d (2005-01 ~ 2010-01)
  ├─ Embargo: 63d buffer
  └─ OOS 예측: 21d (다음 1개월)

[t=2010-02 (21d 슬라이딩)]
  ├─ 학습: 1250d (재학습, 2005-02 ~ 2010-02)
  ├─ Embargo: 63d buffer
  └─ OOS 예측: 21d (다음 1개월)

... 192개월 동안 반복 ... → 펀드 결과의 ret 시계열""",
        language="text",
    )

    st.divider()

    # === 2. TEST / HOLD_OUT 의 정확한 의미 ===
    st.markdown("##### 2. TEST 168m vs HOLD_OUT 24m — 정확한 의미")
    st.info(
        "🔍 **TEST/HOLD_OUT 은 학습/검증 분리가 아닌, walk-forward 결과의 평가 기간 분리**:\n\n"
        "- **TEST 168m** (2010-01 ~ 2023-12) = walk-forward 결과의 in-sample evaluation (config selection 평가)\n"
        "- **HOLD_OUT 24m** (2024-01 ~ 2025-12) = walk-forward 결과의 true out-of-sample (untouched, 사후 평가만)\n\n"
        "→ LSTM 자체는 두 기간 모두 walk-forward 로 학습됨. 차이는 평가 시점."
    )

    st.divider()

    # === 3. LSTM Input / Output 표 ===
    st.markdown("##### 3. Input / Output")
    io_table = pd.DataFrame({
        "Type": ["Input", "Input", "Input", "Output", "Downstream"],
        "Item": [
            "Past 60d returns",
            "Sector dummy (GICS 11)",
            "Market state (SPY rolling vol / VIX)",
            "σ_next (다음 월 변동성)",
            "→ BL Ω (View confidence)",
        ],
        "Description": [
            "과거 60일 일별 수익률",
            "11개 GICS 섹터 one-hot",
            "시장 상태 변수",
            "월별 변동성 예측",
            "BL view confidence 입력 → 사후 평균 산출",
        ],
    })
    st.dataframe(io_table, hide_index=True, use_container_width=True)

    st.divider()

    # === 4. LSTM Cell 수식 (간단) ===
    st.markdown("##### 4. LSTM Cell 수식")
    st.latex(r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)}")
    st.latex(r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)}")
    st.latex(r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)}")
    st.caption(
        "**Hochreiter & Schmidhuber (1997)** — LSTM 원논문. "
        "**Gers, Schmidhuber & Cummins (2000)** — Forget gate 추가. "
        "**Kim & Won (2018)** — 금융 시계열 LSTM 적용."
    )


# ======================================================================
# 영역 6: Factor 분석 (CAPM + FF5) — 조건부
# ======================================================================

def _ols_with_stats(y: np.ndarray, X: np.ndarray) -> dict:
    """
    numpy + scipy 만으로 OLS 회귀 + 표준 통계 (statsmodels 동등).

    Returns: {beta, se, t_stat, p_value, ci_low, ci_high, r_squared, n}
    각 array index 0 = const (intercept), 1+ = features.
    """
    X_full = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    n, k = X_full.shape
    y_hat = X_full @ beta
    resid = y - y_hat
    sse = float((resid ** 2).sum())
    df = n - k
    sigma2 = sse / df
    cov_beta = sigma2 * np.linalg.inv(X_full.T @ X_full)
    se = np.sqrt(np.diag(cov_beta))
    t_stat = beta / se
    p_value = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stat), df=df))
    t_crit = scipy_stats.t.ppf(0.975, df=df)
    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se
    ss_total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - sse / ss_total if ss_total > 0 else np.nan
    return {
        "beta": beta, "se": se, "t_stat": t_stat, "p_value": p_value,
        "ci_low": ci_low, "ci_high": ci_high, "r_squared": float(r_squared), "n": n,
    }


@st.cache_data
def _run_factor_regression(
    fund_ret: pd.Series, ff5_df: pd.DataFrame, model: str = "CAPM"
) -> dict:
    """CAPM 또는 FF5 OLS 회귀 — numpy + scipy 만 사용 (statsmodels 의존성 X)."""
    ff5 = ff5_df.copy()
    ff5["date"] = pd.to_datetime(ff5["date"]) + pd.offsets.MonthEnd(0)
    ff5 = ff5.set_index("date")

    common = fund_ret.index.intersection(ff5.index)
    if len(common) < 24:
        return {}
    y = (fund_ret.loc[common] - ff5.loc[common, "rf"]).values

    if model == "CAPM":
        feature_names = ["mkt_rf"]
    else:  # FF5
        feature_names = ["mkt_rf", "smb", "hml", "rmw", "cma"]
    X = ff5.loc[common, feature_names].values

    res = _ols_with_stats(y, X)

    # const (index 0) = alpha (월별 → 연환산)
    alpha_monthly = float(res["beta"][0])
    return {
        "model": model,
        "alpha_annual": alpha_monthly * 12,
        "alpha_p": float(res["p_value"][0]),
        "alpha_ci": [float(res["ci_low"][0] * 12), float(res["ci_high"][0] * 12)],
        # features (index 1+)
        "params": {f: float(res["beta"][i + 1]) for i, f in enumerate(feature_names)},
        "pvalues": {f: float(res["p_value"][i + 1]) for i, f in enumerate(feature_names)},
        "rsquared": res["r_squared"],
        "n": len(common),
    }


def render_factor_analysis(fund_ret: pd.Series, ff5_df: pd.DataFrame) -> None:
    """CAPM + FF5 회귀 결과 표시."""
    capm = _run_factor_regression(fund_ret, ff5_df, "CAPM")
    ff5 = _run_factor_regression(fund_ret, ff5_df, "FF5")

    if not capm or not ff5:
        st.warning("Factor 회귀 산출 불가 (데이터 부족).")
        return

    # === Annualized Alpha Card ===
    st.markdown("##### Annualized Alpha (CAPM + FF5)")
    cols = st.columns(2)
    for i, (label, res) in enumerate([("CAPM", capm), ("FF5", ff5)]):
        with cols[i]:
            sig = "***" if res["alpha_p"] < 0.01 else ("**" if res["alpha_p"] < 0.05 else ("*" if res["alpha_p"] < 0.1 else ""))
            color = COLORS["accent_green"] if res["alpha_annual"] > 0 else COLORS["accent_red"]
            st.markdown(
                f'<div style="border:2px solid {color};border-radius:6px;padding:14px;'
                f'background:{COLORS["secondary_bg"]};">'
                f'<div style="font-weight:bold;color:#FAFAFA;">{label} α (annualized)</div>'
                f'<div style="font-size:24px;color:{color};font-weight:bold;margin:6px 0;">'
                f'{res["alpha_annual"]*100:+.2f}% {sig}</div>'
                f'<div style="font-size:11px;color:{COLORS["text_muted"]};">'
                f'95% CI: [{res["alpha_ci"][0]*100:+.2f}%, {res["alpha_ci"][1]*100:+.2f}%] / '
                f'p = {res["alpha_p"]:.3f} / R² = {res["rsquared"]:.3f}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # === FF5 Factor Exposure 막대 ===
    st.markdown("##### FF5 Factor Exposure (β)")
    factors = ["mkt_rf", "smb", "hml", "rmw", "cma"]
    factor_labels = {"mkt_rf": "MKT", "smb": "SMB", "hml": "HML", "rmw": "RMW", "cma": "CMA"}
    betas = [ff5["params"].get(f, 0) for f in factors]
    pvalues = [ff5["pvalues"].get(f, 1) for f in factors]
    sigs = ["***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")) for p in pvalues]

    fig = go.Figure(go.Bar(
        x=[factor_labels[f] for f in factors],
        y=betas,
        text=[f"{b:.3f} {s}" for b, s in zip(betas, sigs)],
        textposition="outside",
        marker_color=[COLORS["primary"] if b > 0 else COLORS["accent_red"] for b in betas],
        hovertemplate="%{x}<br>β: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        yaxis_title="β (factor exposure)", height=320,
        margin=dict(t=20, l=0, r=0, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "*** p<0.01 / ** p<0.05 / * p<0.1. "
        "**Jensen (1968)**, **Fama-French (1993, 2015)**, **Carhart (1997)**."
    )



# ======================================================================
# 영역 7: 한계 + 향후 개선 (3개 한계 카드)
# ======================================================================

# 한계 카드 정의 (외부 — 변경 용이)
LIMITATION_CARDS = [
    {
        "key": "ho_decline",
        "icon": "🟧",
        "title": "HO 24m 부진 인정",
        "summary": "SPY 대비 -10.3%p (HO 24m, AI Rally 시기)",
        "detail": (
            "- HO 24m: Fund +21.59% / SPY +40.93% → -19.3%p underperform\n"
            "- 원인: SPY 의 IT 집중 (33%) vs Fund 의 IT under-weight\n"
            "- Sector Watch 페이지 영역 8 narrative 와 일관 (Markowitz 1952)\n"
            "- 학술 정직성 — 단기 sector concentration 시기의 trade-off"
        ),
        "color": LIMITATION_COLORS["ho_decline"],
    },
    {
        "key": "future_work",
        "icon": "🟩",
        "title": "향후 개선 방향",
        "summary": "Multi-factor / 실거래 시뮬 / 추가 학술 검증",
        "detail": (
            "- **Multi-factor 통합**: Momentum + Value + Quality (Fama-French 2015 + Carhart 1997)\n"
            "- **실거래 시뮬레이션**: Slippage + 시장 충격 비용 + 유동성 제약\n"
            "- **Walk-forward 추가 검증**: is_len/embargo sensitivity\n"
            "- **Ablation study**: BL only vs BL+LSTM (LSTM 부가가치 직접 측정)\n"
            "- → Backtesting 페이지 영역 7 sensitivity test 참조"
        ),
        "color": LIMITATION_COLORS["future_work"],
    },
    {
        "key": "practical",
        "icon": "🟥",
        "title": "실무 적용 제약",
        "summary": "가상 펀드 / 운용 규모 / Tax 미반영",
        "detail": (
            "- **가상 펀드** — 실제 운용 X, 학술 backtest 목적\n"
            "- **운용 규모 가정**: 소형 (시장 충격 무시 가능 가정)\n"
            "- **Tax / 유동성 제약 미반영** — 실거래 시 수익 ↓\n"
            "- 거래비용은 One-way 20bp 적용 (Frazzini, Israel & Moskowitz 2018)\n"
            "- 실무 적용 시 추가 제약 검증 필요"
        ),
        "color": LIMITATION_COLORS["practical"],
    },
]


def render_limitations() -> None:
    """3 한계 카드 + 학술 정직성 선언 + Expander."""
    # 학술 정직성 선언
    st.success(
        "✅ **학술 정직성 선언**\n\n"
        "본 펀드는 학술 정직성을 위해 모든 한계를 명시합니다. "
        "한계 인정은 신뢰성 강화의 토대이며, 향후 개선 방향을 통해 지속적 발전을 추구합니다."
    )

    st.markdown("##### 3가지 한계 카드")

    # 3 카드 그리드
    cols = st.columns(3)
    for col, card in zip(cols, LIMITATION_CARDS):
        with col:
            st.markdown(
                f'<div style="border:2px solid {card["color"]};border-radius:8px;'
                f'padding:14px;background:{COLORS["secondary_bg"]};min-height:170px;">'
                f'<div style="font-size:22px;">{card["icon"]}</div>'
                f'<div style="font-weight:bold;color:#FAFAFA;margin:6px 0;">{card["title"]}</div>'
                f'<div style="font-size:12px;color:{COLORS["text_muted"]};">{card["summary"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 자세한 detail (Expander)
    st.markdown("##### 자세한 내용 (Expander)")
    for card in LIMITATION_CARDS:
        with st.expander(f"{card['icon']} {card['title']} — 자세히"):
            st.markdown(card["detail"])
