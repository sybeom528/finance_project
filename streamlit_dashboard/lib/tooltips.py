"""
lib/tooltips.py - 메트릭 정의 dictionary

모든 페이지에서 메트릭 위에 hover 했을 때 보여줄 정의 통일.
참조: decisionlog/04_risk_metrics.md, plan/02_common.md 8절

사용 예시:
  from lib.tooltips import get_tooltip
  st.metric("CAGR", "+9.21%", help=get_tooltip("CAGR"))
"""

METRIC_TOOLTIPS = {
    # ── Pool-1 수익성 ─────────────────────────────────────────────
    "Cumulative Return": "누적 수익률 — 시작 시점부터 현재까지의 총 수익.",
    "CAGR": "연환산 복리 수익률 (Compound Annual Growth Rate). 기하평균 기반.",
    "Net CAGR": "거래비용 차감 후 연환산 수익률 (편측 20bp = 1회 거래당 0.20% 적용).",
    "Arithmetic Mean": "산술 평균 수익률 (월별 → 연환산).",

    # ── Pool-2 위험조정 수익 ───────────────────────────────────────
    "Sharpe": "(R - Rf) / σ. 전체 변동성 대비 초과수익. 1.0 이상이면 양호.",
    "Sortino": "(R - Rf) / σ_downside. 하방 변동성만 페널티 — 변동성 대신 손실에 초점.",
    "Calmar": "CAGR / |MDD|. 최대 낙폭 대비 연수익 — 손실 충격 대비 안정성.",
    "IR": "Information Ratio = (Fund - Benchmark) / Tracking Error. 액티브 운용 효율.",
    "M²": "Modigliani² (1997) — Sharpe 의 % 환산 (벤치마크 변동성에 매칭).",

    # ── Pool-3 위험 ──────────────────────────────────────────────
    "Volatility": "변동성 (연환산 표준편차). 가격 변동의 강도 — 낮을수록 안정적.",
    "MDD": "Maximum Drawdown — 고점 대비 최대 손실. 절대값이 작을수록 좋음.",
    "Downside Deviation": "하방 표준편차 (음수 수익률만). Sortino 분모.",
    "VaR 5%": "Historical VaR 5% — 최악 5% 분위수의 손실 (1개월 기준).",
    "CVaR 5%": "Conditional VaR (Expected Shortfall) — VaR 초과 시 평균 손실.",
    "Beta": "시장 베타 — SPY 대비 시장 노출도. 1보다 작으면 시장 변동에 덜 민감.",
    "Alpha": "Jensen's Alpha — CAPM 회귀 절편. 시장 대비 초과수익.",
    "R²": "결정계수 — 시장으로 펀드 변동을 설명하는 비율. 100% 에 가까울수록 시장 추종.",
    "Tracking Error": "추적오차 — 벤치마크 대비 액티브 위험. 낮을수록 추종, 높을수록 액티브.",
    "Jarque-Bera": "정규성 검정 통계량 — 수익률 분포가 정규분포를 따르는지 평가.",

    # ── Pool-4 운용 효율성 ─────────────────────────────────────────
    "Number of Holdings": "보유 종목 수.",
    "Effective N": "유효 종목 수 = 1 / Σw² — 분산도 척도. 높을수록 분산.",
    "Single Stock HHI": "개별 종목 집중도 = Σw² (Herfindahl-Hirschman Index). 낮을수록 분산.",
    "Sector HHI": "섹터 집중도 = Σ(섹터 weight)². 낮을수록 섹터 분산.",
    "Top 10 Weight": "상위 10 종목의 비중 합계. 낮을수록 분산.",
    "Avg Turnover": "월평균 회전율 — 매매 빈도. 낮을수록 안정 운용 (거래비용 ↓).",

    # ── Pool-5 시장 비교 ──────────────────────────────────────────
    "Win Rate": "양수 수익률 월 비율 — 월 단위로 이긴 비율.",
    "Up Capture": "시장 상승 시 펀드 상승 비율 — 100% 이상이면 시장보다 더 상승.",
    "Down Capture": "시장 하락 시 펀드 하락 비율 — 100% 미만이면 시장보다 덜 하락.",
    "Active Return": "Fund - Benchmark 차이 (월별 또는 연환산).",

    # ── Sector Watch 전용 ──────────────────────────────────────────
    "Avg |Tilt|": "섹터별 (펀드 weight - SPY weight) 절대값 평균 — 액티브 운용 강도.",
    "Active Bets": "|Tilt| > 1% 인 섹터 수 — 높을수록 액티브.",

    # ── Backtesting 전용 ──────────────────────────────────────────
    "Recovery Time": "Drawdown 발생 후 고점 회복까지 걸린 평균 개월 수.",
    "Stress Survival": "위기 구간에서의 펀드 생존율 (사용 폐기 — Avg Recovery Time 으로 대체)",
}


def get_tooltip(metric_name: str) -> str:
    """
    메트릭 이름 → tooltip 텍스트.
    정의가 없는 메트릭은 안내 메시지 반환 (KeyError 방지).
    """
    return METRIC_TOOLTIPS.get(metric_name, f"'{metric_name}' 정의 미정 — lib/tooltips.py 추가 필요.")
