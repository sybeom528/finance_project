"""
lib/search_index.py — 사이드바 영역 검색 기능

각 페이지의 주요 영역을 사전 정의한 index + 키워드 매칭 검색 함수.

설계 원칙:
- Header / Footer / Sub-header 등 모든 페이지 공통 영역은 제외
- 사용자가 검색할 만한 핵심 영역 (KPI / 차트 / 표 / 분석) 만 포함
- 한국어 + 영어 키워드 모두 매칭 (bilingual narrative 반영)
- 점수 기반 정렬 (area 이름 매칭 = 2점, keywords 매칭 = 1점)

사용: lib/page_helpers.py:render_sidebar() 에서 import 후 검색
"""

from __future__ import annotations


SEARCH_INDEX: list[dict] = [
    # ─── Overview (app.py) ────────────────────────────────────────────
    {"page": "Overview", "page_file": "app.py",
     "area": "핵심 성과 지표 (Hero KPI)",
     "keywords": ["kpi", "성과 지표", "hero", "cumulative", "cagr", "sortino",
                  "volatility", "mdd", "누적 수익"]},
    {"page": "Overview", "page_file": "app.py",
     "area": "누적 수익률 — Cumulative Return + Drawdown",
     "keywords": ["누적", "cumulative", "drawdown", "수익률 차트", "낙폭"]},
    {"page": "Overview", "page_file": "app.py",
     "area": "핵심 차별화 — Why this Fund",
     "keywords": ["차별화", "why", "differentiator", "강점", "카드"]},
    {"page": "Overview", "page_file": "app.py",
     "area": "페이지 둘러보기 — Explore",
     "keywords": ["navigation", "둘러보기", "explore", "이동"]},
    {"page": "Overview", "page_file": "app.py",
     "area": "Methodology Overview — BL+LSTM 흐름",
     "keywords": ["methodology", "방법론", "bl+lstm", "sankey", "흐름",
                  "black-litterman", "lstm"]},

    # ─── Investment Simulator ─────────────────────────────────────────
    {"page": "Investment Simulator", "page_file": "pages/02_Investment_Simulator.py",
     "area": "시뮬레이션 입력 (Lump-sum / DCA / Goal-based)",
     "keywords": ["시뮬레이션 입력", "simulator", "lump-sum", "dca",
                  "goal-based", "일시 투자", "적립식 투자", "목표 역산", "월말"]},
    {"page": "Investment Simulator", "page_file": "pages/02_Investment_Simulator.py",
     "area": "시뮬레이션 결과 (KPI 5개)",
     "keywords": ["시뮬레이션 결과", "result", "final value", "최종 자산",
                  "총 수익", "총 투자"]},
    {"page": "Investment Simulator", "page_file": "pages/02_Investment_Simulator.py",
     "area": "누적 자산 곡선",
     "keywords": ["누적 자산", "자산 곡선", "value series", "log scale"]},
    {"page": "Investment Simulator", "page_file": "pages/02_Investment_Simulator.py",
     "area": "인사이트 — Insights",
     "keywords": ["인사이트", "insights", "통찰", "박스"]},

    # ─── Performance ──────────────────────────────────────────────────
    {"page": "Performance", "page_file": "pages/03_Performance.py",
     "area": "핵심 성과 지표 (CAGR / Sortino / Sharpe / IR / Active Return)",
     "keywords": ["성과 kpi", "cagr", "sortino", "sharpe", "ir",
                  "information ratio", "active return", "액티브 수익"]},
    {"page": "Performance", "page_file": "pages/03_Performance.py",
     "area": "누적 수익률 — Performance Trend",
     "keywords": ["누적 수익률", "performance trend", "성과 추세"]},
    {"page": "Performance", "page_file": "pages/03_Performance.py",
     "area": "연간 수익률 — Annual Returns",
     "keywords": ["연간 수익률", "annual returns", "년도별", "막대"]},
    {"page": "Performance", "page_file": "pages/03_Performance.py",
     "area": "액티브 수익 분석 — Active Return Analysis",
     "keywords": ["액티브 수익", "active return", "tracking error", "추적 오차"]},
    {"page": "Performance", "page_file": "pages/03_Performance.py",
     "area": "연환산 Rolling 수익률 — Annualized Rolling Return",
     "keywords": ["rolling", "롤링", "연환산", "1년", "3년", "5년", "annualized"]},
    {"page": "Performance", "page_file": "pages/03_Performance.py",
     "area": "Regime 메트릭 비교 — Regime Heatmap",
     "keywords": ["regime", "heatmap", "시장 국면", "회복기", "확장기", "변동기"]},
    {"page": "Performance", "page_file": "pages/03_Performance.py",
     "area": "분포 통계 — Distribution Statistics",
     "keywords": ["분포 통계", "distribution", "skewness", "kurtosis",
                  "tail ratio", "왜도", "첨도", "꼬리 비율"]},

    # ─── Risk Metrics ─────────────────────────────────────────────────
    {"page": "Risk Metrics", "page_file": "pages/04_Risk_Metrics.py",
     "area": "위험 KPI (Sortino / Sharpe / MDD / Volatility / Calmar)",
     "keywords": ["위험 kpi", "risk kpi", "sortino", "sharpe", "mdd",
                  "volatility", "calmar", "낙폭"]},
    {"page": "Risk Metrics", "page_file": "pages/04_Risk_Metrics.py",
     "area": "Drawdown + Recovery Time",
     "keywords": ["drawdown", "recovery", "낙폭", "회복", "underwater"]},
    {"page": "Risk Metrics", "page_file": "pages/04_Risk_Metrics.py",
     "area": "Regime 메트릭 자세한 비교 (12 메트릭 × 5 Regime)",
     "keywords": ["regime 메트릭", "regime detail", "12 메트릭", "5 regime"]},
    {"page": "Risk Metrics", "page_file": "pages/04_Risk_Metrics.py",
     "area": "Sub-events 분석 — 4 위기",
     "keywords": ["sub-events", "위기", "금융위기", "covid", "유럽 위기",
                  "2008", "2020", "subprime"]},
    {"page": "Risk Metrics", "page_file": "pages/04_Risk_Metrics.py",
     "area": "VaR / CVaR 분포",
     "keywords": ["var", "cvar", "basel", "꼬리 위험", "5%", "value at risk"]},
    {"page": "Risk Metrics", "page_file": "pages/04_Risk_Metrics.py",
     "area": "Rolling 위험 메트릭 — Vol / Sortino / Beta / R² / TE",
     "keywords": ["rolling 위험", "rolling vol", "rolling sortino",
                  "rolling beta", "r²", "tracking error", "te"]},
    {"page": "Risk Metrics", "page_file": "pages/04_Risk_Metrics.py",
     "area": "위험 메트릭 종합 표 (15 메트릭, 4 카테고리)",
     "keywords": ["종합 표", "comprehensive", "15 메트릭",
                  "csv 다운로드", "카테고리"]},

    # ─── Holdings ─────────────────────────────────────────────────────
    {"page": "Holdings", "page_file": "pages/05_Holdings.py",
     "area": "보유 종목 KPI (Top10 / 회전율 / 집중도)",
     "keywords": ["holdings kpi", "보유 종목 kpi", "top10",
                  "회전율", "turnover", "집중도", "hhi"]},
    {"page": "Holdings", "page_file": "pages/05_Holdings.py",
     "area": "Top N Holdings — 선택 시점 snapshot",
     "keywords": ["top n", "top10", "보유 종목", "snapshot", "holdings",
                  "보유 기간"]},
    {"page": "Holdings", "page_file": "pages/05_Holdings.py",
     "area": "시가총액 분포 — Bubble + Treemap",
     "keywords": ["시가총액", "bubble", "treemap", "log_mcap", "size"]},
    {"page": "Holdings", "page_file": "pages/05_Holdings.py",
     "area": "섹터 변천사 — Multi-line",
     "keywords": ["섹터 변천사", "multi-line", "sector evolution"]},
    {"page": "Holdings", "page_file": "pages/05_Holdings.py",
     "area": "시점별 Top N 합 vs Others — 집중도 동적 추세",
     "keywords": ["top n 합", "others", "stacked area", "집중도 추세"]},
    {"page": "Holdings", "page_file": "pages/05_Holdings.py",
     "area": "종목별 기여도 분석 — Tornado",
     "keywords": ["기여도", "contribution", "tornado", "attribution"]},

    # ─── Sector Watch ─────────────────────────────────────────────────
    {"page": "Sector Watch", "page_file": "pages/06_Sector_Watch.py",
     "area": "섹터 KPI",
     "keywords": ["섹터 kpi", "sector summary"]},
    {"page": "Sector Watch", "page_file": "pages/06_Sector_Watch.py",
     "area": "Sector Treemap — Fund vs SPY",
     "keywords": ["sector treemap", "fund vs spy", "섹터 비중"]},
    {"page": "Sector Watch", "page_file": "pages/06_Sector_Watch.py",
     "area": "Sector Decomposition — Latest snapshot",
     "keywords": ["sector decomposition", "섹터 분해", "snapshot"]},
    {"page": "Sector Watch", "page_file": "pages/06_Sector_Watch.py",
     "area": "Sector Tilt vs SPY — 섹터 비중 차이",
     "keywords": ["sector tilt", "tilt", "tornado", "섹터 비중 차이"]},
    {"page": "Sector Watch", "page_file": "pages/06_Sector_Watch.py",
     "area": "Sector Rotation — 시계열 변화",
     "keywords": ["sector rotation", "rotation", "시계열 변화"]},
    {"page": "Sector Watch", "page_file": "pages/06_Sector_Watch.py",
     "area": "Hold Out 24개월 분석 (정당화 narrative)",
     "keywords": ["hold out", "ho", "24개월", "정당화", "ai rally",
                  "ai 빅테크"]},

    # ─── About / FAQ ──────────────────────────────────────────────────
    {"page": "About / FAQ", "page_file": "pages/09_About.py",
     "area": "펀드 소개 (정체성 / 운용 철학 / 프로젝트 메타)",
     "keywords": ["펀드 소개", "about", "정체성", "운용 철학", "프로젝트",
                  "팀", "변동성 인지 적응"]},
    {"page": "About / FAQ", "page_file": "pages/09_About.py",
     "area": "자주 묻는 질문 (FAQ, 13개)",
     "keywords": ["faq", "자주 묻는 질문", "질문", "답변"]},
    {"page": "About / FAQ", "page_file": "pages/09_About.py",
     "area": "데이터 출처",
     "keywords": ["데이터 출처", "yfinance", "yahoo finance",
                  "s&p 500", "membership"]},
    {"page": "About / FAQ", "page_file": "pages/09_About.py",
     "area": "Disclosure (위험 고지 + 5 Risk Factor)",
     "keywords": ["disclosure", "위험 고지", "risk factor", "면책",
                  "backtest", "overfitting"]},
]


def search_areas(query: str, max_results: int = 10) -> list[dict]:
    """
    키워드 매칭으로 영역 검색.

    매칭 점수:
      - area 이름에 query 부분 포함 → 2점
      - keywords 중 하나에 query 부분 포함 → 1점

    Args:
        query: 검색어 (case-insensitive)
        max_results: 최대 결과 수 (기본 10)

    Returns:
        매칭된 영역 list (점수 내림차순, 최대 max_results 개)
    """
    if not query or not query.strip():
        return []

    q = query.lower().strip()
    scored: list[tuple[int, dict]] = []

    for entry in SEARCH_INDEX:
        score = 0
        # area 이름 매칭 (2점)
        if q in entry["area"].lower():
            score += 2
        # keywords 매칭 (1점, 1번만 카운트)
        for kw in entry["keywords"]:
            if q in kw.lower():
                score += 1
                break
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    return [entry for _, entry in scored[:max_results]]
