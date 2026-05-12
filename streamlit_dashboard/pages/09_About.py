"""
pages/09_About.py — About / FAQ 페이지 (6 영역)

영역 구조:
  1. Header
  2. Sub-header (페이지 컨텍스트)
  3. 펀드 소개 (1.1 정체성 / 1.2 운용 철학 / 1.3 프로젝트 메타)
  4. FAQ (2.1 펀드 일반 / 2.2 성과 관련 / 2.3 위험 관리 / 2.4 학술 토대)
  5. 데이터 출처
  6. Disclosure (3.1 표준 disclaimer / 3.2 위험 요인 5가지)
  7. Footer

설계 원칙:
  - 사용자 친화 표현 (코드 변수명 / 학술 용어 최소화)
  - Walk-forward 방식 명시 (사용자 정정 2026-05-12 반영 — overfitting / look-ahead bias 원천 차단)
  - 학술 가상 펀드 명시 (부트캠프 최종 프로젝트)

참조: docs/plan/03_pages/09_about.md, docs/decisionlog/09_about.md
"""

import streamlit as st

from lib.disclosure import init_session_state, render_footer
from lib.page_helpers import (
    inject_custom_css,
    render_page_header,
    render_sidebar,
    render_subheader,
)


# === 페이지 설정 ======================================================
inject_custom_css()
init_session_state()
render_sidebar()


# === 영역 1: Header ===================================================
render_page_header("About / FAQ", "소개 / 자주 묻는 질문")


# === 영역 2: Sub-header ===============================================
render_subheader(
    title_en="About this Fund",
    title_ko="펀드 소개 / FAQ / Disclosure",
    description=(
        "본 펀드의 개요, 운용 철학, 자주 묻는 질문, 데이터 출처, "
        "그리고 본 백테스트 결과를 해석하는 데 필요한 위험 고지 사항을 안내합니다."
    ),
)


# === 영역 3: 펀드 소개 =================================================
st.subheader("펀드 소개")

st.markdown(
    """
    #### 1.1 펀드 정체성

    **Adaptive Volatility Control Fund (변동성 인지 적응 펀드)** 는
    시장의 변동성을 예측하고 그에 맞춰 자산 배분을 조정하는 **학술 목적의 가상 펀드** 입니다.

    - **투자 대상 종목군**: S&P 500 편입 종목 (약 590개)
    - **운용 방식**: 매월 1회 (월말) 종목 비중 재조정
    - **거래비용 가정**: 편측 20bp (mid-cap 포함 액티브 ETF 수준의 보수적 가정)
    - **운용 기간 (백테스트)**: 2010-01 ~ 2025-12 (총 192개월 = 16년)
    """
)

st.markdown(
    """
    #### 1.2 운용 철학

    본 펀드는 두 가지 학술 모델을 결합하여 운용합니다:

    1. **LSTM 변동성 예측** — 시장 데이터를 기반으로 다음 시점의 변동성을 예측
    2. **Black-Litterman 자산 배분 산식** — 변동성 예측을 종목 비중으로 변환

    두 모델 모두 **Walk-forward 방식** 으로 적용됩니다 — 매월 시점의 운용 결정 시
    **그 시점 이전 데이터만 사용**하여 학습 / 산식 적용. 이로써 **미래 정보 누수 (look-ahead bias)**
    와 **백테스트 과적합 (overfitting)** 을 **원천 차단** 했습니다.

    또한 단일 종목 최대 비중을 10% 로 제한하여 **종목 분산** 을 유지하며,
    이는 단일 종목 / 단일 예측에 대한 의존도를 낮춥니다.
    """
)

st.markdown(
    """
    #### 1.3 프로젝트 메타

    - **위치**: 부트캠프 최종 프로젝트
    - **기간**: 2025년 4월 8일 ~ 2025년 5월 13일
    - **목적**: 학술 가상 펀드 — 실제 운용 / 판매 / 투자권유 X
    - **데이터**: 2010-01 ~ 2025-12 (Yahoo Finance + S&P 500 historical membership)
    """
)

st.divider()


# === 영역 4: FAQ ======================================================
st.subheader("자주 묻는 질문 (FAQ)")
st.caption(
    "본 펀드와 백테스트 결과에 대한 가장 자주 묻는 13개 질문을 4 카테고리로 정리했습니다. "
    "각 질문을 클릭하시면 답변이 펼쳐집니다."
)

# --- 2.1 펀드 일반 ---
st.markdown("#### 2.1 펀드 일반")

with st.expander("Q1. 이 펀드는 실제로 가입 가능한가요?"):
    st.markdown(
        "아닙니다. 본 펀드는 부트캠프 최종 프로젝트로 제작된 **학술 목적의 가상 펀드** 입니다. "
        "실제로 운용 / 판매되지 않으며, 투자권유 목적이 아닙니다."
    )

with st.expander("Q2. 어떤 종목을 어떻게 선정하나요?"):
    st.markdown(
        "S&P 500 편입 종목 (약 590개) 을 후보로 두고, 매월 변동성 예측 (LSTM) + "
        "Black-Litterman 산식을 결합하여 종목 비중을 결정합니다. **시점별 S&P 500 편입 종목만 사용**"
        "하여 생존편향 (survivorship bias) 을 차단합니다."
    )

with st.expander("Q3. 리밸런싱은 얼마나 자주 하나요?"):
    st.markdown(
        "**매월 1회, 월말** 에 모든 종목의 변동성 예측을 갱신하고 종목 비중을 재산출합니다."
    )

with st.expander("Q4. 거래비용은 반영되어 있나요?"):
    st.markdown(
        "네. **편측 20bp** (one-way 0.20%) 의 보수적 거래비용이 모든 성과 메트릭에 차감되어 있습니다. "
        "mid-cap 포함 액티브 ETF 수준의 보수적 가정입니다."
    )

# --- 2.2 성과 관련 ---
st.markdown("#### 2.2 성과 관련")

with st.expander("Q5. 왜 최근 24개월 (2024-2025) 에 시장 대비 부진했나요?"):
    st.markdown(
        "2024-2025년은 AI / 빅테크 (Nvidia, Microsoft, Meta 등) 가 시장 상승을 주도한 "
        "**집중 시기** 였습니다. 본 펀드는 변동성 인지 배분 특성상 IT 섹터 비중을 시장 대비 낮게 유지했고, "
        "이로 인해 해당 기간에 시장을 일시 하회했습니다. **변동성 인지 배분 전략의 정상 특성** 이며, "
        "TEST 168개월 운용 구간 (2010-2023, 14년) 에서는 시장 대비 우수한 성과를 보였습니다. "
        "(Sector Watch 페이지에서 자세한 분석을 확인하실 수 있습니다)"
    )

with st.expander("Q6. 백테스트 결과 = 미래 성과인가요?"):
    st.markdown(
        "아닙니다. 본 결과는 **과거 데이터 기반 모의 운용** 이며, 미래 시장 환경이 과거와 다르면 "
        "실제 성과는 보장되지 않습니다. (자세한 사항은 Disclosure 영역을 참고하십시오)"
    )

with st.expander("Q7. CAGR / Sortino / Sharpe — 어느 메트릭을 봐야 하나요?"):
    st.markdown(
        """
        각 메트릭은 다른 측면을 보여줍니다:

        - **CAGR (연평균 복리 수익률)**: 가장 기본적인 수익 지표
        - **Sortino**: 하방 위험 (손실 변동성) 만 고려한 수익률 — 안전성 측정
        - **Sharpe**: 총 변동성 대비 수익률 — 효율성 측정
        - **Active Return**: 시장 (SPY) 대비 초과 수익

        종합적으로 확인하시는 것을 권장합니다.
        """
    )

with st.expander("Q8. SPY 와 비교하는 것 외에 다른 벤치마크는 없나요?"):
    st.markdown(
        "사이드바에서 **균등가중 (EW)** / **역변동성 (IVW)** 벤치마크를 추가로 표시할 수 있습니다. "
        "EW = 모든 종목을 동일 비중, IVW = 변동성의 역수 가중치 — 본 펀드의 **BL+LSTM 효과** 를 "
        "더 정확히 측정하기 위한 비교 기준입니다."
    )

# --- 2.3 위험 관리 ---
st.markdown("#### 2.3 위험 관리")

with st.expander("Q9. 변동성 예측이 틀리면 어떻게 되나요?"):
    st.markdown(
        "본 펀드는 변동성 예측 하나에만 의존하지 않습니다. **Black-Litterman 의 보수적 산식** + "
        "**종목 분산** (단일 종목 최대 비중 10%) 으로 단일 종목 / 단일 예측 의존도를 낮춥니다. "
        "또한 매월 리밸런싱으로 예측 오차가 누적되지 않습니다."
    )

with st.expander("Q10. 최대낙폭 (MDD) -13.65% — 어느 정도 위험인가요?"):
    st.markdown(
        "TEST 168개월 운용 구간 (2010-2023) 의 최악 시점에서 자산 가치가 **-13.65% 하락** 한 경험을 의미합니다. "
        "일반적인 액티브 ETF (-20~-30%) 보다 낮은 수준입니다. 다만 **미래에 더 큰 낙폭이 발생할 "
        "가능성은 항상 존재** 합니다."
    )

with st.expander("Q11. Tail Risk (꼬리 위험) 은 어떻게 관리하나요?"):
    st.markdown(
        "일별 수익률 분포의 **왜도 (Skewness)** / **첨도 (Kurtosis)** / **꼬리 비율 (Tail Ratio)** "
        "및 **VaR / CVaR** 로 꼬리 위험을 측정합니다 "
        "(Performance 페이지 분포 통계, Risk Metrics 페이지 VaR/CVaR 분포 참조). "
        "본 펀드의 꼬리 분포는 시장 (SPY) 대비 양호한 편으로 측정됩니다."
    )

# --- 2.4 학술 토대 ---
st.markdown("#### 2.4 학술 토대")

with st.expander("Q12. 시점별로 어떻게 운용 결정을 내리나요?"):
    st.markdown(
        """
        **Walk-forward 방식** 으로 운용합니다. 매월 시점의 종목 비중을 결정할 때,
        **그 시점 이전 데이터만 사용** 하여:

        1. **LSTM 변동성 예측 모델** 을 그 시점 이전 데이터로 학습
        2. **Black-Litterman 산식** 을 그 시점 이전 약 5년 (1,250 영업일) 데이터로 적용

        이 방식은 **미래 정보 누수 (look-ahead bias)** 와 **백테스트 과적합 (overfitting)** 을
        **원천 차단** 합니다.
        """
    )

with st.expander("Q13. TEST 168개월 / Hold Out 24개월 분리는 어떤 의미인가요?"):
    st.markdown(
        """
        두 구간 모두 walk-forward 방식으로 운용된 결과이며, 분리는 **성과 비교 / 표시 목적** 입니다:

        - **TEST 168개월** (2010-01 ~ 2023-12, 14년): 14년간의 운용 결과
        - **Hold Out 24개월** (2024-01 ~ 2025-12, 2년): 최근 2년의 운용 결과

        두 구간의 성과 차이는 학습 기간에 따른 overfitting 이 아니라 (walk-forward 로 원천 차단),
        **해당 기간의 시장 환경 차이** 를 반영합니다. Hold Out 24개월 는 AI / 빅테크 집중 상승 시기였습니다.
        """
    )

st.divider()


# === 영역 5: 데이터 출처 ===============================================
st.subheader("데이터 출처")

st.markdown(
    """
    본 펀드의 백테스트와 모든 분석에 사용된 데이터의 출처를 안내합니다.

    | 데이터 | 출처 | 용도 |
    |---|---|---|
    | **일별 주가 (Adjusted Close)** | Yahoo Finance (`yfinance`) | 펀드 backtest 의 기초 데이터 + 일별 분포 통계 |
    | **S&P 500 편입 종목 명단 (시점별)** | S&P 공식 historical membership | 생존편향 (survivorship bias) 차단 — 시점별 실제 편입 종목만 사용 |
    | **S&P 500 종목 티커 매핑** | Wikipedia (S&P 500 종목 목록) | 종목 ticker ↔ 회사명 / GICS 섹터 매핑 정보 |
    | **무위험 수익률 (Risk-Free Rate)** | 3M Treasury Bill | Sharpe / Sortino / Information Ratio 산출 시 무위험 수익률 기준 |
    | **시장 벤치마크** | SPY ETF (Yahoo Finance) | 시장 대비 초과 수익 / Beta / Alpha 측정 |

    **데이터 기간**: 2010-01-01 ~ 2025-12-31 (총 16년, 192개월)

    > ℹ️ 본 데이터는 학술 목적으로 사용되었으며, 데이터 source 자체의 정확성은 제공자 (Yahoo Finance 등) 의 정책에 의존합니다.
    """
)

st.divider()


# === 영역 6: Disclosure ===============================================
st.subheader("Disclosure (위험 고지)")

# --- 3.1 표준 disclaimer ---
st.markdown("#### 3.1 표준 위험 고지")

st.warning(
    """
    ※ **Past performance is not indicative of future results.**
    ※ **과거의 운용성과는 미래의 수익을 보장하지 않습니다.**
    ※ 본 결과는 **백테스트 시뮬레이션** 이며 실제 운용 성과를 보장하지 않습니다.
    ※ 본 자료는 **투자권유를 목적으로 작성되지 않았습니다.**
    ※ 본 펀드는 **학술 목적의 가상 펀드** 이며, 실제 가입 / 운용 / 판매되지 않습니다.
    """
)

# --- 3.2 Risk Factor 5가지 ---
st.markdown("#### 3.2 위험 요인 (Risk Factors)")

st.markdown(
    "본 펀드의 백테스트 결과를 해석하실 때 다음 **5가지 위험 요인** 을 고려하셔야 합니다."
)

with st.expander("⚠️ Risk Factor 1 — 시장 위험 (Market Risk)"):
    st.markdown(
        "본 펀드는 S&P 500 종목으로 운용되므로 **미국 주식 시장의 전반적 변동성** 에 노출됩니다. "
        "시장 전체가 하락하는 시기에는 분산 운용 효과에도 불구하고 자산 가치가 함께 하락할 수 있습니다."
    )

with st.expander("⚠️ Risk Factor 2 — 섹터 집중 위험 (Sector Concentration Risk)"):
    st.markdown(
        "본 펀드는 변동성 기반 분산 운용을 추구하나, 특정 시장 환경 (예: AI 집중 상승 시기) 에서는 "
        "특정 섹터 (예: IT) 의 비중이 시장 대비 크게 다를 수 있습니다. 이 경우 **단기적으로 시장 대비 "
        "부진한 성과** 가 발생할 수 있으며, Hold Out 24개월 (2024-2025) 의 부진이 이러한 사례입니다. "
        "(자세한 분석은 Sector Watch 페이지 참조)"
    )

with st.expander("⚠️ Risk Factor 3 — 모델 위험 (Model Risk)"):
    st.markdown(
        "본 펀드는 **LSTM 변동성 예측 + Black-Litterman 산식** 을 기반으로 운용됩니다. "
        "두 모델 모두 학술적 근거가 있는 표준 방법이나, **모든 통계 모델은 본질적으로 가정에 기반** 합니다 "
        "(예: 변동성의 정상성, 분포의 정규성 등). 이 가정이 미래 시장 환경에서 깨질 경우 "
        "모델의 예측력이 저하될 수 있습니다."
    )

with st.expander("⚠️ Risk Factor 4 — 데이터 위험 (Data Risk)"):
    st.markdown(
        "본 펀드의 모든 분석은 **Yahoo Finance 의 Adjusted Close** 와 **S&P 500 historical membership** "
        "데이터에 의존합니다. 데이터 source 자체의 정확성 / 시점 / 분할 (split) / 배당 조정 등에서 "
        "오차가 발생할 가능성이 있으며, 이는 백테스트 결과에 영향을 미칠 수 있습니다."
    )

with st.expander("⚠️ Risk Factor 5 — 백테스트 일반 위험 (Backtest General Risk)"):
    st.markdown(
        """
        **본 펀드의 Walk-forward 방식**: 매월 시점의 운용 결정 시 LSTM 변동성 예측 +
        Black-Litterman 산식 모두 **그 시점 이전 데이터만 사용** 합니다. 이로써 학술적으로 가장
        문제가 되는:

        - **Look-ahead bias** (미래 정보 누수)
        - **Overfitting** (특정 학습 기간에 과도 적합)

        두 위험을 **원천 차단** 했습니다.

        **그래도 남는 일반적 백테스트 한계**:

        1. **모델 가정의 한계** — Black-Litterman / LSTM 모델 자체의 학술적 가정 (정규성, 변동성
           정상성 등) 이 미래 시장에서 깨질 가능성
        2. **데이터 source 한계** — Yahoo Finance Adjusted Close 정확성, S&P 500 편입 종목 한정
        3. **시장 환경 변화 위험 (Regime shift)** — 과거 16년 (2010-2025) 의 시장 환경과 본질적으로
           다른 미래 시장 환경에서는 모델의 적응 한계 가능

        **투자자 시사점**: 본 백테스트 결과는 **walk-forward 방식으로 학술적 정직성을 최대한 확보** 한
        모의 운용 결과이지만, **실제 운용 시 시장 환경 변화에 따라 성과 차이가 발생할 수 있습니다.**
        """
    )

# --- 컴플라이언스 표기 ---
st.caption(
    "본 자료는 **FINRA Rule 2210** (커뮤니케이션의 공정성 / 균형성 / 오해 방지) 의 학술 가상 펀드 "
    "표준 표현을 따르며, 한국 금융감독원의 펀드 광고 표준 표현에도 부합합니다."
)

st.divider()


# === 영역 7: Footer ===================================================
render_footer()
