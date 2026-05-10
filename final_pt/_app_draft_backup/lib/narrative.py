"""
narrative.py — 페이지별 한국어 설명 텍스트 + Top 1 추천 근거.

발표·시연용 스토리텔링 문구를 한 곳에 집약. UI 코드와 콘텐츠 분리.
"""

# ══════════════════════════════════════════════════════════════
# 발표용 Top 1 추천 (mat_eq_eq_lam_pap, 사용자 지정)
# ══════════════════════════════════════════════════════════════
TOP_1_NAME = 'mat_eq_eq_lam_pap'

TOP_1_NARRATIVE = '''
**🥇 Top 1 추천 모델 — `mat_eq_eq_lam_pap`**

본 모델은 **165 실험 중 Sortino 1위** 후보로, 4 슬롯 조합이 다음과 같이
보수적·안정 지향으로 calibrate 되어 있습니다:

| 슬롯 | 값 | 의미 |
|---|---|---|
| **prior** | `capm_eq` | 균등가중 prior — 시총 편향 회피 |
| **p_weight** | `eq` | 균등가중 P — 단일 종목 의존도 최소 |
| **q_mode** | `lambda` | 동적 Q (시장 안정 시 view 강화, 위기 시 약화) |
| **omega_mode** | `ff3_paper` | 직전월 예측오차² Bayesian rolling Ω |

**왜 Sortino 우선인가?** Sharpe 는 변동성 대비 평균이지만, Sortino 는
**하방 손실만 패널티** 합니다. 위험성향별 펀드 후보에서 보수적 투자자가
중시하는 "downside protection" 을 직접 측정. mat_eq_eq_lam_pap 은
HOLD-OUT (2024-2025 24m) 에서도 안정적 sortino 유지 → 학습편향 최소화 검증.
'''


# ══════════════════════════════════════════════════════════════
# 페이지별 hero / intro
# ══════════════════════════════════════════════════════════════
HOME_INTRO = '''
# 🎯 Low-Risk Anomaly + LSTM σ 예측 + Black-Litterman 통합 시연

> S&P 500 종목 대상으로 **LSTM 변동성 예측 + Black-Litterman 단일 view**
> 프레임워크에 **저변동성 anomaly** 를 결합한 위험성향별 펀드 후보 탐색 시스템.

**데이터 기간**: 2010-01-31 ~ 2025-12-31 (총 192 개월)
- 학습/calibration (TEST): 2010-01 ~ 2023-12 (168m)
- 실전 검증 (HOLD-OUT): 2024-01 ~ 2025-12 (24m)

**핵심 후보**: `mat_eq_eq_lam_pap` (Sortino 1위) — eq prior × eq P-weight × λ-dynamic Q × FF3-paper Ω
'''

EXPLORER_INTRO = '''
## 🔍 165 실험 마스터 테이블 + 단일 실험 6 패널 진단

표를 클릭·검색·필터하여 임의 실험을 골라 **누적수익·drawdown·rolling Sharpe·
포트폴리오 분산도·turnover** 6 패널을 즉시 확인.

- **default 정렬**: Sortino 내림차순 (사용자 우선 지표)
- **기간 토글**: TEST / HOLD-OUT / FULL
- **슬롯 필터**: prior × p_weight × q_mode × ω_mode 4 슬롯 multi-select
'''

COMPARE_INTRO = '''
## ⚖️ 다중 실험 비교 — Sortino 6 패널

여러 실험을 한 화면에 겹쳐 **상대 우위** 를 직관적으로 평가.
사용자 우선 지표인 Sortino 를 **3 패널 (bar / rolling / 하방편차)** 으로 강조.

- 누적수익률 (log)  /  Drawdown
- Sortino + CAGR + |MDD| bar  /  12m rolling Sortino
- 12m rolling Sharpe (보조)   /  12m rolling 하방편차
'''

MATRIX_INTRO = '''
## 🗺️ 매트릭스 히트맵 — prior × p_weight × q_mode × ω_mode

LSTM 고정 매트릭스 셀 **3 prior × 3 p_weight × 5 q_mode × 3 ω_mode = 135**
의 (메트릭) 분포를 한 눈에. 행/열 평균으로 단일 슬롯 효과 분리.

- **default metric**: Sortino (사용자 우선)
- **기간 토글**: TEST / HOLD-OUT / FULL
- 셀 클릭 → 사이드바에 누적수익률 미니차트
'''

RISK_PROFILE_INTRO = '''
## 🎯 위험성향별 추천 — 보수형 / 중립형 / 공격형

사용자가 **risk tolerance 슬라이더** 를 조작하면 165 실험 중 적합한
Top 3 후보가 자동으로 추출됩니다. **mat_eq_eq_lam_pap 은 보수형 1순위**
로 고정 노출됩니다 (사용자 지정).

| 위험허용도 | 매핑 | 정렬 키 |
|---|---|---|
| **0~30 (보수형)** | mat_eq_eq_lam_pap 우선 | sortino_TEST + low MDD |
| **30~70 (중립형)** | 균형 | sortino_ir × stability |
| **70~100 (공격형)** | max return | max sortino + alpha |
'''

VOLATILITY_INTRO = '''
## 📈 Phase 1.5 LSTM σ 예측 + Phase 3-2 학술 통계 요약

본 페이지는 BL 의 입력인 **변동성 예측 모델** 과 **사후 학술 검증** 결과를
요약 카드로 제공합니다. 자세한 재현은 03/04 노트북 참조.

**📊 RMSE 표기 — Option A (의미 보존)**
- 학술 표준 (log-RV 공간) RMSE 본값 **유지**
- 발표용 보조 표기 "평균 σ ≈ X%/일" 추가 (Patton 2011 표준 호환)
- log RMSE 와 % 공간 RMSE 는 다른 metric — 의사결정은 log RMSE 사용
'''


# ══════════════════════════════════════════════════════════════
# 슬롯 설명 (tooltip 용)
# ══════════════════════════════════════════════════════════════
SLOT_TOOLTIPS = {
    'prior'      : 'CAPM prior π 가중 — mcap (시총), eq (균등), rp (역변동)',
    'p_mode'     : 'P 분류용 변동성 — trailing_vol21/252 (실현) 또는 lstm_predicted',
    'p_weight'   : 'P 행렬 가중 — mcap, eq, rp, vol_mcap (전체 유니버스)',
    'q_mode'     : 'Q 결정 방식 — fixed, lambda(λ↑→Q↑), inv_lambda, raw_lam, vol_spread',
    'q_value'    : 'Q base 값 (기본 0.003 = 월 0.3%)',
    'omega_mode' : 'Ω 계산 — he_litterman (표준), rmse (LSTM 정확도 적응), ff3_paper (직전월 오차² Bayesian)',
    'tc'         : '편도 거래비용 (기본 0.001 = 10bp)',
    'max_weight' : '단일 종목 상한 (기본 0.10)',
}


# ══════════════════════════════════════════════════════════════
# 위험성향별 카드 description
# ══════════════════════════════════════════════════════════════
RISK_TIER_DESC = {
    'conservative': '''
**보수형** — 안정적 down-side protection 최우선.

핵심: MDD 최소화 + sortino_TEST 안정 + HOLD-OUT 일관성. 변동성 큰 시기에도
손실폭이 작은 모델 선호.
''',
    'balanced': '''
**중립형** — Sharpe·Sortino 균형 + 3-레짐 일관성.

핵심: sortino_ir 우선 (3-레짐 R1/R2/R3 평균/표준편차 비) — 시기별 robust
한 모델만 통과.
''',
    'aggressive': '''
**공격형** — 절대 sortino 와 alpha 우선.

핵심: max sortino_FULL + alpha (Jensen) — 시장 베타 대비 추가 수익을
중시. 변동성·MDD 일부 감수.
''',
}
