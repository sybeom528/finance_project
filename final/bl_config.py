"""
bl_config.py — Black-Litterman 실험 정의

새 실험 추가 방법:
  - 기존 방식의 파라미터만 바꿀 때 → EXPERIMENTS에 dict 한 줄 추가
  - 새 계산 방식 도입 시 → bl_functions.py에 함수 추가 + 여기에 dict + 99_run dispatcher에 분기

슬롯 키 정리:
  p_mode    : 'trailing_vol21' | 'lstm_predicted'
  p_weight  : 'mcap' | 'eq' | 'rp' | 'vol_mcap'
  q_mode    : 'fixed' | 'ff3_regression' | 'realized_spread' | 'regime' | 'lambda' | 'raw_lam' | 'none'
  q_value   : float  (q_mode='fixed'|'lambda'|'raw_lam' 일 때 q_base로 사용, 기본 0.003)
  lam_mean  : float  (q_mode='lambda'|'raw_lam' 일 때 기준 λ, 기본 2.5)
  q_regime_table : dict (q_mode='regime' 일 때 사용)
  omega_mode: 'he_litterman' | 'scaled'
  omega_scale: float (omega_mode='scaled' 일 때 사용, 기본 1.0)
  prior     : 'capm_mcap' | 'capm_eq'
  tc        : float  (거래비용, 편도 turnover 기준, 기본 0.001 = 10bp)
  max_weight: float  (단일 종목 상한, 기본 0.10)
  lstm_pred_path: str | None  (p_mode='lstm_predicted' 또는 omega_mode='rmse' 시 경로)
"""
from pathlib import Path

# ── LSTM 예측 파일 경로 ────────────────────────────────────────────────────────
# 위치에 따라 자동 탐색. 없으면 LSTM 실험은 자동 스킵됨.
_PHASE3_DIR = Path(__file__).parent / 'phase3(data_outputs)'
_LSTM_PRED_DEFAULT = _PHASE3_DIR / 'data' / 'ensemble_predictions_stockwise.csv'

# ── 기준선 실험 (모든 실험의 default 값) ────────────────────────────────────
BASELINE = {
    'name'        : 'baseline',
    'p_mode'      : 'trailing_vol21',   # vol_21d 기준 분류
    'p_weight'    : 'mcap',             # 시총가중 P
    'q_mode'      : 'fixed',
    'q_value'     : 0.003,              # 월 0.3% (연 3.6%)
    'omega_mode'  : 'he_litterman',     # τ·P·Σ·P^T
    'omega_scale' : 1.0,
    'prior'       : 'capm_mcap',        # 시총가중 균형수익률
    'tc'          : 0.001,              # 10bp
    'max_weight'  : 0.10,               # 여기는 팀 합의가 필요한 부분. 몇 %까지 가져갈지에 대한 내용
    'lstm_pred_path': str(_LSTM_PRED_DEFAULT),
}

# ── 실험 목록 ────────────────────────────────────────────────────────────────
EXPERIMENTS = [

    # ── 기준선 (CAPM 시총가중 Prior, P 시총가중, vol21) ───────────────────────
    BASELINE,

    # ── [Prior] CAPM 시총가중 vs 1/N 균등가중 ────────────────────────────────
    {**BASELINE, 'name': 'prior_eq',
     'prior': 'capm_eq'},               # 1/N 균등가중 prior

    # ── [P 슬롯] P 행렬 가중 방식 ───────────────────────────────────────────────
    {**BASELINE, 'name': 'p_rp',
     'p_weight': 'rp'},                 # 1/σ 역변동성 가중

    {**BASELINE, 'name': 'p_eq',
     'p_weight': 'eq'},                 # 동일가중

    {**BASELINE, 'name': 'p_vol_mcap',
     'p_weight': 'vol_mcap'},           # 롱 (1/σ)×mcap, 숏 σ×mcap

    # ── [비교군] BL 없음 ──────────────────────────────────────────────────────
    {**BASELINE, 'name': 'capm_no_bl',
     'q_mode': 'capm'},                 # CAPM prior π 직접 최적화, 전체 유니버스, BL 없음

    {**BASELINE, 'name': 'naive_lowvol',
     'q_mode': 'none'},                 # 저변동 시총가중 직접 보유 (BL 생략)

    # ── [LSTM] CAPM 시총가중 Prior × LSTM 예측 vol ───────────────────────────
    {**BASELINE, 'name': 'p_lstm_mcap',
     'p_mode': 'lstm_predicted'},

    {**BASELINE, 'name': 'p_lstm_eq',
     'p_mode': 'lstm_predicted', 'p_weight': 'eq'},

    {**BASELINE, 'name': 'p_lstm_rp',
     'p_mode': 'lstm_predicted', 'p_weight': 'rp'},

    {**BASELINE, 'name': 'p_lstm_vol_mcap',
     'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap'},

    # ── [LSTM] 1/N Prior × LSTM 예측 vol ─────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_mcap',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_eq',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'eq'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_rp',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'rp'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_vol_mcap',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap'},

    # ── [Q_lambda] 시장 위험회피계수 λ 기반 Q 조절 ───────────────────────────
    # Q = q_base × clip(λ / lam_mean, 0.1, 3.0)
    # 시장 안정(λ↑) → Q 강화 / 시장 불안(λ↓) → Q 약화
    {**BASELINE, 'name': 'q_lambda',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # Q_lambda × LSTM vol P 조합
    {**BASELINE, 'name': 'q_lambda_p_lstm',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'p_mode': 'lstm_predicted'},

    # ── [inv_lambda_Q] 역방향 λ: 위기일수록 Q 강화 ───────────────────────────
    # λ 낮음(위기) → Q 높음, λ 높음(강세) → Q 낮음
    {**BASELINE, 'name': 'q_inv_lambda',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # inv_lambda × LSTM vol P 조합
    {**BASELINE, 'name': 'q_inv_lambda_p_lstm',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'p_mode': 'lstm_predicted'},

    # ── [raw_lam_Q] raw λ 부호 기반 자연 게이팅 ─────────────────────────────
    # SPY 하락 → lam_raw 음수 → Q=0 자연 도달 (하드스탑 없이)
    {**BASELINE, 'name': 'q_raw_lam',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── [최고성과 조합] prior_eq × lstm_rp × q_lambda ────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_rp_q_lambda',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # inv_lambda × LSTM P × prior_eq
    {**BASELINE, 'name': 'prior_eq_q_inv_lambda_p_lstm',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'mcap',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # lambda × LSTM P × prior_eq
    {**BASELINE, 'name': 'prior_eq_q_lambda_p_lstm',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'mcap',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── [논문 FF3] Q = 직전월 실현팩터 (Ω = he_litterman) ────────────────────
    {**BASELINE, 'name': 'q_ff3_paper',
     'q_mode': 'ff3_paper'},

    # ── [논문 완전 구현] Q = 직전월 실현팩터, Ω = 전월 예측오차² ─────────────
    {**BASELINE, 'name': 'q_ff3_paper_omega_paper',
     'q_mode': 'ff3_paper', 'omega_mode': 'ff3_paper'},

    # ── [논문 Ω만] fixed Q + 논문 방식 Ω ──────────────────────────────────────
    {**BASELINE, 'name': 'omega_paper',
     'omega_mode': 'ff3_paper'},

    # ── [논문 Ω + LSTM P] ──────────────────────────────────────────────────────
    {**BASELINE, 'name': 'omega_paper_p_lstm',
     'omega_mode': 'ff3_paper', 'p_mode': 'lstm_predicted'},

    # ════════════════════════════════════════════════════════════════════════
    # [윤서 추가 실험] prior 1/N × p_weight vol_mcap × Q 동적 조합 7개
    # ════════════════════════════════════════════════════════════════════════
    # 목적: baseline 대비 prior·p·q 슬롯별 변경 효과를 OFAT + 조합으로 검증
    # ════════════════════════════════════════════════════════════════════════

    # ── prior=1/N × Q 동적 ────────────────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_q_lambda',
     'prior': 'capm_eq', 'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_eq_q_raw_lam',
     'prior': 'capm_eq', 'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── p=vol_mcap × Q 동적 ───────────────────────────────────────────────
    {**BASELINE, 'name': 'p_vol_mcap_q_lambda',
     'p_weight': 'vol_mcap', 'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'p_vol_mcap_q_raw_lam',
     'p_weight': 'vol_mcap', 'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── prior=1/N × p=vol_mcap (Q fixed) ──────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_vol_mcap',
     'prior': 'capm_eq', 'p_weight': 'vol_mcap'},

    # ── 모두 변경: 1/N + vol_mcap + Q 동적 ────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_vol_mcap_q_lambda',
     'prior': 'capm_eq', 'p_weight': 'vol_mcap',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_eq_p_vol_mcap_q_raw_lam',
     'prior': 'capm_eq', 'p_weight': 'vol_mcap',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

]


def get_changed_slots(cfg: dict, baseline: dict = None) -> set:
    """
    baseline 대비 바뀐 슬롯 이름 반환.
    99_analyze.ipynb에서 조건부 차트 선택에 사용.
    """
    if baseline is None:
        baseline = BASELINE
    ignore = {'name', 'lstm_pred_path'}
    return {k for k in set(cfg) | set(baseline)
            if k not in ignore and cfg.get(k) != baseline.get(k)}
