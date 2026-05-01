"""
bl_config.py — Black-Litterman 실험 정의

새 실험 추가 방법:
  - 기존 방식의 파라미터만 바꿀 때 → EXPERIMENTS에 dict 한 줄 추가
  - 새 계산 방식 도입 시 → bl_functions.py에 함수 추가 + 여기에 dict + 99_run dispatcher에 분기

슬롯 키 정리:
  p_mode    : 'trailing_vol21' | 'trailing_vol252' | 'lstm_predicted'
  p_weight  : 'mcap' | 'eq' | 'rp' | 'asymmetric' | 'vol_mcap'
  q_mode    : 'fixed' | 'ff3_regression' | 'realized_spread' | 'regime' | 'none'
  q_value   : float  (q_mode='fixed' 일 때 사용)
  q_regime_table : dict (q_mode='regime' 일 때 사용)
  omega_mode: 'he_litterman' | 'scaled' | 'rmse'
  omega_scale: float (omega_mode='scaled' 일 때 사용, 기본 1.0)
  prior     : 'capm_mcap' | 'capm_eq'
  tc        : float  (거래비용, 편도 turnover 기준, 기본 0.001 = 10bp)
  max_weight: float  (단일 종목 상한, 기본 0.10)
  lstm_pred_path: str | None  (p_mode='lstm_predicted' 또는 omega_mode='rmse' 시 경로)
"""
from pathlib import Path

# ── LSTM 예측 파일 경로 ────────────────────────────────────────────────────────
# 위치에 따라 자동 탐색. 없으면 LSTM 실험은 자동 스킵됨.
_candidate_phase3 = [
    Path(__file__).parent.parent / '시계열_test' / 'Phase3_Robust_Extensions',            # 서윤범/ 아래
    Path(__file__).parent.parent / '서윤범' / '시계열_test' / 'Phase3_Robust_Extensions',  # finance_project/ 아래
]
_PHASE3_DIR = next((p for p in _candidate_phase3 if p.exists()), _candidate_phase3[0])
_LSTM_PRED_DEFAULT = _PHASE3_DIR / 'results' / 'ensemble_predictions_stockwise.csv'

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

    # ── [P 슬롯] 변동성 측정 기간 ─────────────────────────────────────────────
    {**BASELINE, 'name': 'p_vol252',
     'p_mode': 'trailing_vol252'},      # 252일 장기 실현변동성

    # ── [P 슬롯] P 행렬 가중 방식 (5가지) ────────────────────────────────────
    {**BASELINE, 'name': 'p_rp',
     'p_weight': 'rp'},                 # 1/σ 역변동성 가중

    {**BASELINE, 'name': 'p_eq',
     'p_weight': 'eq'},                 # 동일가중

    {**BASELINE, 'name': 'p_vol_mcap',
     'p_weight': 'vol_mcap'},           # 롱 (1/σ)×mcap, 숏 σ×mcap

    # ── [1/N Prior + P 조합] ─────────────────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_vol252',
     'prior': 'capm_eq', 'p_mode': 'trailing_vol252'},

    {**BASELINE, 'name': 'prior_eq_p_rp',
     'prior': 'capm_eq', 'p_weight': 'rp'},

    {**BASELINE, 'name': 'prior_eq_p_eq',
     'prior': 'capm_eq', 'p_weight': 'eq'},

    {**BASELINE, 'name': 'prior_eq_p_vol_mcap',
     'prior': 'capm_eq', 'p_weight': 'vol_mcap'},

    # ── [비교군] BL 없음 ──────────────────────────────────────────────────────
    {**BASELINE, 'name': 'capm_no_bl',
     'q_mode': 'capm'},                 # CAPM prior π 직접 최적화, 전체 유니버스, BL 없음

    {**BASELINE, 'name': 'naive_lowvol',
     'q_mode': 'none'},                 # 저변동 시총가중 직접 보유 (BL 생략)

    {**BASELINE, 'name': 'naive_lowvol_rp',
     'q_mode': 'none', 'p_weight': 'rp'},  # 저변동 역변동성 가중 (BL 생략)

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
