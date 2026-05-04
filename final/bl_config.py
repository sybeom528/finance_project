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
  omega_mode: 'he_litterman' | 'scaled' | 'rmse' | 'rmse_per_ticker' | 'ewma'
  omega_scale: float (omega_mode='scaled' 일 때 사용, 기본 1.0)
  lambda    : float (omega_mode='ewma' 일 때 사용, EWMA 망각 계수. 기본 0.94)
  prior     : 'capm_mcap' | 'capm_eq' | 'capm_rp'
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

    # ── [비교군] BL 없음 ──────────────────────────────────────────────────────
    {**BASELINE, 'name': 'capm_no_bl',
     'q_mode': 'capm'},                 # CAPM prior π 직접 최적화, 전체 유니버스, BL 없음

    {**BASELINE, 'name': 'naive_lowvol',
     'q_mode': 'none'},                 # 저변동 시총가중 직접 보유 (BL 생략)

    {**BASELINE, 'name': 'naive_lowvol_rp',
     'q_mode': 'none', 'p_weight': 'rp'},  # 저변동 역변동성 가중 (BL 생략)

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

    # ── [Omega-RMSE 옵션1: 시점별 평균 RMSE] CAPM 시총 prior × LSTM ─────────
    # pred_date 이전 12개월 윈도우 안의 모든 종목·일자 abs_err의 sqrt-mean-square를
    # 단일 RMSE로 계산해 omega 스케일에 반영. RMSE 클수록 omega↑ → 뷰 신뢰도↓.
    {**BASELINE, 'name': 'p_lstm_mcap_omega_rmse',
     'p_mode': 'lstm_predicted', 'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'p_lstm_eq_omega_rmse',
     'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'p_lstm_rp_omega_rmse',
     'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'p_lstm_vol_mcap_omega_rmse',
     'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap', 'omega_mode': 'rmse'},

    # ── [Omega-RMSE 옵션1] 1/N prior × LSTM ──────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_mcap_omega_rmse',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_eq_omega_rmse',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_rp_omega_rmse',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_vol_mcap_omega_rmse',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap', 'omega_mode': 'rmse'},

    # ── [Omega-RMSE 옵션2: 종목별 가중 RMSE] CAPM 시총 prior × LSTM ────────
    # 종목별 누적 RMSE를 계산하고 P^2로 가중 결합한 단일 RMSE로 omega 스케일링.
    # 뷰의 실제 종목 구성(가중치 포함)에 따라 omega가 달라짐.
    {**BASELINE, 'name': 'p_lstm_mcap_omega_rmse_pt',
     'p_mode': 'lstm_predicted', 'omega_mode': 'rmse_per_ticker'},

    {**BASELINE, 'name': 'p_lstm_eq_omega_rmse_pt',
     'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'rmse_per_ticker'},

    {**BASELINE, 'name': 'p_lstm_rp_omega_rmse_pt',
     'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'rmse_per_ticker'},

    {**BASELINE, 'name': 'p_lstm_vol_mcap_omega_rmse_pt',
     'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap', 'omega_mode': 'rmse_per_ticker'},

    # ── [Omega-RMSE 옵션2] 1/N prior × LSTM ──────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_mcap_omega_rmse_pt',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'omega_mode': 'rmse_per_ticker'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_eq_omega_rmse_pt',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'rmse_per_ticker'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_rp_omega_rmse_pt',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'rmse_per_ticker'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_vol_mcap_omega_rmse_pt',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap', 'omega_mode': 'rmse_per_ticker'},

    # ── [Risk Parity Prior] vol_21d 기반 1/σ prior × p_weight 4종 ──────────
    # prior 시장가중치를 시총가중도 균등가중도 아닌 1/vol_21d 정규화로 설정.
    # 위험을 균등 분담하는 prior. P 행렬은 trailing_vol21 기준, omega는 he_litterman.
    {**BASELINE, 'name': 'prior_rp_p_mcap',
     'prior': 'capm_rp'},

    {**BASELINE, 'name': 'prior_rp_p_eq',
     'prior': 'capm_rp', 'p_weight': 'eq'},

    {**BASELINE, 'name': 'prior_rp_p_rp',
     'prior': 'capm_rp', 'p_weight': 'rp'},

    {**BASELINE, 'name': 'prior_rp_p_vol_mcap',
     'prior': 'capm_rp', 'p_weight': 'vol_mcap'},

    # ── [Omega-EWMA 시나리오 1: λ=0.825 ('lo')] ─────────────────────────────
    # Pyo & Lee (2018) 식 (17) 의 단순화 EWMA 형태: Ω_t = λΩ_{t-1} + (1-λ)e²_{t-1}
    # λ=0.825 → 반감기 3.6개월, 12개월 후 약 10% 안정화 (워밍업 1년 효과 모사)
    # 노이즈 큼 (ESS≈10) — 데이터 부족 시 단축 워밍업 가능성 검증용
    {**BASELINE, 'name': 'p_lstm_mcap_ewma_lo',
     'p_mode': 'lstm_predicted', 'omega_mode': 'ewma', 'lambda': 0.825},

    {**BASELINE, 'name': 'p_lstm_eq_ewma_lo',
     'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'ewma', 'lambda': 0.825},

    {**BASELINE, 'name': 'p_lstm_rp_ewma_lo',
     'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'ewma', 'lambda': 0.825},

    {**BASELINE, 'name': 'p_lstm_vol_mcap_ewma_lo',
     'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap', 'omega_mode': 'ewma', 'lambda': 0.825},

    # ── [Omega-EWMA 시나리오 2: λ=0.94 ('std')] ─────────────────────────────
    # RiskMetrics 1996 표준값. 반감기 11.2개월, 36개월 후 10% 안정화
    # 워밍업 36개월 OOS 안에 흡수 → 분석 시 2013-01~ 144개월로 trim 권장
    # ESS≈32, 안정성 우수
    {**BASELINE, 'name': 'p_lstm_mcap_ewma_std',
     'p_mode': 'lstm_predicted', 'omega_mode': 'ewma', 'lambda': 0.94},

    {**BASELINE, 'name': 'p_lstm_eq_ewma_std',
     'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'ewma', 'lambda': 0.94},

    {**BASELINE, 'name': 'p_lstm_rp_ewma_std',
     'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'ewma', 'lambda': 0.94},

    {**BASELINE, 'name': 'p_lstm_vol_mcap_ewma_std',
     'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap', 'omega_mode': 'ewma', 'lambda': 0.94},

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
