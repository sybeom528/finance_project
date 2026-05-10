"""
bl_config.py — Black-Litterman 실험 정의

새 실험 추가 방법:
  - 기존 방식의 파라미터만 바꿀 때 → EXPERIMENTS에 dict 한 줄 추가
  - 새 계산 방식 도입 시 → bl_functions.py에 함수 추가 + 여기에 dict + 99_run dispatcher에 분기

슬롯 키 정리 (2026-05-07 갱신):
  p_mode    : 'trailing_vol21' | 'trailing_vol252' | 'lstm_predicted'
  p_weight  : 'mcap' | 'eq' | 'rp' | 'vol_mcap'
  q_mode    : 'fixed' | 'lambda' | 'raw_lam' | 'inv_lambda' | 'vol_spread' | 'ff3_paper' | 'none' | 'capm'
  q_value   : float  (q_mode='fixed' 또는 동적 모드 base, 기본 0.003)
  lam_mean  : float  (q_mode='lambda'|'raw_lam'|'inv_lambda' 일 때 기준 λ, 기본 2.5)
  omega_mode: 'he_litterman' | 'rmse' | 'ff3_paper'
              (scaled는 신뢰성 부족으로 제거됨, 2026-05-07)
  prior     : 'capm_mcap' | 'capm_eq' | 'capm_rp'   # capm_rp = 1/σ 정규화 Risk Parity
  tc        : float  (편측(per-side) 거래비용, 기본 0.003 = 30bp)
                     turnover는 Σ|Δw| ∈ [0,2] (two-way) → TC = turnover × tc
                     매수 한 번 30bp + 매도 한 번 30bp 카운트
  max_weight: float  (단일 종목 상한, 기본 0.10)
  lstm_pred_path: str | None  (p_mode='lstm_predicted' 또는 omega_mode='rmse' 시 경로)
"""
from pathlib import Path

# ── 평가 기간 (Train/Test/Hold-out 분리, 2026-05-07 도입) ───────────────────
# walk_forward 는 PRED_END 까지 수익률 시리즈 생성. 평가는 EVAL_PERIODS 단위로 분리.
PRED_START = '2010-01-01'
PRED_END   = '2025-12-31'    # 기존 2024-12-31 → 12 month hold-out 추가

EVAL_PERIODS = {
    'TEST'    : ('2010-01-01', '2023-12-31'),  # 168m, walk_forward training/calibration
    'HOLD_OUT': ('2024-01-01', '2025-12-31'),  # 24m, 실전 운용 검증
    'FULL'    : ('2010-01-01', '2025-12-31'),  # 192m, 보조 통합 비교
}

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
    'tc'          : 0.003,              # 30bp
    'max_weight'  : 0.10,               # 여기는 팀 합의가 필요한 부분. 몇 %까지 가져갈지에 대한 내용
    'lstm_pred_path': str(_LSTM_PRED_DEFAULT),
}

# ── 실험 목록 ──────────────────────────────────────────────────────────
# 총 156개 = 비매트릭스 21 + 매트릭스 135 (LSTM 고정, prior 3 × pw 3 × q 5 × Ω 3)
EXPERIMENTS = [

    # ═══════════════════════════════════════════════════════════════
    # [0] BASELINE (기준점)
    # prior=capm_mcap, p_mode=trailing_vol21, p_weight=mcap,
    # q_mode=fixed (q_value=0.003), omega=he_litterman
    # ═══════════════════════════════════════════════════════════════
    BASELINE,

    # ═══════════════════════════════════════════════════════════════
    # [1] BL 미사용 비교군 (Q 슬롯으로 BL 우회)
    # ═══════════════════════════════════════════════════════════════
    {**BASELINE, 'name': 'capm_no_bl', 'q_mode': 'capm'},
    {**BASELINE, 'name': 'naive_lowvol', 'q_mode': 'none'},

    # ═══════════════════════════════════════════════════════════════
    # [2] Trailing 단일 슬롯 변형 (prior / p_weight)
    # ═══════════════════════════════════════════════════════════════
    {**BASELINE, 'name': 'prior_eq', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'prior_rp', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'p_eq', 'p_weight': 'eq'},
    {**BASELINE, 'name': 'p_rp', 'p_weight': 'rp'},
    {**BASELINE, 'name': 'p_vol_mcap', 'p_weight': 'vol_mcap'},

    # ═══════════════════════════════════════════════════════════════
    # [3] Trailing × Q 동적 슬롯
    # ═══════════════════════════════════════════════════════════════
    {**BASELINE, 'name': 'q_lambda', 'q_mode': 'lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'q_inv_lambda', 'q_mode': 'inv_lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'q_raw_lam', 'q_mode': 'raw_lam', 'lam_mean': 2.5},

    # ═══════════════════════════════════════════════════════════════
    # [4] Trailing × prior_eq + Q 동적
    # ═══════════════════════════════════════════════════════════════
    {**BASELINE, 'name': 'prior_eq_q_lambda', 'q_mode': 'lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'prior_eq_q_raw_lam', 'q_mode': 'raw_lam', 'prior': 'capm_eq', 'lam_mean': 2.5},

    # ═══════════════════════════════════════════════════════════════
    # [5] Trailing × Ω 변형 (ff3_paper / rmse)
    # ═══════════════════════════════════════════════════════════════
    {**BASELINE, 'name': 'omega_paper', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'omega_rmse', 'omega_mode': 'rmse'},
    {**BASELINE, 'name': 'q_ff3_paper', 'q_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'q_ff3_paper_omega_paper', 'q_mode': 'ff3_paper', 'omega_mode': 'ff3_paper'},

    # ═══════════════════════════════════════════════════════════════
    # [6] LSTM × vol_mcap (vol_mcap은 매트릭스 외 4번째 p_weight)
    # 참고: trailing × vol_mcap = p_vol_mcap (위 [2]에 포함)
    # ═══════════════════════════════════════════════════════════════
    {**BASELINE, 'name': 'p_lstm_vol_mcap', 'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap'},

    # ═══════════════════════════════════════════════════════════════
    # [7] Q 민감도 — baseline 단일 후보 × q_value 4종 (BAB 학술 평균 sweep)
    # ═══════════════════════════════════════════════════════════════
    {**BASELINE, 'name': 'baseline_q55', 'q_value': 0.0055},
    {**BASELINE, 'name': 'baseline_q64', 'q_value': 0.0064},
    {**BASELINE, 'name': 'baseline_q70', 'q_value': 0.007},

    # ═══════════════════════════════════════════════════════════════
    # [7B] Q 민감도 — winner (mat_eq_eq_lam_pap) × q_value sweep
    # 목적: 선정된 winner 슬롯에서 q_value 변화에 대한 robustness 검증
    # 두 sweep:
    #   - Fine-grained: [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
    #   - BAB 학술: [0.0055, 0.0064] (q=0.003, 0.007은 fine-grained sweep과 중복)
    # 기본 winner = q_value 0.003 (이미 mat_eq_eq_lam_pap로 정의됨)
    # naming: winner_qXX 에서 XX = q_value × 10000 (e.g., 0.0055 → q55)
    # ═══════════════════════════════════════════════════════════════
    # Fine-grained sweep (q=0.003은 winner와 중복이라 제외)
    {**BASELINE, 'name': 'winner_q10',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.001},
    {**BASELINE, 'name': 'winner_q20',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.002},
    {**BASELINE, 'name': 'winner_q40',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.004},
    {**BASELINE, 'name': 'winner_q50',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.005},
    {**BASELINE, 'name': 'winner_q60',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.006},
    {**BASELINE, 'name': 'winner_q70',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.007},
    {**BASELINE, 'name': 'winner_q80',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.008},
    {**BASELINE, 'name': 'winner_q90',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.009},
    {**BASELINE, 'name': 'winner_q100', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.010},
    # BAB 학술 평균 (위 grid에 없는 값만)
    {**BASELINE, 'name': 'winner_q55',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.0055},
    {**BASELINE, 'name': 'winner_q64',  'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'q_value': 0.0064},

    # ═══════════════════════════════════════════════════════════════
    # [7C] PCT_GROUP 민감도 — winner × pct_group sweep
    # 목적: P 행렬의 분류 임계값(default 0.30 = 저변동 30% / 고변동 30%)이
    #       sortino/sharpe/MDD에 미치는 영향 검증
    # winner default pct_group=0.30 (mat_eq_eq_lam_pap에 명시 안 되었으므로 글로벌 PCT_GROUP=0.30 사용)
    # ═══════════════════════════════════════════════════════════════
    {**BASELINE, 'name': 'winner_pct10', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'pct_group': 0.10},
    {**BASELINE, 'name': 'winner_pct15', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'pct_group': 0.15},
    {**BASELINE, 'name': 'winner_pct20', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'pct_group': 0.20},
    {**BASELINE, 'name': 'winner_pct25', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'pct_group': 0.25},
    # winner default = pct_group 0.30 (mat_eq_eq_lam_pap)
    {**BASELINE, 'name': 'winner_pct35', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'pct_group': 0.35},
    {**BASELINE, 'name': 'winner_pct40', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5, 'pct_group': 0.40},

    # ═══════════════════════════════════════════════════════════════
    # [8] 매트릭스 (LSTM 고정, mat_{prior}_{pw}_{q}_{Ω}, 총 135 cells)
    # prior ∈ {mcap, eq, rp} × p_weight ∈ {mcap, eq, rp}
    # × q_mode ∈ {fix, lam, raw, inv, vsp} × omega ∈ {he, pap, rms}
    # 정렬: prior → p_weight → q_mode → omega 순
    # ═══════════════════════════════════════════════════════════════
    # ── prior=mcap × pw=mcap ────────────────────────────────
    {**BASELINE, 'name': 'mat_mcap_mcap_fix_he', 'p_mode': 'lstm_predicted'},
    {**BASELINE, 'name': 'mat_mcap_mcap_fix_pap', 'p_mode': 'lstm_predicted', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'mat_mcap_mcap_fix_rms', 'p_mode': 'lstm_predicted', 'omega_mode': 'rmse'},
    {**BASELINE, 'name': 'mat_mcap_mcap_lam_he', 'p_mode': 'lstm_predicted', 'q_mode': 'lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_lam_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_lam_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'lambda', 'omega_mode': 'rmse', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_raw_he', 'p_mode': 'lstm_predicted', 'q_mode': 'raw_lam', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_raw_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_raw_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'raw_lam', 'omega_mode': 'rmse', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_inv_he', 'p_mode': 'lstm_predicted', 'q_mode': 'inv_lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_inv_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_inv_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'inv_lambda', 'omega_mode': 'rmse', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_vsp_he', 'p_mode': 'lstm_predicted', 'q_mode': 'vol_spread'},
    {**BASELINE, 'name': 'mat_mcap_mcap_vsp_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'mat_mcap_mcap_vsp_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'vol_spread', 'omega_mode': 'rmse'},
    # ── prior=mcap × pw=eq ────────────────────────────────
    {**BASELINE, 'name': 'mat_mcap_eq_fix_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq'},
    {**BASELINE, 'name': 'mat_mcap_eq_fix_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'mat_mcap_eq_fix_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'rmse'},
    {**BASELINE, 'name': 'mat_mcap_eq_lam_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_lam_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_lam_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'rmse', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_raw_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_raw_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_raw_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'omega_mode': 'rmse', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_inv_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_inv_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_inv_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'omega_mode': 'rmse', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_vsp_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'vol_spread'},
    {**BASELINE, 'name': 'mat_mcap_eq_vsp_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'mat_mcap_eq_vsp_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'omega_mode': 'rmse'},
    # ── prior=mcap × pw=rp ────────────────────────────────
    {**BASELINE, 'name': 'mat_mcap_rp_fix_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp'},
    {**BASELINE, 'name': 'mat_mcap_rp_fix_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'mat_mcap_rp_fix_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'rmse'},
    {**BASELINE, 'name': 'mat_mcap_rp_lam_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_lam_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_lam_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'lambda', 'omega_mode': 'rmse', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_raw_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_raw_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_raw_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'omega_mode': 'rmse', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_inv_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_inv_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_inv_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'omega_mode': 'rmse', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_vsp_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'vol_spread'},
    {**BASELINE, 'name': 'mat_mcap_rp_vsp_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'mat_mcap_rp_vsp_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'rmse'},
    # ── prior=eq × pw=mcap ────────────────────────────────
    {**BASELINE, 'name': 'mat_eq_mcap_fix_he', 'p_mode': 'lstm_predicted', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_mcap_fix_pap', 'p_mode': 'lstm_predicted', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_mcap_fix_rms', 'p_mode': 'lstm_predicted', 'omega_mode': 'rmse', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_mcap_lam_he', 'p_mode': 'lstm_predicted', 'q_mode': 'lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_lam_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_lam_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'lambda', 'omega_mode': 'rmse', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_raw_he', 'p_mode': 'lstm_predicted', 'q_mode': 'raw_lam', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_raw_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_raw_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'raw_lam', 'omega_mode': 'rmse', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_inv_he', 'p_mode': 'lstm_predicted', 'q_mode': 'inv_lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_inv_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_inv_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'inv_lambda', 'omega_mode': 'rmse', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_vsp_he', 'p_mode': 'lstm_predicted', 'q_mode': 'vol_spread', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_mcap_vsp_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_mcap_vsp_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'vol_spread', 'omega_mode': 'rmse', 'prior': 'capm_eq'},
    # ── prior=eq × pw=eq ────────────────────────────────
    {**BASELINE, 'name': 'mat_eq_eq_fix_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_eq_fix_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_eq_fix_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'rmse', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_eq_lam_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_lam_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_lam_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'rmse', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_raw_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_raw_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_raw_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'omega_mode': 'rmse', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_inv_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_inv_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_inv_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'omega_mode': 'rmse', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_vsp_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_eq_vsp_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_eq_vsp_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'omega_mode': 'rmse', 'prior': 'capm_eq'},
    # ── prior=eq × pw=rp ────────────────────────────────
    {**BASELINE, 'name': 'mat_eq_rp_fix_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_rp_fix_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_rp_fix_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'rmse', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_rp_lam_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_lam_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_lam_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'lambda', 'omega_mode': 'rmse', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_raw_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_raw_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_raw_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'omega_mode': 'rmse', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_inv_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_inv_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_inv_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'omega_mode': 'rmse', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_vsp_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_rp_vsp_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_rp_vsp_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'rmse', 'prior': 'capm_eq'},
    # ── prior=rp × pw=mcap ────────────────────────────────
    {**BASELINE, 'name': 'mat_rp_mcap_fix_he', 'p_mode': 'lstm_predicted', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_mcap_fix_pap', 'p_mode': 'lstm_predicted', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_mcap_fix_rms', 'p_mode': 'lstm_predicted', 'omega_mode': 'rmse', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_mcap_lam_he', 'p_mode': 'lstm_predicted', 'q_mode': 'lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_lam_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_lam_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'lambda', 'omega_mode': 'rmse', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_raw_he', 'p_mode': 'lstm_predicted', 'q_mode': 'raw_lam', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_raw_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_raw_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'raw_lam', 'omega_mode': 'rmse', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_inv_he', 'p_mode': 'lstm_predicted', 'q_mode': 'inv_lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_inv_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_inv_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'inv_lambda', 'omega_mode': 'rmse', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_vsp_he', 'p_mode': 'lstm_predicted', 'q_mode': 'vol_spread', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_mcap_vsp_pap', 'p_mode': 'lstm_predicted', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_mcap_vsp_rms', 'p_mode': 'lstm_predicted', 'q_mode': 'vol_spread', 'omega_mode': 'rmse', 'prior': 'capm_rp'},
    # ── prior=rp × pw=eq ────────────────────────────────
    {**BASELINE, 'name': 'mat_rp_eq_fix_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_eq_fix_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_eq_fix_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'omega_mode': 'rmse', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_eq_lam_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_lam_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_lam_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'rmse', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_raw_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_raw_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_raw_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'omega_mode': 'rmse', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_inv_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_inv_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_inv_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'omega_mode': 'rmse', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_vsp_he', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_eq_vsp_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_eq_vsp_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'omega_mode': 'rmse', 'prior': 'capm_rp'},
    # ── prior=rp × pw=rp ────────────────────────────────
    {**BASELINE, 'name': 'mat_rp_rp_fix_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_rp_fix_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_rp_fix_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'omega_mode': 'rmse', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_rp_lam_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_lam_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_lam_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'lambda', 'omega_mode': 'rmse', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_raw_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_raw_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_raw_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'omega_mode': 'rmse', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_inv_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_inv_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_inv_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'omega_mode': 'rmse', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_vsp_he', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_rp_vsp_pap', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_rp_vsp_rms', 'p_mode': 'lstm_predicted', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'rmse', 'prior': 'capm_rp'},

    # ═══════════════════════════════════════════════════════════════
    # [paper-context] 논문 anchor (mcap_tr_mcap_ff3_pap) 에서 1슬롯만 변경
    # 99_slot_effects.ipynb의 plot_anchor_compare 빨간 line dense화 용
    # paper-strict: prior=mcap, p=tr, pw=mcap, q=ff3, om=pap 중 1개만 변경
    # ═══════════════════════════════════════════════════════════════
    # vary q (prior=mcap, p=tr, pw=mcap, om=pap 고정)
    {**BASELINE, 'name': 'paperctx_q_lam', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'paperctx_q_raw', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'paperctx_q_inv', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'paperctx_q_vsp', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper'},
    # vary prior (p=tr, pw=mcap, q=ff3, om=pap 고정)
    {**BASELINE, 'name': 'paperctx_prior_eq', 'prior': 'capm_eq', 'q_mode': 'ff3_paper', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'paperctx_prior_rp', 'prior': 'capm_rp', 'q_mode': 'ff3_paper', 'omega_mode': 'ff3_paper'},
    # vary pw (prior=mcap, p=tr, q=ff3, om=pap 고정)
    {**BASELINE, 'name': 'paperctx_pw_eq', 'p_weight': 'eq', 'q_mode': 'ff3_paper', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'paperctx_pw_rp', 'p_weight': 'rp', 'q_mode': 'ff3_paper', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'paperctx_pw_volm', 'p_weight': 'vol_mcap', 'q_mode': 'ff3_paper', 'omega_mode': 'ff3_paper'},
    # vary om (prior=mcap, p=tr, pw=mcap, q=ff3 고정) — pap/he는 이미 있음, rms만 추가
    {**BASELINE, 'name': 'paperctx_om_rms', 'q_mode': 'ff3_paper', 'omega_mode': 'rmse'},
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
