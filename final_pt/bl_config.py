"""
bl_config.py — Black-Litterman 실험 정의

새 실험 추가 방법:
  - 기존 방식의 파라미터만 바꿀 때 → EXPERIMENTS에 dict 한 줄 추가
  - 새 계산 방식 도입 시 → bl_functions.py에 함수 추가 + 여기에 dict + 04_BL_Walkforward dispatcher에 분기

슬롯 키 정리 (2026-05-11 갱신):
  p_mode    : 'lstm_predicted'  # 단일 옵션 — trailing 옵션은 2026-05-11 제거됨
  p_weight  : 'mcap' | 'eq' | 'rp'
  q_mode    : 'fixed' | 'lambda' | 'raw_lam' | 'inv_lambda' | 'vol_spread'
  q_value   : float  (q_mode='fixed' 또는 동적 모드 base, 기본 0.003)
  lam_mean  : float  (q_mode='lambda'|'raw_lam'|'inv_lambda' 일 때 기준 λ, 기본 2.5)
  omega_mode: 'he_litterman' | 'ff3_paper'
              (scaled / rmse 는 신뢰성 부족 / 의존성 정리로 제거됨)
  prior     : 'capm_mcap' | 'capm_eq' | 'capm_rp'   # capm_rp = 1/σ 정규화 Risk Parity
  tc        : float  (편측(per-side) 거래비용, 기본 0.003 = 30bp)
                     turnover는 Σ|Δw| ∈ [0,2] (two-way) → TC = turnover × tc
                     매수 한 번 30bp + 매도 한 번 30bp 카운트
  max_weight: float  (단일 종목 상한, 기본 0.10)
  lstm_pred_path: str  (LSTM 예측 csv 경로 — 03b 산출물)

⚠ 변경 이력 (2026-05-11):
  - p_mode: trailing_vol21/252 제거 → lstm_predicted 단일 (30 슬롯 삭제)
  - p_weight: vol_mcap 제거 (연속 가중치, 학술 비교 어려움) → mcap/eq/rp 만 (1 슬롯 삭제)
  - q_mode: ff3_paper 제거 (이미 trailing 와 함께 사라짐)
  - omega_mode: rmse 제거 (LSTM RMSE 의존성 단순화) → he_litterman/ff3_paper 만 (45 슬롯 삭제)
  - 최종: 매트릭스 90 슬롯 (winner_q/pct sweep 은 99_analyze 에서 in-code 처리)
  - 모든 슬롯이 LSTM 예측 변동성 (03b ensemble_predictions_stockwise.csv) 사용
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
# 03b_Volatility_Forecasting.ipynb 산출물. 없으면 모든 실험 스킵.
_PHASE3_DIR = Path(__file__).parent / 'phase3(data_outputs)'
_LSTM_PRED_DEFAULT = _PHASE3_DIR / 'data' / 'ensemble_predictions_stockwise.csv'

# ── 공통 default (모든 실험의 default 값, 더 이상 단독 슬롯 아님) ──────────────
# baseline 슬롯은 trailing 제거로 삭제됨. 본 dict 는 다른 슬롯의 template 으로만 사용.
BASELINE = {
    'p_mode'      : 'lstm_predicted',   # LSTM 예측 변동성 (03b 산출물) — 유일 옵션
    'p_weight'    : 'mcap',             # 시총가중 P
    'q_mode'      : 'fixed',
    'q_value'     : 0.003,              # 월 0.3% (연 3.6%)
    'omega_mode'  : 'he_litterman',     # τ·P·Σ·P^T
    'omega_scale' : 1.0,
    'prior'       : 'capm_mcap',        # 시총가중 균형수익률
    'tc'          : 0.003,              # 30bp
    'max_weight'  : 0.10,
    'lstm_pred_path': str(_LSTM_PRED_DEFAULT),
}

# ── 실험 목록 ──────────────────────────────────────────────────────────
# 총 90개 (매트릭스만 — LSTM 고정):
#   [8] 매트릭스 mat_{prior}_{pw}_{q}_{Ω}  90개  (3 × 3 × 5 × 2)
#
# winner_q*, winner_pct* (q/pct 민감도) 는 99_analyze.ipynb 에서 winner 식별 후
# in-notebook 으로 walk_forward 호출하여 동적 sweep (02b 임계값 민감도와 동일 패턴).
# → pkl 영구 보존 불필요, 결과는 DataFrame 으로 비교.
EXPERIMENTS = [


    # ═══════════════════════════════════════════════════════════════
    # [8] 매트릭스 (LSTM 고정, mat_{prior}_{pw}_{q}_{Ω}, 총 90 cells)
    # prior ∈ {mcap, eq, rp} × p_weight ∈ {mcap, eq, rp}
    # × q_mode ∈ {fix, lam, raw, inv, vsp} × omega ∈ {he, pap}
    # 정렬: prior → p_weight → q_mode → omega 순
    # ═══════════════════════════════════════════════════════════════
    # ── prior=mcap × pw=mcap ────────────────────────────────
    {**BASELINE, 'name': 'mat_mcap_mcap_fix_he'},
    {**BASELINE, 'name': 'mat_mcap_mcap_fix_pap', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'mat_mcap_mcap_lam_he', 'q_mode': 'lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_lam_pap', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_raw_he', 'q_mode': 'raw_lam', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_raw_pap', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_inv_he', 'q_mode': 'inv_lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_inv_pap', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_mcap_vsp_he', 'q_mode': 'vol_spread'},
    {**BASELINE, 'name': 'mat_mcap_mcap_vsp_pap', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper'},
    # ── prior=mcap × pw=eq ────────────────────────────────
    {**BASELINE, 'name': 'mat_mcap_eq_fix_he', 'p_weight': 'eq'},
    {**BASELINE, 'name': 'mat_mcap_eq_fix_pap', 'p_weight': 'eq', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'mat_mcap_eq_lam_he', 'p_weight': 'eq', 'q_mode': 'lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_lam_pap', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_raw_he', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_raw_pap', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_inv_he', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_inv_pap', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_eq_vsp_he', 'p_weight': 'eq', 'q_mode': 'vol_spread'},
    {**BASELINE, 'name': 'mat_mcap_eq_vsp_pap', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper'},
    # ── prior=mcap × pw=rp ────────────────────────────────
    {**BASELINE, 'name': 'mat_mcap_rp_fix_he', 'p_weight': 'rp'},
    {**BASELINE, 'name': 'mat_mcap_rp_fix_pap', 'p_weight': 'rp', 'omega_mode': 'ff3_paper'},
    {**BASELINE, 'name': 'mat_mcap_rp_lam_he', 'p_weight': 'rp', 'q_mode': 'lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_lam_pap', 'p_weight': 'rp', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_raw_he', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_raw_pap', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_inv_he', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_inv_pap', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_mcap_rp_vsp_he', 'p_weight': 'rp', 'q_mode': 'vol_spread'},
    {**BASELINE, 'name': 'mat_mcap_rp_vsp_pap', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper'},
    # ── prior=eq × pw=mcap ────────────────────────────────
    {**BASELINE, 'name': 'mat_eq_mcap_fix_he', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_mcap_fix_pap', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_mcap_lam_he', 'q_mode': 'lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_lam_pap', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_raw_he', 'q_mode': 'raw_lam', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_raw_pap', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_inv_he', 'q_mode': 'inv_lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_inv_pap', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_mcap_vsp_he', 'q_mode': 'vol_spread', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_mcap_vsp_pap', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    # ── prior=eq × pw=eq ────────────────────────────────
    {**BASELINE, 'name': 'mat_eq_eq_fix_he', 'p_weight': 'eq', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_eq_fix_pap', 'p_weight': 'eq', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_eq_lam_he', 'p_weight': 'eq', 'q_mode': 'lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_lam_pap', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_raw_he', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_raw_pap', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_inv_he', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_inv_pap', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_eq_vsp_he', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_eq_vsp_pap', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    # ── prior=eq × pw=rp ────────────────────────────────
    {**BASELINE, 'name': 'mat_eq_rp_fix_he', 'p_weight': 'rp', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_rp_fix_pap', 'p_weight': 'rp', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_rp_lam_he', 'p_weight': 'rp', 'q_mode': 'lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_lam_pap', 'p_weight': 'rp', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_raw_he', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_raw_pap', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_inv_he', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_inv_pap', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_eq_rp_vsp_he', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'prior': 'capm_eq'},
    {**BASELINE, 'name': 'mat_eq_rp_vsp_pap', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_eq'},
    # ── prior=rp × pw=mcap ────────────────────────────────
    {**BASELINE, 'name': 'mat_rp_mcap_fix_he', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_mcap_fix_pap', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_mcap_lam_he', 'q_mode': 'lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_lam_pap', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_raw_he', 'q_mode': 'raw_lam', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_raw_pap', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_inv_he', 'q_mode': 'inv_lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_inv_pap', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_mcap_vsp_he', 'q_mode': 'vol_spread', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_mcap_vsp_pap', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    # ── prior=rp × pw=eq ────────────────────────────────
    {**BASELINE, 'name': 'mat_rp_eq_fix_he', 'p_weight': 'eq', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_eq_fix_pap', 'p_weight': 'eq', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_eq_lam_he', 'p_weight': 'eq', 'q_mode': 'lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_lam_pap', 'p_weight': 'eq', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_raw_he', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_raw_pap', 'p_weight': 'eq', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_inv_he', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_inv_pap', 'p_weight': 'eq', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_eq_vsp_he', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_eq_vsp_pap', 'p_weight': 'eq', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    # ── prior=rp × pw=rp ────────────────────────────────
    {**BASELINE, 'name': 'mat_rp_rp_fix_he', 'p_weight': 'rp', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_rp_fix_pap', 'p_weight': 'rp', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_rp_lam_he', 'p_weight': 'rp', 'q_mode': 'lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_lam_pap', 'p_weight': 'rp', 'q_mode': 'lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_raw_he', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_raw_pap', 'p_weight': 'rp', 'q_mode': 'raw_lam', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_inv_he', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_inv_pap', 'p_weight': 'rp', 'q_mode': 'inv_lambda', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp', 'lam_mean': 2.5},
    {**BASELINE, 'name': 'mat_rp_rp_vsp_he', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'prior': 'capm_rp'},
    {**BASELINE, 'name': 'mat_rp_rp_vsp_pap', 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper', 'prior': 'capm_rp'},
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
