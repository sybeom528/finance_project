"""
신규 추가 기능 단위 검증 스크립트 (1회용, 검증 후 삭제 예정).

검증 대상:
  1. compute_omega_rmse 의 sqrt 보정된 호출 (시점 평균 RMSE)
  2. compute_omega_rmse_per_ticker (종목별 가중 RMSE)
  3. Risk Parity prior 가중 (1/vol 정규화) 수치 검증
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Windows 콘솔에서 한글/유니코드 출력을 위한 인코딩 설정
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# bl_functions 모듈을 import할 수 있도록 final/ 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bl_functions import (
    compute_omega_he,
    compute_omega_rmse,
    compute_omega_rmse_per_ticker,
)

# ── 공통 입력 ──────────────────────────────────────────────────────────────
P = pd.Series([0.5, -0.3, 0.2, 0.0], index=['A', 'B', 'C', 'D'])
Sigma = pd.DataFrame(np.eye(4) * 0.04, index=P.index, columns=P.index)
tau = 0.1

print('=' * 60)
print('1. Omega 옵션1 (시점별 평균 RMSE) 검증')
print('=' * 60)

o_he = compute_omega_he(P, Sigma, tau)
print(f'  he_litterman omega: {o_he:.6f}')

# pred_rmse == base_rmse → scale=1 → he와 동일
o_base = compute_omega_rmse(P, Sigma, tau, pred_rmse=0.39265, base_rmse=0.39265)
assert abs(o_he - o_base) < 1e-10, f'scale=1 검증 실패 ({o_he} vs {o_base})'
print(f'  pred=base 일 때 omega: {o_base:.6f}  ← he와 동일 ✓')

# pred_rmse = 2*base → scale=4 → 4배
o_2x = compute_omega_rmse(P, Sigma, tau, pred_rmse=0.78530, base_rmse=0.39265)
assert abs(o_2x / o_base - 4.0) < 1e-6, f'scale=4 검증 실패 ({o_2x/o_base})'
print(f'  pred=2*base 일 때 omega: {o_2x:.6f}  ← {o_2x/o_base:.2f}x ✓')

# pred_rmse = 0.5*base → scale=0.25 → 1/4
o_half = compute_omega_rmse(P, Sigma, tau, pred_rmse=0.196325, base_rmse=0.39265)
assert abs(o_half / o_base - 0.25) < 1e-6, f'scale=0.25 검증 실패 ({o_half/o_base})'
print(f'  pred=0.5*base 일 때 omega: {o_half:.6f}  ← {o_half/o_base:.2f}x ✓')

print()
print('=' * 60)
print('2. Omega 옵션2 (종목별 가중 RMSE) 검증')
print('=' * 60)

# 정규화 정의: pred_rmse_view = sqrt( Σ P_i² × RMSE_i² / Σ P_i² )
#              scale          = (pred_rmse_view / base_rmse)²
#
# 균등 RMSE = base인 경우:
#   pred_rmse_view = base × sqrt(Σ P_i² × 1 / Σ P_i²) = base
#   scale          = 1                       ← P 크기 무관, he 그대로
#
# 종목별 RMSE 차이만 omega scale에 영향. ablation 비교가 깔끔해짐.

p_sq_sum = float((P ** 2).sum())   # = 0.5² + 0.3² + 0.2² + 0² = 0.38
print(f'  Σ P_i² = {p_sq_sum:.4f}  (정규화 분모로 작용)')

# 모든 종목 RMSE = base → scale = 1 → he와 정확히 동일
rmse_uniform = pd.Series([0.39265] * 4, index=P.index)
o_pt_uniform = compute_omega_rmse_per_ticker(P, Sigma, tau, rmse_uniform, base_rmse=0.39265)
assert abs(o_pt_uniform - o_he) < 1e-10, (
    f'균등 RMSE → he 검증 실패 ({o_pt_uniform} vs {o_he})'
)
print(f'  모든 종목 rmse=base 일 때 omega: {o_pt_uniform:.6f}  ← he와 정확히 동일 ✓')

# P=0인 종목(D)는 결과에 영향 없어야 함
rmse_d_huge = pd.Series([0.39265, 0.39265, 0.39265, 999.0], index=P.index)
o_pt_d = compute_omega_rmse_per_ticker(P, Sigma, tau, rmse_d_huge, base_rmse=0.39265)
assert abs(o_pt_d - o_pt_uniform) < 1e-10, (
    f'P=0 종목 RMSE 영향 검증 실패 ({o_pt_d} vs {o_pt_uniform})'
)
print(f'  D(P=0)의 rmse=999 일 때 omega: {o_pt_d:.6f}  ← P=0 종목 무영향 ✓')

# 모든 종목 RMSE = 2*base → 어떤 P든 scale = 4
rmse_all_2x = pd.Series([0.7853] * 4, index=P.index)
o_pt_all_2x = compute_omega_rmse_per_ticker(P, Sigma, tau, rmse_all_2x, base_rmse=0.39265)
assert abs(o_pt_all_2x / o_he - 4.0) < 1e-6, (
    f'균등 2*base RMSE → scale=4 검증 실패 ({o_pt_all_2x/o_he})'
)
print(f'  모든 종목 rmse=2*base 일 때 omega: {o_pt_all_2x:.6f}  ← {o_pt_all_2x/o_he:.2f}x ✓')

# 종목 A의 RMSE만 2배 → 균등 대비 증가, 균등 2배보단 작음
rmse_a_2x = pd.Series([0.7853, 0.39265, 0.39265, 0.39265], index=P.index)
o_pt_a = compute_omega_rmse_per_ticker(P, Sigma, tau, rmse_a_2x, base_rmse=0.39265)
assert o_pt_uniform < o_pt_a < o_pt_all_2x, (
    f'A만 2배 RMSE 검증 실패 ({o_pt_uniform} < {o_pt_a} < {o_pt_all_2x})'
)
print(f'  A(P=0.5)의 rmse=2*base 일 때 omega: {o_pt_a:.6f}  ← 균등 < A만 2x < 모두 2x ✓')

# 정규화 효과 확인: 다른 P 가중 방식도 균등 RMSE면 모두 he 와 동일
P_mcap_like = pd.Series([0.05, 0.04, 0.03, 0.02], index=P.index)   # 작은 mcap-like P
P_eq_like   = pd.Series([0.10, 0.10, 0.10, 0.10], index=P.index)   # 동일가중-like P

o_he_mcap   = compute_omega_he(P_mcap_like, Sigma, tau)
o_pt_mcap   = compute_omega_rmse_per_ticker(P_mcap_like, Sigma, tau, rmse_uniform, base_rmse=0.39265)
o_he_eq     = compute_omega_he(P_eq_like,   Sigma, tau)
o_pt_eq     = compute_omega_rmse_per_ticker(P_eq_like,   Sigma, tau, rmse_uniform, base_rmse=0.39265)

assert abs(o_pt_mcap - o_he_mcap) < 1e-10, 'mcap-like P 균등 RMSE 검증 실패'
assert abs(o_pt_eq   - o_he_eq)   < 1e-10, 'eq-like P 균등 RMSE 검증 실패'
print(f'  mcap-like P 균등 RMSE: {o_pt_mcap/o_he_mcap:.4f}x he ✓')
print(f'  eq-like P 균등 RMSE  : {o_pt_eq  /o_he_eq  :.4f}x he ✓')
print(f'  → P 가중 방식과 무관하게 균등 RMSE면 omega = he ✓')

print()
print('=' * 60)
print('3. Risk Parity prior 가중 검증 (수동 계산)')
print('=' * 60)

# get_prior_weights_rp 는 노트북 안에서 정의된 함수이므로 동일 로직을 여기서 재현
def rp_weights(vol_dict):
    vol = pd.Series(vol_dict)
    inv_vol = 1.0 / vol
    return inv_vol / inv_vol.sum()

# 변동성: A가 가장 낮음, C가 가장 높음
w = rp_weights({'A': 0.10, 'B': 0.20, 'C': 0.40})
print(f'  vol={{A:0.10, B:0.20, C:0.40}} → w={dict(w.round(4))}')
assert w['A'] > w['B'] > w['C'], 'A(저변동)이 가장 큰 비중이어야 함'
assert abs(w.sum() - 1.0) < 1e-10, '합이 1이 되어야 함'
# 정확한 수치: 1/0.10 : 1/0.20 : 1/0.40 = 10 : 5 : 2.5 → 합 17.5
# A=10/17.5 ≈ 0.5714, B=5/17.5 ≈ 0.2857, C=2.5/17.5 ≈ 0.1429
assert abs(w['A'] - 10/17.5) < 1e-10
assert abs(w['B'] - 5/17.5) < 1e-10
assert abs(w['C'] - 2.5/17.5) < 1e-10
print(f'  → 1/vol 정규화 정확히 일치 ✓')

# 균등 변동성 → 균등 가중
w_eq = rp_weights({'A': 0.20, 'B': 0.20, 'C': 0.20})
assert all(abs(v - 1/3) < 1e-10 for v in w_eq), '균등 변동성 → 1/N 가중'
print(f'  vol=균등 → w={dict(w_eq.round(4))} ← 1/N과 동일 ✓')

print()
print('=' * 60)
print('단위 검증 모두 통과')
print('=' * 60)
