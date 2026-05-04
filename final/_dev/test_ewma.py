"""
EWMA omega 단위 검증 스크립트 (1회용).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bl_functions import compute_omega_ewma, compute_omega_he

# ── 공통 입력 ──────────────────────────────────────────────────────────────
P = pd.Series([0.5, -0.3, 0.2, 0.0], index=['A', 'B', 'C', 'D'])
Sigma = pd.DataFrame(np.eye(4) * 0.04, index=P.index, columns=P.index)
tau = 0.1

print('=' * 70)
print('1. 첫 시점 (prev_e_sq=None) → he_litterman 으로 fallback')
print('=' * 70)

o_he = compute_omega_he(P, Sigma, tau)
o_first = compute_omega_ewma(P, Sigma, tau, prev_e_sq=None, prev_omega=None, lambda_=0.94)
assert abs(o_first - o_he) < 1e-12, f'첫 시점 fallback 실패 ({o_first} vs {o_he})'
print(f'  he_litterman omega = {o_he:.8f}')
print(f'  EWMA 첫 시점 omega = {o_first:.8f}  ← he 와 동일 ✓')

print()
print('=' * 70)
print('2. EWMA 갱신 공식 검증 (λ=0.94)')
print('=' * 70)

# 가상 시나리오: 매월 e² = 0.001 (일정), 초기 Ω_0 = 0.001
prev_omega = 0.001
prev_e_sq  = 0.001
lambda_    = 0.94

print(f'  λ = {lambda_}, 초기 Ω_0 = e² = {prev_e_sq}')
print(f'  → 정상상태(e²이 항상 같으면) Ω_t = e² = {prev_e_sq} 유지')

for t in range(1, 6):
    omega_t = compute_omega_ewma(P, Sigma, tau, prev_e_sq, prev_omega, lambda_)
    expected = lambda_ * prev_omega + (1.0 - lambda_) * prev_e_sq
    assert abs(omega_t - expected) < 1e-12, f't={t}: 공식 불일치 ({omega_t} vs {expected})'
    print(f'  t={t}: Ω_t = {omega_t:.8f}')
    prev_omega = omega_t

print(f'  → 일정 입력 하에 Ω_t 가 e² 와 같음 ✓')

print()
print('=' * 70)
print('3. 충격 후 감쇠 패턴 (반감기 검증)')
print('=' * 70)

# Ω_0 = 1.0, e² = 0 인 경우 → Ω_t = 0.94^t
prev_omega = 1.0
prev_e_sq  = 0.0
lambda_    = 0.94

print(f'  λ = {lambda_}, Ω_0 = 1.0, e² = 0 (계속)')
print(f'  → Ω_t = λ^t = 0.94^t')
print()

for t in [1, 6, 11, 12, 22, 33, 37]:
    # t 회 갱신 시뮬레이션
    omega = 1.0
    for _ in range(t):
        omega = lambda_ * omega + (1.0 - lambda_) * 0.0
    expected = lambda_ ** t
    assert abs(omega - expected) < 1e-12
    print(f'  t={t:3d}월:  Ω_t = {omega:.6f}  (= λ^t)')
    if t == 11:
        print(f'           ↑ 반감기 11.2개월: 50% 도달 (이론: 0.5)')
    elif t == 33:
        print(f'           ↑ 3× 반감기: 12.5% 도달 (이론: 0.125)')
    elif t == 37:
        print(f'           ↑ 10% 안정화 시점 (이론: 0.10)')

print()
print('=' * 70)
print('4. λ 별 안정화 비교')
print('=' * 70)

print()
print(f'  {"λ":<8}{"반감기 H":<12}{"12M 후 영향":<14}{"36M 후 영향":<14}')
print('  ' + '-' * 50)
for lam in [0.825, 0.94, 0.97, 0.99]:
    H = -np.log(2) / np.log(lam)
    inf_12 = lam ** 12
    inf_36 = lam ** 36
    print(f'  {lam:<8.3f}{H:<12.2f}{inf_12*100:<14.2f}{inf_36*100:<14.2f}')

print()
print('  → λ=0.825: 12M 후 약 10% (안정화 빠름, 노이즈 큼)')
print('  → λ=0.94 : 12M 후 약 48% (반감기 도달)')
print('             36M 후 약 11% (10% 안정화 도달)')
print('  → λ=0.97 : 36M 후 약 33% (느린 안정화)')

print()
print('=' * 70)
print('5. lower bound (1e-8) 보장')
print('=' * 70)

# Ω_0 = 0, e² = 0 → 곱셈으로 0 이지만 max(omega, 1e-8)
o_zero = compute_omega_ewma(P, Sigma, tau, prev_e_sq=0.0, prev_omega=0.0, lambda_=0.94)
assert o_zero >= 1e-8
print(f'  Ω_0=0, e²=0 → omega = {o_zero:.2e}  (>= 1e-8) ✓')

print()
print('=' * 70)
print('단위 검증 모두 통과 ✓')
print('=' * 70)
