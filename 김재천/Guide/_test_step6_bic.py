# ============================================================
# Step 6 BIC 탐색 확장 테스트 스크립트 (일회용)
#
# 목적:
#   - 이슈 3 (BIC 공식 수정): k에 (n-1) 초기 상태 분포 자유도 추가
#   - 이슈 4 (n 범위 확장): [2,3,4] → [2,3,4,5,6,7]
#   - 최적 n이 무엇인지 확인하여 Z 옵션 필요 여부 판단
# ============================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.getcwd(), 'data')

# 데이터 로드
df_reg = pd.read_csv(os.path.join(DATA_DIR, 'df_reg_v2.csv'), index_col='Date', parse_dates=True)
HMM_FEATURES = ['VIX_level', 'VIX_contango', 'HY_spread', 'yield_curve', 'Cu_Au_ratio_chg']
X_raw = df_reg[HMM_FEATURES].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

T = len(X_scaled)
d = X_scaled.shape[1]
print(f'HMM 입력: {T}일 x {d}변수')
print(f'피처: {HMM_FEATURES}')
print()
print('=' * 85)
print(f'{"n":>3} | {"LL":>12} | {"k_old":>6} | {"k_new":>6} | {"BIC_old":>12} | {"BIC_new":>12} | {"수렴":>5}')
print('=' * 85)

results = []
for n in [2, 3, 4, 5, 6, 7]:
    best_score = -np.inf
    best_model = None
    converged_count = 0
    for seed in range(10):
        try:
            model = GaussianHMM(
                n_components=n, covariance_type='full',
                n_iter=500, random_state=seed, tol=1e-4
            )
            model.fit(X_scaled)
            score = model.score(X_scaled)
            if model.monitor_.converged:
                converged_count += 1
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as e:
            pass

    # BIC 계산 (이전 공식 vs 수정 공식)
    # 이전: k = A(전이) + μ(평균) + Σ(공분산)
    k_old = n * (n - 1) + n * d + n * d * (d + 1) // 2
    # 수정: + π(초기 상태 분포) 자유도 (n-1)
    k_new = (n - 1) + k_old

    bic_old = -2 * best_score + k_old * np.log(T)
    bic_new = -2 * best_score + k_new * np.log(T)

    print(f'{n:>3} | {best_score:>12,.1f} | {k_old:>6} | {k_new:>6} | {bic_old:>12,.1f} | {bic_new:>12,.1f} | {converged_count:>3}/10')

    results.append({
        'n': n, 'LL': best_score,
        'k_old': k_old, 'k_new': k_new,
        'BIC_old': bic_old, 'BIC_new': bic_new,
        'converged': converged_count,
    })

df_bic = pd.DataFrame(results)
print('=' * 85)
print()

# 최적 n 결정
best_n_old = int(df_bic.loc[df_bic['BIC_old'].idxmin(), 'n'])
best_n_new = int(df_bic.loc[df_bic['BIC_new'].idxmin(), 'n'])

print(f'▶ 이전 BIC 공식 기준 최적 n: {best_n_old} (BIC={df_bic["BIC_old"].min():,.1f})')
print(f'▶ 수정 BIC 공식 기준 최적 n: {best_n_new} (BIC={df_bic["BIC_new"].min():,.1f})')
print()

# BIC 감소폭 분석
print('BIC 감소폭 (n-1 → n, 수정 공식):')
for i in range(1, len(df_bic)):
    delta = df_bic.iloc[i]['BIC_new'] - df_bic.iloc[i-1]['BIC_new']
    arrow = '↓' if delta < 0 else '↑'
    print(f'  n={int(df_bic.iloc[i-1]["n"])} → n={int(df_bic.iloc[i]["n"])}: {arrow}{abs(delta):>8,.1f}')
print()

# 결과 요약
print('=' * 85)
print('결론:')
print('=' * 85)
if best_n_new == 4:
    print(f'  최적 n = 4 (현재와 동일) → Y로 완료, Z 불필요')
    print(f'  → Step 7~11 재실행 불필요')
else:
    print(f'  최적 n = {best_n_new} (현재 n=4에서 변경) → Z 옵션 판단 필요')
    print(f'  → Step 7~11 하류 재실행 영향 있음')

df_bic.to_csv(os.path.join(DATA_DIR, '_step6_bic_test.csv'), index=False)
print()
print(f'[저장] {DATA_DIR}/_step6_bic_test.csv')
