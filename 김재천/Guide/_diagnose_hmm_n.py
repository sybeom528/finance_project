# ============================================================
# Step 6 HMM n 선택 종합 진단 스크립트
#
# 목적: BIC를 넘어서 n 선택을 다기준으로 검증
#   1. 정보 기준: BIC, AIC, ICL (HMM 전용)
#   2. 수렴 안정성: 10 seed 중 수렴 개수
#   3. 레짐 지속성: 평균/최소/최대 duration
#   4. 레짐 크기 분포: 최소 레짐 관측 일수 (rare state 위험)
#   5. 전이 빈도: 연간 전환 횟수 (리밸런싱 비용 지표)
#   6. 해석 가능성: Viterbi vs Forward 차이 (uncertainty)
#   7. 파라미터 대비 샘플: T/k 비율
#   8. 레짐 특성 서명: 각 레짐의 VIX/HY 평균 (해석 가능성 평가)
# ============================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.getcwd(), 'data')

# ── 데이터 로드 ──
df_reg = pd.read_csv(os.path.join(DATA_DIR, 'df_reg_v2.csv'), index_col='Date', parse_dates=True)
HMM_FEATURES = ['VIX_level', 'VIX_contango', 'HY_spread', 'yield_curve', 'Cu_Au_ratio_chg']
X_raw = df_reg[HMM_FEATURES].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

T = len(X_scaled)
d = X_scaled.shape[1]
print(f'HMM 입력: {T}일 × {d}변수')
print(f'피처: {HMM_FEATURES}')
print()

# ── 각 n별 종합 진단 ──
N_RANGE = [2, 3, 4, 5, 6, 7, 8]
N_SEEDS = 10

results = []
models = {}  # 각 n별 최적 모델 저장 (레짐 서명 분석용)

for n in N_RANGE:
    best_score = -np.inf
    best_model = None
    converged_count = 0

    for seed in range(N_SEEDS):
        try:
            model = GaussianHMM(
                n_components=n, covariance_type='full',
                n_iter=500, random_state=seed, tol=1e-4
            )
            model.fit(X_scaled)
            if model.monitor_.converged:
                converged_count += 1
            score = model.score(X_scaled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            pass

    models[n] = best_model

    # ── 1. 정보 기준 ──
    # 파라미터 수 (수정 공식: π 포함)
    k = (n - 1) + n * (n - 1) + n * d + n * d * (d + 1) // 2
    bic = -2 * best_score + k * np.log(T)
    aic = -2 * best_score + 2 * k

    # ── 2. Forward 알고리즘으로 posterior 계산 (ICL용) ──
    log_B = best_model._compute_log_likelihood(X_scaled)
    log_A = np.log(best_model.transmat_ + 1e-300)
    log_alpha = np.zeros((T, n))
    log_alpha[0] = np.log(best_model.startprob_ + 1e-300) + log_B[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(
            log_alpha[t-1][:, None] + log_A, axis=0
        ) + log_B[t]
    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    posterior = np.exp(log_alpha - log_norm)

    # ── 3. ICL (Integrated Classification Likelihood) ──
    # ICL = BIC + 2 × Σ entropy(posterior_t)
    # HMM에서 BIC보다 선호되는 지표
    eps = 1e-12
    entropy_sum = -np.sum(posterior * np.log(posterior + eps))
    icl = bic + 2 * entropy_sum

    # ── 4. Forward vs Viterbi ──
    viterbi_states = best_model.predict(X_scaled)
    forward_states = posterior.argmax(axis=1)
    vf_diff_pct = (viterbi_states != forward_states).mean() * 100

    # ── 5. 레짐 지속성 (1/(1-a_ii)) ──
    durations = []
    for i in range(n):
        a_ii = best_model.transmat_[i, i]
        dur = 1 / (1 - a_ii + 1e-12)
        durations.append(dur)

    # ── 6. 레짐 크기 분포 ──
    regime_counts = np.bincount(forward_states, minlength=n)

    # ── 7. 전이 빈도 ──
    transitions = int((np.diff(forward_states) != 0).sum())
    trans_per_year = transitions / (T / 252)

    # ── 8. 집계 ──
    results.append({
        'n': n,
        'LL': best_score,
        'k': k,
        'BIC': bic,
        'AIC': aic,
        'ICL': icl,
        'conv': f'{converged_count}/10',
        'min_regime_days': int(regime_counts.min()),
        'max_regime_days': int(regime_counts.max()),
        'min_dur': float(min(durations)),
        'mean_dur': float(np.mean(durations)),
        'max_dur': float(max(durations)),
        'trans/yr': trans_per_year,
        'V_F_diff_%': vf_diff_pct,
        'T/k': T / k,
    })

df_diag = pd.DataFrame(results)

# ── 출력 ──
print('=' * 100)
print('【1】 정보 기준 비교 (BIC, AIC, ICL)')
print('=' * 100)
print(df_diag[['n', 'LL', 'k', 'BIC', 'AIC', 'ICL', 'conv']].to_string(index=False, formatters={
    'LL': '{:,.1f}'.format,
    'BIC': '{:,.1f}'.format,
    'AIC': '{:,.1f}'.format,
    'ICL': '{:,.1f}'.format,
}))
print()

print('=' * 100)
print('【2】 각 기준별 최적 n')
print('=' * 100)
print(f"  BIC 최소:  n={int(df_diag.loc[df_diag['BIC'].idxmin(), 'n'])} (BIC={df_diag['BIC'].min():,.1f})")
print(f"  AIC 최소:  n={int(df_diag.loc[df_diag['AIC'].idxmin(), 'n'])} (AIC={df_diag['AIC'].min():,.1f})")
print(f"  ICL 최소:  n={int(df_diag.loc[df_diag['ICL'].idxmin(), 'n'])} (ICL={df_diag['ICL'].min():,.1f})  ← HMM 전문 기준")
print()

print('=' * 100)
print('【3】 n 증가에 따른 BIC/ICL 감소폭 (엘보우 탐색)')
print('=' * 100)
print(f'{"n-1→n":>8} | {"ΔBIC":>12} | {"ΔICL":>12}')
print('-' * 40)
for i in range(1, len(df_diag)):
    dbic = df_diag.iloc[i]['BIC'] - df_diag.iloc[i-1]['BIC']
    dicl = df_diag.iloc[i]['ICL'] - df_diag.iloc[i-1]['ICL']
    prev_n = int(df_diag.iloc[i-1]['n'])
    curr_n = int(df_diag.iloc[i]['n'])
    print(f'  {prev_n}→{curr_n:<4} | {dbic:>+12,.1f} | {dicl:>+12,.1f}')
print()

print('=' * 100)
print('【4】 레짐 실용성 지표')
print('=' * 100)
pretty = df_diag[['n', 'min_regime_days', 'max_regime_days', 'min_dur', 'mean_dur', 'max_dur', 'trans/yr', 'V_F_diff_%', 'T/k']].copy()
print(pretty.to_string(index=False, formatters={
    'min_dur': '{:.1f}'.format,
    'mean_dur': '{:.1f}'.format,
    'max_dur': '{:.1f}'.format,
    'trans/yr': '{:.1f}'.format,
    'V_F_diff_%': '{:.2f}'.format,
    'T/k': '{:.1f}'.format,
}))
print()
print('  min_regime_days: 최소 레짐 관측 일수 (rare state 위험, 50 미만 시 공분산 추정 불안)')
print('  min_dur:         최소 레짐 평균 지속 기간 (5일 미만 시 noise 가능)')
print('  trans/yr:        연간 레짐 전환 횟수 (높을수록 리밸런싱 비용 ↑)')
print('  V_F_diff_%:      Viterbi vs Forward 불일치 (높을수록 posterior 불확실)')
print('  T/k:             파라미터 대비 관측치 (10 미만 시 과적합 위험)')
print()

print('=' * 100)
print('【5】 레짐 특성 서명 (해석 가능성 평가)')
print('=' * 100)
print('각 n에서 레짐을 VIX_level 평균 오름차순으로 나열:')
print()
for n in N_RANGE:
    model = models[n]
    log_B = model._compute_log_likelihood(X_scaled)
    log_A = np.log(model.transmat_ + 1e-300)
    log_alpha = np.zeros((T, n))
    log_alpha[0] = np.log(model.startprob_ + 1e-300) + log_B[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t-1][:, None] + log_A, axis=0) + log_B[t]
    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    posterior = np.exp(log_alpha - log_norm)
    states = posterior.argmax(axis=1)

    # VIX 평균 오름차순으로 정렬
    state_sig = []
    for s in range(n):
        mask = states == s
        if mask.sum() > 0:
            sig = {
                'raw_id': s,
                'days': int(mask.sum()),
                'VIX': float(X_raw.iloc[mask]['VIX_level'].mean()),
                'HY': float(X_raw.iloc[mask]['HY_spread'].mean()),
                'YC': float(X_raw.iloc[mask]['yield_curve'].mean()),
            }
            state_sig.append(sig)

    state_sig.sort(key=lambda x: x['VIX'])

    print(f'  ── n={n} ──')
    print(f'    {"순번":>4} | {"일수":>5} | {"VIX":>6} | {"HY":>6} | {"YC":>7}')
    for rank, s in enumerate(state_sig):
        print(f'    {rank:>4} | {s["days"]:>5} | {s["VIX"]:>6.2f} | {s["HY"]:>6.2f} | {s["YC"]:>+7.2f}')
    print()

# ── 저장 ──
df_diag.to_csv(os.path.join(DATA_DIR, '_hmm_n_diagnosis.csv'), index=False)
print(f'[저장] {DATA_DIR}/_hmm_n_diagnosis.csv')
print()

# ── 최종 scorecard ──
print('=' * 100)
print('【6】 다기준 scorecard (10점 만점, 각 기준 가중치 동일)')
print('=' * 100)

def score_n(row):
    """각 n의 실용성 점수 (높을수록 좋음)"""
    scores = {}

    # (a) BIC 상대 점수 (최소값 대비, 10점 만점)
    max_bic = df_diag['BIC'].max()
    min_bic = df_diag['BIC'].min()
    scores['BIC'] = 10 * (max_bic - row['BIC']) / (max_bic - min_bic + 1e-9)

    # (b) ICL 상대 점수
    max_icl = df_diag['ICL'].max()
    min_icl = df_diag['ICL'].min()
    scores['ICL'] = 10 * (max_icl - row['ICL']) / (max_icl - min_icl + 1e-9)

    # (c) 해석 가능성 (금융 실무 관행)
    interp = {2: 7, 3: 9, 4: 10, 5: 6, 6: 4, 7: 2, 8: 1}
    scores['해석성'] = interp.get(int(row['n']), 5)

    # (d) 수렴 안정성 (10/10이 만점)
    scores['수렴'] = int(row['conv'].split('/')[0])

    # (e) rare state 위험 (최소 레짐 일수가 많을수록 좋음)
    scores['rare'] = min(10, row['min_regime_days'] / 30)  # 300일 이상이면 10점

    # (f) 전이 빈도 (너무 많으면 감점, 5~20회/년이 이상적)
    t = row['trans/yr']
    if 5 <= t <= 20:
        scores['전이적정성'] = 10
    elif t < 5:
        scores['전이적정성'] = 5
    else:
        scores['전이적정성'] = max(0, 10 - (t - 20) / 5)

    # (g) T/k 비율 (10 이상이면 만점)
    scores['T/k'] = min(10, row['T/k'] / 3)

    # (h) 지속성 (평균 duration이 10~100일이 이상적)
    md = row['mean_dur']
    if 10 <= md <= 100:
        scores['지속성'] = 10
    elif md < 10:
        scores['지속성'] = max(0, md)
    else:
        scores['지속성'] = max(0, 10 - (md - 100) / 50)

    return scores

scores_all = {}
for _, row in df_diag.iterrows():
    n = int(row['n'])
    scores_all[n] = score_n(row)

# 출력
criteria = ['BIC', 'ICL', '해석성', '수렴', 'rare', '전이적정성', 'T/k', '지속성']
print(f'{"n":>3} | ' + ' | '.join(f'{c:>8}' for c in criteria) + ' | {:>6}'.format('합계'))
print('-' * 100)
for n in N_RANGE:
    s = scores_all[n]
    line = f'{n:>3} | '
    line += ' | '.join(f'{s[c]:>8.1f}' for c in criteria)
    total = sum(s.values())
    line += f' | {total:>6.1f}'
    print(line)
print()
best_n = max(scores_all, key=lambda n: sum(scores_all[n].values()))
print(f'▶ 종합 점수 최고: n={best_n} ({sum(scores_all[best_n].values()):.1f}점)')
