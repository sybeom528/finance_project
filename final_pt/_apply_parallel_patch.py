"""99_run.ipynb cell-7 을 joblib threading 병렬화로 교체.

변경 전: for 루프 직렬 실행
변경 후: joblib.Parallel(backend='threading') 으로 4 worker 동시 실행

threading 안전성:
- walk_forward 가 monthly_cache, lstm_pred_monthly 를 read-only 로만 사용
- 각 cfg 는 독립
- pkl 저장은 cfg 별 다른 파일 → race condition 없음
- SLSQP/numpy 가 GIL release 하므로 CPU 병렬화 효과적
"""
from __future__ import annotations
import json
from pathlib import Path
import nbformat as nbf

NB = Path('final/99_run.ipynb')

# 새 cell-7 source
NEW_CELL_7_SOURCE = '''# ══════════════════════════════════════════════════════════════
# 전체 실험 실행 + 저장 — joblib threading 병렬 (4 worker)
# ══════════════════════════════════════════════════════════════
# threading 안정성: walk_forward 가 monthly_cache 를 read-only 로만 사용,
# 각 cfg 는 독립, pkl 저장은 다른 파일 → race condition 없음.
# scipy.optimize (SLSQP) 와 numpy BLAS 가 GIL release 하므로 4 worker 효과적.

from joblib import Parallel, delayed

SKIP_IF_EXISTS = True   # True → 이미 저장된 실험은 재실행 않고 스킵
N_JOBS         = 4      # 병렬 worker 수 (CPU core 수 / 메모리 고려하여 조정)

# 실행 대상 결정 (skip 사유 사전 필터링)
def _should_skip(cfg):
    name = cfg['name']
    out_path = RESULTS_DIR / f'{name}.pkl'
    if cfg.get('p_mode') == 'lstm_predicted' and not LSTM_AVAILABLE:
        return f'[SKIP] {name} — LSTM 예측 파일 없음'
    if cfg.get('omega_mode') == 'rmse' and not LSTM_AVAILABLE:
        return f'[SKIP] {name} — LSTM 예측 파일 없음'
    if SKIP_IF_EXISTS and out_path.exists():
        return f'[SKIP] {name} — 이미 저장됨'
    return None  # 실행 대상

run_list, skipped = [], []
for cfg in EXPERIMENTS:
    skip_reason = _should_skip(cfg)
    if skip_reason is None:
        run_list.append(cfg)
    else:
        skipped.append(cfg['name'])
        print(skip_reason)

print(f'\\n실행 예정: {len(run_list)}개 / 전체 {len(EXPERIMENTS)}개  (병렬 worker: {N_JOBS})')
print(f'스킵: {len(skipped)}개\\n')


def _run_one(cfg):
    """worker 함수 — walk_forward 후 pkl 저장."""
    name = cfg['name']
    out_path = RESULTS_DIR / f'{name}.pkl'
    t0 = time.time()
    try:
        result = walk_forward(cfg)
        with open(out_path, 'wb') as f:
            pickle.dump(result, f)
        elapsed = time.time() - t0
        return ('OK', name, elapsed)
    except Exception as e:
        return ('ERR', name, str(e))


_all_t0 = time.time()

# joblib threading 병렬 실행
# verbose=10: 매 시점마다 진행 표시
results = Parallel(n_jobs=N_JOBS, backend='threading', verbose=10)(
    delayed(_run_one)(cfg) for cfg in run_list
)

# 결과 집계
completed, errors = [], []
for status, name, info in results:
    if status == 'OK':
        completed.append((name, info))
        print(f'  ✓ {name}  ({info/60:.1f}분)')
    else:
        errors.append((name, info))
        print(f'  ✗ {name}  ERROR: {info[:100]}')

total_elapsed = (time.time() - _all_t0) / 60
print(f'\\n{"="*60}')
print(f'완료: {len(completed)}개 / 에러: {len(errors)}개 / 스킵: {len(skipped)}개')
print(f'총 소요: {total_elapsed:.1f}분 (병렬 {N_JOBS} worker)')
print(f'저장 위치: {RESULTS_DIR}')
if errors:
    print(f'\\n⚠ 에러 발생 실험:')
    for name, msg in errors:
        print(f'  {name}: {msg[:100]}')
'''

def main():
    with open(NB, 'rb') as f:
        nb = nbf.read(f, as_version=4)

    # cell-7 의 source 교체 (출력은 비움 — 재실행 후 새로 채워짐)
    cell7 = nb['cells'][7]
    assert cell7['cell_type'] == 'code', f"cell-7 이 code 셀이 아님: {cell7['cell_type']}"

    print(f"패치 전 cell-7 source 길이: {len(''.join(cell7['source']))}")
    cell7['source'] = NEW_CELL_7_SOURCE.splitlines(keepends=True)
    cell7['outputs'] = []  # 출력 비움 — 재실행 시 채워짐
    cell7['execution_count'] = None
    print(f"패치 후 cell-7 source 길이: {len(''.join(cell7['source']))}")

    with open(NB, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"\n저장: {NB}")
    print(f"  변경 셀: cell-7 (실험 실행 부분, joblib threading 병렬화)")


if __name__ == "__main__":
    main()
