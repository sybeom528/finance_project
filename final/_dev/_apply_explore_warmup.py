"""
99_explore.ipynb 에 워밍업 자동 trim 기능 추가 (1회용 패치).

변경:
  1. cell-01-loaders 에 warmup_for_lambda / get_warmup_months / fair_period 헬퍼 추가
  2. cell-04-plot-compare 의 plot_compare 에 auto_warmup 옵션 추가
  3. 신규 cell-12-example-9 추가 — auto_warmup=True 사용 예시
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / '99_explore.ipynb'
nb = json.loads(NB.read_text(encoding='utf-8'))


def patch_cell_source(cell, old: str, new: str, must_replace: int = 1):
    src = ''.join(cell['source'])
    cnt = src.count(old)
    assert cnt == must_replace, (
        f"cell {cell.get('id')}: '{old[:60]}...' must_replace={must_replace}, found={cnt}"
    )
    src = src.replace(old, new)
    cell['source'] = src.splitlines(keepends=True)


# ── cell-01-loaders 에 워밍업 헬퍼 함수 추가 ────────────────────────
for cell in nb['cells']:
    if cell.get('id') == 'cell-01-loaders':
        old_tail = (
            "print(f'로드 가능한 실험: {len(list_results())}개')\n"
            "print(f'예: {list_results()[:5]}')"
        )
        new_tail = (
            "# ─── 워밍업 자동 계산 헬퍼 ─────────────────────────────────────\n"
            "def warmup_for_lambda(lambda_, threshold=0.10):\n"
            "    \"\"\"EWMA 의 안정화 기간 (개월). λ^N < threshold 인 최소 N.\n"
            "    threshold=0.10 → 10% 안정화 (3.32× 반감기)\n"
            "    \"\"\"\n"
            "    return int(np.ceil(np.log(threshold) / np.log(lambda_)))\n"
            "\n"
            "\n"
            "def get_warmup_months(cfg, threshold=0.10):\n"
            "    \"\"\"실험 config 에서 권장 워밍업 길이 결정.\n"
            "    omega_mode='ewma' → λ 에 따라 자동, 그 외 → 0\n"
            "    \"\"\"\n"
            "    om = cfg.get('omega_mode', 'he_litterman')\n"
            "    if om == 'ewma':\n"
            "        lam = float(cfg.get('lambda', 0.94))\n"
            "        return warmup_for_lambda(lam, threshold)\n"
            "    return 0\n"
            "\n"
            "\n"
            "def fair_period(names, end='2024-12', start_default='2010-01', threshold=0.10):\n"
            "    \"\"\"여러 실험의 공정 비교 시작 시점 자동 결정.\n"
            "    가장 긴 워밍업을 가진 실험 기준으로 모든 실험 trim.\n"
            "    \"\"\"\n"
            "    max_warmup = 0\n"
            "    for name in names:\n"
            "        try:\n"
            "            cfg = load_result(name)['config']\n"
            "            max_warmup = max(max_warmup, get_warmup_months(cfg, threshold))\n"
            "        except FileNotFoundError:\n"
            "            continue\n"
            "    if max_warmup == 0:\n"
            "        return (start_default, end)\n"
            "    start_dt = pd.Timestamp(start_default) + pd.DateOffset(months=max_warmup)\n"
            "    return (start_dt.strftime('%Y-%m'), end)\n"
            "\n"
            "\n"
            "print(f'로드 가능한 실험: {len(list_results())}개')\n"
            "print(f'예: {list_results()[:5]}')\n"
            "print()\n"
            "print(f'EWMA 워밍업 자동 계산 (10% 안정화 기준):')\n"
            "print(f'  λ=0.825 → {warmup_for_lambda(0.825)}개월')\n"
            "print(f'  λ=0.94  → {warmup_for_lambda(0.94)}개월')\n"
            "print(f'  λ=0.97  → {warmup_for_lambda(0.97)}개월')"
        )
        patch_cell_source(cell, old_tail, new_tail)
        break

# ── cell-04-plot-compare 에 auto_warmup 옵션 추가 ────────────────────
for cell in nb['cells']:
    if cell.get('id') == 'cell-04-plot-compare':
        old_sig = (
            "def plot_compare(names, period=('2010-01', '2024-12'),\n"
            "                 include_spy=True, figsize=(15, 10)):\n"
            "    \"\"\"여러 실험 비교 패널 — 4 panels.\n"
            "\n"
            "    1. 누적수익률 비교\n"
            "    2. Drawdown 비교\n"
            "    3. Sharpe·CAGR·MDD 막대\n"
            "    4. 12M rolling Sharpe 비교\n"
            "    \"\"\""
        )
        new_sig = (
            "def plot_compare(names, period=None, auto_warmup=False,\n"
            "                 include_spy=True, figsize=(15, 10)):\n"
            "    \"\"\"여러 실험 비교 패널 — 4 panels.\n"
            "\n"
            "    1. 누적수익률 비교\n"
            "    2. Drawdown 비교\n"
            "    3. Sharpe·CAGR·MDD 막대\n"
            "    4. 12M rolling Sharpe 비교\n"
            "\n"
            "    Parameters\n"
            "    ----------\n"
            "    period       : (start, end) 직접 지정. None 이면 default 또는 auto.\n"
            "    auto_warmup  : True 시 fair_period(names) 로 자동 trim.\n"
            "                   가장 긴 EWMA 워밍업 기준으로 모든 실험 정합 비교.\n"
            "    \"\"\"\n"
            "    if auto_warmup:\n"
            "        period = fair_period(names)\n"
            "        print(f'⚙ auto_warmup=True → period 자동 결정: {period} (가장 긴 워밍업 기준)')\n"
            "    elif period is None:\n"
            "        period = ('2010-01', '2024-12')"
        )
        patch_cell_source(cell, old_sig, new_sig)
        break

# ── 신규 cell — auto_warmup 사용 예시를 example-6 직후에 추가 ───────
new_cell_source = (
    "# ────────────────────────────────────────────────────────────────────\n"
    "# 예시 9 — auto_warmup=True 공정 비교 (워밍업 자동 trim) ★ 권장\n"
    "# ────────────────────────────────────────────────────────────────────\n"
    "# 같은 P 가중·다른 λ 비교 시 워밍업 차이로 인한 착시를 자동 제거\n"
    "# (lo=12M, std=36M 워밍업이 다르므로 더 긴 std 의 36M trim 으로 통일)\n"
    "names = [\n"
    "    'baseline',\n"
    "    'p_lstm_vol_mcap_ewma_lo',\n"
    "    'p_lstm_vol_mcap_ewma_std',\n"
    "    'p_lstm_mcap_ewma_lo',\n"
    "    'p_lstm_mcap_ewma_std',\n"
    "]\n"
    "\n"
    "plot_compare(names, auto_warmup=True)   # ← period 인자 생략 시 자동 결정"
)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell-15-example-9",
    "metadata": {},
    "outputs": [],
    "source": new_cell_source.splitlines(keepends=True),
}

# example-6 (cell-12) 직후에 추가 — 또는 끝에 추가
# 끝(example-8 = cell-14) 직후에 추가
inserted = False
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'cell-14-example-8':
        nb['cells'].insert(i + 1, new_cell)
        inserted = True
        break

assert inserted, 'cell-14-example-8 못 찾음'

# ── 출력 ────────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'OK: {NB.name} 워밍업 자동 trim 패치 완료')
