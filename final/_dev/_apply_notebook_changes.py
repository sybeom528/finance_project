"""
99_run.ipynb 변경 스크립트 (1회용, 검증 후 삭제 예정).

§3-2 추가:
  - cell-03에 get_prior_weights_rp 함수 추가
  - cell-03의 get_omega 함수에 'rmse_per_ticker' elif 분기 추가
§4-1: cell-00 import 라인에 compute_omega_rmse_per_ticker 추가
§4-2: cell-03 'rmse' 분기 sqrt 보정
§4-3: cell-04 walk_forward의 prior 결정 분기 추가
§4-4: cell-05 스킵 조건 두 곳에 'rmse_per_ticker' 추가
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / '99_run.ipynb'
assert NB.exists(), f'노트북 없음: {NB}'

nb = json.loads(NB.read_text(encoding='utf-8'))


def patch(cell, old: str, new: str, must_replace: int = 1):
    """source list를 join해서 replace 후 splitlines(keepends=True)로 복원."""
    src = ''.join(cell['source'])
    cnt = src.count(old)
    assert cnt == must_replace, (
        f"cell {cell.get('id')}: '{old[:60]}...' must_replace={must_replace}, found={cnt}"
    )
    src = src.replace(old, new)
    cell['source'] = src.splitlines(keepends=True)


for cell in nb['cells']:
    cid = cell.get('id', '')

    # ── §4-1: cell-00 import 라인 ──────────────────────────────────────────
    if cid == 'cell-00':
        patch(
            cell,
            old=(
                "    compute_omega_he, compute_omega_scaled, compute_omega_rmse,\n"
            ),
            new=(
                "    compute_omega_he, compute_omega_scaled,\n"
                "    compute_omega_rmse, compute_omega_rmse_per_ticker,\n"
            ),
        )

    # ── cell-03: §3-2 신규 함수 + 신규 분기, §4-2 sqrt 보정 ────────────────
    elif cid == 'cell-03':
        # (1) §3-2: get_prior_weights 함수 직후에 get_prior_weights_rp 추가
        old_prior = (
            "def get_prior_weights(cfg, valid_tix, mcap):\n"
            "    \"\"\"prior 시장가중치 w_mkt 반환.\"\"\"\n"
            "    prior = cfg.get('prior', 'capm_mcap')\n"
            "    if prior == 'capm_mcap':\n"
            "        return (mcap / mcap.sum()).reindex(valid_tix).fillna(0)\n"
            "    elif prior == 'capm_eq':\n"
            "        n = len(valid_tix)\n"
            "        return pd.Series(1.0 / n, index=valid_tix)\n"
            "    else:\n"
            "        raise ValueError(f'prior={prior!r} 미지원')\n"
        )
        new_prior = old_prior + (
            "\n"
            "\n"
            "def get_prior_weights_rp(valid_tix, month_df):\n"
            "    \"\"\"\n"
            "    Risk Parity prior: w_i ∝ 1 / vol_21d_i, sum=1.\n"
            "    capm_rp prior 전용. 기존 get_prior_weights는 그대로 두고 분기에서 호출.\n"
            "    \"\"\"\n"
            "    vol = month_df['vol_21d'].reindex(valid_tix).replace(0, np.nan).dropna()\n"
            "    inv_vol = 1.0 / vol\n"
            "    w = inv_vol / inv_vol.sum()\n"
            "    return w.reindex(valid_tix).fillna(0)\n"
        )
        patch(cell, old_prior, new_prior)

        # (2) §4-2: get_omega의 'rmse' 분기에서 abs_err.mean() → sqrt(mean(abs_err^2))
        old_rmse_line = (
            "        pred_rmse = float(recent['abs_err'].mean()) if len(recent) > 0 else 0.39\n"
        )
        new_rmse_line = (
            "        pred_rmse = float(np.sqrt((recent['abs_err'] ** 2).mean())) if len(recent) > 0 else 0.39265\n"
        )
        patch(cell, old_rmse_line, new_rmse_line)

        # (3) §3-2: get_omega의 마지막 else 직전에 'rmse_per_ticker' elif 추가
        old_omega_tail = (
            "        pred_rmse = float(np.sqrt((recent['abs_err'] ** 2).mean())) if len(recent) > 0 else 0.39265\n"
            "        return compute_omega_rmse(P, Sigma, TAU, pred_rmse)\n"
            "\n"
            "    else:\n"
            "        raise ValueError(f'omega_mode={mode!r} 미지원')\n"
        )
        new_omega_tail = (
            "        pred_rmse = float(np.sqrt((recent['abs_err'] ** 2).mean())) if len(recent) > 0 else 0.39265\n"
            "        return compute_omega_rmse(P, Sigma, TAU, pred_rmse)\n"
            "\n"
            "    elif mode == 'rmse_per_ticker':\n"
            "        # 옵션 2: 종목별 RMSE → P 가중 결합\n"
            "        if lstm_preds is None:\n"
            "            return compute_omega_he(P, Sigma, TAU)\n"
            "        cutoff = pred_date - pd.DateOffset(months=12)\n"
            "        recent = lstm_preds[\n"
            "            (lstm_preds.index.get_level_values('date') >= cutoff) &\n"
            "            (lstm_preds.index.get_level_values('date') <= pred_date)\n"
            "        ]\n"
            "        if len(recent) == 0:\n"
            "            return compute_omega_he(P, Sigma, TAU)\n"
            "        # 종목별 RMSE = sqrt(mean(err^2 by ticker))\n"
            "        sq = recent['abs_err'] ** 2\n"
            "        rmse_per_ticker = np.sqrt(sq.groupby(level='ticker').mean())\n"
            "        return compute_omega_rmse_per_ticker(P, Sigma, TAU, rmse_per_ticker)\n"
            "\n"
            "    else:\n"
            "        raise ValueError(f'omega_mode={mode!r} 미지원')\n"
        )
        patch(cell, old_omega_tail, new_omega_tail)

    # ── §4-3: cell-04 walk_forward 의 prior 결정 분기 추가 ─────────────────
    elif cid == 'cell-04':
        old_prior_call = (
            "            w_mkt   = get_prior_weights(cfg, valid_tix, mcap)\n"
            "            pi, lam = compute_pi(Sigma, w_mkt, spy_excess, sigma2_mkt)\n"
        )
        new_prior_call = (
            "            if cfg.get('prior') == 'capm_rp':\n"
            "                w_mkt = get_prior_weights_rp(valid_tix, month_df)\n"
            "            else:\n"
            "                w_mkt = get_prior_weights(cfg, valid_tix, mcap)\n"
            "            pi, lam = compute_pi(Sigma, w_mkt, spy_excess, sigma2_mkt)\n"
        )
        patch(cell, old_prior_call, new_prior_call)

    # ── §4-4: cell-05 스킵 조건 두 곳 확장 ─────────────────────────────────
    elif cid == 'cell-05':
        # (1) run_list 필터 부분
        patch(
            cell,
            old="            and not (cfg.get('omega_mode') == 'rmse' and not LSTM_AVAILABLE)]\n",
            new="            and not (cfg.get('omega_mode') in ('rmse', 'rmse_per_ticker') and not LSTM_AVAILABLE)]\n",
        )
        # (2) 루프 안의 SKIP 메시지 부분
        patch(
            cell,
            old="    if cfg.get('omega_mode') == 'rmse' and not LSTM_AVAILABLE:\n",
            new="    if cfg.get('omega_mode') in ('rmse', 'rmse_per_ticker') and not LSTM_AVAILABLE:\n",
        )


# ── 출력 ───────────────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'OK: {NB.name} 패치 완료')
