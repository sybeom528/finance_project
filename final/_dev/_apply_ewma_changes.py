"""
99_run.ipynb 변경 스크립트 — EWMA omega 추가 (1회용).

변경 사항:
  1. cell-00 import 라인에 compute_omega_ewma 추가
  2. cell-04 walk_forward 함수 안에:
     - prev_omega, prev_e_sq 변수 초기화
     - omega 분기에 'ewma' 처리 추가
     - 매월 e_t 계산 후 state 갱신
     - meta_list 에 view_pred_ret, view_real_ret, omega 기록
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / '99_run.ipynb'
assert NB.exists(), f'노트북 없음: {NB}'

nb = json.loads(NB.read_text(encoding='utf-8'))


def patch(cell, old: str, new: str, must_replace: int = 1):
    src = ''.join(cell['source'])
    cnt = src.count(old)
    assert cnt == must_replace, (
        f"cell {cell.get('id')}: '{old[:80]}...' must_replace={must_replace}, found={cnt}"
    )
    src = src.replace(old, new)
    cell['source'] = src.splitlines(keepends=True)


for cell in nb['cells']:
    cid = cell.get('id', '')

    # ── cell-00: import 에 compute_omega_ewma 추가 ──────────────────────
    if cid == 'cell-00':
        patch(
            cell,
            old=(
                "    compute_omega_rmse, compute_omega_rmse_per_ticker,\n"
            ),
            new=(
                "    compute_omega_rmse, compute_omega_rmse_per_ticker,\n"
                "    compute_omega_ewma,\n"
            ),
        )

    # ── cell-04: walk_forward 의 omega state 전파 로직 ───────────────────
    elif cid == 'cell-04':
        # (1) prev_w 초기화 옆에 prev_omega, prev_e_sq 도 추가
        old_init = (
            "    weights_history = {}\n"
            "    prev_w = None\n"
            "    _t0 = time.time()\n"
        )
        new_init = (
            "    weights_history = {}\n"
            "    prev_w = None\n"
            "    prev_omega = None       # EWMA state: Ω_{t-1}\n"
            "    prev_e_sq  = None       # EWMA state: e²_{t-1}\n"
            "    _t0 = time.time()\n"
        )
        patch(cell, old_init, new_init)

        # (2) omega 계산 분기 — 기존 'else' 블록 (BL 정상) 안에서 omega 계산 부분 변경
        old_bl_block = (
            "            else:\n"
            "                Q     = get_Q(cfg, P, valid_tix, train_dates, pred_date, all_dates)\n"
            "                omega = get_omega(cfg, P, Sigma, pred_date)\n"
            "                mu_BL = black_litterman(pi, Sigma, P, Q, omega, TAU)\n"
            "                w     = optimize_portfolio(mu_BL, Sigma, lam, max_w)\n"
            "                mu_meta = Q\n"
        )
        new_bl_block = (
            "            else:\n"
            "                Q     = get_Q(cfg, P, valid_tix, train_dates, pred_date, all_dates)\n"
            "                # omega — 'ewma' 모드는 직전 state(prev_omega/prev_e_sq) 활용\n"
            "                if cfg.get('omega_mode') == 'ewma':\n"
            "                    lambda_ = float(cfg.get('lambda', 0.94))\n"
            "                    omega = compute_omega_ewma(\n"
            "                        P, Sigma, TAU, prev_e_sq, prev_omega, lambda_)\n"
            "                else:\n"
            "                    omega = get_omega(cfg, P, Sigma, pred_date)\n"
            "                mu_BL = black_litterman(pi, Sigma, P, Q, omega, TAU)\n"
            "                w     = optimize_portfolio(mu_BL, Sigma, lam, max_w)\n"
            "                mu_meta = Q\n"
        )
        patch(cell, old_bl_block, new_bl_block)

        # (3) e_t 계산 + state 갱신 + meta 기록 — actual_ret 직후
        old_meta_block = (
            "            meta_list.append({'date': pred_date, 'Q': mu_meta, 'lam': lam})\n"
            "            weights_history[pred_date] = w\n"
            "            prev_w = w\n"
        )
        new_meta_block = (
            "            # ── EWMA state 갱신용 e_t = P · (μ_BL - 실현) 계산 ──\n"
            "            view_pred_ret = np.nan\n"
            "            view_real_ret = np.nan\n"
            "            view_e        = np.nan\n"
            "            if (not is_capm) and (not is_naive):\n"
            "                # P @ μ_BL 와 P @ 실현수익률 — 인덱스 정렬 보장\n"
            "                P_aligned    = P.reindex(valid_tix).fillna(0)\n"
            "                mu_aligned   = mu_BL.reindex(valid_tix).fillna(0)\n"
            "                ret_aligned  = actual_ret.reindex(valid_tix).fillna(0)\n"
            "                view_pred_ret = float((P_aligned * mu_aligned).sum())\n"
            "                view_real_ret = float((P_aligned * ret_aligned).sum())\n"
            "                view_e        = view_pred_ret - view_real_ret\n"
            "                # EWMA state 갱신 (다음 시점 위해)\n"
            "                if cfg.get('omega_mode') == 'ewma':\n"
            "                    prev_e_sq  = float(view_e ** 2)\n"
            "                    prev_omega = float(omega)\n"
            "\n"
            "            meta_list.append({\n"
            "                'date': pred_date, 'Q': mu_meta, 'lam': lam,\n"
            "                'omega': float(omega) if (not is_capm and not is_naive) else np.nan,\n"
            "                'view_pred_ret': view_pred_ret,\n"
            "                'view_real_ret': view_real_ret,\n"
            "                'view_e':        view_e,\n"
            "            })\n"
            "            weights_history[pred_date] = w\n"
            "            prev_w = w\n"
        )
        patch(cell, old_meta_block, new_meta_block)


# ── 출력 ───────────────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'OK: {NB.name} EWMA 패치 완료')
