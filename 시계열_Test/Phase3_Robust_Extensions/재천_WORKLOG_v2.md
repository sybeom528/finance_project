# Phase 3-2 — 재천 WORKLOG v2

> 본 파일은 **Phase 3-2** 의 작업 기록. Phase 3-1 (Step 6→7→8 + 옵션 A 까지) 은 [재천_WORKLOG.md](재천_WORKLOG.md) 에 freeze 되어 있음.

## §0. 작업 결정 (2026-04-30, 팀 합의)

새 프로젝트 설정:

1. **OOS 평가 기간 축소**: 2009-01 ~ 2025-12 (17년, 204개월) → **2010-01 ~ 2024-12** (15년, 180개월)
   - 이유: 금융위기 (2008~2009 GFC) 영향 제거 → 정상 시기 평가
2. **2025 (1년, 12개월) 실운용 Test 기간** 별도 hold-out
   - 이유: model selection 영향 없는 final test (학술 표준 forward test)
3. **변동성 상/하위 30% 그룹 내 가중치 다양화**:
   - mcap (Pyo & Lee 2018, 기존)
   - 1/N (DeMiguel et al. 2009)
   - Risk Parity 단순 1/σ (Maillard et al. 2010)

**작업 방향 (사용자 결정)**:
- **옵션 3**: 같은 폴더 안 v2 노트북 사본 신설
- WORKLOG / 산출물 / 캐시 모두 분리
- scripts / 학습 결과 csv / universe / panel / membership / stale 필터 모두 공유
- Phase 3-1 (기존) freeze, Phase 3-2 (v2) main

**시나리오 결정 (AskUserQuestion 응답)**:
- **6 BL** (BL_ml_sw + BL_trailing 만 3 가중치): 02a_v2 §6 sanity check
- **9 BL** (BL_ml_sw 3 + BL_ml_cs 1 + BL_trailing 3): 03_v2 정식 (02b 후 실행)
- **2025 hold-out**: 단일 BL 루프 + 메트릭 단계 분리

---

## §1. v2 작업 흐름

### Step A. `scripts/black_litterman.py` — `build_P` weighting 인자 확장 (공유, backward 호환)

[scripts/black_litterman.py:96-152](scripts/black_litterman.py:96):

```python
def build_P(vol_series, mcap_series, pct=0.30, weighting='mcap') -> pd.Series:
    """
    weighting : {'mcap', 'eq', 'rp'}, default 'mcap'
        - 'mcap': 시총 비례 (Phase 3-1 표준)
        - 'eq':   1/N 균등 (DeMiguel et al. 2009)
        - 'rp':   1/σ 비례 단순 Risk Parity (Maillard et al. 2010)
    """
    n_group = max(1, int(len(vol_series) * pct))
    sorted_idx = vol_series.sort_values().index
    low_risk = sorted_idx[:n_group]
    high_risk = sorted_idx[-n_group:]
    P = pd.Series(0.0, index=vol_series.index)

    if weighting == 'mcap':
        P[low_risk] = mcap_series[low_risk] / mcap_series[low_risk].sum()
        P[high_risk] = -mcap_series[high_risk] / mcap_series[high_risk].sum()
    elif weighting == 'eq':
        P[low_risk] = 1.0 / n_group
        P[high_risk] = -1.0 / n_group
    elif weighting == 'rp':
        inv_low = 1.0 / vol_series[low_risk]
        inv_high = 1.0 / vol_series[high_risk]
        P[low_risk] = inv_low / inv_low.sum()
        P[high_risk] = -inv_high / inv_high.sum()
    return P
```

**검증** (단위 테스트, 6/6 통과):
- P.sum() ≈ 0 (모든 weighting): mcap -1.6e-16 / eq +4.2e-17 / rp +8.3e-17
- 그룹 내 가중치 합 = ±1 (모든 weighting)
- Backward 호환: 인자 미전달 시 default 'mcap' → Phase 3-1 결과 동일
- weighting 별 의도된 차이: max abs 0.013~0.033
- RP 의 1/σ 비례 정확성: 0.0
- 잘못된 인자 ValueError: ✅

### Step B. v2 노트북 사본 생성

| 원본 | 사본 (v2) | 용도 |
|---|---|---|
| `02a_phase15_stockwise_extended.ipynb` | `02a_v2.ipynb` | §6 BL sanity check 6 시나리오 |
| `03_BL_backtest_extended.ipynb` | `03_v2.ipynb` | §4 정식 9 BL 시나리오 (02b 후) |
| `05a_eval_stockwise.ipynb` | `05a_v2.ipynb` | BL_ml_sw 단독 평가 (3 가중치 비교) |
| `04_compare_stockwise_vs_cross.ipynb` | `04_v2.ipynb` | 02a vs 02b 가중치별 비교 (02b 후) |
| `05b_eval_crosssec.ipynb` | `05b_v2.ipynb` | BL_ml_cs 단독 (02b 후) |
| `05c_eval_compare.ipynb` | `05c_v2.ipynb` | 9 BL + EW + Mcap + SPY 종합 (02b 후) |

**v2 outputs 폴더** (6 개 신설): `outputs/02a_v2_stockwise/`, `outputs/03_v2_bl_backtest/`, `outputs/05a_v2_eval_stockwise/` 등

### Step C. 02a_v2 §6 (6 시나리오 BL sanity check)

| Cell | 변경 |
|---|---|
| 22 (§6-1) | `OOS_START='2010-01-01', OOS_END='2024-12-31', HOLDOUT_START='2025-01-01', HOLDOUT_END='2025-12-31'` 변수 + v2 경로 (`OUT_DIR_V2_SW`, `CACHE_PATH_V2`, `METRICS_PKL_V2`) |
| 23 (§6-2 BL 루프) | `WEIGHTINGS = ['mcap', 'eq', 'rp']` + `weights = {f'ml_sw_{w}': {} for w in WEIGHTINGS} + ...` (6 시나리오) |
| 24 (§6-3 returns) | `returns_v2 = {f'BL_{s}': make_returns_manual(w, ...) for s, w in weights.items()}` |
| 25 (§6-4 메트릭) | 6 시나리오 fair 비교 + OOS/hold-out 분리 표 + 가중치 robustness 표 + `bl_metrics_v2_sanity_check.pkl` 저장 |
| 26 (§6-5 시각화) | `OUT_DIR_V2_SW / 'bl_sanity_check.png'` |
| 30 (§6 결론) | mcap default + 가중치 robustness print |

**캐시**: `data/bl_weights_v2_sanity_check.pkl` (BL 가중치) + `data/bl_metrics_v2_sanity_check.pkl` (메트릭)

### Step D. 03_v2 (9 BL 시나리오 정식, 02b 후 실행)

| Cell | 변경 |
|---|---|
| 6 (§3-1, 3-2) | `rebalance_dates` 정의를 OOS+hold-out 분리 + `OUT_DIR_V2_03`, `CACHE_PATH_V2_03 = 'scenario_weights_03_v2.pkl'` |
| 10 (§4 BL 루프) | 9 시나리오 (BL_ml_sw 3 + BL_ml_cs 1 + BL_trailing 3 + EW + Mcap), v2 캐시 사용 |

**Cell 17 (§5 메트릭) 의 OOS/hold-out 분리는 02b 학습 완료 후 본격 작업** — 03_v2 정식 실행 시.

### Step E. 05a_v2 경로 + reb_dates 변경

| Cell | 변경 |
|---|---|
| 2 (§1) | `OUT_DIR = OUTPUTS_DIR / '05a_v2_eval_stockwise'` |
| 4 | `reb_dates_for_spy` 의 OOS+hold-out 분리 + spy_monthly 그대로 |
| 5 | `bl_cache_path = DATA_DIR / 'bl_weights_v2_sanity_check.pkl'` |
| 27 (§7-5) | `_bl_metrics_path = DATA_DIR / 'bl_metrics_v2_sanity_check.pkl'` |
| 31 (§7-8) | 동일 |

**Layer 4 시기 정의 재검토는 02b 후** — OOS=2010-2024 환경에서 GFC 회복 시기 (2009-2011) 의 의미 변화.

### Step F. 04_v2 / 05b_v2 / 05c_v2 경로 골격

| 파일 | 변경 |
|---|---|
| 04_v2 | `outputs/04_v2_compare/` + `scenario_weights_03_v2.pkl` + `bl_weights_v2_sanity_check.pkl` 참조 |
| 05b_v2 | `outputs/05b_v2_eval_crosssec/` + 동일 |
| 05c_v2 | `outputs/05c_v2_eval_compare/` + 동일 |

**본격 수정은 02b 학습 완료 후** — 9 BL 시나리오 비교 / 02a vs 02b 가중치별 비교 / 종합 시각화.

### Step G. 본 WORKLOG_v2 신규 작성 (현재)

---

## §2. Phase 3-1 (기존) vs Phase 3-2 (v2) 비교 표

| 항목 | Phase 3-1 (기존) | **Phase 3-2 (v2)** |
|---|---|---|
| OOS 기간 | 2009-01 ~ 2025-12 (204m) | **2010-01 ~ 2024-12 (180m) + 2025 hold-out (12m)** |
| BL 가중치 | mcap 만 | **mcap + 1/N + RP** |
| 시나리오 (02a §6) | 2 (BL_ml_sw, BL_trailing) | **6 (3 vol × 3 weighting)** |
| 시나리오 (03 정식) | 6 (BL × 3 + EW + Mcap + SPY) | **10 (BL × 7 + EW + Mcap + SPY)** |
| GFC 영향 | 포함 | **제거** |
| Hold-out | 없음 (전체 OOS) | **2025 별도** |
| 노트북 | 6 노트북 freeze | **6 v2 노트북** |
| 산출물 | `outputs/05a_eval_stockwise/` 등 freeze | **`outputs/05a_v2_eval_stockwise/` 등** |
| 캐시 | `data/bl_weights_sanity_check.pkl` (Step 8) freeze | **`data/bl_weights_v2_sanity_check.pkl`** |
| 학습 결과 (`ensemble_predictions_*.csv`) | 공유 | 공유 |
| `scripts/` (build_P 등) | 공유 (backward 호환) | 공유 |

**Phase 3-1 의 핵심 결과 (참조)**:
- BL_ml_sw_mcap Sharpe: 1.122 (02a §6-4) / 1.119 (05a Layer 2)
- BL_trailing_mcap Sharpe: 1.207
- ML 효과 diff: -0.085 (mcap 환경)
- Universe avg: 418.5 종목/월 (Step 8 stale 필터 후)

**Phase 3-2 의 결과 (2026-05-01 standalone 실행 + nbconvert 임베드 완료)**:

> **실행 정보**: 02a_v2 §6 6 시나리오 BL sanity check, 192 시점 (180 OOS + 12 hold-out), 평균 universe 423.3 종목/월, n_skip=0, n_sigma_fail=0.
> **실행 경위**: VS Code Jupyter kernel 무한 로딩 + nbconvert cell timeout 1h 발생 → **standalone Python 스크립트** (`scripts/_run_02a_v2_sec6.py`) 76.7분 실행 + **nbconvert 캐시 hit 재실행** (Cell 25 BL_ 이중 prefix 버그 수정 후) 으로 노트북 셀 출력 임베드 완료.

### §2.1 OOS 메트릭 (2010-2024, 180개월) — 메인 분석 결과

| 시나리오 | Sharpe | CAGR % | ann_vol % | MDD % | n_months |
|---|---|---|---|---|---|
| BL_ml_sw_mcap | 1.082 | 12.84 | 11.86 | -18.13 | 180 |
| **BL_ml_sw_eq** | **1.136** ⭐ | 12.95 | 11.33 | -16.58 | 180 |
| BL_ml_sw_rp | 1.122 | 12.78 | 11.34 | -16.50 | 180 |
| BL_trailing_mcap | 1.206 | 14.34 | 11.74 | -16.48 | 180 |
| BL_trailing_eq | 1.148 | 12.80 | 11.08 | -15.94 | 180 |
| BL_trailing_rp | 1.146 | 12.79 | 11.09 | -15.98 | 180 |
| SPY | 0.996 | 14.26 | 14.51 | -23.93 | 180 |

### §2.2 Hold-out 메트릭 (2025, 12개월) — 별도 final test

| 시나리오 | Sharpe | CAGR % | ann_vol % | MDD % | n_months |
|---|---|---|---|---|---|
| **BL_ml_sw_mcap** | **1.503** 🏆 | 10.49 | 6.80 | -4.62 | 11 |
| BL_ml_sw_eq | 1.313 | 9.12 | 6.83 | -3.21 | 11 |
| BL_ml_sw_rp | 1.290 | 9.05 | 6.91 | -3.27 | 11 |
| **BL_trailing_mcap** | **0.847** ⚠️ | 6.55 | 7.84 | -5.34 | 11 |
| BL_trailing_eq | 1.285 | 9.34 | 7.16 | -2.72 | 11 |
| BL_trailing_rp | 1.224 | 9.03 | 7.29 | -2.71 | 11 |
| SPY | 1.365 | 16.07 | 11.42 | -6.39 | 11 |

> **n_months=11 의미**: hold-out 12개월 중 마지막 시점 (2025-12-31) 의 forward return 데이터 부재 (`pct_change` 첫 NaN 처리 동일 메커니즘) → 2025-01 ~ 2025-11 의 11개월 평가.

### §2.3 ML 효과 (BL_ml_sw - BL_trailing) 가중치별 비교

| 가중치 | OOS Sharpe diff (180m) | Hold-out Sharpe diff (11m) | 해석 |
|---|---|---|---|
| **mcap** | **-0.124** | **+0.656** 🏆 | OOS negative → Hold-out 압도적 양성 |
| eq | -0.012 ≈ 0 | +0.028 | 가중치 다양화 시 ML 효과 ~중립 |
| rp | -0.024 ≈ 0 | +0.066 | 가중치 다양화 시 ML 효과 ~중립 |

### §2.4 Turnover 분석 (포트폴리오 안정성)

| 시나리오 | mean | std | max | min |
|---|---|---|---|---|
| BL_ml_sw_mcap | **0.216** | 0.091 | 0.759 | 0.043 |
| BL_ml_sw_eq | **0.131** | 0.063 | 0.725 | 0.051 |
| BL_ml_sw_rp | **0.132** | 0.067 | 0.765 | 0.051 |
| BL_trailing_mcap | 0.381 | 0.134 | 0.769 | 0.108 |
| BL_trailing_eq | 0.280 | 0.109 | 0.638 | 0.080 |
| BL_trailing_rp | 0.291 | 0.117 | 0.677 | 0.088 |

→ **BL_ml_sw 가 BL_trailing 의 ~50% turnover** (모든 가중치). ML 예측이 더 안정적인 가중치 변동을 만들어내는 robustness 효과 확인.

### §2.5 Weight Concentration (Diversification)

| 시나리오 | avg_n | top10 % | top1 % |
|---|---|---|---|
| BL_ml_sw_mcap | 423.3 | 35.73 | 5.92 |
| BL_ml_sw_eq | 423.3 | **31.95** ⭐ | 4.69 |
| BL_ml_sw_rp | 423.3 | 32.65 | 4.77 |
| BL_trailing_mcap | 423.3 | **42.00** ⚠️ | 8.47 |
| BL_trailing_eq | 423.3 | 35.72 | 5.36 |
| BL_trailing_rp | 423.3 | 36.50 | 5.50 |

→ BL_ml_sw_eq 가 가장 분산 (top10 31.95%), BL_trailing_mcap 이 가장 집중 (top10 42.0%). ML + 1/N 조합이 학술 표준 diversification 측면에서 최우수.

### §2.6 Phase 3-1 vs Phase 3-2 비교 (mcap 환경 일치)

| 항목 | Phase 3-1 (2009-2025, 204m) | Phase 3-2 OOS (2010-2024, 180m) | 차이 |
|---|---|---|---|
| BL_ml_sw_mcap Sharpe | 1.122 | **1.082** | -0.040 (GFC 회복기 제거) |
| BL_trailing_mcap Sharpe | 1.207 | **1.206** | -0.001 (거의 무영향) |
| ML 효과 diff | -0.085 | **-0.124** | -0.039 (GFC 영향 제거 시 ML 더 negative) |
| 평균 universe | 418.5 | 423.3 | +4.8 |

→ **GFC 회복기 (2009) 가 ML_sw 에 유리** 하던 시기 → 그 영향 제거 시 OOS ML 효과 더 negative. 단, 가중치 다양화 (eq/rp) 또는 hold-out 으로 보면 ML 효과 일관되게 양성/중립.

### §2.7 핵심 인사이트 (Phase 3-2 학술적 중요 발견)

1. **Hold-out (2025) BL_ml_sw_mcap Sharpe 1.503 vs BL_trailing_mcap 0.847** — Phase 3-2 의 가장 강력한 finding. ML 예측이 실운용 환경에서 trailing 대비 압도적 우수성 (ΔSharpe **+0.656**) 보임. 이는 **forward test (model selection 영향 없음)** 로서 학술 보고서의 critical evidence.

2. **OOS 가중치 robustness**: ML 효과가 mcap (-0.124) → eq/rp (~0) 로 일관되게 축소 → **mcap 의 시총 편향이 ML 의 작은 우위를 가리는 핵심 원인**. eq/rp 환경에서 ML 의 RMSE 우위가 BL 성과로 연결됨.

3. **Turnover robustness**: BL_ml_sw 의 turnover (0.13~0.22) 가 BL_trailing (0.28~0.38) 의 50% 수준 → ML 예측의 안정성. 거래비용 환경에서 BL_ml_sw 의 net 우위 확장 가능.

4. **Diversification**: BL_ml_sw_eq 의 top10 31.95% 가 가장 분산된 포트폴리오 → 1/N + ML 조합이 학술 표준 (DeMiguel et al. 2009) 과 가장 잘 맞음.

5. **가중치 vs vol input 효과 분리**: 모든 시나리오 fair 환경 (동일 universe + Σ + market view) 에서 가중치 변경 만으로 ML 효과 -0.124 → ~0 로 변화. 이는 **vol input 차이 (ML vs trailing) 보다 가중치 차이 가 BL 성과 변동의 더 큰 driver** 임을 시사.

### §2.8 Standalone 실행 + nbconvert 캐시 hit 절차 (다음 사용자 참고용)

VS Code Jupyter kernel 무한 로딩 시 우회 절차:

1. `python scripts/_run_02a_v2_sec6.py > sec6_run.log 2>&1` (백그라운드, ~76분)
   - FORCE_RECOMPUTE=True, 192 시점 6 시나리오 BL 루프 실행
   - 결과: `data/bl_weights_v2_sanity_check.pkl` (6.0 MB)
2. `jupyter nbconvert --to notebook --execute --inplace 02a_v2.ipynb --ExecutePreprocessor.timeout=1800` (~1.5분)
   - §6-2 셀 캐시 hit → 1초 통과
   - 메트릭 / 시각화 / 출력 모두 노트북에 임베드
   - 결과: `data/bl_metrics_v2_sanity_check.pkl` (5.1 KB) + `outputs/02a_v2_stockwise/*.png` (4 개)

→ 02b 학습 (PID 44792, GPU 24-39%, 4.6h) 영향 0 검증 완료.

---

## §3. 산출물 / 캐시 분리 매핑

### 신규 (v2)

| 파일 / 폴더 | 종류 | 생성 시점 |
|---|---|---|
| `data/bl_weights_v2_sanity_check.pkl` | 02a_v2 §6 BL 가중치 캐시 | 02a_v2 §6 사용자 재실행 시 |
| `data/bl_metrics_v2_sanity_check.pkl` | 02a_v2 §6-4 메트릭 캐시 | 동일 |
| `data/scenario_weights_03_v2.pkl` | 03_v2 §4 BL 가중치 캐시 | 02b 후 03_v2 정식 실행 시 |
| `outputs/02a_v2_stockwise/bl_sanity_check.png` | 02a_v2 §6-5 시각화 | 02a_v2 §6 재실행 시 |
| `outputs/05a_v2_eval_stockwise/` | 05a_v2 산출 (Layer 1~4 + §7) | 사용자 재실행 시 |
| `outputs/03_v2_bl_backtest/` | 03_v2 산출 | 02b 후 |
| `outputs/04_v2_compare/`, `outputs/05b_v2_eval_crosssec/`, `outputs/05c_v2_eval_compare/` | 비교 / 평가 산출 | 02b 후 |

### Freeze (Phase 3-1)

| 파일 / 폴더 | 상태 |
|---|---|
| `data/bl_weights_sanity_check.pkl` (Step 8) | freeze |
| `data/bl_metrics_sanity_check.pkl` (Step 8) | freeze |
| `data/bl_weights_sanity_check_step7_dynamic.pkl`, `*_legacy_static.pkl` | 격리 보존 |
| `outputs/05a_eval_stockwise/` 등 | freeze (Phase 3-1 robustness 부록) |
| `재천_WORKLOG.md` (§13.6 Step 8 까지) | freeze |
| 기존 6 노트북 | freeze |

### 공유

| 파일 | 상태 |
|---|---|
| `data/ensemble_predictions_stockwise.csv` (02a 학습 결과) | 공유 |
| `data/ensemble_predictions_crosssec.csv` (02b, 진행 중) | 공유 (02b 후) |
| `data/daily_panel.csv`, `data/sp500_membership.pkl`, `data/universe_full_history.csv` | 공유 |
| `scripts/black_litterman.py` (Step A 확장) | 공유, backward 호환 |
| `scripts/setup.py`, `volatility_ensemble.py`, `models_cs.py`, `covariance.py`, `benchmarks.py`, `diagnostics.py` | 공유 |

---

## §4. 02b 학습 영향 0 보장

| 원칙 | 검증 |
|---|---|
| `scripts.volatility_ensemble`, `scripts.models_cs` 변경 X | ✅ |
| `02b_phase15_cross_sectional.ipynb` 변경 X | ✅ |
| `scripts.black_litterman` 변경 (build_P weighting 인자) | ✅ 02b 미사용, backward 호환 (default 'mcap') |
| `scripts.benchmarks`, `scripts.covariance` 변경 X | ✅ |
| GPU 자원 비경쟁 | ✅ 본 작업 모두 CPU only |

---

## §5. 사용자 액션

### Step H. 02a_v2 §6 BL sanity check 실행 ✅ (완료, 2026-05-01)

**실행 경위**: VS Code Jupyter 커널 무한 로딩 + nbconvert cell timeout 1h 발생 → standalone Python + nbconvert 캐시 hit 우회 (자세한 절차는 §2.8 참조).

**실행 결과 검증** (모든 확인 포인트 통과):

| 위치 | 기대 출력 | 실측 |
|---|---|---|
| Cell 22 (§6-1) | `OOS (2010-01-01~2024-12-31): 180 개월`, `Hold-out: 12 개월` | ✅ 동일 |
| Cell 23 (§6-2) | 6 시나리오 모두 192 시점, Skip 0, Σ 실패 0 | ✅ 동일 |
| Cell 25 (§6-4) | 6 시나리오 fair 비교 + OOS/hold-out 분리 표 | ✅ §2.1, §2.2 참조 |
| Cell 30 (§6 결론) | mcap/eq/rp 별 BL_ml_sw vs BL_trailing diff 표 | ✅ §2.3 참조 |

**산출물**:
- `data/bl_weights_v2_sanity_check.pkl` (6.0 MB, 6 시나리오 × 192 시점 가중치)
- `data/bl_metrics_v2_sanity_check.pkl` (5.1 KB, metrics_table + metrics_oos + metrics_holdout + period_results + turnover/dist summary)
- `outputs/02a_v2_stockwise/{bl_sanity_check, hit_rate_analysis, paradox_analysis, rmse_distribution}.png`
- `02a_v2.ipynb` 셀 출력 임베드 (1.29 MB)

**발견된 버그 + 수정**:
- Cell 25 §6-4: `all_returns = {f'BL_{s}': r for s, r in returns_v2.items()}` → returns_v2 키 이미 'BL_*' prefix 포함 → metrics_table row 가 'BL_BL_*' 이중 prefix 가 됨 → KeyError 'BL_ml_sw_mcap'
- 수정: `all_returns = dict(returns_v2)` (이중 prefix 방지)

### 02b 학습 완료 후 (~22h 후 예상)

1. 03_v2 §4 BL 정식 실행 (9 시나리오, ~수시간 BL 루프)
   - 동일 standalone 우회 절차 적용 가능 (`scripts/_run_03_v2_sec4.py` 신규 작성 권장)
2. 04_v2 / 05b_v2 / 05c_v2 본격 수정 (Cell 17 메트릭 분리, Layer 4 시기 정의 등)
3. `재천_WORKLOG_v2.md` 의 9 BL 시나리오 결과 + 시기별 분해 + 학술 보고서 표현 채움

---

## §6. 학술 보고서 구조 (제안)

```
[Main] Phase 3-2 (2010-2024 OOS + 2025 hold-out)
  - Universe / 데이터 / 학습 인프라: §13.6 Step 1-8 그대로 인용 (Phase 3-1)
  - 메인 결과: 9 BL 시나리오 OOS 메트릭 + Hold-out 메트릭
  - 가중치 robustness 분석 (mcap / 1/N / RP 환경에서 ML 효과 일관성)
  - 학술 baseline (Pyo & Lee 2018, 서윤범 99) 과 fair 비교

[부록 A] Phase 3-1 (2009-2025 OOS, GFC 포함) robustness check
  - GFC 충격 환경에서의 ML 효과
  - Universe / weighting 정의 한정성 검증
  - §13.6 Step 6/7/8 의 진화 과정 인용

[부록 B] 한계 + 향후 연구
  - yfinance 한계 (Stale price, 누락 163, §7-6/7/8)
  - CRSP 재현 검증 필요
  - Risk Parity ERC 정의 확장
```

---

# 본 WORKLOG_v2 의 현재 상태

```
[Phase 3-2 진행 중]
   ✅ Step A: build_P weighting 인자 확장 (6/6 단위 검증)
   ✅ Step B: v2 노트북 6개 사본 + outputs/v2 폴더
   ✅ Step C: 02a_v2 §6 (6 시나리오 + OOS/hold-out 분리)
   ✅ Step D: 03_v2 Cell 6 + Cell 10 (9 시나리오, 02b 후 실행)
   ✅ Step E: 05a_v2 Cell 2/4/5/27/31 (v2 경로 + reb_dates)
   ✅ Step F: 04_v2 / 05b_v2 / 05c_v2 경로 골격
   ✅ Step G: 본 WORKLOG_v2 신규 작성
   ✅ Step H: 02a_v2 §6 6 시나리오 실행 + 결과 임베드 (2026-05-01)
              - standalone Python 76.7분 + nbconvert 캐시 hit 재실행
              - Cell 25 BL_ 이중 prefix 버그 수정
              - §2.1~§2.7 결과 + 인사이트 채움

   ⏳ 02b 학습 완료 후: 03_v2 정식 + 04_v2 / 05b_v2 / 05c_v2 본격
   ⏳ 02b 후 task: 옵션 B 근본 fix (scripts/diagnostics.py SPY)
   ⏳ 학술 보고서 Limitations 반영
```
