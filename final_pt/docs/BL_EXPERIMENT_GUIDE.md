# Black-Litterman 실험 프레임워크 — 상세 가이드

> **최종 갱신: 2026-05-11**
> - p_mode: trailing_vol21/252 제거 → lstm_predicted 단일
> - p_weight: vol_mcap 제거 → mcap/eq/rp 만
> - q_mode: ff3_paper/none/capm 비활성 제거 → fixed/lambda/inv_lambda/raw_lam/vol_spread 5종
> - omega_mode: scaled/rmse 제거 → he_litterman/ff3_paper 2종
> - omega_scale 키 제거
> - Q/pct 민감도는 05b_Analyze.ipynb 내 in-code sweep (pkl 영구 저장 안 함)
> - 현재 EXPERIMENTS 총 **90개** (매트릭스 3 × 3 × 5 × 2)

---

## 1. 전체 구조

```
final_pt/
├── bl_config.py              ← EXPERIMENTS 정의 (90개)
├── bl_functions.py           ← BL 수식 핵심 함수 (Σ/π/P/Q/Ω/BL/TC/Metrics)
├── bl_runner.py              ← LSTM 로드 + monthly_cache + walk_forward 엔진
├── master_table.py           ← results/*.pkl → mt/rt 빌더
├── analyze_plots.py          ← 시각화 모듈
│
├── 04_BL_Walkforward.ipynb              ← walk_forward 실행 → results/*.pkl
├── 05b_Analyze.ipynb          ← 분석 단일 진입점 (K_CUT → I → J → K → L → M(Q sens) → N(PCT sens))
├── appendix/99_slot_effects.ipynb  ← 슬롯 차원 효과 라인플롯 (pivot CSV 자동 생성)
│
├── results/                  ← 90 pkl
├── data/                     ← monthly_panel.csv, daily_returns.pkl
└── data/03b_lstm/     ← LSTM 예측 결과
    └── data/ensemble_predictions_stockwise.csv
```

---

## 2. 실험 실행 순서

```
① bl_config.py에 실험 dict 추가 (필요 시)
② 04_BL_Walkforward.ipynb 셀 순서대로 실행
   cell-00 : 패키지 임포트 + bl_runner / bl_config import + 경로 설정
   cell-01 : 데이터 로드 (monthly_panel, daily_returns)
   cell-02 : LSTM 예측 로드 (bl_runner.load_lstm_pred)
   cell-cache : monthly_cache 빌드 (Σ/mcap/spy/π 컴포넌트 — bl_runner.build_monthly_cache)
   cell-05 : 전체 실험 walk_forward 실행 (이미 저장된 pkl 자동 스킵)
③ 05b_Analyze.ipynb 실행 (Setup → K_CUT → I → J → K → L → M → N 순서대로)
   ※ K_CUT은 mandatory pre-step. 모든 후속 분석은 2023-12-31 cutoff 기준.
④ (선택) appendix/99_slot_effects.ipynb 실행 — 슬롯 차원 효과 라인플롯
```

> **자동 스킵**: `(RESULTS_DIR / f'{cfg["name"]}.pkl').exists()` → 이미 있으면 스킵.
> 재실행하려면 해당 pkl 삭제.

---

## 3. 실험 추가 방법

### Case 1 — 기존 슬롯 파라미터만 변경 (가장 흔함)

`bl_config.py`의 `EXPERIMENTS`에 dict 한 줄 추가. **다른 파일 수정 불필요**.

```python
{**BASELINE, 'name': 'q_0007', 'q_value': 0.007},                            # Q값 변경
{**BASELINE, 'name': 'no_tc',  'tc': 0.0},                                   # 거래비용 제거
{**BASELINE, 'name': 'mat_eq_rp_vsp_pap', 'prior': 'capm_eq',
 'p_weight': 'rp', 'q_mode': 'vol_spread', 'omega_mode': 'ff3_paper'},
```

`{**BASELINE, ...}` 패턴: BASELINE 모든 값 상속 + 변경 슬롯만 덮어씀.

### Case 2 — 새 계산 방식 추가

3개 파일을 **이 순서대로** 수정:
1. `bl_functions.py` — 새 함수 추가 (예: `compute_Q_my_method`)
2. `bl_config.py` — 슬롯 주석 + 실험 dict 추가
3. `bl_runner.py` — dispatcher (`get_Q`, `get_omega` 등)에 elif 추가

---

## 4. 슬롯 키 레퍼런스 + 정확한 수식

> **변경 이력**: CAGR 정의를 산술평균×12 → 기하평균(복리)으로 수정. β/α 시차 misalignment 버그 수정.

### 4-0. 슬롯 키 일람표

| 슬롯 | 선택지 | 기본값 | 설명 |
|---|---|---|---|
| `prior` | `capm_mcap` / `capm_eq` / `capm_rp` | `capm_mcap` | Prior π 가중 |
| `p_mode` | `lstm_predicted` | `lstm_predicted` | P 분류용 변동성 (단일 옵션) |
| `p_weight` | `mcap` / `eq` / `rp` | `mcap` | P 행렬 가중 |
| `q_mode` | `fixed` / `vol_spread` / `lambda` / `raw_lam` / `inv_lambda` | `fixed` | Q 결정 방식 |
| `q_value` | float | `0.003` | Q 값(또는 base scaling) |
| `lam_mean` | float | `2.5` | λ 정규화 기준값 |
| `omega_mode` | `he_litterman` / `ff3_paper` | `he_litterman` | Ω 계산 방식 |
| `tc` | float | `0.002` | 편측(per-side) 거래비용 (20bp). turnover=Σ\|Δw\|∈[0,2]이라 TC = turnover×tc로 매수+매도 동시 반영 |
| `max_weight` | float | `0.10` | 단일 종목 상한 |
| `lstm_pred_path` | str | 자동 탐색 | LSTM 예측 파일 경로 |

---

### 4-1. 단일 view BL — 전체 수식 (K=1)

본 프로젝트는 자산별 view(K=N) 대신 **그룹 spread 단일 view(K=1)** 형식을 사용. P, Q, Ω 모두 **스칼라 계산**으로 단순화.

**입력**:
- `Σ` (n×n): Ledoit-Wolf 공분산. 직전 60개월 일별 수익률 1260일에서 추정
- `w_mkt` (n): prior 가중 벡터 (capm_mcap/eq/rp 중)
- `λ` (스칼라): 위험회피계수, `clip(SPY_excess / σ²_mkt, 0.5, 10.0)` 또는 고정 2.5
- `τ` (스칼라): prior 불확실성, **0.05 고정**

**Prior π** (n-vector):
```
π = λ · Σ · w_mkt          ← CAPM 균형수익률 역산
```

**View 슬롯**: 다음 4-3, 4-4, 4-5에서 정의되는 `P` (n-vector), `Q` (스칼라), `Ω` (스칼라).

**사후분포** (Sherman-Woodbury 닫힌해):
```
M     = (P τΣ Pᵀ) + Ω                  ← view 분산 합 (스칼라)
diff  = Q − Pπ                          ← view 오차 (스칼라)
μ_BL  = π + (τΣ Pᵀ) · (diff / M)       ← n-vector
Σ_BL  = τΣ − (τΣ Pᵀ)(P τΣ) / M         ← n×n, rank-1 update
```

**MVO**:
```
maximize  μ_BL^T w − (λ/2) w^T Σ_BL w
s.t.      Σw_i = 1, 0 ≤ w_i ≤ max_weight (기본 0.10)
```

---

### 4-2. Prior π — 3종

`prior` 슬롯이 `w_mkt`를 결정. π = λ Σ w_mkt 공식은 동일.

| `prior` | `w_mkt` 정의 | 의미 |
|---|---|---|
| `capm_mcap` | `w_i = mcap_i / Σ mcap_j` | 시총가중 (He-Litterman 1999 표준) |
| `capm_eq` | `w_i = 1/N` | 균등가중 (naïve baseline) |
| `capm_rp` | `w_i = (1/σ_i) / Σ(1/σ_j)` | 역변동성 가중 (Risk Parity 사상) |

> `capm_rp`의 σ는 **trailing 21일 실현변동성** 사용 (`vol_21d`). prior 단계라 LSTM 예측은 안 씀.

---

### 4-3. P 행렬 — 3종 가중 + 분류

**공통 분류**: LSTM 예측 변동성 σ_pred 를 오름차순 정렬 후
- 하위 `pct=0.30` → **저위험 그룹** (long, P > 0)
- 상위 `pct=0.30` → **고위험 그룹** (short, P < 0)

**σ 정의**: LSTM+HAR 앙상블 예측, `exp(y_pred_ensemble) × √252` (Phase3 산출, 연환산).
LSTM 예측이 있는 종목은 LSTM 값, 나머지는 `vol_21d` 폴백 (둘 다 연환산).

> ⚠️ **단위 가드**: [bl_runner.py](bl_runner.py) `get_vol_series` 에 `pred_slice.median() < 0.05` sanity guard. 누락 시 vol_21d(연환산)와 혼합되어 P 랭킹 왜곡.

**`p_weight`별 가중 수식** (σ_i: 자산 i의 변동성, m_i: 시가총액):

| `p_weight` | Long (저위험 그룹 i ∈ Low) | Short (고위험 그룹 i ∈ High) |
|---|---|---|
| `mcap` | `P_i = +m_i / Σ_{j∈Low} m_j` | `P_i = −m_i / Σ_{j∈High} m_j` |
| `eq` | `P_i = +1 / n_g` | `P_i = −1 / n_g` (n_g = 그룹 크기) |
| `rp` | `P_i = +(1/σ_i) / Σ_{j∈Low}(1/σ_j)` | `P_i = −σ_i / Σ_{j∈High} σ_j` |

**P 합 = 0** (자기자금 조달 long-short, market-neutral 형식).

---

### 4-4. Q (single view 기대 spread) — 5종

매 시점 t의 `λ_t = clip(SPY_excess_t / σ²_mkt,t, 0.5, 10.0)` (compute_pi에서 계산).
`q_base = q_value` (기본 0.003), `lam_mean = 2.5`.

| `q_mode` | Q 수식 | 메커니즘 |
|---|---|---|
| `fixed` | `Q = q_value` | 가장 단순. q_value 고정 |
| `vol_spread` (vsp) | `Q = q_base × clip(spread_t / spread_ref, 0.1, 3.0)`<br>`spread_t = mean(σ_pred[P<0]) − mean(σ_pred[P>0])`<br>`spread_ref = train 기간 expanding median` | LSTM 예측 vol 격차로 view 강도 동적 조절 (위기↑→Q↑) |
| `lambda` | `Q = q_base × clip(λ_t / lam_mean, 0.1, 3.0)` | 시장 안정(λ↑) → Q 강화 |
| `raw_lam` | `Q = max(0, q_base × lam_raw / lam_mean)`<br>`lam_raw = SPY_excess / σ²_mkt` (clip 전) | SPY 하락(lam_raw 음수) → Q=0 자연 게이팅 |
| `inv_lambda` | `Q = q_base × clip(lam_mean / λ_t, 0.1, 3.0)` | 위기(λ↓) → Q 강화 (반대 방향) |

> 📌 **`q_value`는 모든 동적 모드(vsp/lambda/raw_lam/inv_lambda)에서 base로 작용**. 단조 0.003 → 0.0070 sweep해도 절대값 분포만 평행이동(시그널은 동일).

---

### 4-5. Ω (view 분산) — 2종

**τ = 0.05 고정** (prior 불확실성 비율).

| `omega_mode` | Ω 수식 | 메커니즘 |
|---|---|---|
| `he_litterman` (he) | `Ω = max(τ · P Σ Pᵀ, 1e-8)` | He-Litterman 1999 표준. view 분산 = prior 분산 비례 |
| `ff3_paper` (pap) | `Ω_t = max((Q_{t−1} − actual_P_return_{t−1})², 1e-8)` (첫 달은 he fallback) | 직전월 view 예측오차² Bayesian rolling. **walk_forward inline 처리** (직전월 상태 필요), `get_omega` dispatcher 거치지 않음 |

> ⚠️ **`ff3_paper` 명명 주의**: 실제 동작은 직전월 예측오차² 적응형 (Bayesian rolling). 발표 시 그 표현으로 정확히 표기.

---

### 4-6. TC (Transaction Cost) + Metrics

**TC 적용** (walk_forward 안):
```
gross_ret_t = w_t · actual_ret_t
turnover_t  = (1/2) · Σ |w_t,i − w_{t−1},i|
net_ret_t   = gross_ret_t − turnover_t × tc       ← TC 차감 후
```
`ret` 시리즈(pkl 안)는 **net_ret_t**. mt/rt의 모든 sharpe·sortino·cagr·alpha는 TC 차감 후 값.

**핵심 메트릭** ([compute_metrics](bl_functions.py#L547)):
```
excess_t  = ret_t − rf_t
mkt_exc_t = mkt_t − rf_t                       ← mkt = pkl의 spy_ret (forward-aligned)

Sharpe   = mean(excess) / std(excess) × √12
Sortino  = mean(excess) × 12 / (std(excess[excess<0]) × √12)
CAGR     = (cum.iloc[-1])^(12/n) − 1            ← 기하평균 (복리, 2026-05-07 수정)
Vol      = std(ret) × √12
MDD      = min((cum − cum_max)/cum_max)
β        = Cov(excess, mkt_exc) / Var(mkt_exc)
α        = (mean(excess) − β·mean(mkt_exc)) × 12   ← Jensen, 단순 ×12 연환산
```

> ⚠️ **β/α 정합성 (2026-05-07 수정)**: `panel['spy_ret']`은 backward return(과거)이라 forward portfolio와 1-month 시차 발생. `build_master_table`이 각 pkl의 `spy_ret`(walk_forward 저장본, forward-aligned)을 우선 사용하도록 수정. capm_no_bl β ≈ 1.0이 sanity check 통과 기준.

---

## 5. 실험 카테고리 (총 90개)

### 5-1. 매트릭스 (LSTM 고정, 4-token mat_*) — 90개

```
mat_{prior}_{p_weight}_{q}_{omega}
```

| 차원 | 값 |
|---|---|
| prior | mcap / eq / rp |
| p_weight | mcap / eq / rp |
| q_mode | fix / lam / raw / inv / vsp |
| omega | he / pap |

3 × 3 × 5 × 2 = **90 cells** (모두 실현)

### 5-2. 민감도 분석 — in-code (05b_Analyze.ipynb)

Q/PCT_GROUP sweep 은 **pkl 영구 저장 없이** 05b_Analyze.ipynb 내부에서 winner 슬롯에 대해 동적으로 `bl_runner.walk_forward()` 호출하여 진행. (02b 임계값 민감도 패턴과 동일.)

- §M: Q sweep (winner 슬롯의 q_value 변경)
- §N: PCT_GROUP sweep (저/고 그룹 컷오프 변경)

---

## 6. LSTM 실험 전제조건

`p_mode='lstm_predicted'` 실험은 `data/03b_lstm/data/ensemble_predictions_stockwise.csv` 필요. 없으면 자동 스킵 (경고 메시지).

파일 구조:
```
date        ticker  y_pred_lstm  y_pred_har  y_pred_ensemble  y_true  ...
2007-04-23  AAPL    -4.12        -4.36       -4.24            -3.99
```

`y_pred_ensemble`(log-daily-RV) → `np.exp() × √252` → **연환산** σ_pred → 월말 기준 종목 랭킹. `√252` 누락 시 `vol_21d`(연환산)와 단위 혼합 → P 랭킹 왜곡 (cell-04에 sanity guard 있음).

---

## 7. 결과 파일 구조 (`results/{name}.pkl`)

| 키 | 타입 | 내용 |
|---|---|---|
| `config` | dict | 실험 cfg 전체 |
| `ret` | pd.Series | 월별 **순수익률** (`net_ret = gross - turnover×tc`) |
| `gross_ret` | pd.Series | 월별 총수익률 (TC 차감 전) |
| `spy_ret` | pd.Series | 월별 SPY 수익률 |
| `weights` | pd.DataFrame | 월별 × 종목 가중치 |
| `comp` | pd.DataFrame | 월별 구성지표 (eff_n, top10_share, avg_vol, **turnover**, tc_cost 등) |
| `meta` | pd.DataFrame | 월별 Q값, λ |
| `errors` | list | 에러 발생 월 |

> **TC 반영**: `ret` Series는 이미 TC 차감 후. mt/rt의 모든 sharpe/sortino/cagr는 net-of-TC.

---

## 8. 재실행 방법

```bash
# 특정 실험만 재실행
rm final_pt/results/{name}.pkl
# 04_BL_Walkforward.ipynb 실행 (해당 cfg만 walk_forward)
```

전체 재실행: cell-05의 `SKIP_IF_EXISTS = True`를 `False`로 변경.

---

## 9. 주의사항

| 항목 | 내용 |
|---|---|
| **Look-ahead bias** | `fwd_ret_1m`은 평가 전용. BL 입력에 절대 사용 금지 |
| **LSTM 학습** | Phase3에서 walk-forward 사전 생성. 재학습 시 GPU + 수 시간 |
| **rp 가중방식** | Pyo & Lee 2018 — 30% 선별 후 그룹 내 1/σ 가중 |
| **거래비용 단위** | `tc=0.002` = 편측(per-side) 20bp. turnover는 two-way Σ\|Δw\|∈[0,2]이므로 월 TC = `turnover × tc`가 매수+매도 비용 모두 반영 |
| **데이터 선행** | `01_DataCollection.ipynb` 실행 후 `data/` 채워진 상태에서 실행 |
| **monthly_cache** | `04_BL_Walkforward` cell-cache에서 빌드 — 모든 슬롯 공유 (Σ/mcap/spy 등) |
| **vol_pred 단위** | `np.exp(y_pred_ensemble) × √252` 로 연환산. 일별/연환산 혼합 시 P 랭킹 왜곡 (bl_runner `get_vol_series` 에 guard) |

---

## 10. 분석 흐름 (다음 단계)

`04_BL_Walkforward` 완료 → `05b_Analyze` 실행 → 슬롯 효과 + 매트릭스 히트맵 + 3-레짐 안정성 → 위험성향별 최종 후보.

상세: `05b_Analyze.ipynb` 안 markdown 셀 참고.
