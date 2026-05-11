# Black-Litterman 실험 프레임워크 — 상세 가이드

> **최종 갱신: 2026-05-07**
> - omega=scaled 신뢰성 부족으로 제거 (-10 cfg)
> - Q 민감도: 11 후보 × 4 sweep → baseline 단일 후보 × 4 sweep으로 단순화
> - CAGR 정의 산술평균 → 기하평균(복리) 수정
> - β/α 시차 misalignment 버그 수정 (각 pkl의 spy_ret 사용)
> - vol_mcap sparse 변형 12개 추가 제거 (trailing+lstm 각 1개만 보존)
> - 현재 EXPERIMENTS 총 **156개** (매트릭스 135 + 비매트릭스 21), results pkl 156개 완성

---

## 1. 전체 구조

```
final_pt/
├── bl_config.py              ← EXPERIMENTS 정의 (214개)
├── bl_functions.py           ← BL 수식 핵심 함수 (Σ/π/P/Q/Ω/BL/TC/Metrics)
├── master_table.py           ← results/*.pkl → mt/rt 빌더
├── analyze_plots.py          ← 시각화 모듈
│
├── 04_BL_Walkforward.ipynb              ← walk_forward 실행 → results/*.pkl
├── 99_analyze.ipynb          ← 분석 단일 진입점 (K_CUT → I → J → K → L → M → N)
├── 99_slot_effects.ipynb     ← 슬롯 차원 효과 라인플롯 (pivot CSV 자동 생성)
│
├── results/                  ← 214 pkl
├── data/                     ← monthly_panel.csv, daily_returns.pkl, ff3_monthly.csv
└── phase3(data_outputs)/     ← LSTM 예측 결과
    └── data/ensemble_predictions_stockwise.csv
```

---

## 2. 실험 실행 순서

```
① bl_config.py에 실험 dict 추가 (필요 시)
② 04_BL_Walkforward.ipynb 셀 순서대로 실행
   cell-00 : 패키지 임포트 + 경로 설정
   cell-01 : 데이터 로드 (monthly_panel, daily_returns, FF3)
   cell-02 : LSTM 예측 로드 + monthly_cache 빌드 (Σ 등 사전계산)
   cell-03 : Dispatcher 함수 정의 (get_vol_series, get_Q, get_omega 등)
   cell-04 : walk_forward(cfg) 함수 정의
   cell-05 : run_list = 미생성 실험만 → 자동 스킵
   cell-06 : 빠른 성과 확인
③ 99_analyze.ipynb 실행 (Setup → K_CUT → I → J → K → L → M → N 순서대로)
   ※ K_CUT은 mandatory pre-step. 모든 후속 분석은 2023-12-31 cutoff 기준.
④ (선택) 99_slot_effects.ipynb 실행 — 슬롯 차원 효과 라인플롯
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
{**BASELINE, 'name': 'mat_eq_rp_vsp_pap', 'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
 'p_weight': 'rp', 'q_mode': 'vol_spread', 'q_value': 0.003, 'omega_mode': 'ff3_paper'},
```

`{**BASELINE, ...}` 패턴: BASELINE 모든 값 상속 + 변경 슬롯만 덮어씀.

### Case 2 — 새 계산 방식 추가

3개 파일을 **이 순서대로** 수정:
1. `bl_functions.py` — 새 함수 추가 (예: `compute_Q_my_method`)
2. `bl_config.py` — 슬롯 주석 + 실험 dict 추가
3. `04_BL_Walkforward.ipynb` cell-03 — dispatcher (`get_Q`, `get_omega` 등)에 elif 추가

---

## 4. 슬롯 키 레퍼런스 + 정확한 수식

> **변경 이력 (2026-05-07)**: `omega_mode='scaled'` 신뢰성 부족으로 제거. `omega_scale` 키도 사용 안 됨.
> CAGR 정의를 산술평균×12 → 기하평균(복리)으로 수정. β/α 시차 misalignment 버그 수정.

### 4-0. 슬롯 키 일람표

| 슬롯 | 선택지 | 기본값 | 설명 |
|---|---|---|---|
| `prior` | `capm_mcap` / `capm_eq` / `capm_rp` | `capm_mcap` | Prior π 가중 |
| `p_mode` | `trailing_vol21` / `trailing_vol252` / `lstm_predicted` | `trailing_vol21` | P 분류용 변동성 |
| `p_weight` | `mcap` / `eq` / `rp` / `vol_mcap` | `mcap` | P 행렬 가중 |
| `q_mode` | `fixed` / `vol_spread` / `lambda` / `raw_lam` / `inv_lambda` / `none` / `capm` | `fixed` | Q 결정 방식 |
| `q_value` | float | `0.003` | Q 값(또는 base scaling) |
| `lam_mean` | float | `2.5` | λ 정규화 기준값 |
| `omega_mode` | `he_litterman` / `rmse` / `ff3_paper` | `he_litterman` | Ω 계산 방식 |
| `tc` | float | `0.001` | 편측(per-side) 거래비용 (10bp). turnover=Σ\|Δw\|∈[0,2]이라 TC = turnover×tc로 매수+매도 동시 반영 |
| `max_weight` | float | `0.10` | 단일 종목 상한 |
| `lstm_pred_path` | str | 자동 탐색 | LSTM 예측 파일 경로 |

> 📌 `omega_scale`은 deprecated (`scaled` 모드 제거됨). 옛 cfg에서 발견되면 무시.

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

### 4-3. P 행렬 — 4종 가중 + 분류

**공통 분류**: `p_mode`로 결정된 변동성 시리즈 σ를 오름차순 정렬 후
- 하위 `pct=0.30` → **저위험 그룹** (long, P > 0)
- 상위 `pct=0.30` → **고위험 그룹** (short, P < 0)

**`p_mode`별 σ 정의** (모두 **연환산** vol 단위로 통일):
| `p_mode` | σ 출처 | 연환산 변환 |
|---|---|---|
| `trailing_vol21` | `vol_21d` (과거 21일 실현) | `01_DataCollection.ipynb`에서 `lr.rolling(21).std() × √252` |
| `trailing_vol252` | `vol_252d` (과거 252일 실현) | 동일하게 `× √252` |
| `lstm_predicted` | LSTM+HAR 앙상블 예측, `exp(y_pred_ensemble) × √252` (Phase3 산출) | `04_BL_Walkforward.ipynb` cell-03에서 `× √252` |

> ⚠️ **단위 일관성 중요**: `lstm_predicted` 모드는 LSTM 예측이 있는 종목만 `vol_21d`를 LSTM 값으로 덮어쓰고, 나머지는 `vol_21d` 그대로 둠. **두 시리즈가 같은 단위(연환산)** 여야 P 랭킹이 올바름. 과거 버그(2026-05-07 이전): LSTM `vol_pred = exp(y_pred)`만 적용해 일별 단위로 계산되어 vol_21d(연환산)와 혼합 → LSTM 커버 종목이 인위적으로 ~16배 작게 보여 P "저변동 30%"가 사실상 "LSTM 커버리지 dummy"로 변질. 이 버그 수정 후 LSTM 슬롯 모든 backtest 재실행 (`results_backup/`은 옛 결과 보관용).
> [04_BL_Walkforward.ipynb](04_BL_Walkforward.ipynb) cell-04 `get_vol_series`에 `pred_slice.median() < 0.05` sanity guard 추가됨.

**`p_weight`별 가중 수식** (σ_i: 자산 i의 변동성, m_i: 시가총액):

| `p_weight` | Long (저위험 그룹 i ∈ Low) | Short (고위험 그룹 i ∈ High) | 유니버스 |
|---|---|---|---|
| `mcap` | `P_i = +m_i / Σ_{j∈Low} m_j` | `P_i = −m_i / Σ_{j∈High} m_j` | 30% 컷 |
| `eq` | `P_i = +1 / n_g` | `P_i = −1 / n_g` (n_g = 그룹 크기) | 30% 컷 |
| `rp` | `P_i = +(1/σ_i) / Σ_{j∈Low}(1/σ_j)` | `P_i = −σ_i / Σ_{j∈High} σ_j` | 30% 컷 |
| `vol_mcap` | `P_i = +(1−r_i)·m_i / Σ` | `P_i = −r_i·m_i / Σ` (r_i = vol percentile rank) | **전체** (30% 컷 없음) |

**P 합 = 0** (자기자금 조달 long-short, market-neutral 형식).

---

### 4-4. Q (single view 기대 spread) — 5종 활성 + 2종 비활성

매 시점 t의 `λ_t = clip(SPY_excess_t / σ²_mkt,t, 0.5, 10.0)` (compute_pi에서 계산).
`q_base = q_value` (기본 0.003), `lam_mean = 2.5`.

| `q_mode` | Q 수식 | 메커니즘 |
|---|---|---|
| `fixed` | `Q = q_value` | 가장 단순. q_value 고정 |
| `vol_spread` (vsp) | `Q = q_base × clip(spread_t / spread_ref, 0.1, 3.0)`<br>`spread_t = mean(σ_pred[P<0]) − mean(σ_pred[P>0])`<br>`spread_ref = train 기간 expanding median` | LSTM 예측 vol 격차로 view 강도 동적 조절 (위기↑→Q↑) |
| `lambda` | `Q = q_base × clip(λ_t / lam_mean, 0.1, 3.0)` | 시장 안정(λ↑) → Q 강화 |
| `raw_lam` | `Q = max(0, q_base × lam_raw / lam_mean)`<br>`lam_raw = SPY_excess / σ²_mkt` (clip 전) | SPY 하락(lam_raw 음수) → Q=0 자연 게이팅 |
| `inv_lambda` | `Q = q_base × clip(lam_mean / λ_t, 0.1, 3.0)` | 위기(λ↓) → Q 강화 (반대 방향) |
| `none` | BL 스킵, 저변동 그룹 직접 보유 | naive_lowvol 비교군 |
| `capm` | BL 없이 CAPM π로 직접 MVO | capm_no_bl 비교군 |

> 📌 **`q_value`는 모든 동적 모드(vsp/lambda/raw_lam/inv_lambda)에서 base로 작용**. 단조 0.003 → 0.0070 sweep해도 절대값 분포만 평행이동(시그널은 동일).

> ❌ **비활성**: `ff3_regression`, `realized_spread`, `regime`은 코드에 있지만 EXPERIMENTS에서 사용 안 함.

---

### 4-5. Ω (view 분산) — 3종

**τ = 0.05 고정** (prior 불확실성 비율).

| `omega_mode` | Ω 수식 | 메커니즘 |
|---|---|---|
| `he_litterman` (he) | `Ω = max(τ · P Σ Pᵀ, 1e-8)` | He-Litterman 1999 표준. view 분산 = prior 분산 비례 |
| `rmse` (rms) | `Ω = he × (RMSE_t / RMSE_base)²`<br>`RMSE_t = median(\|y_pred_lstm − y_true\|)` 직전 12개월<br>`RMSE_base = 0.39` (전역 기준값) | LSTM 예측 정확도 적응. 정확도↓(RMSE↑)→Ω↑→view 신뢰↓ |
| `ff3_paper` (pap) | `Ω_t = max((Q_{t−1} − actual_P_return_{t−1})², 1e-8)` (첫 달은 he fallback) | 직전월 view 예측오차² Bayesian rolling. **walk_forward inline 처리**, dispatcher 거치지 않음 |

> ⚠️ **`ff3_paper` 명명 주의**: 코드의 `compute_omega_paper`(FF3 회귀 잔차분산)는 **dead code**. 실제 동작은 위 수식. 발표 시 "직전월 예측오차² 적응형 (Bayesian rolling)"으로 정확히 표기.

> ❌ **제거됨 (2026-05-07)**: `omega_mode='scaled'` (omega_scale 배수). 신뢰성 부족 사유로 EXPERIMENTS와 dispatcher에서 제외.

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

## 5. 실험 카테고리 (총 156, 2026-05-07 기준)

### 5-1. 매트릭스 (LSTM 고정, 4-token mat_*) — 135개

```
mat_{prior}_{p_weight}_{q}_{omega}
```

| 차원 | 값 |
|---|---|
| prior | mcap / eq / rp |
| p_weight | mcap / eq / rp |
| q_mode | fix / lam / raw / inv / vsp |
| omega | he / pap / rms |

3 × 3 × 5 × 3 = **135 cells** (모두 실현)

### 5-2. 비매트릭스 (semantic naming) — 21개

| 카테고리 | 이름 | 개수 |
|---|---|---:|
| BASELINE | `baseline` | 1 |
| BL 미사용 비교군 | `capm_no_bl`, `naive_lowvol` | 2 |
| Trailing 단일 슬롯 변형 | `prior_eq`, `prior_rp`, `p_eq`, `p_rp`, `p_vol_mcap` | 5 |
| Trailing × Q 동적 | `q_lambda`, `q_inv_lambda`, `q_raw_lam` | 3 |
| Trailing × prior_eq + Q | `prior_eq_q_lambda`, `prior_eq_q_raw_lam` | 2 |
| Trailing × Ω 변형 | `omega_paper`, `omega_rmse`, `q_ff3_paper`, `q_ff3_paper_omega_paper` | 4 |
| LSTM × vol_mcap | `p_lstm_vol_mcap` | 1 |
| Q 민감도 baseline | `baseline_q55`, `baseline_q64`, `baseline_q70` | 3 |

> ❌ **제거됨 (2026-05-07)**:
> - Scaled Ω 변형 10개 (`omega_scaled_*`) — `omega_mode='scaled'` 폐기, 신뢰성 부족
> - vol_mcap sparse 변형 12개 (`p_vol_mcap_q_lambda`, `prior_eq_p_vol_mcap*`, `*_p_lstm` 등) — 체계적 매트릭스(3×4×5×3) 없이 sparse라 비교 효용 약함. `p_vol_mcap`(trailing) + `p_lstm_vol_mcap`(LSTM) 2개만 보존
> - Q 민감도 다중 후보 33개 — canonical 충돌로 K2 대시보드 혼란

### 5-3. Q 민감도 — baseline 단일 후보 (2026-05-07)

기존 11 후보 × 4 q_value 다중 민감도 분석은 canonical 충돌(같은 이름 중복)로 K2 대시보드 혼란 야기 → **제거**.

대신 **baseline 단일 후보**에 q_value sweep:
- `baseline` (q=0.003, BASELINE 자체) — 기존
- `baseline_q55` (q=0.0055)
- `baseline_q64` (q=0.0064)
- `baseline_q70` (q=0.0070)

학술 근거 (Frazzini-Pedersen 2014, BAB 월평균):
- 0.0055: 4팩터 알파 보수치
- 0.0064: 글로벌 19개국 평균
- 0.0070: 미국 평균

**결과 요약** (K7 셀):
- Sharpe / CAGR / sortino_ir / sharpe_ir 모두 **단조 감소** (q ↑ → 성과 ↓)
- Sortino만 q=0.0055에서 미세 개선 (1.741 vs 1.726)
- → **q=0.003이 우리 setup에 calibrate된 robust optimum**. BAB 학술 평균으로 올릴 동인 약함.

---

## 6. LSTM 실험 전제조건

`p_mode='lstm_predicted'` 실험은 `phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` 필요. 없으면 자동 스킵 (경고 메시지).

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
| **vol_mcap** | 전체 유니버스, 30% 컷 없음 |
| **거래비용 단위** | `tc=0.001` = 편측(per-side) 10bp. turnover는 two-way Σ\|Δw\|∈[0,2]이므로 월 TC = `turnover × tc`가 매수+매도 비용 모두 반영 |
| **데이터 선행** | `01_DataCollection.ipynb` 실행 후 `data/` 채워진 상태에서 실행 |
| **monthly_cache** | `04_BL_Walkforward` cell-02에서 빌드 후 캐시 — 재시작 시 자동 재사용 |
| **vol_pred 단위** | `np.exp(y_pred_ensemble) × √252` 로 연환산. 일별/연환산 혼합 시 P 랭킹 왜곡 (cell-04에 guard) |
| **results_backup/** | 2026-05-07 이전 단위 혼합 버그 결과 (LSTM 슬롯 한정) — **분석엔 절대 사용 금지**, 감사용으로만 보관 |

---

## 10. 분석 흐름 (다음 단계)

`04_BL_Walkforward` 완료 → `99_analyze` 실행 → 슬롯 효과 + 매트릭스 히트맵 + 3-레짐 안정성 → 위험성향별 최종 후보.

상세: `99_analyze.ipynb` 안 markdown 셀 참고.
