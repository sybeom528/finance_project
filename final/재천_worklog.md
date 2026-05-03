# 재천 worklog — Omega RMSE + Risk Parity Prior 실험 추가

> **브랜치**: `feature/omega-rmse-rp-prior`
> **작업 일자**: 2026-05-03
> **목적**: LSTM+HAR 앙상블의 walk-forward 예측 RMSE 를 Black-Litterman 의 Omega 산정에 반영하고, Risk Parity prior 변형을 추가해 BL 모델의 동적 신뢰도 조정과 prior 다양성을 검증.

---

## 1. 작업 배경 (Context)

`final/` 의 BL 실험 프레임워크는 Phase3 의 LSTM+HAR 앙상블 변동성 예측을 P 행렬(`p_mode='lstm_predicted'`)에는 활용하지만, 그 예측의 **정확도 자체는 BL 모델에 반영되지 않고** 있었다.

또한 prior 슬롯이 `capm_mcap` / `capm_eq` 두 가지뿐이라 위험 분산 관점의 자연스러운 비교군인 **Risk Parity (`w_i ∝ 1/σ_i`)** 가 빠져 있었다.

### 코드 사전 점검 결과 (탐색 단계)

| 발견 | 의미 |
|---|---|
| `bl_functions.py:237-254` 에 `compute_omega_rmse` 함수 이미 존재 | 신규 함수가 아닌 기존 함수 활용 가능 |
| `99_run.ipynb` cell-03 dispatcher 에 `omega_mode='rmse'` 분기 이미 존재 | 신규 라우팅이 아닌 기존 분기 활용 가능 |
| `bl_config.py` 에 `omega_mode='rmse'` 실험 0개 | 한 번도 실행된 적 없음 → 실험 dict 추가 필요 |
| dispatcher 가 `recent['abs_err'].mean()` (= **MAE**) 를 RMSE 로 잘못 넘기고 있음 | 함수명·인자명과 호출이 어긋남, 정정 필요 |
| `compute_omega_rmse_per_ticker` 함수 부재 | 종목별 가중 RMSE 활용 불가 |
| `bl_functions.py` 에 RP prior 함수 부재 | 새 함수 + dispatcher 분기 필요 |

### LSTM 학습 데이터 동일성 확인

사용자가 "시계열_Test/Phase3_Robust_Extensions" 의 작업 결과와 final 의 LSTM 데이터가 상이한지 먼저 확인하라고 지시.

| 위치 | 상태 |
|---|---|
| `final/phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` | **현재 시스템에 없음** (outputs/ 폴더만 존재) |
| `시계열_Test/Phase3_Robust_Extensions/data/ensemble_predictions_stockwise.csv` | 존재 (331MB, 2,468,883행, 613종목, 2007-04-23 ~ 2025-12-01) |
| `final/99_run.ipynb` 의 이전 cell-02 출력 LSTM 행수 | 2,468,883행 (정확히 일치) |

→ **두 위치는 동일 파일**. final 폴더의 사본만 사라진 상태이므로 시계열_Test 원본을 final 위치로 복사하면 즉시 사용 가능. LSTM 재학습 불필요.

추가 관찰:
- `w_v4 = w_har = 0.5` 고정 → 단순 평균 앙상블
- `y_pred_ensemble` 는 log(realized variance) 스케일
- `y_true` 일부 -inf 존재 (log(0) 결과) → BL 모델은 `y_pred_ensemble` 만 사용하므로 무영향

---

## 2. 작업 원칙 (사용자 지침 반영)

1. **브랜치 분리 작업**: main 에서 새 작업 브랜치 분기. main 보존.
2. **추가 위주 구현**: 신규 함수·신규 dispatcher 분기·신규 실험 dict 로 기능을 더한다.
3. **기존 코드 수정·삭제는 사전 보고 후 허가**: 본 worklog 의 §5 에 모든 수정 항목과 영향을 분리해 명시.
4. **기존 17개 실험 결과 보존**: `SKIP_IF_EXISTS=True` 기본값에서 재실행되지 않음.

---

## 3. 의사결정 과정

### 3-1. RMSE 산정 단위 결정 (사용자 결정 1)

후보 4가지를 사용자에게 제시:

| 옵션 | 데이터 단위 | scale 변동 요인 | 표본 수 |
|---|---|---|---|
| 1. 시점 평균 RMSE | 모든 종목·일자 한 덩어리 | 시점에만 의존 | 수십만 |
| 2. 종목별 RMSE → 뷰 가중 결합 | 종목별 | 시점 + 뷰 구성 + P 가중 | 종목당 ~250 |
| 3. 뷰 포함 종목만 평균 | 뷰 종목·일자 | 시점 + 뷰 종목구성 | 수만 |
| 4. **1 + 2 동시 비교** | — | — | — |

**사용자 결정: 옵션 4 (시점 평균 + 종목별 가중 모두 비교)**
→ 8개 LSTM 가중 × 2 omega 모드 = **16개 신규 omega 실험**.

### 3-2. Risk Parity prior 도입 (사용자 결정 2~5)

사용자가 RP prior 도입을 추가 요청. 다음 결정점 정리:

**LSTM 파일 경로 처리**:
- 결정: **시계열_Test 의 csv 를 final/phase3(data_outputs)/data/ 로 복사** (코드 변경 0)

**RP 변동성 소스**:
- 후보: `vol_21d`, `vol_252d`, LSTM 예측 변동성, 또는 3종 모두
- 결정: **vol_21d 단일** (P 행렬 기본 변동성과 일관)

**RP 구현 방식 + 실험 조합**:
- A안: `get_prior_weights` 시그니처 확장 (vol 인자 추가) — 기존 코드 수정
- B안: `get_prior_weights_rp` 신규 함수 + walk_forward 분기 1줄 추가 — 추가 위주
- 결정: **B안 + capm_rp × p_weight 4종 = 4개 신규 RP 실험**

→ 신규 실험 총 **20개** (omega 16 + RP 4).

### 3-3. 옵션2 omega 정의 (사용자 결정 6, 단위 검증 후 발견)

단위 검증 중 옵션2 정의의 부작용 발견:

```
현재 정의: pred_rmse_view = sqrt( Σ P_i² × RMSE_i² )
```

**문제**: 균등 RMSE = base 인 경우에도 `scale = Σ P_i²` 가 되어 P 행렬 자체의 크기가 omega scale 에 영향. `p_lstm_mcap_omega_rmse_pt` vs `p_lstm_eq_omega_rmse_pt` 결과 차이가 (a) RMSE 영향인지 (b) P 크기 영향인지 분리 어려움.

**대안 — 정규화 정의**:
```
pred_rmse_view = sqrt( Σ P_i² × RMSE_i² / Σ P_i² )
              = "RMSE_i 의 P²-가중 평균의 sqrt"
```
- 균등 RMSE → scale = 1 → he_litterman 와 정확히 동일
- P 가중 방식 차이는 omega scale 에 무영향
- 종목별 RMSE 분포 차이만 omega 에 반영

**사용자 결정: 정규화 정의로 변경**.

---

## 4. 실행 단계

### Step 0. 브랜치 생성

```bash
git checkout -b feature/omega-rmse-rp-prior
```

### Step 1. LSTM 예측 파일 복사

```powershell
mkdir "final/phase3(data_outputs)/data"
Copy-Item `
  "시계열_Test/Phase3_Robust_Extensions/data/ensemble_predictions_stockwise.csv" `
  "final/phase3(data_outputs)/data/ensemble_predictions_stockwise.csv"
```
- 331,142,770 bytes (315.8 MB) 정상 복사
- `.gitignore` 에 이미 `*.csv` 가 있어 자동 git 제외

### Step 2. 추가 작업 (수정 없음)

#### 2-1. `final/bl_functions.py` — 신규 함수 1개 추가

기존 `compute_omega_rmse` (lines 237-254) 는 손대지 않음. 그 아래에 `compute_omega_rmse_per_ticker` 추가 (lines 257-294).

**최종 정의 (사용자 결정 6 반영, 정규화 정의)**:
```python
def compute_omega_rmse_per_ticker(P, Sigma, tau, rmse_per_ticker, base_rmse=0.39265):
    p_aligned    = P.reindex(rmse_per_ticker.index.union(P.index)).fillna(0)
    rmse_aligned = rmse_per_ticker.reindex(p_aligned.index).fillna(base_rmse)
    p_sq_sum     = float((p_aligned ** 2).sum())
    if p_sq_sum <= 0:
        return compute_omega_he(P, Sigma, tau)
    weighted_sq    = float(((p_aligned ** 2) * (rmse_aligned ** 2)).sum())
    pred_rmse_view = float(np.sqrt(weighted_sq / p_sq_sum))   # ← 정규화 분모
    scale = (pred_rmse_view / base_rmse) ** 2 if base_rmse > 0 else 1.0
    return compute_omega_scaled(P, Sigma, tau, scale)
```

#### 2-2. `final/bl_config.py` — 실험 20개 추가 + 슬롯 주석 업데이트

기존 EXPERIMENTS 17개 끝에 다음 추가 (기존 dict 는 손대지 않음):
- omega_rmse 8개: `p_lstm_{mcap/eq/rp/vol_mcap}_omega_rmse` + `prior_eq_p_lstm_*_omega_rmse`
- omega_rmse_per_ticker 8개: 위와 같은 패턴 + `_pt` 접미사
- RP prior 4개: `prior_rp_p_{mcap/eq/rp/vol_mcap}`

→ 총 17 + 20 = **37개**. 슬롯 주석에 `'rmse_per_ticker'` 와 `'capm_rp'` 추가 (선택 항목, 문서화 일관성).

#### 2-3. `final/99_run.ipynb` cell-03 — 신규 함수 + 신규 elif 분기

`get_prior_weights` 직후에 `get_prior_weights_rp` 신규 함수 추가 (Python 스크립트 `_dev/_apply_notebook_changes.py` 로 일괄 패치).

```python
def get_prior_weights_rp(valid_tix, month_df):
    vol = month_df['vol_21d'].reindex(valid_tix).replace(0, np.nan).dropna()
    inv_vol = 1.0 / vol
    w = inv_vol / inv_vol.sum()
    return w.reindex(valid_tix).fillna(0)
```

`get_omega` 의 `else` 직전에 `'rmse_per_ticker'` elif 추가 (12개월 walk-forward 윈도우, 종목별 RMSE 계산).

### Step 3. 기존 코드 수정 (사용자 사전 보고 + 승인 후)

플랜 §4 에 모든 수정 항목과 영향을 사전 보고. 사용자 승인 후 일괄 진행.

#### 3-1. `cell-00` import 라인

```python
# Before
compute_omega_he, compute_omega_scaled, compute_omega_rmse,
# After
compute_omega_he, compute_omega_scaled,
compute_omega_rmse, compute_omega_rmse_per_ticker,
```

영향: 기존 import 그대로 유지. 신규 함수만 추가. 위험 0.

#### 3-2. `cell-03` 의 기존 `'rmse'` 분기 sqrt 보정

```python
# Before
pred_rmse = float(recent['abs_err'].mean()) if len(recent) > 0 else 0.39
# After
pred_rmse = float(np.sqrt((recent['abs_err'] ** 2).mean())) if len(recent) > 0 else 0.39265
```

원인: 함수명·인자명이 RMSE 인데 MAE 를 넘기고 있었음.
영향: **기존 결과 영향 없음** (기존 17개 실험 중 `omega_mode='rmse'` 사용한 실험 0개). 옵션1 신규 8개 실험에서만 효과.

#### 3-3. `cell-04` walk_forward 의 prior 결정 분기 추가

```python
# Before
w_mkt = get_prior_weights(cfg, valid_tix, mcap)
# After
if cfg.get('prior') == 'capm_rp':
    w_mkt = get_prior_weights_rp(valid_tix, month_df)
else:
    w_mkt = get_prior_weights(cfg, valid_tix, mcap)
```

영향: 기존 prior(capm_mcap, capm_eq) 동작 변화 없음. 신규 prior=='capm_rp' 만 신규 함수로 라우팅.

#### 3-4. `cell-05` 스킵 조건 두 곳 확장

```python
# Before
omega_mode == 'rmse'
# After
omega_mode in ('rmse', 'rmse_per_ticker')
```

영향: 기존 `'rmse'` 모드 스킵 동작 그대로. 신규 mode 추가.

### Step 4. 단위 검증

`final/_dev/test_new_features.py` 임시 스크립트 작성 → 검증 후 삭제 예정.

검증 항목 모두 통과:
1. **옵션1 (시점 평균)**:
   - pred=base → omega = he_litterman ✓
   - pred=2*base → omega = 4×he ✓
   - pred=0.5*base → omega = 0.25×he ✓
2. **옵션2 (종목별 가중, 정규화 정의)**:
   - 모든 종목 RMSE=base → omega = he (P 가중 방식과 무관) ✓
   - P=0 종목 RMSE 변경 → 결과 무영향 ✓
   - 모든 종목 RMSE=2*base → omega = 4×he ✓
   - 한 종목만 RMSE=2*base → 균등 < 결과 < 모두 2배 ✓
   - mcap-like P / eq-like P 모두 균등 RMSE → he 동일 ✓
3. **RP prior**:
   - vol={A:0.10, B:0.20, C:0.40} → w={A:0.5714, B:0.2857, C:0.1429} ✓
   - 균등 vol → 1/N 가중 ✓

---

## 5. 발견 사항 / 시행착오

### 5-1. Windows 콘솔 인코딩 (cp949) 이슈

검증 스크립트의 ✓ 문자가 cp949 로 인코딩 안 됨 → `sys.stdout.reconfigure(encoding='utf-8')` 추가로 해결.

### 5-2. 옵션2 정의의 P 크기 의존성 (3-3 에서 상세)

초기 정의 `sqrt(Σ P_i² × RMSE_i²)` 가 P 행렬 크기에 의존. 정규화 분모 `/ Σ P_i²` 추가로 해결.

### 5-3. final/phase3(data_outputs)/data/ 폴더 부재

이전 실행 환경(`c:\workspace\camp\project\finance_project`) 과 현재 환경(`C:\Users\gorhk\최종 프로젝트\finance_project`) 이 달라 LSTM 데이터 사본이 사라진 상태. 시계열_Test 의 원본을 복사로 해결.

---

## 6. 최종 결과물

### 변경된 파일

| 파일 | 변경 유형 | 라인 수 |
|---|---|---|
| `final/bl_functions.py` | 신규 함수 추가 | +38 / -0 |
| `final/bl_config.py` | 실험 20개 추가 + 슬롯 주석 | +75 / -2 |
| `final/99_run.ipynb` | 신규 함수/분기 + 기존 코드 수정 | +36 / -5 |
| `final/phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` | 시계열_Test 에서 복사 | (gitignore 처리) |

### 신규 임시 파일 (검증용, 작업 후 삭제 예정)

| 파일 | 용도 |
|---|---|
| `final/_dev/_apply_notebook_changes.py` | 노트북 일괄 패치 스크립트 |
| `final/_dev/test_new_features.py` | 단위 검증 스크립트 |

### 신규 실험 20개 (`bl_config.py` EXPERIMENTS 확장)

**Omega-RMSE 옵션1 (시점 평균)** 8개:
- `p_lstm_{mcap/eq/rp/vol_mcap}_omega_rmse` (4)
- `prior_eq_p_lstm_{mcap/eq/rp/vol_mcap}_omega_rmse` (4)

**Omega-RMSE 옵션2 (종목별 가중, 정규화 정의)** 8개:
- `p_lstm_{mcap/eq/rp/vol_mcap}_omega_rmse_pt` (4)
- `prior_eq_p_lstm_{mcap/eq/rp/vol_mcap}_omega_rmse_pt` (4)

**Risk Parity prior** 4개:
- `prior_rp_p_{mcap/eq/rp/vol_mcap}` (4)

---

## 7. 향후 작업 (다음 단계)

1. **노트북 실행 검증**: `99_run.ipynb` 셀 순서대로 실행 → 신규 20개 실험이 SKIP 없이 실행되는지 확인 → `results/` 에 신규 pkl 20개 생성 확인.
2. **효과 분석** (`99_analyze.ipynb`):
   - `p_lstm_mcap` vs `p_lstm_mcap_omega_rmse`: 시점 평균 RMSE 효과
   - `p_lstm_mcap` vs `p_lstm_mcap_omega_rmse_pt`: 종목별 가중 RMSE 효과
   - `*_omega_rmse` vs `*_omega_rmse_pt`: 두 RMSE 산정 방식 비교
   - `baseline` (capm_mcap) vs `prior_rp_p_mcap`: RP prior 효과
   - `prior_eq` vs `prior_rp_p_mcap`: RP prior vs 1/N prior
   - `prior_rp_p_*` 4종 간 비교: RP prior 하의 P 가중 효과
3. **임시 파일 정리**: `_dev/_apply_notebook_changes.py`, `_dev/test_new_features.py` 삭제 (단위 검증 통과 후).
4. **머지 결정**: 사용자 검토 후 main 머지 여부 결정.

---

## 8. Look-ahead Bias 점검

| 항목 | 상태 |
|---|---|
| RMSE 윈도우 | `cutoff = pred_date - 12개월` ~ `pred_date` ✅ |
| `lstm_preds` y_pred_ensemble | Phase3 walk-forward 생성 ✅ |
| `base_rmse=0.39265` | Phase3 in-sample 평가 결과 ⚠️ → 정규화 상수로만 작용해 결과 방향에 영향 없음. 발표 시 한계로 명시 |
| RP prior `vol_21d` | pred_date 이전 21일 ✅ |

---

## 9. 한계 및 노트

- **base_rmse=0.39265 의 in-sample 성격**: Phase3 `eval_metrics_stockwise.json` 의 layer1_overall.rmse 를 그대로 사용. 단, 정규화 상수로만 작용하므로 (모든 실험에 동일하게 적용) 실험 간 상대 비교에는 무영향.
- **base_rmse=0.39265 (옵션2 fallback)**: 종목별 RMSE 시리즈에 결측인 신규 종목은 base_rmse 로 fallback. P 에 들어간 종목 중 결측이 많으면 omega 가 base_rmse 쪽으로 끌림.
- **옵션2 정규화 정의 선택 근거**: P 가중 방식 효과와 RMSE 산정 효과를 분리 비교하기 위함. 대안(현재 정의 = `sqrt(Σ P²·RMSE²)`)은 실험에서 제외했으나, 후속 작업에서 별도 ablation 으로 확인 가능.
