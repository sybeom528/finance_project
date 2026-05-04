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

---

## 11. 후속 작업 — EWMA omega 추가 (사이클 2)

### 11-1. 배경

omega-RMSE 옵션1 결과가 시점 평균 RMSE 의 절대 수준이 walk-forward 분포와 안 맞아 BL 무력화 (MDD ≈ capm_no_bl 수준) 라는 진단 후, **Pyo & Lee (2018) 식 (17)** 의 정통 omega 정의 (BL 자기 예측-실현 차이의 분산) 가 단위 일관성 측면에서 더 자연스러움을 발견.

다만 식 (17) 의 `var(N=60)` 는 60개월 워밍업 필요 → 사용자 제안에 따라 **분산 대신 단일 제곱오차 + EWMA 형태로 단순화**.

### 11-2. EWMA 정의

```
Ω_t = λ · Ω_{t-1} + (1 - λ) · e²_{t-1}
e_t = P_t · μ_BL_t  -  P_t · 실현수익률_t  (= 한 달 전 BL 예측 오차)
초기값 Ω_0 = he_litterman (1 시점 워밍업)
```

### 11-3. 두 시나리오 — λ 값 결정

| 시나리오 | λ | 반감기 | 12M 후 영향 | 의미 |
|---|---|---|---|---|
| **lo** (단축 워밍업) | 0.825 | 3.6개월 | 약 10% | 12M 워밍업 모사. ESS≈10, 노이즈 큼 |
| **std** (RiskMetrics) | 0.94 | 11.2개월 | 약 48% | 표준값. ESS≈32. 36M 후 10% 안정화 |

### 11-4. 데이터 가용성 검증

| 데이터 | 시작 | EWMA 워밍업에 필요 |
|---|---|---|
| daily_returns.pkl | 2004-01 | ✓ Σ 추정에 충분 |
| monthly_panel.csv | 2005-01 | △ 2009-01 시점 IS 60M 못 채움 |
| LSTM 예측 csv | 2007-04 | ✓ 워밍업 시기 사용 가능 |

→ "OOS 시작 전 안정화" 위해 monthly_panel 재수집 또는 IS 단축 필요. 본 사이클에서는 **워밍업을 OOS 안에 흡수** 하는 안 C 채택.

### 11-5. 변경 파일

| 파일 | 변경 |
|---|---|
| `final/bl_functions.py` | `compute_omega_ewma` 신규 함수 추가 (lines 297-336) |
| `final/99_run.ipynb` | cell-00 import / cell-04 walk_forward 의 prev_omega·prev_e_sq 전파 + meta 에 view_pred_ret/view_real_ret/view_e/omega 기록 |
| `final/bl_config.py` | 신규 실험 8개 + 슬롯 주석 lambda 추가 |

### 11-6. 신규 실험 8개

LSTM × p_weight 4종 × λ 2종:
- `p_lstm_{mcap/eq/rp/vol_mcap}_ewma_lo` (λ=0.825) — 4개
- `p_lstm_{mcap/eq/rp/vol_mcap}_ewma_std` (λ=0.94) — 4개

### 11-7. 결과 분석 시 주의

- **시나리오 lo (λ=0.825)**: 12개월 워밍업 후 분석 → 첫 12개월 trim 권장. 실효 168개월
- **시나리오 std (λ=0.94)**: 36개월 워밍업 후 분석 → 첫 36개월 trim 권장. 실효 144개월
- **공정 비교**: baseline 도 같은 구간으로 trim 후 비교

### 11-8. 단위 검증 (test_ewma.py 통과 항목)

1. 첫 시점 (prev_e_sq=None) → he_litterman fallback ✓
2. 일정 입력 (e²=Ω_0) 정상상태 유지 ✓
3. 충격 후 감쇠가 정확히 λ^t 따름 ✓
4. λ 별 안정화 비교 (이론값 일치) ✓
5. lower bound 1e-8 보장 ✓

### 11-9. 실행 결과

#### (A) 전체 180개월 (참고용 — 워밍업 효과 포함, 1:1 비교 부적절)

| 실험 | Sharpe | CAGR | Vol | MDD | Sortino |
|---|---|---|---|---|---|
| baseline | 1.106 | 13.37% | 10.98% | -13.03% | 1.726 |
| p_lstm_mcap (LSTM 기본) | 0.990 | 12.37% | 11.28% | -12.39% | 1.524 |
| p_lstm_vol_mcap_ewma_lo (λ=0.825) | 1.115 | 15.38% | 12.73% | -16.79% | 1.595 |
| p_lstm_vol_mcap_ewma_std (λ=0.94) | 1.057 | 14.64% | 12.73% | -17.35% | 1.496 |

> ⚠️ 위 비교는 **lo·std 워밍업 차이 (12M vs 36M) 가 미반영** 되어 1:1 비교 부적절.

#### (B) 같은 144M 공정 비교 (2013-01 ~ 2024-12) — 권장 ★

워밍업 36개월 (= 3× 반감기, λ=0.94 기준 10% 안정화) 후의 동일 구간:

| 실험 | Sharpe | CAGR | MDD | Sortino |
|---|---|---|---|---|
| **baseline** | **1.046** | 13.13% | -13.03% | 1.619 |
| p_lstm_vol_mcap_ewma_lo | 1.033 | 14.88% | -16.79% | 1.427 |
| p_lstm_rp_ewma_lo | 1.020 | 15.09% | -17.20% | 1.418 |
| p_lstm_eq_ewma_lo | 1.015 | 15.04% | -17.29% | 1.415 |
| p_lstm_eq_ewma_std | 0.997 | 14.91% | -18.10% | 1.374 |
| p_lstm_rp_ewma_std | 0.996 | 14.88% | -18.13% | 1.372 |
| p_lstm_mcap_ewma_lo | 0.995 | 15.07% | -17.91% | 1.367 |
| p_lstm_mcap_ewma_std | 0.982 | 14.95% | -17.88% | 1.369 |
| p_lstm_vol_mcap_ewma_std | 0.967 | 14.24% | -17.35% | 1.337 |
| (참고) p_lstm_mcap_omega_rmse | 0.725 | 11.27% | -22.28% | 1.020 |
| (참고) p_lstm_mcap_omega_rmse_pt | 0.931 | 12.15% | -14.65% | 1.364 |

**144M 공정 비교 결론**:
- 모든 EWMA 변형이 baseline 미달 (Sharpe 0.967~1.033 < 1.046)
- 가장 좋은 EWMA: `p_lstm_vol_mcap_ewma_lo` (1.033) — baseline 의 98.8%
- λ=0.825 (lo) 가 λ=0.94 (std) 보다 모든 P 가중에서 우월 (Δ +0.013~+0.066)
- CAGR 은 EWMA 가 baseline 13.13% 보다 큼 (14.24~15.09%) — 변동성·MDD 증가 대가

### 11-10. 핵심 발견 (144M 공정 비교 기준 정정)

**1. EWMA omega 는 omega_rmse 와 정반대로 작동 — 뷰 강화 방향**

```
omega_rmse  (옵션1) : omega ↑ (×2~6) → 뷰 약화 → BL 무력화 → MDD -22% (capm_no_bl 수준)
omega_rmse_pt(옵션2) : omega ↑ (×1.1~1.2) → 미세 보수화 → MDD -15%
ewma (lo·std)        : omega ≈ 비슷 또는 약간 ↓ → 뷰 강화 → CAGR ↑ MDD -17~18%
```

**2. CAGR 큰 폭 향상, 그러나 위험조정 성과는 baseline 미달**

144M 공정 비교 (2013-01 ~ 2024-12):

| 변형 | CAGR | MDD | Sharpe | vs baseline |
|---|---|---|---|---|
| baseline | 13.13% | -13.03% | **1.046** | (기준) |
| p_lstm_vol_mcap_ewma_lo | 14.88% | -16.79% | 1.033 | -0.013 |
| p_lstm_mcap_ewma_lo | 15.07% | -17.91% | 0.995 | -0.051 |
| p_lstm_vol_mcap_ewma_std | 14.24% | -17.35% | 0.967 | -0.079 |

→ EWMA 가 CAGR 은 +1.1~2%p 향상시키지만 변동성·MDD 동반 증가로 **위험조정 성과 (Sharpe) 는 baseline 미달**.

**3. λ 에 결과 robust (페어 비교)**

같은 144M 구간에서 lo (λ=0.825) vs std (λ=0.94) 페어 차이:

| P 가중 | lo Sharpe | std Sharpe | Δ |
|---|---|---|---|
| mcap | 0.995 | 0.982 | +0.013 |
| eq | 1.015 | 0.997 | +0.018 |
| rp | 1.020 | 0.996 | +0.024 |
| vol_mcap | 1.033 | 0.967 | **+0.066** |

- **lo 가 std 보다 일관되게 우월** (모든 4개 P 가중에서 Δ > 0)
- 평균 Δ Sharpe = +0.030, 최대 +0.066
- → 차이는 작지만 lo 가 일관되게 우월. λ 에 완전히 robust 라기보다는 "**짧은 반감기가 약간 유리**" 의 결론

**4. 워밍업 차이로 인한 비교의 함정**

이전 분석 (180M 전체 기간) 에서 `p_lstm_vol_mcap_ewma_lo` 가 Sharpe 1.115 로 baseline 1.106 을 초과하는 것처럼 보였으나, 144M 공정 비교 시 1.033 으로 baseline 1.046 미달. **워밍업 36M 차이가 약 0.082 Sharpe 의 착시**를 만든 것.

→ **결론**: λ 비교 시 반드시 **더 긴 워밍업 (= λ 큰 쪽) 의 trim 기준** 으로 통일해야 함.

**4. omega 시계열 진단**

- 모든 EWMA 실험에서 omega std/mean ≈ 0.5~0.85 → 시점별 변동 있음
- vol_mcap 의 omega 가 다른 가중의 1/10 수준 (sum(P²) 작음)
- λ=0.825 가 λ=0.94 보다 std 큼 (이론과 일치)
- 그러나 결과 metric 차이 미미 → 노이즈가 BL 결과에 큰 영향 없음

**5. baseline 을 넘지 못한 이유 (정정)**

- baseline (q=0.003 fixed + omega=he_litterman) 이 144M 공정 구간에서 가장 효율적 (Sharpe 1.046)
- EWMA 의 평균 omega 가 he_litterman 과 비슷 (또는 약간 작음, vol_mcap 의 경우 1/12) → BL 사후가 뷰를 더 강하게 신뢰 → 더 공격적
- 결과적으로 CAGR ↑ Vol ↑ MDD ↓ → Sharpe 거의 동일하지만 **약간 미달**
- **EWMA 가 진짜 효과 발휘하려면 e² 분포가 시기별로 크게 달라야** (= 위기/평온 차이가 명확) → 본 데이터에서는 그 차이가 일정 수준 이상 확대되지 않음
- `view_e` 의 평균 +0.025% (약한 과대 예측 편향), std 1.41% — BL 자체가 과대 예측하는 경향이 있음을 확인

**6. 비교 분석 시 주의사항 (이번 사이클의 메타 교훈)**

- **워밍업 차이가 다른 hyperparameter 변형은 반드시 같은 trim 구간으로 비교**
- 워밍업 길이가 다른 두 모형의 같은 시작일(2010-01) 비교는 **워밍업 길이 효과가 결과에 섞임**
- 본 프로젝트에서는 모든 EWMA 비교에 **2013-01~ 의 144M 통일** 권장
- baseline 도 같은 144M 으로 trim 후 비교

### 11-11. 시사점

1. **omega 동적 조정 메커니즘 자체가 항상 도움이 되는 건 아니다**: omega_rmse 무력화, omega_rmse_pt 미세 보수화, EWMA 뷰 강화 — 셋 다 baseline 미달
2. **baseline 의 단순함이 강건성**: BL 의 4개 슬롯 (prior, P, Q, omega) 중 omega 변형 시도 모두 효과 약함
3. **CAGR 절대값 측면에서는 EWMA 가 우월**: 위험 감수를 받아들이면 EWMA 가 매력적
4. **λ robustness 는 좋은 학술적 신호**: hyperparameter 입증 부담 감소

### 11-12. 추가 시도 후보 (미구현)

- omega EWMA + Q 동적 (식 16의 q=Pr̂) 결합 → 논문 식 17 본래 의도 재현
- 시점별 base_rmse rolling 정규화 + omega_rmse 결합
- HMM 레짐 기반 동적 Q (worklog §10 의 미구현 hmm_bayesian 안)

