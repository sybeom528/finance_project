# 윤서_WORKLOG.md

> **작성자**: 김윤서 (myun)
> **규칙**: 설계 결정·파일 변경 시 날짜·내용·판단 근거를 누적 기록한다.

---

## 2026-04-24

### 설계 결정: 타깃을 누적 21일 수익률로 확정

**결정**: `target[t] = log(P[t+21] / P[t])` = 21일 누적 log-return (단순 t+21 하루 수익률 아님)

**두 방식 비교**:

| | 단순 방식 | **누적 방식 (채택)** |
|---|---|---|
| 공식 | `log_ret[t+21]` | `sum(log_ret[t+1:t+22])` |
| 의미 | t+21일 **하루** 수익률 | t → t+21 **21일 전체** 수익률 |
| SPY 예시 (2016-01-11 기준) | `-0.000863` | `-0.036254` = `log(156.60/162.38)` |

**채택 근거**:
- Black-Litterman Q는 *다음 리밸런싱 기간의 기대수익률* → 21일 전체가 필요
- 단순 방식은 미래 특정 하루 하나만 예측 → 노이즈 극대, 리밸런싱 관점에서 무의미
- 누적 방식 = `log(P[t+21]/P[t])` → 실제 보유 수익률과 동일한 정의

**시각화**: `results/cumulative_return_viz.png`

---

### 신규: `scripts/targets.py`

- `build_daily_target_21d(adj_close)` — `log_ret.rolling(21).sum().shift(-21)`. NaN: 첫 1행 + 마지막 21행
- `verify_no_leakage(log_ret, target)` — assert + 육안 표 2단계 검증
- `build_leaky_target_for_test(adj_close)` — 인공 누수 타깃 (§4 대조 실험 전용)

---

### 수정: `scripts/dataset.py` — `build_fold_datasets`

**추가**: `target_series: Optional[np.ndarray] = None`

- 제공 시: `y_tr[j] = target_series[train_idx[j + seq_len - 1]]`, `y_te[k] = target_series[test_idx[k]]`
- `None` 기본값으로 기존 동작 유지 (backward compatible)

**이유**: Setting A 타깃(21일 누적 수익률)은 입력 피처(log_return)와 다른 시계열이므로 외부 주입 방식 채택.

---

### 신규: `02_setting_A_daily21.ipynb`

**확정 파라미터**:

| 항목 | 값 |
|---|---|
| SEQ_LEN | 126 (약 6개월) |
| IS / purge / emb / OOS / step | 231 / 21 / 21 / 21 / 21 |
| 폴드 수 | 106 |
| 훈련 시퀀스/폴드 | 105 (= IS 231 − seq_len 126) |
| 테스트 시퀀스/폴드 | 21 |

**완료**: §1~§6 (환경·데이터·타깃·누수검증·Dataset·Walk-Forward)  
**미완료**: §7~§10 — `scripts/models.py`, `train.py`, `metrics.py`, `plot_utils.py` 필요

---

## 2026-04-25

### 수정: `scripts/models.py` — forget gate bias 옵션화

**변경**: `forget_gate_bias_init=None` (기본값, PyTorch 기본 0 유지). 이전: 항상 1.0 강제 적용.

**근거**: 일별 log-return ACF lag 2+ ≈ 0 → 장기 의존성 부재. `b_f=1` 적용 시 노이즈를 더 멀리 전파해 분산 약 2배 폭증 확인 (재천_WORKLOG.md 2026-04-25).

**사용법**: Setting B(월별) 또는 다변량 입력 도입 후 재실험할 때만 `forget_gate_bias_init=1.0` 명시.

---

### 수정: `02_setting_A_daily21.ipynb` — 1차 Run 결과 기반 전면 개선

#### §4 누수 검증 축소 (3종 → 2종)

- 제거: "인공 누수 대조" sanity check
- 유지: assert 단위 테스트 + 육안 확인 표
- 근거: `scripts/*.py` 단위 테스트로 파이프라인 정상성 이미 확인됨 → 중복 제거

#### §5 SEQ_LEN 단축

| 항목 | 이전 | 이후 |
|---|---|---|
| SEQ_LEN | 126 (6개월) | **63 (3개월)** |
| 훈련 시퀀스/폴드 | 84 (train 80%) | **134** (+60%) |
| val 시퀀스/폴드 | 21 | **33** |

근거: SPY ACF lag 1 = -0.13, lag 2+ ≈ 0 → 126일 중 104일은 예측 기여 없이 샘플 수만 감소.

#### §6 fold 생성 NaN 버그 수정

`build_daily_target_21d` 결과 마지막 21행 NaN → fold 생성 기준을 유효 타깃 수로 변경.

```python
# 이전
folds = walk_forward_folds(n_samples=2514, ...)   # NaN 포함

# 이후
n_samples = min(targets_dict['SPY'].notna().sum(), targets_dict['QQQ'].notna().sum())  # 2493
folds = walk_forward_folds(n_samples, ...)
```

fold 수: 106 → **105**

#### §7 모델 파라미터 축소

| 항목 | 이전 | 이후 |
|---|---|---|
| hidden_size | 128 | **32** |
| num_layers | 2 | **1** |
| dropout | 0.2 | **0.3** |
| 파라미터 수 | 199,297 | **4,513** (1/44) |

#### §8 학습 구성 개선 + 하이퍼파라미터 변수화

- weight_decay: 1e-4 → **1e-3**
- EarlyStop patience: 10 → **5**
- max_epochs: 50 → **30**
- `lr_patience=3` run_all_folds에 새로 노출 (기존 train_one_fold 내부에만 있던 것)
- train/val 예측 수집 추가 (`y_true_train`, `y_pred_train`, `y_true_val`, `y_pred_val`) — §9.F 과적합 진단용
- **하이퍼파라미터 상수 블록** 추가 (`HIDDEN`, `NUM_LAYERS` 등) — 호출부·metrics.json 기록 자동 동기화

#### §9 과적합 진단 시각화 6종 추가

| 섹션 | 내용 | 저장 파일 |
|---|---|---|
| §9.A | 학습곡선 갤러리 (선택 fold 5개) | `learning_curve_gallery.png` |
| §9.B | best_epoch 분포 히스토그램 | `best_epoch_histogram.png` |
| §9.C | 예측 분포 sanity (pred_std/true_std) | `prediction_distribution.png` |
| §9.D | 잔차 시계열 + 부호 혼동행렬 | `residual_and_confusion.png` |
| §9.E | fold별 R²_OOS / Hit Rate 박스플롯 | `metric_boxplots.png` |
| §9.F | Train/Val/Test 동일지표 비교 | `overfit_gap.png` |

---

### 2차·3차 Run 결과 (2026-04-25)

**2차 Run** (위 개선 적용, extra_features 미적용):

| 지표 | SPY | QQQ | 관문 | 판정 |
|---|---|---|---|---|
| hit_rate | 0.6313 ± 0.3350 | 0.5968 ± 0.3352 | > 0.55 | PASS |
| R²_OOS | **-0.2118 ± 1.1812** | **-0.5472 ± 1.9130** | > 0 | ❌ FAIL |

- previous baseline hit_rate: SPY 0.9265 / QQQ 0.9252 — LSTM보다 +30%p
- LSTM MAE가 train_mean baseline MAE보다 큼 → 역사적 평균 예측보다 못함
- 1차 Run 대비 소폭 악화 — 모델 축소로 과적합 억제됐으나 정보 병목(1채널) 미해결

**3차 Run** (hyperparams dict 변수화 버그 수정 후 재실행):
- 수치 동일 (seed 고정, 학습 코드 변경 없음)
- metrics.json hyperparams 기록이 실제 값과 일치하도록 수정

---

### 신규 문서: `논의사항/`

| 파일 | 내용 |
|---|---|
| `2026-04-25_결과분석2.md` | 개선사항 반영 현황 점검 + 2차 Run 결과 분석 |
| `2026-04-25_결과분석3.md` | 3차 Run 결과 + hyperparams 버그 수정 확인 + 4차 Run(피처 확장) 계획 |

---

### 설계 결정: 피처 확장 — 최종 피처 세트 확정 (4차 Run 준비)

#### 왜 피처 확장이 필요한가

3차 Run까지 LSTM이 train_mean baseline조차 이기지 못한 근본 원인:

- **입력 1채널(일별 log_return)** → 모델이 21일 누적 수익률의 강한 자기상관을 직접 관찰 불가
- previous baseline R²_OOS = 0.89 — 이 수치는 인접 타깃 윈도우가 20일 겹치는 구조적 특성에서 나오는 것
- LSTM이 이 신호를 일별 수익률에서 스스로 추출하도록 강제하는 것이 과도한 부담

---

#### 초안 피처 검토 및 기각 이유

초안으로 제안됐던 [Y_trailing, **mean_21**, std_21] 세트를 비판적으로 검토.

**mean_21 기각 — Y_trailing과 수학적으로 동일**

```
mean_21[t] = rolling(21).mean()[t]
           = sum(log_ret[t-20:t+1]) / 21
           = Y_trailing[t] / 21
```

상수 배율(1/21) 차이뿐이므로, StandardScaler 정규화 후:

```
scaled_mean_21 = (mean_21 - E[mean_21]) / std(mean_21)
               = (Y_trailing/21 - E[Y_trailing]/21) / (std(Y_trailing)/21)
               = (Y_trailing - E[Y_trailing]) / std(Y_trailing)
               = scaled_Y_trailing
```

→ 입력 채널 2개를 사용하면서 정보는 1개. LSTM 입력에 동일한 숫자가 두 번 들어가는 꼴이므로 기각.

---

#### 최종 확정 피처 세트 (input_size = 4)

| 채널 | 피처 | 공식 | 의미 | 누수 |
|---|---|---|---|---|
| 0 | log_ret | — | 일별 수익률. 시계열 패턴의 raw 입력 | — |
| 1 | Y_trailing | `rolling(21).sum()` | **중기(21일) 누적 수익률.** 타깃과 동일 시간 스케일로, previous baseline이 0.92를 찍는 자기상관 신호를 모델이 직접 관찰하게 해준다. 이것 하나가 가장 핵심. | 없음 ✅ |
| 2 | std_21 | `rolling(21).std()` | **중기 변동성 레짐.** 변동성은 클러스터링하는 경향(ARCH 효과) — 최근 21일 변동이 클수록 다음 21일도 클 가능성이 높다. Y_trailing과 독립 정보. | 없음 ✅ |
| 3 | Y_trailing_5 | `rolling(5).sum()` | **단기(5일, 1주) 누적 수익률.** Y_trailing(21일)과 다른 시간 스케일이므로 StandardScaler 후에도 독립 정보 유지. 단기 모멘텀과 중기 모멘텀이 같은 방향인지 다른 방향인지 LSTM이 포착 가능. | 없음 ✅ |

**mean_21 대신 Y_trailing_5를 채택한 결정 근거**:
- Y_trailing_5는 Y_trailing과 다른 시간 스케일(5일 vs 21일) → 정규화 후에도 독립
- 단기·중기 모멘텀의 방향 일치/불일치 자체가 추가 신호
- 구현이 동일하게 단순 (`rolling(5).sum()`)

---

#### 피처 세트 2차 검토 — 21일 미래 수익률 예측 관점

코드 수정 전 "이 세 피처가 정말 최선인가"를 추가로 검토.

**타깃 구조 재확인**

```
target[t]    = sum(log_ret[t+1 : t+22])  ← 예측 대상 (미래)
예측 시점 t에서 관찰 가능: log_ret[0 ... t] 까지만
```

**previous baseline 0.92의 실체**

previous baseline = `target[t-1]` = `sum(log_ret[t:t+21])` 을 복사하는 전략.
`target[t-1]`과 `target[t]`는 20일이 겹치기 때문에 hit_rate 0.92가 나오는 것이지, 모델이 무언가를 학습해서가 아니다. 그리고 `target[t-1]` 안에는 `log_ret[t+1:t+21]` — **미래 20일치** 가 포함되어 있어, LSTM이 합법적 입력으로는 이 신호를 완전히 복제할 방법이 없다.
→ **피처 설계 목표를 "0.92 복제"가 아닌 "합법적 입력으로 최대한 관련 신호 제공"으로 재정의.**

**각 피처의 21일 예측 관점 타당성**

| 피처 | 타당성 근거 | 한계 |
|---|---|---|
| Y_trailing | 타깃과 동일 시간 스케일의 가장 관련 높은 합법적 신호. 모멘텀 가설 기반. | target[t]와 겹치는 term 없음 — overlap 자기상관을 직접 복제 불가 |
| std_21 | ARCH 효과로 변동성 클러스터링 포착. Y_trailing과 독립 정보(방향 vs 크기). 조건부 예측력 제공 | 방향 예측보다 크기 예측에 유리 |
| Y_trailing_5 | Y_trailing과 다른 시간 스케일(5일 vs 21일) → 독립. 단기/중기 모멘텀 정렬 여부 포착 | Y_trailing 21일 중 최근 5일이 포함되어 완전 독립은 아님 (부분 overlap) |

**대안 검토 후 Y_trailing_5 유지 결론**

- `z_21 = Y_trailing / (21 × std_21)` — 변동성 정규화 모멘텀. 의미 있지만 Y_trailing·std_21의 조합으로 LSTM이 학습 중 자체 생성 가능 → 명시적 추가 필요성 낮음
- `rolling(5).std()` — 단기 변동성. std_21과 방향 동일, 시간 스케일만 다름 → 방향 정보 없음
- Y_trailing_5는 단기 **방향성** 을 제공하여 대안들보다 독립 정보량이 높음

→ **세 피처 유지 확정.**

---

#### 수정된 파일·셀

| 위치 | 변경 내용 |
|---|---|
| Cell 20 (§8) 상수 블록 | `INPUT_SIZE = 4` 추가 |
| Cell 20 (§8) `run_all_folds` 시그니처 | `extra_features=None`, `input_size=1` 파라미터 추가 |
| Cell 20 (§8) 함수 내부 | `build_fold_datasets` 호출 시 `extra_features` 전달, `LSTMRegressor`의 `input_size` 변수화 |
| Cell 20 (§8) 호출부 | Y_trailing, std_21, Y_trailing_5 계산 후 `extra_features=_extra, input_size=INPUT_SIZE` 전달 |
| Cell 22 (§9) hyperparams | `'input_size': INPUT_SIZE` 추가 |
| Cell 8 (§4) | extra_features 누수 검증 assert 추가 (§8 실행 후 재실행 방식) |

---

### 4차 Run 결과 (2026-04-25)

**실행 조건**: input_size=4 (log_ret + Y_trailing + std_21 + Y_trailing_5), hidden=32, 그 외 동일

| 지표 | SPY | QQQ | 3차 대비 |
|---|---|---|---|
| hit_rate | 0.5510 ± 0.3207 | 0.5596 ± 0.3417 | ↓ 악화 |
| R²_OOS | **-2.1486 ± 6.8904** | **-1.3394 ± 3.3379** | ↓↓ 대폭 악화 |
| R²_OOS min | -48.81 | -20.08 | — |
| MAE (global) | 0.0493 | 0.0574 | ↓ 악화 |
| pred_std/true_std | 1.1929 | 0.9565 | (과분산) |
| R²>0 folds | 40% (42/105) | 43% (45/105) | — |
| 관문 hit_rate>0.55 | ⚠️ BORDERLINE | ✅ | — |
| 관문 R²_OOS>0 | ❌ | ❌ | — |

**모든 지표 악화. 3차 Run(input=1)이 현재까지 최선.**

#### 원인 분석

**핵심 발견: Y_trailing ≠ previous baseline이 사용하는 신호**

```
타깃:            Y[t]   = sum(log_ret[t+1 : t+22])
previous 예측:   Y[t-1] = sum(log_ret[t   : t+21])  ← Y[t]와 20일 겹침
Y_trailing[t]:           = sum(log_ret[t-20: t+1])   ← Y[t]와 0일 겹침
```

- previous baseline의 hit_rate 0.92는 `log_ret[t+1:t+21]` (미래 20일)을 포함하는 Y[t-1]을 사용하는 구조적 아티팩트 → **합법적 피처로 재현 불가**
- Y_trailing이 실제 포착하는 것: 21일 모멘텀(trailing → next). 대형 ETF에서 실증적으로 약한 신호
- 피처 추가 시 오히려 fold-specific 패턴 암기(소규모 훈련 세트 과적합)로 성능 대폭 악화
- fold 12: pred_mean=+21.8% vs true_mean=-0.1% → 모델이 Y_trailing을 타깃 대리 신호로 착각해 역방향 편향 발생

#### 현재 상태 요약

| 시도 | 결과 |
|---|---|
| 1차: hidden=128, seq=126 | hit_rate 0.64, R²=-0.16 |
| 2차: hidden=32, seq=63 (구조 최적화) | hit_rate 0.63, R²=-0.21 (소폭 악화) |
| 3차: 기록 정확도만 수정 | 수치 동일 |
| **4차: 피처 확장 (Y_trailing 등)** | **hit_rate 0.55, R²=-2.15 (대폭 악화)** |

→ 구조 최적화·피처 확장 모두 R²_OOS > 0 관문 통과 실패.

#### 다음 단계 선택지

| 선택지 | 설명 |
|---|---|
| A) input=1로 복원 | 3차 Run 상태로 롤백 (hit_rate 0.63, R²=-0.21) |
| B) 피처 교체 | Y_trailing 제거, vol_ratio/VIX 등 체제 신호 시도 |
| C) Phase 1 종료 선언 | hit_rate > 0.55는 달성. Phase 2에서 GRU/Transformer 시도 |

**참조**: `논의사항/2026-04-25_결과분석4.md`

---

### 4차 변경 롤백 — input=1 복원 (2026-04-25)

4차 Run 결과 분석 후 피처 확장이 오히려 성능을 악화시킨다는 결론에 따라 **3차 Run 상태로 롤백**.

| 위치 | 제거 내용 |
|---|---|
| Cell 20 상수 블록 | `INPUT_SIZE = 4` 제거 |
| Cell 20 `run_all_folds` 시그니처 | `extra_features=None`, `input_size=1` 파라미터 제거 |
| Cell 20 `build_fold_datasets` 호출 | `extra_features=extra_features` 제거 |
| Cell 20 `LSTMRegressor` 생성 | `input_size=input_size` → `input_size=1` 고정 |
| Cell 20 호출부 | Y_trailing/std_21/Y_trailing_5 계산 블록 및 `extra_features=_extra, input_size=INPUT_SIZE` 제거 |
| Cell 22 hyperparams | `'input_size': INPUT_SIZE` 제거 |

Cell 8 extra_features 누수 검증 코드: `_extra_spy` 미정의 시 NameError → SKIP 처리되므로 무해. 유지.

**복원 후 상태**: input_size=1 (log_return 단일 채널), hidden=32, 나머지 하이퍼파라미터 동일 — 3차 Run과 완전히 동일한 구성.

---

### 롤백 후 Run All 재실행 확인 (2026-04-25)

롤백 후 Run All 재실행 → metrics.json 수치가 3차 Run 기록과 완전히 일치함을 확인.

| 지표 | 3차 Run 기록 | 재실행 | 일치 |
|---|---|---|---|
| SPY hit_rate | 0.6313 ± 0.3350 | 0.6313 ± 0.3350 | ✅ |
| SPY R²_OOS | -0.2118 ± 1.1812 | -0.2118 ± 1.1812 | ✅ |
| QQQ hit_rate | 0.5968 ± 0.3352 | 0.5968 ± 0.3352 | ✅ |
| QQQ R²_OOS | -0.5472 ± 1.9130 | -0.5472 ± 1.9130 | ✅ |
| n_folds | 105 | 105 | ✅ |

seed=42 고정이 재현성을 보장. **현재 노트북 상태 = 3차 Run.**

---

### 설계 결정: 다음 피처 전략 — 체제(regime) 신호로 전환 (2026-04-25)

#### 4차 Run 교훈 — 피처 선택 원칙 수정

| 기각 원칙 | 이유 |
|---|---|
| 자산 자체의 과거 수익률 계열 (Y_trailing, Y_trailing_5) | 21일 모멘텀은 대형 ETF에서 약하고, 체제에 따라 방향이 바뀌어 오히려 fold-specific 편향 유발 |
| 수익률에서 직접 파생된 변동성 (std_21) | 방향 예측 기여 없이 입력 복잡도만 증가 |

새 원칙:
- **자산 외부의 독립적 체제 신호** → log_return과 낮은 상관, 진짜 새 정보
- **이미 전처리된 데이터 우선** → 추가 데이터 수집 없이 `long_panel.parquet` 내 매크로 피처 활용

#### 5차 Run 후보 피처

| 우선순위 | 피처 | 출처 | 이유 |
|---|---|---|---|
| ★★★ | **VIX 수준** | `long_panel.parquet` | 시장 체제 직접 표현. 고VIX → 조건부 기대수익률·불확실성 변화. 가장 이론 근거 강함 |
| ★★☆ | **HY 스프레드 변화** | `long_panel.parquet` | 신용 위험 확대 → 주식 하락 선행. 21일 스케일 타당 |
| ★★☆ | **수익률 곡선 기울기** | `long_panel.parquet` | 장단기 역전 → 경기침체 선행 신호 |

**5차 Run 계획**: VIX 단독 추가 (input_size=2). 한 번에 하나씩 추가해 효과 분리.
VIX NaN 처리: 0 대체 아닌 평균 대체 (VIX=0은 물리적으로 불가능).

**참조**: `논의사항/2026-04-25_결과분석4.md` §7

---

### 5차 Run 준비 — VIX 피처 추가 코드 수정 (2026-04-25)

**VIX 데이터**: `김윤서/data/external_prices.csv` → `^VIX` 컬럼, 2016-01-04~2025-12-30
- SPY 인덱스 기준 정렬 후 NaN 1개 (2025-12-31) → ffill 처리

| 위치 | 변경 내용 |
|---|---|
| Cell 20 상수 블록 | `INPUT_SIZE = 2` 추가 (`# ch0: log_ret, ch1: VIX`) |
| Cell 20 `run_all_folds` 시그니처 | `extra_features=None`, `input_size=1` 파라미터 복원 |
| Cell 20 `build_fold_datasets` 호출 | `extra_features=extra_features` 복원 |
| Cell 20 `LSTMRegressor` 생성 | `input_size=input_size` 변수 참조 복원 |
| Cell 20 루프 직전 | `_ext_path`로 external_prices.csv 로드, `_vix_raw` 추출 |
| Cell 20 루프 내부 | 각 ticker 인덱스 기준 VIX 정렬(`reindex + ffill`) → `_extra = _vix[:, np.newaxis]` |
| Cell 20 `run_all_folds` 호출 | `extra_features=_extra, input_size=INPUT_SIZE` 전달 |
| Cell 22 hyperparams | `'input_size': INPUT_SIZE` 추가 |
| Cell 8 | VIX 누수 검증으로 교체 (`_vix_check_aligned[t] == _extra_spy[t, 0]`) |

**입력 구성**: `input_size=2` (ch0=log_ret, ch1=VIX), hidden=32 유지

---

### 버그 수정: Cell 8 AssertionError + Cell 20 pd 임포트 (2026-04-25)

#### 발생한 오류

5차 Run 코드 추가 후 Run All 시 Cell 8에서 아래 오류 발생:

```
AssertionError: VIX 계산 불일치
```

#### 근본 원인 — 커널 잔존 변수 (stale kernel state)

Run All은 커널을 재시작하지 않는다. 4차 Run에서 Cell 20이 실행되며 `_extra_spy`를 Y_trailing 배열로 설정했고, 이 값이 커널 메모리에 남아 있었다.

```
4차 Run _extra_spy[100, 0] = Y_trailing[100] ≈ 0.01   (일별 누적 수익률)
5차 Run Cell 8 비교 대상   = VIX[100]        ≈ 13.43  (공포지수)
→ abs(13.43 - 0.01) > 1e-6 → AssertionError
```

Cell 8이 Cell 20보다 먼저 실행되므로, 새 피처로 갱신되기 전의 `_extra_spy`로 검증을 시도한 것.

#### 수정 내용

**Cell 8** — `try/except NameError` → 3단계 분기로 교체

| 상황 | 이전 동작 | 수정 후 동작 |
|---|---|---|
| `_extra_spy` 미정의 | NameError → SKIP | `not in vars()` → `[SKIP]` |
| `_extra_spy` 있고 값이 다름 (stale) | AssertionError 전파 → **오류** | `except AssertionError` → `[STALE]` 출력 |
| `_extra_spy` 있고 값이 일치 | `[OK]` | `[OK]` |

```python
if '_extra_spy' not in vars():
    print('[SKIP] ...')
else:
    try:
        assert abs(_vix_check_aligned[_t] - _extra_spy[_t, 0]) < 1e-6
        print('[OK] ...')
    except AssertionError:
        print('[STALE] _extra_spy 가 현재 Run 의 VIX 값이 아닙니다 ...')
    except Exception as e:
        print(f'[SKIP] 검증 오류: {e}')
```

**Cell 20** — VIX 로드 블록 상단에 `import pandas as pd` 명시 추가
- `00_setup_and_utils.ipynb` `%run` 으로 pd가 네임스페이스에 로드되지만, 커널 재시작 직후나 셀 단독 실행 시 pd가 없을 수 있어 명시적 임포트 추가

#### 운영 지침

피처 구성이 바뀐 후 Run All 전에는 반드시 **Kernel → Restart Kernel and Run All Cells** 로 실행할 것. 커널 재시작 없이 Run All 하면 이전 세션의 변수가 잔존해 검증 셀이 stale 상태를 잘못 통과하거나 오류를 낼 수 있다.

---

### 5차 Run 결과 (2026-04-25)

**실행 조건**: input_size=2 (log_ret + VIX), hidden=32, 그 외 동일. Restart Kernel and Run All Cells 로 실행.

| 지표 | SPY | QQQ | 관문 | 판정 |
|---|---|---|---|---|
| hit_rate | 0.5338 ± 0.3473 | 0.5406 ± 0.3530 | > 0.55 | ❌ FAIL |
| R²_OOS | **-1.1297 ± 3.2043** | **-1.2398 ± 3.6704** | > 0 | ❌ FAIL |
| R²_OOS min | -26.24 | -31.51 | — | — |
| MAE | 0.0461 | 0.0583 | — | — |
| R²>0 폴드 | 10/105 (9.5%) | 1/105 (0.95%) | — | — |
| R²<-1 폴드 | 72/105 (68.6%) | 71/105 (67.6%) | — | — |

**3차 Run 대비 전면 악화 (VIX 추가 후 오히려 나빠짐)**:

| 지표 | 3차 Run | 5차 Run | 변화 |
|---|---|---|---|
| SPY hit_rate | 0.6313 | 0.5338 | -0.0975 ↓ |
| SPY R²_OOS | -0.2118 | -1.1297 | ↓ (5배 악화) |
| QQQ hit_rate | 0.5968 | 0.5406 | -0.0562 ↓ |
| QQQ R²_OOS | -0.5472 | -1.2398 | ↓ (2배 악화) |

#### 병리적 폴드

fold 14에서 두 자산 동시 붕괴:
- SPY: R²=-131.10, pred_mean=-11.0% vs true_mean=+2.3% (방향 반전)
- QQQ: R²=-250.01, pred_mean=-22.6% vs true_mean=+4.5% (방향 반전)

#### 원인 분석

| 요인 | 설명 |
|---|---|
| VIX-수익률 관계 비선형·비정상성 | 고VIX → 하락이 일반적이나 고VIX 이후 반등도 빈번. 21일 선행에 일관된 방향 신호 아님 |
| 적은 훈련 샘플 (134개/fold) | VIX 레짐별 수익률 방향이 달라지는데 134샘플로는 안정적 학습 불가 |
| 조기 종료 과도 | best_epoch mean ≈ 5.18 — 5에포크만에 중단, 학습 불충분 |
| fold 14 집중 붕괴 | 해당 구간 특유의 VIX 패턴에 소규모 훈련 세트가 과적합 |

**5차 Run까지 시도 결과 누적 요약**:

| 시도 | SPY hit_rate | SPY R²_OOS | 결론 |
|---|---|---|---|
| 1차: hidden=128, seq=126 | 0.6442 | -0.1552 | — |
| 2차/3차: hidden=32, seq=63 | 0.6313 | -0.2118 | 현재 best |
| 4차: + Y_trailing 등 (모멘텀) | 0.5510 | -2.1486 | 대폭 악화 |
| 5차: + VIX (체제 신호) | 0.5338 | -1.1297 | 대폭 악화 |

→ **외부 피처 추가 시 일관되게 악화. 3차 Run(input=1)이 5차 Run 종료 시점 최선.**

공통 원인: **훈련 샘플 수 절대 부족 (134개/fold)**. 이 조건에서 추가 피처는 fold-specific 과적합만 심화.

**참조**: `논의사항/2026-04-25_결과분석5.md`

---

### Phase 1 방향 결정 필요 (2026-04-25)

5번의 Run을 거쳐 다음 중 하나를 선택해야 함:

| 선택지 | 설명 | 장단점 |
|---|---|---|
| A) **3차 Run 결과 수용** | hit_rate=0.63, R²=-0.21을 Phase 1 최종 출력으로 확정 후 Phase 2 진행 | 즉시 진행 가능. R²<0이지만 방향성 신호는 존재 |
| B) **Walk-Forward IS 확대** | IS=231→504로 늘려 n_train=441로 증가 | 근본 원인(샘플 부족) 해결 가능. 폴드 수 감소 |
| C) **모델 교체** | GRU(파라미터 25% 감소) 또는 Ridge 단순화 | 용량 감소로 과적합 완화. 코드 구조 변경 필요 |
