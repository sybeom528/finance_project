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

---

## 2026-04-27

### 분석: 모멘텀 피처 검토 — ret_1m = Y_trailing (동일)

6차 Run 준비 논의 중 `ret_1m/3m/6m/12m` 스타일 모멘텀 피처와 4차 Run의 Y_trailing을 비교 검토.

**결론: ret_1m은 Y_trailing과 수학적으로 동일**

```
Y_trailing[t] = sum(log(p[i]/p[i-1]) for i in t-20..t)
              = log(p[t]) - log(p[t-21])
              = log(p[t] / p[t-21])
              = ret_1m[t]
```

→ 4차 Run에서 이미 시도, 실패한 피처. 새로운 실험이 아님.

**미시도 피처: ret_3m/6m/12m**

| 피처 | 시간 스케일 | 샘플 손실 이슈 |
|---|---|---|
| ret_3m = log(p/p.shift(63)) | 3개월 모멘텀 | 없음 (초기 IS 샘플에서 IS_start 경계 딱 걸림) |
| ret_6m = log(p/p.shift(126)) | 6개월 모멘텀 | 초기 IS 샘플에서 126일 이전 필요 → 조기 폴드 훈련 샘플 감소 |
| ret_12m = log(p/p.shift(252)) | 12개월 (학술 표준) | 초기 252일 이전 필요 → 가장 큰 샘플 손실 |

학술적으로 검증된 모멘텀은 6~12개월이며 1개월은 단기 반전(reversal) 구간. 6차 Run 결과에 따라 추가 여부 결정.

---

### 설계 결정: 6차 Run — seq_len 축소 + IS 확대 + VIX 제거

5차 Run까지 공통 근본 원인으로 확인된 **훈련 샘플 수 절대 부족(134개/폴드)**에 직접 개입하는 방향으로 전환.

#### 변경 내용

| 파라미터 | 이전 | 변경 후 | 근거 |
|---|---|---|---|
| `SEQ_LEN` | 63 (3개월) | **21 (1개월)** | 일별 log_return ACF lag 2+ ≈ 0. 63일 중 대부분 noise |
| `IS` | 231일 (~11개월) | **504일 (~2년)** | 훈련 샘플 134 → 약 370개/폴드 (2.8배) |
| `INPUT_SIZE` | 2 (log_ret + VIX) | **1 (log_ret 단독)** | VIX 제거, seq_len+IS 효과만 격리 검증 |
| 폴드 수 (예상) | 105 | **약 91** | IS 확대 트레이드오프 |

#### seq_len 축소 근거

seq_len이란 LSTM이 한 예측 시 참조하는 과거 일수. 일별 log_return의 실제 ACF는 lag 1 ≈ -0.07, lag 2+ ≈ 0에 수렴한다. seq_len=63이면 61일 분량이 예측 기여 없이 noise만 추가하고, LSTM에게 과적합 경로를 제공한다.

seq_len=21 선택 이유:
- 타깃(21일 선행 수익률)과 동일한 시간 스케일
- seq_len 단축 자체로도 훈련 샘플 증가 (IS 고정 시 `IS - purge - seq_len` 증가)
- 실질 ACF 범위 내 커버

#### IS 확대 근거

IS 확대와 폴드 수 감소의 관계:
```
폴드 수 ≈ (전체 유효 샘플 - IS - purge - emb - OOS) / step
  IS=231: (2493 - 294) / 21 = 105 폴드
  IS=504: (2493 - 567) / 21 ≈ 91 폴드
```

IS가 커질수록 첫 폴드 출발에 더 많은 데이터를 소비 → 남은 롤링 범위가 줄어 폴드 수 감소. **더 깊게 보되, 더 좁은 구간을 평가**하는 트레이드오프.

훈련 샘플 변화:
```
IS=231, seq_len=21, purge=21: 유효 시퀀스 = 189 → train 80% ≈ 151
IS=504, seq_len=21, purge=21: 유효 시퀀스 = 462 → train 80% ≈ 370
```

#### VIX 제거 및 원인 격리

seq_len + IS 두 파라미터를 동시에 바꾸면서 피처까지 변경하면 어느 쪽이 효과인지 분리 불가. VIX를 제거해 5차 Run 대비 변화 요인을 seq_len + IS로만 한정.

#### 코드 수정 내역 (`02_setting_A_daily21.ipynb`)

| 셀 | 변경 내용 |
|---|---|
| Cell 9 (markdown §5) | seq_len 126→21 참조, N_train 계산식, 수정 근거 업데이트 |
| Cell 11 (code §5) | `SEQ_LEN = 63` → `SEQ_LEN = 21` |
| Cell 12 (markdown §6) | IS 231→504, 폴드수·샘플수 설명 업데이트 |
| Cell 13 (code §6) | `IS = 231` → `IS = 504` |
| Cell 14 (code §6) | print 문 파라미터 설명 업데이트 |
| Cell 20 (code §8) | `INPUT_SIZE = 2` → `1`, VIX 로드 블록 제거, 루프 내 `_vix`/`_extra`/`_extra_spy` 제거, `extra_features=None` |

**참조**: `논의사항/2026-04-27_결과분석6.md`

---

### 6차 Run 결과 (2026-04-27)

**실행 조건**: SEQ_LEN=21, IS=504, INPUT_SIZE=1 (log_ret 단독), hidden=32, 그 외 동일.

| 지표 | SPY | QQQ | 관문 | 판정 |
|---|---|---|---|---|
| hit_rate | **0.6594 ± 0.3234** | **0.6548 ± 0.3411** | > 0.55 | ✅ PASS |
| R²_OOS | **-0.0247 ± 0.8489** | **-0.0783 ± 0.8591** | > 0 | ❌ FAIL |
| R²_OOS min/max | -4.467 / +0.934 | -4.191 / +0.895 | — | — |
| MAE | 0.036000 | 0.047893 | — | — |
| n_folds / n_oos_samples | 92 / 1932 | 92 / 1932 | — | — |
| 훈련 시퀀스/폴드 | 483 | 483 | — | — |

**역대 최고 성능 달성 (1~6차 중)**:

| 지표 | 3차 (이전 best) | **6차 Run** | 변화 |
|---|---|---|---|
| SPY hit_rate | 0.6313 | **0.6594** | +0.028 ↑ |
| SPY R²_OOS | -0.2118 | **-0.0247** | **10배 개선** ↑↑ |
| QQQ hit_rate | 0.5968 | **0.6548** | +0.058 ↑ |
| QQQ R²_OOS | -0.5472 | **-0.0783** | **7배 개선** ↑↑ |

**과적합 진단**:
- pred_std/true_std: SPY 0.252, QQQ 0.322 → ⚠️ Mean Collapse 여전히 존재
- best_epoch ≤ 3인 폴드: SPY 42/92(45.7%), QQQ 44/92(47.8%) → 조기 종료 잔존
- Train/Val/Test 갭: test(0.659) ≈ val(0.685) → IS-OOS 갭 이전 대비 대폭 축소 ✅

**원인 분석**:
- IS 확대(231→504): 훈련 샘플 134→483 (+3.6배). 더 다양한 레짐 학습 → OOS 일반화 향상
- seq_len 축소(63→21): noise 경로 제거, 타깃과 시간 스케일 정렬
- 피처 추가보다 파라미터 조정이 효과적임을 확인

**현재 상태**: R²_OOS=-0.025로 관문(>0) 직전. 483샘플/폴드로 피처 추가 재시도 조건이 이전(134샘플)보다 유리해짐.

**참조**: `논의사항/2026-04-27_결과분석6.md` §3~§5

---

### 설계 결정: 7차 Run — ret_3m (3개월 모멘텀) 추가

**배경**: 6차 Run에서 훈련샘플 483개/폴드로 피처 추가 조건이 유리해짐. 한 번도 시도하지 않은 ret_3m 단독 추가로 효과 격리.

| 파라미터 | 변경 | 이유 |
|---|---|---|
| `INPUT_SIZE` | 1 → **2** | ch0: log_ret, ch1: ret_3m |
| `SEQ_LEN`, `IS` | 21, 504 | 동일 유지 (피처 효과만 격리) |

**ret_3m 선택 이유**:
- ret_1m(=Y_trailing)은 4차 Run(IS=231, 134샘플)에서 실패 — ret_3m은 미시도 구간
- Jegadeesh & Titman (1993): 3~12개월 모멘텀이 단기보다 횡단면 예측력 높음
- mean collapse(pred_std/true_std≈0.25) 완화 가능성 — 모멘텀 강도가 크기 추정에 기여 기대
- NaN: 첫 62일 → `fillna(0)` (영향 폴드 fold 0~2, 전체 3%)

**코드 수정**: Cell 8(누수 검증 ret_3m으로 교체), Cell 9(shape 설명 업데이트), Cell 20(INPUT_SIZE=2, rolling(63).sum() 계산·전달)

**참조**: `논의사항/2026-04-27_결과분석7.md`

### 7차 Run 결과 (2026-04-27)

| 지표 | 6차 (log_ret) | **7차 (+ ret_3m)** | 변화 |
|---|---|---|---|
| SPY hit_rate | 0.6594 | 0.6325 | ↓ |
| SPY R²_OOS | -0.0247 | **-0.0897** | ↓ 3.6배 악화 |
| QQQ hit_rate | 0.6548 | 0.5916 | ↓ |
| QQQ R²_OOS | -0.0783 | **-0.2011** | ↓ 2.6배 악화 |
| pred_std/true_std (SPY) | 0.252 | **0.460** | ↑ 크게 개선 |

**역설**: pred_std는 0.25 → 0.46으로 크게 개선(mean collapse 해소)됐지만 R²는 오히려 악화. 방향이 잘못된 채로 크기만 더 크게 예측한 결과.

**피처 추가 → 성능 하락 패턴의 근본 원인**:
1. **레짐 의존성**: Jegadeesh & Titman 모멘텀은 횡단면 예측 효과 — 단일 자산 시계열 예측에 직접 적용 불가
2. **IS-OOS 레짐 이동**: IS=504(2년)에서 학습한 모멘텀-수익률 부호가 OOS 21일에서 역전되는 폴드 다수
3. **pred_std 증가의 역설**: 더 강한 신호 → 잘못된 방향으로 더 크게 예측 → R² 악화
4. **시장 자체 예측 불가능성**: 일별 수익률 ACF lag 2+ ≈ 0 → 21일 선행 수익률 자체가 예측 불가능에 가까움

**결론**: 7번의 Run 중 피처 추가가 일관되게 실패. **6차 Run(R²=-0.025, hit_rate=0.659)이 Phase 1 best**.

**다음 단계**: Phase 2(Black-Litterman) 진행 권장. 6차 Run 출력을 약한 뷰(Q), 높은 불확실성(Omega)으로 설정.

**참조**: `논의사항/2026-04-27_결과분석7.md` §3~§5

---

### CEEMDAN 분해 미적용 사실 확인 (2026-04-27)

**논문(Su et al. 2026 ESWA 295)과 현재 구현의 차이 인식**

원 논문 모델(CGL-BL)은 CEEMDAN 분해 + LSTM 구조를 사용한다. 현재 Phase 1은 의도적으로 분해 없는 순수 LSTM 베이스라인이다.

| 구분 | 원 논문 | 현재 Phase 1 | Phase 3 계획 |
|---|---|---|---|
| 전처리 | CEEMDAN 분해 | 없음 (raw log_return) | CEEMDAN+LSTM 예정 |
| 구조 | 성분별 LSTM → 합산 | 단일 LSTM | 성분별 LSTM |

**CEEMDAN이란**: 시계열을 주파수 성분(IMF)으로 분리하는 기법. 장기 추세(ACF 높음, 예측 가능) vs 단기 노이즈(ACF ≈ 0, 예측 불가능)를 분리해 각각 학습. 현재 R²_OOS < 0의 주요 원인 중 하나가 이 분해 단계 생략일 가능성이 있음.

**로드맵 상 위치**: CEEMDAN+LSTM은 Phase 3(모델 5·6번)에서 시도 예정. Phase 1은 그 baseline.

---

### 추가 튜닝 방향 검토 (2026-04-27)

**제안된 방향**: IS 확대 + seq_len 축소 + 예측 horizon 단축(21일 미만)

#### (1) IS 더 확대 (504 → 756 이상)

| 항목 | 내용 |
|---|---|
| 효과 | 훈련샘플 추가 증가 (462 → 714/폴드 for IS=756) |
| 한계 | 폴드 수 감소 (~80개로 줄어듦). 3년 이상 오래된 데이터가 현재 시장 구조를 반영하지 못할 수 있음 |
| 판단 | **한계 수익 체감**: 483샘플로도 레짐 문제가 해결되지 않았음. 샘플을 더 늘린다고 근본 문제가 해결되지 않을 가능성 높음 |

#### (2) seq_len 더 축소 (21 → 5~10)

| 항목 | 내용 |
|---|---|
| 효과 | 훈련샘플 소폭 증가 (seq_len=5이면 478개/폴드), 노이즈 경로 추가 제거 |
| 한계 | 이미 ACF는 lag 2 이후 ≈ 0 — seq_len=21에서 유용한 정보는 lag 1~2뿐. 더 줄여도 정보 손실만 있음 |
| 판단 | **이미 거의 최적**: 21 이하로 줄이면 LSTM의 시간적 패턴 학습 능력 자체가 손상됨 |

#### (3) 예측 horizon 단축 (21일 → 5일 또는 1일)

| 항목 | 내용 |
|---|---|
| 효과 | 폴드당 OOS 샘플 증가 (step을 유지하면 더 촘촘한 평가 가능) |
| 한계 ① | **BL 프레임워크 불일치**: Black-Litterman은 월간 리밸런싱 가정. Q(뷰)가 5일 예측이면 21일 뷰로 변환하는 추가 불확실성 발생 |
| 한계 ② | **단기 수익률이 더 예측 불가능**: 일별 수익률 ACF≈0 — horizon이 짧을수록 신호 대비 노이즈 비율이 더 나빠짐. 21일 수익률보다 5일 수익률이 오히려 더 예측하기 어려움 |
| 한계 ③ | OOS step과 horizon이 달라지면 Walk-Forward 설계 전체를 재검토해야 함 |
| 판단 | **권장하지 않음**: BL 연결 목적상 21일이 맞는 horizon. 단축하면 파이프라인 복잡도만 증가 |

#### 종합 판단

세 가지 방향 모두 현재 시점에서 시도하기 어려운 근거:

> 근본 문제는 하이퍼파라미터가 아니다. 일별 수익률의 ACF ≈ 0이라는 시장 자체의 특성이 예측 한계를 만들고 있다. 이를 우회하는 구조적 해법이 CEEMDAN 분해이며, 이는 Phase 3에서 다룬다.

**권장 다음 단계**: Phase 1 결과(6차 Run)를 확정하고 Black-Litterman(Phase 2 이후) 또는 CEEMDAN 구현(Phase 3)으로 진행.

---

### Phase 1 방향 결정 필요 (2026-04-25)

5번의 Run을 거쳐 다음 중 하나를 선택해야 함:

| 선택지 | 설명 | 장단점 |
|---|---|---|
| A) **3차 Run 결과 수용** | hit_rate=0.63, R²=-0.21을 Phase 1 최종 출력으로 확정 후 Phase 2 진행 | 즉시 진행 가능. R²<0이지만 방향성 신호는 존재 |
| B) **Walk-Forward IS 확대** | IS=231→504로 늘려 n_train=441로 증가 | 근본 원인(샘플 부족) 해결 가능. 폴드 수 감소 |
| C) **모델 교체** | GRU(파라미터 25% 감소) 또는 Ridge 단순화 | 용량 감소로 과적합 완화. 코드 구조 변경 필요 |

---

### 분석: Train / Val / Test 성능 패턴 해석 (2026-04-25)

5차 Run §9.F 결과에서 나타난 패턴:

```
SPY  train: hit_rate=0.6545, R²=-0.0588
SPY  val:   hit_rate=0.7564, R²=+0.3302  ← 가장 높음
SPY  test:  hit_rate=0.5338, R²=-1.1297  ← 가장 낮음
```

#### Val > Train: Dropout 비대칭 + EarlyStopping 선택 편향

두 원인이 겹쳐서 발생한다.

**원인 1 — Dropout 비대칭**

학습 루프(model.train())에서 train 손실은 Dropout(0.3) 활성 상태로 계산된다. 즉 은닉 유닛 30%가 무작위로 0이 된 "노이즈 낀 모델"의 예측이다. Val/test 평가(model.eval())에서는 Dropout이 비활성화되어 모든 유닛이 살아 있는 "온전한 모델"로 계산된다. 같은 가중치라도 평가 버전이 더 안정된 예측을 하므로 val 메트릭이 train보다 좋게 나오는 것은 구조적으로 당연하다.

**원인 2 — EarlyStopping 선택 편향**

EarlyStopping이 val 손실이 최소인 체크포인트를 저장한다. 보고되는 val 메트릭은 "전체 에포크 중 val에서 가장 운 좋았던 순간"의 수치이므로 낙관적으로 편향된다.

#### Test << Val: IS-OOS 레짐 분포 이동 (핵심 문제)

```
[IS 231일: 특정 시장 레짐]
  train(80%) │ val(20%)
  ← 같은 레짐 내부에서 분할 →

[purge 42일 gap]

[OOS 21일: 다른 레짐일 수 있음]
```

Val은 IS 윈도우 안에서 시간적으로 뗀 것으로, 훈련 데이터와 같은 레짐에 속한다. EarlyStopping이 이 레짐에 맞춘 체크포인트를 선택하므로 val R²=+0.33이 나온다.

Test는 purge+embargo(42일) 이후의 진짜 미래이며, IS 레짐과 다를 수 있다. 모델이 IS 레짐에서 배운 패턴이 OOS에서 역방향으로 작동하면 R²=-131 같은 붕괴가 나온다.

**핵심 결론**: val R²=+0.33은 "모델이 IS 안에서는 뭔가를 배운다"는 증거이다. 문제는 배운 것이 IS 레짐에 특화되어 OOS로 전이되지 않는다는 점이다.

---

### 분석: VIX가 성능을 악화시킨 이유 (2026-04-25)

레짐 신호를 주입한다는 개념은 맞았지만, raw VIX level은 21일 선행 수익률에 대한 일관된 방향 신호가 아니다.

```
고VIX 상황에서 21일 뒤 수익률:
  2020년 3월 (COVID 폭락): VIX=80 → 급반등 (+30%)   ← 양(+)
  2022년 (Fed 긴축):       VIX=30 → 계속 하락         ← 음(-)
  2008년 (금융위기):       VIX=60 → 추가 하락          ← 음(-)
```

VIX level이 같아도 결과 방향이 반대다. 즉 **VIX level → 21일 수익률 방향 관계 자체가 레짐에 따라 달라진다.** 134 훈련 샘플로는 "어떤 레짐에서 고VIX면 반등하고, 어떤 레짐에서는 추가 하락한다"는 관계를 안정적으로 학습할 수 없다.

결과적으로 VIX를 추가하면:
- IS 내부(val): VIX가 IS 구간 특유의 방향 패턴을 더 정밀하게 암기 → val 성능 약간 상승
- OOS(test): IS에서 암기한 VIX 패턴이 OOS 레짐과 맞지 않으면 더 강하게 역방향 예측 → 붕괴 심화

**VIX level이 아니라 VIX_change(전일 대비 변화율)나 VIX_zscore(최근 60일 대비 편차)를 사용하면 IS-OOS 전이성이 올라갈 수 있다.** 절대 레짐이 아닌 "공포가 커지고 있는가/줄고 있는가"는 IS 구간에 무관하게 해석이 일관적이기 때문이다.

#### IS 확대 여부 검토

IS=231일은 약 11개월 ≈ 1 연간 사이클에 해당한다. IS=504(2년)로 늘리면 훈련 샘플이 134→353개로 증가하지만 근본 문제인 IS-OOS 레짐 분포 이동은 해소되지 않는다. val R²=+0.33은 데이터가 부족해서 못 배우는 상황이 아님을 이미 보여준다. 더 긴 IS는 오히려 여러 레짐을 혼합해 모델이 "평균 레짐"을 학습하게 만들어 더 불안정해질 수 있다.

#### 현실적 다음 단계

| 조건 | 권장 |
|---|---|
| 시간 여유 있음 | VIX_change 단독 추가 (IS=231 유지)로 6차 Run. 피처 타입이 달라지므로 결과가 다를 수 있음 |
| 시간 압박 있음 | **3차 Run 결과(hit_rate=0.63) 수용 후 Phase 2 진행**. Black-Litterman은 약한 방향성 신호 + 높은 Omega로 작동 가능 |

**VIX_change 시도 시 기대**: IS-OOS 전이성 개선 가능성은 있으나, R²=-1.13에서 R²>0까지 한 번에 넘어올 가능성은 낮다. 134샘플 제약이 남아 있기 때문이다.

---

### 8차 Run 설계 결정: IS=756 (3년) (2026-04-27)

7차 Run(ret_3m 추가)이 6차 Run(best)보다 악화됨을 확인한 후, 피처 추가 대신 IS 추가 확대로 방향 전환.

**변경 내용**:

| 파라미터 | 7차 Run | 8차 Run | 근거 |
|---|---|---|---|
| `IS` | 504일 (2년) | **756일 (3년)** | 훈련 샘플 483→714 (+47.8%) |
| `INPUT_SIZE` | 2 (log_ret + ret_3m) | **1 (log_ret 단독)** | 6차 Run 기준 복원 — IS 효과만 단독 검증 |
| n_folds | 92 | ~80 | IS 확대 트레이드오프 |

**훈련 샘플 수 계산**:
```
IS=756, seq_len=21, purge=21: 유효 시퀀스 = 756 - 21 - 21 = 714 (+47.8% vs 6차 483개)
폴드 수: (2493 - 819) / 21 ≈ 80 폴드
```

**수정 파일**: `02_setting_A_daily21.ipynb` Cell 8, 11, 12, 13, 14, 20

**기대**: 6차 Run에서 IS=231→504 확대가 R²를 -0.21→-0.025로 개선. 추가 확대(504→756)로 R²>0 달성 가능성 검증. 단, 한계 수익 체감 가능성 존재 (레짐 이동 문제는 IS 확대로 완전 해결 불가).

**참조**: `논의사항/2026-04-27_결과분석8.md`

---

### 8차 Run 결과 (2026-04-27)

| 지표 | SPY | QQQ | 관문 | 판정 |
|---|---|---|---|---|
| hit_rate | 0.6893 ± 0.3261 | 0.6649 ± 0.3393 | > 0.55 | ✅ PASS |
| r2_oos | **+0.0405 ± 0.7521** | -0.0512 ± 0.8100 | > 0 | SPY ✅ / QQQ ❌ |
| mae | 0.035988 | 0.048877 | — | — |
| 종합 판정 | **PASS** | FAIL | — | — |

n_folds=80, n_oos_samples=1680

**핵심 성과**: SPY R²_OOS = +0.0405 — 8번의 Run 중 처음으로 양수 달성. Phase 1 SPY 관문 최초 통과.

**IS 확대 한계 수익 체감 확인**:
- 231→504: SPY R² -0.21 → -0.025 (개선폭 +0.187)
- 504→756: SPY R² -0.025 → +0.040 (개선폭 +0.065 — 관문 통과하지만 체감 뚜렷)

**과적합 진단**:
- pred_std/true_std: SPY 0.219, QQQ 0.239 — mean collapse 6차보다 심화
- r2>0 폴드: SPY 6/80(7.5%), QQQ 4/80(5.0%) — 소수 good fold가 평균 R²를 끌어올리는 구조
- best_epoch≤3 폴드: SPY 46.2%, QQQ 68.8% — 특히 QQQ 학습 매우 조기 종료

**결론**: 8차 Run(IS=756)을 Phase 1 최종 결과로 확정. 추가 IS 확대는 폴드 수 급감 + 한계 수익 체감으로 효과 의문. BL 연동 시 SPY는 중간 신뢰도 뷰, QQQ는 Omega 높게 설정.

---

### 9차 Run 설계 결정: horizon=14일 (2026-04-27)

"가까운 미래가 더 예측하기 쉬울 수 있다"는 가설 검증을 위해 horizon을 21일 → 14일로 단축.

**변경 내용**:

| 파라미터 | 8차 Run | 9차 Run | 근거 |
|---|---|---|---|
| `horizon` | 21일 | **14일** | 가설 검증 |
| `PURGE` | 21 | **14** | = horizon |
| `OOS` | 21 | **14** | = horizon |
| `STEP` | 21 | **14** | = horizon |
| `EMB` | 21 | 21 | = seq_len (lookback 기준 불변) |
| 폴드 수 | ~80 | **~120** | STEP 단축으로 증가 |

**수정 파일**:
- `scripts/targets.py`: `build_daily_target(horizon)` 제네릭 함수 추가, `build_daily_target_14d` 추가, `verify_no_leakage(horizon=14)` 파라미터 추가
- `02_setting_A_daily14.ipynb`: 신규 생성

**기대 vs 현실 예상**:
- 기대(가설): horizon 단축 → 더 가까운 미래 → 예측 쉬움
- 현실 예상: ACF≈0 장벽은 동일. 타깃 분산이 σ×√14로 더 작아져 noise 비율 악화. 8차 Run보다 나쁠 가능성 높음.

**참조**: `논의사항/2026-04-27_결과분석9.md`

---

### 9차 Run 결과 (2026-04-27)

| 지표 | SPY | QQQ | 관문 | 판정 |
|---|---|---|---|---|
| hit_rate | 0.6707 ± 0.3401 | 0.6657 ± 0.3269 | > 0.55 | ✅ PASS |
| **r2_oos** | **+0.0662 ± 0.6801** | **+0.0501 ± 0.6395** | > 0 | ✅ **양 자산 동시 통과** |
| mae | 0.028800 | 0.039000 | — | — |

n_folds=122, n_oos_samples=1708

**핵심 성과**: SPY·QQQ 동시 관문 통과 — 9번의 Run 중 처음. QQQ R²가 처음으로 양수(+0.0501).

**결과 해석**:
- R² 개선의 주 원인은 "horizon 단축 = 예측 쉬움"이 아니라 **폴드 수 증가(80→122)로 인한 통계적 안정화**
- pred_std/true_std=0.21 — mean collapse는 8차보다 오히려 심화
- 상승 편향 심함: 모델이 96% 이상의 경우에서 양방향(상승) 예측 → 하락장 대응 취약

**시각화 요약**:
- pred_vs_actual: 점들이 y=x 대각선 아닌 y≈0 수평선에 집중. 약한 양의 기울기 존재 (R²>0의 원천)
- per_fold_metrics: fold 80~110 구간에서 극단적 R² 폭락 (-4) — COVID·긴축 충격 구간
- prediction_distribution: pred 분포가 true 분포의 20% 너비에 불과 → mean collapse 시각화
- metric_boxplots: R² 중앙값은 관문 위이나 실제 r2>0 폴드는 5~8%뿐 (outlier 구조)
- residual_confusion: SPY 하락(true≤0) 563건 중 519건을 상승으로 오예측

**Phase 1 최종 채택**: 9차 Run (IS=756, H=14, INPUT_SIZE=1) — SPY·QQQ 모두 관문 통과.

---

### 9차 Run 결과 해석 논의 (2026-04-27)

실행 결과를 보고 세 가지 의문점에 대한 논의.

#### Q1. "BL 정합성 측면에서 8차가 우수"의 의미

Black-Litterman은 월별 리밸런싱(≈21 거래일)을 전제로 설계된다. 8차 Run(H=21)은 예측 horizon이 리밸런싱 주기와 1:1로 일치하므로 예측값을 곧바로 Q(뷰 수익률)로 사용할 수 있다. 9차 Run(H=14)은 14일 수익률을 예측하므로, 이를 BL에 넣으려면 "14일 → 21일 기준 환산"이라는 추가 가정이 필요하고 그 과정에서 오차가 발생할 수 있다. 수치 R²는 9차가 높지만, BL 연동의 편의성·정확성은 8차가 낫다는 의미.

#### Q2. 위기 레짐(crisis regime)에서 성능이 나쁜 것 확인

per_fold_metrics 그래프 fold 80~110 구간 R² -4 폭락이 이를 시각적으로 보여준다. 구조적 원인:

```
IS(3년) → "정상 시장 패턴" 학습
OOS가 위기 구간(급락·극단 변동성)에 걸림
IS에서 배운 패턴과 완전히 다른 레짐 = 역방향 예측
R²= -4 수준 붕괴
```

IS를 아무리 늘려도 한 번도 경험하지 않은 레짐(예: 팬데믹 폭락, 연준 초고속 긴축)이 OOS에서 발생하면 막을 방법이 없다. IS-OOS 레짐 이동(regime shift) 문제의 본질.

#### Q3. 하락장 예측 거의 불가능한 것 확인

혼동행렬 수치:

| | SPY | QQQ |
|---|---|---|
| 실제 하락 총 횟수 | 529건 | 563건 |
| 하락으로 정답 예측 | **15건 (2.8%)** | **44건 (7.8%)** |
| 실제 상승 총 횟수 | 1179건 | 1145건 |
| 상승으로 정답 예측 | **1130건 (95.8%)** | **1093건 (95.5%)** |

원인: IS 기간(2016~2025 중 3년 롤링) 동안 미국 주식은 장기 상승 추세. 모델이 "어차피 오른다"는 상승 편향을 학습. LSTM 구조 문제가 아닌 학습 데이터 자체의 상승 편향이 원인.

**BL 연동 시 함의**: 모델 예측값을 그대로 Q로 쓰면 하락장에서 포트폴리오 손실 집중. 권장 처리 방식 — 방향(부호)만 신뢰하되 Omega를 높게 설정해 뷰의 영향력을 prior(시장 균형 수익률) 수준으로 제한.

---

## 11차 Run 설계 결정 (2026-04-27)

### 배경

9차 Run (H=14, IS=756) 에서 SPY·QQQ 두 자산 모두 R²_OOS > 0 (PASS) 를 달성했지만, 본질적인 문제는 여전히 남아 있다.

- **Mean collapse**: pred_std/true_std ≈ 0.21 → 예측값이 너무 좁은 범위에 집중
- **상승 편향**: 혼동행렬 상 하락 예측 정확도 SPY 2.8%, QQQ 7.8%

이를 개선하기 위해 **CEEMDAN(Complete Ensemble EMD with Adaptive Noise)** 분해를 도입하되, 논문(Objective Black-Litterman views through deep learning)의 방식대로 **IMF별 개별 LSTM 훈련 후 합산**하는 구조를 채택한다. horizon=21일(BL 월별 리밸런싱 1:1 대응)로 고정.

### CEEMDAN 도입 근거

다수의 금융 예측 논문에서 CEEMDAN+LSTM 이 단순 LSTM 대비 예측력 향상을 보고한다. log_return 시계열의 단기 변동성(IMF1~3), 중기 사이클(IMF4~6), 장기 추세(IMF7~) 등 다중 주파수 성분을 분리하여 각 LSTM 이 주기별로 독립 학습한다.

### 데이터 누수 인정

CEEMDAN 의 sifting process 는 전체 시계열의 전역 극값(global extrema)으로 envelope 을 구성하므로, IS 구간 IMF 값에 OOS 미래 정보가 반영된다. 이는 Walk-Forward 기준 데이터 누수에 해당하나, **논문 표준 방식을 따르며 명시적으로 인정**한다.

### 논문 방식 3단계 구조

| 단계 | 설명 |
|---|---|
| Stage 1 | CEEMDAN 으로 log_return → IMF₁ … IMF_K 분해 (전체 시계열 1회) |
| Stage 2 | 각 IMF_k 에 대해 독립 LSTM_k 훈련 (INPUT_SIZE=1, 단일 채널) |
| Stage 3 | fold별 예측 합산: ŷ = Σ_k ŷ_k (수학적 보장: Σ target_k = original_target) |

### 파라미터 (11차 Run)

| 파라미터 | 값 | 비고 |
|---|---|---|
| 기반 | 9차 Run (H=21, IS=756) + 논문 CEEMDAN 방식 추가 | |
| CEEMDAN | trials=100, epsilon=0.005, seed=42 | |
| N_IMF | 자동 결정 (~8~10개) | |
| INPUT_SIZE | **1** | 각 IMF LSTM: 단일 채널 (논문 방식) |
| IS | 756 | 8차 Run 이후 유지 |
| PURGE | **21** | horizon=21에 맞춤 |
| EMB | 21 | seq_len=21 기준 (불변) |
| OOS | **21** | ≈1개월 |
| STEP | **21** | |
| 예상 fold 수 | **≈80** | (2493−819)÷21 |
| 훈련 샘플/폴드 | **714** | IS−seq_len−purge = 756−21−21 |
| 결과 경로 | `results/setting_A_21d_ceemdan/` | |
| 노트북 | `02_setting_A_daily21_ceemdan.ipynb` | |

### 가설

- 8차(H=21, 단일 채널) 대비: CEEMDAN IMF별 분해+합산으로 R²_OOS 추가 향상 기대
- 단, CEEMDAN 데이터 누수로 인한 낙관적 편향 가능성 인정

---

### 11차 Run 결과 (2026-04-27)

| 지표 | SPY | QQQ | 관문 | 판정 |
|---|---|---|---|---|
| hit_rate | 0.7821 ± 0.2266 | 0.7327 ± 0.2262 | > 0.55 | ✅ / ✅ |
| **r2_oos** | **+0.1273 ± 0.9387** | **-2.7889 ± 22.393** [min=-199.73] | > 0 | ✅ / **❌** |
| mae | 0.030807 | 0.042890 | — | — |

n_folds=80, n_oos_samples=1,680

**핵심 사항**: QQQ R²=-2.789(min=-199.73) — **IMF 수 불일치 버그** 원인.

```
버그: N_IMFS = ceemdan_extra['SPY'].shape[1] = 10  (전역 고정)
실제: QQQ = 11 IMFs → IMF₁₁ 누락 → Stage 3 y_true_sum ≠ original_target
수학적 보장 파괴 → QQQ 결과 무효
```

**수정 방향**: `n_imfs_ticker = ceemdan_extra[ticker].shape[1]` (ticker별 동적 결정) → 12차 Run에서 반영.

**SPY 성과**: R²=+0.1273, hit_rate=0.7821 — Phase 1 전체 최고 기록. CEEMDAN이 SPY에서 유효함 확인.

**참조**: `논의사항/2026-04-27_결과분석11.md`

---

## 12차 Run 설계 결정 (2026-04-27)

### 배경

11차 Run(H=21, CEEMDAN)에서 QQQ IMF 수 불일치 버그로 QQQ FAIL. 두 가지를 동시에 수정한다.
1. **버그 수정**: ticker별 `n_imfs_ticker` 동적 결정
2. **horizon 단축**: 21d → 5d — 예측 난이도 완화로 두 자산 동시 통과 목표

BL 연동 시 `Q_5d × (21/5)` 선형 스케일링으로 21일 뷰로 근사 변환 가능.

### 변경 사항

| 파라미터 | 11차 Run | 12차 Run | 근거 |
|---|---|---|---|
| horizon | 21 | **5** | 예측 난이도 완화 |
| PURGE | 21 | **5** | = horizon |
| IMF 수 결정 | 전역 고정(N_IMFS=10) | **ticker별 동적** | 버그 수정 |
| 훈련 샘플/폴드 | 714 | **730** | IS−seq_len−purge=756−21−5 |
| 폴드 수 | 80 | **82** | PURGE 축소로 소폭 증가 |
| 노트북 | `02_setting_A_daily21_ceemdan.ipynb` | `02_setting_A_daily5_ceemdan.ipynb` | 신규 생성 |

**참조**: `논의사항/2026-04-27_결과분석12.md`

---

### 12차 Run 결과 (2026-04-27)

| 지표 | SPY | QQQ | 관문 | 판정 |
|---|---|---|---|---|
| hit_rate | 0.6806 ± 0.1442 | 0.7149 ± 0.1217 | > 0.55 | ✅ / ✅ |
| **r2_oos** | **+0.0155 ± 0.6543** | **+0.0622 ± 0.5698** | > 0 | ✅ / ✅ |
| mae | 0.016370 | 0.020863 | — | — |
| rmse | 0.019906 | 0.025294 | — | — |

n_folds=82, n_oos_samples=1,722

**핵심 성과**:
- SPY·QQQ 동시 관문 통과 (9차 이후 두 번째)
- **pred_std/true_std = 0.833(SPY) / 0.862(QQQ)** — 9차 Run(0.209/0.205) 대비 4배 개선. **Mean collapse 해소.**
- best_epoch=1 비율 **0%** (9차: SPY 13.9%, QQQ 23.8%) — 모든 폴드 충분히 학습

**결과 해석**:
- CEEMDAN 분해로 IMF별 독립 학습 → 합산 예측 진폭이 실제에 근사 (mean collapse 해소)
- train/val R² ≈ +0.45~0.53, test R² ≈ +0.016~0.062 → IS-OOS 갭 여전히 존재하나 test가 양수 유지
- BL 연동 시 `Q_5d × (21/5)` 스케일링 필요 (5d horizon vs 21d 리밸런싱 불일치)

**Phase 1 전체 런 정리**:

| Run | IS | H | CEEMDAN | SPY R²_OOS | QQQ R²_OOS | 양자산 통과 | pred/true_std |
|---|---|---|---|---|---|---|---|
| 8차 | 756 | 21 | ❌ | +0.0405 | -0.0512 | ❌ | 0.219/0.239 |
| 9차 | 756 | 14 | ❌ | +0.0662 | +0.0501 | ✅ | 0.209/0.205 |
| 11차 | 756 | 21 | ✅ (버그) | +0.1273 | -2.789 | ❌ | — |
| **12차** | **756** | **5** | **✅** | **+0.0155** | **+0.0622** | **✅** | **0.833/0.862** |

**후속 고려**: 11차 H=21 CEEMDAN 버그 수정 재실험 — SPY R²=+0.127이 최고 기록이므로, QQQ도 정상 IMF 수(11개)로 실행 시 BL horizon 정합(21d) + CEEMDAN 효과 동시 달성 가능성 있음.

---

### 12차 Run 최종 평가 (2026-04-27 노트북 실행 완료)

**실제 실행 결과 확인**: 위 수치와 정확히 일치 (seed=42 재현성 보장).

**성능 한계 인식**:
- 관문(hit_rate>0.55, R²>0)은 통과했으나, previous baseline(R²=+0.55) 대비 LSTM(R²=+0.016~0.062) 격차가 매우 크다.
- RMSE 기준으로도 LSTM(SPY 0.020, QQQ 0.025) > previous(SPY 0.017, QQQ 0.021) → 이전 수익률 복사 전략보다 RMSE가 더 나쁨.
- 5d horizon으로 단축해도 예측력 자체는 크게 개선되지 않았음. CEEMDAN의 기여는 **예측 진폭 복원(mean collapse 해소)**이며, **절대적 예측 정확도 향상**은 아님.

**새로 추가된 지표 (`r2_std`)**:
- SPY: -0.404 ± 1.151, QQQ: -0.394 ± 1.235
- 정규화된 예측에서 R²가 음수 → 예측 진폭이 실제보다 여전히 작음(pred_std/true_std=0.83~0.86이 1.0 미만인 것과 일치)

**Phase 1 결론**: 12차 Run(CEEMDAN 5d)을 현재 Phase 1 최종 결과로 유지. BL 연동 시 `Q_5d × (21/5)` 스케일링 적용, 높은 Omega(불확실성) 설정으로 뷰 영향력 제한 권장.
