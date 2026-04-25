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
