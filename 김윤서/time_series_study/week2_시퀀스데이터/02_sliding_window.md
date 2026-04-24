# 2.2 Sliding Window 기법

> **이 토픽의 목표**
> 2.1에서 본 `(N, F) → (N', T, F)` 변환을 **수식·구현·파라미터 선정 근거**까지 전부 설명할 수 있게 됩니다.
> 특히 `T` 와 `h` 를 고르는 기준을 1.3 실측 ACF 와 프로젝트 현실에 맞춰 정리합니다.

---

## 0. 한 줄 요약

- Sliding Window 는 **세 개의 파라미터**로 정의됩니다: `T` (lookback), `h` (horizon), `stride` (슬라이드 간격).
- 구현은 두 방식이 있는데, 대용량에서는 `np.lib.stride_tricks.sliding_window_view` 가 **메모리를 복사하지 않아서** for-loop 버전보다 수백 배 빠르고 가볍습니다.
- `T` 는 도메인 지식 + ACF 기반 근거 + 계산 비용의 삼각 트레이드오프로 정하고, 현재 프로젝트는 `T=60` 을 1차 후보로 삼습니다.

---

## 1. Sliding Window 의 세 파라미터

```
시점 ──────▶

원본 시계열   :  x_0  x_1  x_2  x_3  x_4  x_5  ...  x_{N-1}
                                                        ▲
                                                      오늘 (t)

샘플 i:
    X_i = x_{t-T+1}, ..., x_t           ← 과거 T일 (입력)
    y_i = f(x_{t+1}, ..., x_{t+h})      ← 미래 h일로 만든 레이블
    stride 만큼 t를 밀면서 반복
```

| 파라미터 | 의미 | 현재 프로젝트 설정 |
|---|---|---|
| **T (lookback window)** | 한 샘플에 포함되는 **과거 관측치 개수** | 미정 — 2주차 후보 T=60 |
| **h (prediction horizon)** | 레이블이 **몇 일 뒤의 결과**인지 | 21일 (`fwd_ret_21d`) |
| **stride** | 인접한 두 샘플의 **시작 시점 간격** | 1 (완전 겹침) — 표준 |

세 값을 확정하면 샘플 수가 결정됩니다.

$$
N' = \left\lfloor \frac{N - T - h + 1}{\text{stride}} \right\rfloor + 1
$$

(h=0 이고 stride=1이면 2.1에서 본 `N - T + 1` 로 돌아갑니다.)

---

## 2. stride 선택: 1 vs > 1

### 2.1 stride=1 (표준)

```
샘플 0: [t=0 .. t=T-1]
샘플 1: [t=1 .. t=T  ]  ← 샘플 0과 T-1 일치 공유
샘플 2: [t=2 .. t=T+1]
```

- 장점: **샘플 수 최대화**
- 단점: 인접 샘플 간 상관 ≈ 1. "독립적인 데이터 포인트"로 셀 수 없음
  - 학습 로스 값이 낮아도 실제 일반화 성능과 괴리가 생길 수 있음
  - 교차검증 시 인접 샘플이 train/val 양쪽에 섞이면 OOS 성능이 과장됨

### 2.2 stride = h

```
샘플 0: [t=0 .. t=T-1]    레이블 t=T 부터 t=T+h-1
샘플 1: [t=h .. t=T+h-1]  레이블 t=T+h 부터 t=T+2h-1
```

- 장점: 레이블 윈도우가 **겹치지 않음** → 샘플 간 독립성 ↑
- 단점: 샘플 수 `1/h` 로 감소

### 2.3 실무에서의 관례

| 상황 | 권장 stride |
|---|---|
| 학습(fit) — 데이터 많이 필요 | stride = 1 |
| 검증(validate) / OOS 평가 | stride ≥ h (레이블 겹침 방지) |
| 예측(inference) | stride = 1 (매일 갱신) |

**현재 프로젝트는 이미 Walk-Forward 구조**여서 OOS 구간에서의 레이블 중복 문제가 약화됩니다. 2.3(look-ahead bias)에서 이 부분을 정밀하게 다룹니다.

---

## 3. 구현 — for-loop vs stride_tricks

### 3.1 for-loop 버전 (2.1 `make_sequences`)

```python
def make_sequences_loop(arr, T):
    N, F = arr.shape
    out = np.empty((N - T + 1, T, F), dtype=arr.dtype)
    for i in range(N - T + 1):
        out[i] = arr[i : i + T]  # ← 매번 T×F 만큼 복사
    return out
```

메모리 사용: `(N - T + 1) × T × F × 4 bytes` (float32 기준)
시간 복잡도: O(N·T·F) — T 가 커지면 선형 증가

### 3.2 stride_tricks 버전

```python
from numpy.lib.stride_tricks import sliding_window_view

def make_sequences_view(arr, T):
    # sliding_window_view 는 (N-T+1, F, T) shape 반환
    # 우리는 (N-T+1, T, F) 가 필요하므로 transpose
    view = sliding_window_view(arr, window_shape=(T,), axis=0)  # (N-T+1, F, T)
    return np.transpose(view, (0, 2, 1))  # (N-T+1, T, F)
```

- **데이터를 복사하지 않음**. 같은 메모리를 가리키는 "뷰"만 반환.
- 메모리 사용: **원본 배열 크기와 거의 동일** (O(N·F))
- 시간 복잡도: O(1) — 뷰 생성만 함
- 주의: 반환된 배열을 **수정하면 원본이 바뀜**. 학습용이면 `.copy()` 한 번 해두면 안전

### 3.3 언제 어떤 걸 쓰나

| 상황 | 권장 |
|---|---|
| 교육·시각화 — 이해가 최우선 | for-loop |
| 실전 학습 파이프라인 | stride_tricks + 필요 시 `.copy()` |
| GPU 업로드 직전 | torch tensor로 바로 변환하면 어차피 복사됨 |

---

## 4. T (lookback) 선정 — 근거 3축

**T를 크게 하면**: 더 긴 히스토리 참조, 샘플 수 감소, 파라미터 학습 난이도 상승
**T를 작게 하면**: 샘플 많이 확보, 단기 동역학만 반영

어떤 T가 적절한지 판단할 때 쓰이는 세 축입니다.

### 4.1 ACF 기반 — "자기상관이 사라지는 lag"

1.3 실습에서 MSFT 에 대해 관측한 것:

| 대상 | 통계적 유의 지속 lag | 해석 |
|---|---|---|
| `log_return_1d` | lag 1~8 정도까지 간헐적 유의 | 수익률 자체 의존성은 짧음 |
| `sq_log_return_1d` (변동성) | **lag 1~20 전체 유의** (p ≈ 0) | 변동성 의존성은 길게 이어짐 |
| `abs_log_return_1d` | lag 1~39 유의 | 절댓값 기준으론 더 길게 남음 |

**기준:** "모델이 잡아야 할 패턴의 자기상관이 유의한 lag 범위"를 T의 하한으로 삼음.
- 수익률만 신경 쓴다면 T ≥ 10 정도
- 변동성 클러스터링까지 학습시키려면 T ≥ 30 이상
- 절댓값 기준이면 T ≥ 50 까지

### 4.2 도메인 지식 — 시장 체제(regime) 주기

- 월 단위 리밸런싱 주기 (h=21) → 과거 1~3 개월치 정보가 가장 관련성 큼
- **분기 실적 발표** 주기 ≈ 63일 → T=60은 한 분기의 정보를 본다
- 변동성 체제(VIX) 전환 주기 ≈ 20~40일

### 4.3 계산 비용 & 샘플 효율

T 가 커지면 `(N', T, F)` 의 T 차원이 커지고, GRU가 시간축을 따라 RNN 연산을 더 많이 반복합니다.
- T=60, F=17 기준 한 샘플당 60번의 RNN step
- T=252 로 가면 같은 정보량 학습에 **4배 계산**

실전 권장:
```
T ∈ [20, 30, 60, 90] 중에서 1차 후보 3개 → 검증
```

### 4.4 이 프로젝트의 1차 후보: T = 60

근거 요약:
1. ACF: 변동성 클러스터링(lag ~20) + 여유 마진
2. 도메인: 한 분기 정보 포함
3. 계산 비용: 252일 대비 1/4로 현실적
4. 샘플 손실: MSFT 단일 종목 기준 4.6%, walk-forward IS(252일) 기준 23%

2.5 이후 GRU 학습까지 간 뒤, 실제 OOS 성능으로 T=30, 60, 90 세 값을 비교해볼 예정입니다.

---

## 5. h (prediction horizon) 처리

현재 프로젝트 `Step3_IT_WalkForward_panel.ipynb` 는 이미 h=21로 `fwd_ret_21d` 컬럼을 계산해두고 있습니다:

```python
df[TARGET_COL] = df["adj_close"].shift(-OOS_DAYS) / df["adj_close"] - 1
```

즉 **레이블은 이미 21일 뒤의 수익률**이고, 우리가 sliding window 를 만들 때는 "오늘까지의 과거 T일 → 오늘의 fwd_ret_21d" 로 짝지으면 됩니다.

```
시점 t 에서의 샘플:
    X_t = x_{t-T+1 : t}          ← 과거 T일 (입력)
    y_t = df["fwd_ret_21d"][t]   ← 이미 계산된 21일 뒤 수익률
```

### 5.1 맨 앞 T-1 일과 맨 뒤 h 일이 잘려나가는 이유

- **앞쪽 T-1 일**: 과거 T일이 없어서 X를 못 만듦
- **뒤쪽 h 일**: 21일 뒤 수익률을 계산할 미래가 없어서 y가 NaN

실제 유효 샘플 수:

$$
N_{\text{valid}} = N - T + 1 - h = N - T - h + 1
$$

MSFT 2020-12 ~ 2025-12 기간 N=1276, T=60, h=21 → **N_valid = 1196 샘플**.

---

## 6. 레이블 변환 — 연속값 → 5분위

현재 프로젝트는 `fwd_ret_21d` 의 **크로스 섹션 5분위**를 클래스 레이블로 씁니다 (10종목 패널 안에서 순위).

```python
def make_quintile_labels(df_panel, target_col, n_classes=5):
    # 각 날짜별로 10종목의 target_col 을 5개 버킷에 배정
    return df_panel.groupby(df_panel.index)[target_col] \
                   .transform(lambda s: pd.qcut(s.rank(method='first'),
                                                n_classes, labels=False))
```

이번 2주차에선 단일 종목 MSFT로 실습하므로, 잠정적으로 **단일 종목 시계열 5분위**(이 종목의 지난 252일 분포 대비 순위)를 쓰거나 **회귀 그대로** (`fwd_ret_21d` 원값)를 쓸 수 있습니다. 2.4 에서 DataLoader 구현할 때 패널 로직으로 확장합니다.

---

## 7. 데이터 누수 사전 경고 (2.3에서 자세히)

Sliding window를 만들 때 실수하기 쉬운 누수 두 가지만 미리 짚고 갑니다.

### 7.1 정규화 누수

피처 정규화(Z-score, min-max 등)를 **전체 기간 통계로** 한 뒤 window 를 자르면,
- 샘플 0의 X 안에 **"미래의 평균·표준편차로 계산된 값"**이 들어감
- OOS 성능이 비현실적으로 좋게 나옴

**해결**: 정규화는 항상 **학습 구간(IS) 통계로만** fit. 이후 IS/OOS 모두에 transform.

### 7.2 target leakage

레이블 `fwd_ret_21d` 에 쓰이는 `adj_close` 값이 X에 포함된다면, **미래 정보가 누설**됩니다.
현재 프로젝트의 17개 피처에는 `adj_close` 자체가 없어서 직접 누수는 없지만, 파생 피처(예: mom_12m) 계산 시점을 잘못 잡으면 발생할 수 있습니다.

**해결**: 파생 피처는 `t` 시점의 피처로만 계산 — `t+k` (k>0)의 정보가 섞이면 안 됨.

---

## 8. 프로젝트 연결 — 완성형 미리 보기

이번 토픽이 끝나면 이런 함수 구조가 됩니다 (2.4 에서 PyTorch Dataset 으로 감쌈):

```python
def build_xy(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    T: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    df: 날짜 오름차순 정렬된 단일 종목 DataFrame.
        feature_cols + target_col 이 모두 결측 없이 들어있어야 함.

    반환:
        X: shape (N', T, F)
        y: shape (N',)      — target_col 값
    """
    arr_X = df[feature_cols].to_numpy(dtype=np.float32)
    arr_y = df[target_col ].to_numpy(dtype=np.float32)

    from numpy.lib.stride_tricks import sliding_window_view
    view = sliding_window_view(arr_X, window_shape=(T,), axis=0)   # (N-T+1, F, T)
    X    = np.transpose(view, (0, 2, 1))[::stride].copy()          # (N', T, F)
    y    = arr_y[T - 1 :: stride]                                  # (N',)

    # NaN 필터 — fwd_ret_21d 가 마지막 h일에서 NaN이므로 여기서 제거
    mask = ~np.isnan(y)
    return X[mask], y[mask]
```

이 함수를 써보면 2.1에서 말한 "얇은 어댑터"의 실체가 보입니다.
2.3 에서 누수 방지 로직을 추가하고, 2.4 에서 `torch.utils.data.Dataset` 으로 포장합니다.

---

## 9. 스스로 점검

1. T=60, h=21, stride=1 일 때 N=1276 이면 유효 샘플 수는 몇 개입니까? 공식으로 답하세요.
2. stride 를 1에서 21로 바꾸면 샘플 수가 어떻게 변합니까? 왜 학습 단계에서 stride=1 이 권장되는지 설명하세요.
3. `np.lib.stride_tricks.sliding_window_view` 가 for-loop 버전보다 메모리를 적게 쓰는 이유는 무엇입니까? `.copy()` 를 추가해야 하는 상황은 언제입니까?
4. T 를 고를 때 "ACF 기반 근거"를 어떻게 활용할 수 있는지, 1.3 MSFT 실측 결과와 연결해 설명하세요.
5. 현재 프로젝트에서 맨 앞 59일 + 맨 뒤 21일이 학습에 쓰이지 못하는 이유를 각각 설명하세요.
6. 정규화 누수(normalization leakage)가 왜 sliding window 구조에서 특히 위험한지 설명하세요.

---

## 부록 A — stride > 1 일 때 샘플 수 공식

stride = s, lookback = T, horizon = h, 원본 길이 = N 일 때:

$$
N' = \left\lfloor \frac{N - T - h}{s} \right\rfloor + 1
$$

예시 (N=1276, T=60, h=21):
- s=1: N' = (1276 - 60 - 21) / 1 + 1 = **1196**
- s=5: N' = ⌊1195/5⌋ + 1 = **240**
- s=21: N' = ⌊1195/21⌋ + 1 = **57**

## 부록 B — `sliding_window_view` 의 차원이 헷갈리는 이유

```python
arr.shape = (N, F)
view = sliding_window_view(arr, window_shape=(T,), axis=0)
# view.shape = (N-T+1, F, T)   ← 주의!
```

`axis=0` 에 대해 윈도우를 밀었더니 T가 **맨 뒤로** 붙습니다. 우리가 원하는 `(N', T, F)` 가 되려면 `np.transpose(view, (0, 2, 1))` 로 재배치가 필요합니다. 이 부분에서 많이 실수합니다.

---
