# 2.4 PyTorch Dataset / DataLoader 구현

> **한 줄 요약**
> `Dataset` 은 "한 샘플을 어떻게 꺼내는가" 를 정의하는 **인덱스 룩업 규약**이고, `DataLoader` 는 그 Dataset 위에 **배치 · 셔플 · 병렬로드** 를 얹는 래퍼입니다. 시계열에서는 2.1~2.3에서 다룬 슬라이딩 윈도우 · look-ahead bias 방지 로직을 Dataset 안에 숨겨 두는 것이 목표입니다.

---

## 0. 이 문서가 다루는 범위

| 구분 | 내용 |
|---|---|
| 개념 | Dataset · DataLoader 의 역할 분담, 3대 필수 메서드 |
| 시계열 특수성 | shuffle 주의, IS-only scaler 주입, walk-forward 와의 결합 |
| 실전 코드 | 우리 프로젝트용 `SequenceDataset` 클래스 설계 |
| 자주 하는 실수 | 3가지 안티패턴과 대응 |

이 문서는 **2.1~2.3의 모든 내용이 하나의 클래스 안에 녹아 들어가는 지점**입니다. build_xy 함수, IS-only fit, walk-forward 윈도우가 `Dataset.__getitem__` 한 줄로 호출 가능해야 합니다.

---

## 1. 왜 Dataset/DataLoader 가 필요한가

지금까지 우리는 `X, y` 를 한꺼번에 numpy 로 만들어왔습니다. 그런데 GRU 학습 단계에서는 아래 네 가지가 **모두** 필요합니다.

1. **배치(batch) 단위로 자르기** — 메모리 한 번에 다 못 올림
2. **셔플(shuffle)** — 학습 안정성 (단, 시계열에서는 조심!)
3. **병렬 로드(num_workers)** — GPU 가 놀지 않도록 CPU 가 미리 배치를 준비
4. **동일한 규약** — 훈련/검증/테스트에서 *같은 인터페이스*로 데이터를 꺼내야 함

numpy 로 이걸 전부 직접 짜면 **학습 루프가 지저분해지고 재사용이 불가능**합니다. PyTorch 는 이 역할을 두 클래스로 나눴습니다.

```
┌──────────────────────┐        ┌──────────────────────┐
│  Dataset             │        │  DataLoader          │
│  ──────────          │        │  ──────────          │
│  len()               │ ◄───── │  배치 묶음           │
│  getitem(i) → (X,y)  │        │  shuffle             │
│  한 샘플 꺼내기      │        │  num_workers 병렬    │
└──────────────────────┘        └──────────────────────┘
       "하나만"                    "한꺼번에 병렬로"
```

역할 분담이 명확하면 **시퀀스 데이터를 다른 용도로 꺼내 써야 할 때(예: 시각화, 디버깅) Dataset 하나만 갖다 쓰면 되고**, DataLoader 설정만 바꿔 배치/병렬/셔플을 조절할 수 있습니다.

---

## 2. Dataset 의 3대 메서드

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, ...):
        # 1. 원본 데이터를 로드/전처리하고 self 에 저장
        # 2. 이 단계에서는 "샘플 꺼낼 준비" 만 함 (아직 배치 X)
        ...

    def __len__(self):
        # 이 Dataset 이 가진 전체 샘플 수를 반환
        # DataLoader 는 len() 만큼 인덱스를 순회함
        return ...

    def __getitem__(self, idx):
        # idx 번째 샘플 한 개를 반환
        # 반환 타입은 보통 (torch.Tensor, torch.Tensor) 또는 dict
        return x, y
```

### 2.1 역할 분담의 핵심 원칙
- `__init__` 에서 **무거운 계산(스케일링, 피처 엔지니어링)** 을 끝내두면 매 `__getitem__` 이 빠르다.
- `__getitem__` 은 **최대한 단순**해야 한다 — 병렬 로드 시 이 메서드가 수천 번 호출되므로.
- 인덱스 매핑(`idx → 실제 데이터 위치`) 은 `__init__` 에서 미리 계산해서 리스트/배열로 들고 있기.

---

## 3. DataLoader 가 해주는 것

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,       # ← 시계열에서는 주의! (§4.1)
    num_workers=4,      # 병렬 로드 (CPU 수)
    pin_memory=True,    # CPU→GPU 복사 최적화
    drop_last=False,    # 마지막 배치 크기 < batch_size 일 때 버릴지
)
```

`DataLoader` 는 내부적으로 이렇게 동작합니다.

1. `range(len(dataset))` 을 셔플(optional) 해서 인덱스 리스트 만들기
2. `batch_size` 개씩 잘라서 **각 인덱스마다 `dataset[i]` 호출**
3. 그 결과들을 **stack** 해서 `(B, ...)` 배치 텐서로 묶기 (= collate)
4. `num_workers` 개 프로세스로 이 작업을 병렬화

중요한 것은 **`DataLoader` 는 Dataset 이 어떤 식으로 샘플을 만드는지 모른다**는 점입니다. Dataset 이 `(X, y)` 튜플만 뱉으면 나머지는 자동입니다.

---

## 4. 시계열에서의 특수 사항 ★

### 4.1 shuffle 주의
- **학습 배치 내부의 순서**는 섞어도 된다 (GRU 는 배치의 순서와 무관하게 하나의 시퀀스를 처리).
- **하지만 `(X, y)` 쌍이 바뀌면 안 된다** — Dataset 이 `(t 시점 시퀀스, t 시점 타겟)` 을 반환하므로, `DataLoader(shuffle=True)` 는 안전하다.
- **금지**: IS/OOS 가 섞이는 shuffle — 그건 Dataset 설계가 아니라 *split 단계*에서 막아야 한다 (2.3 에서 다룸).

### 4.2 IS-only scaler 주입
2.3 §5 에서 봤듯이, `StandardScaler.fit` 은 반드시 **IS 에서만** 일어나야 합니다. 그런데 Dataset 내부에서 `fit_transform` 하면 안 됩니다 — Dataset 은 IS/OOS 둘 다에서 같은 규약으로 데이터를 뱉어야 하기 때문입니다.

**권장 패턴**: Dataset 생성자에 *이미 fit 된* scaler 를 주입하기.

```python
# IS Dataset — fit 은 여기서 끝냄
scaler = StandardScaler().fit(X_is_raw)
ds_is  = SequenceDataset(df_is, scaler=scaler, T=60, h=21)
ds_oos = SequenceDataset(df_oos, scaler=scaler, T=60, h=21)   # 같은 scaler 재사용
```

이렇게 하면 Dataset 은 "변환만" 하고, fit 책임은 밖에서 지게 됩니다.

### 4.3 walk-forward 와의 결합
47 fold 각각에서 `(SequenceDataset_is, SequenceDataset_oos)` 쌍을 만들게 됩니다. 이때 Dataset 생성이 느리면 전체 walk-forward 가 느려지므로, **`__init__` 에서 sliding_window_view 를 한 번만 호출**하고 이후에는 인덱싱만 하는 설계가 최적입니다.

---

## 5. 실전 설계 — `SequenceDataset` 스케치

우리 프로젝트에서 쓸 최종 형태를 미리 봅니다. 이 클래스가 2.1(sliding window) + 2.2(build_xy) + 2.3(leakage-free) 의 집대성입니다.

```python
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

class SequenceDataset(Dataset):
    """
    (N, F) tabular DataFrame 을 (N', T, F) 시퀀스 텐서로 변환하여 제공.

    Parameters
    ----------
    df : pandas.DataFrame                          시간순 정렬된 원본
    feature_cols : list[str]                       입력으로 쓸 컬럼
    target_col : str                               타겟 컬럼 (미래 참조 가능 — shift 된 상태여야 함)
    T : int                                        lookback window
    scaler : sklearn-like Transformer | None       fit 된 scaler (None 이면 raw 값)
    stride : int                                   윈도우 간격 (학습=1, OOS=21 추천)
    """
    def __init__(self, df, feature_cols, target_col, T,
                 scaler=None, stride=1):
        # 1. 원본 배열 확보
        arr_X = df[feature_cols].to_numpy(dtype=np.float32)
        arr_y = df[target_col].to_numpy(dtype=np.float32)

        # 2. IS-only scaler 적용 (주입 방식)
        if scaler is not None:
            arr_X = scaler.transform(arr_X).astype(np.float32)

        # 3. sliding window 로 (N-T+1, T, F) 만들기 (view — 메모리 공유)
        v = sliding_window_view(arr_X, window_shape=(T,), axis=0)  # (N-T+1, F, T)
        X = np.transpose(v, (0, 2, 1))[::stride]                   # (N', T, F)
        y = arr_y[T - 1 :: stride]                                 # (N',)

        # 4. y=NaN (맨 뒤 h일) 제거
        mask = ~np.isnan(y)
        self.X = torch.from_numpy(X[mask].copy())
        self.y = torch.from_numpy(y[mask])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 이미 텐서화 끝났으므로 인덱싱만
        return self.X[idx], self.y[idx]
```

### 5.1 이 설계의 장점
- 한 번 생성하면 `__getitem__` 은 O(1) — 학습 루프가 매 배치마다 느려지지 않음
- `scaler` 를 밖에서 주입하므로 IS/OOS 에 다른 처리가 섞일 여지가 없음
- `stride` 파라미터로 학습(1)/OOS(21)을 같은 클래스로 통제

### 5.2 주의할 점
- 텐서를 `__init__` 에서 만들면 **메모리를 미리 다 차지**합니다. 우리 데이터(1217 × 60 × 17 × 4B ≈ 5MB)는 전혀 문제 없지만, 수십만 샘플 규모로 커지면 `__getitem__` 에서 on-the-fly 변환하는 버전이 필요합니다 (오늘 실습에서는 두 방식을 **속도 비교**로 직접 보여줄 예정).

---

## 6. 자주 하는 실수 3가지

| # | 실수 | 왜 문제인가 | 대응 |
|---|---|---|---|
| 1 | `Dataset.__init__` 안에서 `scaler.fit_transform` | IS/OOS 모두 각자 fit → scaler 다름 → OOS 평가 왜곡 | scaler 는 **밖에서 fit**, Dataset 은 transform 만 |
| 2 | `shuffle=False` 로 놓고 "순서 지켰으니 안전" 이라 착각 | shuffle 은 학습 배치 순서 문제, look-ahead 는 **split 단계** 문제 — 서로 다른 이슈 | split 은 `TimeSeriesSplit` 또는 walk-forward 로, shuffle 은 학습 안정성 관점에서 따로 고려 |
| 3 | `num_workers > 0` 인데 `__getitem__` 이 무거운 연산 | Python 의 GIL 때문에 병렬화 이득이 거의 없고, 메모리 복사 오버헤드만 커짐 | 무거운 일은 `__init__` 으로, `__getitem__` 은 인덱싱만 |

---

## 7. 체크리스트

설계 단계에서 아래 항목을 자문합니다.

- [ ] `__init__` 에서 sliding window / scaler transform 등 한 번으로 끝낼 수 있는 일을 다 처리했는가?
- [ ] scaler 는 **Dataset 외부에서 fit** 된 객체를 주입받는가?
- [ ] `__len__` 이 실제 유효 샘플 수 (NaN 제거 후) 와 일치하는가?
- [ ] `__getitem__` 이 (X, y) 를 **torch.Tensor** 로 반환하는가?
- [ ] walk-forward 의 각 fold 마다 Dataset 을 **새로** 만드는가? (scaler 도 새로 fit)
- [ ] shuffle 설정이 "학습이냐 평가냐" 에 따라 분리돼 있는가? (학습=True, 평가=False)

---

## 8. 자가 점검 질문

스스로 답해보세요 (노트북 Step 끝에서 해설합니다).

1. `Dataset` 과 `DataLoader` 의 책임을 한 문장씩으로 정리하면?
2. 시계열에서 `shuffle=True` 는 되는데 `KFold(shuffle=True)` 는 안 되는 이유는?
3. `num_workers > 0` 으로 늘렸는데 오히려 느려졌다. 가능한 원인 3가지는?
4. fold 마다 scaler 를 새로 fit 해야 하는 이유를 2.3 의 교훈과 연결해서 설명해보세요.

---

## 부록 A. collate_fn — 언제 직접 작성하나

`DataLoader` 의 기본 collate 는 "각 샘플을 `torch.stack`" 입니다. 그래서 샘플마다 shape 이 같으면 건드릴 필요 없습니다. 우리 시퀀스 데이터는 모두 `(T, F)` 로 동일하므로 **기본값으로 충분**합니다.

커스텀 `collate_fn` 이 필요한 경우:
- 가변 길이 시퀀스 (ex. 자연어) → padding 필요
- 여러 종류의 출력을 dict 로 묶어 받고 싶을 때
- 실시간 데이터 증강을 배치 단위로 적용하고 싶을 때

우리 프로젝트(고정 T)에서는 당분간 필요 없습니다.

---

## 다음 단계 → 2.5 텐서 차원의 의미

2.4 에서 만든 `SequenceDataset` 은 배치 텐서를 `(B, T, F)` 로 내보냅니다. 2.5 에서는 **B(batch)·T(time)·F(feature) 세 축의 의미**와 PyTorch GRU 가 기본적으로 요구하는 차원 순서(`batch_first=True` 여부) 를 정리하며, 3주차 GRU 구현으로 넘어갈 준비를 마칩니다.
