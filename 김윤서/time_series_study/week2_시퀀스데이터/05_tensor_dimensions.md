# 2.5 텐서 차원의 의미 — (Batch, Time, Feature)

> **학습 목표**
> 1. `(B, T, F)` 3축이 각각 무엇을 의미하고, 축을 섞으면 왜 의미가 붕괴하는지 설명할 수 있다.
> 2. PyTorch GRU 의 `batch_first` 관례가 존재하는 이유와 전환 방법을 이해한다.
> 3. `permute`, `reshape`/`view`, `unsqueeze`/`squeeze` 를 **언제 어느 것을 써야 하는지** 판단할 수 있다.
> 4. Broadcasting 규칙으로 "축이 안 맞는데 연산이 되는 이유" 를 설명할 수 있다.
> 5. 자주 하는 축 순서 버그를 세 가지 이상 알고 있다.

---

## 왜 이 토픽이 필요한가

지금까지 우리가 만든 배치는 `(32, 60, 17)` 입니다. 세 숫자로만 보이지만 각 위치가 **다른 의미** 를 가집니다.

```
(32,   60,   17)
  │    │    └── Feature  (log_return, mom_1m, vol_20d, ...)
  │    └────── Time      (day 0, day 1, ..., day 59)
  └─────────── Batch     (샘플 #0, 샘플 #1, ..., 샘플 #31)
```

시퀀스 모델 디버깅에서 터지는 버그의 **절반 이상** 이 이 세 축이 섞이면서 발생합니다. 예컨대:

- `permute` 없이 `reshape(32, 17*60)` 으로 평탄화 → Time 과 Feature 가 섞여버림. loss 는 줄어드는데 예측이 무의미.
- GRU 에 `(32, 60, 17)` 을 `batch_first=False` 로 넘김 → GRU 는 "60개 샘플, 32일 시퀀스, 17피처" 로 해석. 결과물은 오류 없이 나오지만 학습이 안 됨.

이런 버그들은 **shape 는 맞지만 의미가 틀린** 부류입니다. 예외가 터지지 않으니 가장 위험하죠. 이 토픽에서는 "축의 의미" 를 코드 구조로 지키는 법을 배웁니다.

---

## §1. 세 축의 의미 — 독립성과 순서 가정

### 1.1 각 축이 허용하는 연산

| 축 | 의미 | 순서 중요? | 섞어도 되나 | 전형적 연산 |
|---|---|---|---|---|
| **B** (Batch) | 샘플 번호 | ❌ | ✅ (shuffle) | 배치 평균, gradient 집계 |
| **T** (Time) | 시점 번호 | ✅ | ❌ 절대 안 됨 | GRU 내부 순환, time pooling |
| **F** (Feature) | 변수 번호 | ❌ | 보통 ❌ | Linear layer, Attention |

핵심 직관 세 가지:

1. **B 축 은 독립 표본 축.** 배치 내 32개 샘플은 서로 영향을 주지 않습니다. shuffle 해도 학습 결과가 바뀌면 안 됩니다(작은 확률적 차이만 있음).
2. **T 축 은 인과 축.** day 5 는 day 4 에 의존하고, day 4 는 day 3 에 의존… 순서를 섞으면 시계열 구조가 붕괴합니다. **GRU/LSTM 은 이 축을 따라 hidden state 를 전파** 합니다.
3. **F 축 은 명명된 축.** col 0 은 log_return, col 1 은 simple_return… 이름이 있습니다. 학습된 가중치는 "0번 피처는 이 역할" 로 암기하므로, 학습 시점과 추론 시점에 피처 순서가 다르면 예측이 무의미해집니다.

### 1.2 "축 계약(contract)" 개념

프로젝트 전체에서 **"3D 텐서는 항상 `(B, T, F)` 다"** 라고 약속하고, 이 약속을 모든 함수의 입출력 문서에 명시하면 버그가 크게 줄어듭니다. 이 약속을 **축 계약** 이라고 부릅니다.

```python
def train_step(x, y):
    """
    Args:
        x: (B, T, F) — B=batch, T=lookback, F=features
        y: (B, 1)    — B=batch, 1=regression target
    """
    ...
```

타입 힌트로는 표현이 어렵지만 docstring 으로라도 남기는 습관이 데이터 과학 코드의 품질을 크게 올립니다.

---

## §2. PyTorch 의 `batch_first` 관례

### 2.1 두 가지 축 순서

PyTorch 의 RNN 계열(`nn.RNN`, `nn.LSTM`, `nn.GRU`)은 입력 순서에 두 가지 선택지를 제공합니다.

| 설정 | 입력 shape | 해석 |
|---|---|---|
| `batch_first=False` (**기본값**) | `(T, B, F)` | Time 이 맨 앞 |
| `batch_first=True` | `(B, T, F)` | Batch 가 맨 앞 |

기본값이 `False` 인 이유는 역사적입니다. 초창기 RNN 구현에서 "시점별 반복" 을 맨 바깥 루프로 두는 게 메모리 접근에 유리했기 때문이에요. 하지만 **현대 대부분 튜토리얼과 실무 코드는 `batch_first=True`** 를 씁니다. Dataset/DataLoader 가 만들어 주는 자연스러운 순서가 `(B, T, F)` 이기 때문이에요.

### 2.2 전환 방법 — permute

두 형태 사이를 오갈 때는 `permute` 를 씁니다.

```python
x_btf = torch.zeros(32, 60, 17)           # (B, T, F)
x_tbf = x_btf.permute(1, 0, 2)            # (T, B, F)
#                     ↑  ↑  ↑
#                     원본의 축 1→새 축 0
#                     원본의 축 0→새 축 1
#                     원본의 축 2→새 축 2
```

`permute(new_order)` 는 **"원래 축 번호를 어떤 순서로 재배열할지"** 를 지정합니다. 숫자가 "원본 축 번호" 이지 "새 축 번호" 가 아닙니다. 헷갈리기 쉬우니 늘 shape 을 찍어 확인.

### 2.3 우리 프로젝트의 선택

**`batch_first=True`** 로 통일합니다. 이유:

1. DataLoader 가 자연스럽게 `(B, T, F)` 를 주므로 `permute` 호출이 사라짐 → 버그 감소.
2. `(B, T, F)` 는 "샘플 → 시간 → 변수" 순으로 읽기 쉬움 (영어 문장처럼 왼쪽→오른쪽으로 의미가 좁아짐).
3. 대부분 최신 논문 구현이 이 관례.

---

## §3. 축을 다루는 네 가지 도구 — `permute`, `reshape`/`view`, `unsqueeze`/`squeeze`, `expand`

### 3.1 한눈에 비교

| 함수 | 하는 일 | shape 변화 예시 | 데이터 재배치 | 메모리 공유 |
|---|---|---|---|---|
| `permute` | 축 순서 바꿈 | (B, T, F) → (T, B, F) | ❌ | ✅ (view) |
| `reshape` / `view` | 원소 재배치 없이 shape 만 재해석 | (B, T, F) → (B, T×F) | ❌ | 보통 ✅ |
| `unsqueeze(d)` | 축 하나 추가 (크기 1) | (B, F) → (B, 1, F) | ❌ | ✅ |
| `squeeze(d)` | 크기 1인 축 제거 | (B, 1, F) → (B, F) | ❌ | ✅ |
| `expand` | 크기 1 축을 반복 (실제 복사 X) | (B, 1, F) → (B, T, F) | ❌ | ✅ (broadcast view) |
| `contiguous()` | 메모리 상에 실제로 재배치 | shape 불변 | ✅ | ❌ (새 텐서) |

### 3.2 `permute` vs `reshape` — 가장 자주 혼동되는 쌍

**permute** 는 **축의 이름표를 바꾸는 것** 입니다. 원소는 그대로, "어느 축을 먼저 읽을지" 만 달라집니다.

**reshape** 은 **원소를 연속된 1차원으로 펼친 뒤 새 shape 로 잘라 담는 것** 입니다. "어떻게 읽을지의 경계" 를 바꿉니다.

```python
a = torch.arange(12).reshape(2, 3, 2)
# shape (2, 3, 2):
# [[[ 0,  1], [ 2,  3], [ 4,  5]],
#  [[ 6,  7], [ 8,  9], [10, 11]]]

# permute: 축만 재배치. 각 원소의 "좌표" 가 달라짐
a_p = a.permute(1, 0, 2)   # (3, 2, 2) — 원본의 [0,1,2] 축을 [1,0,2] 순으로
# a_p[i,j,k] == a[j,i,k]

# reshape: 원소를 평면으로 펴고 다시 자름
a_r = a.reshape(3, 2, 2)   # (3, 2, 2) — shape 은 같지만 내용이 다름!
# a_r[0] = [0,1, 2,3], a_r[1] = [4,5, 6,7], a_r[2] = [8,9, 10,11]
```

**같은 shape 이 나와도 내용이 완전히 다릅니다.** `(B, T, F)` 를 `(T, B, F)` 로 바꾸고 싶을 때 `reshape(T, B, F)` 를 쓰면 재앙이 됩니다. 축 순서 바꾸기는 **무조건 `permute`**.

### 3.3 `reshape` vs `view` — 미묘한 차이

둘 다 shape 재해석 함수인데 차이는:

- `view` : 메모리 상 **연속된 데이터만** 동작. `permute` 한 텐서에 바로 view 를 부르면 에러.
- `reshape` : 필요하면 내부적으로 `contiguous()` 를 호출해서 항상 동작. 안전하지만 가끔 보이지 않는 복사가 발생.

```python
a = torch.arange(12).reshape(3, 4)
b = a.permute(1, 0)           # (4, 3) — permute 결과는 non-contiguous
# b.view(12)                  # ❌ RuntimeError
b.reshape(12)                 # ✅ 내부에서 contiguous() 자동 호출
b.contiguous().view(12)       # ✅ 명시적으로 복사 후 view
```

**실무 지침**: 어지간하면 `reshape` 을 기본으로 쓰세요. `view` 는 성능이 극도로 중요한 루프 내부에서만.

### 3.4 `unsqueeze` / `squeeze` — 배치 축 끼워 넣기

모델은 항상 배치 축을 요구합니다. 샘플 한 개를 추론할 때도 `(1, T, F)` 로 만들어 넣어야 하죠.

```python
single_sample = torch.zeros(60, 17)         # (T, F) — 배치 축 없음
batched = single_sample.unsqueeze(0)        # (1, 60, 17) — 배치 축 추가
pred = model(batched)                       # (1, 1)
result = pred.squeeze()                     # scalar — 불필요한 축 제거
```

`unsqueeze(d)` 는 **d 번째 위치에 크기 1짜리 축을 삽입**. `squeeze(d)` 는 **d 번째 축이 크기 1이면 제거**(아니면 무시). 음수 인덱스도 가능: `unsqueeze(-1)` 은 맨 뒤에 추가.

---

## §4. Broadcasting — 축이 안 맞는데 연산이 되는 이유

### 4.1 규칙

NumPy/PyTorch 는 두 텐서의 shape 을 **오른쪽 정렬** 하고, 각 축이 다음 중 하나면 자동으로 "반복 복사된 것처럼" 연산합니다.

- 축 크기가 같다
- 한쪽 축 크기가 1 이다
- 한쪽에 해당 축이 없다

```
(32, 60, 17)  ← x
   (   ,  17)  ← bias
   ─────────
(32, 60, 17)  결과: bias 가 (1, 1, 17) 로 확장되어 모든 (B, T) 쌍에 더해짐
```

### 4.2 의도된 broadcasting vs 실수 broadcasting

**의도된 예시** — feature 별 편향 더하기:
```python
x    = torch.randn(32, 60, 17)
bias = torch.zeros(17)           # 피처별 상수
y    = x + bias                  # (32, 60, 17)
# → 모든 샘플의 모든 시점에 feature 별로 같은 bias 더해짐 ✅
```

**실수 예시** — shape 타이핑 오류:
```python
x       = torch.randn(32, 60, 17)
y_true  = torch.randn(32)        # 의도는 (B,) 였지만 실제로는 (B, 1) 이어야 함
loss    = (y_pred - y_true) ** 2
# y_pred.shape = (32, 1) 라면 y_true (32,) 와 broadcasting 되어 (32, 32) 가 됨
# → loss shape 이 괴상해져 있고, loss.mean() 값은 나오지만 의미가 엉뚱
```

**방어책**: 연산 전후로 `assert x.shape == (...)` 를 찍어 검증. 또는 loss 직전에 `y_true = y_true.view(-1, 1)` 같은 명시적 reshape.

---

## §5. 자주 하는 실수 다섯 가지

### 5.1 ❌ 축 순서 뒤바뀐 채로 GRU 에 통과

```python
x = torch.randn(32, 60, 17)
gru = nn.GRU(input_size=17, hidden_size=64)   # batch_first 미지정 = False 기본
out, h = gru(x)   # shape 은 맞아서 에러 안 남!
# 하지만 GRU 는 x 를 "T=32, B=60, F=17" 로 해석. 학습이 전혀 안 됨.
```

**예방**: GRU 만들 때 `batch_first=True` 를 **항상 명시**. 에러 대신 침묵하는 버그라 특히 위험.

### 5.2 ❌ `permute` 대신 `reshape` 으로 축 순서 변경

```python
x = torch.randn(32, 60, 17)
x_wrong = x.reshape(60, 32, 17)   # shape 은 (60, 32, 17) — 맞아 보임
x_right = x.permute(1, 0, 2)      # 진짜 축 교환
torch.allclose(x_wrong, x_right)  # False — 완전히 다른 텐서
```

**예방**: 축 순서만 바꾸면 `permute`. shape 이 완전히 달라지면 `reshape`. 둘을 구분하는 기준 = "원소의 좌표가 달라지는가".

### 5.3 ❌ Feature 축에 time pooling 적용

```python
x = torch.randn(32, 60, 17)
# 의도: 시간 평균 (각 피처별 평균) → (32, 17)
x_mean_wrong = x.mean(dim=2)   # ❌ 피처 평균이 나옴 → (32, 60)
x_mean_right = x.mean(dim=1)   # ✅ 시간 평균 → (32, 17)
```

**예방**: `dim=` 에 축 "위치" 를 쓸 때 기억하기: `(B, T, F)` 라면 B=0, T=1, F=2. 헷갈리면 shape 을 print.

### 5.4 ❌ Broadcasting 으로 의도하지 않은 차원 확장

```python
y_pred = model(x)          # (32, 1)
y_true = torch.tensor([0.01, 0.02, ...])   # (32,)
loss = ((y_pred - y_true) ** 2).mean()     # (32, 32) → mean → 잘못된 스칼라
```

**예방**: target 은 모델 출력과 같은 shape 으로. `y_true.view(-1, 1)` 또는 반대로 `y_pred.squeeze(-1)`.

### 5.5 ❌ Dataset `__getitem__` 이 배치 축까지 포함해서 반환

```python
class WrongDataset(Dataset):
    def __getitem__(self, idx):
        return self.X[idx:idx+T].unsqueeze(0)   # ❌ (1, T, F) 반환
# DataLoader 가 stack 하면 (B, 1, T, F) — 축 하나 초과
```

**예방**: `__getitem__` 은 **한 샘플만** 반환 (배치 축 없음). 배치 축은 DataLoader 가 만든다.

---

## §6. 정리 — 우리 프로젝트의 축 계약

```
3D 텐서  : 항상 (B, T, F)    ← 모든 곳에서 통일
2D 텐서  : Dataset __getitem__ 반환값 (T, F) — 배치 전
1D 텐서  : 라벨 스칼라 or 피처별 bias
scalar  : loss 등

축 이름 고정:
  B = Batch size        (배치 축)   shuffle 가능
  T = Time steps        (시간 축)   순서 절대 유지
  F = Features          (변수 축)   이름 순서 고정

GRU 호출 시: nn.GRU(..., batch_first=True)    # 항상
```

이 계약 한 페이지가 앞으로 3~4주차의 GRU 구현에서 나올 모든 shape 버그의 **체크리스트** 가 됩니다.

---

## §7. 자가 점검 질문

실습 전 먼저 답을 생각해보세요. 정답은 실습 Step 8 에서 확인합니다.

1. `(32, 60, 17)` 텐서에 `.mean(dim=1)` 을 부르면 결과 shape 은? 무엇의 평균인가?
2. `(B, T, F)` 를 `nn.GRU(batch_first=False)` 에 넣으면 GRU 는 이 텐서를 어떻게 해석하는가?
3. `x.permute(1, 0, 2)` 와 `x.reshape(T, B, F)` 의 결과 shape 은 같지만 **내용이 다릅니다**. 어느 쪽이 우리가 원하는 축 교환인가?
4. 샘플 1개 `(60, 17)` 를 모델에 넣으려면 어떻게 변환해야 하는가? 그리고 모델 출력 `(1, 1)` 에서 스칼라 값을 꺼내는 방법은?
5. `x.shape = (32, 60, 17)`, `bias.shape = (60,)` 일 때 `x + bias` 는 가능한가? 된다면 bias 가 어떻게 확장되는가? 이것이 의도한 결과인가?

---

## §8. 다음 토픽 예고

2주차는 여기까지입니다. 3주차부터는 실제 시퀀스 모델을 다룹니다.

- 3.1 RNN 의 기본 구조와 한계 (vanishing gradient)
- 3.2 LSTM 의 게이트 메커니즘
- 3.3 GRU — LSTM 의 간소화, 실무에서 더 많이 쓰는 이유
- 3.4 PyTorch 학습 루프 표준 패턴
- 3.5 5분위 확률 출력 구성
- 3.6 학습 안정화 기법 (Dropout, Early Stopping, LR Scheduler)

2주차에서 만든 `SequenceDataset` + `(B, T, F)` 축 계약이 그대로 재사용됩니다.
