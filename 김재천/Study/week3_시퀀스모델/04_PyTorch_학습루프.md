# 3.4 PyTorch 학습 루프 표준 패턴

> **이전 토픽 연결**
> - 3.1~3.3에서 RNN/LSTM/GRU의 **구조**를 배웠습니다.
> - 이 토픽은 **그 구조를 어떻게 학습시키는가** — 실제 학습 루프의 표준 패턴을 다룹니다.
> - 여기서 배우는 패턴은 **모든 PyTorch 모델에 공통** 적용되는 골격입니다.

---

## 목차

1. 학습의 큰 그림 — 4단계 사이클
2. `forward` — 예측 생성
3. `loss` — 얼마나 틀렸나 측정
4. `backward` — 어디를 얼마나 고쳐야 하나 계산
5. `optimizer.step` — 실제 가중치 업데이트
6. `zero_grad` — 왜 매번 초기화하는가
7. `train()` vs `eval()` 모드 — 왜 구분하는가
8. `with torch.no_grad()` — 검증 시 필수
9. Device 관리 — `.to(device)`의 규칙
10. Loss 함수 선택 — 과제별 대응표
11. Optimizer 선택 — SGD, Adam, AdamW
12. Gradient Clipping — 시퀀스 모델의 안전장치
13. Learning Rate Scheduler — 학습률 스케줄링
14. Checkpoint 저장/로드
15. 학습 곡선 모니터링
16. **경량 학습 반복 패턴 — Optuna 최적화를 위한 준비** ⭐
17. 우리 프로젝트의 표준 학습 루프 템플릿
18. 흔한 실수 10가지
19. 부록: 왜 `.item()` / `.detach()`를 써야 하는가

---

## 1. 학습의 큰 그림 — 4단계 사이클

모든 PyTorch 학습 루프는 **매 배치마다 다음 4단계**를 반복합니다.

```
┌─────────────────────────────────────────────┐
│  for batch in dataloader:                   │
│                                             │
│    ① forward:   y_pred = model(x)           │
│                 (입력 → 예측)                │
│                                             │
│    ② loss:      L = criterion(y_pred, y)    │
│                 (예측 vs 정답 차이)          │
│                                             │
│    ③ backward:  L.backward()                │
│                 (gradient 계산 역전파)       │
│                                             │
│    ④ step:      optimizer.step()            │
│                 (가중치 업데이트)            │
│                                             │
│    cleanup:     optimizer.zero_grad()       │
│                 (다음 배치 전 gradient 초기화)│
└─────────────────────────────────────────────┘
```

**비유**: 양궁 연습을 생각해봅시다.
- ① **forward** = 활을 쏘고 화살이 어디 꽂히는지 본다 (예측)
- ② **loss** = "중앙에서 얼마나 빗나갔나" 측정 (오차)
- ③ **backward** = "어깨, 팔꿈치, 손목 각도를 어떻게 고쳐야 중앙에 맞을까" 분석 (gradient)
- ④ **step** = 실제로 자세를 미세하게 조정 (가중치 업데이트)

이 4단계를 수십만 번 반복하면서 조금씩 더 정확해지는 게 학습입니다.

---

## 2. `forward` — 예측 생성

```python
y_pred = model(x)
```

`model(x)`를 호출하면 PyTorch가 내부적으로 `model.forward(x)`를 실행하고, **동시에 계산 그래프(computational graph)를 자동으로 구성**합니다.

### 계산 그래프란

모든 연산을 **노드**로 기록한 방향 그래프입니다. 예를 들어:
```python
h = lstm(x)           # 노드 1
y_pred = linear(h)    # 노드 2
L = loss(y_pred, y)   # 노드 3
```

PyTorch가 뒤에서 `L ← linear ← lstm ← x` 라는 연결 관계를 **자동 기억**합니다. 이 그래프가 있기 때문에 나중에 `L.backward()` 한 줄로 **모든 파라미터의 gradient가 역방향으로 자동 계산**됩니다.

### "자동 미분(autograd)"의 실체

- `requires_grad=True`인 텐서를 연산하면 그래프에 기록됨
- 모델 파라미터(`nn.Parameter`)는 자동으로 `requires_grad=True`
- 입력 `x`는 기본 `requires_grad=False` (gradient 계산 안 함)
- `y_pred.grad_fn`을 찍어보면 그래프의 마지막 연산 노드가 보임

---

## 3. `loss` — 얼마나 틀렸나 측정

```python
criterion = nn.CrossEntropyLoss()  # 분류 과제
L = criterion(y_pred, y_true)
```

### 핵심 원칙

- **scalar** 여야 함 (숫자 한 개)
- **미분 가능** 해야 함 (backward가 동작하려면)
- **낮을수록 좋음** — 학습은 이 값을 최소화하는 방향

### 과제별 표준 loss

| 과제 유형 | loss 함수 | 비고 |
|---|---|---|
| 회귀 (숫자 예측) | `nn.MSELoss()` | (y_pred - y)^2 평균 |
| 이진 분류 | `nn.BCEWithLogitsLoss()` | sigmoid + BCE 합친 안정 버전 |
| 다중 분류 | `nn.CrossEntropyLoss()` | softmax + NLL 합친 안정 버전 |
| 분포 차이 | `nn.KLDivLoss()` | KL divergence |

### 주의: "로짓(logits)"을 바로 넣으세요

`nn.CrossEntropyLoss`와 `nn.BCEWithLogitsLoss`는 **내부에서 softmax/sigmoid를 적용**합니다. 모델 출력을 그대로 넣으면 됩니다. 실수로 softmax를 한 번 더 거쳐서 넣으면 **이중 softmax**가 돼서 학습이 제대로 안 됩니다.

```python
# ❌ 틀림 — 이중 softmax
y_pred_prob = torch.softmax(model(x), dim=-1)
L = nn.CrossEntropyLoss()(y_pred_prob, y)

# ✅ 맞음 — 로짓 그대로 전달
y_pred_logits = model(x)
L = nn.CrossEntropyLoss()(y_pred_logits, y)
```

---

## 4. `backward` — 어디를 얼마나 고쳐야 하나 계산

```python
L.backward()
```

한 줄로 **모든 파라미터의 gradient를 자동 계산**합니다.

### 내부에서 일어나는 일

계산 그래프를 **역방향**으로 순회하면서, 각 노드에 대해:
```
∂L/∂w = ∂L/∂y_pred · ∂y_pred/∂h · ∂h/∂w  ← chain rule
```
를 자동 계산해서 `w.grad` 속성에 저장합니다.

### 확인 방법

```python
L.backward()
for name, param in model.named_parameters():
    print(name, param.grad.abs().mean().item())
```
각 파라미터의 gradient 크기를 찍어볼 수 있습니다. **학습이 안 될 때 제일 먼저 확인해야 할 것** — gradient가 `1e-10` 이하면 소실, `1e+3` 이상이면 폭발.

### 주의: gradient는 **누적**됨

```python
L1.backward()  # w.grad = g1
L2.backward()  # w.grad = g1 + g2  ← 누적!
```

이걸 **의도적으로 쓰는 경우** (gradient accumulation — 큰 배치를 쪼개 처리할 때)도 있지만, **기본적으로는 매 배치마다 초기화**해야 합니다. 이게 다음 단계의 `zero_grad`.

---

## 5. `optimizer.step` — 실제 가중치 업데이트

```python
optimizer.step()
```

각 파라미터에 대해:
```
w ← w − learning_rate · w.grad
```
(기본 SGD). Adam, AdamW 같은 고급 optimizer는 더 복잡한 공식을 씁니다.

### Optimizer의 실제 역할

- 어떤 파라미터를 업데이트할지: `optim.Adam(model.parameters(), lr=0.001)` 에서 지정
- 얼마만큼 업데이트할지: `learning_rate` + 과거 gradient 이력(momentum 등)
- **업데이트 규칙 자체는 optimizer가 정함** — 모델은 gradient만 제공

### 여러 학습률을 섞고 싶을 때

```python
optimizer = optim.Adam([
    {'params': model.lstm.parameters(), 'lr': 1e-4},     # LSTM 본체는 낮게
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # 분류기 헤드는 높게
])
```

Fine-tuning 시 유용.

---

## 6. `zero_grad` — 왜 매번 초기화하는가

```python
optimizer.zero_grad()  # 또는 model.zero_grad()
```

앞서 말했듯 gradient는 누적됩니다. 배치마다 초기화하지 않으면:
- 배치 1의 gradient + 배치 2의 gradient + ... 가 계속 쌓임
- 파라미터가 "모든 과거 배치의 평균 방향"으로 쏠림
- 학습이 불안정해지고 사실상 망가짐

### 어디에 두어야 하는가

두 가지 관례가 있습니다.

```python
# 관례 1: 배치 시작 시 (가장 흔함)
for batch in dataloader:
    optimizer.zero_grad()
    ...
    loss.backward()
    optimizer.step()

# 관례 2: step 직후
for batch in dataloader:
    ...
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

둘 다 맞습니다. 개인 취향. 우리 프로젝트에선 **관례 1** 사용 (배치 시작 시 초기화가 더 직관적).

### `set_to_none=True` 팁

```python
optimizer.zero_grad(set_to_none=True)
```
0으로 채우는 대신 **`None`으로 설정**하면 메모리 사용량 약간 감소 + 다음 backward가 살짝 빠름. PyTorch 1.7+에서 권장 기본값.

---

## 7. `train()` vs `eval()` 모드 — 왜 구분하는가

```python
model.train()   # 훈련 모드
model.eval()    # 평가 모드
```

### 이 한 줄이 바꾸는 것

**Dropout과 BatchNorm의 동작이 달라집니다.**

| 레이어 | `train()` | `eval()` |
|---|---|---|
| `nn.Dropout(p)` | 확률 p로 랜덤하게 0으로 만듦 | 아무것도 안 함 |
| `nn.BatchNorm1d` | 현재 배치의 mean/var 사용 + 이동 평균 업데이트 | 저장된 running mean/var 사용 |

### 왜 중요한가

**학습 시**에는 dropout으로 일부 뉴런을 꺼서 일반화 성능을 높입니다. **추론 시**에는 모든 뉴런을 쓰고 싶죠. `eval()`을 안 해주면:
- 검증 loss가 들쭉날쭉 (매번 랜덤하게 뉴런이 꺼져서)
- 실제 성능보다 낮게 측정됨
- 재현 불가능한 결과

**반드시 지켜야 할 규칙**:
```python
# 훈련 루프
model.train()
for batch in train_loader:
    ...

# 검증 루프
model.eval()
with torch.no_grad():
    for batch in val_loader:
        ...

# 추론 (실제 예측)
model.eval()
with torch.no_grad():
    y_pred = model(x_new)
```

---

## 8. `with torch.no_grad()` — 검증 시 필수

```python
with torch.no_grad():
    y_pred = model(x)
```

### 무엇을 하는가

이 블록 안에서는 **계산 그래프를 구성하지 않습니다.** 즉:
- gradient 계산 불가
- 메모리 사용량 크게 감소 (그래프 안 저장하니까)
- 연산 속도 약간 상승

### 왜 필수인가

검증(validation)과 테스트(test), 실제 추론(inference) 시에는 **gradient가 필요 없습니다.** backward를 안 할 거니까요. 그런데 `torch.no_grad()` 없이 forward를 돌리면:
- 계산 그래프가 메모리에 계속 쌓임
- 큰 검증 데이터셋에서 **OOM(out of memory) 발생**

### `eval()`과 `no_grad()`의 차이

헷갈리는 둘의 차이:
- `model.eval()` = **레이어 동작 모드 변경** (Dropout off, BatchNorm 고정)
- `torch.no_grad()` = **gradient 추적 중지** (메모리/속도)

**둘은 독립적이고 둘 다 필요합니다.** 검증/추론 시에는 항상 둘 다 써야 합니다.

```python
model.eval()                    # ← Dropout off
with torch.no_grad():           # ← gradient 추적 off
    y_pred = model(x_val)
```

---

## 9. Device 관리 — `.to(device)`의 규칙

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
x, y = x.to(device), y.to(device)
```

### 규칙 한 줄 요약

**모델과 입력이 같은 device에 있어야 연산 가능.**

### 실무 패턴

```python
# 1. device 결정 (스크립트 상단에 한 번)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 2. 모델 이동 (모델 생성 직후)
model = MyModel().to(device)

# 3. optimizer는 모델을 device로 옮긴 **후**에 생성
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. 배치마다 입력을 device로 이동
for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    ...
```

### 흔한 실수

```python
# ❌ 틀림 — optimizer를 먼저 만들고 모델을 device로 옮기면 
# optimizer가 CPU 텐서를 참조하고 있음
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model = model.to(device)  # 이후 옮기면 optimizer가 고장

# ✅ 맞음 — 모델 먼저 device로, optimizer 나중에
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

---

## 10. Loss 함수 선택 — 과제별 대응표

### 우리 프로젝트 맥락

현재 XGBoost가 **5분위 확률**(proba[5])을 출력합니다. 본 3.4 이론은 **classification 예제를 기준**으로 설명하며, 이때 loss는 다음과 같습니다:
- 출력: 5개 클래스에 대한 확률 분포
- Loss: `nn.CrossEntropyLoss()`

> **COL-BL 전환 노트**: 본 프로젝트의 최종 모델 **COL-BL (CEEMDAN + Optuna-LSTM + 비선형 앙상블 + Black-Litterman)** 은 **수익률 회귀**를 수행하므로 최종 출력 형식이 다릅니다:
> - 출력: 스칼라 수익률 예측 ŷ ∈ ℝ
> - Loss: `nn.MSELoss()` 또는 `nn.HuberLoss(delta=1.0)` (4.5 실습에서 상세)
> - 각 IMF별 Optuna-LSTM도, LSTM 비선형 앙상블 meta-learner도 모두 이 회귀 loss를 공유
>
> 다만 **학습 루프의 골격(§1~§18)은 분류/회귀 무관하게 동일**하므로, 본 이론은 CrossEntropy 예제로 일관되게 진행합니다. 회귀 전환은 criterion 교체 + 마지막 Linear 층 출력 차원만 바꾸면 되는 기계적 변경입니다.

### CrossEntropyLoss의 수학

```
L = −log(softmax(y_pred)[y_true])
```

즉 "정답 클래스의 확률이 1에 가까워지게, 나머지는 0에 가까워지게" 하는 방향.

### 레이블 형식 주의

```python
# ✅ 맞음 — 정답은 클래스 인덱스 (정수)
y_true = torch.tensor([0, 2, 4, 1])  # shape (B,)
y_pred = model(x)  # shape (B, 5)
L = nn.CrossEntropyLoss()(y_pred, y_true)

# ❌ 틀림 — one-hot 쓰지 말 것 (nn.CrossEntropyLoss는 인덱스 받음)
y_true_onehot = torch.tensor([[1,0,0,0,0], [0,0,1,0,0], ...])
```

### 가중 loss (class imbalance 대응)

우리 프로젝트의 5분위는 어느 정도 균등하지만, 만약 특정 분위에 샘플이 몰려 있다면:
```python
class_weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0]).to(device)  # 3분위 샘플이 적다고 가정
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## 11. Optimizer 선택 — SGD, Adam, AdamW

### 3가지 선택지 요약

| Optimizer | 특징 | 언제 쓰나 |
|---|---|---|
| `SGD` | 단순, 수렴 천천히, 잘 튜닝하면 최고 | CNN, 큰 데이터셋 |
| `Adam` | 적응적 학습률, 빠른 수렴 | 거의 모든 경우 기본 선택 |
| `AdamW` | Adam + 올바른 weight decay | Transformer, 현대 권장 기본값 |

### 우리 프로젝트 권장: **AdamW**

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,  # L2 정규화 — 과적합 방지
    betas=(0.9, 0.999),  # 기본값
)
```

### weight_decay란

**파라미터 크기에 페널티**. 큰 가중치에 벌점을 줘서 **모델이 과도하게 복잡해지는 것 방지**. 금융 시계열처럼 **노이즈 많고 과적합 위험 큰** 데이터에 특히 중요.

### Adam vs AdamW 차이

수학적으로 미묘하지만 중요한 차이가 있습니다. Adam의 weight decay는 gradient에 녹여서 적용되는데, 이게 learning rate scaling과 상호작용하면서 제대로 안 먹히는 경우가 있습니다. AdamW는 **weight decay를 gradient와 분리해서 직접 적용**. 일반적으로 AdamW가 정규화가 더 잘 됩니다.

---

## 12. Gradient Clipping — 시퀀스 모델의 안전장치

### 왜 시퀀스 모델에 특히 필수인가

3.1~3.3에서 봤듯이 RNN 계열은 **gradient 폭발 위험**이 항상 있습니다:
- Vanilla RNN σ=1.5에서 T=60 gradient가 수천~수만 단위
- LSTM도 특정 상황에서 폭발 가능
- 한 번의 폭발 = `optimizer.step()`으로 파라미터가 NaN으로 날아감 → 학습 복구 불가

### 해결: gradient clipping

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← 한 줄
optimizer.step()
```

**작동 원리**: 모든 파라미터의 gradient를 하나의 벡터로 보고 L2 norm을 계산. 이 norm이 `max_norm`(=1.0)을 넘으면 **비율에 따라 축소**.

```
if ||grad||_2 > max_norm:
    grad ← grad · (max_norm / ||grad||_2)
```

방향은 유지하고 크기만 자름 — 폭발만 막고 학습 방향은 보존.

### max_norm 선택

- **1.0** — 보수적, 시퀀스 모델 기본값
- **5.0** — 느슨, Transformer에서 자주
- **0.5** — 매우 보수적, 학습 불안정할 때

### 위치 주의

```python
# ✅ 맞음 — backward 후, step 전
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# ❌ 틀림 — step 이후에 clip해봤자 이미 업데이트됨
loss.backward()
optimizer.step()
torch.nn.utils.clip_grad_norm_(...)
```

### `clip_grad_norm_` vs `clip_grad_value_`

- `clip_grad_norm_` (권장): 전체 gradient 벡터의 norm을 제한 → 방향 보존
- `clip_grad_value_` (비권장): 각 gradient 값을 독립적으로 자름 → 방향 왜곡

---

## 13. Learning Rate Scheduler — 학습률 스케줄링

### 왜 학습률을 바꾸는가

- **초기**: 학습률을 크게 → 최소값 근처로 빠르게 이동
- **후기**: 학습률을 작게 → 정밀 조정, overshoot 방지

고정 학습률보다 스케줄링이 일반적으로 **최종 성능 2~5% 향상**.

### 대표 스케줄러 3가지

```python
# (1) StepLR — 일정 epoch마다 감소
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# 10 epoch마다 lr = lr * 0.5

# (2) CosineAnnealingLR — 코사인 곡선 감소 (현대 권장)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# 50 epoch 동안 cos 곡선으로 0까지 감소

# (3) ReduceLROnPlateau — 검증 loss 정체 시 감소
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# val_loss가 5 epoch 연속 개선 없으면 lr *= 0.5
```

### 호출 위치

```python
# StepLR, CosineAnnealingLR: 매 epoch 끝
for epoch in range(num_epochs):
    train(...)
    validate(...)
    scheduler.step()   # ← epoch 끝에 한 번

# ReduceLROnPlateau: 검증 metric 인자 필요
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)   # ← 검증 loss를 인자로
```

### 우리 프로젝트 권장

**ReduceLROnPlateau** — Walk-Forward 검증에서 매 fold마다 학습 길이가 다를 수 있으므로, **검증 loss 기반 적응적 감소**가 가장 안전.

---

## 14. Checkpoint 저장/로드

### 언제 저장하는가

- 매 epoch 끝 (매번 저장하면 디스크 낭비, 보통 best-val 갱신 시에만)
- 학습 완료 시
- 긴 학습 중간 안전장치 (장애 복구용)

### 표준 저장 형식

```python
# 저장
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
}, 'checkpoint.pt')

# 로드
checkpoint = torch.load('checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### `state_dict()`만 저장하는 이유

**전체 모델 객체를 저장(`torch.save(model, ...)`)**하면 원본 코드 구조에 강하게 의존 → 나중에 클래스 이름 바뀌면 로드 불가. **state_dict만 저장**하면 클래스와 분리된 상태(가중치 텐서들)만 저장 → 코드 변경에 강건.

### Best model 저장 패턴

```python
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f'New best! epoch={epoch}, val_loss={val_loss:.4f}')
```

**검증 성능이 개선될 때만** 모델을 저장 → 최종적으로 `best_model.pt`는 검증 기준 최고 모델.

---

## 15. 학습 곡선 모니터링

### 매 epoch 기록할 것

```python
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'lr': [],
}

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])
```

### 학습 곡선 해석 가이드

학습 곡선을 보면 상태를 진단할 수 있습니다:

| 패턴 | 진단 | 대응 |
|---|---|---|
| train↓, val↓ 함께 감소 | **정상 학습** | 계속 진행 |
| train↓, val↑ | **과적합** | Dropout ↑, weight_decay ↑, Early Stopping |
| train↑, val↑ (둘 다 불안정) | **학습률 너무 높음** | lr ↓, gradient clipping |
| train 평평, val 평평 (높은 상태) | **모델 용량 부족** | hidden 크기 ↑, layers ↑ |
| train 평평, val 평평 (낮은 상태) | **수렴 완료** | Early stop |

### 시각화

```python
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
ax1.set_title('Loss'); ax1.legend()
ax2.plot(history['train_acc'], label='Train'); ax2.plot(history['val_acc'], label='Val')
ax2.set_title('Accuracy'); ax2.legend()
plt.show()
```

---

## 16. 경량 학습 반복 패턴 — Optuna 최적화를 위한 준비 ⭐

### 왜 이 절을 별도로 다루는가

§1~15까지는 **하나의 모델을 한 번 학습**하는 표준 패턴이었습니다. 하지만 4주차부터 다룰 **Optuna-LSTM (Optuna TPE 기반 hyperparameter 최적화)** 은 이 학습 루프를 **n_trials 회 반복 호출** 합니다 (본 프로젝트 기준 IMF당 30~50회, CEEMDAN이 만드는 7~9개 IMF를 합치면 200~450회).

같은 코드를 수십~수백 번 돌리면 **시간 · 메모리 · 재현성** 세 가지가 급격히 나빠지므로, hyperparameter 탐색용 학습 함수(= Optuna objective 내부)는 **§17의 표준 템플릿과는 다른 설계 원칙**을 따릅니다. 여기서는 그 차이를 사전 정리합니다.

> **참고 — COL-BL은 논문 GA-LSTM의 Optuna 변형**: 본 프로젝트(COL-BL)는 원 논문(Su/Lu/Yen 2026)의 **CGL-BL (GA-LSTM 기반)** 을 **Optuna-LSTM으로 치환한 동등 변형**입니다. GA와 Optuna는 탐색 알고리즘만 다르고 *"한 개체(trial)당 빠른 경량 학습 → scalar 지표 반환"* 라는 인터페이스는 완전히 동일합니다. 따라서 §16에서 정의할 `train_one_individual` 함수는 **GA의 fitness 함수이자 Optuna의 objective 함수로 그대로 재사용**됩니다 (§16.3 말미에 변환 예시).

### 16.1 계산 예산 설계 — "한 번 학습"을 얼마나 줄일 것인가

Optuna-LSTM의 총 학습량은 다음 곱으로 결정됩니다 (단일 IMF 기준):

```
total_train_calls = n_trials × epochs_per_trial
                  = 50 × (50 ~ 100)    ← §17의 표준 템플릿 그대로면   2,500 ~ 5,000 epoch
                  = 50 × (5 ~ 10)       ← 경량화하면                    250 ~   500 epoch
```

**10배 차이.** 그리고 CEEMDAN이 보통 7~9개 IMF를 만들어내므로 실제로는 여기에 다시 ×8배(자산당) + ×N_assets(포트폴리오)가 곱해집니다. 따라서 objective 내부에서는 **epoch를 크게 줄이고 early stopping을 공격적으로** 쓰는 게 표준 관례입니다.

> **Optuna가 예산 측면에서 유리한 두 가지 장치**:
> - **MedianPruner**: 진행 중인 trial이 다른 trial들의 중간값보다 나쁘면 중간 epoch에서 조기 종료(prune). 즉 epoch-level early stopping에 더해 **trial-level pruning** 이 중첩 적용됩니다.
> - **TPE sampler**: 과거 trial 결과 분포를 학습해 "유망한 영역"을 집중 탐색 — 같은 n_trials 예산으로 Random search보다 훨씬 빨리 수렴합니다.
>
> 두 장치가 함께 작동하면 **실제 계산량은 이론치의 1/2 ~ 1/5 수준**으로 떨어지는 게 일반적입니다 (Akiba et al., 2019).

### 16.2 Early Stopping — 조기종료로 예산 회수

```python
best_val_loss = float('inf')
patience_counter = 0
PATIENCE = 3   # 3 epoch 연속 개선 없으면 stop

for epoch in range(max_epochs):
    train_one_epoch(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break   # 조기종료 — 남은 epoch 예산 회수
```

Trial당 최대 10 epoch 예산이어도, 대부분 3~5 epoch에서 조기종료되므로 **평균 실제 학습량은 절반 이하**로 내려갑니다. (Optuna의 MedianPruner는 여기에 trial-level 조기 종료를 한 번 더 얹는 구조입니다 — §16.1 참조.)

### 16.3 Fitness / Objective Function — "이 hyperparam이 얼마나 좋은가" 반환

GA든 Optuna든 각 개체/trial(= 한 개의 hyperparam 조합)마다 이 함수를 호출해서 **scalar 숫자 하나**를 받습니다. 이 **단일 인터페이스**가 §16 설계의 핵심 — 탐색 알고리즘을 바꿔도 이 함수는 손대지 않습니다:

```python
def train_one_individual(params: dict, train_loader, val_loader, device,
                         seed: int = 42) -> float:
    """GA·Optuna 공통으로 호출하는 학습 함수. 주어진 params로 LSTM을 학습 후
    best val_loss(낮을수록 좋음)를 반환. 반환값이 그대로:
      - GA의 fitness (minimize)
      - Optuna objective의 반환값 (direction='minimize')
    """
    # --- 재현성 확보 ---
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- 모델 빌드 ---
    model = build_lstm(
        input_size=params['input_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
    ).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=params['lr'],
                            weight_decay=params.get('weight_decay', 1e-4))
    criterion = nn.CrossEntropyLoss()
    # NOTE (COL-BL 회귀 전환): 본 3.4 이론은 classification 예제를 유지합니다.
    #   실제 COL-BL은 수익률 회귀이므로 4.5 실습부터는
    #     criterion = nn.MSELoss()   또는   nn.HuberLoss(delta=1.0)
    #   으로 교체합니다. train/val 루프 구조는 동일.

    # --- 경량 학습 + early stopping ---
    best_val_loss = float('inf')
    patience = 0
    for epoch in range(params.get('max_epochs', 10)):
        # train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # validate
        model.eval()
        val_loss_sum, n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss_sum += criterion(model(x), y).item() * x.size(0)
                n += x.size(0)
        val_loss = val_loss_sum / n
        # early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break

    # --- 메모리 정리 (16.4 참조) ---
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_loss   # 낮을수록 좋음 → GA는 이 값을 minimize
```

**반드시 지킬 5가지 규약**:

| 규약 | 이유 |
|---|---|
| 반환값은 **scalar float** | GA의 fitness는 숫자 하나여야 선택·정렬 가능 |
| **Best val_loss** 반환 | 마지막 epoch loss는 최저가 아닐 수 있음 |
| Minimization 관례 | Sharpe ratio 같은 maximize 지표면 `-sharpe` 반환 |
| Seed 통제 | 같은 hyperparam → 같은 fitness (공정 비교) |
| 메모리 정리 | 수백 번 호출 사이 누수 방지 |

#### Optuna objective로의 자연스러운 변환

위 `train_one_individual` 함수는 Optuna `objective(trial)` 함수로 **10줄 안팎의 얇은 래퍼만 씌우면** 전환됩니다. 원 함수(§16.3)는 **건드리지 않는 것이 핵심** — 덕분에 GA ↔ Optuna 간 비교 실험도 쉬워집니다.

```python
import optuna

def objective(trial: optuna.Trial) -> float:
    # Optuna는 각 trial마다 suggest_* 호출로 hyperparam 조합을 '제안'합니다.
    # 제안된 값은 params dict로 모아서 기존 학습 함수에 그대로 전달.
    params = {
        'input_size':   INPUT_SIZE,                                         # 데이터 고정
        'hidden_size':  trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'num_layers':   trial.suggest_int('num_layers', 1, 3),
        'dropout':      trial.suggest_float('dropout', 0.0, 0.5),
        'lr':           trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'weight_decay': 1e-4,
        'max_epochs':   10,
    }
    # 기존 함수 그대로 재사용 — 반환값(best_val_loss)이 Optuna의 minimize 대상이 됩니다.
    return train_one_individual(params, train_loader, val_loader, device)

study = optuna.create_study(
    direction='minimize',                                  # best_val_loss를 낮추는 방향
    sampler=optuna.samplers.TPESampler(seed=42),           # 재현성 확보된 TPE
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5) # 가망 없는 trial 조기 종료
)
study.optimize(objective, n_trials=50)

best_params = study.best_params      # dict: hyperparam 조합
best_value  = study.best_value       # float: 최소 val_loss
```

**핵심 관찰 3가지**:
1. `train_one_individual` 은 **GA든 Optuna든 그대로 재사용** — 이 호환성이 §16 설계의 본질.
2. `trial.suggest_*` 종류만 바꾸면 탐색 공간이 자동으로 업데이트. **TPE sampler**가 과거 trial 결과 분포를 보고 다음 trial을 지능적으로 제안하고, **MedianPruner**가 가망 없는 trial을 중간에 종료합니다.
3. 회귀 예측(COL-BL)으로 넘어갈 때는 §16.3 함수 내부의 `criterion` 만 `nn.MSELoss()` 또는 `nn.HuberLoss(delta=1.0)` 로 교체하면 되고, **objective 쪽은 손대지 않아도 됩니다** (4.5 실습에서 상세).

> **Walk-Forward와 결합 규칙 (중요)**: Optuna 튜닝은 **초기 training 구간에서 1회만** 실행하고, 얻은 `best_params` 를 모든 fold에 재사용해야 합니다. fold마다 재튜닝하면 **look-ahead bias** (미래 validation 정보를 hyperparam 선택에 쓰는 오염)가 발생합니다. 자세한 내용은 6주차 Walk-Forward 섹션에서 다룹니다.

### 16.4 메모리 관리 — 누적 누수 방지

PyTorch는 GPU 메모리를 **명시적으로 해제하지 않으면 계속 갖고 있습니다.** GA처럼 모델을 수백 번 만들었다 버리는 시나리오에서는 매 개체 끝에 반드시:

```python
del model, optimizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

이걸 생략하면 세대가 진행될수록 가용 VRAM이 줄어들어 결국 OOM으로 GA가 중단됩니다.

### 16.5 재현성 — 같은 hyperparam이 같은 fitness를 내야 공정

```python
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

같은 hyperparam으로 두 번 돌렸을 때 fitness가 다르면 GA의 **선택(selection) 과정이 노이즈에 오염**됩니다. 초기화 랜덤성, 배치 shuffle 순서 등이 모두 seed에 묶여야 합니다.

**주의**: DataLoader의 `shuffle=True`도 seed에 따라 달라집니다. `torch.Generator()`로 seed 주입하거나, 각 개체 전에 DataLoader를 재생성하는 게 안전합니다.

### 16.6 캐싱 (선택) — 동일 hyperparam 재평가 방지

GA의 elitism(엘리트 보존)에서는 상위 개체가 여러 세대에 걸쳐 살아남고, Optuna의 TPE도 유망 영역을 집중 탐색하다 보면 동일 또는 거의 동일한 hyperparam 조합을 재평가할 수 있습니다. 같은 hyperparam을 또 학습시키는 건 낭비이므로:

```python
import hashlib, json

_cache = {}   # 전역 또는 클래스 내부에 둠

def train_one_individual_cached(params, **kwargs):
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    result = train_one_individual(params, **kwargs)
    _cache[key] = result
    return result
```

**단 조건**: seed가 개체마다 고정(공유)돼야 안전. 개체별로 seed를 바꾸고 있다면 캐싱 금물 (같은 params여도 다른 결과가 정당).

### 요약 — 표준 템플릿 vs GA/Optuna 공통 trial 학습 함수

| 항목 | §17 표준 템플릿 | §16 Trial용 학습 함수 (GA/Optuna 공통) |
|---|---|---|
| Epoch 수 | 50 ~ 100 | **5 ~ 10** |
| Early stopping | 선택 | **필수** |
| 반환값 | (학습된 모델 + 메트릭 기록) | **scalar fitness (best val_loss)** |
| 메모리 정리 | 생략 가능 | **필수 (`del` + `empty_cache`)** |
| Seed 통제 | 권장 | **필수** |
| Checkpoint 저장 | 매 best-val 갱신 시 | **미저장** (수백 개체 중 하나일 뿐) |
| Learning curve 시각화 | 필수 | **생략** (로그만 반환, 시각화는 GA 종료 후) |

**4주차 4.3에서 이 `train_one_individual` 함수를 그대로 사용합니다.** Optuna study는 이 함수를 **`n_trials` (본 프로젝트 기준 30~50)** 회 호출해서 각 hyperparam 조합의 `best_val_loss` 를 얻고, **TPE sampler**가 과거 trial 결과를 보고 다음 trial을 지능적으로 제안 + **MedianPruner**가 가망 없는 trial을 조기 종료합니다. 여기서 §16을 미리 이해하고 넘어가면 4.3 Optuna 코드를 처음 봐도 낯설지 않습니다 (그리고 논문 CGL-BL과의 비교 실험을 하고 싶을 때 GA도 같은 함수 하나로 돌릴 수 있습니다 — 인터페이스가 동일하므로).

---

## 17. 우리 프로젝트의 표준 학습 루프 템플릿

위의 모든 개념을 하나의 템플릿으로 정리:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ========== 준비 단계 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 → optimizer → scheduler 순서로 생성
model = MyLSTMClassifier(input_size=17, hidden_size=64, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

history = {'train_loss': [], 'val_loss': [], 'lr': []}
best_val_loss = float('inf')

# ========== 학습 루프 ==========
for epoch in range(num_epochs):

    # --- 훈련 단계 ---
    model.train()
    train_loss_sum = 0.0
    train_n = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_sum += loss.item() * x.size(0)
        train_n += x.size(0)

    train_loss = train_loss_sum / train_n

    # --- 검증 단계 ---
    model.eval()
    val_loss_sum = 0.0
    val_n = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss_sum += loss.item() * x.size(0)
            val_n += x.size(0)
    val_loss = val_loss_sum / val_n

    # --- 스케줄러 + 기록 + 저장 ---
    scheduler.step(val_loss)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

    print(f'Epoch {epoch:3d}/{num_epochs} | '
          f'Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | '
          f'LR: {optimizer.param_groups[0]["lr"]:.1e}')
```

### 이 템플릿에 포함된 모든 요소

- ✅ device 관리
- ✅ train/eval 모드 전환
- ✅ zero_grad → forward → loss → backward → clip → step 순서
- ✅ gradient clipping (max_norm=1.0)
- ✅ AdamW + weight decay
- ✅ ReduceLROnPlateau scheduler
- ✅ with torch.no_grad() 검증
- ✅ Best model checkpoint
- ✅ 학습 곡선 history

**이 코드를 외우세요.** 거의 모든 PyTorch 프로젝트의 출발점입니다.

---

## 18. 흔한 실수 10가지

실무에서 초보 → 중급 넘어가면서 반드시 한 번은 겪는 실수:

1. **zero_grad 누락** → gradient 누적으로 학습 완전 망가짐
2. **model.eval() 누락** → Dropout 때문에 검증 loss 들쭉날쭉
3. **torch.no_grad() 누락** → 검증에서 OOM
4. **loss에 softmax 이중 적용** → CrossEntropyLoss는 내부에서 softmax 함
5. **one-hot 레이블** → CrossEntropyLoss는 인덱스(정수) 받음
6. **model.to(device) 후에 optimizer 생성 안 함** → optimizer가 CPU 텐서 참조
7. **gradient clipping 누락** → 시퀀스 모델에서 NaN 발생
8. **loss.item() 누락** → loss 텐서 자체를 누적하면 계산 그래프가 계속 쌓여 OOM
9. **shuffle=True를 검증에도 적용** → 재현 불가능
10. **checkpoint에서 모델 전체 저장** → 클래스 변경 시 로드 불가

### 실수 8번 상세: `.item()`의 중요성

```python
# ❌ 틀림 — loss가 텐서 상태로 누적되면서 그래프가 남음
total_loss = 0
for x, y in loader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    total_loss += loss   # ← 그래프 참조 유지됨 → OOM

# ✅ 맞음 — .item()으로 스칼라 값만 추출
total_loss = 0.0
for x, y in loader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()   # ← Python float, 그래프와 분리
```

---

## 부록: 왜 `.item()` / `.detach()`를 써야 하는가

### 계산 그래프 관점

PyTorch 텐서는 기본적으로 **계산 그래프에 연결**돼 있습니다. 텐서 하나를 참조하면 **그걸 만들어낸 모든 상위 연산**이 메모리에 남아있습니다.

```python
loss = criterion(model(x), y)   # loss.grad_fn = <...>  ← 그래프 참조
a = loss                         # a도 같은 그래프 참조
b = loss + 1                     # b도 그래프 확장 참조
c = loss.item()                  # c는 Python float — 그래프와 분리
d = loss.detach()                # d는 텐서지만 그래프와 분리
```

### 그래프 분리가 필요한 상황

1. **로그/기록용**: `.item()` (Python scalar 필요)
2. **다른 모델에 입력으로 넘기되 gradient 전파 차단**: `.detach()` (텐서 유지 + 그래프 분리)
3. **반복 누적**: `.item()` 또는 `.detach()` (안 하면 OOM)

### `.detach()` vs `torch.no_grad()`

- `.detach()` = **특정 텐서를 그래프에서 떼어냄** (다른 텐서는 영향 없음)
- `torch.no_grad()` = **블록 전체에서 그래프 구성 자체를 중지** (더 포괄적)

검증/추론은 `torch.no_grad()`, 학습 중 특정 텐서만 분리할 땐 `.detach()`.

---

## 이 토픽에서 배운 것

1. **학습 루프 4단계**: forward → loss → backward → step (+ zero_grad 초기화)
2. **train() / eval() 모드**: Dropout, BatchNorm 동작 변경
3. **torch.no_grad()**: 검증/추론에서 메모리 + 속도
4. **device 관리**: 모델 먼저 `.to(device)`, 그 다음 optimizer 생성
5. **Loss**: 분류는 CrossEntropyLoss(로짓 그대로 전달, 레이블은 인덱스)
6. **Optimizer**: **AdamW + weight_decay** 가 현대 권장 기본값
7. **Gradient clipping**: 시퀀스 모델에선 **필수** (max_norm=1.0)
8. **Scheduler**: ReduceLROnPlateau 가 Walk-Forward에 적합
9. **Checkpoint**: `state_dict`만 저장, best-val 갱신 시에만
10. **흔한 실수 10가지**: 위의 체크리스트를 학습 루프 작성 시마다 점검

---

## 다음 실습에서 확인할 것

1. **최소 학습 루프 실행**: 랜덤 toy 시계열 데이터에 **LSTM**을 학습시켜 loss가 감소하는 것을 확인
2. **zero_grad 누락 실험**: 일부러 빼고 돌려서 학습이 망가지는 것 관찰
3. **eval() 누락 실험**: Dropout 끄지 않은 상태에서 val_loss가 들쭉날쭉한 것 관찰
4. **Gradient clipping 유무 비교**: max_norm 제거 시 NaN 발생하는 케이스 재현
5. **Learning rate scheduler 비교**: StepLR vs CosineAnnealingLR vs ReduceLROnPlateau
6. **과적합 시각화**: train_loss↓ vs val_loss↑ 패턴 재현 + weight_decay로 완화
7. **Checkpoint save/load**: 중간에 끊고 이어 학습하는 시나리오
8. **경량 학습 반복 실습** (§16 내용): `train_one_individual(params)` 함수를 구현하고 5개 random hyperparam 조합으로 호출해 `best_val_loss` 를 비교 — 4주차 Optuna-LSTM 진입 전 warm-up (같은 함수가 나중에 Optuna `objective(trial)` 로 그대로 재사용됨을 체감)

---

## 다음 토픽 예고 — 3.5 학습 안정화 + 정규화

3.4까지 학습 루프의 **일반적 틀**을 배웠습니다. 3.5부터는 **LSTM 계열에서 특히 중요한 안정화·정규화 기법**을 다룹니다:
- LayerNorm vs BatchNorm — 시퀀스 모델에는 왜 LayerNorm이 더 적합한가
- Variational Dropout (Gal & Ghahramani) — LSTM에서 올바른 dropout 적용 방식
- Zoneout — LSTM hidden/cell 상태의 확률적 보존
- Label smoothing — 분류 loss를 부드럽게 만들어 과적합 완화
- Gradient noise injection (선택)
- 우리 프로젝트의 과적합 위험 정량화

**3.4 실습 (LSTM 기반 학습 루프) 완료 후** 3.5로 넘어갑니다.
