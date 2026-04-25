# 3.1 RNN 의 기본 구조와 한계

> **학습 목표**
> 1. "은닉 상태(hidden state)" 가 왜 필요한지, 그것이 시계열에서 무엇을 저장하는지 설명할 수 있다.
> 2. Vanilla RNN 의 수식 `h_t = tanh(W_x x_t + W_h h_{t-1} + b)` 을 그림과 함께 해석할 수 있다.
> 3. "시간축으로 펼친다(unroll)" 는 개념과 BPTT(Backpropagation Through Time) 가 어떻게 동작하는지 이해한다.
> 4. **Vanishing / Exploding gradient** 문제의 수학적 원인을 설명하고, 왜 이 때문에 LSTM/GRU 가 등장했는지 연결할 수 있다.
> 5. 금융 시계열(MSFT 일간 데이터)에 vanilla RNN 을 쓰면 실무적으로 어떤 벽에 부딪히는지 예측할 수 있다.

---

## §0. 왜 "순환(Recurrent)" 이라는 구조가 필요한가

2주차에서 우리는 `(B, T, F)` 텐서를 만들었습니다. 이 입력을 모델에 "어떻게" 통과시킬지 생각해 봅시다.

**선택지 1 — 평탄화 후 MLP:**
```python
x.reshape(B, T*F)   # (32, 60, 17) → (32, 1020)
model = nn.Linear(1020, 5)
```

동작은 합니다. 하지만 **피처 1개당 가중치가 60개** (일자별로) 생기므로 파라미터가 폭발하고, 더 치명적으로 **시점 간 순서를 모델이 "암묵적으로도" 알 수 없습니다**. day0_f0 과 day1_f0 은 그냥 1020차원 벡터의 0번째, 17번째 원소일 뿐이에요.

**선택지 2 — 시간축을 하나씩 처리:**

```
day 0 입력 x_0 ──┐
                 │ → 뭔가 기억할 상태 h_0
day 1 입력 x_1 ──┤ h_0 과 함께 고려 → 새 상태 h_1
day 2 입력 x_2 ──┤ h_1 과 함께 고려 → 새 상태 h_2
                 ...
day T-1 입력 x_{T-1} → h_{T-1} (최종 요약)
```

즉 **"지금까지 본 것의 요약" 이라는 변수** 를 하나 두고, 매 시점마다 그 요약을 업데이트하는 구조. 이 "요약 변수" 가 **은닉 상태(hidden state)** 이고, 업데이트 규칙을 네트워크가 학습하는 구조가 **Recurrent Neural Network(RNN)** 입니다.

순환이라 부르는 이유는 `h_{t-1}` 이 `h_t` 를 만들고, `h_t` 가 다시 `h_{t+1}` 을 만드는 — **같은 함수(셀)가 매 시점 반복 적용** 되기 때문입니다. 시간 축 t 가 반복문처럼 돌아갑니다.

---

## §1. Vanilla RNN 의 수식과 그림

### 1.1 핵심 수식 한 줄

```
h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
```

| 기호 | 뜻 | 우리 프로젝트 차원 |
|---|---|---|
| `x_t` | t 시점 입력 피처 벡터 | `(F,) = (17,)` |
| `h_{t-1}` | 직전 시점 은닉 상태 | `(H,)` — H 는 hidden size, 예 64 |
| `h_t` | 현재 시점 은닉 상태 | `(H,)` |
| `W_x` | 입력 → 은닉 가중치 | `(H, F) = (64, 17)` |
| `W_h` | 은닉 → 은닉 가중치 (**재귀 가중치**) | `(H, H) = (64, 64)` |
| `b` | 편향 | `(H,)` |
| `tanh` | 비선형 활성화 (−1 ~ +1) | 원소별 |

배치를 넣으면 `x_t: (B, F)`, `h_t: (B, H)` 로 한 번에 처리됩니다.

### 1.2 두 개의 가중치 행렬 — `W_x` vs `W_h`

- `W_x` : "현재 입력이 은닉 상태에 얼마나 반영될지" — 새로운 정보 유입
- `W_h` : "과거 요약을 얼마나 이어받을지" — 기억 유지

이 둘은 **모든 시점에서 같은 값** 을 씁니다. 60일짜리 시퀀스를 넣어도 파라미터는 `W_x, W_h, b` 셋뿐. 이게 **파라미터 공유(weight sharing)** 로, RNN 이 긴 시퀀스를 처리해도 파라미터 폭발이 없는 이유입니다.

### 1.3 시간축으로 "펼친" 그림

`h_t = f(h_{t-1}, x_t)` 는 시점 t 에 대해 재귀적이지만, 학습 시점에는 이걸 **시간축으로 쭉 펼쳐서** 하나의 거대한 feedforward 네트워크처럼 봅니다.

```
      x_0       x_1       x_2            x_{T-1}
       │         │         │                │
       ▼         ▼         ▼                ▼
h_0 ─→[셀]─→ h_1 ─→[셀]─→ h_2 ─→ ... ─→ [셀] ─→ h_{T-1} ─→ ŷ
       ▲         ▲         ▲                ▲
     같은 파라미터 W_x, W_h, b 가 반복 사용
```

- 초기 은닉 상태 `h_0` 은 보통 0 벡터로 시작 (또는 학습 가능한 파라미터로 둘 수 있음)
- **출력 선택:** 분류/회귀는 보통 `h_{T-1}` (마지막 은닉 상태) 하나만 써서 `Linear(H → output)` 으로 매핑. 이걸 **many-to-one** 구조라고 함
- 번역/텍스트 생성은 매 시점 출력 `y_t` 가 있는 **many-to-many**

---

## §2. 학습 — BPTT(Backpropagation Through Time)

### 2.1 기본 아이디어

펼친 그림을 보면 일반 feedforward 네트워크와 차이가 없습니다. 따라서 **역전파도 똑같이** 돌립니다 — 단, 같은 파라미터 `W_h` 가 여러 시점에서 쓰였으므로 **각 시점의 gradient 를 모두 더해** 최종 `W_h` 업데이트에 반영합니다.

```
∂L/∂W_h = Σ_{t=0}^{T-1} ∂L_t / ∂W_h
```

이 덧셈 구조가 뒤에서 말할 vanishing gradient 의 원인입니다.

### 2.2 메모리/시간 복잡도

- **메모리**: 펼친 그래프 전체를 저장해야 gradient 계산 가능 → `O(T × H)` 메모리. T=60, H=64, B=32 → 약 12만 개 실수 저장 (감당 가능)
- **시간**: forward `O(T × H²)`, backward 도 동일
- **truncated BPTT**: 너무 긴 시퀀스는 중간에 gradient 전파를 끊음 (ex: 매 20스텝씩). 우리 프로젝트 T=60 은 짧아서 불필요

---

## §3. 치명적 한계 — Vanishing / Exploding Gradient

### 3.1 문제의 직관

마지막 시점 loss 에서 **아주 옛날(t=0)** 입력까지 gradient 가 흘러갈 때, 시간 스텝마다 `W_h` 에 곱해집니다:

```
∂L/∂h_0 ≈ ∂L/∂h_{T-1} · (W_h)^(T-1) · (tanh 미분 항들)
```

여기서 `(W_h)^(T-1)` 이 결정적. `W_h` 의 가장 큰 고유값을 `λ` 라 할 때:

- `|λ| < 1` → `λ^T → 0` : **vanishing gradient** (예: `0.9^60 ≈ 0.0018`)
- `|λ| > 1` → `λ^T → ∞` : **exploding gradient** (예: `1.1^60 ≈ 304`)
- `|λ| = 1` 은 불안정한 경계

게다가 `tanh'(·)` 는 최대 1, 평균 더 작음 — vanishing 쪽으로 편향.

### 3.2 왜 이게 문제인가

- **Vanishing** : t=0 근처 입력이 최종 예측에 거의 영향을 못 줌 → **장기 의존성(long-term dependency) 을 학습 못 함**. 30일 전 충격이 오늘 가격에 미치는 영향을 RNN 은 "느끼지" 못함.
- **Exploding** : gradient 가 너무 커져 weight 가 날아가 학습 불안정. NaN loss 가 뜨는 전형적 증상.

### 3.3 수식으로 더 파 보기 (관심 있는 경우)

Vanilla RNN 의 시간 방향 역전파 핵심 항은:

```
∂h_t / ∂h_{t-1} = diag(tanh'(·)) · W_h
```

T 스텝 거슬러 올라가면:

```
∂h_T / ∂h_0 = Π_{t=1}^{T} diag(tanh'(·_t)) · W_h
```

곱이 쭉 이어지므로 **기하급수적 감쇠/폭발** 이 기본 거동. `W_h` 의 spectral radius(최대 특이값) 를 `σ` 라 하면 곱의 상한이 `σ^T · (tanh 미분의 곱)` 으로 눌리거나 폭발합니다.

### 3.4 완화 기법들 (vanilla RNN 수준)

- **Gradient clipping** — gradient norm 이 역치 넘으면 잘라냄. Exploding 방지에 거의 필수.
- **작은 `W_h` 초기화** — 보통 orthogonal 초기화로 `σ ≈ 1` 근처에서 시작.
- **ReLU RNN** — tanh 대신 ReLU + 항등행렬 초기화 (IRNN). 개선은 있지만 근본 해결은 아님.

근본 해결은 **게이트 구조** 로 gradient path 를 열어두는 것 — 이게 **LSTM(3.2)** 과 **GRU(3.3)** 의 핵심 아이디어입니다.

---

## §4. 코드로 본 vanilla RNN

### 4.1 직접 구현 (개념용)

```python
import torch
import torch.nn as nn

class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_x = nn.Linear(input_size, hidden_size)       # W_x, b_x
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # W_h (bias 는 W_x 에만)
        self.hidden_size = hidden_size

    def forward(self, x_t, h_prev):
        # x_t: (B, F), h_prev: (B, H) → h_t: (B, H)
        return torch.tanh(self.W_x(x_t) + self.W_h(h_prev))
```

매 시점 호출하며 hidden state 를 넘겨주는 식:

```python
cell = SimpleRNNCell(input_size=17, hidden_size=64)
h = torch.zeros(B, 64)           # h_0 = 0 으로 시작
for t in range(T):
    h = cell(x[:, t, :], h)      # x: (B, T, F)
# 이 h 가 h_{T-1} — 마지막 요약
```

### 4.2 PyTorch 내장 `nn.RNN`

실무에서는 이 루프를 직접 짜지 않고:

```python
rnn = nn.RNN(input_size=17, hidden_size=64, batch_first=True, nonlinearity='tanh')
out, h_last = rnn(x)            # x: (B, T, F)
# out: (B, T, H) — 모든 시점의 hidden state
# h_last: (1, B, H) — 마지막 시점만 (layer 수 축 먼저)
```

2주차에서 고정한 `batch_first=True` 관례가 여기서 빛을 발합니다.

---

## §5. "그래서 RNN 이 쓸모 없나?" — 현실적 관점

- 짧은 시퀀스(T < 20), 단순 문제: vanilla RNN 으로 충분한 경우 있음
- 우리 프로젝트 (T=60, 금융, 비선형·비정상): **단독으로는 부족**. 10일 이상 떨어진 시점의 정보를 잡기 어렵기 때문
- 그래서 실무 표준은 **LSTM/GRU** — vanilla RNN 의 gradient path 문제를 **게이트(gate)** 로 우회

**오해 정정:** "RNN 은 낡았고 Transformer 가 이긴다" 는 **텍스트 도메인 한정** 얘기입니다. 시계열 예측은 여전히 TFT, DeepAR 같은 RNN/attention 하이브리드가 강세이고, 샘플 수가 제한된 금융 도메인에서 Transformer 의 자원 소모가 큰 단점이 됩니다.

---

## §6. 자가 점검 질문

실습 Step 8 에서 답을 확인합니다.

1. Vanilla RNN 에서 `W_h` 의 크기는 몇인가? `(F, H)` vs `(H, H)` vs `(H, F)` 중 고르고 근거를 설명.
2. T=60 일짜리 시퀀스 하나를 처리할 때 `W_x, W_h, b` 는 몇 벌 사용되는가?
3. 초기 은닉 상태 `h_0` 을 0 으로 두는 것 vs 학습 가능 파라미터로 두는 것의 차이는?
4. `W_h` 의 가장 큰 특이값(spectral radius) 이 0.9 일 때, t=0 입력이 t=59 출력에 미치는 gradient 크기는 대략 어느 수준? (손계산)
5. Vanishing gradient 가 일어났는지 알아채려면 무엇을 모니터링해야 하는가?

---

## §7. 다음 토픽 예고

- **3.2 LSTM** — "cell state" 라는 **고속도로** 를 추가해 gradient 가 흐를 통로 확보. 3개 게이트(forget/input/output) 의 역할.
- **3.3 GRU** — LSTM 의 간소화. 2개 게이트(update/reset), 실무에서 더 많이 쓰이는 이유.
- **3.4 학습 루프 표준 패턴** — PyTorch 의 `.train()/.eval()/.zero_grad()/.step()` 리듬
- **3.5 분류 출력 — 5분위 확률**
- **3.6 Dropout, Early Stopping, LR Scheduler**

3주차가 끝날 때쯤 **`MSFT_GRU분류기.ipynb`** 가 완성되어 4주차의 Walk-Forward 통합으로 넘어갑니다.

---

## 요약 카드

```
RNN 본질
  h_t = tanh(W_x x_t + W_h h_{t-1} + b)
  같은 (W_x, W_h, b) 가 모든 시점에 반복 사용 (weight sharing)

입출력 축 계약
  nn.RNN(input_size=F, hidden_size=H, batch_first=True)
  in:  x (B, T, F)
  out: output (B, T, H), h_last (1, B, H)

근본 한계
  ∂L/∂h_0 ∝ (W_h)^T  ← T 거듭제곱
  |λ|<1 → vanishing  (장기 의존성 못 배움)
  |λ|>1 → exploding  (NaN loss)

대처
  vanilla: gradient clipping + orthogonal init + 작은 T
  근본 해결: LSTM/GRU 의 게이트 구조 (다음 토픽)
```
