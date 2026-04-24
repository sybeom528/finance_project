# 4.2 BiLSTM — 양방향 LSTM과 Walk-Forward 누수 제어 (Bidirectional LSTM)

> **목표**: 양방향 스캔으로 시퀀스 양쪽 끝의 정보를 모두 쓰는 BiLSTM 의 구조적 이점과, 이 이점이 시계열 예측에서는 **look-ahead bias** 로 쉽게 변질되는 지점을 수식·텐서 차원·실험 설계 세 층위로 드러낸다.
>
> COL-BL v2.4 에서 BiLSTM 은 두 가지 역할을 한다: (1) CEEMDAN IMF 의 "전·후 양쪽 맥락" 을 읽는 특성 추출기, (2) Optuna 탐색 공간의 한 옵션 (`bidirectional` as categorical). 이 장에서는 BiLSTM 을 단순히 "성능 좋은 변형" 으로 소개하지 않고, **왜 시계열에서는 위험한 도구인지 · 어떤 조건에서만 안전하게 쓸 수 있는지** Walk-Forward validation 의 작동 방식과 함께 철저히 해부한다. 많은 튜토리얼과 블로그 포스트가 "BiLSTM 은 LSTM 보다 성능이 좋다" 로 끝나는데, 그건 NLP(자연어 처리, 예: 문장 전체 감성 분류)에서만 맞는 얘기다. 재무 시계열은 정반대 — **기본 설정이 틀렸다**.

---

## 📋 목차

- [§1. 왜 이 토픽이 필요한가 — Stacked 다음이 Bi 인 이유](#1-왜-이-토픽이-필요한가--stacked-다음이-bi-인-이유)
- [§2. BiLSTM 의 수식과 텐서 차원 — Forward + Backward concat](#2-bilstm-의-수식과-텐서-차원--forward--backward-concat)
- [§3. Look-ahead bias — "양방향" 이 "미래 정보 누수" 가 되는 순간](#3-look-ahead-bias--양방향-이-미래-정보-누수-가-되는-순간)
- [§4. Walk-Forward 에서 BiLSTM 을 안전하게 쓰는 유일한 방법 — window 내부 양방향](#4-walk-forward-에서-bilstm-을-안전하게-쓰는-유일한-방법--window-내부-양방향)
- [§5. 재무 시계열에서 BiLSTM 이 의미 있는 이유 — 이동평균·MACD 의 사후 대칭성](#5-재무-시계열에서-bilstm-이-의미-있는-이유--이동평균macd-의-사후-대칭성)
- [§6. 마지막 출력 처리 — concat / last / mean pool / attention pool 비교](#6-마지막-출력-처리--concat--last--mean-pool--attention-pool-비교)
- [§7. 단방향 vs 양방향 — 언제 양방향이 이득인가](#7-단방향-vs-양방향--언제-양방향이-이득인가)
- [§8. 종합 — COL-BL v2.4 에서 BiLSTM 의 위치](#8-종합--col-bl-v24-에서-bilstm-의-위치)

---

## §1. 왜 이 토픽이 필요한가 — Stacked 다음이 Bi 인 이유

### 1.1 Stacked vs Bi 는 "다른 축" 을 건드린다

4.1 에서 다룬 Stacked LSTM 은 **depth 축** 을 건드린다. LSTM cell 여러 개를 수직으로 쌓아 `x → LSTM_1 → LSTM_2 → LSTM_3 → y` 로 깊이를 만드는 것. 한 timestep 안에서 representation 이 여러 번 비선형 변환을 거친다는 의미이고, 시간 축의 흐름 방향은 그대로 **과거 → 현재** 한 방향이다.

BiLSTM 은 정반대다. **time 축 자체의 방향** 을 건드린다. 같은 시퀀스를 한 번은 과거 → 현재 (forward), 한 번은 미래 → 현재 (backward) 로 두 번 읽는다. 깊이는 1층 그대로지만, 각 timestep 의 hidden 이 **양쪽 맥락을 다 담은 vector** 가 된다.

```
Stacked (깊이 축):                    BiLSTM (시간 축):
                                      
x_t ─→ LSTM₁ ─→ LSTM₂ ─→ LSTM₃ → y_t   [x_1, x_2, ..., x_T] 
  (같은 방향으로 3번 변환)               ├─ forward  →→→→  h_fwd
                                        └─ backward ←←←←  h_bwd
                                        → h = [h_fwd ; h_bwd]
```

이 둘은 **조합 가능** 하다. 즉 "3층 BiLSTM" 은 각 층이 양방향이고, 그게 3층 쌓인 구조. PyTorch 로는 `nn.LSTM(..., num_layers=3, bidirectional=True)` 한 줄. 파라미터 수는 Stacked 만 있을 때의 **정확히 2배**다 (forward LSTM + backward LSTM 각각).

### 1.2 NLP 에선 BiLSTM 이 default, 재무에선 **기본적으로 금지**

NLP 의 대표 과제 — 감성 분석 ("이 문장이 긍정인가 부정인가") 을 생각해보자. 입력은 "I absolutely hated this movie" 같은 문장이고, 출력은 단일 레이블. 이때 **문장 전체가 이미 과거에 쓰여진 결과물**이라, 앞뒤 단어를 동시에 보는 것은 자연스럽다. "I absolutely \_\_\_\_ this movie" 의 빈칸을 채우려면 앞 두 단어 "I absolutely" 와 뒤 두 단어 "this movie" 를 모두 봐야 한다. 이게 BiLSTM 이 2015년대 NLP 의 default 가 된 이유다 (ELMo, BERT 의 전신).

재무 시계열은 정반대다. 입력은 `[x_1, x_2, ..., x_T]` 시퀀스이고, 예측 타겟은 `y_{T+1}` (다음 날 수익률). 이때 **T+1 은 아직 존재하지 않는 미래** 다. 실거래 시점에서 "T+2, T+3, ..." 같은 미래 데이터는 당연히 쓸 수 없다. 그런데 BiLSTM 의 backward pass 는 정의상 **시퀀스의 끝부터 거꾸로 정보를 모으는 연산** 이다. 학습 시 train set 의 시퀀스 뒤쪽 값을 backward 가 가져다 쓰면, 모델은 "이 x_t 는 x_{t+1} 과 함께 있을 때 어떤 의미인지" 를 학습한다. 이게 $y_{T+1}$ 예측에 직접 누수된다.

따라서 재무에서 BiLSTM 을 쓰려면 **"window 내부에서만 양방향"** 이라는 제약 조건이 반드시 붙어야 한다. 그 조건을 어떻게 보장하는지가 §4 의 핵심 주제다.

### 1.3 이 장에서 답할 세 가지 질문

1. **BiLSTM 이 수식적으로 정확히 뭘 하는가** — forward 와 backward hidden 이 각각 무엇을 담고, concat 후 차원이 어떻게 바뀌는가 (§2)
2. **look-ahead bias 가 실제 코드에서 어떻게 발생하는가** — "잘못된 사용" 과 "올바른 사용" 을 한 줄씩 대조 (§3, §4)
3. **window 내 양방향이 정말 유익한가** — 재무 시계열의 어떤 구조적 특성이 backward pass 를 의미 있게 만드는가 (§5)

---

## §2. BiLSTM 의 수식과 텐서 차원 — Forward + Backward concat

### 2.1 단방향 LSTM 복습

3.2 에서 배운 표준 LSTM 의 recurrence 를 다시 쓰면:

$$\overrightarrow{h_t} = \text{LSTMCell}_{\text{fwd}}(\overrightarrow{h_{t-1}}, \overrightarrow{c_{t-1}}, x_t)$$

이때 $\overrightarrow{h_t}$ 는 "시점 $t$ 까지의 과거 정보를 요약한 $H$차원 벡터" 다. $t$ 가 커질수록 더 많은 과거가 누적되고, 마지막 시점 $\overrightarrow{h_T}$ 가 "전체 시퀀스를 과거 방향에서 본 요약" 이 된다.

중요한 제약: **$\overrightarrow{h_t}$ 는 $x_{t+1}, x_{t+2}, \ldots$ 를 전혀 모른다.** 재무 예측에서 이 제약은 "현재 시점에서 미래를 모르는 게 당연" 과 부합한다.

### 2.2 Backward LSTM 의 추가

BiLSTM 은 같은 구조의 **두 번째 LSTM cell** 을 추가로 돌리는데, 시간 순서를 뒤집어 읽는다:

$$\overleftarrow{h_t} = \text{LSTMCell}_{\text{bwd}}(\overleftarrow{h_{t+1}}, \overleftarrow{c_{t+1}}, x_t)$$

즉 시점 $T$ 부터 시작해서 $t=T-1, T-2, \ldots, 1$ 로 역방향 전개. 이때 $\overleftarrow{h_t}$ 는 **"시점 $t$ 부터 시퀀스 끝 $T$ 까지의 미래 정보를 요약한 벡터"** 가 된다.

이것이 look-ahead 의 수학적 정체다. 정의상 $\overleftarrow{h_t}$ 는 $x_{t+1}, x_{t+2}, \ldots, x_T$ 를 모두 봤다. 주의: 아직은 수식 자체가 "나쁜" 게 아니다. 이 $T$ 가 무엇이냐, 그리고 이 $\overleftarrow{h_t}$ 를 어디에 쓰느냐에 따라 안전할 수도 누수일 수도 있다 (§3·§4 에서 결판).

### 2.3 Concat — 양쪽 요약을 결합

각 시점 $t$ 에서 BiLSTM 의 최종 hidden 은 두 방향의 연결(concatenation):

$$h_t = [\overrightarrow{h_t} \; ; \; \overleftarrow{h_t}] \in \mathbb{R}^{2H}$$

차원이 $H$ → $2H$ 로 **두 배**가 된다. 이게 "BiLSTM 은 출력 feature dim 이 2배" 라는 말의 출처다.

### 2.4 파라미터 수 — 정확히 2배

단방향 LSTM 1층의 파라미터 수는 `4H(I+H) + 4H(2)` (입력·히든 가중치 + bias). BiLSTM 1층은 forward cell + backward cell 두 개를 **독립적인 파라미터**로 가지므로 정확히 2배다.

```
단방향 1층 (H=64, I=10) : 4·64·(10+64) + 8·64      = 19,456
BiLSTM 1층 (H=64, I=10) : 2 × 19,456                = 38,912
3층 BiLSTM (각 층 H=64)  : 입력층 38,912 + 중간/출력층 (2·64=128 이 입력) 각각 더 큼
                          → 총 ~150,000 근처
```

4.1 3층 단방향 Stacked LSTM 이 ~63,000 개 파라미터였음을 기억하면, 3층 BiLSTM 은 ~**2.4배** 규모. 이 "용량 증가" 가 과적합 위험과 바로 연결된다 — 그래서 **BiLSTM 은 정규화와 함께 써야 함**은 거의 원칙.

### 2.5 PyTorch 구현 — 한 줄 추가

```python
import torch.nn as nn

# 단방향 (4.1 에서 배운 것)
lstm_uni = nn.LSTM(input_size=10, hidden_size=64, num_layers=3,
                   batch_first=True, bidirectional=False)

# 양방향 — 딱 한 글자 바꿈
lstm_bi = nn.LSTM(input_size=10, hidden_size=64, num_layers=3,
                  batch_first=True, bidirectional=True)

# 출력 차원 차이
x = torch.randn(32, 60, 10)  # (batch, time, feat)
out_uni, _ = lstm_uni(x)
out_bi,  _ = lstm_bi(x)
print(out_uni.shape)  # torch.Size([32, 60, 64])    — H=64
print(out_bi.shape)   # torch.Size([32, 60, 128])   — 2H=128
```

PyTorch 는 `out_bi[..., :H]` 가 forward hidden, `out_bi[..., H:]` 가 backward hidden 이 되도록 concat 한다. 이 순서를 기억해두자 — §6 에서 출력 처리할 때 중요.

### 2.6 `h_n`, `c_n` 의 shape 도 바뀐다

```python
out, (h_n, c_n) = lstm_bi(x)
print(h_n.shape)  # (num_layers * 2, batch, H) = (6, 32, 64)
```

즉 `h_n` 의 첫 번째 차원이 `num_layers × 2` 로 확장. 인덱싱 규칙은:
- `h_n[0]`: layer 1 forward 의 마지막 hidden
- `h_n[1]`: layer 1 backward 의 마지막 hidden (= **시퀀스의 첫 시점**까지 되돌아간 결과)
- `h_n[2]`, `h_n[3]`: layer 2 forward / backward
- `h_n[4]`, `h_n[5]`: layer 3 forward / backward

이 인덱싱이 회귀 헤드 설계(§6) 에서 "양 끝의 요약을 어떻게 쓸 것인가" 의 선택지를 만든다.

---

## §3. Look-ahead bias — "양방향" 이 "미래 정보 누수" 가 되는 순간

### 3.1 누수의 정의 — "train 에서 본 정보가 test 시점에 존재하지 않는다"

**Look-ahead bias** (또는 data leakage) 란 "모델이 학습 시점에 접근한 정보 중, **실거래 시점에는 물리적으로 알 수 없는** 정보가 있는 경우" 를 말한다. 2주차 03 (`look_ahead_bias.md`) 에서 다룬 핵심 개념이다. 재무에선 이게 발생하면 백테스트 수익률은 환상이고 실거래 투입 시 완전히 무너진다.

BiLSTM 에서 누수가 발생하는 **가장 흔한 시나리오**는 다음과 같다:

### 3.2 시나리오 A — "전체 시계열을 한 번에 BiLSTM 에 통과"

잘못된 코드 (실제로 많은 튜토리얼에 있음):

```python
# ❌ WRONG: 전체 시계열 [x_1, ..., x_N] 을 그대로 BiLSTM 에
full_series = torch.tensor(prices)  # shape (N, feat), N = 수천 일
out, _ = lstm_bi(full_series.unsqueeze(0))  # (1, N, 2H)

# 각 시점의 예측을 out[0, t, :] 로 뽑아서 loss 계산
for t in range(N-1):
    loss += (linear(out[0, t, :]) - y[t+1]) ** 2
```

이 코드에서 시점 $t$ 의 예측 `linear(out[0, t, :])` 는:

- forward hidden: $x_1, \ldots, x_t$ 까지의 과거 ← 안전
- backward hidden: $x_{t+1}, \ldots, x_N$ 까지의 **미래 전체** ← 완전 누수

backward LSTM 은 $x_N$ 부터 시작해 $t$ 까지 돌아왔으므로, **예측 대상인 $y_{t+1}$ 근처의 가격 정보를 입력으로 쓴 채** 예측을 내는 것이다. train loss 는 극단적으로 낮게 떨어지지만, 실거래 시점(마지막 시점 $N$ 이후) 에는 "$x_{N+1}$ 부터 뒤로 돌아올 정보" 가 존재하지 않아서 모델이 작동 불능이다.

### 3.3 시나리오 B — "train/val split 후에도 여전히 문제"

조금 나아진 코드:

```python
# ❌ STILL WRONG
train_X = prices[:train_end]       # (T_train, feat)
val_X   = prices[train_end:]       # (T_val, feat)

# train 전체를 한 번에
out_train, _ = lstm_bi(train_X.unsqueeze(0))
# ... loss 계산
```

이건 train 구간 내부에서만큼은 시점 $t$ 의 예측이 $t+1, \ldots, T_{\text{train}}$ 을 본다. Validation 시점 이후는 안 보니까 "train vs val" 경계는 지켜진다. 하지만 **train 내부의 label 들** 은 서로가 서로의 입력이 되어 optimization 이 왜곡된다. 모델이 "시퀀스 전체의 통계" 를 학습하고, 그것이 **val 구간의 통계 분포** 와 다를 때 일반화 실패한다.

이 시나리오는 train loss 가 이례적으로 낮은 반면 val loss 가 보통 단방향보다 나쁘게 나오는 패턴으로 관찰된다.

### 3.4 누수의 수학적 증명 — BiLSTM 의 gradient 흐름

Look-ahead 의 핵심은 역전파(backprop) 에서도 드러난다. 시점 $t$ 의 예측 $\hat{y}_t = W h_t$ 가 내는 손실 $L_t = (\hat{y}_t - y_{t+1})^2$ 의 gradient 를 backward LSTM 의 파라미터 $\theta_{bwd}$ 에 대해 계산하면:

$$\frac{\partial L_t}{\partial \theta_{bwd}} = 2(\hat{y}_t - y_{t+1}) \cdot W \cdot \frac{\partial \overleftarrow{h_t}}{\partial \theta_{bwd}}$$

그런데 $\overleftarrow{h_t}$ 는 $\overleftarrow{h_{t+1}}$ 로부터 전파되고, 궁극적으로 $x_{t+1}, x_{t+2}, \ldots$ 를 입력으로 쓴다. 즉 gradient 는 **"미래 입력들이 현재 예측을 더 잘 맞추도록"** 파라미터를 수정한다. train 시점에는 $x_{t+1}$ 이 손에 있으므로 이게 가능하지만, 실거래 시점에는 불가능하다.

### 3.5 진단 — "BiLSTM 을 쓸 때 누수 여부를 어떻게 확인하나"

가장 확실한 방법은 **"미래 피처를 명시적으로 셔플한 뒤 성능 비교"** 다:

```python
# 정상 학습
model_bi = BiLSTMModel()
train(model_bi, X, y)
val_loss_normal = evaluate(model_bi, X_val, y_val)

# 실험적 누수 감지: backward pass 가 보는 "미래 구간" 을 랜덤 셔플
X_shuffled = shuffle_future_half(X)  # 각 window 의 뒷 절반을 셔플
train(model_bi, X_shuffled, y)
val_loss_shuffled = evaluate(model_bi, X_val_shuffled, y_val)

# 만약 두 loss 가 유사 → 뒷 절반이 예측에 기여 안 함 → 누수 없음
# 만약 shuffled 쪽이 크게 악화 → 뒷 절반이 예측에 크게 기여 → 누수 존재
```

이 진단은 §4 의 올바른 구현에선 "두 loss 가 비슷해야" 통과한다. 실습 노트북에서 실제로 돌려볼 것이다.

---

## §4. Walk-Forward 에서 BiLSTM 을 안전하게 쓰는 유일한 방법 — window 내부 양방향

### 4.1 핵심 아이디어 — "window 는 이미 과거로만 잘려 있다"

2주차 `sliding_window.md` 에서 설계한 sliding window dataset 을 떠올리자. 각 샘플은 다음 형태:

```
샘플 i:
  입력 X_i = [x_{i}, x_{i+1}, ..., x_{i+W-1}]    (60일치 과거)
  타겟 y_i = return_{i+W}                         (i+W 날 다음 날 수익률)
```

여기서 **입력 window 자체가 이미 "과거 60일" 로 고정**되어 있다. 이 window 안에서만 BiLSTM 을 돌리면:

- forward: $x_i \to x_{i+1} \to \ldots \to x_{i+W-1}$ (window 내 과거 → 현재)
- backward: $x_{i+W-1} \to x_{i+W-2} \to \ldots \to x_i$ (window 내 현재 → window 내 과거)

backward 가 보는 "미래" 는 **window 내부의 미래** — 즉 실거래 시점 $i+W$ 이전의 값들. **window 경계를 넘지 않음**. 따라서 타겟 $y_i = \text{return}_{i+W}$ 는 입력에 전혀 들어가 있지 않다.

```
시점:  ... i-1 | i  i+1 ... i+W-1 | i+W  i+W+1 ...
           과거  |  window (입력)    |  타겟 및 미지의 미래
                  ←─ backward 스캔 ─→
                  ↑ backward 는 여기서만 움직임 — 경계 안 넘음
```

### 4.2 PyTorch 로는 **추가 코드 0줄**

다음 코드는 자동으로 안전하다. sliding window dataset + `nn.LSTM(bidirectional=True)` 조합이면 경계 위반이 일어날 방법이 없다:

```python
class BiLSTMRegressor(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,   # ← 이 줄 하나
        )
        # 출력 feat dim 이 2배 (forward + backward)
        self.head = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x):
        # x: (batch, window, feat)
        out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers*2, batch, H)
        # 마지막 층 forward 의 마지막 시점 + backward 의 마지막 시점 (=window 시작점)
        h_last_fwd = h_n[-2]  # (batch, H)
        h_last_bwd = h_n[-1]  # (batch, H)
        h_combined = torch.cat([h_last_fwd, h_last_bwd], dim=-1)  # (batch, 2H)
        return self.head(h_combined).squeeze(-1)   # (batch,)
```

여기서 `h_n[-2]` 와 `h_n[-1]` 를 쓰는 게 "양 끝의 요약" 을 가져오는 표준 방식. 자세한 선택지 비교는 §6.

### 4.3 Dataset 설계가 핵심 방어선

BiLSTM 의 안전성은 **모델 내부가 아니라 dataset 경계 설계**에서 나온다. 다음 세 규칙이 지켜지는 한 안전:

1. 각 샘플의 입력 X 는 타겟 y 의 **시점 이전에만** 존재하는 값들로 구성
2. Walk-Forward cross-validation 에서 train 구간과 val 구간이 **시간순으로 분리**
3. Feature engineering (rolling mean, z-score 등) 시 rolling window 가 타겟 시점을 **넘지 않음**

이 중 **3번은 BiLSTM 이전에 걸리는 함정**이다 — 예를 들어 `rolling_mean(close, 30).center=True` 같은 중심 이동평균을 feature 로 넣으면, 그 자체가 미래를 본 feature 가 되어 BiLSTM 과 무관하게 누수. 이건 4.2 가 아닌 feature engineering 토픽(2주차)에서 이미 다룬 내용이지만, BiLSTM 과 합쳐지면 **두 종류의 누수가 중첩**되어 진단이 어려워지므로 주의.

### 4.4 Walk-Forward fold 간의 누수 확인

Walk-Forward validation 의 표준 구조를 떠올리자:

```
fold 1: train = [0, T1],      val = [T1, T2]
fold 2: train = [0, T2],      val = [T2, T3]
fold 3: train = [0, T3],      val = [T3, T4]
...
```

여기서 BiLSTM 모델을 각 fold 마다 **독립적으로 학습** 해야 한다. 만약 같은 모델을 여러 fold 에 이어서 fine-tune 하면, fold 2 학습 시점에 이미 fold 1 의 val 구간 정보가 파라미터에 녹아 있다. Walk-Forward 는 **각 fold 가 독립적인 실험** 이어야 하고, 각 fold 내부에서 BiLSTM + sliding window 규칙을 따르면 그 fold 에서는 안전하다.

### 4.5 정리 — "BiLSTM + sliding window + fold 별 재학습" 3종 세트

COL-BL v2.4 에서 BiLSTM 을 쓰는 모든 경우 이 세 조건을 동시에 만족해야 한다. 하나라도 깨지면 다른 두 개가 의미 없다. Optuna objective 안에서도 반드시 이 패턴. 실습 노트북에서 **"안 지켜진 경우 vs 지켜진 경우"** 의 val loss 를 직접 비교할 것이다.

---

## §5. 재무 시계열에서 BiLSTM 이 의미 있는 이유 — 이동평균·MACD 의 사후 대칭성

### 5.1 "과거만 봐도 예측 가능한가" 질문

여기까지 읽은 독자는 자연스럽게 이런 의문이 들 것이다:

> window 안에서만 양방향이면, 결국 window 의 끝 시점 $i+W-1$ 에서 볼 때 backward pass 는 "이미 아는 과거 값들" 을 한 번 더 섞는 것 아닌가? 단방향 LSTM 이 과거를 정방향으로 흘려도 같은 정보를 담을 텐데, BiLSTM 이 뭐가 다른가?

맞는 지적이다. **정보량 자체는 동일** (둘 다 window 내부의 $x_i, \ldots, x_{i+W-1}$ 만 봄). 차이는 **그 정보를 어떤 시점에서 해석하는가** 에 있다.

### 5.2 단방향이 담는 것 vs 양방향이 담는 것

시점 $t$ (window 중간) 의 hidden 을 비교:

- **단방향** $\overrightarrow{h_t}$: "$x_i, \ldots, x_t$ 까지의 누적 정보" — $t$ 시점에 **실시간으로 보였을** 패턴
- **양방향** $h_t = [\overrightarrow{h_t} ; \overleftarrow{h_t}]$: 실시간 누적 + "$x_t, \ldots, x_{i+W-1}$ 까지의 역방향 누적"

그런데 여기서 "역방향 누적" 은 실거래 당시에는 **볼 수 없었던** 뒤따라온 값들이다. window 안이므로 누수는 아니지만, 의미는 **"사후적 해석"** 이다.

### 5.3 왜 "사후적 해석" 이 예측에 도움이 되나

재무 시계열의 실전 예:

**예 1 — 이동평균 정합성**

전통적 기술적 분석에서 "이 구간의 평균은 100 이었다" 라고 말하려면 구간 끝 시점에 도달해서야 비로소 확정된다. 단방향 LSTM 이 시점 $t$ 에서 보는 평균은 $[x_i, \ldots, x_t]$ 의 부분평균. 양방향 LSTM 의 backward 는 **$t$ 시점에서 window 전체 평균**을 본 것과 동등한 정보를 함께 섞어서 forward hidden 에 추가한다. 이게 예측 head 로 들어가면 "이 window 의 전체 평균을 감안했을 때 $t$ 시점의 움직임이 어떤 위치" 라는 정보가 된다.

**예 2 — 레짐 인식**

window 60일 중 20일차에서 갑자기 큰 가격 변동이 발생하면, 단방향 LSTM 은 그 변동을 "지금 막 발생한 사건" 으로 인코딩한다. BiLSTM 은 backward pass 로 "이 변동 이후 40일 동안 어떻게 반응했는지" 를 함께 인코딩한다. 즉 **"이 이벤트가 반짝이었는지 레짐 전환이었는지" 를 사후 판단** 해서 forward 쪽 표현에 더해준다. 이 정보가 $i+W$ 시점의 수익률 예측에 도움이 될 수 있다.

**예 3 — 노이즈 필터링의 대칭성**

Centered moving average 가 trailing MA 보다 noise 제거 효과가 큰 것은 통계적 사실이다 (시간 대칭 필터는 phase lag 이 0). BiLSTM 의 backward 는 이런 대칭 필터링 효과를 hidden representation 수준에서 근사한다. 특히 **CEEMDAN 의 저주파 IMF (추세 성격)** 에서 이 효과가 커진다 — 추세는 대칭 필터링이 더 정확하고, 고주파 IMF 는 대칭이든 비대칭이든 둘 다 그냥 노이즈.

### 5.4 그럼에도 "무조건 BiLSTM" 은 아니다

위 세 가지 이득은 **조건부 이득**이다:

- window 가 짧으면 (W=20) backward 가 학습할 만한 뒷 구간 정보가 없음 → 이득 미미
- 고주파 IMF 는 사후 해석이 오히려 노이즈 증폭 → 오히려 독
- 파라미터 2배 + 훈련 시간 ~1.8배 → regularization 강화 필수

이 조건들을 Optuna 가 자동 탐색하도록 `bidirectional: categorical(False, True)` 를 HP 공간에 포함시키는 것이 4.3 의 설계. IMF 별로 선택이 다르게 나올 것이 예상되고, 그 **선택 패턴 자체가 가설 검증 신호**다 (§8.3 참고).

### 5.5 간단 비교 — NLP 양방향 vs 재무 양방향

| 축 | NLP 감성 분석 | 재무 window 내 BiLSTM |
|---|---|---|
| 전체 시퀀스 가용성 | 문장 전체가 처음부터 주어짐 | window 내에서만 과거 |
| 실거래 시점 입력 | 항상 "완전한 문장" | 매일 새로 한 step 씩 sliding |
| backward 가 보는 것 | 문장의 뒷 단어들 | window 내 후속 시점들 (**window 밖 미래 아님**) |
| 누수 위험 | 거의 없음 | window 경계 넘으면 즉시 발생 |
| 이득의 성격 | 단어 의미 disambiguation | 레짐 인식 / 사후 해석 / 대칭 필터 |

재무에서 BiLSTM 은 "쓸 수 있지만 조심해야 하는 도구" 다. 조심의 핵심 = window 내부로만 한정.

---

## §6. 마지막 출력 처리 — concat / last / mean pool / attention pool 비교

BiLSTM 의 출력 `out` 은 shape `(batch, window, 2H)`. 여기서 **어떻게 고정 크기 vector 를 뽑아 회귀 head 에 넣을까**가 설계 선택지. 네 가지 대표 방식을 비교한다.

### 6.1 Option A — "forward 의 마지막 + backward 의 마지막" (PyTorch 기본)

```python
h_fwd_last = h_n[-2]  # 마지막 layer forward 의 h_T (shape: batch, H)
h_bwd_last = h_n[-1]  # 마지막 layer backward 의 h_T (= window 시작 시점의 역방향 요약)
h_out = torch.cat([h_fwd_last, h_bwd_last], dim=-1)  # (batch, 2H)
```

**의미**: forward 의 "window 끝에서 본 전체 과거 요약" + backward 의 "window 시작에서 본 전체 미래 요약". 양 끝의 시퀀스 요약 두 개를 결합 → 전체 window 의 "양 끝점에서 각각 압축한 관점" 을 보존.

**장점**: 구현 가장 간단, PyTorch `h_n` 을 그대로 사용, 파라미터 추가 없음.
**단점**: window 가운데의 정보가 상대적으로 희석될 수 있음.

COL-BL v2.4 기본값.

### 6.2 Option B — "out[:, -1, :]" (마지막 timestep 의 양방향 concat)

```python
# out: (batch, window, 2H)
h_out = out[:, -1, :]   # (batch, 2H)
# 이 안에: forward h_{T-1} + backward h_{T-1}
```

**의미**: 마지막 시점에서의 양방향 hidden. forward 는 "지금까지 과거 요약", backward 는 "마지막 시점에서 본 미래 요약 — 그런데 마지막 시점이므로 미래가 없음" → backward part 는 단일 step 만 처리된 값이 됨.

**장점**: 간단.
**단점**: backward 의 정보가 상대적으로 빈약. 특히 마지막 시점 근처의 backward hidden 은 충분히 누적되지 못해 representation 이 얕음.

**이 옵션은 재무 시계열에서 특히 불리** — 우리가 궁금한 건 window 끝 다음 날 수익률인데, backward 는 window 끝 근처에서 거의 초기화 상태.

### 6.3 Option C — Mean Pool (전 시점 평균)

```python
h_out = out.mean(dim=1)   # (batch, 2H), 시간축 평균
```

**의미**: 전체 window 의 forward/backward hidden 을 시점별 평균. 각 시점이 동등한 가중치를 갖도록 강제.

**장점**: 모든 시점 정보가 반영, 특정 시점의 outlier 영향 작음.
**단점**: "최근 시점이 더 중요" 같은 자연스러운 시간 가중이 무시됨. 수익률 예측에서 직전 며칠의 비중이 멀리 과거보다 클 텐데, 그게 사라짐.

시계열에는 잘 안 쓰임 (NLP 에서는 종종 유효 — 문장의 모든 단어가 동등).

### 6.4 Option D — Attention Pool (학습 가능한 시점 가중치)

```python
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    def forward(self, x):
        # x: (batch, window, 2H)
        scores = self.attn(x).squeeze(-1)        # (batch, window)
        weights = torch.softmax(scores, dim=-1)  # (batch, window)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, 2H)

# 사용
h_out = self.pool(out)   # (batch, 2H)
```

**의미**: 각 시점 hidden 의 "중요도" 를 학습 가능한 linear layer 로 계산 후, softmax 가중합.

**장점**: 데이터가 스스로 중요 시점을 찾음. 레짐 변환 시점이 window 안에 있으면 자동으로 높은 가중치.
**단점**: 파라미터 추가 (2H+1 개 정도, 적음), 학습이 불안정할 수 있음, 더 많은 데이터 필요.

COL-BL v2.4 의 ⭐ 5.1 Attention-LSTM 토픽에서 본격적으로 다룰 내용. 4.2 실습에서는 Option A 와 Option C 만 비교.

### 6.5 비교표

| Option | 구현 | 파라미터 추가 | 강점 | 재무 적합성 |
|---|---|---|---|---|
| A. h_n concat | 한 줄 | 0 | 양 끝 요약 보존 | **기본값 추천** |
| B. out[:,-1,:] | 한 줄 | 0 | 간단 | 낮음 (backward 빈약) |
| C. mean pool | 한 줄 | 0 | 모든 시점 반영 | 중간 |
| D. attention pool | 8–10 줄 | 2H+1 | 학습 가능 가중 | 5.1 에서 본격 활용 |

### 6.6 실습에서 쓸 조합

4.2 실습 노트북은 **Option A** 를 기본으로 단방향 vs 양방향 비교. 그 다음 **Option A vs Option C** 직접 비교해서 pooling 방식의 영향을 재본다. Option D 는 5.1 Attention-LSTM 까지 미룸.

---

## §7. 단방향 vs 양방향 — 언제 양방향이 이득인가

### 7.1 "무조건 양방향" 은 NLP 신화

NLP 문헌에서 BiLSTM 이 단방향보다 거의 항상 우수하다는 결과가 많이 보고되지만, 재무에서는 **양방향이 단방향보다 나쁜 경우가 꽤 흔하다**. 이유:

1. **타겟이 window 끝 직후** — forward pass 가 담는 "직전 몇 step 정보" 가 예측에 가장 중요한데, backward 는 그 부분에서 representation 이 얕음 (Option B 문제)
2. **window 가 짧으면 backward 가 학습 못 함** — W=20 이면 backward 가 유의미한 패턴을 배우기에 데이터 부족
3. **파라미터 2배 → 과적합 가속** — 정규화 안 강화하면 train-val gap 벌어짐

### 7.2 양방향이 이득을 내는 조건

반대로 다음 조건들이 충족되면 양방향이 단방향보다 나아진다:

1. **window 가 충분히 긺** (W ≥ 60 정도)
2. **레짐 변동이 시퀀스 내부에 존재** — 가격이 급등락하는 구간이 window 중간에 자주 포함되는 시점
3. **저주파/추세 성격 데이터** — CEEMDAN 의 저주파 IMF, 또는 월봉·주봉
4. **충분한 정규화** (VD + LN + weight_decay)

4.2 실습에서 이 조건들을 하나씩 바꿔가며 단방향 vs 양방향의 val loss 차이를 관찰할 것이다.

### 7.3 정량적 예상치

선행 연구(재무 시계열에 BiLSTM 을 적용한 몇몇 논문) 에서 보고된 이득은:

| 조건 | BiLSTM ΔMSE (vs 단방향) |
|---|---|
| 일봉 + short window + 고주파 IMF | -0.5% ~ +2% (즉 **거의 동일 또는 악화**) |
| 일봉 + long window + 저주파 IMF | -3% ~ -8% (근소한 이득) |
| 월봉 + 전체 시퀀스 | -5% ~ -15% (명확한 이득) |

우리 COL-BL v2.4 는 **일봉 + W=60 + IMF 별 분해** 조합이므로, 저주파 IMF 에서 근소한 이득, 고주파 IMF 에서 이득 없음을 예상. Optuna 가 이 패턴을 발견할지가 4.3 의 검증 포인트.

### 7.4 실습에서 확인할 가설

4.2 실습 노트북에서 다음을 수치로 확인:

**H1**: 단방향 vs 양방향, 동일 파라미터 수 기준 (양방향은 hidden_dim 을 절반으로 줄여 fair comparison)
→ 양방향이 이득 또는 동등

**H2**: BiLSTM with Option A vs BiLSTM with Option C (pooling 차이)
→ Option A 가 유리 (재무 시계열의 최근 시점 비중 때문에)

**H3**: 의도적 누수 BiLSTM (window 경계 위반) vs 정상 BiLSTM
→ 누수 쪽은 train loss 비현실적으로 낮음, val loss 는 비슷하거나 나쁨 (train-val gap 이 극단적)

---

## §8. 종합 — COL-BL v2.4 에서 BiLSTM 의 위치

### 8.1 v2.4 아키텍처 내에서의 역할

COL-BL v2.4 = CEEMDAN + Optuna-LSTM + LSTM ensemble → BL View → Black-Litterman.

여기서 BiLSTM 은 다음 두 지점에서 고려된다:

1. **OLSTM (4.3) 의 탐색 공간 원소**: `bidirectional: categorical(False, True)`. Optuna 가 IMF 별로 최적 선택.
2. **5.1 Attention-LSTM 의 전단**: BiLSTM 위에 attention pooling 을 쌓아 "시점 가중" + "양방향 맥락" 을 결합.

4.1 의 3층 RegularizedStackedLSTM 은 BiLSTM 을 끄고 학습한 버전이었다. 4.2 실습에서는 같은 구조에 `bidirectional=True` 만 넣어 직접 비교한다. 결과가 IMF 별로 일관되지 않다면 (일부 IMF 는 이득, 일부는 손해), 그 자체가 **"IMF 특성에 따라 양방향 가치가 다르다"** 는 사전 가설을 뒷받침하는 evidence 가 된다.

### 8.2 4.3 OLSTM 탐색 공간의 확장

4.1 까지의 탐색 공간은 다음이었다:

```python
hidden_dim  = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
num_layers  = trial.suggest_int("num_layers", 1, 3)
dropout     = trial.suggest_float("dropout", 0.0, 0.5)
window_size = trial.suggest_categorical("window", [20, 40, 60, 90])
lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
batch_size  = trial.suggest_categorical("bs", [32, 64, 128])
```

4.2 를 지나면 여기에 **한 줄**이 추가된다:

```python
bidirectional = trial.suggest_categorical("bidirectional", [False, True])
```

탐색 공간이 정확히 2배가 된다. Optuna TPE 는 이 추가를 잘 다루지만, **총 trial 수를 늘려야** 함 (기본 30 → 50 정도). Pruning 과 조기 종료가 있으니 실행 시간은 선형으로 늘지 않음.

### 8.3 IMF 별 예상 패턴 (가설)

| IMF 특성 | 예상 `bidirectional` 선택 | 이유 |
|---|---|---|
| **고주파 IMF1~2** (노이즈) | False 선호 | backward 가 노이즈를 기억 → 과적합 |
| **중주파 IMF3~5** (단기 패턴) | 혼재 | window 길이에 따라 달라짐 |
| **저주파 IMF6~10** (추세) | True 선호 | 사후 대칭 해석이 추세 추정에 유익 |

4.3 실행 후 Optuna study 의 `best_params` 를 IMF 별로 집계하면 이 패턴이 나타날 것으로 예상. 나타나지 않으면 가설 수정.

### 8.4 정규화 조합 고정

BiLSTM 을 쓸 때는 4.1 에서 고정한 정규화 조합을 **더 강하게** 가져간다:

- **Variational Dropout**: 기본값 `p=0.3` → BiLSTM 에선 `p=0.3~0.4`
- **LayerNorm**: 층별 적용 유지
- **Gradient Clipping**: `max_norm=1.0` 그대로
- **Weight Decay (AdamW)**: 기본 `1e-4` → BiLSTM 에선 `2e-4` 정도 시도

이유는 §2.4 에서 본 파라미터 2배 → overfit 위험 2배. 4.2 실습에서 정규화를 약하게 줬을 때 vs 강하게 줬을 때의 차이도 간단히 비교.

### 8.5 체크포인트

- [ ] §2 수식과 파라미터 수 계산 이해
- [ ] §3 look-ahead 시나리오 3가지 구분 (A: 전체 통과 / B: split 후 내부 / C: feature 자체 누수)
- [ ] §4 sliding window + `bidirectional=True` = 자동 안전 구조
- [ ] §5 재무 시계열에서 BiLSTM 이득의 3가지 구조적 이유 (MA 정합 / 레짐 인식 / 대칭 필터)
- [ ] §6 4가지 pooling 옵션 중 Option A 가 기본, C 는 실습 비교, D 는 5.1 로
- [ ] §7 단방향 > 양방향 경우와 반대 경우의 조건 구분
- [ ] §8 OLSTM 탐색 공간 확장 1줄, IMF 별 예상 패턴

### 8.6 다음 실습 — `02_BiLSTM_실습.ipynb`

다음 노트북에서 다룰 것:

1. **§1** 4.1 Regularized 3층과 동일 구조 + `bidirectional=True` 단일 변경으로 비교 (Option A)
2. **§2** 파라미터 수 공정 비교 — BiLSTM hidden_dim 절반으로 줄여 파라미터 수 매칭 시 성능 변화
3. **§3** 누수 실험 — 의도적으로 window 경계를 넘겨 BiLSTM 을 돌렸을 때 train/val gap 폭발 실증
4. **§4** Option A vs Option C (last concat vs mean pool) 비교
5. **§5** 결론 + 4.3 OLSTM 탐색 공간 확장 설계

### 8.7 다음 토픽 예고

- **4.3 OLSTM (Optuna-LSTM)** ⭐ — 3.4 §16 의 `train_one_individual` 을 Optuna `objective` 로 감싸 IMF 별 자동 탐색. BiLSTM 포함.
- **5.1 Attention-LSTM** — BiLSTM + attention pooling (§6 Option D) 결합. time-step 가중을 학습 가능하게.
