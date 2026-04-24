# 2.3 시계열 누수 (Look-ahead Bias) 방지

> **이 토픽의 목표**
> "sliding window 까지 잘 만들었는데 나중에 OOS 성능이 이상하게 좋은" 함정의 원인을 세 가지 **누수 유형**으로 구분해 설명하고, 현재 프로젝트의 walk-forward 코드가 어떻게 이를 막고 있는지 해부할 수 있게 됩니다.

---

## 0. 한 줄 요약

- **Look-ahead bias** = 모델이 학습 시점에 "알 수 없었어야 할 미래 정보"를 입력으로 받는 오류.
- 누수는 주로 세 경로로 들어옵니다: ① 레이블 자체, ② 전체 기간 통계를 쓴 정규화, ③ train/val 경계에서 샘플의 시간 범위 중복.
- 전통적 K-Fold 는 마지막 경로를 **무조건 열어버리므로**, 시계열에는 반드시 순서 기반 분할을 써야 합니다.
- 현재 프로젝트의 `make_wf_windows` 는 **purge + embargo** 를 모두 적용하고 있어 López de Prado 의 권장 패턴을 충실히 따릅니다.

---

## 1. Look-ahead Bias 가 왜 치명적인가

백테스트에서 조금만 미래 정보가 새어 들어와도 성능 지표가 **극적으로 과장**됩니다.

### 1.1 현실에서 모를 수 없는 것

- 내가 t 시점에 예측을 내릴 때, t+1 일 이후의 어떤 정보도 가질 수 없음
- "오늘의 종가" 조차 장 마감 뒤에야 확정됨 → 장중 예측이라면 전일 종가 기준
- "이번 분기 EPS" → 공시 전까지 사용 불가

### 1.2 누수가 성능을 과장하는 경로

```
올바른 백테스트 Sharpe = 1.1
  ↓ 정규화 누수 한 줄
Sharpe = 1.8
  ↓ 레이블 누수 (future 수익률이 피처에 섞임)
Sharpe = 3.7
  ↓ 실제 운용 → 손실
```

연구자가 "논문의 결과를 재현하지 못하는" 경우 대부분의 원인은 이 두세 줄의 누수입니다.

---

## 2. 세 가지 누수 유형

### 2.1 Label Leakage (레이블 누수)

**정의:** 예측 대상 `y` 계산에 쓰이는 원시값이 입력 `X` 에도 포함되는 경우.

#### 구체적 시나리오

```python
# BAD — log_return_1d 가 X에도 있고, y(내일 수익률)도 같은 값의 함수
X = df[['log_return_1d', ...]]   # t일의 수익률
y = df['log_return_1d'].shift(-1) # t+1일의 수익률
#  → X[t] 와 y[t] 는 무관 (OK)

# BAD — fwd_ret_21d 가 피처에 섞임
X = df[['log_return_1d', 'fwd_ret_21d', ...]]   # 앗
y = df['fwd_ret_21d']
#  → 완벽한 예측 발생. 당연히 Sharpe ∞.
```

현재 프로젝트의 17개 피처 목록엔 `fwd_ret_21d` 가 없으므로 이 직접 누수는 발생하지 않습니다.

#### 간접 누수 — 파생 피처 계산 시점

```python
# BAD — "지난 21일 평균 수익률" 을 rolling 의 center=True 로 계산
df['ret_21d_center'] = df['log_return_1d'].rolling(21, center=True).mean()
# → t 시점의 값이 t-10 ~ t+10 평균이 되어 미래 10일치가 섞임
```

피처 엔지니어링 단계에서 `rolling(..., center=True)` 나 `shift(-k)` 같은 음수 shift 는 **반드시 의심**해야 합니다.

#### 체크리스트

- `X` 계산에 쓰인 원시 시계열 중 어느 하나라도 `t` 이후 시점을 참조하는가?
- 이동평균·지수평활 피처에서 `center=True` 가 들어있는가?
- 팩터 값(`mkt_rf`, `smb` 등)이 **발표일** 기준인가 **계산 대상 일자** 기준인가?

---

### 2.2 Normalization Leakage (정규화 누수)

**정의:** 피처 정규화를 **전체 기간 통계**로 한 뒤 train/test 를 나누는 경우.

#### 수학적 설명

Z-score 스케일링:

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

$\mu$ 와 $\sigma$ 를 전체 기간에서 계산하면, **test 구간의 미래 값들이 $\mu, \sigma$ 계산에 참여**합니다. 이 상태에서 test 샘플을 넣으면:

$$
x^{\text{test}}_{\text{scaled}} = \frac{x^{\text{test}} - \mu_{\text{전체}}}{\sigma_{\text{전체}}}
$$

$\mu_{\text{전체}}$ 는 "아직 알 수 없는" test 값의 평균이 이미 반영된 값입니다.

#### 얼마나 차이날까

시뮬레이션 결과 (실습 노트북에서 재현):
- BAD (전체 기간 μ/σ 사용): test MSE 낮게 측정됨
- GOOD (train 구간 μ/σ 사용): 현실적인 MSE

차이는 데이터의 비정상성(non-stationarity) 정도에 따라 달라지는데, 금융 수익률 처럼 **분산이 시간에 따라 크게 변하는** 경우에는 특히 크게 왜곡됩니다.

#### 올바른 순서

```python
# GOOD
scaler = StandardScaler()
scaler.fit(X_train)                    # ← train 에서만 통계 계산
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # 같은 μ, σ 적용
```

**sklearn 사용자 흔한 실수**
```python
# BAD
scaler = StandardScaler().fit(X)       # 전체 데이터로 fit
X_train = scaler.transform(X[:train_end])
X_test  = scaler.transform(X[train_end:])
```

---

### 2.3 Window Leakage (윈도우 시간 중복)

**정의:** train/test 분할이 시간축상 겹쳐서, train 샘플의 lookback 또는 레이블이 test 샘플과 같은 날짜를 포함하는 경우.

Sliding window 구조에선 **두 가지** 중복이 생길 수 있습니다.

#### 2.3.1 Lookback 중복 (입력 중복)

```
샘플 t-1 의 입력: [x_{t-T}, ..., x_{t-1}]   ← train 에 포함
샘플 t   의 입력: [x_{t-T+1}, ..., x_t  ]   ← test 로 넘어감

→ 두 입력이 T-1 일치 공유. train 샘플이 test 입력을 이미 "본" 상태.
```

#### 2.3.2 레이블 중복 (horizon leakage)

```
샘플 t-10 의 레이블: fwd_ret_21d(t-10) = (price_{t+11} - price_{t-10}) / ...
샘플 t 의 입력 : [x_{t-T+1}, ..., x_t]

→ 샘플 t-10 의 **레이블이 알려주는 미래 정보**가 샘플 t 의 입력 기간과 겹침.
  즉 train 의 y 로부터 test의 X 내용이 유추 가능.
```

이것이 López de Prado 가 "**purge**" 와 "**embargo**" 로 막으려 한 문제입니다.

#### Purge — IS 끝부분 잘라내기

IS 구간 **끝에서 h일을 제거**하면 그 부분의 y 가 OOS 시작 시점 이후의 가격을 참조하지 않게 됩니다.

```
IS 원래:         [ i ─────────── is_end ]  ← is_end - h ~ is_end 의 y 가 OOS 구간 가격 참조
IS purge 후:     [ i ────── is_end - h ]    ← 안전
                                     └── h일 버림
```

현재 프로젝트 코드:
```python
purge = is_end - oos_days    # = is_end - 21
is_w  = dates[i : purge]     # IS 끝 21일 제거
```

#### Embargo — IS 와 OOS 사이 공백

Purge 했어도 **피처 쪽의 lookback**이 OOS 쪽으로 넘어갈 수 있습니다. 추가로 IS 와 OOS 사이에 **embargo_days** 만큼의 완충지대를 둡니다.

```
IS 끝 → [ embargo ] → OOS 시작
           ↑ 이 구간은 아무 샘플도 쓰지 않음
```

현재 프로젝트 코드:
```python
oos_start = is_end + embargo   # embargo = 21
```

#### 왜 K-Fold 는 금융에 쓰면 안 되나

K-Fold 는 샘플을 **무작위로** 5~10개 폴드로 나누고 교차 검증합니다. 시계열에선:

```
[1][2][3][4][5][6][7][8][9][10][11][12]  ← 시간순 샘플
   ↓ K-Fold(k=5)
 train:  [1][2] [4][5][6][7] [9][10][11][12]
 val:            [3]               [8]

val 샘플 [3] 의 lookback 에 [2] 가 들어있음 → leakage
val 샘플 [8] 의 레이블이 [9] 시점의 가격을 참조 → leakage
```

**절대 금물.** 시계열엔 순서를 지키는 분할(TimeSeriesSplit, Walk-Forward, CPCV) 만 써야 합니다.

---

## 3. Walk-Forward (Expanding / Rolling)

### 3.1 개념

```
t 축 →

윈도우 0:  [──────── IS ────────][emb][── OOS ──]
윈도우 1:        [──────── IS ────────][emb][── OOS ──]
윈도우 2:              [──────── IS ────────][emb][── OOS ──]
                step 만큼 우측 이동
```

- 각 윈도우마다 IS 로 학습 → OOS 로 평가
- 모든 OOS 성능을 이어붙여 최종 지표(Sharpe, MDD 등) 계산

### 3.2 Expanding vs Rolling

| 방식 | IS 구간 | 장점 | 단점 |
|---|---|---|---|
| **Rolling (고정 창)** | 항상 `IS_DAYS` 만큼 | 최신 체제 반영, 예전 정보 탈락 | 오래된 체제 변화에 약함 |
| **Expanding (누적)** | 초기값부터 계속 늘어남 | 데이터 많이 활용 | 오래된 정보가 모델을 흐림 |

**현재 프로젝트**: Rolling (`IS_DAYS = 252`, `STEP_SIZE = 21`)
- 한 번에 1년치 IS 로 학습, 1달치 OOS 평가, 1달씩 창을 밀며 반복

### 3.3 TimeSeriesSplit (sklearn) 은 특수 케이스

`sklearn.model_selection.TimeSeriesSplit` 은 expanding + `test_size=1/n_splits` 버전입니다. 간단한 검증엔 유용하지만 **embargo 옵션이 없어** 금융 시계열에선 `make_wf_windows` 같은 수제 함수를 씁니다.

---

## 4. 현재 프로젝트 `make_wf_windows` 해부

`Step3_IT_WalkForward_panel.ipynb` 의 구현을 한 줄씩 뜯어보겠습니다.

```python
def make_wf_windows(dates, is_days=252, embargo=21, oos_days=21, step=21):
    windows, n, i = [], len(dates), 0
    while True:
        is_end    = i + is_days                # (1) IS 끝 인덱스
        oos_start = is_end + embargo           # (2) OOS 시작 = IS 끝 + 공백
        oos_end   = oos_start + oos_days       # (3) OOS 끝
        if oos_end > n:
            break
        purge  = is_end - oos_days             # (4) ★ IS 끝에서 h일 잘라냄
        is_w   = dates[i:purge]                # (5) IS = [i, is_end-21)
        oos_w  = dates[oos_start:oos_end]      # (6) OOS = [is_end+21, is_end+42)
        if len(is_w) > 0 and len(oos_w) > 0:
            windows.append((is_w, oos_w))
        i += step                              # (7) 21일씩 우측 이동
    return windows
```

**주요 파라미터 (현재 설정)**
- `is_days = 252` → 1년치 학습
- `embargo = 21` → IS ↔ OOS 사이 공백 21일
- `oos_days = 21` → 1개월치 평가
- `step = 21` → 1개월씩 윈도우 이동

**실제 IS 길이**: `is_days - oos_days = 252 - 21 = 231` 일
- 즉 "IS" 라고 부르지만 실제로는 **231일** 사용. 뒤 21일은 purge.
- 이유: `fwd_ret_21d` 가 뒤쪽 21일에선 OOS 가격을 참조하므로.

**IS ↔ OOS 총 간격**: `embargo + oos_days = 42` 일
- OOS 시작 시점 기준, IS 의 마지막 데이터는 42일 전 시점.
- 이 간격 덕에 어떤 sliding window 샘플도 train/val 에 동시에 기여하지 않음.

이 구조 덕분에 walk-forward 를 돌려도 **각 OOS 성능이 독립적인 추정치**가 되고, 누적 성능(Sharpe 등)이 왜곡 없이 계산됩니다.

---

## 5. Sequential 모델 교체 시 주의점

XGBoost → GRU 로 교체할 때 추가로 조심할 것이 **두 가지**입니다.

### 5.1 lookback 이 IS 경계를 넘지 않는지

Sliding window (T=60) 를 IS 내부에서 만들면:
- IS 첫 샘플의 lookback 시작 = IS 첫 날
- IS 첫 샘플의 lookback 끝   = IS 첫 날 + 59

IS 앞쪽 59 일은 sequential 샘플이 안 만들어짐 → 샘플 수 감소.
**해결**: IS 시작 전 60일까지 데이터를 당겨서 쓰되, 그 구간은 lookback으로만 쓰고 **샘플·레이블을 만들지 않음**.

```python
# IS 구간이 dates[i:purge] 일 때, lookback 을 고려해 60일 더 앞에서 데이터 추출
buffered_start = max(0, i - T + 1)
lookback_extended = dates[buffered_start : purge]
# sequential 샘플은 lookback_extended 위에서 만들되,
# y index 는 원래 IS 경계 안에만 해당하도록 자름
```

### 5.2 정규화를 IS 안에서만 fit

XGBoost 는 정규화가 필요 없지만, GRU 는 **입력 스케일에 민감**합니다.
- fit: `scaler.fit(IS_X)` — IS 기간 내 피처 값만 사용
- transform: IS, OOS 에 동일한 `scaler.transform`

**절대 하지 말 것:**
```python
# BAD — 전체 기간으로 fit
scaler.fit(pd.concat([IS_X, OOS_X]))
```

시계열 + GRU 조합에서 **거의 모든 연구자들이 한 번 이상 저지르는 실수**입니다.

---

## 6. 최종 체크리스트

2주차 이후 실제 코드를 짤 때 반드시 점검할 항목:

**레이블 쪽**
- [ ] `y` 계산에 쓰이는 원시값 (`adj_close` 등) 이 `X` 에 포함되는가? → 제거
- [ ] 파생 피처에서 `shift(-k)` 나 `rolling(center=True)` 쓴 적 있는가? → 재검토
- [ ] `y` 의 horizon 일수(h) 만큼 IS 끝을 **purge** 했는가?

**정규화 쪽**
- [ ] `scaler.fit(...)` 의 입력은 **IS 구간 한정**인가?
- [ ] `StandardScaler` 대신 전체 기간에 대한 `(X - X.mean()) / X.std()` 를 쓰진 않았는가?

**분할 쪽**
- [ ] K-Fold 나 shuffle=True 가 들어간 분할을 쓰진 않는가?
- [ ] IS ↔ OOS 사이에 **embargo** 를 두었는가?
- [ ] Sliding window 를 만들 때 lookback 이 IS 경계를 넘어가지 않는가? (혹은 경계 외 데이터를 쓸 때 y 는 IS 내부로 제한)

---

## 7. 스스로 점검

1. 전체 기간 통계로 Z-score 정규화를 하면 왜 OOS 성능이 과장되는지, 수식으로 설명하세요.
2. `fwd_ret_21d` 를 예측할 때, IS 끝에서 21일을 purge 하지 않으면 어떤 누수가 발생합니까?
3. K-Fold 로 금융 시계열을 검증하면 왜 위험한지 세 문장으로 정리하세요.
4. Walk-forward 의 rolling 방식과 expanding 방식의 차이점을 장단점 관점에서 설명하세요.
5. 현재 프로젝트에서 실제 IS 학습에 쓰이는 일수는 며칠이며, 왜 명목 IS_DAYS(252)와 차이가 납니까?
6. GRU 로 교체할 때 lookback 때문에 IS 앞부분 샘플이 줄어드는 문제를 어떻게 해결할 수 있습니까?

---

## 부록 A — López de Prado 의 CPCV (참고)

**Combinatorial Purged Cross-Validation** — 여러 OOS 윈도우를 조합해 순서에 덜 민감하게 만드는 기법. 본 프로젝트에선 Walk-Forward 로 충분하지만, 더 엄격한 연구 환경에선 CPCV 를 씁니다.

핵심 아이디어:
1. 시간축을 N 개 구간으로 나눔
2. 그중 k 개를 "test" 로 선택, 나머지는 train
3. test 주변에 embargo 를 두고 train 샘플을 purge
4. 가능한 모든 조합에 대해 반복

단일 walk-forward 보다 훨씬 많은 독립 추정치를 얻을 수 있지만, 연산 비용도 그만큼 커집니다.

---
