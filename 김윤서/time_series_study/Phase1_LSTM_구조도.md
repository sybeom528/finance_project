# Phase 1 — LSTM 베이스라인 폴더 구조도

> **작성 목적**: `시계열_Test/Phase1_LSTM/` 전체 파일의 역할·내부 구성·상호 관계를 한 문서에서 파악하기 위한 구조 가이드.
>
> **작성일**: 2026-04-25 | **작성자**: 김윤서

---

## 1. 이 폴더가 존재하는 이유

전체 프로젝트 파이프라인 중 **Step 3 — DL 모델링** 에 해당한다.  
목적은 Black-Litterman Q 벡터 생성이 *아닌*, **LSTM 수익률 예측 성능의 정량 평가**다.

> 가설: "일별 수익률의 자기상관이 극단적으로 작아 univariate LSTM만으로는 예측력에 본질적 한계가 존재할 것"  
> 이 가설을 검증하는 것이 Phase 1의 핵심 산출물이다.

**6단계 모델 평가 로드맵 중 Phase 1 위치**:

| # | 모델 | 해상도 | Phase |
|---|---|---|---|
| 1 | LSTM | 일별 (21영업일 후) | **Phase 1 ← 현재** |
| 2 | LSTM | 월별 (1개월 후) | **Phase 1 ← 현재** |
| 3 | GRU | 일별 | Phase 2 |
| 4 | GRU | 월별 | Phase 2 |
| 5 | CEEMDAN+LSTM | 일별 | Phase 3 |
| 6 | CEEMDAN+LSTM | 월별 | Phase 3 |

---

## 2. 전체 폴더 구조 (트리)

```
시계열_Test/Phase1_LSTM/
│
├── 📋 문서 파일 (4개)
│   ├── README.md                      ← 협업 진입점 (실행 순서·규약·참고 링크)
│   ├── PLAN.md                        ← ⭐ 전체 구현 계획서 (진실원 팀 공유 사본)
│   ├── scripts_정의서.md              ← ⭐ scripts/*.py 공개 API 정의서
│   ├── 재천_WORKLOG.md                ← 팀 공통 작업·판단 일지
│   └── 윤서_WORKLOG.md                ← 윤서 담당 작업·설계 결정 일지
│
├── 📓 노트북 파일 (3개 완료, 2개 예정)
│   ├── 00_setup_and_utils.ipynb       ← ✅ 환경 설정 노트북 (%run 호환)
│   ├── 01_data_download_and_eda.ipynb ← ✅ 데이터 수집 + EDA + ACF
│   ├── 02_setting_A_daily21.ipynb     ← ✅ 설정 A 전체 완료 (§1~§10, Run All 준비됨)
│   ├── 03_setting_B_monthly.ipynb     ← ⏸ 예정 (설정 B)
│   └── 04_compare_A_vs_B.ipynb        ← ⏸ 예정 (A·B 비교 최종 보고)
│
├── 🐍 scripts/ (재사용 모듈 6개)
│   ├── __init__.py                    ← 패키지 마커
│   ├── setup.py                       ← ✅ 환경 부트스트랩
│   ├── targets.py                     ← ✅ 타깃 생성 + 누수 검증
│   ├── dataset.py                     ← ✅ 텐서 데이터셋 + Walk-Forward
│   ├── models.py                      ← ✅ LSTMRegressor
│   ├── train.py                       ← ✅ 학습 루프
│   └── metrics.py                     ← ✅ 평가 지표
│
└── 📁 results/ (실행 결과만 저장)
    ├── raw_data/                      ← SPY.csv, QQQ.csv (yfinance 원본)
    ├── setting_A/{SPY,QQQ}/           ← metrics.json + PNG 5종 (§9.A~§9.F 산출물)
    ├── setting_B/{SPY,QQQ}/           ← fold별 ckpt (추후)
    ├── cumulative_return_viz.png      ← 누적 vs 단순 수익률 시각화
    └── comparison_report.md           ← 04 노트북에서 자동 생성 (추후)
```

---

## 3. 노트북 실행 흐름

```
  결과: 환경 설정
  ┌──────────────────────────────────┐
  │  00_setup_and_utils.ipynb        │  ← 한글 폰트 · 시드(42) · 경로 상수
  └──────────────────────────────────┘
                  │ %run으로 모든 노트북에서 호출
                  ▼
  ┌──────────────────────────────────┐
  │  01_data_download_and_eda.ipynb  │  ← yfinance SPY·QQQ 다운로드
  └──────────────────────────────────┘   → results/raw_data/{SPY,QQQ}.csv
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  02_setting_A_daily21.ipynb      │  ← 설정 A 전체 (§1~§9)
  └──────────────────────────────────┘   → results/setting_A/{SPY,QQQ}/
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  03_setting_B_monthly.ipynb      │  ← 설정 B 전체 (예정)
  └──────────────────────────────────┘   → results/setting_B/{SPY,QQQ}/
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  04_compare_A_vs_B.ipynb         │  ← A·B 비교 + 관문 판정 (예정)
  └──────────────────────────────────┘   → results/comparison_report.md
```

---

## 4. 노트북별 상세 역할

### 00_setup_and_utils.ipynb
**역할**: 모든 노트북의 공통 환경을 초기화하는 단일 진실원.

`%run ./00_setup_and_utils.ipynb` 한 줄로 아래를 모두 처리:
- OS별 한글 폰트 설정 (`AppleGothic` / `Malgun Gothic` / `NanumGothic`)
- 난수 시드 고정 (Python·NumPy·PyTorch 모두, seed=42)
- 경로 상수 선언 (`RAW_DIR`, `RESULTS_DIR` 등)
- pandas/matplotlib 표시 옵션 적용

```python
# 모든 노트북 첫 번째 셀 패턴
%run ./00_setup_and_utils.ipynb
# → 출력: "Phase 1 — 환경 부트스트랩 완료" 박스
```

---

### 01_data_download_and_eda.ipynb
**역할**: 데이터 수집 + 탐색적 분석 + 자기상관 정량 확인.

| 섹션 | 내용 |
|---|---|
| §1 | yfinance로 SPY·QQQ 일별 OHLCV 다운로드 (2009-01-01~2026-03-31) |
| §2 | 결측·이상치 점검 (Adj Close 기준, ±5σ 임계) |
| §3 | 일별 log-return 분포 (히스토그램·QQ-plot) |
| §4 | ACF/PACF lag 1~30 — 자기상관 약함 가설 정량 확인 |

```python
# 핵심: Adj Close 기준 log-return 계산
import numpy as np
df['log_return'] = np.log(df['Adj Close']).diff()
# 결측 첫 1행 제거 후 분석 구간(2016~2025) 슬라이싱
```

---

### 02_setting_A_daily21.ipynb  ← 주 담당 노트북
**역할**: 설정 A (일별 + 21일 누적 forward log-return) 전체 흐름 실행.

| 섹션 | 내용 | 상태 |
|---|---|---|
| §1 | `%run ./00_setup_and_utils.ipynb` — 환경 초기화 | ✅ |
| §2 | SPY·QQQ CSV 로드 + log_return 계산 + 분석 구간(2016~2025) 슬라이싱 | ✅ |
| §3 | `build_daily_target_21d()` — 21일 누적 forward log-return 타깃 생성 | ✅ |
| §4 | `verify_no_leakage()` — assert + 육안 표 누수 검증 PASS | ✅ |
| §5 | LSTMDataset 임포트 + SEQ_LEN=126 설정, (B,T,F) 축 주의사항 명시 | ✅ |
| §6 | `walk_forward_folds()` — 106개 폴드 생성·검증, `build_fold_datasets()` 호출 | ✅ |
| §7 | `LSTMRegressor(1,128,2,0.2)` 임포트 + smoke test (199,297 params) PASS | ✅ |
| §8 | `build_train_val_loaders` + `run_all_folds` — SPY·QQQ 106폴드 학습 + **train/val 예측 수집** (과적합 진단용, 2026-04-25 추가) | ✅ |
| §9 | `hit_rate`·`r2_oos` 집계 + 베이스라인 비교 + 관문 판정 + `metrics.json` | ✅ |
| §9.A | 학습곡선 갤러리 — 선택 fold (0·25·50·75·마지막) × SPY·QQQ | ✅ |
| §9.B | best_epoch 분포 히스토그램 — 학습 진행도 진단 (`best_ep==1` 비율 확인) | ✅ |
| §9.C | 예측 분포 sanity — mean-prediction collapse 검출 (`pred_std/true_std` 비율) | ✅ |
| §9.D | 잔차 시계열 + 부호 혼동행렬 (2×2) — 체제 편향 검출 | ✅ |
| §9.E | fold별 R²_OOS / Hit Rate 박스플롯 — 이상치 분포 확인 | ✅ |
| §9.F | Train / Val / Test 동일지표 비교 — 과적합 갭 정량화 | ✅ |
| §10 | 결론·메모 + 학습자료_주의사항 준수 현황 표 | ✅ |

---

## 5. scripts/ 모듈 의존 관계

```
                ┌─────────────┐
                │  setup.py   │  환경 (폰트·시드·경로 상수)
                └──────┬──────┘
                       │ from scripts.setup import bootstrap
          ┌────────────┼────────────────────────┐
          │            │                        │
    ┌─────▼─────┐  ┌───▼───────┐  ┌────────────▼────┐
    │ targets.py│  │ dataset.py│  │    models.py     │
    │           │  │           │  │                  │
    │ 타깃 생성 │  │ 텐서 변환 │  │ LSTMRegressor    │
    │ 누수 검증 │  │ 폴드 생성 │  │ (batch_first=T)  │
    └─────┬─────┘  └───┬───────┘  └────────────┬────┘
          │            │                        │
          └────────────┼────────────────────────┘
                       │ 모두 노트북(02,03)에서 import
                  ┌────▼─────┐
                  │ train.py  │  학습 루프
                  └────┬──────┘
                       │ 학습 결과 → 평가
                  ┌────▼──────┐
                  │ metrics.py│  Hit Rate · R²_OOS
                  └───────────┘
```

---

## 6. scripts/ 모듈 상세

### 6-1. `setup.py` — 환경 부트스트랩

**공개 인터페이스 요약**:

```python
from scripts.setup import bootstrap, SEED, BASE_DIR, RESULTS_DIR

bootstrap()
# ============================================================
#   Phase 1 — 환경 부트스트랩 완료
# ============================================================
#   한글 폰트  : AppleGothic
#   시드       : 42
#   결과 경로  : .../시계열_Test/Phase1_LSTM/results
# ============================================================

# 경로 상수 (스크립트 위치 기반 자동 설정)
BASE_DIR       # Phase1_LSTM/
RESULTS_DIR    # Phase1_LSTM/results/
RAW_DATA_DIR   # Phase1_LSTM/results/raw_data/
SETTING_A_DIR  # Phase1_LSTM/results/setting_A/
SETTING_B_DIR  # Phase1_LSTM/results/setting_B/
```

**핵심**: `__file__` 기반으로 경로를 잡아 노트북 실행 위치에 무관하게 안정적.

---

### 6-2. `targets.py` — 타깃 생성 + 누수 검증

**왜 21일 누적인가?**

```
단순 방식 (채택 안 함):
  target[t] = log_ret[t+21]          ← t+21일 '하루' 수익률만 (노이즈 극대)

누적 방식 (채택):
  target[t] = sum(log_ret[t+1:t+22]) = log(P[t+21] / P[t])  ← 21일 전체 보유 수익률
```

Black-Litterman Q는 다음 리밸런싱 기간의 기대수익률 → **21일 전체 누적이 필요**.

```python
from scripts.targets import build_daily_target_21d, verify_no_leakage

# 1. 타깃 생성
target = build_daily_target_21d(adj_close)
# 내부 구현:
# log_ret = np.log(adj_close).diff()           # trailing diff (안전)
# target  = log_ret.rolling(21).sum().shift(-21) # forward 합 (예측 목표)
# → NaN: 마지막 21행 (shift(-21)이 첫 1행 NaN을 범위 밖으로 밀어냄)

# 2. 누수 검증 (2단계)
verify_no_leakage(log_ret, target, n_checks=3, seed=42)
# 단계 1 — Assert: target[t] == log_ret[t+1:t+22].sum() (3개 무작위 시점)
# 단계 2 — 육안 표: 첫 5행의 (날짜, log_ret, target, 직접계산, 일치여부)
```


---

### 6-3. `dataset.py` — 텐서 데이터셋 + Walk-Forward

**Walk-Forward 폴드 구조 (설정 A)**:

```
날짜 축 →
Fold 0: ├─── IS (231일) ───┤ purge(21) │ emb(21) │ OOS(21) ┤
Fold 1:    ├─── IS (231일) ───┤ purge(21) │ emb(21) │ OOS(21) ┤
Fold 2:         ├─── IS (231일) ───┤ purge(21) │ emb(21) │ OOS(21) ┤
...
Fold 105:                                              ... ┤ OOS(21) ┤

총 106개 폴드 (10년, 2016~2025)
```

**각 구간 의미**:

| 구간 | 길이 | 역할 |
|---|---|---|
| IS (In-Sample) | 231일 (~11개월) | 학습 데이터 |
| Purge | 21일 | 타깃(21일 누적) 누수 차단 |
| Embargo | 21일 | autocorrelation 잔존 차단 |
| OOS (Out-of-Sample) | 21일 (~1개월) | 테스트 데이터 |
| Step | 21일 | 다음 폴드 이동 단위 |

```python
from scripts.dataset import walk_forward_folds, build_fold_datasets, LSTMDataset

# 폴드 생성
folds = walk_forward_folds(n=2514, is_len=231, purge=21, emb=21, oos_len=21, step=21)
# → 106개 (train_idx, test_idx) 튜플 리스트

# 폴드 하나의 Dataset 생성
train_ds, test_ds, scaler = build_fold_datasets(
    series=spy_log_return,    # shape (T,)
    train_idx=folds[0][0],    # shape (231,)
    test_idx=folds[0][1],     # shape (21,)
    seq_len=126,               # LSTM 입력 시퀀스 길이 (약 6개월)
    target_series=target.values,  # 외부 타깃 (21일 누적 수익률)
)
# → train_ds.X.shape == (105, 126, 1)  ← (N_train, seq_len, n_features)
# → test_ds.X.shape  == (21, 126, 1)   ← (N_test, seq_len, n_features)

# ⚠️ 누수 방지: scaler는 train_idx 로만 fit, test에는 transform만!
```

**시퀀스 생성 원리 (SEQ_LEN=126 기준)**:

```
날짜: t=0  t=1  ...  t=125 │ t=126  t=127 ...
      └────── X[0] ────────┘    ← seq_len=126개 입력

      t=1  t=2  ...  t=126  │ t=127
      └────── X[1] ────────┘    ← 1일 슬라이딩

IS=231일 → N_train = 231 - 126 = 105개 시퀀스
OOS=21일 → N_test  = 21개 시퀀스
```

---

### 6-4. `models.py` — LSTMRegressor

**구조**:

```
입력 x: (B, T, F) = (batch_size, 126, 1)
         │
    ┌────▼────────────────┐
    │  nn.LSTM            │  hidden=128, num_layers=2, dropout=0.2, batch_first=True
    └────┬────────────────┘
         │  output: (B, T, H) = (B, 126, 128)
         │  output[:, -1, :]  ← 마지막 시점 hidden만 추출
         │  shape: (B, 128)
    ┌────▼────────────────┐
    │  head_dropout       │  num_layers=1일 때만 실제 dropout (학습자료 §4.4 대응)
    │  nn.Dropout(0.2)    │  num_layers>1이면 nn.Identity()
    └────┬────────────────┘
    ┌────▼────────────────┐
    │  nn.Linear(128, 1)  │
    └────┬────────────────┘
         │  .squeeze(-1)
출력 y: (B,)  ← 각 시퀀스의 예측 수익률 scalar
```

```python
from scripts.models import LSTMRegressor, count_parameters

# 설정 A (2-layer)
model = LSTMRegressor(input_size=1, hidden_size=128, num_layers=2,
                      dropout=0.2, batch_first=True)
print(count_parameters(model))  # 199,297개 파라미터

# 설정 B (1-layer, head_dropout 자동 적용)
model_b = LSTMRegressor(input_size=1, hidden_size=64, num_layers=1, dropout=0.3)
# → model_b.head_dropout == nn.Dropout(0.3)  (LSTM dropout 인자 우회)
```

> ⚠️ **핵심 주의사항**: `batch_first=True` 가 기본값.  
> DataLoader 출력 형태가 `(B, T, F)`이므로, `batch_first=False`로 실수하면  
> PyTorch가 `(T, B, F)`를 기대해 silent shape mismatch가 발생한다.

---

### 6-5. `train.py` — 학습 루프

**학습 구성 (PLAN.md 확정 사양)**:

| 요소 | 선택 | 근거 |
|---|---|---|
| Loss | HuberLoss(delta=0.01) | 수익률 이상치 강건 |
| Optimizer | AdamW(lr=1e-3, wd=1e-4) | 가중치 감쇠 포함 |
| Scheduler | ReduceLROnPlateau(patience=5, factor=0.5) | val loss 정체 시 lr 감소 |
| Gradient clip | max_norm=1.0 | LSTM 폭발적 기울기 방지 |
| EarlyStopping | patience=10 | 과적합 방지 |
| Checkpoint | best val loss 기준 | 최적 시점 모델 저장 |

```python
from scripts.train import train_one_fold, save_checkpoint, get_device
from torch.utils.data import DataLoader

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

result = train_one_fold(
    model, train_loader, val_loader,
    max_epochs=100,
    early_stop_patience=10,
    device='auto',   # cuda > mps > cpu 자동 선택
)
# result 키: 'best_state_dict', 'history', 'best_epoch', 'stopped_early'
# (2026-04-25 추가) 'y_true_train', 'y_pred_train', 'y_true_val', 'y_pred_val'
# → §9.F Train/Val/Test 동일지표 비교에 사용

save_checkpoint(result['best_state_dict'], 'results/setting_A/SPY/fold_0.pt')
```

**`run_all_folds` 반환 구조 (02_setting_A_daily21.ipynb §8)**:

```python
fold_out = [
    {
        'k':           fold 번호,
        'y_true':      OOS 실측값,
        'y_pred':      OOS 예측값,
        'y_true_train': train 실측값,   # 과적합 진단용 (2026-04-25 추가)
        'y_pred_train': train 예측값,
        'y_true_val':   val 실측값,
        'y_pred_val':   val 예측값,
        'best_val_loss': 최소 val Huber loss,
        'best_epoch':    최소 val loss 에포크,
        'stopped_early': EarlyStopping 발동 여부,
        'history':       {'train_loss': [...], 'val_loss': [...]},
    },
    ...  # 106개 fold
]
```

**학습 루프 함정 방어 체크리스트** (학습자료_주의사항 §3.6):

```python
for epoch in range(max_epochs):
    model.train()                       # ✅ train 모드 전환
    for xb, yb in train_loader:
        optimizer.zero_grad()           # ✅ 매 배치 기울기 초기화 (누락 시 누적)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✅ 기울기 폭발 방지
        optimizer.step()
        train_loss.append(loss.item())  # ✅ .item() 로 그래프 detach (메모리 누수 방지)

    model.eval()                        # ✅ eval 모드 전환
    with torch.no_grad():               # ✅ 기울기 계산 비활성화
        for xb, yb in val_loader:
            ...
```

---

### 6-6. `metrics.py` — 평가 지표

**Phase 1 관문**:

```
Phase 2 진행 조건: Hit Rate > 0.55  AND  R²_OOS > 0  (둘 다 충족 필요)
```

```python
from scripts.metrics import hit_rate, r2_oos, baseline_metrics, summarize_folds

# 1차 지표 (관문 판정)
hr  = hit_rate(y_test, y_pred)   # 방향 적중률 (관문: > 0.55)
r2  = r2_oos(y_test, y_pred)     # 0 예측 대비 개선 (관문: > 0)

# 공식:
# hit_rate   = mean(sign(y_true) == sign(y_pred))
# r2_oos     = 1 - sum((y - ŷ)²) / sum(y²)   ← Campbell & Thompson (2008)
#              분모: sum(y²)  ≠  표준 R²의 SST (분산)

# 베이스라인 비교 (3종)
bl = baseline_metrics(y_test, y_train)
# {'zero': {...}, 'previous': {...}, 'train_mean': {...}}
# 각각 hit_rate, r2_oos, r2_standard, mae, rmse 포함

# 106폴드 집계
summary = summarize_folds(per_fold_results)
# {metric: {'mean': ..., 'std': ..., 'min': ..., 'max': ..., 'n': ...}}
print(f"Hit Rate: {summary['hit_rate']['mean']:.4f} ± {summary['hit_rate']['std']:.4f}")
```

**R²_OOS vs 표준 R² 차이**:

```
표준 R²  = 1 - SSE / SST_mean    (분모: 평균 예측 baseline 대비)
R²_OOS  = 1 - SSE / sum(y²)      (분모: 0 예측 baseline 대비)

수익률은 평균이 0에 가까움 → sum(y²) ≈ SST_mean
그러나 엄밀하게는 다르며, R²_OOS가 금융 예측 문헌의 표준 (Campbell & Thompson 2008)
```

---

## 7. 핵심 개념: Walk-Forward + Purge + Embargo

### 왜 일반 K-Fold가 아닌가?

```
일반 K-Fold (금융 시계열에서 사용 불가):
  Fold 1: [테스트] [학습] [학습] [학습] [학습]   ← 미래 정보가 과거 예측에 사용됨
  Fold 2: [학습] [테스트] [학습] [학습] [학습]   ← 데이터 누수!

Walk-Forward (시간 순서 엄수):
  Fold 1: [학습] → [테스트]
  Fold 2:   [학습] → [테스트]
  Fold 3:     [학습] → [테스트]
```

### 왜 Purge와 Embargo가 필요한가?

```
타깃 정의: target[t] = log(P[t+21] / P[t])  → t+1 ~ t+21 가격 참조

                IS 끝        OOS 시작
                  ↓              ↓
... [t-3][t-2][t-1][ t ]     [ t+42 ][ t+43 ]...
                    │
                    └─ target[t] = sum(log_ret[t+1:t+22])
                                                  ↑
                                     t+21 가격이 OOS 구간에 포함됨!
                                     → IS의 마지막 21개 샘플 제거 (Purge=21)

Embargo=21: IS 학습 직후 autocorrelation 잔존이 OOS 첫 구간 성능을 부풀리는 것 방지
```

**gap = purge + embargo = 42일**:

```
[── IS 231일 ──][── gap 42일 ──][── OOS 21일 ──]
                 └─purge 21──┘└─embargo 21─┘
```

---

## 8. 데이터 변환 파이프라인 (타깃 → 텐서까지)

```
Adj Close (원본)
    │
    │ np.log().diff()
    ▼
log_return: shape (T,)   ← T = 2514 (2016~2025)
    │
    ├──────────────────────────────┐
    │                              │
    │ build_daily_target_21d()     │ walk_forward_folds() → 106개 fold
    ▼                              ▼
target: shape (T,)         folds[k] = (train_idx, test_idx)
    │ 마지막 21행 NaN           각각 shape (231,), (21,)
    │
    └── build_fold_datasets(series=log_return, target_series=target, ...)
              │
              │ StandardScaler.fit(log_return[train_idx])  ← train only!
              │ scaler.transform(log_return)               ← 전체 transform
              │
              ├── make_sequences(scaled[train_idx], seq_len=126)
              │       → X_tr: (105, 126, 1),  y_tr: target[train_idx[j+125]]
              │
              └── sliding window on test_idx
                      → X_te: (21, 126, 1),   y_te: target[test_idx[k]]
              │
              ▼
    LSTMDataset(X_tr, y_tr)  +  LSTMDataset(X_te, y_te)
              │
    DataLoader(batch_size=32)
              │
              ▼
    배치: x=(32, 126, 1), y=(32,)   ← (B, T, F)
              │
    LSTMRegressor(input_size=1, hidden_size=128, num_layers=2)
              │
              ▼
    예측: ŷ=(32,)
```

---

## 9. 설정 A vs 설정 B 비교

| 항목 | 설정 A (일별) | 설정 B (월별) |
|---|---|---|
| 타깃 | 21영업일 누적 forward log-return | 다음 달 log-return |
| seq_len | **126** (약 6개월) | **24** (개월) |
| hidden_size | **128** | **64** |
| num_layers | **2** | **1** |
| dropout | 0.2 (LSTM 내부) | 0.3 (head_dropout) |
| IS | 231일 | 11개월 |
| purge / emb | 21 / 21 | 1 / 1 |
| OOS | 21일 | 1개월 |
| 폴드 수 | ~106 | ~106 |
| 과적합 위험 | 보통 | 높음 (샘플 수 적음) |
| 타깃 함수 | `build_daily_target_21d` | `build_monthly_target_1m` |
| 노트북 | `02_setting_A_daily21.ipynb` | `03_setting_B_monthly.ipynb` |

---

## 10. 누수 방지 4계층 방어선

| # | 위치 | 위험 | 방어 |
|---|---|---|---|
| 1 | 타깃 shift 부호 | `shift(-21)` 대신 `shift(21)` → 미래 누수 | §4 assert + 육안 표 (2종 검증) |
| 2 | rolling 윈도우 정렬 | 기본 trailing이지만 forward 합이어야 함 | `.shift(-21)` 명시 + 첫 5행 검증 |
| 3 | Scaler fit 범위 | 전체에 fit → val/test 통계 누설 | **train_idx 에만 fit**, 나머지 transform만 |
| 4 | seq_len 경계 | 126일 윈도우가 split 경계를 넘으면 누수 | purge+embargo 구간 42일로 seq_len 126보다 짧아 엄밀하지 않으나, purge가 타깃 누수, embargo가 autocorrelation을 각각 방어 |
| 5 | EarlyStopping val 사용 | val을 HP 선택에도 쓰면 val이 train화 | val은 EarlyStop·LR scheduler에만, 최종 메트릭은 test로만 보고 |

---

## 11. 참고 문서 링크

| 문서 | 위치 | 내용 |
|---|---|---|
| PLAN.md | `시계열_Test/Phase1_LSTM/PLAN.md` | 전체 구현 계획 (진실원) |
| scripts_정의서.md | `시계열_Test/Phase1_LSTM/scripts_정의서.md` | 모든 모듈 API 정의 |
| 학습자료_주의사항.md | `시계열_Test/학습자료_주의사항.md` | 함정·주의사항 체크리스트 |
| 윤서_WORKLOG.md | `시계열_Test/Phase1_LSTM/윤서_WORKLOG.md` | 설계 결정 일지 (윤서 담당) |
| 재천_WORKLOG.md | `시계열_Test/Phase1_LSTM/재천_WORKLOG.md` | 팀 공통 작업 일지 |
| 원 논문 | `김윤서/time_series_study/` PDF | Su et al. 2026, ESWA 295 |
