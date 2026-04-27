# scripts/ 모듈 정의서 (Phase1_GRU)

> **목적**: Phase 1 GRU 협업 팀원이 `scripts/*.py` 의 공개 API · 책임 · 사용법을 한눈에 확인.
>
> **위치**: `시계열_Test/Phase1_GRU/scripts_정의서.md`
>
> **LSTM 대비 변경**: `models.py` (GRURegressor 신규), `dataset.py` (LSTMDataset→SequenceDataset),
> `setup.py` (bootstrap 메시지). 나머지는 Phase1_LSTM 과 동일.
>
> **마지막 갱신**: 2026-04-26

---

## 0. 개요

`scripts/` 는 Phase 1 GRU 노트북이 공통으로 import하는 함수·클래스 모음.

| 모듈 | 책임 | LSTM 대비 변경 | 상태 |
|---|---|---|---|
| `setup.py` | 환경 (한글 폰트·시드·경로 상수) | bootstrap 메시지만 변경 | ✅ |
| `targets.py` | 21일 누적 / 월별 타깃 생성·누수 검증 | 변경 없음 | ✅ |
| `dataset.py` | 텐서 데이터셋·Walk-Forward fold·스케일링 | `LSTMDataset` → `SequenceDataset` | ✅ |
| `models.py` | **GRU 회귀 모델 정의** | **`LSTMRegressor` → `GRURegressor`, nn.LSTM → nn.GRU** | ✅ |
| `train.py` | 학습 루프·체크포인트·device 관리 | 주석만 변경 (로직 동일) | ✅ |
| `metrics.py` | 평가 지표 (Hit Rate, R²_OOS, baseline 비교) | 모듈 docstring 첫 줄만 변경 | ✅ |

---

## 1. `setup.py` — 환경 부트스트랩

**변경 사항**: `bootstrap()` 출력 메시지 "LSTM" → "GRU". `BASE_DIR` 은 `Phase1_GRU/` 를 자동 가리킴.

```python
SEED: int = 42
BASE_DIR: Path        # Phase1_GRU/ (scripts/setup.py 의 parent.parent)
RESULTS_DIR: Path     # BASE_DIR / 'results'
RAW_DATA_DIR: Path    # RESULTS_DIR / 'raw_data'
SETTING_A_DIR: Path   # RESULTS_DIR / 'setting_A'
SETTING_B_DIR: Path   # RESULTS_DIR / 'setting_B'

setup_korean_font() -> str
fix_seed(seed: int = SEED) -> None
apply_display_defaults() -> None
ensure_result_dirs() -> None
bootstrap(seed: int | None = None, verbose: bool = True) -> str
```

---

## 2. `targets.py` — 타깃 시계열 + 누수 검증

**변경 없음**. Phase1_LSTM/scripts/targets.py 와 동일.

```python
build_daily_target_21d(adj_close: pd.Series) -> pd.Series
    # target[t] = sum(log_ret[t+1 : t+22])  — 다음 21영업일 누적 log-return

build_monthly_target_1m(adj_close: pd.Series) -> pd.Series
    # 월말 인덱스. target[m] = log(close[m+1]) - log(close[m])

verify_no_leakage(
    log_ret: pd.Series,
    target: pd.Series,
    n_checks: int = 3,
    seed: int = 42,
) -> None
    # assert + 육안 표 2종 검증
```

---

## 3. `dataset.py` — 텐서 데이터셋 · Walk-Forward 폴드

**변경 사항**: `LSTMDataset` → `SequenceDataset` (모델 무관 이름으로 변경). 로직 동일.

```python
class SequenceDataset(torch.utils.data.Dataset):
    """ (X, y) 텐서 보관. X: (N, seq_len, n_features), y: (N,) """
    def __init__(self, X: np.ndarray, y: np.ndarray): ...

make_sequences(
    arr: np.ndarray,
    seq_len: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]

walk_forward_folds(
    n: int,
    is_len: int,
    purge: int,
    emb: int,
    oos_len: int,
    step: int,
) -> List[Tuple[np.ndarray, np.ndarray]]

build_fold_datasets(
    series: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seq_len: int,
    extra_features: Optional[np.ndarray] = None,
    target_series: Optional[np.ndarray] = None,
) -> Tuple[SequenceDataset, SequenceDataset, StandardScaler]
```

**사용 예**:
```python
from scripts.dataset import SequenceDataset, walk_forward_folds, build_fold_datasets
```

---

## 4. `models.py` — GRU 회귀 모델 ⭐ (핵심 변경)

**변경 사항**: `LSTMRegressor` → `GRURegressor`, `nn.LSTM` → `nn.GRU`, `forget_gate_bias_init` 제거.

```python
class GRURegressor(nn.Module):
    """GRU 기반 회귀 모델 (시퀀스 → scalar 예측).

    Parameters
    ----------
    input_size : int           입력 피처 수
    hidden_size : int = 64     GRU hidden state 차원
    num_layers : int = 1       GRU 층수
    dropout : float = 0.3      Dropout 확률 (1층일 때 head_dropout 으로 우회)
    batch_first : bool = True  입력 형태 (B, T, F)

    Notes
    -----
    - GRU forward 반환: (output, h_n)  ← LSTM 의 (output, (h_n, c_n)) 와 다름
    - forget_gate_bias_init 없음 (GRU 는 forget gate 없음)
    - num_layers=1 dropout 함정: LSTMRegressor 와 동일하게 head_dropout 으로 우회
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GRU: out, _ = self.gru(x)  (LSTM: out, _ = self.lstm(x) 와 동일 패턴)
        # batch_first=True → out[:, -1, :] 로 마지막 시점 추출
        ...

count_parameters(model: nn.Module) -> int
```

**사용 예**:
```python
from scripts.models import GRURegressor, count_parameters

model = GRURegressor(input_size=1, hidden_size=32, num_layers=1, dropout=0.3)
print(count_parameters(model))  # ≈ 3,264 (LSTM hidden=32 대비 25% 감소)
```

**LSTM 과의 파라미터 비교** (hidden=32, input=1):

| 모델 | 파라미터 수 |
|---|---|
| `LSTMRegressor(1, 32, 1, 0.3)` | ≈ 4,352 |
| `GRURegressor(1, 32, 1, 0.3)` | ≈ 3,264 |
| **감소율** | **약 25%** |

---

## 5. `train.py` — 학습 루프

**변경 사항**: 주석/docstring 에서 "LSTM" → "GRU". 학습 루프 로직은 Phase1_LSTM 과 완전 동일.
`GRURegressor` 는 `nn.Module` 이므로 동일 인터페이스로 학습 가능.

```python
train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    max_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 0.01,
    grad_clip: float = 1.0,
    early_stop_patience: int = 10,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    device: str | torch.device = 'auto',
    verbose: bool = True,
    log_every: int = 1,
) -> Dict[str, Any]
    # 반환: best_state_dict, history, best_epoch, best_val_loss, stopped_early

get_device(preference: str = 'auto') -> torch.device
save_checkpoint(state_dict, path) -> None
load_checkpoint(path, device='cpu') -> Dict
```

---

## 6. `metrics.py` — 평가 지표

**변경 없음** (모듈 docstring 첫 줄만 "(GRU)" 추가). Phase1_LSTM 과 완전 동일.

```python
hit_rate(y_true, y_pred, exclude_zero=True) -> float     # 부호 적중률, 관문 > 0.55
r2_oos(y_true, y_pred) -> float                          # Campbell & Thompson 2008, 관문 > 0
r2_standard(y_true, y_pred) -> float
mae(y_true, y_pred) -> float
rmse(y_true, y_pred) -> float
baseline_metrics(y_test, y_train) -> Dict[str, Dict[str, float]]
summarize_folds(per_fold_metrics) -> Dict[str, Dict[str, float]]
```

---

## 변경 이력

| 날짜 | 변경 | 작성자 |
|---|---|---|
| 2026-04-26 | Phase1_GRU 최초 생성 — GRURegressor 신규, SequenceDataset 이름 변경, 나머지 동일 | 윤서 |
