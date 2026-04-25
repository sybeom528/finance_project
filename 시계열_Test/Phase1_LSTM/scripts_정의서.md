# scripts/ 모듈 정의서

> **목적**: Phase 1 협업 팀원이 `scripts/*.py` 의 공개 API · 책임 · 사용법을 한눈에 확인할 수 있도록 정리한 문서.
>
> **위치**: `시계열_Test/Phase1_LSTM/scripts_정의서.md` (협업 폴더 루트, README.md · PLAN.md · 재천_WORKLOG.md 와 같은 레벨)
>
> **갱신 정책**:
>
> - 새 모듈 추가 → 모듈 목록 + 상세 섹션 + 변경 이력 3곳 모두 갱신
> - 공개 인터페이스(함수 시그니처·파라미터) 변경 → 상세 섹션 + 변경 이력 갱신 + 본인 `<이름>_WORKLOG.md` 기록
> - 내부 구현만 변경 → 변경 이력 기록은 선택, 본인 WORKLOG 만 필수
>
> **마지막 갱신**: 2026-04-24 (재천)

---

## 0. 개요

`scripts/` 는 Phase 1 노트북 (`00_~04_*.ipynb`) 이 공통으로 import 하여 재사용하는 함수·클래스 모음입니다. 노트북은 **분석 흐름·시각화·해석** 만 담당하고, 재사용 가능한 함수·클래스·상수는 모두 `scripts/*.py` 에 정의됩니다.

**모듈 작성 원칙**:

- 모든 public 함수: type hints + Numpy-style docstring 필수
- 누수 위험 라인 (shift, rolling, resample, scaler.fit) 에 인라인 주석 의무
- 변수명은 영어 (한글 컬럼명은 유지 가능)

---

## 1. 모듈 목록 (요약)

| 모듈 | 책임 | 담당 | 상태 |
|---|---|---|---|
| `setup.py` | 환경 (한글 폰트·시드·경로 상수) | 재천 | ✅ |
| `targets.py` | 21일 누적 / 월별 타깃 생성·누수 검증 | 재천 (build_monthly만) + 다른 팀원 | ✅ |
| `dataset.py` | 텐서 데이터셋·Walk-Forward fold·스케일링 | 윤서 (+ target_series 인자) | ✅ |
| `models.py` | LSTM 회귀 모델 정의 | 재천 | ✅ |
| `train.py` | 학습 루프·체크포인트·device 관리 | 재천 | ✅ |
| `metrics.py` | 평가 지표 (Hit Rate, R²_OOS, baseline 비교) | 재천 | ✅ |

향후 추가 가능 (현재 미정):

- `plot_utils.py` — 학습 곡선·예측 vs 실측 시각화 (02 §9 에서 언급)
- (Phase 2 진입 시) `models.py` 에 `GRURegressor` 추가

---

## 2. 모듈 상세

### 2.1 `setup.py` — 환경 부트스트랩

**책임**: Phase 1 전반에 공통으로 쓰이는 한글 폰트 · 시드 · 경로 상수를 단일 진실원으로 제공.

**공개 인터페이스**:

```python
SEED: int = 42
BASE_DIR: Path        # Path(__file__).resolve().parent.parent (Phase1_LSTM 위치)
RESULTS_DIR: Path     # BASE_DIR / 'results'
RAW_DATA_DIR: Path    # RESULTS_DIR / 'raw_data'
SETTING_A_DIR: Path   # RESULTS_DIR / 'setting_A'
SETTING_B_DIR: Path   # RESULTS_DIR / 'setting_B'

setup_korean_font() -> str                                  # OS별 한글 폰트 적용
fix_seed(seed: int = SEED) -> None                          # Python·NumPy·PyTorch 시드 + 결정성
apply_display_defaults() -> None                            # pandas/numpy 표시 옵션
ensure_result_dirs() -> None                                # results/ 하위 디렉토리 자동 생성
bootstrap(seed: int | None = None, verbose: bool = True) -> str  # 위 4개 일괄 호출
```

**의존성**: `numpy`, `pandas`, `torch`, `matplotlib`, `platform`, `pathlib`

**사용 예**:

```python
# 노트북에서 (00_setup_and_utils.ipynb 통해 %run)
from scripts.setup import bootstrap, BASE_DIR, RAW_DATA_DIR
bootstrap()
```

---

### 2.2 `targets.py` — 타깃 시계열 + 누수 검증

**책임**: 설정 A (21일 누적) / 설정 B (월별 1개월) 타깃 시계열 생성 + 누수 검증.

**공개 인터페이스**:

```python
build_daily_target_21d(adj_close: pd.Series) -> pd.Series
    # target[t] = sum(log_ret[t+1 : t+22])  — 다음 21영업일 누적 log-return
    # 마지막 21행 NaN

build_monthly_target_1m(adj_close: pd.Series) -> pd.Series
    # 월말 인덱스. target[m] = log(close[m+1]) - log(close[m])  — 다음 달 수익률
    # 마지막 1행 NaN

verify_no_leakage(
    log_ret: pd.Series,
    target: pd.Series,
    n_checks: int = 3,
    seed: int = 42,
) -> None
    # 검증 1 (Assert): n_checks 개 무작위 시점에서 target[t] == log_ret[t+1:t+22].sum()
    # 검증 2 (육안 표): 첫 5개 유효 행의 (날짜, log_ret, target, 직접계산, 일치 O/X) print
    # 실패 시 AssertionError
```

**의존성**: `numpy`, `pandas`

**사용 예**:

```python
from scripts.targets import build_daily_target_21d, verify_no_leakage

target = build_daily_target_21d(df['Adj Close'])
verify_no_leakage(df['log_return'].dropna(), target, n_checks=3, seed=42)
```

**⚠️ Windows 환경 주의**: `verify_no_leakage` 의 print 문에 em-dash(`—`) 가 포함되어 있어, Windows cmd/bash 에서 직접 실행 시 `cp949` 인코딩 에러 발생. 노트북(Jupyter)에서는 정상 동작. CLI 검증 시 `sys.stdout.reconfigure(encoding='utf-8')` 권장.

**이력 메모**: `build_leaky_target_for_test` 함수는 2026-04-25 에 제거되었다. 이유는 변경 이력 참고.

---

### 2.3 `dataset.py` — 텐서 데이터셋 · Walk-Forward 폴드

**책임**: log-return 시계열 → PyTorch 텐서 변환, Walk-Forward 폴드 인덱스 생성, 폴드별 스케일링·LSTMDataset 생성.

**공개 인터페이스**:

```python
class LSTMDataset(torch.utils.data.Dataset):
    """ (X, y) 텐서 보관. X: (N, seq_len, n_features), y: (N,) """
    def __init__(self, X: np.ndarray, y: np.ndarray): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]: ...

make_sequences(
    arr: np.ndarray,
    seq_len: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]
    # 슬라이딩 윈도우 (X, y) 생성. y[i] = arr[i+seq_len+horizon-1, 0]
    # N = T - seq_len - horizon + 1

walk_forward_folds(
    n: int,
    is_len: int,
    purge: int,
    emb: int,
    oos_len: int,
    step: int,
) -> List[Tuple[np.ndarray, np.ndarray]]
    # Rolling Walk-Forward: [IS][purge][emb][OOS] 구조의 (train_idx, test_idx) 목록
    # 설정 A 파라미터 (231/21/21/21/21) → 106 fold

build_fold_datasets(
    series: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seq_len: int,
    extra_features: Optional[np.ndarray] = None,
    target_series: Optional[np.ndarray] = None,
) -> Tuple[LSTMDataset, LSTMDataset, StandardScaler]
    # 폴드 하나의 train/test LSTMDataset 생성
    # StandardScaler 는 train_idx 로만 fit (누수 방지)
    # target_series 가 None 이면 scaled 주 피처를 타깃으로
    # target_series 가 주어지면 그 시계열을 y 로 사용 (스케일링 안 함)
```

**의존성**: `numpy`, `torch`, `sklearn.preprocessing.StandardScaler`

**사용 예 (설정 A)**:

```python
from scripts.dataset import walk_forward_folds, build_fold_datasets
from scripts.targets import build_daily_target_21d

target = build_daily_target_21d(df['Adj Close']).values
folds = walk_forward_folds(len(spy_lr), 231, 21, 21, 21, 21)
tr_idx, te_idx = folds[0]
train_ds, test_ds, scaler = build_fold_datasets(
    spy_lr, tr_idx, te_idx, seq_len=126, target_series=target,
)
```

---

### 2.4 `models.py` — LSTM 회귀 모델

**책임**: 시퀀스 입력 → scalar 회귀 모델 정의. Setting A (2-layer) / Setting B (1-layer) 공통 사용.

**공개 인터페이스**:

```python
class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_first: bool = True,    # ⭐ DataLoader (B,T,F) 호환 필수
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → out: (B,)  (마지막 시점 hidden → Linear → squeeze)

count_parameters(model: nn.Module) -> int
    # 학습 가능 파라미터 수 집계 (보조)
```

**함정 방어**:

- `num_layers=1` 시 PyTorch LSTM dropout 무시 → 별도 `nn.Dropout` 으로 head 앞에 적용 (학습자료_주의사항 §4.4)
- `batch_first=True` 기본값 → `output[:, -1, :]` 인덱스가 올바르게 동작 (학습자료_주의사항 §3.3, 02 §5 명시 요구)

**의존성**: `torch`, `torch.nn`

**사용 예**:

```python
from scripts.models import LSTMRegressor, count_parameters

# 설정 A
model = LSTMRegressor(input_size=1, hidden_size=128, num_layers=2,
                      dropout=0.2, batch_first=True)
print(f'파라미터 수: {count_parameters(model):,}')   # 199,297

# 설정 B (1-layer 시 head_dropout 자동 적용)
model_b = LSTMRegressor(1, 64, 1, dropout=0.3)
```

---

### 2.5 `train.py` — 학습 루프 · 체크포인트

**책임**: 단일 fold 학습 루프 (Huber loss, AdamW, ReduceLROnPlateau, EarlyStopping, gradient clipping) + 체크포인트 I/O + device 자동 감지.

**공개 인터페이스**:

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
    # Returns:
    #   'best_state_dict'  : Dict[str, Tensor]
    #   'history'          : {'train_loss': [], 'val_loss': [], 'lr': []}
    #   'best_epoch'       : int (1-indexed)
    #   'best_val_loss'    : float
    #   'stopped_early'    : bool

get_device(preference: str = 'auto') -> torch.device
    # 'auto' → cuda > mps > cpu
    # 명시값 → torch.device(preference)

save_checkpoint(state_dict: Dict[str, Tensor], path: str | Path) -> None
    # 부모 디렉토리 자동 생성 후 torch.save

load_checkpoint(path: str | Path, device: str | torch.device = 'cpu') -> Dict
    # torch.load(path, map_location=device)
```

**함정 방어** (학습자료_주의사항 §3.6):

- `model.train()` / `model.eval()` 명시 전환
- `optimizer.zero_grad()` 매 배치 호출
- val 단계 `with torch.no_grad():` 컨텍스트
- `loss.item()` 으로 그래프 detach
- `clip_grad_norm_(parameters, max_norm=grad_clip)`

**의존성**: `numpy`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`, `copy.deepcopy`

**사용 예**:

```python
from scripts.train import train_one_fold, save_checkpoint
from scripts.models import LSTMRegressor

model = LSTMRegressor(1, 128, 2, 0.2)
result = train_one_fold(
    model, train_loader, val_loader,
    max_epochs=100, early_stop_patience=10,
)
save_checkpoint(result['best_state_dict'], 'results/setting_A/SPY/fold_0.pt')
```

---

### 2.6 `metrics.py` — 평가 지표

**책임**: Hit Rate · R²_OOS (Campbell & Thompson 2008) · 표준 R² · MAE · RMSE + 3가지 baseline 비교 + fold 별 통계 요약.

**공개 인터페이스**:

```python
hit_rate(y_true, y_pred, exclude_zero: bool = True) -> float
    # 부호 적중률. exclude_zero=True 시 y_true 또는 y_pred 가 0 인 샘플 제외
    # Phase 1 관문: > 0.55

r2_oos(y_true, y_pred) -> float
    # 1 - sum((y - y_hat)**2) / sum(y**2)  (Campbell & Thompson 2008)
    # 0 예측 baseline 대비 개선이면 양수
    # Phase 1 관문: > 0

r2_standard(y_true, y_pred) -> float
    # 1 - SSE / SST_mean  (sklearn r2_score 동일)

mae(y_true, y_pred) -> float
rmse(y_true, y_pred) -> float

baseline_metrics(
    y_test: array-like,
    y_train: array-like,
) -> Dict[str, Dict[str, float]]
    # 3가지 baseline ('zero' / 'previous' / 'train_mean') × 5 메트릭 표
    # zero baseline 의 hit_rate 는 nan (방향 정보 없음)

summarize_folds(
    per_fold_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]
    # fold별 메트릭 list → {metric: {mean, std (ddof=1), min, max, n}}
    # NaN fold 값은 통계에서 제외, n 으로 유효 fold 수 표시
```

**의존성**: `numpy`

**해석 주의** (학습자료_주의사항 §6.5):

- 통계 유의 ≠ 경제 의미. 본 메트릭은 통계적 측정치만 반영, 거래비용·슬리피지 별도 평가 필요
- R²_OOS 음수 시 즉시 보고 → 모델·평가 코드 재검토

**사용 예**:

```python
from scripts.metrics import hit_rate, r2_oos, baseline_metrics, summarize_folds

# 단일 fold 평가
hr = hit_rate(y_test, y_pred)
r2 = r2_oos(y_test, y_pred)
bl = baseline_metrics(y_test, y_train)

# 106 fold 집계
fold_results = [{'hit_rate': hr_k, 'r2_oos': r2_k, ...} for k in range(106)]
summary = summarize_folds(fold_results)
print(f"Hit Rate: {summary['hit_rate']['mean']:.4f} ± {summary['hit_rate']['std']:.4f}")
```

---

## 3. 변경 이력

| 날짜 | 모듈 | 변경 사항 | 작성자 |
|---|---|---|---|
| 2026-04-24 | `setup.py` | 신규 생성 (00 노트북에서 함수·상수 분리) | 재천 |
| 2026-04-24 | `dataset.py` | 신규 생성 (LSTMDataset, make_sequences, walk_forward_folds, build_fold_datasets) | 다른 팀원 |
| 2026-04-24 | `dataset.py` | `build_fold_datasets` 에 `target_series` 인자 추가 (외부 타깃 주입, 하위 호환) | 다른 팀원 |
| 2026-04-24 | `targets.py` | 신규 생성 (`build_daily_target_21d`, `verify_no_leakage`, `build_leaky_target_for_test`) | 다른 팀원 |
| 2026-04-24 | `targets.py` | `build_monthly_target_1m` 추가 (설정 B 용) + 모듈 docstring 갱신 | 재천 |
| 2026-04-24 | `models.py` | 신규 생성 (`LSTMRegressor`, `count_parameters`) — batch_first=True 기본, num_layers=1 dropout 우회 | 재천 |
| 2026-04-24 | `train.py` | 신규 생성 (`train_one_fold`, `get_device`, `save_checkpoint`, `load_checkpoint`) | 재천 |
| 2026-04-24 | `metrics.py` | 신규 생성 (Hit Rate, R²_OOS, R² std, MAE, RMSE, baseline_metrics, summarize_folds) | 재천 |
| 2026-04-25 | `02_setting_A_daily21.ipynb` §4·§7~§10 | 활성화. **실무 관행 반영**: fold 체크포인트(.pt) 저장 제거 → fold 별 y_true/y_pred 배열을 `metrics.json` 에 포함 (seed 고정으로 재학습 복원 가능). GPU/MPS 자동 감지, CUDA 시 `torch.cuda.empty_cache()` 추가 | 재천 |
| 2026-04-25 | `targets.py` · 02 노트북 §4 · PLAN/정의서 | **인공 누수 대조 (identity leak sanity) 완전 제거**: `build_leaky_target_for_test` 함수 삭제 + 02 노트북 §4 검증 3 셀 삭제 + §1·§4·§10 "3종" → "2종". 이유: (a) scripts/*.py 단위 테스트로 파이프라인 정상성이 이미 확인됨, (b) dataset.py 의 train/test 정렬 차이로 identity leak 이 OOS 에서 의미 있게 작동하지 않음 | 재천 |
| 2026-04-25 | `scripts_정의서.md` | §6 "코드 블럭 상세 해설" 추가 — §2 API 레퍼런스 외에 코드 처음 읽는 협업자(사람·Claude) 를 위한 라인별 교육 자료 제공 | 재천 |

---

## 4. 검증 결과 요약 (2026-04-24, 재천)

| 모듈 | 검증 항목 | 결과 |
|---|---|---|
| `targets.py` | daily 유효 2493개 / monthly 119개 / 누수 검증 PASS / 검증 시점 02 노트북과 일치 | ✅ |
| `dataset.py` | 폴드 0 재현: train.X=(105,126,1), scaler.mean=0.00040364, train y mean=0.010401 (02 §6 와 일치) | ✅ |
| `models.py` | 4건 PASS — 설정 A shape (32,)·199k 파라미터, 설정 B Dropout(p=0.3)·warning 0건 | ✅ |
| `train.py` | 4건 PASS — 인공 누수 R²=0.9857, train_loss 감소 0.008→0.0009, EarlyStop 10/50 epoch | ✅ |
| `metrics.py` | 16건 PASS — hit_rate 4건, r2_oos 3건, baseline·summarize 9건 | ✅ |

---

## 5. 참고 문서

- 협업 진입점: [README.md](README.md)
- Phase 1 전체 plan: [PLAN.md](PLAN.md)
- 작업 일지: [재천_WORKLOG.md](재천_WORKLOG.md)
- 학습자료 주의사항: [학습자료_주의사항.md](학습자료_주의사항.md)
- 설정 A 실행 노트북 (TODO 활성화 대기): [02_setting_A_daily21.ipynb](02_setting_A_daily21.ipynb)

---

## 6. 코드 블럭 상세 해설

> **목적**: §2 모듈 상세는 API 레퍼런스로 빠른 참조용이고, 본 §6 은 코드를 처음 읽거나 깊이 이해하려는 협업자(사람·Claude 모두)를 위한 라인별 교육 자료다. 함정 방어와 invariant 의도까지 풀어 설명한다.
>
> **읽는 법**: 각 블럭은 (a) 코드 인용 → (b) 라인별 해부 → (c) 함정/invariant 의의 → (d) 만약 다르게 짰다면 어떻게 깨지나 형식. 노트북 작업 시 §2 만 보고 바로 사용해도 되고, 새 협업자가 코드 베이스를 처음 익힐 때는 §6 을 정독하는 것을 권장.

---

### 6.1 `setup.py` — 환경 부트스트랩

이 모듈은 Phase 1 의 모든 노트북·스크립트가 공통으로 사용하는 환경 설정을 정의한다. **단일 진실원(single source of truth)** 패턴 — 한 곳에서만 시드/경로/폰트를 정의하고 모든 곳이 import.

#### 6.1.1 경로 상수 블록 ([setup.py:36-50](scripts/setup.py))

```python
BASE_DIR: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = BASE_DIR / 'results'
RAW_DATA_DIR: Path = RESULTS_DIR / 'raw_data'
SETTING_A_DIR: Path = RESULTS_DIR / 'setting_A'
SETTING_B_DIR: Path = RESULTS_DIR / 'setting_B'

def ensure_result_dirs() -> None:
    for d in (RAW_DATA_DIR, SETTING_A_DIR, SETTING_B_DIR):
        d.mkdir(parents=True, exist_ok=True)
```

**라인별 해부**:
- `Path(__file__).resolve().parent.parent` — `setup.py` 의 부모(`scripts/`) 의 부모(`Phase1_LSTM/`) 가 `BASE_DIR`. `__file__` 기반이므로 노트북이 어디서 import 하든 동일한 경로를 가리킨다.
- 모든 경로를 `pathlib.Path` 로 통일 — 문자열 + 연산 (`/`) 이 OS 무관하게 동작.
- `ensure_result_dirs()` 의 `parents=True, exist_ok=True` — 디렉토리 트리를 한 번에 생성하면서 이미 존재해도 에러 안 남 (idempotent).

**왜 이렇게 짰나**:
- `os.getcwd()` 를 쓰면 노트북이 어디서 실행됐는지에 따라 경로가 달라짐 → 협업 시 깨짐.
- `__file__.parent.parent` 는 **파일 위치 기준** 이라 호출 컨텍스트 무관.
- 시작 시 디렉토리를 한 번 생성해두면 이후 `np.save(...)`, `model.save(...)` 등이 `FileNotFoundError` 로 죽을 일 없음.

#### 6.1.2 시드 고정 블록 ([setup.py:53-95](scripts/setup.py))

```python
SEED: int = 42

def fix_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f'[setup] 결정적 알고리즘 활성화 실패: {e}')
    except ImportError:
        pass
```

**라인별 해부**:
- `random.seed(seed)` — Python 표준 `random` 모듈. shuffle, choice 등에 영향.
- `np.random.seed(seed)` — NumPy 의 전역 RNG. 단, 이건 함수 안에서만 호출되므로 모듈 import 자체는 깨지지 않음.
- `torch.manual_seed(seed)` — CPU 텐서 생성·dropout·shuffle.
- `torch.cuda.manual_seed_all(seed)` — 모든 GPU 디바이스에 동일 시드 적용.
- `os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'` — CUDA 결정성 요구사항. CUBLAS 의 워크스페이스 메모리를 분리해서 비결정적 연산 차단.
- `torch.use_deterministic_algorithms(True, warn_only=True)` — 비결정 연산 만나면 에러 대신 **경고만**. 학습이 끊기지 않게.

**ImportError 처리**:
- `try ... except ImportError: pass` — 해당 라이브러리가 없는 환경에서도 모듈 import 자체는 성공.
- numpy 만 있고 torch 가 없는 환경에서도 numpy 시드는 적용됨 → 부분 동작.

**왜 완벽한 결정성은 안 되는가**:
- GPU CUDA 의 일부 연산(예: cudnn convolution) 은 알고리즘 자체가 비결정적.
- 멀티스레드 BLAS 의 reduction 순서.
- OS·Python 버전 차이.
- → 비트 단위 재현은 어렵다. 메트릭은 소수점 4자리까지만 비교한다는 운영 정책.

#### 6.1.3 한글 폰트 블록 ([setup.py:101-137](scripts/setup.py))

```python
def setup_korean_font() -> str:
    import matplotlib.pyplot as plt
    os_name = platform.system()
    if os_name == 'Windows':
        font_name = 'Malgun Gothic'
        plt.rcParams['font.family'] = font_name
    elif os_name == 'Darwin':
        font_name = 'AppleGothic'
        plt.rcParams['font.family'] = font_name
    else:
        try:
            import koreanize_matplotlib  # noqa: F401
            font_name = 'NanumGothic'
        except ImportError:
            print('[setup] 경고: koreanize_matplotlib이 설치되어 있지 않습니다.')
            print('       pip install koreanize-matplotlib --break-system-packages')
            font_name = plt.rcParams['font.family']
    plt.rcParams['axes.unicode_minus'] = False
    return font_name
```

**라인별 해부**:
- `platform.system()` — OS 식별. `'Windows'`, `'Darwin'`(macOS), `'Linux'`.
- 각 OS 의 표준 한글 폰트 적용:
  - Windows → `Malgun Gothic` (맑은 고딕, OS 기본 포함)
  - macOS → `AppleGothic` (OS 기본 포함)
  - Linux → `koreanize_matplotlib` 패키지 (NanumGothic 자동 등록)
- `koreanize_matplotlib` 은 `# noqa: F401` 으로 "이 import 는 side effect 가 목적" 임을 린터에게 알림 (실제 사용은 안 함).

**`axes.unicode_minus = False` 의 의의**:
- matplotlib 기본은 마이너스 기호를 유니코드 `−` (U+2212) 로 표기.
- 한글 폰트가 이 문자를 지원 안 하면 □ 박스로 깨짐.
- `False` 로 설정하면 ASCII `-` (U+002D) 사용 → 어떤 폰트에서도 안 깨짐.
- **OS 무관 차단** — 이 한 줄이 모든 OS 에서 마이너스 깨짐을 막음.

**왜 OS 분기인가**:
- Linux 는 한글 폰트가 OS 기본에 없는 경우가 많아 외부 패키지 필요.
- Windows/macOS 는 OS 가 한글 폰트를 기본 제공.
- 사용자가 `pip install koreanize-matplotlib` 을 안 했어도 경고만 출력하고 진행 — 학습이 멈추지 않음.

#### 6.1.4 표시 옵션 + 원샷 부트스트랩 ([setup.py:143-200](scripts/setup.py))

```python
def apply_display_defaults() -> None:
    try:
        import pandas as pd
        pd.set_option('display.float_format', lambda x: f'{x:.6f}')
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_columns', 30)
    except ImportError:
        pass
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (10, 4)
        plt.rcParams['figure.dpi'] = 100
    except ImportError:
        pass


def bootstrap(seed: Optional[int] = None, verbose: bool = True) -> str:
    if seed is None:
        seed = SEED
    font_used = setup_korean_font()
    fix_seed(seed)
    ensure_result_dirs()
    apply_display_defaults()
    if verbose:
        print('=' * 60)
        print('  Phase 1 — 환경 부트스트랩 완료')
        print('=' * 60)
        print(f'  한글 폰트  : {font_used}')
        print(f'  시드       : {seed}')
        print(f'  결과 경로  : {RESULTS_DIR}')
        print('=' * 60)
    return font_used
```

**`apply_display_defaults` 의 선택**:
- `'%.6f'` — 일별 수익률은 0.0001 단위까지 의미가 있어 6자리.
- `display.max_rows=50` — 너무 많이 출력하면 노트북 셀이 길어짐. 50 이 가독성 균형.
- `figsize=(10, 4)` — 시계열 가로형 그래프에 적합.

**`bootstrap()` — 이 모듈의 표면(API)**:
- 사용자는 `bootstrap()` 만 알면 됨. 나머지 4개 함수는 내부 구현.
- 한 줄로 폰트·시드·디렉토리·표시옵션을 모두 적용.
- `verbose` 출력은 환경 정보를 한눈에 확인할 수 있는 박스 — 노트북 첫 셀 결과로 적합.

**왜 이렇게 모았나**:
- 모든 노트북이 첫 셀에서 동일한 환경 보장 → 결과 재현성·비교 가능성.
- 신규 협업자가 "환경 어떻게 세팅하지?" 고민할 필요 없이 `bootstrap()` 한 줄.
- 만약 환경 정책이 바뀌면(예: SEED 를 0 으로) **이 파일 한 곳만** 수정.

---

### 6.2 `targets.py` — 타깃 시계열 + 누수 검증

이 모듈은 **예측 타깃을 만드는 책임** 만 진다. 입력 피처 가공·스케일링은 `dataset.py`, 모델은 `models.py`. 의도적으로 좁게 설계된 모듈.

#### 6.2.1 `build_daily_target_21d` — Setting A 타깃 ([targets.py:15-46](scripts/targets.py))

```python
def build_daily_target_21d(adj_close: pd.Series) -> pd.Series:
    log_ret = np.log(adj_close).diff()                      # 누수: trailing diff
    target = log_ret.rolling(21).sum().shift(-21)            # 누수: forward 21일 합 (예측 목표)
    return target
```

**라인별 해부**:

라인 1 — `np.log(adj_close).diff()`:
- `np.log(adj_close)` : 가격 → 로그가격.
- `.diff()` : `x[t] - x[t-1]` = `log(P_t / P_{t-1})` ≈ 그 날의 수익률.
- 결과: 첫 행 NaN (이전 값 없음), 나머지는 일별 로그수익률.

**왜 단순수익률(`pct_change`)이 아니라 로그수익률인가**:
1. **가법성**: `log_ret_total = log_ret_1 + log_ret_2 + ...` — 다음 라인 `.sum()` 의 정당성.
2. **정규성**: 작은 수익률 영역에서 정규분포에 가까움 → MSE/Huber loss 와 잘 맞음.
3. **대칭성**: +10%/-10% 가 절댓값으로 같음 (단순수익률은 비대칭).

라인 2 — `log_ret.rolling(21).sum().shift(-21)`:

세 단계로 분해:
```
log_ret:                [r1,  r2,  r3,  r4, ..., r21, r22, ..., rN]

rolling(21).sum():      [NaN, NaN, ..., NaN, sum(r1..r21), sum(r2..r22), ...]
                         ↑ 앞 20개 NaN — 21개 모이기 전까지

.shift(-21):            [sum(r22..r42), sum(r23..r43), ..., NaN, NaN, ...]
                         ↑ 위치 t의 값이 t+1 ~ t+21 합으로 바뀜
```

- `rolling(21).sum()` 은 **trailing**(과거를 뒤돌아보는) 21일 합.
- `.shift(-21)` 은 시리즈 전체를 21칸 앞으로 당김 — 위치 t 의 값이 원래 위치 t+21 의 값.
- 두 개 합치면: 위치 t 의 값 = `r[t+1] + r[t+2] + ... + r[t+21]` = **앞으로 21영업일 누적 로그수익률**.
- 결과 NaN: 앞 1개(diff) + 뒤 21개(forward 부족) = 총 22개.

**"누수가 아닌 이유"**:
- 이 함수는 정의상 **미래 값을 가져온다**. 그런데 왜 누수가 아닌가?
- **타깃(y)** 이 미래 값인 건 정상 — 우리가 예측하려는 대상.
- **누수**는 *입력(X)* 에 미래 정보가 섞일 때 발생.
- 모델 학습 시 `X[t]` 가 t 이전 데이터로만 만들어진다면 OK.
- 운영적 보장은 `dataset.py` 의 `walk_forward_folds` (purge + embargo).

**21이라는 숫자**:
- 미국 주식 약 1개월 거래일.
- Phase 1 의 forward horizon. 너무 짧으면 노이즈 지배, 너무 길면 샘플 부족 + 누적 분산 폭주.
- 변경 시 새 함수 (`build_daily_target_5d` 등) 를 만드는 게 안전.

#### 6.2.2 `build_monthly_target_1m` — Setting B 타깃 ([targets.py:49-88](scripts/targets.py))

```python
def build_monthly_target_1m(adj_close: pd.Series) -> pd.Series:
    monthly_close = adj_close.resample('ME').last()
    log_ret_monthly = np.log(monthly_close).diff()
    target = log_ret_monthly.shift(-1)
    return target
```

**Setting A 와의 차이**:

| 항목 | Setting A (일별) | Setting B (월별) |
|---|---|---|
| 입력 빈도 | daily | monthly |
| 타깃 | 21일 **누적** log return | 다음 1달 **단일** log return |
| 다운샘플 | 없음 | `resample('ME').last()` |
| Purge/Embargo 단위 | 일 | 월 (1개월) |
| NaN 위치 | 첫 1 + 끝 21 | 첫 1 + 끝 1 |

**라인별 해부**:

`adj_close.resample('ME').last()`:
- `'ME'` = "Month End" — 각 월의 마지막 거래일에 해당하는 인덱스로 다운샘플.
- pandas 2.0 이전엔 `'M'` 이었으나 2.0+ 에서 deprecated → `'ME'` 사용 (FutureWarning 회피).
- `.last()` : 그 월에 속한 가격들 중 **마지막** 값을 채택.

**왜 last 인가** (mean/median 이 아닌):
- `mean`/`median` 은 그 자체가 미래값을 평균에 포함 (look-ahead bias).
- `last` 는 한 달이 끝난 후에만 알 수 있는 값이지만, 그 시점이 "그 달의 의사결정 가능 시점" 이므로 OK.

**왜 first 가 아닌가**:
- 월초 가격은 직전 월 정보로 다 결정 → 한 달 동안의 가격 변화를 반영 못 함.

`log_ret_monthly.shift(-1)`:
- 위치 m 의 값을 m+1 의 값으로 채움.
- 위치 m(월말) 의 타깃 = **다음 달의 log return**.
- 일별 21일과 달리 누적 합이 아닌 단일 기간 → 분산이 작고 모델이 학습하기 더 쉬울 가능성 (Setting A 실패 후의 백업 카드).

#### 6.2.3 `verify_no_leakage` — 단위 테스트 단계 ([targets.py:91-153](scripts/targets.py))

```python
def verify_no_leakage(log_ret, target, n_checks=3, seed=42) -> None:
    rng = np.random.default_rng(seed)
    valid_pos = [
        i for i in range(len(target))
        if (not np.isnan(target.iloc[i])) and (i + 21 < len(log_ret))
    ]
    if len(valid_pos) < n_checks:
        raise ValueError(...)
    chosen = sorted(rng.choice(valid_pos, size=n_checks, replace=False))

    for pos in chosen:
        t = target.index[pos]
        expected = log_ret.iloc[pos + 1 : pos + 22].sum()
        actual = float(target.iloc[pos])
        diff = abs(actual - expected)
        status = "PASS" if diff < 1e-10 else "FAIL"
        print(f"  [{status}] {str(t.date())}  target={actual:.6f}  직접계산={expected:.6f}  Δ={diff:.2e}")
        assert diff < 1e-10, ...
```

**리턴 None 의 의의**:
- 이 함수는 부수효과(side effect)만 일으킴: 정상 → assert 통과 + print, 비정상 → 예외.
- bool 을 리턴하면 `if verify_no_leakage(...): pass` 같은 무시 코드가 가능 → 강제력 약화.
- None 리턴 + 예외 패턴은 **검증 누락이 불가능** 함을 강제.

**`np.random.default_rng(seed)` 선택**:
- 모듈 레벨 `np.random.seed` 는 전역 RNG 변경 → 다른 코드의 재현성에 영향.
- `default_rng` 는 격리된 RNG 인스턴스 → 이 함수 내부에서만 영향.
- numpy 1.17+ 권장 패턴.

**`valid_pos` 의 두 조건**:
1. `not np.isnan(target.iloc[i])` — 타깃이 NaN 인 위치(첫 1 + 끝 21)는 검증 불가.
2. `i + 21 < len(log_ret)` — `log_ret[i+1:i+22]` 슬라이스를 안전하게 만들 수 있는 위치만.
- 두 조건은 사실상 동치이지만 **방어적 중복** — 타깃 NaN 처리에 버그가 있을 가능성 차단.

**`< n_checks` 가드**:
- 데이터 너무 짧아 유효 위치 < n_checks → silent 통과하면 검증 무의미.
- ValueError 로 **요란하게 실패** — 사용자가 "유효 인덱스 부족" 을 인지.

**`pos + 1 : pos + 22` 범위 (오프바이원 함정)**:
- pandas slice 는 stop **exclusive** → `[pos+1 : pos+22]` 는 `pos+1, pos+2, ..., pos+21` 21개 포함.
- 만약 `[pos : pos+21]` 로 잘못 쓰면 t 시점 자기 자신을 포함 → 결과는 21영업일 *과거 포함* 합 → PASS 가능 (인덱싱 실수가 가려짐).
- 이 줄이 정확히 t **이후** 21개를 가져오는지가 핵심 — 코드 리뷰 1순위 확인 포인트.

**`1e-10` 허용 오차**:
- numpy float64 의 21개 가산 결과는 약 1e-15 단위 잡음.
- `1e-10` 은 그보다 5자리 큰 여유 → 실제 누수와 부동소수점 잡음을 명확히 구분.

**`print` + `assert` 동시 사용**:
- `print` 만 있으면 사용자가 메시지 무시 가능.
- `assert` 만 있으면 PASS 시 침묵 → 검증을 했는지조차 알 수 없음.
- 둘 다: **PASS 도 보이고, FAIL 이면 멈춤** (관찰가능성 + 강제성).

#### 6.2.4 `verify_no_leakage` — 육안 표 단계 ([targets.py:156-171](scripts/targets.py))

```python
print("=== 누수 검증 2 — 육안 확인 표 (첫 5개 유효 행) ===")
first5 = valid_pos[:5]
print(f"  {'날짜':>12}  {'log_ret':>10}  {'target':>10}  {'직접계산':>10}  {'일치':>4}")
print("  " + "-" * 54)
for pos in first5:
    t = target.index[pos]
    lr = float(log_ret.iloc[pos])
    tgt = float(target.iloc[pos])
    direct = log_ret.iloc[pos + 1 : pos + 22].sum()
    match = "O" if abs(tgt - direct) < 1e-10 else "X"
    print(f"  {str(t.date()):>12}  {lr:>10.6f}  {tgt:>10.6f}  {direct:>10.6f}  {match:>4}")
print()
print("[OK] 누수 검증 완료 — 모든 체크포인트 PASS")
```

**왜 assert 가 있는데 표가 또 필요한가**:
- assert 는 "0/1 통과" 정보만 — 통과해도 **수치 자체가 합리적인지** 확인 불가.
- 표를 보면 검증할 수 있는 것:
  - log_ret 단위가 `0.001` 근처(정상 일별 수익률 스케일) 인지 → 100 단위면 % 변환 실수.
  - target 단위가 `0.02` 근처(21일 누적, √21 ~4.6배 분산) 인지 → 너무 크면 단위 오류.
  - 첫 유효 위치의 날짜가 데이터 시작 직후 인지 → 너무 늦으면 NaN 처리 과잉.
- assert 는 **로직 검증**, 표는 **스케일/분포 검증**. 두 차원이 서로 다름.

**`valid_pos[:5]` (앞에서 5개) vs 무작위**:
- assert 단계는 무작위 3개 → 골고루 분포.
- 표 단계는 첫 5개 → **NaN 경계** 직후 값들. 인덱싱 실수가 가장 많이 나는 곳이라 합리적.

**ASCII "O"/"X" 선택**:
- 유니코드 체크 마크는 폰트 따라 □ 박스.
- ASCII 는 어떤 환경에서도 안전.

#### 6.2.5 시그니처 종속성 한계

`verify_no_leakage` 의 시그니처와 내부에 박힌 가정:
- `if i + 21 < len(log_ret)` — **horizon=21 하드코딩**.
- `log_ret.iloc[pos + 1 : pos + 22].sum()` — **누적합 가정**.
- 입력은 `log_ret` (일별), `target` 은 21일 누적.

**즉 이 함수는 사실상 `verify_no_leakage_daily_21d` 다.**

Setting B (월별 1개월) 의 타깃에 그대로 호출하면:
- `log_ret[pos+1:pos+22].sum()` 은 **다음 21개월 합** 을 계산 (월별 시리즈에서).
- 실제 타깃은 **다음 1개월 값** 한 개.
- 두 값이 우연히 일치할 확률 0 → AssertionError 로 다행히 silent fail 은 아님.
- 단, 에러 메시지가 "누수 의심" 으로 떠서 사용자가 **누수가 있다고 오진** 할 위험 — 실제론 검증기 자체가 안 맞는 케이스.

**향후 확장 방향**:
- Setting B 진행 시 **페어 검증 함수** (`verify_monthly_target_1m`) 를 추가하거나, callable 주입형으로 일반화.
- 단기 프로젝트에선 호출 코멘트로 "이 함수는 daily 21d 전용" 명시만으로 운영 가능.

---

### 6.3 `dataset.py` — Walk-Forward + 누수 차단 핵심

이 모듈은 데이터를 LSTM 텐서 쌍으로 가공하는 동시에 **누수 방지의 핵심 invariant** 를 코드 안에 봉인한다. `scripts/` 중에서 가장 민감한 파일.

#### 6.3.1 `LSTMDataset` 클래스 ([dataset.py:22-39](scripts/dataset.py))

```python
class LSTMDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
```

**라인별 해부**:
- `class LSTMDataset(Dataset)` — PyTorch `Dataset` 상속. `__len__`/`__getitem__` 만 구현하면 자동으로 `DataLoader` 와 호환.
- `__init__` — numpy → float32 텐서로 한 번만 변환. dtype 명시로 모델 weight 와 dtype 일치 (LSTM 학습 표준).
- `__len__`, `__getitem__` — DataLoader 가 호출하는 표준 메소드.

**시그니처가 강제하는 책임 분리**:
```python
def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
```
- 인자가 **두 개뿐** — `X`, `y`. 사용자가 "test 도 같이 넣어주세요" 라며 train/test 를 한 객체에 섞는 게 시그니처상 불가능.
- "메타데이터(timestamps) 도 함께 넣자" 같은 변형도 차단 — 새 인자 없으니까.
- 즉 **이 클래스는 "이미 분할되고 스케일링된 (X, y) 만 받는다"** 는 책임 분리를 시그니처로 강제.

**만약 자유롭게 풀면**:
```python
class BadDataset:
    def __init__(self, full_data):
        self.X_train, self.X_val = split(full_data)        # 안에서 자체 분할
        self.scaler = StandardScaler().fit(full_data)       # 전체기간 fit (누수)
```
- 이런 클래스를 만들 수 있는 자유가 누수를 부름.
- LSTMDataset 은 **누수 발생 지점이 이 클래스 밖** 에 있음을 시그니처로 명시.

#### 6.3.2 `make_sequences` — 슬라이딩 윈도우 ([dataset.py:42-80](scripts/dataset.py))

```python
def make_sequences(arr, seq_len, horizon=1):
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    T, n_feat = arr.shape
    N = T - seq_len - horizon + 1
    if N <= 0:
        raise ValueError(
            f"시계열 길이 T={T}이 seq_len({seq_len}) + horizon({horizon})보다 짧습니다."
        )
    X = np.stack([arr[i : i + seq_len] for i in range(N)])
    y = arr[seq_len + horizon - 1 : seq_len + horizon - 1 + N, 0]
    return X, y
```

**라인별 해부**:

`if arr.ndim == 1: arr = arr[:, np.newaxis]`:
- 1D 입력을 2D `(T, 1)` 로 승격 → 이후 코드는 항상 2D 가정 → 분기 한 번만, 본 로직은 단일 경로.

`T, n_feat = arr.shape`:
- 시계열 길이 + 피처 수 분리.

`N = T - seq_len - horizon + 1`:
- 시퀀스 개수.
- 직관: 첫 시퀀스 끝 = `seq_len-1`, 타깃 = 그 위치 + horizon. 마지막 시퀀스 끝 = `T - horizon - 1`. 시작 위치 후보 = `0 ~ T - seq_len - horizon` → +1 해서 N 개.

`if N <= 0: raise ValueError(...)` — **요란한 실패 가드**:
- 시계열이 너무 짧아 시퀀스 단 하나도 안 나오면 즉시 예외.
- 가드 없으면? `np.stack([])` 가 빈 배열, 다음 단계 `DataLoader(empty)` → 학습 epoch 마다 0 배치 → silent failure (모델이 안 학습됨).

`X = np.stack([arr[i : i + seq_len] for i in range(N)])`:
- N 개 시퀀스 stack. 각 시퀀스 = `arr[i:i+seq_len]`.
- 결과 shape: `(N, seq_len, n_feat)` — LSTM 입력 표준 (`batch_first=True`).

`y = arr[seq_len + horizon - 1 : seq_len + horizon - 1 + N, 0]`:
- 첫 시퀀스(i=0) 끝 = `seq_len-1` → 타깃 위치 = `seq_len-1 + horizon`.
- 마지막 시퀀스(i=N-1) 타깃 위치 = `seq_len-1 + horizon + N-1`.
- 슬라이스 길이 = N (정확히).
- `[:, 0]` — **첫 번째 피처만** 타깃으로 (다변량이라도 첫 피처 기준).

#### 6.3.3 `walk_forward_folds` — Walk-Forward CV ([dataset.py:83-129](scripts/dataset.py))

```python
def walk_forward_folds(n, is_len, purge, emb, oos_len, step):
    folds = []
    start = 0
    while True:
        train_end = start + is_len
        test_start = train_end + purge + emb
        test_end = test_start + oos_len
        if test_end > n:
            break
        folds.append((
            np.arange(start, train_end),
            np.arange(test_start, test_end),
        ))
        start += step
    return folds
```

**시각화**:
```
시계열 인덱스:  0 ────── 59 60-61 62-64 65 ─── 74    75 ─── 84 ...
fold 0:        [══train══][purge][emb ][═test═]
                start=0   60-61  62-64  65-74
fold 1:                  [══train══][purge][emb][═test═]
                          start=10  70-71  72-74  75-84
fold 2:        ... step=10 만큼 슬라이딩 ...
test_end > n 이면 종료
```

**라인별 해부**:

`while True ... break`:
- 무한 루프 + 조건부 break — `test_end` 가 매 반복마다 `step` 만큼 증가, 결국 n 초과.
- "최대한 많은 fold 를 만든다" 는 의도 명시.

`train_end = start + is_len` / `test_start = train_end + purge + emb`:
- train 과 test 사이에 **purge + emb 공백 강제**.
- **purge**: train 마지막 라벨이 미래 21일 포함 → 그 21일이 test 와 겹치지 않게 잘라냄.
- **embargo**: 추가 안전 구간. 자기상관 잔여 효과 차단.
- 두 인자를 **양수 정수로 받는다는 사실 자체** 가 누수 방지를 강제.

`np.arange(start, train_end)` / `np.arange(test_start, test_end)`:
- train/test 인덱스를 numpy 배열로 반환.
- 인덱스만 반환 — **이 함수는 데이터를 만지지 않음**.
- 호출자가 데이터를 슬라이싱하는 책임 → 책임 분리.

**시그니처가 누수 방지를 강제**:
```python
def walk_forward_folds(n, is_len, purge, emb, oos_len, step):
```
- `purge`, `emb` 가 **별도 인자**.
- 사용자가 "purge 없이 train→test 붙이고 싶어요" 하려면 `purge=0, emb=0` 을 명시적으로 넣어야 함 → 의식적 위험 감수가 코드에 표시 → 코드 리뷰에서 발견.
- 만약 이 두 인자가 없었다면, 누수 위험 구간이 사용자 머릿속에만 존재 → 잊혀짐 → 누수.

**알려진 버그**:
- `series.dropna()` 길이 미적용 — 만약 `n` 이 NaN 포함 길이로 들어오면 fold 가 NaN 구간을 train/test 에 포함 가능.
- Phase 1 의 fold 분산 폭주 (R² std 0.97~1.73) 의 원인 후보.

#### 6.3.4 `build_fold_datasets` — 시그니처와 입력 결합 ([dataset.py:132-192](scripts/dataset.py))

```python
def build_fold_datasets(
    series: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seq_len: int,
    extra_features: Optional[np.ndarray] = None,
    target_series: Optional[np.ndarray] = None,
) -> Tuple[LSTMDataset, LSTMDataset, StandardScaler]:
    data = series[:, np.newaxis]
    if extra_features is not None:
        data = np.column_stack([series, extra_features])
    ...
```

**시그니처가 강제하는 암묵적 계약**:
1. **`train_idx`/`test_idx` 가 분리된 인자** — 한 통에 섞어 넣을 수 없음.
2. **`extra_features` 와 `series` 길이 일치 강제** — `np.column_stack` 이 길이 불일치 시 즉시 ValueError.
3. **`target_series` 가 Optional** — 기본값 None 일 때는 자체 타깃, 명시적으로 외부 타깃 모드 분리.
4. **반환 `(LSTMDataset, LSTMDataset, StandardScaler)`** — scaler 까지 반환 → 외부에서 inverse_transform 등 재사용 가능.

`data = series[:, np.newaxis]`:
- 주 시계열을 `(T, 1)` 로 승격 → 다변량 처리 통일.
- `extra_features` 있으면 `np.column_stack([series, extra_features])` → `(T, 1+k)`.
- 순서 약속: series 가 0번 열, extra 가 그 뒤. 이후 `[:, 0]` 인덱싱이 series 를 가리킴.

#### 6.3.5 `build_fold_datasets` — 누수 방지 스케일링 (모듈의 핵심 3줄)

```python
scaler = StandardScaler()
scaler.fit(data[train_idx])          # 훈련 구간으로만 fit — 누수 방지
scaled = scaler.transform(data)      # 전체 시계열 transform (테스트 맥락 포함)
```

**이 세 줄이 이 모듈 전체의 누수 방지 핵심.** "함수 안에 invariant 가 박혀 있다" 의 가장 명료한 예.

**라인별 해부**:

`scaler = StandardScaler()`:
- 새 scaler 인스턴스. 이전 fold 의 scaler 를 재사용하지 않음 → fold 독립성 보장.

`scaler.fit(data[train_idx])` ★:
- `data[train_idx]` 만 사용 — train 인덱스로 슬라이싱한 부분만.
- StandardScaler 가 mean/std 를 **이 부분에서만** 계산.
- test 구간 정보는 이 시점에 absolutely 0%.

`scaled = scaler.transform(data)`:
- 전체 시계열에 transform.
- 단, mean/std 는 train 구간 것 → test 가 train 의 분포에 매핑됨 (실제 운영 환경과 동일).

**만약 누가 이 invariant 를 어기려 한다면**:
```python
# 노트북에서 자체 사전 스케일링
scaler = StandardScaler()
scaler.fit(full_data)              # 전체기간 fit (누수)
scaled_data = scaler.transform(full_data)
ds_train, ds_test, _ = build_fold_datasets(scaled_data, ...)
```
- 합법적 코드지만 **누수 발생**.
- `build_fold_datasets` 가 raw 시계열을 가정한다는 점에 의존 — 호출자 신뢰 모델.

**더 안전하게 하려면**:
- 함수 진입 시 `series` 가 이미 스케일된 상태인지 검사 (mean ≈ 0, std ≈ 1 휴리스틱).
- 현재는 협업 단순성을 위해 미적용.

**"함수 안 invariant" 의 의미**:
- 이 세 줄을 매 노트북에서 작성한다면:
  ```python
  # notebook 1
  scaler = StandardScaler()
  scaler.fit(data[train_idx])
  scaled = scaler.transform(data)

  # notebook 2 — 누가 실수
  scaler = StandardScaler()
  scaler.fit(data)                  # train_idx 잊음 → 누수
  scaled = scaler.transform(data)
  ```
- 함수가 이 세 줄을 한 번에 캡슐화 → 사용자는 `build_fold_datasets(...)` 만 호출 → 한 줄 빠뜨릴 가능성 0.

#### 6.3.6 `build_fold_datasets` — 훈련/테스트 시퀀스 ([dataset.py:198-224](scripts/dataset.py))

```python
X_tr, y_tr_default = make_sequences(scaled[train_idx], seq_len)

if target_series is not None:
    n_train = len(X_tr)
    y_tr = np.array([
        float(target_series[train_idx[j + seq_len - 1]])
        for j in range(n_train)
    ])
else:
    y_tr = y_tr_default

X_te_list, y_te_list = [], []
for k, t in enumerate(test_idx):
    if t < seq_len:
        continue  # 충분한 이력 없음 (초반 폴드 방어)
    X_te_list.append(scaled[t - seq_len : t])
    if target_series is not None:
        y_te_list.append(float(target_series[t]))
    else:
        y_te_list.append(float(scaled[t, 0]))

X_te = np.stack(X_te_list)
y_te = np.array(y_te_list)

return LSTMDataset(X_tr, y_tr), LSTMDataset(X_te, y_te), scaler
```

**훈련 시퀀스 — 외부 타깃 매핑의 미세 함정**:

`make_sequences(scaled[train_idx], seq_len)`:
- train 구간만 추출해서 시퀀스화.
- 결과: `X_tr.shape = (is_len - seq_len, seq_len, n_feat)`, `y_tr_default.shape = (is_len - seq_len,)`.

`y_tr = np.array([float(target_series[train_idx[j + seq_len - 1]]) for j in range(n_train)])`:
- j 번째 시퀀스가 끝나는 위치 = `seq_len - 1` 부터 시작 → train 인덱스로는 `train_idx[j + seq_len - 1]`.
- 그 위치의 외부 타깃 값 추출.
- **각 시퀀스의 마지막 시점에 매핑된 21일 누적수익률을 y로**.

**오프바이원 함정**:
- `train_idx[j + seq_len - 1]` (현재 코드): 시퀀스 마지막 시점 → 정의에 충실.
- 만약 `train_idx[j + seq_len]` 으로 잘못 썼다면: 시퀀스 마지막 다음 날 → 1일 미래정보 누수.
- 이 미세한 차이가 누수 vs 정상의 갈림길.

**테스트 시퀀스 — 슬라이스 끝 exclusive**:

`scaled[t - seq_len : t]`:
- 위치 t 의 입력은 **t 직전 seq_len 개**, t 자신은 포함 안 함 (Python slice stop exclusive).
- 누수 방지의 운영적 보장: 입력은 t 이전, 타깃은 t (또는 t 의 미래).

`if target_series is not None: y_te_list.append(float(target_series[t]))` else `float(scaled[t, 0])`:
- 둘 다 위치 **t** — 입력 슬라이스 끝(exclusive) 의 위치.
- 입력은 t 미만, 타깃은 t → 시간 화살표가 항상 **입력 → 타깃** 방향.

**`if t < seq_len: continue` 가드**:
- 만약 `t < seq_len` 이면 `scaled[t - seq_len : t]` 가 음수 시작 → 슬라이스가 비거나 의도와 다름.
- 초반 폴드(start=0 근처) 에서 발생 가능 → 안전 스킵.

**왜 이 28줄을 노트북에서 직접 작성하면 안 되는가**:
- `scaled[t - seq_len : t + 1]` (한 글자 실수) → t 자신을 입력에 포함 → 누수.
- `t < seq_len` 가드 누락 → IndexError 또는 빈 시퀀스.
- target_series 없을 때 폴백 누락 → KeyError.
- scaler 따로 관리 → fold 독립성 깨짐.
- → **모든 함정이 함수 안에 봉인** 되어 사용자는 안전한 인터페이스만 호출.

---

### 6.4 `models.py` — LSTM 함정 방어

이 모듈은 LSTM 회귀 아키텍처를 정의하면서 PyTorch LSTM 의 **4가지 미묘한 함정** 을 클래스 안에 봉인한다 (docstring 의 설계 결정 §1~§4).

#### 6.4.1 `__init__` — dropout 자동 라우팅 ([models.py:87-120](scripts/models.py))

```python
def __init__(
    self,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_first: bool = True,
) -> None:
    super().__init__()
    self.input_size = input_size
    ...
    lstm_dropout = dropout if num_layers > 1 else 0.0
    self.lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=lstm_dropout,
        batch_first=batch_first,
    )
    self.head_dropout = (
        nn.Dropout(dropout) if num_layers == 1 else nn.Identity()
    )
    self.head = nn.Linear(hidden_size, 1)
    self._init_forget_gate_bias(value=1.0)
```

**시그니처**:
- `input_size` 만 필수 (데이터 의존).
- 나머지는 PLAN.md Setting A 기본값 → 사용자가 매번 검색 안 해도 됨.

**`super().__init__()`**:
- `nn.Module` 의 초기화. 빠뜨리면 `register_parameter` 등 PyTorch 내부 메커니즘 망가짐.
- PyTorch 초보자의 1순위 함정 — 이 코드는 명시적으로 통과.

**`lstm_dropout = dropout if num_layers > 1 else 0.0`** ★ **함정 방어 1**:
- PyTorch `nn.LSTM` 의 `dropout` 인자는 **층 사이** 적용 → 1층에선 효과 없음.
- 1층인데 dropout > 0 을 넘기면 **UserWarning** ("dropout option ... non-zero dropout expects num_layers greater than 1").
- 사용자 입장에선:
  - "dropout=0.2 줬는데 정규화 안 됨" — silent 무력화.
  - "warning 정도는 무시해도 되겠지" — 실제론 dropout 자체가 무효.
- 이 코드는 **선제적으로 0.0** 으로 바꿔서 PyTorch에 넘기고, dropout 효과는 별도 레이어로 보장.

**`self.head_dropout = nn.Dropout(dropout) if num_layers == 1 else nn.Identity()`** ★:
- 1층이면 별도 dropout 을 head 앞에 → 의도한 정규화 효과 보존.
- 2층 이상이면 LSTM 내부 dropout 이 동작하므로 head 단계는 `nn.Identity()` (no-op).
- **항상 `head_dropout` 속성** 을 두고 분기를 흡수 → forward 코드 단순화.

**`self.head = nn.Linear(hidden_size, 1)`**:
- LSTM hidden → scalar 매핑.

**`self._init_forget_gate_bias(value=1.0)`** — 함정 방어 3 (다음 절):
- 생성자 마지막 줄에서 **forget gate bias 만 1 로 재초기화**.

#### 6.4.2 `_init_forget_gate_bias` ([models.py:122-137](scripts/models.py))

```python
def _init_forget_gate_bias(self, value: float = 1.0) -> None:
    for name, param in self.lstm.named_parameters():
        if 'bias' in name:
            n = param.size(0)              # = 4 * hidden_size
            hs = n // 4                    # forget gate 구간 길이
            with torch.no_grad():
                param[hs : 2 * hs].fill_(value)
```

**왜 필요한가**:

LSTM 의 4개 게이트 — input(i), forget(f), cell(g), output(o):
```
hidden state h_t = o_t ⊙ tanh(c_t)
cell state   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
                   ↑
             forget gate: 이전 정보를 얼마나 유지할지
             (sigmoid 출력 0~1)
```

**문제**: PyTorch 기본 bias 초기화 = 0 → `f_t = sigmoid(0) = 0.5` → 매 시점마다 **이전 정보의 절반이 지워짐** → 학습 초기부터 vanishing gradient 가속.

**해법** (Jozefowicz et al. 2015):
- forget bias 만 1 로 초기화 → `f_t = sigmoid(1) ≈ 0.73` → 정보 대부분 보존.
- 학습이 진행되면서 모델이 "잊을 곳" 을 직접 배움.

**라인별 해부**:

`for name, param in self.lstm.named_parameters()`:
- LSTM 의 모든 파라미터 순회. 이름 + 텐서 짝.
- 예: `bias_ih_l0`, `bias_hh_l0`, `weight_ih_l0`, ... (num_layers=2 면 `_l1` 도)

`if 'bias' in name`:
- bias 만 골라냄.

`n = param.size(0); hs = n // 4`:
- bias 텐서 길이 = 4 * hidden_size (i, f, g, o 4구간).
- hs = 한 구간 길이.

`with torch.no_grad()`:
- gradient 추적 비활성. 초기화는 학습 그래프에 들어가면 안 됨.
- 누락하면 in-place 연산이 autograd 에 추적되어 다음 backward 에서 RuntimeError.

`param[hs : 2 * hs].fill_(value)`:
- bias 텐서의 [hs, 2*hs) 구간을 1.0 으로. 정확히 forget 구간.
- bias 레이아웃 `[i | f | g | o]` 에서 두 번째.

**`bias_ih_l*` 와 `bias_hh_l*` 둘 다 처리**:
- PyTorch LSTM 은 두 종류의 bias: input→hidden, hidden→hidden.
- 수학적으로 forget gate 는 `b_ih + b_hh` (둘 더해짐) → 한쪽만 1 해도 합 1.
- **둘 다 1** 로 하면 합이 2 → `sigmoid(2) ≈ 0.88`. 더 강한 정보 보존.
- 코드는 `'bias' in name` 으로 둘 다 잡음 (의도적).

#### 6.4.3 `forward` — batch_first 분기 흡수 ([models.py:139-159](scripts/models.py))

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    out, _ = self.lstm(x)                       # (B, T, H) 또는 (T, B, H)
    if self.batch_first:
        last = out[:, -1, :]
    else:
        last = out[-1, :, :]
    last = self.head_dropout(last)
    return self.head(last).squeeze(-1)          # (B, 1) → (B,)
```

**라인별 해부**:

`out, _ = self.lstm(x)`:
- LSTM 호출. 출력 = `(out, (h_n, c_n))`. out 은 모든 시점 hidden, h_n/c_n 은 마지막 cell state.
- `_` 로 무시.
- out shape: `batch_first=True` → `(B, T, H)`, `batch_first=False` → `(T, B, H)`.

`if self.batch_first: last = out[:, -1, :]` else `last = out[-1, :, :]` ★ **함정 방어 2**:
- 마지막 시점 hidden 추출.
- batch_first 에 따라 시간 축 위치가 다름 → 분기.
- 안 맞으면: 잘못된 축에서 -1 → 의미 없는 텐서 또는 shape mismatch RuntimeError.
- 클래스가 자동 처리 → 호출자는 batch_first 신경 안 써도 됨.

**왜 마지막 시점만? (vs 평균/max 풀링)**:
- LSTM 은 순방향으로 정보 누적 → 마지막 hidden 이 전체 시퀀스 요약.
- 평균 풀링은 "초기 시점 영향이 너무 크게 반영" — 먼 과거가 가까운 과거와 동등 가중.
- 수익률 예측은 **최근 정보가 더 중요** → 마지막 시점 우선.

`last = self.head_dropout(last)`:
- 1층 모델에서만 실제 dropout. 그 외엔 nn.Identity (no-op).
- 항상 같은 코드 라인, 분기 없음.

`return self.head(last).squeeze(-1)`:
- Linear(hidden, 1) → `(B, 1)`.
- `.squeeze(-1)` → `(B,)`.
- 호출자가 `loss = criterion(pred, y)` 에서 y shape 와 매칭되도록.

#### 6.4.4 `count_parameters` 유틸 ([models.py:162-180](scripts/models.py))

```python
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**한 줄 함수의 가치**:
- `model.parameters()`: 모든 파라미터 텐서 순회.
- `p.numel()`: 텐서 원소 수 (= 학습 가능 스칼라).
- `if p.requires_grad`: **freezing 된 파라미터 제외** — 전이학습 시 의미.

**왜 함수로 빼는가**:
- 일관성: 모든 노트북이 같은 정의 사용.
- 자기문서화: `count_parameters(model)` 이 의도 명확.
- 확장성: "MB 단위로 환산" 같은 기능 추가하기 쉬움.

**현재 프로젝트 사용 맥락**:
- Setting A 의 `LSTMRegressor(1, 128, 2, 0.2)` 는 약 199k 파라미터.
- Phase 1 의 fold 당 시퀀스 수 84 → ratio = 199000 / 84 ≈ **2370** → 매우 위험한 과적합 영역.
- 이런 진단을 가능하게 하는 게 이 보조 유틸.

---

### 6.5 `train.py` — 학습 루프 5가지 함정 방어

이 모듈은 PyTorch 학습 루프의 **5가지 흔한 함정** 을 한 함수에 봉인한다 (학습자료 §3.6):
1. `model.train()/model.eval()` 명시 전환
2. `optimizer.zero_grad()` 매 배치
3. val 단계 `torch.no_grad()`
4. `loss.item()` 으로 그래프 detach
5. gradient clipping

#### 6.5.1 `get_device` + 체크포인트 I/O ([train.py:46-107](scripts/train.py))

```python
def get_device(preference: str = 'auto') -> torch.device:
    if preference != 'auto':
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_checkpoint(state_dict, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, p)


def load_checkpoint(path, device='cpu'):
    return torch.load(Path(path), map_location=device)
```

**`get_device`**:
- 'auto' → cuda > mps > cpu 우선순위.
- `hasattr(torch.backends, 'mps')` — 구버전 PyTorch 에 mps 속성 없을 수 있음 → 호환.

**`save_checkpoint`**:
- `p.parent.mkdir(parents=True, exist_ok=True)` — 부모 디렉토리 자동 생성. 저장 실패 방지.
- `torch.save(state_dict, p)` — pickle 기반.

**`load_checkpoint`**:
- `map_location=device` — GPU 학습된 모델을 CPU 로 안전 로드 가능.
- map_location 누락 시 GPU 에 자동 로드 → CPU 환경에서 RuntimeError.

#### 6.5.2 `train_one_fold` — 초기화 ([train.py:113-192](scripts/train.py))

```python
def train_one_fold(model, train_loader, val_loader, *,
                   max_epochs=100, lr=1e-3, weight_decay=1e-4,
                   huber_delta=0.01, grad_clip=1.0,
                   early_stop_patience=10, lr_patience=5, lr_factor=0.5,
                   device='auto', verbose=True, log_every=1) -> Dict[str, Any]:
    dev = get_device(device) if isinstance(device, str) else device
    model.to(dev)
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=lr_patience, factor=lr_factor,
    )

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val = float('inf')
    best_state = {}
    best_epoch = 0
    patience_counter = 0
    stopped_early = False
```

**PLAN.md 확정 사양**:
- **Loss = HuberLoss(delta=0.01)** — 일별 수익률 ~1% 스케일에서 outlier 강건. delta 미만은 L2(부드러움), 초과는 L1(robust).
- **Optimizer = AdamW** — Adam + decoupled weight decay. weight_decay 가 lr 과 분리되어 더 안정적.
- **Scheduler = ReduceLROnPlateau** — val loss 정체 시 lr * 0.5. Plateau 패턴에서 효과적.

**키워드-only 인자 (`*`)**:
- 위치 인자 실수 방지. 모든 하이퍼는 키워드로만 전달.
- `train_one_fold(model, train_loader, val_loader, max_epochs=100)` 만 OK.
- `train_one_fold(model, train_loader, val_loader, 100)` 은 TypeError.

**상태 변수 초기화**:
- `best_val = float('inf')` — 첫 epoch 의 val_loss 가 무조건 best 가 되도록.
- `best_state = {}` — best state_dict 보관.
- `patience_counter = 0` — EarlyStopping 카운터.

#### 6.5.3 학습 phase ([train.py:194-209](scripts/train.py))

```python
for epoch in range(1, max_epochs + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb = xb.to(dev)
        yb = yb.to(dev)
        optimizer.zero_grad()                                   # 함정: zero_grad 누락 금지
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        train_losses.append(loss.item())                        # 함정: item() 으로 그래프 detach
    train_loss = float(np.mean(train_losses))
```

**라인별 함정 방어**:

`model.train()`:
- BatchNorm·Dropout 을 학습 모드로. eval 에서 train 모드로 안 돌아오면 dropout 비활성 → 과적합.

`xb.to(dev) / yb.to(dev)`:
- 매 배치마다 device 이동. CPU 에서 GPU 로 매번 보내는 비용은 있지만 명시적.

`optimizer.zero_grad()` ★:
- gradient 누적 방지. 빠뜨리면 매 배치 gradient 가 더해져서 lr 이 사실상 epoch 단위로 폭주.

`pred = model(xb); loss = criterion(pred, yb); loss.backward()`:
- forward → loss → backward. 표준 패턴.

`if grad_clip is not None and grad_clip > 0: nn.utils.clip_grad_norm_(...)` ★:
- LSTM gradient explosion 방지. `max_norm=1.0` (PLAN.md) 가 일반적 안전선.
- None/0 일 때 skip 가능 (디버깅 시 비교용).

`optimizer.step()`:
- weight update.

`train_losses.append(loss.item())` ★:
- `loss.item()` — 텐서를 Python float 로. **그래프 detach** 필수.
- 만약 `train_losses.append(loss)` 만 쓰면 매 배치의 그래프가 리스트에 보관 → epoch 끝까지 메모리 누수.

`train_loss = float(np.mean(train_losses))`:
- epoch 평균. history 에 기록.

#### 6.5.4 검증 phase ([train.py:211-228](scripts/train.py))

```python
    model.eval()                                                # 함정: eval 모드 전환 필수
    val_losses = []
    with torch.no_grad():                                       # 함정: no_grad 로 메모리 절약
        for xb, yb in val_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_losses.append(loss.item())
    val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['lr'].append(current_lr)
```

`model.eval()` ★:
- dropout/BN 추론 모드. dropout 이 모두 통과(스케일 자동 보정), BN running stats 사용.
- 누락 시 평가 시에도 dropout 활성 → 평가 결과 분산 폭주.

`with torch.no_grad()` ★:
- 그래프 미생성 → 메모리·속도 이득.
- 누락 시 forward 그래프가 메모리에 쌓임 → OOM 가능.

`val_loss = ... if val_losses else float('nan')`:
- 빈 로더 방어. NaN 으로 채움.

`scheduler.step(val_loss)`:
- ReduceLROnPlateau 가 val_loss 모니터. 정체 시 lr 감소.
- 학습 phase 가 아닌 **검증 phase 후** 호출해야 함 — val_loss 가 인자.

`current_lr = optimizer.param_groups[0]['lr']`:
- 현재 lr 추출. history 에 기록 → 학습 곡선 시각화에서 lr drop 시점 확인 가능.

#### 6.5.5 best 갱신 + EarlyStopping ([train.py:230-263](scripts/train.py))

```python
    improved = val_loss < best_val - 1e-8
    if improved:
        best_val = val_loss
        best_state = deepcopy(model.state_dict())
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1

    if verbose and (epoch % log_every == 0 or improved):
        flag = '  * best' if improved else ''
        print(f'[ep {epoch:3d}] train={train_loss:.6f}  '
              f'val={val_loss:.6f}  lr={current_lr:.2e}{flag}')

    if patience_counter >= early_stop_patience:
        stopped_early = True
        if verbose:
            print(f'  -> EarlyStopping at epoch {epoch} ...')
        break

return {
    'best_state_dict': best_state,
    'history': history,
    'best_epoch': best_epoch,
    'best_val_loss': best_val,
    'stopped_early': stopped_early,
}
```

**`val_loss < best_val - 1e-8`**:
- `1e-8` epsilon — 부동소수점 잡음 방지. `val_loss == best_val` 같은 미세 변화는 "갱신 아님" 으로 처리.

**`best_state = deepcopy(model.state_dict())`** ★:
- best state 메모리 보관. 이후 학습이 weights 를 덮어써도 안전.
- `model.state_dict()` 만 쓰면 **참조 보관** → 다음 epoch 에서 같이 변경.
- `deepcopy` 로 **값 보관** 필수.

**EarlyStopping**:
- patience 초과 시 학습 중단.
- 누적 history 그대로 반환 → 학습 곡선 시각화 가능.

**리턴 dict 구조**:
- 5개 키 — best_state_dict / history / best_epoch / best_val_loss / stopped_early.
- 호출자가 필요한 것만 골라 사용.

---

### 6.6 `metrics.py` — 게이팅 지표 + 베이스라인

이 모듈은 **Phase 1 게이팅 기준** (hit_rate > 0.55 AND r2_oos > 0) 을 정의하고, LSTM 결과를 3가지 베이스라인과 비교하는 표를 제공한다.

#### 6.6.1 `hit_rate` — 부호 적중률 ([metrics.py:35-64](scripts/metrics.py))

```python
def hit_rate(y_true, y_pred, exclude_zero=True) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if exclude_zero:
        mask = (yt != 0) & (yp != 0)
        yt = yt[mask]
        yp = yp[mask]
    if len(yt) == 0:
        return float('nan')
    return float((np.sign(yt) == np.sign(yp)).mean())
```

**라인별 해부**:

`np.asarray(y_true, dtype=float)`:
- list/Series/ndarray 어떤 입력이든 ndarray 로 통일. dtype=float 명시로 정수 입력도 안전 처리.

`if exclude_zero: mask = (yt != 0) & (yp != 0)`:
- 정확히 0 인 샘플 제외. `np.sign(0) = 0` 이라 `sign(0) == sign(non-zero)` 가 항상 False → 비교 모호.
- 수익률 시계열에서 정확히 0 은 거의 없지만 (drift 가 있으니), 안전 가드.

`if len(yt) == 0: return float('nan')`:
- 마스킹 후 비교 가능 샘플 0 → nan 반환. assert 안 함 (정상 시나리오일 수 있음).

`(np.sign(yt) == np.sign(yp)).mean()`:
- 부호 일치 비율. True/False 평균이 비율.

**Phase 1 게이팅 기준 1**: > 0.55.
- 50% (랜덤) 보다 5%p 이상.

#### 6.6.2 `r2_oos` — Campbell-Thompson ([metrics.py:67-103](scripts/metrics.py))

```python
def r2_oos(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sse = float(((yt - yp) ** 2).sum())
    sum_y2 = float((yt ** 2).sum())
    if sum_y2 == 0:
        return float('nan')
    return 1.0 - sse / sum_y2
```

**공식 해부**:
```
r2_oos = 1 - sum((y - y_hat)^2) / sum(y^2)
```

- 분모가 **`sum(y^2)`** — 표준 R² 와의 차이.
- 표준 R²: `1 - SSE / SST_mean` (분모는 분산 = 평균 빼고 제곱합).
- r2_oos: 분모가 평균 안 빼고 제곱합 = "0 예측 baseline" 의 SSE.
- 의미: **0 예측 baseline 대비 개선률**.

**왜 평균을 안 빼는가**:
- 수익률 평균은 0 에 가까움 (Campbell & Thompson 2008 가정).
- 평균 빼면 거의 변화 없으니, 더 직관적인 "0 예측" 을 baseline 으로.

**Phase 1 게이팅 기준 2**: > 0.
- > 0 → 모델이 0 예측보다 낫다.
- < 0 → 모델이 0 예측보다 못하다 (즉시 재검토).
- 현재 SPY R²_OOS = -0.1552 로 **실패 상태**.

`if sum_y2 == 0: return float('nan')`:
- y 가 모두 0 이면 정의 불가능.

#### 6.6.3 `r2_standard` / `mae` / `rmse` ([metrics.py:106-138](scripts/metrics.py))

```python
def r2_standard(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sse = float(((yt - yp) ** 2).sum())
    sst = float(((yt - yt.mean()) ** 2).sum())
    if sst == 0:
        return float('nan')
    return 1.0 - sse / sst


def mae(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.abs(yt - yp).mean())


def rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(((yt - yp) ** 2).mean()))
```

**`r2_standard`**:
- sklearn `r2_score` 동일.
- 분모 SST = sum((y - mean(y))^2) → "평균 예측 baseline" 대비.
- 수익률에선 mean ≈ 0 이라 `r2_oos` 와 비슷한 값이지만, 일반 시계열엔 다른 척도.

**`mae` (Mean Absolute Error)**:
- 절대 오차 평균. 단위가 수익률과 같아 해석 직관적 ("평균 0.005 만큼 빗나감").

**`rmse` (Root Mean Squared Error)**:
- 제곱 오차 평균의 제곱근. 큰 오차에 더 민감.

#### 6.6.4 `baseline_metrics` — 3종 베이스라인 일괄 ([metrics.py:144-201](scripts/metrics.py))

```python
def baseline_metrics(y_test, y_train) -> Dict[str, Dict[str, float]]:
    yt = np.asarray(y_test, dtype=float)
    ytr = np.asarray(y_train, dtype=float)
    if len(ytr) == 0:
        raise ValueError('y_train 이 비어 있습니다.')

    y_zero = np.zeros_like(yt)
    y_mean = np.full_like(yt, fill_value=float(ytr.mean()))
    y_prev = np.concatenate([[float(ytr[-1])], yt[:-1]])

    baselines = {'zero': y_zero, 'previous': y_prev, 'train_mean': y_mean}
    out = {}
    for name, yp in baselines.items():
        out[name] = {
            'hit_rate': hit_rate(yt, yp),
            'r2_oos': r2_oos(yt, yp),
            'r2_standard': r2_standard(yt, yp),
            'mae': mae(yt, yp),
            'rmse': rmse(yt, yp),
        }
    return out
```

**3가지 베이스라인**:

1. **`zero`** — 항상 0 예측:
   - r2_oos 의 정의상 baseline (분모가 sum(y²) = 0 예측의 SSE).
   - hit_rate 는 항상 nan (sign(0) = 0 → 일치 불가).

2. **`previous`** — 직전 값 유지 (random walk):
   - `np.concatenate([[float(ytr[-1])], yt[:-1]])` — 첫 샘플은 y_train 의 마지막, 나머지는 y_test[i-1].
   - train→test 경계가 자연 연결.
   - 시계열에서 가장 강한 baseline 중 하나.

3. **`train_mean`** — 훈련 기간 평균:
   - 역사적 평균 baseline.
   - 수익률 평균이 0 가까우면 zero 와 거의 같음.

**`raise ValueError(...)` if `len(ytr) == 0`**:
- y_train 이 비면 previous/train_mean 계산 불가 → 즉시 실패.

**현재 프로젝트 결과**:
- Setting A SPY: LSTM hit_rate = 0.6442 vs `previous` hit_rate = 0.92 → **LSTM 이 단순 random walk 보다 못함** → Phase 1 실패 신호.

#### 6.6.5 `summarize_folds` — 폴드 통계 요약 ([metrics.py:207-249](scripts/metrics.py))

```python
def summarize_folds(per_fold_metrics) -> Dict[str, Dict[str, float]]:
    if not per_fold_metrics:
        return {}
    keys = set()
    for d in per_fold_metrics:
        keys.update(d.keys())

    out = {}
    for k in keys:
        raw = [d.get(k, float('nan')) for d in per_fold_metrics]
        arr = np.array([v for v in raw if v is not None and not np.isnan(v)],
                       dtype=float)
        if len(arr) == 0:
            out[k] = {'mean': float('nan'), 'std': float('nan'),
                      'min': float('nan'), 'max': float('nan'), 'n': 0}
        else:
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            out[k] = {
                'mean': float(arr.mean()),
                'std': std,
                'min': float(arr.min()),
                'max': float(arr.max()),
                'n': int(len(arr)),
            }
    return out
```

**라인별 해부**:

`if not per_fold_metrics: return {}`:
- 빈 입력 가드.

`keys = set(); for d in per_fold_metrics: keys.update(d.keys())`:
- 모든 fold 의 키 합집합. 일부 fold 에 없는 키도 안전 처리.

`raw = [d.get(k, float('nan')) for d in per_fold_metrics]`:
- 키 없으면 NaN 으로.

`arr = np.array([v for v in raw if v is not None and not np.isnan(v)], dtype=float)`:
- NaN 과 None 제거. 유효 값만 통계.
- NaN 을 통계에 포함시키면 mean/std 가 NaN 이 되어버림.

`std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0`:
- **`ddof=1`** — 표본 표준편차 (1/(n-1)). 모집단(`ddof=0`) 이 아님.
- fold 1개일 때 `ddof=1` 은 NaN 발생 → 0.0 으로 fallback.

`'n': int(len(arr))`:
- 유효 fold 수 표시.
- 만약 5/106 fold 만 유효하면 n=5 → "거의 다 NaN" 진단 가능.

**현재 프로젝트의 의의**:
- Phase 1 fold 분산 폭주 (R² std 0.97~1.73) 가 이 함수 출력에서 즉시 보임.
- `n=106` 인데 std > mean 의 절대값 → **mean 이 거의 의미 없음** → 모델 불안정.
- Setting A 실패 진단의 한 축.

---

## 6.X 데이터 플로우 한눈에

§6 의 6모듈이 어떻게 협업하는지:

```
[01_data_download.ipynb]
    └─ adj_close (yfinance)
         │
         ▼
[targets.build_daily_target_21d]      → target  (NaN 양끝)
[targets.verify_no_leakage]           → assert PASS
         │
         ▼
[02_setting_A.ipynb 부트스트랩]
    setup.bootstrap()                  → font/seed/dirs
         │
         ▼
[dataset.walk_forward_folds]          → [(train_idx, test_idx), ...]
    for fold in folds:
        [dataset.build_fold_datasets]  → train_ds, test_ds, scaler
                                         (StandardScaler train으로만 fit)
        [models.LSTMRegressor]         → forget bias=1, head_dropout
        [train.train_one_fold]         → best_state_dict, history
        [metrics.hit_rate / r2_oos]    → per-fold metrics
        [metrics.baseline_metrics]     → zero/prev/mean 비교

[metrics.summarize_folds]             → mean/std/min/max
        │
        ▼
results/setting_A/{SPY,QQQ}/metrics.json
results/setting_A/{SPY,QQQ}/fold0_learning_curve.png
```
