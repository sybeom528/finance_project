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
