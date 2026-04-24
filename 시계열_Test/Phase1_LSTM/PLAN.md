# Phase 1 — LSTM 단독 베이스라인 구축 계획 (팀 공유 사본)

> **이 문서는 팀 공유용 사본입니다.**
>
> - **진실원(원본)**: Claude Code의 plan 파일 (`C:\Users\gorhk\.claude\plans\c-users-gorhk-finance-project-study-00-m-frolicking-iverson.md`)
> - **마지막 동기화**: 2026-04-24
> - **동기화 정책**: 진실원이 갱신될 때마다 본 사본도 함께 갱신합니다. 본 사본을 직접 수정하지 마시고, 변경 사항은 `재천_WORKLOG.md` 또는 본인 prefix `<이름>_WORKLOG.md` 에 제안 형태로 기록 후 협의하십시오.
> - **목적**: 팀원이 Claude의 plan 파일에 직접 접근하지 않고도 같은 계획을 한곳에서 확인하기 위함입니다.

---

## Context

팀 프로젝트(COL-BL: Su et al. 2026 ESWA 295 논문 재현)의 6단계 모델 평가 로드맵 중 **Phase 1**을 구축합니다. 학습자료(`김재천/Study/`) 생성은 일시 중단하고 실전 평가로 진입하는 첫 단계입니다.

**Phase 1의 목적**: BL Q 벡터 생성이 아닌, **LSTM 자체의 수익률 예측 성능**을 두 가지 시간 해상도에서 검증하는 것입니다. 사용자 가설은 "수익률의 자기상관이 극단적으로 작아 univariate만으로는 성능이 부족할 것"이며, 이 가설의 정량적 확인이 Phase 1의 핵심 산출물입니다.

평가 결과(R² 등)와 univariate의 한계가 드러나면, Phase 2 GRU 교체 또는 추가 피처 도입으로 이어집니다.

**평가할 6개 모델 로드맵 (Phase 1은 1·2번)**:
1. LSTM × ETF × 월별(1개월 뒤) — 설정 B
2. LSTM × ETF × 일별(21영업일 뒤) — 설정 A
3. GRU × ETF × 월별(1) — Phase 2
4. GRU × ETF × 일별(21) — Phase 2
5. CEEMDAN+LSTM × 월별(1) — Phase 3
6. CEEMDAN+LSTM × 일별(21) — Phase 3

---

## 확정된 의사결정

| 항목 | 결정 |
|---|---|
| 자산군 | **SPY, QQQ** (각각 독립 학습·평가) |
| 데이터 소스 | **yfinance 신규 다운로드** |
| 다운로드 기간 | **2009-01-01 ~ 2026-03-31** (불변) |
| 분석 기간 | **2016-01-01 ~ 2025-12-31** (10년, 2026-04-24 수정) |
| 입력 피처 | **Univariate — log-return 단일 채널** (가설 검증용) |
| 결과물 위치 | `시계열_Test/Phase1_LSTM/` (2026-04-24 finance_project 직하위로 이동) |
| 코드 참고 | 기존 Study 자료의 **패턴/원리만 참고**, 코드는 처음부터 새로 작성 |
| 설정 A | 일별 + 21일 후 누적 log-return / seq_len=126 / hidden=128 / 2-layer |
| 설정 B | 월별 + 1개월 후 log-return / seq_len=24 / hidden=64 / 1-layer |
| 평가 지표 (1차) | **Hit Rate** (방향 적중률), **R²_OOS** (out-of-sample, 0 예측 대비) |
| 평가 지표 (2차/보조) | MAE, RMSE, 표준 R², 베이스라인 비교(직전값/평균) |
| **관문 (Phase 2 진행 조건)** | **Hit Rate > 0.55** AND **R²_OOS > 0** (둘 다 충족 필요) |
| 관문 미달 시 | 설정·피처·모델 재검토 (Phase 2 진행 보류) |

---

## 폴더 구조 — 노트북 + `scripts/*.py` 협업 구조 (2026-04-24 변경)

**배경**: 다른 팀원과 공유 작업으로 전환 → 재사용 가능한 함수·클래스를 `scripts/*.py`로 분리하여 import 가능한 형태로 관리합니다. 노트북은 분석 흐름·시각화·결과 해석에 집중합니다.

**역할 분담 원칙**:
- **`scripts/*.py`**: 재사용 함수·클래스·상수 (Dataset, Model, train loop, metrics, walk-forward fold 생성기 등). 다른 팀원·다른 Phase에서도 import.
- **`*.ipynb`**: 데이터 로드·시각화·결과 분석 흐름. 코드 셀은 주로 `from scripts.X import Y` 후 호출.
- **`00_setup_and_utils.ipynb`**: 환경 노트북은 유지(% run 호환), 같은 내용을 `scripts/setup.py`로도 제공해 `import scripts.setup` 가능.

```
시계열_Test/Phase1_LSTM/
├── README.md                              # 협업 진입점 — 실행 순서·의존 그래프·환경 설치
├── PLAN.md                                # ⭐ 본 문서 (진실원의 팀 공유 사본)
├── 재천_WORKLOG.md                         # 작업·판단 일지 (작성자별 prefix, 모든 결정 누적)
├── 00_setup_and_utils.ipynb               # 환경 노트북 (%run 호환, 인터랙티브 검증용)
├── 01_data_download_and_eda.ipynb         # yfinance + EDA + ACF
├── 02_setting_A_daily21.ipynb             # 설정 A 흐름 (scripts에서 함수 import)
├── 03_setting_B_monthly.ipynb             # 설정 B 흐름
├── 04_compare_A_vs_B.ipynb                # A·B 비교 + 최종 보고
│
├── scripts/                               # ⭐ 재사용 모듈 (협업용)
│   ├── __init__.py                        # 패키지 마커
│   ├── setup.py                           # 한글 폰트·시드·경로 상수 (00 노트북과 동일 로직)
│   ├── data_io.py                         # CSV 로드·분석 기간 절단·워밍업 처리
│   ├── targets.py                         # 21일 누적 / 월별 1개월 타깃 + 누수 검증 단위 함수
│   ├── cv_walkforward.py                  # Walk-Forward fold 생성기 (IS/purge/emb/OOS/step)
│   ├── dataset.py                         # SequenceDataset (PyTorch)
│   ├── models.py                          # LSTMRegressor
│   ├── train.py                           # 학습 루프 (Huber, EarlyStop, scheduler, ckpt)
│   ├── metrics.py                         # Hit Rate, R²_OOS, MAE, RMSE, baseline 비교
│   └── plot_utils.py                      # 한글 폰트 + 표준 플롯 함수
│
└── results/                               # 노트북 실행 결과만 저장 (코드 없음)
    ├── raw_data/                          # SPY.csv, QQQ.csv
    ├── setting_A/{SPY,QQQ}/               # metrics.json, model.pt, *.png
    ├── setting_B/{SPY,QQQ}/               # fold별 ckpt
    └── comparison_report.md               # 04에서 자동 생성
```

**모듈 인터페이스 원칙 (협업 안정성)**:
- 모든 public 함수에 type hints + docstring (Numpy style) 필수
- `scripts/X.py` 변경 시 docstring·signature 변경 사항을 본인 prefix `<이름>_WORKLOG.md` 에 기록
- 노트북에서 사용하는 모든 함수는 import 경로 명시 (`from scripts.targets import build_daily_target_21d`)

**노트북 셀 작성 원칙 (CLAUDE.md 강화 반영, 변경 없음)**:
- 모든 코드 셀 직전에 **마크다운 셀**로 (a) 무엇을 하는지, (b) 왜 이 방식인지, (c) 대안과의 트레이드오프, (d) 주의할 함정을 명시
- 코드 셀 내부도 각 핵심 라인에 인라인 주석 — 특히 누수 위험 라인 (shift, fit, split)
- 함수·클래스 정의 셀에는 docstring 필수 (인자·반환·예제)
- 어려운 개념(Walk-Forward, Purged K-Fold, Huber loss 등)은 마크다운에서 수식·그림 또는 ASCII 다이어그램으로 설명

---

## 구현 단계 (대화·피드백 체크포인트 명시)

CLAUDE.md 지침에 따라 각 단계 종료 시 사용자에게 결과 보고 + 다음 단계 진행 승인을 요청합니다.
**모든 노트북 셀에 마크다운 설명 + 인라인 주석 + 함수 docstring 의무**.

### Step 0. 환경 노트북 (`00_setup_and_utils.ipynb`)
- 한글 폰트 설정(Malgun Gothic), 시드 고정(seed=42, torch deterministic), 경로 상수(`RAW_DIR`, `RESULTS_DIR`)
- 다른 노트북 첫 셀에서 `%run ./00_setup_and_utils.ipynb` 한 줄로 일괄 호출
- **마크다운**: 왜 환경 노트북을 따로 두는가, %run 패턴의 의미, 시드 고정의 한계

### Step 1. 데이터 수집 + EDA (`01_data_download_and_eda.ipynb`)
- §1 yfinance로 SPY, QQQ 일별 OHLCV 다운로드 (2009-01-01 ~ 2026-03-31)
  - 저장: `results/raw_data/{ticker}.csv`
  - 마크다운: 왜 워밍업 1년을 두는가, Adj Close vs Close 선택 근거
- §2 결측·이상치 점검 (Adj Close 기준, ±5σ 임계 보수적)
- §3 일별 log-return 분포 (히스토그램·QQ-plot)
- §4 **ACF/PACF lag 1~30** — 사용자 가설("자기상관 극단적 약함") 정량 확인
  - 마크다운: ACF 해석법, 95% 신뢰구간 의미, 통계적 유의 ≠ 경제적 의미
- **체크포인트**: 결측 처리 방침·이상치 임계 사용자 합의

### Step 2. 설정 A 전체 (`02_setting_A_daily21.ipynb`) — 노트북은 흐름·시각화·해석, 함수·클래스는 `scripts/*.py`
각 절 직전 마크다운 셀로 목적·근거·함정 명시.

**.py 분리 기준 (사용자 지시, 2026-04-24)**: Phase 1 전반에 공통으로 쓰이는 함수·클래스는 `scripts/*.py`에 정의하고 노트북에서 `from scripts.X import Y` 로 import. 한 노트북 안에서만 쓰는 흐름 코드는 노트북 셀에 둠.

- **§1 환경** — `%run ./00_setup_and_utils.ipynb` (또는 `from scripts.setup import bootstrap; bootstrap()`)
- **§2 데이터 로드** — `from scripts.data_io import load_ticker_csv, slice_analysis_period` 호출. 01에서 저장한 CSV를 읽고 분석 구간(2016~2025)으로 잘라냄. 워밍업 데이터는 lookback 채움용으로만 사용.
- **§3 타깃 생성** — `from scripts.targets import build_daily_target_21d` 호출. 내부에서 `log_ret = np.log(adj_close).diff()`; `target = log_ret.rolling(21).sum().shift(-21)` 수행.
  - 마크다운: 왜 21일인가(미국 영업월), shift 부호 그림 설명
- **§4 누수 검증** — `from scripts.targets import verify_no_leakage` 호출 + 노트북 셀에서 육안 검증 표 직접 출력
  - `verify_no_leakage(...)` 내부: 3개 무작위 시점에서 `assert target[t] == log_ret[t+1:t+22].sum()`
  - 첫 5행 시점-타깃 매핑 표 출력 셀 (노트북에서 직접) → 사람이 육안으로 미래 참조 확인
  - 인공 누수 대조: `from scripts.targets import build_leaky_target_for_test` 로 의도적 누수 시계열 생성 → 동일 모델 1 epoch 학습 → R² > 0.9 나오는지 확인
- **§5 SequenceDataset import** — `from scripts.dataset import SequenceDataset`
  - `__getitem__(i)` → `(X[i:i+seq_len], target[i+seq_len-1])`
  - scaler는 외부 주입 패턴 (train fit / val·test transform)
  - 클래스 정의는 `scripts/dataset.py` 안에 docstring·type hints 포함
- **§6 Walk-Forward fold 생성** — `from scripts.cv_walkforward import generate_folds` 호출
  - Rolling Walk-Forward + Purge(21) + Embargo(21): IS 231 / purge 21 / emb 21 / OOS 21 / step 21
  - 각 fold의 IS 안에서 시간순 80/20 train/val 분할 (+ 내부 gap = seq_len)
  - 함수 시그니처 예: `generate_folds(n_total, is_len, purge, embargo, oos_len, step) -> List[Fold]`
  - 노트북에서 fold 루프를 돌며 fold 당 독립 학습·평가 → fold별 메트릭 수집 → 평균±표준편차 보고
  - 마크다운: 왜 Rolling Walk-Forward + Purge + Embargo인가 (López de Prado 2018 근거)
- **§7 LSTMRegressor import** — `from scripts.models import LSTMRegressor`
  - `LSTMRegressor(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)` + 내부 `Linear(128, 1)` 헤드
  - 마크다운: 마지막 시점 hidden 추출 vs 평균 풀링 선택 이유
- **§8 학습 루프 호출** — `from scripts.train import train_one_fold`
  - `train_one_fold(model, train_loader, val_loader, **hp) -> {best_state_dict, history}` 형태
  - Loss: Huber(delta=0.01) / Optimizer: AdamW(lr=1e-3, wd=1e-4) / Scheduler: ReduceLROnPlateau(patience=5, factor=0.5)
  - Gradient clipping max_norm=1.0 / EarlyStopping patience=10
  - Checkpoint: best val loss 기준 저장 (`results/setting_A/{ticker}/fold_{k}_model.pt`)
  - 노트북은 fold 루프·진행률 표시·결과 집계만 담당. 실제 학습 로직은 `scripts/train.py` 함수 안에.
  - 마크다운: 각 컴포넌트 선택 근거, 가능한 함정 (zero_grad 누락 등)
- **§9 평가·시각화** — `from scripts.metrics import hit_rate, r2_oos, baseline_metrics` 호출
  - **1차 메트릭 (관문 판정)**:
    - **Hit Rate** = `mean(sign(y_pred) == sign(y_true))` (test 셋, 0 제외 후) — 관문: > 0.55
    - **R²_OOS** = `1 - SSE_model / SSE_zero_baseline` = `1 - sum((y - y_hat)²) / sum(y²)` — Campbell & Thompson(2008) 정의. 0 예측 대비 개선이면 양수. 관문: > 0
  - **2차 메트릭**: MAE, RMSE, 표준 R²(평균 baseline), 직전값 예측 대비 RMSE 개선폭
  - **베이스라인 비교 (필수)**: 직전 값 예측, 0 예측, 학습 평균 — `baseline_metrics(...)` 한 번 호출로 표 산출
  - 플롯: `from scripts.plot_utils import plot_learning_curve, plot_pred_vs_actual, plot_residuals, plot_sign_confusion` (시각화 함수도 모듈화)
  - 결과 → `results/setting_A/{ticker}/metrics.json` (Hit Rate, R²_OOS, MAE, RMSE, 베이스라인 메트릭 모두 기록)
- **§10 결론·메모** — 관문 통과 여부, 의외 발견 기록 (노트북 마크다운 셀)

### Step 3. 설정 B 전체 (`03_setting_B_monthly.ipynb`) — 02와 동일 모듈 import, 파라미터만 변경
- §1~9는 02와 동일 골격(같은 `scripts/*.py` 모듈을 import)이나 다음 차이:
  - §3 타깃: `from scripts.targets import build_monthly_target_1m` 호출. 내부에서 `monthly_close = adj_close.resample('ME').last()`; `target = np.log(monthly_close).diff().shift(-1)`
  - §6 Walk-Forward: 같은 `generate_folds(...)` 호출, 파라미터만 IS 11개월 / purge 1 / emb 1 / OOS 1 / step 1 (월별 비례 축소) — Step 3 진입 직전 사용자 최종 확인
  - §7 모델: 같은 `LSTMRegressor` 클래스, 인자만 hidden=64, num_layers=1, dropout=0.3 (1-layer는 LSTM dropout 인자 무시 → 클래스 내부에서 Linear 직전 별도 `nn.Dropout` 적용 필요)
  - §8 학습: 같은 `train_one_fold(...)`, 인자 EarlyStopping patience=5
  - §9 평가: fold별 메트릭 + 평균±표준편차 보고 (메트릭 함수 동일)
- 마크다운: 왜 설정 B는 과적합 위험이 큰가, 월별 walk-forward의 기간 매핑 근거
- **이점**: A·B가 같은 코드 베이스를 공유하므로 비교 공정성 자연 확보, 모듈 변경 시 양쪽 동시 적용

### Step 4. 비교 보고 (`04_compare_A_vs_B.ipynb` + `comparison_report.md`)
- 4개 셀(설정 A·B × SPY·QQQ) 메트릭 통합 표 (Hit Rate, R²_OOS, MAE, RMSE)
- Hit Rate · R²_OOS 관문 통과 매트릭스 (4셀 × 2지표 = 8칸 PASS/FAIL)
- **관문 판정 규칙**: Hit Rate > 0.55 AND R²_OOS > 0 둘 다 충족 시 PASS
- 부호 혼동행렬·산점도 비교
- 사용자 가설("자기상관 약함 → 성능 부족") 검증 결론
- Phase 2(GRU) 진행 권고 또는 추가 피처 도입 제안
- `comparison_report.md` 자동 생성 (마크다운 표 + 그림 링크)

---

## Walk-Forward 구조 (2026-04-24 확정, 설정 A·B 공통)

사용자 이미지 명세대로 **Rolling Walk-Forward + Purge + Embargo** 를 설정 A·B 모두에 일관 적용합니다.

```
날짜 축 →
Fold 1: ├── IS (231일) ──┤ purge(21) │ emb(21) │ OOS(21) ┤
Fold 2: (21일 오른쪽 이동) ├── IS (231일) ──┤ purge │ emb │ OOS ┤
Fold 3: ...
... (약 79~100개 fold, 10년치)
```

### 설정 A (일별) — 이미지 직접 적용

| 구성 요소 | 길이 |
|---|---|
| IS (In-Sample, train+val) | **231 영업일** (~11개월) |
| Purge | **21 영업일** (타깃 누수 차단) |
| Embargo | **21 영업일** (autocorrelation 차단) |
| OOS (test) | **21 영업일** |
| Step (fold 간 이동) | **21 영업일** (rolling sliding) |
| 예상 fold 수 | 약 79~100개 (10년 기준) |

### 설정 B (월별) — 비례 축소 (제안, Step 3 진입 직전 사용자 확인)

| 구성 요소 | 길이 |
|---|---|
| IS | **11개월** |
| Purge | **1개월** |
| Embargo | **1개월** |
| OOS | **1개월** |
| Step | **1개월** |
| 예상 fold 수 | 약 106개 (10년 기준, 월별 ~120 샘플) |

### IS 내부 train/val 재분할 (EarlyStopping·LR scheduler 용)
- 각 fold의 IS 안에서 시간순 80/20 분할
  - 설정 A: train 184일 + val 47일 (+ 내부 gap = seq_len=126)
  - 설정 B: train 9개월 + val 2개월 (+ 내부 gap = seq_len=24개월이라 val 양 부족 시 조정)
- val은 EarlyStop·LR 스케줄러에만 사용, OOS는 오로지 test 보고용

### 왜 Purge + Embargo 동시 적용 (학술 근거)
- **Purge** (López de Prado 2018, *Advances in Financial Machine Learning*): IS 샘플의 타깃 생성 구간이 OOS 시작과 겹치는 것을 차단. 설정 A 타깃은 21일 누적 → purge=21 필수.
- **Embargo**: IS 학습 직후 autocorrelation 잔존이 OOS 첫 구간을 부풀리는 것을 방지.
- **금융 시계열 walk-forward 누수 방지의 사실상 학술 표준**.

---

## ⚠️ 데이터 누수 방지 (Phase 1 최우선 원칙)

사용자 지시: **"데이터 누수를 가장 경계하여 코드를 작성하고 작업할 것."**
모든 노트북·셀·함수는 아래 4계층 방어선을 거쳐야 합니다.

### 누수 발생 가능 지점 (체크리스트)

| # | 지점 | 위험 | 방어책 |
|---|---|---|---|
| 1 | **타깃 shift 부호** | `shift(-21)` 대신 `shift(21)` 시 미래값 → 입력에 누설 | §4 누수 검증 셀 (assert + 5행 출력 + 인공 누수 비교) |
| 2 | **rolling 윈도우 정렬** | `rolling(21).sum()`은 기본 trailing(과거 21일 합) — 타깃은 forward 합이어야 함 | `.shift(-21)` 명시 + 첫 5행 직접 검증 |
| 3 | **Scaler fit 범위** | StandardScaler를 전체에 fit 시 val/test 통계 누설 | scaler는 train에만 fit, val·test는 transform만. SequenceDataset은 외부 주입 패턴 강제 |
| 4 | **train/val/test split 경계** | seq_len=126일 윈도우가 split 경계를 넘으면 과거 데이터 일부가 미래 셋으로 흘러감 | split 사이 `gap = seq_len` 강제 (코드에서 슬라이싱 시 명시) |
| 5 | **월별 resample 정렬** | `resample('ME')` 후 shift(-1) 시 월말 close 기준 t→t+1 매핑 검증 필요 | 첫 5행 (date, close, target_t, close_{t+1}) 표 출력 |
| 6 | **EarlyStopping val 사용** | val을 hyperparam 선택에도 쓰면 val이 부분적으로 train화됨 | val은 EarlyStop·LR scheduler에만, 최종 메트릭은 test로만 보고 |
| 7 | **Walk-Forward fold** | fold 사이 embargo 없이 train→test 직결 시 t+1 타깃 누수 | (선택 시) Purged K-Fold + embargo 1개월 |
| 8 | **워밍업 데이터** | 2009 워밍업을 train에 넣으면 분석 구간 밖 정보 사용 | 워밍업은 lookback 채움용으로만, 손실 계산·메트릭 산출에서 제외 |

### 누수 검증 의무 (각 노트북 §4)

설정 A·B 노트북 모두 §4에 다음 3종 검증 셀을 두고, 셀 출력으로 PASS/FAIL 표시합니다.

1. **Assert 단위 테스트**: 3개 무작위 시점 t에 대해 `target[t] == 미래 구간 합`
2. **육안 검증 표**: 첫 5행 (date, log_ret, target_t, 미래 구간 직접 계산값) 나란히 출력
3. **인공 누수 대조**: target에 미래값을 의도적으로 누설시킨 가짜 시계열로 동일 모델 1 epoch 학습 → R² > 0.9 나오는지 확인 (나오면 모델·평가 코드는 정상, 안 나오면 평가 코드가 부서져 있음)

### 코드 작성 규약 (모든 노트북 적용)

- shift·rolling·resample 줄에 `# 누수: ...` 인라인 주석 의무
- Dataset/DataLoader 정의 셀에는 누수 방지 의도 docstring
- 평가 직전에 "test 데이터에 train 정보가 흘러갔는지" 자체 점검 셀
- Pull request 형태가 아니더라도, 사용자에게 누수 검증 PASS 결과를 매 노트북 종료 시 보고

---

## 한글 폰트·환경 (CLAUDE.md 강제)

모든 시각화 셀 상단에 다음 블록 포함:
```python
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    import koreanize_matplotlib
plt.rcParams['axes.unicode_minus'] = False
```

`plot_utils.py`에 `setup_korean_font()` 함수로 추출하여 모든 노트북에서 1줄 호출.

---

## 재사용 가능한 참고 자료 (코드는 새로 작성, 패턴만 참고)

- 학습계획: `김재천/Study/00_학습계획.md` (Phase 1·v2.4 반영 v3 업데이트 검토 별도)
- SequenceDataset 설계 의도 — `김재천/Study/week2_시퀀스데이터/04_pytorch_dataset_실습.ipynb`
- 학습 루프 표준 패턴 — `김재천/Study/week3_시퀀스모델/04_PyTorch_학습루프_실습.ipynb`
- 정규화·EarlyStopping 이론 — `김재천/Study/week3_시퀀스모델/05_학습안정화_정규화.md`
- StackedLSTM 구조 — `김재천/Study/week4_LSTM파생/01_StackedLSTM_실습.ipynb`

위 파일들은 **읽기 전용 참고**. Phase1_LSTM 폴더 내 코드는 처음부터 새로 작성합니다.

---

## End-to-End 검증 방안

1. **누수 검증 셀 (각 노트북 §4)**: assert + 육안 표 + 인공 누수 대조 3종 PASS
2. **노트북 Run All 재현성**: `01_*.ipynb` ~ `04_*.ipynb` 위→아래 1회 실행으로 동일 결과
3. **메트릭 직렬화**: `results/setting_X/{ticker}/metrics.json`에 seed·하이퍼·지표 모두 기록
4. **시각화 산출물**: 학습 곡선, 예측 vs 실측 산점도, 잔차 시계열, ACF (한글 깨짐 없음 육안 확인)
5. **베이스라인 우위 검증**: LSTM이 직전값 예측·0 예측 대비 RMSE 개선 확인 (개선폭 음수면 보고서에 명시)
6. **사용자 단계별 승인**: Step 0~4 종료 시점마다 결과 보고 + 다음 진행 승인 받음

---

## 학습계획 md 업데이트 (Phase 1 진입 메모)

`김재천/Study/00_학습계획.md` 하단 "학습 메모 / 막힌 부분 기록" 섹션에 다음 한 줄 추가:
> 2026-04-24: 학습자료 생성 일시 중단, Phase 1 (시계열_Test/Phase1_LSTM) 실전 구축 진입. 입력은 univariate(log-return) — 자기상관 약함 가설 정량 검증 목적.

---

## 주의사항·리스크

1. **누수 방지 최우선** — 별도 §"⚠️ 데이터 누수 방지" 섹션의 4계층 방어선·검증 의무 모두 준수
2. **Scaler 필요성**: 수익률은 이미 정상화 시계열 → Scaler 생략 가능. EDA 분포 확인 후 결정 (StandardScaler 후보)
3. **R²_OOS 음수 가능성**: 수익률 예측은 본질적 어려움. 음수가 나오면 모델이 "0 예측"보다도 못함을 의미 — 즉시 보고하고 모델 재검토. Hit Rate 0.5 근처면 동전 던지기 수준
4. **설정 B 과적합 신호**: train loss ↓ + val loss ↑ → hidden 32, dropout 0.4까지 축소 검토
5. **결과물 격리**: 모든 산출물은 `시계열_Test/Phase1_LSTM/` 내부에 한정 (CLAUDE.md 김재천 규칙은 2026-04-24 삭제)
6. **변수명 영어 강제**: 컬럼은 한글 유지 가능하나 변수명은 영어 (CLAUDE.md)
7. **대화형 진행**: 각 Step 종료 시 사용자 보고 + 승인 (CLAUDE.md)
8. **협업 모듈화 (2026-04-24 변경)**: 재사용 함수·클래스는 `scripts/*.py`로 분리. 노트북은 흐름·시각화·해석 담당. 인터페이스 변경 시 `재천_WORKLOG.md` 기록
9. **마크다운·주석 풍부**: 각 셀 직전 마크다운 + 인라인 주석 + docstring 의무

---

## 다음 단계 (이 plan 승인 후)

1. 폴더 구조 생성 ✅ (2026-04-24 완료)
2. Step 0 환경 노트북·`scripts/setup.py` ✅
3. Step 1 데이터 다운로드·EDA ✅
4. Step 2-data `scripts/dataset.py` (LSTMDataset / make_sequences / walk_forward_folds / build_fold_datasets + target_series) + `02_tensor_dataset.ipynb` ✅ (다른 팀원)
5. Step 2-target `scripts/targets.py` (`build_daily_target_21d` · `verify_no_leakage` · `build_leaky_target_for_test` 다른 팀원 + `build_monthly_target_1m` 재천) ✅
6. Step 2-exec `scripts/models.py` · `scripts/train.py` · `scripts/metrics.py` (재천 신규, 단위 검증 4+4+16 PASS) ✅
7. Step 2-doc `scripts_정의서.md` (재천 신규, 모듈 API 정의서) ✅
8. Step 2-run `02_setting_A_daily21.ipynb` §1~§6 실행·검증 완료 (다른 팀원), §7~§9 활성화 대기 🟡
9. Step 3 설정 B — 대기 (targets.py 에 `build_monthly_target_1m` 준비됨)
10. Step 4 비교 보고 — 대기
