# Phase 1 — LSTM 단독 베이스라인 작업 일지 (WORKLOG)

> **목적**: Phase 1 구축 과정의 모든 작업·결정·판단 근거를 시간순으로 기록합니다.
> 사용자가 언제든 진행 상황과 의사결정 흐름을 추적할 수 있게 하기 위함입니다.
>
> **위치**: `시계열_Test/Phase1_LSTM/재천_WORKLOG.md`
> **기록 원칙**:
> 1. 모든 의사결정은 (a) 결정 내용, (b) 선택지, (c) 판단 근거, (d) 결정 주체(사용자 / 어시스턴트 제안)를 함께 기록
> 2. 사용자 피드백·지시는 원문 또는 요지를 보존
> 3. 작업 완료 시 산출물 파일 경로와 핵심 변경 사항 기록

---

## 2026-04-24 — Phase 1 착수

### 14:?? · 프로젝트 진입 결정
- **사용자 지시**: 학습 자료 생성 일시 중단, 시계열 모델 실전 평가 6단계 시작
- **6단계 모델 평가 로드맵**:
  1. LSTM × ETF × 월별(1)
  2. LSTM × ETF × 일별(21)
  3. GRU × ETF × 월별(1)
  4. GRU × ETF × 일별(21)
  5. CEEMDAN+LSTM × 월별(1)
  6. CEEMDAN+LSTM × 일별(21)
- **참고 자료**: 사용자 첨부 이미지(LSTM→Black-Litterman Q 벡터 추출 로드맵), `Study/00_학습계획.md`
- **현 단계**: Phase 1 (LSTM 단독 베이스라인) 구축

### 14:?? · 사용자 응답 기반 의사결정 (AskUserQuestion 1회차)

| 항목 | 사용자 결정 | 판단 근거 |
|---|---|---|
| **자산군** | SPY + QQQ 두 개만 | 목적은 BL Q 형성이 아닌 "수익률 예측 자체의 성능 확인" |
| **데이터 소스** | yfinance 신규 다운로드 | 일관된 기간·형식 확보 |
| **결과물 폴더** | `시계열_Test/Phase1_LSTM/` (2026-04-24 시계열_Test가 finance_project 직하위로 이동) | 학습/실전 폴더 분리 |
| **코드 참고 수준** | 패턴/원리만 참고, 코드는 새로 작성 | 사용자 명시 지시 (재사용 X) |

### 14:?? · 추가 의사결정 (AskUserQuestion 2회차)

| 항목 | 사용자 결정 | 판단 근거 |
|---|---|---|
| **데이터 기간** | 다운로드 2009-01-01 ~ 2026-03-31 / 분석 2010-01-01 ~ 2025-12-31 | lookback 채움용 1년 워밍업 확보 |
| **설정 B Walk-Forward** | 사용 가능한 기법 모두 제시 후 선택 | 구현 직전에 옵션 비교 후 결정 |
| **입력 피처** | Univariate (log-return 단일) | 가설 검증("자기상관 약함 → 성능 부족")이 우선 |

### 14:?? · 평가 지표 재정의 (사용자 지시)
- **변경 전**: R² > 0.3 (이미지 명세)
- **변경 후 (사용자 지시)**:
  - **Hit Rate > 0.55** (방향 적중률, 동전 던지기 0.5 대비 의미있는 수준)
  - **R²_OOS > 0** (Campbell & Thompson 2008 정의, 0 예측 baseline 대비 개선)
- **판단 근거**: 수익률 예측의 본질적 어려움 고려 시 R² 0.3은 비현실적 임계. Hit Rate + R²_OOS는 학술 표준이며 더 합리적

### 14:?? · 노트북 구조 원칙 (사용자 피드백 반영)
- **구조 변경**: scripts/ 폴더 폐기 → 모든 함수·클래스를 노트북 셀로 자기완결적 정의
- **마크다운 의무화**: 각 셀 직전 마크다운 셀로 (a) 무엇을 (b) 왜 (c) 트레이드오프 (d) 함정 명시
- **누수 방지 최우선**: 별도 §"⚠️ 데이터 누수 방지" 섹션으로 격상, 4계층 방어선 + 3종 검증 의무
- **WORKLOG 도입 (이 파일)**: 모든 작업·판단 기록 → 사용자 추적 가능

### 14:?? · 최종 plan 승인
- 위치: `C:/Users/gorhk/.claude/plans/c-users-gorhk-finance-project-study-00-m-frolicking-iverson.md`
- ExitPlanMode 후 사용자 승인 완료

---

## 작업 진행 상황

### Step 0. 환경 노트북 (`00_setup_and_utils.ipynb`)
- **상태**: ✅ 완료 (2026-04-24)
- **산출물**: `00_setup_and_utils.ipynb` (셀 구성: 마크다운 6 + 코드 5)
- **포함 기능**:
  - §1 `setup_korean_font()` — Windows/macOS/Linux 분기. Windows는 Malgun Gothic
  - §2 `fix_seed(seed=42)` — Python·NumPy·PyTorch 시드 + `torch.use_deterministic_algorithms(True, warn_only=True)`
  - §3 경로 상수 — `BASE_DIR`, `RESULTS_DIR`, `RAW_DATA_DIR`, `SETTING_A_DIR`, `SETTING_B_DIR` (자동 생성)
  - §4 공통 import — pandas/numpy/matplotlib + 표시 옵션
  - §5 정상 로드 출력 박스
- **사용 방법**: 다른 노트북에서 첫 코드 셀에 `%run ./00_setup_and_utils.ipynb`
- **판단 근거**:
  - **Path.cwd() 기반 경로**: Jupyter는 노트북 위치를 cwd로 잡으므로 절대 경로 하드코딩 회피
  - **`warn_only=True`**: 일부 PyTorch 연산이 결정적 모드에서 에러 대신 경고만 — 학습 멈춤 방지
  - **CUBLAS_WORKSPACE_CONFIG**: CUDA 사용 시 PyTorch 결정성 요구 환경변수
  - **소수점 6자리 표시**: 수익률 0.000xxx 단위 가시성 확보

### Step 1. 데이터 수집 + EDA (`01_data_download_and_eda.ipynb`)
- **상태**: ✅ 완료 (2026-04-24, Run All 검증 완료)
- **산출물**:
  - `01_data_download_and_eda.ipynb` (셀 16개, 출력 포함 301KB)
  - `results/raw_data/SPY.csv` (4,336행, 478KB)
  - `results/raw_data/QQQ.csv` (4,336행, 481KB)
- **다운로드 결과**
  - 기간: 2009-01-02 ~ 2026-03-30 (요청 2009-01-01~2026-03-31, 영업일 자동 보정)
  - 컬럼: Adj Close, Close, High, Low, Open, Volume

#### 분석 기간(2016-01-01~2025-12-31) 핵심 통계

| 항목 | SPY | QQQ |
|---|---:|---:|
| 일별 샘플 수 | 2,514 | 2,514 |
| 평균 log-return (일별) | 0.000546 (≈연 13.7%) | 0.000706 (≈연 17.8%) |
| 표준편차 (일별) | 0.011370 | 0.014101 |
| 왜도(skew) | -0.604 | -0.403 |
| 첨도(excess kurt) | **15.137** | **7.891** |
| 결측치 | 0 | 0 |

→ 둘 다 음의 왜도 + 매우 높은 첨도 = 강한 fat-tail. SPY가 더 극단적.

#### 이상치 (|log_ret| > 5σ, 분석 기간 내)

| 항목 | SPY | QQQ |
|---|---:|---:|
| >3σ 일수 | 34일 | 33일 |
| >5σ 일수 | **9일** | **8일** |
| 주요 사건 | 2020-03-13/24 (COVID), 2020-04-06, 2020-06-11, **2025-04-09 (+8.78σ)** | 2020-03-13/17/24, 2022-11-10, **2025-04-09 (+8.04σ)** |

→ **모두 실제 시장 사건**. CLAUDE.md 보수적 원칙에 따라 **제거하지 않고 그대로 사용**.

#### ACF lag 1~10 (사용자 가설 정량 검증) ⭐

| 지표 | SPY | QQQ |
|---|---:|---:|
| ACF lag 1 | **-0.1329** | **-0.1229** |
| ACF lag 2 | +0.0854 | +0.0484 |
| ACF lag 3 | -0.0250 | -0.0309 |
| 95% CI (±) | 0.0391 | 0.0391 |
| 유의 lag (|ACF| > CI) | [1, 2, 4, 5, 6, 7, 8, 9] | [1, 2, 6, 7, 8, 9, 10] |

#### 사용자 가설 평가
- **가설**: "수익률의 자기상관이 극단적으로 작아 univariate LSTM으로는 성능 부족 예상"
- **정량 결과**:
  - lag 1에서 약 -0.13 정도의 **음의 자기상관** 존재 (평균회귀 신호)
  - **통계적으로 유의** (|ACF| > CI_95)
  - 그러나 **경제적 의미는 작음**: lag 1 ACF=-0.13 → 단순 회귀 R² ≈ 1.7%
  - lag 2 이후는 모두 |0.09| 이하로 더 약함
- **결론**: 가설은 정성적으로 옳음. LSTM이 lag 1 평균회귀 패턴 일부는 잡을 수 있으나 **R²_OOS > 0.05 수준이 현실적 상한**으로 추정. 부호 적중률(Hit Rate)이 0.55를 넘기 어려울 수 있음.
- **시사점**: Phase 1 결과 미달 시 추가 피처(거래량, VIX, 기술지표) 도입 정당화.

#### 추가 메모
- yfinance 1.3.0 설치 (Jupyter Python 3.10 환경에 직접 설치, MiniConda Python과 분리됨에 주의)
- yfinance auto_adjust=False로 OHLC와 Adj Close 모두 받음
- CSV 캐시 정책으로 재실행 시 다운로드 생략 (재현성 확보)

### Step 2. 설정 A 전체 (`02_setting_A_daily21.ipynb`)
- **상태**: ⏸ 대기 (Step 1 완료 후 사용자 승인 필요)

### Step 3. 설정 B 전체 (`03_setting_B_monthly.ipynb`)
- **상태**: ⏸ 대기

### Step 4. 비교 보고 (`04_compare_A_vs_B.ipynb`)
- **상태**: ⏸ 대기

---

## 의사결정 보류 항목

1. ~~**설정 B Walk-Forward 기법**~~ → **확정됨** (아래 2026-04-24 추가 지시 참고)
2. **Scaler 적용 여부**: Step 1 EDA에서 log-return 분포 확인 후 결정 (수익률은 이미 정상화 시계열이므로 생략 가능)
3. **결측 처리 방침·이상치 임계**: Step 1 EDA 결과 사용자 합의
4. **설정 B(월별) Walk-Forward 기간 매핑**: 이미지가 일별 기준(IS 231/purge 21/emb 21/OOS 21)이므로 월별은 비례 축소(IS 11개월/purge 1/emb 1/OOS 1, step 1개월)로 추정 — Step 3 진입 직전 사용자 최종 확인

---

## 2026-04-24 — 중요 지시 반영 (Walk-Forward 구조 + 분석 기간 수정)

### 사용자 지시 (이미지 + 텍스트)
1. **Walk-Forward 구조를 이미지와 동일하게 통일** — 설정 A·B 모두 적용
2. **분석 기간**: 2010-01-01~2025-12-31 → **2016-01-01~2025-12-31 (10년)**로 단축
3. **데이터 수집 기간은 그대로** (2009-01-01~2026-03-31)

### Walk-Forward 구조 확정 (이미지 기반, 일별 기준)

```
날짜 축 →
Fold 1: ├── IS (231일) ──┤ purge(21) │ emb(21) │ OOS(21) ┤
Fold 2: (21일 오른쪽 이동) ├── IS (231일) ──┤ purge │ emb │ OOS ┤
Fold 3: ...
... (약 79개 fold, 10년치)
```

| 구성 요소 | 길이 | 역할 |
|---|---|---|
| **IS** (In-Sample) | 231 영업일 (~11개월) | 학습 + 검증 (train/val) |
| **Purge** | 21 영업일 | t+21 타깃 누수 차단 (IS 끝과 OOS 시작 사이) |
| **Embargo** | 21 영업일 | fold 간 autocorrelation 차단 |
| **OOS** (Out-of-Sample) | 21 영업일 | test (성과 평가) |
| **Step** | 21 영업일 | Fold 간 오른쪽 이동 간격 (rolling sliding) |
| **Fold 수** | 약 79~100개 | 10년 분석 기간 기준 |

### 왜 Purge + Embargo를 동시에 쓰는가 (판단 근거)
- **Purge** (López de Prado 2018): IS 내 샘플의 **타깃 생성 구간**(t+1~t+21)이 OOS 첫 시점과 겹치는 것을 차단. 설정 A는 타깃이 21일 누적이므로 purge=21 필수.
- **Embargo**: IS 학습 직후의 autocorrelation 잔존 효과로 OOS 첫 구간이 부풀려지는 것을 방지. 21일 embargo는 설정 A의 horizon과 매칭.
- **학술 표준**: López de Prado, *Advances in Financial Machine Learning* (2018) §7. Purged K-Fold + Embargo는 금융 시계열 walk-forward 누수 방지의 사실상 표준.

### 분석 기간 단축의 영향
- 이전(2010~2025): 일별 ~4,000, 월별 ~190
- 현재(2016~2025): 일별 ~2,520, 월별 ~120
- **장점**: 최근 시장 체제에 집중, 2010년대 초반 저금리 체제의 bias 제거
- **주의**: 설정 B 월별 샘플 ~120으로 과적합 위험 더 커짐 → dropout/hidden 축소 강화 검토

### 설정 A·B 구조 통합 (변경)
- **이전**: 설정 A는 70/15/15 단일 split, 설정 B만 walk-forward
- **변경 후**: **A·B 모두 Rolling Walk-Forward + Purge + Embargo**
  - 설정 A: IS 231일 / purge 21 / emb 21 / OOS 21 / step 21 (이미지 직접 적용)
  - 설정 B: IS 11개월 / purge 1 / emb 1 / OOS 1 / step 1 (일별→월별 비례 축소 제안) — 사용자 확인 후 확정

### train/val 분리 (IS 내부)
- 각 fold의 IS 231일 안에서 train/val 재분할 필요 (EarlyStopping·LR scheduler용)
- 제안: IS 내부 80/20 시간순 분할 (train 184 + val 47일) + 내부 gap = seq_len
- fold마다 독립 학습 → best_val ckpt로 OOS 평가

### 설정 A 변경 요약
| 항목 | 이전 | 현재 |
|---|---|---|
| 분석 기간 | 2010-01-01~2025-12-31 | **2016-01-01~2025-12-31** |
| Split 방식 | 70/15/15 단일 | **Rolling Walk-Forward + Purge + Embargo** |
| Fold 수 | 1 | **약 79~100** |
| 메트릭 보고 | 단일 test 값 | **fold별 메트릭 + 평균±표준편차** |

### 코드 영향 범위
- `00_setup_and_utils.ipynb`: **변경 없음**
- `01_data_download_and_eda.ipynb`: 분석 기간 상수 수정 (2016-01-01)
- `02_setting_A_daily21.ipynb`: Walk-Forward 루프 구조로 전면 재설계
- `03_setting_B_monthly.ipynb`: 월별 Walk-Forward 파라미터 사용
- `04_compare_A_vs_B.ipynb`: fold별 메트릭 분포 시각화 추가

---

## 2026-04-24 — 협업 구조 전환 (`scripts/*.py` 사용 가능)

### 사용자 지시
> "다른 팀원과 공유하여 작업하게 되었음. 파일 구조에 py 확장자 사용 가능으로 다시 변경."

### 변경 결정
- 이전: 노트북 자기완결 — 모든 함수·클래스를 노트북 셀로 정의 (.py 모듈 폴더 금지)
- 변경: **노트북 + `scripts/*.py` 협업 구조** — 재사용 함수·클래스는 `.py` 모듈로 분리, 노트북은 흐름·시각화·해석 담당

### 모듈 분리 계획 (Step 2 작성 시 점진적 추출)
| `scripts/` 모듈 | 책임 |
|---|---|
| `setup.py` | 한글 폰트·시드·경로 상수 (00 노트북과 동일 로직) |
| `data_io.py` | CSV 로드, 분석 기간 절단, 워밍업 처리 |
| `targets.py` | 21일 누적 / 월별 1개월 타깃 + 누수 검증 |
| `cv_walkforward.py` | Walk-Forward fold 생성기 (IS/purge/emb/OOS/step) |
| `dataset.py` | SequenceDataset (PyTorch) |
| `models.py` | LSTMRegressor |
| `train.py` | 학습 루프 (Huber, EarlyStop, scheduler, ckpt) |
| `metrics.py` | Hit Rate, R²_OOS, MAE, RMSE, baseline 비교 |
| `plot_utils.py` | 한글 폰트 + 표준 플롯 |

### 기존 노트북(00, 01)의 처리
- **현행 유지** (재실행 결과 보존). 함수가 단순하므로 즉시 리팩터링 불필요.
- Step 2 진행하면서 자연스럽게 공통 패턴이 생기면 그 시점에 `scripts/`로 추출.
- 사용자 요청 시 00·01도 즉시 리팩터링 가능.

### 협업 안정성 원칙 (신규)
- 모든 public 함수: type hints + Numpy-style docstring 필수
- `scripts/X.py` 인터페이스 변경 시 WORKLOG에 변경 사항 기록
- 노트북 import는 절대 경로 (`from scripts.targets import ...`)

### 협업 진입점 추가
- **`README.md` 신규 작성**: 실행 순서, 의존 그래프, 환경 설치 안내 (다른 팀원이 첫 진입 시 참고)

### 00 노트북 리팩터링 — `scripts/setup.py` 분리 (2026-04-24, 같은 날 후속)

**사용자 추가 지시**: "00과 같이 프로젝트 전반 기능에 해당하는 부분만 py로 구성"

**적용 범위**:
- ✅ `00_setup_and_utils.ipynb` (환경·시드·폰트·경로 — Phase 1 전반에 공통) → `scripts/setup.py`로 함수·상수 이전
- ❌ `01_data_download_and_eda.ipynb` (데이터 다운로드·EDA 흐름) → 그대로 유지 (EDA 노트북 흐름의 일부)

**`scripts/setup.py` 신규 작성**:
- 단일 진실원(single source of truth)
- public 인터페이스: `setup_korean_font()`, `fix_seed(seed)`, `apply_display_defaults()`, `ensure_result_dirs()`, `bootstrap()`
- public 상수: `SEED=42`, `BASE_DIR`, `RESULTS_DIR`, `RAW_DATA_DIR`, `SETTING_A_DIR`, `SETTING_B_DIR`
- BASE_DIR은 `Path(__file__).resolve().parent.parent`로 import 위치 무관하게 안정적 산출

**00 노트북 변경**:
- 마크다운 셀(교육·설명 가치): **보존**
- 코드 셀: `from scripts.setup import ...` 후 함수 호출만
- 셀 추가: §0에서 `sys.path` 등록 (Phase1_LSTM 디렉토리를 import path에 추가)
- 노트북 크기: 8.7KB → 11.4KB (마크다운 보강), 동작 동일

**검증**:
- 00 노트북 단독 Run All ✅
- 01 노트북 Run All (`%run ./00_setup_and_utils.ipynb` 호출) ✅
- 데이터 캐시 정상 작동 (CSV 재다운로드 안 함)

---

## 2026-04-24 — 폴더 이동 + 협업 개인 일지 prefix 적용

### 사용자 지시
1. `Phase1_LSTM` 폴더를 `김재천/시계열_Test/` 에서 **`finance_project/시계열_Test/` 직하위**로 이동 (다른 팀원과 공유 작업)
2. CLAUDE.md 의 "모든 작업물은 김재천 내에서만 작업·생성" 규칙 **삭제**
3. `WORKLOG.md` → **`재천_WORKLOG.md`** 로 이름 변경 (작성자별 prefix 도입)
4. 모든 문서·코드의 경로 표기 갱신 (단, 김재천 폴더에 남아있는 학습 자료 `김재천/Study/...` 는 실제 위치 그대로 유지)

### 변경 적용 파일
| 파일 | 변경 내용 |
|---|---|
| `C:\Users\gorhk\최종 프로젝트\finance_project\CLAUDE.md` | 김재천 폴더 작업 규칙 1줄 삭제 |
| `시계열_Test/Phase1_LSTM/WORKLOG.md` | → `재천_WORKLOG.md` 로 이름 변경 |
| `재천_WORKLOG.md` (이 파일) | 위치·결과물 폴더·파일명 표기 3곳 갱신 + 본 섹션 추가 |
| `README.md` | WORKLOG 링크 6곳 → `재천_WORKLOG.md`, 경로 표기 2곳 (`김재천/시계열_Test` → `시계열_Test`, cd 명령어), 협업 prefix 안내 추가 |
| `scripts/__init__.py` | docstring 안 `WORKLOG.md` → `재천_WORKLOG.md` |
| `00_setup_and_utils.ipynb` | §3 마크다운 셀 마지막 줄(김재천 폴더 안내) 삭제 |
| `01_data_download_and_eda.ipynb` | 재실행으로 출력 셀의 BASE_DIR 등이 새 경로(`시계열_Test/Phase1_LSTM`)로 자동 갱신 |
| `.claude/plans/c-users-gorhk-finance-project-study-00-m-frolicking-iverson.md` | 결과물 위치 표기·폴더 트리·주의사항 5번 갱신 |

### 보존된 것 (의도)
- `README.md` line 143 `김재천/Study/00_학습계획.md` — 학습 자료는 실제로 김재천 폴더에 그대로 있으므로 정상 참조 유지
- 데이터 캐시(`results/raw_data/SPY.csv`, `QQQ.csv`) 그대로 보존
- BASE_DIR 자동 인식: `Path(__file__)` 기반 설계 덕분에 폴더 이동에도 코드 수정 불필요

### 협업 일지 규칙 (신설)
- 작성자는 본인 이름 prefix 로 별도 일지 파일을 둠 (예: `재천_WORKLOG.md`, `<이름>_WORKLOG.md`)
- 모든 결정·판단·인터페이스 변경은 본인 일지에 시간순 누적
- 공통 문서(README, plan)는 모든 팀원이 협의 후 갱신

---

## 2026-04-24 — PLAN.md 팀 공유 사본 생성 + plan 본문 `.py` 사용 방향 갱신

### 사용자 지시
1. "지금 claude의 플랜 md파일을 복사해서 우리 팀원 모두가 확인할 수 있게 공유 폴더 내에 md파일로 만들어서 넣어줘"
2. "일단 플랜 파일도 py파일 사용하는 것으로 내용 수정하고"

### 작업 내용
- **PLAN.md 신규 작성**: Claude Code plan 파일(`.claude/plans/...`)을 `시계열_Test/Phase1_LSTM/PLAN.md` 로 복사. 상단에 진실원 위치·동기화 정책 헤더 추가.
- **진실원 plan + PLAN.md 동시 수정**: Step 2 본문이 "셀 안에서 직접 정의" 시절 표현이었음 → **`scripts/*.py` 에 정의 + 노트북에서 import** 로 변경.
- **갱신된 절** (Step 2 / 같은 변경 Step 3에도 일관 반영):
  - §2 데이터 로드: `scripts.data_io.load_ticker_csv` import
  - §3 타깃 생성: `scripts.targets.build_daily_target_21d` (B는 `build_monthly_target_1m`)
  - §4 누수 검증: `scripts.targets.verify_no_leakage` + `build_leaky_target_for_test`
  - §5 SequenceDataset: `scripts.dataset.SequenceDataset`
  - §6 Walk-Forward: `scripts.cv_walkforward.generate_folds(n_total, is_len, purge, embargo, oos_len, step)`
  - §7 LSTMRegressor: `scripts.models.LSTMRegressor`
  - §8 학습 루프: `scripts.train.train_one_fold(model, train_loader, val_loader, **hp)`
  - §9 메트릭·시각화: `scripts.metrics.{hit_rate, r2_oos, baseline_metrics}` + `scripts.plot_utils.{plot_learning_curve, ...}`
- 설정 A·B 가 같은 모듈을 공유 → 비교 공정성 자연 확보 (Step 3 본문에 명시)

### README 동기화
- 첫 안내문에 PLAN.md → 재천_WORKLOG.md → 노트북 순서로 읽기 가이드 추가
- 폴더 트리에 `PLAN.md` 한 줄 추가
- §6 참고 문서에 "plan 팀 공유 사본" 링크 추가

### 동기화 정책
- 진실원: `C:\Users\gorhk\.claude\plans\c-users-gorhk-finance-project-study-00-m-frolicking-iverson.md`
- 사본: `시계열_Test/Phase1_LSTM/PLAN.md`
- Claude가 진실원을 갱신하면 사본도 함께 갱신 (수동 동기화). 팀원은 사본만 보면 됨.

---

## 2026-04-24 — 학습자료 주의사항 전수 정리

### 사용자 지시
> "Study 파일 내의 md파일 형식의 학습자료의 각 단계별 주의사항 내용을 전수조사하여 주의사항 md파일로 생성해줘"

### 작업 내용
- `김재천/Study/` 내 모든 md 파일 (week1~week4 + `00_학습계획.md`) 에서 "주의·함정·금지·누수·warning·⚠️" 류 내용을 Explore 에이전트로 1차 추출.
- 38개 항목을 8개 카테고리로 분류하여 [학습자료_주의사항.md](학습자료_주의사항.md) 신규 작성.
- 카테고리: (1) 데이터 누수 방지, (2) 시계열 데이터 처리, (3) 학습·PyTorch, (4) 과적합·정규화, (5) 평가·메트릭, (6) 통계 검정·해석, (7) 코드 작성 일반, (8) 기타 (도메인).
- 하단에 "Phase 1 직접 적용 우선순위 체크리스트" 첨부.

### 한계·TODO 명시
- **week3·week4의 일부 md 세부 항목 누락**: PyTorch 학습 루프 흔한 실수 10가지, Variational Dropout PyTorch 함정, RNN/LSTM/GRU 게이트 초기화 트릭, BiLSTM Walk-Forward 양방향 누수 등.
- Step 2 (`02_setting_A_daily21.ipynb`) 진입 직전 해당 원본 md 직접 정독으로 보강 권장.
- 본 정리는 1차 자동 추출 결과 — 원문 인용 정확성 검증 필요.

### 파일 위치
- `시계열_Test/Phase1_LSTM/학습자료_주의사항.md` (협업 폴더)
- 노트북 작성 시 바로 참조 가능한 위치에 배치

### 협업 규약 추가
- 본 문서는 원본 Study md 갱신 시 **수동 동기화 필요** (자동 갱신 안 됨)
- 팀원이 본 문서의 항목을 검증하거나 보강하려면 본인 `<이름>_WORKLOG.md` 에 제안 기록 후 협의

---

## 산출물 인덱스 (생성 순)

| 파일 | 종류 | 설명 |
|---|---|---|
| `재천_WORKLOG.md` | 문서 | 본 작업 일지 (2026-04-24 협업용으로 `WORKLOG.md` → `재천_WORKLOG.md` 이름 변경) |
| `00_setup_and_utils.ipynb` | 노트북 | 환경 설정·시드·경로·폰트·import |

(이후 진행 시 추가)
