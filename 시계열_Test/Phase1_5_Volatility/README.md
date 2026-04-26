# Phase 1.5 — 변동성 예측 분기 (LSTM)

> **협업 진입점 문서**. 처음 합류한 팀원은 이 README → [PLAN.md](PLAN.md) → [재천_WORKLOG.md](재천_WORKLOG.md) → 노트북 순으로 읽으십시오.
>
> - **PLAN.md**: 전체 구현 계획서 (Claude Code plan 파일의 팀 공유 사본, 진실원과 동기화)
> - **재천_WORKLOG.md**: 작업·판단 일지 (시간순 누적)
> - **노트북**: 실제 분석·학습 흐름

## 1. 분기 위치 및 목적

- **상위 프로젝트**: COL-BL (Su et al. 2026 ESWA 295 논문 재현)
- **현 단계**: **Phase 1.5** — Phase 1 LSTM 의 변형 분기
- **Phase 1 과의 관계**: Phase 1 (`../Phase1_LSTM/`) 결과는 그대로 보존하고, 본 분기에서는 **타깃을 누적수익률 → 실현변동성** 으로 교체한 새 LSTM 을 학습·평가
- **유일한 목적**: **"변동성 예측이 가능한가?"** 단일 질문에 명확한 답을 내는 것

### Phase 1 ⇄ Phase 1.5 비교

| 항목 | Phase 1 | **Phase 1.5** |
|---|---|---|
| 폴더 | `Phase1_LSTM/` (보존, 변경 금지) | `Phase1_5_Volatility/` (본 폴더) |
| 입력 | `log_ret` univariate | `log_ret²` univariate |
| 타깃 | 21일 누적 log-return | 21일 forward log-realized-volatility |
| 손실 | Huber(δ=0.01) | MSE |
| 1차 지표 | Hit Rate · R²_OOS | RMSE on Log-RV · QLIKE · R²_train_mean |
| 베이스라인 | zero / previous / train_mean | **HAR-RV** / EWMA / Naive / Train-Mean |
| 모델·Walk-Forward | LSTM(1,32,1,0.3), 105 fold | **동일** (비교 공정성) |

### 본 단계 평가 비대상 (의도적 분리)

다음은 Phase 1.5 의 PASS/FAIL 판정에 영향을 주지 않습니다. 추후 별도 단계로 미룸:

- 포트폴리오 구축 (Mean-Variance / Black-Litterman)
- 벤치마크 대비 alpha / Sharpe / drawdown
- BL 의 Q/Ω 입력 통합

---

## 2. 핵심 의사결정 요약

자세한 근거는 [PLAN.md](PLAN.md) §2~§7 및 [재천_WORKLOG.md](재천_WORKLOG.md) 참조.

| 항목 | 결정 |
|---|---|
| 자산군 | SPY, QQQ (Phase 1 과 동일) |
| 데이터 | `results/raw_data/{SPY,QQQ}.csv` (Phase 1 결과 복사, 재다운로드 X) |
| 분석 기간 | 2016-01-01 ~ 2025-12-31 (10년) |
| **타깃** | `target = log( log_ret.rolling(21).std() ).shift(-21)` |
| **입력** | `log_ret²` (단일 채널) |
| **모델** | `LSTMRegressor(input_size=1, hidden_size=32, num_layers=1, dropout=0.3)` |
| **손실** | MSE |
| **옵티마** | AdamW(lr=1e-3, weight_decay=1e-3) + ReduceLROnPlateau(patience=3, factor=0.5) |
| **Walk-Forward** | IS 231 / Purge 21 / Embargo 21 / OOS 21 / Step 21 → 105 fold |
| **베이스라인** | HAR-RV (Corsi 2009) · EWMA(λ=0.94) · Naive · Train-Mean |
| **PASS 조건 (3개 모두 충족)** | (1) `LSTM RMSE < HAR-RV RMSE` (2) `R²_train_mean > 0` (3) `pred_std/true_std > 0.5` |

---

## 3. 폴더 구조

```
Phase1_5_Volatility/
├── README.md                              ← 이 문서
├── PLAN.md                                ← ⭐ 전체 구현 계획 (Claude plan 파일 팀 공유 사본)
├── 재천_WORKLOG.md                         ← 작업·판단 일지
│
├── 00_setup_and_utils.ipynb               ← Phase 1 동일 (복사) — 환경 부트스트랩
├── 01_volatility_eda.ipynb                ← 신규 — RV 분포·ACF·정상성·체제 진단 (예정)
├── 02_volatility_lstm.ipynb               ← 신규 — 변동성 LSTM 본 실험 105 fold × 2 ticker (예정)
├── 03_baselines_and_compare.ipynb         ← 신규 — HAR-RV/EWMA/Naive vs LSTM (예정)
│
├── scripts/                               ← 재사용 모듈
│   ├── __init__.py                        ← 패키지 마커
│   ├── setup.py                           ← Phase 1 복사 (한글 폰트·시드·경로)
│   ├── dataset.py                         ← Phase 1 복사 (LSTMDataset + Walk-Forward)
│   ├── models.py                          ← Phase 1 복사 (LSTMRegressor)
│   ├── train.py                           ← Phase 1 복사 (학습 루프, loss_type='mse' 옵션 추가 예정)
│   ├── targets_volatility.py              ← ⭐ 신규 — Log-RV 빌더 + 누수 검증
│   ├── metrics_volatility.py              ← ⭐ 신규 — RMSE/QLIKE/R²_train_mean/MZ
│   └── baselines_volatility.py            ← ⭐ 신규 — HAR-RV/EWMA/Naive
│
└── results/                               ← 노트북 산출물만 (코드 없음)
    ├── raw_data/                          ← SPY.csv, QQQ.csv (Phase 1 사본)
    └── volatility_lstm/{SPY,QQQ}/         ← metrics.json, *.png (예정)
```

---

## 4. 실행 순서

### 4.1 환경 설치 (OS 호환)

본 분기는 **Windows / macOS / Linux** 모두에서 실행 가능하도록 설계되었습니다.

```bash
# 1. Jupyter 가 사용하는 Python 확인 (모든 OS 공통)
python -c "import sys; print(sys.executable)"

# 2. 의존성 설치 (모든 OS 공통)
python -m pip install yfinance statsmodels scipy torch pandas numpy matplotlib jupyter nbconvert

# 3. (Linux 전용) 한글 폰트 패키지 추가 — matplotlib 한글 깨짐 방지
python -m pip install koreanize-matplotlib --break-system-packages
```

> ⚠️ **함정 주의**: `pip install` 이 다른 Python 환경(예: MiniConda)으로 잘못 가는 경우가 있습니다. 반드시 Jupyter 가 쓰는 Python 에서 `python -m pip` 로 설치하십시오.

> 한글 폰트는 `scripts/setup.py` 의 `setup_korean_font()` 가 OS 자동 분기 처리합니다 (Windows: Malgun Gothic / macOS: AppleGothic / Linux: NanumGothic via koreanize-matplotlib).

### 4.2 노트북 실행 순서

```
00_setup_and_utils.ipynb              ← 환경 부트스트랩 (% run 으로 호출됨)
        ↓
01_volatility_eda.ipynb               ← RV 분포·ACF·정상성 검정 (예정)
        ↓
02_volatility_lstm.ipynb              ← 변동성 LSTM 학습 105 fold × 2 ticker (예정)
        ↓
03_baselines_and_compare.ipynb        ← HAR-RV/EWMA/Naive 비교 + 관문 판정 (예정)
```

### 4.3 노트북 자동 실행 (CLI)

```bash
# Bash (Linux/macOS/Git Bash on Windows)
cd "시계열_Test/Phase1_5_Volatility"
jupyter nbconvert --to notebook --execute --inplace 01_volatility_eda.ipynb \
    --ExecutePreprocessor.timeout=600

# PowerShell (Windows)
Set-Location "시계열_Test\Phase1_5_Volatility"
jupyter nbconvert --to notebook --execute --inplace 01_volatility_eda.ipynb `
    --ExecutePreprocessor.timeout=600
```

---

## 5. 협업 규약

### 5.1 모듈 변경
- `scripts/X.py` 의 public 함수 시그니처를 변경할 때는 **반드시 [재천_WORKLOG.md](재천_WORKLOG.md) 에 변경 이유·이전/이후 인터페이스 기록**.
- 모든 public 함수: type hints + Numpy-style docstring.

### 5.2 Phase 1 격리 원칙 (본 분기 특수)
- `Phase1_LSTM/` 의 어떤 파일도 본 분기 작업으로 변경하지 않습니다.
- `Phase1_5_Volatility/scripts/{setup,dataset,models,train}.py` 는 Phase 1 의 사본이며, 사본을 수정하더라도 Phase 1 원본은 무관.
- 본 분기 결과는 `Phase1_5_Volatility/results/` 안에만 저장.

### 5.3 누수 방지 (변동성 특수 함정)
- `shift(-21)` 부호 누락: §4 검증 셀 (assert + 5행 표) 필수.
- `np.log(0)` 또는 `np.log(NaN)`: `rolling` 직후 NaN drop, `assert (rv > 0).all()`.
- HAR-RV 의 monthly(22) lookback: fold 외부 참조 차단 (`train_idx` 한정 슬라이싱).
- EWMA seed: train 의 마지막 EWMA 값으로 고정.
- ddof: pandas `rolling().std()` 기본 1 vs numpy 기본 0 — `assert` 시 `ddof=1` 명시 통일.

### 5.4 작업 기록
- 모든 결정·판단은 [재천_WORKLOG.md](재천_WORKLOG.md) 에 시간순 누적.
- 다른 팀원이 본 분기에 합류 시 본인 prefix(`<이름>_WORKLOG.md`)로 별도 일지 가능.

### 5.5 코드 스타일 (CLAUDE.md 준수)
- 변수명은 영어 (한글 컬럼명은 유지 가능). 한글 변수명 금지.
- 시각화 한글 폰트 설정 필수 (`scripts.setup.setup_korean_font()` 또는 `00` 노트북 호출).
- 어려운 개념은 마크다운 셀에 충분히 설명.
- 데이터 drop 보수적 기준.

---

## 6. 참고 문서

- **상위 plan (진실원)**: `C:\Users\gorhk\.claude\plans\sharded-mapping-puffin.md`
- **plan 팀 공유 사본**: [PLAN.md](PLAN.md) (이 폴더 내, 진실원과 동기화)
- **Phase 1 자산** (참조 전용, 변경 금지):
  - [../Phase1_LSTM/PLAN.md](../Phase1_LSTM/PLAN.md)
  - [../Phase1_LSTM/scripts_정의서.md](../Phase1_LSTM/scripts_정의서.md)
  - [../Phase1_LSTM/02_setting_A_daily21.ipynb](../Phase1_LSTM/02_setting_A_daily21.ipynb) (셀 골격 템플릿)
- **상위 학습계획**: `김재천/Study/00_학습계획.md`
- **학습자료 주의사항**: [../학습자료_주의사항.md](../학습자료_주의사항.md)
- **핵심 학술 인용**:
  - Corsi, F. (2009). *A simple approximate long-memory model of realized volatility.* **Journal of Financial Econometrics, 7(2), 174–196.** (HAR-RV)
  - Patton, A. J. (2011). *Volatility forecast comparison using imperfect volatility proxies.* **Journal of Econometrics, 160(1), 246–256.** (QLIKE)
  - Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). *Modeling and forecasting realized volatility.* **Econometrica, 71(2), 579–625.** (RV 정의)

---

## 7. 진행 상태 (2026-04-26 시점)

| Step | 산출물 | 담당 | 상태 |
|---|---|---|---|
| Step 0 | 폴더 구조 + 4 .py + 1 .ipynb + 2 csv 복사 + 신규 문서 3건 | 재천 | ✅ 완료 |
| Step 1 | `01_volatility_eda.ipynb` (§1~§9) | 재천 | ⏸ 대기 (사용자 Step 0 승인 필요) |
| Step 2 | `targets_volatility.py` · `metrics_volatility.py` · `baselines_volatility.py` + 단위 테스트 | 재천 | ⏸ 대기 |
| Step 3 | `02_volatility_lstm.ipynb` 105 fold × 2 ticker 학습 | 재천 | ⏸ 대기 |
| Step 4 | `03_baselines_and_compare.ipynb` 관문 판정 | 재천 | ⏸ 대기 |

문의는 [재천_WORKLOG.md](재천_WORKLOG.md) 또는 작성자(gorhkdwj@gmail.com)에게 전달 부탁드립니다.
