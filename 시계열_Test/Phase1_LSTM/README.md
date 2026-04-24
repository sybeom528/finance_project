# Phase 1 — LSTM 단독 베이스라인

> **협업 진입점 문서**. 처음 합류한 팀원은 이 README → 재천_WORKLOG → 노트북 순으로 읽으십시오.

## 1. 프로젝트 위치 및 목적

- **상위 프로젝트**: COL-BL (Su et al. 2026 ESWA 295 논문의 CGL-BL 재현, Optuna 변형)
- **현 단계**: Phase 1 — 6단계 모델 평가 로드맵 중 첫 단계
- **목적**: BL Q 벡터 생성이 아닌 **LSTM 자체의 수익률 예측 성능 정량 평가**
- **핵심 가설**: 일별 수익률의 자기상관이 약하므로 univariate LSTM의 예측력에 본질적 한계 존재 → 이를 정량 확인

### 6단계 평가 로드맵

| # | 모델 | 시간 해상도 | 단계 |
|---|---|---|---|
| 1 | LSTM | 일별 (21영업일 후) | **Phase 1 (현재)** |
| 2 | LSTM | 월별 (1개월 후) | **Phase 1 (현재)** |
| 3 | GRU | 일별 (21) | Phase 2 |
| 4 | GRU | 월별 (1) | Phase 2 |
| 5 | CEEMDAN+LSTM | 일별 (21) | Phase 3 |
| 6 | CEEMDAN+LSTM | 월별 (1) | Phase 3 |

---

## 2. 핵심 의사결정 요약

자세한 근거는 [재천_WORKLOG.md](재천_WORKLOG.md) 참조.

| 항목 | 결정 |
|---|---|
| 자산군 | SPY, QQQ (각각 독립 학습·평가) |
| 데이터 소스 | yfinance |
| 다운로드 기간 | 2009-01-01 ~ 2026-03-31 |
| **분석 기간** | **2016-01-01 ~ 2025-12-31** (10년) |
| 입력 피처 | Univariate (log-return 단일) |
| 설정 A | 일별 + 21영업일 후 누적 log-return / seq_len=126 / hidden=128 / 2-layer LSTM |
| 설정 B | 월별 + 1개월 후 log-return / seq_len=24 / hidden=64 / 1-layer LSTM |
| **검증 방법** | **Rolling Walk-Forward + Purge + Embargo (López de Prado 2018)** |
| Walk-Forward (일별) | IS 231 / purge 21 / emb 21 / OOS 21 / step 21 |
| Walk-Forward (월별) | IS 11 / purge 1 / emb 1 / OOS 1 / step 1 (잠정 — Step 3 진입 시 최종 확정) |
| **1차 평가 지표** | **Hit Rate** (방향 적중률), **R²_OOS** (Campbell & Thompson 2008, 0 예측 baseline 대비) |
| 2차 지표 | MAE, RMSE, 표준 R², 직전값/평균 baseline 비교 |
| **관문 (Phase 2 진행 조건)** | **Hit Rate > 0.55** AND **R²_OOS > 0** (둘 다 충족 시 PASS) |

---

## 3. 폴더 구조

```
Phase1_LSTM/
├── README.md                              ← 이 문서
├── 재천_WORKLOG.md                         ← 작업·판단 일지 (모든 결정 누적, 작성자별 prefix)
│
├── 00_setup_and_utils.ipynb               ← 환경 노트북 (한글 폰트·시드·경로)
├── 01_data_download_and_eda.ipynb         ← yfinance 다운로드 + EDA + ACF
├── 02_setting_A_daily21.ipynb             ← 설정 A 흐름 (예정)
├── 03_setting_B_monthly.ipynb             ← 설정 B 흐름 (예정)
├── 04_compare_A_vs_B.ipynb                ← A·B 비교·최종 보고 (예정)
│
├── scripts/                               ← 재사용 모듈 (협업용)
│   ├── __init__.py
│   ├── setup.py                           ← ✅ 한글 폰트·시드·경로 (Phase 1 전반 공통)
│   └── (이후 Step 2~3에서 추가될 모듈은 본 README 갱신 시 등재)
│
└── results/                               ← 노트북 실행 결과만 저장 (코드 없음)
    ├── raw_data/                          ← SPY.csv, QQQ.csv (yfinance 원본)
    ├── setting_A/{SPY,QQQ}/               ← metrics.json, model.pt, *.png
    ├── setting_B/{SPY,QQQ}/               ← fold별 ckpt
    └── comparison_report.md               ← 04 노트북에서 자동 생성
```

---

## 4. 실행 순서

### 4.1 환경 설치

```bash
# Jupyter가 사용하는 Python(이 환경에서는 Python 3.10.8) 확인
python -c "import sys; print(sys.executable)"

# 같은 Python에 의존성 설치 (시스템 pip이 다른 환경일 수 있으므로 `python -m pip` 권장)
python -m pip install yfinance statsmodels scipy torch pandas numpy matplotlib jupyter nbconvert
```

> ⚠️ **함정 주의**: `pip install`이 다른 Python 환경(예: MiniConda)으로 잘못 가는 경우가 있습니다. 반드시 Jupyter가 쓰는 Python에서 `python -m pip`로 설치하십시오.

### 4.2 노트북 실행 순서

```
00_setup_and_utils.ipynb
        ↓ (% run으로 호출됨)
01_data_download_and_eda.ipynb   ← yfinance 다운로드 (1회만 필요, 이후 CSV 캐시)
        ↓ (results/raw_data/{SPY,QQQ}.csv 생성)
02_setting_A_daily21.ipynb       ← 설정 A: Walk-Forward 학습·평가 (예정)
        ↓
03_setting_B_monthly.ipynb       ← 설정 B (예정)
        ↓
04_compare_A_vs_B.ipynb          ← 비교·관문 판정·최종 보고 (예정)
```

### 4.3 노트북 자동 실행 (CLI)

```bash
cd "시계열_Test/Phase1_LSTM"
jupyter nbconvert --to notebook --execute --inplace 01_data_download_and_eda.ipynb \
    --ExecutePreprocessor.timeout=300
```

---

## 5. 협업 규약

### 5.0 `.py` 분리 기준 (사용자 지시, 2026-04-24)
- **`scripts/*.py`로 분리**: Phase 1 **전반**에 공통으로 쓰이는 기능 (예: 환경 설정, 시드, 경로, 시각화 유틸 등)
- **노트북에 둠**: 특정 단계(EDA·학습·평가)의 흐름·시각화·해석 코드
- 즉, "여러 노트북이 import해서 쓸 함수"만 `.py`로, "한 노트북 안에서만 쓰는 흐름"은 노트북 셀로.

### 5.1 모듈 변경
- `scripts/X.py`의 public 함수 시그니처를 변경할 때는 **반드시 [재천_WORKLOG.md](재천_WORKLOG.md)에 변경 이유·이전/이후 인터페이스 기록**.
- 모든 public 함수: type hints + Numpy-style docstring.

### 5.2 누수 방지 (Phase 1 최우선 원칙)
- shift·rolling·resample 줄에는 인라인 주석 `# 누수: ...` 의무.
- 각 노트북 §4 누수 검증 셀 (assert + 육안 표 + 인공 누수 대조) 통과 후 다음 단계 진행.
- 자세한 4계층 방어선: plan 파일의 "데이터 누수 방지" 섹션 참조.

### 5.3 작업 기록
- 모든 결정·판단은 [재천_WORKLOG.md](재천_WORKLOG.md)에 시간순 누적.
- 노트북 실행 결과 핵심 수치(메트릭, 이상치 개수, ACF 등)도 재천_WORKLOG에 요약.
- 다른 팀원은 본인 prefix(`<이름>_WORKLOG.md`)로 별도 일지를 둘 수 있습니다.

### 5.4 코드 스타일
- 변수명은 영어 (한글 컬럼명은 유지 가능). 한글 변수명 금지.
- 시각화 한글 폰트 설정 필수 (`scripts.setup.setup_korean_font()` 또는 `00` 노트북 호출).
- 어려운 개념은 마크다운 셀에 충분히 설명.

---

## 6. 참고 문서

- **상위 plan**: `C:\Users\gorhk\.claude\plans\c-users-gorhk-finance-project-study-00-m-frolicking-iverson.md`
- **상위 학습계획**: `김재천/Study/00_학습계획.md`
- **원 논문**: Su, X., Lu, K., & Yen, J. (2026). *Objective Black-Litterman views through deep learning: A novel hybrid model for enhanced portfolio returns*. **Expert Systems with Applications, 295.**
- **Walk-Forward 누수 방지**: López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. §7.

---

## 7. 진행 상태 (2026-04-24 현재)

| Step | 산출물 | 상태 |
|---|---|---|
| Step 0 | `00_setup_and_utils.ipynb` | ✅ 완료 |
| Step 1 | `01_data_download_and_eda.ipynb` + CSV | ✅ 완료 |
| Step 2 | `02_setting_A_daily21.ipynb` + scripts/*.py | 🟡 진행 예정 |
| Step 3 | `03_setting_B_monthly.ipynb` | ⏸ 대기 |
| Step 4 | `04_compare_A_vs_B.ipynb` | ⏸ 대기 |

문의는 [재천_WORKLOG.md](재천_WORKLOG.md)의 의사결정 보류 항목 또는 작성자(gorhkdwj@gmail.com)에게 전달 부탁드립니다.
