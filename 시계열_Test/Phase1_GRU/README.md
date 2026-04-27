# Phase 1 — GRU 단독 베이스라인

> **협업 진입점 문서**. 처음 합류한 팀원은 이 README → [PLAN.md](PLAN.md) → [scripts_정의서.md](scripts_정의서.md) → 노트북 순으로 읽으십시오.

## 1. 프로젝트 위치 및 목적

- **상위 프로젝트**: COL-BL (Su et al. 2026 ESWA 295 논문의 CGL-BL 재현, Optuna 변형)
- **현 단계**: Phase 1 — 6단계 모델 평가 로드맵 중 LSTM 대체 실험
- **목적**: Phase1_LSTM 결과(과적합, R²_OOS < 0) 개선을 위해 **GRU 로 교체** 후 동일 조건 재실험
- **배경**: 결과분석5.md §6 Option C — "GRU 파라미터 25% 감소 → 134샘플 환경에서 안정적일 수 있음"

### 6단계 평가 로드맵

| # | 모델 | 시간 해상도 | 단계 |
|---|---|---|---|
| 1 | LSTM | 일별 (21영업일 후) | Phase 1 (`Phase1_LSTM/`) |
| 2 | LSTM | 월별 (1개월 후) | Phase 1 (`Phase1_LSTM/`) |
| 3 | **GRU** | 일별 (21) | **Phase 1 (현재 — `Phase1_GRU/`)** |
| 4 | GRU | 월별 (1) | 대기 |
| 5 | CEEMDAN+LSTM | 일별 (21) | Phase 3 |
| 6 | CEEMDAN+LSTM | 월별 (1) | Phase 3 |

---

## 2. GRU vs LSTM 핵심 차이

| 항목 | LSTM | GRU |
|---|---|---|
| 게이트 수 | 4 (input, forget, cell, output) | 2 (reset, update) |
| 파라미터 수 (hidden=32, input=1) | ≈ 4,480 | ≈ 3,264 (약 27% 감소) |
| 셀 상태 (cell state) | 있음 | 없음 |
| `forward` 반환 | `(output, (h_n, c_n))` | `(output, h_n)` |
| forget gate bias init | 있음 (PyTorch 기본 0) | 해당 없음 |
| 과적합 위험 (소규모 데이터) | 상대적으로 높음 | 상대적으로 낮음 |

---

## 3. 핵심 의사결정 요약

| 항목 | 결정 |
|---|---|
| 자산군 | SPY, QQQ (각각 독립 학습·평가) |
| 데이터 소스 | Phase1_LSTM 의 raw_data 공유 (재다운로드 불필요) |
| 분석 기간 | 2016-01-01 ~ 2025-12-31 (10년, LSTM 과 동일) |
| 입력 피처 | Univariate (log-return 단일) — LSTM 과 동일 (공정 비교) |
| 설정 A | 일별 + 21영업일 후 누적 log-return / seq_len=63 / hidden=32 / 1-layer GRU |
| 검증 방법 | Rolling Walk-Forward + Purge + Embargo (López de Prado 2018) |
| Walk-Forward | IS 231 / purge 21 / emb 21 / OOS 21 / step 21 |
| **1차 평가 지표** | **Hit Rate**, **R²_OOS** (Campbell & Thompson 2008) |
| **관문 (Phase 2 진행 조건)** | **Hit Rate > 0.55** AND **R²_OOS > 0** (둘 다 충족 시 PASS) |

---

## 4. 폴더 구조

```
Phase1_GRU/
├── README.md                              ← 이 문서
├── PLAN.md                                ← 전체 구현 계획
├── scripts_정의서.md                       ← scripts/*.py API 정의서
│
├── 논의사항/                               ← 날짜별 논의 기록
│   └── README.md
│
├── 00_setup_and_utils.ipynb               ← 환경 노트북 (한글 폰트·시드·경로)
├── 02_setting_A_daily21.ipynb             ← 설정 A §1~§9.F 완전 구현
│
├── scripts/                               ← 재사용 모듈
│   ├── __init__.py
│   ├── setup.py                           ← ✅ 환경 부트스트랩 (폰트·시드·경로)
│   ├── targets.py                         ← ✅ 타깃 빌더 + 누수 검증 (LSTM 과 동일)
│   ├── dataset.py                         ← ✅ SequenceDataset + Walk-Forward (LSTM LSTMDataset → SequenceDataset)
│   ├── models.py                          ← ✅ GRURegressor (nn.GRU 기반)
│   ├── train.py                           ← ✅ train_one_fold (학습 루프, LSTM 과 동일)
│   └── metrics.py                         ← ✅ hit_rate + r2_oos + baseline_metrics (LSTM 과 동일)
│
└── results/                               ← 노트북 실행 결과만 저장
    ├── raw_data/                          ← (Phase1_LSTM 공유 CSV 사용, 여기 저장 불필요)
    └── setting_A/{SPY,QQQ}/               ← metrics.json, *.png
```

---

## 5. 실행 순서

### 5.1 환경

```bash
# Phase1_LSTM 과 동일 환경 사용 (uv sync 또는 pip install 이미 완료된 경우)
# 추가 의존성 없음
```

### 5.2 노트북 실행

```
00_setup_and_utils.ipynb   ← 환경 부트스트랩
        ↓
02_setting_A_daily21.ipynb ← GRU 학습·평가 전체 (§1~§9.F)
```

> ⚠️ 반드시 `Phase1_GRU/` 디렉토리에서 실행 (`%run ./00_...` 의 상대 경로 때문)

---

## 6. LSTM 과의 코드 차이 요약

| 파일 | 변경 내용 |
|---|---|
| `scripts/models.py` | `LSTMRegressor` → `GRURegressor`, `nn.LSTM` → `nn.GRU`, `forget_gate_bias_init` 제거 |
| `scripts/dataset.py` | `LSTMDataset` → `SequenceDataset` (모델 무관 이름으로 변경) |
| `scripts/setup.py` | bootstrap 메시지 "LSTM" → "GRU" |
| `scripts/train.py` | 주석/docstring 만 "LSTM" → "GRU" (학습 루프 로직 동일) |
| `scripts/metrics.py` | 모듈 docstring 첫 줄만 업데이트 (메트릭 로직 동일) |
| `scripts/targets.py` | 변경 없음 (타깃 생성 모델 무관) |
| `02_setting_A_daily21.ipynb` | `LSTMRegressor` → `GRURegressor`, `LSTMDataset` → `SequenceDataset`, raw_data 경로 수정 |

---

## 7. 참고 문서

- **Phase1_LSTM 결과**: `Phase1_LSTM/논의사항/2026-04-25_결과분석5.md`
- **GRU 이론**: `학습자료_주의사항.md` (Phase1_LSTM 폴더 내)
- **Walk-Forward 설계**: `Phase1_LSTM/PLAN.md` §Walk-Forward 구조
- **원 논문**: Su, X. et al. (2026). ESWA 295.
