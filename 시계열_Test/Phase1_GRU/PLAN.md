# Phase 1 — GRU 단독 베이스라인 구축 계획

> **목적**: Phase1_LSTM 의 과적합 문제(파라미터/샘플 비 과도, 병리적 폴드 다수)를
> 완화하기 위해 GRU 로 모델을 교체하고 동일 조건에서 재실험한다.
>
> **진실원**: Phase1_LSTM/PLAN.md 와 같은 Walk-Forward 설계·관문·데이터를 공유한다.
> 본 문서는 GRU 교체와 관련된 차이점만 기술한다.
>
> **마지막 갱신**: 2026-04-26

---

## Context

Phase1_LSTM 5차 Run(결과분석5.md)에서 다음이 확인되었다:

| 시도 | 결과 |
|---|---|
| hidden 축소 (128→32) | 소폭 악화 |
| Y_trailing 피처 추가 | 대폭 악화 |
| VIX 피처 추가 | 대폭 악화 |
| **공통 결론** | **훈련 샘플 수 절대 부족 (n_train ≈ 134/fold)** |

결과분석5.md §6 Option C:
> GRU는 LSTM 대비 파라미터 수 약 25% 감소 → 134샘플 환경에서 오히려 안정적일 수 있음.

**Phase1_GRU 의 목표**: 동일 하이퍼파라미터·데이터에서 GRU 가 LSTM 대비
과적합 완화 효과를 보이는지 정량 검증.

---

## GRU vs LSTM — 모델 설계 차이

### 게이트 구조

```
LSTM:
  i(t) = σ(W_i·[h_{t-1}, x_t] + b_i)   # input gate
  f(t) = σ(W_f·[h_{t-1}, x_t] + b_f)   # forget gate
  g(t) = tanh(W_g·[h_{t-1}, x_t] + b_g) # cell gate
  o(t) = σ(W_o·[h_{t-1}, x_t] + b_o)   # output gate
  c(t) = f(t)⊙c(t-1) + i(t)⊙g(t)
  h(t) = o(t)⊙tanh(c(t))

GRU (Cho et al. 2014):
  z(t) = σ(W_z·[h_{t-1}, x_t])           # update gate  ← forget+input 통합
  r(t) = σ(W_r·[h_{t-1}, x_t])           # reset gate   ← 이전 hidden 얼마나 잊을지
  n(t) = tanh(W_n·[r(t)⊙h_{t-1}, x_t])  # candidate hidden
  h(t) = (1-z(t))⊙h(t-1) + z(t)⊙n(t)   # cell state 없음
```

### 파라미터 수 비교 (hidden=32, input_size=1)

| 모델 | 가중치 행렬 | 파라미터 수 (근사) |
|---|---|---|
| LSTM (1-layer) | 4 × (H×F + H×H + H) | 4 × (32 + 1024 + 32) = 4,352 |
| GRU  (1-layer) | 3 × (H×F + H×H + H) | 3 × (32 + 1024 + 32) = 3,264 |
| **감소율** | | **약 25%** |

---

## 확정된 의사결정

Phase1_LSTM 과 동일한 항목은 생략. **GRU 전환으로 달라진 항목만 기재**.

| 항목 | Phase1_LSTM | Phase1_GRU |
|---|---|---|
| 모델 클래스 | `LSTMRegressor` (nn.LSTM) | `GRURegressor` (nn.GRU) |
| Dataset 클래스 | `LSTMDataset` | `SequenceDataset` (이름만 변경, 로직 동일) |
| forget_gate_bias_init | 있음 (default None) | 없음 (GRU 는 forget gate 없음) |
| 설정 A 파라미터 | hidden=32, 1-layer, dropout=0.3 | **동일** (공정 비교) |
| raw_data 위치 | `results/raw_data/` | `Phase1_LSTM/results/raw_data/` (공유) |

**변경하지 않은 항목 (공정 비교 보장)**:
- 데이터 기간, 분석 기간, 자산군
- Walk-Forward 설정 (IS/purge/emb/OOS/step = 231/21/21/21/21)
- seq_len=63, batch_size=32, val_ratio=0.2
- 학습 루프 (HuberLoss, AdamW, ReduceLROnPlateau, EarlyStopping patience=5)
- 평가 지표 (hit_rate, r2_oos, 관문 기준)

---

## 폴더 구조

```
시계열_Test/Phase1_GRU/
├── README.md                              # 협업 진입점
├── PLAN.md                                # ⭐ 본 문서
├── scripts_정의서.md                       # scripts/*.py API 정의서
│
├── 논의사항/                               # 날짜별 논의 기록 누적
│   └── README.md
│
├── 00_setup_and_utils.ipynb               # 환경 노트북
├── 02_setting_A_daily21.ipynb             # 설정 A 전체 (§1~§9.F)
│
├── scripts/
│   ├── __init__.py
│   ├── setup.py          # 한글 폰트·시드·경로 (BASE_DIR = Phase1_GRU/)
│   ├── models.py         # ⭐ GRURegressor (핵심 변경)
│   ├── train.py          # 학습 루프 (LSTM 과 동일 로직)
│   ├── dataset.py        # SequenceDataset (LSTMDataset 이름 변경)
│   ├── metrics.py        # 평가 지표 (LSTM 과 동일)
│   └── targets.py        # 타깃 생성 (LSTM 과 동일)
│
└── results/
    ├── raw_data/          # (빈 폴더 — Phase1_LSTM raw_data 공유)
    └── setting_A/{SPY,QQQ}/
```

---

## 구현 완료 상태 (2026-04-26)

| 항목 | 상태 |
|---|---|
| `scripts/models.py` (GRURegressor) | ✅ 완료 |
| `scripts/dataset.py` (SequenceDataset) | ✅ 완료 |
| `scripts/train.py` | ✅ 완료 |
| `scripts/setup.py` | ✅ 완료 |
| `scripts/metrics.py` | ✅ 완료 |
| `scripts/targets.py` | ✅ 완료 |
| `00_setup_and_utils.ipynb` | ✅ 완료 |
| `02_setting_A_daily21.ipynb` | ✅ 완료 (실행 대기) |
| Setting A 실행·결과 확인 | ⏸ 실행 필요 |
| LSTM 대비 비교 분석 | ⏸ 실행 후 진행 |

---

## 예상 결과 시나리오

### 시나리오 A: GRU > LSTM (기대 케이스)
- r2_oos 음수지만 덜 음수 (예: -0.1 수준)
- 병리적 폴드(r2 < -1) 비율 감소
- best_epoch mean 증가 (조기 종료 줄어듦)
- **→ GRU 방향 유지, 추가 피처 or IS 확대 실험**

### 시나리오 B: GRU ≈ LSTM (파라미터 감소 효과 없음)
- 모든 지표 유사
- **→ 모델 구조 변경보다 피처 공학이 근본 해결책**

### 시나리오 C: GRU < LSTM (악화)
- univariate 신호가 너무 약해 어떤 RNN 도 무력
- **→ Option A(3차 Run 결과 수용) 또는 CEEMDAN Phase 3 으로 직행**

---

## 데이터 누수 방지 (LSTM 과 동일 원칙)

Phase1_LSTM/PLAN.md 의 "⚠️ 데이터 누수 방지" 섹션을 그대로 적용.
코드 변경 없이 모델만 교체하므로 누수 위험 추가 없음.

---

## 관련 문서

- [Phase1_LSTM/PLAN.md](../Phase1_LSTM/PLAN.md) — LSTM 전체 계획 (진실원 역할)
- [Phase1_LSTM/논의사항/2026-04-25_결과분석5.md](../Phase1_LSTM/논의사항/2026-04-25_결과분석5.md) — GRU 도입 근거
- [scripts_정의서.md](scripts_정의서.md) — GRU scripts API 정의서
