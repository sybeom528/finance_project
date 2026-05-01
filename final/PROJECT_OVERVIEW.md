# Low-Risk Black-Litterman 실험 프로젝트

---

## 1. 핵심 목표

> **저변동성 종목군에 대해 Black-Litterman 모델을 적용하고,**  
> **Prior / P 행렬 / Q / 비교군 등 슬롯별 실험으로 최적 구성을 찾는다.**

---

## 2. Black-Litterman 구조 요약

```
π (CAPM 균형 수익률, Prior)
+
Q (투자자 뷰 = P 포트폴리오의 기대 초과수익률)
Ω (뷰 불확실성)
─────────────────────────────────────────────
→ μ_BL (사후 기대 수익률) → 포트폴리오 최적화
```

**P 포트폴리오**: `vol_21d` 하위 30% Long / 상위 30% Short (자기자금 조달, sum=0)  
**Prior (π)**: CAPM 균형수익률 — 시총가중(`capm_mcap`) 또는 균등가중(`capm_eq`)  
**Q**: 현재 고정값 0.003 (월 0.3%) → 향후 HMM 기반 동적 Q로 개선 예정

---

## 3. 파일 구조

```
final/
├── bl_config.py          ← 실험 정의 (실험 추가는 여기서만)
├── bl_functions.py       ← BL 계산 핵심 함수
├── 99_run.ipynb          ← 실험 실행 + pkl 저장
├── 99_analyze.ipynb      ← 결과 분석 / 차트
│
├── results/              ← 실험 결과 pkl (99_run 출력)
│   ├── baseline.pkl
│   ├── prior_eq.pkl
│   └── ...
│
├── data/                 ← monthly_panel.csv 등 입력 데이터
├── outputs/99_compare/   ← 차트 PNG 자동 저장
│
├── _dev/                 ← 보관용 (실험 초안 등)
│   └── 99_baseline.ipynb
│
└── PROJECT_OVERVIEW.md   ← 이 파일
```

---

## 4. 실험 슬롯 구조

`bl_config.py`의 `BASELINE` 기준값에서 하나씩 바꾸는 OFAT 방식.

| 슬롯 | 선택지 | 설명 |
|------|--------|------|
| `prior` | `capm_mcap` / `capm_eq` | CAPM 균형수익률 가중방식 |
| `p_mode` | `trailing_vol21` / `trailing_vol252` / `lstm_predicted` | P 행렬 신호 기준 변동성 |
| `p_weight` | `mcap` / `eq` / `rp` / `vol_mcap` | P 행렬 가중방식 |
| `q_mode` | `fixed` / `capm` / `none` / `regime_bayesian`(예정) | Q 결정 방식 |
| `q_value` | float | `q_mode='fixed'`일 때 Q 값 |
| `omega_mode` | `he_litterman` / `scaled` / `rmse` | Ω 계산 방식 |
| `tc` | float | 편도 거래비용 (기본 0.001 = 10bp) |
| `max_weight` | float | 단일 종목 상한 (기본 0.10) |

---

## 5. 현재 실험 목록 (13개)

| 실험명 | 변경 슬롯 | 설명 |
|--------|----------|------|
| `baseline` | — | 기준선 |
| `prior_eq` | prior | 1/N 균등가중 Prior |
| `p_vol252` | p_mode | 252일 장기 변동성 신호 |
| `p_rp` | p_weight | 역변동성 가중 (전체 유니버스) |
| `p_eq` | p_weight | 동일가중 |
| `p_vol_mcap` | p_weight | vol×mcap 가중 (전체 유니버스) |
| `prior_eq_p_vol252` | prior + p_mode | 1/N Prior + 252일 vol |
| `prior_eq_p_rp` | prior + p_weight | 1/N Prior + 역변동성 |
| `prior_eq_p_eq` | prior + p_weight | 1/N Prior + 동일가중 |
| `prior_eq_p_vol_mcap` | prior + p_weight | 1/N Prior + vol×mcap |
| `capm_no_bl` | q_mode | BL 없음 — CAPM π 직접 최적화 (전체 유니버스) |
| `naive_lowvol` | q_mode | BL 없음 — 저변동 시총가중 직접 보유 |
| `naive_lowvol_rp` | q_mode + p_weight | BL 없음 — 저변동 역변동성 가중 |

---

## 6. 실험 추가 방법

### 파라미터만 바꾸는 경우 → `bl_config.py` 한 줄 추가

```python
# 예: max_weight 15%로 올리는 실험
{**BASELINE, 'name': 'max15', 'max_weight': 0.15},
```

이후 `99_run.ipynb` 실행하면 자동으로 `results/max15.pkl` 생성.

### 새 계산 방식 추가 시 → 3개 파일 수정 필요

1. `bl_config.py` — 슬롯 주석 + 실험 dict 추가
2. `bl_functions.py` — 해당 슬롯 처리 함수 추가
3. `99_run.ipynb` — dispatcher에 분기 추가 (cell-04 참고)

---

## 7. 성과 요약 (2004~2025, Sharpe 기준)

| 실험 | Sharpe | CAGR | MDD |
|------|--------|------|-----|
| baseline | 1.106 | 13.4% | -13.0% |
| prior_eq | 1.105 | 14.2% | -13.9% |
| p_vol252 | 1.073 | 13.1% | -13.1% |
| naive_lowvol | 1.061 | 13.9% | -14.7% |
| ... | | | |
| SPY | 0.908 | 14.4% | -23.9% |
| capm_no_bl | 0.899 | 14.8% | -22.2% |

→ **baseline이 Sharpe 1위.** P 행렬 / Prior 변경의 효과는 제한적.  
→ **핵심 개선 여지: Q의 동적 조정** (현재 고정값 = 가장 약한 고리)

---

## 8. 다음 단계 — HMM 기반 동적 Q

**계획**: 김하연 HMM 레짐 모델 + LSTM 변동성 예측을 결합해 Q를 동적으로 결정.

```
HMM posterior P(레짐_t | obs) × 전이행렬
        ↓
P(레짐_{t+1}) prior
        ↓ Bayesian update
LSTM 예측 vol → emission likelihood
        ↓
posterior × Q_per_regime → Q_t (동적)
```

상세 구현 계획: [`hmm_bayesian_q_실험계획.md`](hmm_bayesian_q_실험계획.md) 참고

---

## 9. Look-ahead Bias 체크리스트

| 변수 | 사용 시점 | 상태 |
|------|----------|------|
| `vol_21d` | pred_date 이전 21일 | ✅ 안전 |
| `log_mcap` | pred_date 시점 | ✅ 안전 |
| `fwd_ret_1m` | 평가(성과 계산)에만 사용 | ✅ 안전 |
| LSTM 예측 | walk-forward 생성 | ✅ 안전 |
| HMM 파라미터 | In-sample 학습 (한계) | ⚠️ 명시 필요 |
