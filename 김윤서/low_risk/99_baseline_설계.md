# 99_baseline.ipynb 설계 문서

> 논문 구현: Pyo & Lee (2018) — 변동성 예측 없이 현재 `vol_21d`로 직접 분류한 Black-Litterman Baseline 포트폴리오

---

## 1. 전체 구조

```
[Part 1] BL Baseline (Q_FIXED)
  cell-00  파라미터 및 데이터 로드
  cell-02  BL 구성 함수 정의
  cell-03  Walk-forward (Q_FIXED = 0.003)
  cell-04  성과 분석 (연수익 / 변동성 / Sharpe / MDD)
  cell-05  시각화 → bl_baseline_performance.png

[Part 2] Q 방식별 성능 비교 (2-pass walk-forward)
  7135d45b GARCH vol 로드 / Q_CANDIDATES / get_regime 정의
  ebb251b7 Pass 1: Q 민감도 탐색 + 공유 데이터 캐시 + expanding window Q_OPTIMAL 결정
  b876792f Pass 2: 적응형 Q 방식 8종(q_fixed expanding 포함) + 성능 테이블 출력
  0ac8eb51 시각화 (3-panel) → q_method_comparison.png
           반환 시계열 → q_method_comparison_returns.csv
```

---

## 2. 파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `TRAIN_WINDOW` | 60 | 공분산 추정 rolling 윈도우 (개월) |
| `TAU` | 0.1 | BL 사전 확신도 스케일러 |
| `PCT_GROUP` | 0.30 | 저위험/고위험 그룹 분류 비율 (상하위 30%) |
| `START_PRED` | 2010-01-01 | 관측 기간 시작 (60개월 워밍업 후) |
| `END_PRED` | 2024-12-31 | 관측 기간 상한 |
| `Q_FIXED` | 0.003 | Baseline Q 고정값 (월 0.3%) |
| `LAM_MEAN` | 2.5 | λ 정규화 기준값 |
| `Q_CANDIDATES` | [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020] | Q 민감도 탐색 후보군 |
| `START_PRED_Q` | 2011-01-01 | Q 비교 시작 (GARCH vol_predicted.csv 시작월) |

---

## 3. 4-Step BL 파이프라인

### Step 1 — 자산 분류 (P 벡터 구성)

- 매월 `vol_21d` 오름차순 정렬
- 하위 30% → **저위험 (long)**, 상위 30% → **고위험 (short)**
- P 벡터: 시총 가중 (`log_mcap` → exp → 비율)

```python
def build_P(vol_series, mcap_series, pct=0.30):
    # 하위 pct → long (+), 상위 pct → short (-)
    # 각각 시총 가중 정규화
```

### Step 2 — 사전 분포 계산 (π, Σ, Ω)

**Σ (공분산)**
```python
def compute_sigma(ret_matrix):
    lw = LedoitWolf().fit(ret_matrix.values)
    # Ledoit-Wolf shrinkage estimator
```

**π (균형 기대수익)**
```python
def compute_pi(Sigma, w_mkt, spy_excess_ret, sigma2_mkt):
    lam = spy_excess_ret / sigma2_mkt  # 시장 위험회피계수 λ
    lam = np.clip(lam, 0.5, 10.0)
    pi  = lam * Sigma @ w_mkt          # CAPM 역최적화
```

**Ω (뷰 불확실성) — He-Litterman 공식**
```python
def compute_omega(P, Sigma, tau):
    omega = tau * P @ Sigma @ P        # Ω = τ·P·Σ·P^T (스칼라)
```

> **현재 Ω 설계**: 완전 정적 (static). 뷰 불확실성 = 사전 분포 불확실성의 τ=10% 비례.
> GARCH 신뢰도 / IC 기반 동적 Ω는 미구현 (Q_Estimation_Design.md Section 6 참고).

### Step 3 — BL 사후 분포 (Sherman-Woodbury)

```
μ_BL = π + τΣP^T [P(τΣ)P^T + Ω]^{-1} (Q - Pπ)
```

```python
def black_litterman(pi, Sigma, P, q, omega, tau):
    M      = (tau * P @ Sigma @ P) + omega
    diff   = q - (P @ pi)
    adjust = (tau * Sigma @ P) * (diff / M)
    return pi + adjust
```

### Step 4 — 포트폴리오 최적화 (MVO)

```
min  (λ/2) w^T Σ w − w^T μ_BL
s.t. Σw = 1,  w ≥ 0  (롱온리)
```

```python
def optimize_portfolio(mu_BL, Sigma, lam):
    # SLSQP, 실패 시 1/N 폴백
```

---

## 4. Part 1: BL Baseline Walk-forward

- **기간**: 2010-01 ~ 2024-12 (180개월)
- **Q**: `Q_FIXED = 0.003` (고정)
- **출력 변수**: `bl_ret`, `capm_ret`, `spy_ret`
- **저장**: `data/bl_baseline_returns.csv`, `outputs/99_baseline/bl_baseline_performance.png`

### 비교 대상

| 포트폴리오 | 설명 |
|---|---|
| BL Baseline | vol_21d 분류 + Q_FIXED + He-Litterman Ω |
| CAPM 균형 | BL 없이 μ = π 그대로 MVO |
| S&P 500 | 시장 벤치마크 |

---

## 5. Part 2: Q 방식별 성능 비교 (2-pass)

### 비교 기간

- **시작**: 2011-01-01 (GARCH `vol_predicted.csv` 제공 시점)
- **종료**: 2024-12-31 (`END_PRED` 상한 공유)
- **총 180개월 → GARCH 제약으로 실제 170개월 내외**

### 2-pass 설계 이유

Pass 1에서 Σ, π, P, Ω를 한 번 계산해 캐시 → Pass 2에서 재사용.  
Q 후보 8개와 적응형 방식 8개를 별도 루프로 분리하여 Σ 재계산 비용 절감.

---

### Pass 1: Q 민감도 탐색 + expanding window Q_OPTIMAL 결정

```python
Q_CANDIDATES = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]

# 매월 Σ/π/P/Ω 계산 후 8개 Q 후보 모두 BL+MVO 실행
# 결과는 sens_raw에 누적

# Expanding window Q 선택 (이번 달 실행 전 — 과거 데이터만 사용, 사후 선택 편향 없음)
# i < 24: Q_FIXED 사용 (warmup)
# i >= 24: 누적 sens_raw Sharpe 최대 후보 → q_opt_t
# → _shared_cache[pred_date]['q_optimal']에 저장

# 전 기간 in-sample Q_OPTIMAL은 표시/참고용으로만 출력
best_key  = max(_sharpes_cand, key=_sharpes_cand.get)
Q_OPTIMAL = float(best_key.split('=')[1])  # 참고용
```

**캐시 저장 항목** (`_shared_cache[pred_date]`):

| 키 | 내용 |
|---|---|
| `pi_v` | CAPM 균형 기대수익 벡터 |
| `lam` | clip된 λ (π 계산용, [0.5, 10.0]) |
| `lam_raw` | clip 전 raw λ = SPY_excess_ret / σ²_mkt (음수 가능, Q 스케일링용) |
| `Sigma_np` | numpy 공분산 행렬 (copy) |
| `Sigma_idx` | 유효 ticker 리스트 |
| `P` | 뷰 벡터 |
| `omega` | He-Litterman Ω |
| `actual_ret` | 다음달 실현수익률 |
| `regime` | 3-class 레짐 (`저변동성` / `중간` / `고변동성`) |
| `lam_scale` | λ/LAM_MEAN (클립 0.1~3.0) |
| `spread_t` | \|P·π\| (현재 CAPM 스프레드) |
| `spread_ref` | expanding median of spread_t |
| `vs_t` | GARCH vol 격차 (short leg - long leg) |
| `vs_ref` | expanding median of vs_t |
| `vs_ok` | GARCH 데이터 존재 여부 |
| `q_optimal` | expanding window로 선택된 이번 달 Q (Pass 2 q_fixed 기준값) |

---

### Pass 2: 적응형 Q 방식 8종

캐시를 재사용하므로 Σ 재계산 없음. 모든 방식은 `c['q_optimal']` (expanding window Q)을 기준값으로 사용.

| 방식 | 계산식 | 신호 |
|---|---|---|
| `q_fixed` | `q_opt_t` | expanding window 최적 Q (편향 없음) |
| `q_lambda` | `q_opt_t × clip(λ/LAM_MEAN, 0.1, 3.0)` | 시장 위험회피계수 λ |
| `regime3` | `q_opt_t × {저:1.0, 중:0.5, 고:0.0}` | SPY 변동성 레짐 (하드스탑) |
| `regime_lambda` | `0` (고변동성) or `q_opt_t × lam_scale` (그 외) | 레짐 하드스탑 + λ 연속 스케일 |
| `raw_lam_q` | `max(0, q_opt_t × lam_raw/LAM_MEAN)` | raw λ 부호로 자연 게이팅 (하드스탑 없음) |
| `pi_ratio` | `q_opt_t × clip(spread_t / spread_ref, 0.1, 3.0)` | CAPM 스프레드 비율 |
| `vol_spread` | `0` (고변동성) or `q_opt_t × clip(vs_t/vs_ref, 0.1, 3.0)` (그 외) | GARCH vol 격차 + 레짐 하드스탑 |
| `q_ff3` | `compute_Q_ff3(P, ret_slice, ff3, rf_s)` — Pass 1 캐시 | FF3 팩터 회귀 추정값 (비교 목적) |

#### 레짐 분류 방식 (`get_regime`)

```python
# SPY 12개월 rolling std → expanding percentile rank
pct = (historical_vol12 < current_vol12).mean()

고변동성: pct > 0.67
중간:     0.33 < pct ≤ 0.67
저변동성: pct ≤ 0.33
```

---

## 6. 출력 파일

| 파일 | 내용 |
|---|---|
| `outputs/99_baseline/bl_baseline_performance.png` | BL Baseline vs CAPM vs SPY 누적수익 + Sharpe 차트 |
| `outputs/99_baseline/q_method_comparison.png` | Q 방식별 누적수익 / Sharpe / MDD 3-panel 차트 |
| `data/bl_baseline_returns.csv` | bl / capm / spy 월별 수익률 |
| `data/q_method_comparison_returns.csv` | 8개 Q 방식 + CAPM + SPY 월별 수익률 |

---

## 7. 설계 결정 근거

### 7-1. Q_OPTIMAL 탐색 방식 — 사후 선택 편향 문제

**문제**: 기존 구현은 전 기간 walk-forward 완료 후 `sens_raw`의 누적 수익률로 Q_OPTIMAL을 선택했다.  
각 월의 BL+MVO는 과거 데이터만 사용하므로 walk-forward 자체는 OOS이지만, **Q_OPTIMAL 선택 자체가 전 기간을 본 뒤 내리는 결정**이었다.

```
진짜 OOS라면: 2011년 1월 시점에서 Q_OPTIMAL을 알 수 없음
기존 구현:   2024년 12월까지 다 본 뒤 "0.007이 제일 좋았네" → 소급 적용
```

이 사후 선택 편향은 다음 특성을 갖는다:

| 항목 | 내용 |
|---|---|
| 편향 규모 | 8개 이산 후보 중 최선 소급 선택 → **소폭** (연속 파라미터 과적합 수준 아님) |
| 방식 간 비교 유효성 | 모든 방식이 동일 Q_OPTIMAL 기준값 공유 → **방식 간 상대 비교는 유효** |
| 절대 성과 수치 | 약간 낙관적 편향 → **절대 수치를 보고에 쓸 경우 문제** |

---

### 7-2. Expanding Window Q 선택 — 편향 제거

**해결**: 매월 BL+MVO 실행 직전, 그 시점까지 쌓인 `sens_raw`만으로 Q_OPTIMAL을 결정한다.

```
2011-01 ~ 2013-00 (i < 24): 과거 부족 → Q_FIXED=0.003 fallback
2013-01 (i=24):  과거 24개월 Sharpe → "지금까지는 0.005가 최선" → 0.005 사용
2015-06 (i=54):  과거 54개월 Sharpe → "0.007이 올라옴" → 0.007 사용
2024-12 (i=167): 과거 167개월 Sharpe → 최적 Q 사용
```

**구현 구조**:

```python
# Pass 1 루프 내 — BL+MVO 실행 전에 위치
if i >= 24:
    _past_sharpes = {}
    for _key, _lst in sens_raw.items():          # 이번 달 결과 반영 전
        if len(_lst) >= 12:
            _r   = pd.Series([x['ret'] for x in _lst])
            _exc = _r - _rf_all.reindex(_r.index).fillna(0)
            if _exc.std() > 0:
                _past_sharpes[_key] = float(_exc.mean() / _exc.std() * np.sqrt(12))
    _best_t = max(_past_sharpes, key=_past_sharpes.get) if _past_sharpes else f'Q={Q_FIXED}'
    q_opt_t = float(_best_t.split('=')[1])
else:
    q_opt_t = Q_FIXED
q_optimal_per_month[pred_date] = q_opt_t
_shared_cache[pred_date]['q_optimal'] = q_opt_t
```

**실행 시간 영향**: BL+MVO(SLSQP×8)에 비해 배열 mean/std 계산 8회는 무시 가능 수준 (< 1% 추가).

**Pass 2에서의 활용**: 모든 적응형 방식의 기준값을 `Q_OPTIMAL`(상수) → `c['q_optimal']`(월별 다름)으로 교체.

---

### 7-3. Q 민감도 결과가 실행마다 달라지는 이유

코드 자체는 결정론적(랜덤성 없음). 실행마다 결과가 달라진다면 **데이터 파일이 달라진 것**이 원인이다.

| 변동 요인 | 결과 영향 |
|---|---|
| `monthly_panel.csv` 갱신 | vol_21d, ret_1m, log_mcap 변화 → universe 구성 변화 → Σ 변화 |
| `vol_predicted.csv` 갱신 | vs_t 계산 변화 → vol_spread Q 변화 |
| 동일 데이터 파일 | 완전히 동일한 결과 재현 |

---

### 7-4. BL Baseline vs CAPM vs S&P500 비교 목적

| 비교 쌍 | 보여주는 것 |
|---|---|
| BL Baseline vs CAPM 균형 | BL의 기여도 — 뷰(Q) 추가가 Sharpe를 높이는가 |
| BL Baseline vs S&P 500 | 전략의 시장 대비 초과성과 여부 (알파 존재 여부) |
| CAPM 균형 vs S&P 500 | MVO 최적화 자체의 효과 (지수 단순 추종 vs 최적 가중) |

CAPM 균형은 BL 없이 μ=π 그대로 MVO를 실행한 결과다. CAPM 균형보다 BL Baseline의 Sharpe가 높으면 "Q 뷰 자체가 유효하다"는 근거가 된다.

---

## 8. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-30 | `Q_hist`, `Q_ff3` 제거 (FF3 로드 셀 삭제, `compute_Q_hist` / `compute_Q_ff3` 삭제, GARCH+Q_hist/ff3 walk-forward 셀 삭제) |
| 2026-04-30 | 관측 기간 2010-01 ~ 2024-12로 변경 (`START_PRED`, `END_PRED` 파라미터 추가) |
| 2026-04-30 | Q 방식별 성능 비교 2-pass walk-forward 추가 (Q_FIXED / Q_lambda / Regime3 / Regime+λ / π_ratio / Q_vol_spread+하드스탑) |
| 2026-04-30 | 중복 비교 셀 `dac04d71` 제거 (cell-04 + cell-05로 통합) |
| 2026-05-01 | `raw_lam_q` 신규 추가 (raw λ 부호 기반 자연 게이팅, 하드스탑 없음) — `lam_raw` 캐시 추가, `regime_lambda` 하드스탑 유지하여 둘 다 비교 |
| 2026-05-01 | `Q_ff3` 비교 추가 — `compute_Q_ff3` 함수 및 FF3 데이터 로드 추가, Pass 1 캐시에 `q_ff3` 저장, Pass 2 비교 방식에 포함 |
| 2026-05-01 | `q_fixed` expanding window 전환 — Pass 1에서 매월 과거 sens_raw Sharpe로 `q_opt_t` 결정, `_shared_cache['q_optimal']`에 저장, Pass 2에서 모든 방식의 기준값으로 사용 (사후 선택 편향 제거) |
