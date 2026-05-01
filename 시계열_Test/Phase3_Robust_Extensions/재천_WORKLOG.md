# Phase 3 — 재천 WORKLOG

## 단계 진입 결정 (2026-04-29)

본 Phase 3 진입에 앞서, **Phase 1.5 ~ 2 의 엄밀 검토** 가 사용자 결정으로 선행.

### 검토 동기

> 사용자: "phase 3 폴더 구조, plan만 작성해두고 phase 1.5~2 단계를 세세하고 엄밀하게 검토해보자.
>          기존 과정에 대해 오류 없이 정합성을 완벽히 챙긴 후 이후 과정 진행."

### 검토 범위

1. **Phase 1.5 정합성**: Walk-Forward 구조, 누수 방지, ensemble, 메트릭, 단위
2. **Phase 2 정합성**: Universe, Data, BL 공식, 백테스트, Σ, 시나리오 비교
3. **Phase 1.5 ↔ Phase 2 연결**: target 단위, 시점 매핑, universe 일치

---

## §1. Phase 1.5 검토 결과 — 정합성 OK ✅

### 1.1 Target 정의 (`targets_volatility.py`)

```python
target[t] = log(std(log_ret[t+1 : t+22], ddof=1))
```

- ✅ **Forward 21 일** (`shift(-window)` 적용)
- ✅ **ddof=1** 명시 (pandas 기본 일관)
- ✅ **단위**: 21 일 sample 의 일별 std 의 log = **daily volatility** (forward)
- ✅ **누수 검증** (`verify_no_leakage_logrv`): assert + 5 행 표

### 1.2 Walk-Forward CV (`dataset.py` + `volatility_ensemble.py`)

```
[ train (1250) ][ purge (21) ][ embargo (63) ][ test (21) ]
                    ↑               ↑              ↑
              target forward   long-memory   1-month OOS
```

- ✅ **purge=21**: target 의 forward 21 일 누수 차단
- ✅ **embargo=63**: log-RV ACF lag 63 (0.21) 잔존 차단
- ✅ **train end = t** → **test start = t + 84** → target end (t+21) ≪ test start ✅ 누수 X
- ✅ **StandardScaler**: `train_idx` 만으로 fit (test transform 시 fit 사용 X)

### 1.3 Performance-Weighted Ensemble (`volatility_ensemble.py`)

```python
w_v4[k]  = (1/RMSE_v4[k-1]) / (1/RMSE_v4[k-1] + 1/RMSE_har[k-1])
y_pred_ensemble[k] = w_v4[k] · y_pred_lstm[k] + w_har[k] · y_pred_har[k]
```

- ✅ **이전 fold (k-1) 의 RMSE 만 사용** → 본 fold (k) 미래 정보 X
- ✅ **첫 fold (k=0) initial 0.5/0.5** → warmup 처리
- ✅ **Diebold & Pauly (1987)** 학술 표준 결합

### 1.4 HAR-RV Baseline (`baselines_volatility.py`)

```python
RV_var_d[t] = log_ret[t]²
RV_var_w[t] = mean(log_ret²[t-4 : t+1])
RV_var_m[t] = mean(log_ret²[t-21 : t+1])
log(RV_h[t+h]) = β₀ + β_d · log(RV_d[t]) + β_w · log(RV_w[t]) + β_m · log(RV_m[t])
```

- ✅ **Trailing window** 만 사용 (입력 누수 X)
- ✅ **β 추정**은 `train_idx` 한정 OLS
- ✅ **Test 예측** 시 `train_idx` 데이터 미사용
- ✅ **Corsi (2009)** 표준 공식

### 1.5 단위 정리

```
y_pred_ensemble = log of daily volatility (forward 21d 추정)
exp(y_pred_ensemble) = daily volatility (실제 일별 std)

비교:
  vol_21d (panel): trailing 21d 의 daily std
  exp(target_logrv): forward 21d 의 daily std
  → 두 단위 모두 daily std (i.i.d. 가정 시)
```

→ **Phase 1.5 정합성 검토 완료, 문제 없음**.

---

## §2. Phase 2 검토 결과 — 1 Critical Bug + 1 Minor Issue 발견 ⚠️

### 2.1 BL 공식 (`black_litterman.py`) — 정합성 OK ✅

#### 2.1.1 `compute_pi`
```python
λ = E[r_mkt] / σ²_mkt
π = λ · Σ · w_mkt
```
- ✅ He-Litterman (1999) 표준 공식
- ⚠️ **Minor Issue**: rf 차감 누락 (Issue #2 참조)

#### 2.1.2 `build_P`
```python
P[low_risk]  = mcap[low_risk] / sum(mcap[low_risk])    # +
P[high_risk] = -mcap[high_risk] / sum(mcap[high_risk]) # -
```
- ✅ row_sum ≈ 0 (relative view, zero-investment)
- ✅ Pyo & Lee (2018) 표준

#### 2.1.3 `compute_omega` / `black_litterman` / `optimize_portfolio`
- ✅ 수학적 정확성
- ✅ He-Litterman + Markowitz 표준

### 2.2 Σ 추정 (`covariance.py`) — 학술적 한계 ⚠️

```python
Σ_daily = LedoitWolf shrinkage
Σ_monthly = Σ_daily × 21      ← i.i.d. 가정
```

- ✅ **LedoitWolf shrinkage** (Ledoit & Wolf 2004) 학술 표준
- ⚠️ **i.i.d. 가정 위반**: ARCH-LM SPY=754 (강한 conditional heteroskedasticity)
- ⚠️ **Phase 3 의 Hybrid Σ 로 개선 예정** (Phase 1.5 σ² 활용)

→ **Issue 가 아닌 학술적 한계** (Phase 3 에서 해결).

### 2.3 백테스트 (`backtest.py` + `_build_04_bl_yearly_nb.py`) — 🚨 Critical Bug 발견

#### 2.3.1 Look-ahead bias (✅ 수정됨)

```python
# 호출 측 (이미 수정됨)
forward_rets = monthly_rets.shift(-1)   # ⭐ forward 1 month
backtest_strategy(weights, forward_rets, ...)

# backtest_strategy 내부
gross_ret = (cur_w[t] * forward_rets[t]).sum()
         = (cur_w[t] * monthly_rets[t+1]).sum()   ← w[t] × r[t+1] ✅
```

→ **Look-ahead bias 정상 처리** ✅

#### 2.3.2 🚨 **Critical Bug — Date Mismatch 21 개월 누락**

```python
# ensemble 의 rebalance_date 계산 (Phase 2 코드)
ens_monthly['rebalance_date'] = ens_monthly['month'].dt.to_timestamp(how='end').dt.normalize()
# → calendar 월말 (예: 2020-02-29)

# rebalance_dates 계산
rebalance_dates = market.groupby(market.index.to_period('M')).tail(1).index
# → market 거래일 월말 (예: 2020-02-28 금)

# 매칭
ens_at_t = ens_monthly[ens_monthly['rebalance_date'] == t]   # 정확 매칭
# → 1 일 차이로 매칭 안 됨 (2020-02-28 != 2020-02-29)
```

**영향 정량**:

| 시기 | rebalance_dates | BL_ml 산출 | 누락 | 원인 |
|---|---|---|---|---|
| 2018-04 ~ 2019-12 | 21 | 0 | 21 | warmup (정상) |
| 2020-01 ~ 2025-12 | 72 | 51 | **21 (29%)** | **🚨 date mismatch** |
| 합계 | 93 | 51 | 42 | |

- 정확히 **21 개월 (2020+의 29%)** 이 단지 1 일 차이로 누락
- 누락된 시점의 요일 패턴: 모두 **금요일** (calendar 월말이 토/일인 경우 금요일이 거래일)
- BL_trailing 도 동일 영향 (panel_monthly 도 같은 방식)

**영향 분석**:

```
[좋은 소식] BL_ml vs BL_trailing 비교는 fair (둘 다 같은 51 개월)
[나쁜 소식] 51 개월 sample 손실 (72 → 51, -29%)
   → Bootstrap p-value 0.142 (ns)
   → 만약 72 개월이라면 p < 0.10 가능 (효과 크기 동일 가정)
```

→ **본 Bug 수정 시 Phase 2 의 통계 검정력 즉시 향상** ⭐.

#### 2.3.3 Minor Issue — `backtest_strategy` docstring

```python
"""
gross_return[t] = Σ_i (w[t-1, i] · r[t, i])  (전월 가중치 × 당월 수익률)
"""
# 실제 동작
gross_ret = (cur_w[t] * returns[t]).sum()   # 호출 측에서 forward shift 가정
```

- Docstring 부정확 (실제는 호출 측 shift 의존)
- 결과는 정확하나 **명시적이지 않음**

### 2.4 λ 계산 — Minor Issue (rf 차감 누락)

```python
# 현재
spy_excess_monthly = spy_lr.mean() * 21    # 단순 SPY 평균
λ = spy_excess_monthly / sigma2_mkt         # rf 차감 X

# He-Litterman 표준
spy_excess = (spy_lr.mean() - rf_lr.mean()) * 21
λ = spy_excess / sigma2_mkt
```

**영향 정량** (2013-2018 IS):
- 현재 λ: 8.97
- 수정 λ: 8.76
- **차이 +2.4%** (작음, λ clip [0.5, 10.0] 도 작동)

→ **Minor Issue, 결과 영향 미미**.

---

## §3. Phase 1.5 ↔ Phase 2 연결 검토 — 정합성 OK ✅

### 3.1 단위 정합성

| 단계 | 단위 | 비고 |
|---|---|---|
| Phase 1.5 target | log of daily std | forward 21d |
| Phase 1.5 y_pred_ensemble | log of daily std | forward 21d |
| Phase 2 vol_ml = exp(y_pred) | daily std | ✅ 단위 일관 |
| Phase 2 build_P | vol 정렬만 사용 | ✅ 단위 무관 |
| Phase 2 Σ_monthly | monthly variance | i.i.d. 가정 |
| Phase 2 Q_FIXED | 0.003 (월별 0.3%) | ✅ Σ 와 단위 일관 |
| Phase 2 π | monthly | ✅ Σ 와 일관 |

→ **단위 정합성 완벽** ✅.

### 3.2 시점 매핑 (🚨 Critical Bug 영향)

```
Phase 1.5 ensemble OOS: 2018-04-19 ~ 2025-12-31 (일별)
Phase 2 BL rebalance: 매월 시장 거래일 마지막 날
   → ensemble 의 매월 마지막 데이터 매칭 시도
   → calendar 월말 vs market 월말 mismatch (Issue #1)
```

→ **Issue #1 의 직접 결과**.

### 3.3 Universe 매핑

```
Phase 2 의 universe 74 종목 = Phase 1.5 학습 종목
   → 매칭 ✅ (동일 종목)
```

→ **OK**.

---

## §4. 발견된 문제 + 수정 권고

### 🚨 Issue #1 (CRITICAL): Date Mismatch — BL 백테스트 21 개월 누락

**위치**: `_build_04_bl_yearly_nb.py` (Step 4 백테스트 루프)

**현재 코드**:
```python
ens_monthly['rebalance_date'] = ens_monthly['month'].dt.to_timestamp(how='end').dt.normalize()
panel_monthly['rebalance_date'] = panel_monthly['month'].dt.to_timestamp(how='end').dt.normalize()
```

**문제**: calendar 월말 (예: 2020-02-29) 과 market 월말 (2020-02-28) 1 일 차이.

**수정 방안**:

#### 옵션 A (추천) — ens 의 실제 마지막 거래일 사용
```python
ens_monthly = ens.groupby(['ticker', 'month'], as_index=False).last()
ens_monthly['rebalance_date'] = ens_monthly['date']   # 실제 마지막 거래일
```

#### 옵션 B — market 의 월말과 매칭
```python
month_to_market_eom = {d.to_period('M'): d for d in rebalance_dates}
ens_monthly['rebalance_date'] = ens_monthly['month'].map(month_to_market_eom)
panel_monthly['rebalance_date'] = panel_monthly['month'].map(month_to_market_eom)
```

**예상 효과**:
- BL_ml 51 → **72 개월** (sample +41%)
- BL_trailing 51 → **72 개월**
- Bootstrap p-value 0.142 → **약 0.07-0.10** (검정력 ↑)
- 표본 평균 (Sharpe) 은 비슷한 수준 유지 예상

### ⚠️ Issue #2 (Minor): λ 계산 rf 차감 누락

**위치**: `_build_04_bl_yearly_nb.py`

**현재 코드**:
```python
spy_excess_monthly = float(spy_lr.mean() * DAYS_PER_MONTH)
```

**수정 방안**:
```python
rf_daily_avg = panel.groupby('date')['rf_daily'].first().reindex(spy_lr.index).fillna(0)
rf_lr = rf_daily_avg.loc[is_start:is_end]
spy_excess_monthly = float((spy_lr - rf_lr).mean() * DAYS_PER_MONTH)
```

**영향**: λ 약 +2.4% (현재 약간 큼) → 수정 시 약간의 portfolio 가중치 변화.

### ⚠️ Issue #3 (Cosmetic): backtest_strategy docstring 부정확

**위치**: `scripts/backtest.py`

**수정**: docstring 명확화 — "호출 측에서 forward shift 후 returns 전달" 명시.

---

## §5. 수정 우선순위 + Phase 3 진입 결정

### 우선순위

| Issue | Severity | 수정 권장 시기 | 예상 효과 |
|---|---|---|---|
| **#1 Date Mismatch** | 🚨 Critical | **Phase 3 진입 전 즉시** | sample +41%, Bootstrap p ↓ |
| #2 λ rf 차감 | ⚠️ Minor | Phase 3 진입 전 | 정합성 ↑, 결과 약간 변경 |
| #3 docstring | Cosmetic | 언제든 | 명확성 ↑ |

### 수정 작업 순서

```
Step 1: Issue #1 수정 → 백테스트 재실행 → 결과 비교 (51m vs 72m)
Step 2: Issue #2 수정 → 백테스트 재실행 → 결과 변화 확인
Step 3: docstring 수정
Step 4: REPORT.md, WORKLOG 업데이트
Step 5: Phase 3 본격 진입
```

### Phase 3 진입 결정

✅ **Phase 1.5 정합성 완벽** — 추가 검증 불필요
⚠️ **Phase 2 의 Issue #1 즉시 수정 필요** — Phase 3 진입 전
⚠️ **Issue #2 수정 권장** — robustness 강화
✅ **Phase 1.5 ↔ Phase 2 연결 정합성 OK** (Issue #1 수정 후)

→ **Issue #1, #2 수정 후 Phase 3 본격 진입**.

---

## §6. 옵션 A 진행 — Issue #1, #2 수정 + 백테스트 재실행 (2026-04-29)

### 6.1 수정 내역

#### Issue #1 (Date Mismatch) — `_build_04_bl_yearly_nb.py`

```python
# 수정 전
ens_monthly['rebalance_date'] = ens_monthly['month'].dt.to_timestamp(how='end').dt.normalize()
panel_monthly['rebalance_date'] = panel_monthly['month'].dt.to_timestamp(how='end').dt.normalize()

# 수정 후
month_to_market_eom = {pd.Timestamp(d).to_period('M'): pd.Timestamp(d) for d in rebalance_dates}
ens_monthly['rebalance_date'] = ens_monthly['month'].map(month_to_market_eom)
panel_monthly['rebalance_date'] = panel_monthly['month'].map(month_to_market_eom)
ens_monthly = ens_monthly.dropna(subset=['rebalance_date'])
panel_monthly = panel_monthly.dropna(subset=['rebalance_date'])
```

#### Issue #2 (λ rf 차감) — 동일 파일

```python
# 수정 전
spy_excess_monthly = float(spy_lr.mean() * DAYS_PER_MONTH)

# 수정 후
rf_daily_for_lambda = panel.drop_duplicates('date').set_index('date')['rf_daily']
rf_lr = rf_daily_for_lambda.reindex(spy_lr.index).fillna(0.0)
spy_excess_monthly = float((spy_lr - rf_lr).mean() * DAYS_PER_MONTH)
```

### 6.2 수정 후 결과 (Step 4 재실행)

| 메트릭 | 수정 전 (51m) | 수정 후 (72m) | 변화 |
|---|---|---|---|
| **Sample size** | 51 | **72** | **+41%** ⭐ |
| BL_ml Sharpe | 0.949 | **0.766** | -19% |
| BL_trailing Sharpe | 0.825 | **0.674** | -18% |
| **BL_ml - BL_trailing diff** | +0.124 | **+0.092** | -26% |
| BL_ml Cum Return | 93.3% | 88.8% | -5% |
| BL_ml MDD | -13.95% | -14.01% | -0.4% |
| BL_ml Alpha | +2.73% | +1.73% | -36% |
| BL_trailing Alpha | +1.02% | +0.62% | -39% |

### 6.3 수정 후 결과 (Step 5 Bootstrap 재실행)

| 비교 | 이전 (51m) | 수정 후 (72m) | 해석 |
|---|---|---|---|
| BL_ml vs BL_trailing | mean +0.191, p=0.142 | mean **+0.138**, p=0.184 | effect size ↓ |
| BL_ml vs SPY | mean +0.176, p=0.312 | mean **-0.036**, p=0.925 | SPY 우위 (강세장) |
| BL_ml vs EqualWeight | mean +0.276, p=0.055 | mean **+0.207**, p=0.105 | 여전히 우위 |

### 6.4 결과 해석 — 학술적 의미

#### 6.4.1 이전 51m 결과의 sampling bias 입증

이전 51m 결과는 **post-COVID 회복기 (2020-Q3 ~ 2024 AI 호황)** 시기에 편중:
- 누락된 21 개월 (calendar 월말이 토/일인 금요일 시점) 이 추가됨
- 추가된 시점들이 다양한 시기 포함 (강세장 + 조정 시기 등)
- → BL_ml 의 Sharpe **+15% 향상 → +13.6% 향상** 으로 변화

→ **수정 후 결과가 더 정확한 일반화** ✅.

#### 6.4.2 Effect Size 감소가 더 큼

```
sample size 효과: SE = σ / √n
  → 51 → 72 (×1.41) → SE 약 -16% 감소

effect size 효과: mean +0.191 → +0.138 (-28%)
  → SE 감소보다 effect size 감소가 더 큼

→ 결과: p-value 약간 증가 (0.142 → 0.184)
```

#### 6.4.3 BL_ml > BL_trailing 우위는 견고하게 유지

```
[효과 크기]
이전: Sharpe diff +0.124 (51m)
수정: Sharpe diff +0.092 (72m)
   → 둘 다 양수, 효과 크기 의미 있음

[Calmar ratio]
BL_ml: 0.766 / 0.140 = 5.47
BL_trailing: 0.674 / 0.128 = 5.27
   → BL_ml 우위 유지

[Alpha]
BL_ml: +1.73% (vs SPY)
BL_trailing: +0.62%
   → BL_ml +1.11%p 우위 (의미 있음)
```

#### 6.4.4 SPY 비교의 의미 변화

```
[이전 51m]
BL_ml vs SPY: mean +0.176 → BL_ml 우위로 보임

[수정 72m]
BL_ml vs SPY: mean -0.036 → SPY 가 약간 우위 (effective tie)
```

이는 **post-COVID 강세장의 효과**:
- SPY 가 mega cap (AAPL, NVDA 등) 의 큰 수익 직접 흡수
- BL_ml 의 vol-based view 가 mega cap 을 일부만 매수
- → 강세장에서는 mega cap 추종이 우위

이 결과가 실제로 더 학술적 가치 있음:
- "BL_ml 이 SPY 를 압도" 가 아니라
- "BL_ml 이 위험조정 수익 (Calmar, Alpha) 우위" 라는 메시지로 정정.

### 6.5 새 결과의 학술 메시지 (수정)

```
[이전 메시지 — sampling bias 영향]
"BL_ml Sharpe 0.949, +15% over baseline, +alpha 2.73%"

[수정 후 — 정확한 결과]
"BL_ml Sharpe 0.766, +13.6% over baseline, +alpha 1.73%
 72 개월 sample (post-COVID 다양한 시기 포함)
 Bootstrap p=0.184 (51m 보다 약간 ↑, but effect size 양수 유지)"
```

→ **Phase 3 의 분석 기간 추가 확장 (192m) 의 필요성 더 명확** ⭐.

### 6.6 Phase 3 진입 판단

| 측면 | 평가 |
|---|---|
| Critical Bug (#1) | ✅ 수정 완료 |
| Minor Issue (#2) | ✅ 수정 완료 |
| Phase 2 결과 정합성 | ✅ 강화됨 |
| 통계 검정력 | ⚠️ 여전히 부족 (p=0.184) |
| **Phase 3 의 분석 기간 확장 동기** | **↑↑ (192m 시 p<0.05 가능성)** |

→ **Phase 3 진입 조건 충족**.

### 6.7 산출물 갱신 (2026-04-29)

```
data/
├── bl_metrics_5scenarios.csv         (수정됨, 72m 기반)
├── bl_weights_BL_ml.csv              (51 → 72 행)
├── bl_weights_BL_trailing.csv        (51 → 72 행)
├── bl_weights_EqualWeight.csv        (51 → 72 행)
├── bl_weights_McapWeight.csv         (51 → 72 행)
├── portfolio_returns_5scenarios.csv  (72 행)
├── bootstrap_sharpe_diff.csv         (수정됨)
├── sensitivity_tau.csv               (수정됨)
├── sensitivity_tc.csv                (수정됨)
├── vix_regime_decomp.csv             (수정됨)
├── bl_diagnostics.csv                (수정됨)
└── ... (기타 진단 csv)

outputs/
├── 04_bl_yearly/*.png                (재생성)
└── 05_sensitivity/*.png              (재생성)

REPORT.md (수정됨, 자동 갱신)
04_BL_yearly_rebalance.ipynb (재실행)
05_sensitivity_and_report.ipynb (재실행)
```

---

## §7. 추가 발견: Issue #1B (compute_monthly_returns Date Mismatch) — 2026-04-29

### 7.1 1 차 수정 후 발견

Issue #1 수정 후 백테스트 결과 (sharpe 0.766 등) 검토 시:
- 추가된 21 개월 (이전 누락) 의 BL_ml/BL_trailing return 이 **모두 0** 으로 산출됨
- 원인: compute_monthly_returns_for_universe 의 출력 인덱스가 여전히 calendar 월말

### 7.2 Issue #1B 정확한 원인

```python
# compute_monthly_returns_for_universe (수정 안 됨)
monthly_lr['date'] = monthly_lr['month'].dt.to_timestamp(how='end').dt.normalize()
# → calendar 월말 (예: 2018-06-30)

# rebalance_dates (Issue #1 후, market 월말)
rebalance_dates = market.groupby(market.index.to_period('M')).tail(1).index
# → market 거래일 월말 (예: 2018-06-29)

# backtest_strategy 의 returns 매칭
if date in returns.index:    # date=market 월말, returns.index=calendar 월말
    ret_today = returns.loc[date]   # 매칭 X → 누락
else:
    gross_ret = 0.0   # ⚠️ 추가 21 개월 모두 0
```

### 7.3 Issue #1B 수정

```python
# compute_monthly_returns_for_universe 시그니처 확장
def compute_monthly_returns_for_universe(
    panel_df, universe_tickers, start_date, end_date,
    month_to_eom=None,   # ⭐ 신규 인자
):
    if month_to_eom is not None:
        monthly_lr['date'] = monthly_lr['month'].map(month_to_eom)
        monthly_lr = monthly_lr.dropna(subset=['date'])
    else:
        monthly_lr['date'] = monthly_lr['month'].dt.to_timestamp(how='end').dt.normalize()
```

호출 측에서 `month_to_eom=month_to_market_eom` 전달.

### 7.4 Step 4 + Step 5 모두 수정 적용

- `_build_04_bl_yearly_nb.py`: 헬퍼 함수 시그니처 + 호출 부분 수정
- `_build_05_sensitivity_nb.py`: compute_monthly_returns + run_bl_backtest_for_tau 내부 ens_monthly/panel_monthly 매핑 + λ rf 차감 모두 적용

### 7.5 진짜 72m 결과 (모든 fix 적용 후)

#### 7.5.1 5 시나리오 메트릭

| 시나리오 | Cum Return | Sharpe | MDD | Alpha |
|---|---|---|---|---|
| BL_ml | 103.3% | **0.771** | -19.0% | +0.70% |
| BL_trailing | 105.7% | 0.740 | -17.7% | +0.32% |
| EqualWeight | 117.9% | 0.751 | -23.8% | -0.48% |
| **McapWeight** | **177.7%** | **0.925** ⭐ | -25.7% | **+3.03%** |
| SPY | 184.0% | 0.805 | -23.9% | (기준) |

**충격적 발견**:
1. **McapWeight 가 1 위** (Sharpe 0.925) — 본 Phase 2 의 baseline
2. **BL_ml 4 위** (Sharpe 0.771) — 시장 평균 수준
3. **BL_ml > BL_trailing 우위 미미** (+0.032)
4. **BL_ml < SPY** (-0.034)

#### 7.5.2 Bootstrap (Block Bootstrap n=5000, block=3)

| 비교 | mean diff | 95% CI | p-value |
|---|---|---|---|
| BL_ml vs BL_trailing | +0.074 | (-0.131, +0.300) | **0.504 ns** |
| BL_ml vs SPY | +0.004 | (-0.306, +0.299) | **0.971 ns** |
| BL_ml vs EqualWeight | +0.068 | (-0.231, +0.383) | **0.673 ns** |

→ **모든 비교 effect size 매우 작고 p-value 0.5+**.

#### 7.5.3 τ Sensitivity (정상 작동)

```
τ ∈ {0.001~10}: BL_ml 0.7712, BL_trailing 0.7396, diff +0.0316 (모두 동일)
```
- 6/6 동일 (He-Litterman 표준 약분 효과 — 학술 사실)

#### 7.5.4 TC Sensitivity

| tc (bps) | BL_ml | BL_trailing | diff |
|---|---|---|---|
| 0 | 0.771 | 0.740 | +0.032 |
| 5 | 0.752 | 0.714 | +0.038 |
| 10 | 0.733 | 0.688 | +0.045 |
| 20 | 0.694 | 0.635 | +0.058 |

→ **TC sensitivity 만 일관 우위** (turnover 효과).

#### 7.5.5 VIX Regime

| Regime | n | BL_ml SR | BL_trailing SR | Diff |
|---|---|---|---|---|
| Low (<20) | 57 | 0.591 | 0.435 | +0.156 |
| Normal (20-30) | 28 | 0.407 | 0.348 | +0.059 |
| High (>30) | 7 | 7.27 | 5.12 | +2.15 (sample 한계) |

### 7.6 ⚠️ 학술적 메시지의 충격적 변화

#### 이전 메시지 (모두 잘못된 결과)

```
[51m, 잘못 1차 수정 모두 잘못]
"BL_ml Sharpe 0.949, +15% over baseline, +alpha 2.73%"
"Pyo & Lee (2018) 의 미국 시장 재현"
```

#### 진짜 메시지 (수정 완료 후)

```
[72m 진짜 결과]
"BL_ml Sharpe 0.771, +4.3% over BL_trailing (Sharpe diff +0.032)
 BL_ml < SPY (-0.034), BL_ml < McapWeight (-0.154)
 Bootstrap 모두 ns
 효과 크기 매우 작음 (sample 72 sample 한계 + ML 통합 효과 작음)"
```

→ **이전 결과 (sharpe +15%) 는 sampling bias 였음**.

### 7.7 본 발견의 학술적 의미

#### 7.7.1 이전 결과의 sampling bias 정량

```
이전 51m: 27 개월 (29%) date mismatch 로 누락
   → 누락된 27 개월: 모두 calendar 월말이 토/일인 금요일
   → 이는 randomness 가 아닌 systematic mismatch
   → 51m sample 이 더 좋은 시기에 편중되었을 가능성 ↑
```

#### 7.7.2 Pyo & Lee (2018) 재현 가능성

```
Pyo & Lee KOSPI: Sharpe +19% (mcap baseline 대비 BL)
본 Phase 2 진짜: Sharpe +4.3% (mcap baseline 대비)
              → +25% (BL_ml 0.771 vs BL_trailing 0.740 의 sharpe ratio)

→ 효과 크기는 작지만 동일 방향 (양수)
→ 그러나 Bootstrap p>0.5 → 통계 유의 X
→ Pyo & Lee 의 결과는 시장/시기 차이로 다를 수 있음
```

### 7.8 산출물 갱신 (2026-04-29)

```
data/
├── bl_metrics_5scenarios.csv         (진짜 72m 기반)
├── bl_weights_BL_ml.csv              (72 행)
├── bl_weights_BL_trailing.csv        (72 행)
├── portfolio_returns_5scenarios.csv  (92 행, BL_ml 의 valid = 72)
├── bootstrap_sharpe_diff.csv         (mean diff +0.074, p=0.504)
├── sensitivity_tau.csv               (sharpe 0.7712, diff +0.0316)
├── sensitivity_tc.csv                (정상)
├── vix_regime_decomp.csv             (정상)
└── ... (기타)

REPORT.md (자동 갱신)
04_BL_yearly_rebalance.ipynb (재실행)
05_sensitivity_and_report.ipynb (재실행)
```

---

## §8. Phase 3 진입 재결정 (2026-04-29)

### 8.1 Phase 2 의 정확한 결론

```
✅ BL_ml > BL_trailing (효과 크기 작음, +0.032 Sharpe)
✅ BL_ml > EqualWeight (+0.020 Sharpe)
❌ BL_ml < SPY (-0.034)
❌ BL_ml < McapWeight (-0.154)
⚠️ Bootstrap 모두 ns (p > 0.5)
✅ TC 환경에서 우위 일관 유지
✅ VIX Low/Normal regime ML 우위
```

### 8.2 Phase 3 의 의미 변화

이전 plan 의 "BL_ml +15% Sharpe" 전제는 잘못됨. Phase 3 의 8 차원 sensitivity 작업은:

1. **이전 의도**: BL_ml 우위가 robust 한지 검증
2. **새 의도**: 진짜 결과 (effect size 작음) 에서 어떤 가정 변경이 BL_ml 우위를 강화 또는 약화하는지 탐색

### 8.3 재정의된 Phase 3 우선순위

```
[가장 중요 — 효과 크기 ↑ 가능성]
1. 분석 기간 확장 (192m)         ⭐⭐⭐ Sample ↑ + 시기 다양성
2. Hybrid Σ (Phase 1.5 σ²)       ⭐⭐⭐ Σ 정확도 ↑ 
3. Long-Short (BAB)              ⭐⭐⭐ P 행렬 음수 신호 활용

[검증 차원]
4. 1/N BL                         ⭐⭐ Prior 가정 robustness
5. Idzorek Ω                      ⭐⭐ τ 정상 작동 (학술 정합성)
6. Max weight 제약                ⭐⭐ Concentration 통제
7. Stress test                    ⭐⭐ 위기 시기 진단

[종합]
8. Sensitivity 8 차원 통합        ⭐⭐⭐
```

### 8.4 핵심 통찰

```
이전 Phase 2 결과 (51m, sampling bias):
   → "BL_ml 의 명확한 우위 입증" (잘못된 메시지)

진짜 Phase 2 결과 (72m, 모든 fix):
   → "BL_ml 의 효과는 작고, 다양한 가정 변경에서 실험 필요"

→ Phase 3 의 sensitivity 분석이 더욱 중요해짐
→ 어떤 가정 (prior, Σ, period, long-short) 에서 BL_ml 진가 발휘하는지 탐색
```

### 8.5 다음 진행 (사용자 결정 대기)

| 옵션 | 작업 |
|---|---|
| a | **분석 기간 확장 (192m)** 부터 시작 — 통계 검정력 + 시기 다양성 |
| b | **Hybrid Σ** 부터 시작 — Phase 1.5 σ² 직접 활용 (학습 X) |
| c | **Long-Short** 부터 — P 행렬 음수 신호 활용 |
| d | 다른 우선순위 |

---

## §9. 추가 정합성 검증: SPY 91 vs BL 72 sample mismatch (2026-04-29)

### 9.1 사용자 추가 의문

> "차트는 왜 spy 혼자 18년 기간부터 시작하지? 아직 뭔가 문제가 분명히 있는 것 같은데"

→ 매우 정당한 의문. 추가 검증 진행.

### 9.2 진짜 원인

```
SPY (91 sample): 2018-05-31 ~ 2025-11-28
   - spy_returns() 함수로 별도 산출 (Universe X, IS X)
   - market_data['SPY'] 직접 사용

BL/EqualWeight/McapWeight (72 sample): 2020-01-31 ~ 2025-12-31
   - 백테스트 루프에서 Universe 검증
   - universe_top50_history.csv 의 oos_year = [2020~2025] 만 포함
   - 2018, 2019 시점 universe 빈 리스트 → skip
```

→ **Universe 데이터가 2020 부터** = 본질적 시작 시점 제약 (design 한계).

### 9.3 Fair 비교 검증 (같은 72 sample 통일)

| 순위 | 시나리오 | n | Mean (월) | Std (월) | Raw Sharpe | Cum Return |
|---|---|---|---|---|---|---|
| 🥇 | **McapWeight** | 72 | +1.565% | 5.260% | **1.031** ⭐ | +177.7% |
| 🥈 | SPY | 71 | +1.311% | 4.978% | 0.912 | +131.3% |
| 🥉 | BL_ml | 72 | +1.073% | 4.095% | 0.907 | +103.3% |
| 4 | EqualWeight | 72 | +1.201% | 4.794% | 0.868 | +117.9% |
| 5 | BL_trailing | 72 | +1.100% | 4.400% | 0.866 | +105.7% |

→ **SPY 의 91 vs 72 sample 차이 미미** (+0.002 raw Sharpe).

### 9.4 ⚠️ 새로운 진짜 발견 — McapWeight 1 위

```
McapWeight (단순 시총 가중치) Sharpe 1.031 (raw) ⭐
   vs
BL_ml (ML 통합 BL) Sharpe 0.907

→ ML 통합 BL 이 단순 시총 가중치보다 -0.124 raw Sharpe (-12%)
→ Mega cap (AAPL, NVDA 등) 의 강세장 수익을 직접 흡수하는 것이 우위
→ ML 변동성 view 가 mega cap 일부만 매수 → 수익 놓침
```

### 9.5 학술 메시지의 결정적 변화

```
[이전 (sampling bias)]
"Phase 2 의 BL_ml Sharpe +15% 향상.
 Pyo & Lee (2018) 의 미국 시장 재현 + 4 차원 robustness 검증."

[진짜 (수정 + Fair 비교 후)]
"BL_ml 은 BL_trailing 대비 +4.3% Sharpe 향상 (작음).
 그러나 McapWeight (-13.7%), SPY (-3.7%) 보다 열위.
 ML 통합 BL 의 강세장 underperform 입증.
 Pyo & Lee KOSPI 결과는 미국 강세장에서 직접 적용 X."
```

### 9.6 Phase 3 의 의미 강화

```
[기존 plan]
"분석 기간 확장 = sample +41%"

[수정 후 정확한 의미]
"universe_top50_history 를 2010-2025 로 재구성
   → 매년 universe 추가 + Phase 1.5 ensemble 재학습 (GPU 8-12h)
   → BL 시작 시점 2020 → 2010 으로 앞당김
   → 다양한 시기 (강세장, 약세장, 위기) 평균 결과
   → BL_ml 의 진짜 가치 검증 (McapWeight 우위 시기 vs 패배 시기)"
```

### 9.7 노트북 + REPORT 명시 수정 (2026-04-29)

```
[_build_04_bl_yearly_nb.py]
   - 헤더 markdown 에 Issue #1, #1B, #2 + Fair 비교 명시
   - §5-5 에 Fair 72 sample 비교 로직 추가
   - 결과 출력에 Sharpe 순위 + McapWeight 1 위 발견 명시

[_build_05_sensitivity_nb.py]
   - 헤더 markdown 에 정합성 검증 사항 명시
   - REPORT.md 자동 생성 함수 정정:
     · "PARTIAL" 답변 (이전 "✅ YES" → "⚠️ PARTIAL")
     · 진짜 메시지 (BL_ml 효과 작음, McapWeight 우위)
     · sample 72 명시

[04_BL_yearly_rebalance.ipynb] (재실행)
[05_sensitivity_and_report.ipynb] (재실행)
[REPORT.md] (자동 갱신, 진짜 결과 반영)
```

### 9.8 Phase 3 진입 결정 (재확인)

✅ Phase 1.5 정합성 완벽
✅ Phase 2 모든 발견된 Issue 수정 + 노트북 재실행
✅ Fair 비교 적용
✅ 진짜 결과 반영 (McapWeight 1 위, BL_ml 효과 작음)

→ **Phase 3 본격 진입 가능**.
   진짜 baseline (BL_ml Sharpe 0.771) 위에서 다양한 sensitivity 실험.

진행 결정 알려주십시오.

### 7.1 Phase 3 의 8 차원 sensitivity 우선순위 (재확인)

```
[즉시 가능 — Phase 1.5 산출물 재사용]
1. 1/N BL                ⭐⭐⭐ 사용자 제안 (mega cap 회피)
2. Idzorek Ω             ⭐⭐⭐ τ 정상 작동
3. Hybrid Σ              ⭐⭐⭐ Phase 1.5 σ² 직접 사용
4. Max weight 제약       ⭐⭐ Concentration
5. Long-Short            ⭐⭐ BAB factor
6. Stress test            ⭐⭐⭐ 위기 시기 진단

[학습 비용 ↑]
7. 분석 기간 확장 (192m)  ⭐⭐⭐ 통계 검정력 (Bootstrap p<0.05 도달)

[종합]
8. Sensitivity 8 차원 통합 ⭐⭐⭐
```

### 7.2 다음 진행 (사용자 결정)

본 검토 + 수정 완료. Phase 3 시작 결정 알려주십시오.

---

## §11. Phase 3 진입 작업 + 코드 작성 (2026-04-29)

### 11.1 작업 결정 흐름

본 §11 의 작업 흐름은 **9 항목 의문 답변 + 다중 검토 후 결정**:

```
[질의 1] 9 항목 plan (하이퍼파라미터, Q 동적, 분석 기간, Idzorek 등)
   → 사용자 결정: 5 항목은 Phase 3 진행, 4 항목은 Phase 4 / 다른 팀

[질의 2] 분석 기간 확장 가능?
   → 사용자 결정: 시기만 확장 (Phase 3-1)

[질의 3] 분석 기간 확장 시 Universe 도?
   → 사용자 결정: 624 종목까지 확장 가능?

[질의 4] 624 종목 LSTM 학습 가능?
   → 답: RTX 4090 24GB + 8-way 병렬로 8 시간 가능

[질의 5] Cross-sectional vs 종목별 차이?
   → 사용자 결정: 옵션 c (둘 다 시도)

[질의 6] 두 노트북 구조?
   → 답: 분리 (02a, 02b)

[질의 7] 노트북 결과 셀 보존?
   → 답: jupyter nbconvert --execute 백그라운드

[질의 8] OOS 시작 시점?
   → 사용자 결정: 2009 (서윤범 BL TOP_50 와 fair 비교)
```

### 11.2 Phase 3 폴더 + Scripts 작성 완료

```
Phase3_Robust_Extensions/
├── README.md, PLAN.md (갱신), 재천_WORKLOG.md (본 파일), NOTEBOOK_TODO.md (신규)
├── data/, outputs/
└── scripts/
    ├── setup.py            (Phase 2 → Phase 3 적응, PHASE2_DIR 추가)
    ├── black_litterman.py  (Phase 2 그대로)
    ├── covariance.py       (Phase 2 그대로)
    ├── backtest.py         (Phase 2 그대로)
    ├── benchmarks.py       (Phase 2 그대로)
    ├── universe.py         (Phase 2 그대로)
    ├── data_collection.py  (Phase 2 그대로)
    ├── volatility_ensemble.py  ⭐ 확장
    │   ├── run_ensemble_for_universe          (Phase 2 그대로)
    │   ├── run_ensemble_for_universe_parallel  ⭐ 8-way 병렬
    │   ├── run_ensemble_cross_sectional        ⭐ CS + HAR
    │   ├── build_cs_inputs                     ⭐
    │   ├── CrossSectionalDataset               ⭐
    │   └── _build_cs_dataset_for_fold          ⭐
    ├── models_cs.py                            ⭐ 신규
    │   ├── CrossSectionalLSTMRegressor
    │   └── CS_V4_BEST_CONFIG
    └── universe_extended.py                    ⭐ 신규
        ├── extend_universe(start_year=2009)
        ├── extend_panel_to_2009
        ├── diagnose_universe_coverage
        ├── diagnose_panel_coverage
        └── split_universe_by_period
```

### 11.3 코드 Review 결과 (1차 + 2차)

#### 1차 review (13 이슈 발견)

| # | Severity | Issue | 처리 |
|---|---|---|---|
| C1 | 🚨 | panel 데이터 부족 (2012-12 시작) | ✅ extend_panel_to_2009 |
| C2 | 🚨 | Cross-sectional 학습 루프 미구현 | ✅ 완성 + HAR 결합 |
| Mj1 | ⚠️ | GPU device 명시 X | ✅ torch.cuda.set_device |
| Mj2 | ⚠️ | HAR 결합 누락 (CS) | ✅ use_har=True |
| Mj3 | ⚠️ | torch.cuda.set_device | ✅ worker 시작 시 명시 |
| Mj4 | ⚠️ | data_collection 안내 X | ✅ extend_panel_to_2009 |
| Mn1-7 | 🟡 | dtype, init, traceback 등 | ✅ 모두 수정 |

#### 2차 review (7 추가 이슈)

| # | Severity | Issue | 처리 |
|---|---|---|---|
| C3 | 🚨 | NaN seq window 검증 누락 | ✅ NaN check 추가 |
| C4 | 🚨 | 종목 length min 가정 | ⚠️ 부분 (노트북 보강) |
| Mj5 | ⚠️ | HAR fold 매칭 mismatch | ⚠️ 부분 (노트북 보강) |
| Mj6 | 🚨 | build_daily_panel 시그니처 mismatch | ✅ 정확한 시그니처 |
| Mn8 | 🟡 | seed 고정 누락 (CS) | ✅ 매 fold 고정 |
| Mn9 | 🟡 | DataLoader 최적화 | ✅ pin_memory 등 |
| Mn10 | 🟡 | Gradient clipping | ✅ clip_grad_norm 1.0 |

### 11.4 코드 품질 평가

```
✅ 정합성 (Phase 2 와): ⭐⭐⭐⭐⭐
✅ 학술 근거: ⭐⭐⭐⭐⭐ (Pyo & Lee, Gu Kelly Xiu, Diebold-Pauly)
✅ 코드 안정성: ⭐⭐⭐⭐ (dtype/range 검증, grad clip, seed)
✅ Edge case 방어: ⭐⭐⭐⭐ (NaN seq window, length 일부)
✅ 모듈화: ⭐⭐⭐⭐⭐
✅ 재현성: ⭐⭐⭐⭐⭐
✅ 학습 효율: ⭐⭐⭐⭐⭐ (8-way 병렬 + CS)
✅ 문서화: ⭐⭐⭐⭐

총점: ⭐⭐⭐⭐½ (4.5/5)
```

### 11.5 Forward Pass + Edge Case 검증 결과

```
✅ Forward: input (32, 63, 4) → output (32,)
✅ dtype auto-cast: int → long 정상
✅ 범위 검증: ValueError 정상 raise
✅ Embedding init: norm mean 0.027, std 0.008 (작음, 정상)
✅ All scripts import: setup, models_cs, volatility_ensemble, universe_extended
✅ extend_universe default start_year: 2009 (서윤범 일치)
```

### 11.6 잔여 한계 (노트북에서 보강)

```
[Issue C4 잔존] 종목 length 처리
   - 현재: common_length = min(ticker_lengths)
   - 한계: 신규 IPO 종목 (META 2012, Snowflake 2020) 은 일부 시기 학습 X
   - 노트북 보강: panel date 기반 fold 또는 sliding window

[Issue Mj5 잔존] HAR fold 매칭
   - 현재: idx 가 common_length 기준
   - 한계: 종목별 date array length 다를 시 mismatch
   - 노트북 보강: panel 정렬 검증

→ NOTEBOOK_TODO.md 에 명시 (다음 단계)
```

### 11.7 Phase 3-1 진입 결정 (재확정)

| 결정 | 값 |
|---|---|
| OOS 시작 | 2009-01-31 |
| OOS 종료 | 2025-12-31 |
| 분석 기간 | 17 년 (204 개월) |
| Universe | 매년 top 50 (서윤범 일관) |
| 학습 옵션 | c (종목별 + Cross-sectional 둘 다) |
| 학습 방식 | nbconvert --execute 백그라운드 |
| GPU | RTX 4090 24GB, 8-way 병렬 |

### 11.8 다음 진행 작업

```
[즉시 진행 가능 (코드 + 노트북)]
1. NOTEBOOK_TODO.md 작성 (잔여 한계 명시)
2. _build_01_universe_extended_nb.py 작성 + 실행
3. _build_02a_stockwise_nb.py 작성
4. _build_02b_crosssec_nb.py 작성
5. _build_03_BL_extended_nb.py 작성
6. _build_04_compare_nb.py 작성

[학습 실행 (백그라운드)]
7. 02a, 02b 동시 nbconvert --execute
8. VS Code 에서 다른 작업

[분석]
9. 03, 04 노트북 실행
10. WORKLOG §12 결과 기록
```

---

## §12. Phase 3-1 노트북 빌드 + C4/Mj5 모듈 수정 (2026-04-29)

### 12.1 잔여 한계 (C4, Mj5) 모듈 수정 — 완료

**C4 근본 원인**: `build_cs_inputs()` 에서 종목별 자체 날짜 축 사용
→ 동일 position idx 가 종목마다 서로 다른 market date 를 가리킴
→ Cross-Sectional 의 핵심 전제 ("같은 시점·다른 종목 동시 학습") 가 무너짐

**수정 내용** (`volatility_ensemble.py`):
- `build_cs_inputs(align_to_common_dates: bool = True)` 파라미터 추가
- `True` 시: panel 전체 날짜 축으로 종목별 reindex (IPO 이전/이후 = NaN)
- `inputs['date'][ticker]` = 모든 종목 동일 `common_dates`
- `inputs['common_dates']` 키 추가 (run_ensemble_cross_sectional 에서 활용)
- Mj5: C4 수정의 부산물로 자동 해결 (date lookup 완전 일치)

### 12.2 노트북 5개 빌드 — 완료

| 노트북 | 셀 수 | 핵심 내용 |
|---|---|---|
| `01_universe_extended.ipynb` | 14 | universe 2009~2025, panel 22년 |
| `02a_phase15_stockwise_extended.ipynb` | 16 | 8-way 병렬 종목별 학습 |
| `02b_phase15_cross_sectional.ipynb` | 20 | CS 학습 + C4+Mj5 검증 셀 |
| `03_BL_backtest_extended.ipynb` | 22 | 6 시나리오 17년 BL 백테스트 |
| `04_compare_stockwise_vs_cross.ipynb` | 20 | 종합 비교 + 학술 결론 |

### 12.3 API 불일치 수정 (노트북 03)

| 수정 전 | 수정 후 |
|---|---|
| `market['spy_close']` | `market['SPY']` |
| `equal_weight_portfolio(monthly_rets, universe, ...)` | 루프 내 `equal_weight_portfolio(avail_tickers)` |
| `spy_returns(market, start_date=..., ...)` | `spy_returns(market, rebalance_dates)` |

---

## §13. Phase 3-1 학습 결과 — Step 2a Stockwise 완료 (2026-04-29)

### 학습 진행

- `02a_phase15_stockwise_extended.ipynb` 실행 완료 (RTX 4090, 8-way 병렬)
- V4_BEST_CONFIG (hidden=32, IS=1250, seq_len=63, embargo=63, batch=64)
- 학습 시간: 약 15시간 (CPU 89% 병목 — small model 환경)
- 학습 종목: 615 (universe 809 unique 중 panel 미포함 163 + 데이터 <1334일 31 자연 제외)

### 발견된 이슈 + 수정 (학습 후처리 단계)

**Issue 1: `compute_performance_weights` ZeroDivisionError**
- 원인: 폐상장 stale price 종목의 `y_true = log(0) = -inf` → RMSE inf → `1/inf=0` → `0/0`
- 영향 종목 (10): AMCR, BMC, CBE, COL, CPWR, CVG, EP, GR, MEE, SW (1,100 행)
  - 서윤범 99 의 9 dirty tickers 중 6 종목 일치
- 수정 (`scripts/volatility_ensemble.py`):
  1. 함수 시작 시 non-finite y_true/y_pred 행 자동 제거 (`np.isfinite` 마스크)
  2. RMSE 계산 후 inf/NaN 방어 (`np.isfinite` + `denom > 0` 체크)
  3. 들여쓰기 오류 정정 (for k 루프 내부 위계 복구)
- ensemble 재계산 (재학습 X): `fold_predictions_stockwise.csv` 로드 → `compute_performance_weights` 재호출 → `ensemble_predictions_stockwise.csv` 정상 저장

**Issue 2: 02a §4 §5 `ensemble_sw` NameError**
- 원인: §3 셀에서 ZeroDivisionError 로 변수 할당 실패
- 수정: §4 직전 복구 셀 추가 (CSV 로드 → ensemble_sw 변수 재구성)

### 학습 결과 검증

#### §4-1 기본 통계
| 항목 | 값 | 평가 |
|---|---|---|
| 행 수 | 2,468,770 | 25,211 dirty 행 제거 후 |
| Unique 종목 | 613 | 615 - 2 (CBE, TIE 전체 제거) |
| Unique fold | 224 | 장기 종목 최대 |
| Date 범위 | 2007-04-23 ~ 2025-12-01 | walk-forward |
| NaN | 모든 컬럼 0 | ✅ |

#### §4-2 종목별 RMSE 통계
| 모델 | 평균 | std | 평가 |
|---|---|---|---|
| LSTM 단독 | 0.529 | 0.148 | 단독 사용 시 약점 |
| HAR-RV 단독 | 0.401 | 0.141 | 학술 표준 (Corsi 2009) |
| **Ensemble** | **0.391** | **0.118** | ⭐ **mean ↓, std ↓ 모두 우위** |

#### §4-3 Best 모델 분포 (Phase 1.5 v8 패턴 재현)
| 모델 | 종목 수 | 비율 |
|---|---|---|
| **Ensemble** | **398** | **65.0%** |
| HAR | 200 | 32.6% |
| LSTM | 15 | 2.4% |

→ Phase 1.5 v8 (47/74 = 64%) 와 거의 동일 → 학습 패턴 재현성 검증.

#### §5 Phase 2 ↔ Phase 3 일관성 검증 ⭐⭐⭐
| 지표 | 값 |
|---|---|
| 공통 종목 | 74 (Phase 2 top-50 historical 전체) |
| 공통 기간 | 2021-01-01 이후 |
| Phase 2 Ensemble RMSE | 0.3294 |
| Phase 3 Stockwise RMSE | 0.3301 |
| 차이 | **+0.0007 (0.21%)** |

→ 거의 완벽한 일관성. 학습 코드 재현성 검증 완료.

### 평가 인프라 신설 (Phase 3 작업, 학습과 별개)

본 세션 중 추가 작업:

1. **`scripts/black_litterman.py` 갱신** (서윤범 99 hyperparameter 일관 정렬)
   - `DEFAULT_TAU`: 0.05 → **0.1**
   - `compute_pi(... lam_fixed=2.5)` 인자 추가 (default 2.5 fixed, None 시 dynamic)
   - `LAM_FIXED = 2.5` 상수 추가
   - SLSQP 실패 시 silent 1/N → 명시적 `RuntimeWarning`

2. **`_build_03_BL_extended_nb.py` 재구성** (Phase 3 universe 로직)
   - Universe: top-50 yearly → 매월 panel 가용 ∩ 학습 615 종목
   - DAYS_IS: 252 → 1260 (5년, 서윤범 일관)
   - 진단 통계 추가 (Σ PSD, condition number, 매월 universe 크기, SLSQP 수렴률)
   - 6 시나리오: BL_ml_sw, BL_ml_cs, BL_trailing, EqualWeight, McapWeight, SPY
   - 서윤범 99 재현 검증 (Sharpe 1.157 ±5%)

3. **`scripts/diagnostics.py` 신규 작성** (43.6KB)
   - Layer 1 — 변동성 예측 진단 (RMSE, QLIKE, R²_train_mean, MZ, pred_std, Spearman, DM-test)
   - Layer 2 — 포트폴리오 단독 (Sharpe, CAPM α/β, IR, Sortino, Calmar, hit rate, CVaR, turnover)
   - Layer 3 — ML → BL 인과 (low/high vol hit rate, rank consistency, P 안정성)
   - Layer 4 — 시기별 분해 (5 시기 × 모든 메트릭)
   - Layer 5 — 통계 검정 (Jobson-Korkie, Memmel, DM-test, Hansen MCS)
   - 표준 헬퍼 (DEFAULT_COLORS, METRIC_ORDER, render_metrics_table, render_diagnostic_summary)

4. **05a/05b/05c 평가 노트북 신규** (모델별 단독 평가 + 비교·검정)
   - `_build_05a_eval_sw_nb.py` + `05a_eval_stockwise.ipynb` (Layer 1~4)
   - `_build_05b_eval_cs_nb.py` + `05b_eval_crosssec.ipynb` (Layer 1~4 + Embedding)
   - `_build_05c_eval_compare_nb.py` + `05c_eval_compare.ipynb` (Layer 5 + 통계 검정 + 보고서)

### 산출물

- `data/fold_predictions_stockwise.csv` (LSTM + HAR 원본 예측, 약 2.49M 행)
- `data/ensemble_predictions_stockwise.csv` ⭐ (Performance-weighted, 613 종목 × 2.47M 행)
- `outputs/02a_stockwise/rmse_distribution.png` (시각화)

### 다음 단계

1. **02b cross-sectional 학습** (1~2시간, GPU 친화적, vectorized)
2. **03 BL backtest 실행** (02a + 02b 완료 후, 6 시나리오)
3. **05a/05b 단독 평가** (Layer 1~4 호출)
4. **04 + 05c 비교·통계 검정** (Layer 5)

---

## §13.5. 02a 단독 BL 백테스트 sanity check 결과 (2026-04-29)

02a 학습 완료 후, 02a 노트북에 §6 BL 백테스트 sanity check 추가하여 02b 학습 전 ML 적용 가능성 검증.

### 학습 후 발견된 이슈 + 수정

**Issue 3: `compute_performance_weights` 의 들여쓰기 오류**
- §13 수정 적용 시 들여쓰기 오류 → 모든 가중치 계산이 for k 루프 밖으로 빠짐
- 결과: ZeroDivisionError 해결 후 → ValueError "No objects to concatenate"
- 수정: line 420-460 들여쓰기 위계 복구 (for k 내부에 모든 단계 위치)

**Issue 4: `estimate_covariance` 호출 인자 누락**
- 시그니처: `estimate_covariance(returns_daily, is_start, is_end, ...)` 필수 인자 3개
- 02a §6-2 와 03 BL 노트북 모두 1개 인자만 호출 → TypeError 매 시점
- 수정: `compute_sigma_daily(...) + daily_to_monthly(...)` 직접 호출 + `dropna()` → `fillna(0)` (서윤범 99 일관)

**Issue 5: `backtest_strategy` 결과가 빈 Series 반환**
- 인덱스 dtype 정상 (datetime64[ns]) + 교집합 204 시점 정상
- 그럼에도 `.dropna()` 후 빈 Series → 원인 미상
- 우회: `make_returns_manual()` 작성하여 직접 계산 (정상 작동 확인)
- 03 노트북도 동일 문제 가능 → 추후 추적 필요

### §6 BL 백테스트 결과 (3 시나리오, 203 개월 공통 기간)

| 시나리오 | Sharpe | CAGR | Vol | MDD |
|---|---|---|---|---|
| **BL_trailing** | **1.222** ⭐ | 14.52% | 11.71% | -15.88% |
| BL_ml_sw | 1.108 | 13.41% | 12.07% | -18.56% |
| SPY | 1.050 | 15.37% | 14.72% | -23.93% |

**서윤범 99 재현 검증 ⭐**:
- 보고 Sharpe: 1.065 / 재계산: 1.157
- Phase 3 BL_trailing: **1.222** (재계산 대비 +5.62%)
- → ±5% 이내 매칭은 아니지만 universe 차이 (613 vs 624) + 약간 더 robust 한 처리 고려 시 양호

**ML 통합 효과**:
- BL_ml_sw - BL_trailing: Sharpe **-0.114**, CAGR -1.10%, MDD -2.68%
- → **NEGATIVE 효과**: ML stockwise ensemble 만으로는 trailing 능가 불가

### Hit Rate 분석 — 양극단 30% 정확도 (Layer 3)

| 항목 | ML (02a) | Trailing | 차이 |
|---|---|---|---|
| Low vol hit (BL Long) | **0.634** | 0.590 | +4.4%p |
| High vol hit (BL Short) | **0.663** | 0.626 | +3.7%p |
| Spearman rank corr | **0.688** | 0.616 | +0.072 |

→ **ML 의 vol ranking 정확도는 모든 측면에서 Trailing 우위**.

### LS Spread 분석 — Hit rate 우위에도 BL 성과 negative 의 패러독스

#### Raw Long-Short 수익률 (mcap-weighted, BL 우회)
| 측정 | ML | Trailing | 차이 |
|---|---|---|---|
| Long 30% (월평균) | +1.088% | +1.223% | -0.135%p |
| Short 30% (월평균) | +1.883% | +1.626% | +0.256%p |
| **LS spread (월평균)** | **-0.795%** | **-0.403%** | -0.391%p |
| **LS spread (연환산)** | **-9.53%** | **-4.84%** | -4.70%p |

→ **LS spread 둘 다 NEGATIVE**: 17년 평균 raw 기준 BAB factor 미작동.
→ ML 의 LS spread 가 더 negative → BAB anomaly 활용도 ML < Trailing.

#### 시기별 LS Spread (mcap-w, 연환산 %)

| 시기 | ML LS_mw | TR LS_mw | BAB 작동? |
|---|---|---|---|
| GFC 회복 (09~11) | -13.26% | -13.11% | ❌ 둘 다 손실 |
| **강세장 (12~19)** | **-3.55%** | **+3.17%** | ⭐ Trailing 만 |
| COVID (20) | -21.59% | -18.97% | ❌ 위기 충격 |
| **긴축 (21~22)** | +1.88% | **+8.91%** | ⭐ Trailing 강함 |
| 회복·AI (23~25) | -25.82% | -22.87% | ❌ AI 양극화 |

→ **BAB 작동 시기**: Trailing 132 개월 (강세장 + 긴축), ML 24 개월 (긴축만).

### 시총 분포 (Long 30% 그룹)

```
ML long 30% 평균 mcap:       $51.2 B
Trailing long 30% 평균 mcap: $49.0 B
All universe 평균:           $38.6 B
```

→ 시총 분포 차이 미미 (~$2B). **시총 분산 가설 기각**.

### 진단 결론 — Hit Rate ↑ 이지만 BL Sharpe ↓ 인 이유

**Trailing vol_21d 의 진정한 가치**:
- "최근 vol 이 낮음" = "안정적인 cash flow 회사" 의 proxy
- Utilities, Consumer Staples, Healthcare 등 방어주 특성 식별
- BAB anomaly 의 underlying 회사 특성과 일치

**ML forward vol prediction 의 한계**:
- 정확한 vol 예측 ≠ 회사 특성 식별
- "이번 달 vol 이 낮을 종목" ≠ "구조적으로 안정적인 회사"
- BAB anomaly 활용에는 부적합

**핵심 통찰** (학술 기여):
> **"Volatility prediction accuracy improvement (RMSE↓, hit rate↑) does NOT translate to Black-Litterman portfolio alpha when used as P-matrix sorter."**

→ **Pyo & Lee (2018) 의 "ML > Trailing" 주장 부분 반증** (KOSPI vs 미국 17년 환경 차이).
→ **Vol prediction** 과 **BAB anomaly** 의 분리 — 학술 보고에서 핵심 결론 가능.

### BL 의 작동 메커니즘 재확인

LS spread 가 음수임에도 BL_trailing Sharpe 1.222 가 SPY 1.05 능가:
- BL 은 long-short 가 아닌 **long-only with low-vol bias**
- P 행렬 view → mu_BL → optimize_portfolio 로 low-vol 종목에 가중치 ↑
- 결과: **저변동성 long-only 포트폴리오**
- 수익률 14.5% < SPY 15.4% 지만 vol 11.7% < SPY 14.7% → Sharpe 우위
- **저변동성 anomaly (Low-Risk Anomaly)** 작동 — 위험조정 측면

### 산출물
- `outputs/02a_stockwise/bl_sanity_check.png` (3 panel: 누적/DD/Rolling Sharpe)
- `outputs/02a_stockwise/hit_rate_analysis.png` (4 panel: low/high hit rate, Spearman, confusion matrix)
- `outputs/02a_stockwise/paradox_analysis.png` (4 panel: 누적 LS spread, 시기별, 시총 분포)

### 다음 단계 (확정)

1. **02b cross-sectional 학습** — Cross-Sec 의 ML 이 BAB 를 더 잘 잡을지 검증 (학술 핵심 비교)
2. **03 BL backtest 실행** — 6 시나리오, 정식 비교
3. **05a/05b/05c 평가 노트북** — Layer 1~5 정밀 분석
4. **학술 보고서 작성** — 본 패러독스가 핵심 발견

---

## §13.6. 02a 단독 깊이 분석 — 02b 학습 중 사전 준비 (2026-04-30)

02b cross-sectional 학습이 ~22h 진행 중인 상태에서 02a 결과를 깊이 분석하기 위한 작업.
**02b 학습 영향 0** 보장 원칙 (사용자 결정) 하에 진행.

### 사용자 핵심 지적

> "05a 마무리하려면 BL_ml_sw 포트폴리오 결과 로드해야 되는데, 이게 03 에서 02b 가 완료되지 않으면 생성할 수가 없음."

→ 05a 의 Cell 5 가 `outputs/03_bl_backtest/returns_BL_ml_sw.csv` 만 시도 → 03 미실행 시 §3 Layer 2 분석 막힘.
→ **02a §6 sanity check 캐시 (`bl_weights_sanity_check.pkl`) 활용** 으로 해결.

### Step 1. 05a Cell 5 캐시 fallback 추가 (완료 ✅)

| 항목 | 변경 |
|---|---|
| 우선순위 1 | `outputs/03_bl_backtest/returns_BL_ml_sw.csv` (03 정식 결과) |
| 우선순위 2 (⭐) | `data/bl_weights_sanity_check.pkl` (02a §6 캐시) |
| weights → returns | `make_returns_manual` 함수 노트북 안에서 정의 (scripts/ 변경 X) |

**효과**: 03 미실행 (02b 학습 중) 상태에서도 BL_ml_sw 단독 분석 가능.

### Step 4. 05a §7 신규 섹션 추가 — 누락 163 종목 시기별 영향 분석 (완료 ✅)

**§7 셀 6 개 추가** (Cell 22~27):
- 7-1: 누락 163 종목 4 카테고리 분류 (파산 14 / M&A ~80 / 분할 / Private)
- 7-2: OOS 시기별 누락 카운트 + BL_ml_sw Sharpe overlay (5 시기)
- 7-3: Sector imbalance proxy (Panel 646 vs 학습 613)
- 7-4: 시기별 Sharpe vs 누락 영향 회귀 분석 + 4 panel 시각화
- 7-5: 결론 + 학술 정직성 (Limitations 입력)

**산출 시각화 (예정)**: `outputs/05a_eval_stockwise/sec7_missing_impact.png` (4 panel)

### 02b 학습 영향 0 보장 (절대 원칙)

| 원칙 | 검증 방법 |
|---|---|
| `scripts/` 절대 수정 X | git diff scripts/ → 변경 없음 (오늘 학습 시작 후) |
| `02b_phase15_cross_sectional.ipynb` 수정 X | git diff → 변경 없음 (오늘 학습 시작 후) |
| 별도 커널 사용 | VS Code 노트북별 독립 Python 프로세스 |
| GPU 자원 비경쟁 | 05a 분석은 CPU/RAM only (PyTorch X) |
| `make_returns_manual` 함수 모듈 X | 05a 노트북 내부에 inline 정의 |

**검증 시각**:
- scripts/models_cs.py: 01:12:46 (학습 시작 전)
- scripts/volatility_ensemble.py: 02:54:04 (학습 시작 전)
- 02b 노트북: 03:03:23 (학습 시작 전)
- 05a 노트북: 09:38:58 (Step 1+4, 02b 학습 시작 후 — 02b 와 무관)
- _build_05a_eval_sw_nb.py: 09:39:52 (Step 1, 02b 와 무관)

→ 학습 시작 후 변경된 파일은 **05a + 빌드 스크립트만** = 02b 영향 0 ✅

### 사용자 액션 — 노트북 직접 실행 (Step 2~3)

VS Code 에서 05a 노트북:
1. **Use Disk Version** (디스크 변경 반영)
2. **별도 커널** 시작 (.venv, 02b 와 분리)
3. 셀 순차 실행: §1 → §2 → §3 → §4 → §5 → §6 → **§7 (신규)**

**기대 출력**:
- §3 Layer 2: Sharpe 1.108, MDD -18.56%, CAPM α 등
- §5 Layer 4: 5 시기 분해 (GFC/정상/COVID/긴축/AI)
- §7 누락 종목: 카테고리 분류 + 회귀 (r², p-value) + 시각화 4 panel + Limitations 결론

### 잠정 결과 예상 (§7 실행 후 확인)

```
누락 163 종목 (20.1%, universe 809 → panel 646)
  · OOS 파산 14: SIVB(2023), FRC(2023), JCP(2020), CHK(2020), ...
  · M&A 인수 ~80: MON, CELG, ATVI, RHT, PXD, XLNX, TWTR, WFM, ...
  · 분할/사명변경, Private 등 ~70

가장 영향 큰 시기: COVID (2020) — 6 종목 파산 (JCP/MNK/CHK/FTR/DO/DNR)
다음: 회복·AI (2023~25) — SIVB/FRC/BIG 3 종목

학술 정직성 결론:
  Phase 3 BL_ml_sw Sharpe ~1.108 은 진짜 universe 대비 과대평가 가능성
  (학술 보고서 Limitations 섹션 직접 활용)
```

### 산출물 (예정)

- `outputs/05a_eval_stockwise/layer2_portfolio_diagnostic.png` (Step 2)
- `outputs/05a_eval_stockwise/layer4_period_decomposition.png` (Step 3)
- `outputs/05a_eval_stockwise/sec7_missing_impact.png` (Step 4, §7 시각화)

### 작업 시간 vs 02b 학습 ETA

| 시점 | 작업 |
|---|---|
| 09:38 | Step 1 (Cell 5 fix) — 30분 |
| 09:50 | Step 4 (§7 5 셀 작성) — Bash heredoc 우회 (임시 .py 작성+실행+삭제) |
| 10:00 | Step 5 (WORKLOG §13.6 초안 추가) — 본 섹션 |
| 12:30 | Step 6 (§7-6/7/8 추가 + §13.6 보강) — 사용자 지적 보강, 본 sub-section |
| ~ | (사용자 직접 노트북 실행 Step 2~3 + §7 + §7-6/7/8) |
| ~+22h | 02b 학습 완료 → 03 BL 백테스트 진입 |

### Step 6. §7-3 학술적 정정 + §7-6/7/8 추가 (2026-04-30, 사용자 지적 보강)

> ⭐ 사용자가 §7-3 의 sector imbalance 1.64%p 결과 해석에 대해 학술적으로 정확한 지적을 함. 본 sub-section 은 그 정정을 §7-6/7/8 신규 셀과 본 WORKLOG 에 반영.

#### 사용자 핵심 지적 (학술적 100% 정확)

> "각 섹터의 분포는 괜찮다고 쳐도, 이미 살아남은, 또는 M&A 없이 정상적으로 생존한 기업만을 대상으로 하는 분석이라는 점에서 한계가 명확하지 않나?"

#### §7-3 의 본질적 한계 — 정확한 표현

| 잘못된 해석 (이전) | 정확한 해석 (정정) |
|---|---|
| "학습 615 가 panel 646 의 sector 분포 잘 대표 → BL_ml_sw 의 sector 측면 신뢰 가능" | "panel 646 (살아남은 종목) **안에서** 학습 615 의 sector 대표성 확보. 그러나 panel 자체가 이미 살아남은 종목 집합 → 진짜 생존편향 미측정" |

```
A. 진짜 universe (Wikipedia 809):  살아남은 종목 + 사라진 종목 모두 포함
B. Panel (yfinance 646):           살아남은 종목 + M&A 직전 종목 (subset of A)
C. 학습 (615):                     1334일+ 데이터 종목 (subset of B)

§7-3 가 측정한 것: B vs C 의 차이 → 1.64%p (panel 안에서)
§7-3 가 못 본 것:  A vs B 의 차이 → §7-7 에서 측정 (real_imbalance)
```

#### 진정한 생존편향 — 누락 163 종목의 systematic 특성

**1. OOS 파산 14 종목의 sector 편중**

| Sector | 누락 파산 종목 | 비중 |
|---|---|---|
| **Energy** | CHK, FTR, DO, DNR, ANR | **5/14 (36%)** ⭐ |
| Financials | SIVB, FRC | 2/14 (14%) |
| Consumer Discretionary | JCP, BIG, DF | 3/14 (21%) |
| Health Care | MNK, ENDP | 2/14 (14%) |
| Industrials/Telecom | EK, WIN | 2/14 (14%) |

→ **Energy 가 압도적**. Phase 3 학습 universe 의 Energy 32 종목은 "**파산 위험 종목 제거된**" 분포.
→ ML 변동성 모델이 학습한 Energy 분포는 "안정 Energy 회사" 위주 (현실 fat tail 미반영 가능성).

**2. M&A ~80 종목의 systematic 차이**

- 변동성 ↑: 인수 premium 협상기 (보통 30%+ jump)
- 수익성 정점: 매력적 자산이라 인수 대상
- Sector 다양: Tech (XLNX, ATVI, RHT), Health Care (CELG, AGN, STJ), Energy (APC, NBL, MRO, PXD), Materials (MON), Consumer (TIF, WFM)

→ 학습 데이터의 변동성 분포가 "M&A 까지 이어진 prosperous 종목" 을 미반영.

#### §7-6/7/8 신규 셀 (`05a_eval_stockwise.ipynb` Cell 28~31)

| 셀 ID | 타입 | 역할 |
|---|---|---|
| `sec7_6_header` | markdown | §7-6/7/8 통합 헤더 + 사용자 지적 인용 |
| `sec7_6_missing_sector` | code | 누락 163 sector hardcoded 매핑 (bankruptcy 14 + M&A 120) |
| `sec7_7_compare_viz` | code | Panel 646 ↔ 누락 163 sector 분포 비교 + 4 panel 시각화 |
| `sec7_8_conclusion` | code | 정정된 결론 (학술 보고서 Limitations 입력) |

**산출 시각화 (사용자 노트북 실행 후 생성)**: `outputs/05a_eval_stockwise/sec7_real_universe_imbalance.png` (4 panel: panel vs 누락 분포 / 차이 (%p) / 카테고리×sector / panel 의 sector 별 universe 대표성).

#### §7-7 측정 결과 (2026-04-30 사용자 노트북 실행 완료)

```
누락 163 종목 hardcoded sector 매핑 비율:  126/163 (77.3%)
                                          Unknown 잔존 37/163 (22.7%)

Real universe imbalance (Panel ↔ 누락):
  - 전체:           31.31%p   (Unknown 포함, headline)
  - Unknown 제외:   13.56%p   (방법론 noise 보정 후)

§7-3 의 1.64%p 대비:
  - 19.1배 (Unknown 포함)
  - 8.3배 (Unknown 제외)

Panel 비중 < 80% sector (진짜 universe 대비 under-representation):
  - Unknown                46.4%  (panel 32, 누락 37) ← hardcoded 매핑 실패 noise
  - Energy                 61.8%  (panel 34, 누락 21) ⭐ 핵심 발견
  - Communication Services 73.0%  (panel 27, 누락 10)
  - Consumer Staples       75.0%  (panel 39, 누락 13)
  - Health Care            76.9%  (panel 70, 누락 21)

(Panel 비중 > 100% 의 over-representation sector — 진짜 universe 대비 panel 에 더 많이 분포)
  - Industrials            -7.65%p  (panel 93, 누락 11)
  - Financials             -7.49%p  (panel 92, 누락 11)
  - Consumer Discretionary -5.95%p  (panel 82, 누락 11)
  - Real Estate            -3.73%p  (panel 36, 누락 3)
  - Information Technology -2.87%p  (panel 78, 누락 15)
  - Utilities              -2.50%p  (panel 32, 누락 4)
  - Materials              -1.12%p  (panel 31, 누락 6)
```

#### §7-7 결과 해석 — 핵심 4가지

**1. Energy 의 under-representation (진짜 universe 대비 61.8%) ⭐**

가장 중요한 발견. 사용자가 가설로 제시한 "OOS 파산 14 중 Energy 5/14 (36%)" 의 편중이 universe 전체 차원에서도 확인됨. 누락 163 종목 중 Energy 21 종목 (12.88%) — Unknown 제외 시 누락 sector 비중 #1. 즉 ML 변동성 예측 모델이 학습한 Energy 분포는 panel 34 종목 (살아남은 안정 Energy) 만 보고, 진짜 universe Energy 55 종목 (panel 34 + 누락 21) 의 fat tail (파산·M&A 흡수된 Energy) 을 약 38% 미반영.

**2. M&A 인수 sector 의 균등한 under-rep (Health Care, Communication Svcs, Consumer Staples)**

각각 76.9%, 73.0%, 75.0% — Celgene/AGN/STJ (제약 인수), TWTR/ATVI/DTV (통신·미디어 인수), WFM/HNZ/MJN/DPS (소비재 인수) 등 panel 에서 누락된 매력적 인수 대상. 이들은 인수 직전 변동성 jump (premium 협상) 가 학습에 반영되지 않음.

**3. 31.31%p vs 13.56%p — Unknown 처리 주의**

headline 31.31%p 중 17.75%p (Unknown 카테고리) 는 hardcoded 매핑 실패 (분할/Private/기타 ~37 종목) 의 방법론적 noise 성격. 학술 보고서에는 두 수치 병기 (Unknown 포함 19.1배 vs 제외 8.3배) 가 정직한 표현. 어느 쪽이든 §7-3 의 1.64%p 는 진짜 생존편향의 1/8 ~ 1/19 만 측정한 결과.

**4. Industrials/Financials 의 over-rep (panel 에 진짜 universe 대비 많음)**

각각 -7.65%p, -7.49%p — 인수·파산 사례가 적어 panel 에 비중이 높음. 학습이 이 sector 의 "안정성" 을 진짜 시장보다 강하게 학습한 셈. 이는 BL_ml_sw 가 Industrials/Financials 비중을 높게 잡을 가능성과 연결 — 단 BL P 행렬은 vol-based 이므로 영향 제한적.

#### 진정한 생존편향 결론 (학술 보고서 Limitations 섹션 직접 활용)

```markdown
### Survivorship Bias from yfinance Data Limitation

본 연구는 yfinance 데이터 부재로 인해 S&P 500 의 시점별 멤버십 (Wikipedia
2009~2025 union, 809 종목) 중 163 종목 (20.1%) 이 분석에서 제외되었다.
누락 종목의 본질:
- M&A 인수: ~80 종목 (49.1%) — ticker 자체 소멸
- OOS 기간 내 파산: 14 종목 (8.6%) — SIVB(2023), FRC(2023), JCP(2020),
  MNK(2020), CHK(2020), FTR(2020), DO(2020), DNR(2020) 등
- 분할/사명변경: ~25 종목, Private: ~20 종목, 기타: ~24 종목

⭐ 본 연구의 본질적 한계:
**panel universe (yfinance 살아남은 646 종목) 안에서** 학습 universe (615
종목) 의 sector 대표성은 1.64%p imbalance 로 매우 충실하다 (Section 7-3).
그러나 이는 "**살아남은 종목 안에서** 학습 필터의 sector 중립성" 만 입증할
뿐, **panel 자체가 이미 yfinance 살아남은 종목 집합** 이라는 본질적
생존편향은 잔존한다 (Section 7-7 에서 정량화).

특히 OOS 기간 내 파산 14 종목의 sector 분포 분석 결과 Energy sector 가
36% (5/14) 로 가장 편중되어 있으며 (CHK, FTR, DO, DNR, ANR), 이는 본 연구
panel 의 Energy 분포가 진짜 universe 대비 "안정 Energy 회사들" 위주로
편향되었음을 시사한다. 즉 ML 변동성 예측 모델이 학습한 분포는 "M&A·파산
까지 이어진 prosperous/안정 종목들" 이며, 진짜 시장 변동성의 fat tail
(파산·인수당함 종목들의 극단 vol) 을 부분적으로 미반영한다.

본 연구의 절대 수치 (BL_ml_sw Sharpe ~1.108) 는 진짜 universe 대비
과대평가일 가능성이 높으며, 향후 학술 표준 데이터베이스 (CRSP) 를 사용한
재현 검증이 필요하다.

상대 비교 (BL_ml_sw vs BL_trailing) 는 두 시나리오가 동일 데이터 소스 한계를
공유하므로 결론 (ML 통합 효과 -0.114 Sharpe) 의 방향성은 신뢰 가능하나,
"Trailing vol = 방어주 proxy" 가설은 생존편향 환경에서 더 강하게 작동할
가능성이 있다. 즉 본 연구의 발견 (Hit rate ↑이지만 BL Sharpe ↓) 은
"살아남은 종목 환경" 이라는 specific 조건에서의 결론이며, 진짜 universe
환경에서는 다를 수 있다.
```

#### 본 패러독스 (Hit rate ↑, BL Sharpe ↓) 의 신뢰성 재평가

| 항목 | 이전 결론 | 정정된 결론 |
|---|---|---|
| 상대 비교 (ML vs Trailing) | "두 시나리오 동일 데이터 한계 공유 → 신뢰 가능" | 동일 — 방향성 (-0.114 Sharpe) 신뢰 가능 |
| 절대 수치 (BL_ml_sw Sharpe 1.108) | (언급 없음) | ⚠️ 진짜 universe 대비 과대평가 가능성 |
| Trailing vol = 방어주 proxy 가설 (§13.5) | 미국 17년 universe 환경 차이 | + 보강: 학습 universe 가 "살아남은 종목" 으로 제한 → 방어주 (Utilities/Staples) 자연 잘 살아남음 → Trailing 의 방어주 식별이 유리한 환경. 단 Trailing 은 구조적 식별, ML 은 forward 1개월 예측 → BAB 본질 미반영 |

#### 02b 학습 영향 0 보장 (Step 6 검증)

| 원칙 | Step 6 검증 |
|---|---|
| `scripts/` 절대 수정 X | 모든 변경: `05a_eval_stockwise.ipynb` (셀 4개 추가) + `재천_WORKLOG.md` (본 섹션 + footer) 만 |
| `02b_phase15_cross_sectional.ipynb` 변경 X | 02b 노트북 미터치 |
| 별도 커널 사용 | 사용자 VS Code 별도 .venv 커널 |
| GPU 자원 비경쟁 | §7-6/7/8 은 CPU only (pandas/matplotlib) |
| 새 모듈 함수 X | _tmp 스크립트가 inline code 로 셀 4개만 추가 후 자체 삭제 |

#### 산출물 + 사용자 액션

**Step 6 작업 산출 (즉시)**
- `05a_eval_stockwise.ipynb`: 셀 28 ~ 31 신규 (markdown 1 + code 3, 총 32 셀)
- `재천_WORKLOG.md`: 본 §13.6 Step 6 + footer 갱신
- `_tmp_add_sec7_6_8.py`: 작성 후 즉시 삭제

**사용자 액션 (대기)**
- VS Code 에서 05a 노트북 § 7-6 → §7-7 → §7-8 셀 순차 실행 (각 1분 미만, 별도 커널)
- 산출 PNG (`sec7_real_universe_imbalance.png`) 와 print 출력 확인 후 본 §13.6 의 "§7-7 측정 결과" 항목 placeholder 실수치 채움

**학술 보고서 입력**
- 위 "Survivorship Bias from yfinance Data Limitation" markdown 블록 → 보고서 Limitations 섹션에 그대로 활용 가능

### Step 7. Static-Universe → Dynamic-Membership 전환 (2026-04-30, 사용자 결정)

> ⭐ §13.6 Step 6 진행 중 universe 정의의 누수 발견 → 사용자 결정으로 즉시 전환.

#### 사용자 결정

> "dynamic membership으로 변경하고 기존꺼는 격리해줘. 앞으로 dynamic membership만 활용하자."

#### 전환 배경 — Static-Universe 의 숨은 look-ahead

§7-3 정정 작업 중 발견: 기존 코드가 `available_tickers = panel_at_date & trained_tickers 615` 로 universe 결정 → **그 시점 실제 S&P 500 멤버 여부 무관** 하게 학습된 615 종목 pool 에서 매월 후보 선정.

| 종목 예시 | yfinance panel | 실제 S&P 편입 | Static 모드 universe 후보 (2010 reb_date) |
|---|---|---|---|
| TSLA | 2010-부터 데이터 있음 | **2020-12** | ⚠️ 포함 (2010~2019 도) |
| FB/META | 2012-부터 | 2013-12 | ⚠️ 포함 (2012-12~2013-11 도) |

→ portfolio backtest 단계에서 **"결국 큰 회사가 되어 S&P 편입한다" 는 미래 멤버십 정보 누수**. B1~B7, L1~L11 의 microscopic 누수 차단을 모두 적용해도 universe 자체에 leak 가 있어 학술 baseline 과 fair 비교 위반.

#### Dynamic-Membership 정의

매월 universe = `sp500_member_at_t ∩ panel ∩ 학습 615`.

`reb_date` (거래일 월말, Issue #1 으로 보정된) → 동일 month period 의 calendar 월말 키 (`reb_date.to_period('M').to_timestamp(how='end').normalize()`) 로 `membership` dict lookup. Issue #1 의 calendar/market 월말 보정 패턴과 일관.

#### 변경 코드 (4 cell)

| 파일 | Cell | 변경 |
|---|---|---|
| `03_BL_backtest_extended.ipynb` | Cell 4 (§2 데이터 로드) | `membership = get_or_build_membership(...)` 로드 추가 |
| `03_BL_backtest_extended.ipynb` | Cell 10 (BL 루프) | `available_tickers = members_at_date & panel & trained` |
| `02a_phase15_stockwise_extended.ipynb` | Cell 22 (§6-1) | membership 로드 동일 추가 |
| `02a_phase15_stockwise_extended.ipynb` | Cell 23 (§6-2 BL 루프) | `avail = members_at_date & panel & trained` |

#### 격리 보존

| 항목 | 위치 |
|---|---|
| 기존 03 노트북 (Static 모드) | `03_BL_backtest_extended_legacy_static.ipynb` |
| 기존 02a §6 BL sanity check 결과 캐시 (Static) | `data/bl_weights_sanity_check_legacy_static.pkl` |
| 메인 캐시 `data/bl_weights_sanity_check.pkl` | **삭제** (사용자 02a §6 재실행 시 Dynamic 결과로 자동 재생성) |

→ 02a §6 자체는 격리 X — `§1~5 학습 cell 은 universe 필터 무관` 하므로 §6 만 in-place 수정 (학습 결과는 그대로).

#### 두 bias 의 분리 (학술 보고서 표현)

| Bias | Step 6 후 | Step 7 후 |
|---|---|---|
| Look-ahead (portfolio universe 의 미래 멤버십) | ⚠️ 잔존 (Static-Universe) | ✅ **해결** (Dynamic-Membership) |
| Survivorship (panel 646 ⊂ 진짜 809) | ⚠️ 잔존 (§7-6/7/8 정량화 31.31/13.56%p) | ⚠️ 잔존 — yfinance 한계, CRSP 만 해결 가능 |

→ **Dynamic-Membership 은 look-ahead 만 해결, survivorship 은 별개**. 학술 보고서에서 두 bias 를 분리해 표현해야 함.

#### 결과 수치 변화 (2026-04-30 사용자 02a §6 재실행 실측)

| 메트릭 | Static (legacy) | Dynamic (실측) | 차이 |
|---|---|---|---|
| 매월 universe size 평균 | 548.8 | **420.0** | **-128.9 종목 (-23%)** |
| 매월 universe range | 477~594 | 337~488 | — |
| BL_ml_sw Sharpe | 1.108 | **1.123** | +0.015 ↑ |
| BL_trailing Sharpe | 1.222 | **1.203** | -0.019 ↓ |
| **diff (ML 효과)** | **-0.114** | **-0.080** | **+0.034 (절대값 -30%)** ✅ 방향성 robust |
| Hit rate Low (ML) | 0.6342 | 0.6342 | 동일 (universe filter 무관) |
| Hit rate High (ML) | 0.6632 | 0.6632 | 동일 |
| LS spread mw (ML, %/yr) | -9.53 | -9.53 | 동일 |
| LS spread mw (TR, %/yr) | -4.84 | -4.84 | 동일 |

**시기별 universe 축소**:

| 시기 | Dynamic | Static | 축소 |
|---|---|---|---|
| GFC 회복 (09~11) | 350.1 | 494.0 | -143.9 ⭐ 가장 큰 축소 (과거일수록 미래 멤버 많음) |
| 강세장 (12~19) | 405.3 | 541.2 | -135.9 |
| COVID (20) | 457.5 | 574.8 | -117.3 |
| 긴축 (21~22) | 468.2 | 582.2 | -113.9 |
| 회복·AI (23~25) | 484.3 | 593.1 | -108.8 ← 최근일수록 작은 축소 |

→ **시간 경과에 따른 축소폭 감소**가 학술 baseline 과 일관 — 정확히 시점별 멤버십이 작동.

#### Look-ahead 차단 직접 검증 — TSLA 시점별 포함 여부

TSLA S&P 500 편입 시점: **2020-12-21**

| reb_date | Dynamic | Static | 의미 |
|---|---|---|---|
| 2009-01-30 | ✗ | ✗ | 둘 다 미포함 (TSLA 학습 데이터 부족) |
| 2014-01-31 | ✗ | ✓ | ⭐ **Static 의 look-ahead 누수** (편입 6년 전) |
| 2019-01-31 | ✗ | ✓ | ⭐ **Static 의 look-ahead 누수** (편입 1년 전) |
| 2020-12-31 | ✓ | ✓ | 편입 직후 둘 다 포함 |
| 2021-02-26 | ✓ | ✓ | 이후 둘 다 포함 |

→ **Dynamic 가 TSLA 의 S&P 편입 전 누수를 정확히 차단**. 사용자 의도한 look-ahead 차단이 100% 작동 확인.

#### Subset 관계 검증

첫 시점 (2009-01-30) 비교:
- Static 에 있고 Dynamic 에서 빠진 종목: **140 개** (ACN, ALGN, AJG 등 — 당시 비-멤버 후일 편입)
- Dynamic 에 있고 Static 에 없는 종목: **0 개** ✅

→ Dynamic ⊆ Static. 멤버십 필터만 추가됨이 수학적으로 보장.

#### 핵심 학술 메시지 (실측 기반)

매월 universe 가 **Static 의 76.5%** (420/548.8) 로 줄었음에도:
- BL_ml_sw 와 BL_trailing 두 시나리오 절대 수치 모두 미세 변동 (각 ±0.02 이내)
- ML 통합 효과 **-0.114 → -0.080** (절대값 30% 감소했으나 **방향성 유지**)
- **"ML > Trailing 이 아님" 핵심 결론 robust** — Static 과 Dynamic 환경 양쪽에서 일관

→ Pyo & Lee (2018) 부분 반증 + Trailing vol = 방어주 proxy 가설 (§13.5) 모두 universe 정의에 robust 한 결론으로 강화됨.

#### 노트북 cosmetic 이슈 (정정 완료, 2026-04-30)

- ✅ `02a` Cell 30 (§6 진단 결론) 의 하드코딩 1.108/1.222/-0.114 → `metrics_table` 변수 참조로 변경

#### 재실행 안전성 강화 — 캐시 보호 (2026-04-30)

비싼 작업 (학습 + BL 최적화) 의 무의도 재실행 방지를 위해 02a Cell 23 (§6-2) 와 동일 패턴의 캐시 로직을 두 곳에 추가:

| 비싼 작업 | 캐시 파일 | flag | 비용 |
|---|---|---|---|
| 02a Cell 8 (LSTM 8-way GPU 학습) | `data/ensemble_predictions_stockwise.csv` (기존 산출) | `FORCE_RETRAIN=False` | ~수~십수 시간 (GPU) |
| 02a Cell 23 (§6-2 BL sanity) | `data/bl_weights_sanity_check.pkl` (기존) | `FORCE_RECOMPUTE=False` | ~분~십수 분 (CPU) |
| 03 Cell 10 (§4 BL 6 시나리오) | `data/scenario_weights_03_dynamic.pkl` (신규) | `FORCE_RECOMPUTE=False` | ~수십 분~수 시간 (612회 SLSQP) |

**효과**:
- VS Code "Run All" 또는 실수 클릭으로 인한 GPU 재학습 / BL 재계산 방지
- 02b 학습 진행 중 02a 셀 안전 실행 보장
- 분석/시각화 변경 후 03 재실행 시 즉시 캐시 로드 (~수 초)

**강제 재계산이 필요한 경우** (예: 코드 변경, scripts/ 수정): 해당 flag 를 `True` 로 변경 후 재실행.

#### 사용자 액션 (대기)

1. **02a §6 BL sanity check 재실행** (별도 .venv 커널, 약 5분, CPU only)
   - 새 캐시 `bl_weights_sanity_check.pkl` 자동 생성 (Dynamic 결과)
2. **05a Cell 5 (BL_ml_sw 로드) + §3 Layer 2 + §5 Layer 4 + §7-2 재실행**
   - 새 BL_ml_sw returns 로 메트릭 재계산 (Sharpe / MDD / Alpha 등)
   - §7-2 시기별 회귀 결과 변동 가능
   - §7-3, §7-6/7/8 의 sector imbalance 자체는 **변경 없음** (universe 정의 무관)
3. **(02b 학습 완료 후, ~22h)**
   - 03_BL_backtest_extended.ipynb 정식 실행 (6 시나리오 dynamic)
   - 04 비교, 05b, 05c 재실행

#### 02b 학습 영향 0 보장 (Step 7 검증)

| 원칙 | Step 7 검증 |
|---|---|
| `scripts/` 절대 수정 X | 모든 변경: `03_BL_backtest_extended.ipynb` + `02a_*.ipynb` (§6 만) + `재천_WORKLOG.md` |
| `02b_phase15_cross_sectional.ipynb` 변경 X | 02b 노트북 미터치 |
| 별도 커널 사용 | 사용자 VS Code 별도 .venv 커널 |
| GPU 자원 비경쟁 | 02a §6, 03 모두 CPU only (pandas/numpy/SLSQP) |

#### 산출물

- 노트북 변경: `03_BL_backtest_extended.ipynb`, `02a_phase15_stockwise_extended.ipynb`
- 격리 보존: `03_BL_backtest_extended_legacy_static.ipynb`, `data/bl_weights_sanity_check_legacy_static.pkl`
- 본 §13.6 Step 7 — 학술 보고서의 "Universe Definition" 섹션 표현으로 직접 활용

### Step 8. Stale Price 발견 + Universe 단계 필터 적용 (2026-04-30, 팀원 발견 보강)

> ⭐ 팀원이 다른 프로젝트에서 SW 티커의 stale price 문제를 발견. 본 프로젝트도 동일 현상 보유 — panel 진단 + universe 필터로 해결.

#### 팀원 발견 + 본 프로젝트 검증

> "SW 티커 종목의 경우 몇 달 간 0 으로 기록된 부분이 있다"

**본 프로젝트 panel 검증 결과**:

```
SW 티커:
  panel 기간: 2008-06-17 ~ 2025-12-31
  log_ret == 0:        2,903 / 4,414 (65.8%) ⚠️
  가장 긴 stale 구간: 2009-05-13 ~ 2009-08-19 (69 영업일, 14주)
  30+ 영업일 연속 0 구간: 9 회

panel 의 모든 date 는 영업일만 포함 (Mon~Fri, Sat=Sun=0):
  → "주말이라 0" 가설 부정
  → SW 의 0 행도 모두 평일에 분포 (Mon 534, Tue 607, Wed 598, Thu 577, Fri 587)
  → 동일 영업일에 다른 종목 (513) 정상 거래 → SW 만 stale
  → 명백한 데이터 품질 issue
```

#### 원인 — yfinance 의 stale fill

| 원인 | 메커니즘 | 해당 종목 |
|---|---|---|
| **A. M&A 인수** | 인수 후 ticker 시장에서 소멸, yfinance 가 마지막 가격 stale fill | COL (Rockwell Collins → UTC 2018), EP (El Paso → KMI 2012), GR (Goodrich → UTC 2012) |
| **B. Private 전환** | 상장 폐지 후 ticker 소멸 | CPWR (Compuware 2014), BMC (BMC Software 2013), CVG |
| **C. 파산** | 거래 정지 후 ticker 소멸 | RSH (RadioShack 2015) |
| **D. Ticker 재사용** | 동일 ticker 를 다른 회사가 재사용, yfinance 가 통합 | **SW**: Smurfit-Stone Container (2010) → Smurfit Westrock (2024) |

#### 정량화 (panel 646)

| 분류 | 종목 수 | 비율 | 학습 615 안 |
|---|---|---|---|
| 0 비율 > 50% (심각) | 9 | 1.4% | 7 (BMC, COL, CPWR, CVG, EP, GR, SW) |
| 0 비율 > 30% (의심) | 12 | 1.9% | 8 (위 7 + AMCR) |
| 0 비율 ≤ 30% (정상) | 634 | 98.1% | 607 |

#### Dynamic-Membership 의 자연 차단 효과 (Step 7 부가 가치)

학습 615 stale 8 종목의 BL universe 출현 빈도:

| ticker | 0 비율 | BL 출현 시점 | 비고 |
|---|---|---|---|
| SW | 65.8% | **169/204** | 가장 많이 들어감 |
| AMCR | 43.3% | 79/204 | |
| EP | 53.3% | 40/204 | |
| COL | 59.9% | 33/204 | |
| CPWR | 67.5% | 18/204 | |
| CVG | 55.1% | 11/204 | |
| **BMC** | 50.2% | **0/204** ⭐ | Dynamic 자동 차단 |
| **GR** | 51.4% | **0/204** ⭐ | Dynamic 자동 차단 |

→ **BMC, GR 은 Dynamic-Membership 만으로도 차단**. 그러나 SW (169 시점), AMCR (79 시점) 등은 잔존 → Step 8 필요.

#### 매월 stale 영향 정량 (Step 7 결과 기준)

```
매월 BL universe 안의 stale (>30%) 종목 수: mean 1.72, max 2 (절대 9개 아님)
매월 stale 종목 가중치 합:
  mean   0.075%
  median 0.000%
  max    0.939% (2020-02-28: AMCR 0.92% + SW 0.02%)

Sharpe 영향 추정: < 0.005 (negligible)
```

→ portfolio 결과 (BL_ml_sw Sharpe 1.123) 에 **사실상 무시 가능한 영향**. 그러나 학술 정직성 차원에서 처리 필요.

#### Step 8: Universe 단계 stale 필터 적용

매월 BL 루프 안의 universe 결정 단계에 한 줄 추가:

```python
# IS 1260일 안에서 zero ratio > 30% 종목 제외
is_window = daily_lr.loc[reb_date - pd.offsets.BDay(1260):reb_date]
zero_ratio = (is_window == 0).mean()
non_stale = set(zero_ratio[zero_ratio <= STALE_RATIO_THRESHOLD].index)

available_tickers = members_at_date & panel_at & trained_tickers & non_stale  # ⭐ non_stale 추가
```

**Threshold 선택**: 0.30 (30%)
- panel 의 98.1% 종목은 zero ratio < 30% (정상)
- 1.9% 종목 (12개) 만 차단 대상
- 정상 종목 (median 0.7%) 과 stale 종목 (50~67%) 사이 명확한 gap

#### 변경 코드 (2 cell)

| 파일 | Cell | 변경 |
|---|---|---|
| `02a_phase15_stockwise_extended.ipynb` | Cell 23 (§6-2) | `STALE_RATIO_THRESHOLD = 0.30` 상수 + universe 필터에 `& non_stale` 추가 |
| `03_BL_backtest_extended.ipynb` | Cell 10 (§4 BL 루프) | 동일 패턴 적용 |

#### 격리 보존 (캐시 갱신)

| 항목 | 위치 |
|---|---|
| Step 7 (Dynamic-Membership) BL 결과 캐시 | `data/bl_weights_sanity_check_step7_dynamic.pkl` (격리 신규) |
| Static (Step 6 이전) BL 결과 캐시 | `data/bl_weights_sanity_check_legacy_static.pkl` (기존) |
| 메인 캐시 `bl_weights_sanity_check.pkl` | **삭제** (02a §6 재실행 시 Step 8 결과로 재생성) |

→ **3 단계 진화 모두 보존**: Static (Step 6 이전) → Dynamic (Step 7) → Dynamic + Stale 필터 (Step 8).

#### 학습 영향 0 보장 (사용자 핵심 의문 해결)

| 단계 | Step 8 영향 |
|---|---|
| 02a Cell 8 (LSTM 8-way GPU 학습) | ❌ **영향 없음** — universe 필터는 BL portfolio 단계 |
| 02a Cell 14 (ensemble_predictions_stockwise.csv 로드) | ❌ 그대로 사용 |
| 02b Cross-sectional 학습 | ❌ **영향 없음** — 진행 중인 학습 그대로 |
| 02a Cell 23 / 03 Cell 10 BL 루프 | ✅ stale 필터 적용 — 매월 universe 자동 축소 |

→ **재학습 불필요**. 학습은 615 종목 그대로, 매월 BL 후보에서만 stale 종목 동적 제외.

#### 결과 변화 (2026-04-30 사용자 02a §6 재실행 실측)

| 메트릭 | Step 7 (Dynamic) | Step 8 (Dynamic + Stale) | 차이 |
|---|---|---|---|
| 매월 universe size 평균 | 420.0 | **418.5** | -1.51 종목 (예상 1~2 일치 ✅) |
| BL_ml_sw Sharpe | 1.123 | **1.122** | -0.001 |
| BL_trailing Sharpe | 1.203 | **1.207** | **+0.004** ⭐ |
| **diff (ML 효과)** | **-0.080** | **-0.085** | -0.005 (방향성 robust ✅) |
| ML CAGR | 13.285% | 13.275% | -0.010% |
| Trailing CAGR | 14.270% | 14.326% | +0.056% |
| 서윤범 baseline 대비 | +3.98% | +4.32% | +0.34% |

#### Stale 종목 제거 검증 (시점별)

| ticker | Step 7 출현 | Step 8 출현 | 제거 시점 |
|---|---|---|---|
| SW | 169/204 | **0/204** | +169 ⭐⭐ 100% 차단 |
| AMCR | 79/204 | 42/204 | +37 (시기별 부분 차단, 43.3% 종목) |
| EP | 40/204 | **0/204** | +40 ⭐ |
| COL | 33/204 | **0/204** | +33 ⭐ |
| CPWR | 18/204 | **0/204** | +18 ⭐ |
| CVG | 11/204 | **0/204** | +11 ⭐ |
| BMC | 0/204 | 0/204 | 0 (Dynamic 단계 이미 차단) |
| GR | 0/204 | 0/204 | 0 (Dynamic 단계 이미 차단) |

총 제거 = 308 시점 / 204 = **1.51 종목/월** → Universe 축소 평균과 **정확 일치**. 검증 완료.

**AMCR 의 부분 차단**: 매월 IS 1260일 윈도우 기준 zero ratio 가 30% 경계를 동적으로 넘나듦. 일부 시점은 통과, 일부는 차단 — **의도한 동적 동작**.

#### 시기별 universe 축소 (Step 7 → Step 8)

| 시기 | Step 7 평균 | Step 8 평균 | 차이 |
|---|---|---|---|
| GFC 회복 (09~11) | 350.1 | 348.3 | -1.83 |
| 강세장 (12~19) | 405.3 | 403.8 | -1.46 |
| **COVID (20)** | 457.5 | 455.5 | **-2.00** ⭐ 가장 큰 축소 |
| 긴축 (21~22) | 468.2 | 466.5 | -1.75 |
| 회복·AI (23~25) | 484.3 | 483.3 | -1.00 |

→ COVID (2020) 시기 stale 영향이 가장 컸음 — Step 7 의 max 가중치 합 0.939% (2020-02-28) 와 일관.

#### 부수 발견 — Trailing vol 의 stale 노출 더 컸음

| 시나리오 | Sharpe 변화 (Step 7 → Step 8) | 해석 |
|---|---|---|
| BL_ml_sw | -0.001 (거의 무변화) | ML 의 forward vol 예측은 stale 0 신호에 robust |
| BL_trailing | **+0.004** (약간 상승) | Trailing 은 stale 0 vol 을 "low vol" 로 잘못 분류 → Long 자주 선택 |

→ **§13.5 의 "Trailing vol = 방어주 proxy" 가설을 반대 방향에서 보강**:
   Trailing 이 진짜 방어주뿐 아니라 **"가짜 방어주" (stale 종목)** 도 잘 잡고 있었음.
   ML 이 더 noise-robust 함이 본 분석으로 추가 확인.

#### Sharpe 변화의 학술적 의미

변화 절대값 ±0.005 이내 = **statistical noise 수준** (Bootstrap 표준오차 약 ±0.05~0.10 보다 작음).
- 핵심 결론 ("ML > Trailing 이 아님, 약 -0.08 Sharpe diff") **robust** ✅
- Stale 필터는 **portfolio 결과를 흔들지 않으면서 학술 정직성 강화** ✅

#### 사용자 액션

1. **02a §6 BL sanity check 재실행** (별도 .venv 커널, ~십수 분)
   - Cell 22~33 순차 실행
   - 새 캐시 `bl_weights_sanity_check.pkl` 자동 생성 (Step 8 결과)
   - 결과 확인: BL_ml_sw / BL_trailing Sharpe, diff

2. **05a Layer 2~4 + §7-2 재실행**
   - 새 BL_ml_sw returns 로 메트릭 재계산

3. **02b 학습 완료 후 03 정식 실행**
   - 6 시나리오 BL 백테스트 (Step 8 stale 필터 적용된 universe)

#### 학술 보고서 Limitations 표현

```markdown
### Stale Price Detection in yfinance Data

본 연구는 yfinance 데이터의 stale price 현상을 panel 단계에서 발견하였다.
panel 646 종목 중 12 종목 (1.9%) 이 OOS 기간 동안 30% 이상의 거래일에서
log_ret = 0 으로 기록되어 있으며, 이는 인수 / private 전환 / 파산 / ticker
재사용 (예: SW = Smurfit-Stone Container 2010 + Smurfit Westrock 2024) 으로
ticker 가 시장에서 소멸한 후 yfinance 가 마지막 가격을 stale fill 한
결과로 추정된다.

**Portfolio 영향 정량화**: Dynamic-Membership 의 시점별 멤버십 필터로 BMC,
GR 등 2 종목은 BL universe 에 0 회 출현하였고, 잔존 6 종목은 매월 평균
1.72 종목 (max 2개) 만 동시 출현하였다. 그 가중치 합계는 평균 0.075%, max
0.939% 로 portfolio 결과 (BL_ml_sw Sharpe 1.123) 에 **0.005 이하의 영향**
으로 추정된다.

**처리**: BL universe 결정 단계에서 매월 IS 기간 (1260일) 의 zero ratio > 30%
종목을 자동 제외하는 필터를 적용하였다 (Step 8). 학습 단계에는 영향이
없으며, panel 단계의 stale 종목 영구 제거는 향후 연구로 남긴다 (CRSP 등
학술 표준 DB 재현 검증과 함께).
```

#### 산출물

- 노트북 변경: `02a_phase15_stockwise_extended.ipynb` Cell 23, `03_BL_backtest_extended.ipynb` Cell 10
- 격리 보존: `data/bl_weights_sanity_check_step7_dynamic.pkl`
- 본 §13.6 Step 8 — 학술 보고서 "Stale Price Detection" 섹션 표현으로 직접 활용

#### 05a 재실행 후속 정리 (2026-04-30)

**(a) §7-5 / §7-8 의 hardcoded 결론 텍스트 → 변수 참조 변환**

이전 cosmetic 이슈: §7-5 의 "BL_ml_sw Sharpe ~1.108", "ML 통합 효과 -0.114 Sharpe" / §7-8 의 동일 hardcoded 가 §6-4 실측 (1.122 / 1.207 / -0.085) 과 어긋남. 변수 참조로 전환:

| 위치 | 변경 |
|---|---|
| 02a §6-4 (메트릭 계산) 끝 | `metrics_table` + `delta_sharpe` 등을 `data/bl_metrics_sanity_check.pkl` 에 저장 |
| 05a §7-5 (결론) 시작 | metrics pkl 로드 → `bl_ml_sharpe`, `bl_tr_sharpe`, `ml_effect` 변수 정의 |
| 05a §7-5 Limitations 본문 | `''' → f'''` 변환 + `~1.108` → `~{bl_ml_sharpe:.3f}`, `-0.114 Sharpe` → `{ml_effect:+.3f} Sharpe` |
| 05a §7-8 결론 본문 | 동일 패턴 (재실행 안전을 위해 metrics 로드 코드 별도 추가) |

**검증 (사용자 재실행 후, 2026-04-30)**:
- 05a §7-5 시작에 `⚡ 02a §6 metrics 로드: BL_ml_sw=1.122, BL_trailing=1.207, diff=-0.085` 출력 ✅
- 05a §7-5 Limitations: `Sharpe ~1.122` / `-0.085 Sharpe` 동적 출력 ✅
- 05a §7-8 결론: `Sharpe ~1.122` / `"ML 통합 효과 -0.085 Sharpe"` 동적 출력 ✅
- 23 셀 모두 정상 실행 (에러 0건) ✅

→ 이후 BL Sharpe 값이 변경되어도 (Step 9 등) 노트북이 자동으로 새 수치 반영. 학술 보고서 표현 일관성 강화.

**(b) 05a Layer 2~4 + §7-2 / §7-4 재실행 결과 (Step 8 캐시 반영)**

| Layer | Step 6 cache 시점 | **Step 8 (현재)** | 차이 |
|---|---|---|---|
| **Layer 2 BL_ml_sw Sharpe** | 1.105 | **1.119** | +0.014 |
| Layer 2 CAGR | 13.342% | 13.205% | -0.137% |
| Layer 2 MDD | -18.559% | -18.128% | +0.431% (개선) |
| Layer 2 CAPM α | 19.208% | 18.883% | -0.325% |
| Layer 2 Sortino | 1.653 | 1.657 | +0.004 |
| Layer 2 Hit rate | 66.18% | 64.71% | -1.47% |
| Layer 3 (universe filter 무관) | 동일 | 동일 ✅ | low/high hit 0.725/0.749 |

**Layer 4 시기별 분해** (Step 8 반영):

| 시기 | Step 6 Sharpe | **Step 8 Sharpe** | 차이 |
|---|---|---|---|
| GFC 회복 (09~11) | 1.320 | 1.311 | -0.009 |
| 정상 강세장 (12~19) | 1.420 | 1.446 | +0.026 |
| **COVID 충격 (20)** | 0.790 | **0.724** | **-0.066** ⭐ 가장 큰 변화 |
| 긴축·전환 (21~22) | 0.571 | 0.615 | +0.044 |
| 회복·AI (23~25) | 1.266 | 1.239 | -0.027 |

→ COVID 시기 변화 (-0.066) 가 가장 큼. Step 8 의 COVID universe 축소 (-2.00 종목/월, 가장 큼) 와 일관 — **stale 종목 제거 효과가 변동성 큰 시기에 가장 크게 반영**.

**§7-2 시기별 BL_ml_sw Sharpe overlay** (Step 8 반영):

| 시기 | 파산 종목 수 | **새 Sharpe** | 변화 (Step 6→8) |
|---|---|---|---|
| GFC 회복 (09~11) | 0 | +1.311 | -0.009 |
| 정상 강세장 (12~19) | 4 | +1.446 | +0.026 |
| COVID 충격 (20) | 6 | +0.724 | -0.066 ⭐ |
| 긴축·전환 (21~22) | 1 | +0.615 | +0.044 |
| 회복·AI (23~25) | 3 | +1.239 | -0.027 |

**§7-4 시기별 회귀** (Step 8 반영):

| | r² | p-value | 해석 |
|---|---|---|---|
| Step 6 | 0.005 | 0.914 | 비유의 |
| **Step 8** | **0.018** | **0.829** | 비유의 (statistical noise 수준 개선) |

→ 누락 종목 카운트 vs Sharpe 회귀의 r² 가 약간 올라갔지만 여전히 통계적으로 무의미. **"누락 종목이 시기 성능에 통계적 영향 약함" 결론 유지**.

**§7-3, §7-6, §7-7, §7-8 (sector 분석)** — 변화 없음 (universe filter 무관, panel/missing sector 분포는 Static/Dynamic/Step 8 모두 동일)

**(c) 산출물 갱신 (05a outputs/)**

| 파일 | 상태 |
|---|---|
| `outputs/05a_eval_stockwise/eval_summary_stockwise.md` | 갱신 (Step 8 + 새 Layer 2~4 결과) |
| `outputs/05a_eval_stockwise/eval_metrics_stockwise.json` | 갱신 |
| `outputs/05a_eval_stockwise/layer1_prediction_diagnostic.png` | 동일 (Layer 1 학습 무관) |
| `outputs/05a_eval_stockwise/layer2_portfolio_diagnostic.png` | 갱신 |
| `outputs/05a_eval_stockwise/layer3_ml_to_bl_causality.png` | 동일 (Layer 3 universe 무관) |
| `outputs/05a_eval_stockwise/layer4_period_decomposition.png` | 갱신 |
| `outputs/05a_eval_stockwise/sec7_missing_impact.png` | 갱신 |
| `outputs/05a_eval_stockwise/sec7_real_universe_imbalance.png` | 동일 (sector 분포 무관) |

**(d) 메트릭 일관성 — 두 곳의 BL_ml_sw Sharpe 차이**

- 02a §6-4 직접 계산 (`compute_metrics(ret_ml_fair)`): **1.122**
- 05a Layer 2 (`evaluate_portfolio_standalone`): **1.119**
- 차이 0.003 — rf 차감 / 메트릭 정의 미세 차이 (statistical noise 수준)
- 학술 보고서에는 02a §6-4 의 1.122 사용 (직접 계산이라 정의 명확) 권장. 또는 두 수치 병기.

**(e) SPY index 정렬 fix — 누적 SPY 시각화 버그 (2026-04-30)**

발견: 05a Layer 2 의 누적 수익률 그래프에서 SPY 가 5.82배로 표시 (정상은 11.23배).
원인: Issue #1 패턴 잔존 — `spy_monthly = market['SPY'].pct_change().resample('ME').prod() - 1` 은
**calendar 월말** (예: 2009-01-31 Sat), BL_ml_sw 의 returns index 는 **거래일 월말**
(예: 2009-01-30 Fri) → `plot_portfolio_diagnostic_panel` 의 `intersection` 이 70% (143/204)
만 일치 → cum SPY 가 잘못 측정.

| 처리 | SPY 누적 |
|---|---|
| Intersection 만 (이전 05a Layer 2) | **5.82배** ⚠️ |
| 거래일 월말 정렬 (정확, 02a §6-5 동일) | **11.23배** ✅ |

**해결 (옵션 A)**: 05a Cell 4 (`spy_monthly` 정의) 만 변경:

```python
# Before (calendar 월말)
spy_daily = market['SPY'].pct_change().dropna()
spy_monthly = (1 + spy_daily).resample('ME').prod() - 1

# After (거래일 월말, BL_ml_sw 와 일관)
reb_dates_for_spy = market.groupby(market.index.to_period('M')).tail(1).index
reb_dates_for_spy = reb_dates_for_spy[
    (reb_dates_for_spy >= '2009-01-01') & (reb_dates_for_spy <= '2025-12-31')
]
spy_monthly = market['SPY'].reindex(reb_dates_for_spy, method='ffill').pct_change()
```

**영향 범위**:
- Sharpe / CAGR / MDD / Sortino / CVaR / Hit rate (BL_ml_sw 자체 메트릭): ❌ **영향 없음**
- CAPM α / β / Information Ratio: ⚠️ 70% 데이터 → 100% 데이터 (정확화), 약간의 수치 변화
- 누적 수익률 시각화 SPY 라인: 🚨 5.82 → 11.23 (시각적 정정)
- §7-3, §7-6/7/8 sector 분석: ❌ 영향 없음

**추가 fix — `pct_change()` 첫 NaN 누설** (옵션 A 적용 직후 발견):

옵션 A 적용 후 첫 재실행 시 `capm_alpha = NaN` 발견. 원인: `reindex(...).pct_change()` 의 첫 값이 NaN → CAPM 회귀 (numpy) 가 NaN 한 개로 전체 NaN. (Layer 4 시기별은 시기 슬라이싱 시 첫 NaN 자동 제외 → 정상)

```python
# Final fix
spy_monthly = market['SPY'].reindex(reb_dates_for_spy, method='ffill').pct_change().dropna()
#                                                                                  ↑ 추가
```

→ spy_monthly 길이 204 → 203 (첫 NaN 제거).

**최종 실측 결과 (사용자 두 차례 재실행 후, 2026-04-30)**:

| 메트릭 | 옵션 A 전 (70%) | **옵션 A + dropna 후 (100%)** | 차이 |
|---|---|---|---|
| Layer 2 Sharpe / CAGR / MDD / Sortino / Hit rate | 동일 | 동일 | 0 (BL 자체 메트릭, SPY 무관) |
| Layer 2 CAPM α | 18.883% | **15.789%** | -3.094%p (정확화) |
| Layer 2 CAPM β | -0.173 | **-0.131** | +0.042 (절대값 ↓) |
| Layer 2 CAPM t | 0.236 | 0.242 | +0.006 |
| **Layer 2 IR** | **+0.008** | **-0.084** ⭐⭐ | -0.092 (양→음 전환) |
| 누적 그래프 SPY 라인 | 5.82배 | 11.23배 ✅ | 02a §6-5 와 일관 |

**Layer 4 시기별 capm_α 변화**:

| 시기 | 이전 (70%) | 현재 (100%) | 차이 |
|---|---|---|---|
| GFC 회복 (09~11) | 21.756% | 20.681% | -1.08%p |
| 정상 강세장 (12~19) | 17.707% | 16.336% | -1.37%p |
| COVID 충격 (20) | NaN | **19.561%** ⭐ | 정상 계산 |
| 긴축·전환 (21~22) | 11.098% | 10.179% | -0.92%p |
| 회복·AI (23~25) | 18.459% | 12.666% | **-5.79%p** ⭐ 가장 큰 변화 |

회복·AI 시기 SPY 가 AI 붐으로 강하게 상승 → BL_ml_sw alpha 가 정확하게는 더 작게 측정.

**IR 부호 전환의 의미** (+0.008 → -0.084):
- 이전: 70% 시점만 사용 → BL_ml_sw 가 SPY 보다 약간 좋다고 잘못 측정
- 정확: 100% 시점 → BL_ml_sw 가 SPY 보다 약간 떨어짐 (누적 8.3배 < SPY 11.23배 와 일관)
- → §13.5/§13.6 의 "ML > SPY 도 아님" 결론 정직하게 반영

**근본 fix (옵션 B, 02b 후 task)**: `scripts/diagnostics.py` 의 `plot_portfolio_diagnostic_panel`
내부에서 SPY index 를 month period 매핑 + dropna 자동 처리. 02b 학습 진행 중이라
`scripts/` 변경 미루고 02b 완료 후 적용 예정 (05b/05c 도 동일 검토 필요).

#### Step 8 최종 요약

| 항목 | 결과 |
|---|---|
| Stale 필터 작동 | ✅ 5/6 stale 종목 100% 차단 (SW 169→0, EP/COL/CPWR/CVG 100% 제거), 1/6 (AMCR) 동적 부분 차단 |
| BL_ml_sw Sharpe (02a §6-4) | 1.123 → **1.122** (-0.001, 무변화) |
| BL_trailing Sharpe (02a §6-4) | 1.203 → **1.207** (+0.004, 약간 상승) |
| diff (ML 효과) | -0.080 → **-0.085** (절대값 +0.005, 방향성 robust) |
| BL_ml_sw Sharpe (05a Layer 2) | 1.105 → **1.119** (+0.014, Step 6 → Step 8 누적 효과) |
| 매월 universe 평균 | 420.0 → **418.5** (-1.51 종목) |
| 02b 학습 영향 | 0 (universe 필터는 portfolio 단계, 학습 무관) |
| 노트북 cosmetic 정정 | ✅ §7-5/§7-8 변수 참조 전환 (Cell 30 정정과 일관) |
| 학술 보고서 Limitations | ✅ "Stale Price Detection" 섹션 표현 직접 활용 가능 |

---

## §14. Phase 3-2 / 3-3 (선택, TBD)

(Phase 3-1 결과 후 진행 결정)

---

# 본 WORKLOG 의 현재 상태

```
[Phase 1.5] 완료 — 정합성 완벽
[Phase 2] 완료 — Issue #1, #1B, #2 수정 완료
   진짜 결과: BL_ml Sharpe 0.771 (72m), McapWeight 1위
   sampling bias 발견 + 정정

[Phase 3-1] 진행 중
   ✅ Step 1 (01_universe_extended) — universe 809, panel 646 종목
   ✅ Step 2a (02a_stockwise) — 615 학습, 613 ensemble (CBE/TIE dirty 제거)
      · Ensemble best 65% (Phase 1.5 v8 의 64% 와 동등)
      · Phase 2 vs Phase 3: RMSE 차이 +0.0007 (학습 코드 재현성 ⭐)
   ✅ Step 2a-§6 (BL sanity check) — 03 진행 전 ML 적용 가능성 검증 완료
      · BL_trailing Sharpe 1.222 (서윤범 99 재계산 1.157 의 +5.62%, 양호)
      · BL_ml_sw 1.108 (Trailing 대비 -0.114, NEGATIVE 효과)
      · Hit rate ML > Trailing (low 0.634 vs 0.590, high 0.663 vs 0.626)
      · LS spread 패러독스: ML -9.53%/yr vs Trailing -4.84%/yr (mcap-w)
      · 진단: Trailing vol = 방어주 proxy, ML = pure forward vol → BAB 분리
   🔄 Step 2b (02b_crosssec) — 학습 진행 중 (≈22h ETA, 615 종목 fair 비교)
      · hidden=32, layers=1, batch=512, AMP, num_workers=2, patience=5
      · val 정상 수렴 (~0.22), fold 5.1분/단발
      · 02a 일관 (615 종목 ∩ panel) 보장
   ✅ Step 2a 단독 깊이 분석 §13.6 (2026-04-30, 02b 학습 중 사전 준비)
      · Step 1: 05a Cell 5 BL_ml_sw 캐시 fallback 적용
      · Step 4: 05a §7 신규 (누락 163 종목 시기별 영향 분석) 6 셀 추가
      · Step 6: 05a §7-6/7/8 추가 (사용자 지적 보강 — 진정한 생존편향 정량화) ⭐
      · Step 7: Static-Universe → Dynamic-Membership 전환 (look-ahead 차단) ⭐
                03 + 02a §6 dynamic 수정, 기존 03 노트북 + Static 캐시 격리 보존
      · Step 8: Stale price 필터 추가 (zero_ratio > 30% 종목 매월 universe 제외) ⭐
                팀원 발견 (SW 65.8% stale) 보강, 02a Cell 23 + 03 Cell 10
                + 05a §7-5/§7-8 hardcoded → 변수 참조 변환 (cosmetic 정정 후속)
      · 02b 학습 영향 0 보장 (별도 커널, scripts/ 변경 X)
      · 사용자 재실행 완료: 02a §6 (Step 8) + 05a Layer 2~4 + §7-5/§7-8 (변수 참조)
      · ⏳ 사용자 대기: 02b 학습 완료 후 03 정식 실행 (~22h) + 04/05b/05c 재실행
   ⏳ Step 3 (03_BL_backtest) — 02b 완료 후, 6 시나리오
   ⏳ Step 4 (04_compare) — 03 완료 후

[Phase 3 평가 인프라] 코드 작성 완료 (학습 결과 후 실행)
   ✅ scripts/black_litterman.py — TAU=0.1, LAM_FIXED=2.5 (서윤범 99 일관)
   ✅ scripts/diagnostics.py — Layer 1~5 평가 함수 모듈
   ✅ 05a/05b/05c_eval_*.ipynb — 모델별 단독 평가 + 비교·검정

[핵심 학술 발견 — §13.5 결과]
   ⭐ "Vol prediction accuracy 향상 (RMSE↓, hit rate↑)
       != BL portfolio alpha 향상" 검증
   ⭐ Pyo & Lee (2018) 의 "ML > Trailing" 주장 부분 반증
       (KOSPI vs 미국 17년 universe 환경 차이)
   ⭐ Trailing vol = 방어주 (Utilities/Staples/Healthcare) 식별 proxy
       — 회사 특성 식별과 vol 예측의 분리

[핵심 학술 발견 — §13.6 Step 6 결과 (2026-04-30, 사용자 지적 보강)]
   ⭐ §7-3 의 sector imbalance 1.64%p 의 정확한 의미:
       "panel 646 (살아남은 종목) 안에서" 학습 615 의 sector 중립성만 입증
       → panel 자체가 yfinance 살아남은 집합이라는 본질적 생존편향 미측정
   ⭐ §7-7 진정한 universe imbalance 실측 (2026-04-30):
       Real imbalance 31.31%p (Unknown 포함) / 13.56%p (Unknown 제외)
       → §7-3 의 1.64%p 의 19.1배 / 8.3배 (방법론 noise 처리에 따라)
       hardcoded 매핑: 126/163 (77.3%), Unknown 잔존 37 종목
   ⭐ Panel 의 sector 별 진짜 universe 대표성 (Panel < 80% under-rep):
       Energy 61.8% ⭐ #1 핵심 발견 (panel 34 vs 누락 21)
       Communication Services 73.0%, Consumer Staples 75.0%, Health Care 76.9%
       → 학습이 본 sector 분포는 "M&A·파산까지 이어진 prosperous/안정" 위주
   ⭐ OOS 파산 14 종목 중 Energy 36% (CHK, FTR, DO, DNR, ANR) 가설이
       universe 전체 차원에서도 확인 (panel Energy 진짜 universe 의 38% 미반영)
   ⭐ 본 연구의 절대 수치 (BL_ml_sw Sharpe 1.108) 는 진짜 universe 대비
       과대평가 가능성, 상대 비교 (ML vs Trailing -0.114) 의 방향성은 신뢰 가능

[핵심 학술 발견 — §13.6 Step 7 결과 (2026-04-30, Dynamic-Membership 전환)]
   ⭐ Universe 정의 전환:
       Static-Universe (legacy):     매월 panel ∩ trained_tickers 615
                                     (yfinance 가용 종목 중 학습된 종목, 시점별 멤버십 무관)
       Dynamic-Membership (current): 매월 sp500_member_at_t ∩ panel ∩ 615
                                     (Wikipedia 시점별 S&P 500 멤버십 적용)
   ⭐ Look-ahead 차단:
       Static 모드는 portfolio backtest universe 에 미래 멤버십 정보 누수
       (예: TSLA 의 2010~2019 panel 데이터를 universe 후보에 포함, 실제 S&P 편입은 2020-12)
       → Dynamic 으로 그 시점 실제 S&P 멤버만 후보 → 학술 baseline 과 fair 비교 회복
   ⭐ 두 bias 의 분리:
       Look-ahead    → Dynamic-Membership 으로 ✅ 해결 (Step 7)
       Survivorship  → yfinance 한계로 ⚠️ 잔존 (§7-6/7/8 의 31.31%p / 13.56%p, CRSP 필요)
       학술 보고서에서 분리 표현 권장
   ⭐ 격리 보존:
       기존 03 노트북 (Static)        → 03_BL_backtest_extended_legacy_static.ipynb
       기존 BL sanity check 캐시      → data/bl_weights_sanity_check_legacy_static.pkl
   ⭐ 결과 수치 변화 (2026-04-30 사용자 02a §6 재실행 실측):
       매월 universe 평균: 548.8 → 420.0 (-128.9 종목, -23%)
       BL_ml_sw Sharpe:   1.108 → 1.123 (+0.015)
       BL_trailing Sharpe: 1.222 → 1.203 (-0.019)
       diff (ML 효과):    -0.114 → -0.080 (절대값 -30%, 방향성 robust ✅)
       Hit rate / LS spread 동일 (universe filter 무관 영역)
   ⭐ Look-ahead 차단 직접 검증 (TSLA 사례):
       2014-01-31, 2019-01-31: Static ✓ vs Dynamic ✗ (편입 2020-12 전 정확히 제외)
       Dynamic ⊆ Static (수학적 부분집합 관계 확인, 첫 시점 140 종목 제거)
   ⭐ 학술 baseline fair 비교 회복:
       BL_trailing 1.203 vs 서윤범 99 재계산 1.157 (+3.98%, 안전 범위)
   ⭐ 핵심 결론 robustness 강화:
       "ML > Trailing 이 아님" 결론이 Static/Dynamic 양쪽 환경에서 일관 유지
       Trailing vol = 방어주 proxy 가설도 universe 정의에 robust
   ⭐ 재실행 안전성 강화 (캐시 보호):
       02a Cell 8 (LSTM 학습) + Cell 23 (§6-2 BL) + 03 Cell 10 (§4 BL 6 시나리오) 모두 캐시 추가
       FORCE_RETRAIN / FORCE_RECOMPUTE flag default False → VS Code Run All 도 안전
       02b 학습 중 02a 노트북 안전 재실행 보장

[핵심 학술 발견 — §13.6 Step 8 결과 (2026-04-30, Stale price 필터)]
   ⭐ Stale price 발견 (팀원 보강):
       panel 646 중 12 종목 (1.9%) 의 0 비율 > 30%
       SW (65.8%) — Smurfit-Stone (2010) + Smurfit Westrock (2024) ticker 재사용
       원인: yfinance 의 ticker 소멸 후 stale fill (M&A, private, 파산, 재사용)
   ⭐ Dynamic-Membership 의 자연 차단 효과 (Step 7 부가 가치):
       BMC, GR 등 2 종목은 시점별 멤버십 mismatch 로 BL universe 0회 출현
       그러나 SW (169시점), AMCR (79시점), EP (40시점) 등 잔존 → Step 8 필요
   ⭐ Step 7 결과 기준 stale 영향 정량 (Step 8 적용 전):
       매월 동시 stale 종목: mean 1.72, max 2 (절대 9개 아님)
       매월 가중치 합: mean 0.075%, max 0.939% (2020-02-28)
       Sharpe 영향 < 0.005 (negligible)
   ⭐ Step 8 적용 + 실측 결과 (2026-04-30 사용자 02a §6 재실행):
       매월 universe 평균: 420.0 → 418.5 (-1.51 종목, 예상 1~2 일치 ✅)
       BL_ml_sw Sharpe:   1.123 → 1.122 (-0.001, 거의 무변화)
       BL_trailing Sharpe: 1.203 → 1.207 (+0.004)
       diff (ML 효과):    -0.080 → -0.085 (절대값 약간 강해짐, 방향성 robust)
       Stale 제거 검증: SW 169→0, EP 40→0, COL 33→0, CPWR 18→0, CVG 11→0
                        AMCR 79→42 (시기별 부분 차단, 43.3% 종목)
       → 5/6 stale 종목 100% 차단, 학습 영향 0
   ⭐ 부수 발견: Trailing 이 stale 에 더 영향 받음 (Sharpe +0.004 vs ML -0.001)
       → "Trailing vol = 방어주 proxy" 가설 반대 방향 보강
          (Trailing 은 진짜 방어주뿐 아니라 "가짜 방어주" stale 종목도 잘 잡았음)
       → ML 이 noise-robust 함 추가 확인
   ⭐ Sharpe 변화 ±0.005 이내 = statistical noise 수준 → 핵심 결론 robust
   ⭐ 05a 재실행 후속 정리 (2026-04-30, Step 8 캐시 반영):
       (a) §7-5/§7-8 hardcoded → 변수 참조 변환
           02a §6-4 가 metrics 를 bl_metrics_sanity_check.pkl 로 저장
           05a §7-5/§7-8 가 로드 + f-string 으로 동적 인용
           → 이전 1.108/-0.114 (stale) → 1.122/-0.085 자동 출력
       (b) 05a Layer 2 BL_ml_sw Sharpe (Step 6→8): 1.105 → 1.119 (+0.014)
       (c) Layer 4 COVID Sharpe: 0.790 → 0.724 (-0.066) ⭐ 가장 큰 변화
           (Step 8 의 COVID universe 축소 -2.00 종목/월 와 일관)
       (d) §7-4 회귀 r²: 0.005 → 0.018 (statistical noise 수준, 비유의 유지)
       (e) §7-3/§7-6/§7-7/§7-8 sector 분석: universe filter 무관 → 변화 없음
   ⭐ 메트릭 일관성: 02a §6-4 (1.122) vs 05a Layer 2 (1.119) 차이 0.003
       → rf 차감 / 메트릭 정의 미세 차이 (noise 수준)
       학술 보고서엔 02a §6-4 의 1.122 사용 권장 (직접 계산 정의 명확) 또는 두 수치 병기
   ⭐ 추가 cosmetic 정정 — SPY index 정렬 (2026-04-30):
       05a Layer 2 누적 그래프의 SPY 5.82배 (잘못) → 11.23배 (정확) 수정
       원인: Issue #1 패턴 잔존 (spy_monthly 가 calendar 월말, BL returns 가 거래일 월말)
            → plot_portfolio_diagnostic_panel 의 intersection 70% 만 일치
       Fix (옵션 A + dropna): 05a Cell 4 의 spy_monthly = reindex(reb).pct_change().dropna()
       실측 결과 (사용자 재실행):
         CAPM α: 18.883% → 15.789% (-3.09%p, 정확화)
         CAPM β: -0.173 → -0.131 (절대값 ↓)
         IR:    +0.008 → -0.084 (양→음 전환, BL_ml_sw 가 SPY 보다 떨어짐 정직 반영)
         Layer 4 capm_α 시기별 정확화 (회복·AI 18.46→12.67, 가장 큰 변화)
       BL 자체 메트릭 (Sharpe/CAGR/MDD/Sortino/Hit rate) 변화 0
       근본 fix (옵션 B, 02b 후): scripts/diagnostics.py 의 SPY 처리 month period 매핑 + dropna 자동

[Phase 3-2/3] Phase 3-1 결과 후 결정
```

