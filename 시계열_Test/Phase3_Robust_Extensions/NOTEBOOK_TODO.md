# Phase 3 — 노트북 보완 사항 (Code Review 잔여)

> **목적**: scripts 모듈의 잔여 한계 (코드 review 2 차 발견) 를 노트북에서 보강하기 위한 가이드.
> **작성**: 2026-04-29
> **출처**: WORKLOG.md §11.6 + PLAN.md §11

---

## 0. 현재 진행 상태 (2026-04-29 갱신) ⭐

| Step | 노트북 | 상태 | 산출물 |
|---|---|---|---|
| 1 | `01_universe_extended.ipynb` | ✅ 완료 | universe 809, panel 646 종목 (2002~2025) |
| 2a | `02a_phase15_stockwise_extended.ipynb` | ✅ 완료 | 615 학습 → 613 ensemble (RMSE 0.391) |
| 2a-§6 | (02a 내부) BL sanity check | ✅ 완료 | BL_trailing 1.222 / BL_ml_sw 1.108 / SPY 1.050 |
| 2b | `02b_phase15_cross_sectional.ipynb` | ⏳ 대기 | (학습 1-2h 예상) |
| 3 | `03_BL_backtest_extended.ipynb` | ⏳ 02b 후 | 6 시나리오 비교 |
| 4 | `04_compare_stockwise_vs_cross.ipynb` | ⏳ 03 후 | 종목별 vs CS RMSE 비교 |
| 5a | `05a_eval_stockwise.ipynb` | 빌드 완료 | Layer 1~4 평가 (학습 후 실행) |
| 5b | `05b_eval_crosssec.ipynb` | 빌드 완료 | Layer 1~4 + Embedding 평가 |
| 5c | `05c_eval_compare.ipynb` | 빌드 완료 | Layer 5 통계 검정 + 종합 보고 |

### 02a 학습 후 발견 + 수정 이슈 (학습 코드 X, 후처리 모듈)

- **Issue 3**: `compute_performance_weights` ZeroDivisionError + 들여쓰기 (수정 ✅)
  - 영향: 폐상장 stale price 10 종목 → log(0)=-inf
- **Issue 4**: `estimate_covariance` 호출 인자 누락 (수정 ✅, 우회 적용)
  - 03 노트북도 동일 문제 가능 → 추후 추적 필요
- **Issue 5**: `backtest_strategy` 빈 Series 반환 (우회 ✅)
  - 인덱스 dtype 정상이지만 dropna 후 empty
  - `make_returns_manual()` 작성하여 직접 계산

### ⭐ §6 BL Sanity Check 핵심 발견 (2026-04-29)

| 측정 | ML (02a) | Trailing | 차이 |
|---|---|---|---|
| BL Sharpe | 1.108 | **1.222** | -0.114 |
| Hit rate (low) | **0.634** | 0.590 | +4.4%p |
| Hit rate (high) | **0.663** | 0.626 | +3.7%p |
| LS spread (yr) | -9.53% | -4.84% | -4.70%p |

→ **Hit rate ↑ 인데 BL 성과 ↓ 패러독스 발견**
→ Trailing vol = 방어주 식별 proxy 가설 (자세한 내용 WORKLOG §13.5)
→ 02b cross-sectional 의 BAB 활용도 검증이 핵심 다음 단계

---

## 1. 잔여 한계 정리

### 1.1 코드 review 2 차에서 발견된 부분 해결 issue

| Issue | Severity | 위치 | 잔여 해결 방법 |
|---|---|---|---|
| **C4** | 🚨 | volatility_ensemble.py | 노트북 02b 에서 panel date 기반 fold |
| **Mj5** | ⚠️ | volatility_ensemble.py | 노트북 02b 에서 종목별 date 검증 |

본 issue 들은 scripts 모듈에서 **부분 해결** (NaN seq window, 시그니처 등) 되었으나, **종목 length 차이 처리** 는 노트북에서 직접 보강 권장.

---

## 2. 노트북별 보완 사항

### 2.1 `01_universe_extended.ipynb`

#### 작업 흐름

```python
# §1. 환경
from scripts.setup import bootstrap, DATA_DIR
font_used = bootstrap()

# §2. universe 확장 (2009-2025, top 50)
from scripts.universe_extended import extend_universe, diagnose_universe_coverage

universe_df = extend_universe(
    start_year=2009,
    end_year=2025,
    n_top=50,
    cache_dir=DATA_DIR,
)

# §3. panel 확장 (build_daily_panel 자동 7년)
from scripts.universe_extended import extend_panel_to_2009

panel_df = extend_panel_to_2009(
    cache_dir=DATA_DIR,
    universe_df=universe_df,
)

# §4. 가용성 진단
coverage = diagnose_universe_coverage(universe_df, panel_csv=DATA_DIR / 'daily_panel.csv')
print(coverage.head(20))
```

#### ⚠️ 주의사항

1. **첫 실행 시 30-60 분 소요** (yfinance API)
2. **캐시 활용 시 1-2 분** (재실행)
3. **build_daily_panel 시점**:
   - earliest_cutoff = 2008-12-31 (2009 OOS 시작)
   - panel start = 2001-12-31 (자동, 7 년 buffer)
   - panel end = 2026-01-01

#### 검증 항목

```python
# 검증 1: panel 의 시작/끝 시점
assert panel_df['date'].min() <= pd.Timestamp('2003-12-31'), \
    "panel 시작이 너무 늦음"
assert panel_df['date'].max() >= pd.Timestamp('2025-12-31')

# 검증 2: 17 년 OOS 모두 포함
assert universe_df['oos_year'].min() == 2009
assert universe_df['oos_year'].max() == 2025

# 검증 3: 종목별 panel 시작 시점 검증
panel_start_per_ticker = panel_df.groupby('ticker')['date'].min()
print(f'종목별 panel 시작 분포:')
print(panel_start_per_ticker.value_counts(bins=10))
```

---

### 2.2 `02a_phase15_stockwise_extended.ipynb`

#### 작업 흐름

```python
# §1. 환경
from scripts.setup import bootstrap, DATA_DIR, OUTPUTS_DIR
font_used = bootstrap()

# §2. 종목별 8-way 병렬 학습
from scripts.volatility_ensemble import run_ensemble_for_universe_parallel, V4_BEST_CONFIG

ensemble_df = run_ensemble_for_universe_parallel(
    panel_csv=DATA_DIR / 'daily_panel.csv',
    universe_csv=DATA_DIR / 'universe_top50_history_extended.csv',
    out_dir=DATA_DIR,
    config=V4_BEST_CONFIG,
    n_workers=8,    # ⭐ RTX 4090 24GB
    out_name='ensemble_predictions_stockwise.csv',
    verbose=True,
)
# 학습 시간: 약 1-2 시간

# §3. 결과 검증 + 시각화
print(ensemble_df.shape)
print(ensemble_df.head())
```

#### ⚠️ 잔여 한계 보강 (Issue 해당 사항 적음)

**본 노트북은 종목별 학습 → C4, Mj5 잔여 issue 해당 X**.

#### 권장 추가 분석

1. 종목별 RMSE 분포 시각화
2. Best 모델 분포 (LSTM vs HAR vs Ensemble)
3. Phase 1.5 v8 결과와 비교 (74 종목 6 년 vs 17 년 확장)

---

### 2.3 `02b_phase15_cross_sectional.ipynb` ⭐ **가장 중요한 보완**

#### ⚠️ Issue C4 보강 — 종목 length 처리

**문제**:
```python
# 현재 scripts/volatility_ensemble.py
common_length = min(ticker_lengths.values())
```
- 가장 짧은 종목 기준 → 긴 종목은 일부 시기 학습 X
- 신규 IPO (META 2012, Snowflake 2020 등) 영향

**노트북에서 보완 옵션 1: panel date 기반 fold (권장)**

```python
# 모든 종목 공통 시점 기반 fold
import pandas as pd
import numpy as np
from scripts.volatility_ensemble import (
    run_ensemble_cross_sectional,
    build_cs_inputs,
    CrossSectionalDataset,
    _build_cs_dataset_for_fold,
)
from scripts.models_cs import CrossSectionalLSTMRegressor, CS_V4_BEST_CONFIG

panel = pd.read_csv(DATA_DIR / 'daily_panel.csv', parse_dates=['date'])
universe_df = pd.read_csv(DATA_DIR / 'universe_top50_history_extended.csv', parse_dates=['cutoff_date'])

# ⭐ 모든 종목 공통 date 추출 (가장 늦은 panel 시작 + 가장 이른 panel 끝)
panel_start_per_ticker = panel.groupby('ticker')['date'].min()
panel_end_per_ticker = panel.groupby('ticker')['date'].max()
common_start = panel_start_per_ticker.max()    # 모든 종목이 시작한 시점
common_end = panel_end_per_ticker.min()        # 모든 종목이 끝나는 시점

print(f'공통 시점: {common_start.date()} ~ {common_end.date()}')

# common date range 안의 panel 만 사용
panel_common = panel[
    (panel['date'] >= common_start) & (panel['date'] <= common_end)
].copy()

# 이 panel 으로 cross-sectional 학습
# (run_ensemble_cross_sectional 의 panel_csv 인자에 panel_common 사용)
```

**노트북에서 보완 옵션 2: 종목별 가용 시점 분기 학습 (복잡)**

```python
# IPO 시점 후 종목만 fold 에 포함
# 예: META 의 fold 는 2012-09 (IPO) 이후만
# Snowflake 의 fold 는 2020-09 (IPO) 이후만

# 실용적으로는 옵션 1 권장 (단순 + 안전)
```

#### Cross-sectional 학습 호출

```python
# Cross-sectional 학습
ensemble_df_cs = run_ensemble_cross_sectional(
    panel_csv=DATA_DIR / 'daily_panel.csv',    # 또는 panel_common 의 csv
    universe_csv=DATA_DIR / 'universe_top50_history_extended.csv',
    out_dir=DATA_DIR,
    config=CS_V4_BEST_CONFIG,
    device='cuda',
    use_har=True,    # HAR 결합
    out_name='ensemble_predictions_crosssec.csv',
    verbose=True,
)
# 학습 시간: 약 1-2 시간
```

#### ⚠️ Issue Mj5 보강 — HAR fold 매칭

**문제**:
```python
# scripts/volatility_ensemble.py 의 HAR 결합 부분
date = cs_inputs['date'][ticker][idx]
```
- `idx` 는 common_length 기준
- 종목별 date array length 다르면 mismatch

**노트북에서 검증**:
```python
# 검증: 모든 종목의 date array 가 같은 length 인지
date_lengths = {t: len(cs_inputs['date'][t]) for t in cs_inputs['series']}
unique_lengths = set(date_lengths.values())
print(f'종목별 date array length: {unique_lengths}')

# 만약 unique_lengths 가 1 이 아니면 → mismatch 위험
if len(unique_lengths) > 1:
    print('⚠️ 종목별 date length 다름 → common_length 기준 처리 필요')
    # 옵션 1 적용 후 재실행
```

#### 권장 추가 분석

1. **Cross-sectional 학습 진행 monitoring**:
   - 매 fold 의 train/val loss 곡선
   - 종목별 RMSE 분포
   - Ticker embedding 의 PCA 시각화

2. **종목별 vs Cross-sectional 비교**:
   - 같은 종목의 RMSE 비교 (02a vs 02b)
   - 어떤 종목에서 CS 가 더 좋은가?

3. **신규 IPO 종목 처리 검증**:
   - META, Snowflake 등의 학습 가능 시점 확인
   - 옵션 1 적용 시 손실되는 시기 명시

---

### 2.4 `03_BL_backtest_extended.ipynb`

#### 작업 흐름

```python
# §1. 두 ensemble 결과 로드
import pandas as pd

ens_sw = pd.read_csv(DATA_DIR / 'ensemble_predictions_stockwise.csv', parse_dates=['date'])
ens_cs = pd.read_csv(DATA_DIR / 'ensemble_predictions_crosssec.csv', parse_dates=['date'])

# §2. BL 백테스트 (두 결과 각각)
# Phase 2 의 백테스트 코드 재사용 (Issue #1, #1B, #2 수정 적용)
# - BL_ml_sw: ensemble_predictions_stockwise → P 행렬
# - BL_ml_cs: ensemble_predictions_crosssec → P 행렬
# - 그 외: BL_trailing, EqualWeight, McapWeight, SPY

# §3. 6 시나리오 비교
scenarios = ['BL_ml_sw', 'BL_ml_cs', 'BL_trailing', 'EqualWeight', 'McapWeight', 'SPY']

# §4. Fair 비교 (모든 시나리오 같은 sample)
# 17 년 OOS 의 valid 시점 확인
```

#### ⚠️ 노트북에서 검증할 사항

1. **Date mismatch (Phase 2 의 Issue #1) 재발 방지**:
   ```python
   # rebalance_dates = market 거래일 월말
   # ens_monthly['rebalance_date'] = market eom 매핑 (Phase 2 수정과 일관)
   month_to_market_eom = {pd.Timestamp(d).to_period('M'): pd.Timestamp(d) for d in rebalance_dates}
   ens_monthly['rebalance_date'] = ens_monthly['month'].map(month_to_market_eom)
   ```

2. **forward_rets 처리 (look-ahead bias 방지)**:
   ```python
   forward_rets = monthly_rets.shift(-1)  # ⭐ Phase 2 일관
   ```

3. **λ rf 차감 (Issue #2 일관)**:
   ```python
   spy_excess_monthly = float((spy_lr - rf_lr).mean() * DAYS_PER_MONTH)
   ```

4. **Fair 비교 (모든 시나리오 같은 sample)**:
   ```python
   common_dates = portfolio_returns_dict['BL_ml_sw'].dropna().index
   metrics_fair = {s: compute_metrics(rets.reindex(common_dates).dropna()) ...}
   ```

---

### 2.5 `04_compare_stockwise_vs_cross.ipynb`

#### 작업 흐름

```python
# §1. 두 모델 결과 비교
# RMSE 비교 (종목별)
# Sharpe 비교 (BL_ml_sw vs BL_ml_cs)
# Bootstrap 신뢰구간

# §2. 종목별 특성 분석
# CS 가 더 좋은 종목 identify
# 종목별이 더 좋은 종목 identify
# 신규 IPO 종목 (META 등) 의 결과 분석

# §3. 학술 결론
# Cross-sectional > 종목별?
# 또는 그 반대?
# 시기별 (2009-2014 GFC 회복 vs 2020-2025 강세장) 분해
```

#### 권장 추가 분석

1. **t-test / Wilcoxon for paired comparison**:
   ```python
   from scipy.stats import ttest_rel, wilcoxon
   t_stat, p_value = ttest_rel(rmse_sw, rmse_cs)
   ```

2. **Block Bootstrap (Phase 2 와 일관)**:
   ```python
   # 시계열 구조 보존, n=5000, block_size=3
   ```

3. **시기별 분해**:
   ```python
   periods = {
       'GFC 회복 (2009-2010)': returns.loc['2009':'2010'],
       '정상 강세장 (2011-2019)': returns.loc['2011':'2019'],
       'COVID + AI (2020-2025)': returns.loc['2020':'2025'],
   }
   ```

---

## 3. 학습 실행 명령어 (재정리)

### 3.1 백그라운드 학습 (jupyter nbconvert --execute)

```bash
cd "C:/Users/gorhk/최종 프로젝트/finance_project/시계열_Test/Phase3_Robust_Extensions"

# Step 1: universe + panel 확장 (5 분 캐시 / 30-60 분 첫 실행)
jupyter nbconvert --to notebook --execute \
    01_universe_extended.ipynb \
    --output 01_universe_extended.ipynb \
    --ExecutePreprocessor.timeout=7200

# Step 2: 두 학습 동시 백그라운드 (1-2 시간 동시)
nohup jupyter nbconvert --to notebook --execute \
    02a_phase15_stockwise_extended.ipynb \
    --output 02a_phase15_stockwise_extended.ipynb \
    --ExecutePreprocessor.timeout=86400 \
    > log_stockwise_$(date +%Y%m%d_%H%M).txt 2>&1 &
echo $! > stockwise.pid

nohup jupyter nbconvert --to notebook --execute \
    02b_phase15_cross_sectional.ipynb \
    --output 02b_phase15_cross_sectional.ipynb \
    --ExecutePreprocessor.timeout=7200 \
    > log_crosssec_$(date +%Y%m%d_%H%M).txt 2>&1 &
echo $! > crosssec.pid

# Step 3: 진행 모니터링
tail -f log_stockwise_*.txt log_crosssec_*.txt

# Step 4: 학습 완료 후 BL 백테스트 + 비교
jupyter nbconvert --to notebook --execute 03_BL_backtest_extended.ipynb --output 03_BL_backtest_extended.ipynb
jupyter nbconvert --to notebook --execute 04_compare_stockwise_vs_cross.ipynb --output 04_compare_stockwise_vs_cross.ipynb
```

---

## 4. 검증 체크리스트 (학습 전)

### 4.1 데이터 정합성

- [ ] universe_top50_history_extended.csv 의 oos_year = [2009, 2010, ..., 2025]
- [ ] daily_panel.csv 의 date 시작 ≤ 2003-12-31
- [ ] daily_panel.csv 의 date 끝 ≥ 2025-12-31
- [ ] 종목별 panel 시작 시점 분포 확인 (신규 IPO 식별)

### 4.2 학습 환경

- [ ] RTX 4090 24GB GPU 가용 (`torch.cuda.is_available() == True`)
- [ ] VRAM 여유 ≥ 4 GB (8 worker × 500 MB)
- [ ] CUDA + PyTorch 호환

### 4.3 코드 정합성

- [ ] scripts.volatility_ensemble import OK
- [ ] scripts.models_cs import OK
- [ ] scripts.universe_extended import OK
- [ ] CrossSectionalLSTMRegressor forward pass OK
- [ ] V4_BEST_CONFIG, CS_V4_BEST_CONFIG 정합성

### 4.4 Phase 2 와의 일관성 (BL 백테스트 시)

- [ ] Issue #1 수정: month_to_market_eom 매핑 적용
- [ ] Issue #1B 수정: monthly_rets 의 인덱스 = market eom
- [ ] Issue #2 수정: λ rf 차감
- [ ] Fair 비교: 모든 시나리오 같은 sample

---

## 5. 학습 후 검증 체크리스트

### 5.1 02a (종목별) ✅ 완료 (2026-04-29)

- [x] 산출 csv 의 unique 종목 수 = 613 (universe 615 - dirty 2)
- [x] fold 수 = 224 (장기 종목 최대)
- [x] y_pred_lstm, y_pred_har, y_pred_ensemble 모두 valid
- [x] RMSE 평균 = 0.391 < 0.5 (Phase 1.5 일관) ✅
- [x] Best 모델 분포: Ensemble 65% / HAR 32.6% / LSTM 2.4%
- [x] §6 BL sanity check (BL_trailing 1.222, BL_ml_sw 1.108)

### 5.2 02b (Cross-sectional)

- [ ] 산출 csv 의 unique 종목 수 = inputs 의 종목 수
- [ ] fold 수 일관
- [ ] y_pred_lstm_cs, y_pred_har, y_pred_ensemble valid
- [ ] Ticker embedding norm 분포 (작은 값, < 1)

### 5.3 03 (BL 백테스트)

- [ ] 5+ 시나리오 모두 valid 결과
- [ ] BL_ml_sw, BL_ml_cs 의 sample 수 일관
- [ ] Sharpe 값이 합리적 범위 (0.5 ~ 1.5)
- [ ] 서윤범 BL TOP_50 (1.065) 와 비슷한 수준 도달 여부

### 5.4 04 (비교)

- [ ] RMSE 비교 그래프
- [ ] Sharpe 통계 검정 (paired t-test, Bootstrap)
- [ ] 시기별 분해 (3 시기)
- [ ] 학술 결론 명시

---

## 6. 학습 도중 문제 발생 시 대응

### 6.1 GPU OOM (Out of Memory)

```python
# 대응:
# 1. n_workers 줄임 (8 → 4)
# 2. batch_size 줄임 (256 → 128)
# 3. torch.cuda.empty_cache() 호출
# 4. 학습 중간 저장 활용
```

### 6.2 Multiprocessing 충돌

```python
# 대응:
# 1. spawn 명시 확인 (mp.get_context('spawn'))
# 2. if __name__ == '__main__' 블록 (.py script 변환 시)
# 3. .py script 변환 후 nohup python ... &
```

### 6.3 Cross-sectional 학습 발산

```python
# 대응:
# 1. Gradient clipping 강화 (1.0 → 0.5)
# 2. Learning rate 감소 (1e-3 → 5e-4)
# 3. Embedding init std 감소 (0.01 → 0.005)
# 4. Dropout ↑ (0.3 → 0.5)
```

### 6.4 신규 IPO 종목 학습 X (옵션 1 적용 후)

```python
# 옵션 1: common date range 안의 panel 만 사용 시
# META 2012 등이 common range 시작점을 늦춤
# 대응:
# 1. META 등 신규 종목 universe 에서 제외 (보수적)
# 2. 또는 옵션 2 (종목별 가용 시점 분기) 적용
```

---

## 7. 진행 결정

본 NOTEBOOK_TODO 를 따라 진행:

1. **즉시 진행 가능**:
   - 노트북 빌드 (01, 02a, 02b, 03, 04)
   - 빌드 스크립트 작성

2. **준비 후 진행**:
   - 노트북 빌드 검증 (smoke test)
   - 학습 백그라운드 실행
   - 결과 분석

3. **학습 비용 (RTX 4090 24GB)**:
   - 01 universe + panel: 5-60 분 (캐시 의존)
   - 02a 종목별 8-way: 1-2 시간
   - 02b Cross-sectional: 1-2 시간
   - 03 BL 백테스트: 30 분
   - 04 비교: 30 분
   - **총 약 3-5 시간** (캐시 후) 또는 **5-7 시간** (첫 실행)

---

## 8. 참조

- **PLAN.md**: Phase 3 의 전체 plan
- **재천_WORKLOG.md**: 작업 진행 history
- **scripts/**: 코드 모듈 (review 통과)
- **Phase2_BL_Integration/**: 기존 산출물 재사용

---

본 NOTEBOOK_TODO 는 코드 review 결과의 잔여 한계 + 학습 진행의 실용 가이드. 노트북 빌드 시 본 문서 참조 필수.
