# 🗃️ 데이터 파이프라인 ERD

> **독자**: 엔지니어 / 기술자 (기술자 톤)
> **목적**: 입출력 의존성·스키마·파일 명세 한곳 정리

---

## 📊 전체 ER 다이어그램

```mermaid
erDiagram
    YFINANCE ||--o{ PORTFOLIO_PRICES : "download"
    YFINANCE ||--o{ EXTERNAL_PRICES : "download"
    FRED_API ||--o{ FRED_DATA : "download"

    PORTFOLIO_PRICES ||--|| DF_REG_V2 : "feature_eng"
    EXTERNAL_PRICES ||--|| DF_REG_V2 : "feature_eng"
    FRED_DATA ||--|| DF_REG_V2 : "feature_eng"

    DF_REG_V2 ||--|| PROFILES : "create"
    DF_REG_V2 ||--|| FEATURES : "extract"
    DF_REG_V2 ||--|| GRANGER_RESULTS : "test"
    DF_REG_V2 ||--|| REGIME_HISTORY : "hmm_fit"
    DF_REG_V2 ||--|| ALERT_SIGNALS : "config_abcd"

    PORTFOLIO_PRICES ||--|| OPTIMAL_WEIGHTS : "mv_rp_hrp"
    PROFILES ||--|| OPTIMAL_WEIGHTS : "constraints"

    REGIME_HISTORY ||--|| REGIME_COVARIANCE : "split_is"
    PORTFOLIO_PRICES ||--|| REGIME_COVARIANCE : "returns"

    REGIME_COVARIANCE ||--|| STEP9_RESULTS : "mv_optimize"
    ALERT_SIGNALS ||--|| STEP9_RESULTS : "path1"
    PROFILES ||--|| STEP9_RESULTS : "gamma"

    STEP9_RESULTS ||--|| STEP10_FINAL : "multi_criteria"
    STEP10_FINAL ||--|| STEP11_WEIGHTS : "top10"

    PORTFOLIO_PRICES {
        date Date PK
        float SPY
        float QQQ
        float AGG
        float GLD
        float ...27_others
    }

    DF_REG_V2 {
        date Date PK
        float VIX_level
        float VIX_contango
        float HY_spread
        float HY_spread_chg
        float yield_curve
        float Cu_Au_ratio_chg
        float sahm_indicator
        float claims_zscore
        float SKEW_level
        float rv_neutral
    }

    REGIME_COVARIANCE {
        int window_id PK
        date is_start
        date oos_start
        vector mu
        matrix Sigma_stable
        matrix Sigma_crisis
        str fallback_type
    }

    STEP9_RESULTS {
        str strategy PK "M{mode}_{profile}_{config}"
        series daily_returns
        float sharpe
        float mdd
        float sortino
    }

    STEP11_WEIGHTS {
        str strategy PK
        dataframe weights_daily "30 cols"
        dataframe rebalance_events
        series sigma_selection
        series alert_levels
    }
```

---

## 📁 파일 명세 (입출력 상세)

### Tier 0: 원천 데이터

| 파일 | 출처 | 행 | 열 | 주기 |
|------|------|-----|-----|------|
| yfinance API | Yahoo Finance | - | - | 실시간 |
| FRED API | 세인트루이스 Fed | - | - | 일/주/월 |

### Tier 1: Step 1 산출

| 파일 | 크기 | 행 × 열 | 스키마 |
|------|------|--------|------|
| `portfolio_prices.csv` | 1.4 MB | 2,609 × 31 | Date + 30 tickers |
| `external_prices.csv` | 584 KB | 2,609 × 13 | Date + 12 지표 |
| `fred_data.csv` | 150 KB | 2,609 × 9 | Date + 8 거시 |

### Tier 2: Step 2 산출

| 파일 | 크기 | 행 × 열 |
|------|------|--------|
| `df_reg_v2.csv` | 1.7 MB | 2,328 × 45 |
| `features.csv` | 565 KB | 2,328 × 16 (파생피처 15 + Date) |
| `granger_results.csv` | 2 KB | 43 × 5 (변수, p-value, lag 등) |

### Tier 3: Step 3~6 산출

| 파일 | 스키마 |
|------|------|
| `profiles.csv` | profile, gamma, target_vol, max_equity, min_bond, max_mdd, l1_equity, l1_bond, l1_alt |
| `optimal_weights.csv` | profile, gamma, strategy, ticker, weight (360 rows) |
| `regime_history.csv` | Date, hmm_regime, VIX_level, VIX_contango, HY_spread, yield_curve, Cu_Au_ratio_chg, rv_neutral |
| `alert_signals.csv` | Date, alert_a, alert_b, alert_c, alert_d, stress_score, VIX_level, ... |

### Tier 4: v4.1 산출

| 파일 | 크기 | 구조 |
|------|------|------|
| `regime_covariance_by_window.pkl` | 413 KB | dict[31] of {mu, Σ_stable, Σ_crisis, fallback_type, ...} |
| `regime_covariance_4group.pkl` | 816 KB | dict[31] of {mu, Σ_by_regime, fallback_info} |
| `step9_backtest_results.pkl` | 3.2 MB | dict {results: {67 strategies: Series}, ...} |
| `step9_metrics.csv` | 12 KB | 67 × 8 (strategy × {total_return, ..., cvar_99}) |
| `step10_final_recommendation.csv` | 2 KB | 60 × 14 (각 전략 Multi-criteria rank) |
| `step10_cost_mitigation_decision.pkl` | 160 B | dict with verdict |

### Tier 5: Step 11 산출

| 파일 | 크기 | 구조 |
|------|------|------|
| `step11_top10_weights.pkl` | 6.3 MB | dict {top10_keys, weights[10], daily_returns[10], rebalance_events[10], sigma_selection[10], alert_levels[10], meta} |

---

## 🔑 Primary Keys 및 Join 관계

### 기본 키 (PK)

| 엔티티 | PK |
|-------|-----|
| 시계열 데이터 | `Date` (DatetimeIndex) |
| WF 윈도우 | `window_id` (0 ~ 30) |
| 전략 | `strategy` = `f"{mode}_{profile}_{config}"` |
| 티커 | `ticker` ∈ PORT_TICKERS (30개) |

### 주요 Join 시나리오

**시나리오 1: 특정 시점 포트폴리오 상태 조회**
```python
# 2020-03-20 M1_보수형_ALERT_B의 비중
pkl = load('step11_top10_weights.pkl')
weights = pkl['weights']['M1_보수형_ALERT_B'].loc['2020-03-20']
alert = pkl['alert_levels']['M1_보수형_ALERT_B'].loc['2020-03-20']
# Result: {'SPY': 0.04, 'AGG': 0.25, ..., 'alert_b': 3}
```

**시나리오 2: 특정 윈도우의 Σ 조회**
```python
pkl = load('regime_covariance_by_window.pkl')
w0 = pkl['windows'][0]
# {mu, Σ_stable, Σ_crisis, fallback_type, n_stable, n_crisis, ...}
```

**시나리오 3: 경보-비중 상관성 분석**
```python
alerts = pd.read_csv('alert_signals.csv', ...)
step9 = pickle.load(open('step9_backtest_results.pkl', 'rb'))
strategy_ret = step9['results']['M1_보수형_ALERT_B']
merged = alerts.join(strategy_ret.rename('return'))
# → 경보 레벨별 수익률 분석 가능
```

---

## 📐 데이터 타입 명세

### Datetime 규약
- 모든 시계열: `pd.DatetimeIndex`
- NYSE 영업일 기준 (주말·공휴일 제외)
- 시간대: UTC-4/5 (EST/EDT, 미국 동부)

### 수치 규약
- **가격 (Price)**: `float64`, 달러 단위
- **수익률 (Return)**: `float64`, 소수점 (예: 0.01 = 1%)
- **비중 (Weight)**: `float64`, [0, 1] 범위
- **경보 레벨**: `int64`, {0, 1, 2, 3}
- **레짐 라벨**: `int64`, {0, 1, 2, 3}

### 결측 처리
- 원본 가격: forward-fill → back-fill
- 파생 피처: Step 2에서 결측 제거 (2609 → 2328)
- Granger 결과: 일부 변수 N/A 가능 (lag 미수렴)

---

## 🗂️ 폴더 구조 전체

```
Guide/
├── data/                       ← 모든 CSV·PKL 중앙집중
├── images/                     ← 28+ PNG 시각화
├── docs/                       ← Step1~11 해설 MD
├── quick_reference/            ← 13종 빠른 참조
├── interactive/                ← HTML + Streamlit 앱
│   ├── dashboard.html
│   └── streamlit_app/
│       ├── app.py
│       ├── pages/
│       ├── utils/
│       └── requirements.txt
├── Step1~11_*.ipynb            ← 11개 노트북
├── _build_step*.py             ← 빌더 스크립트
├── report_v3.md                ← 기존 보고서
├── report_v4.md                ← v4.1 보고서
├── report_final.md             ← 통합 보고서
├── decision_log.md             ← v3 설계
├── decision_log_v31.md         ← v4.1 설계
└── stats_model.md              ← 통계 기법 요약
```

---

## 🔗 순환 의존성 및 실행 순서

**엄격한 선행 순서**:
```
Step 1  (데이터 수집)
   ↓
Step 2  (피처 공학) — df_reg_v2 생성
   ↓
┌─────────┼─────────┐
Step 3   Step 4   Step 5   (병렬 가능)
   ↓
Step 6  (HMM + alerts) — alert_signals.csv 필요 ← Step 2 + 수작업 연결
   ↓
Step 7  (v3 Ablation)
   ↓
Step 8  (Regime Σ) — regime_history.csv 필요 ← Step 6
   ↓
Step 9  (64 시뮬) — Step 8 pkl + alerts 필요
   ↓
Step 10 (통계) — Step 9 pkl 필요
   ↓
Step 11 (시각화) — Step 10 csv + Step 9 pkl 필요
```

**실행 재현 가이드**:
```bash
cd Guide/

# 기본 실행 (Step 1~11 순차)
for i in 1 2 3 4 5 6 7 8 9 10 11; do
    jupyter nbconvert --to notebook --execute "Step${i}_*.ipynb" \
            --output "Step${i}_*.ipynb" \
            --ExecutePreprocessor.timeout=1800
done
```

---

## 🧪 데이터 품질 검증

### Invariants (불변식)

| 데이터 | 검증 |
|------|------|
| `portfolio_prices.csv` | 모든 가격 > 0 |
| `df_reg_v2.csv` | VIX > 0, rv_neutral > 0 |
| `alert_signals.csv` | alert_* ∈ {0,1,2,3} |
| `weights_daily` | sum(row) = 1 ± 1e-6, 모든 값 >= 0 |
| `step9_results` | daily_returns Series 길이 일관 |

### 교차 검증 지점

1. **Step 9 vs Step 11**: 동일 전략의 누적수익률이 0.5% 이내 일치
2. **Step 10 recommendation**: Step 9 Sharpe와 일치
3. **Fallback 분포**: 31 윈도우 total = separate + scaled + scaled_reverse + single

---

## 📚 관련 문서

- 구조 개요: `04_pipeline_flowchart.md`
- 실행 가이드: `13_operating_checklist.md`
- 기술 상세: `docs/Step1~11_해설.md`
