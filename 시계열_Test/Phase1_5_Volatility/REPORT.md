# Phase 1.5 변동성 예측 분기 — 종합 진행 보고서

> **작성일**: 2026-04-27
> **작성자**: 재천
> **소속 프로젝트**: COL-BL (Su et al. 2026 ESWA 295 — Constrained Online Learning Black-Litterman 재현·확장)
> **본 단계 위치**: 시계열_Test → Phase1 LSTM 종료 → **Phase 1.5 변동성 분기 신설** (Phase 2 진입 보류)

---

## Executive Summary (한 페이지 요약)

### 단일 질문
> **"변동성 예측이 가능한가?"**

### 한 줄 답변
> **YES — 변동성 예측은 가능. 단 LSTM 으로는 부적합. HAR-RV (Corsi 2009 선형 OLS) 가 학술·실증 표준.**

### 핵심 결과 (90 fold × 2 ticker × 6 모델)

| 순위 | 모델 | SPY RMSE | QQQ RMSE | 비고 |
|---|---|---|---|---|
| **1** | **HAR-RV** | **0.3646** | **0.3308** | best, Corsi 2009 학술 표준 |
| 2 | EWMA | 0.3942 | 0.3582 | RiskMetrics |
| 3 | Naive | 0.4109 | 0.3699 | 직전 RV broadcast |
| 4 | Train-Mean | 0.4320 | 0.4067 | trivial baseline |
| **5** | **LSTM v1** | **0.4688** | **0.4329** | log_ret² 1ch |
| **6** | **LSTM v2** | **0.4798** | **0.4385** | HAR 3ch |

> **충격적 사실**: LSTM 두 모델 모두 가장 단순한 Train-Mean (4위) 보다도 RMSE 가 큼 — "학습이 trivial baseline 보다 못함" 의 정량 증명.

### Phase 1.5 PASS 조건 종합

| 모델 | 관문 1 (RMSE < HAR) | 관문 2 (r2_train_mean > 0) | 관문 3 (pred_std_ratio > 0.5) | 종합 |
|---|---|---|---|---|
| v1 SPY | ❌ FAIL | ❌ FAIL | ❌ FAIL | **0/3** |
| v1 QQQ | ❌ FAIL | ❌ FAIL | ❌ FAIL | **0/3** |
| v2 SPY | ❌ FAIL | ❌ FAIL | ✅ PASS | **1/3** |
| v2 QQQ | ❌ FAIL | ❌ FAIL | ✅ PASS | **1/3** |

→ **모든 LSTM 모델·종목에서 Phase 1.5 PASS 조건 미충족**

### 추후 활용 권고
- 본 결과의 **HAR-RV 예측치 (RMSE 0.33~0.36)** 를 추후 단계의 변동성 입력으로 활용
- 추후 단계: Black-Litterman 모델의 Ω (불확실성 행렬) 입력, Volatility Targeting 등

---

## 1. 프로젝트 배경 및 본 분기 신설 동기

### 1.1 Phase 1 LSTM 마감 결과 (분기 직전 시점)

Phase 1 LSTM 베이스라인이 5회 Run 을 거쳐 다음 결과로 마감:

| Run | 변경 사항 | SPY hit_rate | SPY R²_OOS | 결과 |
|---|---|---|---|---|
| 1차 | 기본 (log_ret 1ch) | 0.5953 | -0.5821 | baseline |
| 2차 | weight_decay 1e-4 | 0.5874 | -0.4123 | 미세 개선 |
| **3차 (best)** | **weight_decay 1e-3** | **0.6313** ✅ | **-0.2118** ❌ | **개선** |
| 4차 | Y_trailing 다변량 (4ch) | 0.5953 | **-2.15** | **대폭 악화** |
| 5차 | VIX 추가 (2ch) | 0.5874 | **-1.13** | **대폭 악화** |

**진단 (윤서)**: 134 훈련 샘플/fold 의 절대 부족 → 다변량 입력은 모두 역효과.

**한편** 김하연님의 §10 ARCH-LM 검정에서 **변동성에는 강한 자기상관** (SPY LM=754, p≈0) 정량적 입증:
- 동일 시계열에서 변동성(ACF max 0.30) 이 수익률 방향(ACF max 0.13) 보다 **2.3배 풍부한 신호**

### 1.2 Phase 1.5 분기 결정 (사용자 확정)

> **Phase 2 (GRU) 로 진행하지 않고 Phase 1.5 분기 신설**:
> - Phase 1 결과는 그대로 **보존** (변경 금지)
> - 예측 대상을 **누적수익률 → 실현변동성** 으로 교체한 새 LSTM 학습·평가
> - 본 단계 **유일한 목표**: "변동성 예측이 가능한가?" 단일 질문에 명확한 답
> - 포트폴리오 구축·BL 통합·벤치마크 비교는 **본 단계 평가 대상 아님** (추후 별도 단계)

### 1.3 Plan 수립 흐름 (5차 plan 까지 사용자 대화)

| 차수 | 사용자 피드백 | 반영 |
|---|---|---|
| 1차 plan | (작성 후 사용자 검토) | 각 결정 사항 직관적 설명 부족 지적 |
| 2차 plan | 타깃·입력·평가지표·베이스라인·관문·loss·옵티마 등 각 항목 직관 설명 추가 | 사용자 검토 |
| 3차 plan | "변동성 예측 → 포트폴리오 구축까지 어떻게 이어지는지 설명 필요" | 효율적 프론티어·BL 모델 설명 추가 |
| 4차 plan | "변동성만으로 벤치마크 이기는 게 불가능한가? 학술 근거 필요" | Moreira & Muir (2017), Harvey et al. (2018) 등 4편 인용 추가 |
| **5차 plan (확정)** | "포트폴리오 가능성은 배경으로, Phase 1.5 는 변동성 예측에만 집중" | 포트폴리오 섹션 → Appendix 이동, Context 의 "핵심 목표" 단일 질문 명시 |

**최종 plan 위치**: `C:\Users\gorhk\.claude\plans\sharded-mapping-puffin.md` (진실원), `PLAN.md` (팀 사본)

---

## 2. 핵심 의사결정 요약

| 항목 | 결정 | 근거 |
|---|---|---|
| 폴더 명 | `Phase1_5_Volatility` | "분기(.5)" 의미 + 타깃 차이 폴더명 노출 |
| 타깃 | **Log-RV 21일 forward** (`log( std(log_ret[t+1:t+22], ddof=1) )`) | log 변환으로 정규화·양수 보장 + Corsi (2009) 와 도메인 일치 |
| 입력 v1 | `log_ret²` (1채널) | 타깃과 dimensional match (variance proxy) |
| 입력 v2 | HAR 3채널 `[\|log_ret\|, RV_w, RV_m]` | Corsi 2009 학술 표준, 변수 부족 가설 검증 |
| Walk-Forward | IS=504 / Purge=21 / Embargo=63 / OOS=21 / Step=21 | EDA §5-확장 결과 기반 정밀도 우선 (90 fold, 훈련 샘플 441/fold) |
| 모델 | LSTM hidden=32, layers=1, dropout=0.3, params=4,513~4,769 | Phase 1 3차 Run 동일 capacity |
| 손실 함수 | **MSE** (Phase 1 의 Huber 에서 변경) | log-RV 가 거의 정규분포 → MSE 의 가우시안 가정 충족 |
| 옵티마이저 | AdamW (lr=1e-3, wd=1e-3) | LSTM 학술 표준 |
| 평가 지표 | rmse / qlike / r2_train_mean / pred_std_ratio / mz_regression | hit_rate·r2_oos 폐기 (변동성 비현실적) |

---

## 3. 방법론

### 3.1 데이터

- **자산**: SPY (S&P 500 ETF), QQQ (Nasdaq-100 ETF)
- **기간**: 2016-01-01 ~ 2025-12-31 (10년, 2,514 영업일)
- **출처**: Phase 1 raw_data 사본 재사용 (재다운로드 X)
- **유효 타깃 수**: 2,493 영업일 (마지막 21일 NaN — forward 21일 산출 불가)

### 3.2 Walk-Forward Cross-Validation (IS 504, Embargo 63)

```
[← IS 504일 →][purge 21][embargo 63][← OOS 21일 →]
 train                                test
```

**구성 근거** (EDA §5-확장 결과):

| lag | SPY ACF | QQQ ACF | 95% CI 밖? |
|---|---|---|---|
| 21 | 0.5566 | 0.5848 | ✓ (강한 잔존) |
| 42 | 0.3695 | 0.3667 | ✓ |
| **63** | **0.2075** | **0.1921** | **✓ (embargo 차단점)** |
| 126 | 0.1292 | 0.1866 | ✓ |
| 252 | -0.1188 | -0.1448 | ✓ (1년 mean-reversion) |
| 95% CI 진입 첫 lag | 222 | 208 | — |

→ **embargo 63** 으로 lag 63 의 ACF 0.21 까지 차단 (이론 권고 lag 200+ 는 fold 수 급감으로 비현실적)

**fold 수**: 90 (이론 105 대비 -15, 통계 권고 30+ 충분 만족)

### 3.3 모델 구조

```python
LSTMRegressor(
    input_size=1,        # v1 (log_ret²)
    # input_size=3,       # v2 (HAR 3ch)
    hidden_size=32,      # Phase 1 동일
    num_layers=1,
    dropout=0.3,
    batch_first=True,
)
# 학습 파라미터: v1 4,513개 / v2 4,769개
```

### 3.4 평가 지표 (Phase 1.5 신규)

| 지표 | 정의 | 방향 | 의미 |
|---|---|---|---|
| **rmse** | √(mean((y_pred − y_true)²)) | ↓ | 평균 예측 오차 |
| **qlike** | mean(σ²_t/σ²_p − log(σ²_t/σ²_p) − 1) | ↓ | Patton 2011 비대칭 손실 (과소예측 더 처벌) |
| **r2_train_mean** | 1 − SSE_model/SSE_train_mean | >0 | trivial baseline 능가 여부 |
| **pred_std_ratio** | std(y_pred)/std(y_true) | ~1.0 | mean-collapse 진단 |
| **mz_regression** | y_true = α + β·y_pred + ε | α=0, β=1 | 편향 진단 (Mincer-Zarnowitz) |

**폐기 지표 (Phase 1 → Phase 1.5)**:
- Hit Rate: 변동성 항상 양수 → trivially 1.0
- R²_OOS (zero baseline): 변동성=0 비현실 + 분모 부풀림

### 3.5 PASS 조건 (3 관문 모두 충족 시 PASS)

1. **관문 1**: LSTM RMSE < HAR-RV RMSE (90 fold 평균)
2. **관문 2**: r2_train_mean > 0
3. **관문 3**: pred_std_ratio > 0.5

---

## 4. 작업 흐름 — Step별 진행

### 4.1 Step 0 (2026-04-26 오전) — 폴더·scripts 격리 보존

```
Phase1_5_Volatility/
├── README.md, PLAN.md, 재천_WORKLOG.md
├── 00_setup_and_utils.ipynb              (Phase 1 사본)
├── scripts/
│   ├── setup.py / dataset.py / models.py / train.py    (Phase 1 사본)
│   ├── targets_volatility.py             (신규 — Log-RV 빌더)
│   ├── metrics_volatility.py             (신규 — 5종 메트릭)
│   └── baselines_volatility.py           (신규 — HAR/EWMA/Naive/TM)
└── results/
    └── raw_data/  (Phase 1 SPY/QQQ.csv 사본)
```

**모듈 단위 테스트 17건 PASS** (`_test_modules.py`):
- targets 4건 (누수 검증, ddof, NaN 처리, log domain)
- metrics 8건 (rmse, qlike, r2_train_mean, mz, baseline_metrics, summarize, edge cases)
- baselines 4건 (HAR 계수, EWMA recursion, Naive shift, fold 격리)
- train.py loss_type 분기 1건

### 4.2 Step 1 (2026-04-26 오전) — `01_volatility_eda.ipynb` (20셀)

**목적**: RV 분포·ACF·정상성·체제 진단

**핵심 결과**:
- Log-RV 분포 정규성 양호 (Shapiro-Wilk p > 0.05 일부 fold)
- **§5 ACF 확장 (lag 1~252)**: long-memory 시그니처 명확 (lag 220+ 까지 잔존)
- 체제 변화 명확: COVID 2020-03 폭락, 2022 긴축 폭등 → fold-level outlier 예고
- **embargo 권고 갱신**: 21 → 63 (lag 63 ACF 0.21 차단)

**사용자 결정 (2026-04-26)**: "정밀도 우선 — IS 504, embargo 63, 1ch 로 진행"

### 4.3 Step 2 (2026-04-26 오후) — 모듈 작성 + 단위 테스트

| 모듈 | 라인 수 | 핵심 함수 |
|---|---|---|
| `targets_volatility.py` | 145 | `build_daily_target_logrv_21d`, `verify_no_leakage_logrv` |
| `metrics_volatility.py` | 280 | rmse, qlike, r2_train_mean, mz_regression, pred_std_ratio, baseline_metrics_volatility, summarize_folds_volatility |
| `baselines_volatility.py` | 230 | fit_har_rv, predict_ewma, predict_naive, predict_train_mean |
| `train.py` (수정) | +10 | loss_type='mse' 분기 추가 |

**검증 (`_test_modules.py`, 17건 PASS / 0 FAIL)**

### 4.4 Step 3 — v1 학습 (2026-04-27 오전, `02_volatility_lstm.ipynb`, 33셀)

**입력**: log_ret² (1ch)
**학습**: SPY 0.8m + QQQ 0.8m = 1.6분
**산출**: `results/volatility_lstm/{SPY,QQQ}_metrics.json` (각 ~2.5MB)

**v1 결과 (90 fold mean)**:

| metric | SPY | QQQ | 관문 |
|---|---|---|---|
| rmse | 0.4688 | 0.4329 | (§03 비교) |
| **r2_train_mean** | **-2.93** | **-2.26** | ❌ FAIL |
| **pred_std_ratio** | **0.348** | **0.353** | ❌ FAIL |
| mz_beta | -0.68 | -0.31 | (편향 진단) |
| best_epoch | 9.69 | 9.09 | ✅ 학습 자체 수렴 |

**진단**: "학습 코드 정상, 모델은 mean-collapse + underfit, fold 마다 결과 매우 다름."

### 4.5 v2 분기 결정 (2026-04-27 오후, `02_v2_volatility_lstm_har3ch.ipynb`)

**검증 가설**: "log_ret² 1ch 의 정보 빈약이 mean-collapse 의 주범인가?" (가설 a)

**v1 ⇄ v2 단 1가지 변경** (통제 실험):
- 입력 채널: log_ret² (1ch) → **HAR 3채널 [|log_ret|, RV_w, RV_m]** (Corsi 2009 표준)
- input_size: 1 → 3
- 결과 폴더: `volatility_lstm/` → `volatility_lstm_har3ch/`
- 그 외 모두 v1 동일 (Walk-Forward / 모델 / 손실 / 옵티마 / seed)

**v2 결과 (90 fold mean)**:

| metric | v1 SPY | **v2 SPY** | Δ | v1 QQQ | **v2 QQQ** | Δ |
|---|---|---|---|---|---|---|
| rmse | 0.4688 | 0.4798 | +0.011 | 0.4329 | 0.4385 | +0.006 |
| **r2_train_mean** | -2.93 | **-4.03** | -1.10 ❌ | -2.26 | **-2.81** | -0.55 ❌ |
| **pred_std_ratio** | 0.348 | **0.634** | **+0.286** ✅ | 0.353 | **0.539** | **+0.187** ✅ |
| **mz_beta** | -0.68 | -2.64 | -1.96 | -0.31 | -2.52 | -2.21 |
| **mz_r2** | 0.125 | 0.274 | +0.149 | 0.115 | 0.260 | +0.146 |

**가설 a 부분 검증**:
- ✅ **mean-collapse 가 변수 부족과 부분 관계**: pred_std_ratio +82%/+53% 회복 → 관문 3 PASS
- ❌ **그러나 정밀도 회복 X (오히려 악화)**: mz_beta 더 음수 (-0.68 → -2.64) — "활발하게 틀린다"

**가설 가중치 갱신** (v2 결과 반영):

| 가설 | 이전 | **갱신** | 이유 |
|---|---|---|---|
| (a) 변수 부족 | 50~60% | **20~30%** | mean-collapse 만 부분 회복 |
| (b) Long-memory 잔존 | 15~20% | **30~40%** | mz_beta 음수 강화 |
| (c) IS 504 다체제 혼합 | 15~20% | **20~30%** | 24개월 IS 가 음의 상관 학습 |
| (d) LSTM 자체 한계 | 5~10% | **10~15%** | HAR 비교 후 결정 (Step 4) |

### 4.6 Step 4 (2026-04-27 저녁, `03_baselines_and_compare.ipynb`, 25셀)

**사용자 결정**: 옵션 A (§03 베이스라인 비교 우선) 선택

**모듈 검토 → 빌드 → 실행**:
1. `baselines_volatility.py`, `metrics_volatility.py` 재검토 + 단위테스트 17건 PASS 재확인
2. `_build_03_compare_nb.py` 작성 (700+ 라인)
3. `03_baselines_and_compare.ipynb` 빌드 (25셀)
4. 1차 실행: §2 assert 임계값 1e-10 가 LSTM float32 정밀도 (~7.3e-08) 초과 → FAIL
5. 수정: assert 1e-10 → 1e-5, np.allclose atol=1e-5 명시
6. 재실행: exit code 0, 모든 셀 정상

**§03 산출**:
- `results/comparison_report.md` (정리 후 8 KB)
- 6 모델 × 5 메트릭 통합 비교 표 (SPY/QQQ 각각)
- 시각화 3종: RMSE 박스플롯, LSTM vs HAR 산점도, 메트릭별 막대그래프

---

## 5. §03 결과 — 전 모델 통합 비교

### 5.1 SPY (90 fold mean ± std)

| 모델 | rmse | qlike | r2_train_mean | pred_std_ratio | mae |
|---|---|---|---|---|---|
| lstm_v1 | 0.4688 ± 0.299 | 0.9270 ± 2.99 | -3.28 ± 18.3 | 0.348 ± 1.37 | 0.4351 ± 0.285 |
| lstm_v2 | 0.4798 ± 0.500 | 0.9210 ± 3.10 | -3.78 ± 21.4 | 0.634 ± 1.53 | 0.4444 ± 0.482 |
| **har** | **0.3646 ± 0.244** | **0.7796 ± 2.84** | **-0.53 ± 2.35** | 0.897 ± 0.66 | **0.3309 ± 0.240** |
| ewma | 0.3942 ± 0.257 | 0.7122 ± 2.53 | -1.85 ± 5.69 | 0.916 ± 0.72 | 0.3597 ± 0.253 |
| naive | 0.4109 ± 0.255 | 0.7525 ± 1.93 | -2.26 ± 7.61 | 1.270 ± 0.97 | 0.3698 ± 0.250 |
| train_mean | 0.4320 ± 0.312 | 1.3578 ± 4.85 | 0.000 ± 0.00 | 0.000 ± 0.00 | 0.4071 ± 0.313 |

### 5.2 QQQ (90 fold mean ± std)

| 모델 | rmse | qlike | r2_train_mean | pred_std_ratio | mae |
|---|---|---|---|---|---|
| lstm_v1 | 0.4329 ± 0.273 | 0.6726 ± 1.79 | -2.73 ± 10.7 | 0.353 ± 1.13 | 0.4026 ± 0.258 |
| lstm_v2 | 0.4385 ± 0.409 | 0.6899 ± 2.16 | -3.65 ± 26.0 | 0.539 ± 0.90 | 0.4112 ± 0.407 |
| **har** | **0.3308 ± 0.209** | 0.5083 ± 1.53 | **-0.26 ± 1.59** | 0.919 ± 0.61 | **0.2972 ± 0.205** |
| ewma | 0.3582 ± 0.235 | **0.5037 ± 1.54** | -1.57 ± 4.93 | 0.917 ± 0.65 | 0.3255 ± 0.231 |
| naive | 0.3699 ± 0.226 | 0.5220 ± 1.25 | -2.11 ± 7.34 | 1.283 ± 0.93 | 0.3313 ± 0.223 |
| train_mean | 0.4067 ± 0.270 | 0.8707 ± 2.39 | 0.000 ± 0.00 | 0.000 ± 0.00 | 0.3847 ± 0.273 |

### 5.3 RMSE 순위 (모든 모델 비교)

```
SPY:  HAR(0.36) < EWMA(0.39) < Naive(0.41) < TrainMean(0.43) < LSTMv1(0.47) < LSTMv2(0.48)
QQQ:  HAR(0.33) < EWMA(0.36) < Naive(0.37) < TrainMean(0.41) < LSTMv1(0.43) < LSTMv2(0.44)
```

**LSTM 의 Train-Mean 대비 격차**:
- LSTM v1: SPY +8.5%, QQQ +6.4%
- LSTM v2: SPY +11.1%, QQQ +7.8%

**LSTM 의 HAR 대비 격차**:
- v1: SPY +28.6%, QQQ +30.9%
- v2: SPY +31.6%, QQQ +32.6%

### 5.4 관문 1 판정 (LSTM RMSE < HAR-RV RMSE)

**4건 모두 FAIL**

| ticker | LSTM v1 RMSE | LSTM v2 RMSE | HAR-RV RMSE | 관문 1 (v1) | 관문 1 (v2) |
|---|---|---|---|---|---|
| SPY | 0.4688 | 0.4798 | 0.3646 | ❌ FAIL | ❌ FAIL |
| QQQ | 0.4329 | 0.4385 | 0.3308 | ❌ FAIL | ❌ FAIL |

---

## 6. Phase 1.5 최종 PASS/FAIL 종합

| 모델 | 관문 1 (RMSE<HAR) | 관문 2 (r2_tm>0) | 관문 3 (psr>0.5) | 종합 |
|---|---|---|---|---|
| v1 SPY | ❌ FAIL | ❌ FAIL | ❌ FAIL | **0/3** |
| v1 QQQ | ❌ FAIL | ❌ FAIL | ❌ FAIL | **0/3** |
| v2 SPY | ❌ FAIL | ❌ FAIL | ✅ PASS | **1/3** |
| v2 QQQ | ❌ FAIL | ❌ FAIL | ✅ PASS | **1/3** |

→ **모든 LSTM 모델·종목에서 PASS 조건 미충족**

---

## 7. Phase 1.5 결론

### 7.1 단일 질문에 대한 답변

> **"변동성 예측이 가능한가?" → YES (HAR-RV 로 가능). 단 LSTM 으로는 부적합.**

### 7.2 명확히 입증된 사실

1. **HAR-RV (4 계수 선형) 가 LSTM (4,500+ 파라미터 비선형) 능가**
   - 변동성 예측에는 단순 선형 모델이 최적 — Corsi (2009) 학술 결과 재확인

2. **LSTM 이 trivial baseline (Train-Mean) 보다도 못함**
   - r2_train_mean 음수의 정량 증명
   - 모든 LSTM 변종 (v1, v2) 에서 동일 패턴

3. **변동성의 강한 자기상관 (lag 1 ACF ≈ 0.99)** 가 단순 모델에 충분 정보 제공
   - LSTM 의 비선형 capacity 가 이 단순 패턴에 noise 첨가

### 7.3 LSTM 부적합 원인 진단 (사후)

| # | 원인 | 근거 |
|---|---|---|
| 1 | Long-memory ACF 잔존 | EDA §5-확장: lag 220+ 까지 잔존, embargo 63 차단 부분 |
| 2 | 변동성 예측의 선형성 | HAR (선형) > LSTM (비선형) → 비선형 capacity 가 noise |
| 3 | 체제 변화 (COVID, 긴축) | LSTM sequential 학습이 외생 충격에 부정적 |
| 4 | 일부 fold 학습 폭주 | val_loss 5~7 outlier (SPY fold 30, 90; QQQ fold 30) |
| 5 | 일별 데이터 한계 | Corsi 의 intraday 환경 (R² 0.55~0.65) 대비 본 환경 (R²_train 0.07~0.08) |

### 7.4 Phase 1.5 의 가치 — Negative Result 가 아닌 명확한 Positive Insight

**"LSTM 으로 변동성 예측 실패" 가 아니라, "변동성 예측의 표준 도구 식별"**:

| 추후 단계에서 활용 | 본 단계 결과 |
|---|---|
| BL 모델의 Ω (불확실성 행렬) 입력 | **HAR-RV 예측치 사용** (RMSE 0.33~0.36) |
| Volatility Targeting | HAR-RV 우선, EWMA 보조 |
| Risk Parity | HAR-RV 자산 간 비율 |

→ **"LSTM 변동성 예측 재시도" 우선순위 ↓** (강력한 학술 + 실증 근거)

---

## 8. 다음 단계 권고 (사용자 결정 사항)

| 옵션 | 의미 | 권고도 |
|---|---|---|
| **A** | **Phase 1.5 종료 + 추후 단계 진입** (HAR-RV 입력 → BL 모델 등) | ⭐ **권고 (높음)** |
| B | LSTM 추가 실험 (IS 단축/확장, embargo 확대 등) | 중간 (학술 가치) |
| C | GARCH 등 통계 모델 추가 baseline 비교 | 낮음 (Phase 1.5 범위 외) |

### 옵션 B 검토 (참고) — IS 변경의 두 방향 비교

| 차원 | 단축 (504→252) | 현재 (504) | 확장 (504→750) |
|---|---|---|---|
| fold 수 | 102 ↑ | 90 (base) | 79 ↓ |
| 훈련 샘플/fold | 189 (Phase 1 134 와 유사) | 441 | 687 (LSTM 권고 충족) |
| 검증 가설 | (c) 다체제 혼합 | base | (a) 추가 검증 (변수+샘플) |
| 위험 | 샘플 부족 재발 | base | 다체제 혼합 더 심함 |

→ **HAR vs LSTM 의 +30% 격차** 는 어떤 IS 변경으로도 좁히기 어려울 가능성 높음.

---

## 9. 산출물 (Phase 1.5 마감 시점)

### 9.1 노트북 (4개, 총 111셀)

| 파일 | 셀 수 | 크기 | 설명 |
|---|---|---|---|
| `00_setup_and_utils.ipynb` | — | 14 KB | 환경·유틸 (Phase 1 사본) |
| `01_volatility_eda.ipynb` | 20 | 952 KB | RV 분포·ACF·정상성·체제 진단 |
| `02_volatility_lstm.ipynb` | 33 | 603 KB | v1 (log_ret² 1ch) — 보존 |
| `02_v2_volatility_lstm_har3ch.ipynb` | 33 | 562 KB | v2 (HAR 3ch) — 신규 |
| `03_baselines_and_compare.ipynb` | 25 | 302 KB | §03 베이스라인 비교 — 신규 |

### 9.2 빌드 스크립트 (5개)
- `_build_01_eda_nb.py`
- `_build_02_lstm_nb.py`
- `_build_02_v2_har_nb.py`
- `_build_03_compare_nb.py`
- `_test_modules.py` (단위 테스트, 17건 PASS)

### 9.3 모듈 (`scripts/`)

| 파일 | 라인 | 설명 |
|---|---|---|
| `setup.py`, `dataset.py`, `models.py` | — | Phase 1 사본 (변경 없음) |
| `train.py` | +10 (수정) | loss_type='mse' 분기 추가 |
| `targets_volatility.py` | 145 | Log-RV 빌더 + 누수 검증 (신규) |
| `metrics_volatility.py` | 280 | 5종 메트릭 + summarize (신규) |
| `baselines_volatility.py` | 230 | HAR/EWMA/Naive/TM (신규) |

### 9.4 결과

```
results/
├── raw_data/  (Phase 1 SPY/QQQ.csv 사본)
├── volatility_lstm/                       (v1 결과)
│   ├── SPY_metrics.json (2,546 KB)
│   └── QQQ_metrics.json (2,545 KB)
├── volatility_lstm_har3ch/                (v2 결과)
│   ├── SPY_metrics.json (2,555 KB)
│   └── QQQ_metrics.json (2,551 KB)
└── comparison_report.md                    (§03 종합 보고서, 8 KB)
```

### 9.5 문서

| 파일 | 라인 | 설명 |
|---|---|---|
| `README.md` | — | 프로젝트 개요 |
| `PLAN.md` | 700+ | 5차 plan 최종본 (팀 사본) |
| `재천_WORKLOG.md` | 720+ | 시간순 작업·결정·근거 일지 |
| **`REPORT.md`** | (본 보고서) | **종합 진행 보고서** |

---

## 10. 부록 — 학술 인용

### A. 본 단계 직접 활용 인용

1. **Corsi, F. (2009)**. A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
   → HAR-RV 모델 원본 (본 단계 1위 모델)

2. **Patton, A. J. (2011)**. Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246-256.
   → QLIKE 손실 함수 (본 단계 2차 메트릭)

3. **Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003)**. Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579-625.
   → Realized Volatility 추정 표준

4. **Müller, U. A., et al. (1997)**. Volatilities of different time resolutions. *Journal of Empirical Finance*, 4(2-3), 213-239.
   → Heterogeneous Market Hypothesis (HAR 의 이론적 근거)

5. **JP Morgan & Reuters (1996)**. RiskMetrics — Technical Document (4th ed.).
   → EWMA λ=0.94 표준

### B. 추후 단계 잠재 활용 인용 (배경)

6. **Moreira, A., & Muir, T. (2017)**. Volatility-Managed Portfolios. *Journal of Finance*, 72(4), 1611-1644.
   → 변동성 단독 vs 시장 Sharpe 0.45 → 0.62

7. **Harvey, C. R., et al. (2018)**. The Impact of Volatility Targeting. *Journal of Portfolio Management*, 45(1), 14-33.

8. **DeMiguel, V., Garlappi, L., & Uppal, R. (2009)**. Optimal Versus Naive Diversification. *Review of Financial Studies*, 22(5), 1915-1953.
   → 1/N 능가의 어려움 (배경 인용)

9. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley.
   → Walk-Forward + Purge + Embargo 표준

---

## 11. 작성자 메모

본 단계는 **단일 질문에 명확한 답** 을 도출했다는 점에서 의의가 큽니다. "LSTM 변동성 예측 실패" 는 부정적 결과로 보일 수 있으나:

1. **학술 정설 (Corsi 2009 의 HAR 우위) 을 본 환경에서 재확인** 함
2. **추후 단계의 변동성 입력으로 HAR-RV 사용** 의 정당화 근거 확보
3. **Phase 1 의 134 샘플 부족 가설** 을 v2 (HAR 3ch + 441 샘플) 로 부분 검증 — 변수가 늘어도 정밀도 회복 X (다른 가설의 영향 ↑)

다음 단계 (Phase 2 또는 BL 통합) 진입 시 본 보고서를 입력 자료로 활용하시면 됩니다.

---

*본 보고서 종료. 사용자 결정 사항 (옵션 A/B/C) 대기.*
