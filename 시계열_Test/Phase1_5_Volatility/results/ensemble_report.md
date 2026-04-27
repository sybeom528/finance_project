# Phase 1.5 v8 — Ensemble Evaluation 보고서

> v4 best LSTM + HAR-RV 의 4 ensemble 변형 비교
> 학습 X — 기존 fold_predictions 의 가중 평균만 계산

## 1. 전 종목 평균 메트릭

| 모델 | avg RMSE | avg QLIKE | avg PSR |
|---|---|---|---|
| lstm_v4 | 0.2988 | 0.2792 | 0.4932 |
| har | 0.3023 | 0.2649 | 0.8261 |
| simple | 0.2944 | 0.2594 | 0.5948 |
| ivw | 0.2944 | 0.2591 | 0.5969 |
| performance | 0.2934 ⭐ | 0.2582 ⭐ | 0.5899 |
| asset_specific | 0.2944 | 0.2594 | 0.5933 |

**RMSE best: performance** | **QLIKE best: performance**

## 2. 종목 × 모델 RMSE 표

| ticker | lstm_v4 | har | simple | ivw | performance | asset_specific | best |
|---|---|---|---|---|---|---|---|
| SPY | 0.3208 | 0.3239 | 0.3172 | 0.3172 | 0.3148 | 0.3172 | **performance** |
| QQQ | 0.2921 | 0.2920 | 0.2858 | 0.2858 | 0.2862 | 0.2858 | **simple** |
| DIA | 0.2963 | 0.3060 | 0.2905 | 0.2906 | 0.2874 | 0.2905 | **performance** |
| EEM | 0.2546 | 0.2662 | 0.2555 | 0.2554 | 0.2560 | 0.2554 | **lstm_v4** |
| XLF | 0.3088 | 0.3164 | 0.3073 | 0.3073 | 0.3060 | 0.3073 | **performance** |
| GOOGL | 0.2827 | 0.2850 | 0.2790 | 0.2790 | 0.2785 | 0.2790 | **performance** |
| WMT | 0.3364 | 0.3269 | 0.3258 | 0.3256 | 0.3246 | 0.3257 | **performance** |

## 3. DM 검정 (Ensemble vs v4 / HAR)

| ticker | variant | DM vs v4 | p_v4 | DM vs HAR | p_har |
|---|---|---|---|---|---|
| SPY | simple | -5.04✓ | 4.56e-07 | -0.53  | 5.95e-01 |
| SPY | ivw | -4.97✓ | 6.69e-07 | -0.62  | 5.32e-01 |
| SPY | performance | -5.71✓ | 1.11e-08 | -1.83  | 6.67e-02 |
| SPY | asset_specific | -5.06✓ | 4.29e-07 | -0.52  | 6.05e-01 |
| QQQ | simple | -6.26✓ | 3.74e-10 | -1.28  | 1.99e-01 |
| QQQ | ivw | -6.16✓ | 7.38e-10 | -1.39  | 1.63e-01 |
| QQQ | performance | -5.84✓ | 5.08e-09 | -0.95  | 3.41e-01 |
| QQQ | asset_specific | -6.26✓ | 3.75e-10 | -1.28  | 1.99e-01 |
| DIA | simple | -5.17✓ | 2.38e-07 | -3.96✓ | 7.40e-05 |
| DIA | ivw | -5.11✓ | 3.14e-07 | -4.04✓ | 5.42e-05 |
| DIA | performance | -5.98✓ | 2.17e-09 | -4.93✓ | 8.19e-07 |
| DIA | asset_specific | -5.22✓ | 1.77e-07 | -3.89✓ | 1.02e-04 |
| EEM | simple | -2.17✓ | 3.02e-02 | -4.71✓ | 2.50e-06 |
| EEM | ivw | -2.23✓ | 2.61e-02 | -4.66✓ | 3.22e-06 |
| EEM | performance | -1.69  | 9.20e-02 | -4.33✓ | 1.49e-05 |
| EEM | asset_specific | -2.25✓ | 2.47e-02 | -4.64✓ | 3.52e-06 |
| XLF | simple | -2.06✓ | 3.91e-02 | -4.51✓ | 6.53e-06 |
| XLF | ivw | -2.13✓ | 3.35e-02 | -4.45✓ | 8.42e-06 |
| XLF | performance | -2.61✓ | 9.10e-03 | -4.89✓ | 9.88e-07 |
| XLF | asset_specific | -2.11✓ | 3.52e-02 | -4.47✓ | 7.76e-06 |
| GOOGL | simple | -3.77✓ | 1.64e-04 | -2.71✓ | 6.65e-03 |
| GOOGL | ivw | -3.75✓ | 1.79e-04 | -2.74✓ | 6.14e-03 |
| GOOGL | performance | -3.80✓ | 1.42e-04 | -2.85✓ | 4.44e-03 |
| GOOGL | asset_specific | -3.78✓ | 1.57e-04 | -2.70✓ | 6.93e-03 |
| WMT | simple | -8.56✓ | 0.00e+00 | +1.59  | 1.13e-01 |
| WMT | ivw | -8.41✓ | 0.00e+00 | +1.43  | 1.52e-01 |
| WMT | performance | -8.93✓ | 0.00e+00 | +1.01  | 3.13e-01 |
| WMT | asset_specific | -8.51✓ | 0.00e+00 | +1.54  | 1.24e-01 |

## 4. Best 모델별 종목 카운트

| 모델 | best 종목 수 |
|---|---|
| performance | 5/7 |
| lstm_v4 | 1/7 |
| simple | 1/7 |
| har | 0/7 |
| ivw | 0/7 |
| asset_specific | 0/7 |

## 5. 결론

**Ensemble 변형 (performance) 가 단일 모델 (v4, HAR) 대비 평균 RMSE 우수.**

→ BL 통합 시 ensemble 권고
