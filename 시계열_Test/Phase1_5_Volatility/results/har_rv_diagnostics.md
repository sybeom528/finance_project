# Phase 1.5 §04 — HAR-RV 자체 진단 보고서

> 90 fold × SPY/QQQ × 6 진단 항목
> 분석 기간: 2016-01-01 ~ 2025-12-31

## 종합 진단 표

| 진단 항목 | SPY | QQQ |
|---|---|---|
| §2_계수안정성 | PASS | PASS |
| §3_잔차정규성 | FAIL | FAIL |
| §3_잔차자기상관 | CAUTION | CAUTION |
| §4_MZ_unbiased | FAIL | FAIL |
| §5_DM_HAR우위 | PASS | PASS |
| §6_체제robust | CAUTION | CAUTION |
| §7_longmemory모방 | FAIL | FAIL |

## 핵심 수치

### SPY

| 항목 | 값 | 해석 |
|---|---|---|
| β_sum 평균 | +0.5241 | 학술 권고 0.7~1.0 vs 본 환경 약 0.2 |
| β_sum > 0 비율 | 100.0% | fold 안정성 |
| 잔차 mean | +0.0638 | 0 근방이면 무편향 |
| 잔차 std | 0.4336 | RMSE 와 비슷 |
| Jarque-Bera p | 1.1867e-97 | < 0.05 면 비정규 |
| Durbin-Watson | 0.0696 | 2 근방 정상 |
| MZ Wald p | 1.4449e-12 | < 0.05 면 편향 |
| DM HAR>EWMA | DM=-5.02, p=5.051e-07 | DM<0 면 HAR 우위 |
| DM HAR>LSTM v1 | DM=-8.61, p=0.000e+00 | |

### QQQ

| 항목 | 값 | 해석 |
|---|---|---|
| β_sum 평균 | +0.5486 | 학술 권고 0.7~1.0 vs 본 환경 약 0.2 |
| β_sum > 0 비율 | 100.0% | fold 안정성 |
| 잔차 mean | +0.0495 | 0 근방이면 무편향 |
| 잔차 std | 0.3875 | RMSE 와 비슷 |
| Jarque-Bera p | 1.0199e-57 | < 0.05 면 비정규 |
| Durbin-Watson | 0.0732 | 2 근방 정상 |
| MZ Wald p | 1.1609e-08 | < 0.05 면 편향 |
| DM HAR>EWMA | DM=-6.22, p=5.037e-10 | DM<0 면 HAR 우위 |
| DM HAR>LSTM v1 | DM=-9.73, p=0.000e+00 | |

## 결론

전체 PASS: 4 / 14

상세 분석은 `04_har_rv_evaluation.ipynb` 의 §2~§7 시각화·표 참조.