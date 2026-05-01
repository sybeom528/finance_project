# Phase 3 최종 결론 보고서

> 생성: 2026-04-29  

> 비교 기간: 2010-01-29 ~ 2025-12-31 (192 개월)


## 1. 예측 정확도 (RMSE)

| 모델 | RMSE 평균 | 비고 |
|---|---|---|
| 종목별 (02a) | 0.3846 | Phase 1.5 v8 아키텍처 |
| Cross-Sec (02b) | 0.4088 | Ticker Embedding (GKX 2020) |

## 2. 포트폴리오 성과 (Fair 비교)

| 시나리오 | Sharpe | Annual Ret | MDD |
|---|---|---|---|
| BL_ml_cs | 1.094 | 12.3% | -17.5% |
| McapWeight | 1.044 | 15.0% | -24.4% |
| SPY | 1.013 | 14.5% | -23.9% |
| EqualWeight | 0.886 | 13.9% | -26.8% |

## 3. 학술 결론

### Phase 3 vs 서윤범 BL TOP_50 (Sharpe 1.065)

- [예측 정확도] 종목별 RMSE (0.3846) ≤ CS RMSE (0.4088): 종목별 방식 동등 또는 우위
- [벤치마크] McapWeight Sharpe: 1.044
-   BL_ml_cs vs McapWeight: +0.050