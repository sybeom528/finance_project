# 02b Cross-Sectional (BL_ml_cs) — 종합 진단 보고서

## Layer 1 — 변동성 예측 진단

- RMSE: 0.4117
- QLIKE: 1.9882
- R²_train_mean: 0.4764
- MZ: α=-0.2529, β=0.9333, R²=0.4800
- pred_std_ratio: 0.742 (mean-collapse 진단)
- Spearman: 0.667
- DM-test vs HAR: stat=-61.484, p=0.0000

## Layer 2 — 포트폴리오 단독 성과

- Sharpe: 1.094
- CAGR: 12.35%
- MDD: -17.55%
- Sortino: 1.694
- Calmar: 0.704
- CAPM α: 17.36% (β=-0.157, t=0.21)
- Information ratio: 0.098
- Hit rate: 64.6%
- CVaR_5: -6.55%

## Layer 3 — ML → BL 인과 추적

- Low vol hit rate: 0.695 (random=0.30)
- High vol hit rate: 0.716
- Rank consistency 평균: 0.765
- P 행렬 turnover 평균: 0.091
