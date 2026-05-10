# 02a Stockwise (BL_ml_sw) — 종합 진단 보고서

## Layer 1 — 변동성 예측 진단

- RMSE: 0.3927
- QLIKE: 2.5687
- R²_train_mean: 0.4851
- MZ: α=-0.6148, β=0.8566, R²=0.5031
- pred_std_ratio: 0.828 (mean-collapse 진단)
- Spearman: 0.697
- DM-test vs HAR: stat=-44.401, p=0.0000

## Layer 2 — 포트폴리오 단독 성과

- Sharpe: 1.119
- CAGR: 13.21%
- MDD: -18.13%
- Sortino: 1.657
- Calmar: 0.728
- CAPM α: 15.79% (β=-0.131, t=0.24)
- Information ratio: -0.084
- Hit rate: 64.7%
- CVaR_5: -6.84%

## Layer 3 — ML → BL 인과 추적

- Low vol hit rate: 0.725 (random=0.30)
- High vol hit rate: 0.749
- Rank consistency 평균: 0.809
- P 행렬 turnover 평균: 0.085
