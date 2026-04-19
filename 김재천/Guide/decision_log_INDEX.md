# Decision Log — INDEX

> 의사결정 진화 흐름을 한눈에 보기 위한 인덱스. 각 버전의 핵심 결정·계기·산출물 요약.
> 상세 내용은 각 파일 본문 참조.

---

## 버전 흐름

| 버전 | 일자 | 파일 | 핵심 변화 | 최종 추천 전략 (Sharpe) |
|------|------|------|----------|----------------------|
| v3   | 2026-04 (초기) | [decision_log.md](decision_log.md) | EW baseline + Config A/B/C/D 경보 + GDELT 실험 | 보수형_ALERT_B (1.473) |
| v4.1 | 2026-04 (중반) | [decision_log_v31.md](decision_log_v31.md) | MV WF + HMM n=4 + 레짐 Σ + Deployment Sim 철학 | M1_보수형_ALERT_B (1.064) |
| v4.2 | 2026-04-19 | [decision_log_v32.md](decision_log_v32.md) | HMM 다기준 평가 (BIC π 자유도 + n∈[2..8] sweep) | (n=4 유지 — 63.7/80) |
| v4.2b | 2026-04-19 | [work_log_2026-04-19.md](work_log_2026-04-19.md) | Step9 alert_lag=1 (look-ahead 제거) → Strict OOS | M2_보수형_ALERT_A (0.847) |
| v4.2c | 2026-04-19 | [work_log_2026-04-19.md](work_log_2026-04-19.md) §9A | Drift-aware turnover + Block Bootstrap + Bonferroni family 명시 | M2_보수형_ALERT_A (0.781) |
| **v4.2d** | 2026-04-19 (현재) | [c-users-gorhk-finance-project-lazy-ember.md](../../C:/Users/gorhk/.claude/plans/c-users-gorhk-finance-project-lazy-ember.md) | 실물경기 트리거 변수 전면 삭제 + 워밍업 가드 + Step5 IS-LW + Sharpe 차분 자동 + Bootstrap 민감도 + Multi-criteria sweep + master_design WF/δ 확정 + BL/ML v5로 분리 | **M2_보수형_ALERT_A (0.814)** |

---

## 주요 의사결정 추적

### 데이터 수집 및 전처리

| 결정 | 버전 | 근거 |
|------|------|------|
| 워밍업 2014~ + 분석 2016~ | v4.1 | claims_zscore 260일 롤링 안정화 |
| FRED PIT 적용 (vintage) | v4.1 | look-ahead bias 근본 제거 |
| BAA10Y로 HY_spread 대체 | v4.1 | BAMLH0A0HYM2 ICE 라이선스 3년 제약 |
| WEI df_reg_v2 제외 | v4.1 | PIT 소급창조 신설 지표 (2020-04 신설) |
| **실물경기 트리거 변수 전 파이프라인 제거** | **v4.2d** | 데이터 수집 제한 — v5에서 UNRATE z-score, claims_zscore 등 후보 검토 |
| Cu_Au_ratio_chg 의도 명시 (과거 모멘텀) | v4.2d | Step2 노트북 주석 |
| 워밍업 NaN assert 가드 추가 | v4.2d | 분석 시작일 이후 잔존 NaN 차단 |

### 레짐 모델 (Step6)

| 결정 | 버전 | 근거 |
|------|------|------|
| HMM full-sample 학습 | v4.1 | 레짐 정의 고정성 + 위기 레짐 추정 안정성 |
| Forward 알고리즘 (vs Viterbi) | v4.2 | 시점별 인과적 추론 → look-ahead 제거 |
| BIC 공식 보정 (π 자유도 (n-1) 추가) | v4.2 | 공식 학술 완전성 |
| n 탐색 범위 [2..8] 확장 | v4.2 | 가짜 엘보우 검출 |
| 다기준 scorecard (8지표) | v4.2 | BIC 단독 판단 한계 보완 — n=4 유지 |
| **HMM 설계 의도 명시** | **v4.2d** | "full-sample 학습 + Forward 추론" 분리 설계의 정당성 |

### 백테스트 시간 정렬

| 결정 | 버전 | 근거 |
|------|------|------|
| Step9 alert_lag=1 (shift) | v4.2b | t close → t+1 close-to-close 적용 — 일별 alert look-ahead -0.241 SR 제거 |
| Step9 drift-aware turnover | v4.2c | 매일 비중 drift → 다음 turnover 정확화 |
| **alert_lag 시간 정렬 의도 주석 잠금** | **v4.2d** | 코드 리뷰 시 명확화 (Step7·9) |

### 포트폴리오·리스크 분석

| 결정 | 버전 | 근거 |
|------|------|------|
| Walk-Forward IS=504/OOS=63 (분기) | v4.1 | 31 윈도우 + 거래비용 절제 |
| **WF 윈도우 master_design에 확정 표기** | **v4.2d** | 기존 미결 항목 정리 |
| 거래비용 편도 15bps + EW에도 동일 적용 | v4.2 | 공정 비교 |
| Step5 risk_contribution IS-LW 시계열 추가 | v4.2d | Step4 LW와 정합성 (단일 시점 sample cov는 참고용) |

### Ablation·통계 검정

| 결정 | 버전 | 근거 |
|------|------|------|
| Block Bootstrap (Stationary, avg_block=21) | v4.2 | volatility clustering 보존 |
| Bonferroni family=3 primary, 64 supplementary | v4.2c | 투자자 의사결정 단위 (성향, Config) |
| **avg_block 민감도 sweep {5,10,21,42}** | **v4.2d** | 결론 robust 확인 |
| **Multi-criteria 가중치 ±10%p sweep (81 조합)** | **v4.2d** | Top1 안정성 검증 (M2_보수형_ALERT_A 74/81) |

### Black-Litterman / ML

| 결정 | 버전 | 근거 |
|------|------|------|
| ML 모델 종류·피처 결정 | (보류) | v5 과제 |
| BL Ω 산출 방법 (A/B/베이지안) | (보류) | v5 과제 |
| **δ 고정값(성향별) 정책 채택** | **v4.2d** | 시장 Sharpe 역산 대비 비교 용이 |
| **master_design §5/§6에 "v5 도입" 라벨 부착** | **v4.2d** | 현 파이프라인 vs 미래 설계 명확화 |

---

## 핵심 산출물 위치

| 항목 | 경로 |
|------|------|
| 마스터 설계 | [`master_design.md`](../master_design.md) |
| 최종 추천 전략 | [`data/step10_final_recommendation.csv`](data/step10_final_recommendation.csv) |
| Step9 백테스트 결과 | [`data/step9_backtest_results.pkl`](data/step9_backtest_results.pkl) |
| Step9 Sharpe 스냅샷 | [`data/step9_metrics_snapshots/`](data/step9_metrics_snapshots/) |
| Top10 비중 시계열 | [`data/step11_top10_weights.pkl`](data/step11_top10_weights.pkl) |
| Block Bootstrap sweep | [`data/step10_block_sweep.csv`](data/step10_block_sweep.csv) |
| Multi-criteria sweep | [`data/step10_multicriteria_sweep.csv`](data/step10_multicriteria_sweep.csv) |
| Family 라벨 결과 | [`data/step10_bootstrap_with_families.csv`](data/step10_bootstrap_with_families.csv) |
| IS-LW Risk Contribution | [`data/rc_by_window.csv`](data/rc_by_window.csv) |

---

## 향후(v5) 과제 요약

1. ML 수익률 예측 (Q 벡터) — Ridge/XGBoost/LightGBM 중 채택
2. ML 피처 확장 — 베타·상대강도·매크로 nowcast
3. Black-Litterman 도입 — Ω 방법 A/B/베이지안 Ablation
4. 실물경기 트리거 보강 (UNRATE z-score, claims_zscore 등) — v4.x에서 제거된 트리거 대체
5. HMM IS 윈도우별 재학습 옵션 (현재는 full-sample 학습 + Forward 추론)
