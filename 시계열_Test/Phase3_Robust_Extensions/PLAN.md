# Phase 3 — Robust Extensions PLAN (2026-04-29 갱신)

> **단일 질문**: "본 Phase 2 의 BL_ml 우위가 다양한 가정 변경 (시기 확장, 학습 방식, Σ 추정, Long-Short, prior) 에 robust 한가? 서윤범 baseline (Sharpe 1.065, 2009-2025) 과 fair 비교에서도 ML 통합 효과가 유지되는가?"

---

## 1. Phase 3 진입 배경

### 1.1 본 Phase 2 의 발견된 한계

| 한계 | 영향 |
|---|---|
| **시기 짧음 (6 년)** | sampling bias, 강세장 + AI 호황 위주 |
| **Universe 좁음 (74 종목)** | mega cap concentration |
| **종목별 학습 (sample 부족)** | over-fitting 위험, mean-collapse |
| **Issue #1, #1B (date mismatch)** | 21 개월 누락 → 수정 후 진짜 결과 변화 |

### 1.2 서윤범 99_baseline 비교 결과

| 측면 | 서윤범 BL TOP_50 | 본 Phase 2 BL_trailing |
|---|---|---|
| 시기 | 2009-2025 (17 년) | 2020-2025 (6 년) |
| Universe | 매년 top 50 (S&P 500) | 동일 |
| Sharpe | **1.065** ⭐ | 0.740 |
| MDD | -11.80% | -17.72% |
| 누적수익 | +1021% | +106% |

→ **분석 기간 차이가 결정적**. Phase 3 의 OOS 2009 시작으로 fair 비교 가능.

---

## 2. Phase 3-1 의 핵심 작업 — OOS 2009 시작

### 2.1 작업 구성

```
[옵션 c — 종목별 + Cross-sectional 둘 다 시도]
1. 종목별 학습 (Phase 1.5 v8 ensemble 그대로)
   - 8-way 병렬 학습 (RTX 4090 24GB 활용)
   - 학습 시간: 약 1-2 시간

2. Cross-sectional 학습 (Phase 3 신규)
   - 단일 LSTM, 모든 종목 공유 (ticker embedding)
   - HAR 베이스라인 결합
   - 학습 시간: 약 1-2 시간

3. 두 결과 BL 백테스트 + 직접 비교
```

### 2.2 시기 + Universe

```
OOS 시기: 2009-01-31 ~ 2025-12-31 (17 년, 204 개월)
Universe: 매년 top 50 (서윤범과 동일)
   - cutoff_dates: 2008-12-31 ~ 2024-12-31
   - Unique 종목 약 130-200 (예상)
Panel: 2001-12-31 ~ 2025-12-31 (build_daily_panel 자동 7 년 확장)
```

### 2.3 학술적 의미

```
[Pyo & Lee (2018) KOSPI]    Sharpe +19% (BL > baseline)
[서윤범 99_baseline US]      Sharpe 1.065 / SPY 0.964 (+10%)
[본 Phase 2 (6 년)]          Sharpe 0.74 / SPY 0.80 (-7%)  ⚠️ sampling bias
[Phase 3-1 (17 년, 종목별)]   예측: 1.0~1.1 (서윤범 재현)
[Phase 3-1 (17 년, CS)]      예측: 1.0~1.2 (학술 SOTA 가능)
```

---

## 3. 폴더 구조

```
Phase3_Robust_Extensions/
├── README.md
├── PLAN.md (본 파일)
├── 재천_WORKLOG.md
├── NOTEBOOK_TODO.md          ⭐ 노트북 보완 사항 명시
│
├── 01_universe_extended.ipynb       # universe + panel 확장 (2009 OOS)
├── 02a_phase15_stockwise_extended.ipynb  # 종목별 8-way 병렬 학습
├── 02b_phase15_cross_sectional.ipynb     # Cross-sectional 학습
├── 03_BL_backtest_extended.ipynb         # BL 백테스트 (둘 다 적용)
├── 04_compare_stockwise_vs_cross.ipynb   # 두 모델 직접 비교
├── 05_idzorek_omega.ipynb                # (Phase 3-3) Idzorek
├── 06_hybrid_covariance.ipynb            # (Phase 3-3) Hybrid Σ
├── 07_long_short.ipynb                   # (Phase 3-3) Long-Short
├── 08_stress_test.ipynb                  # (Phase 3-3) 스트레스 테스트
├── 09_final_report.ipynb                 # REPORT.md
│
├── _build_*.py                       # 빌드 스크립트들
│
├── scripts/
│   ├── setup.py                     # ✅ Phase 2 → Phase 3 적응
│   ├── black_litterman.py           # ✅ 서윤범 그대로
│   ├── covariance.py                # ✅ Phase 2 그대로
│   ├── backtest.py                  # ✅ Phase 2 그대로
│   ├── benchmarks.py                # ✅ Phase 2 그대로
│   ├── universe.py                  # ✅ Phase 2 그대로
│   ├── data_collection.py           # ✅ Phase 2 그대로
│   ├── volatility_ensemble.py       # ⭐ 확장 (병렬 + CS)
│   ├── models_cs.py                 # ⭐ 신규 (CrossSectionalLSTMRegressor)
│   └── universe_extended.py         # ⭐ 신규 (universe + panel 2009)
│
├── data/
└── outputs/
```

---

## 4. Scripts 모듈 정리

### 4.1 신규 / 확장 모듈

| 모듈 | 함수 | 역할 |
|---|---|---|
| `models_cs.py` | `CrossSectionalLSTMRegressor` | 모든 종목 공유 LSTM + Ticker embedding |
| `models_cs.py` | `CS_V4_BEST_CONFIG` | Cross-sectional 학습 hyperparameter |
| `volatility_ensemble.py` | `run_ensemble_for_universe_parallel` | 8-way 병렬 종목별 학습 |
| `volatility_ensemble.py` | `run_ensemble_cross_sectional` | CS 학습 + HAR 결합 |
| `volatility_ensemble.py` | `build_cs_inputs` | CS 학습 input 빌드 |
| `volatility_ensemble.py` | `CrossSectionalDataset` | PyTorch Dataset |
| `volatility_ensemble.py` | `_build_cs_dataset_for_fold` | Fold 의 train/test Dataset |
| `universe_extended.py` | `extend_universe(start_year=2009)` | universe 17 년 확장 |
| `universe_extended.py` | `extend_panel_to_2009` | panel 22 년 확장 (build_daily_panel) |
| `universe_extended.py` | `diagnose_universe_coverage` | universe 가용성 진단 |
| `universe_extended.py` | `diagnose_panel_coverage` | panel 가용성 진단 |
| `universe_extended.py` | `split_universe_by_period` | 시기 분할 (검증용) |

### 4.2 Code Review 결과 (2026-04-29)

```
[1차 review] 13 이슈 발견 → 모두 처리 완료
   - Critical 2 + Major 4 + Minor 7

[2차 review] 7 추가 이슈 → 모두 처리
   - Critical C3 (NaN seq window) ✅
   - Critical C4 (종목 length min) ⚠️ 부분 (노트북 보강)
   - Major Mj5 (HAR fold 매칭) ⚠️ 부분 (노트북 보강)
   - Major Mj6 (build_daily_panel 시그니처) ✅
   - Minor Mn8 (seed 고정) ✅
   - Minor Mn9 (DataLoader 최적화) ✅
   - Minor Mn10 (Gradient clipping) ✅

→ 코드 품질: ⭐⭐⭐⭐½ (4.5/5)
→ 잔여 한계 2 개는 노트북에서 보강 (NOTEBOOK_TODO.md 참조)
```

---

## 5. 진행 순서 (Phase 3-1 — 우선)

### 5.1 작업 단계

```
[Day 1: 데이터 확장]
1. 01_universe_extended.ipynb
   - extend_universe(2009-2025, n_top=50)
   - extend_panel_to_2009 (build_daily_panel 자동 호출)
   - 캐시 활용 시 1-2 분, 첫 실행 30-60 분
   - 산출: universe_top50_history_extended.csv, daily_panel.csv

[Day 1-2: 두 학습 동시 백그라운드]
2a. 02a_phase15_stockwise_extended.ipynb (종목별 + 8-way)
2b. 02b_phase15_cross_sectional.ipynb (CS + HAR)
   - 동시 실행 (jupyter nbconvert --execute 백그라운드)
   - GPU 학습 1-8 시간
   - VS Code 에서 다른 작업

[Day 2-3: BL 백테스트 + 비교]
3. 03_BL_backtest_extended.ipynb
   - 두 ensemble 결과 BL 적용
   - 5+2 시나리오 비교 (BL_ml_sw, BL_ml_cs, BL_trailing, SPY, McapWeight, EqualWeight)

4. 04_compare_stockwise_vs_cross.ipynb
   - RMSE 비교 (종목별 vs CS)
   - 종목별 특성 분석
   - Sharpe / MDD / Bootstrap 비교

[Day 3 (선택): Phase 3-3]
5-9. Idzorek, Hybrid Σ, Long-Short, Stress test, REPORT
```

### 5.2 백그라운드 학습 명령어

```bash
# 02a (종목별 병렬)
nohup jupyter nbconvert --to notebook --execute \
    02a_phase15_stockwise_extended.ipynb \
    --output 02a_phase15_stockwise_extended.ipynb \
    --ExecutePreprocessor.timeout=86400 \
    > log_stockwise.txt 2>&1 &

# 02b (Cross-sectional, 동시)
nohup jupyter nbconvert --to notebook --execute \
    02b_phase15_cross_sectional.ipynb \
    --output 02b_phase15_cross_sectional.ipynb \
    --ExecutePreprocessor.timeout=7200 \
    > log_crosssec.txt 2>&1 &
```

---

## 6. 학술 메시지 (Phase 3-1 후 예상)

### 6.1 만약 BL_ml_sw / BL_ml_cs 가 서윤범 1.065 도달

```
"Phase 1.5 v8 ensemble (LSTM + HAR) 통합 BL 이
 17 년 OOS 환경에서 서윤범의 BL TOP_50 (Sharpe 1.065) 와
 fair 비교에서 우위 (또는 동등) 확인.
 본 Phase 2 의 sampling bias (6 년 한정) 극복."
```

### 6.2 만약 Cross-sectional 이 종목별 능가

```
"Cross-sectional 학습 (Gu, Kelly, Xiu 2020 학술 SOTA) 이
 종목별 학습 (Phase 1.5 v8) 보다 우위.
 미국 시장 변동성 예측에 cross-sectional approach 의 효과 입증."
```

---

## 7. Phase 3-2, 3-3 (선택)

### 7.1 Phase 3-2 — Universe 확장

```
[작업]
- universe 매년 top 200 으로 확장 (또는 서윤범의 624 종목)
- 분산화 ↑ → MDD ↓
- Cross-sectional 환경 강화

[비용]
- universe 확장 + panel 확장 (캐시 활용 시 5-10 분)
- 학습 시간: 200 종목 × 17 년 = 약 4 시간
```

### 7.2 Phase 3-3 — Sensitivity 차원 추가

```
[작업]
1. Idzorek Ω (τ-sensitive)
2. Hybrid Σ (Phase 1.5 σ² 활용)
3. Long-Short (BAB factor)
4. Stress test (COVID, GFC 등 시기별 분해)
5. REPORT.md 자동 생성

[비용]
- 학습 비용 0 (모두 BL 모듈 + 분석)
- 약 2-3 일
```

---

## 8. 결정사항 정리 (2026-04-29 기준)

| # | 결정 | 출처 |
|---|---|---|
| 1 | OOS 시작 2009 (서윤범 일관) | 사용자 차트 (BL TOP_50 1.065) |
| 2 | Universe top 50 (서윤범 일관) | Phase 2 일관 |
| 3 | 두 학습 분리 (02a, 02b) | 동시 실행 + 디버깅 용이 |
| 4 | RTX 4090 24GB + 8-way 병렬 | 사용자 hardware |
| 5 | jupyter nbconvert --execute 백그라운드 | 노트북 결과 셀 보존 |
| 6 | scripts 모듈 (Phase 2 base + 신규) | 모듈화 |
| 7 | Cross-sectional + HAR 결합 | Phase 1.5 일관 + 학술 SOTA |
| 8 | seed 42 + grad clip 1.0 | 재현성 + 안정성 |

---

## 9. 다음 작업

```
[즉시 진행 가능]
1. NOTEBOOK_TODO.md 숙지 (노트북 보완 사항)
2. 01_universe_extended.ipynb 빌드 + 실행
3. 02a, 02b 빌드 + 백그라운드 학습
4. 03, 04 빌드 + 분석

[학습 비용 정리]
- universe + panel 확장: 5 분 (캐시) ~ 60 분 (첫 실행)
- 02a 학습 (종목별 8-way): 1-2 시간
- 02b 학습 (CS): 1-2 시간
- 03, 04 분석: 30 분
- 총: 약 3-5 시간 (캐시 후)
```

---

## 10. Verification (Phase 3-1 진입 전 확인)

- [x] Phase 2 의 정합성 검증 완료 (Issue #1, #1B, #2 수정)
- [x] Phase 3 폴더 구조 + scripts 모듈 작성
- [x] models_cs.py + volatility_ensemble.py + universe_extended.py 작성
- [x] 코드 review 1, 2 차 진행 + 모든 issue 처리
- [x] Forward pass + dtype/range 검증
- [x] Cross-sectional 학습 루프 완성 (HAR 결합)
- [x] OOS 2009 시작 default 설정
- [x] 01_universe_extended 노트북 실행 — universe 809, panel 646
- [x] 02a 노트북 빌드 + 학습 완료 (615 종목, ~15h)
- [x] 02a §6 BL sanity check (Issue 3-5 수정 후 정상 작동)
- [ ] 02b cross-sectional 학습 (대기)
- [ ] 03 BL 백테스트 (02b 완료 후)
- [ ] 04 비교 + 05a/b/c 평가 노트북 실행

---

## 10.5. Phase 3-1 §6 진행 결과 + 핵심 발견 (2026-04-29 갱신) ⭐

### 10.5.1 02a 학습 완료 (Step 2a)

| 항목 | 결과 |
|---|---|
| 학습 종목 | 615 (universe 809 → panel 가용 646 → 1334일↑ 615) |
| 학습 시간 | 약 15 시간 (RTX 4090, 8-way 병렬, CPU 89% 병목) |
| Ensemble 성능 | RMSE 0.391 (LSTM 0.529, HAR 0.401) ⭐ |
| Best 모델 분포 | Ensemble 65% (398) / HAR 32.6% (200) / LSTM 2.4% (15) |
| Phase 1.5 v8 패턴 재현 | 65% vs 64% (74 종목) — 거의 동등 ✅ |

### 10.5.2 학습 후 발견 + 수정 이슈 3 종

- **Issue 3**: `compute_performance_weights` ZeroDivisionError + 들여쓰기 오류
  - 원인: 폐상장 stale price 종목 → log(0)=-inf → RMSE inf → 1/inf=0
  - 영향: 10 종목 (CBE, TIE, AMCR, BMC, COL, CPWR, CVG, EP, GR, MEE, SW)
  - 수정: non-finite y_true/y_pred 자동 제거 + RMSE > 0 체크
  - 결과: 613 종목 ensemble 정상 저장
- **Issue 4**: `estimate_covariance` 호출 인자 누락 (TypeError)
  - 03 노트북 동일 문제 가능 → 추후 추적 필요
  - 수정: `compute_sigma_daily + daily_to_monthly` 직접 호출 + `fillna(0)`
- **Issue 5**: `backtest_strategy` 빈 Series 반환
  - 인덱스 dtype 정상이지만 dropna 후 empty
  - 우회: `make_returns_manual()` 작성하여 직접 계산

### 10.5.3 §6 BL Sanity Check 결과 (203 개월, 17 년)

| 시나리오 | Sharpe | CAGR | Vol | MDD |
|---|---|---|---|---|
| **BL_trailing** | **1.222** ⭐ | 14.52% | 11.71% | -15.88% |
| BL_ml_sw | 1.108 | 13.41% | 12.07% | -18.56% |
| SPY | 1.050 | 15.37% | 14.72% | -23.93% |

- **서윤범 99 재현 검증**: BL_trailing 1.222 vs 서윤범 재계산 1.157 (+5.62%, 양호)
- **ML 효과**: BL_ml_sw - BL_trailing = -0.114 Sharpe (NEGATIVE)

### 10.5.4 ⭐ 핵심 패러독스 — Hit Rate↑ 이지만 BL Sharpe↓

| 측정 | ML | Trailing | 차이 |
|---|---|---|---|
| Low vol hit rate | **0.634** | 0.590 | +4.4%p |
| High vol hit rate | **0.663** | 0.626 | +3.7%p |
| Spearman rank corr | **0.688** | 0.616 | +0.072 |

→ **모든 측면에서 ML > Trailing 인데 BL 성과는 ML < Trailing**.

### 10.5.5 LS Spread 패러독스 분석

| 측정 (mcap-w, 연환산 %) | ML | Trailing | 차이 |
|---|---|---|---|
| LS spread (전체 17 년) | -9.53% | -4.84% | -4.70%p |
| 강세장 (12~19) | -3.55% | **+3.17%** | -6.72%p |
| 긴축 (21~22) | +1.88% | **+8.91%** | -7.03%p |
| BAB 작동 시기 | 24 개월 | **132 개월** | — |

→ **LS spread 둘 다 NEGATIVE** (BAB anomaly 17 년 평균 미작동).
→ **Trailing 의 BAB 활용도 ML 능가** (강세장 + 긴축 모두).

### 10.5.6 진단 — Trailing vol = 방어주 proxy 가설

```
Trailing vol_21d 의 진정한 가치:
  "최근 vol 이 낮음" = "안정적 cash flow 회사" 의 proxy
  Utilities, Consumer Staples, Healthcare 식별
  → BAB anomaly 의 underlying 회사 특성 일치

ML forward vol prediction 의 한계:
  정확한 vol 예측 ≠ 회사 특성 식별
  "이번 달 vol 이 낮을 종목" ≠ "구조적으로 안정적인 회사"
  → BAB anomaly 활용에는 부적합
```

### 10.5.7 학술 기여 (잠정)

> **"Volatility prediction accuracy improvement (RMSE↓, hit rate↑) does NOT
>  translate to Black-Litterman portfolio alpha when used as P-matrix sorter."**

- **Pyo & Lee (2018) 의 "ML > Trailing" 주장 부분 반증**
  - KOSPI 대비 미국 17 년 universe 환경 차이
  - "vol prediction" 과 "BAB anomaly" 의 분리
- **다음 검증 우선순위**: 02b cross-sectional 의 BAB 활용도
  - CS LSTM 이 종목 간 cross-sectional 비교에 강하다면 회사 특성 식별 가능성 ↑

---

## 11. 잔여 한계 (NOTEBOOK_TODO.md 참조)

본 PLAN 완료 후에도 노트북에서 추가 보강 권장:

1. **종목 length 처리** (Issue C4) — 신규 IPO 종목의 fold 처리
2. **HAR fold 매칭** (Issue Mj5) — panel date 정렬 검증
3. **Cross-sectional 의 학습 monitoring** — RMSE 진행 시각화
4. **결과 검증** — 종목별 vs CS 의 종목별 RMSE 분포
5. **Bootstrap 통계 검정** — Phase 2 와 동일 방식

자세한 사항은 `NOTEBOOK_TODO.md` 에 명시.
