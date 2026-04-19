# Final Project 최종 통합 보고서 (v4.2c — 이중 보고 + drift-aware)
## 대안데이터 기반 포트폴리오 시뮬레이터: 전체 파이프라인 한눈 보기

> **프로젝트 기간**: 2016-01 ~ 2025-12 (10년, 2,609 NYSE 영업일)
> **최종 버전**: v4.2c (2026-04-19, **Drift-aware turnover + 통계 family 명시**)
> **이전 버전**: v4.2b (2026-04-19, 이중 보고 체계 도입), v4.1.1 (2026-04-17)
>
> **v4.2c 추가 개선**:
> - Step 9/11 Drift-aware turnover (Option Y 패턴)
> - Step 10 Bonferroni family 정의 명시 (성향·Config별 3 비교)
> - Multi-criteria 가중치 근거 명시 (문헌 + 민감도 분석)

### ⚠️ v4.2b 핵심 개선 — Alert Look-ahead 완전 해결

2026-04-19 엄밀 재검토 중 **alert_t가 close_t 기반 계산되나 같은 날 리밸런싱에 사용되는 look-ahead 패턴** 발견. Step 7/9/11에서 `shift(1)` 적용하여 실무 정합성 확보. 결과는 **두 체계**로 병기:

| 체계 | 설명 | 대표 성과 |
|------|------|----------|
| **Deployment Simulation (v4.1)** | "full-sample 학습 모델을 과거에 배포했다면" 이상적 성과 | **M1_보수형_ALERT_B** Sharpe 1.064 |
| **Strict OOS (v4.2b)** ⭐ | 각 시점에서 이용 가능한 정보만 사용 (실무 기준) | **M2_보수형_ALERT_A** Sharpe 0.847 |

**학술적 / 실무적 권장**: **Strict OOS (v4.2b) 결과를 공식 결론**으로 사용
- Look-ahead 정량: 10년 누적 +53.67% (연평균 +5.37%)
- 특히 위기일(2020 COVID, 2018 Volmageddon)에 look-ahead 이득 집중
- 경보 시스템의 실질 가치는 **거래비용에 의해 상쇄** — Step 10 Block Bootstrap에서 유의하게 M0 하회 실증

> **주요 발견**: 이론적으로 정교해 보이는 경로 2(HMM 레짐→Σ 전환)는 **Strict OOS에서도 부가가치 없음** (M3 ≈ M1). 하지만 경로 1 **단독**(M1)도 M0(순수 MV) 대비 유의하게 악화 → **경로 2 단독(M2) + 보수형**이 유일한 60/40 우위 전략

---

## 📑 목차

- [1. Executive Summary](#1-executive-summary)
- [2. 프로젝트 개요](#2-프로젝트-개요)
- [3. Step 1~7 (v3 파이프라인) 요약](#3-step-1-7-v3-파이프라인-요약)
- [4. Step 8~10 (v4.1 확장) 요약](#4-step-8-10-v41-확장-요약)
- [4.5 Step 11 (v4.1.1 시각화 확장)](#45-step-11-v411-시각화-확장)
- [5. 최종 결론](#5-최종-결론)
- [6. v3 vs v4.1 비교](#6-v3-vs-v41-비교)
- [7. 한계 및 향후 과제](#7-한계-및-향후-과제)
- [8. 부록 — 파일 지도](#8-부록-파일-지도)

---

## 1. Executive Summary

### 🎯 핵심 성과 — 이중 보고 (v4.2b)

#### A. Deployment Simulation (v4.1, 상한 성능)
| 항목 | 값 |
|------|-----|
| 최우수 전략 | **M1_보수형_ALERT_B** |
| Sharpe | **1.064** |
| MDD | **-15.53%** |
| 누적수익률 | +151.4% |
| 연율수익률 | +12.24% |
| vs EW Sharpe | +29.2% 개선 |
| vs SPY Sharpe | +39.8% 개선 |
| ⚠️ 한계 | Alert look-ahead로 연 +5.37% 인위적 이득 포함 |

#### B. Strict OOS (v4.2c, 실무 기준) ⭐

| 항목 | 값 |
|------|-----|
| 최우수 전략 | **M2_보수형_ALERT_A** |
| Sharpe | **0.814** (v4.2b 0.847에서 drift 반영) |
| MDD | **-19.16%** |
| 누적수익률 | +122% |
| 연율수익률 | +11.1% |
| vs 60/40 Sharpe | ≈ 동등 (0.814 vs 0.814) |
| vs SPY Sharpe | -5.2% |
| Multi-criteria 1위 | ✅ (낮은 MDD로 Sharpe 열위 극복) |
| 경로 1 활용 | ❌ (경보 cut 없음) |
| 경로 2 활용 | ✅ (월별 레짐 Σ 전환) |

### 📊 세 체계의 Sharpe 비교 (모드별 평균)

| 모드 | v4.1 | v4.2b | **v4.2c** | 주요 변화 원인 |
|------|------|-------|-----------|---------------|
| M0 (MV only) | 0.796 | 0.796 | **0.781** | drift cost (-0.015) |
| M1 (경보) | **0.950** | 0.709 | **0.699** | alert lag + drift |
| M2 (레짐 Σ) | 0.791 | 0.787 | **0.771** | drift cost |
| M3 (통합) | **0.950** | 0.706 | **0.697** | alert lag + drift |
| BENCH_60_40 | 0.814 | 0.814 | **0.814** | 불변 (단일 비중) |

### 🧬 Look-ahead Bias 정량 분해

| 지표 | 수치 |
|------|------|
| Alert 변경일 (보수형 기준) | 257일 / 2,491일 (10.3%) |
| Look-ahead 유리 비율 | **91.8%** |
| 10년 누적 이득 | **+53.67%** |
| 연평균 이득 | **+5.37%** |
| 최대 단일일 이득 | +2.00% (2018-02-05 Volmageddon) |
| 위기 집중도 | 2020 COVID +12.06%, 2018 Vol +8.58% |

### 💡 5가지 주요 발견

1. **경로 1(일별 경보→주식 축소)의 실전 가치 확인** — M1 평균 Sharpe +0.17 개선
2. **경로 2(레짐 Σ 전환) 무효성 실증** — 2차 재설계 후에도 역효과 (-0.045)
3. **단순성의 가치** — M1(경로 1만)이 M3(통합)보다 우수
4. **보수형 + Config B** — 위험 조정 수익률 기준 최적 조합
5. **Deployment Simulation 철학의 유효성** — Full-sample HMM + IS 기반 의사결정 조합

### 🔢 실무 운용 가이드 요약

```
1. 분기마다 IS 24개월 데이터로 μ, Σ_all 추정 (Ledoit-Wolf)
2. MV 최적화: γ=8, max_equity=35%, min_bond=30%
3. 매일 Config B 경보 확인 (VIX + Contango 기반):
   - L1 → 주식 15% 감축 (→ 채권 70% + 금 30%)
   - L2 → 주식 35% 감축
   - L3 → 주식 60% 감축
4. 편도 15bps 거래비용 가정
5. Σ 레짐 전환 로직 불필요 (실증 무효)
```

---

## 2. 프로젝트 개요

### 2.1 연구 가설

> **대안데이터(VIX 기간구조, HY 스프레드, 수익률 곡선 등)를 포트폴리오에 두 경로로 주입하면 Sharpe Ratio가 개선되는가?**

### 2.2 이중 경로 설계

```
대안데이터
    ├─ 경로 1: 복합 스코어 → 경보 등급 → 주식 비중 축소 (일별)
    └─ 경로 2: HMM 레짐 → Σ_crisis 전환 → 최적화 비중 자동 변화 (월별)
```

### 2.3 v2 실패 교훈 (전제)

| v2 누락 요소 | v3 해결 방안 |
|---------|------|
| Look-ahead bias | Walk-Forward 엄격 분리 |
| 투자자 성향 미반영 | γ=1,2,4,8 4개 프로파일 |
| 계층형 공분산 미구현 | L1 블록 대각 + L2 PCA (설계만) |

### 2.4 10단계 파이프라인

```
Step 1  데이터 수집 (yfinance + FRED)
Step 2  전처리 + EDA + Granger 검정
Step 3  포트폴리오 최적화 (MV/RP/HRP 개념)
Step 4  Walk-Forward 백테스트
Step 5  리스크 분석 (VaR/CVaR)
Step 6  HMM + 경보 Config A/B/C/D
Step 7  EW 기반 Ablation (v3 결과, 1.473)
─────────────────────────────────────────
Step 8  Regime-Aware Covariance (v4.1 신규)
Step 9  Integrated Walk-Forward Backtest (v4.1 신규)
Step 10 Ablation & Statistical Testing (v4.1 신규)
```

---

## 3. Step 1~7 (v3 파이프라인) 요약

### 3.1 Step 1 — 데이터 수집

**목적**: 30개 포트폴리오 자산 + 12개 외부 지표 + 8개 FRED 시리즈를 2016-2025 기간으로 수집

**주요 산출물**:
- `portfolio_prices.csv`: 30자산 × 2,609일
- `external_prices.csv`: VIX 관련 12지표
- `fred_data.csv`: Sahm Rule, ICSA 등 8개 매크로

**30개 포트폴리오 자산**:
- 인덱스 ETF 5 (SPY, QQQ, IWM, EFA, EEM)
- 섹터 ETF 11 (XLK~XLB)
- 개별주 8 (AAPL, MSFT, ..., XOM)
- 채권 4 (TLT, AGG, SHY, TIP)
- 대체 2 (GLD, DBC)

**핵심 데이터 특성**:
- 영업일 인덱스 정렬 (NYSE 기준)
- forward-fill + back-fill로 결측 처리
- 로그 수익률 + 산술 수익률 병용

👉 **상세**: [docs/Step1_해설.md](docs/Step1_해설.md)

### 3.2 Step 2 — 전처리·EDA·Granger

**목적**: 43개 변수의 선행성을 Granger 인과검정으로 검증, 파생 피처 15개 생성

**주요 결과**:
- **df_reg_v2** (2,328×44) 구축 — Step 3 이후 입력
- Granger 검정 **34개 변수 유의** (p<0.05)

**Top 5 선행 지표** (Granger p-value):
| 순위 | 변수 | p-value | 해석 |
|------|------|---------|------|
| 1 | HY_spread_chg | 7.2e-65 | **신용 스트레스 급변이 가장 강력** |
| 2 | VIX_contango_chg | 1.3e-28 | 변동성 기간구조 변화 |
| 3 | VIX_level_chg | 4.8e-22 | 변동성 급변 |
| 4 | yield_curve_inverted | 5.1e-15 | 금리 역전 |
| 5 | Cu_Au_ratio_chg | 8.9e-12 | 경기선행 지표 |

**15개 파생 피처** (일부):
- VIX_contango = VIX3M - VIX (기간구조)
- HY_spread_chg = 5일 변화 (신용 스트레스)
- yield_curve = 10Y - 2Y (경기사이클)
- Cu_Au_ratio = Copper/Gold (경기낙관지수)

👉 **상세**: [docs/Step2_해설.md](docs/Step2_해설.md)

### 3.3 Step 3 — 포트폴리오 최적화

**목적**: MV/RP/HRP 세 가지 최적화를 4개 성향 프로파일에 적용

**핵심 설계**:
- **투자자 성향**: γ=8(보수형) / γ=4(중립형) / γ=2(적극형) / γ=1(공격형)
- **제약**: max_equity, min_bond, target_vol, max_mdd
- **Level 1 배분** (3자산군): 보수형 Equity 29% / Bond 31% / Alt 39%

**성향별 핵심 파라미터**:

| 프로파일 | γ | max_equity | min_bond | 연율 σ 목표 |
|---------|---|---------|---------|-------|
| 보수형 | 8 | 43% | 31% | 10% |
| 중립형 | 4 | 70% | 13% | 17% |
| 적극형 | 2 | 83% | 4% | 20% |
| 공격형 | 1 | 90% | 0% | 22% |

**설계 특이점**: Level 1 배분은 **전체 10년 데이터 기반**이라 Step 4의 동적 WF와 **불일치** (v4에서 재검토)

👉 **상세**: [docs/Step3_해설.md](docs/Step3_해설.md)

### 3.4 Step 4 — Walk-Forward 백테스트

**목적**: 과학적 엄밀성을 갖춘 MV 최적화 기반 성과 측정

**WF 파라미터**:
- IS 24개월 / OOS 3개월 / 슬라이드 3개월
- 총 **31개 윈도우** (2018-01 ~ 2025-09)
- 거래비용 편도 15bps

**성향별 성과**:

| 프로파일 | Total | Sharpe | MDD |
|---------|-------|-------|------|
| 보수형 | +119% | 0.81 | -21% |
| 중립형 | +190% | 0.78 | -27% |
| 적극형 | +258% | 0.77 | -31% |
| 공격형 | +320% | 0.81 | -33% |

**관찰**: MV 최적화 단독은 **벤치마크 대비 미미한 개선** → 대안데이터 주입의 필요성 입증

👉 **상세**: [docs/Step4_해설.md](docs/Step4_해설.md)

### 3.5 Step 5 — 리스크 분석

**목적**: VaR/CVaR + 11개 스트레스 시나리오로 하방 리스크 정량화

**주요 지표**:
- **VaR 95% (일별)**: 보수형 -1.1%, 공격형 -2.5%
- **CVaR 99% (일별)**: 보수형 -2.8%, 공격형 -4.3%
- **역사적 스트레스 6개**: 2018 Volmageddon, 2020 COVID, 2022 긴축, 2023 SVB, 2024 엔캐리 등
- **가상 스트레스 5개**: 금리 +50bp, VIX 80, HY +300bp 등

**COVID 스트레스 결과**:
- 보수형: -6.9% (SPY -34% 대비 강한 방어)
- 공격형: -21.4%

👉 **상세**: [docs/Step5_해설.md](docs/Step5_해설.md)

### 3.6 Step 6 — HMM + 경보 Config

**목적**: HMM 4레짐 분류 + 4가지 경보 Config(A/B/C/D) 설계

**HMM 4레짐** (BIC=5,239 최소 선택):

| 레짐 | VIX 평균 | HY 평균 | 비중 | 해석 |
|------|---------|--------|------|------|
| 0 | 12.3 | 3.63% | 20.2% | 저변동 |
| 1 | 19.2 | 3.17% | 30.2% | 일반 |
| 2 | 19.9 | 4.34% | 31.3% | 신용긴장 |
| 3 | 22.3 | 4.48% | 18.3% | 고변동 |

**4개 경보 Config**:

| Config | 정의 |
|--------|------|
| **A** | VIX 단독 (≥20→L1, ≥28→L2, ≥35→L3) |
| **B** | A + VIX Contango<0 → +1 |
| **C** | 7지표 복합 스코어 + 롤링 분위(p75/p90/p97) + 전문가 트리거 |
| **D** | C + 디바운스 (5일 중 3일 동일 레벨 확인) |

**경보 정밀도** (5일 내 고변동 진입 예측):
- Config A: 62%, Config B: 70%, Config C: 68%, Config D: 72%

👉 **상세**: [docs/Step6_해설.md](docs/Step6_해설.md)

### 3.7 Step 7 — 동적 리밸런싱 + Ablation (v3 결과)

**목적**: EW 기준으로 16개 조합(4성향 × 4Config) + 2 벤치마크 비교

**성향별 주식 감축 테이블**:

| 프로파일 | L1 | L2 | L3 |
|---------|-----|-----|-----|
| 보수형 | 15% | 35% | 60% |
| 중립형 | 10% | 25% | 50% |
| 적극형 | 5% | 15% | 35% |
| 공격형 | 0% | 10% | 25% |

**v3 최우수 전략** (⚠️ v4.1에서 수정됨):

| 순위 | 전략 | Sharpe |
|------|------|-------|
| 1 | 보수형_ALERT_B | **1.473** |
| 2 | 보수형_ALERT_C | 1.376 |
| 3 | 보수형_ALERT_A | 1.364 |

**Bootstrap 검정**: Config B vs A 유의 개선 (보수형·중립형)

**v3의 한계** (v4에서 개선):
- Baseline이 **Equal Weight** → MV 최적화 가치 미반영
- **경로 2 미구현** → decision_log v3 설계와 불일치

👉 **상세**: [docs/Step7_해설.md](docs/Step7_해설.md)

---

## 4. Step 8~10 (v4.1 확장) 요약

> ⚠️ **v4.2b 업데이트 (2026-04-19)**: 본 섹션의 숫자는 **v4.1 Deployment Simulation** 기준. Strict OOS(v4.2b) 결과는 각 Step의 해설.md 또는 [decision_log_v32.md §9](decision_log_v32.md) 참조. 주요 변화:
> - M1/M3 평균 Sharpe **-0.24** (look-ahead 제거로 하락)
> - M0/M2 거의 불변 (alert 미사용 또는 월별만)
> - Top 10 전면 교체 (v4.1 M1/M3 → v4.2b M2/M0)
> - 최종 추천: **M1_보수형_ALERT_B → M2_보수형_ALERT_A**

### 4.1 Step 8 — Regime-Aware Covariance

**목적**: WF 31개 윈도우 각각에 **Σ_stable, Σ_crisis 분리 추정**

**핵심 설계**:
- Full-sample HMM 레짐 라벨을 "고정 렌즈"로 사용
- IS 데이터를 Stable/Crisis로 분할하여 Σ 각각 Ledoit-Wolf 추정
- **대칭 4단계 Fallback** (separate/scaled/scaled_reverse/single)

**실측 Fallback 분포** (31 윈도우):
- separate 71.0%, scaled 12.9%, scaled_reverse 16.1%, single 0.0%

**실증 관찰** (2018-2019 예시):
- Crisis 시 SPY σ: 13.45% → 15.66% (+16%)
- Crisis 시 SPY-AGG 상관: -0.06 → -0.19 (헤지 효과 강화)

👉 **상세**: [docs/Step8_해설.md](docs/Step8_해설.md)

### 4.2 Step 9 — Integrated Walk-Forward Backtest

**목적**: 경로 1 + 경로 2를 MV baseline 위에 통합한 64 시뮬레이션

**4개 모드 설계**:

| 모드 | Σ | 일별 경보 | 의미 |
|------|---|--------|------|
| M0 | Σ_all | ❌ | 순수 MV |
| M1 | Σ_all | ✅ | 경로 1만 |
| M2 | Σ_stable/crisis (**월별 전환**) | ❌ | 경로 2만 |
| M3 | Σ_stable/crisis (**월별 전환**) | ✅ | 통합 |

**v4.1 핵심 변경**: 경로 2 Σ 선택을 OOS 1회 → **월 단위 재전환**으로 재설계 (피드백 반영)

**모드별 평균 Sharpe**:

| 모드 | 평균 | vs M0 |
|------|------|-------|
| M0 | 0.794 | - |
| **M1** | **0.960** | **+0.166** |
| M2 | 0.749 | **-0.045** |
| M3 | 0.920 | +0.126 |

**충격적 발견**: M3 < M1 (경로 2 추가가 **역효과**)

👉 **상세**: [docs/Step9_해설.md](docs/Step9_해설.md)

### 4.3 Step 10 — 통계 검정 및 최종 추천

**목적**: 9개 서브섹션으로 엄밀한 통계 검정 + 최종 추천

**핵심 통계 결과**:

| 비교 | Raw 유의 | FDR 유의 | 평균 ΔSharpe | IR |
|------|--------|--------|------------|-----|
| M1 vs M0 | 8/16 | 3/16 (18.8%) | +0.167 | +0.42 (양호) |
| M2 vs M0 | 0/16 | 0/16 | -0.045 | -0.21 (음) |
| M3 vs M0 | 5/16 | 0/16 | +0.126 | +0.28 |
| **M3 vs M1** | **0/16** | **0/16** | **-0.040** | **-0.29 (음)** |

**v4.1 방법론 개선**:
- Cohen's d (부적합) → **IR + ΔSR 실무 기준**으로 교체
- M0 기준 비교 → **M3 vs M1 직접 검정** 추가

**최종 추천 (Multi-criteria)**:
- 1위: **M1_보수형_ALERT_B** (Sharpe 1.064, MDD -15.53%, Multi-score 4.10)
- 2위: M1_중립형_ALERT_C (Sharpe 1.066, MDD -20.30%)
- 3위: M3_중립형_ALERT_C (Sharpe 1.028)

👉 **상세**: [docs/Step10_해설.md](docs/Step10_해설.md)

---

## 4.5 Step 11 (v4.1.1 시각화 확장)

> ⚠️ **v4.2b 업데이트 (2026-04-19)**: Step 11의 `run_simulation_with_weights`에도 `shift(1)` 적용. Top 10이 전면 교체됨(v4.1 M1/M3 → v4.2b M2/M0 위주). 시각화 8종 모두 v4.2b 결과로 재생성. v4.2b Top 1은 **M2_보수형_ALERT_A**.

### Step 11 — Top 10 전략 자산 구성 시간 변화 시각화

**목적**: Top 10 전략이 **언제·무엇을·왜** 바꿨는지 8종 시각화로 해부

**주요 구현**:
- 신규 `run_simulation_with_weights` 함수 — Step 9 로직 + weights/events/sigma/alert 기록
- Top 10 전략에 대해 재시뮬 (약 3~4분)
- 검증 1: Step 9 누적수익률과 0.5% 이내 일치 (10/10 통과)

**8종 시각화**:

| # | 파일명 | 핵심 전달 |
|---|------|--------|
| 1 | step11_01_dashboard.png | 2×3 통합 대시보드 (최우수 전략) |
| 2 | step11_02_stacked_area_top1.png | Top 1 30자산 비중 흐름 |
| 3 | step11_03_smallmultiples_top10.png | Top 10 자산군 비교 (2×5) |
| 4 | step11_04_weight_heatmap.png | 분기별 자산 비중 히트맵 |
| 5 | step11_05_alert_vs_equity.png | 경보-주식비중 동조 증명 |
| 6 | step11_06_regime_sigma_overlay.png | M1 vs M3 경로 2 무효성 시각적 재확인 |
| 7 | step11_07_turnover_breakdown.png | 리밸런싱 원인 분해 |
| 8 | step11_08_asset_stability.png | 앵커 vs 반응 자산 해부도 |

**핵심 발견 5가지**:
1. **경보 시스템 실전 증명**: L0 주식 40% → L3 주식 16% (체계적 감소)
2. **앵커-반응 이원 구조**: 채권·금이 안정 앵커, 주식이 반응 버퍼
3. **경로 2 무효성 시각적 재확인**: M1과 M3의 비중 흐름 거의 동일
4. **Turnover 주 원인 = 경보**: 분기 리밸런싱보다 일별 경보가 대부분
5. **자산 기여도 불균형**: 실제 성과 동력은 채권·금 앵커

**산출물**:
- `Step11_Top10_Composition_Analysis.ipynb` (22셀)
- `data/step11_top10_weights.pkl` (Top 10 일별 비중 시계열, 6.3 MB)
- `images/step11_01~08_*.png` (8개)

👉 **상세**: [docs/Step11_해설.md](docs/Step11_해설.md)

---

## 5. 최종 결론 (이중 보고 체계, v4.2b)

### 🏆 최종 추천 전략 — 두 체계 병기

#### A. Deployment Simulation (v4.1 — 모델 상한선)
- **M1_보수형_ALERT_B** (Sharpe 1.064, MDD -15.53%)
- ⚠️ **Alert look-ahead 포함** (연 +5.37% 인위적 이득)
- 학술적 탐구용, 실무 배포 시 성과 과대 추정

#### B. Strict OOS (v4.2b — 실무 기준) ⭐ **공식 추천**
- **M2_보수형_ALERT_A** (Sharpe 0.847, MDD -19.44%)
- Alert look-ahead 제거 후 실질 성과
- 60/40 벤치마크 초과 + MDD 열위 아닌 5개 전략 중 1위

### 5.1 Strict OOS 선정 근거 (Multi-criteria)

**M2_보수형_ALERT_A**가 1위인 이유:

| 기준 | 값 | 순위 |
|------|-----|-----|
| Sharpe | 0.847 | 1/64 (non-bench) |
| MDD | -19.44% | Top 5 |
| 연환산 수익률 | 11.43% | 중상위 |
| 60/40 Sharpe 차이 | +0.032 | **유일한 우위** |
| 60/40 MDD 차이 | +1.59%p (개선) | Top |
| **Multi-score** | **5.450 (1위)** | 단독 |

### 5.2 왜 M1 → M2로 바뀌었나? — Look-ahead의 영향

**핵심 발견**: Step 9의 52 전략 중 look-ahead 제거 후 영향 패턴

| 모드 | v4.1 | v4.2b | Δ | 해석 |
|------|------|-------|------|------|
| M0 (MV only) | 0.796 | 0.796 | **0.000** | ✅ alert 미사용 (불변) |
| M1 (일별 경보) | 0.950 | 0.709 | **-0.241** | 🚨 **경로 1이 M0보다 악화** |
| M2 (월별 레짐) | 0.791 | 0.787 | -0.004 | 월별 alert 미미 |
| M3 (통합) | 0.950 | 0.706 | **-0.243** | M1과 거의 동일 악화 |

**실증적 결론 (v4.2b)**:
- **경로 1 유해성 (신규 발견)**: Block bootstrap에서 M1 vs M0 **유의하게 악화** (-0.18 Sharpe, 6건)
- **경로 2 부가가치 없음 (v4.1 유지)**: M3 vs M1 ΔSharpe 평균 -0.005 (유의 0건)
- **M2 단독이 최선**: 월별 레짐 Σ 전환 + 보수형 제약 → 60/40 초과

### 5.3 경보 시스템 실질 가치 재평가

**이론적 설계**: 경보 상승 → 주식 감축 → 위기 회피
**실증 결과** (v4.2b strict OOS):
- **Alert 변경일 257일 중 91.8%에서 look-ahead 유리**
- **거래비용(편도 15bps)이 cut 이익 상쇄** → 실질 가치 없음
- **Block bootstrap에서 M1 < M0 유의** → 통계적으로 **경보 사용이 악화 요인**

### 5.4 Look-ahead Bias 정량 분해 (보수형 기준)

| 지표 | 수치 |
|------|------|
| 10년 누적 이득 | **+53.67%** |
| 연평균 이득 | +5.37% |
| 위기 집중도 | 2020 COVID +12.06%, 2018 Volmageddon +8.58% |
| 최대 단일일 | +2.00% (2018-02-05) |

**교훈**: VIX 급등 = SPY 급락의 강한 음의 상관 (-0.82)으로 인해, 당일 close VIX로 alert 계산 → 그 당일 수익률 평가 시 인위적 "위기 회피" 효과 발생. 이것이 v4.1 결과의 최우수 전략 서사의 주 동인이었음.

### 5.5 시사점 — 이론과 실증의 엄밀한 구분

1. **이론적 정교함 ≠ 실증적 효과**: 경로 2 무효 (v4.1), 경로 1 유해 (v4.2b)
2. **Look-ahead는 은밀함**: 1일 차이가 누적하여 연 5%+ 인위적 이득
3. **실무 배포 검증**: Strict OOS backtest는 논문/학술용으로 필수
4. **가장 정직한 전략**: **M2_보수형 (60/40 소폭 우위)** 또는 **M0_보수형 (60/40과 동등, alert 무관)**

---

## 6. v3 vs v4.1 vs v4.2b 비교

### 6.0 세 버전의 핵심 차이

| 항목 | v3 | v4.1 | **v4.2b** ⭐ |
|------|-----|------|------|
| Baseline | EW 1/30 | MV 최적화 | MV 최적화 |
| 경로 2 | 미구현 | 월별 Σ 전환 구현 | 월별 Σ 전환 (동일) |
| Alert 시차 | Same-day | Same-day (look-ahead) | **shift(1) 적용** |
| Bootstrap | IID | IID | **IID + Block 병행** |
| 벤치마크 비용 | 불일관 | 엄격 | 동일 + Step 7 공정 비교 |
| 통계 검정 | CI만 | Bonferroni + FDR | + Block bootstrap |
| 최우수 | 보수형_ALERT_B (1.473) | M1_보수형_ALERT_B (1.064) | **M2_보수형_ALERT_A (0.847)** |
| 학술 위치 | 개념 증명 | Deployment Sim | **Strict OOS (실무 기준)** |

### 6.0.1 왜 세 버전이 다 다른 숫자를 내는가?

1. **v3 → v4.1**: Baseline EW → MV 변경 + 경로 2 추가 → 절대 Sharpe 기준 변동
2. **v4.1 → v4.2b**: Alert look-ahead 제거 → 연 +5.37% 인위적 이득 제거 → 실질 Sharpe 하락

### 6.0.2 어느 버전을 신뢰해야 하나?

- **학술 논문 / 실무 배포**: **v4.2b (Strict OOS)** — 유일한 정직한 추정
- **모델 역량 탐구**: v4.1 (Deployment Sim) — "이상적" 상한선
- **초기 탐색**: v3 (EW 기반) — 교육용 개념 증명

### 6.1 구조적 차이

| 항목 | v3 | v4.1 |
|------|-----|-----|
| 파이프라인 단계 | 7개 | **10개** (+Step 8,9,10) |
| Baseline | Equal Weight | **MV 최적화** |
| 경로 2 구현 | ❌ | ✅ (무효 확인) |
| 통계 검정 | Bootstrap 95% CI | **+Bonferroni +FDR +IR +ΔSR** |
| M3 vs M1 비교 | 없음 | **추가** |
| Fallback 로직 | 없음 | **대칭 4단계** |
| 문서화 | report_v3 | **+ report_v4 + 해설 10개 + decision_log_v31** |

### 6.2 최우수 전략 비교

| 버전 | 최우수 | Sharpe | 구조 |
|------|------|-------|------|
| v3 | 보수형_ALERT_B | 1.473 | EW + 경로 1 |
| **v4.1** | **M1_보수형_ALERT_B** | **1.064** | **MV + 경로 1** |

### 6.3 주요 발견의 진화

**v3 발견**:
- 경보 기반 동적 리밸런싱의 가치 (EW 대비)
- VIX + Contango (Config B)의 효과

**v4.1 추가 발견**:
- MV 위에서도 경로 1 효과 유지 (감소는 있음)
- **경로 2의 실증적 무효성**
- Cohen's d 재무 부적합 → IR/ΔSR 표준 확립
- 매매비용 완화 장치 검토 필요성 (M1 FDR 18.8%)

---

## 7. 한계 및 향후 과제

### 7.1 한계 (v4.1 시점)

1. **Deployment Simulation의 간접 look-ahead**: Full-sample HMM 학습
2. **M1 FDR 유의율 경계선 (18.8%)**: 비용 30bps 환경에서 효과 감소 가능
3. **Bonferroni 보정의 보수성**: 검정력 희생
4. **단일 시장 (미국)**: 글로벌 분산 미반영
5. **가상 스트레스 단순화**: 정적 배분 근사
6. **8년 OOS의 특이성**: COVID, 2022 긴축 포함 특수 기간

### 7.2 향후 과제 로드맵

**v4.2 (단기, 4~6시간)**:
- 최소 유지 기간 3일 적용 시뮬레이션 (거래비용 완화)
- 4그룹 Σ 별도 추정 실험 (Step 8 산출물 활용)
- 거래비용 감응도 분석 (0/5/15/30/50 bps)

**v5 (중기, 2~3일)**:
- **경로 2 대체안 실험**:
  - 레짐 조건부 max_equity 동적 축소
  - 레짐 조건부 Risk Budgeting
  - μ 전환 (Σ 대신 기대수익률만 레짐별 상이)
- 2006-2015 데이터 확장 → 진정한 10년 rolling HMM
- Expanding Annual Refresh 민감도 검증

**v6 (장기)**:
- 글로벌 확장 (신흥국, 부동산, 원자재)
- LLM 뉴스 감성 분석 피처
- Real-time deployment (paper trading)

---

## 8. 부록 — 파일 지도

### 📂 전체 파일 구조

```
finance_project/김재천/Guide/
│
├── 📓 노트북 (11개)
│   ├── Step1_Data_Collection.ipynb
│   ├── Step2_Preprocessing_EDA.ipynb
│   ├── Step3_Portfolio_Optimization.ipynb
│   ├── Step4_WalkForward_Backtest.ipynb
│   ├── Step5_Risk_Analysis.ipynb
│   ├── Step6_Regime_Alert.ipynb
│   ├── Step7_Dynamic_Rebalancing.ipynb
│   ├── Step8_Regime_Covariance.ipynb                (v4.1)
│   ├── Step9_Integrated_Backtest.ipynb              (v4.1)
│   ├── Step10_Ablation_Final.ipynb                  (v4.1)
│   └── Step11_Top10_Composition_Analysis.ipynb      (v4.1.1, 시각화 확장)
│
├── 📋 설계 기록
│   ├── decision_log.md        (v3 기록)
│   ├── decision_log_v31.md    (v4.1 추가)
│   └── stats_model.md         (통계 기법 요약)
│
├── 📊 보고서
│   ├── report_v3.md           (Step 1~7, 세부)
│   ├── report_v4.md           (Step 8~10, 세부)
│   └── report_final.md        ⭐ (본 문서, 통합)
│
├── 📖 docs/ (비전문가용 해설, 11종)
│   ├── Step1_해설.md
│   ├── Step2_해설.md
│   ├── Step3_해설.md
│   ├── Step4_해설.md
│   ├── Step5_해설.md
│   ├── Step6_해설.md
│   ├── Step7_해설.md
│   ├── Step8_해설.md
│   ├── Step9_해설.md
│   ├── Step10_해설.md
│   └── Step11_해설.md      ⭐ (v4.1.1)
│
├── 📋 quick_reference/ ⭐ (v4.1.1) — 13종 빠른 참조
│   ├── 01_executive_one_pager.md    (30초 요약)
│   ├── 02_investor_summary_card.md  (실전 가이드)
│   ├── 03_v3_v4_changes.md
│   ├── 04_pipeline_flowchart.md
│   ├── 05_path1_vs_path2.md
│   ├── 06_decision_tree.md
│   ├── 07_data_erd.md
│   ├── 08_crisis_case_studies.md
│   ├── 09_timeline_narrative.md
│   ├── 10_day_in_life.md
│   ├── 11_glossary_cheatsheet.md
│   ├── 12_faq_unified.md
│   ├── 13_operating_checklist.md
│   └── README.md
│
├── 🎮 interactive/ ⭐ (v4.1.1) — HTML + Streamlit
│   ├── dashboard.html                 (정적 HTML, 1.1MB)
│   ├── streamlit_app/                 (8페이지 인터랙티브 앱)
│   │   ├── app.py
│   │   ├── pages/ (7개 페이지)
│   │   ├── utils/ (data_loader, theme)
│   │   └── requirements.txt
│   └── README.md
│
├── 💾 data/ (CSV + PKL)
│   ├── portfolio_prices.csv, external_prices.csv, fred_data.csv
│   ├── df_reg_v2.csv, features.csv, granger_results.csv
│   ├── profiles.csv, optimal_weights.csv
│   ├── regime_history.csv, alert_signals.csv
│   ├── regime_covariance_by_window.pkl, regime_covariance_4group.pkl
│   ├── step9_backtest_results.pkl, step9_metrics.csv
│   ├── step10_final_recommendation.csv, step10_cost_mitigation_decision.pkl
│   └── step11_top10_weights.pkl                      ⭐ (v4.1.1)
│
└── 🖼️ images/ (36+ PNG)
    ├── step1~step7_*.png
    ├── step8_01~03_*.png
    ├── step9_01~04_*.png
    ├── step10_01~04_*.png
    ├── step11_01~08_*.png                            ⭐ (v4.1.1)
    ├── infographic_poster.png                        ⭐ (v4.1.1 포스터)
    └── key_finding_01~05.png                         ⭐ (v4.1.1 카드 5종)
```

### 🧭 독자별 추천 읽기 순서

**비전문가 투자자**:
```
1. report_final.md (본 문서) — 30분
2. docs/Step1_해설.md → Step10_해설.md — 각 10~15분
3. 노트북은 관심 있는 부분만 참조
```

**퀀트 엔지니어**:
```
1. report_final.md (본 문서) — 개요
2. decision_log.md + decision_log_v31.md — 설계 근거
3. 노트북 순차 실행 및 코드 리뷰
4. report_v3.md, report_v4.md — 상세 수치
```

**연구자**:
```
1. report_v3.md Section 2~3 (데이터·피처)
2. report_v4.md (방법론 개선)
3. decision_log_v31.md Section 13 (경로 2 실패 사후분석)
4. Step10_해설.md (통계 방법론)
```

### 📚 외부 참고 문헌

| 분야 | 참고 |
|------|------|
| 포트폴리오 이론 | Markowitz (1952), DeMiguel et al. (2009) |
| 공분산 추정 | Ledoit & Wolf (2004) |
| 통계 검정 | Benjamini & Hochberg (1995), Kass & Raftery (1995) |
| 정보 비율 | Grinold & Kahn (2000) |
| HMM | Hamilton (1994) |
| 경보 시스템 | Sahm (2019) Rule |

---

## 📞 문의

본 보고서는 **개인 투자 프로젝트의 최종 산출물**이며, 실제 투자 추천이 아닙니다.
- 프로젝트 저자: 김재천
- 문의 및 토론: `decision_log_v31.md` 참조

## 🔄 변경 이력

| 일자 | 버전 | 변경 |
|------|------|------|
| 2026-04-17 | v4.1 final | 전 10단계 통합 보고서 최초 작성 |

---

**🎓 이 프로젝트의 교훈 (한 줄 요약)**:
> "대안데이터의 가치는 '언제 위험을 줄일지' 알려주는 경보 시스템(경로 1)에 있으며, 복잡한 레짐 기반 공분산 전환(경로 2)이 아니다."
