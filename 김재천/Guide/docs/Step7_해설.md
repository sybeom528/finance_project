# Step 7 해설 — 동적 리밸런싱 + Ablation Study (v3 / v4.2b)

> **독자 대상**: 비전문가 투자자
> **관련 파일**: [`Step7_Dynamic_Rebalancing.ipynb`](../Step7_Dynamic_Rebalancing.ipynb)
> **버전**: v4.2b (2026-04-19, **alert_lag=1 + Block bootstrap + EW 비용**) / v3 (EW baseline)
> **역할**: 경보 효과 격리 검증 (Step 9 M1이 MV 기반 통합 검증 담당)

## ⚠️ v4.2b 주요 업데이트

1. **이슈 1 — Alert 1일 lag 적용**: `shift(1)`로 look-ahead 제거
2. **이슈 2 — EW 벤치마크 비용 차감**: 공정 비교를 위해 EW도 15bps 적용
3. **이슈 5 — Block Bootstrap 추가**: IID와 병행, autocorr 보존
4. **이슈 4 — Config D 해설**: precision vs timing 트레이드오프 명시

### Config D 성과 역전
- v4.1(same-day alert): Config D 최저 성과
- v4.2b(1일 lag): Config D가 오히려 **개선** (lag와 디바운싱 지연이 상쇄)

## 🎯 TL;DR

- **Step 6의 경보를 활용한 동적 리밸런싱 구현** (경로 1 격리 검증)
- **16개 시뮬레이션** (4 성향 × 4 Config) + 2 벤치마크 = 18 전략
- **v3 최우수**: 보수형_ALERT_B (Sharpe 1.473, EW baseline, 일별 lag 없음)
- **v4.2b 최우수**: 적극형_ALERT_C (Sharpe 0.960, alert_lag=1 적용)
- **Step 9 (MV baseline)**: M1_보수형_ALERT_B (v4.1 1.064) → M2_보수형_ALERT_A (v4.2b 0.847)

### 📊 세 버전 Sharpe 비교 (보수형_ALERT_B)

| 버전 | Sharpe | 차이 원인 |
|------|--------|----------|
| v3 (EW + same-day) | 1.473 | EW baseline + look-ahead |
| v4.1 (MV + same-day) | 1.064 | MV baseline + look-ahead |
| **v4.2b (MV + shift 1일)** | **0.644** ⭐ | 모두 제거 (실무 기준) |

---

## 📑 목차

1. [배경과 목적](#1-배경과-목적)
2. [사전 지식](#2-사전-지식)
3. [진행 과정](#3-진행-과정)
4. [주요 개념](#4-주요-개념)
5. [판단 과정](#5-판단-과정)
6. [실행 방법](#6-실행-방법)
7. [결과 해석 (v3 시점)](#7-결과-해석)
8. [v3 vs v4.1 비교](#8-v3-vs-v41-비교)
9. [FAQ](#9-faq)
10. [관련 파일](#10-관련-파일)

---

## 1. 배경과 목적

### 🎯 Step 6의 경보를 "실제 성과"로 전환

Step 6에서 경보 시스템을 설계했으나:
- "경보가 울리면 무엇을 할 것인가?" 미해결

### 🎯 Step 7의 목표

> **"경보 발동 시 자동으로 주식 축소 + 안전자산 이동 → 실제 Sharpe 개선 효과 측정"**

---

## 2. 사전 지식

### 📚 용어 사전

| 용어 | 쉬운 설명 |
|------|---------|
| **Equal Weight (EW)** | 30개 자산 균등 배분 (1/30씩) |
| **동적 리밸런싱** | 시장 상황에 따라 비중 변경 |
| **EQUITY_CUT** | 경보 레벨별 주식 감축 비율 |
| **Ablation Study** | "하나씩 제외하면서 효과 측정"하는 실험 설계 |
| **Bootstrap** | 통계적 유의성 검정 방법 |

---

## 3. 진행 과정

### 🔧 단계별 흐름

```
1. Baseline 설정: EW (30자산 × 1/30)

2. 매일 동작:
   - Config 경보 레벨 확인
   - 성향별 주식 축소율 (EQUITY_CUT) 적용
   - 감축분 → 채권 70% + 금 30% 재배분
   - 거래비용 편도 15bps 차감

3. 16개 조합 실행 (4성향 × 4Config)

4. 2 벤치마크 비교 (EW, SPY)

5. Bootstrap 통계 검정
   - Config A vs B/C/D 비교
   - 95% CI 0 배제 → 유의
```

---

## 4. 주요 개념

### 🎓 개념 1: 성향별 주식 감축 테이블

| 프로파일 | L0 | L1 | L2 | L3 |
|---------|-----|-----|-----|-----|
| 보수형 | 0% | 15% | 35% | 60% |
| 중립형 | 0% | 10% | 25% | 50% |
| 적극형 | 0% | 5% | 15% | 35% |
| 공격형 | 0% | 0% | 10% | 25% |

**해석**:
- 보수형 L3 발동 시 주식 60% 감축 → **채권 70% + 금 30%** 로 이동
- 공격형은 심지어 L3에서도 25%만 감축 (위험 선호 유지)

### 🎓 개념 2: Ablation Study 방법론

**일상 비유 - 요리 실험**:
- 기본 김치찌개 (EW) vs 김치찌개 + 고기 (EW + 경보)
- 두 그룹의 맛(Sharpe) 차이로 고기 효과 측정

**우리 설계**:
- 4 성향 × 4 Config = 16개 결과
- 각각 EW(경보 無)와 비교
- Bootstrap으로 통계적 유의성 검증

### 🎓 개념 3: Bootstrap Sharpe 차이 검정

**원리**:
```
1. 일별 수익률 시계열을 5,000번 복원 추출
2. 각 샘플에서 Sharpe 차이 계산
3. 분포의 2.5~97.5% 지점이 95% 신뢰구간
4. 0을 포함하지 않으면 "유의한 차이"
```

---

## 5. 판단 과정

### 🤔 주요 결정 사항

#### 결정 1: EW를 baseline으로 선택

**v3 당시 이유**:
- "경보의 순수 효과"만 측정 의도
- MV 최적화 효과와 경보 효과 분리 불가능

**한계 (v4에서 드러남)**:
- MV 최적화 가치 미반영
- 실전 기관 투자자는 MV 사용
- → Step 9에서 MV base로 재수행

#### 결정 2: 감축 70/30 (채권/금) 비율

**근거**:
- 채권: 주식과 음의 상관 (헤지)
- 금: 인플레이션·위기 대응
- 70:30은 역사적 안전자산 비율 경험칙

#### 결정 3: Bonferroni 보정 미적용

**v3 한계**: 4 Config × 2~4 성향 = 8~16 비교, 다중 비교 오류 가능
**v4.1 개선**: Step 10에서 Bonferroni + FDR 적용

---

## 6. 실행 방법

### 🔌 입출력

**입력**:
```
data/portfolio_prices.csv
data/profiles.csv
data/alert_signals.csv (Step 6)
```

**출력**:
```
data/step7_results.pkl
images/step7_01~03_*.png
```

### ⏱️ 실행 시간

**약 3~5분**

---

## 7. 결과 해석 (v3 시점)

### 📊 v3 전체 순위 Top 5

| 순위 | 전략 | 누적수익 | 연율수익 | Sharpe | MDD |
|------|------|--------|--------|-------|------|
| **1** | **보수형_ALERT_B** | +286.1% | 15.16% | **1.473** | -16.89% |
| 2 | 보수형_ALERT_C | +241.1% | 13.79% | 1.376 | -15.66% |
| 3 | 보수형_ALERT_A | +263.1% | 14.53% | 1.364 | -16.56% |
| 4 | 중립형_ALERT_B | +265.7% | 14.64% | 1.338 | -18.43% |
| 5 | 중립형_ALERT_C | - | - | 1.310 | - |

### 📊 v3 벤치마크 비교

| 벤치마크 | Sharpe |
|---------|-------|
| EW 1/30 | 0.925 |
| SPY 100% | 0.838 |

**v3 핵심 발견**:
- **모든 동적 전략이 벤치마크 초과** (대안데이터 가치 입증)
- **보수형 상위권 독점** (Sharpe 기준)
- **Config B 약간 우세** (VIX + Contango 조합)

### 📊 v3 Bootstrap 검정

| 비교 | 보수형 | 중립형 |
|------|-------|-------|
| B vs A | 중앙값 +0.11, **유의** | 중앙값 +0.08, **유의** |
| D vs A | -0.19, **유의 악화** | -0.14, **유의 악화** |

**해석**: VIX_contango 추가(B)는 구조적 개선. 디바운스(D)는 **과도한 지연으로 역효과**.

### 📈 v3 시각화 3종

| PNG | 내용 |
|-----|------|
| step7_01_cumulative | 18 전략 누적수익률 |
| step7_02_sharpe_comparison | 성향·Config Sharpe 바차트 |
| step7_03_drawdown | 주요 전략 Drawdown 비교 |

---

## 8. v3 vs v4.1 비교

### 🔄 왜 결과가 변경됐나?

**v3 Step 7** (본 문서):
- Baseline: Equal Weight 1/30
- Sharpe 1.473 (보수형_ALERT_B)

**v4.1 Step 9** (업데이트):
- Baseline: **MV 최적화** (Step 4 재현)
- Sharpe **1.064** (M1_보수형_ALERT_B)

### 📊 차이 원인 분해

| 요인 | 효과 |
|------|------|
| **EW → MV 전환** | Baseline이 이미 최적화된 상태 → 경보 효과 margin 축소 |
| 60/40 벤치마크 비용 정교화 | 절대 Sharpe 기준 재조정 |
| 경로 2 추가 구현 | M2/M3 추가 (M1과 별개) |
| 통계 검정 엄격화 | Bonferroni + FDR로 유의성 재평가 |

### 🎯 어느 것을 믿어야 하나?

**v4.1 Step 9**이 더 엄격한 조건:
- MV baseline (실전 기관 표준)
- 거래비용 정교화
- 경로 2 통합 검증

**v3 Step 7**은 "개념 증명" 단계:
- EW는 교육용 baseline
- 실전 운용에는 MV 사용 권장

**결론**: **v4.1 결과(Sharpe 1.064)를 실전 운용 기준으로 사용**

---

## 9. FAQ

### ❓ Q1. Step 7과 Step 9 중 어떤 걸 봐야 하나요?

**A**: **Step 9 (v4.1)**. Step 7은 EW baseline이라 실전 기준으로 부적합. Step 7은 역사적 기록으로만.

### ❓ Q2. 왜 Config D(디바운스)가 v3에서 오히려 나쁘나요?

**A**: 5일 지연이 위기 초기 대응 놓침. 2020 COVID 같은 급변 시 3~5일이면 이미 큰 손실.

### ❓ Q3. EW가 SPY보다 Sharpe가 높은 이유?

**A**: 분산투자 효과. SPY는 주식 100%, EW는 30자산 분산 (채권·금 포함) → 변동성 ↓.

### ❓ Q4. v3 Step 7과 Step 9의 결과가 다른데 어떻게 신뢰하나요?

**A**: 실험 조건이 다를 뿐, 양쪽 모두 **경로 1의 가치**는 일관되게 입증:
- v3: Config B 효과 +0.11 Sharpe
- v4.1: Config B 효과 +0.17 Sharpe

### ❓ Q5. 왜 Step 7에서 경로 2를 구현하지 않았나요?

**A**: v3 decision_log에서 설계만 명시, 구현은 Step 8/9로 연기. v4.1에서 구현 시도 → 무효 확인.

---

## 10. 관련 파일

```
Guide/
├── Step7_Dynamic_Rebalancing.ipynb (본 해설 대상)
├── data/step7_results.pkl
└── images/step7_01~03_*.png
```

**다음**: [Step8_해설.md](Step8_해설.md) — v4.1 경로 2 구현 시작 (Regime-Aware Covariance)

### 📚 외부 참고

- Maillard, S., Roncalli, T. & Teiletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management*
- Dichtl, H., Drobetz, W. & Wambach, M. (2014). "Testing Rebalancing Strategies." *Journal of Asset Management*

---

## 🔄 변경 이력

| 일자 | 내용 |
|------|------|
| 2026-04-17 | 최초 작성 (v4.1 관점에서 Step 7 회고) |
