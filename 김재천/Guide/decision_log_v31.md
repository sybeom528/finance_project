# Decision Log v3.1 → v4.1 — Step8~10 확장 설계 기록

> **버전**: v4.1 (경로 2 재설계 + 실패 사례 반영, Step8/9/10 전면 재실행)
> **최초 작성**: 2026-04-17
> **v4.1 업데이트**: 2026-04-17 (피드백 반영 대폭 수정)
> **전제 문서**: `decision_log.md` (v3), `report_v3.md`
> **대상 파일**: `Step8_Regime_Covariance.ipynb`, `Step9_Integrated_Backtest.ipynb`, `Step10_Ablation_Final.ipynb`

> ⚠️ **§4 (레짐 수 결정) 및 Appendix A (BIC 계산)는 [`decision_log_v32.md`](decision_log_v32.md)에서 갱신됨 (2026-04-19).**
> - v4.2에서 BIC 공식 수정 + n 탐색 [2~8] 확장 + 다기준 scorecard 도입
> - 최종 결과는 **n=4 유지** (v4.1과 동일)이나 선택 근거가 엄밀화됨
> - 본 문서(v31) 하위 섹션들(§5~§14)은 유효함

## 🚨 v4.1 핵심 변경 사항

### 실증적 결론
- **경로 2(레짐 Σ 전환)는 재설계 후에도 무효 + 역효과 확인**
- **M3 평균 Sharpe < M1 평균 Sharpe** (통합이 경로 1 단독보다 나쁨)
- **최종 추천 전략 변경: M3_보수형_ALERT_C → M1_보수형_ALERT_B**

### 방법론 개선
- Cohen's d → **IR (Information Ratio) + ΔSR 실무 기준**으로 교체
- M3 vs M1 **직접 Bootstrap 검정** 추가 (48 → 64 비교)
- 경로 2 Σ 선택: OOS 시작일 1회 → **OOS 월 단위 전환** (재설계 시도)
- BENCH_60_40 비용 정교화 (drift 기반 실측 turnover)

### 문서 정정
- 윈도우 수 32 → **31** (전체 문서 일괄 수정)
- "경로 1 점진 복원" → **"즉시 반영"** (실제 코드 동작 반영)
- 가상 스트레스 시나리오 컬럼명 정정 ("M3 예상" → "정적배분 근사")

이 문서는 기존 v3 파이프라인(Step1~7)에 Step8~10을 추가하는 과정에서 **설계 결정과 그 근거를 시간순으로 기록**한 문서입니다. 의사결정의 흐름, 고려된 대안, 채택된 방향의 이유를 추적하여 재현성과 유지보수성을 확보합니다.

---

## 목차

- [1. 확장의 배경 및 필요성](#1-확장의-배경-및-필요성)
- [2. 방법론적 기초 (Deployment Simulation vs Strict OOS)](#2-방법론적-기초)
- [3. HMM 레짐 학습 전략 결정](#3-hmm-레짐-학습-전략-결정)
- [4. 레짐 수 결정 (BIC 실측)](#4-레짐-수-결정)
- [5. 공분산 그룹화 전략](#5-공분산-그룹화-전략)
- [6. 관측수 판정 및 Fallback 로직](#6-관측수-판정-및-fallback-로직)
- [7. Step8 설계](#7-step8-설계)
- [8. Step9 설계](#8-step9-설계)
- [9. Step10 설계](#9-step10-설계)
- [10. 매매비용 관리 방침](#10-매매비용-관리-방침)
- [11. 문서화 계획](#11-문서화-계획)
- [12. 향후 과제](#12-향후-과제)
- [Appendix A. BIC 계산 세부](#appendix-a-bic-계산-세부)
- [Appendix B. 레짐 특성 요약](#appendix-b-레짐-특성-요약)

---

## 1. 확장의 배경 및 필요성

### 1.1 기존 v3 파이프라인의 구조적 문제점

프로젝트 검토 과정에서 `decision_log.md`와 실제 구현 간 **중대한 불일치** 가 확인되었습니다.

| 설계 원칙 (decision_log v3) | 실제 구현 (Step4, Step7) |
|----------------------------|------------------------|
| 경로 1: 경보 → 주식 축소 | ✅ Step7 구현 완료 |
| 경로 2: HMM 레짐 → Σ_crisis 전환 → 재최적화 | ❌ **전면 미구현** |
| Step7의 baseline은 Final Project MV 최적화 | ❌ **Equal Weight 1/30으로 단순화됨** |
| 계층형 공분산 (Level 1 블록 대각 + Level 2 PCA) | ❌ 30자산 단일 최적화로 축소 |
| 스트레스 시나리오 11개 | ⚠️ 일부만 Step5에 구현 |
| Bonferroni + Cohen's d 통계 검정 | ❌ Bootstrap 95% CI만 구현 |

### 1.2 통합 목표

1. **경로 2 완전 구현** — HMM 레짐 기반 Σ 전환 메커니즘 추가
2. **MV baseline 복원** — Step4의 MV 최적화를 Step9의 기준선으로 사용
3. **경로 기여도 분해** — 경로 1/경로 2/상호작용을 개별 측정
4. **엄밀한 통계 검정** — Bonferroni, FDR, Cohen's d 적용
5. **위기 강건성 검증** — 11개 스트레스 시나리오 성과 비교

---

## 2. 방법론적 기초

### 2.1 백테스트 철학: Deployment Simulation vs Strict OOS

사용자와의 논의 중 **두 가지 백테스트 철학**의 구분이 명확해졌습니다.

| 철학 | 정의 | 예시 | 채택 여부 |
|------|------|------|----------|
| **Strict OOS Backtest** | "과거 시점 투자자가 실제로 볼 수 있었던 정보만 사용" | 윈도우별 HMM 재훈련 | ❌ |
| **Deployment Simulation** | "지금 이 모델을 배포하면 과거에는 어떻게 작동했을까" | Full-sample HMM 고정 | ✅ 채택 |

### 2.2 채택 근거

사용자 언급:
> "10년 데이터로 먼저 확인하고, 앞으로 1년은 그 학습 결과를 통해 레짐을 분류하여 펀드를 운용한다. 동일한 레짐 분류 기준을 지난 10년에 똑같이 적용했을 때 어떤지 백테스팅하는 목적이니까 문제 없지 않나?"

**방법론적 정당성**:
1. **실전 운용 관점**: 기관 투자자는 일반적으로 모델을 정기(1년) 재학습, 그 사이 기간은 고정 모델 사용. 이 시뮬레이션이 더 현실적.
2. **데이터 가용성**: 2,609영업일 중 Crisis 관측이 100일 수준이므로, WF 24개월 IS에서는 Crisis 관측이 5~15일로 급격히 감소하여 HMM 학습 불가능.
3. **정보 레이어 분리**: HMM 레짐을 "고정 렌즈(fixed lens)"로 취급하되, 실제 의사결정(μ, Σ 추정, 최적화)은 IS 데이터로만 수행하므로 진정한 look-ahead는 회피됨.

### 2.3 Trade-off 명시

**Deployment Simulation의 한계**:
- Academic peer review에서는 "look-ahead bias"로 지적될 수 있음
- 레짐 라벨 생성 시 미래 데이터가 간접적으로 영향
- 완화: Step10에 **Expanding Annual Refresh 민감도 분석** 포함하여 robustness 검증

---

## 3. HMM 레짐 학습 전략 결정

### 3.1 비교된 3가지 대안

| 전략 | 학습 데이터 | Look-Ahead | 실전 부합성 | 관측수 | 채택 |
|------|-----------|-----------|-----------|--------|------|
| A. Full-sample | 2016~2025 전체 | ⚠️ 간접 존재 | 높음 | 충분 | ✅ **기본** |
| B. Rolling 24m WF | 각 IS 24개월 | ❌ 없음 | 낮음 | 심각하게 부족 | ❌ |
| C. Expanding Annual Refresh | 연초까지 누적 | ❌ 없음 | 높음 | 후기만 충분 | ⚠️ **민감도 분석용** |

### 3.2 결정: A안 (Full-sample) 채택 + C안 병행 검증

**이유**:
- A안의 간접 look-ahead를 C안과의 비교로 정량화 가능
- 두 방식의 Sharpe 차이가 작으면 A안 정당성 강화
- Step10-7에 Full vs Annual 비교 섹션 포함

---

## 4. 레짐 수 결정

### 4.1 사용자 제안: 2레짐으로 통일 (Σ와 대칭)

사용자 Q4: "Σ를 2가지로 나눌 거라면 레짐도 2가지로 하는 것이 좋나?"

논리는 타당하나 **데이터 기반 검증 필요**.

### 4.2 BIC 실측 비교 (2016-2025, 5 features, diag covariance)

| n | log-likelihood | 파라미터 수 | BIC | AIC | 레짐 분포 |
|---|--------------|----------|-----|-----|----------|
| 2 | -4,839 | 23 | 9,856 | 9,723 | {989, 1339} |
| 3 | -4,854 | 38 | 10,002 | 9,783 | {687, 687, 954} |
| **4** | **-2,406** | **55** | **5,239** ⭐ | **4,922** | **{449, 746, 337, 796}** |
| 5 | -3,040 | 74 | 6,653 | 6,228 | {569, 568, 488, 536, 167} |

**결과**: BIC(n=2) - BIC(n=4) = **+4,617** → Kass & Raftery (1995) 기준 "매우 강한 증거" (>10)

### 4.3 결정: n=4 레짐 유지

**이유**:
- BIC가 압도적으로 4레짐을 지지 (데이터의 목소리 존중)
- Σ 그룹화는 별도 문제 → 4레짐을 2그룹으로 묶는 것은 허용
- 사용자의 "대칭성" 요구는 Σ 레벨에서 유지

---

## 5. 공분산 그룹화 전략

### 5.1 설계 결정: 4레짐 → 2그룹

| 레짐 분류 | Σ 그룹 | 경보 레벨 대응 |
|---------|-------|-------------|
| 레짐 0, 1 (VIX 평균 낮음) | **Σ_stable** | L0 (정상), L1 (주의) |
| 레짐 2, 3 (VIX 평균 높음) | **Σ_crisis** | L2 (경계), L3 (위기) |

### 5.2 그룹화 기준: VIX 평균 자동 분류

**배경**: HMM의 레짐 번호는 **임의(arbitrary)**. 레짐 0이 반드시 "저변동"은 아님.

**자동 분류 로직**:
```python
# 각 레짐의 Full-sample VIX 평균 계산
regime_stats = {r: regime_df[regime_df['hmm_regime']==r]['VIX_level'].mean()
                for r in [0, 1, 2, 3]}

# VIX 평균 오름차순 정렬
sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1])

# 하위 2개 → Stable, 상위 2개 → Crisis
stable_group = [r for r, v in sorted_regimes[:2]]
crisis_group = [r for r, v in sorted_regimes[2:]]
```

**실측 결과 (2016-2025 Full-sample)**:

| 레짐 | VIX 평균 | HY 평균 | 비중 | 그룹 |
|------|---------|--------|------|------|
| 0 | 12.33 | 3.63% | 20.2% | Stable |
| 1 | 19.19 | 3.17% | 30.2% | Stable |
| 2 | 19.93 | 4.34% | 31.3% | Crisis |
| 3 | 22.26 | 4.48% | 18.3% | Crisis |

### 5.3 추정은 HMM 레짐 기반 / 선택은 경보 레벨 기반

**핵심 이중 구조**:
```
공분산 추정 시점:
  IS 데이터를 HMM 레짐 라벨에 따라 분할
  → Σ_stable = LedoitWolf(IS[regime ∈ {0,1}])
  → Σ_crisis = LedoitWolf(IS[regime ∈ {2,3}])

운용 시점 (OOS):
  현재 경보 레벨에 따라 선택
  → alert ∈ {0,1}: Σ_used = Σ_stable
  → alert ∈ {2,3}: Σ_used = Σ_crisis
```

---

## 6. 관측수 판정 및 Fallback 로직

### 6.1 문제 설정

공분산 행렬 30×30 추정을 위한 최소 관측수:
- 수학적 최소: N > 30 (non-singular)
- 실무 권장: N > 60 (Ledoit-Wolf 수축 시)
- 이상적: N > 90 (수축 계수 안정)

### 6.2 대칭 4단계 Fallback (v3.1 개선)

**원본 v3의 비대칭 로직(Crisis만 체크) 한계**:
- Step8 실제 실행에서 윈도우 12~13 (Stable 30일, Crisis 466일)이 부당하게 single로 분류
- Stable도 부족할 수 있는 상황을 반영 못함

**개선된 대칭 로직**:

```python
# 각 그룹의 충분성을 독립적으로 판정
has_stable = N_stable >= 48
has_crisis = N_crisis >= 48

if has_stable and has_crisis:
    # 1단계: 둘 다 충분 → 정상 분리 추정
    Σ_stable = LedoitWolf(returns_stable).covariance
    Σ_crisis = LedoitWolf(returns_crisis).covariance
    fallback_type = 'separate'

elif has_stable:
    # 2단계: Stable만 충분 → Crisis 프록시
    Σ_stable = LedoitWolf(returns_stable).covariance
    Σ_crisis = Σ_stable × 1.5
    fallback_type = 'scaled'

elif has_crisis:
    # 2'단계: Crisis만 충분 → Stable 역방향 프록시
    Σ_crisis = LedoitWolf(returns_crisis).covariance
    Σ_stable = Σ_crisis / 1.5
    fallback_type = 'scaled_reverse'

else:
    # 3단계: 둘 다 부족 → 단일 Σ (경로 2 비활성)
    Σ_stable = Σ_crisis = LedoitWolf(returns_all).covariance
    fallback_type = 'single'
```

**변경점 요약**:
| 항목 | v3 (원본) | v3.1 (수정) |
|------|---------|----------|
| 임계값 | MIN_SEPARATE=48, MIN_SCALED=20 | **MIN_SEPARATE=48만** |
| 판정 기준 | Crisis 관측수만 | **Stable, Crisis 각각** |
| Fallback 타입 | separate, scaled, single (3개) | separate, scaled, scaled_reverse, single (4개) |
| 대칭성 | 비대칭 (Crisis 중심) | **완전 대칭** |

### 6.3 스케일링 1.5배의 근거

역사적 관찰:
- 2008 글로벌 금융위기: S&P 500 σ가 일반 시기의 1.4~1.7배
- 2020 COVID: σ 1.5~2.0배
- 평균: **약 1.5배** → 보수적 대용치

### 6.4 실측 Fallback 분포 및 Step10 분석 계획

#### 6.4.1 실측된 Fallback 분포 (31 WF 윈도우, v3.1 대칭 로직 적용)

| Fallback | 개수 | 비중 | 발동 윈도우 |
|---------|------|------|----------|
| separate | 22 | 71.0% | 윈도우 5~11, 14~24, 28~31 (정상 분리) |
| scaled | 4 | 12.9% | 윈도우 1~4 (초기, Stable 풍부·Crisis 0) |
| scaled_reverse | 5 | 16.1% | 윈도우 12~13, 25~27 (Stable 부족·Crisis 풍부) |
| single | **0** | **0%** | 해당 없음 |

**v3.0 원본 로직 대비 개선점**:
- 기존: single 7개 (경로 2 비활성 구간 다수)
- 수정 후: single 0개 → **모든 윈도우에서 경로 2 활성화**

#### 6.4.2 각 Fallback 타입의 경로 2 동작

| Fallback | M0 | M1 | M2 | M3 | 경로 2 활성 |
|---------|-----|-----|-----|-----|----------|
| separate | 정상 | 정상 | **경보 따라 Σ 전환** | **M1+M2 통합** | ✅ 완전 |
| scaled | 정상 | 정상 | Σ_crisis=Σ_stable×1.5 전환 | M1+M2 통합 | ⚠️ 프록시 |
| scaled_reverse | 정상 | 정상 | Σ_stable=Σ_crisis/1.5 전환 | M1+M2 통합 | ⚠️ 역방향 프록시 |
| single | 정상 | 정상 | M0과 동일 | M1과 동일 | ❌ 비활성 |

v3.1 대칭 로직 적용 결과 single이 0이므로, **모든 윈도우에서 경로 2의 효과를 최소한 프록시 수준에서는 측정 가능**.

#### 6.4.3 Step10 서브섹션: Fallback 타입별 성과 비교

Single Fallback이 발동되지 않으므로, 당초 계획한 "삼중 분석(전체/서브샘플/초기)"은 불필요. 대신 **Fallback 타입별 4구간 분석**으로 변경:

| 분석 | 샘플 | 측정 목적 |
|------|------|---------|
| **전체 분석** | 31 윈도우 전부 | 실제 배포 시 예상 성과 |
| **separate 구간** | 22 윈도우 | 경로 2의 순수 효과 (독립 추정 Σ) |
| **scaled 구간** | 4 윈도우 | 초기 구간, Stable × 1.5 프록시 효과 |
| **scaled_reverse 구간** | 5 윈도우 | Stable 부족 시 Crisis / 1.5 역프록시 효과 |

#### 6.4.4 Step10 서브섹션 최종 구성 (9개)

- 10-1. 경로 기여도 분해
- 10-2. Bootstrap Sharpe 차이 검정
- 10-3. Bonferroni + FDR 보정
- 10-4. Cohen's d 효과 크기
- 10-5. 스트레스 시나리오 11개
- 10-6. 64 cell Ablation Heatmap
- 10-7. HMM 학습 방식 민감도 분석 (Full-sample vs Expanding Annual)
- 10-8. **Fallback 타입별 성과 비교** (전체/separate/scaled/scaled_reverse)
- 10-9. 최종 추천 전략 (Multi-criteria Decision)

---

## 7. Step8 설계

### 7.1 목적

Walk-Forward 32개 윈도우 각각에 대해 **Σ_stable, Σ_crisis를 IS 데이터 + Full-sample 레짐 라벨로 추정**하여 `.pkl`로 저장. Step9에서 로드하여 사용.

### 7.2 셀 구성 (18개 셀)

| # | 유형 | 내용 |
|---|------|------|
| 0 | MD | 헤더 + 목적 + 목차 |
| 1 | Code | Imports + 한글 폰트 |
| 2 | Code | 데이터 로드 |
| 3 | MD | 8-1. Full-sample HMM 검증 |
| 4 | Code | 4레짐 특성 분석 (VIX/HY 평균) |
| 5 | Code | VIX 기준 Stable/Crisis 자동 그룹화 |
| 6 | Code | 레짐 타임라인 시각화 |
| 7 | MD | 8-2. WF 윈도우 생성 |
| 8 | Code | 32개 윈도우 생성 (IS 24m, OOS 3m) |
| 9 | MD | 8-3. Σ 추정 + Fallback |
| 10 | Code | `estimate_regime_cov()` 함수 |
| 11 | Code | 32개 윈도우 반복 실행 |
| 12 | MD | 8-4. 결과 검증 |
| 13 | Code | Fallback 타입 분포 시각화 |
| 14 | Code | Σ 차이 히트맵 |
| 15 | MD | 8-5. (확장) 4개 Σ 별도 추정 |
| 16 | Code | 4개 Σ 버전 병행 생성 |
| 17 | MD | 8-6. 산출물 저장 |
| 18 | Code | pkl 저장 + 요약 |

### 7.3 산출물

- `data/regime_covariance_by_window.pkl` — 32개 윈도우의 {μ, Σ_stable, Σ_crisis, fallback_type}
- `data/regime_covariance_4group.pkl` — 4개 Σ 버전 (확장 실험용)
- `images/step8_01_regime_timeline.png`
- `images/step8_02_fallback_log.png`
- `images/step8_03_cov_diff.png`

---

## 8. Step9 설계

### 8.1 목적

**경로 1 + 경로 2를 MV baseline 위에 통합**한 Walk-Forward 동적 백테스트. 4개 모드로 경로별 효과 분해 가능한 구조 확보.

### 8.2 4개 모드 정의

| 모드 | Baseline | 경로 1 (일별 경보) | 경로 2 (레짐 Σ) | 의미 |
|------|---------|------------------|----------------|-----|
| **M0** | MV (Step4 재현) | ❌ | ❌ | 순수 MV baseline |
| **M1** | MV | ✅ | ❌ | 경로 1만 추가 |
| **M2** | MV | ❌ | ✅ | 경로 2만 추가 |
| **M3** | MV | ✅ | ✅ | **통합 (최종 제안)** |

### 8.3 핵심 루프 로직

```python
for each window_k (32회):
    # Step8 산출물 로드
    μ, Σ_stable, Σ_crisis = load_window_covariance(window_k)
    
    # 1) OOS 시작일 경보 확인 → Σ 선택 (경로 2)
    if M ∈ {M2, M3}:
        initial_alert = alert_series[oos_start]
        Σ_used = Σ_crisis if initial_alert >= 2 else Σ_stable
    else:
        Σ_used = Σ_stable  # M0, M1은 항상 stable
    
    # 2) MV 최적화 (성향 제약 반영)
    weights_base = optimize_mv_weights(μ, Σ_used, profile, constraints)
    
    # 3) OOS 3개월 보유 (일별 경로 1)
    weights_current = weights_base.copy()
    for day in oos_dates:
        alert_today = alert_series[day]
        
        if M ∈ {M1, M3}:
            cut = EQUITY_CUT[profile][alert_today]
            if cut > 0:
                weights_current = apply_equity_cut(weights_base, cut)
            else:
                weights_current = gradual_restore(weights_current, weights_base)
        
        daily_return = (weights_current * asset_returns[day]).sum() - cost[day]
```

### 8.4 64개 시뮬레이션

- **4 성향** × **4 Config** × **4 모드** = 64 시뮬레이션
- + 3 벤치마크 (EW, SPY 100%, 60/40)
- = 총 **67개 전략**

### 8.5 셀 구성 (24개 셀)

| # | 유형 | 내용 |
|---|------|------|
| 0 | MD | 헤더 + 4모드 설명 |
| 1-3 | Code | Imports + 데이터 로드 + 공통 설정 |
| 4 | MD | 9-1. 4모드 함수 정의 |
| 5-9 | Code | optimize_mv_weights, simulate_M0~M3 |
| 10-11 | Code | 64개 시뮬레이션 실행 루프 |
| 12-13 | Code | 벤치마크 3개 |
| 14-16 | Code | 성과 지표 계산 |
| 17-21 | Code | 4개 주요 시각화 |
| 22-23 | Code | pkl 저장 |

### 8.6 산출물

- `data/step9_backtest_results.pkl` — 67개 전략의 일별 수익률
- `data/step9_metrics.csv` — 성과 지표 요약
- `images/step9_01_cumulative.png`
- `images/step9_02_drawdown.png`
- `images/step9_03_mode_comparison.png`
- `images/step9_04_regime_overlay.png`

---

## 9. Step10 설계

### 9.1 목적

Step9 결과에 대한 **엄밀한 통계 검정 + 경로 기여도 분해 + 스트레스 검증 + 최종 추천**.

### 9.2 8개 서브섹션

#### 10-1. 경로 기여도 분해 (Modular Decomposition)

```
ΔSharpe(M1) = Sharpe_M1 - Sharpe_M0  ← 경로 1 순수 기여
ΔSharpe(M2) = Sharpe_M2 - Sharpe_M0  ← 경로 2 순수 기여
ΔSharpe(M3) = Sharpe_M3 - Sharpe_M0  ← 통합 총 효과
상호작용 = ΔSharpe(M3) - ΔSharpe(M1) - ΔSharpe(M2)
```

각 성향·Config 조합에서 stacked bar로 시각화.

#### 10-2. Bootstrap Sharpe 차이 검정

- 5,000회 복원 추출
- 각 (profile, config)에서 3개 비교 (M1/M2/M3 vs M0)
- 95% CI 계산

#### 10-3. Bonferroni + FDR 다중 비교 보정

- **Bonferroni**: α' = 0.05 / K = 0.0167 (보수적)
- **FDR (Benjamini-Hochberg)**: 거짓 발견율 통제
- 두 방법 비교 테이블 제시

#### 10-4. Cohen's d 효과 크기

| d 범위 | 해석 |
|-------|------|
| < 0.2 | 미미 |
| 0.2~0.5 | 작음 |
| 0.5~0.8 | 중간 |
| ≥ 0.8 | 큼 |

#### 10-5. 스트레스 시나리오 11개

**역사적 6개**:
| 사건 | 기간 |
|------|------|
| 2016 브렉시트 | 2016-06-23 ~ 2016-07-31 |
| 2018 Volmageddon | 2018-02-05 ~ 2018-02-09 |
| 2020 COVID | 2020-02-20 ~ 2020-04-30 |
| 2022 긴축 | 2022-01-01 ~ 2022-10-31 |
| 2023 SVB | 2023-03-09 ~ 2023-03-31 |
| 2024 엔캐리 청산 | 2024-08-05 ~ 2024-08-09 |

**가상 5개**: 금리 +50bp, VIX 80, HY +300bp, 스태그플레이션, 이중 충격

#### 10-6. 64 cell Ablation Heatmap

4 subheatmap (M0, M1, M2, M3 각각), 각 4성향 × 4Config 격자. Sharpe 색상 + 유의 표시(★★★).

#### 10-7. HMM 학습 방식 민감도 분석

- Full-sample HMM (기본)
- Expanding Annual Refresh (민감도 체크)
- 두 방식의 Sharpe 차이 < 0.10이면 A안 정당성 강화

#### 10-8. 최종 추천 전략

```
Score = 0.35·Sharpe_rank + 0.25·MDD_rank + 0.20·Calmar_rank + 0.20·Stress_rank
```

Top 5 전략 선정 + 권고안 도출.

### 9.3 셀 구성 (32개 셀)

Step10 셀 구성 표는 [본 MD 파일 상위 셀 구성 섹션 참조](#2-step8-step9-step10-셀-구성-정리).

### 9.4 산출물

- `images/step10_01_contribution.png` ~ `step10_06_hmm_sensitivity.png`
- `data/step10_ablation_metrics.csv`
- `data/step10_recommendations.json`

---

## 10. 매매비용 관리 방침

### 10.1 기본 방침 (Step9)

- 편도 15bps 일괄 적용 (decision_log v3 Section 12 준수)
- 완화 장치 **미적용** (순수 성과 우선 측정)

### 10.2 조건부 적용 판단 (Step10 이후)

Step10 결과에서 **M3 Sharpe가 M0 대비 Bonferroni 보정 후에도 유의한 개선**이면:
- 현재 구조 그대로 권고 → 완화 장치 불필요

**유의 개선이 없거나 마진이 좁으면** 4가지 완화 장치 순차 적용:

1. **최소 유지 기간** (min_hold=3일) — whipsaw 제거
2. **임계값 기반 리밸런싱** (threshold=5%) — 소액 거래 차단
3. **감응도 분석** (cost ∈ {0, 5, 15, 30} bps) — 붕괴점 탐지
4. **경로별 비용 분해** — Path 1 vs Path 2 ROI 비교

---

## 11. 문서화 계획

### 11.1 문서 구조

| 파일 | 역할 | 생성/수정 |
|------|------|---------|
| `decision_log.md` | v3 설계 기록 (기존) | 유지 |
| `decision_log_v31.md` | **v3.1 확장 기록 (본 문서)** | **신규** |
| `report_v3.md` | v3 결과 보고서 (기존) | 유지 |
| `report_v4.md` | v3.1 통합 결과 보고서 | **Step10 완료 후 신규** |
| `stats_model.md` | 통계 기법 요약 (기존) | Step10 통계 기법 추가 |

### 11.2 report_v4.md 구성 계획

- 1. Executive Summary (Step9 주요 수치 + 경로 기여도)
- 2. v3 대비 개선 요약
- 3. Step8 방법론 (Regime-Aware Covariance)
- 4. Step9 결과 (64 시뮬레이션 + 벤치마크)
- 5. Step10 통계 검정 결과
- 6. 스트레스 시나리오 강건성
- 7. 민감도 분석 (HMM 학습 방식)
- 8. 최종 추천 전략
- 9. 한계 및 향후 과제

---

## 12. 향후 과제

### 12.1 단기 (Step8~10 실행 중 도출될 이슈)

- HMM 레짐 그룹화 기준 재검증 (VIX 외 HY, yield curve도 고려할지)
- Fallback "scaled" 발동 윈도우의 성과가 "separate"보다 떨어지는지
- Config C/D 복합 스코어의 감소된 stationarity로 인한 레짐 오분류 가능성

### 12.2 중기 (v3.2 제안)

- **Level 1 + Level 2 계층 최적화 복원**: decision_log v3 원본 설계 완전 구현 (블록 대각 Σ + PCA 팩터 모형)
- **4개 Σ 별도 추정 실험화**: Step10에서 부가 실험으로 포함했으나, 성과 우수 시 기본 채택 검토
- **2006-2015 데이터 확장**: 진정한 10년 rolling HMM 학습 가능
- **자산 유니버스 재선정**: 2016-2025 생존자 편향 검증 (Step9 결과에 따라)

### 12.3 장기 (v4 이후)

- 글로벌 자산 확장 (신흥국 채권, 부동산, 원자재)
- LLM 기반 뉴스 감성 분석을 β 추정의 추가 피처로
- Real-time deployment (paper trading)

---

## Appendix A. BIC 계산 세부

### A.1 수식

$$\text{BIC} = -2 \ln(\hat{L}) + k \cdot \ln(n)$$

- $\hat{L}$: 최대 우도
- $k$: 파라미터 개수
- $n$: 관측 수

### A.2 HMM 파라미터 수 (5 features, diag covariance)

| n_states | π | A | μ | diag(Σ) | 합계 k |
|---------|---|---|---|--------|--------|
| 2 | 1 | 2 | 10 | 10 | 23 |
| 3 | 2 | 6 | 15 | 15 | 38 |
| 4 | 3 | 12 | 20 | 20 | 55 |
| 5 | 4 | 20 | 25 | 25 | 74 |

### A.3 실측 BIC (N=2,328)

| n | log-L | BIC | AIC |
|---|-------|-----|-----|
| 2 | -4,839 | 9,856 | 9,723 |
| 3 | -4,854 | 10,002 | 9,783 |
| **4** | **-2,406** | **5,239** ⭐ | **4,922** |
| 5 | -3,040 | 6,653 | 6,228 |

### A.4 해석 (Kass & Raftery 1995)

| BIC 차이 | 해석 |
|---------|------|
| 0~2 | 거의 무차이 |
| 2~6 | 약한 증거 |
| 6~10 | 강한 증거 |
| >10 | **매우 강한 증거** |

**결과**: BIC(n=2) - BIC(n=4) = **4,617** → 매우 강한 증거로 n=4 우수

---

## Appendix B. 레짐 특성 요약

### B.1 4개 레짐의 Full-sample 특성

| 레짐 | 영업일 | 비중 | VIX 평균 | HY 평균 | 해석 | Σ 그룹 |
|------|--------|------|---------|--------|------|-------|
| 0 | 470 | 20.2% | 12.33 | 3.63% | 저변동 (고요한 시장) | Stable |
| 1 | 702 | 30.2% | 19.19 | 3.17% | 일반 (건전한 신용) | Stable |
| 2 | 729 | 31.3% | 19.93 | 4.34% | 일반 (신용 긴장) | Crisis |
| 3 | 427 | 18.3% | 22.26 | 4.48% | 고변동 (스트레스) | Crisis |

### B.2 Σ 차이 예시 (2018-2019 IS 기준)

**변동성 비교**:
| 자산 | Stable σ | Crisis σ | 비율 |
|------|---------|---------|------|
| SPY | 13.45% | 15.66% | 1.16× |
| AGG | 4.17% | 4.16% | 1.00× |
| GLD | 9.76% | 11.45% | 1.17× |
| TLT | 9.53% | 11.48% | 1.20× |

**상관 비교**:
| 쌍 | Stable ρ | Crisis ρ | 변화 |
|----|---------|---------|------|
| SPY-AGG | -0.06 | -0.19 | 음의 상관 강화 |
| SPY-TLT | -0.17 | -0.41 | flight to quality |
| SPY-GLD | +0.10 | -0.24 | 금의 안전자산화 |

---

## 13. 경로 2 실패 사후분석 (v4.1 추가)

### 13.1 두 차례 시도 모두 실패

| 시도 | Σ 선택 방식 | 결과 | 비고 |
|------|----------|------|------|
| v4.0 초안 | OOS 시작일 단 1회 경보로 Σ 고정 | M2 평균 Sharpe = 0.810 (M0 대비 +0.016) | 효과 미미 |
| **v4.1 재설계** | **OOS 월 단위 Σ 재전환** | **M2 평균 Sharpe = 0.749** | **오히려 악화 (-0.045)** |

### 13.2 실패 원인 분석

**원인 1: Σ_stable과 Σ_crisis의 차이가 MV 최적화에 충분히 반영 안 됨**
- Ledoit-Wolf 수축이 두 공분산을 유사하게 보정
- 30개 자산 분산투자 효과가 이미 공분산 차이를 상쇄

**원인 2: 월 단위 재최적화로 인한 비중 변동성 증가**
- 월마다 MV 최적화 재실행 → weights 변동 ↑
- 누적 거래비용이 경로 2의 이론적 이득 상회
- 경보 노이즈에 과민 반응

**원인 3: 경로 1이 이미 레짐 전환을 포착**
- 경보 레벨(L0~L3) 자체가 레짐의 프록시 역할
- 경로 2 정보가 경로 1과 **중복**

### 13.3 학술적 교훈

1. **이론적 정교함 ≠ 실증적 효과**: Σ_stable/Σ_crisis 분리 추정은 이론적으로 합리적이나 실제 데이터에서 효과 없음
2. **단순성의 가치**: 경로 1의 즉시 반응(일별 주식 축소)이 복잡한 공분산 전환보다 우수
3. **비용 현실성**: 재최적화 빈도가 높을수록 거래비용 부담 증가 → 이득 상쇄

### 13.4 v5 권고

- **경로 2 개념 자체 폐기 검토**
- 대안: **레짐 조건부 자산 배분 제약**(예: Crisis 시 max_equity 동적 축소)
- 또는 **레짐 조건부 risk budgeting** (자산군별 VaR 한도 조정)

---

## 14. 최종 결론 (v4.1)

### 14.1 확정 사항

1. **최우수 전략**: **M1_보수형_ALERT_B**
   - Sharpe 1.064, MDD -15.53%, Multi-score 4.10
   - 경로 2 없이 경로 1만으로 최고 성과
   
2. **경로 2는 불채택**: 두 차례 재설계 후에도 무효 → 권고 폐기

3. **매매비용 완화 장치 검토 필요**: M1 FDR 유의율 18.8% → 경계선
   - 60/40 비용 정교화 후 이전 56.2%에서 하락
   - v4.2 후속 작업: 최소 유지 기간 3일 적용 시뮬레이션

### 14.2 실용적 권고

**실전 운용 시**:
- **MV 기반 분기 리밸런싱** + **일별 경보 대응(Config B)** 만 구현
- Σ 레짐 전환 로직 **생략**
- 편도 거래비용 15bps 가정

**보고서에 반영**:
- report_v4.md의 핵심 결론이 M3 → M1로 변경됨
- 경로 2 실패는 "교훈"으로 투명하게 기록

---

## 15. Step 11 추가 (v4.1.1)

### 15.1 배경

- Step 10에서 Top 10 전략 확정했으나 성과는 **숫자**(Sharpe·MDD)로만 요약
- 투자자 관점: "**언제·무엇을·왜** 바꿨는지" 시간 궤적과 원인이 핵심
- Step 9 결과물(`step9_backtest_results.pkl`)은 **일별 수익률만** 저장, weights 시계열 누락

### 15.2 Step 11 목적

> **"Top 10 전략의 일별 자산 비중 시계열을 재생성하고, 시간 흐름과 변화 근거를 시각적으로 해부"**

### 15.3 구현 결정사항

| 항목 | 결정 |
|------|------|
| 재시뮬 범위 | Top 10만 (전체 64 재시뮬은 불필요) |
| 함수 설계 | 신규 `run_simulation_with_weights` — Step 9 영향 없음 |
| 시각화 수 | 8종 (대시보드 + 구성변화 3 + 근거 3 + 해부 1) |
| 검증 방식 | 검증 1 (Step 9 누적수익률 일치) + 검증 2 (경보-비중 정합성) |
| Step 9 원본 | **불변** (리스크 회피) |

### 15.4 핵심 발견 (시각화 재확인)

**경로 1의 시각적 증명**:
- 경보 L0(정상) 주식 40% → L3(위기) 주식 16%
- 시각화 5 박스플롯에서 레벨별 체계적 감소 확인

**경로 2의 무효성 시각적 재확인**:
- 시각화 6에서 M1과 M3의 비중 흐름 거의 동일
- Σ 전환 경계에 미세 점프만 존재
- → Step 10 통계 결론(M3-M1 = -0.04)과 정합

**앵커-반응 이원 구조 발견**:
- 채권(AGG, TLT, SHY, TIP) + 금(GLD) = 앵커 (꾸준한 높은 비중)
- 주식 24개 = 반응 버퍼 (경보 따라 변동)

**Turnover 원인 분해**:
- 대부분 turnover = 일별 경보 변동 (주황)
- 분기 리밸런싱(파랑) 소량
- Σ 전환(보라, M3만) 최소

### 15.5 산출물

- `Step11_Top10_Composition_Analysis.ipynb` (22셀)
- `data/step11_top10_weights.pkl` (6.3 MB)
- `images/step11_01~08_*.png` (8개)
- `docs/Step11_해설.md`

### 15.6 Step 11이 v4.1 결론에 주는 영향

**변경 없음 — 보강만**:
- 최우수 전략: M1_보수형_ALERT_B (변동 없음)
- 경로 2 무효성 판정: **강화** (통계에 이어 시각적으로도 확인)
- 매매비용 완화 검토: 시각화 7이 turnover 주 원인 시각화 → 최소 유지 기간 3일 적용 효과 예측 가능

---

## 변경 이력

| 일자 | 버전 | 변경 내용 | 작성자 |
|------|------|---------|--------|
| 2026-04-17 | v3.1 draft | 최초 작성 (Step8~10 설계) | 김재천 + Claude |
| 2026-04-17 | **v4.1** | **경로 2 재설계·실패 확인, M1 최우수 확정, Cohen's d → IR/ΔSR 교체, 피드백 전면 반영** | 김재천 + Claude |
| 2026-04-17 | **v4.1.1** | **Step 11 추가 — Top 10 weights 재시뮬 + 8종 시각화로 경로 1 실증·경로 2 무효성 시각적 재확인** | 김재천 + Claude |

---

## 승인

이 설계 문서는 사용자와의 반복적 논의를 통해 다음 10가지 핵심 결정사항에 대한 명시적 합의를 바탕으로 작성되었습니다:

1. ✅ HMM Full-sample 고정 레짐 분류 채택 (Deployment Simulation 철학)
2. ✅ 레짐 4단계 유지 (BIC 실측 지지)
3. ✅ Σ 2그룹 (Σ_stable, Σ_crisis) 유지
4. ✅ VIX 기반 Stable/Crisis 자동 그룹화
5. ✅ 관측수 판정 3단계 Fallback (≥48 / 20-47 / <20)
6. ✅ 4성향 × 4Config × 4모드 = 64 시뮬레이션
7. ✅ 매매비용 완화 장치는 Step10 결과 후 조건부 적용
8. ✅ Full-sample vs Expanding Annual 민감도 분석 포함
9. ✅ decision_log_v31.md + report_v4.md 동시 업데이트
10. ✅ 폴더 구조 유지 (Guide/ 내부 추가)
