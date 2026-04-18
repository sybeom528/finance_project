# 🔄 v3 → v4.1 변화표

> **독자**: 프로젝트 리뷰어 / 균형 톤
> **목적**: "무엇이 왜 바뀌었는지" 한 표로 파악

---

## 📊 전체 변화 요약

| 범주 | v3 | **v4.1** | 변화 이유 |
|------|-----|---------|---------|
| 파이프라인 Step 수 | 7개 | **11개** | 경로 2 구현 + 시각화 확장 |
| Baseline | Equal Weight 1/30 | **MV 최적화** | 실전 기관 표준 반영 |
| 경로 2 (레짐 Σ) | ❌ 미구현 | ✅ 구현 (무효 확인) | 설계 완결성 |
| Ablation 규모 | 16 조합 | **64 조합** | 4모드 × 4성향 × 4Config |
| 통계 검정 | Bootstrap 95% CI | **+Bonferroni +FDR +IR +ΔSR** | 엄밀성 향상 |
| 최우수 전략 | 보수형_ALERT_B | **M1_보수형_ALERT_B** | 동일 Config·성향, 모드만 다름 |
| 최우수 Sharpe | 1.473 | **1.064** | MV base로 인한 margin 감소 |
| 문서 | report_v3.md | **+ report_v4 + report_final + 11 해설** | 독자별 맞춤 |

---

## 🔍 주요 변화 상세

### 1️⃣ Baseline 전환: EW → MV

**v3 (EW)**:
```python
base_w = np.ones(30) / 30  # 모든 자산 1/30
```
- 30자산 균등
- 최적화 없음 → Sharpe 측정 시 "경보의 순수 효과" 집중
- 한계: 실전 기관은 MV 사용

**v4.1 (MV)**:
```python
weights = optimize_mv_weights(μ, Σ, profile_params, ...)
# γ=8, max_equity=43%, min_bond=31% 제약
```
- IS 24개월 데이터로 MV 최적화
- 성향별 제약 반영
- **변화**: Sharpe 1.47 → 1.06 (margin 감소, 그러나 현실적)

---

### 2️⃣ 경로 2 구현 + 실패 발견

**v3**: 설계 문서(decision_log)에만 명시, 미구현
```
대안데이터 → HMM 레짐 → Σ_crisis 전환 → 최적화 비중 자동 변화
```

**v4.1 첫 시도**: OOS 시작일 1회 경보로 Σ 선택
- M2 Sharpe 0.810 (M0 대비 +0.016, 미미)

**v4.1 재설계**: 월 단위 Σ 전환 (피드백 반영)
- M2 Sharpe **0.749** (M0 대비 **-0.045, 역효과**)
- **결론**: 경로 2는 무효, 폐기 권고

**왜 실패했나**:
1. Σ_stable과 Σ_crisis의 차이가 MV 비중에 미약한 반영
2. 재최적화 비용 > 이론적 이득
3. 경로 1(경보→주식 축소)과 정보 중복

---

### 3️⃣ 통계 검정 강화

**v3**: Bootstrap 5,000 + 95% CI
- 간단하나 다중 비교 오류 미보정

**v4.1 추가**:
| 기법 | 도입 이유 |
|------|---------|
| **Bonferroni 보정** | 48개 비교의 false positive 제어 |
| **FDR (BH)** | 덜 보수적, 검정력 보존 |
| **IR (Information Ratio)** | 재무 실무 표준 (Grinold-Kahn) |
| **ΔSR 실무 임계** | 0.2 이상 실질 개선 판정 |
| **M3 vs M1 직접 검정** | 경로 2 부가가치 실증 |

**Cohen's d 폐기**:
- 일별 수익률 SNR 낮아 항상 "미미"로 분류
- 재무 시계열에 부적합 (심리학 기준)
- → IR + ΔSR로 교체

---

### 4️⃣ 최우수 전략 변경

**v3 Top 3**:
| 순위 | 전략 | Sharpe |
|------|------|-------|
| 1 | 보수형_ALERT_B | 1.473 |
| 2 | 보수형_ALERT_C | 1.376 |
| 3 | 보수형_ALERT_A | 1.364 |

**v4.1 Top 5**:
| 순위 | 전략 | Sharpe | MDD |
|------|------|-------|------|
| 1 | **M1_보수형_ALERT_B** | **1.064** | **-15.53%** |
| 2 | M1_중립형_ALERT_C | 1.066 | -20.30% |
| 3 | M3_중립형_ALERT_C | 1.028 | -19.48% |
| 4 | M1_중립형_ALERT_B | 1.014 | -20.05% |
| 5 | M1_보수형_ALERT_C | 1.072 | -16.97% |

**공통점**:
- 최우수 = **보수형 + Config B**
- 경보 기반 방어 전략의 일관된 우수성

**차이점**:
- v3: baseline이 EW이므로 단순 Sharpe가 큼
- v4.1: MV baseline + 비용 정교화로 현실적 수치
- v4.1 Multi-criteria(Sharpe+MDD+Calmar+Sortino) 사용

---

### 5️⃣ 문서 체계 확장

**v3 산출물**:
- `decision_log.md` (설계)
- `report_v3.md` (결과 1개)
- `stats_model.md` (통계 기법)

**v4.1 산출물**:
- 기존 + `decision_log_v31.md` (v4.1 설계)
- `report_v4.md` (v4.1 결과)
- `report_final.md` (전체 통합, 본 프로젝트의 "얼굴")
- `docs/Step1~11_해설.md` (11개 비전문가 해설)
- `quick_reference/` (13종 보조 자료)
- `interactive/` (Streamlit 앱 + HTML 대시보드)

**변화 이유**: 독자별 맞춤 (경영진·투자자·엔지니어·연구자)

---

### 6️⃣ 새로 추가된 개념

| 개념 | 도입 Step | 의미 |
|------|---------|------|
| **대칭 Fallback 4단계** | Step 8 | Σ 관측수 부족 시 대체 전략 |
| **월 단위 Σ 전환** | Step 9 | 경로 2 재설계 |
| **Deployment Simulation** | Step 8 | HMM Full-sample 학습 철학 |
| **LOO 견고성 검증** | Step 10 | 특정 윈도우 의존성 확인 |
| **IR + ΔSR 실무 기준** | Step 10 | Cohen's d 대체 |
| **Weights 재시뮬** | Step 11 | Top 10 시간 구성 분석 |
| **앵커-반응 이원 구조** | Step 11 | 포트폴리오 해부 관점 |

---

## 📌 버전별 주요 결정 포인트

### v3 결정 (decision_log.md)
- Walk-Forward 방식 채택
- 30자산 유니버스 확정
- 4개 성향 γ=1,2,4,8 수식
- Config A/B/C/D 4개 경보 설계

### v4.1 결정 (decision_log_v31.md)
- Full-sample HMM → Deployment Simulation 철학
- 대칭 Fallback으로 버그 수정
- 경로 2 월 단위 재설계
- Cohen's d → IR/ΔSR 교체
- 경로 2 실증 폐기 결정
- Step 11 시각화 확장

---

## 🎯 종합 판단

**v3의 가치**:
- 경보 시스템의 개념 증명 (EW 기반)
- 대안데이터 선행성 입증 (Granger)

**v4.1의 기여**:
- 실전 기관 환경(MV)에서 검증
- 경로 2 무효성 실증 (중요한 negative result)
- 통계 엄밀성 확보
- 다독자 맞춤 문서화

**실전 적용 권장**:
- **v4.1 결과 기준** (M1_보수형_ALERT_B)
- v3 수치(Sharpe 1.47)는 EW baseline 한계로 이해
- 실전 운용에서는 v4.1 매개변수 사용

---

## 📞 참조 문서

| 목적 | 문서 |
|------|------|
| v3 상세 결과 | `report_v3.md` |
| v4.1 상세 결과 | `report_v4.md` |
| 전체 통합 | `report_final.md` |
| v3 설계 근거 | `decision_log.md` |
| v4.1 설계 근거 | `decision_log_v31.md` |
