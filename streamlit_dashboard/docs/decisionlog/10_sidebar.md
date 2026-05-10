# C-4 사이드바 + C 섹션 완료 요약

> **파일**: `10_sidebar.md`
> **결정 시점**: 2026-05-10
> **상태**: C-4 사이드바 확정 + C 섹션 완료 요약
> **포함**: C-4 사이드바 구성 + 페이지 그룹화 (5 그룹 + 2 토글) / C 섹션 완료 요약 / 다른 페이지 narrative 일괄 정정 (Q-5) / About / FAQ Selection Bias 부록 (Q-B3 메모)

---

## C-4. 사이드바 구성 + 페이지 그룹화 — 확정

### C-4 통합 배경 (Context)

8 페이지 모두 결정 완료. 사이드바 구조 + 페이지 navigation + 토글 구성 확정.

**사용자 누적 결정사항** (사이드바 영향):
- 8 페이지 모두 navigation (C-1)
- 사이드바 토글 — 기간 (FULL/TEST/HO) (Overview 영역 2 통합 결정 ①)
- 사이드바 토글 — 다중 벤치마크 (SPY/EW/IVW) (Performance 영역 3)
- 한/영 병기 (A-3) → 토글 불필요
- 다크 테마 Cobalt Blue (B-4) → 고정

### 결정 항목 C4-1: Navigation 표시 방식

**검토된 옵션**:
- (a) 평면 리스트 (8 페이지)
- (b) 카테고리 그룹
- (c) 카테고리 헤더 + 페이지 (Streamlit pagelink)

**결정**: (c) 카테고리 헤더 + 페이지

**근거**:
1. **정보 위계 명확** — 카테고리 헤더로 페이지 분류 시각
2. (a) 평면은 8 페이지 산만
3. (b) 그룹은 헤더 표시 ↓

### 결정 항목 C4-2: 페이지 그룹화

**결정**: (a) 5 그룹 → ★ **6 그룹으로 갱신** (Investment Simulator 페이지 추가, F 섹션 결정)

**그룹 구성** (6 그룹):
- **개요**: Overview
- **체험** ★ 신규 (F 섹션 결정): Investment Simulator
- **성과**: Performance / Risk Metrics
- **보유**: Holdings / Sector Watch
- **검증**: Methodology / Backtesting
- **메타**: About / FAQ

**변경 이력**:
- *2026-05-10 그룹 1 라벨: "첫인상" → "개요"* — 사용자 피드백 ("첫인상은 어색하다"). 의미 동일 (사이트 진입 시 첫 페이지) 이지만 표현이 더 자연스러움.

**근거**:
1. **6 그룹** = 명확한 정보 위계 + Investment Simulator 별도 그룹
2. (b) 4 그룹은 보유/검증 통합 시 의미 ↓
3. (c) 3 그룹은 성과/보유/검증 한 그룹 → 위계 ↓
4. **★ Investment Simulator** = 인터랙티브 체험 기능 → "체험" 신규 그룹 (F 섹션 결정)

### 결정 항목 C4-3: 펀드명 표시

**결정**: (b) 텍스트 마크 + 메타

**표시 항목**:
- 펀드명 (영문 + 한글)
- Benchmark: SPY
- Last updated: 2025-12

**근거**:
1. **B-5 텍스트 마크** + 메타 강화
2. 사이드바 상단에서 펀드 정체성 + 신뢰성 즉시 노출
3. (c) 펀드 상태 (● Active) 는 Header 영역 1 와 중복

### 결정 항목 C4-4: 토글 구성

**결정**: (a) 기간 + 비교 (2 토글 그룹)

**토글 안**:
- **기간 (Period)**: ● FULL / ○ TEST / ○ HO (radio, 단일 선택)
- **비교 (Benchmark)**: ☑ SPY / ☐ EW / ☐ IVW (checkbox, 다중 선택)

**근거**:
1. **A-3 한/영 병기** → 언어 토글 불필요
2. **B-4 다크 테마** → 테마 토글 불필요
3. (b/c) 추가 토글은 사이드바 산만

### 결정 항목 C4-5: 추가 요소

**결정**: (a) 단순 (페이지 + 토글만)

**근거**:
1. **균형 (B) 적용** — 사이드바 단순
2. **Generate Report** 버튼은 페이지 내 기능으로 (영역별 CSV 다운로드 등)
3. **다운로드 / 공유** 는 Footer 의 GitHub link 로 충분

### 사이드바 시각화 예시 (확정 사항 조합 — 6 그룹)

```
┌──────────────────────────┐
│ Adaptive VolControl Fund │
│ 어댑티브 볼컨트롤 펀드   │
│                          │
│ Benchmark: SPY           │
│ Data: 2025-12            │
├──────────────────────────┤
│ ── 개요 ──               │
│ ◉ Overview               │
│                          │
│ ── 체험 ── ★ 신규         │
│ ◯ Investment Simulator   │
│                          │
│ ── 성과 ──               │
│ ◯ Performance            │
│ ◯ Risk Metrics           │
│                          │
│ ── 보유 ──               │
│ ◯ Holdings               │
│ ◯ Sector Watch           │
│                          │
│ ── 검증 ──               │
│ ◯ Methodology            │
│ ◯ Backtesting            │
│                          │
│ ── 메타 ──               │
│ ◯ About / FAQ            │
├──────────────────────────┤
│ 📅 기간 (Period):        │
│  ● FULL  ○ TEST  ○ HO    │
│                          │
│ 📊 비교 (Benchmark):     │
│  ☑ SPY                   │
│  ☐ EW (펀드 universe)    │
│  ☐ IVW (Naive Low-vol)   │
└──────────────────────────┘
```

### 결과 / 함의

- 사이드바 = `st.sidebar` + `st.page_link` (Streamlit 1.30+)
- 카테고리 헤더 = `st.subheader` 또는 markdown
- 기간 토글 = `st.radio`
- 비교 토글 = `st.checkbox` × 3
- 토글 값 = `st.session_state` 저장 → 모든 페이지 공유
- **C 섹션 (페이지 구조) 완료** — 8 페이지 + 사이드바 모두 확정

---

## ★ C 섹션 (페이지 구조) 완료 요약

| 섹션 | 결정 | 상태 |
|---|---|---|
| C-1. 페이지 진행 방식 | 페이지별 한 대화당 하나씩 | 확정 |
| C-1-1. Overview (6 영역) | Header + Hero + 누적수익 + 강점 + Nav + Footer | 확정 |
| C-1-2. Performance (9 영역) | Sub-header + KPI + Annual Returns + Active Return + Rolling + Regime + 분포 | 확정 |
| C-1-3. Risk Metrics (8 영역) | Sub-header + KPI + DD + VaR/CVaR + Beta + 종합표 + Hill | 확정 |
| C-1-4. Holdings (8 영역) | Sub-header + KPI + Top N + 시가총액 + 변천사 + 기여도 | 확정 |
| C-1-5. Sector Watch (8 영역) | Sub-header + KPI + Treemap + Decomposition + Tilt + Rotation + HO 정당화 | 확정 |
| C-1-6. Methodology (8 영역) | Sub-header + Sankey + BL + LSTM + Factor + 정규성 + 한계 | 확정 |
| C-1-7. Backtesting (7 영역) | Sub-header + KPI + Regime 자세 + Sub-events + Sensitivity | 확정 (Stress 제거) |
| C-1-8. About / FAQ (8 영역) | 메타 결정만 (영역별 = 구현 후 팀 상의) | 부분 확정 |
| C-2. 메트릭 풀 (Universe) | 53개 메트릭 / 9 카테고리 | 확정 |
| C-3. Hero KPI | Overview 영역 2 에서 결정 | 확정 |
| C-4. 사이드바 구성 | 5 그룹 + 2 토글 | 확정 |

### 다른 페이지 narrative 일괄 정정 (Q-5)

**잘못된 표현 → 정확한 표현 일괄 정정**:

| 잘못된 표현 | 정확한 표현 | 영향 페이지 |
|---|---|---|
| "학습 168m / 검증 24m" | "TEST 평가 168m / HOLD_OUT 평가 24m" | 모든 페이지 narrative |
| "Walk-forward 미적용" | (제거 — 사실 적용됨) | Methodology 영역 8 |
| "168m 학습 구간" | "168m TEST 평가 구간 (config selection)" | Sector Watch 영역 8 등 |

→ 구현 시 narrative 일괄 정정 적용

### About / FAQ 페이지 — Selection Bias 부록 추가 (Q-B3 메모)

About 페이지 메타 결정 시 다음 추가 영역 고려:
- **학술 부록 영역**: Selection bias / PBO/DSR / Data snooping 학술 한계
- 위치: About 페이지 후반부 (영역 6)
- 톤: 학술 보고서 형식 (대시보드 본문에서 회피)

→ 09_about.md 메타 결정 단계 (영역 6) 에 반영됨.

---

[← 09_about.md](09_about.md) | [11_dl_sections.md](11_dl_sections.md) →

> 본 파일은 사이드바 (C-4) 결정 + C 섹션 완료 요약을 담고 있습니다. D~L 섹션 결정은 11_dl_sections.md 파일에 기록됩니다.
