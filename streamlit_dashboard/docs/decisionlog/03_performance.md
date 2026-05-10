# C-1-2. Performance 페이지

> **파일**: `03_performance.md`
> **결정 시점**: 2026-05-10
> **상태**: 확정 (페이지 메타 M-1~M-4 + 영역 1~9)
> **포함**: 페이지 메타 결정 / Sub-header / Performance Summary KPI 5개 / Annual Returns 막대 / Active Return 분석 / Annualized Rolling Return / Regime 메트릭 비교 / 분포 통계 카드

---

## C-1-2. Performance 페이지

> **상태**: 진행 중 (9 영역 중 영역 1, 9 는 Overview 동일 / 영역 2-3 확정 / 4-8 미정)

### Performance 페이지 통합 배경 (Context)

성과 분석 (Performance Analysis) 전담 페이지. Overview 의 Hero KPI / 누적수익 곡선의 깊이 있는 버전. 메트릭 풀 (C-2) 의 Pool-1 (수익성), Pool-2 (위험조정수익), Pool-5 (시장 비교), Pool-7 (Regime), Pool-9 (분포 통계) 활용.

**핵심 차별점**:
- vs Overview: 5초 인상 → 깊이 있는 분석
- vs Risk Metrics: 수익 위주 (Sortino/Sharpe/CAGR/Active Return) ↔ 위험 위주 (VaR/CVaR/Beta/Drawdown)

### 페이지 메타 결정 (M-1 ~ M-4)

#### M-1. 영역 개수

**검토된 옵션**:
- (a) 압축 6 영역
- (b) 표준 8 영역 (Header / Sub-KPI / Annual Returns / Active Return rolling 통합 / Regime 표 / Footer)
- (c) 풍부 9-10 영역

**결정**: (b) 표준 8 영역 + 분포 통계 카드 추가 → **9 영역**

**근거**:
1. 분포 통계 카드 (Skewness/Kurtosis/Tail Ratio) = 우리 메트릭 풀 (Pool-9) 핵심 활용
2. fat tail / 비대칭 위험 시각화 → 펀드 정체성 (VolControl) 강화
3. (a) 압축 6 영역은 메트릭 풀 활용 부족
4. (c) 풍부 9-10 은 적정 — 자유 탐색 모드 부합

#### M-2. Sub-header 포함

**결정**: (a) 포함

**근거**: 페이지 컨텍스트 + 사이드바 토글 영향 안내 → 청중 이해도 ↑

#### M-3. Page Summary KPI

**결정**: (a) 5개 — Overview Hero 와 차별화

**근거**: Overview Hero (펀드 정체성 위주) ≠ Performance Summary (액티브 운용 강조)

#### M-4. Regime 표 위치

**검토된 옵션**:
- (a) Performance 만
- (b) Backtesting 만
- (c) 둘 다 (간단/자세 분리)

**결정**: (c) 둘 다 — Performance 간단 (Regime 별 CAGR / Active Return) + Backtesting 자세 (Regime 별 다중 메트릭 + Sub-events)

**근거**:
1. Performance 페이지 일부에 Regime 표 = "성과의 시간 분포" narrative 강화
2. Backtesting 자세 = "검증" narrative
3. 두 페이지가 다른 깊이로 같은 데이터 활용 → 청중 자유 탐색 ↑

### 9 영역 구조

```
1. Header                           (Overview 와 동일)
2. Sub-header                       (페이지 컨텍스트 + 토글 안내)
3. Performance Summary KPI 5개      (액티브 운용 강조)
4. Annual Returns 막대              (Image 6)
5. Active Return 분석               (Image 1 + Image 2)
6. Annualized Rolling Return        (1y/3y/5y)
7. Regime 메트릭 간단 비교          (Backtesting sneak peek)
8. 분포 통계 카드                   (Skewness/Kurtosis/Tail Ratio)
9. Footer                           (Overview 와 동일)
```

### 영역 2: Sub-header — 확정

#### 결정 항목 S2-1: 텍스트 내용

**검토된 옵션**:
- (a) 페이지 정의만
- (b) 정의 + 토글 안내
- (c) 정의 + 토글 + 학술 컨텍스트

**결정**: (b) 정의 + 토글 안내

**근거**:
1. (a) 정의만 으로는 토글 사용법 안내 부족
2. (c) 학술 컨텍스트는 About 페이지에서 자세히 다룸 → 중복 회피
3. 토글 안내가 자유 탐색 모드 (A-2) 의 핵심

**텍스트 안**:
```
펀드의 수익성, 시장 비교, regime별 분해 분석.
사이드바에서 기간 (FULL / TEST / HO) 선택 시 차트 자동 갱신.
비교 벤치마크 (SPY / EW / IVW) 도 선택 가능.
```

#### 결정 항목 S2-2: 디자인

**검토된 옵션**:
- (a) 소형 텍스트 한 줄
- (b) 카드/배너 형식 (background color)
- (c) Info bar (Streamlit `st.info`)

**결정**: (b) 카드/배너 형식

**근거**:
1. Streamlit `st.info` (c) 는 파란색 강제 → Cobalt Blue 팔레트와 다른 톤
2. (a) 소형 텍스트는 시각적 약함
3. 카드/배너는 CSS 자유 조정 → Cobalt Blue 팔레트 일관

#### 결정 항목 S2-3: 한/영 병기

**결정**: (b) 한/영 병기

**근거**: A-3 결정 일관 — 모든 페이지 한/영 병기

**텍스트**: "Performance Analysis (성과 분석)"

#### 결과 / 함의

- 모든 페이지 (Risk Metrics / Holdings 등) 의 Sub-header 패턴 동일 적용
- 카드 디자인 = Cobalt Blue accent + dark background 일관

---

### 영역 3: Performance Summary KPI 5개 — 확정

#### 영역 3 통합 배경 (Context)

Sub-header 아래 페이지 전용 KPI 5개. **Overview Hero 와 차별화** 하여 Performance 페이지의 액티브 운용 가치 강조.

#### 결정 항목 S3-1: KPI 메트릭 선정

**Overview Hero 5개**: Cumulative Return / Net CAGR / Sortino / Volatility / MDD (펀드 정체성 위주)

**검토된 Performance Summary 후보**:
- (a) CAGR / Sortino / Sharpe / Win Rate / Best Year (성과 다각도)
- (b) CAGR / Sortino / Sharpe / Information Ratio / Active Return (액티브 강조)
- (c) Net CAGR / Sortino / Calmar / Win Rate / Up Capture (위험조정 종합)
- (d) CAGR / Sortino / IR / Up/Down Capture (시장 비교)
- (e) 자유 선택

**결정**: (b) CAGR / Sortino / Sharpe / Information Ratio / Active Return

**근거**:
1. Overview Hero (펀드 정체성: Volatility/MDD/Sortino) 와 차별화
2. **액티브 운용 강조**: IR + Active Return → "시장 대비 액티브 운용 가치" narrative
3. CAGR / Sortino / Sharpe = 표준 성과 메트릭 (학술/실무 모두 인지)

#### 결정 항목 S3-2: 표시 기간

**검토된 옵션**:
- (a) Overview 동일 (TEST + HO 별도)
- (b) 사이드바 토글 반응
- (c) FULL 고정 + 토글 추가

**결정**: (b) 사이드바 토글 반응

**근거**:
1. 영역 2 통합 결정 ① (Hero 고정 + 다른 페이지 토글 영향) 과 부합
2. Performance 페이지는 깊이 탐색 모드 → 토글로 기간 변경 가능
3. (a) Overview 동일은 첫인상 강하지만 페이지 의미 약화

#### 결정 항목 S3-3: 카드 디자인

**검토된 옵션**:
- (a) Overview Hero 동일 (큰 숫자 + sparkline)
- (b) 더 단순 (큰 숫자만, sparkline 없음)
- (c) 더 풍부 (+ delta vs SPY)

**결정**: (b) 더 단순 (큰 숫자만)

**근거**:
1. Overview Hero (sparkline) 와 차별화 → 정보 위계 명확
2. Performance 페이지는 차트가 풍부 (영역 4-8) → KPI 단순화로 균형
3. delta 는 S3-4 에서 별도 처리

#### 결정 항목 S3-4: 벤치마크 delta 표시

**검토된 옵션**:
- (a) 표시 안 함
- (b) 모든 카드에 SPY delta
- (c) 일부만 (성과 메트릭만)

**결정**: (b) 모든 카드 + **추가: 다중 벤치마크 토글 통합**

##### 추가 결정 — 다중 벤치마크 토글 (사용자 추가 질문)

**배경**: 영역 3 (누적수익 곡선) 에서 SPY / EW / IVW 토글 결정. KPI delta 도 동일 패턴 적용 가능.

**검토된 옵션**:
- (a) SPY only
- (b) 3개 모두 항상 표시 (복잡도 ↑)
- (c) 사이드바 다중 토글 (선택된 대상만 delta 표시)
- (d) Radio 단일 선택

**결정**: (c) 사이드바 다중 토글

**근거**:
1. **학술 narrative 다각도**:
   - vs SPY = 시장 outperformance
   - vs EW = 액티브 운용 가치 (BL+LSTM 가치)
   - vs IVW = LSTM 가치 (단순 inverse vol vs LSTM 예측)
2. **자유 탐색 모드 부합** (A-2 결정)
3. **카드 복잡도 사용자 자율 선택** — 기본 SPY만, 추가는 사용자 의도
4. **누적수익 곡선 토글과 통합** — 한 토글로 모든 영역 영향, 일관성 ✓

**작동 방식**:
- 사이드바: `[☑ SPY] [☐ EW] [☐ IVW]` 다중 체크박스
- 기본: SPY 만 활성
- 카드 안: 선택된 대상마다 작은 delta 행 동적 추가
- 영향 범위: Performance Summary KPI + 누적수익 곡선 (영역 3) + 다른 페이지 차트

#### 시각화 예시

```
[Sub-header Info bar]

[기간: TEST 168m]  [비교: SPY ✓ / EW ☐ / IVW ☐]
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ CAGR     │ Sortino  │ Sharpe   │ IR       │ Active   │
│ +X.XX%   │ X.XX     │ X.XX     │ X.XX     │ Return   │
│ vs SPY   │ vs SPY   │ vs SPY   │          │ +X.XX%/y │
│ ▲+X.X%   │ ▼-X.XX   │ ▼-X.XX   │          │ vs SPY   │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

EW 추가 시 카드 안에 "vs EW: ▲+X.XX" 행 추가, IVW 동일.

#### 결과 / 함의

- 사이드바에 **2개 토글 그룹** 추가 결정:
  - 기간 토글: FULL / TEST / HO (단일 선택, radio)
  - 벤치마크 토글: SPY / EW / IVW (다중 선택, checkbox)
- 모든 페이지의 비교 차트 / KPI delta 가 이 두 토글 영향
- C-5 (사이드바 구성) 결정 시 이 2개 토글 통합 확정
- IR / Active Return 산출 시 SPY 외 EW / IVW 도 별도 산출 필요 (구현 시)

### 영역 4: Annual Returns 막대 (Image 6) — 확정

#### 영역 4 통합 배경 (Context)

연도별 펀드 vs 벤치마크 막대 차트 — 레퍼런스 PortfolioVisualizer Image 6 패턴. 절대 수익률을 사이드바이로 비교하여 직관적 시각화. Image 1 (Active Return, 차이값) 과 차별화: Image 6 = 절대값.

#### 결정 항목 S4-1: 차트 종류

**검토된 옵션**:
- (a) 사이드바이 막대 (Image 6 동일)
- (b) 누적 막대 (Stacked)
- (c) 그룹 막대 + 라인 오버레이

**결정**: (a) 사이드바이 막대

**근거**:
1. 레퍼런스 Image 6 동일 — 정통 금융 패턴
2. 청중 직관적 이해 (펀드 vs 벤치마크 절대값 비교)
3. 다중 벤치마크 (SPY/EW/IVW) 시 막대 그룹 자연스러움

#### 결정 항목 S4-2: 비교 대상

**결정**: (a) 다중 토글 적용

**근거**: 영역 3 (Performance Summary KPI) 의 다중 벤치마크 토글과 일관 — 한 토글로 모든 차트 영향

**막대 개수 동적 변화**:
- SPY 만 → 2 막대 그룹 (Fund + SPY)
- SPY + EW → 3 막대 그룹
- SPY + EW + IVW → 4 막대 그룹

#### 결정 항목 S4-3: 인터랙션

**검토된 인터랙션 + 결정**:

| 인터랙션 | 결정 |
|---|---|
| Hover tooltip (연도 + 각 수익률 + delta) | ✓ |
| 연도 정렬 (연대순 / 수익률순 토글) | ✓ |
| 양수/음수 색상 자동 (펀드 막대) | ✓ |
| 연도 클릭 → 월별 expand | ✓ (Q-Zoom 결정) |

##### Q-Zoom 결정 (모든 페이지 영향)

**배경**: 사용자가 연도 클릭 시 월별 detail 표시 가능 여부 질문

**구현 옵션 분석**:
- 옵션 1: Plotly 기본 X축 zoom (자동, 코드 0줄)
- 옵션 2: Range slider (1줄)
- 옵션 3: 연도 클릭 → 월별 별도 차트 expand (Streamlit 1.27+ `on_select='rerun'`, 1시간 작업)
- 옵션 4: Crossfilter (복잡, 제외)

**검토된 결정 옵션**:
- (a) 1+2+3 조합 채택 (Phase 1 포함)
- (b) 클릭 expand 만 Phase 2 보류
- (c) 기본 zoom + slider 만

**결정**: (a) 1+2+3 조합 채택 (Phase 1 포함)

**근거**:
1. **자유 탐색 모드 (A-2) 핵심 부합** — 연도 → 월 deep-dive
2. **구현 비용 보통 (1시간)** — Streamlit 1.27+ `on_select` 활용
3. **인터랙션 일관성 원칙** — 모든 시계열 차트에 동일 zoom 패턴

##### Q-Zoom 영향 범위 (모든 페이지)

| 페이지 / 영역 | 적용 |
|---|---|
| Performance 영역 4 (Annual Returns) | 연도 클릭 → 월별 |
| Performance 영역 5 (Active Return) | 연도 클릭 → 월별 |
| Performance 영역 6 (Rolling Return) | 윈도우 클릭 → detail |
| Performance 영역 7 (Regime 표) | Regime 클릭 → 메트릭 expand |
| Risk Metrics (Drawdown) | 시기 클릭 → detail |
| Holdings | 종목 클릭 → detail |
| Sector Watch | 섹터 클릭 → 종목 list |
| Backtesting (Regime 표) | Regime 클릭 → 자세한 메트릭 |

#### 결정 항목 S4-4: Y축 라벨

**검토된 옵션**:
- (a) 막대 위 % 라벨
- (b) 막대만 (hover 시 표시)
- (c) 양수/음수만 라벨

**결정**: (b) 막대만 (hover)

**근거**:
1. 다중 벤치마크 (4 막대 그룹) 시 라벨 혼잡
2. Hover tooltip 으로 정확한 수치 노출
3. 시각적 깔끔

#### 결정 항목 S4-5: 평균선 / 표준편차 영역

**검토된 옵션**:
- (a) 평균선 표시
- (b) 표준편차 영역
- (c) 둘 다
- (d) 표시 안 함

**결정**: (a) 평균선 표시

**근거**:
1. 펀드 평균 + SPY 평균 수평선 → 통계 보강
2. (b) 표준편차 영역은 막대 차트와 어울리지 않음 (선/영역 차트에 적합)
3. 평균선 만으로 통계 narrative 충족

#### 시각화 예시 (확정 사항 조합)

```
[Performance Summary KPI 5개]

┌─ Annual Returns: Adaptive VolControl Fund vs SPY ────────┐
│                                                          │
│  +30%┤        ▓ Fund   ░ SPY                            │
│  +20%┤        ▓░    ▓░       ▓░    ▓░    ▓░             │
│  +10%┤    ▓░  ▓░  ▓░    ▓░  ▓░  ▓░   ▓░  ▓░         ─── Avg(Fund)│
│    0%┤────────────────────────────────────────────  ─── Avg(SPY) │
│  -10%┤              ░                  ░░               │
│  -20%┤      ▓░                              ▓░          │
│       └─────────────────────────────────────────────    │
│        2010  2012  2014  2016  2018  2020  2022  2024   │
│                                                          │
│  Hover: "2024: Fund +X.XX%, SPY +24.8%, Δ -X.XX%"       │
│  Click 2024 → 2024년 월별 차트 expand                    │
│                                                          │
│  [정렬: 연대순 ▼] [☑ SPY] [☐ EW] [☐ IVW]               │
└──────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- **Q-Zoom 결정 = 모든 페이지 일관 적용** (부록 인터랙션 원칙 추가)
- 연도 클릭 expand → Streamlit 1.27+ `on_select='rerun'` 활용
- 다중 벤치마크 토글 + 평균선 + 클릭 expand 종합 = 자유 탐색 모드 (A-2) 강력 구현
- 다른 페이지 (Risk Metrics / Holdings / Sector Watch / Backtesting) 결정 시 클릭 expand 패턴 일관 적용

### 영역 5: Active Return 분석 (Image 1 + Image 2) — 확정

#### 영역 5 통합 배경 (Context)

액티브 운용 가치를 두 차트로 시각화:
- **Image 1**: 연도별 Active Return = (Fund - Benchmark) 막대 — 시간 분포
- **Image 2**: Rolling Active Return + Tracking Error (36m) — 시간 안정성

영역 4 (절대 수익률 사이드바이) 와 차별화: **차이값 위주**.

#### 결정 항목 S5-1: 두 차트 표시 방식

**검토된 옵션**:
- (a) 위아래 배치 — Image 1 위 + Image 2 아래
- (b) 좌우 배치 (반응형)
- (c) Tab 전환
- (d) 하나만 (사용자 선택)

**결정**: (a) 위아래 배치

**근거**:
1. Narrative 자연스러움: 시간 분포 (Image 1) → 시간 안정성 (Image 2)
2. 청중이 두 차트 동시 비교 가능 (Tab/선택은 비교 ↓)
3. 좌우 배치 (b) 는 반응형이지만 두 차트가 서로 다른 X축 (연도 vs 월) 으로 좌우 어울림 ↓

#### 결정 항목 S5-2: Image 1 비교 대상

**결정**: (a) 다중 토글 적용

**근거**:
1. 영역 3, 4 와 일관 — 한 토글로 모든 차트 영향 (인터랙션 일관성 원칙)
2. 토글된 각 벤치마크 (SPY/EW/IVW) 에 대한 Active Return 막대
3. SPY 만 → 1 막대 그룹 / SPY+EW → 2 막대 그룹 / 3개 모두 → 3 막대 그룹

#### 결정 항목 S5-3: Image 2 Rolling 윈도우

**검토된 옵션**:
- (a) 36m only (Image 2 동일)
- (b) 토글 (12m / 36m / 60m)
- (c) 다중 윈도우 동시 표시

**결정**: (b) 토글 (12m / 36m / 60m)

**근거**:
1. 자유 탐색 모드 (A-2) 부합 — 다양한 시간 척도 검증
2. **Annualized Rolling Return (영역 6) 의 1y/3y/5y 결정과 매칭** (12m≈1y, 36m=3y, 60m=5y)
3. (c) 다중 동시는 한 차트에 너무 많은 라인 → 가독성 ↓

#### 결정 항목 S5-4: Tracking Error 표시

**검토된 옵션**:
- (a) 이중 축 (Image 2 동일)
- (b) 별도 차트 (수직 분할)
- (c) Tracking Error 토글 (표시/숨김)
- (d) Tracking Error 제외

**결정**: (a) 이중 축

**근거**:
1. Image 2 동일 패턴 — 학술 표준
2. 한 화면에 정보량 max — Active Return + Tracking Error 동시 비교
3. 좌축 = Active Return 막대 / 우축 = Tracking Error 라인
4. (b) 별도 차트는 영역 분량 ↑

#### 결정 항목 S5-5: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- Plotly 기본 zoom + range slider ✓
- 연도/시기 클릭 → expand ✓ (Q-Zoom)
- 양수/음수 색상 자동 (Image 1) ✓
- Tracking Error accent 색상 (Image 2) ✓

#### 시각화 예시 (확정 사항 조합)

```
[Annual Returns 막대 (영역 4)]

┌─ Active Return Analysis ────────────────────────────────┐
│                                                         │
│ ┌─ Annualized Active Return (Fund - SPY/EW/IVW) ─────┐ │
│ │                                                     │ │
│ │ +15%┤    █ vs SPY                                  │ │
│ │ +10%┤  ██  ██                                       │ │
│ │  +5%┤██  ████  ██                                   │ │
│ │   0%┼────────────────────────────────────────────  │ │
│ │  -5%┤        ██  ██  ██  ██                         │ │
│ │ -10%┤              ██████████                       │ │
│ │       2010  2014  2018  2022                        │ │
│ │  Click: 2024 → 2024 월별 Active Return expand       │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ ┌─ Rolling Active Return + Tracking Error (36m) ─────┐ │
│ │                            좌축: Active   우축: TE  │ │
│ │ +10%┤          ╱╲              ┄┄┄ TE         12% │ │
│ │  +5%┤      ╱╱──── ╲                ───┐       10% │ │
│ │   0%┼────╱─────────╲────────╲╱╲─    │        8%  │ │
│ │  -5%┤  ╱             ╲     ╱   ╲    │        6%  │ │
│ │ -10%┤╱                 ╲╱╲       ╲  │        4%  │ │
│ │       2013  2016  2019  2022  2025                 │ │
│ │  Click: 시기 → detail                              │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│  [Window: 12m / 36m / 60m ▼]  [☑ SPY] [☐ EW] [☐ IVW]  │
└─────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- 영역 6 (Annualized Rolling Return) 의 1y/3y/5y 윈도우와 매칭 → 일관성
- 다중 벤치마크 토글 + Q-Zoom + 양수/음수 색상 = 자유 탐색 모드 강력 구현
- 이중 축 차트 = Plotly `make_subplots` 또는 secondary_y 활용
- Active Return 양수/음수 색상 = Plotly `marker.color` 동적 설정

### 영역 6: Annualized Rolling Return (1y / 3y / 5y) — 확정

#### 영역 6 통합 배경 (Context)

다양한 시간 척도의 rolling 수익률 시계열 — 펀드의 시간 안정성 검증.

**vs Image 2 (영역 5)**:
- 영역 5 = Active Return + Tracking Error (차이값 + 위험)
- 영역 6 = **절대 수익률** rolling (펀드 자체 안정성)

**1y/3y/5y 의미**:
- 1y rolling: 단기 변동
- 3y rolling: 중기 안정
- 5y rolling: 장기 트렌드

#### 결정 항목 S6-1: 표시 방식

**검토된 옵션**:
- (a) 한 차트에 3개 라인
- (b) 3개 분리 차트
- (c) Tab 전환
- (d) 토글로 다중 선택 (☑ 1y ☐ 3y ☐ 5y)

**결정**: (d) 토글로 다중 선택

**근거**:
1. 사용자 자율 — 관심 시간 척도만 선택 가능
2. 모두 활성화 시 (a) 와 동일 (3 라인)
3. 단일 활성 시 단일 라인 (가독성 ↑)
4. 자유 탐색 모드 (A-2) 부합

#### 결정 항목 S6-2: 비교 대상

**결정**: (c) 다중 토글 적용 (SPY/EW/IVW)

**근거**:
1. 영역 3, 4, 5 일관 — 인터랙션 일관성 원칙
2. 사이드바 토글에 따라 라인 추가 (SPY → SPY 1y/3y/5y / EW → EW 1y/3y/5y 등)
3. 라인 혼잡 시 사용자가 윈도우 / 벤치마크 토글로 조절 가능

#### 결정 항목 S6-3: Y축 / 표시 방식

**검토된 옵션**:
- (a) 절대값 % 라인
- (b) 평균 ± 표준편차 영역
- (c) 분위수 영역 (10%/50%/90%)

**결정**: (a) 절대값 % 라인

**근거**:
1. 단순 — 가독성 ↑
2. 표준편차 / 분위수는 영역 8 (분포 통계 카드) 에서 다룸 → 중복 회피
3. 다중 라인 (윈도우 × 벤치마크) 환경에 적합

#### 결정 항목 S6-4: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- Plotly 기본 zoom + slider ✓
- 시기 클릭 → expand ✓ (Q-Zoom)
- 라인 클릭 → 단독 강조 ✓ (Plotly legend 클릭 기본 기능)

#### 결정 항목 S6-5: 추가 표시

**검토된 옵션**:
- (a) 평균선 (수평선)
- (b) 0% 기준선
- (c) Regime 배경색
- (d) 모두

**결정**: (b) 0% 기준선 + (c) Regime 배경색

**근거**:
1. **0% 기준선**: 음수 영역 (loss period) 시각적 강조
2. **Regime 배경색**: 영역 3 (누적수익 곡선) 일관 — 4개 regime 시각 구분
3. (a) 평균선은 윈도우별 다르므로 혼잡 → 제외

#### 시각화 예시 (확정 사항 조합)

```
[Active Return 분석 (영역 5)]

┌─ Annualized Rolling Return (1y / 3y / 5y) ─────────────┐
│                                                         │
│ ┌─R1 회복기─┬──R2 확장기──┬──R3 변동기──┬─HO─┐         │
│ │ (배경색1) │  (배경색2)  │  (배경색3)  │(4) │         │
│ │                                                       │
│ │ +30%┤    ╱──── Fund 1y                                │
│ │ +20%┤  ╱╱╱─                                           │
│ │ +10%┤  ──── Fund 3y                                   │
│ │  +5%┤────── Fund 5y                                   │
│ │   0%┼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ← 0% 기준선│
│ │  -5%┤                          ╲╱  (Fund 1y)          │
│ │ -10%┤                                                 │
│ │       2010  2014  2018  2022  2025                    │
│ │                                                       │
│ │  Hover: "2020-03: Fund 1y +X.XX%, 3y +X.XX%..."      │
│ │  Click: 시기 → detail / 라인 → 단독 강조              │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│  [Window: ☑1y ☑3y ☑5y]   [☑ SPY] [☐ EW] [☐ IVW]       │
└─────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- 영역 5 Image 2 의 rolling 윈도우 (12m/36m/60m) 와 매칭 → 일관성
- 다중 토글 (윈도우 + 벤치마크) 활성 시 최대 9 라인 (3 윈도우 × 3 벤치마크) → 자유 탐색 모드 활용
- Regime 배경색 = 모든 시계열 차트 공통 패턴 (Overview 영역 3, Performance 영역 6 등)

### 영역 7: Regime 메트릭 간단 비교 (Backtesting Sneak Peek) — 확정

#### 영역 7 통합 배경 (Context)

M-4 결정 (둘 다 — Performance 간단 + Backtesting 자세) 의 **Performance 간단 버전**.

Regime 4개 (R1/R2/R3/HO) + FULL 별 핵심 메트릭만 비교. 자세한 분석 (Sub-events / Active Return 분해 / Factor 등) 은 Backtesting 페이지로 위임.

#### 결정 항목 S7-1: 표시 형식

**검토된 옵션**:
- (a) 표 (table)
- (b) 카드 그리드
- (c) Heatmap
- (d) 막대 차트

**결정**: (c) Heatmap

**근거**:
1. **시각적 즉각 이해** — Regime별 강약 색상 표시
2. 정보 밀도 ↑ — 5행 × 4열 = 20 셀 한 화면
3. (a) 표 보다 가상 투자자 친화 (색상 직관)
4. (b) 카드 그리드는 영역 4 (강점 카드) 와 중복

#### 결정 항목 S7-2: 메트릭 선정

**결정**: (a) 4개 — CAGR / Sortino / MDD / Active Return

**근거**:
1. 핵심 4메트릭으로 다각도 평가 (수익 / 위험조정 / 위험 / 시장 비교)
2. (c) 5개는 Heatmap 컬럼 ↑ → 가독성 ↓
3. (b) 3개는 Active Return 또는 MDD 누락 시 narrative 약화

#### 결정 항목 S7-3: 행 구성

**결정**: (a) R1 / R2 / R3 / HO + FULL = 5행

**근거**:
1. **전체 비교 가능** — Regime 4개 + FULL (전체) 동시 노출
2. 청중이 부분 (Regime) vs 전체 (FULL) 비교 즉시 가능
3. (b) 4행 (FULL 제외) 은 전체 평균 부재로 narrative 약화

#### 결정 항목 S7-4: 비교 방식

**결정**: (c) 다중 토글 적용 (SPY/EW/IVW)

**근거**:
1. 영역 3, 4, 5, 6 일관 — 인터랙션 일관성 원칙
2. 단순 펀드 only 보다 비교 narrative ↑

##### S7-Q1 추가 결정 — Heatmap + 다중 토글 통합 방식

**배경**: Heatmap 은 단일 데이터 시각화 기본 → 다중 비교 표시 방식 결정 필요

**검토된 옵션**:
- (A) 단일 Heatmap (Fund) + Hover 비교
- (B) 탭 전환 (Fund / vs SPY / vs EW / vs IVW)
- (C) Diff Heatmap (Active Return 컬럼만 다중)
- (D) 다중 Heatmap 동시 표시

**결정**: (B) 탭 전환

**근거**:
1. **각 비교가 명확** — 탭으로 비교 목적 분리
2. (A) hover 의존은 다중 토글 의미 약화
3. (D) 다중 Heatmap 은 4개 동시 → 시각적 압도
4. **사용자가 의도적으로 비교 선택** — 자유 탐색 모드 부합

**탭 구성**:
- **Fund** 탭: Fund 절대값 Heatmap
- **vs SPY** 탭: (Fund - SPY) Diff Heatmap (사이드바 SPY 활성 시 노출)
- **vs EW** 탭: (Fund - EW) Diff Heatmap (EW 활성 시 노출)
- **vs IVW** 탭: (Fund - IVW) Diff Heatmap (IVW 활성 시 노출)

#### 결정 항목 S7-5: 인터랙션

**결정**:
- Regime 행 클릭 → Backtesting 페이지로 navigation ✓
- 메트릭 컬럼 hover → 정의 tooltip ✓
- HO 행 강조 (배경색) ✓
- sparkline = 보류

**근거**:
1. **Backtesting 자연스러운 navigation** — Performance 간단 → Backtesting 자세
2. **HO 행 강조** = 학술 정직성 (HO 부진 명시 노출)
3. sparkline = 셀 단위 시계열은 구현 복잡 + Heatmap 색상과 중복

#### 시각화 예시 (확정 사항 조합)

```
┌─ Regime 메트릭 비교 (Heatmap) ──────────────────────────┐
│                                                         │
│ [Fund | vs SPY | vs EW | vs IVW]  ← 탭 (활성 토글에 따라)│
│                                                         │
│ Regime      | CAGR  | Sortino | MDD  | Active │        │
│ R1 회복기   | ███   | ███     | ░    | ██     │        │
│ R2 확장기   | ████  | ███     | ░    | ██     │        │
│ R3 변동기   | ██    | ██      | ░░░  | █      │        │
│ HO 홀드아웃 │ █     │ █       │ ░    │ ░░░    │ ← 배경색│
│ FULL        | ███   | ███     | ░░   | ██     │        │
│                                                         │
│  색상: 진하기 = 메트릭 강도 (양수: green / 음수: red)   │
│  Click: Regime 행 → Backtesting 페이지                  │
│  Hover: 메트릭 → 정의 tooltip                           │
│                                                         │
│  [자세한 분석 보기 → Backtesting 페이지 ]               │
└─────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- Backtesting 페이지의 Regime 표는 **자세한 버전** (메트릭 10+ / Sub-events / Factor 분해 등) — 별도 결정
- Heatmap 색상 = green-red diverging colormap (Plotly `RdYlGn` 또는 custom)
- 탭 동적 노출 = 사이드바 토글 활성 벤치마크에 따라 (SPY 활성 시 vs SPY 탭 등)

### 영역 8: 분포 통계 카드 (Distribution Stats) — 확정

#### 영역 8 통합 배경 (Context)

펀드 수익률의 분포 특성 (왜도/첨도/꼬리비율) 시각화. 메트릭 풀 (C-2) 의 Pool-9 활용. **일별 + 월별 모두** (사용자 결정).

**왜 분포 통계인가**: 펀드 정체성 (Adaptive VolControl) 의 hidden risk 검증 — Sharpe 가 같아도 fat tail / 비대칭 분포는 위험 ↑.

#### 결정 항목 S8-1: 표시 형식

**검토된 옵션**:
- (a) 카드 그리드
- (b) 표
- (c) 카드 + 히스토그램/KDE
- (d) Heatmap

**결정**: (c) 카드 + 히스토그램 + KDE 차트

**근거**:
1. 통계 + 시각 동시 (가상 투자자 직관 + 학술 정확성)
2. KDE (Kernel Density Estimation) 추가 = 부드러운 분포 시각
3. (a) 카드만은 시각적 약함, (b) 표는 청중 친화 ↓

#### 결정 항목 S8-2: 일별 / 월별 표시 방식

**결정**: (a) 탭 전환 (Monthly / Daily)

**근거**:
1. 두 셋 동시 표시는 영역 분량 ↑
2. 사용자가 의도적으로 시간 단위 선택
3. 탭 전환 = 영역 7 패턴 일관

#### 결정 항목 S8-3: 통계 메트릭

**검토된 옵션**:
- (a) 3개 그대로 (Skewness / Excess Kurtosis / Tail Ratio)
- (b) +Jarque-Bera test
- (c) +Hill estimator
- (d) +둘 다

**결정**: (a) 3개 그대로

**근거**:
1. **대시보드 적합성**: Jarque-Bera (정규분포 검정) 와 Hill (extreme value theory) 는 가상 투자자에게 난해
2. **위치 분리 권고**:
   - **Jarque-Bera**: About / Methodology 페이지 (학술 검정 narrative)
   - **Hill estimator**: Risk Metrics 페이지 (tail risk 분석)
3. Performance 영역 8 은 가상 투자자 친화 카드 형식 유지
4. 추후 Risk Metrics / About 페이지 결정 시 Jarque-Bera / Hill 추가 검토

#### 결정 항목 S8-4: 비교

**결정**: (c4) Tab 전환 (vs SPY / vs EW / vs IVW)

**근거**:
1. 영역 7 (Regime Heatmap) Tab 전환 패턴과 일관 — 인터랙션 일관성 원칙
2. 분포 통계 카드는 카드 안에 다중 값 표시 시 가독성 ↓
3. 사용자가 의도적으로 비교 선택 — 자유 탐색 모드 부합
4. 사이드바 토글 활성에 따라 Tab 동적 노출 (SPY 활성 시 vs SPY 탭 등)

#### 결정 항목 S8-5: 추가 시각화

**결정**: (a) 히스토그램 + KDE (S8-1 에서 결정)

**근거**:
1. 히스토그램 = 분포 빈도 시각
2. KDE 라인 오버레이 = 부드러운 분포 모양
3. (b) Q-Q plot 은 정규분포 비교 → About 페이지로
4. (c) Box plot 은 분포 요약 → 카드 형식과 어울림 ↓

#### 시각화 예시 (확정 사항 조합)

```
[Regime Heatmap (영역 7)]

┌─ Distribution Statistics ───────────────────────────────┐
│                                                         │
│ [Tab: 월별 (Monthly) | 일별 (Daily)]                    │
│ [Tab: vs SPY | vs EW | vs IVW]  ← 사이드바 토글 활성에 따라│
│                                                         │
│ ┌─────────────┬─────────────┬─────────────┐           │
│ │ Skewness    │ Excess      │ Tail Ratio  │           │
│ │ (왜도)      │ Kurtosis    │ (꼬리 비율) │           │
│ │             │ (초과첨도)  │             │           │
│ │ Fund: -0.XX │ Fund: +X.XX │ Fund: X.XX  │           │
│ │ SPY:  -0.XX │ SPY:  +X.XX │ SPY:  X.XX  │           │
│ │             │             │             │           │
│ │ ┄┄┄ KDE     │ ┄┄┄ KDE     │ ┄┄┄ KDE     │           │
│ │ ▁▂▆█▃▁     │ ▁▁▆█▆▁▁     │ ▂▆█▃▁       │           │
│ │             │             │             │           │
│ │ ⓘ 음수 =   │ ⓘ 양수 =   │ ⓘ >1 = 우측 │           │
│ │  좌측 꼬리 │  fat tail   │  비대칭     │           │
│ └─────────────┴─────────────┴─────────────┘           │
│                                                         │
│  ※ 일별 통계 = ~N 영업일 / 월별 통계 = 192 개월         │
└─────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- **Jarque-Bera test → About / Methodology 페이지로 이동** (학술 검정 narrative)
- **Hill estimator → Risk Metrics 페이지로 이동** (tail risk 분석)
- 히스토그램 + KDE = Plotly `histnorm='probability density'` + KDE overlay
- 일별 portfolio return 산출 = 월별 weight × 일별 ticker return (forward-fill within month)

---

### Performance 페이지 — 전체 확정 (영역 1~9)

#### 페이지 시각화 통합 (확정 사항 모두 조합)

```
┌────────────────────────────────────────────────────────────────┐
│ [영역 1: Header — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 2: Sub-header 카드/배너 — Cobalt Blue 톤]                 │
│ Performance Analysis (성과 분석)                                │
│ 펀드의 수익성 / 시장 비교 / regime별 분해.                       │
│ 사이드바: 기간 + 비교 벤치마크 토글                             │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 3: Performance Summary KPI 5개 — 액티브 강조]              │
│ CAGR / Sortino / Sharpe / IR / Active Return                   │
│ 사이드바 토글 반응 + 다중 벤치마크 delta                        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 4: Annual Returns 막대 (Image 6) — 사이드바이]            │
│ Fund vs 활성 벤치마크 + 평균선 + Q-Zoom                         │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 5: Active Return 분석 (Image 1 + Image 2) — 위아래]       │
│ Image 1: 연도별 차이값 / Image 2: Rolling + Tracking Error     │
│ 윈도우 토글 (12m/36m/60m) + 다중 벤치마크 + 이중 축             │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 6: Annualized Rolling Return (1y/3y/5y)]                 │
│ 윈도우 토글 (다중 선택) + 다중 벤치마크                         │
│ 0% 기준선 + Regime 배경색                                      │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 7: Regime 메트릭 Heatmap — 5행 × 4메트릭]                 │
│ Tab: Fund / vs SPY / vs EW / vs IVW                            │
│ Click Regime → Backtesting / HO 행 강조                        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 8: 분포 통계 카드 + KDE — Skew/Kurt/Tail]                 │
│ Tab 전환 (월별/일별) + Tab 전환 (vs SPY/EW/IVW)                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 9: Footer — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘
```

#### Performance 페이지 결정 결과 / 함의

- 9 영역 모두 확정 → 구현 시 명확한 와이어프레임
- **인터랙션 일관성 원칙** 강력 적용:
  - 다중 벤치마크 토글 (사이드바 SPY/EW/IVW) → 영역 3-7 모든 차트/카드 영향
  - Q-Zoom (1+2+3 조합) → 모든 시계열 차트 적용
  - Regime 배경색 → 영역 6 (Performance) + Overview 영역 3 (Cumulative Return) 일관
  - Tab 전환 (vs SPY/EW/IVW) → 영역 7 (Regime Heatmap) + 영역 8 (분포 통계) 일관
- 다음 페이지 (Risk Metrics / Holdings 등) 결정 시 동일 패턴 활용
- **Jarque-Bera → About / Methodology** / **Hill estimator → Risk Metrics** 메모

### Performance 페이지 → Risk Metrics 페이지로 진행

---


---

[← 02_overview.md](02_overview.md) | [04_risk_metrics.md](04_risk_metrics.md) →
