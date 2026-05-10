# Performance 페이지 — 와이어프레임

> **관련 decisionlog**: `03_performance.md`
> **상태**: 확정
> **결정 수**: 9 영역 (메타 M-1 ~ M-4 + 영역 1~9)

---

## 페이지 역할 정의

성과 분석 (Performance Analysis) 전담 페이지. **Overview Hero KPI / 누적수익 곡선의 깊이 있는 버전**.

**핵심 차별점**:
- vs Overview: 5초 인상 → 깊이 있는 분석
- vs Risk Metrics: 수익 위주 (Sortino/Sharpe/CAGR/Active Return) ↔ 위험 위주 (VaR/CVaR/Beta/Drawdown)

---

## 페이지 영역 구조 (시선 흐름)

```
1. Header                           (Overview 와 동일)
2. Sub-header                       (페이지 컨텍스트 + 토글 안내)
3. Performance Summary KPI 5개      (액티브 운용 강조 — Overview Hero 차별화)
4. Annual Returns 막대              (Image 6 — 사이드바이 막대)
5. Active Return 분석               (Image 1 + Image 2 — 위아래)
6. Annualized Rolling Return        (1y/3y/5y 토글)
7. Regime 메트릭 간단 비교          (Heatmap, Backtesting sneak peek)
8. 분포 통계 카드                   (Skewness/Kurtosis/Tail Ratio)
9. Footer                           (Overview 와 동일)
```

---

## 영역별 와이어프레임

### 영역 1: Header — Overview 동일

→ `02_common.md` 의 `render_page_header()` 호출

---

### 영역 2: Sub-header

**결정사항** (S2-1 ~ S2-3):
- 텍스트: (b) 정의 + 토글 안내
- 디자인: (b) 카드/배너 형식 (Cobalt Blue 톤)
- 한/영 병기: (b)

**텍스트 안**:
```
Performance Analysis (성과 분석)
펀드의 수익성, 시장 비교, regime별 분해 분석.
사이드바에서 기간 (FULL / TEST / HO) 선택 시 차트 자동 갱신.
비교 벤치마크 (SPY / EW / IVW) 도 선택 가능.
```

→ `02_common.md` 의 `render_subheader()` 호출

---

### 영역 3: Performance Summary KPI 5개

**결정사항** (S3-1 ~ S3-4 + 추가 결정):
- KPI: (b) **CAGR / Sortino / Sharpe / IR / Active Return** (액티브 운용 강조)
- 표시 기간: (b) 사이드바 토글 반응
- 카드 디자인: (b) 더 단순 (큰 숫자만, sparkline 없음)
- 벤치마크 delta: (b) 모든 카드 + **다중 벤치마크 토글 통합** (사이드바)

**다중 벤치마크 토글 작동 방식**:
- 사이드바: `[☑ SPY] [☐ EW] [☐ IVW]` 다중 체크박스
- 기본: SPY 만 활성
- 카드 안: 선택된 대상마다 작은 delta 행 동적 추가

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] `st.columns(5)` 반응형
- [ ] 사이드바 `st.session_state.period` 반응 → 기간별 KPI 재계산
- [ ] 사이드바 `show_spy / show_ew / show_ivw` 활성에 따라 delta 행 동적 추가
- [ ] IR / Active Return 산출 시 SPY 외 EW / IVW 도 별도 산출

---

### 영역 4: Annual Returns 막대 (Image 6)

**결정사항** (S4-1 ~ S4-5 + Q-Zoom):
- 차트 종류: (a) 사이드바이 막대 (Image 6 동일)
- 비교: (a) 다중 토글 적용
  - SPY 만 → 2 막대 그룹 (Fund + SPY)
  - SPY + EW → 3 막대 그룹
  - SPY + EW + IVW → 4 막대 그룹
- 인터랙션: 모두 채택 + **Q-Zoom 결정 (★ 모든 페이지 영향)**:
  - 옵션 1: Plotly 기본 X축 zoom
  - 옵션 2: Range slider
  - 옵션 3: 연도 클릭 → 월별 별도 차트 expand (Streamlit 1.27+ `on_select='rerun'`)
- Y축 라벨: (b) 막대만 (hover)
- 평균선: (a) 평균선 표시

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] Plotly `go.Bar` (사이드바이 그룹)
- [ ] 다중 벤치마크 토글 시 막대 그룹 동적 추가
- [ ] 평균선 = `add_hline()` (펀드 평균 + SPY 평균)
- [ ] 양수/음수 색상 자동 (펀드 막대 marker.color)
- [ ] Hover tooltip (연도 + 각 수익률 + delta)
- [ ] 연도 정렬 토글 (연대순 / 수익률순)
- [ ] Q-Zoom: 연도 클릭 → 월별 차트 expand (`render_zoomable_chart()` 활용)

---

### 영역 5: Active Return 분석 (Image 1 + Image 2)

**결정사항** (S5-1 ~ S5-5):
- 두 차트 표시: (a) 위아래 배치 (Image 1 위 / Image 2 아래)
- Image 1 비교: (a) 다중 토글 적용
- Image 2 Rolling 윈도우: (b) 토글 (12m / 36m / 60m)
- Tracking Error 표시: (a) 이중 축 (좌축 = Active Return / 우축 = TE)
- 인터랙션: 모두 채택

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] Image 1: Plotly `go.Bar` (Active Return 막대) + 양수/음수 색상
- [ ] Image 2: Plotly `make_subplots(specs=[[{"secondary_y": True}]])` (이중 축)
- [ ] Rolling 윈도우 토글 (12m / 36m / 60m) — 영역 6 윈도우와 매칭
- [ ] Tracking Error accent 색상 (우축)
- [ ] Q-Zoom: 시기 클릭 → expand

---

### 영역 6: Annualized Rolling Return (1y / 3y / 5y)

**결정사항** (S6-1 ~ S6-5):
- 표시 방식: (d) 토글로 다중 선택 (☑ 1y ☐ 3y ☐ 5y)
- 비교: (c) 다중 토글 적용
- Y축: (a) 절대값 % 라인
- 인터랙션: 모두 채택
- 추가 표시: (b) 0% 기준선 + (c) Regime 배경색

**시각화 예시**:

```
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
│ │  Click: 시기 → detail / 라인 → 단독 강조              │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│  [Window: ☑1y ☑3y ☑5y]   [☑ SPY] [☐ EW] [☐ IVW]       │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] 윈도우 다중 체크박스 (`st.checkbox` × 3)
- [ ] 활성 윈도우 + 활성 벤치마크 모두에 대해 라인 추가 (최대 9 라인)
- [ ] 0% 기준선 = `add_hline(y=0)`
- [ ] Regime 배경색 (`add_regime_backgrounds()` 헬퍼)
- [ ] Plotly legend 클릭 → 단독 강조 (기본 기능)

---

### 영역 7: Regime 메트릭 간단 비교 (Heatmap)

**결정사항** (S7-1 ~ S7-5 + S7-Q1):
- 표시 형식: (c) Heatmap
- 메트릭: (a) 4개 — CAGR / Sortino / MDD / Active Return
- 행 구성: (a) R1 / R2 / R3 / HO + FULL = 5행
- 비교 방식: (c) 다중 토글
- **Heatmap + 다중 토글 통합** (S7-Q1): (B) 탭 전환 (Fund / vs SPY / vs EW / vs IVW)
- 인터랙션: Regime 행 클릭 → Backtesting / 메트릭 hover / HO 행 강조

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] Plotly `go.Heatmap` (또는 `imshow`)
- [ ] Tab 동적 노출 (`st.tabs`) — 활성 벤치마크에 따라
- [ ] 색상: green-red diverging colormap (`RdYlGn` 또는 custom)
- [ ] HO 행 배경색 강조 (학술 정직성 — HO 부진 명시)
- [ ] Regime 행 클릭 → Backtesting 페이지 navigation (`st.switch_page()`)
- [ ] 메트릭 hover → 정의 tooltip

---

### 영역 8: 분포 통계 카드 (Distribution Stats)

**결정사항** (S8-1 ~ S8-5):
- 표시 형식: (c) 카드 + 히스토그램 + KDE 차트
- 일별/월별: (a) 탭 전환
- 통계 메트릭: (a) 3개 (Skewness / Excess Kurtosis / Tail Ratio)
  - **Jarque-Bera** → Methodology 영역 7 로 이동 (LSTM 정당화)
  - **Hill estimator** → Risk Metrics 영역 8 로 이동
- 비교: (c4) Tab 전환 (vs SPY / vs EW / vs IVW)
- 추가 시각화: (a) 히스토그램 + KDE

**시각화 예시**:

```
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

**구현 체크리스트**:
- [ ] `st.tabs(["월별", "일별"])` 외부 + `st.tabs([vs SPY/EW/IVW])` 내부
- [ ] 3 카드 구성 (Skewness / Kurtosis / Tail Ratio)
- [ ] 각 카드 내 KDE mini chart (Plotly `go.Scatter`)
- [ ] 일별 portfolio return 산출 = 월별 weight × 일별 ticker return (forward-fill within month)

---

### 영역 9: Footer — Overview 동일

→ `02_common.md` 의 `render_footer()` 호출

---

## 페이지 데이터 의존성

- monthly_panel.csv (월별 fund/SPY return)
- daily_returns.pkl (일별 분포 통계 + sparkline)
- results/mat_eq_eq_raw_pap.pkl (펀드 결과)
- EW + IVW baseline (`lib/data_loader.py` 캐시)

---

## 메트릭 (C-2 풀)

- Pool-1 수익성: CAGR / Cumulative Return / Arithmetic Mean / Annualized Rolling
- Pool-2 위험조정수익: Sortino / Sharpe / Calmar / IR / M²
- Pool-5 시장 비교: Active Return / Win Rate / Up/Down Capture
- Pool-7 Regime별 성과: Sortino R1/R2/R3/HO
- Pool-9 분포 통계: Skewness / Kurtosis / Tail Ratio

---

## 인터랙션 / 토글 적용

| 영역 | 사이드바 토글 영향 | Q-Zoom |
|---|---|---|
| 영역 3 (KPI) | ✓ 기간 + 다중 벤치마크 delta | ✗ |
| 영역 4 (Annual Returns) | ✓ 다중 벤치마크 (막대 그룹) | ✓ 연도 클릭 → 월별 |
| 영역 5 (Active Return) | ✓ 다중 벤치마크 + 윈도우 토글 | ✓ 시기 클릭 |
| 영역 6 (Rolling Return) | ✓ 다중 벤치마크 + 윈도우 다중 선택 | ✓ 시기 클릭 |
| 영역 7 (Regime Heatmap) | ✓ Tab 동적 노출 | Regime 행 클릭 → Backtesting |
| 영역 8 (분포 통계) | ✓ Tab 동적 노출 (vs SPY/EW/IVW) | ✗ |

---

## 페이지 구현 우선순위

- **Phase 1 (MVP, 1-2주)**: Performance 페이지 (Phase 1 의 핵심 — 4가지 메시지 중 액티브 운용)

---

## 결과 / 함의

- **인터랙션 일관성 원칙** 강력 적용 (다중 벤치마크 + Q-Zoom + Regime 배경 + Tab 전환)
- 모든 시계열 차트에 Q-Zoom 패턴 일관 적용
- Jarque-Bera → Methodology 영역 7 (LSTM 정당화) / Hill estimator → Risk Metrics 영역 8

---

[← 02_simulator.md](02_simulator.md) | [04_risk_metrics.md →](04_risk_metrics.md)
