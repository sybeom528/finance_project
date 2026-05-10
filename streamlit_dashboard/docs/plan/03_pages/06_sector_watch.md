# Sector Watch 페이지 — 와이어프레임

> **관련 decisionlog**: `06_sector_watch.md`
> **상태**: 확정 (★ HO 정당화 narrative 영역 8)
> **결정 수**: 8 영역 (메타 Sector M-1~M-4 + 영역 1~9 위치)

---

## 페이지 역할 정의

섹터 단위 비중 + 분석 전담 페이지. 레퍼런스 PortfolioX360 의 **Sector Watch** 화면 패턴.

**vs Holdings 페이지**:
- Holdings = 개별 종목 위주
- Sector Watch = **섹터 (GICS 11개) 단위 위주**

### ★★★ HO 정당화 narrative (사용자 강조)

**핵심 요구사항**: HO 24m 구간 (2024-2025) 의 SPY 대비 펀드 성능 저하 정당화 가능해야 함.

**narrative**:
- 펀드 = **Sector-balanced** (분산 운용) → 장기 (R1/R2/R3) 안정적 위험조정 수익
- HO 24m: SPY = IT 집중 (AI rally) → +21.2% / 펀드 = IT under-weight → +8.3% (열위)
- "장기 분산의 가치 vs 단기 IT 집중의 trade-off"

→ Sector Watch 페이지가 이 narrative 의 중심 (특히 영역 8)

---

## 페이지 영역 구조 (시선 흐름)

```
1. Header                       (Overview 동일)
2. Sub-header                   (HO narrative 핵심 명시)
3. Sector Summary KPI 5개       (HHI / Avg|Tilt| / Active Bets / Most Over / Most Under)
4. Sector Treemap (현재 비중)   (PortfolioX360 패턴)
5. Sector Decomposition 표      (PortfolioX360 패턴, 9 컬럼)
6. Sector Tilt vs SPY           (Tornado Chart — 액티브 운용 핵심)
7. Sector 시계열 변화           (Sector Rotation 분석)
8. ★★★ HO 24m 분석 + 정당화     (IT/AI Rally narrative + Markowitz 1952 인용)
9. Footer                       (Overview 동일)
```

---

## 영역별 와이어프레임

### 영역 1: Header — Overview 동일

→ `02_common.md` 의 `render_page_header()` 호출

---

### 영역 2: Sub-header (HO narrative 명시)

**결정사항** (S2-1):
- (b) HO narrative 명시 — 사용자 강조 사항

**텍스트 안**:
```
Sector Watch (섹터 분석)
섹터 비중 / 분산 / 시장 비교 분석.
HO 24m (2024-2025) sector rotation 영향과 펀드의
sector 분산 운용의 양면성 분석 포함.
사이드바에서 기간 + 비교 벤치마크 토글 가능.
```

→ `02_common.md` 의 `render_subheader()` 호출

---

### 영역 3: Sector Summary KPI 5개

**결정사항** (Sector M-3 + S3-2 ~ S3-5):
- 5 KPI:
  1. Sector HHI
  2. Avg |Tilt| vs SPY
  3. Number of Active Bets (|Tilt| > 1%)
  4. Most Overweight Sector (이름 + Tilt%)
  5. **Most Underweight Sector (이름 + Tilt%) — HO 정당화 narrative 직접 연결**
- 표시 기간: (c) Latest + 사이드바 토글 평균
- 카드 디자인: (a) 단순 + Most Over/Under 만 다중 값
- 벤치마크 비교: (b) HHI 만 vs SPY
- Hover tooltip: (a) 포함

**시각화 예시**:

```
[Header — Overview 동일]

┌─ ℹ️ Sector Watch (섹터 분석) ───────────────────────────┐
│ 섹터 비중 / 분산 / 시장 비교 분석.                        │
│ HO 24m (2024-2025) sector rotation 영향과 펀드의 sector  │
│ 분산 운용의 양면성 분석 포함.                             │
│ 사이드바에서 기간 + 비교 토글 가능.                       │
└──────────────────────────────────────────────────────────┘

[Latest snapshot: 2025-12]  [기간 평균: TEST 168m]
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ Sector   │ Avg|Tilt|│ Active   │ Most     │ Most     │
│ HHI      │ vs SPY   │ Bets     │ Overweight│ Underwght│
│          │          │          │          │          │
│ X.XX     │ X.XX%    │ XX       │Healthcare│Info Tech │
│ (X.XX 평균)│(X.XX% 평균)│(XX 평균)│ +X.X%    │ -X.X% ★ │
│ vs SPY   │          │          │          │ (HO     │
│ ▼ better │          │          │          │  정당화)│
│ⓘHHI     │ⓘTilt    │ⓘBets    │ⓘOver    │ⓘUnder    │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

**구현 체크리스트**:
- [ ] Most Underweight 카드 = IT 가능성 ↑ HO 시기 (★ 표기)
- [ ] HO narrative 자동 강화 (Most Underweight 에 IT 노출)
- [ ] Tooltip 표준화 (`lib/tooltips.py`)

---

### 영역 4: Sector Treemap (현재 비중)

**결정사항** (S4-1 ~ S4-5):
- Treemap 차원: (d) 토글
  - 기본: (a) 면적=비중, 색상=섹터 (PortfolioX360 일관)
  - (b) Tilt 색상 = HO narrative 강화 (IT 빨강 = under-weight)
  - (c) 12m 수익률 색상
- Sub-sector: (b) Sector + 종목 drill-down
- 비교: (d) 토글
  - 기본: (b) 좌우 분할 (펀드 vs SPY 직관 비교)
  - (c) Tilt Treemap (Diff)
- 표시 기간: (c) 시점 슬라이더
- 인터랙션: 모두 채택

**시각화 예시**:

```
[Sector Summary KPI 5개]

┌─ Sector Treemap ────────────────────────────────────────┐
│                                                         │
│ 색상: [면적=비중,섹터 ▼ / 비중,Tilt / 비중,수익률]     │
│ 보기: [Fund only / Fund vs SPY 좌우 ▼ / Tilt Treemap]  │
│ 시점: 2010 ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2025      │
│                                                         │
│ ┌─ Adaptive VolControl Fund ─┐ ┌─ SPY (S&P 500) ────┐ │
│ │                            │ │                    │ │
│ │ ┌─Healthcare─┬─Tech─┐      │ │ ┌──Tech──┬───┐    │ │
│ │ │ XX%         │ XX%  │     │ │ │  XX%   │HC │    │ │
│ │ ├─Financials─┼─Cons─┤     │ │ ├──Fin───┼───┤    │ │
│ │ │ XX%         │ XX%  │     │ │ │  XX%   │ X%│    │ │
│ │ ├─Industry─┬─Energy─┤     │ │ ├────────┴───┤    │ │
│ │ │ XX%       │ XX%    │    │ │ │ Industry XX% │  │ │
│ │ └───────────┴────────┘     │ │ └──────────────┘  │ │
│ └────────────────────────────┘ └────────────────────┘ │
│                                                         │
│ Hover: "Healthcare: Fund X%, SPY Y%, Tilt +Z%"         │
│ 섹터 Click → 해당 섹터 종목 list expand                 │
│ [⬇ Download]                                           │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Plotly `px.treemap` (path = ['sector', 'ticker'] for drill-down)
- [ ] 좌우 분할 = `make_subplots(cols=2)` 또는 `st.columns(2)`
- [ ] Tilt Treemap = (Fund weight - SPY weight) 색상 (red = under, green = over)
- [ ] 시점 슬라이더 = `st.slider`
- [ ] Q-Zoom: 섹터 클릭 → 종목 list expand

---

### 영역 5: Sector Decomposition 표

**결정사항** (S5-1 ~ S5-5):
- 컬럼: (c) 풍부 9 컬럼:
  1. Sector
  2. Weight
  3. Tilt vs SPY
  4. 12m Return
  5. # Holdings
  6. Volatility
  7. Beta
  8. Sharpe
  9. Contribution
- 비교: (c) 펀드 값 + Tilt 컬럼
- 정렬: (c) 기본 Weight 내림차순 + 사용자 정렬
- 인터랙션: 모두 채택
- 시각 보강: (a) Weight 막대 + (b) Tilt 색상 코딩

**시각화 예시**:

```
[Sector Treemap (영역 4)]

┌─ Sector Decomposition ──────────────────────────────────┐
│ [정렬: Weight 내림차순 ▼]    [⬇ CSV]                    │
│                                                          │
│ Sector       │Weight       │Tilt    │12m R│#H │Vol│β│Shp│Contr│
│ Healthcare   │████ XX.X%   │+X.X% 🟢│+X% │XX│XX%│X│X.X│+X% │
│ Info Tech    │███  XX.X%   │-X.X% 🔴│+X% │XX│XX%│X│X.X│+X% │ ← Tilt red
│ Financials   │███  XX.X%   │+X.X% 🟢│+X% │XX│XX%│X│X.X│+X% │
│ Cons Disc    │██   XX.X%   │-X.X% 🔴│+X% │XX│XX%│X│X.X│+X% │
│ Industry     │██   XX.X%   │+X.X% 🟢│+X% │XX│XX%│X│X.X│+X% │
│ Energy       │█    XX.X%   │-X.X% 🔴│+X% │XX│XX%│X│X.X│+X% │
│ ...                                                      │
│                                                          │
│ Hover: 섹터 detail / Click: 종목 list expand            │
│ 컬럼 헤더 Click: 정렬                                   │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] `st.dataframe` (정렬 + 검색 자동)
- [ ] Weight 막대 = `st.column_config.ProgressColumn`
- [ ] Tilt 색상 = `pandas.Styler.applymap` (red/green)
- [ ] CSV 다운로드

---

### 영역 6: Sector Tilt vs SPY (Tornado Chart)

**결정사항** (S6-1 ~ S6-5):
- 차트 종류: (a) Tornado Chart
- 정렬: (a) Tilt 크기순 (Most Over → Most Under)
- 표시 기간: (c) Latest + 토글 평균 (둘 다)
- 인터랙션: 모두 채택
- 추가 표시: (c) 0% 기준선 + (a) ±1% / ±5% 임계선

**시각화 예시**:

```
[Sector Decomposition (영역 5)]

┌─ Sector Tilt vs SPY (Active Bets) ──────────────────────┐
│                                                         │
│ Latest snapshot: 2025-12  |  TEST 평균: 168m            │
│                                                         │
│ Most Overweight                  Most Underweight       │
│                                                         │
│ Healthcare  │██████ +X.X%   │                           │
│ Industry    │████ +X.X%     │                           │
│ Financials  │███ +X.X%      │                           │
│ ...                                                     │
│                            ┃ 0% 기준선                  │
│ ┄┄ ±1% ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ ━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                            ┃                            │
│ ...                                                     │
│ Cons Disc        │█── -X.X%│                            │
│ Energy           │██── -X.X%│                            │
│ Info Tech        │███───── -X.X% ★ (HO 정당화 narrative)│
│                                                         │
│ ┄┄ ±1% 임계선   ━━ 0% 기준선   ┄┄ ±5% 임계선          │
│                                                         │
│ Hover: "Info Tech: Fund X%, SPY Y%, Tilt -Z%"           │
│ 섹터 Click → 종목 list expand                           │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Plotly `go.Bar` (orientation='h', sorted)
- [ ] 양수/음수 색상 = `marker.color` 동적 (green/red)
- [ ] 0% 기준선 + ±1%/±5% 임계선 = `add_vline`
- [ ] HO narrative 강화 — IT under-weight 강조

---

### 영역 7: Sector 시계열 변화 (Sector Rotation)

**결정사항** (S7-1 ~ S7-5):
- 차트 종류: (e) 토글 (Stacked Area + Multi-line)
  - 기본: Stacked Area (구성 변화 직관)
  - 옵션: Multi-line (개별 섹터 추적)
- 비교: (d) 토글
  - 기본: (a) 펀드 only
  - (b) 펀드 vs SPY 좌우 분할
  - (c) Tilt 시계열 (Fund - SPY)
- 시간 단위: (c) 토글 (월별 / 분기별)
- 인터랙션: 모두 채택
- 추가 표시: (d) 모두 (Regime 배경 + 이벤트 annotation + AI rally 강조)

**시각화 예시**:

```
[Sector Tilt (영역 6)]

┌─ Sector 시계열 변화 (Sector Rotation) ──────────────────┐
│                                                         │
│ [Chart: Stacked Area ▼ / Multi-line]                    │
│ [View: Fund only ▼ / Fund vs SPY 좌우 / Tilt 시계열]   │
│ [Time: 월별 / 분기별 ▼]                                │
│                                                         │
│ ┌─R1─┬───R2────┬──R3──┬─HO─┐ (Regime 배경색)          │
│                                                         │
│ 100%┤████████████████████████████████                   │
│  80%┤▓▓▓ Tech (▓)                  ▓▓▓                  │
│  60%┤░░░ Healthcare (░)         ░░░░░░░                │
│  40%┤▒▒▒ Financials              ▒▒▒                   │
│  20%┤▓▓▓ Cons Disc                                     │
│   0%┴───────────────────────────────                   │
│      2010    2014    2018    2022    2025               │
│                                                         │
│ ▼ Annotation: "2020-03 COVID"  "2022 Bear"             │
│ ▼ "2024-12 AI Rally / IT Rotation" ★ HO 정당화          │
│                                                         │
│ Hover: "2024-Q1: Tech 18%, Healthcare 22%..."           │
│ 섹터 Click: 단독 강조 / 시점 Click: expand              │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Stacked Area = Plotly `go.Scatter(stackgroup='one')`
- [ ] Multi-line = Plotly `go.Scatter` 다중 trace
- [ ] Tilt 시계열 = (Fund - SPY) 시계열
- [ ] 이벤트 annotation = COVID / 2022 Bear / 2024 IT Rotation
- [ ] AI rally 강조 = 영역 8 narrative 와 직접 연결

---

### 영역 8: ★★★ HO 24m 분석 + 정당화 narrative

**결정사항** (S8-1 ~ S8-6):
- Sub-구조: (c) 3개 차트 + narrative 박스
- 차트 1 (S8-2): (c) **IT Mcap (SPY) + Fund IT Tilt 이중 축**
  - 좌축: SPY IT 비중 (%) — 시장 IT 집중도
  - 우축: Fund IT Tilt (%) — 펀드 vs SPY 차이
- 차트 2 (S8-3): (a) HO 24m Sector Contribution Tornado
- 차트 3 (S8-4): (a) Regime 별 Sector HHI 추세
- narrative 박스 (S8-5): (b) **학술 narrative + 인용 (Markowitz 1952)**
- 결론 메시지 (S8-6): (a) **장기 분산의 가치 (자신감)**

**핵심 메시지**:
- HO 24m: SPY +21.2% / 펀드 +8.3% (열위 -12.9%p)
- 원인: SPY 의 IT 집중 (AI rally) → 펀드의 IT under-weight 가 불리하게 작용
- 정당화: "장기 분산의 가치 vs 단기 IT 집중의 trade-off"

**결론 박스 텍스트** (확정):
```
✅ 결론: 장기 Sector 분산 운용의 가치

Markowitz (1952) 의 평균-분산 이론과 Fama-French (1992) 의
factor diversification 관점에서, sector 분산 운용은 idiosyncratic
risk 를 줄이고 장기 위험조정 수익 (Sharpe / Sortino) 향상에 기여
합니다.

본 펀드는 R1 (회복기) / R2 (확장기) / R3 (변동기) 168개월 학습
구간에서 일관된 sector 분산 운용을 통해 우수한 위험조정 성과를
입증했습니다.

HO 24m 의 단기 underperform 은 일시적 sector concentration 시기의
trade-off 이며, 장기 분산 운용의 근본 가치를 손상시키지 않습니다.
```

**보조 학술 narrative 박스** (S8-5 결정):
```
ℹ️ Markowitz (1952) 의 평균-분산 이론에 따르면 sector 분산은
idiosyncratic risk 를 줄이고 장기 위험조정 수익을 향상시킵니다.
그러나 단기 sector concentration 시기 (예: 2024 AI rally) 에는
일시적 underperform 가능. 펀드는 이 trade-off 를 의도한 분산
운용입니다.
```

**시각화 예시**:

```
[Sector 시계열 (영역 7)]

┌─ ★★★ HO 24m 분석 + 정당화 narrative ────────────────────┐
│                                                          │
│ ┌─ ℹ️ 학술 narrative 박스 (Markowitz 1952 인용) ────┐ │
│ │ Markowitz (1952) 의 평균-분산 이론에 따르면 sector  │ │
│ │ 분산은 idiosyncratic risk 를 줄이고 장기 위험조정   │ │
│ │ 수익을 향상시킵니다. 그러나 단기 sector concentration│ │
│ │ 시기 (예: 2024 AI rally) 에는 일시적 underperform   │ │
│ │ 가능. 펀드는 이 trade-off 를 의도한 분산 운용입니다.│ │
│ └────────────────────────────────────────────────────┘ │
│                                                          │
│ ┌─ Chart 1: SPY IT Mcap + Fund IT Tilt (이중 축) ───┐  │
│ │ SPY IT %  ┄┄┄┄┄┄┄┄┄┄┄┄ Fund Tilt %                │  │
│ │  35%┤              ╱╱─AI Rally    +5%             │  │
│ │  30%┤           ╱╱─                                │  │
│ │  25%┤      ╱╱╱─        ─── SPY IT Mcap            │  │
│ │  20%┤───                ┄┄┄ Fund IT Tilt          │  │
│ │       ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄        0%               │  │
│ │       ╲╱╲╱╲╱╲╲╲╲                       -5%        │  │
│ │              ╲╲╲╲╲╲                  -10%        │  │
│ │       2010  2014  2018  2022  2025  ★HO          │  │
│ └─────────────────────────────────────────────────┘  │
│                                                          │
│ ┌─ Chart 2: HO Sector Contribution (Tornado) ───────┐ │
│ │ Healthcare  │██──  +X.X% (분산 운용 가치)           │ │
│ │ Financials  │█──  +X.X%                              │ │
│ │ Industry    │█── +X.X%                               │ │
│ │ ────────────━━━━━━━━━━━━━━━━━━━━━━━ 0%             │ │
│ │ Cons Disc        │█── -X.X%                          │ │
│ │ Energy           │██── -X.X%                          │ │
│ │ Info Tech        │██████ -X.X% ★ (HO 부진 핵심)     │ │
│ └─────────────────────────────────────────────────┘  │
│                                                          │
│ ┌─ Chart 3: Regime 별 Sector HHI 추세 ──────────────┐ │
│ │ HHI                                                │ │
│ │ 0.20┤                                              │ │
│ │ 0.15┤  ███  ████  ████   ████ ← Fund (분산 일관)  │ │
│ │ 0.10┤  ███  ████  ████   ████                       │ │
│ │ 0.20┤████████████████████ ████ ← SPY (집중 ↑ HO)  │ │
│ │ 0.10┤                                              │ │
│ │      R1   R2   R3   HO                             │ │
│ └─────────────────────────────────────────────────┘  │
│                                                          │
│ ┌─ ✅ 결론: 장기 Sector 분산 운용의 가치 ────────────┐ │
│ │ Markowitz (1952) 의 평균-분산 이론과 Fama-French   │ │
│ │ (1992) 의 factor diversification 관점에서, sector  │ │
│ │ 분산 운용은 idiosyncratic risk 를 줄이고 장기      │ │
│ │ 위험조정 수익 향상에 기여합니다.                    │ │
│ │                                                     │ │
│ │ 본 펀드는 R1/R2/R3 168개월 학습 구간에서 일관된    │ │
│ │ sector 분산을 통해 우수한 위험조정 성과 입증.      │ │
│ │ HO 24m 단기 underperform 은 일시적 trade-off 이며, │ │
│ │ 장기 분산 운용의 근본 가치를 손상시키지 않습니다.  │ │
│ └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] 학술 narrative 박스 = `st.info` 또는 custom card (Cobalt Blue accent)
- [ ] Chart 1 (이중 축) = Plotly `make_subplots(specs=[[{"secondary_y": True}]])`
- [ ] Chart 2 (Tornado) = Plotly `go.Bar` (orientation='h')
- [ ] Chart 3 (HHI 비교) = Plotly `go.Bar` (Fund vs SPY 사이드바이)
- [ ] 결론 박스 = `st.success` 또는 custom card (Cobalt Blue accent)
- [ ] 학술 인용 (Markowitz 1952, Fama-French 1992) 명시

---

### 영역 9: Footer — Overview 동일

→ `02_common.md` 의 `render_footer()` 호출

---

## 페이지 데이터 의존성

- results/mat_eq_eq_raw_pap.pkl (펀드 weights)
- universe.csv (gics_sector)
- monthly_panel.csv (SPY sector weights)

---

## 메트릭 (C-2 풀)

- Pool-4 운용 효율성: Sector HHI
- Pool-5 시장 비교: Sector Tilt
- 추가 (영역 5 9 컬럼): Volatility / Beta / Sharpe / Contribution (sector level)
- 영역 8 narrative: Markowitz (1952) + Fama-French (1992) factor diversification

---

## 인터랙션 / 토글 적용

| 영역 | 사이드바 토글 영향 | Q-Zoom |
|---|---|---|
| 영역 3 (KPI) | ✓ 기간 평균 | ✗ |
| 영역 4 (Treemap) | 시점 슬라이더 | ✓ 섹터 클릭 → 종목 list |
| 영역 5 (Decomposition 표) | ✗ | ✓ 섹터 클릭 |
| 영역 6 (Tilt Tornado) | ✓ 기간 토글 | ✓ 섹터 클릭 |
| 영역 7 (Rotation) | ✗ | ✓ 시점 클릭 |
| 영역 8 (HO 정당화) | ✗ (HO 자체 분석) | ✗ |

---

## 페이지 구현 우선순위

- **Phase 2 (확장, 2-3주)**: Sector Watch 페이지 (Phase 2 의 핵심 — 5분 demo 1.5분 차지, K-2 결정)

---

## 결과 / 함의

- **HO 정당화 narrative** 가 영역 2, 3 (Most Underweight), 6 (Tilt), 7 (AI Rally annotation), 8 (학술 정당화) 에 일관 분산
- **5분 demo 의 1.5분** 차지 (K-2 결정) — Sector Watch 핵심 페이지
- 학술 인용: **Markowitz (1952)**, **Fama-French (1992)**

---

[← 05_holdings.md](05_holdings.md) | [07_methodology.md →](07_methodology.md)
