# Holdings 페이지 — 와이어프레임

> **관련 decisionlog**: `05_holdings.md`
> **상태**: 확정
> **결정 수**: 8 영역 (메타 Hold M-1~M-4 + 영역 1~8)

---

## 페이지 역할 정의

보유 종목 + 비중 + 시가총액 분포 + 변천사 분석. 메트릭 풀 (C-2) 의 **Pool-4 (운용 효율성)** + **Pool-5 (시장 비교)** 일부 활용.

**vs Sector Watch 페이지**:
- Holdings = **개별 종목 위주**
- Sector Watch = 섹터 단위 위주

---

## 페이지 영역 구조 (시선 흐름)

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트)
3. Holdings Summary KPI 6개     (Number / Eff N / Single HHI / Sector HHI / Top Weights / Avg Turnover)
4. Top N Holdings 표 + 비중     (현재 보유)
5. 시가총액 분포 (Bubble / Treemap)
6. 보유 종목 변천사 (Multi-line)
7. 종목별 기여도 분석 (Tornado Chart)
8. Footer                       (Overview 동일)
```

---

## 영역별 와이어프레임

### 영역 1: Header — Overview 동일

→ `02_common.md` 의 `render_page_header()` 호출

---

### 영역 2: Sub-header

**결정사항** (H2-1):
- (a) Performance 패턴 + Footnote 보조 표시 (yfinance 명시)

**텍스트 안**:
```
Holdings (보유 종목)
보유 종목 / 비중 / 시가총액 분포 / 변천사 분석.
사이드바에서 기간 + 비교 벤치마크 토글 가능.
```

**Footnote** (영역 7 또는 영역 8): "※ 회사명 / 시가총액 / 섹터 매핑: yfinance 기반"

---

### 영역 3: Holdings Summary KPI 6개

**결정사항** (Hold M-3 + H3-1 ~ H3-4):
- **6 KPI**: Number / Effective N / Single HHI / Sector HHI / **Top Weights (Top 1/5/10 통합)** / Avg Turnover
- 표시 기간: (c) Latest snapshot + 사이드바 토글 평균 (양쪽)
- 카드 디자인: (a-2) Top Weights 만 다중 값 예외
- 벤치마크 비교: (b) 의미 있는 KPI 만 vs SPY (Eff N / Single HHI / Sector HHI / Top Weights)
- Hover tooltip: (a) 포함

**Tooltip 텍스트** (`lib/tooltips.py` 참조)

**시각화 예시**:

```
[Header — Overview 동일]

┌─ ℹ️ Holdings (보유 종목) ────────────────────────────────┐
│ 보유 종목 / 비중 / 시가총액 분포 / 변천사 분석.           │
│ 사이드바에서 기간 + 비교 벤치마크 토글 가능.              │
└──────────────────────────────────────────────────────────┘

[Latest snapshot: 2025-12]  [기간 평균: TEST 168m]
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Number   │ Eff N    │ Single   │ Sector   │ Top      │ Avg      │
│ Holdings │          │ HHI      │ HHI      │ Weights  │ Turnover │
│          │          │          │          │ T1:  X%  │          │
│ XX       │ XX.X     │ X.XX     │ X.XX     │ T5:  XX% │ XX%/y    │
│ (XX 평균)│ (XX 평균)│ (X.XX 평균)│ (X.XX 평균)│ T10: XX% │ (XX 평균)│
│          │ vs SPY   │ vs SPY   │ vs SPY   │ vs SPY   │          │
│          │ ▼ better │ ▼ better │ ▼ better │ T5 ▼ better│          │
│ ⓘNumber  │ ⓘEff N   │ⓘSingle  │ⓘSector  │ⓘTop W    │ⓘTurnover│
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

**구현 체크리스트**:
- [ ] `st.columns(6)` 반응형 (작은 화면 3+3 분할)
- [ ] Latest snapshot = 최신 월 (예: 2025-12) 의 weight 기반
- [ ] 기간 평균 = 사이드바 토글의 월별 weight 평균
- [ ] Top Weights 카드 = 다중 값 예외 (T1 / T5 / T10)
- [ ] 의미 있는 4 KPI 만 vs SPY delta 표시

---

### 영역 4: Top N Holdings 표 + 비중

**결정사항** (H4-1 ~ H4-6):
- Top N: (c) Top N 토글 (10/20/50/All)
- 컬럼: (b) 표준 7 컬럼:
  1. Rank
  2. Ticker + Company (yfinance 매핑)
  3. Sector (GICS)
  4. Weight (현재 비중)
  5. Market Cap ($B)
  6. 12m Return
  7. ΔWeight (vs 전월 변화)
- 표시 기간: (a) Latest only
- 정렬: (c) 비중 default + 사용자 정렬
- 인터랙션: 모두 채택 (검색, 정렬, CSV, Q-Zoom)
- 시각 보강: (a) Weight 막대 + (b) 섹터 색상 코딩

**시각화 예시**:

```
[Holdings Summary KPI 6개]

┌─ Top N Holdings — Latest snapshot (2025-12) ──────────────────────┐
│ [Top: 10 / 20 / 50 / All ▼]  [🔍 Search...]  [⬇ CSV]              │
│                                                                    │
│ Rank | Ticker     | Sector       | Weight        | Mcap  | 12m R│ ΔW│
│  1   | AAPL —     │Info Tech     │ ████ X.X%     │ $XXXB │ +X% │+X%│
│      │ Apple Inc. │ (배경 색상)   │               │       │     │   │
│  2   | MSFT —     │Info Tech     │ ███ X.X%      │ $XXXB │ +X% │-X%│
│      │ Microsoft  │              │               │       │     │   │
│  3   | NVDA —     │Info Tech     │ ███ X.X%      │ $XXXB │ +X% │+X%│
│      │ Nvidia     │              │               │       │     │   │
│ ...                                                                │
│                                                                    │
│ Hover: 종목 detail tooltip                                         │
│ 종목 클릭 → 종목 detail expand                                     │
│ 컬럼 헤더 클릭 → 정렬                                              │
└────────────────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Streamlit `st.dataframe` 또는 `st.data_editor` (정렬 + 검색 자동)
- [ ] Weight 막대 = `st.column_config.ProgressColumn` (Streamlit 1.30+)
- [ ] 섹터 색상 코딩 = `pandas.Styler.background_gradient` 또는 `lib/colors.py` 의 SECTOR_COLORS
- [ ] ticker → 회사명 = `data/ticker_company_map.csv` 캐시 (yfinance 1회 수집)
- [ ] 종목 클릭 expand = `st.dataframe(on_select='rerun')` (Streamlit 1.27+)
- [ ] CSV 다운로드 = `st.download_button`

---

### 영역 5: 시가총액 분포 (Bubble / Treemap)

**결정사항** (H5-1 ~ H5-5):
- 차트 종류: (d) Bubble + Treemap (탭 전환)
- Treemap 차원: (d) 토글 (면적/색상 사용자 선택)
  - 기본: 면적=비중, 색상=섹터
  - 추가: 면적=비중, 색상=12m 수익률 (red-green)
  - 추가: 면적=시가총액, 색상=섹터
- Bubble 차원: (d) 사용자 선택 (X/Y/크기/색 모두 토글)
- 표시 기간: (c) 시점 슬라이더
- 인터랙션: 모두 채택

**시각화 예시**:

```
[Top N Holdings 표 (영역 4)]

┌─ 시가총액 분포 ─────────────────────────────────────────┐
│                                                         │
│ [Tab: Bubble Chart | Treemap]                           │
│ [시점 슬라이더: 2010 ●━━━━━━━━━━━━━━━━━━━━━━━━ 2025] │
│                                                         │
│ ─── Treemap (선택 시) ───────────────                   │
│ 색상: [면적=비중,섹터 ▼ / 면적=비중,수익률 / 면적=시가총액,섹터]│
│ ┌───────────────┬─────────┬─────────┐                  │
│ │ AAPL          │ MSFT    │ NVDA    │                  │
│ │ ████████      │ ██████  │ █████   │                  │
│ │ X.X%          │ X.X%    │ X.X%    │                  │
│ ├───────────┬───┴─────┬───┴─────┐                      │
│ │ AMZN      │ GOOGL   │ META    │                      │
│ └───────────┴─────────┴─────────┘                      │
│                                                         │
│ ─── Bubble (선택 시) ───────────────                    │
│ X축: [시가총액 ▼]  Y축: [비중 ▼]                       │
│ 크기: [시가총액 ▼]  색상: [섹터 ▼]                     │
│                                                         │
│ Hover: 종목 detail / Click: expand / 섹터 필터          │
│ [⬇ PNG] [⬇ CSV]                                        │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] `st.tabs(["Bubble Chart", "Treemap"])`
- [ ] Treemap = Plotly `px.treemap` (path = ['sector', 'ticker'] for drill-down)
- [ ] Bubble = Plotly `px.scatter` (size 파라미터)
- [ ] 시점 슬라이더 = `st.slider` (월 단위)
- [ ] 사용자 선택 (X/Y/크기/색) = `st.selectbox` × 4
- [ ] 색상 팔레트 = `lib/colors.py` SECTOR_COLORS (GICS 11개)

---

### 영역 6: 보유 종목 변천사 (Holdings History)

**결정사항** (H6-1 ~ H6-5):
- 차트 종류: (e) Multi-line
- 종목 그룹화: (d) 토글 (섹터 / Top N + Others)
- 시간 단위: (a) + (b) 토글 (월별 / 분기별)
- 인터랙션: 모두 채택 (Hover, 종목/섹터 클릭, 시점 클릭, 검색)
- 추가 표시: (d) 모두 (Regime 배경 + 신규/제외 마커 + 이벤트 annotation)

**시각화 예시**:

```
[시가총액 분포 (영역 5)]

┌─ 보유 종목 변천사 (Multi-line) ─────────────────────────┐
│                                                         │
│ [Group: 섹터 ▼ / Top 10 + Others / Top 20 + Others]    │
│ [Time: 월별 / 분기별 ▼]   [🔍 종목 검색]              │
│                                                         │
│ ┌─R1─┬───R2────┬──R3──┬─HO─┐ (Regime 배경색)          │
│                                                         │
│ 비중 │                                                  │
│  10%┤    ╱──── AAPL                                     │
│   8%┤  ╱╱─                                              │
│   6%┤  ╱  ★(2020-04 신규) ╲    ▼COVID-19              │
│   4%┤╱      MSFT─────╱─────╲╱  (annotation)            │
│   2%┤              NVDA──╱──    ▼2022 Bear              │
│   0%┴───────────────────────────────────────            │
│      2010    2014    2018    2022    2025               │
│                                                         │
│  ★ 신규 진입 마커 / ✗ 제외 종목 마커                   │
│                                                         │
│  Hover: "2024-Q1: AAPL 7.2%, NVDA 5.1%..."             │
│  Click: 종목 강조 / 시점 detail expand                  │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Plotly `go.Scatter` (mode='lines') 다중 trace
- [ ] Regime 배경색 (`add_regime_backgrounds()` 헬퍼)
- [ ] 신규/제외 마커 = `add_annotation` 또는 별도 `go.Scatter` (mode='markers')
- [ ] 이벤트 annotation = COVID/2022 Bear/2024 dictionary
- [ ] 검색 = `st.text_input` + 라인 강조 동적
- [ ] Q-Zoom: 시점 클릭 → 해당 시점 detail expand

---

### 영역 7: 종목별 기여도 분석 (Performance Attribution)

**결정사항** (H7-1 ~ H7-6):
- Attribution 방법: (a) Simple Contribution ($w \times R$)
- 표시 형식: (a) Tornado Chart (가로 막대 양수/음수)
- Top N: (a) Top N 양수 + Top N 음수
- 표시 기간: (a) 사이드바 토글 반응
- 인터랙션: 모두 채택
- 추가 표시: (d) 모두 (섹터 합계 + 양수/음수 색상 + 총수익률 검증)

**시각화 예시**:

```
[보유 종목 변천사 (영역 6)]

┌─ 종목별 기여도 분석 (Performance Attribution) ─────────┐
│                                                         │
│ [Top: 10 / 20 / 50 ▼]   [기간: TEST 168m]              │
│                                                         │
│ Top Contributors (양수)         Bottom Contributors    │
│                                                         │
│ NVDA  │██████████  +X.X%       │ SOMECO │██──  -X.X%  │
│ AAPL  │████████    +X.X%       │ STOCK2 │██─    -X.X% │
│ MSFT  │██████      +X.X%       │ STOCK3 │██     -X.X% │
│ GOOGL │█████       +X.X%       │ STOCK4 │█      -X.X% │
│ ...                                                     │
│                                                         │
│ ─── Sector 합계 (보조) ─────────                       │
│ Tech: +X.X% / Healthcare: +X.X% / Financials: -X.X%    │
│ → Sector Watch 페이지에서 자세히                       │
│                                                         │
│ ※ 검증: Σ Contribution = Fund Return = +X.X% ✓         │
│                                                         │
│ Hover: "NVDA — Nvidia / Weight: X% / Return: +X%"      │
│ Click: 종목 detail expand                               │
│ [⬇ CSV]                                                │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Tornado chart = Plotly `go.Bar` (orientation='h', sorted)
- [ ] 양수/음수 색상 = `marker.color` 동적 (green/red)
- [ ] Sector 합계 = pandas `groupby('sector').sum()`
- [ ] 총수익률 검증 = mathematical assertion 표시
- [ ] Top N 토글 (10/20/50)
- [ ] CSV 다운로드

---

### 영역 8: Footer — Overview 동일 + Footnote (yfinance)

→ `02_common.md` 의 `render_footer()` 호출

**Footnote 추가**: `※ 회사명 / 시가총액 / 섹터: yfinance 기반`

---

## 페이지 데이터 의존성

- results/mat_eq_eq_raw_pap.pkl (펀드 weights + returns)
- universe.csv (gics_sector)
- ticker_company_map.csv (yfinance 회사명)
- daily_returns.pkl (시가총액 변화 + 12m return)

---

## 메트릭 (C-2 풀)

- Pool-4 운용 효율성: Number / Effective N / Single HHI / Sector HHI / Avg Turnover
- Pool-5 시장 비교 일부: Top Weights
- 추가 (영역 4 컬럼): Market Cap / 12m Return / ΔWeight
- 추가 (영역 7 기여도): Simple Contribution = $w \times R$

---

## 인터랙션 / 토글 적용

| 영역 | 사이드바 토글 영향 | Q-Zoom |
|---|---|---|
| 영역 3 (KPI) | ✓ 기간 평균 표시 | ✗ |
| 영역 4 (Top N 표) | ✗ (Latest only) | ✓ 종목 클릭 → detail |
| 영역 5 (시가총액 분포) | 시점 슬라이더 | ✓ 종목 클릭 |
| 영역 6 (변천사) | ✗ | ✓ 시점 클릭 → detail |
| 영역 7 (기여도) | ✓ 기간 토글 | ✓ 종목 클릭 |

---

## 페이지 구현 우선순위

- **Phase 2 (확장, 2-3주)**: Holdings 페이지

---

## 결과 / 함의

- **yfinance 의존**: ticker → 회사명 매핑 = `data/ticker_company_map.csv` 캐시 (1회 수집)
- **인터랙션 일관성 원칙** 적용
- 다음 페이지 (Sector Watch) 결정 시 동일 패턴 활용 + 섹터 narrative 강화

---

[← 04_risk_metrics.md](04_risk_metrics.md) | [06_sector_watch.md →](06_sector_watch.md)
