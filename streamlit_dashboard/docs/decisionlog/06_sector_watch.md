# C-1-5. Sector Watch 페이지

> **파일**: `06_sector_watch.md`
> **결정 시점**: 2026-05-10
> **상태**: 확정 (페이지 메타 Sector M-1~M-4 + 영역 1~9, ★ HO 정당화 narrative 포함)
> **포함**: 페이지 메타 결정 / Sub-header (HO narrative 명시) / Sector Summary KPI 5개 / Sector Treemap / Sector Decomposition 표 / Sector Tilt vs SPY / Sector 시계열 변화 (Sector Rotation) / ★★★ HO 24m 분석 + 정당화 narrative

---

## C-1-5. Sector Watch 페이지

> **상태**: 진행 중 (메타 결정 확정 / 영역 1, 9 는 Overview 동일 / 2-8 미정)

### Sector Watch 페이지 통합 배경 (Context)

섹터 단위 비중 + 분석 전담 페이지. 레퍼런스 PortfolioX360 의 **Sector Watch** 화면 (Tree Map / Decomposition / Sector Allocation) 패턴 적용.

**vs Holdings 페이지**:
- Holdings = 개별 종목 위주
- Sector Watch = 섹터 (GICS 11개) 단위 위주

**핵심 메트릭**:
- Sector HHI (Holdings 영역 3 KPI 와 다른 깊이)
- 섹터별 비중 (현재 + 시계열)
- **Sector Tilt vs SPY** = 액티브 운용 핵심
- 섹터별 contribution / return

#### **★★★ HO 정당화 narrative (사용자 강조)**

**핵심 요구사항**: HO 24m 구간 (2024-2025) 의 SPY 대비 펀드 성능 저하 정당화 가능해야 함.

**narrative**:
- 펀드 = **Sector-balanced** (분산 운용) → 장기 (R1/R2/R3) 안정적 위험조정 수익
- HO 24m: SPY = IT 집중 (AI rally) → +21.2% / 펀드 = IT under-weight → +8.3% (열위)
- "장기 분산의 가치 vs 단기 IT 집중의 trade-off"

→ Sector Watch 페이지가 이 narrative 의 중심 (특히 영역 8)

### 페이지 메타 결정 (Sector M-1 ~ M-4)

#### Sector M-1. 영역 개수

**검토된 옵션**:
- (a) 압축 5 영역
- (b) 표준 7 영역
- (c) 풍부 8 영역

**결정**: (c) 풍부 8 영역

**근거**:
1. **HO 정당화 narrative 필수** = 별도 영역 8 추가 필요
2. 섹터 분석 깊이 (현재 + 시계열 + 비교 + HO 분석) 충분 활용
3. 자유 탐색 모드 부합

#### Sector M-2. Sub-header

**결정**: (a) 포함

**근거**: 일관 + HO narrative 핵심 명시 필수

#### Sector M-3. Sector Summary KPI

**검토된 옵션**:
- (a) 섹터 분산 강조: Number / HHI / Largest / Smallest
- (b) 시장 비교 강조: HHI / Active Bets / Avg Tilt / Most Over/Under
- (c) 종합
- (d) 자유

**결정**: (b) 시장 비교 강조 + HO 정당화 narrative

**근거**:
1. **Sector Tilt** = 펀드 정체성 핵심 (액티브 운용 narrative)
2. **Most Overweight / Underweight** = HO 정당화 narrative 와 직접 연결 (IT under-weight 명시)
3. (a) 분산 강조는 Holdings 영역 3 (Sector HHI KPI) 와 중복

**5개 KPI 안**:
1. Sector HHI (분산도, Holdings KPI 와 다른 깊이)
2. Avg |Tilt| vs SPY (Active Bets 강도)
3. Number of Active Bets (|Tilt| > 1% 섹터 수)
4. Most Overweight Sector (이름 + Tilt%)
5. Most Underweight Sector (이름 + Tilt% — IT 가능성 ↑ HO 시기)

#### Sector M-4. Holdings KPI 중복 처리

**검토된 옵션**:
- (a) 중복 표시
- (b) Sector Watch 에서 제외
- (c) Sector Watch 에서 자세히

**결정**: (c) Sector Watch 에서 자세히

**근거**:
1. **Holdings KPI** = Sector HHI 단순값 (정적 snapshot)
2. **Sector Watch KPI** = HHI + Tilt + Active Bets 등 다차원 (페이지 깊이)
3. 차별화 + 일관성 양립

### Sector Watch 페이지 8 영역 구조 (HO 정당화 반영)

```
1. Header                       (Overview 동일)
2. Sub-header                   (HO narrative 핵심 명시)
3. Sector Summary KPI 5개       (HHI / |Tilt| / Active Bets / Over / Under)
4. Sector Treemap (현재 비중)   (PortfolioX360 패턴)
5. Sector Decomposition 표      (PortfolioX360 패턴)
6. Sector Tilt vs SPY           (Active Bets — 액티브 운용 핵심)
7. Sector 시계열 변화           (Sector Rotation 분석)
8. ★★★ HO 24m 분석 + 정당화     (IT/AI Rally narrative)
9. Footer                       (Overview 동일)
```

### 영역 2: Sub-header (HO narrative 명시) — 확정

#### 결정 항목 S2-1: 텍스트 내용

**검토된 옵션**:
- (a) 일반 패턴
- (b) HO narrative 명시
- (c) 학술 narrative 강조

**결정**: (b) HO narrative 명시

**근거**:
1. **사용자 강조 사항** (Sector M-3) — HO 정당화 가능해야 함
2. Sub-header 에서부터 HO narrative 노출 → 페이지 전체 컨텍스트 명확
3. (c) 학술 narrative 는 영역 8 에서 자세히

**텍스트 안**:
```
Sector Watch (섹터 분석)
섹터 비중 / 분산 / 시장 비교 분석.
HO 24m (2024-2025) sector rotation 영향과 펀드의
sector 분산 운용의 양면성 분석 포함.
사이드바에서 기간 + 비교 벤치마크 토글 가능.
```

---

### 영역 3: Sector Summary KPI 5개 — 확정

#### 영역 3 통합 배경 (Context)

5 KPI 사전 확정 (Sector M-3 b):
1. Sector HHI
2. Avg |Tilt| vs SPY
3. Number of Active Bets (|Tilt| > 1%)
4. Most Overweight Sector (이름 + Tilt%)
5. Most Underweight Sector (이름 + Tilt%) — HO 정당화 narrative 직접 연결

#### 결정 항목 S3-2: 표시 기간

**결정**: (c) Latest + 사이드바 토글 평균

**근거**: Holdings 영역 3 일관 — 정적 + 동적 양립

#### 결정 항목 S3-3: 카드 디자인

**결정**: (a) 단순 + Most Over/Under 카드만 (b) 다중 값

**Most Over/Under 카드 구조**:
```
┌── Most Overweight ──┐
│ Healthcare          │
│ +X.X%               │
└─────────────────────┘
```

**근거**:
1. 단일 값 (Tilt% 만) 은 섹터명 손실
2. 섹터명 + Tilt% = 두 정보 한 카드
3. Holdings 영역 3 (Top Weights) 패턴 일관

#### 결정 항목 S3-4: 벤치마크 비교

**결정**: (b) HHI 만 vs SPY

**근거**:
1. **Tilt 자체가 vs SPY 메트릭** → delta 추가 표시 의미 ↓
2. **HHI** = SPY 와 비교 의미 있음 (Sector HHI 비교)
3. Active Bets / Most Over/Under 는 펀드 자체 특성 → 비교 불필요

#### 결정 항목 S3-5: Hover tooltip

**결정**: (a) 포함

**근거**:
1. **Tilt / HHI / Active Bets** 정의 안내 필수 (가상 투자자 친화)
2. Performance / Risk / Holdings 일관

**Tooltip 텍스트 안**:
- Sector HHI: "섹터 집중도 = Σ(섹터 weight)². 낮을수록 분산"
- Avg |Tilt|: "섹터별 (펀드 weight - SPY weight) 절대값 평균. Active 운용 강도"
- Active Bets: "|Tilt| > 1% 인 섹터 수. 높을수록 액티브 운용"
- Most Over/Under: "SPY 대비 펀드의 가장 큰 over/under-weight 섹터"

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- HO 시기 IT under-weight 가 KPI Most Underweight 에 자연 노출 → narrative 자동 강화
- Latest snapshot 표시 = 펀드의 현재 sector 운용 상태
- 토글 평균 표시 = 기간별 sector 운용 안정성
- Tooltip 표준화 (Sector 메트릭 dictionary)

### 영역 4: Sector Treemap (현재 비중) — 확정

#### 영역 4 통합 배경 (Context)

11개 GICS 섹터의 현재 비중 시각화. 레퍼런스 PortfolioX360 의 Sector Tree Map 패턴.

**vs Holdings 영역 5 (Bubble + Treemap)**:
- Holdings 영역 5 = 종목 단위 Treemap
- Sector Watch 영역 4 = 섹터 단위 Treemap (집계)

#### 결정 항목 S4-1: Treemap 차원

**검토된 옵션**:
- (a) 면적=비중, 색상=섹터
- (b) 면적=비중, 색상=Tilt vs SPY
- (c) 면적=비중, 색상=12m 수익률
- (d) 토글 (a/b/c)

**결정**: (d) 토글

**근거**:
1. **자유 탐색** — 다차원 분석 자유 선택
2. 기본: (a) 면적=비중, 색상=섹터 (PortfolioX360 일관)
3. (b) Tilt 색상 = HO narrative 강화 (IT 빨강 = under-weight)
4. (c) 수익률 색상 = 섹터별 성과 분석

#### 결정 항목 S4-2: Sub-sector

**검토된 옵션**:
- (a) Sector level only
- (b) Sector + 종목 drill-down
- (c) Sector + Industry

**결정**: (b) Sector + 종목 drill-down

**근거**:
1. **섹터 클릭 → 종목 detail expand** = 자유 탐색 깊이
2. Q-Zoom 패턴 일관 (인터랙션 일관성)
3. (c) Industry 는 GICS 24개로 시각 혼잡

#### 결정 항목 S4-3: 비교

**검토된 옵션**:
- (a) 펀드 only
- (b) 펀드 vs SPY 좌우 분할
- (c) Tilt Treemap (Diff)
- (d) 토글

**결정**: (d) 토글

**근거**:
1. **자유 탐색** — 사용자 자율 선택
2. 기본: (b) 좌우 분할 (펀드 vs SPY 직관 비교)
3. (c) Tilt Treemap = HO narrative 강화 (Diff 시각)

#### 결정 항목 S4-4: 표시 기간

**결정**: (c) 시점 슬라이더

**근거**: Holdings 영역 5 일관 — 시점별 sector 변화 관찰 가능

#### 결정 항목 S4-5: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- 섹터 클릭 → 종목 detail expand ✓ (Q-Zoom)
- 섹터 필터 ✓
- 다운로드 ✓

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- Treemap 구현 = Plotly `px.treemap` (path = ['sector', 'ticker'] 으로 drill-down)
- 좌우 분할 = `make_subplots(cols=2)` 또는 `st.columns(2)`
- Tilt Treemap = (Fund weight - SPY weight) 색상
- 시점 슬라이더 = Plotly `animation_frame` 또는 Streamlit `st.slider`
- 섹터 클릭 expand = `st.plotly_chart(on_select='rerun')`

### 영역 5: Sector Decomposition 표 — 확정

#### 영역 5 통합 배경 (Context)

레퍼런스 PortfolioX360 의 **Sector Decomposition 표** 패턴 — 11개 GICS 섹터 종합 표.

**vs 영역 4 (Treemap)**: 영역 4 = 시각 (직관) / 영역 5 = 표 (정확 수치)

#### 결정 항목 S5-1: 컬럼

**검토된 옵션**:
- (a) 핵심 5컬럼
- (b) 표준 7컬럼
- (c) 풍부 9컬럼
- (d) 자유

**결정**: (c) 풍부 9컬럼

**컬럼 구성**:
1. Sector
2. Weight (펀드 비중)
3. Tilt vs SPY (Active 운용)
4. 12m Return (섹터 12개월 수익률)
5. # Holdings (섹터 내 보유 종목 수)
6. Volatility (섹터 변동성)
7. Beta (섹터 β vs SPY)
8. Sharpe (섹터 위험조정수익)
9. Contribution (섹터의 펀드 수익 기여도)

**근거**:
1. **종합 분석** = 가상 투자자 + 학술 모두 만족
2. (a) 5컬럼은 위험/Sharpe/Contribution 누락 → 정보 ↓
3. (c) 9컬럼 = Sector Decomposition 의 학술 표준
4. CSV 다운로드 가능 → 청중이 자유 분석

#### 결정 항목 S5-2: 비교

**검토된 옵션**:
- (a) 펀드 only
- (b) 펀드 + SPY 분리 컬럼
- (c) 펀드 값 + Tilt 컬럼

**결정**: (c) 펀드 값 + Tilt 컬럼

**근거**:
1. **Decomposition 표준 패턴** — Tilt 가 vs SPY 비교 직접 표시
2. (b) SPY 분리는 컬럼 ↑↑ → 가독성 ↓
3. Tilt 컬럼이 SPY 비교 narrative 충족

#### 결정 항목 S5-3: 정렬

**결정**: (c) 기본 Weight 내림차순 + 사용자 정렬

**근거**:
1. Holdings 영역 4 패턴 일관
2. 기본 Weight 정렬 = 표준
3. 사용자 자율 정렬 = 다차원 분석 (Tilt 정렬 → HO under-weight 강조)

#### 결정 항목 S5-4: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- 섹터 클릭 → 종목 detail expand ✓ (Q-Zoom)
- 컬럼 정렬 ✓
- CSV 다운로드 ✓

#### 결정 항목 S5-5: 시각 보강

**검토된 옵션**:
- (a) Weight 컬럼 막대
- (b) Tilt 컬럼 색상 코딩 (red/green)
- (c) HO 시기 IT 강조
- (d) 모두

**결정**: (a) Weight 막대 + (b) Tilt 색상 코딩

**근거**:
1. **Weight 막대** = Holdings 영역 4 패턴 일관
2. **Tilt 색상** = red (under-weight) / green (over-weight) 직관 시각
3. (c) HO IT 강조 = 영역 8 (HO 정당화) 에서 자세히 → 영역 5 는 일반 표 유지

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- 표 구현 = Streamlit `st.dataframe` (정렬 + 검색 자동)
- Weight 막대 = `st.column_config.ProgressColumn`
- Tilt 색상 = `pandas.Styler.applymap` 또는 cell formatting
- 9컬럼 = 가로 스크롤 가능 (Streamlit 자동)
- CSV 다운로드 = `st.download_button`

### 영역 6: Sector Tilt vs SPY (Active Bets) — 확정

#### 영역 6 통합 배경 (Context)

Active Bets 의 핵심 시각화 — 각 섹터의 Tilt (Fund Weight - SPY Weight) 직관 표시. 펀드 정체성 (액티브 운용) narrative 핵심.

**HO 정당화 narrative 의 일부**:
- IT under-weight 시각적 노출 (HO 24m 부진 원인)
- 다른 섹터 over-weight 의 균형 narrative

#### 결정 항목 S6-1: 차트 종류

**검토된 옵션**:
- (a) Tornado Chart
- (b) Diverging Bar (수직)
- (c) Bullet Chart
- (d) Lollipop Chart

**결정**: (a) Tornado Chart

**근거**:
1. **Holdings 영역 7 (Contribution) 패턴 일관** — 인터랙션 일관성 원칙
2. 양수/음수 즉시 시각 — 가로 막대 양옆 분리
3. (b) Diverging Bar 도 유사하지만 가로보다 세로 시각이 직관 ↓

#### 결정 항목 S6-2: 정렬

**결정**: (a) Tilt 크기순 (Most Over → Most Under)

**근거**:
1. **HO narrative 강화** — Most Underweight (IT) 명확히 노출
2. 표준 Tornado 정렬 (Most Over 위 → Most Under 아래)
3. (b) 알파벳 / (c) Weight 순은 narrative 약화

#### 결정 항목 S6-3: 표시 기간

**결정**: (c) Latest + 토글 평균 (둘 다)

**근거**:
1. **Latest** = 현재 sector 운용 상태
2. **토글 평균** = 기간별 평균 Tilt (사이드바 토글 반응)
3. Holdings 영역 3 (Summary KPI) 패턴 일관

#### 결정 항목 S6-4: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- 섹터 클릭 → 종목 detail expand ✓ (Q-Zoom)
- Top N Tilt 강조 ✓

#### 결정 항목 S6-5: 추가 표시

**결정**: (c) 0% 기준선 + (a) ±1% / ±5% 임계선

**근거**:
1. **0% 기준선** = 양/음 명확
2. **±1% / ±5% 임계선** = Active Bet 강도 시각 기준 (1% = mild / 5% = strong)
3. (b) 평균 |Tilt| 는 KPI 카드 (영역 3) 에서 이미 표시 → 중복 회피

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- Tornado 구현 = Plotly `go.Bar` (orientation='h', sorted)
- 양수/음수 색상 = `marker.color` 동적 (green/red)
- 임계선 = `add_vline` 활용 (±1%, ±5%, 0%)
- 영역 8 (HO 정당화) 와 narrative 연결 (IT under-weight 강조)

### 영역 7: Sector 시계열 변화 (Sector Rotation) — 확정

#### 영역 7 통합 배경 (Context)

시간에 따른 섹터 비중 변화 시각화 — 펀드의 sector rotation 패턴 분석.

**vs Holdings 영역 6 (보유 종목 변천사)**: 종목 단위 ↔ 섹터 단위 (집계)
**vs 영역 4 (Treemap)**: 단일 시점 ↔ 시간 흐름

#### 결정 항목 S7-1: 차트 종류

**검토된 옵션**:
- (a) Stacked Area Chart
- (b) Stream Graph
- (c) Multi-line (11개 섹터 라인)
- (d) Heatmap
- (e) 토글 (Stacked Area + Multi-line)

**결정**: (e) 토글 (Stacked Area + Multi-line)

**근거**:
1. **두 시각화 모두 의미** — 구성 (Stacked Area) + 추적 (Multi-line)
2. 자유 탐색 모드 부합
3. 기본: Stacked Area (구성 변화 직관) / 옵션: Multi-line (개별 섹터 추적)

#### 결정 항목 S7-2: 비교

**검토된 옵션**:
- (a) 펀드 only
- (b) 펀드 vs SPY 좌우 분할
- (c) Tilt 시계열 (Fund - SPY)
- (d) 토글

**결정**: (d) 토글

**근거**:
1. **자유 탐색** — 다중 비교 관점 자유 선택
2. (c) Tilt 시계열 = HO narrative 강화 (IT under-weight 추세)
3. 토글 = 청중이 의도적 비교 선택

#### 결정 항목 S7-3: 시간 단위

**결정**: (c) 토글 (월별 / 분기별)

**근거**: Holdings 영역 6 일관 — 인터랙션 일관성 원칙

#### 결정 항목 S7-4: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- 섹터 클릭 → 단독 강조 ✓
- 시점 클릭 → expand ✓ (Q-Zoom)
- 섹터 검색 / 필터 ✓

#### 결정 항목 S7-5: 추가 표시

**결정**: (d) 모두 (Regime 배경 + 이벤트 annotation + AI rally 강조)

**근거**:
1. **Regime 배경색** = 다른 페이지 일관
2. **이벤트 annotation** = "2020-03 COVID", "2022 Bear", "2024-12 AI Rally"
3. **AI rally 강조** = HO narrative 직접 시각 (영역 8 와 narrative 연결)

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- Stacked Area = Plotly `go.Scatter(stackgroup='one')`
- Multi-line = Plotly `go.Scatter` 다중 trace
- Tilt 시계열 = (Fund - SPY) 시계열
- 이벤트 annotation = dictionary 정의 (시점 + 라벨)
- 영역 8 (HO 정당화) 와 narrative 직접 연결

### 영역 8: ★★★ HO 24m 분석 + 정당화 narrative — 확정

#### 영역 8 통합 배경 (Context)

**Sector Watch 페이지의 핵심 narrative 영역** — HO 24m (2024-2025) 펀드 부진을 sector 분산 운용의 양면성으로 학술/실무 정당화.

**핵심 메시지**:
- HO 24m: SPY +21.2% / 펀드 +8.3% (열위 -12.9%p)
- 원인: SPY 의 IT 집중 (AI rally) → 펀드의 IT under-weight 가 불리하게 작용
- 정당화: "장기 분산의 가치 vs 단기 IT 집중의 trade-off"

#### 결정 항목 S8-1: 영역 sub-구조

**검토된 옵션**:
- (a) 단일 차트 + narrative 박스
- (b) 2개 차트 + narrative 박스
- (c) 3개 차트 + narrative 박스
- (d) 학술 보고서 형식 (다중 sub-section)

**결정**: (c) 3개 차트 + narrative 박스

**근거**:
1. **3축 분석** = IT Tilt 시계열 / HO Sector Contribution / Regime 별 분산 효과
2. 깊이 있는 narrative + 적정 페이지 분량
3. (d) 학술 보고서는 청중 부담 ↑

#### 결정 항목 S8-2: 차트 1 — IT Tilt 시계열

**검토된 옵션**:
- (a) HO 24m IT Tilt 막대 (월별)
- (b) FULL 192m IT Tilt 시계열 + HO 강조
- (c) IT 시가총액 비중 (SPY) + Fund IT Tilt 이중 축

**(b) vs (c) 비교**:

| 차원 | (b) FULL+HO | (c) IT Mcap+Tilt 이중 축 |
|---|---|---|
| 시각화 단순성 | ★★★ | ★★ |
| 펀드 IT 운용 일관성 | ★★★ | ★★ |
| AI rally 원인 시각화 | ★ | ★★★ |
| Cause-Effect 분석 | ★★ | ★★★ |

**결정**: (c) IT Mcap (SPY) + Fund IT Tilt 이중 축

**근거**:
1. **HO 정당화 narrative 핵심** = "SPY IT 비중 ↑↑ 시기에 Fund IT Tilt ↓" 동시 시각
2. (b) Fund 만 보여주면 "왜 Tilt 가 음수?" 가 모호 → (c) 는 SPY 변화가 원인임을 직접 노출
3. 이중 축 복잡도는 narrative 박스 + tooltip 으로 보완

**구현**:
- 좌축: SPY IT 비중 (%) — 시장 IT 집중도
- 우축: Fund IT Tilt (%) — 펀드 vs SPY 차이
- 시계열 동시 표시 + AI rally 시기 강조

#### 결정 항목 S8-3: 차트 2 — HO Sector Contribution

**결정**: (a) HO 24m Sector Contribution Tornado

**근거**:
1. Holdings 영역 7 (Performance Attribution Tornado) 패턴 일관 — 인터랙션 일관성
2. Negative contributors (IT under-weight 등) 즉시 시각
3. (b) BHB 분해는 학술 깊이 ↑ but 청중 어려움
4. (c) Fund vs SPY Contribution 차이는 (a) 의 한 형태

#### 결정 항목 S8-4: 차트 3 — Regime 별 분산 효과

**결정**: (a) Regime 별 Sector HHI 추세

**근거**:
1. **분산도 시간 변화 + 가치 시각** — R1/R2/R3/HO 별 펀드 vs SPY HHI 비교
2. HO 시기 SPY HHI ↑↑ (IT 집중) vs Fund HHI 안정 (분산 일관) 시각
3. (b) Regime별 Contribution = 영역 7 (Performance) 와 중복
4. (c) "trade-off" 추상 시각화는 narrative 박스로 충분

#### 결정 항목 S8-5: narrative 박스

**결정**: (b) 학술 narrative + 인용 (Markowitz 1952 등)

**근거**:
1. **학술 정직성 + 심사위원 어필**
2. Markowitz (1952) 평균-분산 이론 = 분산 운용의 학술 토대
3. Fama-French (1992) factor diversification = factor 다양성 가치
4. (a) 단순 설명은 학술 깊이 ↓ / (c) Q&A 는 영역 분량 ↑

#### 결정 항목 S8-6: 결론 메시지

**검토된 텍스트** (3개 옵션 사용자 비교):

(a) "장기 분산의 가치" — 자신감 강조
(b) "HO 부진 + 학습 교훈" — 솔직 인정
(c) 균형 (a + b)

**결정**: (a) 장기 분산의 가치 (자신감)

**근거**:
1. **펀드 정체성 (VolControl) 자신감 narrative**
2. HO 부진은 "trade-off" 로 정당화 (이미 narrative 박스에서 학술적 인정)
3. 결론 박스는 펀드의 가치를 명확 강조 → 청중 신뢰 ↑
4. (b) 솔직 인정은 학술 정직성 ↑ but 마케팅 정체성 ↓
5. (c) 균형은 메시지 모호

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

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- **HO 정당화 narrative 의 중심** — 펀드 정체성 보존 + 학술 정직성 양립
- 학술 인용 = Markowitz (1952), Fama-French (1992) 추가 (부록 학술 근거 일람)
- 이중 축 차트 = Plotly `make_subplots(specs=[[{"secondary_y": True}]])`
- 결론 박스 = `st.info` 또는 custom card (Cobalt Blue accent)
- About 페이지 / Methodology 페이지에서 narrative 추가 강화 가능

---

### Sector Watch 페이지 — 전체 확정 (영역 1~9)

#### 페이지 시각화 통합

```
┌────────────────────────────────────────────────────────────────┐
│ [영역 1: Header — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 2: Sub-header — HO narrative 명시]                        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 3: Sector Summary KPI 5개]                                │
│ HHI / Avg|Tilt| / Active Bets / Most Over / Most Under         │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 4: Sector Treemap]                                        │
│ 토글 (차원/비교/시점) + drill-down                             │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 5: Sector Decomposition 표 9컬럼]                         │
│ Weight 막대 + Tilt 색상 + 정렬 + CSV                           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 6: Sector Tilt vs SPY (Tornado)]                          │
│ Tilt 크기순 + 0% 기준선 + ±1%/5% 임계선                        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 7: Sector 시계열 변화 (Sector Rotation)]                  │
│ 토글 (Stacked/Multi-line) + Regime + AI Rally annotation       │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 8: ★★★ HO 24m 분석 + 정당화]                              │
│ 학술 narrative + 3차트 (IT 이중축/Contribution/HHI) + 결론 박스 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 9: Footer — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘
```

#### Sector Watch 페이지 결정 결과 / 함의

- **8 영역 모두 확정** → 구현 시 명확한 와이어프레임
- **HO 정당화 narrative** 가 영역 2, 3 (Most Underweight), 6 (Tilt), 7 (AI Rally annotation), 8 (학술 정당화) 에 일관 분산
- 학술 인용 = Markowitz (1952), Fama-French (1992)
- 인터랙션 일관성 원칙 적용 (사이드바 토글 + Q-Zoom + Tab 전환)

### Sector Watch 페이지 → Methodology 페이지로 진행

---


---

[← 05_holdings.md](05_holdings.md) | [07_methodology.md](07_methodology.md) →
