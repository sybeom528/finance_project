# C-1-4. Holdings 페이지

> **파일**: `05_holdings.md`
> **결정 시점**: 2026-05-10
> **상태**: 확정 (페이지 메타 Hold M-1~M-4 + 영역 1~9)
> **포함**: 페이지 메타 결정 / Sub-header / Holdings Summary KPI 6개 / Top N Holdings 표 + 비중 / 시가총액 분포 (Bubble / Treemap) / 보유 종목 변천사 / 종목별 기여도 분석 (Performance Attribution)

---

## C-1-4. Holdings 페이지

> **상태**: 진행 중 (메타 결정 확정 / 영역 1, 8 는 Overview 동일 / 2-7 미정)

### Holdings 페이지 통합 배경 (Context)

보유 종목 + 비중 + 시가총액 분포 + 변천사 분석. 메트릭 풀 (C-2) 의 **Pool-4 (운용 효율성)** + **Pool-5 (시장 비교)** 일부 활용.

**vs Sector Watch 페이지**:
- Holdings = 개별 종목 위주
- Sector Watch = 섹터 단위 위주

### 추가 결정 (Hold-Q1, 2026-05-10): final 정합성 + 학술 표준 메트릭 매핑

**배경**: 사용자 지시 — "본 페이지의 각 지표들도 final 과 비교하여 정확하게 동일한 결과가 나올 수 있도록 구성하고 검증까지 진행" (Phase 2.1 Risk-Q1 동일 패턴).

**Final 에 직접 산출된 메트릭** (`fund.pkl["comp"]` DataFrame, 192m × 10 컬럼 — `final/_extend_bl_to_2025.py:235~260` + `final/master_table.py:225~227`):

| Holdings 메트릭 | Final ground truth | 우리 lib | 정합 |
|---|---|---|---|
| Number of Holdings | `comp.n_stocks` | snapshot/avg | ✅ 직접 사용 |
| Effective N | `comp.eff_n` (= 1/Σw²) | snapshot 직접 / avg = `comp.eff_n.mean()` | ✅ |
| Top 1 Weight | `comp.top1_weight` | snapshot 직접 산출 (단순 max) | ✅ |
| Top 10 Share | `comp.top10_share` | snapshot 직접 산출 (단순 합) | ✅ |
| Avg Turnover | `comp.turnover.mean()` | `calc_avg_turnover` | ✅ final/master_table 정확 재현 |

**Final 에 부재 → 학술 표준 정의 (출처 명시)**:

| 메트릭 | 정의 | 출처 |
|---|---|---|
| **Single Stock HHI** | $\sum w_i^2$ (= 1/Effective N) | Hirschman (1945) "National Power and the Structure of Foreign Trade" |
| **Sector HHI** | $\sum_s (\sum_{i \in s} w_i)^2$ (sector level) | Hirschman (1945) — sector aggregate |
| **Top 5 Share** | $\sum_{i=1}^{5} w_{(i)}$ (Top 5 weight 합) | 표준 집중도 분석 |
| **Simple Contribution** | $C_i = \sum_t w_{i,t} \times R_{i,t}$ (per ticker) | Brinson, Hood & Beebower (1986) "Determinants of Portfolio Performance" Financial Analysts Journal |
| **ΔWeight** | $w_{i,t} - w_{i,t-1}$ (월 단위 변화) | 표준 운용 분석 |
| **12m Return** | $\prod_{k=t-11}^{t}(1 + r_{i,k}) - 1$ (rolling 12m cumprod) | 표준 누적 수익률 |
| **Market Cap** | $\exp(\text{log\_mcap})$ (monthly_panel 직접) | final 데이터 그대로 (D-1 정합) |

**SPY 벤치마크 처리** (Hold M-3 H3-3 정합):
- SPY 직접 weight 데이터 부재 → "펀드 universe 균등가중 (1/N)" 을 비교 기준선으로 사용
- 학술 정직성: "SPY 503 가정" 명시적 회피, "vs 균등 분산" 라벨로 표시
- 의미 있는 4 KPI 만 비교 (EffN / SingleHHI / SectorHHI / TopW)
- Number / Latest Turnover 는 비교 의미 ↓ → 펀드 자체값만

**Simple Contribution 검증**:
- Σ Contribution ≈ Fund 누적 수익률 (월별 선형 합)
- 차이 ε > 0 일반적 — 누적 수익률은 복리 (1+r)·(1+r')... 이지만 Simple Contribution 은 Σ(w·r) 단순 합 → 학술 한계 명시 (Brinson 1986 알려진 근사)
- 페이지 영역 7 끝에 ε 차이 명시 표시

**검증 결과** (실제 데이터 fund.weights + fund.comp + monthly_panel):
- Latest snapshot (2025-12): n_stocks=490 / eff_n=402.17 / top1=0.40% (ED) / top10=3.79% / turnover=26.9%
- 평균 (FULL 192m): n_stocks=421 / eff_n=215 / turnover=99% — final/master_table.py turnover_avg / eff_n_avg 정확 일치
- Single HHI = 1/eff_n 수학적 동치 검증 ✓

**결과 / 함의**:
- `lib/metric_calculators.py` 보강 — 8 신규 함수 (Top N share, Sector HHI, Holdings KPI snapshot/avg, Simple Contribution, Market Cap from panel, 12m Return, ΔWeight)
- `lib/holdings_charts.py` 신규 — 5 영역 함수 (KPI / Top N table / Treemap-Bubble / 변천사 / Tornado)
- 모든 결과가 final 정합 또는 학술 출처 명시 → 정직성 확보

### 페이지 메타 결정 (Hold M-1 ~ M-4)

#### Hold M-1. 영역 개수

**검토된 옵션**:
- (a) 압축 5 영역
- (b) 표준 7 영역
- (c) 풍부 8 영역
- (d) **풍부+ 9 영역** (구현 후 보강 결정 2026-05-10 — 영역 7 집중도 동적 추세 추가)

**결정**: (d) 풍부+ 9 영역 (1차 결정 c → 보강)

**근거**:
1. 운용 효율성 메트릭 (Pool-4) + 종목별 기여도 분석 모두 활용
2. 자유 탐색 모드 (A-2) 부합
3. 종목 변천사 + 기여도 분석 = 펀드 운용 narrative 깊이
4. **9영역 보강 (2026-05-10)**: 영역 6 (섹터 변천사) 와 동등한 가치의 종목 단위 동적 narrative 분리. 영역 6 = 거시 (섹터) / 영역 7 = 미시 (종목 집중도). sub-numbering "6.5" 보다 정식 영역 승격이 명료.

**변경 이력**:
- 1차 (2026-05-10): 8 영역 (Hold M-1 옵션 c)
- 2차 (2026-05-10 구현 후): 9 영역 — 영역 7 (집중도 동적 추세) 정식 승격, 기존 영역 7 (Tornado) → 영역 8, 기존 영역 8 (Footer) → 영역 9

#### Hold M-2. Sub-header 포함

**결정**: (a) 포함

**근거**: Performance/Risk Metrics 일관 — 인터랙션 일관성 원칙

#### Hold M-3. Holdings Summary KPI

**검토된 옵션** (메트릭 그룹):
- (a) 운용 효율성: Number / Eff N / Single HHI / Sector HHI / Avg Turnover
- (b) 집중도+분산: Number / Eff N / Top 1 / Top 5 / Top 10
- (c) 다각도
- (d) 자유

**1차 결정**: (a) + (b) 통합 → 8개 메트릭

**2차 결정 — 카드 수 명확화**:
- (A) 8개 카드 / **(B) 6개 카드 (Top Weights 통합)** / (C) 7개 / (D) 5개

**최종 결정**: 옵션 B (6개 카드)

**근거**:
1. **Top Weights (1/5/10) 통합 카드** — 집중도 narrative 자연 연결
2. **6개 = 반응형 한 줄** (큰 화면) / 3+3 분할 (작은 화면)
3. (A) 8개는 카드 너무 많음 / (D) 5개는 정보 손실

**6개 KPI**:
1. Number of Holdings
2. Effective N
3. Single Stock HHI
4. Sector HHI
5. **Top Holdings Weight (Top 1 / Top 5 / Top 10 combined)**
6. Avg Turnover

#### Hold M-4. 회사명 표시

**검토된 옵션**:
- (a) ticker only
- (b) ticker + 회사명
- (c) ticker + 회사명 + logo
- (d) 회사명만 추가

**yfinance 확인 결과**:
- yfinance Python 라이브러리는 회사명 (`longName` / `shortName`) 제공
- 예: `yf.Ticker("AAPL").info["longName"]` → "Apple Inc."
- ⚠ Rate limiting 회피 위해 한 번 수집 후 CSV 캐시 권장

**결정**: (b) ticker + 회사명

**근거**:
1. **가상 투자자 친화** — "AAPL" 보다 "Apple Inc." 인지 ↑
2. yfinance 활용 가능 (외부 의존 최소)
3. **CSV 캐시** = `data/ticker_company_map.csv` 한 번 수집 후 재사용 → rate limiting 회피
4. (c) logo 는 별도 데이터 수집 비용 ↑↑ → 제외

### Holdings 페이지 8 영역 구조

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트)
3. Holdings Summary KPI 6개     (Number / Eff N / Single HHI / Sector HHI / Top Weights / Avg Turnover)
4. Top N Holdings 표 + 비중     (현재 보유)
5. 시가총액 분포 (Bubble / Treemap)
6. 보유 종목 변천사 (Sankey / 시간별 weight)
7. 종목별 기여도 분석 (수익률 기여도)
8. Footer                       (Overview 동일)
```

### 영역 2: Sub-header — 확정

#### 결정 항목 H2-1: 텍스트 내용

**검토된 옵션 + 장단점**:

| 옵션 | 페이지 일관성 | 데이터 출처 노출 | 페이지별 특수성 |
|---|---|---|---|
| (a) Performance 패턴 동일 | ★★★ | ★ (Footer 의존) | ★ |
| (b) 더 자세 (yfinance 명시) | ★★ Holdings 만 다름 | ★★★ 즉시 명시 | ★★★ |

**결정**: (a) Performance 패턴 + Footnote 보조 표시

**근거**:
1. **8 페이지 일관성** 우선 — Sub-header 동일 패턴
2. **yfinance 컨텍스트 손실 방지** — Footnote (영역 7 끝 또는 영역 8 Footer) 에 작은 글씨 "회사명 매핑: yfinance 기반" 추가
3. 청중 인지 부담 ↓ + 학술 정직성 보존

**텍스트 안**:
```
Holdings (보유 종목)
보유 종목 / 비중 / 시가총액 분포 / 변천사 분석.
사이드바에서 기간 + 비교 벤치마크 토글 가능.
```

**Footnote** (영역 7 또는 영역 8): "※ 회사명 / 시가총액 / 섹터 매핑: yfinance 기반"

---

### 영역 3: Holdings Summary KPI 6개 — 확정

#### 영역 3 통합 배경 (Context)

이미 메트릭 확정 (Number / Eff N / Single HHI / Sector HHI / Top Weights / Avg Turnover) → 표시 방식만 결정.

#### 결정 항목 H3-1: 표시 기간

**검토된 옵션**:
- (a) Latest snapshot only
- (b) 사이드바 토글 평균
- (c) Latest snapshot + 사이드바 토글 평균 (양쪽)

**결정**: (c) Latest snapshot + 사이드바 토글 평균 (양쪽)

**근거**:
1. **Latest snapshot** = "현재 보유 상태" (정적 특성)
2. **기간 평균** = "기간별 운용 안정성" (동적 특성)
3. 두 값 동시 표시 → Holdings 의 본질 (정적 + 동적) 모두 노출
4. (a) Latest only 는 운용 변화 trend 손실 / (b) 평균 only 는 현재 상태 모호

**표시 방식**:
- 카드 안: 큰 숫자 (Latest) + 작은 숫자 ("XX 평균" 보조)
- 사이드바 토글에 따라 평균값 갱신

#### 결정 항목 H3-2: 카드 디자인

**검토된 옵션**:
- (a) Performance/Risk 일관 (단순)
- (b) Top Weights 카드만 다중 값
- (c) 추가 미니 시각화

**1차 결정**: (a) 단순

**2차 결정 — Top Weights 카드 명확화**:
- (a-1) 모든 카드 단일 값 (Top Weights = Top 5 만)
- (a-2) Top Weights 만 다중 값 예외 (★ 채택)
- (a-3) Top Weights 만 미니 시각화

**최종 결정**: (a-2) Top Weights 만 예외 (다중 값)

**근거**:
1. **Top Weights 본질** = 집중도 narrative (Top 1/5/10 함께 의미)
2. 단일 값 (Top 5 만) 은 정보 손실
3. 다른 카드는 단일 값으로 단순 유지 → 페이지 일관성 + 정보량 균형

#### 결정 항목 H3-3: 벤치마크 비교

**Holdings 메트릭별 SPY 비교 의미**:

| KPI | SPY 비교 의미 |
|---|---|
| Number of Holdings | 약함 (SPY=503) |
| Effective N | 의미 있음 |
| Single Stock HHI | 의미 있음 |
| Sector HHI | 강함 |
| Top Weights | 의미 있음 |
| Avg Turnover | 약함 (SPY ETF turnover 매우 낮음) |

**결정**: (b) 의미 있는 KPI 만 vs SPY (Eff N / Single HHI / Sector HHI / Top Weights)

**근거**:
1. Number / Avg Turnover 는 SPY 비교 의미 ↓ → 펀드 자체값만
2. 의미 있는 4 KPI 만 SPY 비교 → 정보 정확성
3. (c) 모든 KPI vs SPY 는 비교 의미 없는 KPI 까지 강제 → 청중 혼동

#### 결정 항목 H3-4: Hover tooltip

**결정**: (a) 포함

**근거**:
1. Performance / Risk Metrics 일관 — 모든 페이지 KPI tooltip 표준화
2. **Holdings 메트릭 정의** = 가상 투자자 친화 필수 (HHI, Effective N 등 어려움)

**Tooltip 텍스트 안**:
- Number of Holdings: "보유 종목 수"
- Effective N: "유효 종목 수 = 1 / Σw² (분산도 척도). 높을수록 분산"
- Single Stock HHI: "개별 종목 집중도 = Σw² (Herfindahl-Hirschman Index). 낮을수록 분산"
- Sector HHI: "섹터 집중도 = Σ(섹터 weight)² . 낮을수록 섹터 분산"
- Top Weights: "상위 N 종목의 비중 합계. 낮을수록 분산"
- Avg Turnover: "월평균 회전율. 낮을수록 안정 운용"

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- 카드 6개 = 반응형 한 줄 / 작은 화면 3+3 분할
- Latest snapshot = 최신 월 (예: 2025-12) 의 weight 기반 산출
- 기간 평균 = 사이드바 토글 (FULL/TEST/HO) 의 월별 weight 평균
- SPY 비교 KPI (Eff N / Single HHI / Sector HHI / Top Weights) 만 delta 표시
- Tooltip 텍스트 표준화 (구현 시 dictionary)
- Footnote (영역 7 끝 또는 영역 8): "회사명 매핑 yfinance" 추가

### 영역 4: Top N Holdings 표 + 비중 — 확정

#### 영역 4 통합 배경 (Context)

현재 보유 종목 Top N 의 자세한 표 — ticker / 회사명 / 섹터 / 시가총액 / 비중 표시. 펀드의 핵심 보유 종목 직관적 노출.

#### 결정 항목 H4-1: Top N 개수

**검토된 옵션**:
- (a) Top 10 (표준)
- (b) Top 20
- (c) Top N 토글 (10/20/50/All)
- (d) Pagination

**결정**: (c) Top N 토글

**근거**:
1. **자유 탐색 모드 (A-2)** 부합 — 사용자 자율 선택
2. 기본 Top 10 (간결) → 사용자가 깊이 탐색 시 50/All
3. (d) Pagination 은 페이지 분할로 한눈에 파악 ↓

#### 결정 항목 H4-2: 표 컬럼

**결정**: (b) 표준 7컬럼

**컬럼 구성**:
1. Rank
2. Ticker + Company (yfinance 매핑)
3. Sector (GICS)
4. Weight (현재 비중)
5. Market Cap ($B)
6. 12m Return (최근 12개월 수익률)
7. ΔWeight (vs 전월 비중 변화)

**근거**:
1. **종목 detail 충분** — 청중이 한 표에서 종목 정보 종합 이해
2. (a) 5컬럼은 변화 (ΔW) / 수익률 (12m R) 누락 → 트렌드 인지 ↓
3. (c) 9컬럼은 영역 7 (기여도 분석) 와 중복

#### 결정 항목 H4-3: 표시 기간

**결정**: (a) Latest only

**근거**:
1. **Top N 표 = 정적 snapshot** 적합
2. 시간별 변화는 영역 6 (보유 종목 변천사) 에서 자세히
3. (b)/(c) 시점 선택은 영역 6 에서 다룸

#### 결정 항목 H4-4: 정렬

**결정**: (c) 비중 default + 사용자 정렬

**근거**:
1. **기본 비중 내림차순** (Top N 의 표준)
2. **사용자 자율 정렬** — 모든 컬럼 클릭 정렬 가능
3. 자유 탐색 모드 부합

#### 결정 항목 H4-5: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- 종목 클릭 → 종목 detail expand ✓ (Q-Zoom 일관)
- 컬럼 정렬 ✓ (H4-4 와 연결)
- CSV 다운로드 ✓
- 검색 (ticker / 회사명 검색) ✓

#### 결정 항목 H4-6: 시각 보강

**결정**: (a) Weight 막대 + (b) 섹터 색상 코딩

**근거**:
1. **Weight 막대** = 비중 시각화 (셀 안 막대)
2. **섹터 색상 코딩** = row 배경색 또는 sector 컬럼 색상
3. (c) 시가총액 시각화는 영역 5 (Bubble/Treemap) 에서 자세히 → 중복 회피

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- 표 구현 = Streamlit `st.dataframe` 또는 `st.data_editor` (정렬 / 검색 자동 지원)
- Weight 막대 = `st.column_config.ProgressColumn` (Streamlit 1.30+ 신기능)
- 섹터 색상 = `pandas.Styler.background_gradient` 또는 cell 단위 styling
- ticker → 회사명 매핑 = `data/ticker_company_map.csv` 캐시 (yfinance 1회 수집 후)
- 종목 클릭 expand = `st.dataframe(on_select='rerun')` (Streamlit 1.27+)

### 영역 5: 시가총액 분포 (Bubble / Treemap) — 확정

#### 영역 5 통합 배경 (Context)

각 보유 종목의 시가총액 + 비중 + 섹터를 시각적으로 한눈에 파악. Top N 표 (영역 4) 의 정보를 시각화로 보강.

레퍼런스 PortfolioX360 의 **Sector Tree Map** + Holdings 분석 표준 패턴.

#### 결정 항목 H5-1: 차트 종류

**검토된 옵션**:
- (a) Bubble Chart
- (b) Treemap
- (c) Sunburst (계층형)
- (d) Bubble + Treemap (탭 전환)

**결정**: (d) Bubble + Treemap (탭 전환)

**근거**:
1. **두 시각화 모두 유의미**:
   - Treemap = 비중 직관 (면적 = 비중)
   - Bubble = 다차원 분석 (X/Y/크기/색)
2. 자유 탐색 모드 (A-2) 부합 — 사용자 자율 선택
3. (c) Sunburst 는 영역 6 (변천사) 와 시각 유사 가능 → 차별화 필요

#### 결정 항목 H5-2: Treemap 차원

**검토된 옵션**:
- (a) 면적=비중 / 색상=섹터
- (b) 면적=비중 / 색상=12m 수익률 (red-green)
- (c) 면적=시가총액 / 색상=섹터
- (d) 토글 (a/b/c)

**결정**: (d) 토글

**근거**:
1. **자유 탐색** — 사용자가 분석 관점 자유 선택
2. 기본: (a) 면적=비중, 색상=섹터 (가장 직관)
3. 토글로 (b) 수익률 강도, (c) 시장 비중 vs 펀드 비중 비교 가능

#### 결정 항목 H5-3: Bubble 차원

**검토된 옵션**:
- (a) X=시가총액, Y=비중, 크기=시가총액, 색=섹터
- (b) X=12m 수익률, Y=비중, 크기=시가총액, 색=섹터
- (c) X=Beta, Y=비중, 크기=시가총액, 색=섹터
- (d) 사용자 선택 (X/Y/크기/색 모두 토글)

**결정**: (d) 사용자 선택

**근거**:
1. **자유 탐색 max** — X/Y/크기/색 4축 자유 조합
2. 기본: (a) 시가총액-비중 관계
3. 사용자가 다양한 차원 (수익률, Beta, Volatility 등) 자유 비교 가능

#### 결정 항목 H5-4: 표시 기간

**검토된 옵션**:
- (a) Latest only
- (b) 사이드바 토글 시점
- (c) 시점 선택 슬라이더 (월 단위)
- (d) 시간 애니메이션

**결정**: (c) 시점 선택 슬라이더

**근거**:
1. **월별 시점 자유 선택** — 시간 변화 관찰 가능
2. (d) 애니메이션은 구현 복잡 + 청중 통제 ↓
3. 슬라이더 = Plotly 기본 기능 (간단 구현)

#### 결정 항목 H5-5: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip (종목 detail) ✓
- 종목 클릭 → 종목 detail expand ✓ (Q-Zoom)
- 섹터 필터 (클릭으로 섹터만 강조) ✓
- Zoom / Pan ✓ (Plotly 기본)
- 다운로드 (PNG / CSV) ✓

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- Treemap 구현 = Plotly `px.treemap`
- Bubble 구현 = Plotly `px.scatter` (size 파라미터)
- 시점 슬라이더 = Plotly `animation_frame` 또는 Streamlit `st.slider`
- 사용자 선택 (X/Y/크기/색) = Streamlit `st.selectbox` × 4
- 색상 팔레트 = GICS 11개 섹터 표준 색상

### 영역 6: 보유 종목 변천사 (Holdings History) — 확정

#### 영역 6 통합 배경 (Context)

시간에 따른 보유 종목 + 비중 변화 시각화. 펀드의 적응형 운용 narrative 강화.

**핵심 메시지**:
- "BL+LSTM 이 시점별로 어떤 종목에 더/덜 비중을 두었는지"
- "Sector 시기별 rotation"
- "신규 종목 / 제외 종목 흐름"

**vs 영역 4 (Top N 표)**: 영역 4 = 정적 snapshot / 영역 6 = 동적 변화

#### 결정 항목 H6-1: 차트 종류

**검토된 옵션**:
- (a) Stacked Area Chart
- (b) Stream Graph
- (c) Sankey Diagram
- (d) Heatmap (시간 × 종목)
- (e) Multi-line (Top N 종목 라인)
- (f) 토글 (Stacked + Multi-line)

**결정**: (e) Multi-line

**근거**:
1. **개별 종목 추적** = 각 Top N 종목의 비중 변화 명확 추적
2. (a) Stacked Area = 누적 (전체 구성), 개별 변화 추적 ↓
3. (c) Sankey = 시기 흐름 시각이지만 시간 척도 ↓
4. Multi-line + 토글 (H6-2) = 섹터별 / Top N 별 자유 선택

#### 결정 항목 H6-2: 종목 그룹화

**검토된 옵션**:
- (a) 섹터별 그룹화 (11개 섹터)
- (b) Top N 종목 + Others
- (c) 모든 보유 종목
- (d) 토글 (a/b)

**결정**: (d) 토글 (섹터 / Top N + Others)

**근거**:
1. **자유 탐색** — 사용자가 섹터 vs 개별 종목 자유 선택
2. (a) 섹터만 = 종목 변화 손실
3. (b) Top N 만 = 섹터 narrative 손실
4. 토글로 두 관점 모두 가능

#### 결정 항목 H6-3: 시간 단위

**검토된 옵션**:
- (a) 월별 (192 시점)
- (b) 분기별 (64 시점)
- (c) 연도별 (16 시점)
- (d) 토글

**결정**: (a) + (b) 토글 (월별 / 분기별)

**근거**:
1. **월별** = 가장 정밀 (BL 월별 rebalancing 과 매칭)
2. **분기별** = 중간 척도 (시각 깔끔)
3. (c) 연도별은 정보 손실 ↑ → 제외
4. 사용자가 정밀도 자유 선택

#### 결정 항목 H6-4: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip (시점 + 비중) ✓
- 종목/섹터 클릭 → 단독 강조 ✓
- 시점 클릭 → 해당 시점 detail expand ✓ (Q-Zoom)
- 종목 검색 / 필터 ✓

#### 결정 항목 H6-5: 추가 표시

**결정**: (d) 모두 (Regime 배경색 + 신규/제외 마커 + 이벤트 annotation)

**근거**:
1. **Regime 배경색** = 영역 3 (Overview) / 영역 6 (Performance) 일관
2. **신규/제외 마커** = 펀드 종목 진입/퇴출 시각 (적응형 운용 narrative)
3. **이벤트 annotation** = COVID-19 / 2022 Bear / 2024 Sector Rotation 시점
4. 모두 채택 = 정보량 max + 학술 정직성

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- Multi-line 구현 = Plotly `go.Scatter` (mode='lines') 다중 trace
- Regime 배경색 = `add_vrect` 활용
- 신규/제외 마커 = `add_annotation` 또는 별도 `go.Scatter` (mode='markers')
- 이벤트 annotation = COVID/2022 Bear/2024 dictionary 정의
- 검색 = Streamlit `st.text_input` + 라인 강조 동적

### 영역 7: 시점별 Top N 합 vs Others (집중도 동적 추세) — 확정 (2026-05-10 보강)

#### 영역 7 통합 배경 (Context)

영역 6 (섹터 변천사) 와 동등한 가치의 **종목 단위 동적 narrative** 영역. 펀드 집중도가 시점별로 어떻게 변하는지 시각화 — 시장 stress 시 (예: 2015-05 utility 도피, 2022-08 Bear) 펀드가 Top N 종목에 얼마나 집중하는지 직접 표현.

**vs 영역 6 (섹터)**:
- 영역 6 = 11 GICS 섹터 weight 시계열 (거시)
- 영역 7 = Top N (시점별 동적 ticker) weight 합 시계열 (미시)

#### 결정 항목 H7c-1: 차트 종류

**검토된 옵션**:
- (a) Multi-line (Top N 합 + Others 두 라인)
- (b) **100%-stacked area** (Top N + Others 누적)
- (c) Heatmap

**결정**: (b) 100%-stacked area

**근거**:
1. Top N + Others = 100% → 두 라인의 거꾸로 대칭 중복 회피 (사용자 지적 2026-05-10)
2. 면적 시각화로 집중도 변화 직관적 인지 (큰 spike = 큰 면적)
3. (a) Multi-line 은 두 라인이 거꾸로 대칭이라 정보 중복

#### 결정 항목 H7c-2: Top N 토글

**결정**: 1 / 5 / 10 / 20 토글

**근거**:
1. **Top 10 default** — final.comp.top10_share 직접 정합 (max diff 0)
2. **Top 1** — final.comp.top1_weight 직접 정합 (집중도 극단 표현)
3. **Top 5 / 20** — 직접 산출 (각 시점 nlargest sum)
4. 단일 토글로 다양한 집중도 관점 자유 탐색

#### 결정 항목 H7c-3: Hover 정보

**결정**: 시점별 Top N 종목 list + 각 비중 표시

**근거**:
1. 영역 6 (섹터) 와 차별화 — 종목 narrative 본질
2. 사용자가 hover 만으로 그 시점의 Top N 종목 (예: 2020-04 COVID Defensive 종목 list) 확인 가능
3. 시점 슬라이더 등 별도 UI 불필요 — 마우스 hover 만으로 인터랙티브 탐색

**Hover 형식 예시**:
```
2015-05 — Top 10 합: 79.47%
  1. ED: 10.00%
  2. DUK: 10.00%
  3. SO: 10.00%
  ...
```

#### 결과 / 함의

- 100%-stacked area = 종목 집중도 동적 추세를 한눈에
- final.comp 직접 정합 → 별도 산출 오류 0
- Hover = 시점별 Top N 종목 list 인터랙티브 탐색 (영역 6 의 정적 차트와 차별화)
- **영역 6 (거시) ↔ 영역 7 (미시) Top-Down narrative 흐름 완성**

---

### 영역 8: 종목별 기여도 분석 (Performance Attribution) — 확정

#### 영역 8 통합 배경 (Context)

각 종목이 펀드 수익률에 얼마나 기여했는지 정량 분석.

**학술 배경**:
- **Simple Contribution**: $\text{Contribution}_i = w_i \times R_i$
- **Brinson-Hood-Beebower (BHB) Attribution**: Allocation + Selection + Interaction 분해 (Brinson et al. 1986)
- **Active Return Attribution**: (Fund - Benchmark) 종목별 분해

#### 결정 항목 H7-1: Attribution 방법

**검토된 옵션**:
- (a) Simple Contribution
- (b) Brinson-Hood-Beebower (BHB)
- (c) Active Return Attribution
- (d) Simple + BHB
- (e) Simple + Active

**결정**: (a) Simple Contribution

**근거**:
1. **단순 + 청중 친화** — 가상 투자자가 즉시 이해 ($w \times R$)
2. **수학적 검증 가능** — Σ Contribution = 펀드 총 수익률
3. (b) BHB 는 학술 깊이 ↑ but 구현 복잡 + 청중 어려움
4. (c) Active Return Attribution 은 SPY 비교 필요 (다중 토글 시 복잡)

#### 결정 항목 H7-2: 표시 형식

**결정**: (a) Tornado Chart (가로 막대 양수/음수)

**근거**:
1. **양수/음수 즉시 시각** — Top contributors / Bottom contributors 분리
2. 표준 Attribution 시각화 패턴
3. (b) Waterfall 은 누적 시각, 종목 비교 ↓
4. (d) 표는 시각적 인상 ↓

#### 결정 항목 H7-3: Top N

**결정**: (a) Top N 양수 + Top N 음수

**근거**:
1. **학술 정직성** — Bottom contributors 도 노출 (감추지 않음)
2. 청중에게 "성공/실패 종목 모두" 보여줌
3. (b) 절대값 기준은 음수 종목 누락 가능
4. 기본: Top 10 + Bottom 10 (사용자 토글 가능)

#### 결정 항목 H7-4: 표시 기간

**결정**: (a) 사이드바 토글 반응

**근거**:
1. **인터랙션 일관성 원칙** 준수
2. 기간별 (FULL/TEST/HO) 기여도 변화 가능
3. (c) Regime 별 분해는 Backtesting 페이지의 Regime 분석에 위임

#### 결정 항목 H7-5: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- 종목 클릭 → 종목 detail expand ✓ (Q-Zoom)
- Top N 토글 (10/20/50) ✓
- CSV 다운로드 ✓

#### 결정 항목 H7-6: 추가 표시

**결정**: (d) 모두 (섹터 합계 + 양수/음수 색상 + 총수익률 검증)

**근거**:
1. **섹터 합계** = Sector Watch 페이지로의 자연 navigation
2. **양수/음수 색상** = green/red 즉시 시각
3. **총수익률 검증** = "Σ Contribution = Fund Return" 학술 정확성

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- Tornado chart 구현 = Plotly `go.Bar` (orientation='h', sorted)
- 양수/음수 색상 = `marker.color` 동적 (green/red)
- Sector 합계 = pandas `groupby('sector').sum()` 활용
- 총수익률 검증 = mathematical assertion (검증 결과 표시)

---

### Holdings 페이지 — 전체 확정 (영역 1~8)

#### 페이지 시각화 통합 (확정 사항 모두 조합)

```
┌────────────────────────────────────────────────────────────────┐
│ [영역 1: Header — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 2: Sub-header 카드/배너]                                  │
│ Holdings (보유 종목) — 보유 종목 / 비중 / 시가총액 / 변천사     │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 3: Holdings Summary KPI 6개]                              │
│ Number / Eff N / Single HHI / Sector HHI / Top Weights / Turnover│
│ Latest snapshot + 기간 평균 + Tooltip                          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 4: Top N Holdings 표 + 비중]                              │
│ Top 토글 (10/20/50/All) + 7컬럼 + Weight 막대 + 섹터 색상     │
│ ticker + 회사명 (yfinance)                                     │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 5: 시가총액 분포]                                         │
│ Bubble + Treemap (탭) + 사용자 차원 선택 + 시점 슬라이더       │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 6: 보유 종목 변천사 (Multi-line)]                         │
│ 섹터/Top N+Others 토글 + 월별/분기별 토글                      │
│ Regime 배경 + 신규/제외 마커 + 이벤트 annotation              │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 7: 종목별 기여도 (Tornado Chart)]                         │
│ Top + Bottom + 섹터 합계 + 총수익률 검증                       │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 8: Footer — Overview 동일]                                │
│ ※ 회사명 / 시가총액 / 섹터: yfinance 기반 (Footnote)           │
└────────────────────────────────────────────────────────────────┘
```

#### Holdings 페이지 결정 결과 / 함의

- **8 영역 모두 확정** → 구현 시 명확한 와이어프레임
- **yfinance 의존**: ticker → 회사명 매핑 = `data/ticker_company_map.csv` 캐시 (한 번 수집)
- **인터랙션 일관성 원칙** 적용:
  - 사이드바 토글 (기간) → 영역 3, 7 영향
  - Q-Zoom → 영역 4 (종목 클릭 expand) / 영역 5 (종목 클릭) / 영역 6 (시점 클릭) / 영역 7 (종목 클릭)
- **다음 페이지 (Sector Watch)** 결정 시 동일 패턴 활용 + 섹터 narrative 강화

### Holdings 페이지 → Sector Watch 페이지로 진행

---


---

[← 04_risk_metrics.md](04_risk_metrics.md) | [06_sector_watch.md](06_sector_watch.md) →
