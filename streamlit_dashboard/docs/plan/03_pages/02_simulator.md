# Investment Simulator 페이지 — 와이어프레임

> **관련 decisionlog**: `11_dl_sections.md` F-6 (★ 신규 페이지)
> **상태**: 확정
> **결정 수**: 7 영역 (메타 Sim M-1 ~ M-4 + 영역 1~7)

---

## 페이지 역할 정의

**사용자 요청 핵심**: "사용자가 실제로 기간과 금액을 입력해서 '내가 이때 얼마를 넣었으면 언제까지 또는 지금까지 얼마를 벌었겠구나' 를 느낄 수 있는 기능."

**5분 demo 흐름 (K-2)** 에서 1.5분 차지 — 가상 투자자 친화 + 마케팅 핵심 기능.

**옵션 4 (종합) 채택**:
- Lump-sum (일시 투자)
- DCA (Dollar Cost Averaging, 분산 투자)
- Goal-based (목표 금액 역산)

---

## 페이지 영역 구조 (시선 흐름)

```
1. Header                 → Overview 동일
2. Sub-header             → "내 투자 시뮬레이션" 친근 톤
   (+ Sim disclaimer 박스 — I-5 결정)
3. Input 영역             → Tab (Lump-sum / DCA / Goal) + 다단 입력
4. Result KPI 카드 5개    → 최종자산 / 총수익 / CAGR / MDD / 총투자
5. 누적 자산 곡선         → Fund vs SPY/EW/IVW + DCA 누적 + Regime + 위기 annotation
6. Insight 박스           → 정적 템플릿 카드 그리드 (4-8개 조건부)
7. Footer                 → Overview 동일
```

---

## 영역별 와이어프레임

### 영역 1: Header

**결정사항**: Overview 영역 1 동일 패턴

→ `02_common.md` 의 `render_page_header()` 호출

---

### 영역 2: Sub-header (Sim2-1 친근 톤)

**결정사항** (Sim2-1):
- (b) 친근 톤 — 가상 투자자 친화 (마케팅 핵심 기능)

**텍스트**:
```
"내가 이때 얼마를 투자했더라면?"
실제 수익을 시뮬레이션해 보세요.
```

**+ I-5 Sim disclaimer 박스** (영역 2 하단 / 영역 3 상단 사이):

```
⚠️ 본 시뮬레이션은 가상의 백테스트 결과이며, 실제 투자권유 또는
   투자 자문을 목적으로 하지 않습니다. 과거의 성과는 미래의 수익을
   보장하지 않습니다.
```

**구현 체크리스트**:
- [ ] `render_subheader()` 호출 (`02_common.md` 참조)
- [ ] `render_simulator_disclaimer()` 호출 (`lib/disclosure.py`)

---

### 영역 3: Input 영역

**결정사항** (Sim3-1 ~ Sim3-4):
- 시나리오 토글: (a) Tab 전환 (Lump-sum / DCA / Goal)
- 입력 필드: (b) 표준 (시작 / 종료 / 초기 / DCA 매월 / Goal 금액)
- 시점 입력: (a) Date input 2개
- 금액 입력: (c) Number + Slider 조합 ($100 ~ $1M)

**시각화 예시**:

```
┌────────────────────────────────────────────────────────────────┐
│ [Lump-sum | DCA | Goal-based]   ← Tab 전환                     │
│                                                                │
│ ── 공통 입력 ──                                                │
│ 시작 시점: [2015-01-01 ▼]    종료 시점: [2025-12-31 ▼]        │
│                                                                │
│ 초기 금액: [$10,000          ] [Slider ──●──────]              │
│                                                                │
│ ── DCA Tab 활성 시 ──                                          │
│ 매월 추가 투자: [$500    ] [Slider ──●──────]                  │
│                                                                │
│ ── Goal Tab 활성 시 ──                                         │
│ 목표 금액: [$1,000,000  ]                                      │
│                                                                │
│ [▶ 시뮬레이션 실행] (또는 자동 실행 — Streamlit reactive)      │
└────────────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] `st.tabs(["Lump-sum", "DCA", "Goal-based"])`
- [ ] 각 Tab 안에 입력 필드:
  - 공통: `st.date_input("시작 시점")`, `st.date_input("종료 시점")`
  - 공통: `st.number_input("초기 금액", min_value=100, max_value=1_000_000)` + `st.slider`
  - DCA: `st.number_input("매월 추가 투자")` + `st.slider`
  - Goal: `st.number_input("목표 금액")`
- [ ] Streamlit reactive — 입력 변경 시 자동 재계산 (별도 버튼 X)
- [ ] 시작 ≤ 종료 검증
- [ ] 데이터 범위 (2010-01 ~ 2025-12) 안에서만 가능

**코드 snippet (예시)**:
```python
def render_input_section():
    """영역 3 Input 영역."""
    tabs = st.tabs(["Lump-sum (일시 투자)", "DCA (분산 투자)", "Goal-based (목표 역산)"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작 시점", value=pd.Timestamp("2015-01-01"))
        with col2:
            end_date = st.date_input("종료 시점", value=pd.Timestamp("2025-12-31"))

        initial_amount = st.number_input(
            "초기 금액 ($)",
            min_value=100, max_value=1_000_000,
            value=10_000, step=100
        )

        scenario = "lump_sum"
        sim_input = {
            "scenario": scenario,
            "start_date": start_date,
            "end_date": end_date,
            "initial_amount": initial_amount
        }
    # DCA, Goal Tabs 도 유사
    return sim_input
```

---

### 영역 4: Result KPI 카드 5개

**결정사항** (Sim4-1, Sim4-2):
- KPI 5개: 최종자산 / 총수익 / 연환산 CAGR / MDD / 총투자금액
- 디자인: (a) Performance 영역 3 단순 패턴 (큰 숫자 + tooltip)

**시각화 예시**:

```
┌────────────┬────────────┬────────────┬────────────┬────────────┐
│ 최종 자산  │ 총 수익    │ 연환산     │ 최대낙폭   │ 총 투자    │
│ Final      │ Total      │ CAGR       │ Max DD     │ Invested   │
│ $XXX,XXX   │ +$XX,XXX   │ +X.X%      │ -X.X%      │ $XX,XXX    │
│ ⓘ Final    │ ⓘ Profit   │ ⓘ CAGR     │ ⓘ MDD      │ ⓘ Invested │
└────────────┴────────────┴────────────┴────────────┴────────────┘
```

**구현 체크리스트**:
- [ ] `st.columns(5)` 반응형
- [ ] 각 카드 = 큰 숫자 + tooltip (`get_tooltip()` 활용)
- [ ] DCA 시나리오: 총 투자 = 초기 + (매월 × 개월수)
- [ ] Goal 시나리오: 목표 도달 시점 + 추가 분석

---

### 영역 5: 누적 자산 곡선

**결정사항** (Sim5-1 ~ Sim5-3):
- 차트 구성: (b) Fund + 사이드바 토글 활성 벤치마크
- 추가 표시: (d) 모두 (DCA 누적 투자금액 라인 + Regime 배경 + 위기 annotation)
- 인터랙션: 모두 채택 (Hover / Zoom + Slider / Y축 토글 / 위기 annotation)

**시각화 예시**:

```
┌────────────────────────────────────────────────────────────────┐
│ [영역 5: 누적 자산 곡선 — 사이드바 토글 반응]                   │
│ Y축: [Linear ▼]   기간 슬라이더: 2010 ●━━━━━━━━━━━━━━ 2025    │
│                                                                │
│ ┌─R1─┬─R2─┬─R3─┬─HO─┐ (Regime 배경색)                         │
│                                                                │
│ 자산 │                                                         │
│  $50K┤                       ╱─── Fund                         │
│  $40K┤                    ╱─                                   │
│  $30K┤                 ╱──                                     │
│  $20K┤              ╱──         ─── SPY (사이드바 토글)        │
│  $10K┤        ╱─────                                           │
│   $5K┤  ─────                ─── EW / IVW (사이드바 토글)      │
│      │                                                         │
│   ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ DCA 누적 투자금액 (DCA Tab 활성)    │
│      │                                                         │
│      │ ▼COVID  ▼2022 Bear  ▼2024 IT                           │
│      └───────────────────────────────────────                 │
│        2015    2018    2022    2025                            │
└────────────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Plotly `go.Scatter` 다중 라인:
  - Fund (Cobalt Blue, 메인)
  - SPY / EW / IVW (사이드바 토글 활성에 따라 동적)
  - DCA 누적 투자금액 (점선, DCA Tab 활성 시)
- [ ] Regime 배경색 (`add_regime_backgrounds()` 헬퍼)
- [ ] 위기 annotation (`add_event_annotations()` 헬퍼)
- [ ] Y축 토글 (Linear / Log)
- [ ] Range slider
- [ ] Hover tooltip

---

### 영역 6: Insight 박스

**결정사항** (Sim6-1, Sim6-2, Sim6-Q):
- 내용: (b) 표준 (수익 + 비교 narrative)
- 형식: (b) 카드 그리드
- 다국어: (a) 한/영 병기 (A-3 일관)

**구현 파일**: `lib/insight_generator.py` 활용 (`02_common.md` 참조)

**조건부 카드 (4-8개)**:

| 카드 | 표시 조건 | 예시 |
|---|---|---|
| 💰 누적 수익 | 항상 | "$10,000 → $XX,XXX (+$X,XXX, +X.X%)" |
| 📈 연환산 CAGR | 항상 | "+X.X% per year" |
| 📊 vs SPY | SPY 토글 활성 | "SPY 대비 +X.X%" |
| 📊 vs EW | EW 토글 활성 | "EW 대비 +X.X%" |
| 📊 vs IVW | IVW 토글 활성 | "IVW 대비 +X.X%" |
| ⚠️ 최대 손실 / 회복 | 항상 | "COVID-19 -X.X% / 회복 5개월" |
| 🔄 DCA 효과 | DCA Tab 활성 | "매월 $500 분산 투자 / 일시 투자 대비 +$X" |
| 🎯 Goal 달성 분석 | Goal Tab 활성 | "$1M 목표 / 달성 시점: 20XX-XX" |

**시각화 예시**:

```
┌────────────────────────────────────────────────────────────────┐
│ [영역 6: Insight 박스 — 카드 그리드]                            │
│ ┌──────────┬──────────┬──────────┐                            │
│ │💰 누적    │📈 연환산  │📊 vs SPY  │                            │
│ │수익       │CAGR      │           │                            │
│ │$10K→$XXK │+X.X%/yr  │+X.X% 우월 │                            │
│ ├──────────┼──────────┼──────────┤                            │
│ │⚠️ 최대   │🔄 DCA    │🎯 Goal    │ (조건부)                    │
│ │손실/회복 │효과      │달성       │                            │
│ │COVID -X% │+$X 효과  │달성 시점  │                            │
│ └──────────┴──────────┴──────────┘                            │
└────────────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] `generate_insight_cards(sim_result, benchmarks, scenario)` 호출
- [ ] `render_insight_grid(cards)` 호출
- [ ] 한/영 병기 (각 카드 title)
- [ ] 색상 코딩 (양수 = green / 음수 = red / 중립 = blue)
- [ ] **LLM 미사용** (정적 템플릿 + 동적 값 채움)

**구현 비용**: 2-3시간 (코드 50-100 lines)

---

### 영역 7: Footer

**결정사항**: Overview 영역 6 동일 패턴

→ `02_common.md` 의 `render_footer()` 호출

---

## 페이지 데이터 의존성

- **Fund 월별 수익률** (D 섹션 캐시 활용 — `lib/data_loader.py`)
- **SPY / EW / IVW 월별 수익률** (D 섹션 캐시 활용)
- **위기 시기 dictionary** (영역 5 annotation — `02_common.md` `add_event_annotations()`)

### DCA / Goal-based 산출 로직

- **DCA**:
  - 매월 추가 매수 → 누적 자산 (월별 시뮬레이션)
  - `total_invested = initial + monthly * months`
- **Goal-based**:
  - 역산: 주어진 시작/종료 + 목표 금액 → 필요 초기 금액 (binary search 또는 closed-form)

```python
def simulate_lump_sum(start_date, end_date, initial_amount, fund_returns):
    """일시 투자 시뮬레이션."""
    period_returns = fund_returns.loc[start_date:end_date]
    cumulative = (1 + period_returns).cumprod()
    final_value = initial_amount * cumulative.iloc[-1]
    return {
        "final_value": final_value,
        "total_profit": final_value - initial_amount,
        "cagr": (final_value / initial_amount) ** (12 / len(period_returns)) - 1,
        "mdd": compute_mdd(cumulative),
        "total_invested": initial_amount,
        "value_series": initial_amount * cumulative,
    }

def simulate_dca(start_date, end_date, initial_amount, monthly_amount, fund_returns):
    """DCA (Dollar Cost Averaging) 시뮬레이션."""
    period_returns = fund_returns.loc[start_date:end_date]
    value = initial_amount
    value_series = []
    for ret in period_returns:
        value = value * (1 + ret) + monthly_amount  # 매월 말 추가 투자
        value_series.append(value)
    total_invested = initial_amount + monthly_amount * len(period_returns)
    final_value = value
    return {
        "final_value": final_value,
        "total_profit": final_value - total_invested,
        "cagr": (final_value / total_invested) ** (12 / len(period_returns)) - 1,
        "mdd": ...,
        "total_invested": total_invested,
        "value_series": pd.Series(value_series, index=period_returns.index),
        "dca_monthly": monthly_amount,
    }
```

---

## 메트릭 (C-2 풀에서 picking)

- KPI 5개: Final Value (custom) / Total Profit (custom) / CAGR (Pool-1) / MDD (Pool-3) / Total Invested (custom)

---

## 인터랙션 / 토글 적용

| 영역 | 사이드바 토글 영향 | Q-Zoom |
|---|---|---|
| 영역 1 (Header) | ✗ | ✗ |
| 영역 2 (Sub-header + Disclaimer) | ✗ | ✗ |
| 영역 3 (Input) | ✗ | ✗ |
| 영역 4 (KPI 5개) | ✗ (Sim 자체 입력 우선) | ✗ |
| 영역 5 (자산 곡선) | ✓ 비교 토글 (SPY/EW/IVW) (G-2 결정) | ✓ 시기 클릭 → expand |
| 영역 6 (Insight) | ✓ 활성 벤치마크에 따라 카드 동적 추가 | ✗ |
| 영역 7 (Footer) | ✗ | ✗ |

**G-2 결정**: Investment Simulator 도 사이드바 토글 영향 (인터랙션 일관성 원칙)

---

## 페이지 구현 우선순위

- **Phase 2 (확장, 2-3주)**: Investment Simulator (Phase 2 의 핵심 — 5분 demo 1.5분 차지)
- 의존성: D 섹션 (data_loader) 완료 필요
- `lib/insight_generator.py` 별도 작성 필요

---

## 결과 / 함의

- **마케팅 친화 핵심 기능** = 5분 demo 의 1.5분 차지 (K-2 결정)
- **Insight 박스** = LLM 미사용 (정적 템플릿) → 빠른 응답 + 일관된 메시지
- **인터랙션 일관성** = 사이드바 토글 + Q-Zoom 모두 적용
- **데이터 의존**: Fund 월별 수익률 + SPY/EW/IVW (D 섹션 캐시)

---

[← 01_overview.md](01_overview.md) | [03_performance.md →](03_performance.md)
