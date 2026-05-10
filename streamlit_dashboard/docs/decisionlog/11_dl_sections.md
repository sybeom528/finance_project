# D~L 섹션 — 데이터 / Disclosure / 시뮬레이션 / UX / 디자인 / 컴플라이언스 / 기술 / Storytelling / 한계

> **파일**: `11_dl_sections.md`
> **결정 시점**: 2026-05-10 ~ (진행 중)
> **상태**: D 확정 / E 확정 / F~L 미정 또는 부분 확정
> **포함**: D 데이터 / E HO Disclosure / F 시뮬레이션 / G 인터랙션 UX / H 디자인 / I 컴플라이언스 / J 기술 / K Storytelling / L 한계

---

# D 섹션 — 데이터 준비 (Data Layer) — 확정

> **결정 시점**: 2026-05-10 / **상태**: 확정

## D 섹션 통합 배경 (Context)

대시보드 구현 시 필요한 데이터 파일 + 캐싱 + 회사명 매핑 등 기술적 결정.

**이미 부분 확정** (다른 섹션 결정에 의해):
- ticker 회사명 매핑 = yfinance 1회 수집 후 CSV 캐시 (Holdings M-4)
- 거래비용 20bp + Net 메트릭 산출 (C-2-4)
- EW + IVW baseline 산출 (Overview 영역 3)

## D-1. 데이터 파일 위치

**검토된 옵션**:
- (a) final/data/ 참조 (path 지정)
- (b) streamlit_dashboard/data/ 에 복사
- (c) 핵심만 복사 + 나머지 참조

**결정**: (c) 핵심만 복사 + 나머지 참조

**근거**:
1. **핵심 데이터 = 자주 접근** → 복사로 IO 부담 ↓
2. **나머지 (156 config pkl) = 참조** → 데이터 중복 회피
3. (a) 모두 참조는 streamlit_dashboard 폴더 독립성 ↓
4. (b) 모두 복사는 데이터 중복 ↑↑

**핵심 복사 대상**:
- `monthly_panel.csv` (rf, spy_ret, sector, log_mcap)
- `daily_returns.pkl` (822 ticker × 6099 영업일)
- `ff5_monthly.csv` (Fama-French 5-factor)
- `mat_eq_eq_raw_pap` 결과 pkl (우리 펀드)
- `universe.csv` (833 ticker + sector)

**참조 대상** (필요 시 final/data/ 직접 접근):
- 다른 155 config pkl (Sensitivity Test 영역에서만)
- prices_raw.pkl / shares_outstanding.pkl

## D-2. 회사명 매핑

**Universe 확인 결과**: `final/data/universe.csv` = 833 ticker, 컬럼 = `ticker, gics_sector` (회사명 없음)

**검토된 옵션**:
- (a) yfinance 1회 수집 → CSV 캐시
- (b) 사전 정의된 파일 사용 (없음)
- (c) 수집 안 함

**결정**: (a) yfinance 1회 수집

**근거**:
1. universe.csv 에 회사명 없음 → 보강 필수
2. yfinance `Ticker.info["longName"]` 활용
3. 1회 수집 후 CSV 캐시 → rate limiting 회피

**구현 안**:
```python
# scripts/build_ticker_company_map.py
import yfinance as yf
import pandas as pd

tickers = pd.read_csv("final/data/universe.csv")["ticker"].tolist()
mapping = {}
for t in tickers:
    try:
        mapping[t] = yf.Ticker(t).info.get("longName", t)
    except Exception:
        mapping[t] = t  # fallback to ticker

pd.DataFrame(
    {"ticker": list(mapping.keys()), "company_name": list(mapping.values())}
).to_csv("streamlit_dashboard/data/ticker_company_map.csv", index=False)
```

**산출물**: `streamlit_dashboard/data/ticker_company_map.csv` (ticker, company_name)

## D-3. 캐싱 전략

**검토된 옵션**:
- (a) `@st.cache_data` 일괄 적용
- (b) `@st.cache_resource` 활용
- (c) 함수별 적절히

**결정**: (c) 함수별 적절히

**근거**:
1. **데이터 로딩** = `@st.cache_data` (DataFrame, dict, JSON-serializable)
2. **모델/연결** = `@st.cache_resource` (singleton, 모든 세션 공유)
3. **외부 API (yfinance)** = `@st.cache_data(ttl=86400 * 30)` (30일 TTL)

**적용 패턴**:
```python
# 데이터
@st.cache_data
def load_monthly_panel():
    return pd.read_csv("streamlit_dashboard/data/monthly_panel.csv")

@st.cache_data
def load_pkl_results(config_name: str):
    return pickle.load(open(f"final/results/{config_name}.pkl", "rb"))

# 모델 / 연결 (필요 시)
@st.cache_resource
def load_lstm_model():
    return torch.load("model.pt")
```

## D-4. 데이터 갱신 주기

**결정**: (a) 정적 (배포 시점 고정)

**근거**:
1. **학술 백테스트** = 시점 고정 데이터 필수
2. **가상 펀드** = 실제 운용 X → 실시간 갱신 불필요
3. 데이터 갱신 시 = 새 배포 (Streamlit Cloud rebuild)

## D-5. 데이터 무결성 검증

**검토된 옵션**:
- (a) Startup check (앱 시작 시 1회)
- (b) 페이지별 가벼운 check
- (c) 명시적 button
- (d) (a) + (b) 결합

**결정**: (a) Startup check

**근거**:
1. **단순 + 명확** — 1회 검증으로 전체 데이터 무결성 보장
2. **`@st.cache_data` 활용 시** (b) 와 리소스 차이 거의 없음
3. **실패 시** = 앱 시작 안 됨 (명확한 에러)

**Startup check 항목**:
```python
def startup_data_check():
    required = [
        "streamlit_dashboard/data/monthly_panel.csv",
        "streamlit_dashboard/data/daily_returns.pkl",
        "streamlit_dashboard/data/ff5_monthly.csv",
        "streamlit_dashboard/data/ticker_company_map.csv",
        # ...
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        st.error(f"필수 데이터 파일 누락: {missing}")
        st.stop()
```

## D 섹션 결과 / 함의

- 데이터 폴더 구조:
  ```
  streamlit_dashboard/data/
  ├── monthly_panel.csv          # 복사
  ├── daily_returns.pkl          # 복사
  ├── ff5_monthly.csv            # 복사
  ├── universe.csv               # 복사
  ├── ticker_company_map.csv     # yfinance 수집
  └── results/
      └── mat_eq_eq_raw_pap.pkl  # 복사 (우리 펀드)
  
  final/data/                    # 참조 (필요 시)
  └── (다른 155 config pkl)
  ```
- 캐싱 표준 정의 (lib/data_loader.py 작성 시 적용)
- Startup check (lib/validators.py 작성 시 적용)

---

# E 섹션 — HO 24m 부진 Disclosure 전략 — 확정

> **결정 시점**: 2026-05-10 / **상태**: 확정 (페이지별 narrative 일관성 점검 완료)

## E 섹션 통합 배경 (Context)

HO 24m (2024-2025) 펀드 부진 narrative — 페이지별로 이미 적용됨. **일관성 점검 + 통합 narrative 결정**.

**이미 적용된 HO narrative** (페이지별):

| 페이지 / 영역 | 적용 narrative |
|---|---|
| Overview 영역 1 (Header) | "● Active (Simulated)" 표시 |
| Overview 영역 2 (Hero KPI) | TEST + HO 별도 표시 (학술 정직성) |
| Overview 영역 6 (Footer) | "HO 24m 부진 — Backtesting 페이지 참조" |
| Performance 영역 7 (Regime Heatmap) | HO 행 강조 (배경색) |
| Risk Metrics 영역 4 (Drawdown) | 2024 위기 시기 mark |
| Sector Watch 영역 8 (HO 정당화) | ★★★ Markowitz 1952 narrative + IT under-weight 시각 |
| Methodology 영역 8 (한계) | 🟧 HO 부진 인정 카드 |
| Backtesting 영역 4 (Regime 자세) | Worst Regime: HO 강조 |
| Backtesting 영역 5 (Sub-events) | 2024 IT Rotation worst event |

→ **9개 영역에 HO narrative 분산 적용됨**

## E-1. 페이지간 HO 표현 통일

**검토된 옵션**:
- (a) 현재 혼용 ("HO 24m" / "HOLD_OUT" / "2024-2025")
- (b) "HOLD_OUT 24m (2024-2025)" 통일
- (c) "검증 24m (2024-2025)" 통일
- (d) 컨텍스트별 다름

**결정**: (b) "HOLD_OUT 24m (2024-2025)" 통일

**근거**:
1. **학술 정확성** — `bl_config.py` 의 EVAL_PERIODS 키 ("HOLD_OUT") 와 일관
2. **명시적** — 기간 (2024-2025) 명확
3. (c) "검증" 은 "validation" 으로 오역 가능 (학습/검증 분리 X 명확화)
4. (d) 컨텍스트별은 일관성 ↓

## E-2. HO 부진 narrative 톤

**검토된 옵션**:
- (a) Sector Watch 영역 8 (자신감) 톤 통일
- (b) 페이지별 다른 톤
- (c) 균형 톤 통일

**결정**: (a) Sector Watch 자신감 통일 — 단, **Methodology 영역 8 한계 카드는 솔직 인정 톤 유지**

**근거**:
1. **펀드 홍보 정체성** = 자신감 톤 우선 (균형 옵션 B 와 부합)
2. Sector Watch 영역 8 결론 박스 = "장기 분산의 가치" (Markowitz 1952 인용)
3. Methodology 영역 8 한계 = "🟧 HO 부진 인정" 카드 = 학술 정직성 (한계 차원이라 솔직 톤 유지)
4. **자신감 + 한계 인정의 balance** = 균형 옵션 (B) 부합

**적용 narrative**:
- Overview / Performance / Risk / Holdings / Sector / Backtesting = 자신감 톤
  - "장기 분산 운용의 가치 / HO 24m 단기 trade-off"
- Methodology 영역 8 한계 카드 = 솔직 인정 톤
  - "HO 24m 부진 = SPY 대비 -12.9%p / Sector trade-off 인정"

## E-3. HO Disclosure 위치 일관성

**검토된 옵션**:
- (a) Footer 표현 통일 (모든 페이지 동일)
- (b) 페이지별 컨텍스트 다른 disclosure
- (c) 추가 Disclosure 박스 (모든 페이지 상단)

**결정**: (a) Footer 표현 통일

**근거**:
1. **모든 페이지 Footer 동일 disclosure** = 일관성
2. **Overview 영역 6 Footer 결정** 그대로 적용
3. (c) 상단 박스는 첫인상 부정적 강조 (마케팅 ↓)
4. (b) 페이지별 다름은 일관성 ↓

**Footer Disclosure 통일 텍스트**:
```
※ 본 결과는 백테스트 시뮬레이션이며 실제 운용 성과를 보장하지 않습니다.
   데이터 기간: 2010-01 ~ 2025-12 (TEST 평가 168m + HOLD_OUT 24m)
   ※ HOLD_OUT 24m (2024-2025) 구간에서 SPY 대비 부진 (Net CAGR +X.XX%
     vs SPY +21.2%) — 자세한 분석은 Backtesting 페이지 참조
```

## E-4. HO 부진 정량 표현 통일

이미 결정된 수치:
- Net CAGR Fund +8.3% / SPY +21.2% / 차이 -12.9%p
- Sortino Fund 0.685 / SPY 2.333

**검토된 옵션**:
- (a) Net CAGR 차이 (-12.9%p) 강조
- (b) Sortino 비교 (0.685 vs 2.333) 강조
- (c) 둘 다 강조
- (d) 페이지별 컨텍스트

**결정**: (c) 둘 다 강조 (Net CAGR + Sortino)

**근거**:
1. **Net CAGR** = 가장 직관 (가상 투자자 친화)
2. **Sortino** = 펀드 정체성 (위험조정수익) 강조 — VolControl narrative 부합
3. **둘 다 표시** = 컨텍스트별 활용 (Footer = Net CAGR / Sector Watch 영역 8 = Sortino)

## E 섹션 결과 / 함의

- **HO 표현 통일** ("HOLD_OUT 24m (2024-2025)") = 모든 페이지 narrative 적용
- **자신감 톤 + 한계 인정 톤 분리** — Sector Watch (자신감) + Methodology 한계 카드 (솔직 인정)
- **Footer Disclosure 통일** = 모든 페이지 동일 텍스트
- **정량 표현** = Net CAGR + Sortino 둘 다 활용 (컨텍스트별)
- 구현 시 = `lib/disclosure.py` 등 standardized text dictionary 작성

---

# F 섹션 — 시뮬레이션 레이어 (Fund Operations) — 확정

> **결정 시점**: 2026-05-10 / **상태**: 확정 + Investment Simulator 페이지 신규 추가

## F 섹션 통합 배경 (Context)

펀드 운용 시뮬레이션 레이어. 이미 부분 확정 (TER 미사용 + One-way TC 20bp, C-2-4) + 가상 AUM / Performance Fee / Net 메트릭 / 인터랙티브 시뮬레이터 결정.

## F-1. 가상 AUM 표시

**결정**: (a) 표시 안 함

**근거**: "가상 펀드" 정체성 정직 + 사용자 슬라이더 산만

## F-2. Performance Fee 표시

**결정**: (b) "0% (no performance fee)" 명시

**근거**: 학술 정직성 — 모든 비용 가정 명시

## F-3. Net 메트릭 표시 위치 일관성

**결정**: (d) Hero = Net / 다른 곳 = Gross + Net 둘 다

**적용**:
- Overview 영역 2 Hero KPI: "Net CAGR (TC 20bp)" (이미 결정)
- Performance 영역 3 Summary KPI: CAGR (gross) — Net 미표시
- Risk Metrics 영역 7 종합 표: Net Sharpe / Net Sortino 포함
- 다른 페이지 = 컨텍스트별

## F-4. 거래비용 (One-way 20bp) 명시 위치

**결정**: (d) 모두 (Hero + Methodology + Footer)

**위치**:
1. Overview 영역 2 Hero KPI Net CAGR 카드 안에 "TC 20bp"
2. Methodology 영역 5/8 에서 자세히 (학술 근거 인용)
3. Footer disclosure 에 명시

## F-5. 펀드 운영 시뮬레이션 추가 narrative

**결정**: (c) 운영 가정 박스 (About 페이지)

**About 페이지 영역 5 (데이터 출처) 보완 — 운영 가정 박스 추가**:
```
펀드 운영 가정:
- 운용보수 (TER): 0%
- Performance Fee: 0%
- 거래비용 (One-way): 20bp
- 매월 rebalancing
- Tax / Slippage 미반영
```

---

## ★ F-6. Investment Simulator 페이지 신규 추가 (사용자 추가 요청)

### 배경

사용자 요청: **"사용자가 실제로 기간과 금액을 입력해서 '내가 이때 얼마를 넣었으면 언제까지 또는 지금까지 얼마를 벌었겠구나' 를 느낄 수 있는 기능"** — 인터랙티브하게 펀드를 체감할 수 있는 기능.

### Claude 구현 비용 분석

옵션 4 (종합 — Lump-sum + DCA + Goal-based + 비교 시나리오 + 풍부 시각화):
- 코드 약 310-480 lines
- Claude 답변 4-6회 (자동 생성 → 사용자 시간 비용 적정)

### 결정

**옵션 4 (종합) 채택** + **별도 페이지 (Investment Simulator)** + **사이드바 "체험" 신규 그룹** (옵션 2)

### Sim 페이지 메타 결정 (Sim M-1 ~ M-4)

| 항목 | 결정 |
|---|---|
| **Sim M-1 영역 개수** | (b) 표준 7 영역 |
| **Sim M-2 Sub-header** | (a) 포함 |
| **Sim M-3 Input 디자인** | (b) 다단 + Lump-sum/DCA/Goal 토글 |
| **Sim M-4 비교 시나리오** | (a) 사이드바 토글 (SPY/EW/IVW) 활용 |

### Investment Simulator 페이지 7 영역 구조

```
1. Header                       (Overview 동일)
2. Sub-header                   ("내 투자 시뮬레이션 — Investment Simulator")
3. Input 영역                   (시작/종료/초기금액/DCA/Goal 토글)
4. Result KPI 카드 4-5개         (최종자산/수익/연환산/MDD/총투자)
5. 누적 자산 곡선               (Fund vs SPY vs EW vs IVW — 사이드바 토글 반응)
6. Insight 박스                 (자세한 분석 + 시나리오 비교)
7. Footer                       (Overview 동일)
```

### Sim 페이지 영역별 결정 — 확정 (사용자 추가 진행)

#### 영역 2: Sub-header

**Sim2-1**: (b) 친근 톤

**텍스트**:
```
"내가 이때 얼마를 투자했더라면?"
실제 수익을 시뮬레이션해 보세요.
```

**근거**: 가상 투자자 친화 (마케팅 핵심 기능 — 친근 톤이 인터랙티브 narrative 강화)

#### 영역 3: Input 영역

**Sim3-1 시나리오 토글**: (a) Tab 전환 (Lump-sum / DCA / Goal)
- 명확 시나리오 분리

**Sim3-2 입력 필드**: (b) 표준
- 시작 시점 / 종료 시점 / 초기 금액 / DCA 매월 금액 / Goal 금액

**Sim3-3 시점 입력**: (a) Date input 2개
- Streamlit `st.date_input` × 2

**Sim3-4 금액 입력**: (c) Number + Slider 조합
- Number input ($100 ~ $1M) + Slider 동시 (사용자 자유 선택)

#### 영역 4: Result KPI 카드 5개

**Sim4-1 KPI 메트릭**: (b) 5개
1. 최종 자산 (Final Value)
2. 총 수익 (Total Profit)
3. 연환산 수익률 (CAGR)
4. 최대낙폭 (Max DD)
5. 총 투자금액 (Total Invested)

**Sim4-2 디자인**: (a) Performance 영역 3 단순 패턴 (큰 숫자 + tooltip)

#### 영역 5: 누적 자산 곡선

**Sim5-1 차트 구성**: (b) Fund + 사이드바 토글 활성 벤치마크
- 인터랙션 일관성 원칙 (Sim M-4 + G-2 결정)

**Sim5-2 추가 표시**: (d) 모두
- DCA 누적 투자금액 라인 (DCA Tab 활성 시 — "내가 넣은 돈 vs 자산" 시각)
- Regime 배경색 (다른 페이지 일관)
- 위기 시기 annotation (COVID / 2022 / 2024 마커)

**Sim5-3 인터랙션**: 모두 채택
- Hover tooltip ✓
- Plotly zoom + range slider ✓
- Y축 토글 (선형 / 로그) ✓
- 위기 annotation ✓

#### 영역 6: Insight 박스

**Sim6-1 Insight 내용**: (b) 표준 (수익 + 비교 narrative)

**Sim6-2 형식**: (b) 카드 그리드

**Sim6-Q 다국어**: (a) 한/영 병기 (A-3 일관)

#### Insight 박스 구현 패턴 (정적 템플릿 + 카드 그리드)

**구현 파일**: `lib/insight_generator.py`

**조건부 카드** (4-8개):

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

**구현 특징**:
- LLM 미사용 (정적 템플릿 + 동적 값 채움)
- 색상 코딩 (양수 = green / 음수 = red / 중립 = blue)
- 한/영 병기 (A-3 일관)

**구현 비용**: 2-3시간 (코드 50-100 lines)

### Investment Simulator 페이지 — 전체 확정 (영역 1~7)

#### 페이지 시각화 통합

```
┌────────────────────────────────────────────────────────────────┐
│ [영역 1: Header — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 2: Sub-header 카드/배너 — 친근 톤]                        │
│ "내가 이때 얼마를 투자했더라면?"                                │
│ 실제 수익을 시뮬레이션해 보세요.                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 3: Input 영역 — Tab + 다단]                               │
│ [Lump-sum | DCA | Goal-based]                                  │
│                                                                │
│ 시작: [2015-01-01]  종료: [2025-12-31]                         │
│ 초기 금액: [$10,000] [Slider]                                  │
│ (DCA Tab) 매월 추가: [$500] [Slider]                           │
│ (Goal Tab) 목표 금액: [$1,000,000]                             │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 4: Result KPI 5개 — 단순 패턴]                            │
│ ┌────┬────┬────┬────┬────┐                                    │
│ │최종│총수│연환│MDD │총투│                                    │
│ │자산│익  │CAGR│    │자  │                                    │
│ └────┴────┴────┴────┴────┘                                    │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 5: 누적 자산 곡선 — 사이드바 토글 반응]                   │
│ ┌─R1─┬─R2─┬─R3─┬─HO─┐ (Regime 배경색)                         │
│ │ ── Fund (cumulative value)                                   │
│ │ ── SPY (사이드바 토글)                                       │
│ │ ── EW / IVW (사이드바 토글)                                  │
│ │ ┄┄ DCA 투자금액 (DCA Tab)                                   │
│ │ ▼ COVID  ▼ 2022 Bear  ▼ 2024 IT                              │
│ └────────────────────────────────────────────────────────────┘ │
│ [Linear ▼] [기간 슬라이더]                                     │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 6: Insight 박스 — 카드 그리드 (4-8개 조건부)]              │
│ ┌──────┬──────┬──────┐                                        │
│ │💰 누적│📈 CAGR│📊 SPY │                                       │
│ ├──────┼──────┼──────┤                                        │
│ │⚠️ 손실│🔄 DCA │🎯 Goal│ (조건부 표시)                          │
│ └──────┴──────┴──────┘                                        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 7: Footer — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘
```

#### Investment Simulator 페이지 결과 / 함의

- **7 영역 모두 확정** — 구현 시 명확한 와이어프레임
- **Insight 박스 구현** = `lib/insight_generator.py` 작성 (정적 템플릿 + 카드 그리드)
- **인터랙션 일관성** = 사이드바 토글 (G-2 결정) + Q-Zoom (G-1 결정)
- **데이터 의존**:
  - Fund 월별 수익률 (D 섹션 캐시 활용)
  - SPY / EW / IVW 월별 수익률 (D 섹션 캐시 활용)
  - 위기 시기 dictionary (영역 5 annotation)
- **DCA / Goal-based 산출 로직**:
  - DCA: 매월 추가 매수 → 누적 자산 (월별 시뮬레이션)
  - Goal-based: 역산 (binary search 또는 closed-form)

### 사이드바 그룹 변경 (5 → 6 그룹)

C-4 결정 갱신: 5 그룹 → 6 그룹 (체험 신규 그룹 추가, Investment Simulator 단일 페이지)
→ `10_sidebar.md` 갱신됨

### 페이지 갯수 변경 (8 → 9 페이지)

| 페이지 | 위치 |
|---|---|
| 1. Overview | 첫인상 |
| 2. **Investment Simulator** ★ 신규 | 체험 |
| 3. Performance | 성과 |
| 4. Risk Metrics | 성과 |
| 5. Holdings | 보유 |
| 6. Sector Watch | 보유 |
| 7. Methodology | 검증 |
| 8. Backtesting | 검증 |
| 9. About / FAQ | 메타 |

## F 섹션 결과 / 함의

- **Investment Simulator 페이지 신규 추가** = 마케팅 친화 핵심 기능
- **사이드바 6 그룹** (10_sidebar.md 갱신)
- **9 페이지** = 전체 구조 (00_README.md 인덱스 갱신 필요)
- 영역별 자세한 결정 = 구현 단계 또는 별도
- Sim 페이지 데이터 = 우리 펀드 월별 수익률 + SPY + EW + IVW (D 섹션 캐시 활용)

---

# G 섹션 — 인터랙션 / UX — 확정

> **결정 시점**: 2026-05-10 / **상태**: 확정

## G 섹션 통합 배경

이미 부분 확정 (인터랙션 일관성 원칙, 00_README.md 부록):
- Q-Zoom (1+2+3 조합): Plotly 기본 + Range slider + 클릭 expand
- 다중 토글: 사이드바 (기간 + 비교) → 모든 페이지
- Tab 전환: 영역 7 / 영역 8 등

추가 결정: 인터랙션 디테일

## G-1. Q-Zoom (클릭 → expand) 구현 디테일

**결정**: (b) 같은 페이지 expand 영역

**근거**:
1. 자유 탐색 부합 — 페이지 흐름 유지
2. (a) Modal popup = 흐름 차단
3. (c) 별도 페이지 navigation = 컨텍스트 손실

**구현**: `st.expander` 또는 `st.dataframe(on_select='rerun')` + 차트 동적 추가

## G-2. Sim 페이지 토글 영향

**결정**: (b) Investment Simulator 도 사이드바 토글 영향

**근거**:
1. Sim M-4 결정 부합 (사이드바 토글 활용)
2. 인터랙션 일관성 원칙 (모든 페이지 동일 토글)

## G-3. 페이지간 navigation 디테일

**결정**: (a) 현재 결정 그대로

**근거**:
1. 사이드바 page_link (C-4) + Overview 영역 4 강점 카드 → 페이지 이동 + Regime 행 클릭 → Backtesting 이동 = 충분
2. (b) contextual link 추가는 디자인 복잡도 ↑
3. (c) Breadcrumb은 사이드바 highlighted 와 중복

## G-4. 모바일 반응형

**결정**: (a) Streamlit 기본 반응형 활용

**근거**:
1. `st.columns` 자동 반응 (모바일에서 세로 배치)
2. 발표 환경 = 데스크톱 우선
3. 추가 CSS 비용 대비 효과 ↓

## G-5. 키보드 / 접근성

**결정**: (a) Streamlit 기본

**근거**:
1. Streamlit 표준 keyboard navigation
2. (b) 단축키 / (c) ARIA = 추가 비용 ↑ 대비 효과 ↓

## G 섹션 결과 / 함의

- 같은 페이지 expand = 모든 차트의 클릭 인터랙션 표준 패턴
- Sim 페이지 = 사이드바 토글 영향 (Investment Simulator 도 일관성)
- Streamlit 기본 활용으로 추가 라이브러리 최소화 (J 섹션 영향)

---

# H 섹션 — 디자인 / UX 디테일 — 확정 (실행 안정성 검토 완료)

> **결정 시점**: 2026-05-10 / **상태**: 확정 + 안정성 보강 권고 포함

## H 섹션 통합 배경

이미 부분 확정 (Cobalt Blue 팔레트, B-4):
- Primary: `#3B82F6` / Green: `#10B981` / Red: `#EF4444` / Background: `#0E1117`

추가 결정: 폰트 / 레이아웃 표준 / CSS / 차트 색상 / 다크 테마 디테일.

**★ 사용자 강조**: 실행 안정성 최우선 → 각 결정 사항 안정성 검토 완료.

## H-1. 폰트

**결정**: (b) Pretendard (한국어 친화)

**안정성 평가**: ★★ 중간 (CDN 의존)

**Fallback chain 적용 (안정성 보강)**:
```css
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", -apple-system, sans-serif;
```

**근거**:
1. **Pretendard** = 한국어 가독성 ★★★ (무료 웹 폰트)
2. **Fallback chain** = CDN 차단 시 자동 fallback → 기능 정상
3. (a) Streamlit 기본은 한글 깨짐 가능
4. (c) Noto Sans KR + Inter 도 좋지만 Pretendard 더 한국어 친화

## H-2. 레이아웃 표준

**결정**: (a) Streamlit 기본 (자동 비율)

**안정성 평가**: ★★★ 안전

**근거**:
1. **`st.columns` 자동 반응** = 표준 + 안정
2. (b) 황금비율은 추가 CSS 비용
3. (c) 페이지별 자유는 일관성 ↓

## H-3. CSS 커스터마이징

**결정**: (b) 표준 CSS

**안정성 평가**: ★★ 중간 (CSS 충돌 가능)

**최소화 원칙 (안정성 보강)**:
1. **카드 / Insight 박스 / accent 색상만** custom CSS
2. **Streamlit 자체 컴포넌트 우선** (`st.metric`, `st.expander`, `st.dataframe`)
3. **inline style 우선** (`st.markdown` 내 `<div style="...">`)
4. **복잡한 layout = `st.columns`** (CSS 직접 X)

**근거**:
1. (a) Streamlit 기본만은 카드 / Insight 박스 디자인 한계
2. (c) 풍부 CSS = 충돌 위험 ↑↑
3. **표준 CSS = 최소 필요한 부분만** → 안정성 + 디자인 균형

## H-4. 차트 색상 팔레트

**결정**: (c) Custom (Cobalt Blue accent + Plotly 보조)

**안정성 평가**: ★★★ 안전

**팔레트 정의**:
- **Primary 라인** (Fund): `#3B82F6` (Cobalt Blue)
- **벤치마크 라인**:
  - SPY: `#6B7280` (Gray)
  - EW: `#10B981` (Green)
  - IVW: `#8B5CF6` (Purple)
- **수익 색상**:
  - Positive: `#10B981` (Green)
  - Negative: `#EF4444` (Red)
- **Regime 배경색** (Overview 영역 3 일관):
  - R1 (회복기): `#1F2937` (Dark Gray)
  - R2 (확장기): `#0E1117` (Background, 기본)
  - R3 (변동기): `#1F2937` (Dark Gray)
  - HO (홀드아웃): `#374151` (Slightly Lighter Gray)

**GICS 11개 섹터 색상** (Holdings Treemap / Sector Watch Treemap):
- IT (Information Technology): `#3B82F6` (Blue)
- Healthcare: `#10B981` (Green)
- Financials: `#F59E0B` (Amber)
- Consumer Discretionary: `#EC4899` (Pink)
- Consumer Staples: `#8B5CF6` (Purple)
- Industrials: `#06B6D4` (Cyan)
- Energy: `#EF4444` (Red)
- Materials: `#84CC16` (Lime)
- Communication Services: `#F97316` (Orange)
- Real Estate: `#A855F7` (Violet)
- Utilities: `#14B8A6` (Teal)

→ `lib/colors.py` 에 dictionary 정의 (안정성 ★★★)

## H-5. 다크 테마

**결정**: (b) 커스텀 다크 (Cobalt Blue 강조)

**안정성 평가**: ★★★ 안전

**`.streamlit/config.toml` 설정**:
```toml
[theme]
base = "dark"
primaryColor = "#3B82F6"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1F2937"
textColor = "#FAFAFA"
font = "sans serif"
```

**근거**:
1. **B-4 결정 (Cobalt Blue) 일관**
2. Streamlit `config.toml` = 표준 안정
3. (a) Streamlit 기본은 Cobalt Blue accent 미적용
4. (c) 라이트 / 다크 토글은 C-4 결정 (다크 고정) 위배

## 추가 안정성 보강 권고 (J 섹션 영향)

**Streamlit 버전 핀 권고**:
- `streamlit>=1.30,<2.0` (1.30+ 의 신기능 활용 + 2.0 메이저 변경 회피)

**외부 라이브러리 최소화**:
- **핵심**: streamlit / plotly / pandas / numpy
- **추가**: yfinance / streamlit-card (선택)
- **회피**: streamlit-elements / streamlit-aggrid 등 무거운 라이브러리

**Fallback / Graceful degradation**:
- 폰트 로딩 실패 → fallback chain
- 외부 데이터 (yfinance) 실패 → ticker 만 표시
- Plotly 차트 에러 → `st.error` + 메시지

## H 섹션 결과 / 함의

- **모든 결정 안정성 ★★ 이상** — 실행 안정성 최우선 부합
- **CSS 최소화** = 카드 / Insight 박스 / accent 색상만
- **색상 팔레트** = `lib/colors.py` dictionary 정의
- **Pretendard fallback chain** = CDN 차단 환경에서도 기능 정상
- **다크 테마** = `.streamlit/config.toml` 설정
- J 섹션 (기술 스택) = 외부 라이브러리 최소화 + 버전 핀

---

# I 섹션 — 컴플라이언스 / Disclosure — 확정

> **결정 시점**: 2026-05-10 / **상태**: 확정

## I 섹션 통합 배경

이미 부분 확정 (E 섹션 — HO Disclosure):
- HO 표현 통일 ("HOLD_OUT 24m (2024-2025)")
- Footer Disclosure 통일 텍스트
- About 페이지 영역 7 (자세한 Disclosure) — 메타만 결정

**참조 표준**:
- FINRA Rule 2210 (미국 마케팅 규제)
- 한국 금융감독원 (펀드 관련 표시 의무)
- 학술 가상 펀드 표준

## I-1. Disclosure 표준 텍스트

**결정**: (c) FINRA + 한국 금감원 + 학술 가상 펀드 (둘 다)

**근거**:
1. **국제 표준 부합** — FINRA Rule 2210
2. **한국 청중 친화** — 금융감독원 표준 표현
3. **학술 정직성** — 가상 펀드 / 백테스트 명시

**표준 표현**:
```
※ Past performance is not indicative of future results.
※ 과거의 운용성과는 미래의 수익을 보장하지 않습니다.
※ 본 결과는 백테스트 시뮬레이션이며 실제 운용 성과를 보장하지 않습니다.
※ 본 자료는 투자권유를 목적으로 작성되지 않았습니다.
```

## I-2. Footer Disclosure 길이

**결정**: (a) 짧음 (3-4줄, E-3 결정 그대로)

**근거**:
1. Footer = 모든 페이지에 표시 → 공간 부담 ↓
2. 자세한 Disclosure 는 About 영역 7 에서

**Footer 텍스트** (E-3 결정 + I-1 표준 통합):
```
※ 본 결과는 백테스트 시뮬레이션이며 실제 운용 성과를 보장하지 않습니다.
   데이터 기간: 2010-01 ~ 2025-12 (TEST 평가 168m + HOLD_OUT 24m)
※ HOLD_OUT 24m (2024-2025) 구간에서 SPY 대비 부진 (Net CAGR +X.XX%
   vs SPY +21.2%) — 자세한 분석은 Backtesting 페이지 참조
   자세한 Disclosure / Risk factors: About 페이지 참조
```

## I-3. About 페이지 자세한 Disclosure (영역 7)

**결정**: (b) 표준 (+ Risk factors)

**About 영역 7 콘텐츠**:
1. 학술 가상 펀드 표준 disclaimer
2. FINRA Rule 2210 부합 표현
3. 한국 금감원 표준 표현
4. **Risk factors** (추가):
   - Market risk (시장 변동성)
   - Sector concentration risk (sector 분산 한계)
   - Model risk (LSTM / BL 모델 한계)
   - Data risk (yfinance 데이터 정확성)
   - Backtest overfitting risk (Methodology 영역 8 / About Selection Bias 부록 참조)

## I-4. Risk factors 위치

**결정**: (b) About 영역 7 + Footer 짧은 link

**구현**:
- About 영역 7: 자세한 Risk factors (5가지 차원)
- Footer: "Risk factors: About 페이지 참조" 짧은 link

**근거**:
1. **자세한 Risk factors** = About 영역 7 (학술 정직성)
2. **모든 페이지 Footer 짧은 link** = 일관성 + 접근성
3. (c) Methodology 한계 와 중복 회피 (Methodology 영역 8 = 모델 한계 / About 영역 7 = 종합 Risk factors)

## I-5. Investment Simulator 추가 disclaimer

**결정**: (a) Sim 페이지 상단 disclaimer 박스

**시뮬레이터 disclaimer 텍스트**:
```
⚠️ 본 시뮬레이션은 가상의 백테스트 결과이며, 실제 투자권유 또는
   투자 자문을 목적으로 하지 않습니다. 과거의 성과는 미래의 수익을
   보장하지 않습니다.
```

**근거**:
1. **시뮬레이터 = 사용자 입력 → 가상 결과** → 투자권유 오해 방지 필수
2. **상단 박스** = 사용자가 입력 전 인지
3. (b) footnote 만은 누락 가능
4. **법적 보호 + 학술 정직성**

**위치**: Sim 영역 2 (Sub-header) 하단 + 영역 3 (Input) 상단 사이

## I 섹션 결과 / 함의

- **모든 페이지 Footer** = E-3 + I-2 통일 텍스트
- **About 영역 7** = 자세한 Disclosure + Risk factors (5가지)
- **Investment Simulator** = 추가 상단 disclaimer (투자권유 X 명시)
- **규제 부합** = FINRA Rule 2210 + 한국 금감원 + 학술
- 구현 시 = `lib/disclosure.py` 표준 텍스트 dictionary 작성

---

# J 섹션 — 기술 스택 + 배포 — 확정

> **결정 시점**: 2026-05-10 / **상태**: 확정

## J 섹션 통합 배경

이미 부분 결정 (페이지별 + H 섹션 권고):
- Streamlit + Plotly (메인)
- streamlit-card (Holdings 영역 4)
- Pretendard 폰트 (H-1)
- 외부 라이브러리 최소화 (H 권고)

## J-1. 라이브러리 최종 list

**결정**: (b) 핵심 + streamlit-card

**라이브러리 list**:
| 카테고리 | 라이브러리 | 용도 |
|---|---|---|
| 메인 | streamlit | 프레임워크 |
| 차트 | plotly | 모든 차트 (Sankey 포함) |
| 데이터 | pandas, numpy | DataFrame |
| 데이터 수집 | yfinance | 회사명 매핑 (1회) |
| 통계 | scipy | Jarque-Bera, Hill estimator |
| 회귀 | statsmodels | CAPM / FF5 회귀 |
| 카드 UI | streamlit-card | Overview 영역 4 강점 카드 |

**근거**:
1. **streamlit-card** = Holdings 영역 4 (M4-4 결정) 부합 + 시각 우수
2. (a) 핵심만은 카드 클릭 인터랙션 한계
3. (c/d) streamlit-extras 는 의존성 ↑ → 안정성 ↓

## J-2. Streamlit 버전

**결정**: (a) `>=1.30,<2.0`

**근거**:
1. **1.30+ 신기능**: `st.dataframe(on_select='rerun')`, `st.page_link` 등
2. **<2.0**: 메이저 버전 변경 회피 (안정성)
3. (b) Latest = 갑작스런 변경 위험 / (c) 고정 = 보안 패치 누락

## J-3. 배포 환경

**결정**: (a) Streamlit Cloud (Community 무료)

**근거**:
1. **표준 배포** — GitHub 연동 자동 배포
2. **무료** — 학술 / 가상 펀드 적합
3. **Streamlit 표준 환경** — 호환성 ★★★
4. (b) HF Spaces 도 좋지만 Streamlit Cloud 가 더 표준
5. (c) 자체 서버 = 비용 ↑↑
6. (d) 로컬 only = A-2 (자유 탐색 + URL 공유) 결정 위배

**제약사항**:
- 메모리 제한 1GB (Community)
- 슬립 모드 (1주 미사용 시)
- → 메모리 효율 (D-3 캐싱) + 발표 전 warmup 필요

## J-4. URL

**결정**: (b) Custom subdomain

**URL 안**:
- `https://volcontrol.streamlit.app/`
- 또는 `https://adaptive-volcontrol.streamlit.app/`

**근거**:
1. **펀드 정체성 (VolControl) 노출** — 마케팅 효과
2. (a) 기본 URL = `[username]-[repo]-app.streamlit.app` 길고 어색
3. (c) Custom domain = 비용 ↑ + 학술 프로젝트 과함

## J-5. requirements.txt 작성 방향

**결정**: (b) Range versions

**`requirements.txt` 안**:
```
streamlit>=1.30,<2.0
plotly>=5.18,<6.0
pandas>=2.0,<3.0
numpy>=1.24,<2.0
yfinance>=0.2,<1.0
scipy>=1.11,<2.0
statsmodels>=0.14,<1.0
streamlit-card>=1.0,<2.0
```

**근거**:
1. **Range** = 안정성 + 유연성 (보안 패치 자동)
2. **Major version 핀** = 메이저 변경 회피
3. (a) Pinned (특정 버전) = 보안 패치 누락
4. (c) Latest = 갑작스런 변경 위험

## J 섹션 결과 / 함의

- **8개 라이브러리** = `requirements.txt` 작성 시 적용
- **Streamlit Cloud 배포** = GitHub 연동 (자동 배포)
- **URL** = `https://volcontrol.streamlit.app/` (또는 유사)
- **메모리 효율 + 캐싱** = D-3 결정과 연결 (1GB 제한 대응)
- **발표 전 warmup** = 슬립 모드 회피 (발표 30분 전 1회 접속)

---

# K 섹션 — 차별화 포인트 (Storytelling) — 확정

> **결정 시점**: 2026-05-10 / **상태**: 확정 + 발표 컨텍스트 명확화

## K 섹션 통합 배경

이미 부분 확정 (페이지별 narrative):
- Overview 영역 4 강점 카드 = LSTM + 3-Regime + Net Costs (A + C + E)
- Sector Watch 영역 8 = HO 정당화 narrative (Markowitz)
- Methodology 영역 8 = 한계 인정
- Investment Simulator = 친근 톤 (체험)

**★ K-4 사용자 결정 — 발표 컨텍스트 명확화 (반드시 명시)**:

> **가상 투자자 대상 홍보물 컨셉으로 프로젝트 발표 진행.
> 대시보드 시연 시간: 5분 제한.
> 학술적 분석 내용: PPT 통해 10~15분 분량 사용.
> 대시보드에 발표 스크립트는 포함되지 않음.**

→ 모든 K 섹션 결정 + 다른 D~L 섹션은 위 컨텍스트 반영하여 검토.

## K-1. 발표 시나리오

**결정**: (d) 모두 — 5분 demo / 15분 자유 탐색 안내 (실제는 5분 demo 가 메인)

**근거 (K-4 컨텍스트)**:
1. **대시보드 시연 = 5분 (메인)** → demo 시나리오 핵심
2. **15분 deep-dive = PPT 가 담당** (학술 분석)
3. **자유 탐색 = 발표 후 URL 공유** (가상 투자자 추가 탐색)
4. 따라서 대시보드는 **5분 시연용 + 자유 탐색용** 양립

## K-2. Storytelling 흐름 (5분 demo 기준)

**결정**: (a) Overview → Sim → Sector → Methodology

**5분 흐름 안**:
1. **Overview** (1분): Hero KPI + 누적수익 곡선 + 핵심 강점 3카드
2. **Investment Simulator** (1.5분): 사용자 입력 → 즉각 결과 ("$10,000 투자 시 X")
3. **Sector Watch (영역 8 HO 정당화)** (1.5분): "왜 HO 부진? 분산의 가치"
4. **Methodology** (1분): "왜 이 펀드인가" (Sankey + 4-slot)

**근거**:
1. **Sim 우선** = 가상 투자자 친화 (체험 즉각)
2. **Sector HO 정당화** = 학술 정직성 (단점도 솔직)
3. **Methodology 마지막** = 신뢰성 마무리
4. (b) 표준 (Performance → Risk → Methodology) 은 마케팅 친화 ↓
5. (c) Sim 우선 + Sector 우선 = 균형

## K-3. 핵심 메시지

**결정**: (c) 3가지 강점 + HO 정당화 추가 (4가지)

**4가지 핵심 메시지**:
1. 🟦 **Volatility-Aware Allocation** (LSTM 변동성 예측 + BL 적응형)
2. 🟪 **Validated Across Market Regimes** (3-regime + OOS)
3. 🟧 **Net of Conservative Costs** (20bp 거래비용 차감)
4. 🟩 **Sector 분산의 가치 (HO 정당화)** — Markowitz 1952 인용

**근거**:
1. **Overview 영역 4** 3가지 카드 + Sector Watch 영역 8 결론 박스 일관
2. (a) 3가지만은 HO 정당화 narrative 누락 → 5분 demo 흐름 (Sector 강조) 와 부합 ↓
3. (b) Sim 추가는 Sim 자체가 narrative 보강 (5분 demo 에서 1.5분 차지)

## K-4. 청중별 tone 조정

**결정**: 가상 투자자 홍보물 컨셉 (★ 사용자 결정)

**적용 원칙**:
1. **대시보드 = 마케팅 친화 톤 일관** (균형 옵션 B 유지)
2. **About 페이지 = 발표 스크립트 X** (사용자 결정)
3. **About 페이지 영역 6 (Selection Bias 부록)** = Q-B3 결정 그대로 (학술 부록만)
4. **PPT 분리** = 학술 깊이는 PPT 가 담당 → 대시보드 학술 깊이 추가 강조 불필요

**제거된 옵션**:
- (b) "발표자 모드" 토글 → 불필요 (대시보드 자체에 스크립트 X)
- (c) "About 페이지 발표 가이드" → **단순 메타 정보로 단순화** (스크립트 X)

## K 섹션 결과 / 함의

- **5분 demo 흐름** = Overview → Sim → Sector → Methodology
- **4가지 핵심 메시지** = LSTM + Regime + Net + Sector 분산
- **PPT 분리** = 대시보드 = 마케팅 / PPT = 학술
- **About 페이지** = 단순 메타 정보 (발표 스크립트 X)
- 균형 옵션 (B) 유지 — 학술 깊이는 PPT 위임
- 다른 D~L 섹션 결정은 위 컨텍스트 (가상 투자자 홍보물) 반영하여 일관 적용

---

# L 섹션 — 한계 명시 (학술적 진실성) — 확정

> **결정 시점**: 2026-05-10 / **상태**: 확정 (일관성 점검 완료)

## L 섹션 통합 배경

이미 부분 확정 (Methodology 영역 8 — Q-A1):
- 5가지 → 3가지 차원 축소
- HO 부진 / 향후 개선 / 실무 제약

## 한계 narrative 분포 점검

| 위치 | 한계 내용 | 톤 |
|---|---|---|
| Methodology 영역 8 카드 1 (HO 부진) | -12.9%p / Sector trade-off | 솔직 인정 (E-2) |
| Methodology 영역 8 카드 2 (개선 방향) | Multi-factor / WFV | 향후 개선 |
| Methodology 영역 8 카드 3 (실무 제약) | 가상 펀드 / 운용 규모 | 표준 disclosure |
| About 영역 6 (Selection Bias 부록) | PBO/DSR / Data snooping | 학술 부록 (Expander) |
| About 영역 7 (자세한 Disclosure) | Risk factors 5가지 | 자세한 |
| Sector Watch 영역 8 (HO 정당화) | 분산 trade-off | 자신감 |
| Footer (모든 페이지) | HO 부진 짧은 언급 | 표준 |

## L-1. 한계 narrative 통합 일관성

**결정**: (a) 현재 결정 그대로

**근거**:
1. 페이지별 narrative 이미 충분 일관 (7개 위치)
2. (b) 추가 통합 영역은 페이지 분량 ↑↑ + 균형 옵션 (B) 위배
3. (c) 단순화는 학술 정직성 ↓

## L-2. 학술 인용 일관성

**결정**: (b) 페이지별 학술 인용 → 00_README.md 학술 근거 일람 일괄 갱신

**근거**:
1. **일관성 강화** — 모든 학술 인용 한 곳에서 확인 가능
2. About 영역 5 (데이터 출처 + 학술 인용 일람) = 00_README 와 동기화

**갱신할 학술 인용** (페이지별 → 00_README 일람):
- Markowitz (1952) — Sector Watch 영역 8
- Black-Litterman (1990, 1992) — Methodology 영역 4
- He-Litterman (1999) — Methodology 영역 4
- Idzorek (2005) — Methodology 영역 4
- Hochreiter & Schmidhuber (1997) — Methodology 영역 5 LSTM
- Gers, Schmidhuber, Cummins (2000) — Methodology 영역 5
- Kim & Won (2018) — Methodology 영역 5
- Jensen (1968) — Methodology 영역 6
- Fama-French (1993, 2015) — Methodology 영역 6
- Carhart (1997) — Methodology 영역 6
- Jarque-Bera (1980) — Methodology 영역 7
- Cont (2001) — Methodology 영역 7 fat tail
- Hill (1975) — Risk Metrics 영역 8
- Embrechts, Klüppelberg, Mikosch (1997) — Methodology 영역 7
- Lopez de Prado (2018) — Methodology 영역 5/8 walk-forward
- Engle (2002) — Methodology 영역 8 DCC-GARCH (Expander)
- AQR Frazzini, Israel, Moskowitz (2018) — Overview 영역 4 + Methodology 영역 8
- Bailey & Lopez de Prado (2014) — About 영역 6 PBO/DSR
- Frazzini-Pedersen (2014) — Overview 영역 3 IVW

## L-3. "학술 정직성 선언" 박스 위치

**결정**: (a) Methodology 영역 8 만

**근거**:
1. **Methodology 영역 8 = 한계 영역의 결론** — 학술 정직성 선언 자연 위치
2. **About 영역 7 = 다른 톤** (자세한 Disclosure / Risk factors 형식)
3. **Sector Watch 영역 8 = 자신감 결론 박스** (다른 톤 유지)
4. (b/c) 통일은 페이지별 narrative tone 일관성 위배 (E-2 결정)

## L 섹션 결과 / 함의

- **현재 7개 위치 한계 narrative 일관 유지**
- **00_README.md 학술 근거 일람 일괄 갱신** (L-2 결정 적용 후)
- **학술 정직성 선언 = Methodology 영역 8 만**
- D~L 섹션 전체 확정 완료 → plan.md 작성 단계로 이동

---

# ★ D~L 섹션 전체 완료 요약

| 섹션 | 주제 | 상태 |
|---|---|---|
| **D** | 데이터 준비 (핵심 복사 + yfinance 매핑 + 함수별 캐싱 + 정적 + Startup check) | ✅ 확정 |
| **E** | HO Disclosure (HOLD_OUT 24m 통일 + 자신감 톤 + Footer 통일 + Net CAGR + Sortino) | ✅ 확정 |
| **F** | 시뮬레이션 (TER 미사용 + One-way TC 20bp + Net 메트릭 위치 + Investment Simulator 페이지 신규) | ✅ 확정 |
| **G** | 인터랙션 / UX (같은 페이지 expand + Sim 토글 + Streamlit 기본 반응) | ✅ 확정 |
| **H** | 디자인 / UX 디테일 (Pretendard + Streamlit 기본 + 표준 CSS + Custom 색상 + 커스텀 다크) | ✅ 확정 |
| **I** | 컴플라이언스 / Disclosure (FINRA + 한국 금감원 + 학술 + Footer 짧음 + About 영역 7 + Sim 상단 disclaimer) | ✅ 확정 |
| **J** | 기술 스택 + 배포 (8개 라이브러리 + Streamlit Cloud + Custom subdomain + Range versions) | ✅ 확정 |
| **K** | Storytelling (5분 demo + Sim 우선 흐름 + 4가지 메시지 + 가상 투자자 홍보물 + PPT 분리) | ✅ 확정 |
| **L** | 한계 (현재 그대로 + 학술 인용 일괄 + Methodology 영역 8 만 정직성 선언) | ✅ 확정 |

→ **D~L 9 섹션 모두 확정 완료**
→ **다음 단계**: plan.md + 페이지 와이어프레임 작성

---

[← 10_sidebar.md](10_sidebar.md) | [00_README.md](00_README.md) (인덱스) →

> 본 파일은 D~L 섹션 결정을 담고 있습니다. D 확정 / E 확정 / F~L 진행 예정.
