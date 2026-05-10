# C-1-3. Risk Metrics 페이지

> **파일**: `04_risk_metrics.md`
> **결정 시점**: 2026-05-10
> **상태**: 확정 (페이지 메타 Risk M-1~M-4 + 영역 1~9, Hill 영역은 옵션 C 축소형 → Q-F2 추가 축소)
> **포함**: 페이지 메타 결정 / Sub-header / Risk Summary KPI 5개 / Drawdown + Recovery / VaR / CVaR 분포 / Beta + R² + Tracking Error 시계열 / Risk Metrics 종합 표 / Tail Risk 분석 (Hill Estimator)

---

## C-1-3. Risk Metrics 페이지

> **상태**: 진행 중 (메타 결정 확정 / 영역 1, 9 는 Overview 동일 / 2-8 미정)

### Risk Metrics 페이지 통합 배경 (Context)

위험 지표 전담 페이지. Performance 페이지가 수익 위주라면 Risk Metrics 는 **위험 위주**.

메트릭 풀 (C-2) 의 **Pool-3 (위험)** 중심:
- Volatility / MDD / Downside Deviation / Historical VaR / CVaR / Beta / R² / Tracking Error
- 추가: **Hill estimator** (S8-3 결정에서 이동)

**핵심 차별점**:
- vs Performance: 수익률 → 위험률
- 펀드 정체성 (Adaptive VolControl) 의 **위험 통제 narrative** 강화

### 추가 결정 (Risk-Q1, 2026-05-10): final 정합성 + 학술 표준 메트릭 매핑

**배경**: 사용자 지시 — "본 페이지의 각 지표들도 final 과 비교하여 정확하게 동일한
결과가 나올 수 있도록 구성하고 검증까지 진행."

**Final 에 직접 정의된 메트릭** (이미 검증 — Phase 1.5 사전 작업):

| Risk Metrics 메트릭 | Final ground truth | 우리 lib | 검증 |
|---|---|---|---|
| Volatility | `compute_metrics.vol` | `calc_volatility` | ✅ 1e-3 일치 |
| MDD | `compute_metrics.mdd` | `calc_mdd` | ✅ |
| Beta | `compute_metrics.beta` (excess 기준) | `calc_beta` | ✅ |
| Sortino | `compute_metrics.sortino` | `calc_sortino` | ✅ |
| Sharpe | `compute_metrics.sharpe` | `calc_sharpe` | ✅ |
| CVaR 5% | `compute_metrics.cvar_5` (pandas quantile) | `calc_cvar` | ✅ |
| Skewness | `compute_metrics.skewness` (pandas .skew) | `calc_skewness` | ✅ |
| Excess Kurtosis | `timeseries_lib.heavy_tail_stats` (scipy.stats) | `calc_excess_kurtosis` | ✅ |
| MDD Duration | `compute_metrics.mdd_duration` | `calc_mdd_duration` | ✅ |
| Alpha (CAPM) | `compute_metrics.alpha` | `calc_alpha` | ✅ |

**Final 에 부재 → 학술 표준 정의 (출처 명시)**:

| 메트릭 | 정의 | 출처 |
|---|---|---|
| **R²** | `corr(fund, mkt)²` (Pearson² = OLS R² 동치) | Sharpe (1964) CAPM |
| **Correlation** | Pearson correlation | 학술 표준 |
| **Tracking Error** | `std(fund - bench) × √12` | Grinold & Kahn (1999) |
| **VaR 5%** | `ret.quantile(0.05)` (Historical method) | Jorion (2007) |
| **Recovery Time** | DD trough → 신고가 회복 개월 수 | Magdon-Ismail & Atiya (2004) |
| **Top N DD** | local minima 식별 (depth 기준 정렬) | 표준 DD 분해 |
| **Hill ξ̂** | $\frac{1}{k}\sum \log X_{(i)} - \log X_{(k+1)}$ | Hill (1975) |

**검증 결과** (실제 데이터 fund.ret + fund.spy_ret + panel.rf_1m):
- Volatility 0.1345 / MDD -0.1289 / Beta 0.730 / Sortino 1.862 / Sharpe 1.103 / CVaR -0.0687 — **모두 final 1e-3 이내 일치**
- R² 0.5984 (= corr² 0.5984, 수학적 동치 검증 True)
- TE 9.41% / VaR -5.31%
- Top 3 DD: #1 2011-08 (-12.9%, dur 4m, rec 3m) / #2 2022-08 (-12.8%) / #3 2010-05 (-10.5%)
- Hill ξ̂: Fund 0.356 / SPY 0.378 → **Fund tail risk 가 SPY 보다 적음** (학술 narrative 검증)

**결과 / 함의**:
- `lib/metric_calculators.py` 보강 — 11 신규 함수 (R², Correlation, Top N DD, Recovery Time, Hill, Rolling 5종)
- `lib/risk_charts.py` 신규 — 6 영역 함수
- 모든 결과가 final 정합 또는 학술 출처 명시 → 정직성 확보

### 페이지 메타 결정 (Risk M-1 ~ M-4)

#### Risk M-1. 영역 개수

**검토된 옵션**:
- (a) 압축 6 영역
- (b) 표준 8 영역
- (c) 풍부 9-10 영역

**결정**: (b) 표준 8 영역

**근거**:
1. 위험 메트릭 (Pool-3) 충분 활용
2. Drawdown / VaR-CVaR / Beta-R²-TE / Tail Risk 분리 = 학술 표준
3. (a) 압축은 Tail Risk (Hill) 별도 영역 부재로 narrative 약화

#### Risk M-2. Sub-header 포함

**결정**: (a) 포함

**근거**: Performance 페이지 일관 — 토글 영향 안내 청중 친화

#### Risk M-3. Risk Summary KPI 5개

**검토된 옵션**:
- (a) 위험 다각도: Volatility / MDD / Downside Dev / VaR 95% / CVaR 95%
- (b) 위험 + 시장 노출: Volatility / MDD / Beta / R² / Tracking Error
- (c) 종합 균형: Volatility / MDD / Beta / VaR 95% / Down Capture
- (d) 자유 선택

**결정**: (b) 위험 + 시장 노출

**근거 (자세한 비교)**:

| 차원 | (a) 위험 다각도 | (b) 위험+시장 노출 |
|---|---|---|
| 영역 5 (VaR 분포) 와 중복 | ★★★ 직접 중복 | ✗ |
| 영역 6 (Beta 시계열) 와 중복 | ✗ | ★★ 부분 중복 (시계열 vs 카드) |
| 영역 8 (Tail Risk Hill) 와 중복 | △ tail 중복 | ✗ |
| VolControl 정체성 매칭 | ★★ | ★★★ |
| 학술 표준 부합 (CFA Total/Systematic/Active) | △ Total only | ★★★ 모두 커버 |
| 청중 인식 | "위험만 강조" | "다각도 위험 분석" |

**최종 근거**:
1. **펀드 정체성 (VolControl) 직접 매칭** — Volatility/MDD (절대) + Beta/R² (시장) + TE (액티브)
2. **영역 5/8 과 중복 회피** — VaR/CVaR/Hill 은 자세한 영역에서 다룸
3. **학술 표준 부합** — CFA Institute 위험 분류 (Total/Systematic/Active) 모두 커버
4. **다차원 위험 narrative** — 청중이 "위험을 다양한 차원에서" 인식

#### Risk M-4. Drawdown 시계열 표시 방식

**검토된 옵션**:
- (a) 단일 DD 영역 차트
- (b) 이중 차트 (누적수익 + DD) — Overview 영역 3 동일
- (c) DD + Recovery Time 분석
- (d) DD heatmap (월×년)

**결정**: (c) DD + Recovery Time 분석

**근거**:
1. Overview 영역 3 (이중 차트) 와 차별화 — 깊이 있는 분석
2. Recovery Time = "DD 발생 → 회복 기간" → 위험 통제 narrative
3. (d) Heatmap 은 Risk Metrics 종합 표 (영역 7) 와 중복

### Risk Metrics 페이지 8 영역 구조

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트 + 토글 안내)
3. Risk Summary KPI 5개         (Vol / MDD / Beta / R² / TE)
4. Drawdown 시계열 + Recovery Time
5. VaR / CVaR 분포              (히스토그램 + 임계선)
6. Beta + R² + Tracking Error 시계열  (Rolling)
7. Risk Metrics 종합 표         (Image 4)
8. Tail Risk 분석               (Hill estimator + extreme value theory)
9. Footer                       (Overview 동일)
```

### 영역 2: Sub-header — 확정

#### 결정 항목 R2-1: 텍스트 내용

**검토된 옵션**:
- (a) Performance 패턴 동일 적용
- (b) 더 자세 (Hill estimator 등 언급)
- (c) 자유 작성

**결정**: (a) Performance 패턴 동일

**근거**:
1. 페이지 일관성 — Performance / Risk Metrics 등 모든 페이지 Sub-header 패턴 통일
2. 위험 narrative 강화 텍스트로 적용

**텍스트 안**:
```
Risk Metrics (위험 지표)
펀드의 위험 통제 / 시장 노출 / tail risk 분석.
사이드바에서 기간 (FULL / TEST / HO) + 비교 벤치마크 (SPY/EW/IVW) 토글 가능.
```

**디자인 / 한/영 병기**: Performance 영역 2 일관 적용 (카드/배너 + 한/영 병기)

#### 결과 / 함의

- 모든 페이지 Sub-header 동일 패턴 (Performance / Risk Metrics / Holdings 등)

---

### 영역 3: Risk Summary KPI 5개 — 확정

#### 영역 3 통합 배경 (Context)

이미 메트릭 확정 (Vol/MDD/Beta/R²/TE — Risk M-3) → 표시 방식만 결정.

#### 결정 항목 R3-1: 표시 기간

**결정**: (b) 사이드바 토글 반응

**근거**:
1. Performance 영역 3 패턴 일관 — 인터랙션 일관성 원칙
2. 페이지 안에서 깊이 탐색 가능
3. (a) Overview Hero 동일은 첫인상 강하지만 페이지 의미 약화

#### 결정 항목 R3-2: 카드 디자인

**검토된 옵션**:
- (a) Overview Hero 동일 (큰 숫자 + sparkline)
- (b) Performance 영역 3 동일 (큰 숫자만, 단순)
- (c) 추가 시각화 (큰 숫자 + delta + 미니 트렌드)

**결정**: (b) Performance 동일 단순

**근거 — (c) 난잡성 분석**:

| 요소 | (b) 단순 | (c) 추가 시각화 |
|---|---|---|
| 다중 토글 활성 시 카드 안 정보 | 적정 | ★★ 과부하 |
| 카드 높이 | 보통 | ★★ 길어짐 |
| 영역 6 (Beta 시계열) 와 중복 | ✗ | ★★ sparkline 중복 |
| 정보 위계 (Hero ≠ Summary) | ★★★ 명확 | △ 무너짐 |

**최종 근거**:
1. 다중 토글 활성 시 카드 안 정보 과부하 회피
2. 영역 6 (Beta+R²+TE Rolling 시계열) 가 sparkline 역할 대체
3. Tooltip (R3-4) 으로 추가 정보 hover 노출 가능
4. 정보 위계 원칙 — Overview Hero (풍부) / Performance-Risk Summary (단순)

#### 결정 항목 R3-3: 벤치마크 delta

**결정**: (b) 다중 토글 적용

**근거**:
1. Performance 일관 — 인터랙션 일관성 원칙
2. 위험 KPI 의 "vs SPY 위험 비교" narrative — 펀드 위험이 SPY 보다 낮음 → 좋음 표시 (↓ 화살표)
3. EW / IVW 비교로 "단순 baseline 대비 위험 통제" narrative 강화

#### 결정 항목 R3-4: Hover tooltip

**결정**: (a) Tooltip 포함

**근거**:
1. 가상 투자자 친화 — R² / Tracking Error 등 이해 어려움
2. Tooltip 으로 메트릭 정의 + 좋음/나쁨 방향 안내
3. 카드 위 ⓘ 아이콘 + hover 시 노출

**Tooltip 텍스트 안**:
- Volatility: "변동성 (연환산 표준편차) — 펀드 가격 변동의 강도. 낮을수록 안정적."
- MDD: "최대낙폭 — 고점 대비 최대 손실. 절대값이 작을수록 좋음."
- Beta: "시장 베타 — SPY 대비 시장 노출도. 1보다 작으면 시장 변동에 덜 민감."
- R²: "결정계수 — Beta 의 설명력. 100% 에 가까울수록 시장 변동으로 펀드 변동 설명 가능."
- Tracking Error: "추적오차 — 벤치마크 대비 액티브 운용 위험. 낮을수록 벤치마크 추적, 높을수록 액티브."

#### 시각화 예시 (확정 사항 조합)

```
[Header — Overview 동일]

┌─ ℹ️ Risk Metrics (위험 지표) ───────────────────────────┐
│ 펀드의 위험 통제 / 시장 노출 / tail risk 분석.           │
│ 사이드바에서 기간 + 비교 벤치마크 토글 가능.             │
└──────────────────────────────────────────────────────────┘

[기간: TEST 168m]  [비교: SPY ✓ / EW ☐ / IVW ☐]
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│Volatility│  MDD     │  Beta    │  R²      │  TE      │
│ XX.X%    │ -XX.X%   │  X.XX    │  XX.X%   │ X.X%     │
│ vs SPY   │ vs SPY   │ vs SPY   │          │          │
│ ▼-X.X%↓  │ ▲+X.X%↓  │ <1 ✓    │          │          │
│  (좋음)  │  (좋음)  │  (낮음)  │          │          │
│ⓘVolatility│ⓘMDD     │ⓘBeta    │ⓘR²      │ⓘTE       │
└──────────┴──────────┴──────────┴──────────┴──────────┘

(↓ 화살표 = "낮을수록 좋음" / 펀드 정체성 = 위험 ↓ = 좋음)
```

#### 결과 / 함의

- 모든 페이지 Summary KPI 디자인 패턴 = 단순 (큰 숫자 + delta + tooltip)
- Hero KPI (Overview) = 풍부 (sparkline) — 정보 위계 일관 유지
- Tooltip 텍스트는 모든 메트릭에 표준화 (구현 시 dictionary 참조)

### 영역 4: Drawdown 시계열 + Recovery Time — 확정

#### 영역 4 통합 배경 (Context)

Risk M-4 결정 (DD + Recovery Time 분석) 의 자세한 시각화. **단순 DD 시계열 (Overview 영역 3)** 과 차별화: Recovery Time 추가 분석.

**Recovery Time**: DD 발생 → 이전 고점 회복까지 소요 기간 (개월). 학술 표준 (Pain Index 보조).

**핵심 narrative**: 펀드 = "DD 적게 발생 + 회복 빠름" → 위험 통제 강점.

#### 결정 항목 R4-1: 차트 종류

**검토된 옵션**:
- (a) 단일 DD 영역 차트
- (b) DD 영역 차트 + Recovery 마커
- (c) DD + Duration/Recovery 막대 이중 차트
- (d) Underwater plot

**결정**: (c) DD + Duration/Recovery 막대 이중 차트

**근거**:
1. **이중 차트 = 시각 풍부** — 위: DD 시계열 / 아래: 각 DD 의 Duration + Recovery 막대
2. Overview 영역 3 (이중 차트: 누적수익 + DD) 와 차별화 — Recovery 분석 추가
3. (a) 단순 DD 는 Recovery 분석 부재
4. (d) Underwater plot 은 직관성 ↓ (가상 투자자 친화 ↓)

#### 결정 항목 R4-2: Top N DD 강조

**결정**: (a) Top 3 DD 강조 (라벨 + Recovery Time)

**근거**:
1. 핵심 3개만 강조 — 시각 깔끔
2. 나머지 DD 는 hover 로 detail 노출
3. (b) Top 5, (c) Top 10 은 시각 혼잡

#### 결정 항목 R4-3: Recovery Time 표시 방식

**검토된 옵션**:
- (a) 화살표 + 개월 수
- (b) 별도 표
- (c) 차트 마커 + 표 결합

**결정**: (c) 차트 마커 + 표 결합

**근거**:
1. **시각 + 데이터 둘 다** — 마커로 시각, 표로 정확 수치
2. (a) 화살표만은 데이터 정확성 ↓
3. (b) 표만은 시각적 인상 ↓

#### 결정 항목 R4-4: 비교

**결정**: (b) 펀드 vs SPY

**근거**:
1. **위험 통제 narrative 핵심** — Fund DD < SPY DD = 우월성
2. 다중 토글 (SPY/EW/IVW) 은 영역 6 (Beta 시계열) / 영역 7 (종합 표) 에서 자세히
3. DD 차트는 한 화면에 두 라인 + Top 3 마커로 충분 (3-4 라인 시 혼잡)

#### 결정 항목 R4-5: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip (DD 깊이 + 회복 기간) ✓
- Plotly 기본 zoom + slider ✓
- DD 클릭 → 해당 시기 detail expand ✓ (Q-Zoom)
- Top N DD 강조 색상 (레드 톤) ✓

#### 결정 항목 R4-6: 추가 표시

**검토된 옵션**:
- (a) 평균 DD 깊이 라인
- (b) Pain Index 표시
- (c) Regime 배경색
- (d) 모두

**결정**: (a) 평균 DD 라인 + (c) Regime 배경색

**근거**:
1. **평균 DD 라인** = 통계 보강 (Top 3 vs 평균 비교)
2. **Regime 배경색** = 영역 3 (누적수익) 일관 + 위험 발생 시점의 regime 컨텍스트
3. (b) Pain Index 는 영역 7 (종합 표) 에 별도 메트릭으로

#### 시각화 예시 (확정 사항 조합)

```
[Risk Summary KPI 5개]

┌─ Drawdown + Recovery Time Analysis ─────────────────────┐
│                                                         │
│ ┌─ Drawdown 시계열 ──────────────────────────────────┐ │
│ │  ┌─R1─┬───R2────┬──R3──┬─HO─┐ (Regime 배경색)     │ │
│ │   0%──────────────────────────────                 │ │
│ │ -10%─    ╲╱─    ╲╱  ╲╱╲  ╲╱   ╲╱                 │ │
│ │       ★Top1(-X%) ★Top2(-Y%)  ★Top3(-Z%)            │ │
│ │ -20%─        ╲    ╲           ─── Avg DD (라인)   │ │
│ │ -30%─                                               │ │
│ │       Fund ───  SPY ┄┄┄                            │ │
│ │  Click DD → 해당 시기 detail expand (Q-Zoom)       │ │
│ └────────────────────────────────────────────────────┘ │
│                                                         │
│ ┌─ Top 3 DD: Duration + Recovery 표 + 막대 ─────────┐ │
│ │ Rank | Date    | DD%  | Duration | Recovery       │ │
│ │ #1   | 2020-03 | -X%  | 2개월    | 5개월          │ │
│ │ #2   | 2022-09 | -Y%  | 9개월    | X개월          │ │
│ │ #3   | 2024-12 | -Z%  | 1개월    | (진행 중)      │ │
│ │                                                     │ │
│ │ (옆 막대: Duration 막대 + Recovery 막대 색 구분)    │ │
│ └────────────────────────────────────────────────────┘ │
│                                                         │
│ Hover: "2020-03: DD -X.X%, Recovery 5 months"           │
└─────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- 평균 DD 라인 = `numpy.mean(drawdowns)` 단순 계산
- Recovery Time = DD bottom 이후 신고가 갱신까지의 개월 수
- Top 3 DD 식별 = `pd.Series.nsmallest(3)` 기반
- 진행 중 DD (회복 안 됨) = "(진행 중)" 표시 + 기간 구분
- Regime 배경색 = 영역 3 (Overview), 영역 6 (Performance) 동일 색상 일관

### 영역 5: VaR / CVaR 분포 — 확정

#### 영역 5 통합 배경 (Context)

펀드 수익률 분포에 **VaR + CVaR (Expected Shortfall)** 임계선 시각화.

**메트릭** (확정 풀):
- Historical VaR 5% — 과거 분포의 5% 분위수
- CVaR 5% — VaR 초과 시 평균 손실

**vs Performance 영역 8 (분포 통계 카드)**:
- 영역 8 = 분포 모양 통계 (Skewness/Kurtosis/Tail Ratio)
- 여기 영역 5 = **tail risk 정량 임계선** (VaR/CVaR)

**vs Risk Metrics 영역 8 (Tail Risk Hill)**:
- 영역 5 = 분포 시각 + 임계선 (직관)
- 영역 8 = Extreme Value Theory (Hill estimator, 학술 깊이)

#### 결정 항목 R5-1: 표시 형식

**검토된 옵션**:
- (a) 히스토그램 + 임계선
- (b) KDE + 임계선
- (c) 히스토그램 + KDE + 임계선
- (d) Box plot
- (e) Violin plot

**결정**: (c) 히스토그램 + KDE 오버레이 + 임계선

**근거**:
1. **히스토그램 = 빈도 시각** (가상 투자자 직관)
2. **KDE = 부드러운 분포 모양** (학술 정확)
3. Performance 영역 8 (KDE only) 와 차별화 — 히스토그램 추가
4. (d) Box plot, (e) Violin 은 정보량 ↓ + tail 시각 약함

#### 결정 항목 R5-2: VaR / CVaR 임계선 표시

**결정**: (d) 모두 (수직선 + Tail 영역 강조 + 라벨 + 값)

**근거**:
1. **수직선 + Tail 영역 강조** = 시각 즉시 이해 (red 영역 = worst 5%)
2. **라벨 + 값** = 정확 수치 노출
3. 시각 + 정확성 모두 충족

#### 결정 항목 R5-3: 일별 / 월별 표시 방식

**결정**: (a) 탭 전환 (월별 / 일별)

**근거**:
1. Performance 영역 8 패턴 일관 — 인터랙션 일관성 원칙
2. 두 셋 동시 표시는 영역 분량 ↑
3. 사용자가 의도적으로 시간 단위 선택

#### 결정 항목 R5-4: 비교

**검토된 옵션**:
- (a) 펀드 only
- (b) 펀드 vs SPY 오버레이
- (c) 다중 토글
- (d) 좌우 분할 (subplot)

**결정**: (b) 펀드 vs SPY 오버레이

**근거**:
1. **두 분포 직접 비교** — 펀드 vs SPY tail risk 차이 즉시 시각
2. 다중 토글 (SPY/EW/IVW) 은 영역 7 (종합 표) 에서 자세히
3. (d) 좌우 분할은 분포 비교 직관성 ↓ (오버레이가 더 명확)

#### 결정 항목 R5-5: 인터랙션

**결정**: 드래그 제외 모두 채택

**채택 인터랙션**:
- Hover tooltip (수익률 구간 + 빈도) ✓
- VaR/CVaR 임계선 hover (정의 tooltip) ✓
- Tail 영역 클릭 → 해당 시기 detail expand ✓ (Q-Zoom)
- 임계선 드래그 = 보류 (구현 복잡, 가치 ↓)

#### 결정 항목 R5-6: 추가 표시

**검토된 옵션**:
- (a) 정규분포 비교 라인
- (b) Statistics 보조 표
- (c) 다른 percentile (1% VaR 등)
- (d) 모두

**결정**: (a) 정규분포 비교 + (b) Statistics 보조 표

**근거**:
1. **정규분포 비교 라인** = fat tail 시각화 (실제 분포 vs 정규분포 가정 차이)
2. **Statistics 보조 표** = Mean/Median/Std/VaR/CVaR 정확 수치 (펀드 vs SPY)
3. (c) 다른 percentile (1% VaR) 은 영역 8 (Hill) 에서 자세히

#### 시각화 예시 (확정 사항 조합)

```
[Drawdown + Recovery (영역 4)]

┌─ VaR / CVaR 분포 ───────────────────────────────────────┐
│                                                         │
│ [Tab: 월별 (Monthly) | 일별 (Daily)]                    │
│                                                         │
│ ┌─ 펀드 vs SPY 분포 (오버레이) ─────────────────────┐ │
│ │  Frequency                                          │ │
│ │  ┌─Tail (red 영역)─┐                                │ │
│ │  │░░░░░░          │     ▓▓▓                        │ │
│ │  │░░░░░░░░        │   ▓▓▓▓▓▓▓                      │ │
│ │  │░░░░░░░░░░      │ ▓▓▓▓▓▓▓▓▓▓▓                    │ │
│ │  │  ░░░░░░░       │▓▓▓▓▓▓▓▓▓▓▓▓▓                   │ │
│ │  │   ░░░░░  Norm  ─KDE                             │ │
│ │  └────│──┘────────────────────────                  │ │
│ │       │  │                                          │ │
│ │  CVaR -X.X%   VaR -Y.Y% (수직선 + 라벨 + 값)        │ │
│ │                                                     │ │
│ │  ▓ Fund   ░ SPY  ┄┄┄ 정규분포 비교                  │ │
│ │  Click Tail 영역 → 해당 시기 detail expand          │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ ┌─ Statistics (보조 표) ────────────────────────────┐  │
│ │ Stat        | Fund    | SPY                       │  │
│ │ Mean        | +X.XX%  | +X.XX%                    │  │
│ │ Median      | +X.XX%  | +X.XX%                    │  │
│ │ Std         | X.XX%   | X.XX%                     │  │
│ │ VaR 5%      | -X.XX%  | -Y.YY%                    │  │
│ │ CVaR 5%     | -X.XX%  | -Y.YY%                    │  │
│ └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- 정규분포 비교 라인 = `scipy.stats.norm.pdf(x, mean, std)` 활용
- 히스토그램 + KDE 오버레이 = Plotly `go.Histogram` + `go.Scatter` (KDE)
- Tail 영역 강조 = Plotly `add_vrect` 또는 fill_between
- Statistics 표 = pandas DataFrame → Streamlit `st.dataframe`

### 영역 6: Beta + R² + Tracking Error 시계열 (Rolling) — 확정

#### 영역 6 통합 배경 (Context)

3개 시장 노출 / 액티브 위험 메트릭의 Rolling 시계열.

**메트릭** (확정 풀):
- Beta (rolling)
- R² (rolling)
- Tracking Error (rolling)

**Image 2 (Performance 영역 5) 와 차별성**:
- Performance 영역 5 = Active Return + Tracking Error
- Risk Metrics 영역 6 = **Beta + R²** 추가 (시장 노출 분석)

#### 결정 항목 R6-1: 표시 방식

**검토된 옵션**:
- (a) 한 차트 3개 라인 (다중 축)
- (b) 3개 분리 차트 (위아래)
- (c) Tab 전환
- (d) 토글 다중 선택

**결정**: (b) 3개 분리 차트 (위아래)

**근거**:
1. **스케일 차이 ↑** — Beta(0~1.5) / R²(0~100%) / TE(0~10%) 다중 축 시 가독성 ↓
2. (a) 다중 축은 청중 혼란 가능
3. 분리 차트로 각 메트릭의 시간 변화 명확

#### 결정 항목 R6-2: Rolling 윈도우

**결정**: (b) 토글 (12m / 36m / 60m)

**근거**:
1. Performance 영역 5 (Image 2) / 영역 6 (Annualized Rolling) 일관
2. 다양한 시간 척도 검증 가능
3. 인터랙션 일관성 원칙

#### 결정 항목 R6-3: 비교

**검토된 옵션**:
- (a) 펀드 only
- (b) 다중 토글

**결정**: (a) 펀드 only

**근거**:
1. **Beta/R²/TE 는 본질적으로 "벤치마크 대비" 메트릭** — SPY 와 비교가 디폴트 (Beta=Cov(P,SPY)/Var(SPY))
2. 다중 벤치마크는 영역 7 (종합 표) 에서 자세히
3. 시계열에 다중 벤치마크 추가는 시각 혼잡

#### 결정 항목 R6-4: 추가 표시

**검토된 옵션**:
- (a) Beta=1 기준선
- (b) Beta=0 기준선
- (c) Regime 배경색
- (d) 모두

**결정**: (a) Beta=1 기준선만

**근거**:
1. **Beta=1 = 시장 동행 기준** — Beta < 1 (방어적) / Beta > 1 (공격적) 즉시 시각
2. (b) Beta=0 은 Market neutral 기준이지만 long-only equity 펀드와 거리
3. (c) Regime 배경색 = 영역 4 (Drawdown) 에서 이미 적용 → 영역 6 에서는 단순화
4. **단순한 시각 — Beta 시각화의 핵심 메시지 강화**

#### 결정 항목 R6-5: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- Plotly 기본 zoom + slider ✓
- 시기 클릭 → expand ✓ (Q-Zoom)
- 라인 클릭 → 단독 강조 ✓ (Plotly legend 기본)

#### 결정 항목 R6-6: β 음수 처리

**배경**: 사용자 명시 — 우리 포트폴리오 β 음수 가능성 (Treynor 제외 결정 근거)

**검토된 옵션**:
- (a) 0 기준선과 함께 자연 표시
- (b) β < 0 강조 (alert 영역)
- (c) β < 0 시 R² 함께 표시 (신뢰성 평가)

**결정**: (c) R² 함께 표시 (신뢰성 평가)

**근거**:
1. **β < 0 자체가 noise 가능성** — R² 낮으면 Beta estimate 신뢰성 ↓
2. **학술 정직성** — β 의 신뢰성 평가는 R² 동시 확인 표준
3. **Tooltip / annotation**: "β < 0 시기의 R² = X% (신뢰성 검토 필요)"
4. (b) alert 는 청중에게 부정적 인상 강요 → noise 일 가능성 명시가 더 정직

#### 시각화 예시 (확정 사항 조합)

```
[VaR/CVaR 분포 (영역 5)]

┌─ Beta + R² + Tracking Error (Rolling) ─────────────────┐
│                                                         │
│ [Window: 12m / 36m / 60m ▼]                             │
│                                                         │
│ ┌─ Beta (Rolling) ──────────────────────────────────┐  │
│ │  1.5┤                                              │  │
│ │  1.0┼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Beta=1 │  │
│ │  0.5┤────╱───╱╲╱─╲╱╲                              │  │
│ │  0.0┼─────────────────────────────────────         │  │
│ │ -0.5┤                              ╲╱╲ ← β<0     │  │
│ │       2013  2016  2019  2022  2025                │  │
│ │       Tooltip: "2024-12: β=-0.12, R²=18%          │  │
│ │                  (신뢰성 검토 필요)"               │  │
│ └─────────────────────────────────────────────────┘  │
│                                                         │
│ ┌─ R² (Rolling) ────────────────────────────────────┐  │
│ │ 100%┤   ──────────                                 │  │
│ │  50%┤───        ──────╲╱╲                          │  │
│ │   0%┼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │  │
│ │     ※ β<0 시기에 R² 낮음 → noise 가능성           │  │
│ └────────────────────────────────────────────────────┘ │
│                                                         │
│ ┌─ Tracking Error (Rolling) ────────────────────────┐  │
│ │  10%┤                                              │  │
│ │   5%┤───╱╲╱─╲╱╲                                    │  │
│ │   0%┼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │  │
│ └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- 3개 분리 차트 = Plotly `make_subplots(rows=3)` 활용
- Beta/R²/TE Rolling 산출 = pandas `.rolling(window).apply(lambda x: ...)` 활용
- β<0 시 R² 함께 표시 = annotation + tooltip 동적 조정 (β<0 감지 시 R² 노출)

### 영역 7: Risk Metrics 종합 표 (Image 4) — 확정

#### 영역 7 통합 배경 (Context)

레퍼런스 PortfolioVisualizer **Image 4 (Risk and Return Metrics 표)** 스타일의 종합 메트릭 표.

**메트릭 풀 (C-2) 의 핵심 메트릭들을 한 표에 종합** — 청중이 한 화면에서 모든 위험/수익조정/시장 노출 메트릭을 펀드 vs 벤치마크 비교 가능.

#### 결정 항목 R7-1: 메트릭 선정

**검토된 옵션**:
- (a) Image 4 그대로 (15+ 메트릭)
- (b) Pool-2 + Pool-3 (위험조정 + 위험)
- (c) 메트릭 풀 전체 (50+)
- (d) 핵심만 10개
- (e) 카테고리별 그룹화

**결정**: (e) 카테고리별 그룹화

**근거**:
1. **메트릭 풀 활용 의의** — 53개 메트릭 풀 중 적절히 picking
2. **카테고리 구조** = 학술 표준 (Performance / Risk-Adj / Risk / Market / Factor / Distribution)
3. (a) Image 4 그대로는 중복 메트릭 多 / (c) 전체는 너무 많음 / (d) 10개는 정보량 ↓

**카테고리별 메트릭 안**:

| 카테고리 | 메트릭 |
|---|---|
| **Performance Returns** | Cumulative Return / CAGR / Arithmetic Mean (annualized) |
| **Risk-adjusted Return** | Sharpe / Sortino / Calmar / IR / M² |
| **Risk** | Volatility / MDD / Downside Dev / Historical VaR 5% / CVaR 5% |
| **Market Exposure** | Beta / R² / Correlation / Tracking Error |
| **Factor Analysis** | CAPM Alpha / FF5 Alpha (선택) |
| **Distribution** | Skewness / Kurtosis / Tail Ratio |

→ 약 22-25 메트릭

#### 결정 항목 R7-2: 표시 형식

**결정**: (b) 그룹화 표 + 확장 가능 (`st.expander`)

**근거**:
1. **Streamlit `st.expander`** 활용 — 카테고리별 펼침/접기
2. 청중이 관심 카테고리만 펼쳐 볼 수 있음 (자유 탐색)
3. (a) 단순 표는 22-25 메트릭 한 화면에 표시 시 스크롤 ↑
4. (c) Heatmap 은 영역 7 (Performance Regime Heatmap) 와 중복

#### 결정 항목 R7-3: 비교

**검토된 옵션**:
- (a) Fund vs SPY
- (b) 다중 토글 (Fund / SPY / EW / IVW)
- (c) Fund + SPY + Diff column
- (d) 다중 토글 + Diff column

**결정**: (d) 다중 토글 + Diff column

**근거**:
1. **다중 토글** = Performance / Risk Metrics 다른 영역 일관 (인터랙션 일관성 원칙)
2. **Diff column 추가** = 정확한 비교 narrative (Fund - Benchmark)
3. (a) 단순 SPY 비교는 다른 영역 (영역 5/6) 보다 정보 ↓

**컬럼 구조** (예: SPY + EW 활성 시):
```
| Metric | Fund | SPY | Diff vs SPY | EW | Diff vs EW |
```

#### 결정 항목 R7-4: 정렬 / 그룹화

**결정**: (a) 카테고리 헤더 그룹화

**근거**:
1. R7-1 (카테고리별) 결정과 일관
2. **학술 표준** — 메트릭을 의미별 그룹으로 분류
3. (b) 알파벳 순은 의미적 연결성 ↓
4. (d) Image 4 순서는 우리 카테고리 분류와 다름

#### 결정 항목 R7-5: 인터랙션

**결정**: 권장대로 — hover tooltip / 정렬 토글 / CSV 다운로드 / 좋음나쁨 색상 (메트릭 클릭 보류)

**채택 인터랙션**:
- **Hover tooltip** = 메트릭 정의 즉시 노출 (R7-6 Footnote 와 보완)
- **정렬 토글** = 오름/내림차순 (Diff 기준 정렬 등)
- **CSV 다운로드** = 청중이 원본 데이터 export 가능
- **좋음/나쁨 색상** = Fund vs SPY 비교 시 우월 = green / 열위 = red 자동 색상

**보류**:
- **메트릭 클릭 → 차트 이동** = 구현 복잡 + Footnote / Tooltip 으로 충분

#### 결정 항목 R7-6: 추가 표시

**결정**: (a) Footnote 메트릭 정의 일괄

**근거**:
1. **표 아래 정의 모음** — 전체 메트릭 정의 한 곳에서 확인
2. Hover tooltip (R7-5) 과 보완 — Hover = 빠른 확인 / Footnote = 자세한 정의
3. (b) 카테고리 설명은 카테고리 헤더 (R7-4) 에 통합 가능
4. (c) RC 컬럼 (위험 기여도) = 학술 깊이 ↑ 하지만 복잡도 ↑ → 영역 8 (Hill) 와 중복

#### 시각화 예시 (확정 사항 조합)

```
[Beta+R²+TE 시계열 (영역 6)]

┌─ Risk Metrics 종합 표 ───────────────────────────────────┐
│ [기간: TEST 168m]   [☑ SPY  ☐ EW  ☐ IVW] [⬇ CSV]      │
│                                                          │
│ Metric              | Fund   | SPY    | Diff vs SPY    │
├──────────────────────┼────────┼────────┼────────────────┤
│ ▼ Performance Returns                                   │
│   Cumulative Return | +X%🟢  | +X%    | +X%🟢          │
│   CAGR              | +X%    | +X%    | +X%            │
│   Arithmetic Mean   | +X%    | +X%    | +X%            │
├──────────────────────┼────────┼────────┼────────────────┤
│ ▼ Risk-adjusted Return                                  │
│   Sharpe Ratio      | X.XX🔴 | X.XX   | -X.XX🔴        │
│   Sortino Ratio     | X.XX   | X.XX   | -X.XX          │
│   Calmar Ratio      | X.XX   | X.XX   | -X.XX          │
│   IR                | X.XX   | -      | -              │
│   M²                | +X%    | +X%    | -X%            │
├──────────────────────┼────────┼────────┼────────────────┤
│ ▶ Risk (펼침/접기 가능)                                 │
│ ▶ Market Exposure                                       │
│ ▶ Factor Analysis                                       │
│ ▶ Distribution                                          │
│                                                          │
│ ※ Footnote (메트릭 정의):                               │
│  - Sharpe = (R - Rf) / σ                                │
│  - Sortino = (R - Rf) / σ_downside                      │
│  - Calmar = CAGR / |MDD|                                │
│  ... (전체 정의)                                        │
└──────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- 카테고리 헤더 = `st.expander` 활용 (펼침/접기 가능)
- 좋음/나쁨 색상 = Fund vs SPY 자동 비교 → green/red highlighting
- CSV 다운로드 = `st.download_button` 활용
- Footnote = `st.markdown` 으로 LaTeX 수식 포함 가능
- 동적 컬럼 (다중 토글) = pandas DataFrame `pd.concat([fund_col, spy_col, diff_col, ew_col, ...], axis=1)` 동적 생성

### 영역 8: Tail Risk 분석 (Hill Estimator) — 확정 (옵션 C 축소형 → Q-F2 추가 축소)

> **★ 균형 옵션 (B) Q-F2 적용**: Jarque-Bera 제거 — Methodology 영역 7 (LSTM 정당화 narrative) 로 통합.
> **Hill estimator 만 유지** — 영역 7 (Methodology Jarque-Bera) 와 중복 회피.

#### 영역 8 통합 배경 (Context)

**Extreme Value Theory (EVT) 기반 tail risk 정량 분석**. Performance 영역 8 (S8-3) 결정에서 Risk Metrics 페이지로 이동 (Jarque-Bera).

**vs 영역 5 (VaR/CVaR 분포)**:
- 영역 5 = 분포 시각 + 임계선 (직관)
- 영역 8 = **EVT 기반 학술 깊이** (Hill estimator + 정규분포 검정)

#### 사용자 의문 — 필요성 검토 + 결정

**검토된 필요성**:

| 차원 | Pro | Con |
|---|---|---|
| 펀드 정체성 (VolControl) 강화 | ★★★ | — |
| 학술 정직성 (심사위원) | ★★★ | — |
| 차별화 (일반 fact sheet 에 없음) | ★★★ | — |
| VaR/CVaR 보완 | ★★ | 영역 5 와 일부 중복 |
| 가상 투자자 친화 | — | ★★★ Hill / GPD 어려움 |
| 페이지 분량 | — | ★★ 8 영역 길이 |
| 구현 복잡도 | — | ★★ scipy/arch 의존 |

**3가지 옵션 검토**:
- **A**: 영역 8 전체 유지 (Hill + Jarque-Bera + GPD + Block Maxima 모두)
- **B**: 영역 8 제거 (VaR/CVaR 만으로 충족)
- **C**: 영역 8 축소 (Hill + Jarque-Bera 만, GPD/Block Maxima 제외)

**결정**: 옵션 C (축소형)

**근거**:
1. **학술 깊이 유지** + 페이지 부담 ↓ (균형)
2. **Hill estimator** = 학술 표준 tail 측정 + Hill Plot 시각화 표준
3. **Jarque-Bera** = 정규분포 검정 (Performance S8-3 결정에서 이동)
4. **GPD / Block Maxima 제외**: 학술 깊이 ↑↑ 하지만 가상 투자자에게 과도하게 어려움

#### 학술 배경 (Hill Estimator + EVT 설명)

**Hill Estimator (Hill 1975)**:

$$\hat{\xi}_k = \frac{1}{k}\sum_{i=1}^k \log(X_{(i)}) - \log(X_{(k+1)})$$

- $\hat{\xi}_k$ = tail index 추정치
- ξ > 0: Fat tail (극단값 자주) — 주식 일반 ξ≈0.2-0.4
- ξ = 0: Exponential tail (정규분포)
- ξ < 0: Bounded tail

**Hill Plot**: k (order statistics 수) vs ξ̂ — plateau 영역에서 ξ 안정 → 추정치

**EVT (Extreme Value Theory)**:
- Fisher-Tippett-Gnedenko theorem (1928) 토대
- 금융 응용: Embrechts, Klüppelberg, Mikosch (1997), McNeil et al. (2005)

#### 결정 항목 R8-1 ~ R8-6

##### R8-1: 핵심 시각화

**결정**: (c) Hill Plot + ξ 카드

**근거**: 학술 표준 (Hill Plot) + 청중 친화 (ξ 카드 단일 값)

##### R8-2: k 선택

**결정**: (a) auto plateau detection

**근거**: 사용자 부담 ↓ + 알고리즘 자동 결정 (가장 안정한 plateau 영역)

##### R8-3: 일별 / 월별

**결정**: (a) 일별 only

**근거**: tail risk 정확도는 일별 (large N) 이 우월. 월별은 N 작아 tail estimate noise ↑

##### R8-4: 비교

**결정**: (b) 펀드 vs SPY

**근거**: 핵심 narrative ("Fund tail risk < SPY"). 다중 토글은 가독성 ↓

##### R8-5: 학술 narrative

**결정**: (c) Footnote + 학술 설명 박스

**근거**:
1. **학술 설명 박스**: 가상 투자자에게 Hill estimator 의미 즉시 안내
2. **Footnote**: Hill (1975), Jarque-Bera (1980) 인용 (학술 정확성)

##### R8-6: 추가 EVT

**결정**: (b) +Jarque-Bera test (옵션 C 의 핵심)

**근거**:
1. **Performance S8-3 결정** — Jarque-Bera 를 About/Methodology 가 아닌 Risk Metrics 영역 8 로 통합 결정
2. **GPD / Block Maxima 제외**: 학술 깊이 ↑↑ 하지만 가상 투자자 부담 ↑↑
3. Hill + Jarque-Bera = 적정 학술 깊이

#### 시각화 예시 (옵션 C 축소형)

```
[Risk Metrics 종합 표 (영역 7)]

┌─ Tail Risk Analysis (Hill + Normality Test) ──────────┐
│                                                        │
│ ┌─ ℹ️ 학술 설명 박스 ─────────────────────────────┐  │
│ │ Hill estimator 는 분포 tail 의 두께를 측정합니다.│  │
│ │ ξ > 0 = fat tail (극단 손실 자주 발생).        │  │
│ │ 펀드 ξ 가 SPY 보다 낮으면 tail risk 가 적음.   │  │
│ └──────────────────────────────────────────────────┘  │
│                                                        │
│ ┌──────────────┬──────────────┐                       │
│ │ Fund ξ̂       │ SPY ξ̂        │                       │
│ │ X.XX         │ X.XX         │                       │
│ │ (auto k=N)  │ (auto k=M)   │                       │
│ │ ⓘ Fat tail  │ ⓘ Fat tail   │                       │
│ │  여부        │  여부        │                       │
│ └──────────────┴──────────────┘                       │
│                                                        │
│ ┌─ Hill Plot ────────────────────────────────────┐    │
│ │ ξ̂                                              │    │
│ │ 0.5┤                                            │    │
│ │ 0.4┤    ╱╲╱─────  ─── plateau (Fund ξ̂)        │    │
│ │ 0.3┤  ╱──    ╲╱╲                               │    │
│ │ 0.2┤        ╱──    ─── plateau (SPY ξ̂)        │    │
│ │ 0.1┤      ╱                                    │    │
│ │ 0.0┴─────────────────────────                  │    │
│ │     k=10  50  100  200  500                    │    │
│ │  Fund ───  SPY ┄┄┄                              │    │
│ │  Plateau 영역 = 안정 → ξ 추정치                 │    │
│ └─────────────────────────────────────────────────┘   │
│                                                        │
│ ┌─ Jarque-Bera Test (정규분포 검정) ─────────────┐    │
│ │ Series | JB    | p-value | 정규분포 가설      │    │
│ │ Fund   | X.XX  | 0.XX    | 기각/채택          │    │
│ │ SPY    | X.XX  | 0.XX    | 기각/채택          │    │
│ │ ※ p<0.05 → 정규분포 가설 기각 (fat tail 시사)  │    │
│ └─────────────────────────────────────────────────┘   │
│                                                        │
│ ※ Footnote:                                           │
│ - Hill, B.M. (1975). "A simple general approach to   │
│   inference about the tail of a distribution."        │
│   Annals of Statistics, 3, 1163-1174.                 │
│ - Jarque, C.M. & Bera, A.K. (1980). "Efficient tests  │
│   for normality, homoscedasticity and serial          │
│   independence of regression residuals."              │
│   Economics Letters, 6(3), 255-259.                   │
└────────────────────────────────────────────────────────┘
```

#### 결과 / 함의

- Hill estimator 구현 = `numpy` 기반 직접 계산 (외부 라이브러리 불필요)
- Hill Plot = Plotly `go.Scatter` (k vs ξ̂)
- Auto plateau detection = 슬라이딩 윈도우 표준편차 최소 영역
- Jarque-Bera = `scipy.stats.jarque_bera()` 활용
- 부록 학술 근거 일람에 추가:
  - Hill (1975)
  - Jarque-Bera (1980)
  - Embrechts et al. (1997)
  - McNeil et al. (2005)

---

### Risk Metrics 페이지 — 전체 확정 (영역 1~9)

#### 페이지 시각화 통합 (확정 사항 모두 조합)

```
┌────────────────────────────────────────────────────────────────┐
│ [영역 1: Header — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 2: Sub-header 카드/배너]                                  │
│ Risk Metrics (위험 지표) — 위험 통제 / 시장 노출 / tail risk    │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 3: Risk Summary KPI 5개]                                  │
│ Volatility / MDD / Beta / R² / Tracking Error                  │
│ 다중 토글 delta + Tooltip                                      │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 4: Drawdown + Recovery Time]                              │
│ DD 시계열 + Recovery 막대 + Top 3 강조 + 평균 + Regime 배경    │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 5: VaR / CVaR 분포]                                       │
│ 히스토그램 + KDE + Tail 영역 강조 + 정규분포 비교 + Stats 표   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 6: Beta + R² + TE Rolling 시계열]                         │
│ 3개 분리 차트 + Beta=1 기준선 + β<0 시 R² 신뢰성 평가          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 7: Risk Metrics 종합 표]                                  │
│ 카테고리별 그룹화 + 다중 토글 + Diff column + CSV 다운로드     │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 8: Tail Risk 분석 (옵션 C 축소)]                          │
│ Hill Plot + ξ 카드 + Jarque-Bera + 학술 narrative              │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 9: Footer — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘
```

#### Risk Metrics 페이지 결정 결과 / 함의

- **8 영역 모두 확정** → 구현 시 명확한 와이어프레임
- **인터랙션 일관성 원칙** 강력 적용:
  - 사이드바 토글 (기간 + 다중 벤치마크) → 모든 영역 영향
  - Q-Zoom (1+2+3) → 영역 4-6 시계열 차트
  - Tab 전환 → 영역 5 (월별/일별)
- **Performance 와 차별화**: 수익 → 위험 위주
- **학술 깊이 추가**: 영역 8 (Hill estimator + Jarque-Bera) → 심사위원 어필
- **다음 페이지 (Holdings)** 결정 시 동일 패턴 활용

### Risk Metrics 페이지 → Holdings 페이지로 진행

---


---

[← 03_performance.md](03_performance.md) | [05_holdings.md](05_holdings.md) →
