# Risk Metrics 페이지 — 와이어프레임

> **관련 decisionlog**: `04_risk_metrics.md`
> **상태**: 확정 (Hill 영역 = 옵션 C 축소형 → Q-F2 추가 축소)
> **결정 수**: 8 영역 (메타 Risk M-1~M-4 + 영역 1~9)

---

## 페이지 역할 정의

위험 지표 전담 페이지. **Performance 페이지가 수익 위주라면 Risk Metrics 는 위험 위주**.

**핵심 차별점**:
- vs Performance: 수익률 → 위험률
- 펀드 정체성 (Adaptive VolControl) 의 **위험 통제 narrative** 강화

---

## 페이지 영역 구조 (시선 흐름)

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트 + 토글 안내)
3. Risk Summary KPI 5개         (Vol / MDD / Beta / R² / TE)
4. Drawdown 시계열 + Recovery Time
5. VaR / CVaR 분포              (히스토그램 + 임계선)
6. Beta + R² + Tracking Error 시계열  (Rolling)
7. Risk Metrics 종합 표         (Image 4)
8. Tail Risk 분석               (Hill estimator + Jarque-Bera 옵션 C 축소)
9. Footer                       (Overview 동일)
```

---

## 영역별 와이어프레임

### 영역 1: Header — Overview 동일

→ `02_common.md` 의 `render_page_header()` 호출

---

### 영역 2: Sub-header

**결정사항** (R2-1):
- (a) Performance 패턴 동일

**텍스트 안**:
```
Risk Metrics (위험 지표)
펀드의 위험 통제 / 시장 노출 / tail risk 분석.
사이드바에서 기간 (FULL / TEST / HO) + 비교 벤치마크 (SPY/EW/IVW) 토글 가능.
```

→ `02_common.md` 의 `render_subheader()` 호출

---

### 영역 3: Risk Summary KPI 5개

**결정사항** (Risk M-3 + R3-1 ~ R3-4):
- KPI: (b) **Volatility / MDD / Beta / R² / Tracking Error** (위험 + 시장 노출)
- 표시 기간: (b) 사이드바 토글 반응
- 카드 디자인: (b) 단순 (Performance 동일)
- 벤치마크 delta: (b) 다중 토글 적용
- Hover tooltip: (a) 포함 (가상 투자자 친화)

**Tooltip 텍스트** (`lib/tooltips.py` 참조):
- Volatility / MDD / Beta / R² / Tracking Error 각 정의

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] `st.columns(5)` 반응형
- [ ] 위험 KPI 의 "vs SPY 위험 비교" narrative — 펀드 위험이 SPY 보다 낮음 → 좋음 표시 (↓ 화살표)
- [ ] EW / IVW 비교로 "단순 baseline 대비 위험 통제" narrative 강화
- [ ] Tooltip ⓘ 아이콘 + hover

---

### 영역 4: Drawdown 시계열 + Recovery Time

**결정사항** (R4-1 ~ R4-6):
- 차트: (c) DD + Duration/Recovery 막대 이중 차트
- Top N 강조: (a) Top 3 DD 강조 (라벨 + Recovery Time)
- Recovery Time: (c) 차트 마커 + 표 결합
- 비교: (b) 펀드 vs SPY (다중 토글 X — DD 차트 단순 유지)
- 인터랙션: 모두 채택
- 추가 표시: (a) 평균 DD 라인 + (c) Regime 배경색

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] Plotly 이중 차트 (`make_subplots(rows=2)`)
- [ ] DD 시계열 (Fund + SPY) + Top 3 마커 + 평균 DD 라인 + Regime 배경
- [ ] Recovery Time 표 + 막대
- [ ] Top 3 DD 식별: `pd.Series.nsmallest(3)`
- [ ] Recovery Time = DD bottom 이후 신고가 갱신까지의 개월 수
- [ ] 진행 중 DD (회복 안 됨) = "(진행 중)" 표시
- [ ] Q-Zoom: DD 클릭 → 해당 시기 detail expand

---

### 영역 5: VaR / CVaR 분포

**결정사항** (R5-1 ~ R5-6):
- 표시 형식: (c) 히스토그램 + KDE 오버레이 + 임계선
- 임계선 표시: (d) 모두 (수직선 + Tail 영역 강조 + 라벨 + 값)
- 일별/월별: (a) 탭 전환
- 비교: (b) 펀드 vs SPY 오버레이
- 인터랙션: 드래그 제외 모두 채택
- 추가 표시: (a) 정규분포 비교 라인 + (b) Statistics 보조 표

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] Plotly `go.Histogram` + `go.Scatter` (KDE)
- [ ] `add_vrect` 또는 fill_between (Tail 영역 강조)
- [ ] `add_vline` (VaR / CVaR 수직선 + 라벨)
- [ ] 정규분포 비교 라인 = `scipy.stats.norm.pdf(x, mean, std)`
- [ ] Statistics 보조 표 = `st.dataframe`
- [ ] Tab 전환 (월별 / 일별)

---

### 영역 6: Volatility + Sortino + Beta + R² + Tracking Error 시계열 (Rolling)

**결정사항** (R6-1 ~ R6-6 + 2026-05-10 보강):
- 표시 방식: (b) 5개 분리 차트 (위아래) — **Volatility / Sortino 추가**
- Rolling 윈도우: (b) 토글 (12m / 36m / 60m)
- 비교: (a) 펀드 only (Beta/R²/TE 본질적으로 vs SPY 메트릭)
- 추가 표시: (a) Beta=1 기준선만
- 인터랙션: 모두 채택
- **β 음수 처리** (R6-6): (c) R² 함께 표시 (신뢰성 평가)

**보강 근거 (Q-G, 2026-05-10)**: Overview Hero KPI sparkline 제거 (옵션 C)
시 Sortino / Volatility 의 시간 추이가 다른 페이지에 명시되어야 함. 기존 영역 6
은 Beta/R²/TE 만 → Volatility / Sortino 추가하여 5 메트릭 통합 시계열 페이지로 확장.

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] Plotly `make_subplots(rows=3)` (3개 분리 차트)
- [ ] Beta=1 기준선 = `add_hline(y=1)`
- [ ] β<0 시 R² 함께 표시 (annotation + tooltip 동적 조정)
- [ ] Rolling 산출 = `pandas.rolling(window).apply(...)` 활용
- [ ] Q-Zoom: 시기 클릭 → expand

---

### 영역 7: Risk Metrics 종합 표 (Image 4)

**결정사항** (R7-1 ~ R7-6):
- 메트릭 선정: (e) 카테고리별 그룹화 (~22-25 메트릭)
- 표시 형식: (b) 그룹화 표 + 확장 가능 (`st.expander`)
- 비교: (d) 다중 토글 + Diff column
- 정렬/그룹화: (a) 카테고리 헤더 그룹화
- 인터랙션: hover tooltip / 정렬 / CSV / 좋음나쁨 색상 (메트릭 클릭 보류)
- 추가 표시: (a) Footnote 메트릭 정의 일괄

**카테고리별 메트릭**:

| 카테고리 | 메트릭 |
|---|---|
| **Performance Returns** | Cumulative Return / CAGR / Arithmetic Mean (annualized) |
| **Risk-adjusted Return** | Sharpe / Sortino / Calmar / IR / M² |
| **Risk** | Volatility / MDD / Downside Dev / Historical VaR 5% / CVaR 5% |
| **Market Exposure** | Beta / R² / Correlation / Tracking Error |
| **Factor Analysis** | CAPM Alpha / FF5 Alpha (선택) |
| **Distribution** | Skewness / Kurtosis / Tail Ratio |

**시각화 예시**:

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

**구현 체크리스트**:
- [ ] `st.expander` (카테고리별 펼침/접기)
- [ ] 다중 토글 활성에 따라 컬럼 동적 추가 (SPY / EW / IVW + 각 Diff)
- [ ] 좋음/나쁨 색상 = `pandas.Styler.background_gradient` 또는 cell 단위
- [ ] CSV 다운로드 = `st.download_button`
- [ ] Footnote = `st.markdown` (LaTeX 수식 가능)

---

### 영역 8: Tail Risk 분석 (Hill Estimator)

**결정사항** (옵션 C 축소형 + Q-F2 추가 축소):
- ★ **균형 옵션 (B) Q-F2 적용**: Jarque-Bera 제거 → Methodology 영역 7 (LSTM 정당화) 통합
- **Hill estimator 만 유지** (옵션 C 축소)
- R8-1: (c) Hill Plot + ξ 카드
- R8-2: (a) auto plateau detection
- R8-3: (a) 일별 only (tail risk 정확도)
- R8-4: (b) 펀드 vs SPY
- R8-5: (c) Footnote + 학술 설명 박스

**학술 배경 (Hill 1975)**:

$$\hat{\xi}_k = \frac{1}{k}\sum_{i=1}^k \log(X_{(i)}) - \log(X_{(k+1)})$$

- ξ > 0: Fat tail (극단값 자주) — 주식 일반 ξ≈0.2-0.4
- ξ = 0: Exponential tail (정규분포)

**시각화 예시 (Q-F2 적용 — Jarque-Bera 제거)**:

```
[Risk Metrics 종합 표 (영역 7)]

┌─ Tail Risk Analysis (Hill estimator) ──────────────────┐
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
│ ※ Footnote:                                           │
│ - Hill, B.M. (1975). "A simple general approach to   │
│   inference about the tail of a distribution."        │
│   Annals of Statistics, 3, 1163-1174.                 │
│ - Embrechts, Klüppelberg, Mikosch (1997)              │
└────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Hill estimator 직접 구현 (`numpy` 기반):
  ```python
  def hill_estimator(returns, k):
      sorted_losses = np.sort(-returns[returns < 0])[::-1]  # 큰 손실 순
      log_losses = np.log(sorted_losses[:k+1])
      xi_hat = np.mean(log_losses[:k]) - log_losses[k]
      return xi_hat
  ```
- [ ] Hill Plot = Plotly `go.Scatter` (k vs ξ̂)
- [ ] Auto plateau detection = 슬라이딩 윈도우 표준편차 최소 영역
- [ ] 학술 설명 박스 + Footnote

**Jarque-Bera 는 Methodology 영역 7 로 이동** (Q-F2 적용)

---

### 영역 9: Footer — Overview 동일

→ `02_common.md` 의 `render_footer()` 호출

---

## 페이지 데이터 의존성

- monthly_panel.csv (월별 fund/SPY return)
- daily_returns.pkl (영역 5/8 일별 분포 / Hill estimator)
- results/mat_eq_eq_raw_pap.pkl (펀드 결과)

---

## 메트릭 (C-2 풀)

- Pool-3 위험: Volatility / MDD / Downside Dev / VaR / CVaR / Beta / R² / Tracking Error
- Pool-9 분포 통계: Skewness / Kurtosis / Tail Ratio
- 추가: **Hill estimator** (Risk Metrics 페이지 추가, Hill 1975)

---

## 인터랙션 / 토글 적용

| 영역 | 사이드바 토글 영향 | Q-Zoom |
|---|---|---|
| 영역 3 (KPI) | ✓ 기간 + 다중 벤치마크 delta | ✗ |
| 영역 4 (Drawdown) | ✓ 기간 (Fund vs SPY 단순) | ✓ DD 클릭 → expand |
| 영역 5 (VaR/CVaR) | Tab 전환 (월별/일별) | ✓ Tail 영역 클릭 → expand |
| 영역 6 (Beta/R²/TE) | ✓ Rolling 윈도우 토글 | ✓ 시기 클릭 |
| 영역 7 (종합 표) | ✓ 다중 벤치마크 + Diff column | ✗ |
| 영역 8 (Hill) | ✗ (Hill 자체) | ✗ |

---

## 페이지 구현 우선순위

- **Phase 2 (확장, 2-3주)**: Risk Metrics (Phase 2 의 핵심)

---

## 결과 / 함의

- **Performance 와 차별화**: 수익 → 위험 위주
- **학술 깊이 추가**: 영역 8 (Hill estimator) → 심사위원 어필
- **Q-F2 적용**: Jarque-Bera → Methodology 영역 7 (LSTM 정당화)
- 학술 인용: Hill (1975), Embrechts et al. (1997), McNeil et al. (2005)

---

[← 03_performance.md](03_performance.md) | [05_holdings.md →](05_holdings.md)
