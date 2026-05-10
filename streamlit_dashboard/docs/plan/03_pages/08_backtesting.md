# Backtesting 페이지 — 와이어프레임

> **관련 decisionlog**: `08_backtesting.md`
> **상태**: 확정 (★ 균형 옵션 (B) 적용 + Stress Test 제거 → 9 영역 → 7 영역)
> **결정 수**: 7 영역 (메타 BT M-1~M-4 + 영역 1~7)

---

## 페이지 역할 정의

백테스트 검증 전담 페이지 — Walk-forward / Regime 자세한 분석 / Sensitivity / Stress Test (제거).

**vs 다른 페이지**:
- Performance / Risk / Holdings / Sector Watch = **결과 위주**
- Methodology = **과정 / 토대 위주**
- **Backtesting = 검증 / 강건성 위주**

**핵심 narrative**:
- Performance 영역 7 의 Regime Heatmap **자세한 버전** (M-4 결정)
- Sub-events / 단기 위기 분석
- Sensitivity Test (4-slot 변경 시 결과)

---

## ★ 균형 옵션 (B) 적용 — 9 영역 → 7 영역 재정의

| 변경 | 이유 |
|---|---|
| **9 → 8 영역** (Q-D1-x5) | Walk-forward 검증 영역 제거 (이미 LSTM + BL 모두 walk-forward 적용됨) |
| **영역 6 (Walk-forward) 제거** (Q-D2-b) | LSTM + BL 모두 walk-forward 적용됨 → 별도 검증 불필요 |
| **영역 7 (Sensitivity) 단순화** (Q-D3-a) | 4-slot 비교만 (학술 깊이 ↓) |
| **영역 8 (Stress) 제거** (Q-Stress A) | 사용자 지적: 정확한 시뮬레이션 구현 매우 어려움 |
| **KPI 5개 재정의** | "Walk-forward Sortino" 제거 → "Regime 일관성" 추가 |

---

## 페이지 영역 구조 (시선 흐름)

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트)
3. Backtest Summary KPI 5개     (TEST/HO Gap / Sensitivity / 4-slot / Recovery / Regime 일관성)
4. Regime 메트릭 자세한 비교    (Performance 영역 7 자세한 버전, 12 메트릭)
5. Sub-events 분석              (4 위기 — 2018Q4 / COVID / 2022 Bear / 2024 IT Rotation)
6. Sensitivity Test (단순)      (Top 10 config + 우리 펀드 강조)
7. Footer                       (Overview 동일)
```

---

## 영역별 와이어프레임

### 영역 1: Header — Overview 동일

→ `02_common.md` 의 `render_page_header()` 호출

---

### 영역 2: Sub-header

**결정사항** (BT2-1):
- (b) 균형 narrative — 모델 구조 정확화 반영 (TEST/HOLD_OUT 의 정확한 의미)

**텍스트 안**:
```
Backtesting (백테스트 검증)
Regime / Sub-events / Sensitivity 분석.
TEST 평가 168m + HOLD_OUT 평가 24m walk-forward 결과의
깊이 있는 검증.
```

→ `02_common.md` 의 `render_subheader()` 호출

---

### 영역 3: Backtest Summary KPI 5개

**결정사항** (BT M-3 + 균형 옵션 B 재정의 + BT3-2 ~ BT3-4):
- 5 KPI 재정의:
  1. **TEST/HO Gap** — 학습편향 검증 (TEST sortino - HO sortino, 작을수록 robust)
  2. **Sensitivity Robustness** — 4-slot 변경 시 Top 1 일관성 (%)
  3. **4-slot Robustness Score** — 156 config 의 sortino 분산
  4. **Avg Recovery Time** — 영역 5 Sub-events 의 Recovery Time 평균
  5. **Regime 일관성 Score** — R1/R2/R3/HO Sortino 일관성 (mean / std)
- 표시 기간: (a) FULL only (검증 메트릭 특성)
- 카드 디자인: (c) 단순 + 좋음/나쁨 색상 (🟢🟧🔴)
- Hover tooltip: (a) 포함

**색상 기준**:
- TEST/HO Gap > 1.0 → 🔴 (학습편향 의심)
- Sensitivity Robustness < 50% → 🟧 (Top 1 변동 큼)
- 4-slot Robustness > 0.30σ → 🟢 (안정)
- Avg Recovery Time < 6m → 🟢
- Regime 일관성 > 0.50 → 🟢

**Tooltip 텍스트**:
- TEST/HO Gap: "TEST sortino - HO sortino. 작을수록 학습편향 ↓ (robust)"
- Sensitivity Robustness: "4-slot 일부 변경 시 Top 1 유지 비율"
- 4-slot Robustness Score: "156 config 의 sortino 분산 (mean / std). 높을수록 robust"
- Avg Recovery Time: "Sub-events 회복 시간 평균"
- Regime 일관성: "R1/R2/R3/HO Sortino 의 일관성 (mean / std)"

**시각화 예시**:

```
[Header — Overview 동일]

┌─ ℹ️ Backtesting (백테스트 검증) ────────────────────────┐
│ Regime / Sub-events / Sensitivity 분석.    │
│ TEST 평가 168m + HOLD_OUT 평가 24m walk-forward          │
│ 결과의 깊이 있는 검증.                                    │
└──────────────────────────────────────────────────────────┘

[Backtest Summary KPI 5개 — FULL only]
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│TEST/HO   │Sensitivity│4-slot   │Avg       │Regime    │
│Gap       │Robustness│ Robust   │Recovery  │일관성    │
│ X.XX     │ XX%      │ X.XX σ   │ X 개월   │ X.XX     │
│ 🟧 주의 │ 🟢 안정  │ 🟢 robust│ 🟢 빠름  │ 🟢 일관  │
│ⓘGap     │ⓘSensit. │ⓘ4-slot   │ⓘRecovery  │ⓘRegime    │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

**구현 체크리스트**:
- [ ] `st.columns(5)` 반응형
- [ ] 색상 코딩 = `lib/colors.py` (🟢🟧🔴 dictionary)
- [ ] 색상 임계값 = decisionlog 부록 또는 코드 dictionary 정의
- [ ] Tooltip ⓘ 아이콘 + hover

---

### 영역 4: Regime 메트릭 자세한 비교

**결정사항** (BT4-1 ~ BT4-5):
- 표시 형식: (c) 표 + 보조 시각화 (Regime별 Sortino 막대)
- 메트릭: (b) 풍부 12개 — CAGR / Sortino / Sharpe / MDD / Volatility / Active Return / IR / Win Rate / Calmar / VaR 5% / CVaR 5% / Beta
- 비교: (b) 다중 토글 (Fund / SPY / EW / IVW)
- 인터랙션: 모두 채택
- 추가 분석: (d) 모두 (통계 + Active Return 컬럼 + Best/Worst 강조)

**vs Performance 영역 7**:
- Performance = 4 메트릭 Heatmap (간단)
- Backtesting = **12 메트릭 표 + 통계 + 시각 (자세한)**

**시각화 예시**:

```
[Backtest Summary KPI (영역 3)]

┌─ Regime 메트릭 자세한 비교 ──────────────────────────────┐
│ [☑Fund ☑SPY ☐EW ☐IVW]                       [⬇ CSV]     │
│                                                          │
│ Metric        | R1     | R2     | R3     | HO    | FULL │
│ CAGR          | +X%    | +X%    | +X%    | +X%   | +X%  │
│ Sortino       | X.X 🟢 | X.X 🟢🌟│ X.X 🟢 | 0.69  | X.X  │
│ Sharpe        | X.X    | X.X    | X.X    | X.X   | X.X  │
│ MDD           | -X%    | -X%    | -X% 🔴 | -X%   | -X%  │
│ Volatility    | XX%    | XX%    | XX%    | 10.3% | XX%  │
│ Active Return | +X%    | +X%    | +X% 🟢 | -12.9%🔴 | -X% │
│ IR            | X.X    | X.X    | X.X    | -X.X  | X.X  │
│ Win Rate      | XX%    | XX%    | XX%    | XX%   | XX%  │
│ Calmar        | X.X    | X.X    | X.X    | X.X   | X.X  │
│ VaR 5%        | -X%    | -X%    | -X%    | -X%   | -X%  │
│ CVaR 5%       | -X%    | -X%    | -X%    | -X%   | -X%  │
│ Beta          | X.X    | X.X    | X.X    | X.X   | X.X  │
│                                                          │
│ ─── Regime 통계 ───                                      │
│ Sortino mean / std / IR (일관성)                         │
│                                                          │
│ ★ Best Regime: R2 (Sortino X.X)                          │
│ 🔴 Worst Regime: HO (Sortino 0.69)                       │
│                                                          │
│ ─── Regime별 Sortino 막대 ───                            │
│ R1 │██████ X.X                                           │
│ R2 │████████ X.X                                         │
│ R3 │██████ X.X                                           │
│ HO │██ 0.69                                              │
│                                                          │
│ Hover: 메트릭 정의 / Click: Regime → Sub-events          │
└──────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] 12 메트릭 × 5 Regime 표 = `st.dataframe`
- [ ] Best / Worst 강조 = cell formatting (🟢🌟 / 🔴)
- [ ] Sortino IR (일관성 score) = Regime 통계 보강
- [ ] Regime별 Sortino 막대 = Plotly `go.Bar`
- [ ] 다중 토글 활성 시 컬럼 추가
- [ ] Regime 행 Click → 영역 5 (Sub-events) navigation
- [ ] CSV 다운로드

---

### 영역 5: Sub-events 분석 (단기 위기)

**결정사항** (BT5-1 ~ BT5-5):
- 표시 형식: (d) 표 + Timeline 둘 다
- Sub-events: (b) 핵심 4개:
  1. **2018 Q4 Sell-off** (2018-09 ~ 2018-12)
  2. **COVID-19 Crash** (2020-02 ~ 2020-03)
  3. **2022 Inflation Bear** (2022-01 ~ 2022-10)
  4. **2024-12 Sector Rotation** (2024-12)
- 메트릭: (b) 표준 (Fund / SPY / Active Return / MDD / Recovery Time)
- 비교: (a) Fund vs SPY only
- 인터랙션: 모두 채택

**시각화 예시**:

```
[Regime 자세한 비교 (영역 4)]

┌─ Sub-events 분석 (단기 위기) ───────────────────────────┐
│                                                         │
│ ┌─ Timeline (위기 마커) ────────────────────────────┐  │
│ │ 2010 ──── 2014 ──── 2018 ──── 2022 ──── 2025      │  │
│ │                       ▼2018Q4 ▼COVID  ▼2022 Bear  │  │
│ │                       (-X%)   (-X%)   (-X%)       │  │
│ │                                              ▼2024│  │
│ │                                              (-X%)│  │
│ └─────────────────────────────────────────────────┘  │
│                                                         │
│ ┌─ 위기별 표 ───────────────────────────────────────┐  │
│ │ Event       | Period      | Fund | SPY  | Active| MDD| Recv│
│ │ 2018 Q4     | 09-18~12-18 | -X%  | -19% | +X%🟢 | -X%| Xm  │
│ │ COVID-19    | 02-20~03-20 | -X%  | -34% | +X%🟢 | -X%| Xm  │
│ │ 2022 Bear   | 01-22~10-22 | -X%  | -25% | +X%🟢 | -X%| Xm  │
│ │ 2024 IT Rot | 12-24       | -X%  | -8%  | -X%🔴 | -X%| Xm  │
│ │                                                    │  │
│ │ ★ Best 방어: COVID-19 (Active +X%)                  │  │
│ │ 🔴 Worst: 2024 IT Rotation (Active -X%)            │  │
│ └─────────────────────────────────────────────────┘  │
│                                                         │
│ Hover: 위기 detail / Click: 해당 시기 차트 expand       │
│ 정렬: [시기순 ▼ / 손실 크기순]                         │
│ [⬇ CSV]                                                │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] Timeline = Plotly `go.Scatter` (시점 + annotation 마커)
- [ ] 위기별 표 = `st.dataframe`
- [ ] Best / Worst 강조 = cell formatting
- [ ] Recovery Time = 위기 후 회복 narrative (마케팅 친화)
- [ ] 2024 IT Rotation = HO 정당화 narrative 와 직접 연결
- [ ] Q-Zoom: Event 클릭 → 해당 시기 차트 expand

---

### 영역 6: Sensitivity Test (단순 — D3-a)

**결정사항** (BT6-1 ~ BT6-5):
- 표시 형식: (d) 표 + 막대 둘 다
- 비교 대상: (d) Top 10 + 우리 펀드 강조
- 메트릭: (b) Sortino + CAGR + MDD
- 인터랙션: hover + Methodology 클릭 + 정렬 (CSV 제외 — 영역 4 와 중복)
- 추가 표시: (a) 우리 펀드 강조 + (b) Top 1-Top 10 차이 시각

**시각화 예시**:

```
[Sub-events 분석 (영역 5)]

┌─ Sensitivity Test (4-slot 단순 비교) ───────────────────┐
│                                                         │
│ ┌─ Top 10 config 표 ───────────────────────────────┐   │
│ │ Rank | Config              | Sortino | CAGR | MDD │  │
│ │ 1    | mat_eq_eq_raw_pap ★ | X.XX    | +X%  | -X% │ ← 우리 펀드│
│ │ 2    | mat_eq_mcap_lam_he  | X.XX    | +X%  | -X% │  │
│ │ 3    | mat_mcap_eq_raw_pap | X.XX    | +X%  | -X% │  │
│ │ ...                                                │  │
│ │ 10   | rp_eq_eq_lam_pap    | X.XX    | +X%  | -X% │  │
│ │                                                    │  │
│ │ Hover: config detail / Click: Methodology 영역 4   │  │
│ │ 정렬: Sortino / CAGR / MDD 컬럼 헤더 클릭          │  │
│ └────────────────────────────────────────────────────┘  │
│                                                         │
│ ┌─ Top 1 - Top 10 Sortino 차이 ────────────────────┐  │
│ │ Top 1 ★ ████████ X.XX                             │  │
│ │ Top 2  ███████  X.XX                              │  │
│ │ Top 3  ███████  X.XX                              │  │
│ │ Top 5  ██████   X.XX                              │  │
│ │ Top 10 █████    X.XX                              │  │
│ │ → Top 1-10 차이 작음 = robust                     │  │
│ └────────────────────────────────────────────────────┘ │
│                                                         │
│ ✓ 우리 펀드 (Top 1) 와 Top 10 차이 X.XX → robust       │
└─────────────────────────────────────────────────────────┘
```

**구현 체크리스트**:
- [ ] 156 config 의 Sortino 정렬 → Top 10 추출
- [ ] `st.dataframe` (Top 10 표)
- [ ] 우리 펀드 (mat_eq_eq_raw_pap) ★ 마커 + Cobalt Blue accent
- [ ] Plotly `go.Bar` (Top 1-10 Sortino 차이 막대)
- [ ] Hover: config detail
- [ ] Click: Methodology 영역 4 navigation (`st.switch_page()`)
- [ ] 데이터: `final/data/results/` 의 다른 155 config pkl (D-1 참조)

---

### 영역 7: Footer — Overview 동일

→ `02_common.md` 의 `render_footer()` 호출

---

## ★ Stress Test 영역 — 제거 결정 (Q-Stress A)

**제거 근거**:
1. **사용자 지적**: "Stress Test 의 학술적/실무적 정확한 시뮬레이션 구현 매우 어려움"
2. **학술 정직성** — 단순 β 기반 시뮬레이션은 fat tail / 상관관계 변화 / 유동성 미반영
3. **Methodology 영역 7 (Jarque-Bera fat tail) narrative 와 모순** — 정규분포 가정 stress test 부적절
4. **Sub-events (영역 5) + Sensitivity (영역 6) 로 위기 안정성 narrative 충분**
5. **"감추지 않는 펀드"** 정신 — 못 하는 것을 하는 척 X

→ 영역 7 = Footer (Overview 동일) 로 단순화

---

## 페이지 데이터 의존성

- results/mat_eq_eq_raw_pap.pkl (펀드 결과 — 영역 4, 5)
- final/data/results/ (다른 155 config pkl — 영역 6 Sensitivity)
- monthly_panel.csv (SPY 비교)
- daily_returns.pkl (영역 5 위기 시점 detail)

---

## 메트릭 (C-2 풀)

- 영역 4 (12 메트릭): CAGR / Sortino / Sharpe / MDD / Volatility / Active Return / IR / Win Rate / Calmar / VaR 5% / CVaR 5% / Beta
- Pool-7 Regime별 성과: Sortino R1/R2/R3/HO + IR (일관성)
- 영역 6 (Sensitivity): Sortino / CAGR / MDD 3 메트릭

---

## 인터랙션 / 토글 적용

| 영역 | 사이드바 토글 영향 | Q-Zoom |
|---|---|---|
| 영역 3 (KPI) | ✗ (FULL only) | ✗ |
| 영역 4 (Regime 자세) | ✓ 다중 벤치마크 (컬럼 추가) | Regime 행 클릭 → Sub-events |
| 영역 5 (Sub-events) | ✗ | ✓ Event 클릭 → 차트 expand |
| 영역 6 (Sensitivity) | ✗ | config 클릭 → Methodology 영역 4 |

---

## 페이지 구현 우선순위

- **Phase 3 (검증, 1-2주)**: Backtesting 페이지 (학술 정직성 강화)

---

## 결과 / 함의

- **균형 옵션 (B) 적용 완료**:
  - Walk-forward 검증 영역 제거 (LSTM + BL 자체 적용)
  - Stress Test 영역 제거 (구현 정확성 어려움 + 학술 정직성)
  - Sensitivity Test 단순화
  - KPI 재정의
- **인터랙션 일관성 원칙** 적용:
  - 다중 토글 (영역 4)
  - Q-Zoom (영역 5 Event 클릭)
  - config 클릭 → Methodology 영역 4 navigation (영역 6)
  - Regime 행 클릭 → Sub-events navigation (영역 4 → 영역 5)
- **HO 정당화 narrative 보완**:
  - 영역 5 (2024 IT Rotation 위기) — Sector Watch 영역 8 와 narrative 연결
  - KPI Avg Recovery Time — 위기 후 회복 narrative

---

[← 07_methodology.md](07_methodology.md) | [09_about.md →](09_about.md)
