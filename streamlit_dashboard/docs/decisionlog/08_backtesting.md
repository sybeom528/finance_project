# C-1-7. Backtesting 페이지

> **파일**: `08_backtesting.md`
> **결정 시점**: 2026-05-10
> **상태**: 확정 (페이지 메타 BT M-1~M-4 + 영역 1~7, ★ 균형 옵션 (B) 적용 + Stress Test 제거)
> **포함**: 페이지 메타 결정 / 균형 옵션 (B) narrative (9 → 8 영역 재정의) / Stress Test 제거 narrative (8 → 7) / Sub-header / Backtest Summary KPI 5개 / Regime 메트릭 자세한 비교 / Sub-events 분석 (단기 위기) / Sensitivity Test (단순 — D3-a) / 영역 7 (Stress Test) — 제거

---

## C-1-7. Backtesting 페이지

> **상태**: 진행 중 (메타 결정 확정 / 영역 1, 9 는 Overview 동일 / 2-8 미정)

### Backtesting 페이지 통합 배경 (Context)

백테스트 검증 전담 페이지 — Walk-forward / Regime 자세한 분석 / Sensitivity / Stress Test.

**vs 다른 페이지**:
- Performance / Risk / Holdings / Sector Watch = **결과 위주**
- Methodology = **과정 / 토대 위주**
- **Backtesting = 검증 / 강건성 위주**

**핵심 narrative**:
- Performance 영역 7 의 Regime Heatmap **자세한 버전** (M-4 결정)
- Sub-events / 단기 위기 분석
- Sensitivity Test (4-slot 변경 시 결과)
- Walk-forward OOS 검증
- Methodology 영역 8 한계 (Walk-forward 미적용) → **여기서 일부 적용**

### 추가 결정 (BT-Q1, 2026-05-11): 메트릭 정합성 + 156 config 데이터 포함

**배경**: Phase 3.2 구현 (Phase 2.1/2.2/2.3 동일 패턴 — 신모델 mat_eq_mcap_raw_he 기준).

**Final 정합 메트릭** (subperiod 함수 사용):
- Regime별 CAGR / Sortino / Sharpe / MDD = `calc_*_subperiod` 함수 (master_table 1:1)
- Volatility / Beta / VaR / CVaR / Win Rate = `calc_*` 함수 (compute_metrics 정합)

**학술 표준 정의 (출처 명시)**:

| 메트릭 | 정의 | 출처 |
|---|---|---|
| **TEST/HO Gap** | Sortino_TEST − Sortino_HO | 학습편향 검증 표준 |
| **Regime 일관성 Score** | Sortino mean / std (R1/R2/R3/HO) | Reverse CV 표준 |
| **Sub-events Recovery** | 위기 시작 직전 peak → 종료 후 신고가 회복 개월 수 | Magdon-Ismail & Atiya (2004) |
| **Active Return** | Fund 누적 - SPY 누적 (위기 기간) | Active Management 표준 |
| **IR (Information Ratio)** | Active Return / Tracking Error | Grinold & Kahn (1999) |
| **Calmar** | CAGR / |MDD| | Young (1991) |
| **156 config Sensitivity** | Sortino 분포 mean/std (= 1/CV) | Robustness 표준 |

**데이터 의존성 결정 — 156 config 직접 git 포함**:
- 156 pkl 합 ~145MB → GitHub 권장 한계 (1GB) 내
- CSV summary 캐시 vs 156 pkl 직접 포함 비교 → **직접 포함 채택**
- 사유: build 스크립트 불필요 + 메트릭 추가 자유 + pkl 내부 데이터 (weights/comp) 활용 가능
- `.gitignore` 에 `!streamlit_dashboard/data/results/*.pkl` negation
- `lib/data_loader.list_available_configs()` 추가 — main 에서도 156 config 접근 가능

**검증 결과** (신모델 mat_eq_mcap_raw_he):
- TEST/HO Gap: 1.265
- Regime 일관성 score: 2.73 (R1=2.23/R2=2.07/R3=1.86/HO=0.81)
- Avg Recovery: 2.5 개월 (2024 IT Rotation = 7m, HO narrative)
- 156 config Sortino Rank: **5/159 (Top 3.1%)** — 매우 우수 + robust
- 156 config Sortino mean=1.59 / std=0.18 → Robustness=8.71 (안정)

**결과 / 함의**:
- `lib/metric_calculators.py` 보강 — 5 신규 함수 + REGIME_METRICS / SUB_EVENTS / REGIME_PERIODS 외부 dict (변경 용이)
- `lib/backtesting_charts.py` 신규 — 4 영역 함수
- 모든 결과 final 정합 또는 학술 출처 명시 → 정직성 확보
- **초안** 으로 작성됨 — 향후 영역 추가 / 제거 / 메트릭 변경 용이한 구조 (모듈화)

### 페이지 메타 결정 (BT M-1 ~ M-4)

#### BT M-1. 영역 개수

**검토된 옵션**:
- (a) 압축 6 영역
- (b) 표준 7 영역
- (c) 풍부 9 영역

**결정**: (c) 풍부 9 영역

**근거**:
1. 모든 검증 측면 (Regime / Sub-events / Walk-forward / Sensitivity / Stress) 다룸
2. **Methodology 영역 8 한계** (Walk-forward 미적용) **와 직접 연결** → Backtesting 페이지에서 일부 적용
3. 학술 정직성 ★★★

#### BT M-2. Sub-header

**결정**: (a) 포함

**근거**: 모든 페이지 일관 — 인터랙션 일관성 원칙

#### BT M-3. Backtest Summary KPI

**검토된 옵션**:
- (a) 검증 메트릭만
- (b) 성과 요약
- (c) 종합
- (d) 자유

**결정**: (c) 종합

**5개 KPI**:
1. **Walk-forward Sortino** (Rolling Window 평균)
2. **In-sample / OOS Gap** (학습편향 검증 — TEST sortino - HO sortino)
3. **Sensitivity Robustness** (4-slot 변경 시 결과 안정성)
4. **4-slot Robustness Score** (Top N config 일관성)
5. **Stress Test Survival** (위기 시나리오 생존율)

**근거**:
1. **검증 다각도** — Walk-forward + Sensitivity + Stress Test 모두 KPI 화
2. (a) 검증만 5개는 KPI 가독성 ↓
3. (b) 성과 요약은 Performance 페이지와 중복

#### BT M-4. Walk-forward 방식

**검토된 옵션**:
- (a) 단순 적용 (1번 OOS, 168m/24m)
- (b) Rolling Window Walk-forward (월별)
- (c) Expanding Window Walk-forward
- (d) 둘 다

**결정**: (b) Rolling Window Walk-forward (월별)

**근거**:
1. **학술 표준** (Lopez de Prado 2018)
2. **Rolling Window** = 일관된 윈도우 크기 (예: 60m 학습 + 12m OOS, 월별 슬라이드)
3. (c) Expanding Window 는 학술 정직성 ↑↑ but 구현 복잡 ↑↑
4. (d) 둘 다 = 정보 과부하
5. **Methodology 영역 8 한계 (Walk-forward 미적용) 의 일부 해결** — 학술 정직성 narrative 강화

**구현 안**:
- 학습 윈도우: 60개월 (5년)
- OOS 윈도우: 12개월 (1년)
- 슬라이드: 월별 (총 ~120개 OOS 결과)
- 평균 Sortino + 표준편차

### ★ 균형 옵션 (B) 적용 — Backtesting 페이지 9 → 8 영역 재정의

**모델 구조 정확화 + 균형 옵션 (B)** 반영하여 페이지 메타 재정의:

| 변경 | 이유 |
|---|---|
| **9 → 8 영역** (Q-D1-x5) | Walk-forward 검증 영역 제거 (이미 모델에 적용됨) |
| **영역 6 (Walk-forward) 제거** (Q-D2-b) | LSTM + BL 모두 walk-forward 적용됨 → 별도 검증 불필요 |
| **영역 7 (Sensitivity) 단순화** (Q-D3-a) | 4-slot 비교만 (학술 깊이 ↓) |
| **영역 8 (Stress) 단순화** (Q-D4-a) | 가상 시나리오만 |
| **KPI 5개 재정의** | "Walk-forward Sortino" 제거 → "Regime 일관성" 추가 |
| **BT M-4 결정 무효화** | 영역 6 자체 제거로 walk-forward 방식 결정 무관 |

### ★ Stress Test 영역 추가 제거 (Q-Stress A) — 8 → 7 영역

**사용자 지적**: "Stress Test 의 학술적/실무적 정확한 시뮬레이션 구현 매우 어려움"

**제거 근거**:
1. **학술 정직성** — 단순 β 기반 시뮬레이션은 fat tail / 상관관계 변화 / 유동성 미반영
2. **Methodology 영역 7 (Jarque-Bera fat tail) narrative 와 모순** — 정규분포 가정 stress test 부적절
3. **Sub-events (영역 5) + Sensitivity (영역 6) 로 위기 안정성 narrative 충분**
4. **"감추지 않는 펀드"** 정신 — 못 하는 것을 하는 척 X
5. Q-D1 사용자 의도 (9→7) 와 부합

### Backtesting 페이지 7 영역 구조 (최종)

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트)
3. Backtest Summary KPI 5개     (TEST/HO Gap / Sensitivity / 4-slot / Recovery Time / Regime 일관성)
4. Regime 메트릭 자세한 비교    (Performance 영역 7 자세한 버전)
5. Sub-events 분석              (4 위기 — 2018Q4 / COVID / 2022 Bear / 2024 IT Rotation)
6. Sensitivity Test (단순)      (4-slot 변경 결과 비교)
7. Footer                       (Overview 동일)
```

= 7 영역 (Header + 본문 4 + Footer)

### KPI 5개 재정의 (Stress Survival → Avg Recovery Time)

1. **TEST/HO Gap** — 학습편향 검증 (TEST sortino - HO sortino, 작을수록 robust)
2. **Sensitivity Robustness** — 4-slot 변경 시 Top 1 일관성 (%)
3. **4-slot Robustness Score** — 전체 156 config 의 분산
4. **Avg Recovery Time** — 영역 5 Sub-events 의 Recovery Time 평균 (위기 후 회복 속도)
5. **Regime 일관성 Score** — R1/R2/R3/HO Sortino 의 일관성 (mean / std)

### 영역 2: Sub-header — 확정

#### 결정 항목 BT2-1: 텍스트 내용

**검토된 옵션**:
- (a) 단순 검증 narrative
- (b) 균형 narrative (TEST/HO walk-forward 명시)
- (c) 학술 정직성 강조 (Lopez de Prado 인용)

**결정**: (b) 균형 narrative

**근거**:
1. **모델 구조 정확화 반영** — TEST/HOLD_OUT 의 정확한 의미 (walk-forward 결과 평가) 명시
2. (a) 단순 narrative 는 검증 의미 ↓
3. (c) 학술 인용은 Methodology 영역 8 narrative 와 중복

**텍스트 안** (Q-Stress A — Stress Test 영역 제거 반영):
```
Backtesting (백테스트 검증)
Regime / Sub-events / Sensitivity 분석.
TEST 평가 168m + HOLD_OUT 평가 24m walk-forward 결과의
깊이 있는 검증.
```

---

### 영역 3: Backtest Summary KPI 5개 — 확정

#### 영역 3 통합 배경 (Context)

KPI 5개 사전 확정 (메타 결정 단계 BT M-3 + 균형 옵션 B 재정의 + Q-Stress A 적용):
1. **TEST/HO Gap** — 학습편향 검증 (TEST sortino - HO sortino)
2. **Sensitivity Robustness** — 4-slot 변경 시 Top 1 일관성 (%)
3. **4-slot Robustness Score** — 156 config 분산
4. **Avg Recovery Time** — 영역 5 Sub-events Recovery Time 평균 (Q-Stress A: Stress Test Survival 대체)
5. **Regime 일관성 Score** — R1/R2/R3/HO Sortino 일관성

#### 결정 항목 BT3-2: 표시 기간

**결정**: (a) FULL only

**근거**:
1. **검증 메트릭 특성** — 전체 기간 평가가 표준
2. (b) 사이드바 토글은 검증 의미 모호 (TEST/HO Gap 은 자동으로 두 기간 비교)

#### 결정 항목 BT3-3: 카드 디자인

**결정**: (c) 단순 + 좋음/나쁨 색상

**근거**:
1. **균형 (B) 적용** — 일관 + 검증 시각
2. **좋음/나쁨 색상** = 🟢 (좋음) / 🟧 (주의) / 🔴 (나쁨)
3. (b) ✓/⚠/✗ 마커는 강조 ↑↑ → 부정적 인상 가능

**색상 기준 안**:
- TEST/HO Gap > 1.0 → 🔴 (학습편향 의심)
- Sensitivity Robustness < 50% → 🟧 (Top 1 변동 큼)
- 4-slot Robustness > 0.30σ → 🟢 (안정)
- Avg Recovery Time < 6m → 🟢 (Q-Stress A: Stress Survival 대체)
- Regime 일관성 > 0.50 → 🟢

#### 결정 항목 BT3-4: Hover tooltip

**결정**: (a) 포함

**근거**:
1. 일관성 — 모든 페이지 KPI tooltip 표준
2. **검증 메트릭** = 가상 투자자 친화 필수 (TEST/HO Gap 등 어려움)

**Tooltip 텍스트 안**:
- TEST/HO Gap: "TEST sortino - HO sortino. 작을수록 학습편향 ↓ (robust)"
- Sensitivity Robustness: "4-slot 일부 변경 시 Top 1 유지 비율"
- 4-slot Robustness Score: "64 config 의 sortino 분산 (mean / std). 높을수록 robust"
- Avg Recovery Time: "Sub-events 회복 시간 평균. 짧을수록 위기 후 회복 빠름" (Q-Stress A: Stress Survival 대체)
- Regime 일관성: "R1/R2/R3/HO Sortino 의 일관성 (mean / std)"

#### 시각화 예시 (확정 사항 조합)

```
[Header — Overview 동일]

┌─ ℹ️ Backtesting (백테스트 검증) ────────────────────────┐
│ Regime / Sub-events / Sensitivity / Stress Test 분석.    │
│ TEST 평가 168m + HOLD_OUT 평가 24m walk-forward          │
│ 결과의 깊이 있는 검증.                                    │
└──────────────────────────────────────────────────────────┘

[Backtest Summary KPI 5개 — FULL only]
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│TEST/HO   │Sensitivity│4-slot   │Stress    │Regime    │
│Gap       │Robustness│ Robust   │ Survival │일관성    │
│ X.XX     │ XX%      │ X.XX σ   │ X/X 통과 │ X.XX     │
│ 🟧 주의 │ 🟢 안정  │ 🟢 robust│ 🟧 부분  │ 🟢 일관  │
│ⓘGap     │ⓘSensit. │ⓘ4-slot   │ⓘStress    │ⓘRegime    │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

#### 결과 / 함의

- TEST/HO Gap = 학습편향의 정량 지표 (selection bias 와 유사하지만 톤 부드럽게)
- 4-slot Robustness Score = 156 config 의 sortino 분포 통계
- Regime 일관성 Score = Performance 영역 7 (Regime Heatmap) 의 통계 종합
- 색상 코딩 임계값 = decisionlog 부록 또는 코드 dictionary 정의

### 영역 4: Regime 메트릭 자세한 비교 — 확정

#### 영역 4 통합 배경 (Context)

Performance 영역 7 의 간단 Heatmap (4 메트릭) 의 자세한 버전 — 12+ 메트릭 + 통계 + Best/Worst 분석.

**vs Performance 영역 7**:
- Performance = 4 메트릭 Heatmap (간단)
- Backtesting = 12 메트릭 표 + 통계 + 시각 (자세한)

#### 결정 항목 BT4-1: 표시 형식

**결정**: (c) 표 + 보조 시각화 (Regime별 Sortino 막대)

**근거**:
1. **표** = 정확 12 메트릭 한눈
2. **보조 막대** = Regime별 핵심 (Sortino) 시각
3. (b) Heatmap 은 Performance 영역 7 와 중복

#### 결정 항목 BT4-2: 메트릭

**결정**: (b) 풍부 12개

**12개 메트릭**:
1. CAGR
2. Sortino
3. Sharpe
4. MDD
5. Volatility
6. Active Return
7. IR (Information Ratio)
8. Win Rate
9. Calmar
10. VaR 5%
11. CVaR 5%
12. Beta

**근거**:
1. **자세한 분석** — Performance 영역 7 (4 메트릭) 와 차별화
2. (a) 8개는 위험조정 메트릭 부족
3. (c) 15+ 은 정보 과부하

#### 결정 항목 BT4-3: 비교

**결정**: (b) 다중 토글 (Fund / SPY / EW / IVW)

**근거**:
1. **인터랙션 일관성 원칙** — 다른 페이지 다중 토글 일관
2. 자유 탐색 모드 부합
3. (a) Fund+SPY Tab 만은 EW/IVW 비교 손실

#### 결정 항목 BT4-4: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- Regime 행 클릭 → 영역 5 (Sub-events) navigation ✓
- 컬럼 정렬 ✓
- CSV 다운로드 ✓
- 색상 코딩 (양수/음수) ✓

#### 결정 항목 BT4-5: 추가 분석

**결정**: (d) 모두 (통계 + Active Return 컬럼 + Best/Worst 강조)

**근거**:
1. **Regime 통계** = mean / std / Sortino IR (일관성 score)
2. **Best / Worst Regime 강조** = 시각 narrative
3. (a/b/c) 단독은 정보 ↓

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- 다중 토글 (SPY/EW/IVW) 활성 시 컬럼 추가 (각 벤치마크별)
- Sortino IR (일관성 score) = Regime 통계 보강
- Best / Worst 강조 = HO 의 부진 명확 노출 (학술 정직성)
- Regime 행 Click → 영역 5 (Sub-events) 의 해당 Regime 위기 detail

### 영역 5: Sub-events 분석 (단기 위기) — 확정

#### 영역 5 통합 배경 (Context)

우리 데이터 (2010-2025) 시기의 단기 위기 이벤트별 펀드 vs 벤치마크 비교. 펀드의 위기 안정성 narrative.

**vs Sector Watch 영역 7/8**:
- Sector Watch = sector 단위 분석
- Backtesting Sub-events = **위기 이벤트 단위 분석**

#### 결정 항목 BT5-1: 표시 형식

**결정**: (d) 표 + Timeline 둘 다

**근거**:
1. **표** = 정확 수치
2. **Timeline** = 시간 컨텍스트 (위기 시기 시각)
3. (a) 표만은 시간 narrative 손실
4. (b) Tornado 만은 정확 수치 손실

#### 결정 항목 BT5-2: Sub-events 선정

**결정**: (b) 핵심 4개

**4개 위기**:
1. **2018 Q4 Sell-off** (2018-09 ~ 2018-12)
2. **COVID-19 Crash** (2020-02 ~ 2020-03)
3. **2022 Inflation Bear** (2022-01 ~ 2022-10)
4. **2024-12 Sector Rotation** (2024-12)

**근거**:
1. 핵심 위기 균형 (각 Regime 1-2개)
2. (a) 6개 모두는 정보 과부하 + Eurozone/China-Oil 영향 ↓
3. (c) 3개는 2018 Q4 누락 → R2 위기 미반영

#### 결정 항목 BT5-3: 메트릭

**결정**: (b) 표준 (Fund / SPY / Active Return / MDD / Recovery Time)

**근거**:
1. **5 메트릭** = 위기 분석 표준
2. **Recovery Time** = 위기 후 회복 narrative (펀드 안정성)
3. (a) 핵심 3개는 MDD/Recovery 누락
4. (c) 풍부는 정보 과부하

#### 결정 항목 BT5-4: 비교

**결정**: (a) Fund vs SPY only

**근거**:
1. **위기 시기 단순 비교** = 직관적
2. 다중 토글은 영역 4 (Regime 자세한 비교) 에서 다룸
3. 위기 시기 narrative 는 시장 (SPY) 대비 직접 비교가 핵심

#### 결정 항목 BT5-5: 인터랙션

**결정**: 모두 채택

**채택 인터랙션**:
- Hover tooltip ✓
- Event 클릭 → 해당 시기 차트 expand ✓ (Q-Zoom)
- 정렬 (시기순 / 손실 크기순) ✓
- CSV 다운로드 ✓

#### 시각화 예시 (확정 사항 조합)

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

#### 결과 / 함의

- 4개 위기 = 펀드의 위기 시 안정성 narrative (3개 위기 outperform + 1개 underperform 솔직 노출)
- Recovery Time = 위기 후 회복 narrative (마케팅 친화)
- 2024 IT Rotation = HO 정당화 narrative 와 직접 연결
- Timeline = 시간 컨텍스트 시각

### 영역 6: Sensitivity Test (단순 — D3-a) — 확정

#### 영역 6 통합 배경 (Context)

4-slot config 변경 시 결과 비교 — 우리 펀드 config 의 강건성 narrative.

**vs Methodology 영역 4 (4-slot config 표)**:
- Methodology = 4-slot 옵션 정의 (구조)
- Backtesting = 각 config 의 결과 비교 (성과)

**균형 (B) D3-a 적용**: 단순 4-slot 비교 (학술 깊이 X)

#### 결정 항목 BT6-1: 표시 형식

**결정**: (d) 표 + 막대 둘 다

**근거**:
1. **표** = 정확 (Top 10 config 수치)
2. **막대** = Top 1-Top 10 차이 시각 (robustness)
3. (a) 표만은 시각 ↓ / (b) 막대만은 정확성 ↓

#### 결정 항목 BT6-2: 비교 대상

**결정**: (d) Top 10 + 우리 펀드 강조

**근거**:
1. **Top 10** = 충분한 비교 (Top 5 보다 강건성 ↑)
2. **우리 펀드 강조** = mat_eq_eq_raw_pap 위치 즉시 인지 (Methodology 영역 4 일관)
3. (c) slot별 변경은 Methodology 영역 4 와 중복

#### 결정 항목 BT6-3: 메트릭

**결정**: (b) Sortino + CAGR + MDD

**근거**:
1. **3 메트릭 균형** — 위험조정 + 수익 + 위험
2. (a) Sortino only 는 단일 차원 → robustness 검증 부족
3. (c) +Active Return 은 정보 과부하

#### 결정 항목 BT6-4: 인터랙션

**결정**: hover + Methodology 클릭 + 정렬 (CSV 제외)

**채택 인터랙션**:
- Hover tooltip ✓
- config 클릭 → Methodology 영역 4 (4-slot config) navigation ✓
- 정렬 (Sortino / CAGR / MDD 컬럼) ✓
- CSV 다운로드 ✗ (영역 4 종합 표 와 중복)

#### 결정 항목 BT6-5: 추가 표시

**결정**: (a) 우리 펀드 강조 + (b) Top 1-Top 10 차이 시각

**근거**:
1. **우리 펀드 강조** = ★ 마커 + Cobalt Blue accent
2. **Top 1-Top 10 차이 막대** = robustness 시각 narrative
3. (c) slot별 분포는 Methodology 와 중복

#### 시각화 예시

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

#### 결과 / 함의

- 156 config (또는 64 → 156) 의 Sortino 정렬 → Top 10 추출
- Top 1-Top 10 차이 작음 = robust (마케팅 친화 narrative)
- Methodology 영역 4 와 navigation 연결
- CSV 제외 — 영역 4 (Regime 자세한 비교) 의 CSV 와 차별화

### 영역 7 (Stress Test) — 제거 (Q-Stress A 적용)

**제거 결정**: 사용자 지적 (구현 정확성 어려움) + 균형 옵션 (B) + 학술 정직성

영역 7 = Footer (Overview 동일) 로 단순화.

---

### Backtesting 페이지 — 전체 확정 (영역 1~7)

#### 페이지 시각화 통합

```
┌────────────────────────────────────────────────────────────────┐
│ [영역 1: Header — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 2: Sub-header 카드/배너]                                  │
│ Backtesting (백테스트 검증) — Regime / Sub-events / Sensitivity│
│ TEST 168m + HOLD_OUT 24m walk-forward 결과 깊이 있는 검증       │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 3: Backtest Summary KPI 5개]                              │
│ TEST/HO Gap / Sensitivity / 4-slot Robust / Recovery / Regime  │
│ 단순 + 좋음/나쁨 색상 + Tooltip                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 4: Regime 메트릭 자세한 비교]                             │
│ 12 메트릭 표 + Regime별 Sortino 막대 + Best/Worst 강조         │
│ 다중 토글 (Fund/SPY/EW/IVW)                                    │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 5: Sub-events 분석 (역사적 위기)]                         │
│ 4 위기 표 + Timeline + Recovery Time                           │
│ 2018 Q4 / COVID / 2022 Bear / 2024 IT Rotation                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 6: Sensitivity Test (단순)]                               │
│ Top 10 config 표 + Top 1-Top 10 차이 막대 + 우리 펀드 강조      │
│ Sortino + CAGR + MDD                                           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ [영역 7: Footer — Overview 동일]                                │
└────────────────────────────────────────────────────────────────┘
```

#### Backtesting 페이지 결정 결과 / 함의

- **7 영역 모두 확정** (Stress Test 영역 제거)
- **균형 옵션 (B) 적용 완료**:
  - Walk-forward 검증 영역 제거 (LSTM + BL 자체 적용)
  - Stress Test 영역 제거 (구현 정확성 어려움 + 학술 정직성)
  - Sensitivity Test 단순화
  - KPI 재정의 (Stress Survival → Avg Recovery Time)
- **인터랙션 일관성 원칙** 적용:
  - 다중 토글 (영역 4)
  - Q-Zoom (영역 5 Event 클릭)
  - config 클릭 → Methodology 영역 4 navigation (영역 6)
  - Regime 행 클릭 → Sub-events navigation (영역 4 → 영역 5)
- **HO 정당화 narrative 보완**:
  - 영역 5 (2024 IT Rotation 위기) — Sector Watch 영역 8 와 narrative 연결
  - KPI Avg Recovery Time — 위기 후 회복 narrative
- 다음 페이지 (About / FAQ) 결정 시:
  - Selection Bias / PBO/DSR 학술 부록 위치 (Q-B3 결정)

### Backtesting 페이지 → About / FAQ 페이지로 진행

---


---

[← 07_methodology.md](07_methodology.md) | [09_about.md](09_about.md) →
