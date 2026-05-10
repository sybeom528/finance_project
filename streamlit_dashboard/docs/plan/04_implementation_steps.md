# 04. 구현 단계 — Phase 1/2/3 + 우선순위

> **관련 decisionlog**: 모든 결정 종합
> **상태**: 확정
> **목적**: Streamlit 대시보드 구현 단계별 계획 + 페이지 의존성 + 우선순위

---

## 1. 전체 일정 개요

| Phase | 기간 | 목표 | 핵심 산출물 |
|---|---|---|---|
| **Phase 1: MVP** | 1-2주 | 기본 대시보드 가동 | Setup + Sidebar + Overview + Performance |
| **Phase 2: 확장** | 2-3주 | 모든 분석 페이지 + 시뮬레이터 | Risk + Holdings + Sector + Investment Simulator + Methodology (간략) |
| **Phase 3: 검증** | 1-2주 | 학술 깊이 + 검증 + About | Methodology (완성) + Backtesting + About + 검증 |

**총 예상 기간**: 4-7주

---

## 2. Phase 1: MVP (1-2주)

### 2.1 목표

5분 demo 의 첫 1분 (Overview) + 핵심 메시지 진입점 가동.

### 2.2 산출물

#### A. Setup (1-2일)

- [ ] **폴더 구조 생성** (`01_setup.md` 1.1 참조):
  - `streamlit_dashboard/` 전체 구조
  - `pages/`, `lib/`, `data/`, `scripts/`, `assets/`, `docs/`

- [ ] **`requirements.txt`** (J-5 Range versions, `01_setup.md` 3.2):
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

- [ ] **`.streamlit/config.toml`** (H-5 다크 테마 + Cobalt Blue):
  ```toml
  [theme]
  base = "dark"
  primaryColor = "#3B82F6"
  backgroundColor = "#0E1117"
  secondaryBackgroundColor = "#1F2937"
  textColor = "#FAFAFA"
  ```

- [ ] **데이터 복사** (D-1):
  - `final/data/` → `streamlit_dashboard/data/`:
    - monthly_panel.csv
    - daily_returns.pkl
    - ff5_monthly.csv
    - universe.csv
    - results/mat_eq_eq_raw_pap.pkl

- [ ] **회사명 매핑 수집** (D-2):
  - `scripts/build_ticker_company_map.py` 한 번 실행
  - 산출: `data/ticker_company_map.csv`

#### B. lib/* 공통 컴포넌트 (3-4일)

- [ ] **`lib/data_loader.py`** (D-3 캐싱 표준)
  - `load_monthly_panel()`, `load_daily_returns()`, `load_fund_results()` 등
  - `equal_weight_returns()`, `ivw_returns()` baseline 산출

- [ ] **`lib/validators.py`** (D-5 Startup check)
  - `startup_data_check()` 필수 데이터 파일 검증

- [ ] **`lib/colors.py`** (B-4 + H-4)
  - COLORS / BENCHMARK_COLORS / REGIME_COLORS / SECTOR_COLORS / LIMITATION_COLORS / SANKEY_GROUP_COLORS

- [ ] **`lib/disclosure.py`** (E-3 + I-2 + I-5)
  - FOOTER_DISCLOSURE 통일 텍스트
  - `render_footer()`, `render_simulator_disclaimer()`
  - `init_session_state()` 토글 값 초기화

- [ ] **`lib/tooltips.py`** (메트릭 정의 dictionary)
  - METRIC_TOOLTIPS dictionary
  - `get_tooltip(metric_name)` 헬퍼

- [ ] **`lib/plot_helpers.py`** (Plotly 공통)
  - `add_regime_backgrounds()`, `add_event_annotations()`, `bilingual_label()`

- [ ] **`lib/interactions.py`** (G-1 Q-Zoom)
  - `render_zoomable_chart()` 같은 페이지 expand 헬퍼

#### C. 사이드바 + Page Routing (1일)

- [ ] **`app.py`** (`02_common.md` 1.2 참조):
  - `st.set_page_config(layout="wide")`
  - Pretendard CSS 주입 (`inject_custom_css()`)
  - Startup check 호출
  - Session state 초기화
  - 사이드바 렌더 (펀드명 + 6 그룹 + 2 토글)

- [ ] **Streamlit multi-page 자동 routing**:
  - `pages/01_Overview.py` ~ `pages/09_About.py`
  - Numeric prefix 로 사이드바 자동 그룹화

#### D. Overview 페이지 (3-4일)

- [ ] **`pages/01_Overview.py`** (`03_pages/01_overview.md` 참조):
  - 영역 1: Header
  - 영역 2: Hero KPI 5개 (반응형 + sparkline)
  - 영역 3: 누적수익 곡선 (이중 차트 + Regime + EW/IVW 토글 + Q-Zoom)
  - 영역 4: 핵심 강점 카드 3개 (`streamlit-card` 활용)
  - 영역 5: Navigation cards 7개
  - 영역 6: Footer

#### E. Performance 페이지 (3-4일)

- [ ] **`pages/03_Performance.py`** (`03_pages/03_performance.md` 참조):
  - 영역 1: Header / 영역 2: Sub-header
  - 영역 3: Performance Summary KPI 5개 (액티브 운용 강조)
  - 영역 4: Annual Returns 막대 (다중 벤치마크 + Q-Zoom)
  - 영역 5: Active Return 분석 (Image 1 + Image 2 위아래)
  - 영역 6: Annualized Rolling Return (1y/3y/5y 토글)
  - 영역 7: Regime 메트릭 Heatmap (Tab 전환)
  - 영역 8: 분포 통계 카드 (Skewness/Kurtosis/Tail Ratio + Tab 전환)
  - 영역 9: Footer

### 2.3 Phase 1 검증

- [ ] Streamlit 로컬 실행 정상 (`streamlit run streamlit_dashboard/app.py`)
- [ ] 사이드바 6 그룹 + 2 토글 동작
- [ ] Overview 페이지 모든 영역 렌더
- [ ] Performance 페이지 모든 영역 렌더
- [ ] 사이드바 토글 (기간 + 다중 벤치마크) 차트 갱신 정상

### 2.4 Phase 1 시간 추정

| 작업 | 기간 |
|---|---|
| Setup (A) | 1-2일 |
| lib/* 공통 (B) | 3-4일 |
| 사이드바 + Routing (C) | 1일 |
| Overview (D) | 3-4일 |
| Performance (E) | 3-4일 |
| **합계** | **11-15일 (1-2주)** |

---

## 3. Phase 2: 확장 (2-3주)

### 3.1 목표

5분 demo 의 모든 흐름 + 인터랙티브 시뮬레이터 가동.

### 3.2 산출물

#### A. Risk Metrics 페이지 (3-4일)

- [ ] **`pages/04_Risk_Metrics.py`** (`03_pages/04_risk_metrics.md` 참조):
  - 영역 1-2: Header / Sub-header
  - 영역 3: Risk Summary KPI 5개 (Vol/MDD/Beta/R²/TE)
  - 영역 4: Drawdown + Recovery Time (이중 차트 + Top 3)
  - 영역 5: VaR/CVaR 분포 (히스토그램 + KDE + 임계선)
  - 영역 6: Beta + R² + TE Rolling (3개 분리 차트 + β<0 R² 신뢰성)
  - 영역 7: Risk Metrics 종합 표 (카테고리 그룹화 + Diff column)
  - 영역 8: Tail Risk 분석 (Hill estimator 옵션 C 축소 — Q-F2)
  - 영역 9: Footer

- [ ] **`lib/metric_calculators.py`** 추가:
  - `hill_estimator()` (Hill 1975)
  - `compute_var_cvar()` (Historical VaR + CVaR)
  - `compute_recovery_time()` (DD bottom 이후 신고가 갱신)

#### B. Holdings 페이지 (3-4일)

- [ ] **`pages/05_Holdings.py`** (`03_pages/05_holdings.md` 참조):
  - 영역 1-2: Header / Sub-header
  - 영역 3: Holdings Summary KPI 6개 (Number/Eff N/Single HHI/Sector HHI/Top Weights/Turnover)
  - 영역 4: Top N Holdings 표 (Top 토글 + Weight 막대 + 섹터 색상)
  - 영역 5: 시가총액 분포 (Bubble + Treemap 탭)
  - 영역 6: 보유 종목 변천사 (Multi-line + 신규/제외 마커)
  - 영역 7: 종목별 기여도 분석 (Tornado Chart)
  - 영역 8: Footer (+ yfinance Footnote)

#### C. Sector Watch 페이지 (3-4일)

- [ ] **`pages/06_Sector_Watch.py`** (`03_pages/06_sector_watch.md` 참조):
  - 영역 1-2: Header / Sub-header (HO narrative 명시)
  - 영역 3: Sector Summary KPI 5개 (HHI/Avg|Tilt|/Active Bets/Most Over/Most Under)
  - 영역 4: Sector Treemap (좌우 분할)
  - 영역 5: Sector Decomposition 표 (9 컬럼)
  - 영역 6: Sector Tilt vs SPY (Tornado + 임계선)
  - 영역 7: Sector Rotation (Stacked Area / Multi-line 토글)
  - 영역 8: ★★★ HO 24m 분석 + 정당화 (3 차트 + Markowitz 1952 narrative)
  - 영역 9: Footer

#### D. Investment Simulator 페이지 (3-4일)

- [ ] **`pages/02_Investment_Simulator.py`** (`03_pages/02_simulator.md` 참조):
  - 영역 1-2: Header / Sub-header (친근 톤) + Disclaimer 박스 (I-5)
  - 영역 3: Input (Tab 전환 — Lump-sum / DCA / Goal)
  - 영역 4: Result KPI 5개 (단순 패턴)
  - 영역 5: 누적 자산 곡선 (Fund + 사이드바 토글 + DCA 누적 + Regime + 위기)
  - 영역 6: Insight 박스 (정적 템플릿 카드 그리드)
  - 영역 7: Footer

- [ ] **`lib/insight_generator.py`** (Sim 영역 6 핵심):
  - `generate_insight_cards(sim_result, benchmarks, scenario)` (조건부 4-8 카드)
  - `render_insight_grid(cards)` (반응형 3-column)

- [ ] **시뮬레이션 로직** (`pages/02_Investment_Simulator.py`):
  - `simulate_lump_sum()` 일시 투자
  - `simulate_dca()` 매월 추가 투자
  - `simulate_goal_based()` 역산 (binary search)

#### E. Methodology 간략 (1-2일)

- [ ] **`pages/07_Methodology.py`** (Phase 2 의 간략 버전):
  - 영역 1-3: Header / Sub-header / Sankey
  - 영역 4-5: BL 상세 (간략) / LSTM 상세 (간략)
  - 영역 9: Footer
  - **Phase 3 에서 영역 6, 7, 8 완성 예정**

### 3.3 Phase 2 검증

- [ ] 5분 demo 흐름 정상 (Overview → Sim → Sector Watch → Methodology)
- [ ] 모든 사이드바 토글 영향 페이지 갱신 정상
- [ ] Q-Zoom 모든 시계열 차트 작동
- [ ] yfinance 회사명 매핑 정상 표시
- [ ] HO 정당화 narrative (Sector Watch 영역 8) 정상 렌더

### 3.4 Phase 2 시간 추정

| 작업 | 기간 |
|---|---|
| Risk Metrics (A) | 3-4일 |
| Holdings (B) | 3-4일 |
| Sector Watch (C) — HO 정당화 핵심 | 3-4일 |
| Investment Simulator (D) | 3-4일 |
| Methodology 간략 (E) | 1-2일 |
| **합계** | **13-18일 (2-3주)** |

---

## 4. Phase 3: 검증 (1-2주)

### 4.1 목표

학술 정직성 강화 + 검증 페이지 + About 메타 + 검증 작업.

### 4.2 산출물

#### A. Methodology 완성 (3-4일)

- [ ] **`pages/07_Methodology.py`** 완성 (`03_pages/07_methodology.md` 참조):
  - 영역 6: Factor 분석 (CAPM + FF5) — 조건부, 결과 확인 후 재평가
  - 영역 7: 정규성 검정 (Jarque-Bera) — LSTM 정당화 + 동적 narrative
  - 영역 8: 한계 + 향후 개선 — 3개 카드 + Expander + 동적 추가

- [ ] **동적 narrative logic**:
  - 영역 7 Jarque-Bera 결과 → 영역 8 한계 카드 동적 추가 (`st.session_state.lstm_value_unproven`)

#### B. Backtesting 페이지 (3-4일)

- [ ] **`pages/08_Backtesting.py`** (`03_pages/08_backtesting.md` 참조):
  - 영역 1-2: Header / Sub-header
  - 영역 3: Backtest Summary KPI 5개 (TEST/HO Gap / Sensitivity / 4-slot / Recovery / Regime 일관성)
  - 영역 4: Regime 메트릭 자세한 비교 (12 메트릭 + Best/Worst)
  - 영역 5: Sub-events 분석 (4 위기 + Timeline)
  - 영역 6: Sensitivity Test (Top 10 + 우리 펀드 강조)
  - 영역 7: Footer

- [ ] **Sensitivity Test 데이터**:
  - `final/data/results/` 의 다른 155 config pkl 참조 (D-1)
  - Top 10 추출 (Sortino 기준)

#### C. About 페이지 (메타만 — 1-2일)

- [ ] **`pages/09_About.py`** (`03_pages/09_about.md` 참조):
  - 영역 1-2: Header / Sub-header (메타만)
  - 영역 3-5: 펀드 소개 / FAQ / 데이터 출처 (메타만 — 팀 상의 후 자세히 작성)
  - 영역 6: Selection Bias 학술 부록 (Expander + 학술 인용 link)
  - 영역 7: Disclosure 자세한 버전 (I-3 — Risk factors 5가지)
  - 영역 8: Footer

- [ ] **★ 영역별 자세한 결정 = 구현 후 팀 상의** (사용자 결정)

#### D. 검증 + 한계 (1-2일)

- [ ] **`05_validation.md` 참조**:
  - 데이터 무결성 검증 (D-5 Startup check)
  - Streamlit Cloud 배포 전 체크리스트
  - 한계 (L 섹션)
  - 향후 개선 (Future Work)

- [ ] **L-2 결정 적용**:
  - 페이지별 학술 인용 → 00_README.md 학술 근거 일람 일괄 갱신

#### E. 배포 + Warmup (1일)

- [ ] **GitHub Push** (`01_setup.md` 4.3):
  - `streamlit_dashboard/` 폴더 전체 commit

- [ ] **Streamlit Cloud 연결**:
  - share.streamlit.io 접속 → GitHub repo 연동
  - main branch + `streamlit_dashboard/app.py` 지정
  - Custom subdomain 설정 (예: `volcontrol.streamlit.app`)

- [ ] **Warmup 발표 30분 전 1회 접속** (J-3 슬립 모드 회피)

### 4.3 Phase 3 검증

- [ ] Methodology 영역 6 Factor 결과 확인 → 재평가 (유지/축소/제거)
- [ ] 영역 7 Jarque-Bera 동적 narrative (Case A/B) 정상 동작
- [ ] 영역 8 동적 카드 추가 (Case B 시) 정상 동작
- [ ] Backtesting 영역 6 Sensitivity Test (156 config) 메모리 효율
- [ ] About 페이지 영역 6 Expander 학술 인용 link 정상
- [ ] Streamlit Cloud 배포 정상

### 4.4 Phase 3 시간 추정

| 작업 | 기간 |
|---|---|
| Methodology 완성 (A) | 3-4일 |
| Backtesting (B) | 3-4일 |
| About 메타 (C) | 1-2일 |
| 검증 + 한계 (D) | 1-2일 |
| 배포 + Warmup (E) | 1일 |
| **합계** | **9-13일 (1-2주)** |

---

## 5. 페이지별 의존성

```
Setup (A) ────┬──> lib/* (B) ──> 사이드바 + Routing (C) ──┬──> Overview (D)
              │                                            │
              └──> data/ 복사 ──┬──> ticker_company_map.csv │
                                │  (yfinance 1회 수집)      │
                                │                           │
                                ├──> EW + IVW baseline      ├──> Performance (E)
                                │  (lib/data_loader.py)     │
                                │                           │
                                └──> Hill estimator,        └──> 다른 페이지들
                                     Jarque-Bera 등
                                     (lib/metric_calculators.py)
```

**핵심 의존성**:
- 모든 페이지 → `lib/data_loader.py` 캐싱
- 모든 페이지 → `lib/colors.py` 팔레트
- 모든 페이지 → `lib/disclosure.py` Footer
- 모든 페이지 → `lib/tooltips.py` 메트릭 정의
- 시계열 차트 → `lib/plot_helpers.py` Regime 배경 + 위기 annotation
- Investment Simulator → `lib/insight_generator.py`

---

## 6. 각 단계 검증 항목

### 6.1 Phase 1 검증 항목

| 항목 | 방법 |
|---|---|
| Streamlit 정상 실행 | `streamlit run streamlit_dashboard/app.py` |
| 사이드바 6 그룹 + 2 토글 동작 | UI 확인 |
| Overview Hero KPI 5개 정상 | 반응형 + sparkline 확인 |
| Overview 누적수익 곡선 | EW/IVW 토글 + Q-Zoom + Regime 배경 |
| Performance 모든 영역 | 다중 벤치마크 토글 갱신 |
| Pretendard 폰트 적용 | 한글 가독성 확인 |
| 다크 테마 + Cobalt Blue | 모든 페이지 일관 |

### 6.2 Phase 2 검증 항목

| 항목 | 방법 |
|---|---|
| 5분 demo 흐름 (K-2) | Overview → Sim → Sector → Methodology |
| Risk Metrics Hill estimator | 직접 구현 + plateau detection |
| Holdings yfinance 회사명 | ticker_company_map.csv 참조 |
| Sector Watch HO 정당화 | 영역 8 narrative + 학술 인용 |
| Investment Simulator | Lump-sum / DCA / Goal 모두 정상 |
| Insight 박스 동적 카드 | 사이드바 토글 + Tab 활성에 따라 |
| Q-Zoom 모든 시계열 차트 | 클릭 expand 정상 |

### 6.3 Phase 3 검증 항목

| 항목 | 방법 |
|---|---|
| Methodology Sankey 노드 클릭 | 영역 4-7 navigation |
| Factor 분석 결과 | 재평가 (양수 alpha → 유지 / 음수 → 축소) |
| Jarque-Bera 동적 narrative | Case A/B 자동 표시 |
| 영역 8 동적 카드 추가 | Case B 시 LSTM 가치 미입증 카드 |
| Backtesting Sensitivity | 156 config Top 10 정상 |
| About Selection Bias 부록 | Expander + 학술 인용 link |
| Streamlit Cloud 배포 | URL 접속 정상 |

---

## 7. 우선순위 매트릭스

### 7.1 5분 demo 핵심 페이지 (우선순위 ★★★)

| 페이지 | Phase | 시간 비중 (5분 demo) |
|---|---|---|
| Overview | Phase 1 | 1분 |
| Investment Simulator | Phase 2 | 1.5분 |
| Sector Watch (HO 정당화) | Phase 2 | 1.5분 |
| Methodology (간략) | Phase 2 | 1분 |

### 7.2 자유 탐색 페이지 (우선순위 ★★)

| 페이지 | Phase |
|---|---|
| Performance | Phase 1 |
| Risk Metrics | Phase 2 |
| Holdings | Phase 2 |
| Backtesting | Phase 3 |

### 7.3 메타 페이지 (우선순위 ★)

| 페이지 | Phase |
|---|---|
| About | Phase 3 (메타만, 영역별 = 팀 상의) |

---

## 8. 리스크 + 대응

### 8.1 데이터 의존 리스크

| 리스크 | 대응 |
|---|---|
| yfinance rate limiting | 1회 수집 + CSV 캐시 (D-2) |
| 데이터 파일 누락 | D-5 Startup check |
| pickle 호환성 | requirements.txt range 내 pandas/numpy |

### 8.2 Streamlit Cloud 리스크

| 리스크 | 대응 |
|---|---|
| 메모리 1GB 제한 | D-3 캐싱 + Sensitivity Test Top 10 만 메모리 |
| 슬립 모드 (1주 미사용) | 발표 전 30분 warmup |
| streamlit-card 라이브러리 미사용 | 옵션 2 (HTML + 별도 버튼) fallback |

### 8.3 학술 결과 리스크

| 리스크 | 대응 |
|---|---|
| Methodology 영역 6 Factor alpha 음수 | 영역 축소 (B) 또는 제거 (C) 검토 |
| 영역 7 Jarque-Bera Case B (정규분포 채택) | 영역 8 한계 카드 동적 추가 |

---

## 9. 다음 단계

→ `05_validation.md`: 검증 / 테스트 / 한계 / Future Work

---

[← 03_pages/09_about.md](03_pages/09_about.md) | [05_validation.md →](05_validation.md)
