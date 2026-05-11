# 업데이트 로그 (Update Log)

> **파일 역할**: 최초 decisionlog (2026-05-10) 생성 이후의 모든 변경 사항을 **시간 순 / 카테고리별** 로 종합 관리.
> **참조 원칙**:
>   - 페이지별 결정 이력 → 각 페이지 decisionlog (`02_overview.md` ~ `09_about.md`) 의 변경 박스
>   - 본 파일 → **전체 변경 사항의 시간 순 일람**
> **작성 시작**: 2026-05-12

---

## 사용 방법

- 신규 변경 발생 시 본 파일 상단에 새 날짜 섹션 추가
- 각 변경 항목은 다음 4요소 형식:
  1. **카테고리** (UX / Narrative / Code Structure / Data / Page Structure)
  2. **변경 내역** — Before / After
  3. **사유** — 왜 변경했는지
  4. **영향 파일** — 변경된 핵심 파일 경로

---

## 📅 2026-05-12

### 모델 4차 변경 — `mat_eq_eq_raw_pap` 회귀
- **카테고리**: Data / Model
- **변경 내역**: 3차 `mat_eq_mcap_raw_he` → 4차 (최종) **`mat_eq_eq_raw_pap`** (1차 모델로 회귀, 사용자 결정)
- **4-slot 차원**: prior=capm_eq / **p_weight=eq** (균등 가중) / q_mode=raw_lam / **omega_mode=ff3_paper**
- **영향 파일**: `lib/data_loader.py` (default), `lib/backtesting_charts.py` (default 매개변수), `app.py` + `pages/*.py` 주석, `docs/decisionlog/00_README.md` (이력)

### Performance 페이지 영역 4 신규 — 누적 수익률 차트
- **카테고리**: Page Structure / New Feature
- **변경 내역**: Performance 페이지 9 → 10 영역. KPI 직후 위치에 **누적 수익률 (Performance Trend)** 영역 신규.
- **사유**: 성과 분석 페이지에 가장 기본인 누적 수익률 차트가 빠져있어 페이지 정체성 약함.
- **차별화**: Overview 의 이중 차트 (누적 + Drawdown) 와 다른 단일 차트 (Drawdown 은 Risk Metrics 영역 4 별도).
- **영향 파일**: `lib/performance_charts.py` (`render_cumulative_only` 신규), `pages/03_Performance.py`, `docs/decisionlog/03_performance.md`

### Holdings 영역 4 — 시점 슬라이더 + 보유 (월) 컬럼 추가
- **카테고리**: UX / New Feature
- **변경 내역**:
  - Top N Holdings 표에 시점 슬라이더 추가 (기본 = 최신 월)
  - "보유 (월)" 컬럼 신규 — 선택 시점부터 거꾸로 끊김 없이 연속 보유한 개월 수
- **사유**:
  - 영역 5 (시가총액 분포) 와 일관성 (이미 시점 슬라이더 사용)
  - 펀드 turnover 101% 환경에서 "핵심 holding 안정성 vs 회전 종목" 구분 필요
- **영향 파일**: `lib/metric_calculators.py` (`calc_holding_period` 신규), `lib/holdings_charts.py`, `pages/05_Holdings.py`

### Top N Holdings 표 / Sector Decomposition 표 — % 산식 정정
- **카테고리**: Code / Bug Fix
- **변경 내역**: `st.column_config` 의 printf format `%.Nf%%` 가 Python `:.N%` 와 달리 ×100 자동 처리 안함. 표시용 사본 (`df_display`) 만들어 ×100 적용.
- **사유**: 모든 % 컬럼이 100배 작게 표시되는 버그 (예: 4.23% → 0.04% 로 표시).
- **영향 파일**: `lib/holdings_charts.py:render_top_n_table`, `lib/sector_charts.py:render_sector_decomposition_table`

### Sector Watch KPI 변경 — Active Bets → 섹터 비중 최대 차이
- **카테고리**: Narrative / Metric Replacement
- **변경 내역**: KPI 3 의 "Active Bets" (|Tilt| > 1%p 섹터 수) → **"섹터 비중 최대 차이 (vs SPY)"** = max |Tilt|
- **사유**:
  - 본 펀드 (turnover 101%, dynamic weighting) 의 Active Bets 가 거의 항상 9-11 / 11 → 정보 가치 낮음
  - Max |Tilt| 가 본 펀드의 핵심 narrative (IT under-weight -24.9%p) 직접 표현
  - "베팅" 표현 (도박 톤) 제거
- **영향 파일**: `lib/sector_charts.py:render_sector_kpi`, `pages/06_Sector_Watch.py`

### Sector 한글 매핑 (KPI 카드 한정)
- **카테고리**: UX / Localization
- **변경 내역**: GICS 11 영문 섹터명 → 한글 매핑 (`SECTOR_KO_MAP`). KPI 3/4/5 의 sector value 표시에만 적용.
- **사유**: 좁은 KPI 카드 공간에서 "Information Technology" 같은 긴 영문 섹터명이 잘리는 문제.
- **보존**: 다른 차트 (Treemap, Decomposition 표, Tilt Tornado, Rotation) 는 영문 GICS 표준 유지.
- **영향 파일**: `lib/sector_charts.py` (`SECTOR_KO_MAP`, `_sector_ko`)

### Sector Watch 영역 8 — Hold Out 분석 narrative 강화
- **카테고리**: Narrative / New Chart
- **변경 내역**:
  - Subheader: "HO 24m 분석 + 정당화 narrative" → **"Hold Out 24m 분석"** (부차적 표현 제거)
  - **Chart 2 신규 (4 → 5 chart)**: "IT 섹터 변동성 − 시장 평균 (Spread, %p)" — LSTM 변동성 인지 운용의 IT under-weight 학술 근거 시각화
  - 도입 narrative + 결론 박스 한글화 (학자 인용 / vol-target / risk-aware 등 영어 학술 용어 제거)
- **사유**: IT under-weight 의 근거 부족했던 narrative 보강. 일반 사용자 직관성 향상.
- **영향 파일**: `lib/sector_charts.py:render_ho_justification`, `pages/06_Sector_Watch.py`

### Holdings 영역 8 — Attribution 라벨 한글화
- **카테고리**: UX / Localization
- **변경 내역**:
  - "Attribution 방법" → **"기여도 계산 방식"**
  - "Simple (Brinson 1986)" → **"단순 합 (월별 합산)"**
  - "Carino Smoothed (Carino 1999)" → **"복리 보정 (장기 일치)"**
  - 검증 박스 narrative 도 한글화 (학술 출처는 help tooltip 에 보존)
- **사유**: 학자명 + 영어 학술 용어가 일반 사용자에게 직관적이지 않음.
- **영향 파일**: `lib/holdings_charts.py:render_attribution_tornado`, `pages/05_Holdings.py`

### Holdings KPI 5 — Top Weights 라벨 명확화
- **카테고리**: UX
- **변경 내역**: 라벨 "Top Weights" → **"Top 10 비중"**, caption "T1: X · T5: Y" → "Top 1: X · Top 5: Y", help tooltip 에 산식 + 예시 상세 추가.
- **사유**: "Top Weights" 가 메인 값이 T1/T5/T10 중 무엇인지 불명확.
- **영향 파일**: `lib/holdings_charts.py:render_holdings_kpi`

### Sector Watch KPI — 베팅 / Active Bets 표현 일괄 제거
- **카테고리**: Narrative
- **변경 내역**:
  - KPI 라벨 "최대 베팅 크기" → "섹터 비중 최대 차이"
  - 영역 6 subheader "Sector Tilt vs SPY — Active Bets" → "Sector Tilt vs SPY — 섹터 비중 차이"
  - Caption "활성 베팅 / 큰 베팅" → "의미 있는 차이 / 큰 차이"
- **사유**: "베팅" 은 학술 용어 (Active Bet) 의 직역이라 한글에서는 도박 톤. 펀드 운용 narrative 와 어울리지 않음.
- **영향 파일**: `lib/sector_charts.py`, `pages/06_Sector_Watch.py`

### Sector Watch KPI 5 — Most Underweight ★ 제거
- **카테고리**: UX
- **변경 내역**: IT 일 때 라벨에 ★ 추가하던 로직 제거 ("Most Underweight ★" → "Most Underweight" 일관).
- **사유**: 영역 8 (Hold Out 분석) 에서 IT narrative 충분히 다뤄지므로 KPI 카드 별표 강조 불필요.
- **영향 파일**: `lib/sector_charts.py:render_sector_kpi`

### Footer Meta 정리 — Built with 제거 + 부트캠프 표기
- **카테고리**: Narrative
- **변경 내역**:
  - "Built with: Streamlit + Plotly" 제거
  - "학술 / 경진대회 목적" → "부트캠프 최종 프로젝트"
  - Last updated 갱신
- **사유**: 기술 스택 정보는 Footer 부적합. 실제 프로젝트 목적은 부트캠프 최종 과제.
- **영향 파일**: `lib/disclosure.py`

### docs "경진대회" 표현 정리 (옵션 B — inline 정정)
- **카테고리**: Narrative / Docs
- **변경 내역**:
  - `docs/decisionlog/01_meta_A_B_C.md` 상단에 사후 정정 박스 추가 + 본문 5건 inline 정정 (L39 결정 / L42 근거 / L76 근거 / L93 배경 / L143 근거)
  - `docs/plan/00_README.md` 상단에 사후 정정 박스 추가 + 본문 2건 inline 정정 (2.1 절)
  - 형식: `~~취소선~~ → [정정 2026-05-12: ...]` — 원본 보존 + 정정 명시
- **유지**: 검토된 옵션 표 (A-1 의 `(b) 경진대회` 행) + 변경 이력 인용 부분 (`updatelog.md`, `10_sidebar.md`) — 의사결정 이력 보존
- **사유**: 결정 당시 ("경진대회 + 가상 투자자") 와 실제 컨텍스트 ("부트캠프 최종 프로젝트") 의 괴리 해소. 원본 보존 + 정정 사실 명시 (CLAUDE.md "사후 변경 → 페이지별 박스 추가" 지침 부합).
- **영향 파일**: `docs/decisionlog/01_meta_A_B_C.md`, `docs/plan/00_README.md`

### SPY NaN 보강 + HO docs 정정 (Task 2) — `lib/data_loader.py` + decisionlog/plan
- **카테고리**: Data / Bug Fix / Narrative 정합성
- **배경**:
  - HO SPY CAGR 의 docs claim (21.2%) 과 dashboard 산출치 (20.58%) 사이 0.62%p 차이 발견
  - 추적 결과: `panel.spy_ret` 의 2개 시점 (2024-12-31, 2025-12-31) 이 **NaN**
  - 원인: `monthly_panel.spy_ret` 도 `fwd_ret_1m` 패턴 (다음 21 영업일 cumprod) 으로 산출 → panel 생성 시점에 boundary 데이터 부족 시 NaN 발생
  - 결과: dashboard 의 HO 24m SPY CAGR = 22m 데이터로 산출 (NaN dropna) → **Fund 24m vs SPY 22m 비대칭 비교** ⚠️
- **해결 — `lib/data_loader.py` 의 `load_fund_results` 에 SPY NaN 보강 로직 추가**:
  - 신규 helper 함수 `_fill_spy_ret_nan(spy_ret)`:
    - daily_returns.pkl 의 SPY 로그 수익률 → 산술 변환 (`exp(log) - 1`)
    - NaN 시점 t 에 대해 다음 21 영업일 cumprod 산출 (panel 산식 일치)
    - 21 영업일 데이터 부족 시 → NaN 유지 (안전 fallback)
  - `load_fund_results(fill_spy_nan=True)` 파라미터 추가 (기본 True)
  - 보강 후 `config["spy_filled_dates"]` 에 채워진 날짜 list 저장
- **검증 결과 (6 시나리오 PASS)**:
  - Scenario 1: default 시 `spy_ret.isna().sum() = 0` (목표 0)
  - Scenario 2: `fill_spy_nan=False` 시 원본 NaN 보존 (목표 2)
  - Scenario 3: HO 24m SPY CAGR 20.58% (22m) → **21.07% (24m)** — 대칭 회복
  - Scenario 4: Fund vs SPY 모두 24m 같은 기간 비교 — `차이 -13.87%p`
  - Scenario 5: TEST 168m 영향 없음 / FULL 192m CAGR 14.25% → 14.37% (+0.12%p)
  - Scenario 6: 보강값 검증 — `spy_ret[2024-12-31] = +1.9946%`, `spy_ret[2025-12-31] = +1.9782%`
- **docs 일괄 갱신**:
  - **decisionlog**: `00_README.md`, `02_overview.md`, `06_sector_watch.md`, `11_dl_sections.md` (E-3, E-4, I-1 Footer)
  - **plan**: `00_README.md`, `02_common.md`, `03_pages/06_sector_watch.md`
  - 정정 패턴: `~~원본 수치~~ → **새 수치** [정정 2026-05-12]`
  - 원본 결정 기록은 보존 + 사후 정정 박스 / inline 마커로 갱신 사실 명시
- **최종 정합 수치 (편측 20bp + SPY NaN 보강)**:
  - HO 24m: Fund **+7.20%** / SPY **+21.07%** / 차이 **−13.87%p**
  - (이전 docs: Fund +8.3% / SPY +21.2% / 차이 −12.9%p)
- **영향 파일**: `lib/data_loader.py`, 7개 docs 파일

### 일별 데이터 source 차이 정직성 caption 도입 (옵션 β — Tier B) — Performance/Risk Metrics
- **카테고리**: Narrative / 학술 정직성
- **배경**:
  - 검증 과정에서 `daily_returns.pkl` 과 `monthly_panel.fwd_ret_1m` 가 **별도 source** 임을 발견
    - `monthly_panel.fwd_ret_1m`: 21-day fixed forward window 의 **산술 수익률** (BL backtest 입력 의도)
    - `daily_returns.pkl`: **로그 수익률** (LW 공분산 추정 입력 의도)
  - 두 데이터는 다른 분석 목적으로 설계되어 단일 ticker level 외에는 완벽 일치 X
  - Streamlit 의 일별 사용 3개 영역 (Performance 영역 9, Risk Metrics 영역 7/10) 이 펀드 backtest 와 다른 source 사용 중
- **결정**: 옵션 β (caption + expander 로 학술 정직성 확보)
  - 데이터 재생성 (γ) 의 위험 (펀드 결과 변동) 과 작업 비용 (1-2주) 을 고려
  - 실무 펀드 분석의 표준 패턴 (월별 NAV + 일별 risk metric) 으로 narrative 정당화
  - 펀드 backtest 결과 자체에는 영향 없음
- **변경 내역**:
  - **Performance 영역 9 (분포 통계 — 왜도/첨도/꼬리비율)**:
    - 기존 caption 끝에 1줄 추가: `ℹ️ 일별 = 시장 데이터 기반 (펀드 backtest 의 월별 source 와 별도).`
    - `st.expander("ℹ️ 데이터 source 안내 (분포 통계)")` 추가 — 학술 근거 (Cont 2001) + 산식 + 펀드 관계 명시
  - **Risk Metrics 영역 7 (VaR / CVaR)**:
    - 기존 caption 끝에 1줄 추가: `ℹ️ Basel III 표준 = 일별 (시장 데이터 기반). 펀드 누적 수익률과는 별도 source.`
    - `st.expander("ℹ️ 데이터 source 안내 (VaR / CVaR)")` 추가 — Basel III/EU CRR + 산식 + 펀드 관계 명시
  - **Risk Metrics 영역 10 (Hill Estimator)**:
    - 기존 caption 끝에 1줄 추가: `ℹ️ Hill 은 일별 필수 (꼬리 sample 필요). 시장 데이터 기반 — 펀드 backtest 와 별도 source.`
    - `st.expander("ℹ️ 데이터 source 안내 (Hill Estimator)")` 추가 — Hill (1975), Embrechts (1997) EVT 표준 + 일별 필수 근거 + 펀드 관계 명시
- **영향 파일**: `pages/03_Performance.py`, `pages/04_Risk_Metrics.py`
- **영향 범위**:
  - 변경 X: 펀드 backtest 결과 / KPI / 차트 / lib 모듈 / 데이터 파일
  - 변경 O: 3개 영역의 caption (1줄 추가) + expander (펼치기 박스)
- **학술 정직성**:
  - 검토자/평가자 질문 ("왜 source 가 다른가?") 대비 강화
  - Cont 2001, Hill 1975, Embrechts 1997 등 학술 표준 인용으로 근거 명시
  - 실무 패턴 (월별 NAV + 일별 risk) 으로 narrative 정당화

### TC override 도입 (편측 10bp → 편측 20bp, 펀드 의도값) — `lib/data_loader.py`
- **카테고리**: Data / Bug Fix / 학술 정직성
- **발견 경위**:
  - 시뮬레이션 일별 시점 지원 검토 중 일별 데이터 source 차이 (monthly_panel vs daily_returns) 발견 → 추가 검증 진행
  - 검증 과정에서 `(gross_ret − ret) / turnover = 0.001` (모든 192월 일관) 확인 → 펀드 backtest 가 **편측 10bp** 로 산출되었음을 발견
  - 그러나 dashboard / docs 의 모든 표기는 **편측 20bp** 로 안내 → 표기와 실제 데이터 불일치
- **팀 확인 결과 (2026-05-12)**:
  - 펀드의 **의도된 거래비용 = 편측 20bp (tc=0.002)**
  - 초기 config 설정은 tc=0.001 (편측 10bp) 였으나 차후 의도가 0.002 로 변경됨
  - **pkl 재실행은 불필요** — dashboard 사용 시점에 net 재산출로 의도값 반영
- **변경 내역**:
  - `lib/data_loader.py` 의 `load_fund_results(config_name, tc_override=0.002)` 시그니처에 `tc_override` 파라미터 추가
  - pkl 로드 후 자동으로 `net = gross_ret − turnover × tc_override` 재산출
  - `config["tc"]` = 새 값으로 갱신, `config["tc_original_in_pkl"]` = 원본 tc 보존
  - 모든 페이지 (`app.py` + 5 페이지) 의 `fund_ret = fund["ret"]` 자동으로 새 net 반영
- **검증 결과 (6 시나리오 모두 PASS)**:
  - SCENARIO 1: `(gross − ret) / turnover = 0.002` 일관 (CV=0, mean=0.002000000)
  - SCENARIO 2: `tc_override=None` 시 원본 (편측 10bp) 보존
  - SCENARIO 3: `tc_override=0.001` 명시 시 원본과 동일
  - SCENARIO 4: pkl 파일 자체 무수정 (size / mtime 변화 X)
  - SCENARIO 5: FULL/TEST/HO 모든 KPI 자동 갱신
  - SCENARIO 6: dashboard subperiod 함수 (`calc_*_subperiod`) 자동 반영
- **KPI 변화** (편측 10bp → 편측 20bp):
  | 기간 | CAGR | Sortino | Sharpe | MDD |
  |---|---|---|---|---|
  | FULL 16년 | 16.440% → **15.080%** (−1.36%p) | 1.911 → 1.724 | 1.103 → 1.014 | −12.89% → −13.65% |
  | TEST 168m | 17.657% → **16.251%** (−1.41%p) | 2.039 → 1.853 | 1.185 → 1.096 | −12.89% → −13.65% |
  | HO 24m | 8.268% → **7.202%** (−1.07%p) | 0.685 → 0.516 | 0.397 → 0.302 | −8.25% → −8.95% |
- **영향 범위**:
  - 모든 페이지의 펀드 net 기반 메트릭 (CAGR / Sortino / Sharpe / Calmar / MDD / Active Return / Tracking Error / Information Ratio) 자동 갱신
  - 누적 수익률 차트 / Drawdown 차트 / Investment Simulator 결과 / Regime 분석 / Sub-events / Holdings Attribution 모두 자동 반영
  - Vol / 분포 통계 (Skew/Kurt) 는 거의 영향 없음 (TC 가 변동성에 미치는 영향 미미)
  - SPY / EW / IVW 벤치마크는 영향 없음 (TC 미적용)
- **영향 파일**: `lib/data_loader.py` (단일 파일 수정으로 모든 페이지 자동 반영)
- **장점**:
  - pkl 원본 보존 (재실행 X, 변경 X)
  - 모든 페이지 코드 변경 없이 자동 반영
  - `tc_override=None` 으로 검증 / 디버깅 가능
  - dashboard 표기 ("편측 20bp") 와 실제 적용 일치

### lib/backtesting_charts.py 미사용 함수 cleanup (옵션 A — 완전 제거)
- **카테고리**: Code Structure / Cleanup
- **변경 내역**: 567 라인 → 169 라인 (약 70% 감소)
  - **제거된 deprecated 함수 3개**: `render_backtest_kpi`, `render_cumulative_comparison`, `render_sensitivity_test`
  - **제거된 미사용 헬퍼 5개**: `_fmt_pct`, `_fmt_ratio`, `_color_by_threshold`, `_compute_all_config_metrics`, `_compute_all_config_cumulative`
  - **제거된 imports 7개**: `numpy`, `BENCHMARK_COLORS`, `list_available_configs`, `load_fund_results`, `load_monthly_panel`, `add_event_annotations`, `add_regime_backgrounds` (모두 제거된 함수만 의존)
- **유지**: USED 함수 2개 (`render_regime_detail_table`, `render_sub_events`) + 핵심 imports (`pandas`, `plotly.graph_objects`, `streamlit`, `metric_calculators`, `COLORS`)
- **검증**: `python -c "from lib.backtesting_charts import render_regime_detail_table, render_sub_events"` 정상 동작 확인
- **복원 방법**: `git log -p lib/backtesting_charts.py` 에서 복구 가능 (모듈 docstring 에 명시)
- **사유**: 2026-05-11 Backtesting 페이지 통합 삭제 후 두 함수만 Risk Metrics 페이지로 이전됨. 나머지는 deprecated 상태로 남아 파일 비대 + 신규 개발자 혼동 우려.
- **영향 파일**: `lib/backtesting_charts.py` 단독

---

## 📅 2026-05-11

### Methodology 페이지 통합 삭제 → Sankey 만 Overview 영역 6 으로
- **카테고리**: Page Structure / Deletion
- **변경 내역**:
  - 삭제: `pages/07_Methodology.py`, `lib/methodology_charts.py`
  - 이전: Methodology Overview Sankey → `lib/overview_charts.py:render_methodology_sankey`
  - Overview 페이지 6 → 7 영역 (영역 6 신규)
- **사유**: BL+LSTM 흐름은 단일 Sankey 로 충분 표현. 별도 페이지 운영보다 Overview 마지막 영역으로 통합이 자연스러운 동선.
- **영향 파일**: `app.py`, `lib/overview_charts.py`, `lib/page_helpers.py` (사이드바)
- **자세한 이력**: `docs/decisionlog/07_methodology.md` (DEPRECATED)

### Backtesting 페이지 통합 삭제 → Regime + Sub-events 만 Risk Metrics 영역 5/6 으로
- **카테고리**: Page Structure / Deletion
- **변경 내역**:
  - 삭제: `pages/08_Backtesting.py`
  - 보존: `lib/backtesting_charts.py` (이전된 두 함수만 import 됨)
  - 이전: `render_regime_detail_table`, `render_sub_events` → Risk Metrics 영역 5/6
  - Risk Metrics 페이지 9 → 11 영역
- **사유**: Backtesting 의 학술 검증 narrative 2개는 Risk Metrics 의 자연스러운 흐름 (Drawdown → Regime → Sub-events → VaR/CVaR) 과 더 일치. 156 config sensitivity 등 학술 robustness 메트릭은 가상 투자자 친화 목적과 부합도 낮음.
- **영향 파일**: `pages/04_Risk_Metrics.py`, `app.py` (Navigation cards 6 → 5), `lib/page_helpers.py` (검증 그룹 제거)
- **자세한 이력**: `docs/decisionlog/08_backtesting.md` (DEPRECATED)

### 사이드바 통합 정리
- **카테고리**: UX / Sidebar
- **변경 내역**:
  - 기간 토글 순서: `FULL/TEST/HO` → `TEST/HO/FULL`
  - 기본값: `FULL` → **`TEST`**
  - 표시: `HO` → `Hold Out` (format_func, 내부 value 는 `"HO"` 유지)
  - 벤치마크 라벨 한글화: `SPY` → `SPY (S&P 500 ETF)` / `EW (펀드 universe)` → `균등가중` / `IVW (Naive Low-vol)` → `역변동성 가중`
  - 그룹 변경: 6 그룹 → 5 그룹 (검증 그룹 사라짐 — Methodology + Backtesting 통합 결과)
  - Data 기간 표시: "Data: 2025-12" → "Data: 2010-01 ~ 2025-12"
  - Widget state 보존 (page navigation 시): source-of-truth 패턴 + on_change callback
- **영향 파일**: `lib/page_helpers.py`, `lib/disclosure.py` (init_session_state default)

### KPI 디자인 통일 — st.metric 으로 모든 페이지 통일
- **카테고리**: UX / Visual Consistency
- **변경 내역**:
  - Performance / Risk Metrics / Overview 의 KPI 카드 = `st.markdown + 수동 HTML` → **`st.metric`** (Holdings 와 동일 native 컴포넌트)
  - help tooltip 으로 ⓘ 정보 표시 (이전 수동 HTML ⓘ 제거)
  - delta_color (normal / inverse / off) 활용으로 시각적 일관성
  - KPI 라벨 폰트: 14px → **17px** (CSS 주입)
- **사유**: 페이지마다 다른 디자인 → 사용자 인지 부담. 통일성 + 학습 효과 ↑.
- **영향 파일**: `lib/performance_charts.py`, `lib/risk_charts.py`, `lib/overview_charts.py`, `lib/page_helpers.py` (CSS)

### KPI tooltip ⓘ 일괄 추가
- **카테고리**: UX / Help
- **변경 내역**: Overview / Performance / Risk Metrics / Holdings 모든 KPI 카드에 마우스 hover tooltip 추가 (`lib/tooltips.py:METRIC_TOOLTIPS` 활용).
- **사유**: KPI 카드의 메트릭 정의를 hover 만으로 즉시 확인 가능 (caption 길이 ↓).

### Performance CAGR — Gross + TC 누적 표시
- **카테고리**: Narrative / Transparency
- **변경 내역**: CAGR 카드 하단에 "Gross +X% · TC −Y% (편측 20bp = 0.20%/거래)" 표시.
- **사유**: Net CAGR 의 거래비용 누락 의심 ("Overview Net CAGR = Performance CAGR" 일치) 해소. 거래비용 차감 전후를 한 카드에서 확인.

### 거래비용 표기 정정 — 1회 거래당 0.10% → 0.20% (편측 20bp)
- **카테고리**: Code / Bug Fix (표기 오류)
- **변경 내역**:
  - 원본: "One-way 20bp = 거래 1회당 0.10%" ❌
  - 정정: "편측 20bp = 1회 거래당 0.20%" ✓
- **사유**: `final/bl_functions.py:apply_tc` 의 `tc` 는 편측(per-side) rate. 20bp = 0.20%/거래. 원본 표기 잘못.
- **영향 파일**: `pages/03_Performance.py`, `lib/performance_charts.py`, `lib/overview_charts.py`, `lib/tooltips.py`

### HOLD_OUT → Hold Out 표기 통일 (사용자 표시 영역)
- **카테고리**: UX / Localization
- **변경 내역**: 사용자에게 표시되는 모든 위치에서 `HOLD_OUT` → `Hold Out`. 내부 dict key (`EVAL_PERIODS["HOLD_OUT"]`) 는 호환성 보존.
- **영향 파일**: `app.py`, `lib/disclosure.py`, `lib/holdings_charts.py`, `lib/sector_charts.py`, `lib/overview_charts.py`

### 각 페이지 caption 직관적 한글로 일괄 변경 (40곳)
- **카테고리**: Narrative / Localization
- **변경 내역**: 모든 페이지의 sub-header description + 영역별 caption 을 학술/기술 영어 위주 → 직관적 한글로 변경.
- **제거된 표현**:
  - "final/master_table 정합", "decisionlog Q-E"
  - "Frazzini, Israel & Moskowitz 2018 Trading Costs 표준"
  - "R1 회복 / R2 확장 / R3 변동" → "회복기 / 확장기 / 변동기"
  - "TC 차감 후 Net" → "거래비용 차감 후"
  - "Hill (1975) 학술 표준. ξ̂ > 0 = fat tail"
- **보존된 학술 용어**: Sharpe / Sortino / VaR / CVaR / HHI / β / R² (펀드 표준 메트릭)
- **영향 파일**: `app.py`, `pages/*.py`

### 부차적 설명 일괄 제거
- **카테고리**: Narrative / Cleanup
- **변경 내역**:
  - "(학술 정직성)", "(학술 정직 — ...)" → 제거
  - "자세한 정당화 narrative 는 ... 페이지 참조" → 제거
  - "자세한 분석은 ... 페이지" → 제거
  - "★ ..." 강조 표시 (불필요한 곳) → 제거

### Performance 영역 9 / Risk Metrics 영역 7 — 분포 통계 / VaR/CVaR 일별 only
- **카테고리**: Narrative / Statistical Accuracy
- **변경 내역**: 월별 / 일별 Tab 표시 → **일별만 표시** 단순화.
- **사유**:
  - 월별 (192 sample) 의 5% 분위수는 단 ~9.6개 → 통계 신뢰성 매우 낮음
  - 학술/실무 VaR 표준 = 일별 (Basel III, J.P. Morgan RiskMetrics)
  - 월별 분포 통계 (Skew/Kurt) 는 CLT 효과로 정규에 수렴 → 의미 약함
- **영향 파일**: `lib/performance_charts.py:render_distribution_stats`, `lib/risk_charts.py:render_var_cvar_distribution`

### VaR/CVaR + 분포 통계 차트 — x축 동적 range
- **카테고리**: UX / Visualization
- **변경 내역**: 자동 범위 (±12%) → **0.5% / 99.5% 분위수 기반 동적 range × 1.15** (99% 데이터 포함, 극단 outlier 제외).
- **사유**: 극단 outlier 때문에 정작 99% 데이터가 몰린 중앙 영역이 압축되어 보이는 가독성 문제.
- **영향 파일**: `lib/risk_charts.py:_render_distribution_chart`, `lib/performance_charts.py:_render_distribution_card`

### Risk Metrics Rolling 메트릭 — 벤치마크 비교선 추가
- **카테고리**: Narrative / New Feature
- **변경 내역**:
  - Vol / Sortino: 절대 메트릭 → SPY/EW/IVW 자체 rolling 값 비교선
  - β / R² / TE: SPY 기준 메트릭 → EW/IVW 의 SPY 대비 rolling 비교선
  - Fund 라인: 메트릭별 다른 색상 (Vol=Blue, Sortino=Green, Beta=Amber, R²=Purple, TE=Red), 굵은 실선
  - 벤치마크 라인: 벤치마크별 색상 + 점선 (SPY=dot, EW=dash, IVW=dashdot)
- **사유**: 단일 Fund 라인만 보여줘서 시장 대비 비교 어려움. 벤치마크와의 차이 시계열로 학술 정보 ↑.
- **영향 파일**: `lib/risk_charts.py:render_rolling_risk_metrics`

### 동명 함수 충돌 해결 — `calc_avg_recovery_time`
- **카테고리**: Code / Bug Fix
- **변경 내역**: `lib/metric_calculators.py` 에 동명 함수 2개 (Series 용, DataFrame 용) 가 있어 Python 후정의 override 로 Series 버전이 사라진 문제. 두 번째 함수를 `calc_avg_recovery_from_subevents` 로 rename.
- **영향 파일**: `lib/metric_calculators.py`, `lib/backtesting_charts.py` (호출부)

### Risk Metrics Drawdown 평균 Recovery 에러 수정
- **카테고리**: Code / Bug Fix
- **변경 내역**: `'Series' object has no attribute 'columns'` 에러 — `calc_avg_recovery_time` 의 두 함수 충돌 결과. rename 으로 해결.

### Sector Watch / Holdings — 사이드바 진입 시 벤치마크 reset 문제 해결
- **카테고리**: Code / Bug Fix
- **변경 내역**: Streamlit multipage 의 widget unmount/remount 시 session_state reset 문제. **Source-of-truth 패턴** 적용 — Widget key 와 session_state key 분리 + on_change callback 으로 source 업데이트.
- **영향 파일**: `lib/page_helpers.py:render_sidebar`

### KPI 라벨 폰트 확대 — 14px → 17px
- **카테고리**: UX / Typography
- **변경 내역**: `st.metric` 의 라벨 폰트가 작아 가독성 ↓. CSS 주입 (`[data-testid="stMetricLabel"]`) 으로 일괄 확대.
- **영향 파일**: `lib/page_helpers.py:_PRETENDARD_CSS`

### Page Header — 우상단 메타 정보 제거
- **카테고리**: UX / Cleanup
- **변경 내역**: 모든 페이지 우상단의 "● Active (Simulated) / Benchmark: S&P 500 / Data as of: 2025-12-31" 제거. 페이지명만 단일 column 으로 표시.
- **사유**: 사이드바 + Footer 와 정보 중복. 우상단 공간 활용도 낮음.
- **영향 파일**: `lib/page_helpers.py:render_page_header`

### Investment Simulator — Insight 카드 2×2 그리드 + 동일 높이
- **카테고리**: UX / Visualization
- **변경 내역**:
  - 카드 그리드: 3-col (4 카드 시 3 + 1 불균형) → **2-col** (4 카드 시 완벽 2×2)
  - `min-height: 180px` 추가 — 카드 내용 (delta / subtitle 유무) 무관하게 동일 높이
- **영향 파일**: `lib/insight_generator.py`, `pages/02_Investment_Simulator.py`

### Methodology 페이지의 ★ 강조 일괄 제거 (페이지 삭제 전)
- **카테고리**: Narrative
- **변경 내역**: 영역 4/5 의 "본 펀드 ★ 강조" 같은 별표 표시 모두 제거.
- **참고**: 이 변경 후 페이지 자체가 통합 삭제됨.

---

## 📝 향후 작업 TODO (Pending Tasks)

### 시뮬레이션 일별 시점 지원 결정
- **상태**: 보류
- **배경**: Investment Simulator 의 시작/종료 시점이 월별만 의미 있게 변경됨 (펀드 backtest 가 월별 rebalancing).
- **옵션**:
  - A. 월별 명확화 (UI 를 월말 select_slider 로 변경)
  - B. Lump-sum 만 일별 지원 (일별 수익률 데이터 활용, but TC 미적용으로 ~2.4%p/연 과대평가)
  - C. 현재 유지 + caption 안내
- **결정 필요**: 정확성 (A) vs 정밀 timing (B) trade-off
- **자세한 분석**: 본 대화 (2026-05-12) "시뮬레이션 일별 시뮬레이션 부정확성" 내용 참조

### About / FAQ 페이지 구현 (Phase 4)
- **상태**: 메타 결정만 확정 (영역별 구현 X)
- **자세한 이력**: `docs/decisionlog/09_about.md`

### About 페이지 — 데이터 source 종합 안내 박스 추가 (옵션 β 후속)
- **상태**: 보류 (Tier B 옵션 β 의 후속 작업)
- **배경**: 2026-05-12 일별 데이터 source 차이 caption (Performance 영역 9 + Risk Metrics 영역 7/10) 도입. 각 영역의 expander 는 효과적이나, **한 곳에서 종합 reference** 도 필요.
- **목적**:
  - 학술 / 평가자 청중을 위한 "데이터 source 종합 안내" 한 페이지 reference
  - 펀드 backtest (`monthly_panel`) 과 시장 risk metric (`daily_returns`) 의 본질적 분리 명시
  - yfinance / retroactive adjustment / NaN 처리 등 데이터 한계 narrative
- **구현 예시**:
  - About 페이지의 한 영역으로 "데이터 source 와 한계" 박스 추가
  - 또는 Methodology 페이지 (만약 부활 시) 의 하위 섹션
  - 또는 Sidebar 의 "Data Notes" expander
- **의존성**: About / FAQ 페이지 구현 (Phase 4) 완료 후 진행 권장
- **자세한 분석**: 본 파일의 "일별 데이터 source 차이 정직성 caption 도입 (옵션 β)" 항목 참조

### ~~lib/backtesting_charts.py 미사용 함수 cleanup~~ → ✅ 완료 (2026-05-12)
- **상태**: ✅ 완료 (2026-05-12)
- **배경**: Backtesting 페이지 통합 삭제 후 두 함수 (`render_regime_detail_table`, `render_sub_events`) 만 사용됨. 나머지 (`render_backtest_kpi`, `render_cumulative_comparison`, `render_sensitivity_test`) 는 deprecated 상태로 남아있음.
- **결정**: 옵션 A — 완전 제거 (567 → 169 라인, 약 70% 감소)
  - deprecated 함수 3개 + 미사용 헬퍼 5개 + 불필요 imports 7개 제거
  - 복원 필요 시 `git log -p lib/backtesting_charts.py` 에서 복구
- **상세 변경 이력**: 본 파일 상단 2026-05-12 섹션 참조

### ~~docs 의 "경진대회" 표현 정리 (선택)~~ → ✅ 완료 (2026-05-12)
- **상태**: ✅ 완료 (2026-05-12)
- **배경**: 결정 당시 (2026-05-10) 의 청중 컨텍스트 ("경진대회 심사위원 + 가상 투자자") 가 docs 에 남아있음.
- **결정**: 옵션 B — 본문 inline 정정 (`~~취소선~~ → [정정 ...]` 마커) + 각 파일 상단 사후 정정 박스 추가. 원본 텍스트는 취소선으로 보존하고 정정 사항을 명시.
- **유지**: 검토된 옵션 표 (A-1 의 `(b) 경진대회` 행) + 변경 이력 인용 부분 (`updatelog.md`, `10_sidebar.md`) — 의사결정 이력 보존 차원
- **변경 파일**: `docs/decisionlog/01_meta_A_B_C.md` (6건) + `docs/plan/00_README.md` (2건)
- **상세 변경 이력**: 본 파일 상단 2026-05-12 섹션 마지막 항목 참조
