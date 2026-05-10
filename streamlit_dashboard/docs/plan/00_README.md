# Adaptive VolControl Fund — Streamlit 대시보드 구현 계획 (Plan)

> **프로젝트**: Adaptive VolControl Fund (어댑티브 볼컨트롤 펀드) Streamlit 펀드 홍보 대시보드
> **모델**: `mat_eq_eq_raw_pap` (별도 기준 선정 Top 1 config)
> **작성 시점**: 2026-05-10
> **작성 목적**: decisionlog 의 모든 결정사항을 구현 가능한 plan 으로 변환

---

## 1. 프로젝트 개요

본 plan 은 `streamlit_dashboard/docs/decisionlog/` 의 12개 의사결정 파일을 토대로 작성된 구현 계획입니다.

### 1.1 펀드 정체성

- **영문**: Adaptive VolControl Fund
- **한글**: 어댑티브 볼컨트롤 펀드
- **슬로건**: "변동성 예측 기반 적응형 자산배분 — Volatility-Aware Adaptive Allocation"
- **핵심 메커니즘**: Black-Litterman (BL) + LSTM 변동성 예측 + 4-slot config 검증
- **선정 config**: `mat_eq_eq_raw_pap` (prior=mat / p_weight=eq / q_mode=eq / omega_mode=pap)

### 1.2 펀드 기준 메트릭 (HOLD_OUT 24m)

| 메트릭 | HO 24m | 비고 |
|---|---|---|
| sortino | 0.685 | HO 부진 |
| sharpe | 0.397 | HO 부진 |
| cagr | +8.3% | SPY +21.2% 대비 열위 |
| mdd | -8.3% | SPY -7.6% 와 유사 |
| vol | 10.3% | — |

---

## 2. 청중 / 발표 컨텍스트

### 2.1 타겟 청중 (A 섹션 결정)

- **(b) 경진대회 + (c) 가상 투자자 중점**
- 학교 최종 프로젝트 + 경진대회 출품용
- 가상 투자자 친화 narrative + 학술 정직성 양립

### 2.2 발표 컨텍스트 (K-4 사용자 결정)

> **가상 투자자 대상 홍보물 컨셉**
> - **대시보드 시연**: 5분 제한
> - **학술 분석**: PPT 통해 10~15분 (별도)
> - **대시보드에 발표 스크립트는 포함 X**

### 2.3 5분 demo 흐름 (K-2 결정)

```
1. Overview            (1분)   — Hero KPI + 누적수익 + 핵심 강점
2. Investment Simulator (1.5분) — 사용자 입력 → 즉각 결과
3. Sector Watch (영역 8) (1.5분) — HO 정당화 narrative
4. Methodology         (1분)   — Sankey + 4-slot config
```

### 2.4 사용 환경 (A-4 결정)

- 데스크톱 우선 (Streamlit 다크 테마)
- 발표자 데모 + 청중 셀프 탐색 (URL 공유) 양립

---

## 3. 4가지 핵심 메시지 (K-3 결정)

| # | 메시지 | 학술 근거 | 적용 위치 |
|---|---|---|---|
| 1 | 🟦 **Volatility-Aware Allocation** | LSTM (Hochreiter 1997) + BL (Black-Litterman 1990) | Overview 영역 4 카드 1 |
| 2 | 🟪 **Validated Across Market Regimes** | OOS validation | Overview 영역 4 카드 2 |
| 3 | 🟧 **Net of Conservative Costs** | AQR Frazzini et al. (2018) | Overview 영역 4 카드 3 |
| 4 | 🟩 **Sector 분산의 가치 (HO 정당화)** | Markowitz (1952) | Sector Watch 영역 8 |

---

## 4. 14 파일 인덱스

```
plan/
├── 00_README.md                   # ★ 본 파일 — 인덱스 + 개요
├── 01_setup.md                    # 폴더 구조 + 데이터 + 라이브러리 + 배포
├── 02_common.md                   # lib/* + 사이드바 + 디자인 + Disclosure
├── 03_pages/
│   ├── 01_overview.md             # Overview 페이지 와이어프레임
│   ├── 02_simulator.md            # Investment Simulator 와이어프레임 (★ 신규)
│   ├── 03_performance.md          # Performance 와이어프레임
│   ├── 04_risk_metrics.md         # Risk Metrics 와이어프레임
│   ├── 05_holdings.md             # Holdings 와이어프레임
│   ├── 06_sector_watch.md         # Sector Watch 와이어프레임
│   ├── 07_methodology.md          # Methodology 와이어프레임
│   ├── 08_backtesting.md          # Backtesting 와이어프레임
│   └── 09_about.md                # About 메타만
├── 04_implementation_steps.md      # 구현 단계 (Phase 1/2/3) + 우선순위
└── 05_validation.md               # 검증 / 테스트 / 한계
```

### 4.1 파일별 짧은 설명

| 파일 | 설명 | 참조 decisionlog |
|---|---|---|
| `00_README.md` | 본 파일. 프로젝트 개요 + 핵심 메시지 + 인덱스 | `00_README.md` |
| `01_setup.md` | 폴더 구조, 데이터 layer, 라이브러리, 배포 환경 | `11_dl_sections.md` (D, J) |
| `02_common.md` | 공통 컴포넌트 (lib/*), 사이드바, 디자인 표준, Disclosure | `10_sidebar.md`, `11_dl_sections.md` (G, H, I) |
| `03_pages/01_overview.md` | Overview 페이지 6 영역 와이어프레임 | `02_overview.md` |
| `03_pages/02_simulator.md` | Investment Simulator 7 영역 와이어프레임 (신규) | `11_dl_sections.md` (F-6) |
| `03_pages/03_performance.md` | Performance 페이지 9 영역 와이어프레임 | `03_performance.md` |
| `03_pages/04_risk_metrics.md` | Risk Metrics 페이지 8 영역 와이어프레임 | `04_risk_metrics.md` |
| `03_pages/05_holdings.md` | Holdings 페이지 8 영역 와이어프레임 | `05_holdings.md` |
| `03_pages/06_sector_watch.md` | Sector Watch 페이지 8 영역 (HO 정당화) | `06_sector_watch.md` |
| `03_pages/07_methodology.md` | Methodology 페이지 8 영역 (LSTM walk-forward) | `07_methodology.md` |
| `03_pages/08_backtesting.md` | Backtesting 페이지 7 영역 (Stress 제거) | `08_backtesting.md` |
| `03_pages/09_about.md` | About 페이지 메타만 (영역별 = 구현 후 팀 상의) | `09_about.md` |
| `04_implementation_steps.md` | Phase 1/2/3 구현 단계, 페이지 의존성 | (모든 결정 종합) |
| `05_validation.md` | 데이터 무결성, Streamlit Cloud 체크리스트, 한계 | (모든 결정 종합) |

---

## 5. 결정 일람 (decisionlog → plan 매핑)

### 5.1 메타 결정

| decisionlog 섹션 | 주제 | plan 반영 위치 |
|---|---|---|
| A 섹션 | 청중 / 시나리오 / 언어 / 환경 | `00_README.md` (본 파일) |
| B 섹션 | 펀드 정체성 / 슬로건 / 색상 / 로고 | `00_README.md` + `02_common.md` |
| C-1 ~ C-3 | 페이지 진행 / 메트릭 풀 / Hero KPI | `03_pages/01_overview.md` |
| C-4 | 사이드바 (6 그룹 + 2 토글) | `02_common.md` |

### 5.2 페이지별 결정

| 페이지 | decisionlog 파일 | plan 파일 |
|---|---|---|
| Overview (6 영역) | `02_overview.md` | `03_pages/01_overview.md` |
| Investment Simulator (7 영역) | `11_dl_sections.md` F-6 | `03_pages/02_simulator.md` |
| Performance (9 영역) | `03_performance.md` | `03_pages/03_performance.md` |
| Risk Metrics (8 영역, Hill 축소) | `04_risk_metrics.md` | `03_pages/04_risk_metrics.md` |
| Holdings (8 영역) | `05_holdings.md` | `03_pages/05_holdings.md` |
| Sector Watch (8 영역, HO 정당화) | `06_sector_watch.md` | `03_pages/06_sector_watch.md` |
| Methodology (8 영역, walk-forward) | `07_methodology.md` | `03_pages/07_methodology.md` |
| Backtesting (7 영역, Stress 제거) | `08_backtesting.md` | `03_pages/08_backtesting.md` |
| About (메타만) | `09_about.md` | `03_pages/09_about.md` |

### 5.3 D~L 섹션 결정

| 섹션 | 주제 | plan 반영 위치 |
|---|---|---|
| D | 데이터 (핵심 복사 + yfinance + 함수별 캐싱 + Startup check) | `01_setup.md` + `02_common.md` |
| E | HO Disclosure (HOLD_OUT 24m 통일 / 자신감 톤 + 한계 인정) | `02_common.md` (lib/disclosure.py) |
| F | 시뮬레이션 (TC 20bp + Investment Simulator 신규) | `02_common.md` + `03_pages/02_simulator.md` |
| G | 인터랙션 (같은 페이지 expand / Sim 토글 / 반응형) | `02_common.md` |
| H | 디자인 (Pretendard + Cobalt Blue + GICS 색상) | `02_common.md` |
| I | Disclosure (FINRA + 금감원 + Footer 통일) | `02_common.md` (lib/disclosure.py) |
| J | 기술 (8 라이브러리 + Streamlit Cloud + Custom subdomain) | `01_setup.md` |
| K | Storytelling (5분 demo + 4 메시지 + PPT 분리) | `00_README.md` (본 파일) |
| L | 한계 (현재 그대로 + 00_README 학술 인용 일람 갱신) | `05_validation.md` |

---

## 6. 9 페이지 구조

| # | 페이지 | 영역 | 그룹 | 핵심 narrative |
|---|---|---|---|---|
| 1 | Overview | 6 | 첫인상 | 5초 인상 + 4가지 메시지 진입점 |
| 2 | **Investment Simulator** ★ | 7 | 체험 | 사용자 입력 → 즉각 시뮬레이션 |
| 3 | Performance | 9 | 성과 | 액티브 운용 + 다중 벤치마크 |
| 4 | Risk Metrics | 8 | 성과 | 위험 통제 + Tail Risk (Hill) |
| 5 | Holdings | 8 | 보유 | 종목 detail + 기여도 분석 |
| 6 | Sector Watch | 8 | 보유 | ★ HO 정당화 (Markowitz 1952) |
| 7 | Methodology | 8 | 검증 | BL + LSTM walk-forward 구조 |
| 8 | Backtesting | 7 | 검증 | Regime / Sub-events / Sensitivity |
| 9 | About / FAQ | 8 (메타만) | 메타 | FAQ + 학술 부록 (구현 후 팀 상의) |

---

## 7. 다음 단계

1. **Phase 1 (MVP, 1-2주)**: Setup + 사이드바 + Overview + Performance — `04_implementation_steps.md` 참조
2. **Phase 2 (확장, 2-3주)**: + Risk Metrics + Holdings + Sector Watch + Investment Simulator
3. **Phase 3 (검증, 1-2주)**: + Methodology + Backtesting + About + 검증 — `05_validation.md` 참조

---

## 8. 결정 원칙 일람 (00_README decisionlog 부록)

### 8.1 일관성 원칙

1. **언어**: 한/영 병기
2. **색상**: Cobalt Blue 팔레트 준수
3. **톤**: 학술 정직성 + 가상 투자자 친화 양립
4. **HO 부진 처리**: HOLD_OUT 24m (2024-2025) 통일
5. **메트릭 풀 일관성**: 53개 메트릭 / 9 카테고리 (C-2-6)
6. **인터랙션 일관성** (Q-Zoom 결정): 모든 시계열 차트에 zoom 패턴 일관
7. **다중 벤치마크 토글**: 사이드바 토글 (SPY/EW/IVW) 모든 비교 차트 + KPI delta 동시 영향

### 8.2 회피 원칙

1. "Alpha" 무비판 사용 회피 (B-1)
2. "Guard / Shield / Defense" 수동적 표현 회피 (B-1)
3. 학술 부정확성 회피 (Sortino vs Sharpe 혼동 등)
4. 펀드 정체성과 거리 먼 메트릭 회피 (Withdrawal Rate, 배당 등)
5. β 음수 시 Treynor 표시 회피 (혼동 위험)

---

## 9. 학술 근거 일람 (L-2 결정 갱신)

decisionlog `00_README.md` 의 학술 근거 일람을 그대로 적용:

- **분산 / 평균-분산**: Markowitz (1952), Frazzini-Pedersen (2014)
- **Black-Litterman**: Black-Litterman (1990, 1992), He-Litterman (1999), Idzorek (2005)
- **LSTM**: Hochreiter & Schmidhuber (1997), Gers et al. (2000), Kim & Won (2018)
- **Factor**: Jensen (1968), Fama-French (1993, 2015), Carhart (1997)
- **분포 검정**: Jarque-Bera (1980), Cont (2001), Hill (1975), Embrechts et al. (1997)
- **거래비용**: AQR Frazzini, Israel, Moskowitz (2018), Korajczyk-Sadka (2004)
- **Walk-forward**: Lopez de Prado (2018)
- **Backtest overfitting**: Bailey & Lopez de Prado (2014)
- **위험조정**: Modigliani² (1997), DCC-GARCH Engle (2002)

→ 자세한 인용 위치는 페이지별 plan 참조

---

[01_setup.md →](01_setup.md)
