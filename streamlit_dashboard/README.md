# streamlit_dashboard/ — Adaptive VolControl Fund Dashboard

Adaptive VolControl Fund의 백테스트 성과를 시각화하는 **대화형 Streamlit 대시보드**. 5분 데모용 마케팅 자료 + 학술 투명성 제공이 목표.

> 분석 메서드와 결과 산출 경로는 [`../final_pt/`](../final_pt/), 프로젝트 전체 개요는 [상위 README](../README.md) 참고.

## 빠른 시작

```bash
cd streamlit_dashboard
uv run streamlit run app.py
```

기본 진입: **Overview** 페이지가 자동 로드됩니다.

### 데이터 동기화 (선택)

대시보드는 `streamlit_dashboard/data/`에 사본 데이터를 사용합니다. `final_pt/`의 데이터·결과가 변경되었을 때만 동기화 필요:

```bash
uv run python scripts/copy_data.py
```

## 페이지 구성 (6개 + Overview)

| 파일 | 페이지 | 내용 |
|---|---|---|
| `app.py` | **Overview** | 펀드 메타 헤더 → 5 KPI 카드 → 누적수익 곡선(Regime+비교선) → 3 강점 카드 → 네비게이션 → Sankey 다이어그램(BL+LSTM 파이프라인 시각화) |
| [`pages/02_Investment_Simulator.py`](pages/02_Investment_Simulator.py) | **Investment Simulator** ⭐ | Lump-sum / DCA / Goal-based 3가지 시뮬레이션 탭 → 결과 KPI + 누적 자산 곡선 + 자동 인사이트 카드 (4-8개 조건부) |
| [`pages/03_Performance.py`](pages/03_Performance.py) | **Performance** | CAGR/Sortino/Sharpe/IR/Active Return + 누적수익 + 연간수익률 + Active Return + Rolling Return(1y/3y/5y) + Regime 히트맵 + 분포 통계 (Skewness/Kurtosis/Tail Ratio) |
| [`pages/04_Risk_Metrics.py`](pages/04_Risk_Metrics.py) | **Risk Metrics** | Vol/MDD/Beta/R²/TE KPI + Drawdown 시계열·회복시간 + VaR/CVaR 분포 + 5개 메트릭 Rolling + 종합 표 (~22 메트릭 CSV 다운로드) |
| [`pages/05_Holdings.py`](pages/05_Holdings.py) | **Holdings** | 보유 수 / Eff N / HHI / Turnover KPI + Top N 표 + 시가총액 분포(Bubble+Treemap) + 11 섹터 변천사 + 기여도 Tornado |
| [`pages/06_Sector_Watch.py`](pages/06_Sector_Watch.py) | **Sector Watch** | Sector HHI/Tilt KPI + Treemap (Fund vs SPY) + Sector 분해 표 + Tornado (Tilt) + Stacked Area + HO 정당화 narrative (Markowitz + Fama-French) |
| [`pages/09_About.py`](pages/09_About.py) | **About / FAQ** | 펀드 정체성·철학·프로젝트 메타 + FAQ 4개 + 데이터 출처 + Disclosure (표준 disclaimer + 5개 위험요인) |

## 폴더 구조

```
streamlit_dashboard/
├── app.py                       Overview 진입점
├── pages/                       6개 페이지 모듈 (Streamlit 자동 사이드바)
├── lib/                         차트·계산·헬퍼 모듈
├── data/                        대시보드용 데이터 사본 (gitignored 큰 파일)
├── scripts/                     데이터 복사·매핑 빌더·QA 스크립트
├── assets/                      이미지·정적 자원 (현재 비어있음, inline CSS)
├── docs/                        구현 가이드 + 의사결정 로그 + 페이지 plan
├── requirements.txt             Streamlit Cloud 배포용 (UV 외)
└── README.md
```

## `lib/` — 모듈 (19개)

**핵심 차트·렌더링**:
| 모듈 | 역할 |
|---|---|
| `overview_charts.py` | Overview 페이지 영역 컴포넌트 (KPI / 곡선 / Sankey) |
| `performance_charts.py` | Performance 페이지 영역 (Annual / Rolling / 분포) |
| `risk_charts.py` | Risk Metrics 차트 (Drawdown / VaR 분포 / Rolling) |
| `holdings_charts.py` | Holdings 영역 (Top N / Treemap / 기여도) |
| `sector_charts.py` | Sector Watch 영역 (Tilt / HO 정당화 narrative) |
| `simulator_charts.py` | Investment Simulator 결과 시각화 |
| `backtesting_charts.py` | Risk Metrics 페이지 영역 5/6 차트 (12 메트릭 × 5 Regime + Sortino 막대) |

**계산·로직**:
| 모듈 | 역할 |
|---|---|
| `metric_calculators.py` | Sharpe / Sortino / VaR / CVaR / Beta / Tracking Error 등 메트릭 계산 |
| `simulator.py` | Investment Simulator 시뮬레이션 로직 (Lump-sum / DCA / Goal-based) |
| `insight_generator.py` | 자동 인사이트 텍스트 생성 (시뮬레이션 결과용) |

**데이터·검증**:
| 모듈 | 역할 |
|---|---|
| `data_loader.py` | `@st.cache_data` 캐싱된 데이터 로더 |
| `validators.py` | 시작 시 데이터 무결성 검증 |
| `search_index.py` | 사이드바 영역 검색 기능 (페이지별 영역 인덱스 + 키워드 매칭) |

**UI·공통 헬퍼**:
| 모듈 | 역할 |
|---|---|
| `page_helpers.py` | 페이지 헤더·서브헤더·CSS(Pretendard 폰트 fallback) 주입 |
| `plot_helpers.py` | Plotly 공통 헬퍼 (Regime 배경색, 위기 annotation, 한영 병기 라벨) |
| `colors.py` | 색상 팔레트 단일 진실 공급원 (B-4 + H-4 + M3-2 결정 통합) |
| `tooltips.py` | 메트릭 정의 dictionary (hover 시 표시) |
| `interactions.py` | Q-Zoom 인터랙션 (같은 페이지 expand 패턴) |
| `disclosure.py` | 세션 초기화 + Footer disclaimer 렌더링 |

## 데이터 의존성

`data/` 폴더의 파일 출처는 두 갈래입니다.

**(A) `scripts/copy_data.py`가 복사하는 6개** (`final_pt/data/` + `final_pt/results/` 원본):

| 파일 | 원본 위치 | 용도 |
|---|---|---|
| `daily_returns.pkl` | `final_pt/data/` | 822 ticker × 6099 영업일 일별 수익률 |
| `monthly_panel.csv` | `final_pt/data/` | 월별 패널 (rf, spy_ret, sector, log_mcap) |
| `ff5_monthly.csv` | `final_pt/data/` | Fama-French 5-factor |
| `universe.csv` | `final_pt/data/` | 833 ticker + GICS 섹터 |
| `sp500_membership.pkl` | `final_pt/data/` | 시점별 S&P500 편입 종목 |
| `results/mat_eq_eq_raw_pap.pkl` | `final_pt/results/` | Winner 백테스트 결과 (Top 1 config) |

**(B) `scripts/build_ticker_company_map.py`가 별도로 생성하는 파일**:

| 파일 | 생성 방법 | 용도 |
|---|---|---|
| `ticker_company_map.csv` | yfinance API 호출로 ticker→회사명 매핑 빌드 | UI에서 종목명 표시용 |

> 백테스트 90개 슬롯 비교가 필요한 페이지가 있다면 `final_pt/results/`의 다른 pkl도 수동 복사 필요. 현재는 winner pkl 하나만 사용.

## `scripts/`

| 스크립트 | 역할 |
|---|---|
| `copy_data.py` | `final_pt/data/`·`final_pt/results/` → `streamlit_dashboard/data/` 사본 생성 (재실행 안전, 동일 크기 시 skip) |
| `build_ticker_company_map.py` | 티커 ↔ 회사명 매핑 CSV 생성 (UI 표시용) |
| `dev_test.py`, `test_lib.py` | 개발 중 sanity check |

## 기술 스택

| 패키지 | 버전 | 용도 |
|---|---|---|
| streamlit | 1.30+ | 메인 프레임워크 |
| plotly | 6.0+ | 대화형 차트 (누적곡선, 히트맵, Sankey, Tornado) |
| pandas / numpy | 2.0+ / 2.0+ | 데이터 처리 (numpy 2.x pickle 호환 필요) |
| scipy / statsmodels | latest | 통계 (VaR, Sharpe, Rolling 메트릭) |
| streamlit-card | 1.0+ | KPI 카드 UI |
| yfinance | 0.2+ | `scripts/`에서만 사용 (외부 데이터 fetch) |

전체 의존성은 [`requirements.txt`](requirements.txt) 참고. 로컬 실행은 루트 `pyproject.toml`의 `uv sync`만으로 충분.

## 배포

Streamlit Community Cloud (무료):
1. https://share.streamlit.io 접속, GitHub 연결
2. Repository: `sybeom528/finance_project`
3. Branch: `main`
4. Main file path: `streamlit_dashboard/app.py`
5. Advanced settings → Python version 3.12

배포 후 URL을 루트 README의 "Live Demo" 섹션에 추가하세요.

## 더 자세한 가이드

- [`docs/IMPLEMENTATION_GUIDELINES.md`](docs/IMPLEMENTATION_GUIDELINES.md) — 구현 가이드라인
- [`docs/plan/`](docs/plan/) — 페이지별 설계 plan
- [`docs/decisionlog/`](docs/decisionlog/) — 의사결정 로그
