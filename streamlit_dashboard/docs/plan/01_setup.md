# 01. Setup — 폴더 구조 + 데이터 + 라이브러리 + 배포

> **관련 decisionlog**: `11_dl_sections.md` (D 섹션 데이터 + J 섹션 기술 스택)
> **상태**: 확정
> **목적**: 대시보드 구축에 필요한 모든 인프라 (폴더 / 데이터 / 의존성 / 배포 환경) 정의

---

## 1. 폴더 구조

### 1.1 전체 streamlit_dashboard/ 구조

```
streamlit_dashboard/
├── app.py                          # Streamlit 진입점 (사이드바 + page routing)
├── requirements.txt                # J-5 Range versions
├── README.md                       # 프로젝트 설명 (배포 시 참조)
├── .streamlit/
│   └── config.toml                 # H-5 다크 테마 + Cobalt Blue (B-4)
│
├── pages/                          # Streamlit multi-page (자동 navigation)
│   ├── 01_Overview.py              # 4가지 메시지 진입점
│   ├── 02_Investment_Simulator.py  # 사용자 입력 → 시뮬레이션 (★ 신규)
│   ├── 03_Performance.py           # 성과 분석
│   ├── 04_Risk_Metrics.py          # 위험 지표
│   ├── 05_Holdings.py              # 보유 종목
│   ├── 06_Sector_Watch.py          # 섹터 분석 (★ HO 정당화)
│   ├── 07_Methodology.py           # 방법론 (BL + LSTM)
│   ├── 08_Backtesting.py           # 검증 (Regime / Sub-events)
│   └── 09_About.py                 # FAQ + 학술 부록
│
├── lib/                            # 공통 모듈 (02_common.md 참조)
│   ├── __init__.py
│   ├── data_loader.py              # D-3 캐싱 표준 + 데이터 로딩
│   ├── validators.py               # D-5 Startup check
│   ├── colors.py                   # H-4 Cobalt Blue + GICS 11개 섹터 색상
│   ├── disclosure.py               # E-3 + I-2 Footer + I-5 Sim disclaimer
│   ├── insight_generator.py        # Sim 영역 6 정적 템플릿 카드 그리드
│   ├── tooltips.py                 # 메트릭 정의 dictionary (모든 페이지 공유)
│   ├── interactions.py             # G-1 Q-Zoom expand 헬퍼
│   ├── metric_calculators.py       # Net 메트릭 / Sortino / VaR / CVaR 등
│   └── plot_helpers.py             # Plotly 공통 헬퍼 (Regime 배경색 등)
│
├── data/                           # D-1 핵심 복사 (final/data/ 에서 복사)
│   ├── monthly_panel.csv           # rf, spy_ret, sector, log_mcap (월별)
│   ├── daily_returns.pkl           # 822 ticker × 6099 영업일
│   ├── ff5_monthly.csv             # Fama-French 5-factor
│   ├── universe.csv                # 833 ticker + gics_sector
│   ├── ticker_company_map.csv      # D-2 yfinance 1회 수집 (회사명)
│   └── results/
│       └── mat_eq_eq_raw_pap.pkl   # 우리 펀드 결과
│
├── scripts/                        # 보조 스크립트
│   └── build_ticker_company_map.py # D-2 yfinance → CSV 캐시 (1회 실행)
│
├── assets/                         # 정적 파일 (필요 시)
│   └── images/                     # 로고, 아이콘 등
│
└── docs/                           # 본 문서들
    ├── decisionlog/                # 12 decisionlog 파일
    └── plan/                       # 13 plan 파일
```

### 1.2 final/data/ 참조 (D-1 결정)

핵심 외 data 는 **참조** (복사하지 않음):

```
final/data/                         # 참조 (필요 시 직접 접근)
├── results/
│   ├── (다른 155 config pkl)       # Backtesting 영역 6 Sensitivity Test 시
│   ├── prices_raw.pkl              # 원본 가격 데이터
│   └── shares_outstanding.pkl
└── ...
```

**Sensitivity Test 영역 (Backtesting 영역 6)** 에서만 필요 시 직접 접근.

---

## 2. 데이터 Layer (D 섹션)

### 2.1 D-1. 데이터 파일 위치 (핵심 복사 + 나머지 참조)

**핵심 복사 대상 (`streamlit_dashboard/data/`)**:

| 파일 | 크기 추정 | 용도 |
|---|---|---|
| `monthly_panel.csv` | ~수 MB | rf / spy_ret / sector / log_mcap (월별 패널) |
| `daily_returns.pkl` | ~수십 MB | 822 ticker × 6099 영업일 일별 수익률 |
| `ff5_monthly.csv` | ~KB | Fama-French 5-factor (Methodology 영역 6) |
| `universe.csv` | ~KB | 833 ticker + gics_sector (Holdings / Sector Watch) |
| `results/mat_eq_eq_raw_pap.pkl` | ~MB | 우리 펀드 backtest 결과 (weights, returns) |

### 2.2 D-2. 회사명 매핑 (yfinance 1회 수집)

**산출물**: `streamlit_dashboard/data/ticker_company_map.csv` (ticker, company_name)

**구현**: `scripts/build_ticker_company_map.py` 한 번 실행

```python
# scripts/build_ticker_company_map.py
import yfinance as yf
import pandas as pd
from pathlib import Path

tickers = pd.read_csv("final/data/universe.csv")["ticker"].tolist()
mapping = {}
for t in tickers:
    try:
        mapping[t] = yf.Ticker(t).info.get("longName", t)
    except Exception:
        mapping[t] = t  # fallback to ticker

output_path = Path("streamlit_dashboard/data/ticker_company_map.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {"ticker": list(mapping.keys()), "company_name": list(mapping.values())}
).to_csv(output_path, index=False)

print(f"Saved {len(mapping)} ticker → company mappings to {output_path}")
```

**주의**:
- yfinance rate limiting 회피 위해 1회만 실행
- 실패 시 ticker 자체를 회사명으로 fallback
- 결과 CSV 는 git 에 commit 권장

### 2.3 D-3. 캐싱 전략 (함수별 적절히)

**적용 패턴** — `lib/data_loader.py` 참조:

```python
import streamlit as st
import pandas as pd
import pickle

# 데이터 로딩 (DataFrame, dict, JSON-serializable)
@st.cache_data
def load_monthly_panel():
    return pd.read_csv("streamlit_dashboard/data/monthly_panel.csv")

@st.cache_data
def load_daily_returns():
    with open("streamlit_dashboard/data/daily_returns.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_pkl_results(config_name: str):
    with open(f"streamlit_dashboard/data/results/{config_name}.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_ff5_monthly():
    return pd.read_csv("streamlit_dashboard/data/ff5_monthly.csv")

@st.cache_data
def load_ticker_company_map():
    return pd.read_csv("streamlit_dashboard/data/ticker_company_map.csv")

# 외부 API (yfinance 추가 호출 시) — TTL 30일
@st.cache_data(ttl=86400 * 30)
def fetch_yfinance_info(ticker: str):
    import yfinance as yf
    return yf.Ticker(ticker).info

# 모델 / 연결 (singleton, 모든 세션 공유)
@st.cache_resource
def load_lstm_model():
    import torch
    return torch.load("model.pt")
```

**캐싱 결정 기준**:
- `@st.cache_data`: DataFrame / dict / JSON / pickle 결과
- `@st.cache_resource`: 모델 / DB 연결 / 무거운 singleton
- `ttl` 파라미터: 외부 API 결과 (yfinance 추가 호출 시 30일)

### 2.4 D-4. 데이터 갱신 주기 (정적, 배포 시점 고정)

- 학술 백테스트 = 시점 고정 데이터 필수
- 가상 펀드 = 실시간 갱신 불필요
- 데이터 갱신 시 = 새 배포 (Streamlit Cloud rebuild)

### 2.5 D-5. Startup check (앱 시작 시 1회)

**구현**: `lib/validators.py`

```python
import streamlit as st
from pathlib import Path

REQUIRED_DATA_FILES = [
    "streamlit_dashboard/data/monthly_panel.csv",
    "streamlit_dashboard/data/daily_returns.pkl",
    "streamlit_dashboard/data/ff5_monthly.csv",
    "streamlit_dashboard/data/universe.csv",
    "streamlit_dashboard/data/ticker_company_map.csv",
    "streamlit_dashboard/data/results/mat_eq_eq_raw_pap.pkl",
]

def startup_data_check():
    """앱 시작 시 1회 실행. 필수 데이터 파일 존재 검증."""
    missing = [p for p in REQUIRED_DATA_FILES if not Path(p).exists()]
    if missing:
        st.error(f"필수 데이터 파일 누락: {missing}")
        st.error("다음 명령어로 ticker_company_map.csv 를 먼저 생성하세요:")
        st.code("python scripts/build_ticker_company_map.py")
        st.stop()
```

**`app.py` 에서 호출**:
```python
from lib.validators import startup_data_check
startup_data_check()  # 모든 페이지 진입 전 검증
```

---

## 3. 라이브러리 (J-1 결정)

### 3.1 8개 라이브러리 list

| 카테고리 | 라이브러리 | 용도 | 비고 |
|---|---|---|---|
| 메인 | streamlit | 프레임워크 | 1.30+ 신기능 활용 |
| 차트 | plotly | 모든 차트 (Sankey 포함) | Sankey, Treemap 등 모두 지원 |
| 데이터 | pandas, numpy | DataFrame / 수치 | 기본 |
| 데이터 수집 | yfinance | 회사명 매핑 (1회) | scripts/ 에서만 |
| 통계 | scipy | Jarque-Bera, Hill estimator | Risk Metrics / Methodology |
| 회귀 | statsmodels | CAPM / FF5 회귀 | Methodology 영역 6 |
| 카드 UI | streamlit-card | Overview 영역 4 강점 카드 | Hover + 클릭 인터랙션 |

### 3.2 J-5. requirements.txt (Range versions)

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

**근거** (J-5):
- **Range versions** = 안정성 + 유연성 (보안 패치 자동)
- **Major version 핀** = 메이저 변경 회피
- (a) Pinned (특정 버전) = 보안 패치 누락
- (c) Latest = 갑작스런 변경 위험

### 3.3 라이브러리 선정 원칙 (H-7 안정성 보강)

**핵심**: streamlit / plotly / pandas / numpy
**추가**: yfinance / streamlit-card (선택)
**회피**: streamlit-elements / streamlit-aggrid 등 무거운 라이브러리

**Fallback / Graceful degradation**:
- 폰트 로딩 실패 → fallback chain (H-1)
- 외부 데이터 (yfinance) 실패 → ticker 만 표시
- Plotly 차트 에러 → `st.error` + 메시지

---

## 4. 배포 (J 섹션)

### 4.1 J-3. Streamlit Cloud (Community 무료)

**선택 근거**:
- 표준 배포 (GitHub 연동 자동 배포)
- 무료 (학술 / 가상 펀드 적합)
- Streamlit 표준 환경 (호환성 ★★★)

**제약사항**:
- 메모리 제한 1GB (Community)
- 슬립 모드 (1주 미사용 시)
- → 메모리 효율 (D-3 캐싱) + 발표 전 warmup 필요

### 4.2 J-4. URL — Custom subdomain

**URL 안**:
- `https://volcontrol.streamlit.app/`
- 또는 `https://adaptive-volcontrol.streamlit.app/`

**근거** (J-4):
- 펀드 정체성 (VolControl) 노출 → 마케팅 효과
- 기본 URL = `[username]-[repo]-app.streamlit.app` 길고 어색

### 4.3 배포 절차

1. **GitHub Push**:
   - `streamlit_dashboard/` 폴더 전체 commit
   - `data/` 폴더는 git LFS 또는 외부 저장소 권장 (크기에 따라)

2. **Streamlit Cloud 연결**:
   - share.streamlit.io 접속
   - GitHub repo 연동
   - main branch + `streamlit_dashboard/app.py` 지정

3. **Custom subdomain 설정**:
   - Settings → General → App URL → `volcontrol`

4. **Secrets / 환경 변수**:
   - 현재 외부 API 미사용 (yfinance 는 scripts/ 에서만)
   - 향후 API 키 필요 시 `.streamlit/secrets.toml` 사용

5. **Warmup (발표 전 30분)**:
   - 슬립 모드 회피
   - 발표 30분 전 1회 접속

### 4.4 README.md (배포 시 참조)

`streamlit_dashboard/README.md` 에 다음 내용 포함 권장:
- 프로젝트 한 줄 설명
- 라이브 데모 URL
- 로컬 실행 방법
- 데이터 준비 (yfinance 1회 수집)

---

## 5. 환경 변수 / Secrets

**현재 미사용**:
- yfinance: scripts/ 에서만 (1회 실행, 인증 불필요)
- 데이터: 정적 (배포 시점 고정)

**향후 추가 가능**:
- 외부 API (Finnhub, Alpha Vantage 등) 사용 시 → `.streamlit/secrets.toml`

---

## 6. 초기 실행 가이드

### 6.1 로컬 실행 (개발 환경)

```bash
# 1. 의존성 설치
pip install -r streamlit_dashboard/requirements.txt

# 2. 데이터 준비 (한 번만)
python streamlit_dashboard/scripts/build_ticker_company_map.py

# 3. 데이터 복사 (final/data/ → streamlit_dashboard/data/)
# (수동 또는 별도 스크립트)

# 4. Streamlit 실행
streamlit run streamlit_dashboard/app.py
```

### 6.2 첫 실행 시 검증

- Startup check 통과 (D-5 — `lib/validators.py`)
- 사이드바 6 그룹 + 2 토글 정상 표시 (C-4)
- Overview 페이지 정상 렌더 (Hero KPI / 누적수익 곡선)

### 6.3 디버깅

- 데이터 누락 시 → Startup check 에러 메시지 확인
- 차트 미표시 시 → Plotly 에러 + 데이터 NaN 확인
- 폰트 깨짐 시 → Pretendard fallback chain (H-1, `02_common.md` 참조)

---

## 7. 주의사항

### 7.1 데이터 무결성

- `data/` 의 모든 파일은 D-5 Startup check 에서 존재 검증
- pickle 파일 호환성 = pandas / numpy 버전 일치 (requirements.txt range 내)

### 7.2 메모리 효율 (Streamlit Cloud 1GB 제한)

- D-3 캐싱 활용 (모든 데이터 로딩에 `@st.cache_data`)
- 큰 pickle (daily_returns.pkl, 156 config pkl) 은 함수별로 lazy loading
- Sensitivity Test (Backtesting 영역 6) 는 Top 10 만 메모리 로드

### 7.3 슬립 모드 회피

- 발표 전 30분 warmup
- 향후 자동 ping 스크립트 검토 (선택)

---

## 8. 다음 단계

→ `02_common.md`: 공통 컴포넌트 (lib/* + 사이드바 + 디자인 + Disclosure) 정의

---

[← 00_README.md](00_README.md) | [02_common.md →](02_common.md)
