# 🎮 Final Project — Streamlit 앱

> **대안데이터 기반 포트폴리오 시뮬레이터 인터랙티브 탐색 도구**

## 🚀 실행 방법

### 1. 가상환경 생성 (권장)

```bash
cd Guide/interactive/streamlit_app
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 앱 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 열립니다: **http://localhost:8501**

---

## 📑 8개 페이지 구성

| # | 페이지 | 설명 |
|---|------|------|
| 🏠 | **Home** | 프로젝트 개요 + KPI + 탐색 가이드 |
| 📊 | **Overview** | 11단계 파이프라인 + 초기 vs 최종 + 벤치마크 |
| 🏆 | **Top 10** | 최우수 전략 상세 비교 + 순위 표 + Sharpe×MDD 산점도 |
| 📈 | **Composition** | 선택 전략의 30자산 비중 시계열 (일별) |
| ⚠️ | **Alerts** | 4 Config 경보 시스템 비교 |
| 🌪️ | **Crisis** | 11 스트레스 시나리오 (COVID, 2022 긴축 등) |
| 🎮 | **Simulator** | 파라미터 조정 → 실시간 결과 |
| 📚 | **Learn** | 용어 사전(검색) + FAQ + 학습 경로 |

---

## 🎨 테마

사이드바에서 3가지 테마 선택 가능:
- 🌞 **라이트** (기본)
- 🌙 **다크**
- 🤖 **자동** (시스템 설정 따라감)

---

## 📊 데이터 의존성

앱은 다음 파일을 자동 로드합니다 (`../../data/` 참조):

| 파일 | 용도 |
|------|------|
| `portfolio_prices.csv` | 30자산 가격 |
| `alert_signals.csv` | 경보 4 Config |
| `profiles.csv` | 성향 파라미터 |
| `regime_history.csv` | HMM 레짐 |
| `step9_metrics.csv` | 64 전략 지표 |
| `step9_backtest_results.pkl` | 일별 수익률 |
| `step10_final_recommendation.csv` | Top 10 |
| `step11_top10_weights.pkl` | Top 10 비중 시계열 |
| `regime_covariance_by_window.pkl` | Σ by 윈도우 |

**사전 조건**: 노트북 Step 1~11이 모두 실행되어 산출물이 생성되어 있어야 함.

---

## 🧩 디렉터리 구조

```
streamlit_app/
├── app.py                    # 홈 페이지 (엔트리)
├── requirements.txt
├── README.md                 # 본 파일
├── pages/                    # 다중 페이지
│   ├── 1_📊_Overview.py
│   ├── 2_🏆_Top_10.py
│   ├── 3_📈_Composition.py
│   ├── 4_⚠️_Alerts.py
│   ├── 5_🌪️_Crisis.py
│   ├── 6_🎮_Simulator.py
│   └── 7_📚_Learn.py
└── utils/
    ├── __init__.py
    ├── data_loader.py        # 캐시된 데이터 로더
    └── theme.py              # 테마 관리 (라이트/다크/자동)
```

---

## 🛠️ 주요 기능

### 데이터 캐싱
- `@st.cache_data`: CSV / DataFrame
- `@st.cache_resource`: PKL / 큰 객체

### 인터랙티브 요소
- 📌 전략 선택 dropdown/multiselect
- 📅 기간 선택 date_input
- 🎚️ 파라미터 slider (거래비용 등)
- 🔍 용어 검색

### 시각화
- Plotly 기반 (줌·hover·저장 지원)
- 테마 자동 매핑 (plotly_white / plotly_dark)

---

## 🚨 문제 해결

### 앱이 시작 안 됨
```bash
# 포트 충돌 시
streamlit run app.py --server.port 8502
```

### 한글 깨짐
- Windows: Malgun Gothic 폰트 설치 확인
- Mac: AppleGothic (기본 포함)
- Linux: `pip install koreanize-matplotlib` (노트북 용)

### 데이터 파일 없음 오류
Step 1~11 노트북 먼저 실행:
```bash
cd ../../
jupyter notebook
# 순차 실행
```

### 느림
- 첫 로드 후 캐시됨 (이후 빨라짐)
- 메모리 8GB 이상 권장 (Step 11 pkl 6.3MB)

---

## 📚 관련 자료

- **HTML 정적 대시보드**: `../dashboard.html` (브라우저로 바로 열기)
- **보고서**: `../../report_final.md`
- **해설**: `../../docs/Step1~11_해설.md`
- **빠른 참조**: `../../quick_reference/` (13종)

---

## 🔄 버전

- **Final Project** (2026-04-17): 초판 출시
  - 8개 페이지
  - 3개 테마
  - 20+ 인터랙티브 차트

## 📞 문의

프로젝트 저자: 김재천 · Guide/ 폴더 내 문서 참조
