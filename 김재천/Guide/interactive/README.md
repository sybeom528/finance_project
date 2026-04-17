# 🎮 Interactive — HTML 대시보드 + Streamlit 앱

> **정적 탐색 (HTML) + 인터랙티브 앱 (Streamlit) 2종**

---

## 📂 구성

### 1. `dashboard.html` (1.1 MB)
**즉시 사용 가능한 정적 HTML 대시보드**
- 브라우저에서 더블클릭으로 실행
- 서버 불필요, 오프라인 가능
- Plotly 기반 6종 차트
- 공유·프레젠테이션 용도 우수

**기능**:
- 📊 Top 5 전략 누적수익률
- 🏆 Top 10 Sharpe × MDD 버블
- 📈 최우수 전략 자산군 비중
- 🚦 경보 레벨 시계열
- 🎛️ 4 모드 평균 Sharpe
- 📉 Drawdown 비교

**실행**:
```bash
# Windows
start dashboard.html

# Mac
open dashboard.html

# Linux
xdg-open dashboard.html
```

---

### 2. `streamlit_app/` (8 페이지)
**풀 기능 인터랙티브 탐색 앱**

**실행**:
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

**페이지**:
| # | 페이지 | 핵심 기능 |
|---|------|---------|
| 🏠 | Home | KPI 카드 + 탐색 가이드 |
| 📊 | Overview | 파이프라인 + 초기 vs 최종 |
| 🏆 | Top 10 | 전략 순위 + 다중 선택 비교 |
| 📈 | Composition | 선택 전략 비중 + 날짜 필터 |
| ⚠️ | Alerts | 4 Config 경보 비교 |
| 🌪️ | Crisis | 11 스트레스 시나리오 |
| 🎮 | Simulator | 파라미터 조정 시뮬 |
| 📚 | Learn | 용어 검색 + FAQ |

**테마**: 🌞 라이트 / 🌙 다크 / 🤖 자동 3종 선택

---

## 🔀 HTML vs Streamlit 비교

| 항목 | HTML | Streamlit |
|------|------|---------|
| **설치 필요** | ❌ | ✅ (requirements.txt) |
| **서버 필요** | ❌ | ✅ (로컬/클라우드) |
| **인터랙션** | 제한 (Plotly 기본) | 풀 기능 |
| **공유** | 파일 1개 전송 | URL 또는 배포 |
| **용도** | 빠른 공유, 프레젠테이션 | 심층 탐색, 시뮬 |
| **파일 크기** | 1.1 MB | 소스 + 데이터 의존 |

---

## 🧰 의존 데이터

두 시각화 모두 아래 파일에 의존 (Guide/data/):

- `step9_backtest_results.pkl` — 64 전략 일별 수익률
- `step9_metrics.csv` — 성과 지표
- `step10_final_recommendation.csv` — Top 10
- `step11_top10_weights.pkl` — Top 10 비중 시계열
- `portfolio_prices.csv`, `alert_signals.csv`, `profiles.csv`, `regime_history.csv`

**Step 1~11 노트북 전체 실행이 선행되어야 함**.

---

## 🆕 업데이트 이력

### Final Project (2026-04-17)
- HTML 대시보드 초판 (6 차트)
- Streamlit 앱 초판 (8 페이지, 3 테마)
- 사용자 파라미터 기반 Simulator 구현
- 용어 검색 + FAQ 통합

### 예정 (v4.2)
- Streamlit Cloud 배포 (공개 URL)
- 실시간 VIX 연동 (yfinance)
- 사용자 포트폴리오 업로드 기능

---

## 📚 관련 자료

- 보고서: `../report_final.md`
- 해설: `../docs/Step1~11_해설.md`
- 빠른 참조: `../quick_reference/`
