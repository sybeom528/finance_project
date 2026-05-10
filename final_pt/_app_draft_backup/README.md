# Streamlit 대시보드 — Low-Risk BL 시연 도구

> **`final/` 작업물의 통합 시연 도구. 6 페이지 멀티페이지 Streamlit.**
> 발표·시연용 + 팀 내부 자유 탐색 겸용.

## 🚀 빠른 실행

```bash
cd final
pip install streamlit          # 한 번만
streamlit run app/streamlit_app.py
```

브라우저가 자동으로 `http://localhost:8501` 을 엽니다.

## 📂 페이지 목록

| 파일 | 역할 |
|---|---|
| `streamlit_app.py` | 진입점 + Home (프로젝트 소개) |
| `pages/1_📊_Overview.py` | KPI hero + Top 1 추천 + 누적수익률 |
| `pages/2_🔍_Experiment_Explorer.py` | 165 실험 마스터 테이블 + 단일 실험 6 패널 |
| `pages/3_⚖️_Compare.py` | 다중 실험 비교 6 패널 (Sortino 강조) |
| `pages/4_🗺️_Matrix_Heatmap.py` | 매트릭스 3 prior × 3 pw × 5 q × 3 ω = 135 cells |
| `pages/5_🎯_Risk_Profile.py` | 보수/중립/공격 슬라이더 + Top 3 추천 |
| `pages/6_📈_Volatility_Stats.py` | LSTM σ + 04 학술 통계 요약 |

## 🧱 lib/ 모듈

| 파일 | 역할 |
|---|---|
| `lib/data_loader.py` | 캐시된 pkl/master_table/panel/spy/rf 로더 |
| `lib/plot_helpers.py` | 99_explore plot 함수 추출 + Compare 6 패널 확장 |
| `lib/narrative.py` | 페이지별 한국어 설명 + Top 1 추천 근거 |
| `lib/recommendations.py` | 위험성향별 Top 3 매핑 (mat_eq_eq_lam_pap 우선) |

## 📌 핵심 설계 결정

1. **Sortino 우선** — master_table default sort = `sortino`. Compare 페이지에 12m
   rolling sortino + 12m rolling 하방편차 차트 추가 (사용자 요청).
2. **TEST/HOLD-OUT 분리** — 모든 페이지에서 `2010-01~2023-12 (168m, TEST)` vs
   `2024-01~2025-12 (24m, HOLD-OUT)` 토글 가능.
3. **mat_eq_eq_lam_pap Top 1** — Home/Risk-Profile 보수형 1순위에 자동 노출
   (사용자 지정).
4. **RMSE Option A** — log-RMSE 본값 보존 + 평균 σ ≈ X%/일 보조 표기 (학술
   표준 호환 + 발표 친숙도 균형).

## 🗂️ 데이터 의존성

본 대시보드는 read-only. 다음 데이터를 import 만 하므로 기존 final/ 산출물에
영향 없음:

| 경로 | 역할 |
|---|---|
| `final/results/*.pkl` | 165 BL 실험 결과 |
| `final/data/monthly_panel.csv` | rf, spy_ret, sector |
| `final/phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` | LSTM 예측 |
| `final/outputs/04_statistics/B3_*.png` | 04 노트북 시각화 (있으면) |

데이터가 없으면 진입점에서 명확한 에러 메시지로 안내합니다.

## ⚙ 캐시 전략

```python
@st.cache_resource     # 165 pkl (메모리 한 번만)
def load_all_results(): ...

@st.cache_data         # master_table (sort_by 변경 시만 재계산)
def get_master_table(sort_by='sortino_TEST'): ...
```

첫 로드 5~8 초 (165 pkl), 이후 모든 페이지 cache hit. 기간/sort 변경
시에만 cache miss 1 회.

## 🐛 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `FileNotFoundError: monthly_panel.csv` | 01_DataCollection 미실행 | `python final/_rebuild_panel_2025.py` |
| `results/ 부족` | 99_run.ipynb 미실행 | 99_run cell-7 실행 |
| HOLD-OUT 페이지 빈값 | pkl 이 2024-12 까지만 | `python final/_extend_bl_to_2025.py` |
| 한글 깨짐 | matplotlib 폰트 미설정 | OS 별 자동 설정됨 (CLAUDE.md 표준) |
| port 8501 이미 사용 | 다른 streamlit 실행 중 | `streamlit run app/streamlit_app.py --server.port 8502` |

## 🔧 개발 메모

- 새 페이지 추가: `pages/N_<emoji>_<이름>.py` 생성 → 자동 탐지
- 페이지 순서 = 파일명 prefix 숫자 (1~6)
- 사이드바 글로벌 옵션은 각 페이지에서 독립 (st.session_state 미사용)
- `st.metric` `delta` 인자로 TEST vs HOLD-OUT 차이 시각화

## 📜 의존 패키지

```
streamlit >= 1.28
pandas    >= 2.0
numpy     >= 1.24
matplotlib >= 3.7
scipy     >= 1.10
nbformat  (선택, _extend_bl_to_2025.py 용)
```

이미 99_run/99_analyze 환경에 모두 포함되어 있어 추가 설치 불필요한 경우
대부분.

## 🏁 시연 시나리오 (3 분)

1. **Home** (30초) — 프로젝트 한 줄 요약 + 페이지 안내
2. **Overview** (40초) — Top 1 KPI hero + TEST vs HOLD-OUT 정합성
3. **Risk-Profile** (40초) — 슬라이더 → Top 3 카드 자동 갱신 시연
4. **Compare** (50초) — mat_eq_eq_lam_pap vs SPY 6 패널 (rolling sortino 강조)
5. **Matrix Heatmap** (20초) — 135 cells 시각화 마무리
