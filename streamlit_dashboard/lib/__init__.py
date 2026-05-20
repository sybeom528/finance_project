"""
Adaptive VolControl Fund - 대시보드 공통 라이브러리

대시보드 모든 페이지에서 공유하는 모듈 모음.

각 모듈의 역할:
  - colors              : 색상 팔레트 (Cobalt Blue + GICS 11 섹터)
  - tooltips            : 메트릭 정의 dictionary (~30 메트릭)
  - disclosure          : Footer / Disclaimer / Session state
  - page_helpers        : 페이지 헤더 + 서브헤더 + CSS 주입
  - plot_helpers        : Plotly 공통 헬퍼 (Regime 배경 / Event annotation)
  - metric_calculators  : 16 메트릭 계산 함수 (Sortino / VaR / CVaR / Beta 등)
  - data_loader         : Streamlit 캐싱 + 데이터 로딩
  - validators          : Startup check (필수 데이터 파일 검증)
  - insight_generator   : Sim 페이지 카드 그리드 (F-6)
"""
