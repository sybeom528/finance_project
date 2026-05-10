"""
lib/colors.py - 색상 팔레트 (B-4 + H-4 + M3-2 결정 통합)

모든 차트 / 카드 / 배경에서 일관된 색상 사용을 위한 단일 진실 공급원.
HEX 코드 변경 시 이 파일만 수정 → 전체 대시보드에 자동 반영.

참조: docs/decisionlog/11_dl_sections.md G/H 섹션, plan/02_common.md 3절
"""

# === Primary 팔레트 (B-4 결정: Cobalt Blue) ===========================
# 다크 테마 + 차분한 신뢰감 (PortfolioX360 스타일)
COLORS = {
    "primary": "#3B82F6",        # Cobalt Blue - 메인 강조 (KPI / 펀드 라인)
    "accent_green": "#10B981",   # Positive return / 수익 / 성공
    "accent_red": "#EF4444",     # Drawdown / 손실 / 경고
    "accent_amber": "#F59E0B",   # 중간 / 주의
    "background": "#0E1117",     # Streamlit 다크 기본 배경
    "secondary_bg": "#1F2937",   # 카드 / 사이드바 배경
    "text": "#FAFAFA",           # 본문 텍스트 (off-white)
    "text_muted": "#9CA3AF",     # 보조 / 캡션 텍스트
    "border": "#374151",         # 카드 테두리 / 구분선
}


# === 벤치마크 라인 색상 (H-4 결정) =====================================
# 4개 비교군이 동시 표시될 때 한눈에 구분 가능하도록 색상 분리
BENCHMARK_COLORS = {
    "Fund": "#3B82F6",    # Cobalt Blue - 우리 펀드
    "SPY": "#6B7280",     # Gray - 표준 시장 벤치마크
    "EW": "#10B981",      # Green - Equal Weight (펀드 universe 1/N)
    "IVW": "#8B5CF6",     # Purple - Inverse Volatility (Naive Low-vol)
}


# === Regime 배경색 (Overview 영역 3 일관) ==============================
# 시계열 차트 배경에 4개 구간 표시 (회복기 / 확장기 / 변동기 / 홀드아웃)
REGIME_COLORS = {
    "R1": "#1F2937",   # Dark Gray - 회복기 (2010-01 ~ 2012-06)
    "R2": "#0E1117",   # Background - 확장기 (2012-07 ~ 2019-12, 기본 배경)
    "R3": "#1F2937",   # Dark Gray - 변동기 (2020-01 ~ 2023-12)
    "HO": "#374151",   # Lighter Gray - 홀드아웃 (2024-01 ~ 2025-12)
}


# === GICS 11개 섹터 색상 (H-4 결정) ====================================
# Holdings / Sector Watch 페이지의 섹터별 차트 일관성
SECTOR_COLORS = {
    "Information Technology": "#3B82F6",     # Blue
    "Health Care": "#10B981",                # Green
    "Financials": "#F59E0B",                 # Amber
    "Consumer Discretionary": "#EC4899",     # Pink
    "Consumer Staples": "#8B5CF6",           # Purple
    "Industrials": "#06B6D4",                # Cyan
    "Energy": "#EF4444",                     # Red
    "Materials": "#84CC16",                  # Lime
    "Communication Services": "#F97316",     # Orange
    "Real Estate": "#A855F7",                # Violet
    "Utilities": "#14B8A6",                  # Teal
}


# === Methodology 영역 8 한계 카드 색상 ================================
# 5가지 한계 (데이터 / 모델 / HO 부진 / Future work / 실무) 별 강조색
LIMITATION_COLORS = {
    "data": "#3B82F6",         # Blue
    "model": "#8B5CF6",        # Purple
    "ho_decline": "#F59E0B",   # Amber
    "future_work": "#10B981",  # Green
    "practical": "#EF4444",    # Red
}


# === Methodology Sankey 4그룹 색상 (M3-2) =============================
# 데이터 → BL → LSTM → Optimizer 흐름 표현
SANKEY_GROUP_COLORS = {
    "data": "#3B82F6",       # Cobalt Blue
    "bl": "#10B981",         # Green
    "lstm": "#8B5CF6",       # Purple
    "optimizer": "#F59E0B",  # Amber
}
