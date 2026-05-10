"""
lib/validators.py - Startup check (D-5)

앱 시작 시 1회 실행 — 필수 데이터 파일 존재 검증.
파일 누락 시 사용자 친화적 에러 메시지 + 실행 중단 (st.stop()).

참조: docs/plan/01_setup.md 2.5절
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


# 모듈 위치 기준 절대 경로 (cwd 무관)
DASHBOARD_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DASHBOARD_DIR / "data"


# === 필수 데이터 파일 (없으면 앱 중단) ================================
REQUIRED_DATA_FILES: list[Path] = [
    DATA_DIR / "monthly_panel.csv",
    DATA_DIR / "daily_returns.pkl",
    DATA_DIR / "ff5_monthly.csv",
    DATA_DIR / "universe.csv",
    DATA_DIR / "sp500_membership.pkl",  # EW/IVW universe (look-ahead 회피)
    DATA_DIR / "results" / "mat_eq_eq_raw_pap.pkl",  # Top 1 config
]


# === 선택 데이터 파일 (없어도 graceful degradation) ===================
# 회사명 매핑이 없으면 ticker 자체를 회사명으로 사용
OPTIONAL_DATA_FILES: list[Path] = [
    DATA_DIR / "ticker_company_map.csv",
]


def startup_data_check() -> None:
    """
    앱 시작 시 1회 실행. 필수 파일 누락 시 에러 + st.stop().
    선택 파일 누락 시 경고만 표시 후 진행.

    호출 위치: app.py 의 set_page_config() 직후
    """
    # 1. 필수 파일 검증
    missing_required = [p for p in REQUIRED_DATA_FILES if not p.exists()]
    if missing_required:
        st.error("필수 데이터 파일이 누락되었습니다:")
        for p in missing_required:
            st.code(str(p))
        st.error("다음 명령으로 데이터를 준비하세요:")
        st.code("python streamlit_dashboard/scripts/copy_data.py")
        st.stop()

    # 2. 선택 파일 검증 (경고만)
    missing_optional = [p for p in OPTIONAL_DATA_FILES if not p.exists()]
    if missing_optional:
        for p in missing_optional:
            st.warning(
                f"선택 데이터 파일 누락: `{p.name}` — 회사명 대신 ticker 자체로 표시됩니다."
            )
            st.caption(
                "다음 명령으로 생성 가능: "
                "`python streamlit_dashboard/scripts/build_ticker_company_map.py`"
            )
