"""
copy_data.py — final/data/ + final/results/ 에서 대시보드용 핵심 데이터 복사

목적:
  대시보드 (streamlit_dashboard/) 가 final/ 원본 데이터에 의존하지 않도록
  필요한 파일만 streamlit_dashboard/data/ 로 복사한다.

  원본은 read-only 로 유지 (덮어쓰기/수정 절대 불가).
  대시보드는 사본만 사용 → 원본 데이터 무결성 보장.

복사 대상 (D-1 결정):
  - monthly_panel.csv    : rf, spy_ret, sector, log_mcap (월별 패널)
  - daily_returns.pkl    : 822 ticker × 6099 영업일 일별 수익률
  - ff5_monthly.csv      : Fama-French 5-factor (Methodology 페이지)
  - universe.csv         : 833 ticker + gics_sector
  - results/mat_eq_eq_raw_pap.pkl : 우리 펀드 backtest 결과 (Top 1 config)

실행 방법 (어느 디렉터리에서나 OK):
  python streamlit_dashboard/scripts/copy_data.py

참조: docs/plan/01_setup.md 2.1
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


# 스크립트 자신의 위치 기준 경로 (cwd 무관하게 작동)
SCRIPT_DIR = Path(__file__).resolve().parent              # .../streamlit_dashboard/scripts/
DASHBOARD_DIR = SCRIPT_DIR.parent                          # .../streamlit_dashboard/
PROJECT_ROOT = DASHBOARD_DIR.parent                        # .../finance_project/

SOURCE_DATA_DIR = PROJECT_ROOT / "final" / "data"
SOURCE_RESULTS_DIR = PROJECT_ROOT / "final" / "results"
TARGET_DATA_DIR = DASHBOARD_DIR / "data"
TARGET_RESULTS_DIR = TARGET_DATA_DIR / "results"


# 복사 매핑: (source 절대 경로, target 절대 경로, 설명)
COPY_PLAN: list[tuple[Path, Path, str]] = [
    (SOURCE_DATA_DIR / "monthly_panel.csv",
     TARGET_DATA_DIR / "monthly_panel.csv",
     "월별 패널 (rf / spy_ret / sector / log_mcap)"),

    (SOURCE_DATA_DIR / "daily_returns.pkl",
     TARGET_DATA_DIR / "daily_returns.pkl",
     "일별 수익률 (822 ticker × 6099 영업일)"),

    (SOURCE_DATA_DIR / "ff5_monthly.csv",
     TARGET_DATA_DIR / "ff5_monthly.csv",
     "Fama-French 5-factor (월별)"),

    (SOURCE_DATA_DIR / "universe.csv",
     TARGET_DATA_DIR / "universe.csv",
     "Universe (833 ticker + gics_sector)"),

    (SOURCE_DATA_DIR / "sp500_membership.pkl",
     TARGET_DATA_DIR / "sp500_membership.pkl",
     "S&P500 시점별 편입 종목 (EW/IVW universe, look-ahead 회피)"),

    (SOURCE_RESULTS_DIR / "mat_eq_eq_raw_pap.pkl",
     TARGET_RESULTS_DIR / "mat_eq_eq_raw_pap.pkl",
     "우리 펀드 backtest 결과 (Top 1 config)"),
]


def ensure_target_dirs() -> None:
    """대상 폴더 생성 (이미 존재해도 OK)"""
    TARGET_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def copy_one(source: Path, target: Path, description: str) -> bool:
    """
    파일 1개 복사. 원본 부재 시 False 반환 (graceful 처리 — 다른 파일은 계속 시도).

    shutil.copy2 = 메타데이터(수정 시간 등) 보존.
    """
    if not source.exists():
        print(f"  [SKIP] 원본 없음: {source}")
        return False

    # 인간 친화적 크기 표시
    size_mb = source.stat().st_size / (1024 * 1024)

    # 이미 동일 크기 파일 존재 시 → skip (재실행 효율화)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        print(f"  [SKIP] 이미 동일 크기 존재: {target.name} ({size_mb:.2f} MB)")
        return True

    shutil.copy2(source, target)
    print(f"  [OK]   {source.name:30s} -> {target.relative_to(DASHBOARD_DIR)}  ({size_mb:.2f} MB) - {description}")
    return True


def main() -> int:
    print("=" * 72)
    print("Adaptive VolControl Fund - 대시보드 데이터 복사")
    print("=" * 72)
    print(f"  원본:   {SOURCE_DATA_DIR}")
    print(f"          {SOURCE_RESULTS_DIR}")
    print(f"  대상:   {TARGET_DATA_DIR}")
    print()

    ensure_target_dirs()

    success_count = 0
    fail_count = 0

    for source, target, description in COPY_PLAN:
        if copy_one(source, target, description):
            success_count += 1
        else:
            fail_count += 1

    print()
    print("=" * 72)
    print(f"  완료: {success_count}/{len(COPY_PLAN)} 성공, {fail_count} 실패/스킵")
    print("=" * 72)

    # 1개라도 실패 시 비정상 종료 코드 반환 (CI/스크립트 chain 대응)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
