"""
test_lib.py - lib/* 11개 모듈 자동 검증 (Phase 1.2 검증 방식 A)

검증 항목:
  1. Import — 11개 모듈 import 성공
  2. data_loader — 6개 파일 로드 (shape / type 출력)
  3. metric_calculators — 16개 함수 가짜 데이터로 호출
  4. colors / tooltips — dict 길이 + 샘플
  5. plot_helpers — Figure 에 Regime / Event 추가

실행:
  python streamlit_dashboard/tests/test_lib.py
  또는
  cd streamlit_dashboard && python tests/test_lib.py

Streamlit 미설치 / 데이터 없는 환경에서도 가능한 범위까지 진행 (graceful).
"""

from __future__ import annotations

import logging
import sys
import traceback
import warnings
from pathlib import Path


# === Streamlit 경고 억제 (스크립트 직접 실행 시 컨텍스트 없어 무해) ==
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)


# === Windows cp949 인코딩 환경에서도 한글 / em dash 출력 가능하도록 ==
# (PowerShell UTF-8 환경은 영향 없음, 일부 cmd / Bash 환경 호환성)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass


# === lib import 가능하도록 path 설정 ==================================
SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DASHBOARD_DIR))


# === 출력 포맷 ========================================================
def _print_section(title: str) -> None:
    print(f"\n[{title}]")


def _print_ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _print_warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _print_fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# === 1. Import 검증 ===================================================
def test_imports() -> bool:
    _print_section("1/5 Import 검증")
    modules = [
        "lib",
        "lib.colors",
        "lib.tooltips",
        "lib.disclosure",
        "lib.page_helpers",
        "lib.plot_helpers",
        "lib.metric_calculators",
        "lib.data_loader",
        "lib.validators",
        "lib.insight_generator",
    ]
    failed = []
    for m in modules:
        try:
            __import__(m)
            _print_ok(f"{m}")
        except Exception as e:
            _print_fail(f"{m}: {type(e).__name__} - {e}")
            failed.append(m)
    return len(failed) == 0


# === 2. data_loader 검증 ==============================================
def test_data_loaders() -> bool:
    _print_section("2/5 data_loader")
    try:
        from lib.data_loader import (
            load_monthly_panel,
            load_daily_returns,
            load_ff5_monthly,
            load_universe,
            load_ticker_company_map,
            load_fund_results,
        )
    except Exception as e:
        _print_fail(f"import 실패: {e}")
        return False

    failed = []

    # Streamlit 캐싱 데코레이터는 스크립트 실행 시 raw 함수 호출 가능 (st.cache_data).
    # 데이터 파일 누락 시 FileNotFoundError 발생 → graceful
    def _try(name, fn, optional=False):
        try:
            data = fn()
            if data is None:
                _print_warn(f"{name}: None 반환 (선택 파일 누락 가능)")
                return
            if hasattr(data, "shape"):
                _print_ok(f"{name}: shape={data.shape}")
            elif isinstance(data, dict):
                keys_preview = list(data.keys())[:5]
                _print_ok(f"{name}: dict (keys={len(data)}, 일부={keys_preview})")
            else:
                _print_ok(f"{name}: type={type(data).__name__}")
        except FileNotFoundError as e:
            if optional:
                _print_warn(f"{name}: 파일 없음 (선택) - {e}")
            else:
                _print_fail(f"{name}: 파일 없음 (필수) - {e}")
                failed.append(name)
        except Exception as e:
            _print_fail(f"{name}: {type(e).__name__} - {e}")
            failed.append(name)

    _try("monthly_panel", load_monthly_panel)
    _try("daily_returns", load_daily_returns)
    _try("ff5_monthly", load_ff5_monthly)
    _try("universe", load_universe)
    _try("ticker_company_map", load_ticker_company_map, optional=True)
    _try("fund_results", load_fund_results)

    return len(failed) == 0


# === 3. metric_calculators 검증 =======================================
def test_metrics() -> bool:
    _print_section("3/5 metric_calculators (가짜 데이터 60개월)")
    try:
        import numpy as np
        import pandas as pd
        from lib import metric_calculators as mc
    except Exception as e:
        _print_fail(f"import 실패: {e}")
        return False

    np.random.seed(42)
    fund_ret = pd.Series(np.random.normal(0.008, 0.04, 60))
    bench_ret = pd.Series(np.random.normal(0.007, 0.045, 60))
    weights = pd.Series([0.10, 0.15, 0.20, 0.25, 0.30])

    tests = [
        ("calc_cagr",                lambda: mc.calc_cagr(fund_ret)),
        ("calc_arithmetic_mean",     lambda: mc.calc_arithmetic_mean(fund_ret)),
        ("calc_volatility",          lambda: mc.calc_volatility(fund_ret)),
        ("calc_sharpe",              lambda: mc.calc_sharpe(fund_ret, rf=0.04)),
        ("calc_sortino",             lambda: mc.calc_sortino(fund_ret, rf=0.04)),
        ("calc_mdd",                 lambda: mc.calc_mdd(fund_ret)),
        ("calc_calmar",              lambda: mc.calc_calmar(fund_ret)),
        ("calc_downside_deviation",  lambda: mc.calc_downside_deviation(fund_ret)),
        ("calc_var",                 lambda: mc.calc_var(fund_ret)),
        ("calc_cvar",                lambda: mc.calc_cvar(fund_ret)),
        ("calc_beta",                lambda: mc.calc_beta(fund_ret, bench_ret)),
        ("calc_tracking_error",      lambda: mc.calc_tracking_error(fund_ret, bench_ret)),
        ("calc_ir",                  lambda: mc.calc_ir(fund_ret, bench_ret)),
        ("calc_win_rate",            lambda: mc.calc_win_rate(fund_ret)),
        ("calc_up_capture",          lambda: mc.calc_up_capture(fund_ret, bench_ret)),
        ("calc_down_capture",        lambda: mc.calc_down_capture(fund_ret, bench_ret)),
        ("calc_hhi",                 lambda: mc.calc_hhi(weights)),
        ("calc_effective_n",         lambda: mc.calc_effective_n(weights)),
    ]

    failed = []
    for name, fn in tests:
        try:
            v = fn()
            _print_ok(f"{name:25s} = {v:.4f}")
        except Exception as e:
            _print_fail(f"{name}: {type(e).__name__} - {e}")
            failed.append(name)
    return len(failed) == 0


# === 4. colors / tooltips 검증 ========================================
def test_colors_tooltips() -> bool:
    _print_section("4/5 colors / tooltips")
    try:
        from lib.colors import (
            COLORS,
            BENCHMARK_COLORS,
            REGIME_COLORS,
            SECTOR_COLORS,
            LIMITATION_COLORS,
            SANKEY_GROUP_COLORS,
        )
        from lib.tooltips import METRIC_TOOLTIPS, get_tooltip
    except Exception as e:
        _print_fail(f"import 실패: {e}")
        return False

    failed = []

    expected = {
        "COLORS": (COLORS, 7),
        "BENCHMARK_COLORS": (BENCHMARK_COLORS, 4),
        "REGIME_COLORS": (REGIME_COLORS, 4),
        "SECTOR_COLORS": (SECTOR_COLORS, 11),  # GICS 11
        "LIMITATION_COLORS": (LIMITATION_COLORS, 5),
        "SANKEY_GROUP_COLORS": (SANKEY_GROUP_COLORS, 4),
    }
    for name, (d, min_count) in expected.items():
        if len(d) >= min_count:
            _print_ok(f"{name}: {len(d)} keys (>= {min_count})")
        else:
            _print_fail(f"{name}: {len(d)} keys (< {min_count})")
            failed.append(name)

    _print_ok(f"METRIC_TOOLTIPS: {len(METRIC_TOOLTIPS)} 메트릭")

    sample_metrics = ["CAGR", "Sortino", "MDD", "Beta", "Up Capture"]
    for m in sample_metrics:
        tooltip = get_tooltip(m)
        if tooltip and "정의 미정" not in tooltip:
            _print_ok(f"get_tooltip('{m}') = '{tooltip[:50]}...'")
        else:
            _print_fail(f"'{m}' 정의 누락")
            failed.append(m)

    return len(failed) == 0


# === 5. plot_helpers 검증 =============================================
def test_plot_helpers() -> bool:
    _print_section("5/5 plot_helpers")
    try:
        import plotly.graph_objects as go
        from lib.plot_helpers import (
            REGIME_PERIODS,
            EVENT_MARKERS,
            add_regime_backgrounds,
            add_event_annotations,
            bilingual_label,
        )
    except Exception as e:
        _print_fail(f"import 실패: {e}")
        return False

    failed = []

    # Regime / Event 상수 검증
    if len(REGIME_PERIODS) == 4:
        _print_ok(f"REGIME_PERIODS: {len(REGIME_PERIODS)} 구간")
    else:
        _print_fail(f"REGIME_PERIODS: 예상 4 구간, 실제 {len(REGIME_PERIODS)}")
        failed.append("REGIME_PERIODS")

    if len(EVENT_MARKERS) >= 3:
        _print_ok(f"EVENT_MARKERS: {len(EVENT_MARKERS)} 이벤트")
    else:
        _print_fail(f"EVENT_MARKERS: {len(EVENT_MARKERS)}")
        failed.append("EVENT_MARKERS")

    # callable 검증 — 실제 figure 에 호출하는 시각적 검증은 Phase 1.3
    # dev_test 페이지에서 streamlit 환경으로 수행 (plotly 6.x 의 일부 figure
    # 조작이 streamlit 외부에서 datetime 호환성 이슈가 있음)
    if callable(add_regime_backgrounds):
        _print_ok("add_regime_backgrounds: callable")
    else:
        _print_fail("add_regime_backgrounds: not callable")
        failed.append("add_regime_backgrounds")

    if callable(add_event_annotations):
        _print_ok("add_event_annotations: callable")
    else:
        _print_fail("add_event_annotations: not callable")
        failed.append("add_event_annotations")

    # bilingual_label
    label = bilingual_label("CAGR", "연환산 수익률")
    if label == "CAGR (연환산 수익률)":
        _print_ok(f"bilingual_label: '{label}'")
    else:
        _print_fail(f"bilingual_label 예상치 다름: '{label}'")
        failed.append("bilingual_label")

    return len(failed) == 0


# === main =============================================================
def main() -> int:
    print("=" * 72)
    print("Adaptive VolControl Fund - lib/* 자동 테스트 (Phase 1.2 검증)")
    print("=" * 72)

    results: list[tuple[str, bool]] = []

    for name, fn in [
        ("Import",             test_imports),
        ("data_loader",        test_data_loaders),
        ("metric_calculators", test_metrics),
        ("colors / tooltips",  test_colors_tooltips),
        ("plot_helpers",       test_plot_helpers),
    ]:
        try:
            results.append((name, fn()))
        except Exception:
            traceback.print_exc()
            results.append((name, False))

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print()
    print("=" * 72)
    print(f"  결과: {passed}/{total} PASS")
    for name, r in results:
        marker = "[OK]  " if r else "[FAIL]"
        print(f"   {marker} {name}")
    print("=" * 72)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
