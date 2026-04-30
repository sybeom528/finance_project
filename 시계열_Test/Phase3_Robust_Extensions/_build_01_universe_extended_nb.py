"""1회용 빌드 스크립트 — 01_universe_extended.ipynb 생성.

Phase 3 Step 1 — Universe & Panel 확장 (2009~2025, 17 년).

서윤범 BL TOP_50 (Sharpe 1.065, 2009-2025) 와의 fair 비교를 위해
OOS 시작을 2009 년으로 맞춤.
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

NB = nbf.v4.new_notebook()
NB.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
}

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))


# ============================================================================
# 헤더
# ============================================================================
md("""# Phase 3 Step 1 — Universe & Panel 확장 (`01_universe_extended.ipynb`)

> **목적**: OOS 2009 시작 기준 17 년 (2009~2025) universe (매년 top 50) +
>          daily_panel (2003~2025, ~22 년) 을 확장·구축한다.

## 배경

Phase 2 의 Phase 3 진입 분석에서:
- **McapWeight 가 실질 1 위** (Sharpe 1.031) — Phase 2 의 6 년 (2020~2025) 표본이 경기 확장기에 편중
- **서윤범 BL TOP_50 (Sharpe 1.065, 2009~2025)** 를 fair 비교하려면 동일 기간 필요
- **Phase 2 의 sampling bias 한계**: 2020 이전 GFC 회복기·정상 강세장 제외

→ OOS 를 2009 로 소급 확장하여 fair 비교 확보.

## 산출물

| 파일 | 내용 |
|---|---|
| `data/universe_top50_history_extended.csv` | 매년 top 50, 2009~2025 |
| `data/daily_panel.csv` (갱신) | 2003~2025 일별 OHLCV 통합 panel |

## 셀 구성

| § | 내용 |
|---|---|
| §1 | 환경 부트스트랩 |
| §2 | Universe 확장 (2009~2025) |
| §3 | Panel 확장 (2003~2025) |
| §4 | 가용성 진단 (coverage, IPO 종목 식별) |
| §5 | 검증 + 시각화 |

⚠️ **첫 실행 시 30~60 분 소요** (yfinance API 다운로드).
   재실행 시 캐시 활용 → 1~2 분.
""")


# ============================================================================
# §1 환경
# ============================================================================
md("""## §1. 환경 부트스트랩""")

code("""%load_ext autoreload
%autoreload 2

import sys, time
from pathlib import Path

NB_DIR = Path.cwd()
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform

from scripts.setup import bootstrap, BASE_DIR, DATA_DIR, OUTPUTS_DIR, PHASE2_DIR

font_used = bootstrap()
print(f'NB_DIR: {NB_DIR}')
print(f'DATA_DIR: {DATA_DIR}')
print(f'PHASE2_DIR: {PHASE2_DIR}')""")


# ============================================================================
# §2 Universe 확장
# ============================================================================
md("""## §2. Universe 확장 (2009~2025, 매년 top 50)

`extend_universe()` — `universe.py` 의 `build_universe_history()` 를 2009~2025 로 확장.

| Phase | 연도 | unique 종목 |
|---|---|---|
| Phase 2 | 2020~2025 | ~74 |
| **Phase 3** | **2009~2025** | ~130~200 (예상)** |

> ⚠️ 약 15분 소요 (캐시 없을 시). 캐시 활용 시 즉시 완료.
""")

code("""from scripts.universe_extended import extend_universe

t0 = time.time()
universe_df = extend_universe(
    start_year=2009,
    end_year=2025,
    n_top=50,
    cache_dir=DATA_DIR,
    out_name='universe_top50_history_extended.csv',
    verbose=True,
)
print(f'\\n⏱️ 소요 시간: {time.time()-t0:.1f}초')
print(f'universe_df.shape: {universe_df.shape}')
print(f'unique 종목: {universe_df["ticker"].nunique()}')
print(f'oos_year 범위: {universe_df["oos_year"].min()} ~ {universe_df["oos_year"].max()}')
universe_df.head(10)""")

code("""# 연도별 universe 종목 수 확인
year_counts = universe_df.groupby('oos_year')['ticker'].count()
print('연도별 universe 크기:')
print(year_counts)

# 신규 IPO 종목 (2009 이후 등장한 종목들)
all_tickers = universe_df['ticker'].unique()
print(f'\\n총 unique 종목 수: {len(all_tickers)}')""")


# ============================================================================
# §3 Panel 확장
# ============================================================================
md("""## §3. Panel 확장 (2003~2025)

`extend_panel_to_2009()` — Phase 2 의 6 년 panel 을 22 년 (2003~2025) 으로 확장.

- **시작**: 2003-12-31 (OOS 2009 - 5 년 IS 여유)
- **끝**: 2026-01-01

> ⚠️ 첫 실행 시 30~60 분 소요 (yfinance API). 이미 존재하면 skip.
""")

code("""from scripts.universe_extended import extend_panel_to_2009

t0 = time.time()
panel_df = extend_panel_to_2009(
    cache_dir=DATA_DIR,
    universe_df=universe_df,
)
print(f'\\n⏱️ 소요 시간: {time.time()-t0:.1f}초')
print(f'panel_df.shape: {panel_df.shape}')
print(f'date 범위: {panel_df["date"].min().date()} ~ {panel_df["date"].max().date()}')
print(f'unique 종목: {panel_df["ticker"].nunique()}')
panel_df.head()""")


# ============================================================================
# §4 가용성 진단
# ============================================================================
md("""## §4. 가용성 진단 (Coverage & IPO 식별)""")

code("""# 종목별 panel 시작 시점
panel_start_per_ticker = panel_df.groupby('ticker')['date'].min().sort_values()
panel_end_per_ticker = panel_df.groupby('ticker')['date'].max()

print('=== 종목별 panel 시작 시점 분포 ===')
print(panel_start_per_ticker.describe())

# IPO 종목 (panel 시작이 2009 이후인 종목 = 신규 IPO 또는 늦은 데이터)
ipo_tickers = panel_start_per_ticker[panel_start_per_ticker >= '2009-01-01']
print(f'\\nIPO 이후 데이터 종목 ({len(ipo_tickers)} 개):')
print(ipo_tickers.head(20))""")

code("""# 연도별 데이터 커버리지 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: 연도별 unique 종목 수 (universe)
year_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('연도별 Universe 종목 수 (top 50)', fontsize=12)
axes[0].set_xlabel('OOS 연도')
axes[0].set_ylabel('종목 수')
axes[0].axhline(50, color='red', linestyle='--', label='목표 50')
axes[0].legend()

# Plot 2: 종목별 panel 시작 연도 분포
start_years = panel_start_per_ticker.dt.year.value_counts().sort_index()
start_years.plot(kind='bar', ax=axes[1], color='darkorange', edgecolor='white')
axes[1].set_title('종목별 panel 시작 연도 분포', fontsize=12)
axes[1].set_xlabel('시작 연도')
axes[1].set_ylabel('종목 수')

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / '01_universe_coverage.png', dpi=100, bbox_inches='tight')
plt.show()
print('시각화 저장 완료')""")


# ============================================================================
# §5 검증
# ============================================================================
md("""## §5. 검증 (Assertions)

| 검증 항목 | 기대값 |
|---|---|
| panel 시작 ≤ 2003-12-31 | OOS 2009 - 5 년 IS |
| panel 끝 ≥ 2025-12-31 | 최근 데이터 포함 |
| oos_year 범위 2009~2025 | 17 년 OOS |
| unique 종목 ≥ 100 | 확장 universe |
""")

code("""# 검증 1: panel 시작 시점
assert panel_df['date'].min() <= pd.Timestamp('2003-12-31'), \\
    f"panel 시작이 너무 늦음: {panel_df['date'].min()}"
print(f'✅ panel 시작: {panel_df["date"].min().date()} (≤ 2003-12-31)')

# 검증 2: panel 끝 시점
assert panel_df['date'].max() >= pd.Timestamp('2025-12-31'), \\
    f"panel 끝이 너무 이름: {panel_df['date'].max()}"
print(f'✅ panel 끝: {panel_df["date"].max().date()} (≥ 2025-12-31)')

# 검증 3: oos_year 범위
assert universe_df['oos_year'].min() == 2009, \\
    f"oos_year 최솟값 오류: {universe_df['oos_year'].min()}"
assert universe_df['oos_year'].max() == 2025, \\
    f"oos_year 최댓값 오류: {universe_df['oos_year'].max()}"
print(f'✅ oos_year: {universe_df["oos_year"].min()} ~ {universe_df["oos_year"].max()}')

# 검증 4: unique 종목 수
n_unique = universe_df['ticker'].nunique()
assert n_unique >= 100, f"unique 종목 수 부족: {n_unique}"
print(f'✅ unique 종목 수: {n_unique} (≥ 100)')

print('\\n🎉 모든 검증 PASS')""")

code("""# 최종 요약
print('=== Phase 3 Step 1 완료 ===')
print(f'universe: {universe_df.shape[0]} 행, {universe_df["ticker"].nunique()} unique 종목, {universe_df["oos_year"].nunique()} 연도')
print(f'panel: {panel_df.shape[0]:,} 행, {panel_df["ticker"].nunique()} 종목, {panel_df["date"].min().year}~{panel_df["date"].max().year}')
print()
print('다음 단계:')
print('  02a_phase15_stockwise_extended.ipynb   (8-way 병렬 종목별 학습)')
print('  02b_phase15_cross_sectional.ipynb      (Cross-Sectional 학습 — C4+Mj5 수정 적용)')""")


# ============================================================================
# 저장
# ============================================================================
NB.cells = cells

OUT_PATH = Path(__file__).parent / '01_universe_extended.ipynb'
nbf.write(NB, str(OUT_PATH))
print(f'✅ 저장 완료: {OUT_PATH}')
