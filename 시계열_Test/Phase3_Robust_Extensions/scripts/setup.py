"""Phase 3 — 환경 부트스트랩 (한글 폰트·시드·경로 상수).

Phase 2 의 setup.py 를 Phase 3 폴더 구조 (data/, outputs/) 에 맞게 적응.
Phase 2 산출물 (universe, ensemble 등) 도 재사용 가능하도록 PHASE2_DIR 추가.

사용 예시
---------
from scripts.setup import bootstrap, BASE_DIR, DATA_DIR, OUTPUTS_DIR
font_used = bootstrap()

설계 원칙
---------
- BASE_DIR 은 `__file__` 기반 (import 위치 무관 안정성).
- 시드 42 (Phase 1, 1.5 와 동일).
- 한글 폰트 OS 자동 분기.
"""
from __future__ import annotations

import os
import platform
import random
from pathlib import Path
from typing import Optional


# =============================================================================
# 경로 상수
# =============================================================================
# scripts/setup.py 의 부모(scripts/) 의 부모(Phase2_BL_Integration/) 가 BASE_DIR
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / 'data'
PRICES_DIR: Path = DATA_DIR / 'prices_daily'
OUTPUTS_DIR: Path = BASE_DIR / 'outputs'

# Phase 1.5 결과 (재사용)
PHASE15_DIR: Path = BASE_DIR.parent / 'Phase1_5_Volatility'
PHASE15_RESULTS_DIR: Path = PHASE15_DIR / 'results'

# Phase 2 결과 재사용 (universe, ensemble 등)
PHASE2_DIR: Path = BASE_DIR.parent / 'Phase2_BL_Integration'
PHASE2_DATA_DIR: Path = PHASE2_DIR / 'data'

# 서윤범 코드 참조
SEOYUN_DIR: Path = BASE_DIR.parent.parent / '서윤범' / 'low_risk'


def ensure_result_dirs() -> None:
    """결과 저장용 디렉토리들이 없으면 생성한다."""
    for d in (DATA_DIR, PRICES_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 시드 고정
# =============================================================================
SEED: int = 42  # Phase 1, 1.5 와 동일 (팀 표준)


def fix_seed(seed: int = SEED) -> None:
    """Python·NumPy·PyTorch 난수 시드 고정 + 결정적 알고리즘 강제."""
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f'[setup] 결정적 알고리즘 활성화 실패: {e}')
    except ImportError:
        pass


# =============================================================================
# 한글 폰트
# =============================================================================
def setup_korean_font() -> str:
    """matplotlib 한글 폰트 OS 자동 분기.

    Windows: Malgun Gothic
    Darwin (Mac): AppleGothic
    Linux: NanumGothic via koreanize_matplotlib
    """
    import matplotlib.pyplot as plt

    os_name = platform.system()
    if os_name == 'Windows':
        font_name = 'Malgun Gothic'
        plt.rcParams['font.family'] = font_name
    elif os_name == 'Darwin':
        font_name = 'AppleGothic'
        plt.rcParams['font.family'] = font_name
    else:
        try:
            import koreanize_matplotlib  # noqa: F401
            font_name = 'NanumGothic'
        except ImportError:
            print('[setup] 경고: koreanize_matplotlib 미설치')
            print('       pip install koreanize-matplotlib --break-system-packages')
            font_name = plt.rcParams['font.family']

    plt.rcParams['axes.unicode_minus'] = False
    return font_name


# =============================================================================
# 표시 옵션
# =============================================================================
def apply_display_defaults() -> None:
    """pandas·matplotlib 표시 옵션."""
    try:
        import pandas as pd
        pd.set_option('display.float_format', lambda x: f'{x:.6f}')
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_columns', 30)
    except ImportError:
        pass

    try:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (10, 4)
        plt.rcParams['figure.dpi'] = 100
    except ImportError:
        pass


# =============================================================================
# 원샷 부트스트랩
# =============================================================================
def bootstrap(seed: Optional[int] = None, verbose: bool = True) -> str:
    """환경 한 번에 부트스트랩 (폰트·시드·디렉토리·표시옵션)."""
    if seed is None:
        seed = SEED

    font_used = setup_korean_font()
    fix_seed(seed)
    ensure_result_dirs()
    apply_display_defaults()

    if verbose:
        print('=' * 60)
        print('  Phase 3 Robust Extensions — 환경 부트스트랩 완료')
        print('=' * 60)
        print(f'  한글 폰트   : {font_used}')
        print(f'  시드        : {seed}')
        print(f'  데이터 경로  : {DATA_DIR}')
        print(f'  Phase 1.5  : {PHASE15_DIR}')
        print(f'  Phase 2    : {PHASE2_DIR}')
        print('=' * 60)

    return font_used
