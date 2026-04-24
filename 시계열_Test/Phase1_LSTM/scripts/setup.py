"""Phase 1 — 환경 부트스트랩 (한글 폰트·시드·경로 상수).

이 모듈은 Phase 1의 **모든 노트북·스크립트가 공통으로 사용**하는
환경 설정을 정의한다. 단일 진실원(single source of truth) 역할을 한다.

사용 예시
---------
# 노트북 / 다른 .py 파일에서:
from scripts.setup import (
    setup_korean_font, fix_seed, SEED,
    BASE_DIR, RESULTS_DIR, RAW_DATA_DIR, SETTING_A_DIR, SETTING_B_DIR,
)
font_used = setup_korean_font()
fix_seed(SEED)

# 또는 원샷 부트스트랩:
from scripts.setup import bootstrap
font_used = bootstrap()

설계 원칙
---------
- BASE_DIR 은 `__file__` 기반으로 잡아 import 위치에 무관하게 안정적이다.
- 시드는 Python·NumPy·PyTorch 모두 고정하고 `torch.use_deterministic_algorithms(True, warn_only=True)`
  로 가능한 결정성을 추구한다.
- 모든 public 인터페이스에 type hints + Numpy-style docstring 필수.
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
# scripts/setup.py 의 부모(scripts/) 의 부모(Phase1_LSTM/) 가 BASE_DIR
BASE_DIR: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = BASE_DIR / 'results'
RAW_DATA_DIR: Path = RESULTS_DIR / 'raw_data'
SETTING_A_DIR: Path = RESULTS_DIR / 'setting_A'
SETTING_B_DIR: Path = RESULTS_DIR / 'setting_B'


def ensure_result_dirs() -> None:
    """결과 저장용 디렉토리들이 없으면 생성한다 (있으면 무시)."""
    for d in (RAW_DATA_DIR, SETTING_A_DIR, SETTING_B_DIR):
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 시드 고정
# =============================================================================
SEED: int = 42  # 팀 내 표준 시드 (관행 — The Hitchhiker's Guide to the Galaxy)


def fix_seed(seed: int = SEED) -> None:
    """Python·NumPy·PyTorch 난수 시드를 고정하고 결정적 알고리즘을 강제한다.

    Parameters
    ----------
    seed : int, default=SEED
        고정할 시드 값. 기본값 42는 팀 내 표준이다.

    Notes
    -----
    GPU CUDA 비결정적 연산, 멀티스레드 BLAS, OS·Python 버전 차이로
    완벽한 비트 단위 재현은 어렵다. 메트릭은 소수점 4자리까지만 비교한다.

    `torch.use_deterministic_algorithms(True, warn_only=True)` 로 일부 연산이
    결정적이지 않을 때 에러 대신 경고만 발생시켜 학습이 멈추지 않도록 한다.
    """
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
        # CUDA 결정성 요구사항 (CUBLAS 워크스페이스 분리)
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
    """matplotlib에서 한글이 깨지지 않도록 폰트를 OS별로 설정한다.

    Returns
    -------
    str
        설정에 사용된 폰트 이름.

    Notes
    -----
    Linux 환경에서 `koreanize_matplotlib` 패키지가 없으면 다음 명령으로 설치한다.

        pip install koreanize-matplotlib --break-system-packages

    `axes.unicode_minus = False` 설정으로 마이너스 기호(−)가
    □ 박스로 깨지는 현상을 OS 무관하게 방지한다.
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
            print('[setup] 경고: koreanize_matplotlib이 설치되어 있지 않습니다.')
            print('       pip install koreanize-matplotlib --break-system-packages')
            font_name = plt.rcParams['font.family']

    plt.rcParams['axes.unicode_minus'] = False
    return font_name


# =============================================================================
# pandas / matplotlib 표시 옵션
# =============================================================================
def apply_display_defaults() -> None:
    """pandas·matplotlib 의 기본 표시 옵션을 적용한다.

    수익률은 소수점 6자리까지 보이도록 하고, matplotlib 기본 figure 크기를
    가독성 있는 (10, 4) 로 설정한다.
    """
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
    """환경을 한 번에 부트스트랩한다 — 폰트·시드·디렉토리·표시옵션.

    Parameters
    ----------
    seed : int, optional
        시드. None 이면 모듈 상수 SEED 사용.
    verbose : bool, default=True
        부트스트랩 결과 한 줄 박스를 출력할지 여부.

    Returns
    -------
    str
        설정에 사용된 폰트 이름.
    """
    if seed is None:
        seed = SEED

    font_used = setup_korean_font()
    fix_seed(seed)
    ensure_result_dirs()
    apply_display_defaults()

    if verbose:
        print('=' * 60)
        print('  Phase 1 — 환경 부트스트랩 완료')
        print('=' * 60)
        print(f'  한글 폰트  : {font_used}')
        print(f'  시드       : {seed}')
        print(f'  결과 경로  : {RESULTS_DIR}')
        print('=' * 60)

    return font_used
