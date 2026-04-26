"""Phase 1.5 — 변동성 예측 분기 재사용 모듈 패키지.

이 패키지의 모든 하위 모듈은 노트북(`01_*` ~ `03_*`)에서
`from scripts.<module> import <name>` 형태로 import 합니다.

분기 특성
---------
- Phase 1 LSTM 의 타깃을 **누적 수익률 → 실현변동성(realized volatility)** 으로 교체
- 입력: ``log_ret²`` univariate (instantaneous variance proxy)
- 평가지표: RMSE on Log-RV · QLIKE · R²_train_mean · MZ regression
- 베이스라인: HAR-RV (Corsi 2009) · EWMA(λ=0.94) · Naive · Train-Mean

파일 출처
---------
- ``setup.py``, ``dataset.py``, ``models.py``, ``train.py`` 는
  ``Phase1_LSTM/scripts/`` 에서 **그대로 복사한 사본** (2026-04-26 시점).
  Phase 1 자산은 격리 보존하기 위해 원본 변경 없이 복사만 수행.
- ``targets_volatility.py`` · ``metrics_volatility.py`` · ``baselines_volatility.py``
  는 Phase 1.5 신규 작성.

협업 안정성 원칙
----------------
- 모든 public 함수: type hints + Numpy-style docstring 필수
- 인터페이스 변경 시 재천_WORKLOG.md 에 변경 사항 기록
- 모듈은 책임 단위로 분리 (setup/dataset/model/train + 변동성 전용 3종)
"""
