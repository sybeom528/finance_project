"""Phase 1 — LSTM 단독 베이스라인 재사용 모듈 패키지.

이 패키지의 모든 하위 모듈은 노트북(`01_*` ~ `04_*`)에서
`from scripts.<module> import <name>` 형태로 import한다.

협업 안정성 원칙
----------------
- 모든 public 함수: type hints + Numpy-style docstring 필수
- 인터페이스 변경 시 재천_WORKLOG.md 에 변경 사항 기록
- 모듈은 책임 단위로 분리 (data/target/cv/dataset/model/train/metrics/plot)
"""
