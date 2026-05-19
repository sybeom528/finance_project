# 발표 후 정리 항목 (2026-05-13 기준)

## 즉시 삭제 가능

~~### 1. results_backup_pre_spy_fix/ (83MB)~~ — 2026-05-19 삭제 완료

## 발표 후 검토 사항

~~### 2. .py 모듈 → lib/ 폴더 분리~~ — 2026-05-19 완료 (`final_pt/lib/`)
- 7개 .py → `final_pt/lib/` 이동, `__init__.py` 추가 (패키지화)
- 내부 cross-import 3건은 상대 import (`from .bl_functions` 등)
- `Path(__file__).parent` → `.parent.parent` 수정: `bl_config.py`, `bl_runner.py`, `lstm_pipeline.py` (3곳)
- 5개 노트북 import 일괄 갱신 (03a, 03b, 04, 05b, 06)
- import sanity check 통과. 단, 노트북 풀 재실행 검증은 사용자가 직접 수행 필요

~~### 3. results/_backup_before_unit_fix/ 검토~~ — 2026-05-19 삭제 완료 (final/, final_pt/ 양쪽, git 추적되어 있어 복구 가능)
