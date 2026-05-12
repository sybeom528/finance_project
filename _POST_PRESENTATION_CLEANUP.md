# 발표 후 정리 항목 (2026-05-13 기준)

## 즉시 삭제 가능

### 1. results_backup_pre_spy_fix/ (83MB)
- SPY NaN 패치 안전망 (2026-05-13 생성)
- 패치 검증 완료, 더 이상 불필요
- 삭제 명령: `rm -rf results_backup_pre_spy_fix/`

## 발표 후 검토 사항

### 2. .py 모듈 → lib/ 폴더 분리
- 7개 .py 파일 (bl_*.py, master_table.py, analyze_plots.py, lstm_pipeline.py, timeseries_lib.py)
- 필요 작업:
  - lib/ 생성 + 7개 .py 이동
  - 모든 노트북 import 변경 (예: `from bl_runner` → `from lib.bl_runner`)
  - bl_config.py 의 `Path(__file__).parent` → `Path(__file__).parent.parent` 수정
  - 12개 노트북 재실행 + 검증
- 예상 소요: 30분~1시간

### 3. results/_backup_before_unit_fix/ 검토
- 옛 LSTM 단위 버그 (√252 누락) 시기 결과 보관
- CLAUDE.md: "분석엔 사용 금지" 명시
- 삭제해도 안전 (단, git history 미추적이므로 신중)
