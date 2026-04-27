# 논의사항 인덱스

> Phase 1 진행 중 발생하는 설계·결과 해석·개선 방향 등의 논의 기록을 날짜별로 누적합니다.
>
> **파일명 규약**: `YYYY-MM-DD_주제.md`
> **누적 원칙**: 기존 문서 수정 대신 새 파일로 추가 (논의 이력 보존)
> **작성 주체**: 각 팀원 본인 prefix 없이 날짜 기준으로 통합. 본문 시작에 작성자 명시.

---

## 논의 목록 (최신순)

| 날짜 | 파일 | 주제 | 핵심 결론 | 후속 task |
|---|---|---|---|---|
| 2026-04-28 | [결과분석13](2026-04-28_결과분석13.md) | 13차 Run: CEEMDAN H=1d (논문 원설계 재현) | SPY·QQQ 모두 R²_OOS FAIL (-0.92/-0.79). QQQ train R²=-0.003 — 1일 수익률에 학습 가능 패턴 없음. 5d 대비 구조적으로 더 어려운 문제 확인. | 12차(5d) 결과를 Phase 1 채택 결과로 유지. H=21 재실험 또는 BL Step 4 진행 |
| 2026-04-27 | [결과분석12](2026-04-27_결과분석12.md) | 12차 Run: CEEMDAN H=5d, IMF 버그 수정 | SPY·QQQ 동시 PASS. **pred_std/true_std=0.83~0.86 — mean collapse 해소** (9차 0.21 대비 4배). best_epoch=1 비율 0%. | H=21 CEEMDAN 버그 수정 재실험(11차) 또는 5d 결과로 BL 연동 진행 |
| 2026-04-27 | [결과분석11](2026-04-27_결과분석11.md) | 11차 Run: CEEMDAN H=21d, QQQ IMF 수 버그 발견 | SPY R²=+0.127(Phase 1 최고). QQQ는 IMF 수 불일치 버그(10개 고정, 실제 11개)로 R²=-2.789 FAIL | IMF 수 동적 결정으로 버그 수정 → 12차 Run |
| 2026-04-25 | [Run결과분석](2026-04-25_Run결과분석.md) | §8·§9 Run 결과 FAIL 원인 분석 + 개선 3축 (모델 축소 · seq_len 재조정 · 입력 피처 확장) | SPY·QQQ 모두 R²_OOS 관문 FAIL. previous baseline 이 LSTM 보다 Hit Rate 높음 → 모델이 trivial 규칙보다 못 배움. 파라미터/샘플 비 약 2,372배로 과적합 심각 | 모델 축소 + 정규화 강화 + seq_len 축소 + 다변량 입력 재 Run |

---

## 상태 색인 (주제별)

### 🔴 진행 중 논의
- **11차 재실험 (H=21 CEEMDAN, QQQ 버그 수정)**: `02_setting_A_daily21_ceemdan.ipynb` QQQ ticker루프 `n_imfs_ticker` 동적 수정 후 재실행 — BL horizon 정합(21d) 달성 목표. 보류 중.

### 🟢 결론·반영 완료
- **13차 Run 실행 완료 (CEEMDAN 1d)** (2026-04-28): `02_setting_A_daily1_ceemdan.ipynb` Run All 완료. R²_OOS FAIL 확인 — 1d 수익률 예측이 SPY·QQQ ETF에서 구조적으로 불가능함을 검증. Phase 1 채택 결과는 12차(5d) 유지.
- **Phase 1 Run 결과 개선** (2026-04-25 → 2026-04-27): 12차 Run(CEEMDAN 5d)으로 SPY·QQQ 동시 관문 통과 + mean collapse 해소. 12차 결과로 BL 연동 진행 가능 상태.
- **12차 Run 실행 완료** (2026-04-27): `02_setting_A_daily5_ceemdan.ipynb` Run All 완료. 수치 재현 확인.

### ⏸ 장기·보류
- 데이터 풀링 (멀티 티커 공동 학습) — Phase 1.5 또는 Phase 2 이후
- 앙상블 (seed 다중 학습) — 시간 여유 시
- 사전 학습 (pretrain-finetune) — Phase 4+ 또는 별도 연구

---

## 논의 기록 시 체크리스트

새 논의 문서 추가 시:
1. 파일명 `YYYY-MM-DD_주제.md` 규약 준수
2. 문서 상단에 작성자·작성일·배경 명시
3. **사실(What) + 근거(Why) + 대안(How) + 평가(Trade-off)** 4요소 모두 기록
4. 본 인덱스 표에 1줄 추가 (날짜·파일링크·주제·결론·후속 task)
5. 해당 논의가 코드·문서에 반영되면 `상태 색인` 의 "결론·반영 완료" 로 이동

---

## 관련 문서

- [../README.md](../README.md) — 협업 진입점
- [../PLAN.md](../PLAN.md) — Phase 1 전체 계획
- [../재천_WORKLOG.md](../재천_WORKLOG.md) — 작업·판단 일지
- [../scripts_정의서.md](../scripts_정의서.md) — 모듈 API 정의서
- [../학습자료_주의사항.md](../학습자료_주의사항.md) — Study 자료 주의사항
