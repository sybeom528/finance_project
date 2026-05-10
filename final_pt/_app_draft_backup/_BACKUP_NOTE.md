# Streamlit 초안 백업 (2026-05-08)

## 백업 사유

본 디렉터리는 **첫 시안 Streamlit 대시보드** 입니다. 사용자 결정 (2026-05-08) 으로
**처음부터 단계적 재구축** 하기로 함에 따라 보존 목적으로 이동되었습니다.

원본 위치: `final/app/`
이동 위치: `final/_app_draft_backup/`

## 백업 시점 상태

| 항목 | 값 |
|---|---|
| 총 라인 수 | 약 1,723 라인 |
| 페이지 수 | 6 (Overview / Explorer / Compare / Matrix / Risk-Profile / Volatility) |
| lib 모듈 | 4 (data_loader / plot_helpers / narrative / recommendations) |
| 백업 일자 | 2026-05-08 |
| 백업 사유 | 첫 시안 → 단계적 재구축 결정 |

## 원본 commit 참조 (git history 보존)

| Commit | 설명 |
|---|---|
| `345dbe5` | Phase 1: Streamlit 대시보드 6 페이지 구현 (발표·시연용) |
| `5849a40` | Fix Streamlit dashboard bugs (5건) |
| `a65b6b7` | Merge feature/streamlit-dashboard: Streamlit 대시보드 + 192m BL 확장 |
| `c5dabba` | (현재) 4번째 LSTM 재학습 + BL 156 pkl 갱신 |

## 재구축 시 참고 사항

### 검증된 패턴 (재사용 가치 있음)
- **캐시 전략** (`lib/data_loader.py`):
  - `@st.cache_resource` 로 165 pkl 한 번만 메모리 로드
  - `@st.cache_data` 로 master_table 기간 토글 시만 재계산
- **TEST/HOLD-OUT 분리** (`master_table.EVAL_PERIODS` 활용)
- **Sortino 우선** (`master_table.build_master_table(sort_by='sortino')`)
- **Diebold-Pauly 6 패널 비교** (`lib/plot_helpers.py` 의 plot_compare)

### 알려진 정합성 문제 (재구축 시 해결 필요)
1. **`lib/narrative.py` 28번 줄 모순**:
   ```
   "mat_eq_eq_lam_pap 은 HOLD-OUT (2024-2025 24m) 에서도 안정적 sortino 유지"
   ```
   → 4번째 학습 후 실제 sortino_HOLD_OUT = 0.685 (Top 134/153) 부진. 학습편향 가능성.

2. **`lib/recommendations.py`** 의 `TOP_1_NAME = 'mat_eq_eq_lam_pap'` 강제 우선
   → Top 1 cfg 재검토 task 결과 반영 필요 (등록된 spawn_task 참조).

3. **2025/2026 데이터 갱신 안 됨 시점에 작성** — 4번째 학습 후 streamlit cache 무효화 필요.

### 단계적 재구축 권장 순서
1. **lib/data_loader.py** 부터 (순수 데이터 액세스 레이어)
2. **streamlit_app.py + Home** (최소 기능 검증)
3. **pages/1_Overview** (Top 1 KPI hero — narrative 갱신 필요)
4. **pages/3_Compare** (Sortino 6 패널 비교)
5. **pages/4_Matrix_Heatmap** (135 cells)
6. **pages/2_Experiment_Explorer** (마스터 테이블)
7. **pages/5_Risk_Profile** (Top 1 재검토 결과 반영)
8. **pages/6_Volatility_Stats** (RMSE Option A 보조 표기)

각 단계 완료 시 사용자 검증 후 다음 단계 진행.

## 사용 금지

본 백업 디렉터리의 파일은 **참고 용도** 입니다. 직접 실행 시:
- `streamlit run _app_draft_backup/streamlit_app.py` — 동작은 가능하지만
  Top 1 narrative 모순 + cache 정합성 미검증 상태이므로 발표·시연 부적절
- 재구축 진행 시 `final/app/` 새 디렉터리에서 처음부터 작성 권장
