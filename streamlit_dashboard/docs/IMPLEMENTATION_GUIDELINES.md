# Streamlit 대시보드 구현 가이드라인

> **프로젝트**: Adaptive VolControl Fund — 펀드 홍보 대시보드
> **작성일**: 2026-05-10
> **상태**: 구현 시 반드시 준수
> **주체**: 사용자 (`gorhk`) + Claude (자동 생성)

본 문서는 사용자의 6가지 요구사항을 자세히 명시한 구현 가이드라인입니다. 모든 구현 단계에서 반드시 준수합니다.

---

## 1. plan.md 전체 검토 + 정합성 완벽화 이후 구현 시작

### 검토 대상

`streamlit_dashboard/docs/plan/` 13 파일:
- 00_README.md
- 01_setup.md
- 02_common.md
- 03_pages/01_overview.md ~ 09_about.md (9 파일)
- 04_implementation_steps.md
- 05_validation.md

### 검토 항목

| 항목 | 점검 내용 |
|---|---|
| **결정사항 누락** | decisionlog 12 파일의 모든 결정이 plan 에 반영되었는가 |
| **흐름 정합성** | 페이지간 navigation / 의존성 일관성 |
| **논리적 어긋남** | 잘못된 메트릭 / 잘못된 path / 모순된 결정 |
| **안정성 우려** | 외부 의존 / 폰트 / 데이터 / 리소스 |

### 검토 절차

1. Claude 가 13 파일 전체 검토 (Agent 위임 가능)
2. 검토 결과 사용자에게 보고
3. 수정 필요 사항 → 즉시 수정 + 재보고
4. 사용자 승인 후 → 구현 시작

---

## 2. 안정적 구동 최우선

### 안정성 원칙

| 원칙 | 적용 |
|---|---|
| **Streamlit 표준 컴포넌트 우선** | `st.metric`, `st.expander`, `st.dataframe` 등 |
| **외부 라이브러리 최소화** | 8개 라이브러리만 (J-1 결정) |
| **Streamlit 버전 핀** | `>=1.30,<2.0` (J-2) |
| **Fallback chain** | 폰트 (Pretendard → Noto Sans KR → Malgun → sans-serif) |
| **try-except + graceful degradation** | 모든 외부 호출 (yfinance / file IO / CSS) |
| **Startup data check** | 앱 시작 시 데이터 무결성 검증 (D-5) |

### 안정성 검증 절차

각 파일 작성 후:
1. **Syntax 검증** (Python AST parse)
2. **Import 검증** (필요 라이브러리만)
3. **외부 의존 try-except**
4. **사용자 직접 실행** (각 페이지 / 기능)

---

## 3. 모듈 + 함수 형식 / 리소스 최소화

### 모듈 구조 원칙

| 원칙 | 예시 |
|---|---|
| **단일 책임 원칙 (SRP)** | `lib/data_loader.py` = 데이터 로딩만 / `lib/colors.py` = 색상만 |
| **DRY** | 공통 로직 = `lib/*` 함수로 추출 (페이지 X) |
| **느슨한 결합** | 페이지간 직접 import X → `lib/*` 통해서만 |
| **명시적 의존성** | 함수 인자로 의존성 전달 (global state X) |

### 변경 영향 최소화 패턴

```python
# 좋음: 함수 형식 + 명시적 의존성
def load_fund_returns(config_name: str = "mat_eq_eq_raw_pap") -> pd.Series:
    """우리 펀드의 월별 수익률 로딩."""
    return load_pkl_results(config_name).portfolio_return

# 나쁨: 글로벌 변수
FUND_RETURNS = pd.read_pickle(...)  # 모듈 로딩 시 즉시 실행 (변경 시 영향 큼)
```

### 리소스 최소화 체크리스트

- [ ] 함수 1개 = 책임 1개
- [ ] 캐싱 (`@st.cache_data`) = 데이터 / 계산 결과
- [ ] 변경 시 영향 범위 = 해당 함수 + 직접 호출자만
- [ ] 추가 시 = 새 함수 (기존 수정 X)
- [ ] 제거 시 = `git rm` + 호출자 fallback

---

## 4. 각 파일 생성 전 상세 플랜 보고

### 표준 보고 form (필수)

```markdown
## 파일 생성 보고: `[파일 경로]`

### 역할
- 한 줄 설명

### 데이터 출처
- final/data/X.csv → ...
- streamlit_dashboard/data/Y.pkl → ...

### 함수 list (모듈인 경우)
| 함수 | 입력 | 출력 | 캐싱 |
|---|---|---|---|
| `load_X()` | - | DataFrame | `@st.cache_data` |
| `compute_Y(df)` | DataFrame | float | - |

### 의존성
- pandas, numpy, streamlit
- lib/data_loader.py (load_pkl_results)

### 레퍼런스 (plan + decisionlog)
- plan/02_common.md 1.3절
- decisionlog 11_dl_sections.md D-3

### 예상 라인 수
- ~150줄

### 사용자 승인 사항
- [ ] 함수 list 적정?
- [ ] 의존성 적정?
- [ ] 캐싱 전략 적정?
```

### 보고 후 흐름

1. 사용자 검토 + 수정 요청 (필요 시)
2. 사용자 승인
3. Claude: 파일 생성
4. 사용자: 직접 실행 (`streamlit run app.py` 등)
5. 사용자 검토 + 승인
6. 다음 단계 진행

---

## 5. 세부 단계별 사용자 검토 + 승인

### 단계 정의

`plan/04_implementation_steps.md` 의 Phase 1 / 2 / 3 안에서 **세부 단계** 분할:

```
Phase 1 (MVP)
├── 1.1 Setup
│   ├── 1.1.1 폴더 구조 + .streamlit/config.toml
│   ├── 1.1.2 requirements.txt
│   ├── 1.1.3 데이터 복사 스크립트
│   └── 1.1.4 yfinance 회사명 매핑 1회 수집
├── 1.2 lib/* 공통 컴포넌트
│   ├── 1.2.1 lib/data_loader.py
│   ├── 1.2.2 lib/colors.py
│   ├── 1.2.3 lib/disclosure.py
│   ├── 1.2.4 lib/insight_generator.py
│   ├── 1.2.5 lib/plot_helpers.py
│   ├── 1.2.6 lib/validators.py
│   └── 1.2.7 lib/metrics.py
├── 1.3 app.py + 사이드바
├── 1.4 Overview 페이지
└── 1.5 Performance 페이지
```

### 한 단계 = 한 답변 (★ 절대 준수)

- 한 답변 = 1 파일 또는 1 sub-단계 (예: 1.2.1 만)
- 여러 단계 한 번에 진행 ✗
- 단계 완료 후 사용자 검토 + 승인 ✓

### 단계 흐름 표준

```
[Claude] 단계 X.Y.Z 상세 플랜 보고
   ↓
[사용자] 검토 + 수정 요청 / 승인
   ↓
[Claude] 파일 생성 (Write / Edit)
   ↓
[사용자] 직접 실행 / 검토
   - 예: pip install / streamlit run app.py / 페이지 진입
   ↓
[사용자] 승인 / 수정 요청
   ↓
[Claude] 다음 단계 X.Y.(Z+1) 시작
```

---

## 6. compact 대비 영구 기록

### 기록 위치

| 위치 | 내용 |
|---|---|
| `finance_project/CLAUDE.md` | 6가지 원칙 짧은 요약 (자동 로드) |
| `streamlit_dashboard/docs/IMPLEMENTATION_GUIDELINES.md` | 본 파일 (자세한 가이드) |
| `streamlit_dashboard/docs/decisionlog/` | 의사결정 로그 (12 파일) |
| `streamlit_dashboard/docs/plan/` | 구현 plan (13 파일) |

### compact 후 작업 재개 절차

1. CLAUDE.md 자동 로드 → 6가지 원칙 인지
2. 본 파일 읽고 자세한 가이드 확인
3. plan/ 13 파일 + decisionlog/ 12 파일 필요 시 참조
4. **사용자 명시적 승인 없이 새 단계 진행 금지**

---

## 부록 A: 표준 응답 패턴

### 단계 시작 시

```markdown
# 단계 X.Y.Z: [작업명]

## 상세 플랜 보고
- 파일 경로: ...
- 역할: ...
- 데이터 출처: ...
- 함수 list: ...
- 의존성: ...
- 레퍼런스: ...
- 예상 라인 수: ...

## 사용자 승인 부탁드립니다
```

### 단계 완료 후

```markdown
# 단계 X.Y.Z 완료

## 작업 내용
- 생성 파일: ...
- 라인 수: ...

## 사용자 검토 + 실행 안내
1. ...
2. ...

## 다음 단계: X.Y.(Z+1)
- 사용자 승인 후 진행
```

---

## 부록 B: 검증 체크리스트

각 단계 완료 후 사용자가 확인:

- [ ] 파일이 정상 생성되었는가?
- [ ] Syntax 오류 없는가? (Python: `python -m py_compile`)
- [ ] 의존성 import 정상?
- [ ] 함수 단위 테스트 (수동)
- [ ] Streamlit 실행 정상? (`streamlit run app.py`)
- [ ] 페이지 진입 정상?
- [ ] 토글 / 인터랙션 정상?
- [ ] 데이터 무결성 (Startup check)

---

[관련 문서]
- `CLAUDE.md` (프로젝트 글로벌)
- `decisionlog/00_README.md` (의사결정 로그 인덱스)
- `plan/00_README.md` (구현 plan 인덱스)
