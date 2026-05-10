# C-1-8. About / FAQ 페이지

> **파일**: `09_about.md`
> **결정 시점**: 2026-05-10
> **상태**: 부분 확정 (페이지 메타 About M-1~M-4 만 결정, 영역별 = 구현 후 팀 상의)
> **포함**: 페이지 메타 결정 / 8 영역 구조 / 영역 2~7 — 차후 결정 (★ 사용자 결정) / About 페이지 메타 결정 → C-4 사이드바 결정으로 진행

---

## C-1-8. About / FAQ 페이지

> **상태**: 메타 결정만 확정 / 영역별 자세한 결정은 **구현 후 팀 상의**

### About / FAQ 페이지 통합 배경 (Context)

펀드 메타 정보 + FAQ + 학술 부록 — 마지막 페이지로 신뢰성 보강.

**핵심 콘텐츠**:
1. 펀드 정체성 / 팀 정보
2. FAQ (자주 묻는 질문)
3. 데이터 출처 + 학술 인용 일람
4. ★ Selection Bias / PBO/DSR 학술 부록 (Q-B3 결정에서 이동)
5. Disclosure 자세한 버전 (Footer 단순 버전 의 상세)
6. Contact (선택)

### 페이지 메타 결정 (About M-1 ~ M-4)

#### About M-1. 영역 개수

**검토된 옵션**:
- (a) 압축 4 영역
- (b) 표준 6 영역
- (c) 풍부 8 영역

**결정**: (c) 풍부 8 영역

**근거**:
1. **Selection Bias 학술 부록** (Q-B3) 포함 필수 → 별도 영역 필요
2. 신뢰성 보강 영역들 (펀드 소개 / FAQ / 데이터 출처 / Disclosure) 충분 활용
3. About 페이지는 마지막 페이지 → 분량 부담 ↓

#### About M-2. Sub-header

**결정**: (a) 포함

**근거**: 모든 페이지 일관 — 인터랙션 일관성 원칙

#### About M-3. FAQ 깊이

**검토된 옵션**:
- (a) 핵심 5-7개
- (b) 표준 10-15개
- (c) 풍부 20+

**결정**: (b) 표준 10-15개

**근거**:
1. **균형 (B) 적용** — 청중 부담 ↓ + 정보 충분
2. (a) 5-7개는 핵심 질문 누락 가능
3. (c) 20+ 은 학술 보고서 수준 (마케팅 ↓)

#### About M-4. Selection Bias 학술 부록 — 표시 형식

**검토된 옵션**:
- (a) Expander 안에 텍스트만
- (b) 별도 영역 + 학술 narrative
- (c) PDF 다운로드 link 만
- (d) Expander + 학술 인용 link

**결정**: (d) Expander + 학술 인용 link

**근거**:
1. **균형 (B) Q-B3** 부합 — 부드러운 노출 (대시보드 본문에서 회피)
2. **Expander** = 청중이 의도적 클릭 시만 노출
3. **학술 인용 link** = Bailey-Lopez de Prado (2014) PBO/DSR 등 인용

**Expander 콘텐츠 안**:
- Selection Bias / Data Snooping 학술 정의
- 우리 펀드 156 config 평가 → Top 1 선정 절차의 학술 한계 인정
- PBO (Probability of Backtest Overfitting) 적용 방안
- DSR (Deflated Sharpe Ratio) 적용 방안
- 학술 인용: Bailey & Lopez de Prado (2014), Lopez de Prado (2018)

### About / FAQ 페이지 8 영역 구조

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트)
3. 펀드 소개 / 팀 정보          (브랜딩 narrative)
4. FAQ                          (10-15개 표준 질문)
5. 데이터 출처 + 학술 인용 일람  (Methodology 보완)
6. ★ Selection Bias 학술 부록    (Expander + 학술 인용)
7. Disclosure 자세한 버전        (Footer 보완)
8. Footer                       (Overview 동일)
```

### 영역 2~7 — 차후 결정 (★ 사용자 결정)

> **사용자 결정 (2026-05-10)**: "영역별 자세한 구축은 앞 범위 모두 구현 완료한 뒤 팀원들과 상의하여 세부 결정 및 작성, 구현"

**Action items** (구현 완료 후 진행):
- 영역 2: Sub-header 텍스트
- 영역 3: 펀드 소개 / 팀 정보 콘텐츠
- 영역 4: FAQ 10-15개 질문 작성 (팀 상의)
- 영역 5: 데이터 출처 + 학술 인용 일람 (Methodology 보완)
- 영역 6: Selection Bias 학술 부록 (Expander 콘텐츠)
- 영역 7: Disclosure 자세한 버전

### About / FAQ 페이지 결정 결과 / 함의

- **메타 결정만 확정** — 영역별 자세한 결정은 구현 후 팀 상의
- 8 영역 구조 = plan.md 작성 시 명확한 와이어프레임
- Selection Bias 부록 (Q-B3) 위치 = 영역 6 확정
- 다음 단계:
  - C-4 (사이드바 + 페이지 그룹화) 결정
  - D~L 섹션 (데이터 / HO disclosure / 시뮬레이션 / UX / 컴플라이언스 / 기술 / 스토리) 결정
  - plan.md + 페이지 와이어프레임 작성
  - Streamlit 대시보드 구현
  - **구현 완료 후 About 페이지 영역별 자세한 결정 (팀 상의)**

### About 페이지 메타 결정 → C-4 사이드바 결정으로 진행

---


---

[← 08_backtesting.md](08_backtesting.md) | [10_sidebar_dl.md](10_sidebar_dl.md) →
