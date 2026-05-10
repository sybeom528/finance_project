# A·B·C 메타 + C-2 메트릭 풀

> **파일**: `01_meta_A_B_C.md`
> **결정 시점**: 2026-05-10
> **상태**: 확정 (A 4개 + B 5개 + C-1 메타 + C-2 6개)
> **포함**: A 섹션 (청중) + B 섹션 (브랜딩) + C 섹션 헤더 + C-1 페이지 진행 방식 (메타) + C-2 메트릭 풀 (Universe)

---

# A 섹션 — 발표 맥락 + 청중

> **결정 시점**: 2026-05-10 / **상태**: 확정

## A 섹션 통합 배경 (Context)

대시보드 구축 plan 의 가장 첫 결정은 **누구에게 무엇을 어떻게 보여줄 것인가** 입니다. 청중 / 시간 / 언어 / 환경의 4축이 결정되어야 페이지 분량 / 인터랙션 깊이 / narrative 톤이 정해집니다.

이 4축이 **모든 후속 결정의 전제 조건** 으로 작용합니다.

---

## A-1. 타겟 청중

### 배경

펀드 홍보 대시보드의 타겟이 누구인지에 따라 학술 정확성 vs 직관적 시각화의 비중이 결정됩니다. 청중 유형에 따라 메트릭 노출 방식 / hover tooltip 필요성 / 검증 절차 페이지 필요 여부가 갈립니다.

### 검토된 옵션

| 옵션 | 설명 | 함의 |
|---|---|---|
| (a) 학교 발표 | 교수/조교/동료 | 학술 정확성 강조 |
| (b) 경진대회 | 심사위원 (실무+학계) | 차별화 + 실무 적용성 |
| (c) 가상 투자자 | 일반 투자자 가정 | 직관적 시각화 + 성과 |
| (d) 종합 | 위 3가지 모두 | 페이지 분량 ↑ + 토글 필요 |

### 선택 + 근거

**결정**: (b) 경진대회 + (c) 가상 투자자 중점

**근거**:
1. 본 프로젝트가 학교 최종 프로젝트이자 경진대회 출품용이므로 두 청중 동시 대응 필요
2. 가상 투자자 중점 = 발표 narrative 의 마케팅성 강화 필요
3. 학술 정확성은 별도 페이지 (Methodology / Backtesting) 로 분리 가능

### 결과 / 함의

- 메트릭 hover tooltip **필수** (가상 투자자 이해도 ↑)
- 한/영 병기 필요 (학술 용어 + 직관적 한글)
- 검증 절차 (FF5 alpha, walk-forward, regime 분석) 별도 페이지 분리
- Hero KPI 는 직관적 메트릭 우선 (CAGR / Sortino / MDD / Volatility)
- A-3 (언어), C-1-1 (Hero KPI), I 섹션 (Disclosure) 결정에 영향

---

## A-2. 사용 시나리오

### 배경

발표 시간 + 사용 방식이 결정되어야 페이지 분량과 인터랙션 깊이가 정해집니다. 짧은 데모용이면 MVP 4페이지로 충분하지만, 자유 탐색용이면 전체 8페이지 풀 구축이 필요합니다.

### 검토된 옵션

| 옵션 | 시간 / 방식 | MVP 페이지 수 |
|---|---|---|
| (a) 5분 lightning demo | 발표자 핵심만 시연 | 3-4 페이지 |
| (b) 15분 deep-dive | 페이지 순회 | 6-7 페이지 |
| (c) 자유 탐색형 | 청중 셀프 클릭 | 8 페이지 + 풍부한 인터랙션 |
| (d) 데모 + 자유 탐색 혼합 | 발표 후 링크 공유 | 8 페이지 풀 구축 |

### 선택 + 근거

**결정**: (d) 데모 + 자유 탐색 혼합 + 인터랙션 풍부하게

**근거**:
1. 경진대회 + 가상 투자자 두 청중 동시 대응 시 한 형식으로는 부족
2. 발표 후 URL 공유로 활용도 ↑ (청중이 발표 후에도 탐색 가능)
3. 8 페이지 풀 구축 = 학술 정직성 (Methodology, Backtesting 별도) + 마케팅 (Overview, Performance 강조) 양립

### 결과 / 함의

- **8 페이지 풀 구축** 결정 → C-1 페이지 목록 8개로 확정
- 인터랙션 풍부 결정 → 사이드바에 기간 선택 / 벤치마크 토글 등 추가 (C-5)
- 발표자 동선 + 청중 셀프 탐색 양립 가능한 구조 필요
- J 섹션 (배포) 결정 시 안정적 환경 필요 (Streamlit Cloud 등)

---

## A-3. 언어

### 배경

타겟 청중이 가상 투자자 중점이지만 경진대회 심사위원 (학계 포함) 도 대응해야 합니다. 한국어 단독은 학술 용어 정확성 ↓, 영문 단독은 가상 투자자 친화 ↓.

### 검토된 옵션

| 옵션 | 방식 |
|---|---|
| (a) 한국어 단독 | 가장 간단, 학술 용어 부정확 |
| (b) 영문 단독 | 국제 컨퍼런스용, 한국 청중에 부적합 |
| (c) 한/영 병기 | 핵심 라벨만 영문 (예: "Sortino Ratio (소르티노 비율)") |
| (d) 토글 스위치 | 사이드바 언어 전환 (개발 비용 ↑) |

### 선택 + 근거

**결정**: (c) 한/영 병기

**근거**:
1. 학술 용어 (Sortino, Sharpe, MDD 등) 는 영문이 표준 → 영문 유지
2. 한국어 보조 라벨로 가상 투자자 이해 보조
3. 토글 스위치 (d) 는 개발 비용 ↑ 대비 효과 ↓ → 병기로 통일이 효율적

### 결과 / 함의

- 모든 메트릭명 / 페이지명 / 슬로건에 한/영 병기 적용
- 차트 축 라벨 / hover tooltip 도 한/영 병기 (가능 시)
- 펀드명 (B-1/B-2) 결정 시 영문 + 한글 두 형태 모두 결정 필요
- i18n 라이브러리 불필요

---

## A-4. 사용 환경

### 배경

발표자 데모 only 와 청중 셀프 탐색 only 는 배포 환경 / UX 디자인이 다릅니다. 발표자 데모는 노트북에서 로컬 실행해도 되지만, 셀프 탐색은 안정적 배포 + URL 공유 필요.

### 검토된 옵션

| 옵션 | 설명 |
|---|---|
| (a) 발표자 데모 only | 로컬 실행, URL 공유 안 함 |
| (b) 청중 셀프 탐색 only | URL 공유, 발표자 동선 X |
| (c) 둘 다 | 발표 + 사후 링크 공유 |

### 선택 + 근거

**결정**: (c) 발표자 데모 + 청중 셀프 탐색 (URL 공유)

**근거**:
1. A-2 의 "데모 + 자유 탐색 혼합" 결정과 일관
2. 발표 후 URL 공유로 활용도 ↑
3. 경진대회 후속 평가 시 URL 만으로 검토 가능

### 결과 / 함의

- 안정적 배포 환경 필요 → J 섹션에서 Streamlit Cloud / 자체 서버 결정
- URL 공유 가능한 형태 (private link 또는 public)
- 캐싱 (`@st.cache_data`) 활용으로 청중 다중 접속 시 속도 보장

---

# B 섹션 — 펀드 정체성 (Branding)

> **결정 시점**: 2026-05-10 / **상태**: 확정

## B 섹션 통합 배경 (Context)

가상 투자자 중점 청중 (A-1) 에게 펀드를 홍보하려면 신뢰감 있는 브랜드명 + 일관된 톤이 필수입니다. 펀드명 / 슬로건 / 색상 / 로고 4축은 모든 페이지에서 일관 적용되어야 합니다.

또한 펀드의 학술적 정체성 (BL + LSTM 기반 변동성 통제) 을 정확히 반영하면서, 가상 투자자에게 친근한 표현을 찾아야 합니다.

---

## B-1. 펀드명 (영문)

### 배경

펀드명은 펀드 정체성을 5초 안에 전달하는 핵심 브랜딩 요소. 우리 펀드의 차별화 포인트는:
1. BL (Black-Litterman) 적응형 자산배분
2. LSTM 변동성 예측
3. 4-slot config 의 robust 한 검증

이 중 어떤 측면을 펀드명에 반영할지 결정 필요.

### 검토된 옵션 (1차)

| 카테고리 | 후보 | 강조점 |
|---|---|---|
| 방법론 직설 | BL-LSTM Adaptive Fund | 학술 정확성 |
| Adaptive 강조 | Adaptive Alpha Fund | 변동성 적응형 |
| Regime 강조 | RegimeGuard Strategic Fund | 3-regime 검증 |
| 방어성 강조 | Resilient Alpha Fund | downside risk 관리 |
| Hybrid 강조 | Hybrid Equity Navigator | BL+LSTM 결합 |
| Smart Beta | SmartBeta Adaptive Fund | ETF 친숙 |

### 사용자 피드백 반영

사용자 요청: "방어성보다는 변동성을 줄이는데 초점을 둔 용어"

→ "Guard / Shield / Defense" 등 **수동적 표현 제외**, **능동적 변동성 관리** 표현 위주로 재구성.

### 검토된 옵션 (2차 — 변동성 감소 능동적)

| 후보 | 의미 |
|---|---|
| VolSmart Adaptive Fund | Smart + Adaptive |
| SmoothAlpha Adaptive Fund | Smooth (변동성 ↓) + Alpha |
| Adaptive VolControl Fund | 능동적 변동성 통제 |
| VolNavigator Fund | 변동성 항해 |
| Volatility-Optimized Alpha | 직설적 학술 |

### Alpha 용어 사용에 대한 의문 제기

사용자 질문: "Alpha 가 펀드명에 흔히 들어가는 이유는? 저위험 프리미엄 때문인가?"

**개념 정리**:
- **Alpha** (Jensen 1968) = 시장으로 설명되지 않는 잔여 수익 (CAPM 기반)
- **Low Volatility Anomaly** (Ang et al. 2006, Frazzini-Pedersen 2014) = 저변동성 종목이 위험조정 후 outperform 하는 factor premium
- 두 개념은 **별개** — Alpha 는 운용자 skill, Low Vol Premium 은 systematic factor exposure
- 우리 펀드는 **저변동성 factor + LSTM 예측 활용** = Smart Beta 성격에 가까움
- "Alpha" 는 검증되지 않은 상태에서 마케팅 용어로 흔히 쓰임 → 학술 정직성 우려

### 선택 + 근거

**결정**: `Adaptive VolControl Fund`

**근거**:
1. **Adaptive**: LSTM 기반 적응형 학습 강조 (펀드 메커니즘 정확 반영)
2. **VolControl**: 능동적 변동성 통제 (사용자 요청 반영, 수동적 "Guard" 회피)
3. **Alpha 미사용**: 검증되지 않은 alpha 표현 회피 → 학술 정직성 ↑
4. **균형**: 학술 정확성 (심사위원) + 직관성 (가상 투자자) 양립

### 결과 / 함의

- 슬로건 (B-3) 은 "변동성 예측 기반 적응형 자산배분" 으로 정체성 직접 매칭
- Hero KPI (C-3) 에서 Volatility 메트릭 강조 (펀드 정체성 어필)
- 페이지 narrative 에서 "VolControl" 키워드 일관 사용
- 메트릭 풀 (C-2) 에서 Down Capture / Volatility / Sortino 강조 메트릭 추가

---

## B-2. 펀드명 (한글)

### 배경

A-3 에서 한/영 병기 결정 → 영문 펀드명에 대응되는 한글명 필요.

### 선택 + 근거

**결정**: `어댑티브 볼컨트롤 펀드`

**근거**: 영문 음역 (Adaptive → 어댑티브, VolControl → 볼컨트롤) 으로 일관성 유지. 한국어 의역 ("적응형 변동성 통제 펀드") 은 길어서 가독성 ↓.

### 결과 / 함의

- 모든 페이지 헤더에 한/영 병기 표시 (예: "Adaptive VolControl Fund / 어댑티브 볼컨트롤 펀드")

---

## B-3. 슬로건

### 배경

슬로건은 펀드 정체성을 한 줄로 요약하는 마케팅 핵심 텍스트. Header 영역 (Overview 영역 1) 에서 펀드명 바로 아래 표시.

### 검토된 옵션

| 톤 | 후보 |
|---|---|
| 기술 강조 | "변동성 예측 기반 적응형 자산배분 / Volatility-Aware Adaptive Allocation" |
| 방어성 강조 | "하방 위험을 줄이는 똑똑한 베타 / Smart Beta with Downside Protection" |
| 성과 강조 | "다양한 시장 국면에서 검증된 알파 / Alpha Validated Across Market Regimes" |
| 간결형 | "Adaptive. Resilient. Validated." |

### 선택 + 근거

**결정**: "변동성 예측 기반 적응형 자산배분 / Volatility-Aware Adaptive Allocation"

**근거**:
1. 펀드명 (Adaptive VolControl) 과 직접 매칭
2. 학술적 정확성 — "변동성 예측" (LSTM) + "적응형" (BL) 메커니즘 정확 반영
3. "Alpha" 미사용 — 학술 정직성 일관 (B-1 결정과 부합)
4. 한/영 병기 (A-3 결정 부합)

### 결과 / 함의

- Header 영역 (1-2) 에 대시(—) 로 한/영 구분하여 표시
- 페이지 narrative 의 핵심 메시지로 일관 사용

---

## B-4. 색상 팔레트

### 배경

다크 테마 (레퍼런스 PortfolioX360 일관성) 를 기본으로, 정통 금융 톤 vs fintech 톤 vs 럭셔리 톤 중 선택 필요.

### 검토된 옵션

| 후보 | Primary | 느낌 |
|---|---|---|
| (a) Cobalt Blue | `#3B82F6` | 정통 금융, 신뢰감 |
| (b) Teal & Mint | `#14B8A6` | 현대적, fintech |
| (c) Purple Indigo | `#6366F1` | 독창적, 기술 |
| (d) Gold & Charcoal | `#F59E0B` | 프리미엄 |

### 선택 + 근거

**결정**: (a) Cobalt Blue

**상세 팔레트**:
- Primary: `#3B82F6` (밝은 블루)
- Accent Green: `#10B981` (positive return)
- Accent Red: `#EF4444` (drawdown / negative)
- Background: `#0E1117` (Streamlit 다크 기본)

**근거**:
1. 정통 금융 톤 = 가상 투자자에게 신뢰감 ↑
2. 레퍼런스 PortfolioX360 와 일관성 (사용자가 레퍼런스로 제시)
3. 그린/레드 accent 는 손익 시각화 표준

### 결과 / 함의

- 모든 차트 / 카드 / 배지 색상 위 팔레트 준수
- Streamlit `config.toml` 또는 `st.set_page_config()` 에 적용
- H 섹션 (디자인 디테일) 에서 폰트 / 레이아웃 결정 시 일관성 유지

---

## B-5. 로고 / 심볼

### 배경

펀드 brand identity 강화를 위해 로고 필요. 단, 디자인 작업량 vs 비용 trade-off.

### 검토된 옵션

| 옵션 | 작업량 |
|---|---|
| (a) 텍스트 마크 only | 즉시 |
| (b) 단순 아이콘 + 텍스트 | 1-2시간 |
| (c) 정식 로고 디자인 | 별도 작업 (외부 툴) |

### 선택 + 근거

**결정**: (a) 텍스트 마크 (펀드명 자체를 굵은 폰트 + accent 색)

**근거**:
1. 비용 대비 효과 ↑ — 별도 디자인 작업 불필요
2. 펀드명 자체가 길지 않음 (Adaptive VolControl Fund) → 텍스트 만으로 충분
3. 빠른 구축 가능

### 결과 / 함의

- Header 영역 (1-1) 에서 펀드명을 큰 굵은 폰트 + Cobalt Blue accent 색으로 표시
- 사이드바 상단에도 동일 텍스트 마크 표시

---

# C 섹션 — 페이지 구조 (Architecture)

> **상태**: 진행 중

## C 섹션 통합 배경 (Context)

A 섹션 (8 페이지 풀 구축 + 인터랙션 풍부) 결정에 따라 8 페이지 구조 필요. 각 페이지의 역할 / 콘텐츠 / 시각화를 결정해야 합니다.

페이지 진행 방식은 **한 대화당 한 페이지씩 자세히** 진행하는 방식 채택 (사용자 요청).

---

## C-1. 페이지 진행 방식 (메타)

### 배경

8 페이지 모두 한 번에 결정하면 누락/혼동 위험. 각 페이지마다 영역 (Header / KPI / 차트 등) 이 5-7개 있고, 각 영역마다 결정 항목이 3-5개 → 총 200+ 결정 사항.

### 선택 + 근거

**결정**: 페이지별로 한 대화당 하나씩 자세하게 진행. TodoWrite 로 추적.

**근거**:
1. 누락 방지 — 각 페이지의 모든 영역 결정 후 다음 진행
2. 사용자 피드백 즉시 반영 가능
3. decisionlog.md 작성 시 결정 시점 명확히 기록

### 결과 / 함의

- 8 페이지 결정 = 약 8-10 대화 소요 예상
- 각 대화 후 decisionlog.md 업데이트
- TodoWrite 로 페이지별 진행 상황 추적

### 8 페이지 목록

1. **Overview** — 랜딩 (Hero KPI + 누적수익 + 강점)
2. **Performance** — 성과 분석
3. **Risk Metrics** — 위험 지표
4. **Holdings** — 보유 종목
5. **Sector Watch** — 섹터 분석
6. **Methodology** — 방법론
7. **Backtesting** — 백테스트 검증
8. **About / FAQ** — 펀드 정보

---
## C-2. 메트릭 풀 (Universe)

> **결정 시점**: 2026-05-10 / **상태**: 확정

### C-2 통합 배경 (Context)

Hero KPI / Performance / Risk Metrics 등 모든 페이지에서 사용할 메트릭 풀을 **먼저** 확정해야 페이지별 메트릭 분배 시 일관성 + 누락 방지가 가능합니다.

이전 폐기된 노트북 (`06_Top1_Selection.ipynb`) 에서 검증된 메트릭 + 추가 가능한 메트릭 + 레퍼런스 이미지 (PortfolioVisualizer) 메트릭을 모두 검토합니다.

---

### C-2-1. 추가 메트릭 결정 (Treynor 제외 외)

#### 배경

기본 풀 (Sortino / Sharpe / MDD / Volatility / VaR 등) 외에 학술 표준 / 레퍼런스 이미지에 등장하는 메트릭 추가 검토 필요.

#### 검토된 추가 메트릭

| 메트릭 | 추천 강도 | 학술 근거 |
|---|---|---|
| Arithmetic Mean | ★★ | Volatility Drag 시각화 (CAGR vs 산술평균 차이) |
| R² | ★★ | Beta 의 설명력 (시장 노출 신뢰도) |
| M² (Modigliani²) | ★★ | Modigliani & Modigliani (1997) — Sharpe 의 % 환산 |
| Treynor Ratio | ★ | Sharpe 의 systematic risk 버전 |
| Active Return (연도별 + Rolling) | ★★★ | PortfolioVisualizer 표준 |

#### 선택 + 근거

**결정**: Arithmetic Mean ✓ / R² ✓ / M² ✓ / **Treynor ❌** / Active Return ✓

**근거 — 추가 4개**:
1. **Arithmetic Mean**: CAGR 와의 차이로 Volatility Drag 시각화 가능 → Performance 페이지 보조
2. **R²**: Beta 의 설명력 (시장 노출 신뢰도) → Beta 단독 표시의 한계 보완
3. **M²**: Sharpe 의 % 환산으로 직관성 ↑ (학술 표준)
4. **Active Return**: 레퍼런스 Image 1, 2 의 핵심 차트 → Performance 페이지 메인

**근거 — Treynor 제외**:
1. **β 음수 가능성**: 사용자 명시 — 우리 포트폴리오 β 음수 발생 가능
2. **β < 0 시 Treynor 해석 모호**:
   - $R_p > R_f$ + $\beta < 0$ → Treynor 음수 (해석: "음의 시장노출에서 양의 초과수익")
   - $R_p < R_f$ + $\beta < 0$ → Treynor 양수 (오해 가능)
3. **학술 컨센서스**:
   - CFA Institute: "Treynor is not appropriate when β is close to zero or negative"
   - Bodie, Kane, Marcus (Investments): β < 0 시 Treynor 사용 비추천
   - AQR / DFA fact sheets: long-only equity 펀드에 Treynor 거의 표시 안 함
4. **R² 로 대체 가능**: Beta + R² 표기로 시장 노출 + 신뢰도 충분 전달

#### 결과 / 함의

- Performance 페이지에 Arithmetic Mean 보조 표시 (CAGR 와 함께)
- Risk Metrics 페이지에 R² 추가 (Beta 옆)
- 종합 메트릭 표 (Image 4 스타일) 에 M² 포함
- Treynor 표시 안 함 → β < 0 혼동 방지 + R² 로 대체

---

### C-2-2. VaR 종류 결정

#### 배경

VaR 산출 방식 3종 (Historical / Analytical / Conditional) 중 어떤 것을 사용할지 결정 필요. 각 방식의 장단점 차이.

#### 검토된 옵션

| VaR 종류 | 장점 | 단점 |
|---|---|---|
| Historical VaR (5%) | 분포 가정 X, fat tail 반영 | 샘플 수 적으면 noise ↑ |
| Analytical (Parametric) VaR | 적은 샘플로 산출 가능 | 정규분포 가정 (fat tail 과소평가) |
| Conditional VaR (CVaR) | tail risk 정확 (Basel III 표준) | 계산 약간 복잡 |

#### 선택 + 근거

**결정**: Historical + CVaR (Analytical 제외)

**근거**:
1. **Historical VaR**: 우리 데이터 192개월 샘플 충분 → noise 우려 ↓
2. **CVaR**: Basel III, EU CRR 표준 → 학술/실무 신뢰성 ↑
3. **Analytical 제외**: 정규분포 가정은 금융 수익률에 부적합 (fat tail 과소평가) → 학술 정확성 ↓

#### 결과 / 함의

- Risk Metrics 페이지에 Historical VaR + CVaR 두 개 표시
- 분포 통계 (Pool-9) 의 Skewness / Kurtosis 와 함께 fat tail 시각화

---

### C-2-3. Withdrawal Rate 제외 결정

#### 배경

레퍼런스 Image 5 (PortfolioVisualizer 분포 통계) 에 Safe Withdrawal Rate (SWR) + Perpetual Withdrawal Rate (PWR) 가 포함되어 있어 채택 여부 검토.

#### 메트릭 정의

- **Safe Withdrawal Rate**: 30년간 자산 고갈 없이 인출 가능한 최대 연간 인출률 (Bengen 1994 — 4% rule)
- **Perpetual Withdrawal Rate**: 영구히 자산 가치 유지하며 인출 가능한 비율

#### 사용자 질문

"실무에서도 펀드 성격과 거리 먼 지표는 대시보드 / 홍보물에서 제외하나?"

#### 실무 관행 분석

**펀드 카테고리별 메트릭 매칭 (Vanguard / Fidelity / BlackRock fact sheet 패턴)**:

| 펀드 카테고리 | 강조 메트릭 | 제외 메트릭 |
|---|---|---|
| 성장형 | CAGR, Sharpe, MDD, Sector Allocation | Withdrawal Rate, Yield |
| 배당형 | Yield, Distribution, Yield on Cost | Sortino, MDD (덜 강조) |
| 은퇴형 (TDF) | **Glide Path, Withdrawal Rate**, Income Replacement | Beta, Tracking Error |
| Low Vol / Defensive | **Beta, Down Capture, Sortino, MDD** | Yield |
| Smart Beta | **Factor Exposures, Active Return, IR** | Withdrawal Rate |

#### 우리 펀드 매칭

- Adaptive VolControl Fund = Smart Beta + Low Vol 하이브리드
- Withdrawal Rate 는 **은퇴형 전용 메트릭** → 우리 펀드와 거리 ↑

#### 선택 + 근거

**결정**: Withdrawal Rate 제외 (SWR + PWR 둘 다)

**근거**:
1. **펀드 정체성과 거리**: 은퇴자금 관점 메트릭 → Adaptive VolControl 과 무관
2. **혼란 방지**: 가상 투자자가 잘못된 메트릭으로 펀드 판단 가능
3. **포커스**: 펀드 정체성 narrative 강화
4. **규제 부합**: SEC / FCA / 한국 금감원 — 펀드 카테고리 부적합 메트릭 표시 시 misrepresentation 우려
5. **법적 책임**: FINRA Rule 2210 (마케팅 규제) — 잘못된 메트릭 → 마케팅 위반 사례
6. **실무 표준**: 거리 먼 지표는 표시 안 함 (Vanguard / Fidelity / BlackRock 패턴 일관)

#### 결과 / 함의

- Risk Metrics 페이지의 분포 통계 표에서 Withdrawal Rate 제외
- 동일 원칙으로 다른 거리 먼 메트릭 (예: Yield, Income Distribution) 도 제외
- "펀드 카테고리에 부합하는 메트릭만 사용" 일관성 원칙 (부록 회피 원칙에 등재)

---

### C-2-4. 거래비용 20bp 결정

#### 배경

펀드의 Net 메트릭 (Net CAGR / Net Sharpe / Net Sortino) 산출 시 거래비용 가정 필요. 학술/실무 표준 범위 내에서 우리 펀드에 적합한 수준 결정.

#### 학술 근거

| 문헌 | 산출 비용 (one-way) |
|---|---|
| AQR / Frazzini, Israel, Moskowitz (2018) "Trading Costs of Asset Pricing Anomalies" | 미국 대형주 평균 10-15bp / high-turnover 15-25bp |
| Korajczyk, Sadka (2004) | S&P 500 7bp / 소형주 30bp |
| Engle, Ferstenberg (2007) "Execution Risk" | 일반 ETF/Mutual fund 5-15bp / 액티브 10-30bp |
| Almgren, Chriss (2001) | Implementation Shortfall framework |

#### 실무 데이터

| 펀드 유형 | One-way 거래비용 |
|---|---|
| 대형 패시브 ETF (VTI, SPY) | 2-5bp |
| 대형 액티브 ETF | 5-15bp |
| mid-cap 포함 액티브 펀드 | 15-30bp |
| 소형주 펀드 | 30-50bp |
| 신흥국 펀드 | 50-100bp |

#### 우리 펀드 적용

- Universe: ~822 ticker (대형주 + mid-cap 추정)
- Rebalancing: 월별
- 운용 규모: 가상 — 시장 영향 ↓

#### 검토된 옵션

| bp | 학술/실무 위치 | 정직성 |
|---|---|---|
| 5bp | 대형 패시브 ETF | ↓ (낙관적) |
| 10bp | AQR 학술 평균 | 중도 |
| 15bp | 액티브 ETF 평균 | 균형 |
| **20bp** | mid-cap 포함 액티브 | **보수적** |
| 30bp | 소형주 포함 | 매우 보수적 |

#### 선택 + 근거

**결정**: One-way 20bp

**근거**:
1. **보수적 가정**: AQR 평균 (10-15bp) 보다 보수적 → 학술 정직성 ↑
2. **mid-cap 적용**: 우리 universe 가 mid-cap 포함 가능성 → 15-30bp 범위 내
3. **발표 narrative**: "보수적 거래비용 가정으로 net 성과 시연" 가능
4. **정당화 가능**: 학술 (AQR) + 실무 (액티브 펀드) 모두 부합

#### 결과 / 함의

- 연간 거래비용 = Annual Turnover × 40bp (round-trip)
- Net CAGR / Net Sharpe / Net Sortino 모두 산출
- Performance 페이지에 Gross vs Net 비교 표시 가능
- 운용보수 (TER) 는 미사용 결정 → 거래비용만이 "수수료"

---

### C-2-5. 차트 / 표 형태 (레퍼런스 적용)

#### 배경

PortfolioVisualizer 스타일 7장 이미지 중 어떤 차트/표를 채택할지 결정. 우리 펀드 데이터 (2010-2025, 16년) 에 적용 가능 여부도 검증 필요.

#### 검토된 이미지

| Image | 컨셉 | 평가 |
|---|---|---|
| 1. Annualized Active Return | 연도별 펀드 - SPY 차이값 막대 | ★★★ 채택 |
| 2. Rolling Active Return + Tracking Error | 36m rolling | ★★★ 채택 |
| 3. Portfolio Income | 연도별 배당 | ❌ 제외 (yfinance Adj Close 사용 시 자동 반영) |
| 4. Risk and Return Metrics | 종합 표 (Arithmetic Mean / Geometric Mean / Sharpe / Sortino 등) | ★★★ 채택 |
| 5. 분포 통계 + Withdrawal Rate | Skewness / Kurtosis / VaR / Withdrawal Rate | ★★ 부분 채택 (Withdrawal Rate 제외) |
| 6. Annual Returns 사이드바이 | 연도별 펀드 vs SPY 막대 (절대값) | ★★★ 채택 |
| 7. Historical Stress Periods | Asian Crisis / Russian Default / Dotcom 등 | ❌ 제외 (우리 데이터 시작 2010 → 위기 데이터 없음) |

#### Image 7 대체 — Regime 기준 재구성

#### 배경

사용자 요청: "역사적 위기 구간이 아닌, 우리의 각 regime 기준으로 구성"

#### 검토된 옵션 (Regime 표 형식)

| 옵션 | 형식 | 정보량 |
|---|---|---|
| A | Regime 4개 단순 비교 (4행) | 낮음 |
| B | Regime + 다중 메트릭 (CAGR/Sortino/MDD/Active Return) | 중간 |
| C | Regime + Sub-events (단기 위기 6개) | 높음 |

#### 선택 + 근거

**결정**: 옵션 B (Regime 기준 다중 메트릭 비교 표)

**근거**:
1. **사용자 요청 부합**: 역사적 위기 X, regime 기준 ✓
2. **다중 메트릭 강조**: 각 regime 별 펀드 정체성 (Sortino, MDD) + 시장 비교 (Active Return)
3. **C 옵션 (sub-events) 회피**: 사용자 의도 ("역사적 위기 구간이 아닌") 와 거리
4. **Backtesting 페이지 핵심 표**: 학술 검증 narrative

#### Regime 정의

**결정**: 이전 노트북 정의 그대로 사용

```
R1 회복기  : 2010-01 ~ 2012-06 (30개월)
R2 확장기  : 2012-07 ~ 2019-12 (90개월)
R3 변동기  : 2020-01 ~ 2023-12 (48개월)
HO 홀드아웃: 2024-01 ~ 2025-12 (24개월)
```

**근거**:
1. 이전 검증된 정의 → 학술 일관성
2. R3 → 2023-12 종료 = HO 분리 완전 (학습편향 방지)
3. 12개월 + 24개월 균형

#### 결과 / 함의

- Performance 페이지에 Image 1, 2, 6 차트 추가
- Risk Metrics 페이지에 Image 4, 5 (일부) 종합 표
- Backtesting 페이지에 **Regime 기준 다중 메트릭 표** (Image 7 대체)
- 5개 행 (R1 / R2 / R3 / HO / FULL) × 다중 메트릭 컬럼

---

### C-2-6. 메트릭 풀 최종 (확정)

#### 카테고리별 (총 53개)

##### Pool-1. 수익성 (Performance Returns)
- Cumulative Return (누적수익률)
- CAGR (= Geometric Mean)
- Arithmetic Mean (monthly + annualized) — 보조
- Annualized Rolling Return (1y / 3y / 5y) — **사용자 결정**
- Net CAGR (after TC)

##### Pool-2. 위험조정수익 (Risk-adjusted Return)
- Sortino Ratio
- Sharpe Ratio
- Calmar Ratio
- Information Ratio
- Modigliani² (M²)
- Net Sharpe / Net Sortino (after TC)

##### Pool-3. 위험 (Risk)
- Volatility (annualized)
- Max Drawdown (MDD)
- Downside Deviation
- Historical VaR (5%)
- CVaR (5%, Expected Shortfall)
- Beta
- R²
- Tracking Error

##### Pool-4. 운용 효율성 (Operational)
- Turnover
- Effective N
- Sector HHI
- Number of Holdings
- Single Stock HHI

##### Pool-5. 시장 비교 (Market Comparison)
- Win Rate
- Win/Loss Ratio
- Best Year / Worst Year
- Best Month / Worst Month
- Up Capture Ratio
- Down Capture Ratio
- Active Return (연도별 + Rolling 36m)

##### Pool-6. Factor 분석 (Academic Validation)
- CAPM Alpha (Jensen's alpha)
- FF5 Alpha (Fama-French 5-factor)
- Factor Exposures (β_MKT, SMB, HML, RMW, CMA)

##### Pool-7. Regime별 성과 (Stability)
- Sortino R1 (회복기 30m)
- Sortino R2 (확장기 90m)
- Sortino R3 (변동기 48m)
- Sortino HO (홀드아웃 24m)
- Sortino IR (regime stability)

##### Pool-8. 시뮬레이션 (Fund Operations)
- Transaction Cost (One-way 20bp)
- Net CAGR / Net Sharpe / Net Sortino (after TC)

##### Pool-9. 분포 통계 (Distribution Stats) — 월별 + 일별
- Skewness
- Excess Kurtosis
- Tail Ratio (95th / 5th percentile)

#### 결과 / 함의

- 페이지별 메트릭 분배 가능 (Hero KPI 결정 시 풀에서 picking)
- 누락 방지 + 일관성 확보
- 추가 메트릭 필요 시 별도 결정 (부록 일관성 원칙 적용)


---

[← 00_README.md](00_README.md) | [02_overview.md](02_overview.md) →
