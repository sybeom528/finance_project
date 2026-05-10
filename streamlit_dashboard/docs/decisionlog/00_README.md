# Streamlit 대시보드 의사결정 로그 — 인덱스

> **프로젝트**: Adaptive VolControl Fund — Streamlit 펀드 홍보 대시보드
> **모델**: `mat_eq_eq_raw_pap` (별도 기준으로 선정된 Top 1)
> **작성 목적**: 모든 의사결정의 배경 / 근거 / 결과 (함의) 를 명확히 기록하여 향후 검토 가능하게 함
> **작성 시작**: 2026-05-10
> **분리 일자**: 2026-05-10 (단일 7,469 줄 파일 → 11개 파일 → 12개 파일 [D~L 분리])

---

## 결정 기록 형식

각 결정은 다음 4요소로 기록합니다:

1. **배경 (Context)**: 왜 이 결정이 필요했는지, 어떤 상황에서 결정이 이루어졌는지
2. **검토된 옵션 (Options)**: 어떤 대안들이 있었는지
3. **선택 + 근거 (Decision + Rationale)**: 무엇을 선택했고 왜 선택했는지 (학술/실무 근거 포함)
4. **결과 / 함의 (Outcome / Implications)**: 이 결정으로 인해 무엇이 결정되었고, 다른 결정에 어떤 영향을 미치는지

---

## 메타 컨텍스트

### 프로젝트 배경

본 대시보드는 BL (Black-Litterman) + LSTM 변동성 예측 모델로 구축한 가상 펀드의 **홍보용 대시보드** 입니다.

**선행 작업의 폐기**:
- 이전 단계 `06_Top1_Selection.ipynb` 분석 (다중 메트릭 + lexicographic + Decision Matrix + sensitivity test) 은 사용자에 의해 **폐기** 되었습니다 (commit `cb899bd`).
- 사용자는 별도 기준으로 `mat_eq_eq_raw_pap` 을 최종 Top 1 으로 결정.
- 본 대시보드는 이 모델 기준으로 구축됩니다.

**펀드 모델 핵심 메트릭** (참조용):

| 메트릭 | TEST 168m | HOLD_OUT 24m | 비고 |
|---|---|---|---|
| sortino | (별도 산출) | 0.685 | HO 부진 |
| sharpe | (별도 산출) | 0.397 | HO 부진 |
| cagr | (별도 산출) | +8.3% | SPY +21.2% 대비 열위 |
| mdd | (별도 산출) | -8.3% | SPY -7.6% 와 유사 |
| vol | (별도 산출) | 10.3% | — |

### 레퍼런스 이미지 (사용자 제공 총 7장)

- **PortfolioX360** 스타일 다크 테마 대시보드 (Risk Metrics + Sector Watch 화면 2장)
- **HOLD_OUT 24m 메트릭 비교 표** (펀드 후보 6개 비교)
- **PortfolioVisualizer** 스타일:
  - Annual Returns (연도별 펀드 vs 벤치마크 막대)
  - Active Return (연도별 vs 벤치마크 차이값)
  - Rolling Active Return + Tracking Error (36m)
  - Risk and Return Metrics 종합 표
  - 분포 통계 + Withdrawal Rate
  - Stress Periods (역사적 위기)

---

## 파일 인덱스

| 파일 | 주제 | 결정 갯수 | 상태 |
|---|---|---|---|
| [01_meta_A_B_C.md](01_meta_A_B_C.md) | A 청중 + B 브랜딩 + C-1 메타 + C-2 메트릭 풀 | A: 4, B: 5, C-1 메타: 1 (8 페이지 목록), C-2: 6 | 확정 |
| [02_overview.md](02_overview.md) | C-1-1. Overview 페이지 (영역 1~6) | 영역 6개 | 확정 |
| [03_performance.md](03_performance.md) | C-1-2. Performance 페이지 (영역 1~9) + 페이지 메타 (M-1~M-4) | 메타 4 + 영역 9 | 확정 |
| [04_risk_metrics.md](04_risk_metrics.md) | C-1-3. Risk Metrics 페이지 (영역 1~9) + 페이지 메타 (Risk M-1~M-4) | 메타 4 + 영역 8 | 확정 |
| [05_holdings.md](05_holdings.md) | C-1-4. Holdings 페이지 (영역 1~8) + 페이지 메타 (Hold M-1~M-4) | 메타 4 + 영역 7 | 확정 |
| [06_sector_watch.md](06_sector_watch.md) | C-1-5. Sector Watch 페이지 (영역 1~9) + HO 정당화 narrative (영역 8) | 메타 4 + 영역 8 | 확정 |
| [07_methodology.md](07_methodology.md) | C-1-6. Methodology 페이지 (영역 1~9) + Factor 분석 / LSTM walk-forward / 5→3 한계 | 메타 4 + 영역 8 | 확정 |
| [08_backtesting.md](08_backtesting.md) | C-1-7. Backtesting 페이지 (영역 1~7) + 균형 옵션 (B) + Stress Test 제거 narrative | 메타 4 + 영역 7 (8→7 축소) | 확정 |
| [09_about.md](09_about.md) | C-1-8. About / FAQ 페이지 메타 결정 (영역별 = 구현 후 팀 상의) | 메타 4 + 영역 부분 확정 | 부분 확정 |
| [10_sidebar.md](10_sidebar.md) | C-4 사이드바 구성 + C 섹션 완료 요약 + Q-5 narrative 정정 + Selection Bias 부록 메모 | C4: 5 | 확정 |
| [11_dl_sections.md](11_dl_sections.md) | D~L 섹션 (데이터 / Disclosure / 시뮬 / UX / 디자인 / 컴플라이언스 / 기술 / 스토리 / 한계) | D: 5 / E: 4 / F~L: 미정 또는 부분 확정 | D, E 확정 / F~L 진행 예정 |

---

## 결정 일람 표 (각 섹션별 주요 결정 요약)

### A 섹션 — 청중

| 결정 | 결과 |
|---|---|
| A-1 타겟 청중 | 가상 투자자 (학술 정직성 + 친화적 톤 양립) |
| A-2 사용 시나리오 | 발표 / 자율 탐색 양립 (인터랙션 풍부) |
| A-3 언어 | 한/영 병기 (메트릭명 / 페이지명 / 슬로건 모두) |
| A-4 사용 환경 | 데스크톱 우선 (Streamlit 다크 테마) |

### B 섹션 — 브랜딩

| 결정 | 결과 |
|---|---|
| B-1 펀드명 (영문) | Adaptive VolControl Fund (Alpha 표현 회피) |
| B-2 펀드명 (한글) | 적응형 변동성 제어 펀드 |
| B-3 슬로건 | (확정 슬로건) |
| B-4 색상 팔레트 | Cobalt Blue 계열 다크 테마 |
| B-5 로고 / 심볼 | (확정 심볼) |

### C 섹션 — 페이지 구조 + 메트릭 풀

| 항목 | 결과 |
|---|---|
| C-1 페이지 진행 방식 | 페이지별 한 대화당 하나씩 (8 페이지) |
| C-1-1 ~ C-1-8 | 8 페이지 모두 확정 (영역 6+9+8+8+8+8+7+8) |
| C-2 메트릭 풀 | 53개 메트릭 / 9 카테고리 확정 |
| C-3 Hero KPI | Overview 영역 2 에서 결정 |
| C-4 사이드바 | 5 그룹 + 2 토글 → **6 그룹** (Investment Simulator 추가, F 섹션 결정) |
| **F-6 Investment Simulator (신규 페이지)** | **9번째 페이지 추가 (체험 그룹)** |
| C-5 사이드바 추가 | (전체 페이지 결정 후) |

### D~L 섹션 — 진행 중 (11_dl_sections.md)

| 섹션 | 주제 | 상태 |
|---|---|---|
| D | 데이터 준비 (Data Layer) — 핵심 복사 + 참조 / yfinance 매핑 / 함수별 캐싱 / 정적 / Startup check | ✅ 확정 |
| E | HO 24m 부진 disclosure 전략 — HOLD_OUT 24m (2024-2025) 통일 / 자신감 톤 + 한계 인정 / Footer 통일 / Net CAGR + Sortino | ✅ 확정 |
| F | 시뮬레이션 레이어 — TER 미사용 + One-way TC 20bp 결정됨 (C-2-4) | 부분 확정 |
| G | 인터랙션 / UX — 인터랙션 일관성 원칙 결정됨 | 부분 확정 |
| H | 디자인 / UX 디테일 — 색상 팔레트 (B-4) 확정 | 부분 확정 |
| I | 컴플라이언스 / Disclosure | 미정 |
| J | 기술 스택 + 배포 | 미정 |
| K | 차별화 포인트 (Storytelling) | 부분 확정 |
| L | 한계 명시 — Methodology 영역 8 (3개 차원) 확정 | 부분 확정 |

---

# 부록: 결정 원칙

## 일관성 원칙

1. **언어**: 한/영 병기 (메트릭명 / 페이지명 / 슬로건 모두) — A-3 결정
2. **색상**: Cobalt Blue 팔레트 준수 (모든 페이지) — B-4 결정
3. **톤**: 학술 정직성 + 가상 투자자 친화 양립 — A-1 결정
4. **HO 부진 처리**: E 섹션 결정 후 모든 페이지 일관 적용
5. **메트릭 풀 일관성**: C-2 확정 풀 외 메트릭 추가 시 별도 결정 필요
6. **인터랙션 일관성** (Q-Zoom 결정): 모든 시계열 차트에 다음 zoom 패턴 일관 적용
   - Plotly 기본 X축 zoom (자동)
   - Range slider (필요 시)
   - 클릭 → expand (시간/Regime/종목/섹터 단위)
7. **다중 벤치마크 토글 일관성**: 사이드바 토글 (SPY/EW/IVW) 이 모든 비교 차트 + KPI delta 에 동시 영향

## 회피 원칙

1. **회피**: "Alpha" 무비판적 사용 — 검증되지 않은 alpha 표현 (B-1 근거)
2. **회피**: "Guard / Shield / Defense" 등 수동적 표현 (B-1 사용자 요청)
3. **회피**: 학술적 부정확성 — Sortino vs Sharpe 혼동 등
4. **회피**: 펀드 정체성과 거리 먼 메트릭 — Withdrawal Rate, 배당 등 (C-2-3 근거)
5. **회피**: β 음수 시 Treynor 표시 — 혼동 위험 (C-2-1 근거)

## 검증 원칙

1. 각 결정은 **배경 + 옵션 + 선택 + 근거 + 결과/함의** 형식으로 기록
2. 다음 결정에 영향 미치는 결정은 **결과/함의** 항목에 명시
3. 미정 항목은 **상태: 미정** 표기
4. 학술/실무 근거 명시 (논문 / 규제 / 표준)

## 학술/실무 근거 일람 (L-2 결정 적용 후 갱신)

### 분산 투자 / 평균-분산 이론

| 주제 | 근거 | 위치 |
|---|---|---|
| 평균-분산 이론 | Markowitz (1952) | Sector Watch 영역 8 |
| Low Vol Anomaly | Ang, Hodrick, Xing, Zhang (2006) / Frazzini-Pedersen (2014) | Overview 영역 3 (IVW) |

### Black-Litterman

| 주제 | 근거 | 위치 |
|---|---|---|
| BL 원조 | Black & Litterman (1990, 1992) | Methodology 영역 4 |
| Ω intuition | He & Litterman (1999) | Methodology 영역 4 |
| BL 실무 가이드 | Idzorek (2005) | Methodology 영역 4 |

### LSTM / 변동성 예측

| 주제 | 근거 | 위치 |
|---|---|---|
| LSTM 원조 | Hochreiter & Schmidhuber (1997) | Methodology 영역 5 |
| Forget gate | Gers, Schmidhuber, Cummins (2000) | Methodology 영역 5 |
| 금융 LSTM 적용 | Kim & Won (2018) | Methodology 영역 5 |

### Factor 분석

| 주제 | 근거 | 위치 |
|---|---|---|
| Alpha (CAPM) | Jensen (1968) | Methodology 영역 6 |
| FF3 / FF5 | Fama-French (1993, 2015) | Methodology 영역 6 |
| Carhart 4-factor | Carhart (1997) | Methodology 영역 6 |

### 분포 / 정규성 검정

| 주제 | 근거 | 위치 |
|---|---|---|
| Jarque-Bera | Jarque & Bera (1980) | Methodology 영역 7 |
| Fat tail stylized fact | Cont (2001) | Methodology 영역 7 (LSTM 정당화) |
| Hill estimator | Hill (1975) | Risk Metrics 영역 8 |
| Extreme value theory | Embrechts, Klüppelberg, Mikosch (1997) | Methodology 영역 7 |
| DCC-GARCH (공분산) | Engle (2002) | Methodology 영역 8 (Expander) |

### 거래비용 / 검증

| 주제 | 근거 | 위치 |
|---|---|---|
| Walk-forward validation | Lopez de Prado (2018) | Methodology 영역 5/8 |
| 거래비용 (대형주) | AQR Frazzini, Israel, Moskowitz (2018) | Overview 영역 4 + Methodology 영역 8 |
| 거래비용 보조 | Engle-Ferstenberg (2007) / Korajczyk-Sadka (2004) / Almgren-Chriss (2001) | Methodology 영역 8 |
| Backtest overfitting (PBO/DSR) | Bailey & Lopez de Prado (2014) | About 영역 6 (학술 부록) |

### 위험조정수익 / 표준 메트릭

| 주제 | 근거 | 위치 |
|---|---|---|
| M² (Modigliani²) | Modigliani & Modigliani (1997) | Risk Metrics 영역 7 |
| Active Return / Rolling Return | PortfolioVisualizer 표준 | Performance 영역 5 / 6 |
| Treynor 부적합 (β<0) | CFA Institute / Bodie-Kane-Marcus / AQR fact sheets | Methodology 영역 6 (제외 근거) |

### 규제 / 표준

| 주제 | 근거 | 위치 |
|---|---|---|
| VaR / CVaR | Basel III / EU CRR | Risk Metrics 영역 5 |
| 펀드 카테고리 메트릭 | SEC / FINRA Rule 2210 / 한국 금융감독원 | I 섹션 (Disclosure) |
| 4% Rule (참조) | Bengen (1994) | (사용 안 함, 제외 근거) |

---

[01_meta_A_B_C.md](01_meta_A_B_C.md) →
