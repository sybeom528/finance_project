# About / FAQ 페이지 — 와이어프레임 (메타만)

> **관련 decisionlog**: `09_about.md`
> **상태**: **부분 확정** (페이지 메타 About M-1~M-4 만 결정, 영역별 = **구현 후 팀 상의**)
> **결정 수**: 8 영역 구조 (메타만)

---

## 페이지 역할 정의

펀드 메타 정보 + FAQ + 학술 부록 — **마지막 페이지로 신뢰성 보강**.

**핵심 콘텐츠**:
1. 펀드 정체성 / 팀 정보
2. FAQ (자주 묻는 질문)
3. 데이터 출처 + 학술 인용 일람
4. ★ Selection Bias / PBO/DSR 학술 부록 (Q-B3 결정에서 이동)
5. Disclosure 자세한 버전 (Footer 단순 버전 의 상세)
6. Contact (선택)

---

## 페이지 영역 구조 (8 영역 — 메타만 확정)

```
1. Header                       (Overview 동일)
2. Sub-header                   (페이지 컨텍스트) ← 영역별 자세 = 팀 상의
3. 펀드 소개 / 팀 정보          (브랜딩 narrative) ← 영역별 자세 = 팀 상의
4. FAQ                          (10-15개 표준 질문) ← 팀 상의
5. 데이터 출처 + 학술 인용 일람  (Methodology 보완) ← 영역별 자세 = 팀 상의
6. ★ Selection Bias 학술 부록    (Expander + 학술 인용) ← 메타 확정
7. Disclosure 자세한 버전        (Footer 보완) ← 영역별 자세 = 팀 상의
8. Footer                       (Overview 동일)
```

---

## 페이지 메타 결정 (About M-1 ~ M-4)

### About M-1. 영역 개수

**결정**: (c) 풍부 8 영역

**근거**:
1. **Selection Bias 학술 부록** (Q-B3) 포함 필수 → 별도 영역 필요
2. 신뢰성 보강 영역들 (펀드 소개 / FAQ / 데이터 출처 / Disclosure) 충분 활용
3. About 페이지는 마지막 페이지 → 분량 부담 ↓

### About M-2. Sub-header

**결정**: (a) 포함 (모든 페이지 일관)

### About M-3. FAQ 깊이

**결정**: (b) 표준 10-15개

**근거**:
1. **균형 (B) 적용** — 청중 부담 ↓ + 정보 충분
2. (a) 5-7개는 핵심 질문 누락 가능
3. (c) 20+ 은 학술 보고서 수준 (마케팅 ↓)

### About M-4. Selection Bias 학술 부록 — 표시 형식

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

---

## 영역 2~7 — 차후 결정 (★ 사용자 결정)

> **사용자 결정 (2026-05-10)**: "영역별 자세한 구축은 앞 범위 모두 구현 완료한 뒤 팀원들과 상의하여 세부 결정 및 작성, 구현"

**Action items** (구현 완료 후 진행):
- 영역 2: Sub-header 텍스트
- 영역 3: 펀드 소개 / 팀 정보 콘텐츠
- 영역 4: FAQ 10-15개 질문 작성 (팀 상의)
- 영역 5: 데이터 출처 + 학술 인용 일람 (Methodology 보완)
- 영역 6: Selection Bias 학술 부록 (Expander 콘텐츠)
- 영역 7: Disclosure 자세한 버전

---

## 영역별 와이어프레임 (메타 + 영역 6, 7 결정사항만)

### 영역 1: Header — Overview 동일

→ `02_common.md` 의 `render_page_header()` 호출

---

### 영역 2: Sub-header (메타만 — 팀 상의)

**결정사항**: (a) 포함 (Performance 패턴 동일)

**텍스트 안 (팀 상의 후 확정)**:
```
About / FAQ
펀드 소개 / 팀 정보 / FAQ / 학술 부록.
```

→ 자세한 텍스트는 구현 후 팀 상의

---

### 영역 3: 펀드 소개 / 팀 정보 (팀 상의)

**결정 차후**:
- 펀드 정체성 narrative
- 팀 구성 / 역할
- Contact 정보
- (K-4 결정: 발표 스크립트 X — 단순 메타 정보로 단순화)

---

### 영역 4: FAQ (10-15개 — 팀 상의)

**결정 차후**:
- 가상 투자자 친화 FAQ
- 학술 정직성 보강 FAQ (HO 부진, Walk-forward, Selection Bias 등)

**예상 FAQ 카테고리**:
1. 펀드 일반 (목적 / 메커니즘 / 운용 가정)
2. 성과 (HO 부진 이유 / 장기 분산 가치)
3. 위험 관리 (LSTM 변동성 예측 / Tail Risk)
4. 학술 토대 (BL / LSTM / Factor)

---

### 영역 5: 데이터 출처 + 학술 인용 일람 (팀 상의 + L-2 결정 자동 적용)

**L-2 결정 적용**: 페이지별 학술 인용 → 00_README.md 학술 근거 일람 일괄 갱신

**갱신할 학술 인용 (00_README 일람)**:
- Markowitz (1952) — Sector Watch 영역 8
- Black-Litterman (1990, 1992) — Methodology 영역 4
- He-Litterman (1999) — Methodology 영역 4
- Idzorek (2005) — Methodology 영역 4
- Hochreiter & Schmidhuber (1997) — Methodology 영역 5 LSTM
- Gers, Schmidhuber, Cummins (2000) — Methodology 영역 5
- Kim & Won (2018) — Methodology 영역 5
- Jensen (1968) — Methodology 영역 6
- Fama-French (1993, 2015) — Methodology 영역 6
- Carhart (1997) — Methodology 영역 6
- Jarque-Bera (1980) — Methodology 영역 7
- Cont (2001) — Methodology 영역 7 fat tail
- Hill (1975) — Risk Metrics 영역 8
- Embrechts, Klüppelberg, Mikosch (1997) — Methodology 영역 7
- Lopez de Prado (2018) — Methodology 영역 5/8 walk-forward
- Engle (2002) — Methodology 영역 8 DCC-GARCH (Expander)
- AQR Frazzini, Israel, Moskowitz (2018) — Overview 영역 4 + Methodology 영역 8
- Bailey & Lopez de Prado (2014) — About 영역 6 PBO/DSR
- Frazzini-Pedersen (2014) — Overview 영역 3 IVW
- Modigliani² (1997) — Risk Metrics 영역 7
- Sharpe (1966, 1994) — 표준 메트릭

**+ F-5 결정 — 운영 가정 박스 추가** (영역 5):
```
펀드 운영 가정:
- 운용보수 (TER): 0%
- Performance Fee: 0%
- 거래비용 (One-way): 20bp
- 매월 rebalancing
- Tax / Slippage 미반영
```

---

### 영역 6: ★ Selection Bias 학술 부록 (Q-B3 — 메타 확정)

**결정사항** (About M-4):
- (d) **Expander + 학술 인용 link**

**Expander 콘텐츠 안 (구현 시 작성)**:

```
▼ Selection Bias / Data Snooping (학술 부록) [Expander]

학술 정의:
- Selection bias: 156 config 중 Top 1 선정 시 발생 가능한 통계 오류
- Data snooping bias: 동일 데이터로 다중 가설 검정 시 우연한 outperformance 가능

우리 펀드의 한계 인정:
- 156 config 평가 → Top 1 (mat_eq_eq_raw_pap) 선정 절차
- True out-of-sample = HOLD_OUT 24m 만 (config selection 시 미사용)
- 학술 정직성 — selection bias 가능성 명시

PBO (Probability of Backtest Overfitting) 적용 방안:
- Bailey & Lopez de Prado (2014) 의 PBO 측정 방법
- 우리 펀드: PBO ≈ X% (계산 가능 시 추가)
- → PBO < 0.5 = 강건성 ↑

DSR (Deflated Sharpe Ratio) 적용 방안:
- Multi-config 검정의 보정 Sharpe
- 우리 펀드: DSR = X (계산 가능 시 추가)

학술 인용:
- Bailey, D.H. & Lopez de Prado, M. (2014).
  "The Deflated Sharpe Ratio: Correcting for Selection Bias,
  Backtest Overfitting, and Non-Normality."
  Journal of Portfolio Management.
- Lopez de Prado, M. (2018).
  "Advances in Financial Machine Learning."
  Wiley.
```

**구현 체크리스트**:
- [ ] `st.expander("Selection Bias / Data Snooping 학술 부록")`
- [ ] 학술 인용 link (Google Scholar / DOI)
- [ ] PBO / DSR 수치 = 구현 후 계산 가능 시 추가

---

### 영역 7: Disclosure 자세한 버전 (팀 상의 + I-3 결정 자동 적용)

**I-3 결정**: (b) 표준 (+ Risk factors)

**About 영역 7 콘텐츠** (I-3 결정):
1. 학술 가상 펀드 표준 disclaimer
2. FINRA Rule 2210 부합 표현
3. 한국 금감원 표준 표현
4. **Risk factors** (5가지):
   - Market risk (시장 변동성)
   - Sector concentration risk (sector 분산 한계)
   - Model risk (LSTM / BL 모델 한계)
   - Data risk (yfinance 데이터 정확성)
   - Backtest overfitting risk (Methodology 영역 8 / About Selection Bias 부록 참조)

**I-1 표준 표현**:
```
※ Past performance is not indicative of future results.
※ 과거의 운용성과는 미래의 수익을 보장하지 않습니다.
※ 본 결과는 백테스트 시뮬레이션이며 실제 운용 성과를 보장하지 않습니다.
※ 본 자료는 투자권유를 목적으로 작성되지 않았습니다.
```

**구현 체크리스트**:
- [ ] 표준 disclaimer 텍스트 (I-1)
- [ ] Risk factors 5가지 (각각 expander 또는 카드)
- [ ] Methodology 영역 8 / About 영역 6 navigation link

---

### 영역 8: Footer — Overview 동일

→ `02_common.md` 의 `render_footer()` 호출

---

## 페이지 데이터 의존성

- (영역 6) 156 config 결과 (PBO 계산 시 — 구현 후 추가)
- (영역 5) 학술 인용 일람 (00_README 갱신 자동)

---

## 메트릭

- (영역 6) PBO (Probability of Backtest Overfitting)
- (영역 6) DSR (Deflated Sharpe Ratio)

---

## 인터랙션 / 토글 적용

| 영역 | 사이드바 토글 영향 | Q-Zoom |
|---|---|---|
| 모든 영역 | ✗ (메타 페이지) | ✗ |
| 영역 6 | Expander 펼침/접기 + 학술 인용 link click | ✗ |

---

## 페이지 구현 우선순위

- **Phase 3 (검증, 1-2주)**: About 페이지 (메타만 — Phase 3 마지막)
- **구현 완료 후 팀 상의 단계** (★ 사용자 결정): 영역 2~7 영역별 자세한 결정

---

## 결과 / 함의

- **메타 결정만 확정** — 영역별 자세한 결정은 **구현 후 팀 상의**
- 8 영역 구조 = 명확한 와이어프레임 (메타)
- Selection Bias 부록 (Q-B3) 위치 = 영역 6 확정
- L-2 결정 적용 → 00_README 학술 근거 일람 일괄 갱신
- F-5 운영 가정 박스 → 영역 5 자동 추가
- I-3 Risk factors 5가지 → 영역 7 자동 적용

---

## 다음 단계 (구현 완료 후)

1. 팀 상의로 영역 2~7 영역별 자세한 결정
2. FAQ 10-15개 질문 작성
3. 펀드 소개 / 팀 정보 작성
4. PBO / DSR 수치 계산 (영역 6)
5. 학술 인용 일람 검증 (영역 5)
6. Risk factors 5가지 자세한 작성 (영역 7)

---

[← 08_backtesting.md](08_backtesting.md) | [04_implementation_steps.md →](../04_implementation_steps.md)
