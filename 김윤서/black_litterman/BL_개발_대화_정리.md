# BL + 모멘텀 개발 대화 정리

> 작성 기준: 2026-04-22 (최종 업데이트: 회의 결정 반영 — 통합 BL + XGBoost Ω 취소) | 관련 파일: Step4~Step5 시리즈

---

## 1. 핵심 질문과 답변

### Q1. 수익률 자기상관이 거의 0인데 모멘텀을 Q로 사용해도 되나?

**결론: 가능하다. 스케일과 측정 대상이 다른 현상이다.**

| 개념 | 스케일 | 대상 | 결론 |
|------|--------|------|------|
| 자기상관 ≈ 0 (EMH) | 일별 | 절대 수익률 예측 | 내일 주가 방향 맞추기 불가능 |
| 모멘텀 효과 | 3~12개월 | 횡단면 상대 순위 | 잘 오른 종목이 계속 잘 오름 |

**근거 논문**: Jegadeesh & Titman (1993) *JF*, Carhart (1993), Asness et al. (2013) *JF*, Fama & French (2008), Moskowitz et al. (2012)

---

### Q2. Lee & Bae (2024) 논문의 2010-2019 상승장 편향 우려

**우리 분석 기간(2016-2025)이 더 robust하다.**

| 기간 | 특성 |
|------|------|
| 2010-2019 (논문) | 역사적 최장 강세장, 금융위기 회복 이후 |
| **2016-2025 (우리)** | **코로나 급락(2020) + 금리인상 하락장(2022) + 관세쇼크(2025) 포함** |

---

### Q3. Ω를 논문과 다르게 구현한 이유

| 방식 | 수식 | 반영 |
|------|------|------|
| 논문 (Lee & Bae) | %pos/%neg 상승일 비율 | 방향만 |
| **우리 구현** | **횡단면 z-score → sigmoid → Idzorek(2004)** | **방향 + 횡단면 강도** |

```
z = (Q_i - mean(Q)) / std(Q)
confidence = 0.10 + 0.80 × (sigmoid(|z|) - 0.5) / 0.5
Ω_ii = ((1 - c_i) / c_i) × τ × Σ_ii
```

> **팀장님 반응**: "Z스코어는 좋네요" — 회의에서 명시적으로 긍정 확인

---

### Q4. SPY 대비 수익률이 낮은가?

**프레이밍**: "본 연구 목적은 시장 초과 수익이 아닌, 투자자 위험 성향(γ)에 최적화된 위험 조정 포트폴리오 구축이다."

| 전략 | 목적 | SPY 대비 | 채권 비중 |
|------|------|---------|---------|
| SPY B&H | 최대 수익 | 기준선 | 0% |
| BL Aggressive | 위험조정 수익 | 유사/상회 가능 | 최대 20% |
| BL Neutral | 균형 | 낮음, Sharpe 개선 | 최대 35% |
| BL Conservative | 하락 방어 | 수익 낮음, MDD 크게 낮음 | 최대 55% |

---

## 2. 모멘텀 가설 학술 근거

| 논문 | 핵심 주장 |
|------|---------|
| Jegadeesh & Titman (1993) *JF* | 3-12개월 과거 승자 → 초과수익 (원조) |
| Carhart (1993) | 4팩터 모델에 모멘텀(UMD) 추가 |
| Fama & French (2008) | EMH 진영에서도 모멘텀 이상현상 인정 |
| Asness et al. (2013) *JF* | **주식·채권·통화·상품 전반에 모멘텀** — 채권 ETF에 Q 적용 근거 |
| Moskowitz et al. (2012) | Time-series momentum |
| Lee & Bae (2024) | BL + 모멘텀 통합 (직접 참고) |

---

## 3. 확장 방안 최종 상태 (회의 결정 반영)

| 확장안 | 상태 | 이유 |
|--------|------|------|
| **A) 듀얼 모멘텀** | ✅ 유지 (Step5A) | 하락장 방어, 팀 합의 |
| ~~B) XGBoost Ω~~ | ❌ **취소** | 팀장님: "얘는 안 됩니다" |
| **C) 상대뷰 (P행렬)** | ✅ 유지 (Step5C) | 이론적으로 가장 정확 |

---

## 4. 통합 BL 설계 (회의 결정: 2단계 MVO → 단일 BL)

### 전환 이유

> "모멘텀으로 하면 그냥 다 하면 안되나요? 채권도 그렇고." — 팀장님

모멘텀 Q는 주식뿐 아니라 채권·대안 ETF에도 동일하게 계산 가능 (Asness et al. 2013 근거).  
→ 2단계로 분리할 이유 없음, **모든 자산을 단일 BL로 통합**.

### 이전 방식(2단계) vs 현재 방식(통합)

| 항목 | ~~2단계 MVO (폐기)~~ | **통합 BL (현재)** |
|------|-------------------|-----------------|
| 구조 | BL(주식) → MVO(자산군 배분) | 단일 BL 전체 |
| Prior | 시가총액 (주식만) | **Risk Parity (전 자산)** |
| Q | 주식만 | **주식 + 채권 + 대안** |
| Omega | Z-score (주식만) | **Z-score (전 자산)** |

### Risk Parity Prior (Maillard et al. 2010)

```
목표: RC_i = w_i × (Σw)_i / σ_p  →  모든 i에 대해 동일 (Equal Risk Contribution)
π = λΣw_rp

한계: 저변동성 자산(채권, GLD)에 높은 비중 → π가 채권에 주식보다 높은 implied return 부여 가능
해결: MVO 단계 max_bond_total 상한으로 보정
```

### 채권 4개 자산 선택 근거

| 자산 | 역할 | 주식과 상관 | 출처 |
|------|------|-----------|------|
| **TLT** (장기국채 20Y+) | flight-to-safety, 디플레 헤지 | −0.35 | Litterman (2003), Ilmanen (2011) |
| **SHY** (단기국채 1-3Y) | 현금성 안전자산, Conservative 방어 | −0.12 | Swensen (2000) Yale Endowment |
| **TIP** (물가연동채) | 인플레 헤지 — AGG 혼합 팩터 분리 | −0.18 | Asness et al. (2012) |
| **GLD** (금) | 지정학적 위기·통화 약세 헤지 | +0.04 | Dalio All Weather |

> AGG 미포함: 듀레이션+신용 팩터 혼합 → SHY+TIP+TLT로 팩터 명확히 분리

### 프로파일별 MVO 제약

| 프로파일 | max 개별주식 | max 섹터 | max 채권합계 |
|---------|-----------|--------|-----------|
| Aggressive | 15% | 50% | 20% |
| Neutral | 8% | 35% | 35% |
| Conservative | 5% | 25% | 55% |

---

## 5. 모멘텀 윈도우 최적화 EDA (신규 추가)

### 배경

> "자산별로 모멘텀이 달라지는 애들이 있을 거에요. EDA나 통계검정으로 확인해 보자." — 팀장님

### 방법

- **IC (Information Coefficient)**: Spearman 상관계수(모멘텀, 미래 21일 수익률)
- **t-검정**: IC 분포의 유의성 (H₀: IC = 0)
- 윈도우 후보: 21, 42, 63, 126, 189, 252 거래일 (1M~12M)
- 결과: `OPTIMAL_WINDOWS` 변수에 저장 → 백테스트에서 자동 사용

---

## 6. ETF → 개별종목 전환 이유와 설계

### 왜 개별종목인가

채권 통합 이후에도 개별 주식 유지 이유:
- 주식 부분은 시가총액 기반 동적 유니버스 → 섹터 다각화 보장
- BL의 Prior는 Risk Parity로 전환했으나, 주식 선정 방식(Top N per sector)은 유지

### 설계 구조

```
통합 유니버스 = 주식(동적) ∪ 채권ETF(고정)
  주식: 11개 GICS 섹터 × 5종목 = 55종목 (Wikipedia 필터 + 연간 재선정)
  채권: TLT, SHY, TIP, GLD (고정)
  총합: ~59자산
```

---

## 7. 생존편향(Survivorship Bias)

### 우리의 해결 방법

1. **연도별 동적 선정**: 매년 그 시점 시가총액으로 Top5 재선정
2. **Wikipedia 역산 필터**: "target_date 이후 편입된 종목" 제거

```python
def filter_sector_candidates_at_date(target_date, sector_candidates, changes_df):
    # target_date 이후 S&P 500에 추가된 종목을 후보 풀에서 제거
    added_after = changes_df[changes_df['date'] > target_date]['added']
    valid_pool = all_tickers - set(added_after)
    return filtered_candidates
```

**한계**: Wikipedia는 완전한 이력 없음 (현재 없는 종목 = 파산/제외된 종목은 복구 불가) → 명시적 언급 필요

---

## 8. 생성된 파일 (최종)

| 파일 | 상태 | 설명 |
|------|------|------|
| [Step4_BlackLitterman_Momentum.ipynb](Step4_BlackLitterman_Momentum.ipynb) | ✅ | ETF 기반 BL (등가중 Prior) |
| [Step4_코드해설.md](Step4_코드해설.md) | ✅ | Step4 전체 코드 해설 |
| [Step5_Stocks_Universe_BL.ipynb](Step5_Stocks_Universe_BL.ipynb) | ✅ | **통합 BL (Risk Parity Prior + 주식+채권 단일 BL)** |
| [Step5A_DualMomentum.ipynb](Step5A_DualMomentum.ipynb) | ✅ | 확장A: 듀얼 모멘텀 |
| ~~Step5B_XGBoost_Omega.ipynb~~ | ❌ **삭제** | XGBoost Ω — 팀 회의에서 취소 결정 |
| [Step5C_RelativeViews.ipynb](Step5C_RelativeViews.ipynb) | ✅ | 확장C: 횡단면 상대뷰 |

---

## 9. Step5 실행 순서

```bash
# 1. 베이스 노트북 (필수 먼저 — 캐시 생성)
Step5_Stocks_Universe_BL.ipynb
  ↳ 셀 8-B: EDA로 OPTIMAL_WINDOWS 결정 (백테스트에서 사용됨)

# 2. 확장 노트북 (순서 무관)
Step5A_DualMomentum.ipynb      # 하락장 방어
Step5C_RelativeViews.ipynb     # 상대뷰 + 전략 통합 비교
```

---

## 10. 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `TOP_N` | 5 | 섹터당 선정 종목 수 |
| `REBALANCE_FREQ` | 21거래일 | BL 기반 비중 리밸런싱 주기 |
| `UNIVERSE_UPDATE_FREQ` | 252거래일 | 유니버스 갱신 주기 (1년) |
| `COV_WIN` | 252 | 공분산 추정 윈도우 (거래일) |
| `LAM` | 2.5 | 위험회피계수 λ (Risk Parity Prior용) |
| `TAU` | 1/252 | Prior 불확실성 τ |
| `BOND_TICKERS` | TLT, SHY, TIP, GLD | 통합 BL 채권/대안 자산 (고정) |
| `OPTIMAL_WINDOWS` | EDA로 결정 | 모멘텀 윈도우 (셀 8-B 실행 후 자동 설정) |

---

## 11. 미완료 항목

- [ ] Step5 백테스트 실행 및 결과 확인 (통합 BL, Risk Parity Prior)
- [ ] EDA (셀 8-B) 실행: 자산별 최적 모멘텀 윈도우 확인
- [ ] Conservative 프로파일 MDD 개선 확인 (채권 통합 효과)
- [ ] Step5A/C 확장 노트북에 통합 BL 동일 적용 (현재 Step5 기반 수정 필요)
- [ ] 생존편향 완화 전후 성능 비교
