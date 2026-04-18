# ❓ 통합 FAQ — 모든 궁금증 한곳에

> **독자**: 모든 독자 (균형 톤)
> **목적**: Step별 해설의 FAQ를 주제별로 재구성

---

## 📑 주제별 목차

- [🎯 전략 개요](#-전략-개요)
- [📊 성과·결과](#-성과결과)
- [⚙️ 운영·실전](#-운영실전)
- [🔬 방법론](#-방법론)
- [💻 기술·구현](#-기술구현)
- [📚 학습·심화](#-학습심화)

---

## 🎯 전략 개요

### Q. 이 프로젝트의 한 문장 요약은?

**A**: **"대안데이터 기반 경보 시스템으로 주식 비중을 조절하면 Sharpe 1.06, MDD -15.5%의 Walk-Forward 성과 달성"**

### Q. "경로 1", "경로 2"가 뭔가요?

**A**:
- **경로 1**: 매일 경보 레벨 체크 → 주식 비중 축소 (성공)
- **경로 2**: HMM 레짐 기반 공분산 전환 (실패, 무효)

→ 자세한 내용: `05_path1_vs_path2.md`

### Q. 왜 M1_보수형_ALERT_B가 최우수?

**A**:
- Multi-criteria 종합 1위 (Sharpe + MDD + Calmar + Sortino)
- Sharpe 1.064 (Top 3 내)
- **MDD -15.53%** (Top 10 중 최저)
- Config B 구성 단순 (VIX + Contango 2변수)

### Q. 다른 성향(중립·적극·공격)은 어때?

**A**: 총수익률은 높지만 위험 대비 수익(Sharpe)은 비슷:
- 공격형 M1: Total +428%, Sharpe 0.97
- 보수형 M1: Total +151%, Sharpe 1.06
- **같은 위험당 수익**은 보수형 우수
- 총수익률 극대화 원하면 공격형 선택 가능 (MDD -22% 감수)

---

## 📊 성과·결과

### Q. Sharpe 1.064는 좋은 수치?

**A**: **상당히 우수**:
- 헤지펀드 평균: 0.5~0.8
- S&P 500 장기: ~0.5
- **Sharpe ≥ 1.0 = 상위 수준**

### Q. 왜 SPY보다 Total 낮은가? (151% vs 180%)

**A**: **변동성을 크게 낮춰서**:
- SPY: +180% 수익 / 17% 변동성
- 우리: +151% 수익 / 11% 변동성 (65%)
- **위험 조정 후 우리가 우수 (Sharpe 1.06 vs 0.76)**

### Q. 미국 시장 외 어떤 경우에 적용될까?

**A**: 현재는 미검증. v5에서 검토 예정:
- 글로벌 분산 (신흥국, 선진국 non-US)
- 부동산 (REIT)
- 원자재 확장

### Q. 8년 성과가 미래를 보장?

**A**: **아니요**. 특이 기간 포함:
- 2020 COVID (역대급 베어)
- 2022 긴축 (40년만 인플레)
- 일반 장에서도 동일 성과 보장 ❌
- 하지만 **구조적 방어력**은 유지될 가능성 높음

---

## ⚙️ 운영·실전

### Q. 최소 자본?

**A**: **$30,000 (약 4천만원)** 권장
- 자산당 $1,000 × 30 자산
- 거래비용 효율 고려
- 이상: $100,000+

### Q. 하루 얼마나 시간 걸려?

**A**: **평균 15분/일**:
- 평상시: 10분 (VIX 체크만)
- 경보 발동: 30분
- 분기 리밸런싱: 1시간

→ 자세히: `10_day_in_life.md`

### Q. 자동화 가능?

**A**: **네**. 방법:
1. Python + yfinance로 VIX 자동 조회
2. 경보 판정 로직 스크립트화
3. 거래 API 연동 (주의: 오류 위험)
4. 일단 **수동 + 알림만** 자동화 추천

### Q. 거래비용 15bps가 현실적?

**A**: **기관 수준**. 소매 환경:
- 한국 증권사 해외주식: 수수료 0.25%~ (25bps)
- 미국 ETF: 거의 0 (Robinhood 등)
- 실제 환경 맞춤: Streamlit Simulator에서 비용 조정 가능

### Q. 세금은?

**A**: **한국 기준**:
- 양도소득세 22% (해외주식)
- 연 250만원 공제
- 배당세 15% (원천징수)
- **세금 후 Sharpe 0.8~0.9로 하락** 가능

---

## 🔬 방법론

### Q. 왜 Walk-Forward?

**A**: **look-ahead bias 방지**:
- 과거 IS 24개월로 μ, Σ 추정
- 미래 OOS 3개월로 검증
- 미래 정보 절대 안 씀
- 실전 운용과 동일 환경

### Q. HMM이 뭔가요?

**A**: **Hidden Markov Model**:
- 시장의 **숨은 상태** (저변동/일반/고변동/위기) 자동 분류
- 관측 가능한 지표(VIX, HY 등)로 상태 역추정
- 4레짐이 BIC 최적 (Step 6)

### Q. 경로 2가 왜 실패?

**A**: **3가지 원인**:
1. Σ 차이가 MV 비중에 미약하게 반영
2. 월 재최적화 비용 > 이론적 이득
3. 경로 1과 정보 중복

→ 상세: `05_path1_vs_path2.md`

### Q. Cohen's d는 왜 폐기?

**A**: **재무 일별 수익률에 부적합**:
- 일별 SNR 낮음 → d 항상 <0.01
- 모든 전략 "미미"로 분류 → 해석 불가
- **IR + ΔSR** 실무 기준으로 교체

### Q. Bootstrap 5000번이면 충분?

**A**: **일반적 표준**:
- 더 정확: 10,000+
- 계산 비용 trade-off
- 재현 가능성: `seed=42` 고정

### Q. Bonferroni vs FDR?

**A**:
- **Bonferroni**: 매우 보수적 (α/K)
- **FDR (Benjamini-Hochberg)**: 덜 보수적, 검정력 보존
- **둘 다 적용**하여 일관성 확인

---

## 💻 기술·구현

### Q. 사용된 라이브러리?

**A**:
```
pandas, numpy, scipy, scikit-learn
matplotlib, seaborn
yfinance, fredapi
hmmlearn (HMM)
streamlit, plotly (앱)
```

### Q. 전체 실행 시간?

**A**: **Step 1~11 순차 실행 약 1.5시간**:
- Step 9가 가장 오래 (10~18분)
- Step 1은 네트워크 속도 의존

### Q. 재현 가능?

**A**: **네**. 재현 방법:
1. 노트북 순차 실행 (`jupyter nbconvert --execute`)
2. `seed=42` 고정
3. yfinance 데이터 변동 가능하나 크지 않음

### Q. Streamlit 앱은 어떻게?

**A**:
```bash
cd Guide/interactive/streamlit_app
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

### Q. 노트북 실행 시 메모리 요구?

**A**: **약 4~8 GB RAM**:
- Step 8 Σ 추정에서 가장 큼
- 64bit Python 필수

### Q. Windows / Mac / Linux 지원?

**A**: **모두 지원**:
- 한글 폰트만 OS별 다름:
  - Windows: Malgun Gothic
  - Mac: AppleGothic
  - Linux: NanumGothic (koreanize-matplotlib)

---

## 📚 학습·심화

### Q. 완전 초보인데 어디부터 봐야?

**A**: **추천 순서**:
1. `report_final.md` (30분) — 전체 개요
2. `quick_reference/01_executive_one_pager.md` (1분)
3. `quick_reference/09_timeline_narrative.md` — 스토리
4. `quick_reference/02_investor_summary_card.md` — 실전
5. `docs/Step1_해설.md` → `Step11_해설.md` (순차)

### Q. 퀀트 관점의 기술 심화는?

**A**:
1. `decision_log.md` + `decision_log_v31.md` — 설계 근거
2. 노트북 Step 1~11 실행 + 코드 리뷰
3. `report_v3.md` + `report_v4.md` — 상세 수치
4. `stats_model.md` — 통계 기법
5. 외부 참고:
   - Markowitz (1952)
   - Ledoit & Wolf (2004)
   - López de Prado (2016, HRP)

### Q. 이 프로젝트를 개선하려면?

**A**: **v5 후보**:
- 경로 2 대체안 (Risk Budgeting)
- 글로벌 확장
- LLM 감성 분석 통합
- Real-time deployment

→ 상세: `report_final.md` Section 7.2

### Q. 학회·논문 발표 가능?

**A**: **가능**:
- **Negative result**: "경로 2의 실증적 무효성"
- Multi-regime covariance에 관한 실증 기여
- 참고 : 부적합한 Cohen's d 대신 IR/ΔSR 제안

---

## 🔗 특정 주제 심화

| 주제 | 참조 문서 |
|------|---------|
| 경보 시스템 설계 | `docs/Step6_해설.md` |
| HMM 레짐 분류 | `docs/Step6_해설.md` + `decision_log.md` |
| MV 최적화 수학 | `stats_model.md` + `docs/Step3_해설.md` |
| Walk-Forward 이론 | `docs/Step4_해설.md` |
| VaR/CVaR | `docs/Step5_해설.md` |
| Bootstrap 통계 | `docs/Step10_해설.md` |
| 위기 대응 사례 | `quick_reference/08_crisis_case_studies.md` |
| 실전 운용 가이드 | `quick_reference/13_operating_checklist.md` |

---

## 💬 지원 채널

- **Streamlit 앱 Q&A 페이지**: interactive/streamlit_app/pages/8_Learn.py
- **노트북 내 Markdown 주석**: 각 셀별 설명
- **체크리스트**: `quick_reference/13_operating_checklist.md`
