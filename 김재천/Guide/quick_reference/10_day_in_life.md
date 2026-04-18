# 🌅 Day in the Life — 경보가 울린 어느 날

> **독자**: 실전 운용 궁금증 (비전문가 톤)
> **목적**: 가상 시나리오로 하루 일과 체험

---

## 📅 시나리오 A: 평범한 월요일

### **2024-07-22 (월) — 조용한 여름날**

#### 🕘 오전 9:00 (장 시작 30분 전)
- 김재천 투자자, 커피 한 잔
- 노트북 열고 **매일 체크 루틴** 시작
- Python 스크립트 실행:

```python
import yfinance as yf
vix = yf.download('^VIX', period='1d')['Close'].iloc[-1]
vix3m = yf.download('^VIX3M', period='1d')['Close'].iloc[-1]
contango = vix3m - vix

print(f'VIX: {vix:.2f}, VIX3M: {vix3m:.2f}, Contango: {contango:+.2f}')
# 출력: VIX: 14.35, VIX3M: 15.80, Contango: +1.45
```

#### 🕘 오전 9:05
- **경보 판정**: VIX 14.35 < 20 → **L0**
- Contango > 0 → 조정 없음
- 📝 노트: "오늘도 L0, 조용한 날"

#### 🕘 오전 9:10
- 이번 주 분기 리밸런싱 예정일 확인 → 아직 (7월 1일에 이미 완료)
- 현재 비중 확인 (전일 기준)
- 아침 메모: "정상 비중 유지, 할 일 없음"

#### 🕐 오후
- 일상 업무 (다른 일)
- 포트폴리오 **신경 쓸 필요 없음**

#### 🕕 오후 6:00 장 마감 후
- 5분 점검:
  - 오늘 수익률: +0.35%
  - 누적 수익: +147% (8년 기준 추정)
  - 경보 변동 없음
- 📝 노트: "평범한 하루, Sharpe 유지"

**오늘 총 투자 시간**: **10분**

---

## 📅 시나리오 B: 경보 발동일

### **2024-08-05 (월) — 엔캐리 청산일**

#### 🕘 오전 7:00 (한국 시간)
- 알림: 일본 증시 폭락 뉴스
- Nikkei -12%, 사상 최대 일일 하락
- "느낌 안 좋다..."

#### 🕘 오전 8:30
- Yahoo Finance 앱 확인
- 미국 선물 대폭 하락 (-4%)
- 😰 긴장 모드

#### 🕘 오전 9:00 (미국 장 시작 30분 전)
- Python 스크립트 실행 준비

#### 🕘 오후 10:30 (한국 시간, 미국 장 시작)
- VIX 급등: 16 → **35** (pre-market)
- Contango < 0 (백워데이션)

#### 🕘 오후 10:35
- **경보 판정**:
  - VIX 35 ≥ 28 → **L2** (경계)
  - Contango < 0 → **+1** → **L3** (위기)
- 📝 긴급 노트: "L3 발동, 주식 60% 감축 필요"

#### 🕘 오후 10:40 (행동)
- 현재 비중:
  - 주식 40% (SPY 8%, QQQ 5%, XLK 4%, ... 총 24개 자산)
  - 채권 45%
  - 금 15%

- **감축 계산** (Python):
```python
def calculate_L3(current_equity):
    cut = 0.60
    equity_reduction = current_equity * cut
    to_bond = equity_reduction * 0.70
    to_gold = equity_reduction * 0.30
    return {
        'new_equity': current_equity - equity_reduction,
        'equity_reduction': equity_reduction,
        'to_bond': to_bond,
        'to_gold': to_gold,
    }

result = calculate_L3(0.40)
# {'new_equity': 0.16, 'equity_reduction': 0.24,
#  'to_bond': 0.168, 'to_gold': 0.072}
```

- 결과:
  - 주식 40% → **16%** (24%p 감축)
  - 채권 45% → **61.8%** (+16.8%p)
  - 금 15% → **22.2%** (+7.2%p)

#### 🕙 오후 10:50 (거래 실행)
- 24개 주식 자산을 **각각 60% 매도**
- 받은 현금으로 채권·금 매수
- 총 거래: 주식 매도 24건 + 채권 4건 매수 + 금 2건 매수
- 예상 거래비용: turnover 약 48% × 15bps = **약 7bps (0.07%)**

#### 🕕 오후 6:00 미국 장 마감 (한국 시간 다음날 새벽)
- SPY 종가: -3.0%
- **우리 포트폴리오**: -1.2% (경보 감축 후)
- **방어 성공!** (SPY 대비 2.5배 덜 손실)

#### 다음 날 (2024-08-06 화)

**오전 9:00 체크**:
- VIX 27 → 경보 L2 (정상화 중)
- 어떻게 할까?

- **규칙**: 당일 즉시 복원 (v4.1 로직)
- 주식 16% → 26% (L2 감축 수준)

**오전 9:05 행동**:
- 주식 추가 매수 10%p
- 채권·금 일부 매도

#### 8월 8일 (목)
- VIX 22 → 경보 L1 → 주식 34%
- VIX 18 → 경보 L0 → 주식 40% (완전 복구)

**3일 만에 원상복구**. 총 손실 -1.2% (SPY는 -3.5% 유지).

**일주일 총 투자 시간**: **약 1시간**

---

## 📅 시나리오 C: 분기 리밸런싱 날

### **2025-10-01 (수) — 4분기 리밸런싱**

#### 🕘 오전 9:00
- "오늘은 분기 리밸런싱 날!"
- 평소보다 조금 길게 시간 확보 (30분)

#### 🕘 오전 9:05
- **최근 24개월 IS 데이터 다운로드** (2023-10-01 ~ 2025-09-30)

```python
import yfinance as yf
tickers = ['SPY', 'QQQ', ...]  # 30개
data = yf.download(tickers, start='2023-10-01', end='2025-09-30')
log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
```

#### 🕘 오전 9:10
- **μ, Σ 계산**:

```python
mu = log_returns.mean().values * 252  # 연율
from sklearn.covariance import LedoitWolf
cov = LedoitWolf().fit(log_returns.values).covariance_ * 252
```

#### 🕘 오전 9:15
- **MV 최적화 실행** (성향 제약 반영):

```python
from scipy.optimize import minimize

def neg_utility(w, mu, cov, gamma):
    return -(w @ mu - gamma / 2 * w @ cov @ w)

# 성향: 보수형 (γ=8, max_equity=0.43, min_bond=0.31)
constraints = [
    {'type': 'eq', 'fun': lambda w: w.sum() - 1},
    {'type': 'ineq', 'fun': lambda w: 0.43 - w[eq_idx].sum()},  # max_equity
    {'type': 'ineq', 'fun': lambda w: w[bd_idx].sum() - 0.31},  # min_bond
]
result = minimize(neg_utility, x0=np.ones(30)/30, method='SLSQP',
                   args=(mu, cov, 8), bounds=[(0,1)]*30,
                   constraints=constraints)
new_weights = result.x / result.x.sum()
```

#### 🕘 오전 9:20
- **결과 확인**:
  - 새 비중이 이전 비중과 얼마나 다른지 (turnover)
  - 예: turnover 약 12% (정상 분기 수준)
  - 예상 거래비용: 12% × 15bps = **1.8bps (0.018%)**

#### 🕘 오전 9:25
- VIX 확인 → 16, 경보 L0
- 이제 리밸런싱 실행 OK

#### 🕘 오전 9:30 (장 시작)
- 실제 거래 실행:
  - 비중 증가 자산 매수
  - 비중 감소 자산 매도
- 분할 매매로 임팩트 최소화

#### 🕙 오전 10:00 (30분 후)
- 모든 거래 체결 완료
- 새 비중으로 전환
- 📝 기록: "Q4 리밸런싱 완료, turnover 12%, 비용 0.018%"

#### 🕕 오후 6:00
- 평소처럼 일일 점검
- 추가 경보 없음

**리밸런싱 날 총 시간**: **약 1시간**

---

## 🔧 필요한 도구 Set

### 필수 (매일)
- ☕ 커피
- 💻 Python 환경 (yfinance, pandas)
- 📱 Yahoo Finance 앱 (실시간 VIX)
- 📝 Excel/노트 (비중 기록)

### 분기별
- 🧮 MV 최적화 스크립트 (본 프로젝트 코드 활용)
- 📊 거래 플랫폼 (대량 주문)

### 월간 점검
- 📈 성과 추적 스프레드시트
- 🎯 체크리스트 (`13_operating_checklist.md`)

---

## ⏱️ 시간 투자 요약

| 활동 | 빈도 | 소요 시간 | 월 합계 |
|------|------|--------|--------|
| 매일 VIX 체크 | 일 | 10분 | 약 3.3시간 |
| 경보 발동 대응 | 월 평균 5회 | 30분/회 | 2.5시간 |
| 분기 리밸런싱 | 분기 | 1시간 | 0.3시간/월 |
| 월말 성과 점검 | 월 | 30분 | 0.5시간 |
| **총 예상** | | | **약 6.6시간/월** |

→ **하루 평균 15분 미만**

---

## 😊 심리적 팁

### 첫 3개월 (입문)
- 규칙 **엄격 준수** 연습
- 의심되면 데이터 재검증
- 커뮤니티 참여 (추천)

### 첫 1년
- 첫 경보 발동 경험 중요
- **Whipsaw(감축 후 반등)** 한두 번 겪어야 진짜 배움
- 규칙 신뢰 형성

### 장기 (3년+)
- 자동화 고도화 (Streamlit 앱 활용)
- 주위에 공유 가능 수준
- 개인 맞춤 튜닝 (성향 변경 등)

---

## 💬 흔한 순간의 대화

### 💬 "경보 L3인데 정말 60% 감축?"
> "규칙 준수. 일관성이 장기 성공."

### 💬 "감축했는데 반등하네..."
> "Whipsaw는 보험료. 대공포 방어의 대가."

### 💬 "강세장에 왜 이렇게 보수적?"
> "Sharpe 보라. MDD 낮으면 장기 누적 승리."

---

## 🎯 요약

> **하루 10분, 경보 발동 시 30분, 분기 1시간**
> 이것이 M1_보수형_ALERT_B 운용의 전부.
> 규칙을 믿고, 감정을 배제하고, 꾸준히 실행하는 것이 성공의 열쇠.
