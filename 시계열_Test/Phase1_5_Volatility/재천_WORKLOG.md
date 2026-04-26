# Phase 1.5 — 변동성 예측 분기 작업 일지 (재천 WORKLOG)

> **목적**: Phase 1.5 구축 과정의 모든 작업·결정·판단 근거를 시간순으로 기록합니다.
> 사용자가 언제든 진행 상황과 의사결정 흐름을 추적할 수 있게 하기 위함입니다.
>
> **위치**: `시계열_Test/Phase1_5_Volatility/재천_WORKLOG.md`
> **기록 원칙**:
> 1. 모든 의사결정은 (a) 결정 내용, (b) 선택지, (c) 판단 근거, (d) 결정 주체(사용자 / 어시스턴트 제안)를 함께 기록
> 2. 사용자 피드백·지시는 원문 또는 요지를 보존
> 3. 작업 완료 시 산출물 파일 경로와 핵심 변경 사항 기록

---

## 2026-04-26 — 분기 신설 결정 + Step 0 완료

### 1. 분기 신설 배경

Phase 1 LSTM 베이스라인이 5회 Run 끝에 다음 결과로 마감되었습니다.

- **3차 Run (현재 best)**: SPY hit_rate=0.6313 ✅ / R²_OOS=-0.2118 ❌
- **4차 Run (Y_trailing 다변량)**: 대폭 악화 (R²_OOS -2.15)
- **5차 Run (VIX 추가)**: 더 악화 (R²_OOS -1.13)

윤서님의 진단으로 **134 훈련 샘플/fold 환경의 절대 부족** 이 root cause 로 확인되었습니다. 한편 김하연님의 §10 ARCH-LM 검정에서 **변동성에는 강한 자기상관** (SPY LM=754, p≈0)이 정량적으로 입증되어, 동일 시계열에서 **변동성(ACF max 0.30)이 수익률 방향(ACF max 0.13)보다 훨씬 풍부한 신호** 임이 확인되었습니다.

### 2. 사용자 결정 (확정)

**Phase 2(GRU)로 진행하지 않고 Phase 1.5 분기를 신설**:
- Phase 1 결과는 그대로 보존 (변경 금지)
- Phase 1.5 는 예측 대상을 **누적수익률 → 실현변동성(realized volatility)** 으로 교체한 새 LSTM 학습·평가
- **본 단계 유일한 목표**: "변동성 예측이 가능한가?" 단일 질문에 명확한 답
- 포트폴리오 구축·BL 통합·벤치마크 비교는 **본 단계 평가 대상 아님** (추후 별도 단계로 미룸)

### 3. plan 수립 흐름 (사용자 대화)

| 차수 | 사용자 피드백 | 반영 |
|---|---|---|
| 1차 plan | (작성 후 사용자 검토) | 각 결정 사항 직관적 설명 부족 지적 |
| 2차 plan | 타깃·입력·평가지표·베이스라인·관문·loss·옵티마 등 각 항목에 직관적 설명 추가 | 사용자 검토 |
| 3차 plan | 사용자 질문: "변동성 예측 → 포트폴리오 구축까지 어떻게 이어지는지 일반 사용자 관점에서 설명 필요" | 효율적 프론티어·BL 모델·시너지 다이어그램 추가 |
| 4차 plan | 사용자 질문: "변동성만으로 벤치마크를 이기는 게 불가능한가? 학술 근거 필요" | Moreira & Muir (2017, JF), Harvey et al. (2018), DeMiguel et al. (2009, RFS), Asness et al. (2012) 인용 추가 |
| 5차 plan (최종) | 사용자 결정: "포트폴리오 가능성은 배경으로 남기고, Phase 1.5 는 '변동성 예측 가능성'에만 집중" | 포트폴리오 섹션을 Appendix 로 이동, Context 의 "핵심 목표" 단일 질문 명시 |

**최종 plan 위치**: `C:\Users\gorhk\.claude\plans\sharded-mapping-puffin.md` (진실원), [PLAN.md](PLAN.md) (팀 공유 사본)

### 4. 핵심 의사결정 요약

| 항목 | 결정 | 근거 |
|---|---|---|
| 폴더 명 | `Phase1_5_Volatility` | "분기(.5)" 의미 + 타깃 기준 차이 폴더명 노출 |
| scripts 전략 | **독립 복사 + 변동성 전용 신규 추가** | Phase 1 자산 격리 보존 (옵션 a/b 보다 우월) |
| 타깃 | **Log Realized Volatility** = `log(rolling(21).std(log_ret)).shift(-21)` | (1) Log 변환 후 정규 분포 → MSE 정합 (2) Corsi(2009) HAR-RV 학술 표준 (3) `exp(pred)` 양수 보장 (4) Gradient 안정성 |
| 입력 | **`log_ret²` univariate** | (1) instantaneous variance proxy — 타깃과 dimensional match (2) Run 4·5 다변량 실패 교훈 회피 |
| 손실 | **MSE** (Huber 에서 변경) | Log-RV 거의 정규분포 → 가우시안 가정 충족, HAR-RV OLS 와 정합 |
| 평가지표 | **RMSE on Log-RV / QLIKE / R²_train_mean** + MZ regression | Hit Rate 폐기 (변동성 항상 양수 → trivially 1.0), R²_OOS zero baseline 폐기 (R² 인공 증가 함정) |
| 베이스라인 | **HAR-RV / EWMA(λ=0.94) / Naive / Train-Mean** | Corsi 2009 학술 표준 + RiskMetrics 산업 표준 + sanity check |
| 관문 (3개 모두 충족) | (1) LSTM RMSE < HAR-RV RMSE / (2) R²_train_mean > 0 / (3) pred_std/true_std > 0.5 | 학술 표준 능가 + 단순 baseline 능가 + mean-collapse 회피 |
| 모델·Walk-Forward | **Phase 1 동일** (LSTM 1·32·1·0.3, 105 fold) | 비교 공정성 — 타깃만 변경하여 신호 차이 분리 측정 |

### 5. Step 0 작업 기록

#### 5.1 사용자 지시 (3 항목)
1. 즉시 시작
2. 본 환경(Windows)에서 진행하지만, 다른 환경(macOS/Linux/타 팀원)에서도 실행 가능하도록 호환성 고려
3. 작성 주체: 재천 (재천_WORKLOG.md prefix)

#### 5.2 폴더 생성 + 파일 복사 (Bash)

```bash
BASE="시계열_Test"
mkdir -p "$BASE/Phase1_5_Volatility/scripts"
mkdir -p "$BASE/Phase1_5_Volatility/results/raw_data"

# scripts 4개 복사 (격리 보존: Phase 1 원본 변경 X)
cp "$BASE/Phase1_LSTM/scripts/setup.py"   "$BASE/Phase1_5_Volatility/scripts/"
cp "$BASE/Phase1_LSTM/scripts/dataset.py" "$BASE/Phase1_5_Volatility/scripts/"
cp "$BASE/Phase1_LSTM/scripts/models.py"  "$BASE/Phase1_5_Volatility/scripts/"
cp "$BASE/Phase1_LSTM/scripts/train.py"   "$BASE/Phase1_5_Volatility/scripts/"

# 환경 노트북 복사 (Phase 1 동일)
cp "$BASE/Phase1_LSTM/00_setup_and_utils.ipynb" "$BASE/Phase1_5_Volatility/"

# 데이터 복사 (Phase 1 다운로드 결과 그대로 활용)
cp "$BASE/Phase1_LSTM/results/raw_data/SPY.csv" "$BASE/Phase1_5_Volatility/results/raw_data/"
cp "$BASE/Phase1_LSTM/results/raw_data/QQQ.csv" "$BASE/Phase1_5_Volatility/results/raw_data/"
```

#### 5.3 신규 문서 3건 작성
- `scripts/__init__.py` — 패키지 마커, 분기 특성·파일 출처 명시
- `README.md` — 협업 진입점, Phase 1 ⇄ Phase 1.5 비교 표, OS 호환 설치 안내
- `재천_WORKLOG.md` (본 파일) — 작업 일지

#### 5.4 PLAN.md 동기화 (예정)
진실원(`C:\Users\gorhk\.claude\plans\sharded-mapping-puffin.md`)을 Phase1_5_Volatility/PLAN.md 로 복사. 진실원이 갱신될 때 사본도 함께 갱신.

### 6. 호환성 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 한글 폰트 | `scripts/setup.py` 의 `setup_korean_font()` 가 OS 자동 분기 (Windows: Malgun Gothic / macOS: AppleGothic / Linux: NanumGothic via koreanize-matplotlib) | Phase 1 의 setup.py 그대로 복사 — 이미 OS 호환 구현됨 |
| 경로 | `Path(__file__).resolve().parent.parent` 패턴 (setup.py BASE_DIR) | 절대경로 하드코딩 X, OS 무관 |
| 데이터 | Phase 1 의 `results/raw_data/{SPY,QQQ}.csv` 를 본 분기에 복사 (참조 X) | 분기 격리 + 자기완결적 실행 |
| 라인 종료 | LF (git autocrlf 권장 설정) | 크로스 OS 호환 |
| README 설치 안내 | Bash + PowerShell + Linux 한글 폰트 패키지 모두 명시 | 다른 팀원 OS 무관 실행 |

### 7. Step 0 산출물 인덱스

| 파일 | 종류 | 출처 / 비고 |
|---|---|---|
| `00_setup_and_utils.ipynb` | 노트북 | Phase 1 복사 |
| `scripts/__init__.py` | 모듈 | **신규 작성** |
| `scripts/setup.py` | 모듈 | Phase 1 복사 (변경 없음) |
| `scripts/dataset.py` | 모듈 | Phase 1 복사 (변경 없음) |
| `scripts/models.py` | 모듈 | Phase 1 복사 (변경 없음) |
| `scripts/train.py` | 모듈 | Phase 1 복사 (Step 2~3 에서 `loss_type='mse'` 옵션 추가 예정) |
| `results/raw_data/SPY.csv` | 데이터 | Phase 1 복사 (4,336행) |
| `results/raw_data/QQQ.csv` | 데이터 | Phase 1 복사 (4,336행) |
| `README.md` | 문서 | **신규 작성** — 협업 진입점 |
| `재천_WORKLOG.md` | 문서 | **신규 작성** (본 파일) |
| `PLAN.md` | 문서 | **신규 작성** — Claude plan 사본 |

### 8. 다음 단계 (Step 1 — 사용자 승인 대기)

**Step 1**: `01_volatility_eda.ipynb` (§1~§9) 작성

| § | 내용 |
|---|---|
| §1 | 환경 부트스트랩 (`bootstrap()`) |
| §2 | 데이터 로드 — `results/raw_data/{SPY,QQQ}.csv` |
| §3 | log_ret · log_ret² · \|log_ret\| 시계열 + 변동성 클러스터링 육안 확인 |
| §4 | RV (rolling 21) 계산 + 분포 진단 (히스토그램·QQ-plot, log 변환 전후) |
| §5 | ACF/PACF on log(RV) — 변동성 자기상관 정량 (lag 1~30) |
| §6 | 정상성 검정 (ADF, KPSS) on log(RV) |
| §7 | 체제 진단 (저변동/고변동 구간 RV 분위수 표시) |
| §8 | 타깃 누수 검증 (`verify_no_leakage_logrv` + 5행 육안 표) |
| §9 | 결론 — 사용자 체크포인트 |

**사용자 체크포인트**: §4 RV 분포 결과 보고 → log 변환 채택 합의, §5 ACF 결과 → embargo=21 충분성 합의, §8 누수 검증 PASS 보고.

---

## 의사결정 보류 항목

(없음 — 본 단계 plan 에서 모든 결정 확정)

---

## TODO (Step 2 진입 시)

- `scripts/train.py` 에 `loss_type: str = 'huber'` 옵션 인자 추가 (기본값 huber 유지, Phase 1 호환)
- Phase 1.5 노트북에서는 `loss_type='mse'` 명시
