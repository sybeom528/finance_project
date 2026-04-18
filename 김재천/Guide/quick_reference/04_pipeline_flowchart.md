# 🗺️ 파이프라인 플로우차트 — Step 1~11 전체 구조

> **독자**: 프로젝트 이해를 원하는 모든 독자 (균형 톤)
> **목적**: 11단계 파이프라인을 한 장 다이어그램으로

---

## 📊 전체 파이프라인 (Mermaid 다이어그램)

```mermaid
flowchart TB
    subgraph "📥 Phase 1: 데이터 준비 (Step 1-2)"
        S1[Step 1: 데이터 수집<br/>yfinance + FRED<br/>30자산 + 12시장지표 + 8매크로]
        S2[Step 2: 전처리 + EDA + Granger<br/>파생피처 15개 생성<br/>34개 변수 p<0.05 선행성 확인]
        S1 --> S2
    end

    subgraph "🏗️ Phase 2: 최적화 기반 (Step 3-5)"
        S3[Step 3: 포트폴리오 최적화 개념<br/>MV/RP/HRP<br/>4개 성향 γ=1,2,4,8]
        S4[Step 4: Walk-Forward MV<br/>IS 24m → OOS 3m<br/>31개 윈도우, Sharpe 0.81]
        S5[Step 5: 리스크 분석<br/>VaR/CVaR<br/>11 스트레스 시나리오]
        S3 --> S4
        S4 --> S5
    end

    subgraph "🔔 Phase 3: 경보 시스템 (Step 6-7)"
        S6[Step 6: HMM 4레짐<br/>+ 경보 Config A/B/C/D<br/>BIC 최적화]
        S7[Step 7: EW 기반 Ablation<br/>v3 결과 Sharpe 1.473]
        S6 --> S7
    end

    subgraph "⚡ Phase 4: v4.1 확장 (Step 8-10)"
        S8[Step 8: Regime-Aware Σ<br/>Σ_stable / Σ_crisis<br/>대칭 4단계 Fallback]
        S9[Step 9: 64 시뮬레이션<br/>4모드 × 4성향 × 4Config<br/>경로 2 무효성 발견]
        S10[Step 10: 통계 검정<br/>Bootstrap + FDR + IR<br/>최종 추천 M1_보수형_ALERT_B]
        S8 --> S9
        S9 --> S10
    end

    subgraph "🎨 Phase 5: 시각화 확장 (Step 11)"
        S11[Step 11: Top 10 구성 시각화<br/>8종 시각화로 메커니즘 해부]
    end

    S2 --> S3
    S5 --> S6
    S7 --> S8
    S10 --> S11

    style S1 fill:#E3F2FD
    style S2 fill:#E3F2FD
    style S3 fill:#F3E5F5
    style S4 fill:#F3E5F5
    style S5 fill:#F3E5F5
    style S6 fill:#FFF3E0
    style S7 fill:#FFF3E0
    style S8 fill:#E8F5E9
    style S9 fill:#E8F5E9
    style S10 fill:#E8F5E9
    style S11 fill:#FCE4EC
```

---

## 📦 데이터 흐름 다이어그램

```mermaid
flowchart LR
    subgraph "원천 데이터"
        YF[yfinance<br/>주식·ETF 가격]
        FRED[FRED API<br/>거시지표]
    end

    subgraph "Step 1 산출물"
        PP[portfolio_prices.csv<br/>30자산 × 2609일]
        EP[external_prices.csv<br/>12지표]
        FD[fred_data.csv<br/>8매크로]
    end

    subgraph "Step 2 산출물"
        DF[df_reg_v2.csv<br/>2328 × 44]
        FT[features.csv]
        GR[granger_results.csv]
    end

    subgraph "Step 3~6 산출물"
        PR[profiles.csv]
        OW[optimal_weights.csv]
        RH[regime_history.csv]
        AS[alert_signals.csv]
    end

    subgraph "v4.1 산출물"
        RC[regime_covariance_by_window.pkl]
        SR[step9_backtest_results.pkl]
        FR[step10_final_recommendation.csv]
        TW[step11_top10_weights.pkl]
    end

    YF --> PP
    YF --> EP
    FRED --> FD

    PP --> DF
    EP --> DF
    FD --> DF

    DF --> PR
    DF --> FT
    DF --> GR
    PP --> OW
    DF --> RH
    DF --> AS

    PP --> RC
    RH --> RC
    RC --> SR
    AS --> SR
    SR --> FR
    FR --> TW

    style YF fill:#FFF9C4
    style FRED fill:#FFF9C4
    style PP fill:#E3F2FD
    style DF fill:#F3E5F5
    style RC fill:#E8F5E9
    style TW fill:#FCE4EC
```

---

## 🎯 단계별 핵심 역할 요약

| Step | 역할 | 핵심 산출 |
|------|------|---------|
| **1** | 원천 데이터 확보 | `portfolio_prices.csv` |
| **2** | 피처 공학 + 선행성 검증 | `df_reg_v2.csv` |
| **3** | 최적화 개념 시연 | `profiles.csv`, `optimal_weights.csv` |
| **4** | MV baseline 성과 (WF) | `wf_results.pkl` |
| **5** | 리스크 정량화 | `risk_metrics.csv` |
| **6** | HMM 레짐 + 경보 신호 | `regime_history.csv`, `alert_signals.csv` |
| **7** | v3 Ablation (EW 기반) | `step7_results.pkl` |
| **8** | 레짐 조건부 공분산 | `regime_covariance_by_window.pkl` |
| **9** | 통합 백테스트 (64 전략) | `step9_backtest_results.pkl` |
| **10** | 통계 검정 + 최종 추천 | `step10_final_recommendation.csv` |
| **11** | Top 10 시각적 해부 | `step11_top10_weights.pkl` |

---

## 🔄 의존성 계층 (5-tier)

```
Tier 0 (원천):      yfinance + FRED
Tier 1 (Step 1):    원본 가격 CSV
Tier 2 (Step 2-3):  피처 + 프로파일
Tier 3 (Step 4-7):  v3 결과 (EW 기반)
Tier 4 (Step 8-10): v4.1 통합 결과
Tier 5 (Step 11):   시각화·해부
```

**위 → 아래 의존**: 각 Tier는 상위 Tier 산출물 필요
**병렬 가능 Step**: Step 3과 Step 4 일부, Step 5 일부

---

## ⏱️ 전체 실행 시간 (순차)

| Step | 시간 | 누적 |
|------|------|------|
| Step 1 | 3~5분 | 5분 |
| Step 2 | 10~15분 | 20분 |
| Step 3 | 3~5분 | 25분 |
| Step 4 | 5~8분 | 33분 |
| Step 5 | 3~5분 | 38분 |
| Step 6 | 5~10분 | 48분 |
| Step 7 | 3~5분 | 53분 |
| Step 8 | 5~10분 | 63분 |
| Step 9 | 10~18분 | 81분 |
| Step 10 | 3~5분 | 86분 |
| Step 11 | 3~5분 | **91분 (약 1.5시간)** |

---

## 📚 상세 참조

각 Step의 상세 내용은 `docs/Step{n}_해설.md` 참조
전체 프로젝트 요약은 `report_final.md` 참조
