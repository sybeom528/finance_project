"""
lib/insight_generator.py - Investment Simulator 영역 6 카드 그리드

F-6 결정 (Sim6-1 + Sim6-2): 정적 템플릿 카드 그리드 (LLM X — 안정성 우선).
시뮬레이션 결과 + 활성 벤치마크 + 시나리오 (Lump-sum / DCA / Goal) 에 따라
조건부 4-8개 카드 생성.

참조: docs/plan/02_common.md 5절, decisionlog/11_dl_sections.md F 섹션
"""

from __future__ import annotations

import streamlit as st


# 카드 색상 매핑 (테두리 + 강조)
_COLOR_MAP = {
    "green": "#10B981",
    "red": "#EF4444",
    "blue": "#3B82F6",
    "orange": "#F59E0B",
    "purple": "#8B5CF6",
}


def generate_insight_cards(
    sim_result: dict,
    benchmarks: dict,
    scenario: str = "lump_sum",
) -> list[dict]:
    """
    Sim 결과 → 카드 정보 list 생성.

    Args:
        sim_result: {
            "total_invested": float,        # 총 투자 원금
            "final_value": float,            # 최종 평가액
            "total_profit": float,           # 누적 수익 (final - invested)
            "cagr": float,                   # 연환산
            "mdd": float,                    # 최대 낙폭 (음수)
            "recovery_months": int | None,  # COVID 회복 개월
            "dca_monthly": float | None,    # DCA 월 금액 (해당 시)
            "dca_advantage": float | None,  # DCA 일시 대비 차이
            "goal_amount": float | None,    # Goal 시나리오 목표
            "goal_achievement_date": str | None,
        }
        benchmarks: {"SPY": {"cagr": float}, "EW": {...}, "IVW": {...}}
                    활성 벤치마크만 포함 (사이드바 토글 반영)
        scenario: "lump_sum" / "dca" / "goal"

    Returns:
        cards: list[dict] — 각 dict: {"icon", "title", "value", "delta?", "subtitle?", "color"}
    """
    cards: list[dict] = []

    # 1. 누적 수익 (항상)
    profit_pct = (
        sim_result["total_profit"] / sim_result["total_invested"] * 100
        if sim_result["total_invested"] > 0
        else 0.0
    )
    cards.append({
        "icon": "💰",
        "title": "누적 수익 — Total Profit",
        "value": f"${sim_result['total_invested']:,.0f} → ${sim_result['final_value']:,.0f}",
        "delta": f"+${sim_result['total_profit']:,.0f} ({profit_pct:+.1f}%)",
        "color": "green" if sim_result["total_profit"] > 0 else "red",
    })

    # 2. 연환산 CAGR (항상)
    cards.append({
        "icon": "📈",
        "title": "연환산 수익률 — CAGR",
        "value": f"{sim_result['cagr'] * 100:+.2f}% per year",
        "color": "green" if sim_result["cagr"] > 0 else "red",
    })

    # 3-5. vs Benchmark (활성 벤치마크만) — CAGR delta + Final value delta 동시 표시
    for name in ("SPY", "EW", "IVW"):
        if name in benchmarks:
            bench = benchmarks[name]
            delta_cagr = sim_result["cagr"] - bench.get("cagr", 0.0)
            delta_final = sim_result["final_value"] - bench.get("final_value", 0.0)
            cards.append({
                "icon": "📊",
                "title": f"vs {name}",
                "value": f"CAGR {delta_cagr * 100:+.3f}%p",
                "subtitle": f"Final value 차이: ${delta_final:+,.0f}",
                "color": "green" if delta_cagr > 0 else "red",
            })

    # 6. 최대 손실 / 회복 (항상)
    recovery_label = (
        f"COVID-19 회복: {sim_result['recovery_months']}개월"
        if sim_result.get("recovery_months")
        else "회복 정보 미산출"
    )
    cards.append({
        "icon": "⚠️",
        "title": "최대 손실 / 회복 — Max Drawdown",
        "value": f"{sim_result['mdd'] * 100:.1f}%",
        "subtitle": recovery_label,
        "color": "orange",
    })

    # 7. DCA 효과 (DCA Tab 활성 시만)
    if scenario == "dca" and sim_result.get("dca_monthly"):
        dca_advantage = sim_result.get("dca_advantage", 0.0)
        cards.append({
            "icon": "🔄",
            "title": "DCA 효과 — Dollar Cost Averaging",
            "value": f"매월 ${sim_result['dca_monthly']:,.0f} 분산 투자",
            "subtitle": f"일시 투자 대비 {dca_advantage:+,.0f}",
            "color": "blue",
        })

    # 8. Goal 달성 분석 (Goal Tab 활성 시만)
    if scenario == "goal" and sim_result.get("goal_amount"):
        req = sim_result.get("required_initial")
        goal = sim_result["goal_amount"]
        cum_factor = (goal / req) if (req and req > 0) else None
        cards.append({
            "icon": "🎯",
            "title": "Goal 역산 — Goal-based",
            "value": f"필요 초기 ${req:,.0f}" if req else "산출 불가",
            "subtitle": (
                f"목표 ${goal:,.0f} / 누적 factor {cum_factor:.2f}배"
                if cum_factor else f"목표 ${goal:,.0f}"
            ),
            "color": "purple",
        })

    return cards


def render_insight_grid(cards: list[dict], cols_per_row: int = 3) -> None:
    """
    카드 그리드 렌더링 (반응형 N-column).

    cards 가 비어있으면 안내 메시지 표시.
    """
    if not cards:
        st.info("표시할 인사이트가 없습니다 — 시뮬레이션을 먼저 실행해주세요.")
        return

    for i in range(0, len(cards), cols_per_row):
        row = cards[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, card in enumerate(row):
            with cols[j]:
                _render_card(card)


def _render_card(card: dict) -> None:
    """
    단일 카드 렌더링 (HTML + inline style).

    NOTE: HTML 을 한 줄로 합쳐서 markdown 의 4-space 들여쓰기 code block 해석
    회피 (들여쓴 HTML 이 escape 되어 raw text 로 출력되는 케이스 방지).
    """
    border_color = _COLOR_MAP.get(card.get("color", "blue"), "#3B82F6")

    subtitle_html = (
        f'<div style="font-size:13px;color:#9CA3AF;margin-top:6px;">{card["subtitle"]}</div>'
        if card.get("subtitle")
        else ""
    )
    delta_html = (
        f'<div style="font-size:14px;color:{border_color};margin-top:4px;">{card["delta"]}</div>'
        if card.get("delta")
        else ""
    )

    card_html = (
        f'<div style="border:2px solid {border_color};border-radius:8px;'
        f'padding:16px;margin-bottom:8px;background-color:#1F2937;">'
        f'<div style="font-size:24px;">{card["icon"]}</div>'
        f'<div style="font-weight:bold;margin:8px 0;color:#FAFAFA;">{card["title"]}</div>'
        f'<div style="font-size:18px;color:{border_color};">{card["value"]}</div>'
        f"{delta_html}"
        f"{subtitle_html}"
        f"</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)
