"""
lib/plot_helpers.py - Plotly 공통 헬퍼

Regime 4 구간 배경색 / 위기 이벤트 annotation / 한영 병기 라벨 등
모든 페이지의 시계열 차트가 공유하는 컴포넌트.

참조: docs/plan/02_common.md 9절, decisionlog/02_overview.md (Regime 정의)
"""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go

from lib.colors import REGIME_COLORS


# === Regime 4 구간 정의 (Overview 영역 3 표준) ========================
# (id, 시작일, 종료일, 한글 라벨)
# - R1 회복기: 2010-01 ~ 2012-06 (글로벌 금융위기 회복)
# - R2 확장기: 2012-07 ~ 2019-12 (저금리 + 안정 성장)
# - R3 변동기: 2020-01 ~ 2023-12 (COVID + 인플레)
# - HO 홀드아웃: 2024-01 ~ 2025-12 (true OOS 평가)
#
# datetime 객체 사용 — plotly 6.x 가 string date 를 annotation 옵션과 함께
# 처리할 때 내부 string+int 연산에서 실패하는 케이스 회피.
REGIME_PERIODS: list[tuple[str, datetime, datetime, str]] = [
    ("R1", datetime(2010, 1, 1),  datetime(2012, 6, 30), "회복기"),
    ("R2", datetime(2012, 7, 1),  datetime(2019, 12, 31), "확장기"),
    ("R3", datetime(2020, 1, 1),  datetime(2023, 12, 31), "변동기"),
    ("HO", datetime(2024, 1, 1),  datetime(2025, 12, 31), "홀드아웃"),
]


# === 위기 이벤트 (Performance / Risk 페이지 공유) =====================
# 차트 위에 점선 + 라벨로 표시
EVENT_MARKERS: list[tuple[datetime, str]] = [
    (datetime(2020, 3, 1),  "▼ COVID-19"),
    (datetime(2022, 9, 1),  "▼ 2022 Bear"),
    (datetime(2024, 12, 1), "▼ AI Rally / IT Rotation"),
]


def add_regime_backgrounds(fig: go.Figure, with_labels: bool = True) -> go.Figure:
    """
    시계열 차트에 4개 Regime 배경색 추가.

    Args:
        fig: Plotly Figure (이미 데이터 trace 가 그려져 있어야 함)
        with_labels: True 면 각 Regime 좌상단에 ID + 한글 라벨 표시

    Returns:
        같은 fig (in-place 수정 + 반환)

    Note:
        - shape (vrect) 와 annotation 을 분리 호출하여 plotly 6.x 의
          axis_spanning_shape_annotation 자동 계산 (datetime sum() 시도) 회피
        - update_xaxes(type="date") 로 x축 type 명시
        - annotation 의 xref="x" + yref="paper" 명시로 좌표계 충돌 방지
    """
    fig.update_xaxes(type="date")
    for regime_id, start, end, label in REGIME_PERIODS:
        # 배경 사각형 (annotation 없이)
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=REGIME_COLORS[regime_id],
            opacity=0.3,
            layer="below",
            line_width=0,
        )
        # 라벨은 별도 annotation (plotly 자동 계산 로직 우회)
        if with_labels:
            fig.add_annotation(
                x=start,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"{regime_id} {label}",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=10, color="#9CA3AF"),
            )
    return fig


def add_event_annotations(fig: go.Figure) -> go.Figure:
    """
    위기 이벤트 점선 + 라벨 추가 (3개: COVID / 2022 Bear / AI Rally).

    Returns:
        같은 fig (in-place 수정 + 반환)

    Note:
        add_vline 의 annotation_text 옵션 사용 시 plotly 6.x 가 datetime 에
        대해 sum() 시도하다 실패 → vline 과 annotation 분리 호출.
    """
    fig.update_xaxes(type="date")
    for date, label in EVENT_MARKERS:
        # 점선 (annotation 없이)
        fig.add_vline(
            x=date,
            line_dash="dot",
            line_color="#EF4444",
            line_width=1,
        )
        # 라벨은 별도 annotation
        fig.add_annotation(
            x=date,
            y=1.0,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color="#EF4444"),
        )
    return fig


def bilingual_label(en: str, ko: str) -> str:
    """
    한/영 병기 라벨 헬퍼 (A-3 결정).

    예: bilingual_label("CAGR", "연환산 수익률") → "CAGR (연환산 수익률)"
    """
    return f"{en} ({ko})"
