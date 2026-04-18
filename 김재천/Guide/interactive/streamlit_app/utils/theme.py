"""테마 관리 (라이트/다크/자동)."""
import streamlit as st


THEMES = {
    'light': {
        'name': '🌞 라이트',
        'bg': '#FFFFFF',
        'secondary_bg': '#F5F5F5',
        'text': '#212121',
        'muted': '#616161',
        'primary': '#1976D2',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'danger': '#F44336',
        'plotly_template': 'plotly_white',
    },
    'dark': {
        'name': '🌙 다크',
        'bg': '#0E1117',
        'secondary_bg': '#1E1E2E',
        'text': '#FAFAFA',
        'muted': '#B0B0B0',
        'primary': '#448AFF',
        'success': '#66BB6A',
        'warning': '#FFA726',
        'danger': '#EF5350',
        'plotly_template': 'plotly_dark',
    },
    'auto': {
        'name': '🤖 자동',
        'bg': '#FFFFFF',
        'secondary_bg': '#F5F5F5',
        'text': '#212121',
        'muted': '#616161',
        'primary': '#1976D2',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'danger': '#F44336',
        'plotly_template': None,  # Streamlit 자동 따라감
    },
}


def get_current_theme():
    """현재 선택된 테마 반환."""
    theme_key = st.session_state.get('theme', 'light')
    return THEMES[theme_key]


def apply_custom_css():
    """테마에 맞춰 CSS 적용."""
    t = get_current_theme()
    css = f"""
    <style>
        /* KPI 카드 스타일 */
        .metric-card {{
            background: {t['secondary_bg']};
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid {t['primary']};
            margin-bottom: 10px;
        }}
        .metric-card .label {{
            font-size: 12px;
            color: {t['muted']};
        }}
        .metric-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: {t['primary']};
        }}
        /* 배지 */
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            color: white;
        }}
        .badge-success {{ background: {t['success']}; }}
        .badge-warning {{ background: {t['warning']}; }}
        .badge-danger {{ background: {t['danger']}; }}
        .badge-primary {{ background: {t['primary']}; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_theme_selector():
    """사이드바에 테마 선택 위젯 렌더링."""
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'light'
    st.sidebar.markdown('### 🎨 테마')
    options = {'light': '🌞 라이트', 'dark': '🌙 다크', 'auto': '🤖 자동'}
    choice = st.sidebar.radio(
        '테마 선택',
        options=list(options.keys()),
        format_func=lambda k: options[k],
        index=list(options.keys()).index(st.session_state['theme']),
        key='theme_radio',
        label_visibility='collapsed',
    )
    if choice != st.session_state['theme']:
        st.session_state['theme'] = choice
        st.rerun()
