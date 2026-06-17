import streamlit as st
from utils import inject_css

st.set_page_config(
    page_title="Régions Inégales",
    page_icon="🇫🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

st.sidebar.markdown(
    '<div class="ri-sidebar-brand">'
    '<div class="ri-sidebar-title">Régions Inégales</div>'
    '<div class="ri-sidebar-sub">French regional entrepreneurship</div>'
    '<hr class="ri-sidebar-rule">'
    '</div>',
    unsafe_allow_html=True,
)

pg = st.navigation(
    [
        st.Page("pages/overview.py",     title="Overview"),
        st.Page("pages/map.py",          title="Map"),
        st.Page("pages/model.py",        title="Model"),
        st.Page("pages/explore.py",      title="Explore"),
        st.Page("pages/data_methods.py", title="Methods"),
    ]
)

pg.run()
