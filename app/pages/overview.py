import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
from utils import hero_header, stats_strip, plotly_defaults, BLUE, RED, GRAY

hero_header(
    "Régions Inégales",
    "French regional entrepreneurship · 2012-2021",
)

st.markdown(
    """
    Education and income predict where firms form across French departments. Unemployment does not.
    An XGBoost model on 960 department-years assigns opportunity factors **60% of predictive weight**
    versus 15% for necessity. Unemployment ranks last of 8 features and runs negative in the full panel.
    **The necessity hypothesis does not hold for French departments in this period.**
    """
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<h3 class="ri-section-h">Predictive importance by theory group</h3>', unsafe_allow_html=True)
st.markdown('<p class="ri-caption">Mean |SHAP| grouped by theory · XGBoost · 960 dept-years · 2012-2021</p>', unsafe_allow_html=True)

groups = ["Opportunity", "Necessity", "Other"]
shares = [60, 15, 25]
shap_totals = [2.914, 0.727, 1.183]
colors = [BLUE, RED, GRAY]

fig_hero = go.Figure()
fig_hero.add_trace(
    go.Bar(
        x=groups,
        y=shares,
        marker_color=colors,
        text=[f"{s}%<br><span style='font-size:11px'>SHAP {t:.3f}</span>" for s, t in zip(shares, shap_totals)],
        textposition="outside",
        textfont=dict(size=14, color="#1A1A2E"),
        width=0.45,
        hovertemplate="<b>%{x}</b><br>Share: %{y}%<extra></extra>",
    )
)

fig_hero.add_annotation(
    x=0.5, xref="paper",
    y=72, yref="y",
    text="<b>4.0× more important</b><br><span style='color:#666;font-size:11px'>opportunity vs necessity</span>",
    showarrow=True,
    arrowhead=2,
    ax=0, ay=-40,
    font=dict(size=13, color="#0055A4"),
    bgcolor="white",
    bordercolor="#0055A4",
    borderwidth=1,
    borderpad=6,
)

fig_hero.update_layout(
    **plotly_defaults(),
    height=380,
    yaxis=dict(
        title="SHAP importance share (%)",
        range=[0, 82],
        gridcolor="#EBEBF0",
        zeroline=False,
    ),
    xaxis=dict(title=""),
    showlegend=False,
)
fig_hero.update_yaxes(ticksuffix="%")

st.plotly_chart(fig_hero, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    stats_strip([
        ("960",      "dept-years"),
        ("96",       "departments"),
        ("6",        "source datasets"),
        ("REJECTED", "necessity hypothesis", True),
    ]),
    unsafe_allow_html=True,
)
