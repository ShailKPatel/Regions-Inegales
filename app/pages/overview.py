import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_CHOROPLETH = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "firm_rate_choropleth.png")

import streamlit as st
import plotly.graph_objects as go
from utils import hero_header, stats_strip, plotly_defaults, render_footer, BLUE, RED, WHITE

hero_header(
    "Régions Inégales",
    "French regional entrepreneurship · 2012-2021",
)

st.markdown(
    '<p style="font-size:1.5rem;font-weight:700;color:#1A1A2E;margin-bottom:0.5rem">'
    "The necessity hypothesis is rejected."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    Education and income predict where firms form across French departments. Unemployment does not.
    An XGBoost model on 960 department-years assigns opportunity factors **58% of predictive weight**
    versus 20% for necessity. Unemployment ranks last of 8 features and runs negative in the full panel.
    """
)

st.markdown("<br>", unsafe_allow_html=True)

_NH_TIP = (
    "Necessity hypothesis: unemployment pushes people into self-employment.&#10;"
    "&#10;"
    "&#x2022; Unemployment ranks last of 8 SHAP features.&#10;"
    "&#x2022; SHAP values are predominantly negative.&#10;"
    "&#x2022; Opportunity factors carry 3.0&#215; more weight."
)

st.markdown(
    stats_strip([
        ("10",       "years"),
        ("96",       "departments"),
        ("9",        "source datasets"),
        ("REJECTED", f'necessity hypothesis <span class="ri-info-tip" data-tip="{_NH_TIP}">i</span>', True),
    ]),
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<h3 class="ri-section-h">Where firms form: a geography of creation rates</h3>', unsafe_allow_html=True)
st.markdown('<p class="ri-caption">Mean firm creations per 1,000 residents · 96 metropolitan departments · 2012-2021</p>', unsafe_allow_html=True)

if os.path.exists(_CHOROPLETH):
    col_map, col_txt = st.columns([2, 1])
    with col_map:
        st.image(_CHOROPLETH, use_container_width=True)
    with col_txt:
        st.markdown(
            """
            Firm creation rates span a **5x range** across metropolitan France.
            Paris (75) averages 28.4 new firms per 1,000 residents each year, while
            rural departments in the northeast (Meuse, Haute-Marne) and the Massif
            Central (Cantal) sit below 6. The Riviera coast and the Ile-de-France
            ring are consistently elevated.

            This sharp spatial gradient is the puzzle the model addresses: is it
            driven by opportunity (income, education, urban density) or by necessity
            (unemployment, poverty)? The map makes clear the pattern is structural,
            not cyclical, persisting across the entire 2012-2021 period.
            """
        )

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<h3 class="ri-section-h">Predictive importance by theory group</h3>', unsafe_allow_html=True)
st.markdown('<p class="ri-caption">Mean absolute SHAP grouped by theory · XGBoost · 960 dept-years · 2012-2021</p>', unsafe_allow_html=True)

groups = ["Opportunity", "Other", "Necessity"]
shares = [58, 22, 20]
shap_totals = [2.7463, 1.0426, 0.9236]
colors = [BLUE, WHITE, RED]

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
    text="<b>3.0&#215; more important</b><br><span style='color:#666;font-size:11px'>opportunity vs necessity</span>",
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

st.markdown(
    "This runs against the necessity-entrepreneurship hypothesis, which holds that unemployment "
    "and hardship push people into self-employment. In French departments, the opposite pattern dominates. "
    "This pattern holds across the period: the necessity channel does not strengthen over 2012-2021, it weakens."
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h3 class="ri-section-h">Robustness</h3>', unsafe_allow_html=True)
st.markdown(
    """
- **Urban vs rural:** opportunity dominates in both density subsets. Rural departments show a 4.19× Opp/Nec ratio, higher than urban (1.95×), the opposite of what a necessity account predicts.
- **Over time:** unemployment's relationship with firm creation weakens across 2012–2021. Opportunity features (income, education) strengthen. The finding is not a period artefact.
- **OLS confirmation:** unemployment rate runs **negative** in both unweighted (coef = −0.30, p = 0.044) and population-weighted (coef = −0.66, p = 0.001) OLS specs. Higher unemployment accompanies fewer firms, not more.
"""
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h3 class="ri-section-h">Demographic appendix</h3>', unsafe_allow_html=True)
st.markdown(
    "A secondary model targeting **birth rate** (not firm creation) uses the same LODO framework "
    "with marriage rate, income, urban structure, and other indicators. LODO R² = 0.715. "
    "Unemployment is again the weakest predictor (p = 0.63 in OLS). "
    "This is a methodological extension, not a co-finding. "
    "Full details on the **Model** page."
)

render_footer()
