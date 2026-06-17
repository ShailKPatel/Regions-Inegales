import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
from utils import page_header, plotly_defaults, BLUE, RED, GRAY

page_header(
    "Model",
    "XGBoost · SHAP attribution · 96 departments · 2012-2021",
)

# ── 1. FEATURE SHAP IMPORTANCE ────────────────────────────────────────────
st.markdown('<h3 class="ri-section-h">Feature importance (mean |SHAP|)</h3>', unsafe_allow_html=True)
st.markdown('<p class="ri-caption">Sorted by contribution · colors encode theory group</p>', unsafe_allow_html=True)

features = [
    ("Median income",       "Opportunity", 1.1762),
    ("Higher-ed share",     "Opportunity", 1.1727),
    ("Wage income share",   "Other",       0.7725),
    ("Poverty rate",        "Necessity",   0.6175),
    ("Gini coefficient",    "Other",       0.4110),
    ("Doctor density",      "Opportunity", 0.3845),
    ("% Urban",             "Opportunity", 0.1809),
    ("Unemployment rate",   "Necessity",   0.1096),
]
GROUP_COLOR = {"Opportunity": BLUE, "Necessity": RED, "Other": GRAY}

feat_names   = [f[0] for f in reversed(features)]
feat_groups  = [f[1] for f in reversed(features)]
feat_vals    = [f[2] for f in reversed(features)]
feat_colors  = [GROUP_COLOR[g] for g in feat_groups]

fig_shap = go.Figure()
fig_shap.add_trace(
    go.Bar(
        x=feat_vals,
        y=feat_names,
        orientation="h",
        marker_color=feat_colors,
        text=[f"{v:.4f}" for v in feat_vals],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP| = %{x:.4f}<extra></extra>",
        width=0.6,
    )
)
for group, color in GROUP_COLOR.items():
    fig_shap.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=color, size=10, symbol="square"),
        name=group,
        showlegend=True,
    ))

fig_shap.update_layout(
    **plotly_defaults(),
    height=370,
    xaxis=dict(title="Mean |SHAP| value", gridcolor="#EBEBF0", zeroline=False),
    yaxis=dict(title=""),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)",
    ),
    bargap=0.3,
)

st.plotly_chart(fig_shap, use_container_width=True)

st.markdown(
    "> Opportunity features dominate. Unemployment rate ranks **last of 8**: "
    "mean |SHAP| 0.110, under 4% of total importance, predominantly negative SHAP values."
)

st.divider()

# ── 2. VALIDATION TABLE ───────────────────────────────────────────────────
st.markdown('<h3 class="ri-section-h">Cross-validation performance</h3>', unsafe_allow_html=True)

st.markdown(
    """
<table class="ri-val-table">
  <thead>
    <tr>
      <th>Scheme</th>
      <th>R²</th>
      <th>MAE</th>
      <th>Note</th>
    </tr>
  </thead>
  <tbody>
    <tr class="headline">
      <td>Leave-One-Dept-Out (LODO) ★</td>
      <td>0.674</td>
      <td>1.308</td>
      <td>Generalisation to <em>unseen departments</em> — headline number</td>
    </tr>
    <tr>
      <td>Leave-One-Year-Out (LOYO)</td>
      <td>0.906</td>
      <td>0.847</td>
      <td>Generalisation to unseen years</td>
    </tr>
    <tr>
      <td>Random 10-fold (KFold)</td>
      <td>0.921</td>
      <td>0.786</td>
      <td>Leaky baseline — do <strong>not</strong> over-interpret</td>
    </tr>
  </tbody>
</table>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="ri-caption" style="margin-top:0.5rem">'
    "★ <strong>LODO R² = 0.674</strong>: the model explains 67% of firm-rate variance "
    "in departments held out from training. The gap versus random KFold (&#916;0.247) "
    "reflects between-department variation not captured by these 8 features."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# ── 3. URBAN vs RURAL SPLIT ───────────────────────────────────────────────
st.markdown('<h3 class="ri-section-h">Urban vs rural: opportunity holds everywhere</h3>', unsafe_allow_html=True)
st.markdown(
    '<p class="ri-caption">'
    "SHAP group shares by density subset · Grille de densité 2025 · "
    "Urban/Intermediate = 45 depts · Rural = 51 depts"
    "</p>",
    unsafe_allow_html=True,
)

contexts  = ["Full panel", "Urban/Intermediate", "Rural"]
opp_vals  = [60, 59, 61]
nec_vals  = [15, 16, 13]
oth_vals  = [25, 25, 26]
ratios    = [4.0, 3.8, 4.6]
lodo_r2   = [0.674, 0.548, 0.653]

col_left, col_right = st.columns(2)

def make_split_bar(subset_idx_list: list, title: str) -> go.Figure:
    ctx  = [contexts[i] for i in subset_idx_list]
    opp  = [opp_vals[i] for i in subset_idx_list]
    nec  = [nec_vals[i] for i in subset_idx_list]
    oth  = [oth_vals[i] for i in subset_idx_list]
    rat  = [ratios[i] for i in subset_idx_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Opportunity", x=ctx, y=opp, marker_color=BLUE,
                         text=[f"{v}%" for v in opp], textposition="auto"))
    fig.add_trace(go.Bar(name="Necessity",   x=ctx, y=nec, marker_color=RED,
                         text=[f"{v}%" for v in nec], textposition="auto"))
    fig.add_trace(go.Bar(name="Other",       x=ctx, y=oth, marker_color=GRAY,
                         text=[f"{v}%" for v in oth], textposition="auto"))

    for i, (c, r) in enumerate(zip(ctx, rat)):
        fig.add_annotation(
            x=c, y=max(opp[i], nec[i], oth[i]) + 12,
            text=f"<b>{r}× ratio</b>",
            showarrow=False,
            font=dict(size=11, color=BLUE),
        )

    fig.update_layout(
        **plotly_defaults(),
        title=dict(text=title, font=dict(size=13)),
        height=350,
        barmode="group",
        yaxis=dict(range=[0, 85], gridcolor="#EBEBF0", title="SHAP share (%)", ticksuffix="%"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    return fig

with col_left:
    fig_urban = make_split_bar([0, 1], "Urban / Intermediate departments")
    st.plotly_chart(fig_urban, use_container_width=True)

with col_right:
    fig_rural = make_split_bar([0, 2], "Rural departments")
    st.plotly_chart(fig_rural, use_container_width=True)

m1, m2, m3 = st.columns(3)
for col, label, r2 in zip(
    [m1, m2, m3],
    ["Full panel", "Urban/Intermediate", "Rural"],
    lodo_r2,
):
    col.metric(f"LODO R² · {label}", f"{r2:.3f}")

st.markdown(
    '<div class="ri-note">'
    "<strong>Rural subset:</strong> OLS finds a positive unemployment coefficient in rural "
    "departments (coef = +0.214, p &lt; 0.001). SHAP still ranks unemployment last of 8 "
    "in the rural-only model, and a pooled interaction test finds no significant urban/rural "
    "difference (p = 0.870)."
    "</div>",
    unsafe_allow_html=True,
)
