import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
from utils import page_header, plotly_defaults, render_footer, BLUE, RED, WHITE

page_header(
    "Model",
    "XGBoost · SHAP attribution · 96 departments · 2012-2021",
)

# ── 1. FEATURE SHAP IMPORTANCE ────────────────────────────────────────────
st.markdown('<h3 class="ri-section-h">Feature importance (mean absolute SHAP)</h3>', unsafe_allow_html=True)
st.markdown('<p class="ri-caption">Sorted by contribution · colors encode theory group</p>', unsafe_allow_html=True)

features = [
    ("Median income",       "Opportunity", 1.1140),
    ("Higher-ed share",     "Opportunity", 1.0505),
    ("Poverty rate",        "Necessity",   0.7383),
    ("Wage income share",   "Other",       0.6329),
    ("Gini coefficient",    "Other",       0.4096),
    ("Doctor density",      "Opportunity", 0.3957),
    ("% Urban",             "Opportunity", 0.1860),
    ("Unemployment rate",   "Necessity",   0.1853),
]
GROUP_COLOR = {"Opportunity": BLUE, "Other": WHITE, "Necessity": RED}

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
        hovertemplate="<b>%{y}</b><br>Abs. SHAP = %{x:.4f}<extra></extra>",
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
    xaxis=dict(title="Mean absolute SHAP", gridcolor="#EBEBF0", zeroline=False),
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
    "mean abs. SHAP 0.185, 4% of total importance, predominantly negative SHAP values."
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
      <td>0.678</td>
      <td>1.4951</td>
      <td>Generalisation to <em>unseen departments</em>; primary result</td>
    </tr>
    <tr>
      <td>Leave-One-Year-Out (LOYO)</td>
      <td>0.929</td>
      <td>0.7564</td>
      <td>Generalisation to unseen years; inflated by cross-sectional overlap with training departments, not a test of generalisation to new units</td>
    </tr>
    <tr>
      <td>Random 10-fold (KFold)</td>
      <td>0.932</td>
      <td>0.7479</td>
      <td>Leaky baseline; inflated upper bound</td>
    </tr>
  </tbody>
</table>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="ri-caption" style="margin-top:0.5rem">'
    "★ <strong>LODO R² = 0.678</strong>: the model explains 68% of firm-rate variance "
    "in departments held out from training. The gap versus random KFold (&#916;0.254) "
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
opp_vals  = [58, 54, 61]
nec_vals  = [20, 27, 15]
oth_vals  = [22, 19, 25]
ratios    = [2.97, 1.95, 4.19]
lodo_r2   = [0.678, 0.573, 0.603]

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
    fig.add_trace(go.Bar(name="Other",       x=ctx, y=oth, marker_color=WHITE,
                         text=[f"{v}%" for v in oth], textposition="auto"))
    fig.add_trace(go.Bar(name="Necessity",   x=ctx, y=nec, marker_color=RED,
                         text=[f"{v}%" for v in nec], textposition="auto"))

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
    "Opportunity dominates in both contexts, and markedly more so in rural departments (4.19x) "
    "than urban (1.95x), contrary to a necessity-driven account of rural entrepreneurship."
)

st.markdown(
    '<div class="ri-note">'
    "<strong>Rural subset:</strong> OLS finds a positive unemployment coefficient in rural "
    "departments (coef = +0.081, clustered p = 0.568 unweighted / 0.761 pop-weighted), "
    "well short of conventional significance after clustering standard errors by department. "
    "SHAP still ranks unemployment last of 8 in the rural-only model, and a pooled interaction "
    "test finds no significant urban/rural difference (clustered p = 0.231 unweighted, p = 0.401 "
    "pop-weighted)."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "**Necessity over time — mixed:** year-interaction OLS on the full 960-row panel shows "
    "unemployment weakening (unemployment × year p = 0.033 UW, p < 0.001 WT; coef = −0.083 per year) "
    "while poverty strengthens (poverty × year p < 0.001 both specs; coef = +0.237 UW). The two "
    "necessity features move in opposite directions, so the necessity channel as a whole is not "
    "simply flat or fading. Opportunity features also strengthen (income × year and edu × year "
    "both significant). Opportunity dominates in all 10 year-by-year SHAP models regardless."
)

st.divider()

st.markdown('<h3 class="ri-section-h">Additional tests</h3>', unsafe_allow_html=True)
st.markdown(
    """
- **Inequality (Gini)** was tested as a predictor and found inconclusive: it ranks 5th of 8, its importance weakens when Île-de-France is excluded, and its sign depends on the weighting scheme.
"""
)

st.divider()

st.markdown('<h3 class="ri-section-h">Cross-model robustness</h3>', unsafe_allow_html=True)
st.markdown(
    "Four models — ElasticNetCV, RandomForest, LightGBM, XGBoost — were retrained on the identical "
    "8-feature matrix and identical LODO folds (GroupKFold, 96 departments). Top-3 features "
    "(higher-ed share, median income, poverty rate) agree across all four, and unemployment ranks "
    "in the bottom half for every model. Minimum pairwise Spearman rank correlation across the four "
    "models: +0.81. On pooled out-of-fold predictions, ElasticNetCV (linear) generalizes best "
    "(R² = 0.714), ahead of tuned XGBoost (R² = 0.676), LightGBM (R² = 0.634), and RandomForest "
    "(R² = 0.647) — findings don't depend on tree-model nonlinearity."
)
st.markdown(
    """
| Model | LODO R² | Top-3 match | Unemployment rank (of 8) |
|---|---|---|---|
| ElasticNetCV | 0.714 | Yes | 6 |
| XGBoost | 0.676 | Yes | 7 |
| RandomForest | 0.647 | Yes | 8 |
| LightGBM | 0.634 | Yes | 8 |
"""
)

st.divider()

# ── 4. BIRTH RATE MODEL ───────────────────────────────────────────────────
st.markdown('<h3 class="ri-section-h">Appendix: birth rate determinants (secondary analysis)</h3>', unsafe_allow_html=True)
st.markdown(
    '<p class="ri-caption">'
    "Not a co-finding, a methodological extension applying the same LODO + OOF SHAP framework "
    "to a different target: birth rate (live births per 1,000 inhab.) · "
    "2012–2021 · 960 dept-years · uses marriage rate, deaths, and births from three new sources"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "Three channels tested: **Social** (marriage rate), **Economic** (income, unemployment, poverty), "
    "and **Structural** (education, urbanisation, doctor density, Gini)."
)

birth_features = [
    ("% Urban",            "Structural", 1.1452),
    ("Poverty rate",       "Economic",   0.3797),
    ("Marriage rate",      "Social",     0.2695),
    ("Median income",      "Economic",   0.2610),
    ("Doctor density",     "Structural", 0.2213),
    ("Unemployment rate",  "Economic",   0.1380),
    ("Gini coefficient",   "Structural", 0.0939),
    ("Higher-ed share",    "Structural", 0.0700),
]

BIRTH_COLOR = {"Social": "#1565c0", "Economic": "#2e7d32", "Structural": "#6a1b9a"}

b_names  = [f[0] for f in reversed(birth_features)]
b_groups = [f[1] for f in reversed(birth_features)]
b_vals   = [f[2] for f in reversed(birth_features)]
b_colors = [BIRTH_COLOR[g] for g in b_groups]

fig_birth = go.Figure()
fig_birth.add_trace(go.Bar(
    x=b_vals, y=b_names,
    orientation="h",
    marker_color=b_colors,
    text=[f"{v:.4f}" for v in b_vals],
    textposition="outside",
    textfont=dict(size=12),
    hovertemplate="<b>%{y}</b><br>Abs. SHAP = %{x:.4f}<extra></extra>",
    width=0.6,
))
for group, color in BIRTH_COLOR.items():
    fig_birth.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=color, size=10, symbol="square"),
        name=group,
        showlegend=True,
    ))

fig_birth.update_layout(
    **plotly_defaults(),
    height=350,
    xaxis=dict(title="Mean absolute SHAP", gridcolor="#EBEBF0", zeroline=False),
    yaxis=dict(title=""),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)",
    ),
    bargap=0.3,
)
st.plotly_chart(fig_birth, use_container_width=True)

# CV table
st.markdown(
    """
<table class="ri-val-table">
  <thead>
    <tr><th>Scheme</th><th>R²</th><th>MAE</th></tr>
  </thead>
  <tbody>
    <tr class="headline">
      <td>Leave-One-Dept-Out (LODO) ★</td><td>0.715</td><td>0.8406</td>
    </tr>
    <tr>
      <td>Leave-One-Year-Out (LOYO)</td><td>0.952</td><td>0.3417</td>
    </tr>
    <tr>
      <td>Random 10-fold (KFold)</td><td>0.948</td><td>0.3517</td>
    </tr>
  </tbody>
</table>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="ri-caption" style="margin-top:0.5rem">'
    "★ LODO R² = 0.715, stronger generalisation than the firm-rate model (0.678)."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
**Group shares (OOF SHAP):** Structural 59% · Economic 30% · Social 10%.

**Key OLS findings** (department-clustered SE):
- Marriage rate: +0.43 per 1,000, p = 0.001 (robust, both weighting specs)
- Median income: −0.0006, p < 0.001 (demographic transition: richer departments have fewer births)
- % Urban: +0.08, p < 0.001 (urban departments have higher birth rates, driven by younger age structure)
- Unemployment rate: −0.05, p = 0.63 (not significant, parallel to the firm-rate model)

**Structural note:** % Urban alone accounts for 44% of total OOF SHAP in this model, reflecting persistent demographic concentration in metropolitan France. Marriage rate is the most interpretable individual predictor (OLS p = 0.001).
"""
)

render_footer()
