import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from utils import page_header, plotly_defaults, BLUE, RED, WHITE

from data_loader import load_panel, get_dept_names

page_header(
    "Explore",
    "Firm creation rate vs. structural indicators · 960 dept-years",
)

# ── Row 1: primary controls ────────────────────────────────────────────────
X_FEATURES = {
    "Median income (€)":         "q2_disp",
    "Higher-ed share (%)":       "edu_share_sup",
    "Unemployment rate (%)":     "unemployment_rate",
    "Poverty rate (%)":          "poverty_rate_disp",
    "Gini coefficient":          "gini_disp",
    "Doctor density (per 100k)": "doctor_density_per_100k",
    "% Urban":                   "pct_urban",
    "Wage income share (%)":     "pct_wages",
}
COLOR_BY = {
    "Density class (urban/rural)": "density_class",
    "Year":                        "year",
    "None":                        "none",
}

ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 1])
with ctrl1:
    x_label = st.selectbox("X axis (feature):", list(X_FEATURES.keys()))
with ctrl2:
    color_label = st.selectbox("Color by:", list(COLOR_BY.keys()))
with ctrl3:
    year_filter = st.selectbox("Year:", ["All"] + list(range(2012, 2022)))

x_key     = X_FEATURES[x_label]
color_key = COLOR_BY[color_label]

# ── Load data ──────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df_full = load_panel()

dept_names     = get_dept_names(df_full)
all_dept_codes = sorted(df_full["dep_code"].unique())

def _fmt(c):
    return f"{c} · {dept_names.get(c, c)}"

# ── Row 2: highlight / hide multiselects ───────────────────────────────────
col_hl, col_hide = st.columns(2)
with col_hl:
    highlight_depts = st.multiselect(
        "Highlight departments:",
        options=all_dept_codes,
        format_func=_fmt,
        placeholder="None selected (mark in gold)",
    )
with col_hide:
    hide_depts = st.multiselect(
        "Hide departments:",
        options=all_dept_codes,
        format_func=_fmt,
        placeholder="None selected (excludes from plot)",
    )

hide_set      = set(hide_depts)
highlight_set = set(highlight_depts) - hide_set   # hidden takes priority

# ── Filter data ────────────────────────────────────────────────────────────
df = df_full.copy()
if year_filter != "All":
    df = df[df["year"] == int(year_filter)]

plot_df  = df.dropna(subset=[x_key, "firm_rate"])
plot_df  = plot_df[~plot_df["dep_code"].isin(hide_set)]
base_df  = plot_df[~plot_df["dep_code"].isin(highlight_set)]
hl_df    = plot_df[plot_df["dep_code"].isin(highlight_set)]

x_vals    = plot_df[x_key].values
firm_rate = plot_df["firm_rate"].values
N         = len(plot_df)

# ── Trendline ──────────────────────────────────────────────────────────────
def compute_trend(x, y):
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_range = np.linspace(x.min(), x.max(), 200)
    return slope, r, x_range, intercept + slope * x_range

slope, r_val, x_range, trend_y = compute_trend(x_vals, firm_rate)

# r for all non-hidden departments (full reference, ignoring highlight)
r_all_val = r_val
if hide_set:
    df_all = df.dropna(subset=[x_key, "firm_rate"])
    x_a = df_all[x_key].values
    y_a = df_all["firm_rate"].values
    r_all_val = compute_trend(x_a, y_a)[1] if len(x_a) > 1 else float("nan")

# ── Hover helper ───────────────────────────────────────────────────────────
def make_hover(sub, suffix=""):
    return [
        f"<b>{dept_names.get(r.dep_code, r.dep_code)}</b> ({r.dep_code}) · {r.year}"
        f"<br>{x_label}: {getattr(r, x_key):.2f}"
        f"<br>Firm rate: {r.firm_rate:.2f}"
        + (f"<br><em>{suffix}</em>" if suffix else "")
        for r in sub.itertuples()
    ]

# ── Build scatter ──────────────────────────────────────────────────────────
DENSITY_COLOR = {"urban": BLUE, "intermediate": "#6B8FD4", "rural": WHITE}

fig = go.Figure()

if color_key == "density_class":
    for group, clr in DENSITY_COLOR.items():
        sub = base_df[base_df["density_class"] == group]
        fig.add_trace(go.Scatter(
            x=sub[x_key], y=sub["firm_rate"],
            mode="markers", name=group.capitalize(),
            marker=dict(color=clr, size=5, opacity=0.6),
            text=make_hover(sub),
            hovertemplate="%{text}<extra></extra>",
        ))
elif color_key == "year":
    fig.add_trace(go.Scatter(
        x=base_df[x_key], y=base_df["firm_rate"],
        mode="markers",
        marker=dict(
            color=base_df["year"], colorscale="Blues",
            size=5, opacity=0.65,
            colorbar=dict(title="Year", len=0.6, thickness=12),
        ),
        name="Department-year",
        text=make_hover(base_df),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
else:
    fig.add_trace(go.Scatter(
        x=base_df[x_key], y=base_df["firm_rate"],
        mode="markers",
        marker=dict(color=BLUE, size=5, opacity=0.55),
        name="Department-year",
        text=make_hover(base_df),
        hovertemplate="%{text}<extra></extra>",
    ))

# Trendline
fig.add_trace(go.Scatter(
    x=x_range, y=trend_y, mode="lines",
    line=dict(color=RED, width=2, dash="dash"),
    name="OLS trendline", hoverinfo="skip",
))

# Highlighted departments overlay
if len(hl_df) > 0:
    fig.add_trace(go.Scatter(
        x=hl_df[x_key], y=hl_df["firm_rate"],
        mode="markers", name="Highlighted",
        marker=dict(
            color="#E8A020", size=8, opacity=0.9, symbol="diamond",
            line=dict(color="#B07010", width=0.8),
        ),
        text=make_hover(hl_df, suffix="Highlighted"),
        hovertemplate="%{text}<extra></extra>",
    ))

# Annotation: slope + r
slope_txt = f"beta={slope:+.4f}" if abs(slope) < 1 else f"beta={slope:+.2f}"
fig.add_annotation(
    xref="paper", yref="paper", x=0.98, y=0.05,
    text=f"<b>{slope_txt}  r={r_val:.3f}</b>",
    showarrow=False,
    bgcolor="white", bordercolor="#DDD", borderwidth=1, borderpad=5,
    font=dict(size=12, color="#333"), xanchor="right",
)

hide_note = f" · {len(hide_set)} dept(s) hidden" if hide_set else ""
yr_note   = f" · {year_filter}" if year_filter != "All" else " · all years"
fig.update_layout(
    **plotly_defaults(),
    height=480,
    xaxis=dict(title=x_label, gridcolor="#EBEBF0", zeroline=False),
    yaxis=dict(title="Firm creation rate (per 1,000 inhab.)", gridcolor="#EBEBF0", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    title=dict(
        text=f"Firm rate vs {x_label}{hide_note}{yr_note} · N={N}",
        font=dict(size=12, color="#888"),
    ),
)

st.plotly_chart(fig, use_container_width=True)

# ── r comparison ───────────────────────────────────────────────────────────
st.markdown("---")
col_r1, col_r2 = st.columns(2)
with col_r1:
    lbl = "r (current selection)" if hide_set else "r (all departments)"
    st.metric(lbl, f"{r_val:.3f}")
with col_r2:
    if hide_set and not np.isnan(r_all_val):
        delta = r_all_val - r_val
        st.metric("r (all departments)", f"{r_all_val:.3f}", delta=f"{delta:+.3f}")
    else:
        st.metric("r (all departments)", f"{r_all_val:.3f}")
