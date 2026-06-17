import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import page_header, plotly_defaults, BLUE, RED, GRAY

page_header(
    "Explore",
    "Firm creation rate vs. structural indicators · 960 dept-years",
)

# ── Controls ───────────────────────────────────────────────────────────────
X_FEATURES = {
    "Median income (€)":             "median_income",
    "Higher-ed share (%)":           "edu_share_sup",
    "Unemployment rate (%)":         "unemployment_rate",
    "Poverty rate (%)":              "poverty_rate_disp",
    "Gini coefficient":              "gini_disp",
    "Doctor density (per 100k)":     "doctor_density_per_100k",
    "% Urban":                       "pct_urban",
    "Wage income share (%)":         "pct_wages",
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
    exclude_idf = st.checkbox("Exclude Île-de-France", value=False)

x_key     = X_FEATURES[x_label]
color_key = COLOR_BY[color_label]

# ── Dummy data generation (TODO-REAL: replace with CSV load) ───────────────
N = 960 if not exclude_idf else 880  # TODO-REAL: actually filter dept codes 75,77,78,91-95

rng = np.random.default_rng(seed=42 + hash(x_key) % 1000)

# Generate dummy x values with rough realistic ranges per feature
RANGES = {
    "median_income":          (18_000, 38_000),
    "edu_share_sup":          (15,     52),
    "unemployment_rate":      (3.5,    16),
    "poverty_rate_disp":      (8,      28),
    "gini_disp":              (0.24,   0.34),
    "doctor_density_per_100k":(160,    420),
    "pct_urban":              (10,     100),
    "pct_wages":              (48,     78),
}
lo, hi = RANGES.get(x_key, (0, 100))
x_vals = rng.uniform(lo, hi, N)

# Dummy firm rate correlated with x (direction varies by feature)
DIRECTION = {
    "median_income": +1, "edu_share_sup": +1, "unemployment_rate": -1,
    "poverty_rate_disp": +0.3, "gini_disp": 0, "doctor_density_per_100k": +0.6,
    "pct_urban": +0.8, "pct_wages": -0.4,
}
d = DIRECTION.get(x_key, 0)
base_rate = 12 + d * (x_vals - x_vals.mean()) / x_vals.std() * 4
firm_rate  = base_rate + rng.normal(0, 2.5, N)
firm_rate  = np.clip(firm_rate, 3, 35)

# Dummy color column
if color_key == "density_class":
    density_choices = rng.choice(["Urban", "Intermediate", "Rural"],
                                  size=N, p=[0.15, 0.31, 0.54])
    color_vals  = density_choices.tolist()
    color_map   = {"Urban": BLUE, "Intermediate": "#6B8FD4", "Rural": GRAY}
    color_array = [color_map[v] for v in color_vals]
    color_title = "Density class"
elif color_key == "year":
    year_vals   = rng.integers(2012, 2022, N)
    color_vals  = year_vals.tolist()
    color_array = None
    color_title = "Year"
else:
    color_vals  = None
    color_array = [BLUE] * N
    color_title = None

# ── Trendline via numpy polyfit ────────────────────────────────────────────
z   = np.polyfit(x_vals, firm_rate, 1)
p   = np.poly1d(z)
x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
trend_y = p(x_range)

df_scatter = pd.DataFrame({
    "x":         x_vals,
    "firm_rate": firm_rate,
    "color":     color_vals if color_vals is not None else [BLUE] * N,
})

# ── Build figure ───────────────────────────────────────────────────────────
fig = go.Figure()

if color_key == "none" or color_key == "density_class":
    if color_key == "density_class":
        for group, clr in color_map.items():
            mask = df_scatter["color"] == group
            fig.add_trace(go.Scatter(
                x=df_scatter.loc[mask, "x"],
                y=df_scatter.loc[mask, "firm_rate"],
                mode="markers",
                name=group,
                marker=dict(color=clr, size=5, opacity=0.6),
                hovertemplate=f"<b>{group}</b><br>{x_label}: %{{x:.2f}}<br>Firm rate: %{{y:.2f}}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=x_vals, y=firm_rate, mode="markers",
            marker=dict(color=BLUE, size=5, opacity=0.55),
            name="Department-year",
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>Firm rate: %{{y:.2f}}<extra></extra>",
        ))
elif color_key == "year":
    fig.add_trace(go.Scatter(
        x=x_vals, y=firm_rate, mode="markers",
        marker=dict(
            color=color_vals, colorscale="Blues",
            size=5, opacity=0.6,
            colorbar=dict(title="Year", len=0.6, thickness=12),
        ),
        name="Department-year",
        hovertemplate=f"{x_label}: %{{x:.2f}}<br>Firm rate: %{{y:.2f}}<br>Year: %{{marker.color}}<extra></extra>",
        showlegend=False,
    ))

# Trendline
fig.add_trace(go.Scatter(
    x=x_range, y=trend_y, mode="lines",
    line=dict(color=RED, width=2, dash="dash"),
    name="OLS trendline",
    hoverinfo="skip",
))

# Annotation: slope direction
slope_txt = f"β ≈ {z[0]:+.3f}" if abs(z[0]) < 1 else f"β ≈ {z[0]:+.1f}"
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.98, y=0.05,
    text=f"<b>{slope_txt}</b>",
    showarrow=False,
    bgcolor="white",
    bordercolor="#DDD",
    borderwidth=1,
    borderpad=5,
    font=dict(size=12, color="#333"),
    xanchor="right",
)

idf_note = " · Île-de-France excluded" if exclude_idf else ""
fig.update_layout(
    **plotly_defaults(),
    height=480,
    xaxis=dict(title=x_label, gridcolor="#EBEBF0", zeroline=False),
    yaxis=dict(title="Firm creation rate (per 1,000 inhab.)", gridcolor="#EBEBF0", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    title=dict(
        text=f"Firm rate vs {x_label}{idf_note} · N={N} dept-years",
        font=dict(size=12, color="#888"),
    ),
)

st.plotly_chart(fig, use_container_width=True)

