import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import page_header, plotly_defaults

from data_loader import (
    load_panel, get_dept_names, get_year_slice, get_dept_year, COLUMN_MAP
)

page_header(
    "Map",
    "Department-level indicators across metropolitan France · 2012-2021",
)

# ── GeoJSON (bundled locally) ──────────────────────────────────────────────
_GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "departements.geojson")

@st.cache_data(show_spinner=False)
def load_geojson():
    try:
        with open(_GEOJSON_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

geojson = load_geojson()

# ── Data ───────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df = load_panel()

dept_names = get_dept_names(df)

# ── Controls row ───────────────────────────────────────────────────────────
COLOR_OPTIONS = {
    "Firm rate (per 1,000 inhab.)":  "firm_rate",
    "Median income (disposable, €)": "q2_disp",
    "Higher-ed share (%)":           "edu_share_sup",
    "Unemployment rate (%)":         "unemployment_rate",
    "Doctor density (per 100k)":     "doctor_density_per_100k",
    "% Urban":                       "pct_urban",
    "Poverty rate (%)":              "poverty_rate_disp",
    "Gini coefficient":              "gini_disp",
}

PROFILE_VARS = [
    ("Firm rate",       "firm_rate",              ".1f"),
    ("Median income",   "q2_disp",                ",.0f"),
    ("Higher-ed share", "edu_share_sup",           ".1f"),
    ("Unemployment",    "unemployment_rate",       ".1f"),
    ("Doctor density",  "doctor_density_per_100k", ".0f"),
    ("% Urban",         "pct_urban",               ".1f"),
    ("Poverty rate",    "poverty_rate_disp",       ".1f"),
    ("Gini",            "gini_disp",               ".4f"),
]

col_ctrl, col_yr = st.columns([3, 1])
with col_ctrl:
    color_label = st.selectbox("Color by:", list(COLOR_OPTIONS.keys()))
with col_yr:
    year = st.slider("Year:", 2012, 2021, 2021)

color_col = COLOR_OPTIONS[color_label]

# ── Build choropleth data ──────────────────────────────────────────────────
df_yr = get_year_slice(df, year)
df_map = df_yr[["dep_code", "dep_name", color_col]].copy()
df_map = df_map.rename(columns={"dep_code": "code", "dep_name": "name", color_col: "value"})

# ── Map + profile panel ────────────────────────────────────────────────────
map_col, profile_col = st.columns([3, 1])

with map_col:
    if geojson is None:
        st.error(
            "**GeoJSON not found.** Expected at `app/assets/departements.geojson`.\n\n"
            "Fetch it from:\n"
            "`https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/"
            "departements-version-simplifiee.geojson`"
        )
    else:
        fig_map = px.choropleth(
            df_map,
            geojson=geojson,
            locations="code",
            featureidkey="properties.code",
            color="value",
            hover_name="name",
            hover_data={"code": True, "value": ":.2f"},
            color_continuous_scale="Blues",
            labels={"value": color_label, "code": "Dept."},
        )
        fig_map.update_geos(
            fitbounds="locations",
            visible=False,
            projection_type="mercator",
        )
        fig_map.update_layout(
            **plotly_defaults(),
            height=580,
            coloraxis_colorbar=dict(
                title=dict(text=color_label, font=dict(size=11)),
                len=0.7,
                thickness=12,
            ),
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        fig_map.update_traces(
            marker_line_color="white",
            marker_line_width=0.5,
        )
        st.markdown(
            f'<p class="ri-caption"><strong>{color_label}</strong> · {year}</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_map, use_container_width=True)

with profile_col:
    st.markdown("#### Department profile")
    dept_codes_sorted = sorted(df["dep_code"].unique())
    dept_select = st.selectbox(
        "Select department:",
        options=dept_codes_sorted,
        format_func=lambda c: f"{c} - {dept_names.get(c, c)}",
        key="dept_profile",
    )
    name = dept_names.get(dept_select, dept_select)
    row = get_dept_year(df, dept_select, year)

    if row is not None:
        # National rank on the chosen variable (lower rank = higher value)
        col_vals = df_yr[color_col].dropna()
        val_here = row[color_col]
        rank = int((col_vals > val_here).sum()) + 1
        n_total = len(col_vals)

        profile_rows = ""
        for label, col, fmt in PROFILE_VARS:
            v = row.get(col, None)
            if pd.isna(v):
                formatted = "—"
            elif col == "q2_disp":
                formatted = f"€{v:,.0f}"
            else:
                formatted = f"{v:{fmt}}"
            profile_rows += (
                f'<div class="ri-profile-row">'
                f'<span>{label}</span>'
                f'<span class="ri-profile-val">{formatted}</span>'
                f"</div>"
            )

        density = row.get("density_class", "—")
        st.markdown(
            f'<div class="ri-profile-card">'
            f'<div class="ri-profile-title">{dept_select} · {name}</div>'
            f"{profile_rows}"
            f'<div class="ri-profile-row" style="margin-top:0.4rem">'
            f'<span>Density class</span>'
            f'<span class="ri-profile-val">{density}</span>'
            f"</div>"
            f'<div class="ri-profile-row">'
            f'<span>Rank ({color_label[:18]})</span>'
            f'<span class="ri-profile-val">#{rank} / {n_total}</span>'
            f"</div>"
            f'<p style="font-size:0.72rem;color:#aaa;margin-top:0.6rem">{year} data</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(f"No data for {dept_select} in {year}.")
