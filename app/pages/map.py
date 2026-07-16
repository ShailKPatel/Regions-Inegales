import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import page_header, plotly_defaults, render_footer

from data_loader import (
    load_panel, get_dept_names, get_year_slice, get_dept_year, COLUMN_MAP
)

page_header(
    "Map",
    "Department-level indicators across metropolitan France · 2012-2021",
)

# ── GeoJSON ────────────────────────────────────────────────────────────────
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

# ── Controls ───────────────────────────────────────────────────────────────
COLOR_OPTIONS = {
    "Firm rate (per 1,000 inhab.)":  "firm_rate",
    "Birth rate (per 1,000 inhab.)": "birth_rate",
    "Marriage rate (per 1,000)":     "marriage_rate",
    "Death rate (per 1,000 inhab.)": "death_rate",
    "Median income (disposable, €)": "q2_disp",
    "Higher-ed share (%)":           "edu_share_sup",
    "Unemployment rate (%)":         "unemployment_rate",
    "Doctor density (per 100k)":     "doctor_density_per_100k",
    "% Urban":                       "pct_urban",
    "Poverty rate (%)":              "poverty_rate_disp",
    "Gini coefficient":              "gini_disp",
}

# Profile: demography section
PROFILE_DEMO = [
    ("Birth rate",    "birth_rate",   ".2f"),
    ("Live births",   "live_births",  ",.0f"),
    ("Marriage rate", "marriage_rate",".2f"),
    ("Marriages",     "marriages",    ",.0f"),
    ("Death rate",    "death_rate",   ".2f"),
    ("Deaths",        "deaths",       ",.0f"),
]

# Profile: economy / model features section
PROFILE_ECON = [
    ("Firm rate",       "firm_rate",              ".2f"),
    ("Median income",   "q2_disp",                ",.0f"),
    ("Higher-ed share", "edu_share_sup",           ".1f"),
    ("Unemployment",    "unemployment_rate",       ".1f"),
    ("Doctor density",  "doctor_density_per_100k", ".0f"),
    ("% Urban",         "pct_urban",               ".1f"),
    ("Poverty rate",    "poverty_rate_disp",       ".1f"),
    ("Gini",            "gini_disp",               ".4f"),
]

ctrl_col, yr_col = st.columns([3, 1])
with ctrl_col:
    color_label = st.selectbox("Color by:", list(COLOR_OPTIONS.keys()))
with yr_col:
    year = st.slider("Year:", 2012, 2021, 2021)

color_col = COLOR_OPTIONS[color_label]

# Department selector ABOVE columns so dept_select is available when building the map
dept_codes_sorted = sorted(df["dep_code"].unique())
dept_select = st.selectbox(
    "Highlight department:",
    options=dept_codes_sorted,
    format_func=lambda c: f"{c} – {dept_names.get(c, c)}",
    key="dept_profile",
)

# ── Data slice ─────────────────────────────────────────────────────────────
df_yr = get_year_slice(df, year)
df_map = df_yr[["dep_code", "dep_name", color_col]].copy()
df_map = df_map.rename(columns={"dep_code": "code", "dep_name": "name", color_col: "value"})

# ── Map + profile ──────────────────────────────────────────────────────────
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
        # Colorscale: light blue → brand dark blue; avoids white at low-value end
        _SCALE = [[0, "#bee3f8"], [0.4, "#4a8fc1"], [1, "#0055A4"]]

        fig_map = px.choropleth(
            df_map,
            geojson=geojson,
            locations="code",
            featureidkey="properties.code",
            color="value",
            hover_name="name",
            hover_data={"code": True, "value": ":.2f"},
            color_continuous_scale=_SCALE,
            labels={"value": color_label, "code": "Dept."},
        )
        fig_map.update_geos(
            fitbounds="locations",
            visible=False,
            projection_type="mercator",
        )
        _defaults = plotly_defaults()
        _defaults["paper_bgcolor"] = "#eef2f7"  # light blue-gray; makes low-value depts visible
        fig_map.update_layout(
            **_defaults,
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

        # Gold border overlay for selected department
        fig_map.add_trace(go.Choropleth(
            geojson=geojson,
            locations=[dept_select],
            z=[1],
            featureidkey="properties.code",
            showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker_line_color="#E8A020",
            marker_line_width=3.5,
            hoverinfo="skip",
        ))

        hl_name = dept_names.get(dept_select, dept_select)
        st.markdown(
            f'<p class="ri-caption"><strong>{color_label}</strong> · {year}'
            f' · <span style="color:#E8A020;font-weight:700">■</span>'
            f' highlighted: {dept_select} – {hl_name}</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_map, use_container_width=True)

with profile_col:
    name  = dept_names.get(dept_select, dept_select)
    row   = get_dept_year(df, dept_select, year)

    if row is not None:
        col_vals = df_yr[color_col].dropna()
        val_here = row[color_col]
        rank     = int((col_vals > val_here).sum()) + 1
        n_total  = len(col_vals)

        def _row_html(label, col, fmt):
            v = row.get(col, None)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return (
                    f'<div class="ri-profile-row">'
                    f'<span>{label}</span>'
                    f'<span class="ri-profile-val">—</span>'
                    f'</div>'
                )
            formatted = f"€{v:,.0f}" if col == "q2_disp" else f"{v:{fmt}}"
            return (
                f'<div class="ri-profile-row">'
                f'<span>{label}</span>'
                f'<span class="ri-profile-val">{formatted}</span>'
                f'</div>'
            )

        def _section(text):
            return (
                f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.07em;'
                f'color:#888;text-transform:uppercase;margin:0.55rem 0 0.2rem">'
                f'{text}</div>'
            )

        demo_rows = "".join(_row_html(l, c, f) for l, c, f in PROFILE_DEMO)
        econ_rows = "".join(_row_html(l, c, f) for l, c, f in PROFILE_ECON)
        density   = row.get("density_class", "—")

        st.markdown(
            f'<div class="ri-profile-card">'
            f'<div class="ri-profile-title" '
            f'style="border-left:3px solid #E8A020;padding-left:0.5rem">'
            f'{dept_select} · {name}</div>'
            f'{_section("Demography")}'
            f'{demo_rows}'
            f'{_section("Economy & Model")}'
            f'{econ_rows}'
            f'<div class="ri-profile-row" style="margin-top:0.5rem">'
            f'<span>Density class</span>'
            f'<span class="ri-profile-val">{density}</span>'
            f'</div>'
            f'<div class="ri-profile-row">'
            f'<span>Rank ({color_label[:16]})</span>'
            f'<span class="ri-profile-val">#{rank}&thinsp;/&thinsp;{n_total}</span>'
            f'</div>'
            f'<p style="font-size:0.72rem;color:#aaa;margin-top:0.6rem">{year} data</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(f"No data for {dept_select} in {year}.")

render_footer()
