import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from utils import page_header, plotly_defaults, BLUE

page_header(
    "Map",
    "Department-level indicators across metropolitan France · 2012-2021",
)

# ── 96 metropolitan department codes ──────────────────────────────────────
METRO_DEPT_CODES = (
    [f"{i:02d}" for i in range(1, 20)]   # 01–19
    + ["2A", "2B"]                         # Corsica
    + [f"{i:02d}" for i in range(21, 96)] # 21–95
)

DEPT_NAMES = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes",
    "09": "Ariège", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze",
    "2A": "Corse-du-Sud", "2B": "Haute-Corse",
    "21": "Côte-d'Or", "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne",
    "25": "Doubs", "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir",
    "29": "Finistère", "30": "Gard", "31": "Haute-Garonne", "32": "Gers",
    "33": "Gironde", "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre",
    "37": "Indre-et-Loire", "38": "Isère", "39": "Jura", "40": "Landes",
    "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire", "44": "Loire-Atlantique",
    "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne", "48": "Lozère",
    "49": "Maine-et-Loire", "50": "Manche", "51": "Marne", "52": "Haute-Marne",
    "53": "Mayenne", "54": "Meurthe-et-Moselle", "55": "Meuse", "56": "Morbihan",
    "57": "Moselle", "58": "Nièvre", "59": "Nord", "60": "Oise",
    "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin",
    "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône", "71": "Saône-et-Loire",
    "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie", "75": "Paris",
    "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines",
    "79": "Deux-Sèvres", "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne",
    "83": "Var", "84": "Vaucluse", "85": "Vendée", "86": "Vienne",
    "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne", "90": "Territoire de Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise",
}

COLOR_BY_OPTIONS = {
    "Firm rate (per 1,000 inhab.)": "firm_rate",
    "Median income (disposable)":   "median_income",
    "Education share (higher-ed %)":"education",
    "Unemployment rate (%)":        "unemployment",
    "Doctor density (per 100k)":    "doctor_density",
    "% Urban":                      "pct_urban",
}

# ── Controls row ───────────────────────────────────────────────────────────
col_ctrl, col_yr = st.columns([3, 1])
with col_ctrl:
    color_label = st.selectbox("Color by:", list(COLOR_BY_OPTIONS.keys()))
with col_yr:
    year = st.selectbox("Year:", list(range(2012, 2022)), index=9)

color_key = COLOR_BY_OPTIONS[color_label]

# ── Dummy data (TODO-REAL: replace with master CSV lookup in Phase 2) ─────
rng = np.random.default_rng(seed=hash(color_key) % (2**31))
dummy_values = rng.uniform(10, 90, size=len(METRO_DEPT_CODES))

df_map = pd.DataFrame({
    "code":  METRO_DEPT_CODES,
    "name":  [DEPT_NAMES.get(c, c) for c in METRO_DEPT_CODES],
    "value": dummy_values,
})
# TODO-REAL: load from filosofi_panel.csv filtered to `year`, pivot by color_key

# ── GeoJSON load with fallback ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=86400)
def load_geojson():
    url = (
        "https://raw.githubusercontent.com/gregoiredavid/"
        "france-geojson/master/departements-version-simplifiee.geojson"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

with st.spinner("Loading map data…"):
    geojson = load_geojson()

# ── Map + profile panel ────────────────────────────────────────────────────
map_col, profile_col = st.columns([3, 1])

with map_col:
    if geojson is None:
        st.markdown(
            '<div class="ri-placeholder">'
            "⚠️ <strong>Map unavailable</strong><br>"
            "Could not fetch the departments GeoJSON (network error).<br>"
            "The choropleth will appear here in Phase 2 once the GeoJSON is bundled locally."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        fig_map = px.choropleth(
            df_map,
            geojson=geojson,
            locations="code",
            featureidkey="properties.code",
            color="value",
            hover_name="name",
            hover_data={"code": True, "value": ":.1f"},
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
    dept_select = st.selectbox(
        "Select department:",
        options=METRO_DEPT_CODES,
        format_func=lambda c: f"{c} — {DEPT_NAMES.get(c, c)}",
        key="dept_profile",
    )
    name = DEPT_NAMES.get(dept_select, dept_select)

    # TODO-REAL: replace all dummy values below with master CSV lookup
    rng2 = np.random.default_rng(seed=hash(dept_select) % (2**31))
    dummy_profile = {
        "Firm rate":        f"{rng2.uniform(5, 25):.1f} / 1,000",
        "Median income":    f"€{rng2.integers(18000, 38000):,}",
        "Education share":  f"{rng2.uniform(18, 50):.1f}%",
        "Unemployment":     f"{rng2.uniform(4, 14):.1f}%",
        "Doctor density":   f"{rng2.uniform(150, 400):.0f} / 100k",
        "% Urban":          f"{rng2.uniform(10, 100):.1f}%",
    }

    rows = "".join(
        f'<div class="ri-profile-row">'
        f'<span>{k}</span>'
        f'<span class="ri-profile-val">{v}</span>'
        f"</div>"
        for k, v in dummy_profile.items()
    )
    st.markdown(
        f'<div class="ri-profile-card">'
        f'<div class="ri-profile-title">{dept_select} · {name}</div>'
        f"{rows}"
        f'<p style="font-size:0.72rem;color:#aaa;margin-top:0.6rem">Placeholder values · {year}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

