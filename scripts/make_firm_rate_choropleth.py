"""
Generate firm_rate_choropleth.pdf + .png from the master panel.
No geopandas needed -- draws directly from the bundled GeoJSON.
Projection: approximate Lambert-conformal by scaling lon * cos(46.5°).
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import sys
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import PANEL_START, PANEL_END

MASTER  = os.path.join(BASE, "merged",  "france_panel_master.csv")
POP     = os.path.join(BASE, "sources", "population_insee.csv")
GEOJSON = os.path.join(BASE, "app", "assets", "departements.geojson")
FIGURES = os.path.join(BASE, "figures")

# ── Data ──────────────────────────────────────────────────────────────────────
df  = pd.read_csv(MASTER, sep=";", dtype={"dep_code": str})
pop = pd.read_csv(POP,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')
df  = df.merge(pop[["dep_code", "year", "pop_jan1"]],
               on=["dep_code", "year"], how="left")
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)
df["firm_rate"] = df["total_firm_creations"] / df["pop_jan1"] * 1000
dept_mean = df.groupby("dep_code")["firm_rate"].mean()

# ── GeoJSON ───────────────────────────────────────────────────────────────────
with open(GEOJSON) as f:
    gj = json.load(f)

geojson_codes = {feat["properties"]["code"] for feat in gj["features"]}
panel_codes   = set(dept_mean.index)
unmatched_geo = geojson_codes - panel_codes
unmatched_pan = panel_codes   - geojson_codes
print("GeoJSON codes not in panel:", unmatched_geo or "none")
print("Panel codes not in GeoJSON:", unmatched_pan or "none")

# ── Projection: scale lon by cos(46.5°) ──────────────────────────────────────
SCALE_X = np.cos(np.radians(46.5))

def _ring_to_xy(ring):
    arr = np.array(ring)
    return arr[:, 0] * SCALE_X, arr[:, 1]

def geom_to_path(geom):
    """Return a single matplotlib Path covering all rings of a (Multi)Polygon."""
    gtype = geom["type"]
    polys = geom["coordinates"] if gtype == "MultiPolygon" else [geom["coordinates"]]
    verts, codes = [], []
    for poly in polys:
        for ring in poly:
            xs, ys = _ring_to_xy(ring)
            pts = list(zip(xs, ys))
            verts += pts + [(0.0, 0.0)]
            codes += ([mpath.Path.MOVETO]
                      + [mpath.Path.LINETO] * (len(pts) - 1)
                      + [mpath.Path.CLOSEPOLY])
    return mpath.Path(np.array(verts), codes)

# ── Colormap & normalisation ──────────────────────────────────────────────────
vmin, vmax = dept_mean.min(), dept_mean.max()
norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap  = cm.YlGnBu

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_aspect("equal")
ax.axis("off")

rendered = 0
all_x, all_y = [], []

for feat in gj["features"]:
    code  = feat["properties"]["code"]
    value = dept_mean.get(code)
    if value is None:
        print(f"  WARNING: {code} missing from panel mean")
        continue
    path  = geom_to_path(feat["geometry"])
    color = cmap(norm(value))
    patch = mpatches.PathPatch(path,
                               facecolor=color,
                               edgecolor="#cccccc",
                               linewidth=0.35,
                               zorder=2)
    ax.add_patch(patch)
    rendered += 1
    # accumulate bounds
    vs = path.vertices
    mask = path.codes != mpath.Path.CLOSEPOLY
    all_x.append(vs[mask, 0])
    all_y.append(vs[mask, 1])

print(f"Departments rendered: {rendered}")

# Fit view
all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
pad_x = (all_x.max() - all_x.min()) * 0.02
pad_y = (all_y.max() - all_y.min()) * 0.02
ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

# Colorbar
sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01, shrink=0.6)
cbar.set_label("Firm creations per 1,000 residents\n(2012–2021 mean)",
               fontsize=8, labelpad=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout(pad=0.3)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(FIGURES, exist_ok=True)
out_png = os.path.join(FIGURES, "firm_rate_choropleth.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
print(f"PNG: {out_png}")

# ── Summary stats ─────────────────────────────────────────────────────────────
codes_to_name = df[["dep_code", "dep_name"]].drop_duplicates().set_index("dep_code")["dep_name"].to_dict()
top3    = dept_mean.nlargest(3)
bottom3 = dept_mean.nsmallest(3)
print("\nHighest firm-creation rate (per 1,000):")
for code, val in top3.items():
    print(f"  {code} {codes_to_name.get(code, '?'):30s}  {val:.2f}")
print("\nLowest firm-creation rate (per 1,000):")
for code, val in bottom3.items():
    print(f"  {code} {codes_to_name.get(code, '?'):30s}  {val:.2f}")
