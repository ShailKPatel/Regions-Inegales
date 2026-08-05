"""
FIG 1 -- Choropleth of mean firm creation rate per 1,000 inhabitants by
department, averaged over 2012-2021, 96 metropolitan departments.

Canonical panel: merged/france_panel_master.csv + sources/population_insee.csv
(PANEL_START/PANEL_END from scripts/panel_config.py). No model dependency,
this is a descriptive figure of the target variable only.

Output: paper/figures/fig1_firm_rate_choropleth.pdf, single IEEE column
(3.5in wide). France's bounding box is close to square, so a full-width
map would run ~7in tall for no legibility gain -- this figure carries no
small text beyond the colorbar, so single-column is the right size.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.cm as cm
import matplotlib.colors as mcolors

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import PANEL_START, PANEL_END
from fig_common import IEEE_COL_WIDTH, SEQUENTIAL_CMAP, report_fig

MASTER  = os.path.join(BASE, "merged",  "france_panel_master.csv")
POP     = os.path.join(BASE, "sources", "population_insee.csv")
GEOJSON = os.path.join(BASE, "app", "assets", "departements.geojson")
OUT     = os.path.join(BASE, "paper", "figures", "fig1_firm_rate_choropleth.pdf")

# ── Data (canonical panel, target variable only) ────────────────────────────
df  = pd.read_csv(MASTER, sep=";", dtype={"dep_code": str})
pop = pd.read_csv(POP,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')
df  = df.merge(pop[["dep_code", "year", "pop_jan1"]], on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0, "unmatched pop rows"
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)
df["firm_rate"] = df["total_firm_creations"] / df["pop_jan1"] * 1000

dept_mean = df.groupby("dep_code")["firm_rate"].mean()
assert dept_mean.shape[0] == 96, f"expected 96 metropolitan departments, got {dept_mean.shape[0]}"

print("Firm creation rate per 1,000 (2012-2021 mean), 96 metro departments:")
print(f"  min={dept_mean.min():.3f}  max={dept_mean.max():.3f}  "
      f"mean={dept_mean.mean():.3f}  median={dept_mean.median():.3f}")
top3, bottom3 = dept_mean.nlargest(3), dept_mean.nsmallest(3)
names = df[["dep_code", "dep_name"]].drop_duplicates().set_index("dep_code")["dep_name"].to_dict()
print("  top3:", [(c, names.get(c, "?"), round(v, 2)) for c, v in top3.items()])
print("  bottom3:", [(c, names.get(c, "?"), round(v, 2)) for c, v in bottom3.items()])
print()

# ── GeoJSON ──────────────────────────────────────────────────────────────────
with open(GEOJSON) as f:
    gj = json.load(f)

geojson_codes = {feat["properties"]["code"] for feat in gj["features"]}
panel_codes   = set(dept_mean.index)
unmatched_geo = geojson_codes - panel_codes
unmatched_pan = panel_codes - geojson_codes
if unmatched_geo:
    print("WARNING geojson codes not in panel:", unmatched_geo)
if unmatched_pan:
    print("WARNING panel codes not in geojson:", unmatched_pan)

SCALE_X = np.cos(np.radians(46.5))

def _ring_to_xy(ring):
    arr = np.array(ring)
    return arr[:, 0] * SCALE_X, arr[:, 1]

def geom_to_path(geom):
    gtype = geom["type"]
    polys = geom["coordinates"] if gtype == "MultiPolygon" else [geom["coordinates"]]
    verts, codes = [], []
    for poly in polys:
        for ring in poly:
            xs, ys = _ring_to_xy(ring)
            pts = list(zip(xs, ys))
            verts += pts + [(0.0, 0.0)]
            codes += ([mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(pts) - 1)
                      + [mpath.Path.CLOSEPOLY])
    return mpath.Path(np.array(verts), codes)

vmin, vmax = dept_mean.min(), dept_mean.max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = matplotlib.colormaps[SEQUENTIAL_CMAP]

# ── First pass: compute bounds to fix aspect ratio at final print size ─────
paths, bounds_x, bounds_y = [], [], []
for feat in gj["features"]:
    code = feat["properties"]["code"]
    if code not in panel_codes:
        continue
    p = geom_to_path(feat["geometry"])
    paths.append((code, p))
    vs = p.vertices
    mask = p.codes != mpath.Path.CLOSEPOLY
    bounds_x.append(vs[mask, 0])
    bounds_y.append(vs[mask, 1])

all_x = np.concatenate(bounds_x)
all_y = np.concatenate(bounds_y)
x_range = all_x.max() - all_x.min()
y_range = all_y.max() - all_y.min()
map_aspect = y_range / x_range  # height/width of the map body alone

FIG_W = IEEE_COL_WIDTH
# Reserve a slim margin for the colorbar strip beside the map so the
# overall figure (map + colorbar) still fits in FIG_W exactly.
MAP_FRACTION = 0.88
map_w_in = FIG_W * MAP_FRACTION
map_h_in = map_w_in * map_aspect
FIG_H = map_h_in + 0.35  # small pad for nothing (no title; colorbar is inline)

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0.02, 0.02, 0.80, 0.96])
ax.set_aspect("equal")
ax.axis("off")

for code, p in paths:
    value = dept_mean[code]
    color = cmap(norm(value))
    patch = mpatches.PathPatch(p, facecolor=color, edgecolor="#999999",
                                linewidth=0.25, zorder=2)
    ax.add_patch(patch)

pad_x = x_range * 0.02
pad_y = y_range * 0.02
ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cax = fig.add_axes([0.85, 0.15, 0.05, 0.70])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Firm creations / 1,000 inhabitants\n(2012–2021 mean)", fontsize=8)
cbar.ax.tick_params(labelsize=8)

w_in, h_in, min_pt = report_fig(
    fig, OUT,
    extra_note=f"cmap={SEQUENTIAL_CMAP} (ColorBrewer colorblind-safe, sequential); "
               f"{len(paths)} departments rendered; no dept labels; no title in image."
)
plt.close(fig)
