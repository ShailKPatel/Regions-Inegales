"""
9-variable Pearson correlation heatmap (lower triangle, model features + target).
Saves figures/feature_correlation_8.pdf and .png at 300 dpi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import PANEL_START, PANEL_END

MASTER  = os.path.join(BASE, "merged",  "france_panel_master.csv")
POP     = os.path.join(BASE, "sources", "population_insee.csv")
FIGURES = os.path.join(BASE, "figures")

COLS = [
    "q2_disp",
    "edu_share_sup",
    "pct_urban",
    "doctor_density_per_100k",
    "unemployment_rate",
    "poverty_rate_disp",
    "gini_disp",
    "pct_wages",
    "firm_rate",
]

LABELS = [
    "Median income",
    "Higher-ed share",
    "% Urban",
    "Doctor density",
    "Unemployment rate",
    "Poverty rate",
    "Gini",
    "Wage share",
    "Firm rate",          # target, kept last
]

# ── Data ──────────────────────────────────────────────────────────────────────
df  = pd.read_csv(MASTER, sep=";", dtype={"dep_code": str})
pop = pd.read_csv(POP,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')
df  = df.merge(pop[["dep_code", "year", "pop_jan1"]],
               on=["dep_code", "year"], how="left")
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)
df["firm_rate"] = df["total_firm_creations"] / df["pop_jan1"] * 1000

corr = df[COLS].corr(method="pearson")
print(f"Rows used: {df[COLS].dropna().shape[0]}  (of {len(df)})")

# ── Mask upper triangle (keep diagonal for reference) ─────────────────────────
n = len(COLS)
mask = np.triu(np.ones((n, n), dtype=bool), k=1)   # True = hide

# ── Plot ──────────────────────────────────────────────────────────────────────
CELL   = 1.05   # inches per cell
FIG_SZ = n * CELL + 1.4

fig, ax = plt.subplots(figsize=(FIG_SZ, FIG_SZ))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

cmap = plt.get_cmap("RdBu_r")
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

# Draw cells
for i in range(n):
    for j in range(n):
        if mask[i, j]:
            continue
        val   = corr.iloc[i, j]
        color = cmap(norm(val))
        rect  = plt.Rectangle([j, n - 1 - i], 1, 1, facecolor=color,
                               edgecolor="white", linewidth=1.2)
        ax.add_patch(rect)
        # Annotation: dark text on light cells, white on dark
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        txt_color = "white" if lum < 0.45 else "#1a1a1a"
        ax.text(j + 0.5, n - 1 - i + 0.5, f"{val:.2f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=txt_color)

ax.set_xlim(0, n)
ax.set_ylim(0, n)
ax.set_aspect("equal")

# Ticks
ax.set_xticks([x + 0.5 for x in range(n)])
ax.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=11)
ax.set_yticks([y + 0.5 for y in range(n)])
ax.set_yticklabels(reversed(LABELS), fontsize=11)
ax.tick_params(length=0)

# Spine off
for spine in ax.spines.values():
    spine.set_visible(False)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.7)
cbar.set_label("Pearson r", fontsize=11, labelpad=8)
cbar.ax.tick_params(labelsize=10)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

plt.tight_layout(pad=0.6)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(FIGURES, exist_ok=True)
out_png = os.path.join(FIGURES, "feature_correlation_8.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
print(f"PNG: {out_png}")

# ── Top 5 off-diagonal absolute correlations ──────────────────────────────────
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((LABELS[i], LABELS[j], corr.iloc[i, j]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print("\nTop 5 absolute off-diagonal correlations:")
for a, b, r in pairs[:5]:
    print(f"  {a:<22}  <->  {b:<22}  r = {r:+.3f}")
