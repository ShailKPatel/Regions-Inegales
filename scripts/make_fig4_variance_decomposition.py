"""
FIG 4 -- Within-department vs between-department variance share for each
of the 8 predictors, stacked horizontal bars summing to 100%, ordered by
within-share descending.

Decomposition formula reused exactly from model/fixed_effects_test.py
(xtsum-style: pct_within = within_sd^2 / (within_sd^2 + between_sd^2)),
recomputed here directly from the canonical panel rather than reading a
stale figure.

pct_urban's within-share is exactly 0% (density_is_static=True, one
Grille de densite classification per department applied uniformly across
all years) -- this is a structural property of the source data, not a
plotting artefact. The caption in main.tex already states this, so no
in-image annotation is added here.

Output: paper/figures/fig4_variance_decomposition.pdf, single IEEE
column (3.5in).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import PANEL_START, PANEL_END
from fig_common import IEEE_COL_WIDTH, COLOR_WITHIN, COLOR_BETWEEN, report_fig

MASTER_PATH = os.path.join(BASE, "merged", "france_panel_master.csv")
POP_PATH    = os.path.join(BASE, "sources", "population_insee.csv")
OUT         = os.path.join(BASE, "paper", "figures", "fig4_variance_decomposition.pdf")

FEATURES = [
    "q2_disp", "gini_disp", "poverty_rate_disp", "unemployment_rate",
    "doctor_density_per_100k", "edu_share_sup", "pct_urban", "pct_wages",
]
TARGET = "firm_rate"
FEATURE_DISPLAY = {
    "q2_disp":                 "Median income",
    "gini_disp":               "Gini coefficient",
    "poverty_rate_disp":       "Poverty rate",
    "unemployment_rate":       "Unemployment rate",
    "doctor_density_per_100k": "Doctor density",
    "edu_share_sup":           "Higher-ed share",
    "pct_urban":               "% Urban",
    "pct_wages":               "Wage income share",
}

# ── Load canonical panel ────────────────────────────────────────────────────
master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
pop    = pd.read_csv(POP_PATH,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')
df = master.merge(pop, on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0, "unmatched pop rows"
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)
df[TARGET] = df["total_firm_creations"] / df["pop_jan1"] * 1000
df = df.sort_values(["dep_code", "year"]).reset_index(drop=True)
assert len(df) == 960 and df["dep_code"].nunique() == 96

# ── xtsum-style within/between decomposition (identical formula to
# model/fixed_effects_test.py STEP 2) ───────────────────────────────────────
decomp = {}
for f in FEATURES:
    total_sd  = df[f].std(ddof=0)
    dept_means = df.groupby("dep_code")[f].transform("mean")
    between_sd = df.groupby("dep_code")[f].mean().std(ddof=0)
    within_sd  = (df[f] - dept_means + df[f].mean()).std(ddof=0)
    pct_within = (within_sd**2 / (within_sd**2 + between_sd**2)) * 100
    decomp[f] = dict(total_sd=total_sd, between_sd=between_sd,
                      within_sd=within_sd, pct_within=pct_within,
                      pct_between=100 - pct_within)

print("Within/between variance decomposition (xtsum-style), all 8 features:")
print(f"  {'Feature':<28} {'Total SD':>10} {'Between SD':>12} {'Within SD':>11} {'% Within':>9}")
for f in FEATURES:
    d = decomp[f]
    print(f"  {f:<28} {d['total_sd']:>10.4f} {d['between_sd']:>12.4f} "
          f"{d['within_sd']:>11.4f} {d['pct_within']:>8.1f}%")
print()

# sanity check: pct_urban must be exactly 0% within (time-invariant by construction)
_urban_range = df.groupby("dep_code")["pct_urban"].apply(lambda s: s.max() - s.min())
assert (_urban_range < 1e-9).all(), "pct_urban is not exactly time-invariant -- re-check source"
assert decomp["pct_urban"]["pct_within"] < 1e-6, "pct_urban within-share should be ~0%"

order = sorted(FEATURES, key=lambda f: decomp[f]["pct_within"], reverse=True)
print("Order (within-share descending):", [FEATURE_DISPLAY[f] for f in order])
print()

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

n = len(order)
fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 3.4))
y_pos = np.arange(n)

within_vals  = [decomp[f]["pct_within"] for f in order]
between_vals = [decomp[f]["pct_between"] for f in order]

ax.barh(y_pos, within_vals, color=COLOR_WITHIN, edgecolor="white",
        height=0.68, label="Within-dept.")
ax.barh(y_pos, between_vals, left=within_vals, color=COLOR_BETWEEN,
        edgecolor="white", height=0.68, label="Between-dept.")

for i, f in enumerate(order):
    w = decomp[f]["pct_within"]
    if w >= 8:
        ax.text(w / 2, i, f"{w:.0f}%", ha="center", va="center",
                fontsize=8, color="white")
    else:
        ax.text(w + 1.5, i, f"{w:.0f}%", ha="left", va="center", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels([FEATURE_DISPLAY[f] for f in order], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of variance (%)", fontsize=8)
ax.tick_params(axis="x", labelsize=8)

# Reserve explicit margins rather than relying on tight_layout/
# bbox_inches="tight" -- report_fig saves at the exact figsize, so
# anything placed outside a manually reserved margin gets silently
# clipped by the page boundary. Legend uses fig.legend in figure
# coordinates (full 3.5in width) rather than ax.legend (axes width only,
# which overflowed the right edge when centered via bbox_to_anchor).
fig.subplots_adjust(left=0.40, right=0.98, top=0.98, bottom=0.22)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
           ncol=2, fontsize=8, frameon=False, handlelength=1.4,
           columnspacing=1.2)

report_fig(
    fig, OUT,
    extra_note="decomposition formula identical to model/fixed_effects_test.py; "
    "pct_urban's 0% within-share asserted structural (exactly time-invariant "
    "source classification), consistent with caption."
)
plt.close(fig)
