"""
FIG 3 -- Coefficient forest plot of the pooled linear specification.
All 8 coefficients, unweighted and population-weighted shown as paired
markers, 95% CI from department-clustered SE. Vertical line at zero.

Standardisation choice: predictors (X) are z-scored before fitting;
the outcome (firm_rate, per 1,000 inhabitants) is left in its native
units. Each plotted coefficient then reads as "effect of a one-SD
increase in the predictor, in firm-creations-per-1,000 units" --
interpretable off the axis. Z-scoring y as well would produce a
unitless quantity with no natural reading.

Built-in correctness check: standardising X only rescales each
coefficient and its SE by the same factor (the predictor's SD), so
every t-statistic and p-value must be numerically identical, to double
precision, to the raw-unit fit. This is asserted below to 6 decimals;
if the assertion fails, the refit is wrong, not the theory.

Canonical model: identical 960-row panel, identical 8-feature matrix,
identical department-clustered OLS/WLS construction as model/final_model.py
STEP 5.

Output: paper/figures/fig3_coefficient_forest.pdf, wide two-column
(7.16in) -- 8 features x 2 weighting specs with CI whiskers needs more
horizontal room than a single column allows at 8pt+ text.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import PANEL_START, PANEL_END
from fig_common import (IEEE_PAGE_WIDTH, COLOR_UNWEIGHTED, COLOR_WEIGHTED,
                         report_fig)

MASTER_PATH = os.path.join(BASE, "merged", "france_panel_master.csv")
POP_PATH    = os.path.join(BASE, "sources", "population_insee.csv")
OUT         = os.path.join(BASE, "paper", "figures", "fig3_coefficient_forest.pdf")

FEATURES = [
    "q2_disp", "gini_disp", "poverty_rate_disp", "unemployment_rate",
    "doctor_density_per_100k", "edu_share_sup", "pct_urban", "pct_wages",
]
TARGET = "firm_rate"
FEATURE_DISPLAY = {
    "q2_disp":                 "Median income",
    "edu_share_sup":           "Higher-ed share",
    "poverty_rate_disp":       "Poverty rate",
    "pct_wages":                "Wage income share",
    "gini_disp":               "Gini coefficient",
    "doctor_density_per_100k": "Doctor density",
    "pct_urban":               "% Urban",
    "unemployment_rate":       "Unemployment rate",
}
# Locked SHAP-importance order from FINDINGS.md (descending mean |SHAP|),
# reused here so the reading order lines up with FIG 2.
DISPLAY_ORDER = [
    "q2_disp", "edu_share_sup", "poverty_rate_disp", "pct_wages",
    "gini_disp", "doctor_density_per_100k", "pct_urban", "unemployment_rate",
]

# ── Load canonical panel ────────────────────────────────────────────────────
master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
pop    = pd.read_csv(POP_PATH,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')
df = master.merge(pop, on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0, "unmatched pop rows"
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)
df[TARGET] = df["total_firm_creations"] / df["pop_jan1"] * 1000

X = df[FEATURES].copy()
y = df[TARGET].copy()
groups_dep = df["dep_code"].values
weights = df["pop_jan1"].values
assert len(df) == 960 and df["dep_code"].nunique() == 96

# ── Raw-unit fit (matches FINDINGS.md / final_model.py exactly) ────────────
X_raw = sm.add_constant(X)
ols_uw_raw = sm.OLS(y, X_raw).fit(cov_type="cluster", cov_kwds={"groups": groups_dep})
ols_wt_raw = sm.WLS(y, X_raw, weights=weights).fit(cov_type="cluster", cov_kwds={"groups": groups_dep})

print("RAW-UNIT department-clustered OLS/WLS (cross-check against FINDINGS.md):")
for f in FEATURES:
    print(f"  {f:<26} UW coef={ols_uw_raw.params[f]:+.4f} p={ols_uw_raw.pvalues[f]:.3e}   "
          f"WT coef={ols_wt_raw.params[f]:+.4f} p={ols_wt_raw.pvalues[f]:.3e}")
print()

# ── Standardised-X fit (X z-scored, y left in native units) ────────────────
X_mean = X.mean()
X_std  = X.std(ddof=0)
X_z    = (X - X_mean) / X_std

X_z_c = sm.add_constant(X_z)
ols_uw_z = sm.OLS(y, X_z_c).fit(cov_type="cluster", cov_kwds={"groups": groups_dep})
ols_wt_z = sm.WLS(y, X_z_c, weights=weights).fit(cov_type="cluster", cov_kwds={"groups": groups_dep})

# ── Correctness assertion: standardising X rescales coef & SE by the same
# factor, so t-stats (and therefore p-values) must be IDENTICAL to the
# raw-unit fit, for every feature, to 6 decimals. ──────────────────────────
print("Equivalence check: t-statistics, raw-unit fit vs standardised-X fit "
      "(must match to 6dp):")
max_t_diff = 0.0
for f in FEATURES:
    t_raw_uw, t_z_uw = ols_uw_raw.tvalues[f], ols_uw_z.tvalues[f]
    t_raw_wt, t_z_wt = ols_wt_raw.tvalues[f], ols_wt_z.tvalues[f]
    d_uw = abs(t_raw_uw - t_z_uw)
    d_wt = abs(t_raw_wt - t_z_wt)
    max_t_diff = max(max_t_diff, d_uw, d_wt)
    print(f"  {f:<26} UW t: raw={t_raw_uw:+.6f} std={t_z_uw:+.6f} diff={d_uw:.2e}   "
          f"WT t: raw={t_raw_wt:+.6f} std={t_z_wt:+.6f} diff={d_wt:.2e}")
    assert round(t_raw_uw, 6) == round(t_z_uw, 6), f"{f} UW t-stat mismatch: refit is wrong"
    assert round(t_raw_wt, 6) == round(t_z_wt, 6), f"{f} WT t-stat mismatch: refit is wrong"
    assert round(ols_uw_raw.pvalues[f], 6) == round(ols_uw_z.pvalues[f], 6), f"{f} UW p-value mismatch"
    assert round(ols_wt_raw.pvalues[f], 6) == round(ols_wt_z.pvalues[f], 6), f"{f} WT p-value mismatch"
print(f"PASSED: all 8 features, both specs, t-stats identical to 6dp "
      f"(max abs diff = {max_t_diff:.2e}). Standardisation is a pure "
      f"rescaling, not a different regression.")
print()

# ── Build forest-plot data (standardised coefficients, i.e. per-1-SD) ──────
rows = []
for f in DISPLAY_ORDER:
    coef_uw, ci_uw = ols_uw_z.params[f], ols_uw_z.conf_int().loc[f]
    coef_wt, ci_wt = ols_wt_z.params[f], ols_wt_z.conf_int().loc[f]
    rows.append(dict(
        feature=f, display=FEATURE_DISPLAY[f],
        coef_uw=coef_uw, lo_uw=ci_uw[0], hi_uw=ci_uw[1], p_uw=ols_uw_z.pvalues[f],
        coef_wt=coef_wt, lo_wt=ci_wt[0], hi_wt=ci_wt[1], p_wt=ols_wt_z.pvalues[f],
    ))

print("Standardised (per-1-SD) coefficients used in the forest plot:")
for row in rows:
    print(f"  {row['display']:<22} UW={row['coef_uw']:+.4f} [{row['lo_uw']:+.4f}, {row['hi_uw']:+.4f}]  "
          f"WT={row['coef_wt']:+.4f} [{row['lo_wt']:+.4f}, {row['hi_wt']:+.4f}]")
print(f"R² (UW, standardised) = {ols_uw_z.rsquared:.4f}  R² (WT, standardised) = {ols_wt_z.rsquared:.4f}  "
      f"(identical to raw-unit R²: {ols_uw_raw.rsquared:.4f} / {ols_wt_raw.rsquared:.4f})")
print()

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

n = len(rows)
fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH, 3.6))

y_uw = np.arange(n) + 0.14
y_wt = np.arange(n) - 0.14

for i, row in enumerate(rows):
    ax.plot([row["lo_uw"], row["hi_uw"]], [y_uw[i], y_uw[i]],
            color=COLOR_UNWEIGHTED, lw=1.3, zorder=2)
    ax.plot([row["lo_wt"], row["hi_wt"]], [y_wt[i], y_wt[i]],
            color=COLOR_WEIGHTED, lw=1.3, zorder=2)

ax.scatter([row["coef_uw"] for row in rows], y_uw, marker="o", s=32,
           color=COLOR_UNWEIGHTED, edgecolor="white", linewidth=0.5, zorder=3,
           label="Unweighted")
ax.scatter([row["coef_wt"] for row in rows], y_wt, marker="s", s=28,
           color=COLOR_WEIGHTED, edgecolor="white", linewidth=0.5, zorder=3,
           label="Population-weighted")

ax.axvline(0, color="#333333", lw=1.0, ls="--", zorder=1)

ax.set_yticks(np.arange(n))
ax.set_yticklabels([row["display"] for row in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Standardised coefficient (effect per 1-SD increase in predictor,\n"
              "firm creations per 1,000 inhabitants)", fontsize=8.5)
ax.tick_params(axis="x", labelsize=8.5)
ax.tick_params(axis="y", labelsize=9)
ax.legend(loc="lower right", fontsize=8.5, frameon=False, markerscale=1.1)
ax.set_ylim(n - 0.5, -0.5)

fig.tight_layout(pad=0.4)

report_fig(
    fig, OUT,
    extra_note="predictors z-scored, outcome left in native units (firm-creations "
    "per 1,000); t-stats/p-values verified identical to raw-unit fit to 6dp "
    "(see printed equivalence check above)."
)
plt.close(fig)
