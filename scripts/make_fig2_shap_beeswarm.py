"""
FIG 2 -- SHAP beeswarm for all 8 features, using the out-of-fold SHAP
matrix from the LODO folds (NOT an in-sample explainer). Features
ordered by mean |SHAP|, descending (unemployment last).

Canonical model: XGBoost, 8 locked features, identical params and LODO
GroupKFold construction as model/final_model.py STEP 4. This script
recomputes OOF SHAP directly from the canonical panel rather than
reading a stale figure, per the "do not trust existing figure scripts"
instruction.

Output: paper/figures/fig2_shap_beeswarm.pdf, single IEEE column (3.5in).
cmap forced to viridis (not SHAP's default red/blue), which fails common
colour-vision deficiencies and collapses to near-identical greys in print.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import GroupKFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import PANEL_START, PANEL_END
from fig_common import IEEE_COL_WIDTH, report_fig

MASTER_PATH = os.path.join(BASE, "merged", "france_panel_master.csv")
POP_PATH    = os.path.join(BASE, "sources", "population_insee.csv")
OUT         = os.path.join(BASE, "paper", "figures", "fig2_shap_beeswarm.pdf")

RNG = 42
FEATURES = [
    "q2_disp", "gini_disp", "poverty_rate_disp", "unemployment_rate",
    "doctor_density_per_100k", "edu_share_sup", "pct_urban", "pct_wages",
]
TARGET = "firm_rate"
FEATURE_DISPLAY = {
    "edu_share_sup":           "Higher-ed share",
    "q2_disp":                 "Median income",
    "pct_urban":               "% Urban",
    "doctor_density_per_100k": "Doctor density",
    "unemployment_rate":       "Unemployment rate",
    "poverty_rate_disp":       "Poverty rate",
    "gini_disp":               "Gini coefficient",
    "pct_wages":               "Wage income share",
}

# ── Load canonical panel (identical construction to final_model.py) ────────
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
assert X.isna().sum().sum() == 0 and y.isna().sum() == 0
assert len(df) == 960 and df["dep_code"].nunique() == 96

xgb_params = dict(
    max_depth=4, n_estimators=300, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG,
)

gkf = GroupKFold(n_splits=df["dep_code"].nunique())
lodo_splits = list(gkf.split(X, y, groups=groups_dep))
assert len(lodo_splits) == 96, "expected 96 LODO folds"

# ── OOF SHAP via LODO (primary; NOT in-sample) ──────────────────────────────
shap_values = np.zeros((len(X), len(FEATURES)), dtype=float)
for tr, te in lodo_splits:
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
    shap_values[te] = shap.TreeExplainer(m).shap_values(X.iloc[te])

mas = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
shap_order = mas.sort_values(ascending=False).index.tolist()

print("Mean |SHAP| (OOF, LODO), descending:")
for f in shap_order:
    print(f"  {FEATURE_DISPLAY[f]:<22} {mas[f]:.4f}")
print(f"Unemployment rank: {shap_order.index('unemployment_rate') + 1}/8 "
      f"({'last, as expected' if shap_order[-1] == 'unemployment_rate' else 'NOT last -- check'})")
print()

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

feature_names = [FEATURE_DISPLAY[f] for f in FEATURES]
shap.summary_plot(
    shap_values, X, feature_names=feature_names,
    cmap="viridis",
    plot_size=(IEEE_COL_WIDTH, 3.2),
    show=False,
)
fig = plt.gcf()
ax = fig.axes[0]
ax.set_xlabel("SHAP value (impact on firm creation rate)", fontsize=8)
ax.tick_params(axis="both", labelsize=8)
for tick in ax.get_yticklabels():
    tick.set_fontsize(8)
# colorbar (feature value) is fig.axes[1] when color_bar=True
if len(fig.axes) > 1:
    cax = fig.axes[1]
    cax.tick_params(labelsize=8)
    cax.set_ylabel(cax.get_ylabel(), fontsize=8)

fig.set_size_inches(IEEE_COL_WIDTH, 3.2)

report_fig(
    fig, OUT,
    extra_note="cmap=viridis (explicit, not SHAP default red/blue); "
    "SHAP matrix is out-of-fold via 96-fold LODO, not an in-sample TreeExplainer; "
    "ordered by mean |SHAP| descending, unemployment last."
)
plt.close(fig)
