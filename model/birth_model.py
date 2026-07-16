"""
Birth-rate model: determinants of departmental fertility in France.
Reads merged/france_panel_master.csv (read-only).
Writes figures/birth_*.png and model/findings_birth.md.

Target: birth_rate (live births per 1,000 inhabitants, pre-computed in master CSV).
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from panel_config import PANEL_START, PANEL_END

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb
import shap
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

# ── constants ──────────────────────────────────────────────────────────────
RNG         = 42
MASTER_PATH = "merged/france_panel_master.csv"
FIG_DIR     = "figures"
MODEL_DIR   = "model"

TARGET = "birth_rate"

# 8-feature matrix mirroring final_model.py structure
FEATURES = [
    "marriage_rate", "q2_disp", "unemployment_rate", "edu_share_sup",
    "pct_urban", "doctor_density_per_100k", "poverty_rate_disp", "gini_disp",
]

SOCIAL    = ["marriage_rate"]
ECONOMIC  = ["q2_disp", "unemployment_rate", "poverty_rate_disp"]
STRUCTURAL = ["edu_share_sup", "pct_urban", "doctor_density_per_100k", "gini_disp"]

GROUP_LABEL = {}
for f in SOCIAL:     GROUP_LABEL[f] = "Social"
for f in ECONOMIC:   GROUP_LABEL[f] = "Economic"
for f in STRUCTURAL: GROUP_LABEL[f] = "Structural"

FEATURE_DISPLAY = {
    "marriage_rate":           "Marriage rate",
    "q2_disp":                 "Median income",
    "unemployment_rate":       "Unemployment rate",
    "edu_share_sup":           "Higher-ed share",
    "pct_urban":               "% Urban",
    "doctor_density_per_100k": "Doctor density",
    "poverty_rate_disp":       "Poverty rate",
    "gini_disp":               "Gini coefficient",
}

SOC_COLORS = ["#1565c0"]
ECO_COLORS = ["#2e7d32", "#66bb6a", "#a5d6a7"]
STR_COLORS = ["#6a1b9a", "#ab47bc", "#ce93d8", "#e1bee7"]

GROUP_COLOR = {
    "marriage_rate":           SOC_COLORS[0],
    "q2_disp":                 ECO_COLORS[0],
    "unemployment_rate":       ECO_COLORS[1],
    "poverty_rate_disp":       ECO_COLORS[2],
    "edu_share_sup":           STR_COLORS[0],
    "pct_urban":               STR_COLORS[1],
    "doctor_density_per_100k": STR_COLORS[2],
    "gini_disp":               STR_COLORS[3],
}

report = []
def r(line=""):
    print(line)
    report.append(str(line))

# ── STEP 1: Load ───────────────────────────────────────────────────────────
r("=" * 70)
r("STEP 1 — LOAD DATA")
r("=" * 70)

master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
df = master[(master["year"] >= PANEL_START) & (master["year"] <= PANEL_END)].reset_index(drop=True)

assert df[TARGET].isna().sum() == 0, "birth_rate has nulls"
assert df[FEATURES].isna().sum().sum() == 0, "feature nulls"

X = df[FEATURES].copy()
y = df[TARGET].copy()
groups_dep  = df["dep_code"].values
groups_year = df["year"].values
weights     = df["n_persons"].values  # population weight

r(f"Rows: {len(df)}  Features: {len(FEATURES)}  Target: {TARGET}")
r(f"birth_rate range: {y.min():.2f} – {y.max():.2f} per 1,000")
r()

# ── STEP 2: CV ─────────────────────────────────────────────────────────────
xgb_params = dict(
    max_depth=4, n_estimators=300, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG,
)

def run_cv(X_, y_, splits):
    oof = np.full(len(y_), np.nan)
    for tr, te in splits:
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_.iloc[tr], y_.iloc[tr], verbose=False)
        oof[te] = m.predict(X_.iloc[te])
    return r2_score(y_, oof), mean_absolute_error(y_, oof)

r("=" * 70)
r("STEP 2 — CROSS-VALIDATION")
r("=" * 70)

loyo_splits  = [(np.where(groups_year != yr)[0], np.where(groups_year == yr)[0])
                for yr in sorted(df["year"].unique())]
gkf          = GroupKFold(n_splits=df["dep_code"].nunique())
lodo_splits  = list(gkf.split(X, y, groups=groups_dep))
kfold_splits = list(KFold(n_splits=10, shuffle=True, random_state=RNG).split(X, y))

r("Running LOYO ...")
r2_loyo, mae_loyo   = run_cv(X, y, loyo_splits)
r(f"  LOYO  R²={r2_loyo:.3f}  MAE={mae_loyo:.4f}")
r("Running LODO (96 folds) ...")
r2_lodo, mae_lodo   = run_cv(X, y, lodo_splits)
r(f"  LODO  R²={r2_lodo:.3f}  MAE={mae_lodo:.4f}")
r("Running KFold ...")
r2_kfold, mae_kfold = run_cv(X, y, kfold_splits)
r(f"  KFold R²={r2_kfold:.3f}  MAE={mae_kfold:.4f}")
r()

# ── STEP 3: OOF SHAP (LODO) ────────────────────────────────────────────────
r("=" * 70)
r("STEP 3 — OOF SHAP (LODO)")
r("=" * 70)

shap_values = np.zeros((len(X), len(FEATURES)), dtype=float)
for tr, te in lodo_splits:
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
    shap_values[te] = shap.TreeExplainer(m).shap_values(X.iloc[te])

mas = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
shap_order = mas.sort_values(ascending=False).index.tolist()

r("Mean |SHAP| by feature (OOF):")
for f in shap_order:
    r(f"  {FEATURE_DISPLAY[f]:<30} {mas[f]:.4f}  [{GROUP_LABEL[f]}]")
r()

soc_total = mas[SOCIAL].sum()
eco_total = mas[ECONOMIC].sum()
str_total = mas[STRUCTURAL].sum()
grand     = soc_total + eco_total + str_total

r("GROUP TOTALS (OOF mean |SHAP|):")
r(f"  SOCIAL     (marriage_rate)         : {soc_total:.4f}  {soc_total/grand*100:.0f}%")
r(f"  ECONOMIC   (income, unemp, poverty) : {eco_total:.4f}  {eco_total/grand*100:.0f}%")
r(f"  STRUCTURAL (edu, urban, health, gini): {str_total:.4f}  {str_total/grand*100:.0f}%")
r()

# ── STEP 4: OLS ────────────────────────────────────────────────────────────
r("=" * 70)
r("STEP 4 — OLS (full panel, department-clustered SE)")
r("=" * 70)

X_ols = sm.add_constant(X)
ols_uw = sm.OLS(y, X_ols).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})
ols_wt = sm.WLS(y, X_ols, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})

for feat in ["marriage_rate", "unemployment_rate", "q2_disp", "edu_share_sup", "pct_urban"]:
    c_uw = ols_uw.params[feat]; p_uw = ols_uw.pvalues[feat]
    c_wt = ols_wt.params[feat]; p_wt = ols_wt.pvalues[feat]
    r(f"  {feat}:")
    r(f"    UW: coef={c_uw:+.4f}  p={p_uw:.4f}")
    r(f"    WT: coef={c_wt:+.4f}  p={p_wt:.4f}")
r()

# ── STEP 5: Figures ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Fig 1: Grouped SHAP bar
soc_sorted = sorted(SOCIAL,    key=lambda f: mas[f], reverse=True)
eco_sorted = sorted(ECONOMIC,  key=lambda f: mas[f], reverse=True)
str_sorted = sorted(STRUCTURAL,key=lambda f: mas[f], reverse=True)
ordered    = soc_sorted + eco_sorted + str_sorted

bar_colors = ([GROUP_COLOR[f] for f in soc_sorted] +
              [GROUP_COLOR[f] for f in eco_sorted] +
              [GROUP_COLOR[f] for f in str_sorted])
bar_vals   = [mas[f] for f in ordered]
bar_names  = [FEATURE_DISPLAY[f] for f in ordered]

n_soc = len(soc_sorted); n_eco = len(eco_sorted); n_str = len(str_sorted)
n_all = len(ordered)

fig1, ax1 = plt.subplots(figsize=(9, 5.5))
y_pos = list(range(n_all))
ax1.barh(y_pos, bar_vals, color=bar_colors, edgecolor="white", height=0.72)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(bar_names, fontsize=10)
ax1.invert_yaxis()
ax1.axhline(n_soc - 0.5,          color="#aaaaaa", lw=1.0, ls="--")
ax1.axhline(n_soc + n_eco - 0.5,  color="#aaaaaa", lw=1.0, ls="--")

ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks([
    (n_soc - 1) / 2,
    n_soc + (n_eco - 1) / 2,
    n_soc + n_eco + (n_str - 1) / 2,
])
ax2.set_yticklabels([
    f"SOCIAL\ntotal = {soc_total:.3f}",
    f"ECONOMIC\ntotal = {eco_total:.3f}",
    f"STRUCTURAL\ntotal = {str_total:.3f}",
], fontsize=9)
for tick, color in zip(ax2.get_yticklabels(), [SOC_COLORS[0], ECO_COLORS[0], STR_COLORS[0]]):
    tick.set_color(color); tick.set_fontweight("bold")
ax2.tick_params(axis="y", length=0)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax1.set_xlabel("Mean |SHAP| value", fontsize=10)
ax1.set_title(
    f"Birth Rate Determinants — Feature Importance (OOF SHAP, LODO)\n"
    f"Social {soc_total:.2f}  |  Economic {eco_total:.2f}  |  Structural {str_total:.2f}",
    fontsize=11, fontweight="bold", pad=10,
)
fig1.tight_layout()
fig1.savefig(f"{FIG_DIR}/birth_grouped_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
r("Saved figures/birth_grouped_shap_bar.png")

# Fig 2: SHAP beeswarm
plt.figure(figsize=(10, 5.5))
shap.summary_plot(shap_values, X, feature_names=[FEATURE_DISPLAY[f] for f in FEATURES], show=False, plot_size=None)
ax_bee = plt.gca()
ax_bee.set_title("SHAP Beeswarm — Birth Rate Determinants", fontsize=11, fontweight="bold")
legend_patches = [
    mpatches.Patch(color=SOC_COLORS[0], label="Social"),
    mpatches.Patch(color=ECO_COLORS[0], label="Economic"),
    mpatches.Patch(color=STR_COLORS[0], label="Structural"),
]
ax_bee.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/birth_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
r("Saved figures/birth_shap_beeswarm.png")

# Fig 3: SHAP dependence — marriage_rate
plt.figure(figsize=(7, 5))
shap.dependence_plot("marriage_rate", shap_values, X, interaction_index=None, show=False)
ax4 = plt.gca()
ax4.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6)
ax4.set_title("SHAP Dependence: Marriage Rate\nHigher marriage rate → higher birth rate",
              fontsize=10, fontweight="bold")
ax4.set_xlabel("Marriage rate (per 1,000)")
ax4.set_ylabel("SHAP value for marriage rate")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/birth_shap_dependence_marriage.png", dpi=150, bbox_inches="tight")
plt.close()
r("Saved figures/birth_shap_dependence_marriage.png")

r()

# ── STEP 6: findings_birth.md ──────────────────────────────────────────────
findings_md = f"""# findings_birth.md — Régions Inégales
_Generated by model/birth_model.py_

---

## Research Question

What predicts birth rate variation across French departments (2012–2021)?
Three channels tested: social institution (marriage), economic conditions
(income, unemployment, poverty), and structural characteristics
(education, urbanisation, healthcare access, inequality).

---

## Grouped SHAP Importance

XGBoost on 960 department-years (96 depts × 10 years). OOF SHAP via LODO.

| Group | Features | Total mean |SHAP| | Share |
|---|---|---|---|
| **SOCIAL** | marriage_rate | **{soc_total:.3f}** | {soc_total/grand*100:.0f}% |
| **ECONOMIC** | income, unemployment, poverty | **{eco_total:.3f}** | {eco_total/grand*100:.0f}% |
| **STRUCTURAL** | education, urban, doctors, Gini | **{str_total:.3f}** | {str_total/grand*100:.0f}% |

Per-feature breakdown (OOF mean |SHAP|, sorted):

| Feature | Group | Mean |SHAP| |
|---|---|---|
{"".join(f"| {FEATURE_DISPLAY[f]} | {GROUP_LABEL[f]} | {mas[f]:.4f} |" + chr(10) for f in shap_order)}

---

## Validation

| Scheme | R² | MAE |
|---|---|---|
| Leave-One-Year-Out (LOYO) | {r2_loyo:.3f} | {mae_loyo:.4f} |
| **Leave-One-Dept-Out (LODO) ★** | **{r2_lodo:.3f}** | **{mae_lodo:.4f}** |
| Random 10-fold (KFold) | {r2_kfold:.3f} | {mae_kfold:.4f} |

★ Headline. KFold leaky baseline.

---

## Key OLS Results (department-clustered SE)

| Feature | UW coef | UW p | WT coef | WT p |
|---|---|---|---|---|
{"".join(f"| {FEATURE_DISPLAY[f]} | {ols_uw.params[f]:+.4f} | {ols_uw.pvalues[f]:.4f} | {ols_wt.params[f]:+.4f} | {ols_wt.pvalues[f]:.4f} |" + chr(10) for f in ["marriage_rate","unemployment_rate","q2_disp","edu_share_sup","pct_urban"])}

---

## Figures

| File | Content |
|---|---|
| `figures/birth_grouped_shap_bar.png` | Headline SHAP by group |
| `figures/birth_shap_beeswarm.png` | Beeswarm all features |
| `figures/birth_shap_dependence_marriage.png` | Marriage rate dependence plot |
"""

with open(f"{MODEL_DIR}/findings_birth.md", "w", encoding="utf-8") as fh:
    fh.write(findings_md)
r(f"Written: model/findings_birth.md")
r("Done.")
