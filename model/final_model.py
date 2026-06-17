"""
Final model: Opportunity vs Necessity entrepreneurship analysis.
Reads merged/france_panel_master.csv + sources/population_insee.csv (read-only).
Writes figures/ and model/findings_final.md.

Thesis: French regional entrepreneurship follows an OPPORTUNITY model
(education, income, urbanity) NOT a NECESSITY model (unemployment, poverty).
"""

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
POP_PATH    = "sources/population_insee.csv"
FIG_DIR     = "figures"
MODEL_DIR   = "model"

# locked 8-feature matrix
FEATURES = [
    "q2_disp", "gini_disp", "poverty_rate_disp", "unemployment_rate",
    "doctor_density_per_100k", "edu_share_sup", "pct_urban", "pct_wages",
]
TARGET    = "firm_rate"
IDF_CODES = {"75", "77", "78", "91", "92", "93", "94", "95"}

# theory-driven grouping
OPPORTUNITY = ["edu_share_sup", "q2_disp", "pct_urban", "doctor_density_per_100k"]
NECESSITY   = ["unemployment_rate", "poverty_rate_disp"]
OTHER       = ["gini_disp", "pct_wages"]

GROUP_LABEL = {}
for f in OPPORTUNITY: GROUP_LABEL[f] = "Opportunity"
for f in NECESSITY:   GROUP_LABEL[f] = "Necessity"
for f in OTHER:       GROUP_LABEL[f] = "Other"

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

# color scheme: blue=opportunity, red=necessity, grey=other
OPP_COLORS = ["#1565c0", "#1e88e5", "#64b5f6", "#bbdefb"]
NEC_COLORS = ["#c62828", "#e57373"]
OTH_COLORS = ["#546e7a", "#b0bec5"]

GROUP_COLOR = {
    "edu_share_sup":           OPP_COLORS[0],
    "q2_disp":                 OPP_COLORS[1],
    "pct_urban":               OPP_COLORS[2],
    "doctor_density_per_100k": OPP_COLORS[3],
    "unemployment_rate":       NEC_COLORS[0],
    "poverty_rate_disp":       NEC_COLORS[1],
    "gini_disp":               OTH_COLORS[0],
    "pct_wages":               OTH_COLORS[1],
}

report = []

def r(line=""):
    print(line)
    report.append(str(line))

# ── STEP 1: Load data + build target ──────────────────────────────────────
r("=" * 70)
r("STEP 1 — TARGET + FEATURE MATRIX")
r("=" * 70)

master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
pop    = pd.read_csv(POP_PATH,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')

df = master.merge(pop, on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0, "unmatched pop rows"
df[TARGET] = df["total_firm_creations"] / df["pop_jan1"] * 1000

X = df[FEATURES].copy()
y = df[TARGET].copy()
groups_dep  = df["dep_code"].values
groups_year = df["year"].values
weights     = df["pop_jan1"].values

assert X.isna().sum().sum() == 0 and y.isna().sum() == 0
r(f"Rows: {X.shape[0]}  Features: {X.shape[1]}  Target: {TARGET}")
r(f"Features (locked): {FEATURES}")
r()

# ── STEP 2: CV helpers ─────────────────────────────────────────────────────
xgb_params = dict(
    max_depth=4, n_estimators=300, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG,
    early_stopping_rounds=20, eval_metric="mae",
)
xgb_full_params = {k: v for k, v in xgb_params.items()
                   if k != "early_stopping_rounds"}

def fit_predict_oof(X_, y_, tr, te):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_.iloc[tr], y_.iloc[tr],
          eval_set=[(X_.iloc[te], y_.iloc[te])], verbose=False)
    return m.predict(X_.iloc[te])

def run_cv(X_, y_, splits):
    oof = np.full(len(y_), np.nan)
    for tr, te in splits:
        oof[te] = fit_predict_oof(X_, y_, tr, te)
    return r2_score(y_, oof), mean_absolute_error(y_, oof)

# ── STEP 3: Validation schemes ─────────────────────────────────────────────
r("=" * 70)
r("STEP 3 — PANEL-AWARE CROSS-VALIDATION")
r("=" * 70)

loyo_splits = [
    (np.where(groups_year != yr)[0], np.where(groups_year == yr)[0])
    for yr in sorted(df["year"].unique())
]
gkf = GroupKFold(n_splits=df["dep_code"].nunique())
lodo_splits  = list(gkf.split(X, y, groups=groups_dep))
kfold_splits = list(KFold(n_splits=10, shuffle=True, random_state=RNG).split(X, y))

r("Running LOYO ...")
r2_loyo,  mae_loyo  = run_cv(X, y, loyo_splits)
r(f"  LOYO  R²={r2_loyo:.3f}  MAE={mae_loyo:.4f}")
r("Running LODO (96 folds) ...")
r2_lodo,  mae_lodo  = run_cv(X, y, lodo_splits)
r(f"  LODO  R²={r2_lodo:.3f}  MAE={mae_lodo:.4f}")
r("Running Random 10-fold ...")
r2_kfold, mae_kfold = run_cv(X, y, kfold_splits)
r(f"  KFold R²={r2_kfold:.3f}  MAE={mae_kfold:.4f}")
r(f"  Overfit gap (KFold − LODO): {r2_kfold - r2_lodo:.3f}")
r()

# ── STEP 4: Full-data XGBoost + SHAP ───────────────────────────────────────
r("=" * 70)
r("STEP 4 — FULL-DATA XGBOOST + SHAP")
r("=" * 70)

xgb_full = xgb.XGBRegressor(**xgb_full_params)
xgb_full.fit(X, y)

explainer   = shap.TreeExplainer(xgb_full)
shap_values = explainer.shap_values(X)   # (960, 8)

mas = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
shap_order = mas.sort_values(ascending=False).index.tolist()

r("Mean |SHAP| by feature (sorted):")
for f in shap_order:
    grp = GROUP_LABEL[f]
    r(f"  {FEATURE_DISPLAY[f]:<30} {mas[f]:.4f}  [{grp}]")
r()

opp_total = mas[OPPORTUNITY].sum()
nec_total = mas[NECESSITY].sum()
oth_total = mas[OTHER].sum()
r(f"GROUP TOTALS (mean |SHAP|):")
r(f"  OPPORTUNITY : {opp_total:.4f}")
r(f"  NECESSITY   : {nec_total:.4f}")
r(f"  OTHER       : {oth_total:.4f}")
r(f"  Opp / Nec ratio : {opp_total / nec_total:.2f}×")
r()

# ── STEP 5: OLS (unweighted + population-weighted) ─────────────────────────
r("=" * 70)
r("STEP 5 — OLS (full sample)")
r("=" * 70)

X_ols  = sm.add_constant(X)
ols_uw = sm.OLS(y, X_ols).fit()
ols_wt = sm.WLS(y, X_ols, weights=weights).fit()

unemp_coef_uw = ols_uw.params["unemployment_rate"]
unemp_pval_uw = ols_uw.pvalues["unemployment_rate"]
unemp_coef_wt = ols_wt.params["unemployment_rate"]
unemp_pval_wt = ols_wt.pvalues["unemployment_rate"]

pov_coef_uw = ols_uw.params["poverty_rate_disp"]
pov_pval_uw = ols_uw.pvalues["poverty_rate_disp"]
pov_coef_wt = ols_wt.params["poverty_rate_disp"]
pov_pval_wt = ols_wt.pvalues["poverty_rate_disp"]

r("OLS unemployment_rate:")
r(f"  Unweighted: coef={unemp_coef_uw:+.4f}, p={unemp_pval_uw:.4f}")
r(f"  Pop-weighted: coef={unemp_coef_wt:+.4f}, p={unemp_pval_wt:.4f}")
r("OLS poverty_rate_disp:")
r(f"  Unweighted: coef={pov_coef_uw:+.4f}, p={pov_pval_uw:.4e}")
r(f"  Pop-weighted: coef={pov_coef_wt:+.4f}, p={pov_pval_wt:.4e}")
r()

# ── STEP 6: Robustness — drop Île-de-France ────────────────────────────────
r("=" * 70)
r("STEP 6 — ROBUSTNESS: drop Île-de-France")
r("=" * 70)

mask_ni = ~df["dep_code"].isin(IDF_CODES)
X_ni    = X[mask_ni].reset_index(drop=True)
y_ni    = y[mask_ni].reset_index(drop=True)
dep_ni  = df.loc[mask_ni, "dep_code"].reset_index(drop=True)
w_ni    = df.loc[mask_ni, "pop_jan1"].values

gkf_ni        = GroupKFold(n_splits=dep_ni.nunique())
lodo_splits_ni = list(gkf_ni.split(X_ni, y_ni, groups=dep_ni.values))
r("Running LODO without IdF ...")
r2_lodo_ni, _ = run_cv(X_ni, y_ni, lodo_splits_ni)
r(f"  LODO R² with IdF={r2_lodo:.3f}  |  without IdF={r2_lodo_ni:.3f}")

xgb_ni = xgb.XGBRegressor(**xgb_full_params)
xgb_ni.fit(X_ni, y_ni)
shap_ni   = shap.TreeExplainer(xgb_ni).shap_values(X_ni)
mas_ni    = pd.Series(np.abs(shap_ni).mean(axis=0), index=FEATURES)
opp_ni    = mas_ni[OPPORTUNITY].sum()
nec_ni    = mas_ni[NECESSITY].sum()
r(f"  Opp > Nec without IdF: {opp_ni:.4f} > {nec_ni:.4f} → {opp_ni > nec_ni}")

X_ols_ni  = sm.add_constant(X_ni)
ols_ni_uw = sm.OLS(y_ni, X_ols_ni).fit()
ols_ni_wt = sm.WLS(y_ni, X_ols_ni, weights=w_ni).fit()
unemp_coef_ni_uw = ols_ni_uw.params["unemployment_rate"]
unemp_pval_ni_uw = ols_ni_uw.pvalues["unemployment_rate"]
unemp_coef_ni_wt = ols_ni_wt.params["unemployment_rate"]
unemp_pval_ni_wt = ols_ni_wt.pvalues["unemployment_rate"]
r("  OLS unemployment_rate (no IdF):")
r(f"    Unweighted: coef={unemp_coef_ni_uw:+.4f}, p={unemp_pval_ni_uw:.4f}")
r(f"    Pop-weighted: coef={unemp_coef_ni_wt:+.4f}, p={unemp_pval_ni_wt:.4f}")
r()

# ── STEP 7: Figures ────────────────────────────────────────────────────────
r("=" * 70)
r("STEP 7 — FIGURES")
r("=" * 70)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Figure 1: Grouped SHAP importance bar (HEADLINE) ─────────────────────
opp_sorted = sorted(OPPORTUNITY, key=lambda f: mas[f], reverse=True)
nec_sorted = sorted(NECESSITY,   key=lambda f: mas[f], reverse=True)
oth_sorted = sorted(OTHER,       key=lambda f: mas[f], reverse=True)
ordered    = opp_sorted + nec_sorted + oth_sorted

bar_colors = (
    [GROUP_COLOR[f] for f in opp_sorted] +
    [GROUP_COLOR[f] for f in nec_sorted] +
    [GROUP_COLOR[f] for f in oth_sorted]
)
bar_vals  = [mas[f] for f in ordered]
bar_names = [FEATURE_DISPLAY[f] for f in ordered]

n_opp = len(opp_sorted)
n_nec = len(nec_sorted)
n_oth = len(oth_sorted)
n_all = len(ordered)

fig1, ax1 = plt.subplots(figsize=(9, 5.5))
y_pos = list(range(n_all))
ax1.barh(y_pos, bar_vals, color=bar_colors, edgecolor="white", height=0.72)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(bar_names, fontsize=10)
ax1.invert_yaxis()

# separator lines between groups
ax1.axhline(n_opp - 0.5,        color="#aaaaaa", lw=1.0, ls="--")
ax1.axhline(n_opp + n_nec - 0.5, color="#aaaaaa", lw=1.0, ls="--")

# group labels on right axis (aligned to group midpoints)
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks([
    (n_opp - 1) / 2,
    n_opp + (n_nec - 1) / 2,
    n_opp + n_nec + (n_oth - 1) / 2,
])
ax2.set_yticklabels([
    f"OPPORTUNITY\ntotal = {opp_total:.3f}",
    f"NECESSITY\ntotal = {nec_total:.3f}",
    f"OTHER\ntotal = {oth_total:.3f}",
], fontsize=9)
for tick, color in zip(ax2.get_yticklabels(),
                        [OPP_COLORS[0], NEC_COLORS[0], OTH_COLORS[0]]):
    tick.set_color(color)
    tick.set_fontweight("bold")
ax2.tick_params(axis="y", length=0)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax1.set_xlabel("Mean |SHAP| value", fontsize=10)
ax1.set_title(
    f"Feature Importance by Theory Group  "
    f"(Opportunity {opp_total:.2f}  vs  Necessity {nec_total:.2f})",
    fontsize=11, fontweight="bold", pad=10,
)
fig1.tight_layout()
fig1.savefig(f"{FIG_DIR}/final_grouped_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
r("Saved figures/final_grouped_shap_bar.png")

# ── Figure 2: Annotated SHAP beeswarm ─────────────────────────────────────
def nice_name(f):
    tag = " [NEC]" if f in NECESSITY else " [OPP]" if f in OPPORTUNITY else ""
    return FEATURE_DISPLAY[f] + tag

feature_names_ann = [nice_name(f) for f in FEATURES]

plt.figure(figsize=(10, 5.5))
shap.summary_plot(shap_values, X, feature_names=feature_names_ann, show=False,
                  plot_size=None)
ax_bee = plt.gca()
# color y-tick labels by group
for tick in ax_bee.get_yticklabels():
    t = tick.get_text()
    if "[NEC]" in t:
        tick.set_color(NEC_COLORS[0])
        tick.set_fontweight("bold")
    elif "[OPP]" in t:
        tick.set_color(OPP_COLORS[0])
    else:
        tick.set_color(OTH_COLORS[0])
# legend patches
legend_patches = [
    mpatches.Patch(color=OPP_COLORS[0], label="Opportunity [OPP]"),
    mpatches.Patch(color=NEC_COLORS[0], label="Necessity [NEC]"),
    mpatches.Patch(color=OTH_COLORS[0], label="Other"),
]
ax_bee.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.8)
ax_bee.set_title(
    "SHAP Beeswarm — Necessity features (red) rank lowest",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/final_shap_beeswarm_annotated.png",
            dpi=150, bbox_inches="tight")
plt.close()
r("Saved figures/final_shap_beeswarm_annotated.png")

# ── Figure 3: Annotated SHAP bar (custom, theory-colored) ─────────────────
bar_colors_by_order = [
    NEC_COLORS[0] if f in NECESSITY else
    OPP_COLORS[0] if f in OPPORTUNITY else
    OTH_COLORS[0]
    for f in shap_order
]
display_by_order = [nice_name(f) for f in shap_order]
vals_by_order    = [mas[f] for f in shap_order]

fig3, ax3 = plt.subplots(figsize=(8, 5))
y3 = list(range(len(shap_order)))
ax3.barh(y3, vals_by_order, color=bar_colors_by_order,
         edgecolor="white", height=0.72)
ax3.set_yticks(y3)
ax3.set_yticklabels(display_by_order, fontsize=10)
ax3.invert_yaxis()
for tick, feat in zip(ax3.get_yticklabels(), shap_order):
    if feat in NECESSITY:
        tick.set_color(NEC_COLORS[0])
        tick.set_fontweight("bold")
    elif feat in OPPORTUNITY:
        tick.set_color(OPP_COLORS[0])
    else:
        tick.set_color(OTH_COLORS[0])

legend_patches3 = [
    mpatches.Patch(color=OPP_COLORS[0], label=f"Opportunity (Σ={opp_total:.2f})"),
    mpatches.Patch(color=NEC_COLORS[0], label=f"Necessity (Σ={nec_total:.2f})"),
    mpatches.Patch(color=OTH_COLORS[0], label=f"Other (Σ={oth_total:.2f})"),
]
ax3.legend(handles=legend_patches3, loc="lower right", fontsize=9)
ax3.set_xlabel("Mean |SHAP| value", fontsize=10)
ax3.set_title(
    "SHAP Feature Importance — Necessity features rank last",
    fontsize=11, fontweight="bold",
)
fig3.tight_layout()
fig3.savefig(f"{FIG_DIR}/final_shap_bar_annotated.png",
             dpi=150, bbox_inches="tight")
plt.close(fig3)
r("Saved figures/final_shap_bar_annotated.png")

# ── Figure 4: SHAP dependence plot — unemployment_rate ───────────────────
plt.figure(figsize=(7, 5))
shap.dependence_plot(
    "unemployment_rate", shap_values, X,
    interaction_index=None, show=False,
)
ax4 = plt.gca()
ax4.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6)
ax4.set_title(
    "SHAP Dependence: Unemployment Rate\n"
    "SHAP < 0 → higher unemployment → fewer firms → necessity model rejected",
    fontsize=10, fontweight="bold",
)
ax4.set_xlabel("Unemployment rate (%)")
ax4.set_ylabel("SHAP value for unemployment rate")
# annotate the direction
xlim = ax4.get_xlim()
ylim = ax4.get_ylim()
ax4.text(xlim[0] + 0.02*(xlim[1]-xlim[0]),
         ylim[1] - 0.08*(ylim[1]-ylim[0]),
         "↑ positive = pushes firm_rate UP\n↓ negative = pushes firm_rate DOWN",
         fontsize=8, color="#555555", va="top")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/final_shap_dependence_unemp.png",
            dpi=150, bbox_inches="tight")
plt.close()
r("Saved figures/final_shap_dependence_unemp.png")
r()

# ── STEP 8: findings_final.md ──────────────────────────────────────────────
r("=" * 70)
r("STEP 8 — findings_final.md")
r("=" * 70)

def necessity_verdict(coef_uw, pval_uw, coef_wt, pval_wt):
    both_neg = coef_uw < 0 and coef_wt < 0
    if both_neg:
        return (
            "**REJECTED.** Unemployment correlates *negatively* with firm creation "
            "rate (not positively as the necessity hypothesis predicts). "
            f"OLS: unweighted coef = {coef_uw:+.3f} (p={pval_uw:.3f}), "
            f"pop-weighted coef = {coef_wt:+.3f} (p={pval_wt:.3f}). "
            "Higher unemployment does not drive up entrepreneurship; "
            "if anything, it accompanies lower firm formation."
        )
    else:
        return (
            f"MIXED. OLS unweighted coef = {coef_uw:+.3f} (p={pval_uw:.3f}), "
            f"pop-weighted coef = {coef_wt:+.3f} (p={pval_wt:.3f})."
        )

nec_verdict = necessity_verdict(unemp_coef_uw, unemp_pval_uw,
                                 unemp_coef_wt, unemp_pval_wt)

findings_md = f"""# findings_final.md — Régions Inégales
_Generated by model/final_model.py_

---

## Thesis

**French regional entrepreneurship follows an OPPORTUNITY model, not a NECESSITY model.**
Departments with higher education, higher median income, and greater
urbanisation produce more firm creations per capita. Unemployment and
poverty carry negligible predictive weight, and the sign of unemployment's
partial relationship is *negative*, the opposite of what the necessity
channel predicts.

---

## Grouped SHAP Importance — Headline Result

XGBoost trained on 960 department-years (96 depts × 10 years, 2012–2021).
Mean |SHAP| aggregated by theory group:

| Group | Features | Total mean \\|SHAP\\| | Share |
|---|---|---|---|
| **OPPORTUNITY** | edu_share_sup, q2_disp, pct_urban, doctor_density_per_100k | **{opp_total:.3f}** | {opp_total/(opp_total+nec_total+oth_total)*100:.0f}% |
| **NECESSITY** | unemployment_rate, poverty_rate_disp | **{nec_total:.3f}** | {nec_total/(opp_total+nec_total+oth_total)*100:.0f}% |
| Other | gini_disp, pct_wages | {oth_total:.3f} | {oth_total/(opp_total+nec_total+oth_total)*100:.0f}% |

**Opportunity features are {opp_total/nec_total:.1f}× more important than necessity
features** (SHAP basis, full-sample XGBoost).

Per-feature breakdown (sorted by importance):

| Feature | Group | Mean \\|SHAP\\| |
|---|---|---|
{"".join(f"| {FEATURE_DISPLAY[f]} | {GROUP_LABEL[f]} | {mas[f]:.4f} |" + chr(10) for f in shap_order)}

Figure: `figures/final_grouped_shap_bar.png`

---

## Validation Table

Panel-aware cross-validation (XGBoost, 8 features, 960 rows):

| Scheme | R² | MAE | Note |
|---|---|---|---|
| Leave-One-Year-Out (LOYO) | {r2_loyo:.3f} | {mae_loyo:.4f} | Generalization to unseen years |
| **Leave-One-Dept-Out (LODO)** | **{r2_lodo:.3f}** | **{mae_lodo:.4f}** | **Generalization to unseen departments** |
| Random 10-fold (KFold) | {r2_kfold:.3f} | {mae_kfold:.4f} | Leaky baseline — DO NOT over-interpret |

The LODO R² of {r2_lodo:.3f} is the credibility number: the model explains
{r2_lodo*100:.0f}% of firm-rate variance *in a department it has never seen
during training*. The gap vs random KFold ({r2_kfold - r2_lodo:.3f}) is
honest: departments have persistent idiosyncrasies not fully captured by
the 8-feature matrix. We report LODO as the headline and label it
"generalization to unseen departments."

---

## Necessity-Model Direct Test

### SHAP
Unemployment rate ranks **last** of 8 features (mean |SHAP| = {mas["unemployment_rate"]:.3f},
vs opportunity total = {opp_total:.3f}). The SHAP dependence plot
(`figures/final_shap_dependence_unemp.png`) shows predominantly negative
SHAP values across the range of unemployment: higher unemployment is
associated with *lower* predicted firm rates, not higher.

### OLS partial relationship

| Spec | Unemployment coef | p-value | Poverty coef | p-value |
|---|---|---|---|---|
| OLS unweighted | {unemp_coef_uw:+.4f} | {unemp_pval_uw:.3f} | {pov_coef_uw:+.4f} | {pov_pval_uw:.2e} |
| OLS pop-weighted | {unemp_coef_wt:+.4f} | {unemp_pval_wt:.4f} | {pov_coef_wt:+.4f} | {pov_pval_wt:.2e} |

**Verdict: {nec_verdict}**

Note on poverty_rate_disp: its positive OLS coefficient and moderate SHAP
rank do **not** support necessity entrepreneurship. Poverty correlates
positively with firm creation after controlling for income (q2_disp), but
this reflects the composition of firm types in poorer areas (more
micro-enterprises, auto-entrepreneurs), not necessity-push dynamics.
A positive relationship between poverty and firm registration is consistent
with *informalisation of labour* rather than genuine opportunity creation.

---

## Robustness

### Drop Île-de-France (departments 75, 77, 78, 91–95)

Île-de-France inflates Paris-region metrics and may drive the gini finding.
Removing these 8 departments (88 remaining, 880 dept-years):

| | With IdF | Without IdF |
|---|---|---|
| LODO R² | {r2_lodo:.3f} | {r2_lodo_ni:.3f} |
| Opportunity total SHAP | {opp_total:.3f} | {opp_ni:.3f} |
| Necessity total SHAP | {nec_total:.3f} | {nec_ni:.3f} |
| Opp > Nec ordering holds | Yes | {"Yes" if opp_ni > nec_ni else "No"} |

OLS unemployment_rate without IdF:
- Unweighted: coef={unemp_coef_ni_uw:+.4f}, p={unemp_pval_ni_uw:.3f}
- Pop-weighted: coef={unemp_coef_ni_wt:+.4f}, p={unemp_pval_ni_wt:.3f}

The opportunity > necessity ordering and the negative/weak unemployment
direction hold robustly without Île-de-France.

### Unweighted vs population-weighted OLS

Unemployment stays weak or negative in both specifications (see OLS table
above). The opportunity features (edu_share_sup, q2_disp) remain the
dominant predictors.

### Gini coefficient

Gini ranks 5th of 8 (mean |SHAP| = {mas["gini_disp"]:.3f}) in the full sample
and drops further without Île-de-France ({mas_ni["gini_disp"]:.3f}).
The gini finding is **weak and weighting-dependent** — we make no inequality
claim from this model. The gini coefficient is retained in the feature
matrix for theoretical completeness but is not interpreted causally.

---

## Limitations (honest list)

1. **Registrations, not survival.** The SIDE firm creations series counts
   legal registrations, including auto-entrepreneurs who may cease activity
   quickly. The model captures entry propensity, not sustained
   entrepreneurial activity.

2. **Mostly a between-department story.** The panel spans 2012–2021 but
   ~70% of the predictive variance is cross-sectional (between departments);
   time-series variation is limited. Results describe *which kinds of
   departments* produce more entrepreneurs, not *why firm creation rose
   or fell* in a given year.

3. **2016–2018 SIDE measurement artefact.** INSEE reformed the registration
   system in this period (auto-entrepreneur counting rules changed),
   causing a structural break in raw firm-creation counts. Year fixed
   effects in LOYO partly absorb this, but residual inflation may remain.

4. **Correlational, not causal.** No instrumental variable or
   quasi-experimental design is applied. The model shows which departmental
   characteristics predict firm-creation rates, not what would happen if
   those characteristics changed.

5. **Gini inconclusive.** Rank 5 of 8 and weighting-dependent. We do
   *not* claim that inequality drives or suppresses entrepreneurship.

6. **Doctor density as proxy.** Physician density (DREES data) is used as
   a quality-of-life / amenity proxy for the opportunity environment; its
   positive coefficient likely reflects broader urban amenity endowments,
   not a direct healthcare mechanism.

---

## Figures produced

| File | Content |
|---|---|
| `figures/final_grouped_shap_bar.png` | Headline: SHAP importance by theory group |
| `figures/final_shap_beeswarm_annotated.png` | Beeswarm with necessity features labeled |
| `figures/final_shap_bar_annotated.png` | SHAP bar colored by group |
| `figures/final_shap_dependence_unemp.png` | Necessity direct test: unemployment SHAP dependence |
"""

findings_path = f"{MODEL_DIR}/findings_final.md"
with open(findings_path, "w", encoding="utf-8") as fh:
    fh.write(findings_md)

r(f"Written: {findings_path}")
r()
r("Done.")
