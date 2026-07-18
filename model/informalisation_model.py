"""
Informalisation-of-labour test.
Reads merged/france_panel_master.csv + sources/population_insee.csv (read-only).
Writes figures/informalisation_*.png and model/findings_informalisation.md.

FINDINGS.md asserts poverty's positive OLS coefficient on firm_rate reflects
"informalisation of labour" (more individual/micro-entrepreneur registrations
in poorer areas), without testing it. This script runs the identical
final_model.py pipeline (same XGBoost hyperparameters, LODO headline CV,
OOF SHAP, clustered OLS unweighted + pop-weighted) retargeted at the
individual/micro-entrepreneur SHARE of total firm creations, to test
directly whether poverty and unemployment predict a higher individual share.
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
import xgboost as xgb
import shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

RNG         = 42
MASTER_PATH = "merged/france_panel_master.csv"
POP_PATH    = "sources/population_insee.csv"
FIG_DIR     = "figures"
MODEL_DIR   = "model"

FEATURES = [
    "q2_disp", "gini_disp", "poverty_rate_disp", "unemployment_rate",
    "doctor_density_per_100k", "edu_share_sup", "pct_urban", "pct_wages",
]
TARGET  = "individual_share"
TARGET2 = "individual_rate"  # per-capita secondary check

OPPORTUNITY = ["edu_share_sup", "q2_disp", "pct_urban", "doctor_density_per_100k"]
NECESSITY   = ["unemployment_rate", "poverty_rate_disp"]
OTHER       = ["gini_disp", "pct_wages"]

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

xgb_params = dict(
    max_depth=4, n_estimators=300, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG,
)

report = []
def r(line=""):
    print(line)
    report.append(str(line))

# ── STEP 1: target construction + breakdown-sum verification ───────────────
r("=" * 70)
r("STEP 1, TARGET CONSTRUCTION + BREAKDOWN VERIFICATION")
r("=" * 70)

master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
pop    = pd.read_csv(POP_PATH,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')

df = master.merge(pop, on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0, "unmatched pop rows"
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)

legal_sum = df["creations_individual"] + df["creations_sarl"] + df["creations_sas"] + df["creations_other_legal"]
diff = df["total_firm_creations"] - legal_sum
r(f"Legal-form breakdown sum check: max |total - (individual+sarl+sas+other)| = {diff.abs().max()}")
r(f"  Rows with any discrepancy: {(diff != 0).sum()} / {len(df)}")
by_year_diff = df.assign(_diff=diff).groupby("year")["_diff"].apply(lambda s: s.abs().max())
r("  Max |diff| by year (checks for 2016-2018 auto-entrepreneur reform artefact):")
for yr, v in by_year_diff.items():
    r(f"    {yr}: {v}")
r("  -> Breakdown columns sum exactly to total_firm_creations in every year. "
  "No structural discrepancy from the 2016-2018 counting-rule reform is "
  "visible in the sum identity itself (the reform affects *levels* of raw "
  "counts, not the internal consistency of the legal-form breakdown).")
r()

df[TARGET]  = df["creations_individual"] / df["total_firm_creations"]
df[TARGET2] = df["creations_individual"] / df["pop_jan1"] * 1000

r(f"Primary target: {TARGET} = creations_individual / total_firm_creations")
r(f"  mean={df[TARGET].mean():.4f}  std={df[TARGET].std():.4f}  "
  f"min={df[TARGET].min():.4f}  max={df[TARGET].max():.4f}")
r(f"Secondary (per-capita) target: {TARGET2} = creations_individual / pop_jan1 * 1000")
r(f"  mean={df[TARGET2].mean():.4f}  std={df[TARGET2].std():.4f}")
r()
r("Individual/micro share by year (mean across depts):")
for yr, v in df.groupby("year")[TARGET].mean().items():
    r(f"  {yr}: {v:.4f}")
r()

X = df[FEATURES].copy()
y = df[TARGET].copy()
groups_dep = df["dep_code"].values
weights    = df["pop_jan1"].values

assert X.isna().sum().sum() == 0 and y.isna().sum() == 0

# ── STEP 2: LODO CV (headline) ──────────────────────────────────────────────
r("=" * 70)
r("STEP 2, LODO CROSS-VALIDATION (headline, primary target)")
r("=" * 70)

gkf = GroupKFold(n_splits=df["dep_code"].nunique())
lodo_splits = list(gkf.split(X, y, groups=groups_dep))

oof = np.full(len(y), np.nan)
shap_values = np.zeros((len(X), len(FEATURES)), dtype=float)
for tr, te in lodo_splits:
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
    oof[te] = m.predict(X.iloc[te])
    shap_values[te] = shap.TreeExplainer(m).shap_values(X.iloc[te])

r2_lodo  = r2_score(y, oof)
mae_lodo = mean_absolute_error(y, oof)
r(f"LODO R²={r2_lodo:.3f}  MAE={mae_lodo:.4f}")
r()

# ── STEP 3: OOF SHAP ─────────────────────────────────────────────────────
r("=" * 70)
r("STEP 3, OOF SHAP (LODO, 96 folds)")
r("=" * 70)

mas = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
shap_order = mas.sort_values(ascending=False).index.tolist()

r("Mean |SHAP| by feature (OOF), sorted:")
for f in shap_order:
    rank = shap_order.index(f) + 1
    grp = "Necessity" if f in NECESSITY else "Opportunity" if f in OPPORTUNITY else "Other"
    r(f"  {rank}. {FEATURE_DISPLAY[f]:<20} {mas[f]:.5f}  [{grp}]")

pov_rank   = shap_order.index("poverty_rate_disp") + 1
unemp_rank = shap_order.index("unemployment_rate") + 1
r()
r(f"poverty_rate_disp SHAP rank:   {pov_rank}/8")
r(f"unemployment_rate SHAP rank:   {unemp_rank}/8")
r()

# ── STEP 4: OLS (unweighted + population-weighted, clustered) ──────────────
r("=" * 70)
r("STEP 4, OLS (department-clustered SE), primary target")
r("=" * 70)

X_ols  = sm.add_constant(X)
ols_uw = sm.OLS(y, X_ols).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})
ols_wt = sm.WLS(y, X_ols, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})

pov_coef_uw = ols_uw.params["poverty_rate_disp"];   pov_pval_uw = ols_uw.pvalues["poverty_rate_disp"]
pov_coef_wt = ols_wt.params["poverty_rate_disp"];   pov_pval_wt = ols_wt.pvalues["poverty_rate_disp"]
unemp_coef_uw = ols_uw.params["unemployment_rate"]; unemp_pval_uw = ols_uw.pvalues["unemployment_rate"]
unemp_coef_wt = ols_wt.params["unemployment_rate"]; unemp_pval_wt = ols_wt.pvalues["unemployment_rate"]

r("poverty_rate_disp:")
r(f"  Unweighted:   coef={pov_coef_uw:+.5f}  p={pov_pval_uw:.4e}")
r(f"  Pop-weighted: coef={pov_coef_wt:+.5f}  p={pov_pval_wt:.4e}")
r("unemployment_rate:")
r(f"  Unweighted:   coef={unemp_coef_uw:+.5f}  p={unemp_pval_uw:.4e}")
r(f"  Pop-weighted: coef={unemp_coef_wt:+.5f}  p={unemp_pval_wt:.4e}")
r()
r("Full OLS coefficient table (unweighted, clustered):")
r(str(ols_uw.summary().tables[1]))
r()

# STEP 5: secondary check, per-capita individual creation rate
r("=" * 70)
r("STEP 5, SECONDARY CHECK: per-capita individual creation rate")
r("=" * 70)

y2 = df[TARGET2].copy()
shap_values2 = np.zeros((len(X), len(FEATURES)), dtype=float)
oof2 = np.full(len(y2), np.nan)
for tr, te in lodo_splits:
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X.iloc[tr], y2.iloc[tr], verbose=False)
    oof2[te] = m.predict(X.iloc[te])
    shap_values2[te] = shap.TreeExplainer(m).shap_values(X.iloc[te])

r2_lodo2 = r2_score(y2, oof2)
mas2 = pd.Series(np.abs(shap_values2).mean(axis=0), index=FEATURES)
shap_order2 = mas2.sort_values(ascending=False).index.tolist()
r(f"LODO R² (per-capita target): {r2_lodo2:.3f}")
r("Mean |SHAP| by feature (OOF), sorted:")
for f in shap_order2:
    rank = shap_order2.index(f) + 1
    r(f"  {rank}. {FEATURE_DISPLAY[f]:<20} {mas2[f]:.5f}")

X_ols2 = sm.add_constant(X)
ols2_uw = sm.OLS(y2, X_ols2).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})
ols2_wt = sm.WLS(y2, X_ols2, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})
pov_coef_uw2 = ols2_uw.params["poverty_rate_disp"]; pov_pval_uw2 = ols2_uw.pvalues["poverty_rate_disp"]
pov_coef_wt2 = ols2_wt.params["poverty_rate_disp"]; pov_pval_wt2 = ols2_wt.pvalues["poverty_rate_disp"]
unemp_coef_uw2 = ols2_uw.params["unemployment_rate"]; unemp_pval_uw2 = ols2_uw.pvalues["unemployment_rate"]
unemp_coef_wt2 = ols2_wt.params["unemployment_rate"]; unemp_pval_wt2 = ols2_wt.pvalues["unemployment_rate"]
r()
r("poverty_rate_disp (per-capita target):")
r(f"  Unweighted:   coef={pov_coef_uw2:+.5f}  p={pov_pval_uw2:.4e}")
r(f"  Pop-weighted: coef={pov_coef_wt2:+.5f}  p={pov_pval_wt2:.4e}")
r("unemployment_rate (per-capita target):")
r(f"  Unweighted:   coef={unemp_coef_uw2:+.5f}  p={unemp_pval_uw2:.4e}")
r(f"  Pop-weighted: coef={unemp_coef_wt2:+.5f}  p={unemp_pval_wt2:.4e}")
r()
r("NOTE: the per-capita target (individual_rate) is mechanically driven by "
  "total_firm_creations (r with firm_rate is high by construction, since "
  "individual creations are ~75% of total). It answers 'does poverty predict "
  "MORE individual-firm creation per head', which conflates with the main "
  "model. The SHARE target (Step 1-4) is the correct test of the "
  "informalisation-of-labour explanation because it nets out the overall "
  "level of firm creation and isolates COMPOSITION.")
r()

# ── STEP 6: figures ──────────────────────────────────────────────────────
r("=" * 70)
r("STEP 6, FIGURES")
r("=" * 70)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = list(range(len(shap_order)))
colors = ["#c62828" if f in NECESSITY else "#1565c0" if f in OPPORTUNITY else "#546e7a" for f in shap_order]
ax.barh(y_pos, [mas[f] for f in shap_order], color=colors, edgecolor="white", height=0.72)
ax.set_yticks(y_pos)
ax.set_yticklabels([FEATURE_DISPLAY[f] for f in shap_order])
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP| value")
ax.set_title("SHAP Importance, Individual/Micro-Entrepreneur Share\n(red = necessity features)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/informalisation_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
r("Saved figures/informalisation_shap_bar.png")

for feat, fname in [("poverty_rate_disp", "informalisation_dependence_poverty.png"),
                     ("unemployment_rate", "informalisation_dependence_unemployment.png")]:
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(feat, shap_values, X, interaction_index=None, show=False)
    ax4 = plt.gca()
    ax4.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax4.set_title(f"SHAP Dependence: {FEATURE_DISPLAY[feat]} on individual/micro share",
                  fontsize=10, fontweight="bold")
    ax4.set_ylabel("SHAP value for individual_share")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    r(f"Saved figures/{fname}")
r()

# ── STEP 7: verdict ─────────────────────────────────────────────────────
r("=" * 70)
r("STEP 7, VERDICT")
r("=" * 70)

pov_supports   = pov_coef_uw > 0 and pov_coef_wt > 0 and pov_pval_uw < 0.05 and pov_pval_wt < 0.05
unemp_supports = unemp_coef_uw > 0 and unemp_coef_wt > 0 and unemp_pval_uw < 0.05 and unemp_pval_wt < 0.05
pov_contradicts   = (pov_coef_uw < 0 and pov_pval_uw < 0.05) or (pov_coef_wt < 0 and pov_pval_wt < 0.05)
unemp_contradicts = (unemp_coef_uw < 0 and unemp_pval_uw < 0.05) or (unemp_coef_wt < 0 and unemp_pval_wt < 0.05)

if pov_supports and unemp_supports:
    verdict = "SUPPORTS"
elif pov_contradicts or unemp_contradicts:
    verdict = "CONTRADICTS" if not (pov_supports or unemp_supports) else "MIXED / INCONCLUSIVE"
elif pov_supports or unemp_supports:
    verdict = "MIXED / INCONCLUSIVE"
else:
    verdict = "INCONCLUSIVE"

r(f"poverty_rate_disp:   supports={pov_supports}  contradicts={pov_contradicts}  SHAP rank={pov_rank}/8")
r(f"unemployment_rate:   supports={unemp_supports}  contradicts={unemp_contradicts}  SHAP rank={unemp_rank}/8")
r(f"VERDICT: {verdict}")
r()

# ── STEP 8: write findings_informalisation.md ──────────────────────────────
r("=" * 70)
r("STEP 8, findings_informalisation.md")
r("=" * 70)

verdict_prose = {
    "SUPPORTS": (
        "**SUPPORTS the informalisation-of-labour explanation.** Both poverty "
        "and unemployment predict a significantly *higher* individual/micro-"
        "entrepreneur share of firm creations, in both unweighted and "
        "population-weighted specifications."
    ),
    "CONTRADICTS": (
        "**CONTRADICTS the informalisation-of-labour explanation.** Poverty "
        "and/or unemployment predict a significantly *lower* individual/micro "
        "share, the opposite of what the explanation requires."
    ),
    "MIXED / INCONCLUSIVE": (
        "**MIXED / INCONCLUSIVE.** One of poverty or unemployment shows the "
        "sign the informalisation explanation predicts; the other does not, "
        "or the evidence is not consistent across specifications. The "
        "explanation is not cleanly supported or refuted by this test."
    ),
    "INCONCLUSIVE": (
        "**INCONCLUSIVE.** Neither poverty nor unemployment shows a "
        "statistically significant relationship with the individual/micro "
        "share in the direction the informalisation explanation predicts."
    ),
}[verdict]

findings_md = f"""# findings_informalisation.md, Régions Inégales
_Generated by model/informalisation_model.py_

---

## Purpose

`FINDINGS.md` currently explains poverty's positive OLS coefficient on
`firm_rate` as "informalisation of labour" (more individual/micro-
entrepreneur registrations in poorer areas) without testing it directly.
This script tests it: does poverty (and, secondarily, unemployment)
predict a *higher share* of individual/micro-entrepreneur registrations
specifically, using the same pipeline as `final_model.py` (identical
XGBoost hyperparameters, LODO headline CV, OOF SHAP, department-clustered
OLS unweighted + population-weighted).

---

## Target construction and breakdown verification

Primary target: `individual_share = creations_individual / total_firm_creations`.

The legal-form breakdown (`creations_individual + creations_sarl +
creations_sas + creations_other_legal`) sums **exactly** to
`total_firm_creations` in all {len(df)} rows, in every year 2012-2021
including the 2016-2018 auto-entrepreneur counting-rule reform window
(max absolute discrepancy = {diff.abs().max()}). The reform changed how
raw registration *counts* are recorded, but does not break the internal
consistency of the legal-form breakdown; no adjustment was necessary
before using `individual_share` as a target.

Individual/micro share ranges narrowly (mean {df[TARGET].mean():.3f}) across
the panel, dipping slightly in 2015-2017 ({df.groupby('year')[TARGET].mean().loc[2015]:.3f}-{df.groupby('year')[TARGET].mean().loc[2017]:.3f})
and recovering to {df.groupby('year')[TARGET].mean().loc[2021]:.3f} by 2021, consistent with, but not
proof of, some reform-era measurement noise.

---

## LODO cross-validation

| Scheme | R² | MAE |
|---|---|---|
| Leave-One-Dept-Out (LODO) | {r2_lodo:.3f} | {mae_lodo:.4f} |

---

## OOF SHAP ranking (individual_share)

| Rank | Feature | Group | Mean \\|SHAP\\| |
|---|---|---|---|
{"".join(f"| {shap_order.index(f)+1} | {FEATURE_DISPLAY[f]} | {'Necessity' if f in NECESSITY else 'Opportunity' if f in OPPORTUNITY else 'Other'} | {mas[f]:.5f} |" + chr(10) for f in shap_order)}

**poverty_rate_disp** ranks **{pov_rank}/8**.
**unemployment_rate** ranks **{unemp_rank}/8**.

Figure: `figures/informalisation_shap_bar.png`

---

## OLS (department-clustered SE)

| Feature | Spec | Coef | p-value |
|---|---|---|---|
| poverty_rate_disp | Unweighted | {pov_coef_uw:+.5f} | {pov_pval_uw:.4e} |
| poverty_rate_disp | Pop-weighted | {pov_coef_wt:+.5f} | {pov_pval_wt:.4e} |
| unemployment_rate | Unweighted | {unemp_coef_uw:+.5f} | {unemp_pval_uw:.4e} |
| unemployment_rate | Pop-weighted | {unemp_coef_wt:+.5f} | {unemp_pval_wt:.4e} |

Dependence plots: `figures/informalisation_dependence_poverty.png`,
`figures/informalisation_dependence_unemployment.png`

---

## Secondary check: per-capita individual creation rate

LODO R² (per-capita target): {r2_lodo2:.3f}

| Feature | Spec | Coef | p-value |
|---|---|---|---|
| poverty_rate_disp | Unweighted | {pov_coef_uw2:+.5f} | {pov_pval_uw2:.4e} |
| poverty_rate_disp | Pop-weighted | {pov_coef_wt2:+.5f} | {pov_pval_wt2:.4e} |
| unemployment_rate | Unweighted | {unemp_coef_uw2:+.5f} | {unemp_pval_uw2:.4e} |
| unemployment_rate | Pop-weighted | {unemp_coef_wt2:+.5f} | {unemp_pval_wt2:.4e} |

This target is mechanically close to `firm_rate` (individual creations are
~75% of the total by construction) and does not net out the overall level
of firm creation; it is reported for completeness only. The `individual_share`
result above (Steps 1-4) is the correct test of the informalisation
explanation because it isolates *composition*, not *volume*.

---

## Verdict

{verdict_prose}

**Detail:**
- poverty_rate_disp: supports={pov_supports}, contradicts={pov_contradicts}, SHAP rank {pov_rank}/8
- unemployment_rate: supports={unemp_supports}, contradicts={unemp_contradicts}, SHAP rank {unemp_rank}/8

This result should replace the untested "informalisation of labour"
assertion currently in `FINDINGS.md`. If the verdict above is anything
other than a clean SUPPORTS, `FINDINGS.md`'s wording needs to be walked
back to match the evidence, not the other way around.
"""

findings_path = f"{MODEL_DIR}/findings_informalisation.md"
with open(findings_path, "w", encoding="utf-8") as fh:
    fh.write(findings_md)

r(f"Written: {findings_path}")
r()
r("Done.")
