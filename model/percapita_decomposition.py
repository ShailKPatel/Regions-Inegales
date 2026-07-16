"""
Per-capita decomposition of the informalisation-of-labour effect.
Reads merged/france_panel_master.csv + sources/population_insee.csv (read-only).
Writes figures/percapita_*.png and appends to model/findings_informalisation.md.

findings_informalisation.md showed poverty predicts a higher individual/
micro registration SHARE. A share can rise because individual registrations
go up, or because company (SARL/SAS) registrations go down. This script
tests the two per-capita levels separately, using the identical
final_model.py pipeline, to determine which side of the ratio poverty is
actually moving.

Target A: creations_individual / population * 100000
Target B: (creations_sarl + creations_sas) / population * 100000
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
    line = str(line)
    print(line)
    report.append(line)

# ── load ─────────────────────────────────────────────────────────────────
r("=" * 70)
r("PER-CAPITA DECOMPOSITION: individual vs company registrations")
r("=" * 70)

master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
pop    = pd.read_csv(POP_PATH,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')

df = master.merge(pop, on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0, "unmatched pop rows"
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)

df["target_individual_pc"] = df["creations_individual"] / df["pop_jan1"] * 100000
df["target_company_pc"]    = (df["creations_sarl"] + df["creations_sas"]) / df["pop_jan1"] * 100000

r(f"Target A (individual, per 100k pop): mean={df['target_individual_pc'].mean():.2f}  std={df['target_individual_pc'].std():.2f}")
r(f"Target B (company SARL+SAS, per 100k pop): mean={df['target_company_pc'].mean():.2f}  std={df['target_company_pc'].std():.2f}")
r()

X = df[FEATURES].copy()
groups_dep = df["dep_code"].values
weights    = df["pop_jan1"].values
gkf = GroupKFold(n_splits=df["dep_code"].nunique())
lodo_splits = list(gkf.split(X, df["target_individual_pc"], groups=groups_dep))

def run_target(target_col, label):
    r("=" * 70)
    r(f"TARGET: {label} ({target_col})")
    r("=" * 70)

    y = df[target_col].copy()

    oof = np.full(len(y), np.nan)
    shap_values = np.zeros((len(X), len(FEATURES)), dtype=float)
    for tr, te in lodo_splits:
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        oof[te] = m.predict(X.iloc[te])
        shap_values[te] = shap.TreeExplainer(m).shap_values(X.iloc[te])

    r2_lodo  = r2_score(y, oof)
    mae_lodo = mean_absolute_error(y, oof)
    r(f"LODO R2={r2_lodo:.3f}  MAE={mae_lodo:.4f}")

    mas = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
    shap_order = mas.sort_values(ascending=False).index.tolist()
    r("Mean |SHAP| by feature (OOF), sorted:")
    for f in shap_order:
        rank = shap_order.index(f) + 1
        r(f"  {rank}. {FEATURE_DISPLAY[f]:<20} {mas[f]:.4f}")
    pov_rank   = shap_order.index("poverty_rate_disp") + 1
    unemp_rank = shap_order.index("unemployment_rate") + 1
    r(f"poverty_rate_disp SHAP rank:   {pov_rank}/8")
    r(f"unemployment_rate SHAP rank:   {unemp_rank}/8")

    X_ols  = sm.add_constant(X)
    ols_uw = sm.OLS(y, X_ols).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})
    ols_wt = sm.WLS(y, X_ols, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})

    pov_coef_uw = ols_uw.params["poverty_rate_disp"];   pov_pval_uw = ols_uw.pvalues["poverty_rate_disp"]
    pov_coef_wt = ols_wt.params["poverty_rate_disp"];   pov_pval_wt = ols_wt.pvalues["poverty_rate_disp"]
    unemp_coef_uw = ols_uw.params["unemployment_rate"]; unemp_pval_uw = ols_uw.pvalues["unemployment_rate"]
    unemp_coef_wt = ols_wt.params["unemployment_rate"]; unemp_pval_wt = ols_wt.pvalues["unemployment_rate"]

    r("poverty_rate_disp:")
    r(f"  Unweighted:   coef={pov_coef_uw:+.4f}  p={pov_pval_uw:.4e}")
    r(f"  Pop-weighted: coef={pov_coef_wt:+.4f}  p={pov_pval_wt:.4e}")
    r("unemployment_rate:")
    r(f"  Unweighted:   coef={unemp_coef_uw:+.4f}  p={unemp_pval_uw:.4e}")
    r(f"  Pop-weighted: coef={unemp_coef_wt:+.4f}  p={unemp_pval_wt:.4e}")
    r()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})
    NECESSITY = ["unemployment_rate", "poverty_rate_disp"]
    OPPORTUNITY = ["edu_share_sup", "q2_disp", "pct_urban", "doctor_density_per_100k"]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c62828" if f in NECESSITY else "#1565c0" if f in OPPORTUNITY else "#546e7a" for f in shap_order]
    ax.barh(range(len(shap_order)), [mas[f] for f in shap_order], color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(range(len(shap_order)))
    ax.set_yticklabels([FEATURE_DISPLAY[f] for f in shap_order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title(f"SHAP Importance, {label} (per 100k pop)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fname = f"percapita_shap_{target_col.replace('target_', '').replace('_pc', '')}.png"
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    r(f"Saved figures/{fname}")
    r()

    return dict(
        label=label, r2_lodo=r2_lodo, mas=mas, shap_order=shap_order,
        pov_rank=pov_rank, unemp_rank=unemp_rank,
        pov_coef_uw=pov_coef_uw, pov_pval_uw=pov_pval_uw,
        pov_coef_wt=pov_coef_wt, pov_pval_wt=pov_pval_wt,
        unemp_coef_uw=unemp_coef_uw, unemp_pval_uw=unemp_pval_uw,
        unemp_coef_wt=unemp_coef_wt, unemp_pval_wt=unemp_pval_wt,
        fig=fname,
    )

resA = run_target("target_individual_pc", "Target A: individual registrations")
resB = run_target("target_company_pc", "Target B: company (SARL+SAS) registrations")

# ── verdict ─────────────────────────────────────────────────────────────
r("=" * 70)
r("VERDICT")
r("=" * 70)

A_pos_sig = resA["pov_coef_uw"] > 0 and resA["pov_pval_uw"] < 0.05 and resA["pov_coef_wt"] > 0 and resA["pov_pval_wt"] < 0.05
B_neg_sig = resB["pov_coef_uw"] < 0 and resB["pov_pval_uw"] < 0.05 and resB["pov_coef_wt"] < 0 and resB["pov_pval_wt"] < 0.05
B_pos_sig = resB["pov_coef_uw"] > 0 and resB["pov_pval_uw"] < 0.05 and resB["pov_coef_wt"] > 0 and resB["pov_pval_wt"] < 0.05
B_null    = not B_neg_sig and not B_pos_sig

if A_pos_sig and B_null:
    verdict = "CLOSED"
elif A_pos_sig and B_neg_sig:
    verdict = "CLOSED"
elif A_pos_sig and B_pos_sig:
    verdict = "PARTIAL"
elif not A_pos_sig:
    verdict = "CONTRADICTED"
else:
    verdict = "PARTIAL"

r(f"Target A (individual, per capita): poverty positive+significant = {A_pos_sig}")
r(f"Target B (company, per capita):    poverty null = {B_null}  negative+significant = {B_neg_sig}  positive+significant = {B_pos_sig}")
r(f"VERDICT: {verdict}")
r()

# ── append to findings_informalisation.md ──────────────────────────────
verdict_prose = {
    "CLOSED": (
        "**CLOSED: poverty raises individual registrations per capita and "
        "not company registrations, the level effect is fully accounted "
        "for by the informalisation channel.**"
    ),
    "PARTIAL": (
        "**PARTIAL: poverty predicts both individual and company "
        "registrations per capita, informalisation explains some but not "
        "all of the level effect.**"
    ),
    "CONTRADICTED": (
        "**CONTRADICTED: the pattern does not match informalisation.** "
        "Poverty does not significantly raise individual registrations per "
        "capita, so the share effect from the primary test cannot be "
        "attributed to a rise in individual/micro registrations."
    ),
}[verdict]

section = f"""

---

## Per-capita decomposition

_Appended by model/percapita_decomposition.py_

The share test above (individual_share) showed poverty predicts a higher
individual/micro registration share. A share can rise because individual
registrations increase, or because company registrations decrease. This
section tests the two per-capita levels separately with the identical
final_model.py pipeline (same XGBoost hyperparameters, LODO headline CV,
OOF SHAP, department-clustered OLS unweighted + population-weighted).

- **Target A**: `creations_individual / pop_jan1 * 100000`
- **Target B**: `(creations_sarl + creations_sas) / pop_jan1 * 100000`

### LODO cross-validation

| Target | LODO R2 |
|---|---|
| A: individual per capita | {resA['r2_lodo']:.3f} |
| B: company (SARL+SAS) per capita | {resB['r2_lodo']:.3f} |

### OOF SHAP ranking

| Rank | Target A feature | Mean \\|SHAP\\| | Target B feature | Mean \\|SHAP\\| |
|---|---|---|---|---|
{"".join(f"| {i+1} | {FEATURE_DISPLAY[fa]} | {resA['mas'][fa]:.4f} | {FEATURE_DISPLAY[fb]} | {resB['mas'][fb]:.4f} |" + chr(10) for i, (fa, fb) in enumerate(zip(resA['shap_order'], resB['shap_order'])))}

poverty_rate_disp rank: Target A {resA['pov_rank']}/8, Target B {resB['pov_rank']}/8.
unemployment_rate rank: Target A {resA['unemp_rank']}/8, Target B {resB['unemp_rank']}/8.

Figures: `figures/{resA['fig']}`, `figures/{resB['fig']}`

### OLS (department-clustered SE)

| Target | Feature | Spec | Coef | p-value |
|---|---|---|---|---|
| A (individual) | poverty_rate_disp | Unweighted | {resA['pov_coef_uw']:+.4f} | {resA['pov_pval_uw']:.4e} |
| A (individual) | poverty_rate_disp | Pop-weighted | {resA['pov_coef_wt']:+.4f} | {resA['pov_pval_wt']:.4e} |
| A (individual) | unemployment_rate | Unweighted | {resA['unemp_coef_uw']:+.4f} | {resA['unemp_pval_uw']:.4e} |
| A (individual) | unemployment_rate | Pop-weighted | {resA['unemp_coef_wt']:+.4f} | {resA['unemp_pval_wt']:.4e} |
| B (company) | poverty_rate_disp | Unweighted | {resB['pov_coef_uw']:+.4f} | {resB['pov_pval_uw']:.4e} |
| B (company) | poverty_rate_disp | Pop-weighted | {resB['pov_coef_wt']:+.4f} | {resB['pov_pval_wt']:.4e} |
| B (company) | unemployment_rate | Unweighted | {resB['unemp_coef_uw']:+.4f} | {resB['unemp_pval_uw']:.4e} |
| B (company) | unemployment_rate | Pop-weighted | {resB['unemp_coef_wt']:+.4f} | {resB['unemp_pval_wt']:.4e} |

### Verdict

{verdict_prose}

**Evidence:**
- Target A (individual, per capita): poverty coef {resA['pov_coef_uw']:+.4f} (p={resA['pov_pval_uw']:.3e}) unweighted, {resA['pov_coef_wt']:+.4f} (p={resA['pov_pval_wt']:.3e}) pop-weighted. SHAP rank {resA['pov_rank']}/8.
- Target B (company, per capita): poverty coef {resB['pov_coef_uw']:+.4f} (p={resB['pov_pval_uw']:.3e}) unweighted, {resB['pov_coef_wt']:+.4f} (p={resB['pov_pval_wt']:.3e}) pop-weighted. SHAP rank {resB['pov_rank']}/8.

This decomposition should be read together with the share-level verdict
above (MIXED/INCONCLUSIVE overall, because unemployment does not show
the informalisation signature). This section speaks only to poverty's
share effect, not unemployment's.
"""

findings_path = f"{MODEL_DIR}/findings_informalisation.md"
with open(findings_path, "a", encoding="utf-8") as fh:
    fh.write(section)

r(f"Appended to: {findings_path}")
r("Done.")
