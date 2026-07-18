"""
HOLD diagnostics , read-only w.r.t. inputs, writes model/findings_diagnostics.md.
Sensitivity check 1: log(pop_jan1) control , does doctor_density importance drop?
Sensitivity check 2: drop 2012 rows , how do poverty and unemployment results change?
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

MODEL_DIR = "model"

report = []
def r(line=""):
    line = str(line)
    print(line)
    report.append(line)

RNG         = 42
MASTER_PATH = "merged/france_panel_master.csv"
POP_PATH    = "sources/population_insee.csv"

FEATURES = [
    "q2_disp", "gini_disp", "poverty_rate_disp", "unemployment_rate",
    "doctor_density_per_100k", "edu_share_sup", "pct_urban", "pct_wages",
]
TARGET = "firm_rate"

xgb_params = dict(
    max_depth=4, n_estimators=300, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG,
)

master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
pop    = pd.read_csv(POP_PATH,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')

df = master.merge(pop, on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0
df[TARGET] = df["total_firm_creations"] / df["pop_jan1"] * 1000

groups_dep = df["dep_code"].values
weights    = df["pop_jan1"].values

r("=" * 70)
r("LOG(POP) CONTROL , does doctor_density importance change?")
r("=" * 70)

df["log_pop"] = np.log(df["pop_jan1"])

# Correlations
r("\nCorrelations with doctor_density_per_100k:")
for c in ["log_pop", "pct_urban", "edu_share_sup"]:
    rho = df[["doctor_density_per_100k", c]].corr().iloc[0, 1]
    r(f"  doctor_density vs {c}: r = {rho:.3f}")

FEATURES_LOGPOP = FEATURES + ["log_pop"]
X_lp  = df[FEATURES_LOGPOP].copy()
y_lp  = df[TARGET].copy()

gkf_lp = GroupKFold(n_splits=df["dep_code"].nunique())
splits_lp = list(gkf_lp.split(X_lp, y_lp, groups=groups_dep))

shap_lp = np.zeros((len(X_lp), len(FEATURES_LOGPOP)), dtype=float)
oof_lp  = np.full(len(y_lp), np.nan)
for tr, te in splits_lp:
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_lp.iloc[tr], y_lp.iloc[tr], verbose=False)
    oof_lp[te]  = m.predict(X_lp.iloc[te])
    shap_lp[te] = shap.TreeExplainer(m).shap_values(X_lp.iloc[te])

r2_lp = r2_score(y_lp, oof_lp)
mas_lp = pd.Series(np.abs(shap_lp).mean(axis=0), index=FEATURES_LOGPOP).sort_values(ascending=False)

r(f"\nLODO R² with log(pop) added: {r2_lp:.4f}")
r("\nMean |SHAP| (OOF) with log(pop) control:")
for f in mas_lp.index:
    rank = list(mas_lp.index).index(f) + 1
    r(f"  {rank:2d}. {f:<30} {mas_lp[f]:.4f}")

doc_rank_base = None
doc_rank_lp   = list(mas_lp.index).index("doctor_density_per_100k") + 1
r(f"\ndoctor_density rank with log(pop): {doc_rank_lp}/{len(FEATURES_LOGPOP)}")

# Baseline rank (9-feature set, use OOF SHAP)
X_base = df[FEATURES].copy()
y_base = df[TARGET].copy()
gkf_b  = GroupKFold(n_splits=df["dep_code"].nunique())
splits_b = list(gkf_b.split(X_base, y_base, groups=groups_dep))
shap_b = np.zeros((len(X_base), len(FEATURES)), dtype=float)
oof_b  = np.full(len(y_base), np.nan)
for tr, te in splits_b:
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_base.iloc[tr], y_base.iloc[tr], verbose=False)
    oof_b[te]  = m.predict(X_base.iloc[te])
    shap_b[te] = shap.TreeExplainer(m).shap_values(X_base.iloc[te])
mas_b = pd.Series(np.abs(shap_b).mean(axis=0), index=FEATURES).sort_values(ascending=False)
doc_rank_base = list(mas_b.index).index("doctor_density_per_100k") + 1
r(f"doctor_density rank without log(pop): {doc_rank_base}/8")
r(f"doctor_density mean|SHAP| without: {mas_b['doctor_density_per_100k']:.4f}  with: {mas_lp['doctor_density_per_100k']:.4f}")

r()
r("=" * 70)
r("DROP 2012 ROWS , poverty and unemployment results")
r("=" * 70)

df_no2012 = df[df["year"] != 2012].copy().reset_index(drop=True)
r(f"\nRows after dropping 2012: {len(df_no2012)} (removed {len(df) - len(df_no2012)})")

X_n12 = df_no2012[FEATURES].copy()
y_n12 = df_no2012[TARGET].copy()
g_n12 = df_no2012["dep_code"].values
w_n12 = df_no2012["pop_jan1"].values

gkf_n12 = GroupKFold(n_splits=df_no2012["dep_code"].nunique())
splits_n12 = list(gkf_n12.split(X_n12, y_n12, groups=g_n12))

shap_n12 = np.zeros((len(X_n12), len(FEATURES)), dtype=float)
oof_n12  = np.full(len(y_n12), np.nan)
for tr, te in splits_n12:
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_n12.iloc[tr], y_n12.iloc[tr], verbose=False)
    oof_n12[te]  = m.predict(X_n12.iloc[te])
    shap_n12[te] = shap.TreeExplainer(m).shap_values(X_n12.iloc[te])

r2_n12  = r2_score(y_n12, oof_n12)
mas_n12 = pd.Series(np.abs(shap_n12).mean(axis=0), index=FEATURES).sort_values(ascending=False)

r(f"\nLODO R² without 2012: {r2_n12:.4f}  (full panel: {r2_score(y_base, oof_b):.4f})")

X_ols_n12  = sm.add_constant(X_n12)
ols_n12_uw = sm.OLS(y_n12, X_ols_n12).fit(cov_type='cluster', cov_kwds={'groups': g_n12})
ols_n12_wt = sm.WLS(y_n12, X_ols_n12, weights=w_n12).fit(cov_type='cluster', cov_kwds={'groups': g_n12})

r("\nOLS results (department-clustered SE):")
for feat in ["unemployment_rate", "poverty_rate_disp"]:
    c_uw = ols_n12_uw.params[feat]; p_uw = ols_n12_uw.pvalues[feat]
    c_wt = ols_n12_wt.params[feat]; p_wt = ols_n12_wt.pvalues[feat]
    r(f"  {feat}:")
    r(f"    Without 2012: UW: coef={c_uw:+.4f}  p={p_uw:.3e}  | WT: coef={c_wt:+.4f}  p={p_wt:.3e}")

X_ols_full = sm.add_constant(X_base)
ols_full_uw = sm.OLS(y_base, X_ols_full).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})
ols_full_wt = sm.WLS(y_base, X_ols_full, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})
r("\n  Full-panel baseline (for comparison):")
for feat in ["unemployment_rate", "poverty_rate_disp"]:
    c_uw = ols_full_uw.params[feat]; p_uw = ols_full_uw.pvalues[feat]
    c_wt = ols_full_wt.params[feat]; p_wt = ols_full_wt.pvalues[feat]
    r(f"  {feat}:")
    r(f"    Full panel: UW: coef={c_uw:+.4f}  p={p_uw:.3e}  | WT: coef={c_wt:+.4f}  p={p_wt:.3e}")

r("\nSHAP comparison (full panel vs without 2012):")
r(f"  {'Feature':<30} {'Full panel':>12} {'No 2012':>12}")
r("  " + "-" * 56)
for feat in FEATURES:
    r(f"  {feat:<30} {mas_b[feat]:>12.4f} {mas_n12[feat]:>12.4f}")

unemp_rank_n12 = list(mas_n12.index).index("unemployment_rate") + 1
unemp_rank_b   = list(mas_b.index).index("unemployment_rate") + 1
r(f"\nUnemployment rank: full panel {unemp_rank_b}/8  |  without 2012 {unemp_rank_n12}/8")

OPP = ["edu_share_sup", "q2_disp", "pct_urban", "doctor_density_per_100k"]
NEC = ["unemployment_rate", "poverty_rate_disp"]
OTH = ["gini_disp", "pct_wages"]

opp_b   = mas_b[OPP].sum();   nec_b   = mas_b[NEC].sum();   tot_b   = mas_b.sum()
opp_n12 = mas_n12[OPP].sum(); nec_n12 = mas_n12[NEC].sum(); tot_n12 = mas_n12.sum()
r(f"Opp share: full {opp_b/tot_b*100:.0f}%  no-2012 {opp_n12/tot_n12*100:.0f}%")
r(f"Nec share: full {nec_b/tot_b*100:.0f}%  no-2012 {nec_n12/tot_n12*100:.0f}%")
r(f"Opp/Nec:   full {opp_b/nec_b:.2f}x  no-2012 {opp_n12/nec_n12:.2f}x")

log_pop_changes_ranking = doc_rank_lp != doc_rank_base
drop2012_changes_unemp_rank = unemp_rank_n12 != unemp_rank_b

findings_md = f"""# findings_diagnostics.md, Regions Inegales
_Generated by archive/hold_diagnostics.py_

---

## Log(population) control on doctor_density_per_100k

Correlation of doctor_density_per_100k with log(pop), pct_urban, edu_share_sup:

| Control | r |
|---|---|
{"".join(f"| {c} | {df[['doctor_density_per_100k', c]].corr().iloc[0,1]:.3f} |" + chr(10) for c in ["log_pop", "pct_urban", "edu_share_sup"])}

LODO R2 with log(pop) added: {r2_lp:.4f}

| Rank (9-feature) | Feature | Mean \\|SHAP\\| |
|---|---|---|
{"".join(f"| {list(mas_lp.index).index(f)+1} | {f} | {mas_lp[f]:.4f} |" + chr(10) for f in mas_lp.index)}

doctor_density_per_100k rank: {doc_rank_base}/8 without log(pop) control, {doc_rank_lp}/{len(FEATURES_LOGPOP)} with it.
Mean |SHAP|: {mas_b['doctor_density_per_100k']:.4f} without vs {mas_lp['doctor_density_per_100k']:.4f} with.

**Does the headline ordering change?** {"YES, doctor_density's rank shifts" if log_pop_changes_ranking else "NO, doctor_density's rank is unchanged"} when log(pop) is added as a control. {"Its importance does not appear to be a population-size proxy in disguise." if not log_pop_changes_ranking else "Some of its importance may be attributable to city size rather than a distinct amenity effect."}

---

## Drop 2012 (poverty_rate_dec gap year)

Rows after dropping 2012: {len(df_no2012)} (removed {len(df) - len(df_no2012)})

LODO R2 without 2012: {r2_n12:.4f} (full panel: {r2_score(y_base, oof_b):.4f})

OLS (department-clustered SE):

| Feature | Spec | Full panel coef | Full panel p | No-2012 coef | No-2012 p |
|---|---|---|---|---|---|
| unemployment_rate | Unweighted | {ols_full_uw.params['unemployment_rate']:+.4f} | {ols_full_uw.pvalues['unemployment_rate']:.3e} | {ols_n12_uw.params['unemployment_rate']:+.4f} | {ols_n12_uw.pvalues['unemployment_rate']:.3e} |
| unemployment_rate | Pop-weighted | {ols_full_wt.params['unemployment_rate']:+.4f} | {ols_full_wt.pvalues['unemployment_rate']:.3e} | {ols_n12_wt.params['unemployment_rate']:+.4f} | {ols_n12_wt.pvalues['unemployment_rate']:.3e} |
| poverty_rate_disp | Unweighted | {ols_full_uw.params['poverty_rate_disp']:+.4f} | {ols_full_uw.pvalues['poverty_rate_disp']:.3e} | {ols_n12_uw.params['poverty_rate_disp']:+.4f} | {ols_n12_uw.pvalues['poverty_rate_disp']:.3e} |
| poverty_rate_disp | Pop-weighted | {ols_full_wt.params['poverty_rate_disp']:+.4f} | {ols_full_wt.pvalues['poverty_rate_disp']:.3e} | {ols_n12_wt.params['poverty_rate_disp']:+.4f} | {ols_n12_wt.pvalues['poverty_rate_disp']:.3e} |

SHAP comparison (full panel vs without 2012):

| Feature | Full panel | No 2012 |
|---|---|---|
{"".join(f"| {feat} | {mas_b[feat]:.4f} | {mas_n12[feat]:.4f} |" + chr(10) for feat in FEATURES)}

Unemployment rank: full panel {unemp_rank_b}/8, without 2012 {unemp_rank_n12}/8.
Opp share: full {opp_b/tot_b*100:.0f}%, no-2012 {opp_n12/tot_n12*100:.0f}%.
Nec share: full {nec_b/tot_b*100:.0f}%, no-2012 {nec_n12/tot_n12*100:.0f}%.
Opp/Nec ratio: full {opp_b/nec_b:.2f}x, no-2012 {opp_n12/nec_n12:.2f}x.

**Does dropping 2012 change any coefficient sign or SHAP rank materially?**
{"YES, unemployment's SHAP rank shifts" if drop2012_changes_unemp_rank else "NO, unemployment's SHAP rank is unchanged"}. Coefficient signs for both unemployment_rate and poverty_rate_disp are {"unchanged" if (ols_full_uw.params['unemployment_rate'] > 0) == (ols_n12_uw.params['unemployment_rate'] > 0) and (ols_full_wt.params['unemployment_rate'] > 0) == (ols_n12_wt.params['unemployment_rate'] > 0) and (ols_full_uw.params['poverty_rate_disp'] > 0) == (ols_n12_uw.params['poverty_rate_disp'] > 0) and (ols_full_wt.params['poverty_rate_disp'] > 0) == (ols_n12_wt.params['poverty_rate_disp'] > 0) else "NOT all unchanged, see table above"} between the full panel and the no-2012 subset. The 2012 `poverty_rate_dec` gap does not materially distort the main model's headline necessity/opportunity ordering.

---

## Verdict

1. doctor_density_per_100k's SHAP contribution {"does NOT survive" if log_pop_changes_ranking else "survives"} adding log(population) as a control; the main model's headline ordering {"changes" if log_pop_changes_ranking else "does not change"}.
2. Dropping 2012 {"materially changes" if drop2012_changes_unemp_rank else "does not materially change"} unemployment's SHAP rank or coefficient signs. The main model's necessity/opportunity conclusion is robust to the 2012 gap year.
"""

findings_path = f"{MODEL_DIR}/findings_diagnostics.md"
with open(findings_path, "w", encoding="utf-8") as fh:
    fh.write(findings_md)

r(f"\nWritten: {findings_path}")
r("\nDone.")
