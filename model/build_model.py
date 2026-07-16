"""
DEPRECATED - superseded by model/final_model.py.
This is the earlier draft of the same pipeline (same OOF-SHAP-via-LODO
methodology, same numbers). It predates final_model.py's polished
robustness sections and figures, and its output (model/metrics.md) is
not cited anywhere in FINDINGS.md. Kept for reference only; cite
model/final_model.py + model/findings_final.md as canonical.

Feature engineering + baseline model for firm_rate (firm creations per 1000 pop).
Reads merged/france_panel_master.csv (read-only) + sources/population_insee.csv.
Writes figures to figures/, metrics to model/metrics.md.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from panel_config import PANEL_START, PANEL_END

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

RNG = 42
MASTER_PATH = "merged/france_panel_master.csv"
POP_PATH = "sources/population_insee.csv"
FIG_DIR = "figures"
MODEL_DIR = "model"

FEATURES = [
    "q2_disp",
    "gini_disp",
    "poverty_rate_disp",
    "unemployment_rate",
    "doctor_density_per_100k",
    "edu_share_sup",
    "pct_urban",
    "pct_wages",
]
TARGET = "firm_rate"

IDF_CODES = {"75", "77", "78", "91", "92", "93", "94", "95"}

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(msg)


# ---------------------------------------------------------------------------
# STEP 1 - TARGET + FEATURE MATRIX
# ---------------------------------------------------------------------------
log("## STEP 1 - Target + feature matrix\n")

master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
pop = pd.read_csv(POP_PATH, sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')

df = master.merge(pop, on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0, "unmatched pop_jan1 rows after merge"
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)

df[TARGET] = df["total_firm_creations"] / df["pop_jan1"] * 1000

X = df[FEATURES].copy()
y = df[TARGET].copy()

log(f"Final feature list ({len(FEATURES)}): {FEATURES}")
log(f"Target: {TARGET} = total_firm_creations / pop_jan1 * 1000\n")
log(f"X shape: {X.shape}")
log(f"y shape: {y.shape}")
null_counts = X.isna().sum()
log(f"Nulls in X:\n{null_counts.to_string()}")
log(f"Nulls in y: {y.isna().sum()}")
assert X.isna().sum().sum() == 0, "unexpected nulls in feature matrix"
assert y.isna().sum() == 0, "unexpected nulls in target"

groups_dep = df["dep_code"].values
groups_year = df["year"].values
weights = df["pop_jan1"].values

# ---------------------------------------------------------------------------
# STEP 2 - VALIDATION DESIGN
# ---------------------------------------------------------------------------
log("\n## STEP 2 - Validation design (panel-aware CV)\n")

xgb_params = dict(
    max_depth=4,
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RNG,
)


def fit_predict_oof(X, y, train_idx, test_idx):
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
    pred = model.predict(X.iloc[test_idx])
    return pred


def run_cv_scheme(X, y, splits, label):
    oof_pred = np.full(len(y), np.nan)
    for train_idx, test_idx in splits:
        oof_pred[test_idx] = fit_predict_oof(X, y, train_idx, test_idx)
    r2 = r2_score(y, oof_pred)
    mae = mean_absolute_error(y, oof_pred)
    log(f"{label}: OOF R2 = {r2:.4f}, OOF MAE = {mae:.4f}")
    return r2, mae, oof_pred


# (a) Leave-One-Year-Out
years = sorted(df["year"].unique())
loyo_splits = []
for yr in years:
    test_idx = np.where(groups_year == yr)[0]
    train_idx = np.where(groups_year != yr)[0]
    loyo_splits.append((train_idx, test_idx))

# (b) Leave-One-Department-Out via GroupKFold (96 groups -> 96 folds)
gkf = GroupKFold(n_splits=df["dep_code"].nunique())
lodo_splits = list(gkf.split(X, y, groups=groups_dep))

# Random KFold (leaky baseline, for contrast only)
kf = KFold(n_splits=10, shuffle=True, random_state=RNG)
kfold_splits = list(kf.split(X, y))

r2_loyo, mae_loyo, _ = run_cv_scheme(X, y, loyo_splits, "Leave-One-Year-Out (XGB)")
r2_lodo, mae_lodo, _ = run_cv_scheme(X, y, lodo_splits, "Leave-One-Department-Out (XGB)")
r2_kfold, mae_kfold, _ = run_cv_scheme(X, y, kfold_splits, "Random 10-fold (XGB) [OPTIMISTIC/LEAKY BASELINE]")

# ---------------------------------------------------------------------------
# STEP 3 - XGBOOST BASELINE + OLS
# ---------------------------------------------------------------------------
log("\n## STEP 3 - XGBoost baseline + OLS\n")

log("XGBoost OOF metrics table:")
log(f"{'scheme':<45}{'R2':>10}{'MAE':>10}")
log(f"{'Leave-One-Year-Out':<45}{r2_loyo:>10.4f}{mae_loyo:>10.4f}")
log(f"{'Leave-One-Department-Out':<45}{r2_lodo:>10.4f}{mae_lodo:>10.4f}")
log(f"{'Random 10-fold (leaky baseline)':<45}{r2_kfold:>10.4f}{mae_kfold:>10.4f}")

overfit_gap = r2_kfold - r2_lodo
log(f"\nOverfitting gap (random KFold R2 - LODO R2): {overfit_gap:.4f}")

# OLS with department-clustered standard errors (primary)
X_ols = sm.add_constant(X)
_ols_nc_uw     = sm.OLS(y, X_ols).fit()
_ols_nc_wt     = sm.WLS(y, X_ols, weights=weights).fit()
ols_unweighted = sm.OLS(y, X_ols).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})
ols_weighted   = sm.WLS(y, X_ols, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': groups_dep})

log("\nOLS (unweighted, department-clustered SE) summary:")
log(str(ols_unweighted.summary()))

log("\nOLS (population-weighted, department-clustered SE) summary:")
log(str(ols_weighted.summary()))

log("\nOLS coefficient comparison (unweighted vs weighted, department-clustered SE):")
coef_compare = pd.DataFrame({
    "coef_unweighted": ols_unweighted.params,
    "pval_unweighted": ols_unweighted.pvalues,
    "coef_weighted":   ols_weighted.params,
    "pval_weighted":   ols_weighted.pvalues,
})
log(coef_compare.to_string())

log("\nC-1 NON-CLUSTERED vs DEPARTMENT-CLUSTERED SE (full panel):")
for _feat in ["unemployment_rate", "poverty_rate_disp"]:
    log(f"  {_feat}:")
    log(f"    UW non-clustered: coef={_ols_nc_uw.params[_feat]:+.4f}  p={_ols_nc_uw.pvalues[_feat]:.3e}")
    log(f"    UW clustered:     coef={ols_unweighted.params[_feat]:+.4f}  p={ols_unweighted.pvalues[_feat]:.3e}")
    log(f"    WT non-clustered: coef={_ols_nc_wt.params[_feat]:+.4f}  p={_ols_nc_wt.pvalues[_feat]:.3e}")
    log(f"    WT clustered:     coef={ols_weighted.params[_feat]:+.4f}  p={ols_weighted.pvalues[_feat]:.3e}")

# ---------------------------------------------------------------------------
# STEP 4 - SHAP (OOF via LODO, 96 folds)
# ---------------------------------------------------------------------------
log("\n## STEP 4 - SHAP (OOF via LODO, 96 folds)\n")

# In-sample SHAP for side-by-side comparison only
_xgb_is = xgb.XGBRegressor(**xgb_params)
_xgb_is.fit(X, y)
_shap_insample = shap.TreeExplainer(_xgb_is).shap_values(X)
_mas_insample  = pd.Series(np.abs(_shap_insample).mean(axis=0), index=FEATURES)

# OOF SHAP, primary
shap_values = np.zeros((len(X), len(FEATURES)), dtype=float)
for tr, te in lodo_splits:
    _m = xgb.XGBRegressor(**xgb_params)
    _m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
    shap_values[te] = shap.TreeExplainer(_m).shap_values(X.iloc[te])

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)

_OPP = ["edu_share_sup", "q2_disp", "pct_urban", "doctor_density_per_100k"]
_NEC = ["unemployment_rate", "poverty_rate_disp"]
_OTH = ["gini_disp", "pct_wages"]

log("C-2 IN-SAMPLE vs OOF MEAN |SHAP| COMPARISON:")
log(f"  {'Feature':<30} {'In-sample':>10} {'OOF':>10}")
log("  " + "-" * 52)
for _feat in FEATURES:
    log(f"  {_feat:<30} {_mas_insample[_feat]:>10.4f} {mean_abs_shap[_feat]:>10.4f}")

_opp_is  = _mas_insample[_OPP].sum()
_nec_is  = _mas_insample[_NEC].sum()
_oth_is  = _mas_insample[_OTH].sum()
_tot_is  = _opp_is + _nec_is + _oth_is
_opp_oof = mean_abs_shap[_OPP].sum()
_nec_oof = mean_abs_shap[_NEC].sum()
_oth_oof = mean_abs_shap[_OTH].sum()
_tot_oof = _opp_oof + _nec_oof + _oth_oof

log("\nGroup shares:")
log(f"  OPPORTUNITY : in-sample {_opp_is/_tot_is*100:.0f}%  OOF {_opp_oof/_tot_oof*100:.0f}%")
log(f"  NECESSITY   : in-sample {_nec_is/_tot_is*100:.0f}%  OOF {_nec_oof/_tot_oof*100:.0f}%")
log(f"  OTHER       : in-sample {_oth_is/_tot_is*100:.0f}%  OOF {_oth_oof/_tot_oof*100:.0f}%")
log(f"  Opp/Nec     : in-sample {_opp_is/_nec_is:.2f}x  OOF {_opp_oof/_nec_oof:.2f}x")
_unemp_rank_oof = list(mean_abs_shap.index).index("unemployment_rate") + 1
log(f"  Unemployment rank (OOF): {_unemp_rank_oof}/8  opportunity dominant: {_opp_oof > _nec_oof}")

gini_rank = list(mean_abs_shap.index).index("gini_disp") + 1
log(f"\ngini_disp SHAP importance rank (OOF): {gini_rank} of {len(FEATURES)}")
log("\nMean |SHAP| by feature (OOF):")
log(mean_abs_shap.to_string())

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Gini confound check: drop Ile-de-France (OOF SHAP)
mask_no_idf = ~df["dep_code"].isin(IDF_CODES)
X_no_idf   = X[mask_no_idf].reset_index(drop=True)
y_no_idf   = y[mask_no_idf].reset_index(drop=True)
dep_no_idf = df.loc[mask_no_idf, "dep_code"].reset_index(drop=True)

gkf_no_idf      = GroupKFold(n_splits=dep_no_idf.nunique())
lodo_splits_no_idf = list(gkf_no_idf.split(X_no_idf, y_no_idf, groups=dep_no_idf.values))

shap_no_idf     = np.zeros((len(X_no_idf), len(FEATURES)), dtype=float)
oof_pred_no_idf = np.full(len(y_no_idf), np.nan)
for tr, te in lodo_splits_no_idf:
    _m = xgb.XGBRegressor(**xgb_params)
    _m.fit(X_no_idf.iloc[tr], y_no_idf.iloc[tr], verbose=False)
    oof_pred_no_idf[te] = _m.predict(X_no_idf.iloc[te])
    shap_no_idf[te]     = shap.TreeExplainer(_m).shap_values(X_no_idf.iloc[te])

r2_lodo_no_idf  = r2_score(y_no_idf, oof_pred_no_idf)
mae_lodo_no_idf = mean_absolute_error(y_no_idf, oof_pred_no_idf)
log(f"\nLeave-One-Department-Out, NO IdF (XGB): OOF R2 = {r2_lodo_no_idf:.4f}, OOF MAE = {mae_lodo_no_idf:.4f}")

mean_abs_shap_no_idf = pd.Series(
    np.abs(shap_no_idf).mean(axis=0), index=FEATURES
).sort_values(ascending=False)
gini_rank_no_idf = list(mean_abs_shap_no_idf.index).index("gini_disp") + 1

log("\nGINI CONFOUND CHECK (OOF) - with vs without Ile-de-France (75,77,78,91,92,93,94,95):")
log(f"  WITH IdF:    gini_disp mean|SHAP| = {mean_abs_shap['gini_disp']:.4f}, rank = {gini_rank}/{len(FEATURES)}")
log(f"  WITHOUT IdF: gini_disp mean|SHAP| = {mean_abs_shap_no_idf['gini_disp']:.4f}, rank = {gini_rank_no_idf}/{len(FEATURES)}")
log(f"  Gini rank stable OOF: {gini_rank == gini_rank_no_idf}")
log(f"\nLODO R2 with IdF: {r2_lodo:.4f}  |  LODO R2 without IdF: {r2_lodo_no_idf:.4f}")

# ---------------------------------------------------------------------------
# STEP 5 - REPORT
# ---------------------------------------------------------------------------
log("\n## STEP 5 - Summary report\n")

summary_md = []
summary_md.append("# Model report - firm_rate baseline\n")
summary_md.append(f"Rows: {X.shape[0]}, features: {X.shape[1]}\n")
summary_md.append(f"Features: {FEATURES}\n")
summary_md.append("\n## CV metrics (XGBoost)\n")
summary_md.append("| scheme | R2 | MAE |\n|---|---|---|\n")
summary_md.append(f"| Leave-One-Year-Out | {r2_loyo:.4f} | {mae_loyo:.4f} |\n")
summary_md.append(f"| Leave-One-Department-Out | {r2_lodo:.4f} | {mae_lodo:.4f} |\n")
summary_md.append(f"| Random 10-fold (leaky baseline) | {r2_kfold:.4f} | {mae_kfold:.4f} |\n")
summary_md.append(f"\nOverfitting gap (random KFold R2 - LODO R2): {overfit_gap:.4f}\n")

summary_md.append("\n## OLS R2\n")
summary_md.append(f"- Unweighted OLS R2: {ols_unweighted.rsquared:.4f}\n")
summary_md.append(f"- Population-weighted OLS R2: {ols_weighted.rsquared:.4f}\n")

summary_md.append("\n## OLS coefficients (unweighted vs weighted, department-clustered SE)\n")
summary_md.append(coef_compare.to_markdown() + "\n")

summary_md.append("\n## SHAP top features (mean |SHAP|, OOF via LODO)\n")
summary_md.append(mean_abs_shap.to_markdown() + "\n")

summary_md.append("\n## Gini confound check (OOF SHAP, with vs without Ile-de-France)\n")
summary_md.append(f"- WITH IdF: gini_disp mean|SHAP| = {mean_abs_shap['gini_disp']:.4f}, rank {gini_rank}/{len(FEATURES)}; LODO R2 = {r2_lodo:.4f}\n")
summary_md.append(f"- WITHOUT IdF: gini_disp mean|SHAP| = {mean_abs_shap_no_idf['gini_disp']:.4f}, rank {gini_rank_no_idf}/{len(FEATURES)}; LODO R2 = {r2_lodo_no_idf:.4f}\n")

summary_md.append("\n## Figures\n")
summary_md.append("- figures/model_shap_beeswarm.png\n")
summary_md.append("- figures/model_shap_bar.png\n")

with open(f"{MODEL_DIR}/metrics.md", "w") as f:
    f.write("".join(summary_md))

log(f"\nSaved report to {MODEL_DIR}/metrics.md")
log(f"Saved figures to {FIG_DIR}/model_shap_beeswarm.png and {FIG_DIR}/model_shap_bar.png")
