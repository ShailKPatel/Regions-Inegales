"""
Feature engineering + baseline model for firm_rate (firm creations per 1000 pop).
Reads merged/france_panel_master.csv (read-only) + sources/population_insee.csv.
Writes figures to figures/, metrics to model/metrics.md.
"""

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
    early_stopping_rounds=20,
    eval_metric="mae",
)


def fit_predict_oof(X, y, train_idx, test_idx):
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        eval_set=[(X.iloc[test_idx], y.iloc[test_idx])],
        verbose=False,
    )
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

# Full-data XGBoost fit (used later for SHAP); no early stopping eval set needed,
# but XGBRegressor requires eval_set if early_stopping_rounds is set -> reuse all data.
xgb_full_params = dict(xgb_params)
xgb_full_params.pop("early_stopping_rounds")
xgb_full = xgb.XGBRegressor(**xgb_full_params)
xgb_full.fit(X, y)

# OLS - unweighted
X_ols = sm.add_constant(X)
ols_unweighted = sm.OLS(y, X_ols).fit()
log("\nOLS (unweighted) summary:")
log(str(ols_unweighted.summary()))

# OLS - population weighted
ols_weighted = sm.WLS(y, X_ols, weights=weights).fit()
log("\nOLS (population-weighted) summary:")
log(str(ols_weighted.summary()))

log("\nOLS coefficient comparison (unweighted vs weighted):")
coef_compare = pd.DataFrame({
    "coef_unweighted": ols_unweighted.params,
    "pval_unweighted": ols_unweighted.pvalues,
    "coef_weighted": ols_weighted.params,
    "pval_weighted": ols_weighted.pvalues,
})
log(coef_compare.to_string())

# ---------------------------------------------------------------------------
# STEP 4 - SHAP
# ---------------------------------------------------------------------------
log("\n## STEP 4 - SHAP (global, full-data XGBoost fit)\n")

explainer = shap.TreeExplainer(xgb_full)
shap_values = explainer.shap_values(X)

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
log("Mean |SHAP| by feature (full data):")
log(mean_abs_shap.to_string())

gini_rank = list(mean_abs_shap.index).index("gini_disp") + 1
log(f"\ngini_disp SHAP importance rank: {gini_rank} of {len(FEATURES)}")

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

# Gini confound check: drop Ile-de-France
mask_no_idf = ~df["dep_code"].isin(IDF_CODES)
X_no_idf = X[mask_no_idf].reset_index(drop=True)
y_no_idf = y[mask_no_idf].reset_index(drop=True)
dep_no_idf = df.loc[mask_no_idf, "dep_code"].reset_index(drop=True)

xgb_no_idf = xgb.XGBRegressor(**xgb_full_params)
xgb_no_idf.fit(X_no_idf, y_no_idf)

explainer_no_idf = shap.TreeExplainer(xgb_no_idf)
shap_values_no_idf = explainer_no_idf.shap_values(X_no_idf)
mean_abs_shap_no_idf = pd.Series(
    np.abs(shap_values_no_idf).mean(axis=0), index=FEATURES
).sort_values(ascending=False)
gini_rank_no_idf = list(mean_abs_shap_no_idf.index).index("gini_disp") + 1

log("\nGINI CONFOUND CHECK - with vs without Ile-de-France (75,77,78,91,92,93,94,95):")
log(f"  WITH IdF:    gini_disp mean|SHAP| = {mean_abs_shap['gini_disp']:.4f}, rank = {gini_rank}/{len(FEATURES)}")
log(f"  WITHOUT IdF: gini_disp mean|SHAP| = {mean_abs_shap_no_idf['gini_disp']:.4f}, rank = {gini_rank_no_idf}/{len(FEATURES)}")

# OOF R2 without IdF, using LODO on remaining departments
gkf_no_idf = GroupKFold(n_splits=dep_no_idf.nunique())
lodo_splits_no_idf = list(gkf_no_idf.split(X_no_idf, y_no_idf, groups=dep_no_idf.values))
r2_lodo_no_idf, mae_lodo_no_idf, _ = run_cv_scheme(
    X_no_idf, y_no_idf, lodo_splits_no_idf, "Leave-One-Department-Out, NO IdF (XGB)"
)

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

summary_md.append("\n## OLS coefficients (unweighted vs weighted)\n")
summary_md.append(coef_compare.to_markdown() + "\n")

summary_md.append("\n## SHAP top features (mean |SHAP|, full data)\n")
summary_md.append(mean_abs_shap.to_markdown() + "\n")

summary_md.append("\n## Gini confound check (with vs without Ile-de-France)\n")
summary_md.append(f"- WITH IdF: gini_disp mean|SHAP| = {mean_abs_shap['gini_disp']:.4f}, rank {gini_rank}/{len(FEATURES)}; LODO R2 = {r2_lodo:.4f}\n")
summary_md.append(f"- WITHOUT IdF: gini_disp mean|SHAP| = {mean_abs_shap_no_idf['gini_disp']:.4f}, rank {gini_rank_no_idf}/{len(FEATURES)}; LODO R2 = {r2_lodo_no_idf:.4f}\n")

summary_md.append("\n## Figures\n")
summary_md.append("- figures/model_shap_beeswarm.png\n")
summary_md.append("- figures/model_shap_bar.png\n")

with open(f"{MODEL_DIR}/metrics.md", "w") as f:
    f.write("".join(summary_md))

log(f"\nSaved report to {MODEL_DIR}/metrics.md")
log(f"Saved figures to {FIG_DIR}/model_shap_beeswarm.png and {FIG_DIR}/model_shap_bar.png")
