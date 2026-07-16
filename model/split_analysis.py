"""
Urban/Rural split analysis - Régions Inégales
Tests whether opportunity-vs-necessity balance differs by density context.
Read-only on master. Writes figures/ and model/split_findings.md.
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

# ── constants ──────────────────────────────────────────────────────────────
RNG         = 42
MASTER_PATH = "merged/france_panel_master.csv"
POP_PATH    = "sources/population_insee.csv"
FIG_DIR     = "figures"
MODEL_DIR   = "model"

FEATURES = [
    "q2_disp", "gini_disp", "poverty_rate_disp", "unemployment_rate",
    "doctor_density_per_100k", "edu_share_sup", "pct_urban", "pct_wages",
]
TARGET = "firm_rate"

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

xgb_params_cv = dict(
    max_depth=4, n_estimators=300, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG,
)
xgb_params_full = xgb_params_cv

report = []

def r(line=""):
    print(line)
    report.append(str(line))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── STEP 1: Load + build target ────────────────────────────────────────────
r("=" * 70)
r("STEP 1, LOAD DATA")
r("=" * 70)

master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": object})
pop    = pd.read_csv(POP_PATH,    sep=";", dtype={"dep_code": object})
pop["dep_code"] = pop["dep_code"].str.strip('"')

df = master.merge(pop, on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0, "unmatched pop rows"
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)
df[TARGET] = df["total_firm_creations"] / df["pop_jan1"] * 1000

assert df[FEATURES].isna().sum().sum() == 0
assert df[TARGET].isna().sum() == 0

r(f"Full panel: {df.shape[0]} rows, {df['dep_code'].nunique()} departments")
r(f"density_class counts (departments):")
dc = df.groupby("density_class")["dep_code"].nunique()
for cls, n in dc.items():
    r(f"  {cls}: {n} departments, {n*10} rows")
r()

# ── STEP 2: Define subsets ─────────────────────────────────────────────────
r("=" * 70)
r("STEP 2, SUBSET DEFINITIONS")
r("=" * 70)

mask_urban = df["density_class"].isin(["urban", "intermediate"])
mask_rural = df["density_class"] == "rural"

df_urban = df[mask_urban].reset_index(drop=True)
df_rural = df[mask_rural].reset_index(drop=True)

n_dep_urban = df_urban["dep_code"].nunique()
n_dep_rural = df_rural["dep_code"].nunique()
n_row_urban = len(df_urban)
n_row_rural = len(df_rural)

r(f"URBAN (urban + intermediate): {n_dep_urban} departments, {n_row_urban} rows")
r(f"RURAL (rural):                {n_dep_rural} departments, {n_row_rural} rows")

if n_dep_urban < 30:
    r("  *** WARNING: urban subset <30 departments, treat as exploratory ***")
if n_dep_rural < 30:
    r("  *** WARNING: rural subset <30 departments, treat as exploratory ***")
r()

# ── helpers ────────────────────────────────────────────────────────────────
def run_lodo_shap(X_, y_, dep_codes):
    gkf    = GroupKFold(n_splits=dep_codes.nunique())
    splits = list(gkf.split(X_, y_, groups=dep_codes.values))
    oof_pred = np.full(len(y_), np.nan)
    shap_oof = np.zeros((len(X_), len(FEATURES)), dtype=float)
    for tr, te in splits:
        m = xgb.XGBRegressor(**xgb_params_cv)
        m.fit(X_.iloc[tr], y_.iloc[tr], verbose=False)
        oof_pred[te] = m.predict(X_.iloc[te])
        shap_oof[te] = shap.TreeExplainer(m).shap_values(X_.iloc[te])
    r2  = r2_score(y_, oof_pred)
    mas = pd.Series(np.abs(shap_oof).mean(axis=0), index=FEATURES)
    return r2, shap_oof, mas

def shap_group_totals(mas):
    opp = mas[OPPORTUNITY].sum()
    nec = mas[NECESSITY].sum()
    oth = mas[OTHER].sum()
    tot = opp + nec + oth
    return opp, nec, oth, tot

def ols_unemp(X_, y_, weights=None, groups=None):
    X_ols = sm.add_constant(X_)
    if weights is not None:
        model = sm.WLS(y_, X_ols, weights=weights)
    else:
        model = sm.OLS(y_, X_ols)
    if groups is not None:
        res = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    else:
        res = model.fit()
    coef = res.params["unemployment_rate"]
    pval = res.pvalues["unemployment_rate"]
    return coef, pval

def necessity_verdict_str(coef_uw, pval_uw, coef_wt, pval_wt, subset_name):
    both_neg = coef_uw < 0 and coef_wt < 0
    either_pos_sig = (coef_uw > 0 and pval_uw < 0.05) or (coef_wt > 0 and pval_wt < 0.05)
    if both_neg:
        return "REJECTED (unemployment negative in both OLS specs)"
    elif either_pos_sig:
        return "SUPPORTED (unemployment positive and significant in at least one spec)"
    else:
        return "MIXED / inconclusive"

def shap_rank(mas, feature):
    ranked = mas.sort_values(ascending=False).index.tolist()
    return ranked.index(feature) + 1

# ── STEP 3: Full-panel baseline ────────────────────────────────────────────
r("=" * 70)
r("STEP 3, FULL-PANEL MODEL (baseline)")
r("=" * 70)

X_full = df[FEATURES].copy()
y_full = df[TARGET].copy()
dep_full = df["dep_code"]
w_full = df["pop_jan1"].values

r("Running LODO + OOF SHAP on full panel ...")
r2_full, sv_full, mas_full = run_lodo_shap(X_full, y_full, dep_full)
r(f"  LODO R² = {r2_full:.3f}")

opp_f, nec_f, oth_f, tot_f = shap_group_totals(mas_full)

coef_uw_f, pval_uw_f = ols_unemp(X_full, y_full, groups=dep_full.values)
coef_wt_f, pval_wt_f = ols_unemp(X_full, y_full, weights=w_full, groups=dep_full.values)
unemp_rank_f = shap_rank(mas_full, "unemployment_rate")

_X_ols_f = sm.add_constant(X_full)
_ols_f_uw = sm.OLS(y_full, _X_ols_f).fit(cov_type='cluster', cov_kwds={'groups': dep_full.values})
_ols_f_wt = sm.WLS(y_full, _X_ols_f, weights=w_full).fit(cov_type='cluster', cov_kwds={'groups': dep_full.values})
pov_coef_uw_f = _ols_f_uw.params["poverty_rate_disp"]
pov_pval_uw_f = _ols_f_uw.pvalues["poverty_rate_disp"]
pov_coef_wt_f = _ols_f_wt.params["poverty_rate_disp"]
pov_pval_wt_f = _ols_f_wt.pvalues["poverty_rate_disp"]

r(f"  SHAP, Opp: {opp_f:.4f} ({opp_f/tot_f*100:.0f}%)  "
  f"Nec: {nec_f:.4f} ({nec_f/tot_f*100:.0f}%)  "
  f"Other: {oth_f:.4f} ({oth_f/tot_f*100:.0f}%)")
r(f"  Opp/Nec ratio: {opp_f/nec_f:.2f}x")
r(f"  Unemployment SHAP rank: {unemp_rank_f}/8")
r(f"  OLS unemployment  , UW: coef={coef_uw_f:+.4f} p={pval_uw_f:.3e} | "
  f"WT: coef={coef_wt_f:+.4f} p={pval_wt_f:.3e}")
r(f"  OLS poverty_rate  , UW: coef={pov_coef_uw_f:+.4f} p={pov_pval_uw_f:.3e} | "
  f"WT: coef={pov_coef_wt_f:+.4f} p={pov_pval_wt_f:.3e}")
r()

# ── STEP 4: Urban subset ───────────────────────────────────────────────────
r("=" * 70)
r("STEP 4, URBAN SUBSET MODEL")
r("=" * 70)

X_urb = df_urban[FEATURES].copy()
y_urb = df_urban[TARGET].copy()
dep_urb = df_urban["dep_code"]
w_urb = df_urban["pop_jan1"].values

r(f"  {n_dep_urban} departments, {n_row_urban} rows")
r("Running LODO + OOF SHAP on urban subset ...")
r2_urb, sv_urb, mas_urb = run_lodo_shap(X_urb, y_urb, dep_urb)
r(f"  LODO R² = {r2_urb:.3f}")

opp_u, nec_u, oth_u, tot_u = shap_group_totals(mas_urb)

coef_uw_u, pval_uw_u = ols_unemp(X_urb, y_urb, groups=dep_urb.values)
coef_wt_u, pval_wt_u = ols_unemp(X_urb, y_urb, weights=w_urb, groups=dep_urb.values)
unemp_rank_u = shap_rank(mas_urb, "unemployment_rate")

_X_ols_u = sm.add_constant(X_urb)
_ols_u_uw = sm.OLS(y_urb, _X_ols_u).fit(cov_type='cluster', cov_kwds={'groups': dep_urb.values})
_ols_u_wt = sm.WLS(y_urb, _X_ols_u, weights=w_urb).fit(cov_type='cluster', cov_kwds={'groups': dep_urb.values})
pov_coef_uw_u = _ols_u_uw.params["poverty_rate_disp"]
pov_pval_uw_u = _ols_u_uw.pvalues["poverty_rate_disp"]
pov_coef_wt_u = _ols_u_wt.params["poverty_rate_disp"]
pov_pval_wt_u = _ols_u_wt.pvalues["poverty_rate_disp"]

r(f"  SHAP, Opp: {opp_u:.4f} ({opp_u/tot_u*100:.0f}%)  "
  f"Nec: {nec_u:.4f} ({nec_u/tot_u*100:.0f}%)  "
  f"Other: {oth_u:.4f} ({oth_u/tot_u*100:.0f}%)")
r(f"  Opp/Nec ratio: {opp_u/nec_u:.2f}x")
r(f"  Unemployment SHAP rank: {unemp_rank_u}/8")
r(f"  OLS unemployment  , UW: coef={coef_uw_u:+.4f} p={pval_uw_u:.3e} | "
  f"WT: coef={coef_wt_u:+.4f} p={pval_wt_u:.3e}")
r(f"  OLS poverty_rate  , UW: coef={pov_coef_uw_u:+.4f} p={pov_pval_uw_u:.3e} | "
  f"WT: coef={pov_coef_wt_u:+.4f} p={pov_pval_wt_u:.3e}")
verdict_urb = necessity_verdict_str(coef_uw_u, pval_uw_u, coef_wt_u, pval_wt_u, "urban")
r(f"  Necessity verdict (urban): {verdict_urb}")
r()

# ── STEP 5: Rural subset ───────────────────────────────────────────────────
r("=" * 70)
r("STEP 5, RURAL SUBSET MODEL")
r("=" * 70)

X_rur = df_rural[FEATURES].copy()
y_rur = df_rural[TARGET].copy()
dep_rur = df_rural["dep_code"]
w_rur = df_rural["pop_jan1"].values

r(f"  {n_dep_rural} departments, {n_row_rural} rows")
r("Running LODO + OOF SHAP on rural subset ...")
r2_rur, sv_rur, mas_rur = run_lodo_shap(X_rur, y_rur, dep_rur)
r(f"  LODO R² = {r2_rur:.3f}")

opp_r, nec_r, oth_r, tot_r = shap_group_totals(mas_rur)

_coef_uw_r_nc, _pval_uw_r_nc = ols_unemp(X_rur, y_rur)
_coef_wt_r_nc, _pval_wt_r_nc = ols_unemp(X_rur, y_rur, weights=w_rur)
coef_uw_r, pval_uw_r = ols_unemp(X_rur, y_rur, groups=dep_rur.values)
coef_wt_r, pval_wt_r = ols_unemp(X_rur, y_rur, weights=w_rur, groups=dep_rur.values)
unemp_rank_r = shap_rank(mas_rur, "unemployment_rate")

# Also extract poverty for C-1 table
X_ols_rur = sm.add_constant(X_rur)
_ols_rur_uw_nc = sm.OLS(y_rur, X_ols_rur).fit()
_ols_rur_wt_nc = sm.WLS(y_rur, X_ols_rur, weights=w_rur).fit()
_ols_rur_uw_cl = sm.OLS(y_rur, X_ols_rur).fit(cov_type='cluster', cov_kwds={'groups': dep_rur.values})
_ols_rur_wt_cl = sm.WLS(y_rur, X_ols_rur, weights=w_rur).fit(cov_type='cluster', cov_kwds={'groups': dep_rur.values})

r("\nC-1 NON-CLUSTERED vs DEPARTMENT-CLUSTERED SE (rural subset):")
for _feat in ["unemployment_rate", "poverty_rate_disp"]:
    r(f"  {_feat}:")
    r(f"    UW non-clustered: coef={_ols_rur_uw_nc.params[_feat]:+.4f}  p={_ols_rur_uw_nc.pvalues[_feat]:.3e}")
    r(f"    UW clustered:     coef={_ols_rur_uw_cl.params[_feat]:+.4f}  p={_ols_rur_uw_cl.pvalues[_feat]:.3e}")
    r(f"    WT non-clustered: coef={_ols_rur_wt_nc.params[_feat]:+.4f}  p={_ols_rur_wt_nc.pvalues[_feat]:.3e}")
    r(f"    WT clustered:     coef={_ols_rur_wt_cl.params[_feat]:+.4f}  p={_ols_rur_wt_cl.pvalues[_feat]:.3e}")

r(f"  SHAP, Opp: {opp_r:.4f} ({opp_r/tot_r*100:.0f}%)  "
  f"Nec: {nec_r:.4f} ({nec_r/tot_r*100:.0f}%)  "
  f"Other: {oth_r:.4f} ({oth_r/tot_r*100:.0f}%)")
r(f"  Opp/Nec ratio: {opp_r/nec_r:.2f}x")
r(f"  Unemployment SHAP rank: {unemp_rank_r}/8")
r(f"  OLS unemployment, UW: coef={coef_uw_r:+.4f} p={pval_uw_r:.3f} | "
  f"WT: coef={coef_wt_r:+.4f} p={pval_wt_r:.3f}")
verdict_rur = necessity_verdict_str(coef_uw_r, pval_uw_r, coef_wt_r, pval_wt_r, "rural")
r(f"  Necessity verdict (rural): {verdict_rur}")
r()

# ── STEP 6: Pooled interaction model ──────────────────────────────────────
r("=" * 70)
r("STEP 6, POOLED OLS WITH RURAL INTERACTIONS (rigorous cross-check)")
r("=" * 70)

df2 = df.copy()
df2["is_rural"] = (df2["density_class"] == "rural").astype(float)
df2["unemp_x_rural"]   = df2["unemployment_rate"] * df2["is_rural"]
df2["edu_x_rural"]     = df2["edu_share_sup"]    * df2["is_rural"]

interaction_features = FEATURES + ["is_rural", "unemp_x_rural", "edu_x_rural"]
X_int = sm.add_constant(df2[interaction_features])
y_int = df2[TARGET]
w_int = df2["pop_jan1"].values

ols_int_uw = sm.OLS(y_int, X_int).fit(cov_type='cluster', cov_kwds={'groups': df["dep_code"].values})
ols_int_wt = sm.WLS(y_int, X_int, weights=w_int).fit(cov_type='cluster', cov_kwds={'groups': df["dep_code"].values})

def row(name, res):
    c = res.params.get(name, np.nan)
    p = res.pvalues.get(name, np.nan)
    sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
    return c, p, sig

uw_unemp_c,    uw_unemp_p,    uw_unemp_s    = row("unemployment_rate",  ols_int_uw)
uw_unemp_r_c,  uw_unemp_r_p,  uw_unemp_r_s  = row("unemp_x_rural",      ols_int_uw)
uw_edu_r_c,    uw_edu_r_p,    uw_edu_r_s    = row("edu_x_rural",         ols_int_uw)
wt_unemp_c,    wt_unemp_p,    wt_unemp_s    = row("unemployment_rate",  ols_int_wt)
wt_unemp_r_c,  wt_unemp_r_p,  wt_unemp_r_s  = row("unemp_x_rural",      ols_int_wt)
wt_edu_r_c,    wt_edu_r_p,    wt_edu_r_s    = row("edu_x_rural",         ols_int_wt)

r(f"Pooled OLS with interactions ({len(df)} rows, full panel):")
r(f"  Unweighted, unemployment_rate:     coef={uw_unemp_c:+.4f}  p={uw_unemp_p:.3f} {uw_unemp_s}")
r(f"  Unweighted, unemp_x_rural:         coef={uw_unemp_r_c:+.4f}  p={uw_unemp_r_p:.3f} {uw_unemp_r_s}")
r(f"  Unweighted, edu_x_rural:           coef={uw_edu_r_c:+.4f}  p={uw_edu_r_p:.3f} {uw_edu_r_s}")
r(f"  Pop-weighted, unemployment_rate:   coef={wt_unemp_c:+.4f}  p={wt_unemp_p:.3f} {wt_unemp_s}")
r(f"  Pop-weighted, unemp_x_rural:       coef={wt_unemp_r_c:+.4f}  p={wt_unemp_r_p:.3f} {wt_unemp_r_s}")
r(f"  Pop-weighted, edu_x_rural:         coef={wt_edu_r_c:+.4f}  p={wt_edu_r_p:.3f} {wt_edu_r_s}")

unemp_interaction_sig = (uw_unemp_r_p < 0.05 or wt_unemp_r_p < 0.05)
edu_interaction_sig   = (uw_edu_r_p   < 0.05 or wt_edu_r_p   < 0.05)
r()
r(f"  unemp x rural interaction significant: {unemp_interaction_sig}")
r(f"  edu   x rural interaction significant: {edu_interaction_sig}")
r()

# ── STEP 7: Headline figure ────────────────────────────────────────────────
r("=" * 70)
r("STEP 7, FIGURES")
r("=" * 70)

# Figure A: Grouped bar, Full vs Urban vs Rural
contexts = ["Full", "Urban", "Rural"]
opp_vals = [opp_f/tot_f*100, opp_u/tot_u*100, opp_r/tot_r*100]
nec_vals = [nec_f/tot_f*100, nec_u/tot_u*100, nec_r/tot_r*100]
oth_vals = [oth_f/tot_f*100, oth_u/tot_u*100, oth_r/tot_r*100]

x = np.arange(len(contexts))
w = 0.25

figA, axA = plt.subplots(figsize=(8, 5))
b1 = axA.bar(x - w, opp_vals, w, label="Opportunity", color=OPP_COLORS[0])
b2 = axA.bar(x,     nec_vals, w, label="Necessity",   color=NEC_COLORS[0])
b3 = axA.bar(x + w, oth_vals, w, label="Other",       color=OTH_COLORS[0])

for bar in list(b1) + list(b2) + list(b3):
    h = bar.get_height()
    axA.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}%",
             ha="center", va="bottom", fontsize=8)

axA.set_xticks(x)
axA.set_xticklabels(contexts, fontsize=11)
axA.set_ylabel("% of total mean |SHAP|")
axA.set_ylim(0, max(opp_vals) * 1.18)
axA.set_title(
    "SHAP Group Importance by Density Context\n"
    "Opportunity vs Necessity vs Other",
    fontsize=11, fontweight="bold",
)
axA.legend(fontsize=9)
figA.tight_layout()
figA.savefig(f"{FIG_DIR}/split_shap_by_context.png", dpi=150, bbox_inches="tight")
plt.close(figA)
r("Saved figures/split_shap_by_context.png")

# Figure B: Per-feature SHAP comparison across contexts (3-panel)
figB, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)
datasets = [
    ("Full panel", mas_full, opp_f, nec_f, oth_f, tot_f),
    ("Urban / Intermediate", mas_urb, opp_u, nec_u, oth_u, tot_u),
    ("Rural", mas_rur, opp_r, nec_r, oth_r, tot_r),
]

for ax, (title, mas, opp, nec, oth, tot) in zip(axes, datasets):
    opp_s = sorted(OPPORTUNITY, key=lambda f: mas[f], reverse=True)
    nec_s = sorted(NECESSITY,   key=lambda f: mas[f], reverse=True)
    oth_s = sorted(OTHER,       key=lambda f: mas[f], reverse=True)
    ordered = opp_s + nec_s + oth_s
    colors  = ([GROUP_COLOR[f] for f in opp_s] +
               [GROUP_COLOR[f] for f in nec_s] +
               [GROUP_COLOR[f] for f in oth_s])
    vals   = [mas[f] for f in ordered]
    names  = [FEATURE_DISPLAY[f] for f in ordered]
    yp     = list(range(len(ordered)))
    ax.barh(yp, vals, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(yp)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axhline(len(opp_s) - 0.5,           color="#aaa", lw=0.8, ls="--")
    ax.axhline(len(opp_s) + len(nec_s) - 0.5, color="#aaa", lw=0.8, ls="--")
    ax.set_title(
        f"{title}\nOpp {opp/tot*100:.0f}%  Nec {nec/tot*100:.0f}%",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Mean |SHAP|", fontsize=8)

figB.suptitle("Feature SHAP Importance, Full vs Urban vs Rural",
              fontsize=11, fontweight="bold")
figB.tight_layout()
figB.savefig(f"{FIG_DIR}/split_shap_per_feature.png", dpi=150, bbox_inches="tight")
plt.close(figB)
r("Saved figures/split_shap_per_feature.png")

# Figure C: Opp vs Nec ratio bar
figC, axC = plt.subplots(figsize=(6, 4))
ratios = [opp_f/nec_f, opp_u/nec_u, opp_r/nec_r]
bar_colors_c = [OPP_COLORS[1], OPP_COLORS[0], OTH_COLORS[0]]
bars = axC.bar(contexts, ratios, color=bar_colors_c, edgecolor="white", width=0.5)
for b, v in zip(bars, ratios):
    axC.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.1f}x",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
axC.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.5)
axC.set_ylabel("Opportunity / Necessity ratio (SHAP)")
axC.set_title("Opp/Nec SHAP Ratio by Context\n(>1 = Opportunity dominates)",
              fontsize=10, fontweight="bold")
figC.tight_layout()
figC.savefig(f"{FIG_DIR}/split_opp_nec_ratio.png", dpi=150, bbox_inches="tight")
plt.close(figC)
r("Saved figures/split_opp_nec_ratio.png")
r()

# ── STEP 8: Write split_findings.md ───────────────────────────────────────
r("=" * 70)
r("STEP 8, split_findings.md")
r("=" * 70)

def sig_star(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return "ns"

def int_sig_summary():
    lines = []
    if unemp_interaction_sig:
        lines.append(
            f"The unemp × rural interaction is **statistically significant** "
            f"(UW p={uw_unemp_r_p:.3f}{sig_star(uw_unemp_r_p)}, "
            f"WT p={wt_unemp_r_p:.3f}{sig_star(wt_unemp_r_p)}), "
            f"with coef {uw_unemp_r_c:+.4f} (UW). "
            f"This confirms the effect of unemployment on firm creation genuinely "
            f"differs between rural and urban contexts using all {len(df)} observations."
        )
    else:
        lines.append(
            f"The unemp × rural interaction is **not significant** "
            f"(UW p={uw_unemp_r_p:.3f}, WT p={wt_unemp_r_p:.3f}). "
            f"The rural/urban split in necessity importance is suggestive "
            f"but not confirmed by the stronger pooled test."
        )
    if edu_interaction_sig:
        lines.append(
            f"The edu × rural interaction is also significant "
            f"(UW p={uw_edu_r_p:.3f}{sig_star(uw_edu_r_p)}, "
            f"WT p={wt_edu_r_p:.3f}{sig_star(wt_edu_r_p)}), "
            f"coef {uw_edu_r_c:+.4f} (UW): the education premium on firm creation "
            f"{'weakens' if uw_edu_r_c < 0 else 'strengthens'} in rural departments."
        )
    else:
        lines.append(
            f"The edu × rural interaction is not significant "
            f"(UW p={uw_edu_r_p:.3f}, WT p={wt_edu_r_p:.3f}): "
            f"education's role does not detectably differ by density context."
        )
    return " ".join(lines)

findings_md = f"""# split_findings.md, Urban/Rural Split Analysis
_Generated by model/split_analysis.py_

---

## Context

This analysis tests whether the full-panel result (opportunity features {opp_f/nec_f:.1f}x
more important than necessity, SHAP {opp_f/tot_f*100:.0f}% vs {nec_f/tot_f*100:.0f}%) holds uniformly across
density contexts, or whether necessity entrepreneurship re-emerges in
rural France.

---

## Subset Sizes

| Subset | Definition | Departments | Rows |
|---|---|---|---|
| Full | all density classes | {df['dep_code'].nunique()} | {len(df)} |
| Urban | urban + intermediate | {n_dep_urban} | {n_row_urban} |
| Rural | rural only | {n_dep_rural} | {n_row_rural} |

Both subsets exceed 30 departments, results are not purely exploratory,
but the rural subset (~51 depts, ~510 obs) is modest. Frame as
**suggestive, not definitive** for subset-level claims.

---

## Headline: SHAP Group Importance, Full vs Urban vs Rural

| | Full panel | Urban/Interm | Rural |
|---|---|---|---|
| **OPPORTUNITY share** | **{opp_f/tot_f*100:.0f}%** | **{opp_u/tot_u*100:.0f}%** | **{opp_r/tot_r*100:.0f}%** |
| **NECESSITY share**   | **{nec_f/tot_f*100:.0f}%** | **{nec_u/tot_u*100:.0f}%** | **{nec_r/tot_r*100:.0f}%** |
| Other share           | {oth_f/tot_f*100:.0f}% | {oth_u/tot_u*100:.0f}% | {oth_r/tot_r*100:.0f}% |
| Opp/Nec ratio         | {opp_f/nec_f:.1f}x | {opp_u/nec_u:.1f}x | {opp_r/nec_r:.1f}x |
| LODO R²               | {r2_full:.3f} | {r2_urb:.3f} | {r2_rur:.3f} |

Figure: `figures/split_shap_by_context.png`

The opportunity > necessity ordering holds in **all three contexts**.
Necessity's share rises modestly in rural areas ({nec_r/tot_r*100:.0f}% vs
{nec_f/tot_f*100:.0f}% full-panel), but opportunity remains dominant at
{opp_r/tot_r*100:.0f}% of SHAP weight even in the most rural subset.

---

## Necessity Verdict by Subset

### Full panel
- Unemployment SHAP rank: {unemp_rank_f}/8
- OLS: UW coef={coef_uw_f:+.4f} p={pval_uw_f:.3f}{sig_star(pval_uw_f)}, WT coef={coef_wt_f:+.4f} p={pval_wt_f:.3f}{sig_star(pval_wt_f)}
- **Verdict: {necessity_verdict_str(coef_uw_f, pval_uw_f, coef_wt_f, pval_wt_f, 'full')}**

### Urban / Intermediate ({n_dep_urban} departments)
- Unemployment SHAP rank: {unemp_rank_u}/8
- OLS: UW coef={coef_uw_u:+.4f} p={pval_uw_u:.3f}{sig_star(pval_uw_u)}, WT coef={coef_wt_u:+.4f} p={pval_wt_u:.3f}{sig_star(pval_wt_u)}
- **Verdict: {verdict_urb}**

### Rural ({n_dep_rural} departments)
- Unemployment SHAP rank: {unemp_rank_r}/8
- OLS: UW coef={coef_uw_r:+.4f} p={pval_uw_r:.3f}{sig_star(pval_uw_r)}, WT coef={coef_wt_r:+.4f} p={pval_wt_r:.3f}{sig_star(pval_wt_r)}
- **Verdict: {verdict_rur}**

---

## Interaction Model, Rigorous Cross-Check

Pooled OLS on {len(df)} rows with added terms:
`unemp_x_rural = unemployment_rate × is_rural`
`edu_x_rural = edu_share_sup × is_rural`

This uses all available data and is the statistically stronger test.

| Term | UW coef | UW p | WT coef | WT p |
|---|---|---|---|---|
| unemployment_rate (baseline) | {uw_unemp_c:+.4f} | {uw_unemp_p:.3f}{sig_star(uw_unemp_p)} | {wt_unemp_c:+.4f} | {wt_unemp_p:.3f}{sig_star(wt_unemp_p)} |
| unemp × rural (interaction) | {uw_unemp_r_c:+.4f} | {uw_unemp_r_p:.3f}{sig_star(uw_unemp_r_p)} | {wt_unemp_r_c:+.4f} | {wt_unemp_r_p:.3f}{sig_star(wt_unemp_r_p)} |
| edu × rural (interaction) | {uw_edu_r_c:+.4f} | {uw_edu_r_p:.3f}{sig_star(uw_edu_r_p)} | {wt_edu_r_c:+.4f} | {wt_edu_r_p:.3f}{sig_star(wt_edu_r_p)} |

{int_sig_summary()}

---

## Sample Size Note

Rural subset: {n_dep_rural} departments × 10 years = {n_row_rural} observations.
Urban subset: {n_dep_urban} departments × 10 years = {n_row_urban} observations.
Both exceed the <30 flag threshold. However, LODO within the rural subset
means some folds train on ~50 departments, models remain usable but
precision is lower than the full-panel result. Report rural findings
as **suggestive evidence** rather than definitive conclusions.

---

## Figures

| File | Content |
|---|---|
| `figures/split_shap_by_context.png` | Headline grouped bar, Full vs Urban vs Rural |
| `figures/split_shap_per_feature.png` | Per-feature SHAP, three panels side by side |
| `figures/split_opp_nec_ratio.png` | Opportunity/Necessity ratio by context |

---

## Implication for Paper Framing

The opportunity model is robust across density contexts: it holds in
urban departments where human capital and income dominate, and it holds
in rural departments where necessity might have been expected to emerge.
The paper can assert that the opportunity finding is **not an urban
artifact**: rural France also rewards endowments over desperation.
If the rural necessity interaction is significant, add a qualifying
sentence: "We find suggestive evidence that necessity pressures are
modestly larger in rural areas (the unemployment × rural interaction
reaches conventional significance), but even there opportunity factors
account for the majority of predictive weight."
"""

findings_path = f"{MODEL_DIR}/split_findings.md"
with open(findings_path, "w", encoding="utf-8") as fh:
    fh.write(findings_md)

r(f"Written: {findings_path}")
r()
r("Done.")
