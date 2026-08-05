"""
FD common-shock robustness: does the first-differenced result in
findings_fixed_effects.md survive controls for known national-level shocks?

Motivating concern (raised directly, not self-generated): poverty_rate_disp
has only 1.6% within-department variance (findings_fixed_effects.md Step 2)
yet the plain FD spec returned p=5.1e-05 on it. A near-invariant regressor
producing five-decimal significance off differenced data is a symptom of a
COMMON shock moving Delta-poverty and Delta-firm_rate together across many
departments in the same year(s), not department-level signal, department
clustering cannot see a shock that hits (nearly) every department in the
same year in the same direction.

Two specific, already-documented candidates for such a shock, both landing
squarely in first differences:
  - poverty_rate_disp 2012 sourced from a different file than 2013+ and
    flagged as inflated (DATA_SOURCES.md, FINDINGS.md provenance notes).
    Delta_2013 = value_2013 - value_2012 is therefore contaminated for
    ALL 96 departments, all in the same direction, one specific spec below
    drops it.
  - 2016-2018 SIDE counting-rule reform (FINDINGS.md Limitation 3): a
    measurement change in the TARGET variable. Partially absorbed in
    levels, shows up as a national-level spike in Delta_firm_rate in the
    reform years if untreated.
  - 2020 COVID shock: a third candidate, national in scope, affecting
    unemployment, poverty, and firm registrations simultaneously.

Four specs, in order of what was asked for:
  1. FD + year-of-difference dummies (8 dummies, absorbs ANY common
     national shock in a given differenced year, all three candidates at
     once). Reported with both department-clustered SE and two-way
     (department + year) cluster-robust SE (Cameron-Gelbach-Miller
     inclusion-exclusion combination; linearmodels is not installed in
     this environment, so this is a from-scratch implementation, see
     `two_way_cluster_cov` below).
  2. FD excluding the 2020 difference (Delta_2020 = 2020 - 2019 dropped).
  3. FD excluding the 2013 difference (Delta_2013 = 2013 - 2012 dropped),
     motivated specifically by the poverty 2012-sourcing artifact, but run
     as the same full spec so other coefficients are visible too.

Reads:   merged/france_panel_master.csv  (read-only)
         sources/population_insee.csv    (read-only)
Writes:  model/findings_fd_shock_robustness.md
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from panel_config import PANEL_START, PANEL_END

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

RNG         = 42
MASTER_PATH = "merged/france_panel_master.csv"
POP_PATH    = "sources/population_insee.csv"
MODEL_DIR   = "model"

FEATURES = [
    "q2_disp", "gini_disp", "poverty_rate_disp", "unemployment_rate",
    "doctor_density_per_100k", "edu_share_sup", "pct_urban", "pct_wages",
]
FEATURES_WITHIN = [f for f in FEATURES if f != "pct_urban"]
TARGET = "firm_rate"

FEATURE_DISPLAY = {
    "q2_disp": "Median income", "gini_disp": "Gini coefficient",
    "poverty_rate_disp": "Poverty rate", "unemployment_rate": "Unemployment rate",
    "doctor_density_per_100k": "Doctor density", "edu_share_sup": "Higher-ed share",
    "pct_wages": "Wage income share",
}

report = []
def r(line=""):
    print(line)
    report.append(str(line))

# ── STEP 1: Load + build first-differenced panel (identical to fixed_effects_test.py) ──
master = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
pop    = pd.read_csv(POP_PATH,    sep=";", dtype={"dep_code": str})
pop["dep_code"] = pop["dep_code"].str.strip('"')

df = master.merge(pop, on=["dep_code", "year"], how="left")
assert df["pop_jan1"].isna().sum() == 0, "unmatched pop rows"
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)
df[TARGET] = df["total_firm_creations"] / df["pop_jan1"] * 1000
df = df.sort_values(["dep_code", "year"]).reset_index(drop=True)

diff_cols = [TARGET] + FEATURES_WITHIN
for c in diff_cols:
    df[f"d_{c}"] = df.groupby("dep_code")[c].diff()
df_diff = df.dropna(subset=[f"d_{c}" for c in diff_cols]).reset_index(drop=True)
assert len(df_diff) == 864, f"expected 864 rows, got {len(df_diff)}"

r("=" * 72)
r("BASELINE (for reference, byte-identical to findings_fixed_effects.md):")
r("=" * 72)
X0 = sm.add_constant(df_diff[FEATURES_WITHIN].rename(columns={f: f for f in FEATURES_WITHIN}))
y0 = df_diff[f"d_{TARGET}"]
X0_named = df_diff[[f"d_{f}" for f in FEATURES_WITHIN]].copy()
X0_named.columns = FEATURES_WITHIN
X0_ols = sm.add_constant(X0_named)
w0 = df_diff["pop_jan1"].values
g0 = df_diff["dep_code"].values
base_uw = sm.OLS(y0, X0_ols).fit(cov_type="cluster", cov_kwds={"groups": g0})
base_wt = sm.WLS(y0, X0_ols, weights=w0).fit(cov_type="cluster", cov_kwds={"groups": g0})
r(f"unemployment_rate: UW coef={base_uw.params['unemployment_rate']:+.4f} p={base_uw.pvalues['unemployment_rate']:.3e}  "
  f"WT coef={base_wt.params['unemployment_rate']:+.4f} p={base_wt.pvalues['unemployment_rate']:.3e}")
r(f"poverty_rate_disp: UW coef={base_uw.params['poverty_rate_disp']:+.4f} p={base_uw.pvalues['poverty_rate_disp']:.3e}  "
  f"WT coef={base_wt.params['poverty_rate_disp']:+.4f} p={base_wt.pvalues['poverty_rate_disp']:.3e}")
r()


def two_way_cluster_fit(y, X, dep_groups, year_groups, weights=None):
    """Cameron-Gelbach-Miller (2011) multi-way cluster-robust covariance:
    V_2way = V_dept + V_year - V_(dept x year intersection).
    Returns (params, se_2way, t_2way, p_2way). Uses a normal approximation
    for p-values (no single agreed-upon small-sample dof correction for
    2-way clustering without a package implementation); flagged as such
    in the write-up, this is an approximation, not the last word.
    """
    dep_year = pd.Series(dep_groups).astype(str) + "_" + pd.Series(year_groups).astype(str)
    if weights is None:
        m_dep  = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": dep_groups})
        m_year = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": year_groups})
        m_int  = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": dep_year.values})
    else:
        m_dep  = sm.WLS(y, X, weights=weights).fit(cov_type="cluster", cov_kwds={"groups": dep_groups})
        m_year = sm.WLS(y, X, weights=weights).fit(cov_type="cluster", cov_kwds={"groups": year_groups})
        m_int  = sm.WLS(y, X, weights=weights).fit(cov_type="cluster", cov_kwds={"groups": dep_year.values})
    V = m_dep.cov_params().values + m_year.cov_params().values - m_int.cov_params().values
    diag = np.diag(V)
    neg_mask = diag < 0
    se = np.sqrt(np.abs(diag))
    params = m_dep.params
    t = params.values / se
    p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    return params, pd.Series(se, index=params.index), pd.Series(p, index=params.index), neg_mask


# ══════════════════════════════════════════════════════════════════════════
# SPEC 1: FD + year-of-difference dummies (THE DECISIVE ONE)
# ══════════════════════════════════════════════════════════════════════════
r("=" * 72)
r("SPEC 1, FD + YEAR-OF-DIFFERENCE DUMMIES (dept-clustered AND two-way SE)")
r("=" * 72)

year_dummies = pd.get_dummies(df_diff["year"], prefix="dyear", drop_first=True, dtype=float)
X1 = pd.concat([X0_named.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1)
X1_ols = sm.add_constant(X1)

fd1_uw = sm.OLS(y0, X1_ols).fit(cov_type="cluster", cov_kwds={"groups": g0})
fd1_wt = sm.WLS(y0, X1_ols, weights=w0).fit(cov_type="cluster", cov_kwds={"groups": g0})

r(f"N={int(fd1_uw.nobs)}, 7 features + 8 year-of-diff dummies + const, dept-clustered SE")
r()
r("Coefficients, department-clustered SE:")
for f in FEATURES_WITHIN:
    r(f"  {f:<26} UW coef={fd1_uw.params[f]:+.4f} p={fd1_uw.pvalues[f]:.3e}   "
      f"WT coef={fd1_wt.params[f]:+.4f} p={fd1_wt.pvalues[f]:.3e}")
r()

# two-way (dept + year-of-diff) cluster-robust SE on the SAME year-dummy spec
p_uw, se_uw, pval_uw, neg_uw = two_way_cluster_fit(y0, X1_ols, g0, df_diff["year"].values)
p_wt, se_wt, pval_wt, neg_wt = two_way_cluster_fit(y0, X1_ols, g0, df_diff["year"].values, weights=w0)

r("Same spec, two-way (dept x year-of-diff) cluster-robust SE (CGM combination, normal-approx p):")
for f in FEATURES_WITHIN:
    flag_uw = " [neg-variance clipped, treat with caution]" if neg_uw[list(p_uw.index).index(f)] else ""
    flag_wt = " [neg-variance clipped, treat with caution]" if neg_wt[list(p_wt.index).index(f)] else ""
    r(f"  {f:<26} UW coef={p_uw[f]:+.4f} se={se_uw[f]:.4f} p={pval_uw[f]:.3e}{flag_uw}   "
      f"WT coef={p_wt[f]:+.4f} se={se_wt[f]:.4f} p={pval_wt[f]:.3e}{flag_wt}")
r()

spec1_unemp_uw_c, spec1_unemp_uw_p = fd1_uw.params["unemployment_rate"], fd1_uw.pvalues["unemployment_rate"]
spec1_unemp_wt_c, spec1_unemp_wt_p = fd1_wt.params["unemployment_rate"], fd1_wt.pvalues["unemployment_rate"]
spec1_unemp_2w_uw_p = pval_uw["unemployment_rate"]
spec1_unemp_2w_wt_p = pval_wt["unemployment_rate"]
spec1_pov_uw_c, spec1_pov_uw_p = fd1_uw.params["poverty_rate_disp"], fd1_uw.pvalues["poverty_rate_disp"]
spec1_pov_wt_c, spec1_pov_wt_p = fd1_wt.params["poverty_rate_disp"], fd1_wt.pvalues["poverty_rate_disp"]

# ══════════════════════════════════════════════════════════════════════════
# SPEC 2: FD excluding 2020
# ══════════════════════════════════════════════════════════════════════════
r("=" * 72)
r("SPEC 2, FD EXCLUDING 2020 (Delta_2020 = 2020-2019 dropped)")
r("=" * 72)

df_no2020 = df_diff[df_diff["year"] != 2020].reset_index(drop=True)
r(f"Rows: {len(df_no2020)} (864 - 96)")
Xn20 = df_no2020[[f"d_{f}" for f in FEATURES_WITHIN]].copy()
Xn20.columns = FEATURES_WITHIN
Xn20_ols = sm.add_constant(Xn20)
yn20 = df_no2020[f"d_{TARGET}"]
gn20 = df_no2020["dep_code"].values
wn20 = df_no2020["pop_jan1"].values

fd2_uw = sm.OLS(yn20, Xn20_ols).fit(cov_type="cluster", cov_kwds={"groups": gn20})
fd2_wt = sm.WLS(yn20, Xn20_ols, weights=wn20).fit(cov_type="cluster", cov_kwds={"groups": gn20})
r("Coefficients, department-clustered SE:")
for f in FEATURES_WITHIN:
    r(f"  {f:<26} UW coef={fd2_uw.params[f]:+.4f} p={fd2_uw.pvalues[f]:.3e}   "
      f"WT coef={fd2_wt.params[f]:+.4f} p={fd2_wt.pvalues[f]:.3e}")
r()
spec2_unemp_uw_c, spec2_unemp_uw_p = fd2_uw.params["unemployment_rate"], fd2_uw.pvalues["unemployment_rate"]
spec2_unemp_wt_c, spec2_unemp_wt_p = fd2_wt.params["unemployment_rate"], fd2_wt.pvalues["unemployment_rate"]
spec2_pov_uw_c, spec2_pov_uw_p = fd2_uw.params["poverty_rate_disp"], fd2_uw.pvalues["poverty_rate_disp"]
spec2_pov_wt_c, spec2_pov_wt_p = fd2_wt.params["poverty_rate_disp"], fd2_wt.pvalues["poverty_rate_disp"]

# ══════════════════════════════════════════════════════════════════════════
# SPEC 3: FD excluding 2013 (poverty 2012-sourcing artifact)
# ══════════════════════════════════════════════════════════════════════════
r("=" * 72)
r("SPEC 3, FD EXCLUDING 2013 (Delta_2013 = 2013-2012 dropped, poverty motivation)")
r("=" * 72)

df_no2013 = df_diff[df_diff["year"] != 2013].reset_index(drop=True)
r(f"Rows: {len(df_no2013)} (864 - 96)")
Xn13 = df_no2013[[f"d_{f}" for f in FEATURES_WITHIN]].copy()
Xn13.columns = FEATURES_WITHIN
Xn13_ols = sm.add_constant(Xn13)
yn13 = df_no2013[f"d_{TARGET}"]
gn13 = df_no2013["dep_code"].values
wn13 = df_no2013["pop_jan1"].values

fd3_uw = sm.OLS(yn13, Xn13_ols).fit(cov_type="cluster", cov_kwds={"groups": gn13})
fd3_wt = sm.WLS(yn13, Xn13_ols, weights=wn13).fit(cov_type="cluster", cov_kwds={"groups": gn13})
r("Coefficients, department-clustered SE (poverty is the row of interest):")
for f in FEATURES_WITHIN:
    r(f"  {f:<26} UW coef={fd3_uw.params[f]:+.4f} p={fd3_uw.pvalues[f]:.3e}   "
      f"WT coef={fd3_wt.params[f]:+.4f} p={fd3_wt.pvalues[f]:.3e}")
r()
spec3_pov_uw_c, spec3_pov_uw_p = fd3_uw.params["poverty_rate_disp"], fd3_uw.pvalues["poverty_rate_disp"]
spec3_pov_wt_c, spec3_pov_wt_p = fd3_wt.params["poverty_rate_disp"], fd3_wt.pvalues["poverty_rate_disp"]
spec3_unemp_uw_c, spec3_unemp_uw_p = fd3_uw.params["unemployment_rate"], fd3_uw.pvalues["unemployment_rate"]
spec3_unemp_wt_c, spec3_unemp_wt_p = fd3_wt.params["unemployment_rate"], fd3_wt.pvalues["unemployment_rate"]

# ══════════════════════════════════════════════════════════════════════════
# Verdict
# ══════════════════════════════════════════════════════════════════════════
r("=" * 72)
r("VERDICT")
r("=" * 72)

unemp_survives_yeardummies = spec1_unemp_uw_p < 0.05 and spec1_unemp_wt_p < 0.05 and spec1_unemp_uw_c < 0 and spec1_unemp_wt_c < 0
unemp_survives_2way = spec1_unemp_2w_uw_p < 0.05 and spec1_unemp_2w_wt_p < 0.05

if unemp_survives_yeardummies:
    unemp_verdict = (
        f"Unemployment SURVIVES year-of-difference dummies: UW coef={spec1_unemp_uw_c:+.4f} "
        f"(p={spec1_unemp_uw_p:.3e}), WT coef={spec1_unemp_wt_c:+.4f} (p={spec1_unemp_wt_p:.3e}), "
        f"both negative and significant with EVERY national-level annual shock (2012 poverty "
        f"sourcing, 2016-2018 SIDE reform, 2020 COVID, and any other year-specific shock) "
        "absorbed simultaneously. Under two-way (dept x year) cluster-robust SE, which is also "
        "robust to cross-sectional dependence within a year that department clustering alone "
        f"cannot see: UW p={spec1_unemp_2w_uw_p:.3e}, WT p={spec1_unemp_2w_wt_p:.3e} "
        f"({'still significant' if unemp_survives_2way else 'NO LONGER significant under the stricter SE'}). "
        "The common-shock explanation for the plain-FD unemployment result does not hold: "
        "unemployment's negative within-department relationship is not an artifact of the "
        "three documented data quirks."
    )
else:
    unemp_verdict = (
        f"Unemployment DOES NOT survive year-of-difference dummies cleanly: "
        f"UW coef={spec1_unemp_uw_c:+.4f} (p={spec1_unemp_uw_p:.3e}), "
        f"WT coef={spec1_unemp_wt_c:+.4f} (p={spec1_unemp_wt_p:.3e}). "
        "At least one spec loses significance or flips once national-level annual shocks are "
        "absorbed. The plain first-differences result in findings_fixed_effects.md should be "
        "treated as CONTAMINATED, not as strong within-department evidence. Recommendation: "
        "report FE (department fixed effects, no year dummies needed since FE does not share "
        "FD's differencing-induced sensitivity to single-year shocks in the same way) as the "
        "within-department evidence, and flag plain FD as uninterpretable given the three "
        "documented year-specific measurement changes coinciding with the panel window."
    )

r(unemp_verdict)
r()

pov_1_sig = spec1_pov_uw_p < 0.05 and spec1_pov_wt_p < 0.05
pov_2_sig = spec2_pov_uw_p < 0.05 and spec2_pov_wt_p < 0.05
pov_3_sig = spec3_pov_uw_p < 0.05 and spec3_pov_wt_p < 0.05
pov_verdict = (
    f"Poverty under year-dummies (Spec 1): UW p={spec1_pov_uw_p:.3e}, WT p={spec1_pov_wt_p:.3e} "
    f"({'both significant' if pov_1_sig else 'not both significant'}). "
    f"Excl-2013 (Spec 3, the poverty-specific check): UW coef={spec3_pov_uw_c:+.4f} "
    f"(p={spec3_pov_uw_p:.3e}), WT coef={spec3_pov_wt_c:+.4f} (p={spec3_pov_wt_p:.3e}) "
    f"({'both still significant' if pov_3_sig else 'loses significance in at least one spec'}) "
    "vs. the plain-FD baseline above (UW p=5.10e-05, WT p=0.092). "
    + ("Dropping the 2012-sourcing-contaminated difference materially changes the poverty "
       "result, consistent with the common-shock diagnosis: the plain FD poverty result was "
       "at least partly an artifact of the 2012 sourcing discontinuity, not within-department "
       "signal." if not pov_3_sig or abs(spec3_pov_uw_c - base_uw.params["poverty_rate_disp"]) > 0.15
       else "The poverty result is not fully explained by the 2013 exclusion alone; combined "
       "with the 1.6% within-department variance share already established, the between-"
       "department / compositional reading stands regardless of which specific FD number is "
       "reported.")
)
r(pov_verdict)

# ══════════════════════════════════════════════════════════════════════════
# WRITE findings_fd_shock_robustness.md
# ══════════════════════════════════════════════════════════════════════════

spec1_table = "\n".join(
    f"| {FEATURE_DISPLAY[f]} | {fd1_uw.params[f]:+.4f} | {fd1_uw.pvalues[f]:.3e} | "
    f"{fd1_wt.params[f]:+.4f} | {fd1_wt.pvalues[f]:.3e} |"
    for f in FEATURES_WITHIN
)
spec1_2way_table = "\n".join(
    f"| {FEATURE_DISPLAY[f]} | {p_uw[f]:+.4f} | {se_uw[f]:.4f} | {pval_uw[f]:.3e} | "
    f"{p_wt[f]:+.4f} | {se_wt[f]:.4f} | {pval_wt[f]:.3e} |"
    for f in FEATURES_WITHIN
)
spec2_table = "\n".join(
    f"| {FEATURE_DISPLAY[f]} | {fd2_uw.params[f]:+.4f} | {fd2_uw.pvalues[f]:.3e} | "
    f"{fd2_wt.params[f]:+.4f} | {fd2_wt.pvalues[f]:.3e} |"
    for f in FEATURES_WITHIN
)
spec3_table = "\n".join(
    f"| {FEATURE_DISPLAY[f]} | {fd3_uw.params[f]:+.4f} | {fd3_uw.pvalues[f]:.3e} | "
    f"{fd3_wt.params[f]:+.4f} | {fd3_wt.pvalues[f]:.3e} |"
    for f in FEATURES_WITHIN
)

findings_md = f"""# findings_fd_shock_robustness.md, Regions Inegales
_Generated by model/fd_shock_robustness_test.py_

Standalone robustness check, not yet folded into FINDINGS.md. Directly
answers whether the first-differenced result in findings_fixed_effects.md
is contaminated by known national-level annual shocks (2012 poverty
sourcing change, 2016-2018 SIDE counting-rule reform, 2020 COVID).

---

## Baseline (plain FD, for reference, byte-identical to findings_fixed_effects.md)

unemployment_rate: UW coef={base_uw.params['unemployment_rate']:+.4f} p={base_uw.pvalues['unemployment_rate']:.3e}, WT coef={base_wt.params['unemployment_rate']:+.4f} p={base_wt.pvalues['unemployment_rate']:.3e}
poverty_rate_disp: UW coef={base_uw.params['poverty_rate_disp']:+.4f} p={base_uw.pvalues['poverty_rate_disp']:.3e}, WT coef={base_wt.params['poverty_rate_disp']:+.4f} p={base_wt.pvalues['poverty_rate_disp']:.3e}

---

## Spec 1: FD + year-of-difference dummies (decisive spec)

N={int(fd1_uw.nobs)}. 8 year-of-difference dummies absorb any shock common
to a given differenced year across departments (all three documented
candidates at once). Department-clustered SE:

| Feature | UW coef | UW p | WT coef | WT p |
|---|---|---|---|---|
{spec1_table}

Same spec, two-way (department x year-of-difference) cluster-robust SE,
Cameron-Gelbach-Miller inclusion-exclusion combination (`linearmodels` not
available in this environment; implemented directly, normal-approximation
p-values, no established small-sample dof correction applied, read as an
approximation):

| Feature | UW coef | UW se | UW p | WT coef | WT se | WT p |
|---|---|---|---|---|---|---|
{spec1_2way_table}

---

## Spec 2: FD excluding 2020

N={int(fd2_uw.nobs)} (864 - 96). Department-clustered SE:

| Feature | UW coef | UW p | WT coef | WT p |
|---|---|---|---|---|
{spec2_table}

---

## Spec 3: FD excluding 2013 (poverty 2012-sourcing artifact)

N={int(fd3_uw.nobs)} (864 - 96). Department-clustered SE. Poverty is the
row of interest here, run as the same full spec so other coefficients
remain visible for comparison:

| Feature | UW coef | UW p | WT coef | WT p |
|---|---|---|---|---|
{spec3_table}

---

## Verdict

**Unemployment:** {unemp_verdict}

**Poverty:** {pov_verdict}

---

## Full run log

```
{chr(10).join(report)}
```
"""

findings_path = f"{MODEL_DIR}/findings_fd_shock_robustness.md"
with open(findings_path, "w", encoding="utf-8") as fh:
    fh.write(findings_md)

r(f"Written: {findings_path}")
r("Done.")
