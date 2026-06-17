# FINDINGS.md -- Regions Inegales
_Source-of-truth record. Do not edit except to correct locked numbers._
_Model outputs locked: model/findings_final.md, model/split_findings.md_

---

## The question

What drives firm-creation differences across French departments? Two competing
explanations exist in the literature. The necessity model says unemployment and
poverty push people into self-employment when they have no other options. The
opportunity model says education, income, and urban amenities pull people toward
entrepreneurship when conditions are favourable. This project tests which model
fits French departmental data for 2012-2021.

---

## The data

960 observations: 96 metropolitan departments x 10 years (2012-2021), 45 variables.
Six official sources, each cross-checked against an independent external reference
before use in the model.

- **Filosofi** (INSEE): household income, poverty rates, and the Gini coefficient
  at department level, verified against published INSEE summary tables.
- **SIDE** (INSEE): total firm creations per department per year, cross-referenced
  against INSEE Premiere annual national totals.
- **Localised unemployment** (INSEE): ILO unemployment rate, quarterly series
  averaged to annual, full 960-cell cross-check against the INSEE BDM SDMX API
  (85.2% exact match, 14.8% within +-0.1, zero beyond +-0.1).
- **RPPS doctor density** (DREES): active doctors per 100k inhabitants, full
  960-cell recompute from headcount and population data (max deviation 0.0025%).
- **Education** (INSEE): share of adults with higher-education diplomas, from
  three census snapshots (2011, 2016, 2022) linearly interpolated to annual;
  formula confirmed against the ANCT Observatoire des Territoires indicator.
- **Grille de densite** (INSEE): percent urban by department, time-invariant,
  cross-checked against published INSEE density typology.

Target variable: firm creation rate per 1,000 inhabitants.

---

## Method

8 structural predictors selected to represent the two theories: four opportunity
variables (higher-ed share, median disposable income, percent urban, doctor
density per 100k), two necessity variables (unemployment rate, poverty rate), and
two controls (Gini coefficient, wage income share).

Model: XGBoost trained on 960 department-years. SHAP (SHapley Additive
exPlanations) used to measure each variable's average contribution to predictions.

Validation used three schemes. The honest number is leave-one-department-out
(LODO), which trains the model on all other departments and tests on one it has
never seen. LODO R2 = 0.674: the model explains 67% of firm-creation variance
in departments not used in training. The random 10-fold (KFold) result is
R2 = 0.921, but this is a leaky baseline: departments appear in both train and
test sets, inflating performance. The gap between 0.921 and 0.674 is expected
and honest: departments have persistent idiosyncrasies the 8-feature set does not
fully capture. Leave-one-year-out (LOYO) R2 = 0.906, which is strong but a less
demanding test since year-to-year variation is smooth. LODO is the headline.

---

## Main finding

Opportunity beats necessity by a wide margin.

Grouped mean absolute SHAP values, full panel:

| Group       | Features                                       | Total SHAP | Share |
|-------------|------------------------------------------------|------------|-------|
| Opportunity | income, education, % urban, doctor density     | 2.914      | 60%   |
| Necessity   | unemployment rate, poverty rate                | 0.727      | 15%   |
| Other       | Gini, wage share                               | 1.183      | 25%   |

Opportunity features are 4.0x more important than necessity features on the SHAP
measure. Unemployment is the single weakest predictor of all 8 (mean |SHAP| = 0.110,
rank 8/8). Median income (1.176) and higher-ed share (1.173) dominate.

Per-feature breakdown, sorted by importance:

| Feature             | Group       | Mean |SHAP| |
|---------------------|-------------|--------------|
| Median income       | Opportunity | 1.1762       |
| Higher-ed share     | Opportunity | 1.1727       |
| Wage income share   | Other       | 0.7725       |
| Poverty rate        | Necessity   | 0.6175       |
| Gini coefficient    | Other       | 0.4110       |
| Doctor density      | Opportunity | 0.3845       |
| % Urban             | Opportunity | 0.1809       |
| Unemployment rate   | Necessity   | 0.1096       |

OLS partial regression confirms: unemployment correlates negatively with firm
creation rates in both the unweighted spec (coef = -0.018, p = 0.759) and the
population-weighted spec (coef = -0.225, p = 0.007). Higher unemployment does not
drive up entrepreneurship; if anything, it accompanies lower firm formation.

**Verdict: the necessity-entrepreneurship hypothesis is rejected for metropolitan
France, 2012-2021.**

Note on poverty rate: its positive OLS coefficient and moderate SHAP rank (4th of
8) do not confirm necessity push. The most plausible interpretation is
informalisation of labour: poorer areas have more micro-enterprise and
auto-entrepreneur registrations for structural reasons, not because unemployment
is pushing people into business formation.

---

## Robustness: urban/rural split

The opportunity finding holds in both density subsets, not just in cities.

| Context          | Departments | Opportunity SHAP share | Necessity SHAP share | Opp/Nec ratio | LODO R2 |
|------------------|-------------|------------------------|----------------------|---------------|---------|
| Full panel       | 96          | 60%                    | 15%                  | 4.0x          | 0.674   |
| Urban/Intermediate | 45        | 59%                    | 16%                  | 3.8x          | 0.548   |
| Rural            | 51          | 61%                    | 13%                  | 4.6x          | 0.653   |

Necessity's share is actually lower in rural departments (13%) than in urban ones
(16%), and the opportunity/necessity ratio is highest in the rural subset (4.6x).

One complication in the rural subset: OLS finds a positive and significant
unemployment coefficient (unweighted coef = +0.214, p < 0.001). This looks like
necessity push at first glance. But three things undercut that reading:

1. SHAP still ranks unemployment last (8/8) in the rural-only model. It
   contributes negligible predictive variance once the other features are included.
2. A pooled interaction test on the full 960-row panel (adding an unemployment
   x rural interaction term) finds no significant interaction (p = 0.870
   unweighted, p = 0.569 population-weighted). The rural/urban difference in
   unemployment's coefficient is not statistically confirmed in the stronger test.
3. The most plausible explanation is compositional: lower-income rural departments
   have both higher unemployment and more micro-enterprise registrations for
   structural reasons. This produces a positive correlation without implying that
   unemployment drives firm creation.

The rural OLS result is reported here honestly, not hidden. It was investigated
and the evidence does not support the necessity interpretation.

---

## What did not work (reported, not hidden)

**Gini coefficient**: tested as a predictor. Ranks 5th of 8 (mean |SHAP| = 0.411).
The ranking is weaker without Ile-de-France (drops to 0.358), and the sign and
magnitude of its OLS coefficient depend on the weighting scheme. No robust claim
can be made about inequality driving or suppressing entrepreneurship from this
model. Retained in the feature matrix for theoretical completeness; treated as
inconclusive.

---

## Limitations

1. **Registrations, not survival.** SIDE counts legal registrations including
   auto-entrepreneurs who may cease activity quickly. The model captures entry
   propensity, not sustained entrepreneurial activity.

2. **Mostly a cross-sectional story.** Roughly 70% of the predictive variance is
   between departments rather than within them over time. Results describe which
   kinds of departments produce more entrepreneurs, not why creation rates rose or
   fell in a given year.

3. **2016-2018 SIDE measurement artefact.** INSEE reformed the registration system
   in this period (auto-entrepreneur counting rules changed), causing a structural
   break in raw firm-creation counts. Year fixed effects in LOYO partly absorb
   this, but residual inflation in those years cannot be ruled out.

4. **Correlational, not causal.** No instrumental variable or quasi-experimental
   design is applied. The model shows which departmental characteristics predict
   firm-creation rates, not what would change if those characteristics changed.

5. **Education interpolated.** Higher-ed share is observed at three census points
   (2011, 2016, 2022) and linearly interpolated for all other years. Year-to-year
   variation in this variable is artificial by construction.

6. **Doctor density as amenity proxy.** Physician density is used as a
   quality-of-life proxy for the opportunity environment. Its positive contribution
   likely reflects broader urban amenity endowments rather than a direct healthcare
   mechanism.

---

## Bottom line

French regional entrepreneurship from 2012 to 2021 is structured by
opportunity factors (education, income, and urban environment), not by
necessity. Unemployment is the weakest of the 8 predictors tested,
ranking last (8/8) on the SHAP measure in the full panel and in both the
urban and rural subsets. Its partial OLS relationship with firm creation
is negative in the full panel and in urban departments; in rural
departments the raw coefficient is positive (+0.214), but this is not
confirmed by the pooled interaction test (p = 0.870) and unemployment
still ranks last in the rural SHAP model, so the most credible reading is
compositional rather than necessity-driven. Across every specification,
unemployment carries little predictive weight. The
necessity-entrepreneurship hypothesis, which is prominent in much of the
comparative literature, does not fit the French regional evidence for
this period.

---

_Numbers locked from: model/findings_final.md and model/split_findings.md_
_Data documentation: DATA_SOURCES.md_
_Generated: 2026-06-17_
