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

960 observations: 96 metropolitan departments x 10 years (2012-2021), 51 variables.
Nine official sources, each cross-checked against an independent external reference
before use in the model.

- **Filosofi** (INSEE): household income, poverty rates, and the Gini coefficient
  at department level, verified against published INSEE summary tables.
- **SIDE** (INSEE): total firm creations per department per year, cross-referenced
  against INSEE Premiere annual national totals.
- **Localised unemployment** (INSEE): ILO unemployment rate, quarterly series
  averaged to annual, full cross-check against the INSEE BDM SDMX API
  (85.2% exact match, 14.8% within +-0.1, zero beyond +-0.1).
- **RPPS doctor density** (DREES): active doctors per 100k inhabitants, full
  recompute from headcount and population data (max deviation 0.0025%).
- **Education** (INSEE): share of adults with higher-education diplomas, from
  three census snapshots (2011, 2016, 2022) linearly interpolated to annual;
  formula confirmed against the ANCT Observatoire des Territoires indicator.
- **Grille de densite** (INSEE): percent urban by department, time-invariant,
  cross-checked against published INSEE density typology.
- **Live births** (INSEE DS_NAISSANCES_FECONDITE_SERIES): live births at place of
  residence per department per year; verified 960/960 pairs, 0 nulls.
- **Deaths** (INSEE DS_ETAT_CIVIL_DECES_COMMUNES): deaths per department per year;
  verified 960/960 pairs, 0 nulls.
- **Marriages** (INSEE DEP6 annual files): total marriages per department per year,
  2012-2021; verified 960/960 pairs, national totals consistent with published figures.

Target variable: firm creation rate per 1,000 inhabitants.

---

## Method

8 structural predictors selected to represent the two theories: four opportunity
variables (higher-ed share, median disposable income, percent urban, doctor
density per 100k), two necessity variables (unemployment rate, poverty rate), and
two controls (Gini coefficient, wage income share).

**Higher-ed share is 90% interpolated.** `edu_share_sup` is observed at only
three census snapshots (2011, 2016, 2022); within the 2012-2021 panel, only
year 2016 (96 of 960 cells) is a real observation, the other 864 cells (9 of
10 years) are linearly interpolated between anchors. Linear interpolation
between two points per department produces a value that is close to a
straight-line function of department identity, i.e. close to a
cross-sectional variable rather than a genuinely time-varying one. This
matters because `edu_share_sup` is one of the two features carrying the
headline result (second-highest SHAP, 1.0505, of 8 features) in a panel
where ~70% of predictive variance is already between-department rather
than within-department (Limitation 2). The cross-sectional reading of the
main finding ("higher-education departments create more firms") is not
weakened by this. The temporal reading ("rising education raises firm
creation") should be treated with more caution than the SHAP number alone
implies, there is little genuine year-to-year variation in this variable
for the model to have learned that pattern from. See Limitation 5.

Model: XGBoost trained on 960 department-years. SHAP (SHapley Additive
exPlanations) used to measure each variable's average contribution to predictions.

Validation used three schemes. The honest number is leave-one-department-out
(LODO), which trains the model on all other departments and tests on one it has
never seen. LODO R2 = 0.678: the model explains 68% of firm-creation variance
in departments not used in training. The random 10-fold (KFold) result is
R2 = 0.932, but this is a leaky baseline: departments appear in both train and
test sets, inflating performance. The gap between 0.932 and 0.678 is expected
and honest: departments have persistent idiosyncrasies the 8-feature set does not
fully capture. Leave-one-year-out (LOYO) R2 = 0.929, which is inflated by cross-sectional overlap
with training departments: all 96 departments appear in training from the eight
non-held-out years, so LOYO tests temporal extrapolation for known units, not
generalization to new units. LODO is the headline.

---

## Main finding

Income and human capital dominate predicted firm-creation rates; the necessity/unemployment-push channel is rejected.

Grouped mean absolute SHAP values, full panel (OOF):

| Group       | Features                                       | Total SHAP | Share |
|-------------|------------------------------------------------|------------|-------|
| Opportunity | income, education, % urban, doctor density     | 2.7463     | 58%   |
| Necessity   | unemployment rate, poverty rate                | 0.9236     | 20%   |
| Other       | Gini, wage share                               | 1.0426     | 22%   |

Within the opportunity group, median income (1.114) and higher-ed share (1.051)
together account for 2.165 of the 2.746 total opportunity SHAP (79%). Doctor
density (0.396) and percent urban (0.186) make up the remaining 21%. The
accurate reading is that income and human capital dominate the opportunity
signal, not that urban amenities are doing comparable work under the same
label. Doctor density is a quality-of-life proxy for the opportunity
environment here, not a direct healthcare mechanism (see Limitation 8).
Unemployment is the single weakest predictor of all 8 (mean |SHAP| = 0.185,
rank 8/8).

Per-feature breakdown, sorted by importance (OOF mean |SHAP|):

| Feature             | Group       | Mean |SHAP| |
|---------------------|-------------|--------------|
| Median income       | Opportunity | 1.1140       |
| Higher-ed share     | Opportunity | 1.0505       |
| Poverty rate        | Necessity   | 0.7383       |
| Wage income share   | Other       | 0.6329       |
| Gini coefficient    | Other       | 0.4096       |
| Doctor density      | Opportunity | 0.3957       |
| % Urban             | Opportunity | 0.1860       |
| Unemployment rate   | Necessity   | 0.1853       |

OLS partial regression (department-clustered SE) confirms: unemployment correlates
negatively with firm creation rates in both the unweighted spec (coef = -0.304,
p = 0.044) and strongly so in the population-weighted spec (coef = -0.660,
p = 0.001). Higher unemployment does not drive up entrepreneurship; it accompanies
lower firm formation. (A lagged-unemployment robustness check bearing on the
reverse-causality reading of this coefficient is summarized in Limitation 10.)

The unemployment/poverty rows above come from a regression that already
includes all 8 locked features; the full coefficient table (department-clustered
SE, same 960-row sample) is:

| Feature | Group | UW coef | UW p | WT coef | WT p |
|---|---|---|---|---|---|
| Median income | Opportunity | +0.0009 | 8.69e-08 | +0.0011 | 4.66e-05 |
| Gini coefficient | Other | +6.2095 | 0.551 | -11.9444 | 0.409 |
| Poverty rate | Necessity | +0.5968 | 2.77e-09 | +0.8593 | 2.98e-12 |
| Unemployment rate | Necessity | -0.3038 | 0.044 | -0.6599 | 7.40e-04 |
| Doctor density | Opportunity | +0.0058 | 0.182 | +0.0113 | 0.035 |
| Higher-ed share | Opportunity | +0.2041 | 0.005 | +0.1579 | 0.063 |
| % Urban | Opportunity | +0.0210 | 0.290 | +0.0160 | 0.527 |
| Wage income share | Other | -0.1531 | 0.006 | -0.1482 | 0.036 |

R² (UW) = 0.7637, R² (WT) = 0.8188, N = 960. Full breakdown:
model/findings_final.md. % urban carries real SHAP weight but is not
significant in either OLS spec: it is exactly time-invariant (Limitation
6), so under department-clustered errors it effectively has 96
observations, not 960, and the clustered SE correctly reports that its
between-department variance is not distinguishable from noise across 96
units, the tree, unconstrained by that clustering correction, is free to
exploit the same variance directly, so SHAP and OLS are reading the same
signal two different, both honest, ways, not disagreeing. This also
sharpens the opportunity-group story: of the four opportunity features,
only income and education carry weight that survives clustered-SE
scrutiny, the 58/20 group-level split was never resting on % urban or
doctor density (see the 79% income+education share noted in the SHAP
breakdown above) and is not weakened by this. Gini stays non-significant
in both specs, consistent with the "inconclusive" read already given to
it elsewhere in this document.

Unemployment was tested four ways, not just on the main model, and fails the
necessity signature every time (model/findings_informalisation.md):

1. **Volume** (main model, full panel): SHAP rank 8/8 (mean |SHAP| = 0.185);
   OLS coefficient negative, unweighted (-0.304, p = 0.044) and pop-weighted
   (-0.660, p = 0.001).
2. **Composition** (individual/micro registration share): SHAP rank 6/8; OLS
   coefficient not significant, unweighted (+0.00294, p = 0.158) or
   pop-weighted (+0.00009, p = 0.969). Unemployment does not predict a
   higher individual/micro share.
3. **Per-capita individual registrations** (per 100k population): SHAP rank
   7/8; OLS coefficient negative, unweighted (-18.51, p = 0.079) and
   pop-weighted (-45.40, p = 0.0010).
4. **Per-capita company (SARL/SAS) registrations**: SHAP rank 8/8; OLS
   coefficient negative, unweighted (-11.57, p = 0.024) and pop-weighted
   (-20.79, p = 0.0027).

Across volume, composition, and both per-capita components, unemployment never
shows the positive, significant signature the necessity model predicts.

**Verdict: the necessity-entrepreneurship hypothesis is rejected for metropolitan
France, 2012-2021.**

Note on poverty rate: its positive OLS coefficient and moderate SHAP rank (3rd of
8) do not confirm necessity push on their own, and the informalisation-of-labour
explanation previously asserted here has now been tested directly
(model/findings_informalisation.md). Poverty predicts a higher individual/micro
registration share (SHAP rank 2/8 on individual_share; OLS coef +0.00255,
p = 0.037 unweighted, +0.00331, p = 0.0072 pop-weighted). The per-capita
decomposition shows poverty raises individual per-capita registrations more
strongly, and higher-ranked in SHAP (rank 3/8; coef +44.99, p = 1.5e-12
unweighted, +63.30, p = 1.3e-15 pop-weighted), than company per-capita
registrations (rank 5/8; coef +14.75, p = 2.3e-04 unweighted, +22.29,
p = 2.1e-06 pop-weighted). But company (SARL/SAS) formation also rises
significantly with poverty in both specifications, so informalisation does not
fully account for the effect: the overall share-level test is MIXED/INCONCLUSIVE,
because unemployment does not show the same signature (share SHAP rank 6/8, OLS
not significant, see above). The mechanism behind poverty's positive coefficient
remains partially unresolved.

**Poverty's positive coefficient is a between-department pattern, not a
within-department one, and the data cannot really speak to the latter.**
Poverty's within-department variance is 1.6% of its total variance
(model/findings_fixed_effects.md), the lowest share of any of the 8
features besides doctor density and wage share. Under department fixed
effects, poverty's coefficient is not significant either weighting
(UW p=0.171, WT p=0.505, WT even flips sign), which is consistent with
that near-total lack of within-department variation, not a contradiction
of the pooled result: there is barely any real within-department movement
in poverty for a within-estimator to find a coefficient in, so an unstable,
non-significant within-department estimate is the expected outcome, not
evidence against the pooled, well-powered, cross-sectional finding.
(A first-differenced spec also exists but should not be cited here: it
turned out to be substantially driven by the 2012 poverty-sourcing
discontinuity and other year-specific shocks rather than department-level
signal, see model/findings_fd_shock_robustness.md.) Read plainly: poverty
predicts *which departments* see more registrations, not *within a
department*, that rising poverty over time raises them.

---

## Robustness: urban/rural split

The opportunity finding holds in both density subsets, not just in cities.

| Context          | Departments | Opportunity SHAP share | Necessity SHAP share | Opp/Nec ratio | LODO R2 |
|------------------|-------------|------------------------|----------------------|---------------|---------|
| Full panel       | 96          | 58%                    | 20%                  | 2.97x         | 0.678   |
| Urban/Intermediate | 45        | 54%                    | 27%                  | 1.95x         | 0.573   |
| Rural            | 51          | 61%                    | 15%                  | 4.19x         | 0.603   |

Necessity's share is lower in rural departments (15%) than in urban ones (27%),
and the opportunity/necessity ratio is highest in the rural subset (4.19x).

One complication in the rural subset: OLS finds a positive unemployment coefficient
in the rural-only specification (unweighted coef = +0.081, clustered p = 0.568;
pop-weighted coef = +0.044, clustered p = 0.761). The coefficients are positive
but well short of conventional significance after clustering standard errors by
department. Three additional pieces of evidence undercut a necessity reading:

1. SHAP still ranks unemployment last (8/8) in the rural-only model. It
   contributes negligible predictive variance once the other features are included.
2. A pooled interaction test on the full 960-row panel (adding an unemployment
   x rural interaction term) finds no significant interaction (clustered p = 0.231
   unweighted, p = 0.401 population-weighted). The rural/urban difference in
   unemployment's coefficient is not statistically confirmed in the stronger test.
3. The pattern is compositional: lower-income rural departments
   have both higher unemployment and more micro-enterprise registrations for
   structural reasons. This produces a positive correlation without implying that
   unemployment drives firm creation.

The rural OLS result is inconclusive and does not support the necessity interpretation.

---

## Robustness: does the balance shift over time?

Tested whether the necessity/opportunity balance changes across 2012-2021
using year-interaction terms on the full 960-row panel (not year-by-year
subsets, which are too small to trust). Full detail: model/temporal_findings.md.

**The two necessity features move in opposite directions, both significantly.**
Unemployment's partial relationship with firm_rate weakens over the period
(unemployment x year, UW coef = -0.083, p = 0.033; pop-weighted coef = -0.160,
p < 0.001). But poverty's partial relationship *strengthens* over the same
period, and more strongly: poverty x year, UW coef = +0.237, p < 0.001;
pop-weighted coef = +0.287, p < 0.001 -- the single most significant
interaction term in the whole test. Because poverty and unemployment are
both NECESSITY features in this project's grouping, it is not accurate to
say "the necessity channel does not strengthen over the decade": one of
its two components clearly does.

- Opportunity features also strengthen: education x year (UW coef = +0.071,
  p = 0.077; pop-weighted coef = +0.146, p < 0.001) and income x year
  (UW coef = +0.295, p < 0.001; pop-weighted coef = +0.175, p < 0.001), both
  positive, income significant in both specs.
- A pre/post test (2012-2014 vs 2019-2021, dropping 2015-2018 for clean
  separation from the SIDE-affected years) finds *both* unemployment x late
  (UW coef = +0.356, p = 0.049; pop-weighted coef = +0.720, p = 0.040) and
  edu x late (UW coef = +1.764, p < 0.001; pop-weighted coef = +2.074,
  p < 0.001) significant and positive. Poverty was not included as an
  interaction term in this specific test, so it cannot corroborate or
  contradict the poverty-strengthens result above. The late-period intercept
  shift (firm_rate +1.59 WT, p < 0.001) reflects the general secular rise in
  registrations.

**What holds:** the opportunity > necessity SHAP-share ordering does not
flip in any of the 10 years (Test 3, descriptive/noisy, 96 rows per
year-model). **What does not hold:** the claim that the necessity channel
as a whole is flat over time. Poverty's rising partial association with
firm_rate over 2012-2021 is a genuine, statistically significant pattern
in this data and is not resolved by this project's existing robustness
battery; it is flagged here rather than folded into the unemployment
result. It should be read alongside the poverty mechanism already flagged
as unresolved in the Main finding above (informalisation test: MIXED /
PARTIAL, not a clean confirmation).

---

## Robustness: additional diagnostics

Two further checks were run (model/findings_diagnostics.md):

- **log(population) control on doctor_density.** Adding log(pop) as a ninth
  feature does not change doctor_density's SHAP rank: 6th of 8 without the
  control, 6th of 9 with it (mean |SHAP| 0.396 without vs 0.324 with). Doctor
  density's importance does not appear to be a population-size proxy in
  disguise. LODO R2 with the added control: 0.646.
- **Dropping 2012** (the year with a structural poverty_rate_dec gap).
  Removing 2012 (864 rows remain) changes no coefficient sign and no SHAP
  rank: unemployment stays 8/8 (LODO R2 without 2012: 0.688, vs 0.678 full
  panel), poverty and unemployment OLS signs are unchanged, and the
  opportunity/necessity SHAP shares stay materially the same (60%/20% without
  2012 vs 58%/20% full panel; ratio 3.05x vs 2.97x).

---

## Robustness: does a simpler model find the same thing?

Reviewer question: would a plain linear model, with no tree structure and no
SHAP, reach the same conclusion? Four models (ElasticNetCV, RandomForest,
LightGBM, XGBoost) were trained on the identical 8-feature locked matrix,
identical target, and identical LODO folds (GroupKFold, 96 splits, grouped
on dep_code) used throughout this document (model/model_comparison.py; full
breakdown in model/findings_model_comparison.md).

**Tuning budget is not identical across model families, stated plainly
rather than left implicit.** ElasticNetCV gets its own native exhaustive
alpha/l1_ratio path search over its 2 hyperparameters, standard practice
for that estimator. The three tree models (RandomForest, LightGBM,
XGBoost) each get a `RandomizedSearchCV` budget of `n_iter=20` draws from a
36-point grid (3 max_depth × 4 learning_rate × 3 n_estimators, or the
RandomForest equivalent), i.e. roughly 56% of their own grid, sampled per
outer fold from train-only inner CV. This budget was raised from an initial
n_iter=4 (~11% of the grid) specifically to test whether a wider search
would close the R2 gap seen at the smaller budget; running it across all 96
outer LODO folds for three tree models took several hours of active compute
(Reproducibility section, model/findings_model_comparison.md), confirming
the original tractability concern that kept the budget small in the first
place. The remaining asymmetry with ElasticNetCV's exhaustive
2-hyperparameter search is smaller now but not zero: the tree models'
reported LODO R2 is still a lower bound on what a fully exhaustive search
might reach, not their ceiling. What the asymmetry does not touch: the
qualitative result below (feature ranking, unemployment's bottom-half
position, cross-model agreement) is identical between the n_iter=4 and
n_iter=20 runs.

Stated plainly: ElasticNetCV still generalizes marginally better than tuned
XGBoost on these folds, but the gap narrowed sharply with the larger tuning
budget. LODO R2 = 0.7142 for ElasticNetCV versus 0.7014 for tuned XGBoost
(gap -0.0128). That gap now falls within this document's own 0.03 "mostly
linear" tolerance band, so
the four-model comparison script classifies this as MOSTLY LINEAR rather
than the earlier LINEAR OUTPERFORMS. XGBoost is not the best-generalizing
model on this panel for raw predictive accuracy, but it is close.

XGBoost remains the headline model in this document anyway, for three
reasons. First, the feature-attribution story throughout this document is
built on tree-based SHAP; moving the headline to a linear model would mean
rebuilding that attribution machinery from scratch, not swapping one line of
code. Second, the full robustness battery above (urban/rural split,
Ile-de-France drop, log-population control, dropping 2012, the temporal
interaction tests) was run against the XGBoost pipeline; re-deriving all of
it for ElasticNet is out of scope for this project. Third, what this
document actually claims is that opportunity factors dominate and
unemployment is weak, not that XGBoost is the best predictive model, and
that qualitative claim is what the four-model comparison actually bears on.

That comparison is the real robustness evidence, and it points the other
way from the R2 gap: all four models, including the linear one, agree on
the same top-3 features (higher-ed share, median income, poverty rate),
unemployment lands in the bottom half of importance for all four, and the
minimum pairwise Spearman rank correlation across all six model pairs is
+0.81. A finding that survives a switch from gradient-boosted trees to a
plain linear model is better supported than one that only appears under one
model family. That agreement, not the R2 comparison, is what should
reassure a reader that the opportunity-over-necessity result is not an
XGBoost artefact.

Full breakdown, including per-model tuning grids and the rank matrix:
model/findings_model_comparison.md.

---

## What did not work

**Gini coefficient**: tested as a predictor. Ranks 5th of 8 (mean |SHAP| = 0.410).
The ranking is weaker without Ile-de-France (drops further), and the sign and
magnitude of its OLS coefficient depend on the weighting scheme. No robust claim
can be made about inequality driving or suppressing entrepreneurship from this
model. It remains in the feature matrix; the result is inconclusive.

---

## Limitations

1. **Registrations, not survival or net growth.** SIDE counts legal
   registrations, including auto-entrepreneurs who may cease activity
   quickly. The model captures entry propensity, not survival or net
   business growth. High-poverty departments could partly reflect higher
   business turnover (more entries and more exits) rather than durable
   growth in entrepreneurial activity; this data cannot distinguish the two,
   which is part of why the poverty mechanism (see Main finding) remains
   unresolved.

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

5. **Education interpolated against a post-panel anchor.** The higher
   education share is not observed annually. It is linearly interpolated
   between three census anchors (2011, 2016 and 2022), so 864 of 960
   department-year values are constructed and only the 96 values for 2016
   are direct observations. Interpolation removes idiosyncratic year-to-year
   variation, leaving a variable that is close to a department-level mean
   plus a smooth trend. Its contribution should therefore be read as
   cross-sectional, consistent with the paper's framing throughout, and not
   as evidence that changes in local education drive changes in firm
   creation. The 2022 anchor also lies outside the panel window, so values
   from 2017 onward embed information from beyond the observation period;
   the leave-one-year-out 2021 fold is contaminated on this account.
   Quantified: full-panel LOYO R² = 0.929; recomputed over the same
   out-of-fold predictions with the 2021 fold excluded, LOYO R² = 0.936
   (MAE 0.6714 vs 0.7564 full, model/findings_final.md). The excl.-2021
   number is not lower than the full figure, if anything marginally
   higher, so the 2021 fold does not appear to be an outlier that is
   inflating the reported LOYO score. This does not clear the underlying
   concern (2017-2021 values still embed post-panel information by
   construction), but it means the contamination is not visibly showing
   up as an anomalously easy test fold. LOYO is not this project's
   headline validation scheme in any case (LODO is).

6. **pct_urban is a single-vintage time-invariant classification.** The density
   classification (Grille de densité, RP2021/2025 vintage) is applied uniformly
   across all years and contributes only cross-sectional signal. It is a forward
   look-ahead for 2012–2020.

7. **pct_wages uses a different income concept from the other income variables.**
   pct_wages is derived from the DEC income concept, while q2_disp, gini_disp,
   and poverty_rate_disp use the DISP (disposable income) concept.

8. **Doctor density as amenity proxy.** Physician density is used as a
   quality-of-life proxy for the opportunity environment. Its positive contribution
   captures broader urban amenity endowments, not a direct healthcare effect.

9. **Ecological inference.** The necessity/opportunity framework, as usually
   stated, is a claim about individual decision-making: why a given person
   chooses to register a firm. This model only has department-year
   aggregates. A department-level association between low unemployment and
   high firm-registration rates cannot distinguish "unemployment doesn't
   push individuals into entrepreneurship" from purely compositional
   explanations (e.g. departments with more already-entrepreneurial people
   sorting into low-unemployment areas for unrelated reasons). The results
   here support claims about which *kinds of departments* have higher
   registration rates; they do not, on their own, establish the individual-level
   push/pull mechanism the necessity/opportunity literature is usually about.

10. **Reverse causality / simultaneity.** The headline necessity-rejection
    argument leans on unemployment's negative partial coefficient on
    firm_rate. An alternative reading of that same negative sign is that
    causation runs the other way: departments with more firm creation have
    more hiring, which mechanically lowers local unemployment. A lagged test
    (model/findings_lagged_robustness.md) speaks to this directly: on the
    864-row 2013-2021 subset (2012 dropped for lack of a 2011 lag),
    replacing unemployment_rate(t) with unemployment_rate(t-1) in the
    identical 8-feature specification, the coefficient stayed negative and
    grew slightly larger rather than shrinking toward zero, unweighted moved
    from -0.2405 (p=0.148) same-year to -0.2961 (p=0.057) lagged, and
    population-weighted moved from -0.5917 (p=0.006) same-year to -0.6501
    (p=0.001) lagged, remaining significant.

    **This is a consistency check, not a mitigation, and the reverse-causality
    threat remains unmitigated.** unemployment_rate(t) and
    unemployment_rate(t-1) correlate at r=0.9777 pooled across the 864-row
    subset (model/findings_lagged_robustness.md). Unemployment is highly
    persistent year-to-year, so for the large majority of department-years
    the lag is nearly a relabeled copy of the same-year value, not a
    genuinely different regressor. At that level of persistence, a
    coefficient surviving the swap from same-year to lag1 is close to
    guaranteed whether or not the same-year result is partly a simultaneity
    artifact, the test has almost no power to distinguish the two
    explanations. What it does establish, and all it establishes: the
    coefficient does not flip sign or collapse under lagging, which would
    itself have been informative had it happened. It did not happen, that
    is worth recording, but a near-guaranteed outcome under high
    persistence cannot be read as evidence *for* the negative-coefficient
    interpretation over the reverse-causality one. This project still has
    no instrument and no natural experiment. Reverse causality is not
    narrowed by this test; it is an open threat to interpretation with no
    mitigation currently in the repo.

---

## Future work

The poverty mechanism (informalisation vs. genuine growth vs. churn) is
unresolved and is the natural next test, building on the per-capita
individual/company breakdown already run in
model/findings_informalisation.md. The birth-rate model (Appendix) is a
complete secondary analysis that has not received the same robustness
battery (urban/rural split, IdF-drop, temporal interaction) as the main
firm-rate claim; extending that battery is future work, not a second main
claim of this project.

---

## Bottom line

French regional entrepreneurship from 2012 to 2021 is structured by
opportunity factors (education, income, and urban environment), not by
necessity. Unemployment is the weakest of the 8 predictors tested,
ranking last (8/8) on the SHAP measure in the full panel and in both the
urban and rural subsets. Its partial OLS relationship with firm creation
is negative in the full panel (unweighted coef = -0.304, p = 0.044; pop-weighted
coef = -0.660, p = 0.001) and in urban departments; in rural
departments the raw coefficient is positive (+0.081), but this is not
confirmed by the pooled interaction test (clustered p = 0.231 unweighted,
p = 0.401 pop-weighted) and unemployment still ranks last in the rural SHAP
model, so the pattern is compositional, not necessity-driven. Across every
specification, unemployment carries little predictive weight. The
necessity-entrepreneurship hypothesis, which is prominent in much of the
comparative literature, does not fit the French regional evidence for
this period.

---

_Numbers locked from: model/findings_final.md and model/split_findings.md_
_Data documentation: DATA_SOURCES.md_
_Generated: 2026-07-16_

---

## Appendix: birth rate determinants (secondary analysis, not a co-finding)

**This is a methodological extension only.** It applies the same LODO + OOF SHAP
framework to a second target variable (birth rate) using three additional data
sources (births, deaths, marriages). It does not produce a second main claim.
The main claim of this project is and remains the necessity hypothesis rejection above.

An exploratory XGBoost + SHAP model using the same 8-feature LODO scheme,
now targeting birth rate (live births per 1,000 inhabitants). Features:
marriage rate (Social), median income / unemployment rate / poverty rate (Economic),
higher-ed share / % urban / doctor density / Gini (Structural).

### Validation

| Scheme | R2 | MAE |
|--------|-----|-----|
| Leave-One-Year-Out (LOYO) | 0.952 | 0.3417 |
| **Leave-One-Dept-Out (LODO) ★** | **0.715** | **0.8406** |
| Random 10-fold (KFold) | 0.948 | 0.3517 |

LODO R2 = 0.715: 72% of birth-rate variance explained in held-out departments.
Stronger than the firm-rate model (0.678).

### SHAP feature importance (OOF, LODO)

| Feature | Group | Mean |SHAP| |
|---|---|---|
| % Urban | Structural | 1.1452 |
| Poverty rate | Economic | 0.3797 |
| Marriage rate | Social | 0.2695 |
| Median income | Economic | 0.2610 |
| Doctor density | Structural | 0.2213 |
| Unemployment rate | Economic | 0.1380 |
| Gini coefficient | Structural | 0.0939 |
| Higher-ed share | Structural | 0.0700 |

Group totals: Structural 59%, Economic 30%, Social 10%.

### Key OLS findings (department-clustered SE)

- marriage_rate: +0.43, p = 0.0009 (strong positive, robust to weighting)
- q2_disp (median income): -0.0006, p < 0.001 (negative, demographic transition:
  richer departments have fewer births per capita)
- pct_urban: +0.08, p < 0.001 (positive, urban departments have higher birth rates,
  likely driven by younger population structure and immigration in IDF and major cities)
- edu_share_sup: +0.13, p < 0.001 (positive, OLS runs counter to SHAP rank,
  possibly via age structure confounding)
- unemployment_rate: -0.05, p = 0.63 (not significant)

### Interpretation

Urban structure dominates (% urban alone accounts for 1.14 of total SHAP), capturing
the demographic concentration and younger age structure in metropolitan departments.
Marriage rate is the strongest individually interpretable social predictor (OLS p = 0.001).
Income is negative after conditioning on urbanisation, consistent with the demographic
transition: higher-income departments have lower fertility once urban effects are removed.
Unemployment is again the weakest predictor (ranked 6/8 by SHAP) and is not
significant in OLS, a parallel to the firm-rate model's finding.

Figures generated: figures/birth_grouped_shap_bar.png, figures/birth_shap_beeswarm.png,
figures/birth_shap_dependence_marriage.png.

_Numbers locked from: model/findings_birth.md (generated 2026-07-16)_
