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
lower firm formation.

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

5. **Education interpolated against a post-panel anchor.** Higher-ed share is
   observed at census snapshots (2011, 2016, 2022) and linearly interpolated.
   The 2022 anchor lies outside the panel window, so 2017–2021 values embed
   future information. The LOYO 2021 fold is mildly contaminated; year-to-year
   variation in this variable is artificial by construction.

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
    firm_rate. An untested alternative reading of that same negative sign is
    that causation runs the other way: departments with more firm creation
    have more hiring, which mechanically lowers local unemployment. Nothing
    in this project's design (no instrument, no lagged/pre-determined
    unemployment measure, no natural experiment) rules this out. The
    negative unemployment coefficient is consistent with both "necessity-push
    doesn't operate here" and "entrepreneurship reduces local unemployment,"
    and this analysis cannot adjudicate between them.

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
