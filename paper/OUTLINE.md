# Paper Outline — Opportunity, Not Necessity

Draft skeleton. Every section below is content-mapped from FINDINGS.md / DATA_SOURCES.md
unless marked BLOCKED. Nothing here is final prose, it's the rough shape to draft against.
Numbers are locked from FINDINGS.md, do not restate from memory when drafting, copy from source.

---

## 1. Title

Working (substantive framing):
> Opportunity, Not Necessity: Sub-National Evidence on Regional Entrepreneurship
> from French Departments, 2012-2021

Alt (methodological framing, safer if lit search shows substantive claim is old news):
> Leakage-Honest Evidence on Opportunity vs. Necessity Entrepreneurship:
> French Departments, 2012-2021

Decide after lit search (Section 3 below). Don't lock now.

---

## 2. Abstract — DRAFT LAST

Draft exists (TOPIC_ASSESSMENT.md), reproduced here as a starting point only,
needs a final pass once Introduction/Related Work are written, not before:

> We test the necessity-versus-opportunity model of entrepreneurship using a
> department-year panel covering all 96 metropolitan French departments from
> 2012 to 2021, combining nine independently verified administrative sources.
> Using leave-one-department-out cross-validation and SHAP attribution across
> four model families, we find that opportunity factors, education and
> income, account for roughly three times the predictive weight of necessity
> factors, unemployment and poverty, and that unemployment's partial
> relationship with firm creation is negative rather than positive across
> every specification and every one of four alternative operationalizations
> tested. The result holds outside Ile-de-France, in both urban and rural
> department subsets, and is not an artifact of model choice, though we
> cannot rule out reverse causality running from local firm creation to lower
> unemployment, which remains this study's central open threat to
> interpretation.

Do not touch until Sections 3-4 exist. Abstract written first is how you end up
promising a claim the finished paper doesn't actually support.

---

## 3. Introduction — BLOCKED on lit search

Cannot draft real prose yet. What it needs to do, once unblocked:

1. State the necessity/opportunity framework (2-3 sentences, cite GEM framework
   and Fritsch & Storey or equivalent canonical cite).
2. State what's already known at cross-country level (cite it, don't assert
   from memory the way TOPIC_ASSESSMENT.md's Phase 2 explicitly flagged as
   UNVERIFIED).
3. State the gap this paper fills — either (a) first sub-national French test
   of this specific question, if lit search confirms that's true, or (b) a
   methodological gap (leakage-honest validation not standard in this
   literature), if (a) turns out false.
4. One paragraph roadmap of the paper.

Placeholder only until Section headed "Lit search results" below is filled in.

---

## 4. Related Work — BLOCKED on lit search

Structure once populated:
- Necessity vs. opportunity entrepreneurship, foundational framing (GEM,
  Verheul et al., Fritsch & Storey or whatever the search turns up).
- Cross-country empirical evidence for opportunity dominance in high-income
  countries.
- France-specific / French regional entrepreneurship literature, INSEE and
  France Stratégie work likely exists here, this is the search that
  determines novelty framing.
- Validation methodology in regional panel work (does anyone else use LODO
  vs pooled/K-fold for this kind of question — this is the methodological
  novelty check).

---

## 5. Data

Source: DATA_SOURCES.md, mostly reformat + trim internal-only detail
(script filenames, internal crosscheck CSV names can go to a data appendix
or footnote, not main text).

Content to carry over:
- Panel shape: 960 obs, 96 metropolitan departments x 10 years (2012-2021).
- 9 sources: Filosofi (income/poverty/Gini), SIDE (firm creations, target
  var), Localised unemployment, RPPS doctor density, Education (interpolated
  from 3 census snapshots), Grille de densité (urban/rural), Live births,
  Deaths, Marriages (last 3 feed the Appendix birth model only, not main
  text unless Appendix is included).
- Verification methodology per source: external cross-checks, deviation
  percentages (unemployment 85.2% exact match, doctor density max deviation
  0.0025%, etc). This is a genuine strength, worth a full paragraph or a
  compact table, most department-panel papers don't document provenance this
  thoroughly, it's citable as methodological rigor.
- Target variable: firm creation rate per 1,000 inhabitants (SIDE).
- 8 structural predictors: 4 opportunity (higher-ed share, median disposable
  income, percent urban, doctor density per 100k), 2 necessity (unemployment
  rate, poverty rate), 2 controls (Gini, wage income share).
- Known data quirks worth a footnote, not main text: poverty_rate_disp 2012
  sourced differently (inflated vs later years), 2016-2018 SIDE
  counting-rule reform, education interpolation caveat (2022 anchor is
  outside panel window, LOYO 2021 fold mildly contaminated).

---

## 6. Method

Source: FINDINGS.md "Method" section, near-verbatim structure works:

- Model: XGBoost, 8-feature matrix, 960 department-years.
- Attribution: SHAP (SHapley Additive exPlanations), grouped into
  Opportunity / Necessity / Other for the headline comparison, plus
  per-feature ranking.
- Validation, three schemes, explain why each exists and why LODO is
  headline:
  - LODO (leave-one-department-out): trains on 95 depts, tests on the
    96th, never seen. Headline metric, no department-identity leakage.
  - KFold (random 10-fold): explicitly labeled leaky baseline, departments
    appear in both train/test across different years, inflates R2.
    Reported for contrast, not as a competing headline.
  - LOYO (leave-one-year-out): tests temporal extrapolation for known
    units, not generalization to new units, all 96 depts already seen in
    training years. Caveated, not comparable to LODO.
- Confirmatory spec: department-clustered OLS/WLS (population-weighted),
  run alongside SHAP as a second, independent method on the same question.
- Four-model comparison (ElasticNetCV, RandomForest, LightGBM, XGBoost),
  identical LODO folds, identical feature matrix, fair tuning budget per
  model. This IS the robustness section, but the setup belongs here in
  Method since it's a validation design choice, not a result.

---

## 7. Results

Source: FINDINGS.md "Main finding" + all "Robustness" subsections. Suggest
mirroring FINDINGS.md's own structure, it's already organized as a paper
would be:

### 7.1 Main finding
- Grouped SHAP table: Opportunity 58% / Necessity 20% / Other 22%.
- Per-feature SHAP ranking table (8 rows).
- OLS confirmation: unemployment coef -0.304 (p=0.044) unweighted, -0.660
  (p=0.001) pop-weighted.
- Four-way unemployment operationalization test (volume, composition,
  per-capita individual, per-capita company), none show the necessity
  signature.
- Verdict statement: necessity hypothesis rejected.

### 7.2 Poverty's coefficient (brief, honestly flagged as unresolved)
- Positive OLS coef, moderate SHAP rank (3rd/8).
- Informalisation test: MIXED/INCONCLUSIVE. State plainly, don't oversell,
  don't hide either, this is a real finding (a null/mixed one) and reviewers
  respect it being reported as such.

### 7.3 Robustness battery
One subsection each, tight, these are mostly table + 2-3 sentences:
- Urban/rural split (58/20 full, 54/27 urban, 61/15 rural).
- Temporal interaction (unemployment weakens over time, poverty strengthens,
  don't conflate the two despite both being "necessity" features).
- Diagnostics: log(pop) control on doctor density, drop-2012 sensitivity.
- Four-model comparison: ElasticNet LODO R2=0.7142 vs XGBoost 0.6759, state
  plainly, explain why XGBoost stays headline anyway (SHAP infrastructure,
  full battery already run against it), this reconciliation is already
  written in FINDINGS.md's "Robustness: does a simpler model find the same
  thing?" section, near copy-paste ready.
- Lagged-unemployment check (reverse-causality mitigation): same-year vs
  t-1 coefficient comparison, stayed negative and grew slightly rather than
  shrinking toward zero.

### 7.4 What did not work
- Gini coefficient: inconclusive, rank 5/8, sign/magnitude weighting-
  dependent. Worth a short paragraph, negative results build credibility.

---

## 8. Discussion

Not really drafted anywhere yet as prose, needs original writing, but the
content to synthesize already exists across FINDINGS.md:
- What the pattern of results collectively says: strong evidence against
  necessity-push, weaker/more qualified evidence on what explains poverty's
  independent positive coefficient.
- Where this sits relative to Related Work (depends on Section 4).
- What would change the interpretation: an instrument or natural experiment
  for reverse causality, a firm-survival dataset instead of registrations-
  only, individual-level (not ecological) data.

---

## 9. Limitations

Source: FINDINGS.md "Limitations", 10 numbered items, already publication-
ready prose, minimal editing needed:
1. Registrations, not survival/net growth.
2. Mostly cross-sectional (≈70% of variance is between-department).
3. 2016-2018 SIDE measurement artefact.
4. Correlational, not causal.
5. Education interpolation, 2022 anchor outside panel window.
6. pct_urban time-invariant, single-vintage classification.
7. pct_wages uses a different income concept (DEC vs DISP) from other vars.
8. Doctor density as amenity proxy, not literal healthcare-access claim.
9. Ecological inference (department-level, not individual-level decisions).
10. Reverse causality / simultaneity, lagged test narrows but doesn't close
    this.

---

## 10. Conclusion

Short, restates Section 7's verdict plainly, one paragraph on future work
(poverty mechanism, birth-rate model extension to full robustness battery,
explicitly framed as separate future papers per FINDINGS.md's own framing,
not folded in here).

---

## 11. Appendix (optional, decide during drafting)

- Birth-rate model (FINDINGS.md Appendix): LODO R2=0.715, %urban dominant
  (59% structural share). Include only if it strengthens the paper without
  diluting the single main claim, current recommendation (Phase 2 of
  TOPIC_ASSESSMENT.md) is keep it OUT of this paper, save for a separate
  paper once it gets its own robustness battery.
- Four-model tuning grids and rank-correlation matrix
  (model/findings_model_comparison.md), full detail, main text only needs
  the summary numbers.

---

## 12. References — BLOCKED on lit search

Empty until Section 3/4 unblock. No .bib file exists yet anywhere in the repo.

---

## Suggested draft order (see prior chat message for the "why")

1. Method (Section 6) — no blocker, near-mechanical.
2. Results (Section 7) — no blocker, near-mechanical.
3. Data (Section 5) — no blocker, mostly reformat.
4. Limitations (Section 9) — no blocker, near copy-paste.
5. **Lit search** — unblocks everything below.
6. Related Work (Section 4).
7. Introduction (Section 3).
8. Discussion (Section 8).
9. Conclusion (Section 10).
10. Title + Abstract (Sections 1-2) — finalize last.
11. References (Section 12) — compiled as you cite through steps 5-9.
