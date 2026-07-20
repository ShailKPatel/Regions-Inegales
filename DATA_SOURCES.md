# Régions Inégales, Data Sources & Methods

## Master panel

- Shape: 960×51, 96 metropolitan departments × 10 years (2012–2021).
- Keys: `dep_code` (2-char string, leading zeros preserved, incl. 2A/2B), `year` (int). Scope: metropolitan France; overseas departments excluded throughout.
- 9 merged sources: Filosofi (income/poverty/inequality), Firm Creations (SIDE), Unemployment, Doctor Density, Education (interpolated annual panel), Urban/Rural Density, Live Births, Deaths, Marriages.
- Every merge validated to exactly 960 rows, 0 duplicate keys, 96 rows per year.
- Column inventory: 2 keys, 7 metadata, 33 candidate features, 9 target-related; full roles and collinearity map in `merged/_column_roles.md`.
- Pre-demographic backup: `merged/france_panel_master_pre_birth.csv` (45 columns, before births/deaths/marriages added).

Auxiliary asset not merged into master: departmental population estimates (`sources/population_insee.csv`, INSEE, 1975–2026 series), used as denominator cross-check for doctor density; kept for feature engineering.

---

## Source 1, Filosofi: Income, Poverty & Inequality

**Source / producer**, INSEE, Fichier Localisé Social et Fiscal (Filosofi), "Structure et distribution des revenus, inégalité des niveaux de vie".

**Download URLs**, one ZIP per year, "Base niveau administratif" (covers EPCI, arrondissement, département, région, France métropolitaine):

| Year | URL |
|------|-----|
| 2012 | https://www.insee.fr/fr/statistiques/2043745 |
| 2013 | https://www.insee.fr/fr/statistiques/2388413 |
| 2014 | https://www.insee.fr/fr/statistiques/3126151 |
| 2015 | https://www.insee.fr/fr/statistiques/3560118 |
| 2016 | https://www.insee.fr/fr/statistiques/4190006 |
| 2017 | https://www.insee.fr/fr/statistiques/4291712 |
| 2018 | https://www.insee.fr/fr/statistiques/5009218 |
| 2019 | https://www.insee.fr/fr/statistiques/6036907 |
| 2020 | https://www.insee.fr/fr/statistiques/6692220 |
| 2021 | https://www.insee.fr/fr/statistiques/7756855 |

**File**, "Base niveau administratif" ZIP per year. Accessed: 2026.

**Variables**, income deciles and quartiles (declared `_dec` and disposable `_disp` series), Gini coefficient, S80/S20, D9/D1, poverty rates, all at department level. 27 Filosofi-derived columns in total; see `merged/_column_roles.md` for the full list and collinearity groupings.

**Key methodological decisions**

- Two income concepts retained in parallel: declared (`_dec`) and disposable (`_disp`). Disposable (post-tax, post-transfer) is the policy-relevant concept for inequality research; use one family consistently.
- Department-level rows only extracted from each year's base; regional and national aggregates excluded.
- `poverty_rate_disp` and `poverty_rate_dec` use the 60%-of-median threshold. See provenance notes for the `poverty_rate_dec` null structure in 2012.

**Verification**, 96-department sweep for median income and Gini across representative years; spot anchors against published INSEE summary tables. No full-column external cross-check (BDM does not expose a Filosofi SDMX series at department level).

---

## Source 2, Firm Creations (SIDE)

**Source / producer**, INSEE, SIDE system (Système d'Information sur les Démographies d'Entreprises), "Démographie des entreprises : créations d'entreprises et d'établissements de 2012 à 2024".

**Download URL**, https://www.insee.fr/fr/statistiques/8557644

**File**, `DS_SIDE_CREA_DEP_REG_NAT_2024_CSV_FR.zip`, données nationales/régionales/départementales, A21 sector breakdown. Accessed: 2026.

**Variables**, `total_firm_creations`; legal-form breakdown: `creations_individual` (EI: all individual entrepreneurs), `creations_sarl`, `creations_sas`, `creations_other_legal`; sector breakdown: `creations_sector_industry`, `creations_sector_construction`, `creations_sector_trade`, `creations_sector_services`. All 9 are TARGET-RELATED, do not use as features simultaneously with each other.

**Key methodological decisions**

- LEGAL_FORM code 10 = "Entrepreneur individuel" covers all individual entrepreneurs (micro/auto-entrepreneurs and classic EI combined). SIDE provides no sub-code to separate micro-entrepreneurs; `creations_individual` is the unified total and this is appropriate.
- Our metropole (96-dept) sums are byte-exact against the raw SIDE dataset (`DS_SIDE_CREA_ENT_DEP_REG_NAT_CJ_2024_CSV_FR.zip`, GEO_OBJECT=DEP, LEGAL_FORM=_T) for all 10 years, confirming the panel column is a faithful extraction. Published INSEE Première headlines undershoot this panel by 2-8%, but that gap is a **vintage/revision effect, not a definitional one**: INSEE revises créations d'entreprises totals upward for 2-5 years after first release as late SIRENE registrations get backdated to their true creation date. Compared against the *original* headline (e.g. Insee Première n1734, "691 300" for 2018), our panel looks too high; compared against the *restated* total for the same year (Insee Première n1984, Feb 2024: "749,3" thousand for 2018), the gap flips to our panel being 2.6-3.6% *below* the national figure, consistent with a stable France-entière DOM/Mayotte share. See Verification below.
- Sector classification A21 (21 sectors) aggregated to 4 broad groups for panel columns.

**Verification**, Two checks. (1) Internal-consistency cross-check (`scripts/make_side_crosscheck.py` -> `sources/_side_firm_creations_crosscheck.csv`): `total_firm_creations` equals the sum of its own legal-form breakdown and the sum of its own sector breakdown in all 960/960 rows, including the 2016-2018 counting-rule reform window (max |diff| = 0 both ways). (2) External anchor against published INSEE national totals (`scripts/check_side_external.py` -> `sources/_side_external_anchor.csv`): all 10 years (2012-2021) classify CONSISTENT (published France-entière total exceeds our metropole-only sum, implied DOM/Mayotte share stable at 2.6-3.6%, spread 0.96pp across the decade, well inside the 1-6%/2pp tolerance). Coverage: 10/10 years; 4/10 (2018-2021) independently double-sourced (raw SIDE dataset + a second, later publication, Insee Première n1984 "Les créations d'entreprises en 2023", restating those years); 6/10 (2012-2017) rest on the raw SIDE dataset's own national total only, not cross-confirmed by a second independent publication for those specific years. Department-level published-table spot check was attempted but not achieved: INSEE Première regional breakdowns give percentage growth, not absolute counts, so no independently-published department-level anchor was reachable. Remaining unverified: 2012-2017 single-source status, and department-level accuracy.

---

## Source 3, Unemployment Rate

**Source / producer**, INSEE, "Estimations de taux de chômage localisés", série longue trimestrielle depuis 1982.

**Download URL**, https://www.insee.fr/fr/statistiques/2012804

**File**, `sl_etc_2025T4.xls`, Q4-2025 vintage, published 2026-03-20. Sheet "Département". Stored at project root.

**Variable**, `unemployment_rate`: annual mean of 4 quarterly ILO/BIT localised unemployment rates (CVS, en moyenne trimestrielle, en %), rounded to 1 decimal. `unemployment_rate_raw` retains the unrounded mean for internal checks. Champ: France hors Mayotte from Q1-2014; France métropolitaine before Q1-2014.

**Key methodological decisions**

- Quarterly T1–T4 averaged to annual mean. All 960 (dep_code, year) combinations had exactly 4 non-null quarterly values; no imputation applied.
- Q4-2025 vintage used throughout; values may differ by ≤0.1 from contemporaneous 2012–2021 publications due to retrospective seasonal-adjustment revisions applied by INSEE.
- DOM departments (971 Guadeloupe, 972 Martinique, 973 Guyane, 974 La Réunion) present in source and excluded.

**Verification**, Full-column external cross-check (2026-06-11): all 96 metro department annual-average series fetched independently from the INSEE BDM SDMX API (SERIES_BDM idbanks 001784552–001784707, "Taux de chômage localisés (moyenne annuelle), Ensemble", verified against TITLE_FR and REF_AREA). Coverage: 960/960 (dep, year) pairs. Result: 818/960 exact (85.2%), 142/960 within ±0.1 (14.8%), 0 beyond ±0.1; max |diff| = 0.1. Non-exact cases explained by rounding: BDM annual series are computed from unrounded quarterly intermediates, while our panel averages the 1-decimal values as published in the quarterly XLS. Panel kept as-is. Cross-check file: `sources/_unemployment_bdm_crosscheck.csv`.

---

## Source 4, Doctor Density

**Source / producer**, DREES (Direction de la recherche, des études, de l'évaluation et des statistiques), "La démographie des professionnels de santé depuis 2012", RPPS register (Répertoire partagé des professionnels de santé, via the Ordres professionnels). Dataset identifier: `la-demographie-des-professionnels-de-sante-depuis-2012` on data.drees.sante.gouv.fr.

**Download URL**, data.drees.sante.gouv.fr, dataset `la-demographie-des-professionnels-de-sante-depuis-2012`.

**File**, `doctors_raw/Médecins RPPS 2012-2025.xlsx`, vintage 28/07/2025. Sheet `Densités` (density per 100k) used for merge; sheet `Effectifs` (headcount) used for cross-check only.

**Variable**, `doctor_density_per_100k`: density of active doctors per 100,000 inhabitants, source precision to 2 decimal places. Stock at 1 January of year Y → assigned to panel year Y.

**Key methodological decisions**

- Scope: *actifs occupés* = at least one ongoing liberal or salaried activity (including replacement) at 1 January. Excludes volunteers, students, interns, junior doctors, and those working exclusively for the Services de santé des Armées. No age cap for médecins in RPPS (the <62 cap applies only to ADELI professions). This definition runs ~1% below DREES "en activité" totals (definitional difference, not error).
- Filter applied: `territoire='1-France métropolitaine'`, `specialites_agregees='00-Ensemble'`, `specialites='00-Ensemble'`, `exercice='0-Ensemble'`, `tranche_age='00-Ensemble'`, `sexe='0-Ensemble'`: all doctors, all specialties, all modes d'exercice, all ages, both sexes.
- Population denominator: standard INSEE departmental population as computed by DREES.
- DOM/region/France aggregate rows excluded (`territoire='2-DROM'`; `departement='000-Ensemble'`).

**Verification**, Full-column external cross-check (2026-06-13): `density_check = n_doctors / pop_jan1 × 100,000` recomputed for all 960 cells from `sources/population_insee.csv` (INSEE pop estimates, same vintage) and the `Effectifs` sheet. Result: 960/960 within ±0.5%; max |rel_diff| = 0.0025%; mean diff ≈ 0. Confirms DREES uses the same INSEE population denominators. Cross-check file: `sources/_doctor_density_pop_crosscheck.csv`.

---

## Source 5, Education (Higher-Ed Share, Interpolated Annual Panel)

**Source / producer**, INSEE, "Diplômes et formation 2022, Base communale des diplômes et formations". Source page: insee.fr/fr/statistiques/8581488.

**Download URL**, https://www.insee.fr/fr/statistiques/8581488

**File**, `education_raw/base-cc-diplomes-formation-2022_csv/base-cc-diplomes-formation-2022.CSV` (separator `;`, encoding latin-1). Three rolling-census snapshots: reference years 2010-window → labelled 2011; 2015-window → 2016; 2022-window → 2022. Accessed: 2026.

**Variable**, `edu_share_sup`: share (%) of non-schooled population aged 15+ holding any higher-education diploma (Bac+2 and above), population-weighted aggregate from commune to department:
- 2022: (P22_NSCOL15P_SUP2 + P22_NSCOL15P_SUP34 + P22_NSCOL15P_SUP5) / P22_NSCOL15P × 100
- 2016: P16_NSCOL15P_SUP / P16_NSCOL15P × 100
- 2011: (P11_NSCOL15P_BACP2 + P11_NSCOL15P_SUP) / P11_NSCOL15P × 100

Flag `edu_is_interpolated`: `False` only for year 2016 (the single within-panel observed census anchor); `True` for all other years (2012–2015, 2017–2021).

**Key methodological decisions**

- Census snapshots 2011/2016/2022 only; **linearly interpolated** to fill the full 2012–2021 panel using `numpy.interp`. 2012–2015: straight-line between 2011 and 2016 observed values; 2016: observed value carried through unchanged; 2017–2021: straight-line between 2016 and 2022 observed values. Only 2016 is a real within-panel observation.
- Sub-level breakdowns (SUP2/SUP34/SUP5 in 2022 vs. single SUP in 2016) are not cross-year comparable and are not used. Total supérieur share only.
- Do not over-interpret year-to-year variation in `edu_share_sup`; variation is linear by construction except at 2016.
- Commune-level numerator and denominator summed before computing the department share (not averaging share estimates).

**Verification**, Internal-exhaustive: 288/288 department re-derivation (3 anchor years × 96 departments); 34,787-commune gender-and-parts reconciliation; monotonic trend confirmed across all 96 departments. Definition independently confirmed against Observatoire des Territoires (ANCT) indicator `p_diplsup15`: identical formula `100*(nb_diplsup15/nb_nonsco15)`, scope BTS/DUT–doctorat, dept level, years 2011/2016/2022. Exact-level numeric anchor not performed: Geoclip portal values are JS-gated (JS-blocked at time of check). Ranking corroborated against published INSEE young-graduate shares.

---

## Source 6, Urban/Rural Density (Grille de Densité)

**Source / producer**, INSEE, "Grille de densité 2025, Fichier de diffusion au 1er janvier 2026", RP2021-based. Source page: insee.fr/fr/information/8571524.

**Download URL**, https://www.insee.fr/fr/information/8571524

**File**, `density_raw/fichier_diffusion_2026.xlsx`, RP2021-based, post-July-2025 correction of departmental rural population share. Sheet `Maille départementale` (5 header rows + 101 data rows). Accessed: 2026.

**Variable**, `pct_urban`: continuous model feature; computed as `100 − P_RURAL` where `P_RURAL` is the departmental share of population living in rural communes (%), rounded to 6 decimal places. Range: 10.268% (Creuse, dep 23) to 100.000% (Paris, dep 75). Time-invariant: one grille classification per department, assigned identically to all panel years 2012–2021.

Metadata column `density_class` (urban / intermediate / rural, from `LIBDENS` verbatim): 14 urban, 31 intermediate, 51 rural departments. **Interpretation-only metadata (not a model feature).**

Flag `density_is_static = True` on all 960 rows.

**Key methodological decisions**

- Framework: **Grille de densité** (1 km² density grid, RP2021), NOT the RP-based "zonage en unités urbaines". The grille classifies more population as rural in rural departments; e.g., Creuse reads 10.3% urban under the grille vs ~22% under unités urbaines. This distinction matters for interpreting `pct_urban` values. `density_class` is interpretation-only metadata and is not a model feature.
- Time-invariant structural variable: `density_is_static = True` on all 960 rows (one RP2021 classification per department, replicated across years).
- Hauts-de-Seine (92): source `P_RURAL = 1.0e-15` (floating-point artifact in source file), treated as 0 → `pct_urban = 100.0%`.
- DOM rows (971–976) present in source and excluded.

**Verification**, Build-confirmed: 0 nulls, 0 duplicate keys, 96 rows per year. `pct_urban` range and department ranking cross-checked against published INSEE density typology. Framework distinction (grille vs unités urbaines) documented and confirmed against published Creuse rural-share figures.

---

## Source 7, Live Births (INSEE DS_NAISSANCES_FECONDITE_SERIES)

**Source / producer**, INSEE, "Naissances et fécondité, séries longues".

**Download URLs**
- Long series (data.gouv): https://www.data.gouv.fr/datasets/naissances-et-fecondite-series-longues
- Per commune flat CSV (data.gouv): https://www.data.gouv.fr/fr/datasets/nombre-de-naissances-par-commune/
- Departmental CSVs (INSEE): https://www.insee.fr/fr/statistiques/8582142

**File**, `DS_NAISSANCES_FECONDITE_SERIES_CSV_FR/DS_NAISSANCES_FECONDITE_SERIES_data.csv` (raw, now deleted after extraction). Clean extracted file: `sources/births_insee.csv`.

**Variables added to panel**
- `live_births`: live births at place of residence (`LVB_PLACE_RES`, measure code, total age `_T`). Raw integer count per department per year.
- `birth_rate`: `live_births / pop_jan1 × 1000`. Computed at merge time using `sources/population_insee.csv` as denominator. Stored as float, 4 decimal places.

**Key methodological decisions**
- `LVB_PLACE_RES` (place of residence) used rather than `LVB_PLACE_REG` (place of registration). Residence is the correct geographic attribution for a panel of departmental characteristics.
- Total age group (`AGE = _T`) only; age-specific fertility rates not used.
- Overseas departments (971, 972, 973, 974, 976) excluded to match the metropolitan panel scope.

**Verification**, 960/960 (dep_code, year) pairs present; 0 nulls; 0 duplicate keys; 96 metro departments matched exactly against master panel dep_code list. Rate range: 6.5–18.8 per 1,000 (Creuse low, Seine-Saint-Denis high), consistent with published INSEE departmental natality profiles.

---

## Source 8, Deaths (INSEE DS_ETAT_CIVIL_DECES_COMMUNES)

**Source / producer**, INSEE, "Décès et mortalité, séries longues".

**Download URLs**
- Long series (data.gouv): https://www.data.gouv.fr/datasets/deces-et-mortalite-series-longues/
- Per commune flat CSV (data.gouv): https://www.data.gouv.fr/fr/datasets/nombre-de-deces-par-commune/

**File**, `DS_ETAT_CIVIL_DECES_COMMUNES_CSV_FR/DS_ETAT_CIVIL_DECES_COMMUNES_data.csv` (raw, now deleted after extraction). Clean extracted file: `sources/deaths_insee.csv`.

**Variables added to panel**
- `deaths`: number of deaths (`EC_MEASURE = DTH`, `OBS_STATUS = A` only). Raw integer count per department per year.
- `death_rate`: `deaths / pop_jan1 × 1000`. Computed at merge time using `sources/population_insee.csv` as denominator. Stored as float, 4 decimal places.

**Key methodological decisions**
- `GEO_OBJECT = DEP` rows only; commune, region, and other geographic levels excluded.
- `OBS_STATUS = M` (missing/not applicable) rows dropped; only `OBS_STATUS = A` (normal) retained.
- Overseas departments (971, 972, 973, 974, 976) excluded to match the metropolitan panel scope.

**Verification**, 960/960 (dep_code, year) pairs present; 0 nulls; 0 duplicate keys; 96 metro departments matched exactly against master panel dep_code list. Rate range: 5.3–17.4 per 1,000, consistent with published INSEE departmental mortality profiles.

---

## Source 9, Marriages (INSEE DEP6 annual files)

**Source / producer**, INSEE, civil registration state ("État civil"), DEP6 tables: monthly marriages by department and region of marriage.

**Download URLs (by year)**
- 2012: https://www.insee.fr/fr/statistiques/2020625
- 2013: https://www.insee.fr/fr/statistiques/2020490
- 2014: https://www.insee.fr/fr/statistiques/1913527
- 2015: https://www.insee.fr/fr/statistiques/2561535
- 2016: https://www.insee.fr/fr/statistiques/3317603
- 2017: https://www.insee.fr/fr/statistiques/3704307
- 2018: https://www.insee.fr/fr/statistiques/4273672
- 2019: https://www.insee.fr/fr/statistiques/5012966
- 2020: https://www.insee.fr/fr/statistiques/6045407
- 2021: https://www.insee.fr/fr/statistiques/6790710

**Files**, Raw files now deleted after extraction. Clean extracted file: `sources/marriages_insee.csv`.

**Variables added to panel**
- `marriages`: total marriages (all types: HF, HH, FF) per department per year. Integer count (number of weddings, not persons).
- `marriage_rate`: `marriages / pop_jan1 × 1000`. Computed at merge time using `sources/population_insee.csv`. Stored as float, 4 decimal places.

**Key methodological decisions**
- Files span three formats across years: XLS with single EFF sheet (2012–2013), XLS/XLSX with separate HF/HH/FF sheets (2014–2017, 2020), and CSV with REGDEP_MAR codes (2018, 2019, 2021). All three sheet types summed to get the total across marriage types.
- **Critical: CSV files (2018, 2019, 2021) count married persons (2 per wedding), not weddings.** NBMAR was divided by 2 to convert to wedding counts consistent with the XLS files. Verified against published INSEE national totals (e.g., 2019 national total 228,000 weddings confirmed via CSV total ÷ 2).
- REGDEP_MAR codes in CSV files follow the format `[2-digit region code][2-digit or 2A/2B dept code]` (e.g., "1175" = region 11 + dept 75 = Paris). Regional aggregates (ending "XX"), national rows ("FM", "FE"), and overseas departments (3-digit codes 971–976) excluded.
- Geography of marriage, not residence: counts marriages registered in each department, not where the spouses reside. Standard for this series.
- 2022 microdata folder (`irsomar2022_dd7_csv`) contained only 2022 cross-sectional tables, not used, deleted.

**Verification**, 960/960 (dep_code, year) pairs present; 0 nulls; 0 duplicate keys. National totals per year consistent with known INSEE figures: 239,840 (2012), ~228,000 (2018–2019), 150,545 (2020 COVID drop), ~213,000 (2021 recovery). Rate range: 1.8–4.5 per 1,000.

---

## Provenance notes

- **Cantal 4-commune blanks**: INSEE suppressed education data for 4 Cantal communes (15031, 15035, 15047, 15171). The likely cause is a commune restructuring (these communes appear to have been folded into Neussargues-Moissac, 15141); this was not independently confirmed against INSEE commune records. This is a restructuring artefact, NOT confidentiality suppression. The Cantal (dep 15) `edu_share_sup` is computed from the remaining communes; the suppressed communes' combined population weight is unquantifiable from this census file alone.

- **`poverty_rate_dec` null for all 2012**: 96 null values (all 96 departments, year 2012 only) in the declared-income poverty rate. Structural: the declared-income Pauvres file did not exist for 2012. Use `poverty_rate_disp` as the consistent alternative for cross-year analysis, or exclude 2012 from any `_dec`-based analyses. The nulls are expected; they are not a merge error.

- **`poverty_rate_disp` 2012 source**: Sourced from INSEE Filosofi "Revenus et pauvreté des ménages en 2012" (dataset 1895078, URL: https://www.insee.fr/fr/statistiques/2043745). This is the standard Filosofi release for 2012 at department level and is the consistent alternative to `poverty_rate_dec` for cross-year analysis.

- **INSEE firm-stat micro/non-micro split coding error 2015+**: Affects some SIDE sub-tables in the source data. The total creation counts (which `total_firm_creations` and our breakdowns use) are unaffected. This further supports the choice of `creations_individual` as the unified EI column rather than attempting a micro/non-micro sub-split.
