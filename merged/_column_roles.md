# Column Roles, france_panel_master.csv (960 × 51)

Generated: 2026-07-16  
Source file covers: 96 French departments × 10 years (2012–2021)
Model panel: 960 rows (2012–2021)

---

## Classification Table

| column | role | dtype | notes |
|--------|------|-------|-------|
| dep_code | KEY | str | 2-char string; 01–95, 2A, 2B; no float coercion |
| year | KEY | int64 | 2012–2021, exactly 96 rows per year |
| dep_name | METADATA | str | Human-readable label; not a feature |
| n_households | METADATA | float64 | Size weight; use as offset/weight, not feature |
| n_persons | METADATA | float64 | Size weight |
| n_uc | METADATA | float64 | Consumption-unit count (size weight) |
| density_class | METADATA | str | urban / intermediate / rural; static per dept; dist: 140/310/510 |
| density_is_static | METADATA | bool | Always True on all 960 rows; drop before modelling |
| edu_is_interpolated | METADATA | bool | False for 2016 (96 rows), True for all other years; audit flag |
| q1_dec | CANDIDATE FEATURE | float64 | 1st quartile, declared income; ↔ q1_disp (r≈0.93+) |
| q2_dec | CANDIDATE FEATURE | float64 | Median, declared |
| q3_dec | CANDIDATE FEATURE | float64 | 3rd quartile, declared |
| d1_dec | CANDIDATE FEATURE | float64 | 1st decile, declared |
| d9_dec | CANDIDATE FEATURE | float64 | 9th decile, declared |
| gini_dec | CANDIDATE FEATURE | float64 | Gini, declared; r=0.937 with gini_disp |
| s80s20_dec | CANDIDATE FEATURE | float64 | S80/S20 ratio, declared; r=0.986 with d9_d1_dec, near-duplicate |
| d9_d1_dec | CANDIDATE FEATURE | float64 | D9/D1 ratio, declared; r=0.986 with s80s20_dec, near-duplicate |
| q1_disp | CANDIDATE FEATURE | float64 | 1st quartile, disposable income |
| q2_disp | CANDIDATE FEATURE | float64 | Median, disposable |
| q3_disp | CANDIDATE FEATURE | float64 | 3rd quartile, disposable; r=0.981 with d9_disp |
| d1_disp | CANDIDATE FEATURE | float64 | 1st decile, disposable |
| d9_disp | CANDIDATE FEATURE | float64 | 9th decile, disposable |
| gini_disp | CANDIDATE FEATURE | float64 | Gini, disposable; r=0.978 with d9_d1_disp, r=0.978 with s80s20_disp |
| s80s20_disp | CANDIDATE FEATURE | float64 | S80/S20 ratio, disposable; r=0.986 with d9_d1_disp, near-duplicate |
| d9_d1_disp | CANDIDATE FEATURE | float64 | D9/D1 ratio, disposable; r=0.986 with s80s20_disp, near-duplicate |
| poverty_rate_disp | CANDIDATE FEATURE | float64 | % below 60% median, disposable; 960 non-null |
| poverty_rate_dec | CANDIDATE FEATURE | float64 | % below 60% median, declared; 96 nulls (all 2012), expected |
| pct_wages | CANDIDATE FEATURE | float64 | % of income from wages; Σ pct_* = 100 ± 0.1 confirmed |
| pct_unemployment | CANDIDATE FEATURE | float64 | % of income from unemployment benefits; r=0.792 with unemployment_rate |
| pct_capital_gains | CANDIDATE FEATURE | float64 | % of income from capital |
| pct_pensions | CANDIDATE FEATURE | float64 | % of income from pensions; r=0.928 with pct_wages (anti-collinear pair) |
| pct_other | CANDIDATE FEATURE | float64 | % of income from other sources |
| unemployment_rate | CANDIDATE FEATURE | float64 | ILO unemployment rate %; r=0.792 with pct_unemployment |
| doctor_density_per_100k | CANDIDATE FEATURE | float64 | GPs per 100k inhabitants; range [171, 894] |
| edu_share_sup | CANDIDATE FEATURE | float64 | % with higher-education diploma; interpolated 2012–2021 except anchor 2016; monotonic per dept |
| pct_urban | CANDIDATE FEATURE | float64 | % population in urban units; static per dept; range [10.3, 100.0] |
| live_births | CANDIDATE FEATURE | int64 | Live births at place of residence (LVB_PLACE_RES); source: INSEE DS_NAISSANCES_FECONDITE_SERIES |
| birth_rate | CANDIDATE FEATURE | float64 | Live births per 1,000 inhabitants (live_births / pop_jan1 × 1000); computed at merge time |
| deaths | CANDIDATE FEATURE | int64 | Deaths at place of residence (DTH); source: INSEE DS_ETAT_CIVIL_DECES_COMMUNES |
| death_rate | CANDIDATE FEATURE | float64 | Deaths per 1,000 inhabitants (deaths / pop_jan1 × 1000); computed at merge time |
| marriages | CANDIDATE FEATURE | int64 | Total marriages (HF+HH+FF) by dept of marriage; source: INSEE DEP6 annual files 2012–2021 |
| marriage_rate | CANDIDATE FEATURE | float64 | Marriages per 1,000 inhabitants (marriages / pop_jan1 × 1000); computed at merge time |
| total_firm_creations | TARGET-RELATED | int64 | Primary outcome candidate; must NOT be used as feature |
| creations_individual | TARGET-RELATED | int64 | Legal-form breakdown of total; collinear with total |
| creations_sarl | TARGET-RELATED | int64 | Legal-form breakdown |
| creations_sas | TARGET-RELATED | int64 | Legal-form breakdown |
| creations_other_legal | TARGET-RELATED | int64 | Legal-form breakdown |
| creations_sector_industry | TARGET-RELATED | int64 | Sector breakdown of total; collinear with total |
| creations_sector_construction | TARGET-RELATED | int64 | Sector breakdown |
| creations_sector_trade | TARGET-RELATED | int64 | Sector breakdown |
| creations_sector_services | TARGET-RELATED | int64 | Sector breakdown |

---

## Role Summary

| role | count |
|------|-------|
| KEY | 2 |
| METADATA | 7 |
| CANDIDATE FEATURE | 33 |
| TARGET-RELATED | 9 |
| **Total** | **47** |

---

## Collinearity Groups (do not use all members simultaneously)

| group | members | max |r| | recommendation |
|-------|---------|---------|----------------|
| Inequality (disp) | gini_disp, s80s20_disp, d9_d1_disp | 0.986 | Pick one; gini_disp preferred |
| Inequality (dec) | gini_dec, s80s20_dec, d9_d1_dec | 0.986 | Pick one; gini_dec preferred |
| Inequality cross | gini_disp ↔ gini_dec | 0.937 | Choose one income concept |
| Income level (disp) | q1–q3, d1, d9 disp variants | 0.981 | q2_disp (median) as representative |
| Income level (dec) | q1–q3, d1, d9 dec variants | ~0.95 | q2_dec as representative |
| disp ↔ dec mirrors | q2_disp ↔ q2_dec, etc. | ~0.93+ | Choose one income concept globally |
| Joblessness | pct_unemployment ↔ unemployment_rate | 0.792 | unemployment_rate preferable (ILO) |
| Income composition | pct_wages ↔ pct_pensions | 0.928 | Sum constraint: only 4 of 5 are free |
| Firm sub-columns | creations_* ↔ total_firm_creations | ~0.9+ | All are TARGET-RELATED; don't mix with features |

---

## Notes for Feature Engineering

1. **Income concept choice**: the `_disp` (disposable) and `_dec` (declared) series are highly correlated within-measure but represent different economic concepts. Disposable (post-transfer) is the more policy-relevant concept for inequality research. Pick one family and be consistent.

2. **Inequality representation**: gini is the most internationally comparable; s80s20 and d9_d1 are near-duplicates. Use gini only, or gini + poverty_rate as a second dimension.

3. **edu_share_sup caution**: 2016 is the census anchor; other years are linear interpolations. Flag this in any paper or results.

4. **pct_urban vs density_class**: both encode urbanisation. pct_urban is continuous and preferable for regression; density_class is useful for fixed-effects or subgroup analysis.

5. **poverty_rate_dec in 2012**: 96 nulls are structural (source data unavailability), not a merge error. Either drop 2012 from dec-based analyses or use poverty_rate_disp as the consistent alternative.

6. **Size scaling**: n_households / n_persons should be used as regression weights or offsets rather than features, to account for the scale differences between Paris (dep 75) and Lozère (dep 48).
