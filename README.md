# Régions Inégales

XGBoost analysis of firm-creation rates across 96 French departments, 2012–2021. Two competing explanations exist for why rates differ: the necessity hypothesis (unemployment and poverty push people into self-employment) and the opportunity hypothesis (education and income pull people toward entrepreneurship).

**Result:** the necessity hypothesis is rejected. Opportunity features (education, income) carry 58% of predictive SHAP weight; necessity features (unemployment, poverty) carry 20%. Unemployment ranks last of 8 features and runs negative in the full panel. LODO R² = 0.678.

## Live app

[regions-inegales.streamlit.app](https://regions-inegales.streamlit.app/)

## Running locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Requires Python 3.10+.

## Pages

| Page | Content |
|------|---------|
| Overview | Headline result and SHAP group shares |
| Map | Choropleth of any panel variable by year |
| Model | Feature importance, cross-validation, urban/rural split, birth rate appendix |
| Explore | Scatter of firm rate vs any feature, with OLS trendline |
| Methods | Data sources, limitations, citation |

## Data

Nine official sources, all publicly available:

| Source | Producer | Variable |
|--------|----------|----------|
| Filosofi | INSEE | Income deciles, Gini, poverty rates |
| SIDE | INSEE | Firm creations by department |
| Estimations de taux de chômage localisés | INSEE | ILO unemployment rate |
| RPPS | DREES | Doctor density per 100k |
| Diplômes et formation | INSEE | Higher-education share |
| Grille de densité 2025 | INSEE | % Urban |
| DS_NAISSANCES_FECONDITE_SERIES | INSEE | Live births, birth rate |
| DS_ETAT_CIVIL_DECES_COMMUNES | INSEE | Deaths, death rate |
| État civil DEP6 (annual files) | INSEE | Marriages, marriage rate |

Panel: 960 observations (96 metropolitan departments × 10 years), 51 columns. Download URLs and verification notes are in [DATA_SOURCES.md](DATA_SOURCES.md).

## Model

XGBoost trained on 8 structural predictors. SHAP (SHapley Additive exPlanations) used for feature attribution. Three cross-validation schemes; the headline is leave-one-department-out (LODO) at R² = 0.678.

| Validation | R² | MAE |
|------------|-----|-----|
| Leave-One-Dept-Out (LODO) ★ | 0.678 | 1.4951 |
| Leave-One-Year-Out (LOYO) | 0.929 | 0.7564 |
| Random 10-fold (KFold) | 0.932 | 0.7479 |

★ Primary result. KFold is a leaky baseline: departments appear in both train and test sets.

## Repository layout

```
app/
  app.py                  Entry point
  pages/                  Five Streamlit pages
  assets/                 GeoJSON for choropleth
  data_loader.py          Panel loading and caching
  utils.py                CSS, colour constants, shared components
model/                    Model scripts and output files
figures/                  Output figures (PNG)
FINDINGS.md               Main results document
DATA_SOURCES.md           Data provenance and verification
requirements.txt          Python dependencies
```

The panel dataset (`merged/france_panel_master.csv`) and raw sources (`sources/`) are not committed. See DATA_SOURCES.md to rebuild from scratch.

## Citation

Author (2026). *Régions Inégales: Opportunity vs. Necessity Entrepreneurship Across French Departments, 2012–2021.* Working paper. Preprint forthcoming on SSRN/EconPapers.

## Source code

[github.com/ShailKPatel/Regions-Inegales](https://github.com/ShailKPatel/Regions-Inegales)
