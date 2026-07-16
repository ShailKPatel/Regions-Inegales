> **DEPRECATED.** This is the earlier draft of the same OOF-SHAP-via-LODO
> pipeline (same methodology, same numbers as `model/findings_final.md`),
> superseded by `model/final_model.py`. Not cited anywhere in
> `FINDINGS.md`. Kept for reference only.

# Model report - firm_rate baseline
Rows: 960, features: 8
Features: ['q2_disp', 'gini_disp', 'poverty_rate_disp', 'unemployment_rate', 'doctor_density_per_100k', 'edu_share_sup', 'pct_urban', 'pct_wages']

## CV metrics (XGBoost)
| scheme | R2 | MAE |
|---|---|---|
| Leave-One-Year-Out | 0.9290 | 0.7564 |
| Leave-One-Department-Out | 0.6780 | 1.4951 |
| Random 10-fold (leaky baseline) | 0.9316 | 0.7479 |

Overfitting gap (random KFold R2 - LODO R2): 0.2535

## OLS R2
- Unweighted OLS R2: 0.7637
- Population-weighted OLS R2: 0.8188

## OLS coefficients (unweighted vs weighted, department-clustered SE)
|                         |   coef_unweighted |   pval_unweighted |   coef_weighted |   pval_weighted |
|:------------------------|------------------:|------------------:|----------------:|----------------:|
| const                   |     -16.6589      |       0.000237064 |    -16.9038     |     0.00639601  |
| q2_disp                 |       0.000897988 |       8.68875e-08 |      0.00109879 |     4.66105e-05 |
| gini_disp               |       6.20949     |       0.550514    |    -11.9444     |     0.408613    |
| poverty_rate_disp       |       0.596795    |       2.76561e-09 |      0.859335   |     2.98455e-12 |
| unemployment_rate       |      -0.303787    |       0.0437988   |     -0.659857   |     0.000740342 |
| doctor_density_per_100k |       0.0057666   |       0.182138    |      0.0112704  |     0.0345312   |
| edu_share_sup           |       0.204089    |       0.00486438  |      0.157929   |     0.0633131   |
| pct_urban               |       0.0210113   |       0.290229    |      0.0159521  |     0.526887    |
| pct_wages               |      -0.15314     |       0.00554931  |     -0.148231   |     0.0355252   |

## SHAP top features (mean |SHAP|, OOF via LODO)
|                         |        0 |
|:------------------------|---------:|
| q2_disp                 | 1.11404  |
| edu_share_sup           | 1.05052  |
| poverty_rate_disp       | 0.738319 |
| pct_wages               | 0.632915 |
| gini_disp               | 0.409649 |
| doctor_density_per_100k | 0.395729 |
| pct_urban               | 0.186019 |
| unemployment_rate       | 0.185286 |

## Gini confound check (OOF SHAP, with vs without Ile-de-France)
- WITH IdF: gini_disp mean|SHAP| = 0.4096, rank 5/8; LODO R2 = 0.6780
- WITHOUT IdF: gini_disp mean|SHAP| = 0.4064, rank 5/8; LODO R2 = 0.7126

## Figures
- figures/model_shap_beeswarm.png
- figures/model_shap_bar.png
