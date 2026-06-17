# Model report - firm_rate baseline
Rows: 960, features: 8
Features: ['q2_disp', 'gini_disp', 'poverty_rate_disp', 'unemployment_rate', 'doctor_density_per_100k', 'edu_share_sup', 'pct_urban', 'pct_wages']

## CV metrics (XGBoost)
| scheme | R2 | MAE |
|---|---|---|
| Leave-One-Year-Out | 0.9062 | 0.8465 |
| Leave-One-Department-Out | 0.6744 | 1.3075 |
| Random 10-fold (leaky baseline) | 0.9211 | 0.7862 |

Overfitting gap (random KFold R2 - LODO R2): 0.2466

## OLS R2
- Unweighted OLS R2: 0.7346
- Population-weighted OLS R2: 0.7666

## OLS coefficients (unweighted vs weighted)
|                         |   coef_unweighted |   pval_unweighted |   coef_weighted |   pval_weighted |
|:------------------------|------------------:|------------------:|----------------:|----------------:|
| const                   |      -16.1098     |       4.35811e-12 |    -16.4697     |     1.91572e-08 |
| q2_disp                 |        0.0010981  |       1.00587e-26 |      0.00124929 |     9.9867e-22  |
| gini_disp               |       -2.21237    |       0.655486    |    -20.6489     |     0.000734444 |
| poverty_rate_disp       |        0.422804   |       8.98262e-42 |      0.589618   |     4.94136e-51 |
| unemployment_rate       |       -0.0182259  |       0.758657    |     -0.224522   |     0.00698651  |
| doctor_density_per_100k |        0.00316984 |       0.0239077   |      0.00892755 |     2.45682e-08 |
| edu_share_sup           |        0.254964   |       4.87131e-20 |      0.212781   |     6.66386e-11 |
| pct_urban               |        0.0273327  |       7.37914e-05 |      0.032773   |     7.44765e-05 |
| pct_wages               |       -0.218802   |       8.6783e-30  |     -0.209082   |     8.32142e-19 |

## SHAP top features (mean |SHAP|, full data)
|                         |        0 |
|:------------------------|---------:|
| q2_disp                 | 1.17617  |
| edu_share_sup           | 1.17269  |
| pct_wages               | 0.772501 |
| poverty_rate_disp       | 0.617539 |
| gini_disp               | 0.410983 |
| doctor_density_per_100k | 0.384493 |
| pct_urban               | 0.180885 |
| unemployment_rate       | 0.109581 |

## Gini confound check (with vs without Ile-de-France)
- WITH IdF: gini_disp mean|SHAP| = 0.4110, rank 5/8; LODO R2 = 0.6744
- WITHOUT IdF: gini_disp mean|SHAP| = 0.3577, rank 5/8; LODO R2 = 0.7679

## Figures
- figures/model_shap_beeswarm.png
- figures/model_shap_bar.png
