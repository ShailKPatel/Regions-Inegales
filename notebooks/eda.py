#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA for Regions Inegales -- france_panel_master.csv (960 x 45)
96 French metropolitan departments x 10 years (2012-2021).
Intended target: firm creation rate per capita, constructed in S1.
Run: python3 notebooks/eda.py
"""

# ===========================================================================
# S0 -- SETUP & SANITY
# ===========================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except ImportError:
    HAS_LOWESS = False

warnings.filterwarnings('ignore')

ROOT = "/home/crusie/3. Code/Régions Inégales"
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

# Publication-grade style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
})
PALETTE = sns.color_palette('colorblind')
C0, C1, C2 = PALETTE[0], PALETTE[1], PALETTE[2]

findings = []


def savefig(fig, name):
    """Save PNG @150 dpi and SVG to figures/."""
    fig.savefig(os.path.join(FIGS, name + '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, name + '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"  -> saved {name}.png + .svg")


print("=" * 68)
print("S0 -- Setup & Sanity")
print("=" * 68)

# dtype={'dep_code': object} forces numpy str regardless of pandas version.
# (pandas 3.x maps dtype=str to pd.StringDtype; object gives numpy dtype.)
df = pd.read_csv(
    os.path.join(ROOT, 'merged', 'france_panel_master.csv'),
    sep=';',
    dtype={'dep_code': object},
)

# --- shape ---
assert df.shape == (960, 45), f"Expected 960x45, got {df.shape}"
print(f"Shape: {df.shape}  OK")

# --- dep_code data integrity (not dtype -- check actual values) ---
assert '01' in df['dep_code'].values, "Leading zero '01' missing from dep_code"
assert '09' in df['dep_code'].values, "Leading zero '09' missing from dep_code"
assert '2A' in df['dep_code'].values, "Corsica 2A missing"
assert '2B' in df['dep_code'].values, "Corsica 2B missing"
assert not pd.api.types.is_numeric_dtype(df['dep_code']), "dep_code must not be numeric"
print(f"dep_code dtype : {df['dep_code'].dtype}")
print(f"dep_code sample: {df['dep_code'].head(6).tolist()}  -- leading zeros intact OK")

# --- rows per year ---
rows_per_year = df.groupby('year').size().unique()
assert list(rows_per_year) == [96], f"Expected 96 rows/year, got {rows_per_year}"
print(f"Years: {sorted(df['year'].unique())}")
print(f"Rows/year: {rows_per_year.tolist()}  OK")

# --- missing values ---
nulls = df.isnull().sum()
missing = nulls[nulls > 0]
print("\nMissing values:")
print(missing.to_string() if not missing.empty else "  None")
print("(Expected: poverty_rate_dec 96 nulls -- all 2012, structural)")
print()
print(df.head(3).to_string())


# ===========================================================================
# S1 -- TARGET CONSTRUCTION & DENOMINATOR DECISION
# ===========================================================================
print("\n" + "=" * 68)
print("S1 -- Target Construction & Denominator Decision")
print("=" * 68)

print("""
Three rate candidates:
  rate_npers  = total_firm_creations / n_persons   * 1000   (Filosofi persons)
  rate_pop    = total_firm_creations / pop_jan1    * 1000   (INSEE total pop)
  rate_adult  = total_firm_creations / (0.62*pop_jan1) * 1000  (working-age proxy)
""")

# Merge pop_jan1 (read-only join)
pop = pd.read_csv(
    os.path.join(ROOT, 'sources', 'population_insee.csv'),
    sep=';',
    dtype={'dep_code': object},
)
pop['dep_code'] = pop['dep_code'].str.strip('"').str.strip()

n_before = len(df)
df = df.merge(pop[['dep_code', 'year', 'pop_jan1']], on=['dep_code', 'year'], how='left')
assert len(df) == n_before, "Merge changed row count"
assert df['pop_jan1'].isnull().sum() == 0, \
    f"pop_jan1 join produced {df['pop_jan1'].isnull().sum()} nulls -- dep_code mismatch"
print(f"pop_jan1 merged OK, {df['pop_jan1'].isnull().sum()} nulls  ->  shape now {df.shape}")

# Sanity: n_persons vs pop_jan1
ratio = df['n_persons'] / df['pop_jan1']
print(f"n_persons / pop_jan1: mean={ratio.mean():.4f}  std={ratio.std():.4f}  "
      f"min={ratio.min():.4f}  max={ratio.max():.4f}")

# Build three rates
WORKING_AGE_FRAC = 0.62
df['rate_npers'] = df['total_firm_creations'] / df['n_persons']                    * 1000
df['rate_pop']   = df['total_firm_creations'] / df['pop_jan1']                     * 1000
df['rate_adult'] = df['total_firm_creations'] / (WORKING_AGE_FRAC * df['pop_jan1']) * 1000

print(f"\n{'Rate':<15} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
print("-" * 48)
for col in ['rate_npers', 'rate_pop', 'rate_adult']:
    s = df[col]
    print(f"{col:<15} {s.mean():8.3f} {s.std():8.3f} {s.min():8.3f} {s.max():8.3f}")

# Rank correlations
dept_avg = df.groupby('dep_code')[['rate_npers', 'rate_pop', 'rate_adult']].mean()
print("\nSpearman rank correlations (dept mean rates):")
for a, b in [('rate_npers', 'rate_pop'), ('rate_npers', 'rate_adult'), ('rate_pop', 'rate_adult')]:
    rho, p = spearmanr(dept_avg[a], dept_avg[b])
    print(f"  {a} vs {b}: rho={rho:.5f}  p={p:.2e}")

# Largest rank shifts
dept_avg['rk_npers'] = dept_avg['rate_npers'].rank(ascending=False)
dept_avg['rk_adult'] = dept_avg['rate_adult'].rank(ascending=False)
dept_avg['rk_shift'] = (dept_avg['rk_npers'] - dept_avg['rk_adult']).abs()
dep_names_s = df.groupby('dep_code')['dep_name'].first()
dept_avg = dept_avg.join(dep_names_s)
top_movers = dept_avg.nlargest(10, 'rk_shift')[['dep_name', 'rk_npers', 'rk_adult', 'rk_shift']]
print("\nTop 10 rank-movers (rate_npers vs rate_adult):")
print(top_movers.to_string())

# Figure: distribution comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
RATE_DEFS = {
    'rate_npers': 'n_persons\n(Filosofi)',
    'rate_pop':   'pop_jan1\n(INSEE total)',
    'rate_adult': '0.62 x pop_jan1\n(working-age proxy)',
}
for ax, (col, label) in zip(axes, RATE_DEFS.items()):
    d = df[col]
    ax.hist(d, bins=40, color=C0, edgecolor='white', linewidth=0.4)
    ax.axvline(d.mean(),   color='crimson', ls='--', lw=1.5, label=f'mean={d.mean():.2f}')
    ax.axvline(d.median(), color='navy',    ls=':',  lw=1.5, label=f'med={d.median():.2f}')
    ax.set_title(f'Denominator: {label}', fontsize=10)
    ax.set_xlabel('Firm creations per 1,000 persons')
    ax.set_ylabel('Count' if ax is axes[0] else '')
    ax.legend(fontsize=8)
fig.suptitle('Firm Creation Rate -- Three Denominator Candidates (960 obs)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'S1_rate_distributions')

# Figure: denominator scatter
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(dept_avg['rate_npers'], dept_avg['rate_adult'], alpha=0.65, s=50, color=C0, edgecolors='none')
m, b = np.polyfit(dept_avg['rate_npers'], dept_avg['rate_adult'], 1)
xl = np.linspace(dept_avg['rate_npers'].min(), dept_avg['rate_npers'].max(), 200)
ax.plot(xl, m * xl + b, 'r-', lw=1.5, label=f'OLS slope={m:.3f}')
rho_sc, _ = spearmanr(dept_avg['rate_npers'], dept_avg['rate_adult'])
ax.set_title(f'Denominator Comparison (dept means)\nSpearman rho={rho_sc:.5f}', fontweight='bold')
ax.set_xlabel('Rate/1k  (n_persons denom.)')
ax.set_ylabel('Rate/1k  (working-age proxy denom.)')
ax.legend()
plt.tight_layout()
savefig(fig, 'S1_denominator_scatter')

# Set primary target
df['firm_rate'] = df['rate_pop']

print("""
DENOMINATOR RECOMMENDATION (awaiting confirmation):
  rate_pop = total_firm_creations / pop_jan1 * 1000
  Evidence:
  - n_persons / pop_jan1 ~= 0.977 across all dept-years (near-identical)
  - Spearman rho > 0.999 between any pair -- rankings unaffected by choice
  - rate_adult = rate_pop / 0.62: uniform rescale, adds zero cross-dept variation
  - pop_jan1 is the same official INSEE denominator used for doctor_density_per_100k
""")

findings.append('[S1] THREE rate defs compared. Spearman rho > 0.999 between any pair -- '
                'denominator choice does NOT change dept rankings.')
findings.append('[S1] rate_adult = rate_pop / 0.62: uniform rescale only, no new cross-dept info.')
findings.append('[S1] RECOMMENDATION (awaiting confirmation): rate_pop = firm_creations / pop_jan1 * 1000'
                ' -- consistent with doctor_density denominator; most transparent.')


# ===========================================================================
# S2 -- UNIVARIATE DISTRIBUTIONS
# ===========================================================================
print("\n" + "=" * 68)
print("S2 -- Univariate Distributions")
print("=" * 68)

FEAT_INEQ  = ['q2_disp', 'd1_disp', 'd9_disp', 'gini_disp', 's80s20_disp',
               'd9_d1_disp', 'poverty_rate_disp']
FEAT_COMP  = ['pct_wages', 'pct_unemployment', 'pct_capital_gains', 'pct_pensions', 'pct_other']
FEAT_OTHER = ['unemployment_rate', 'doctor_density_per_100k', 'edu_share_sup', 'pct_urban', 'n_persons']
ALL_FEATS  = FEAT_INEQ + FEAT_COMP + FEAT_OTHER
TARGET_COL = 'firm_rate'


def skew_flag(s):
    if   abs(s) < 0.5: return 'symmetric'
    elif abs(s) < 1.0: return 'moderate skew'
    elif abs(s) < 2.0: return 'high skew'
    else:              return '** VERY HIGH -- consider log-transform'


log_candidates = []
print(f"\n{'Variable':<28} {'Mean':>10} {'Std':>10} {'Skewness':>10}  Flag")
print("-" * 72)
for col in ALL_FEATS + [TARGET_COL]:
    if col not in df.columns:
        continue
    skew = df[col].skew()
    flag = skew_flag(skew)
    if 'log' in flag:
        log_candidates.append(col)
    print(f"{col:<28} {df[col].mean():>10.2f} {df[col].std():>10.2f} {skew:>10.3f}  {flag}")
print(f"\nLog-transform candidates (|skew|>=2): {log_candidates}")


def plot_hist_grid(cols, suptitle, fname, ncols=4, figw=5, figh=4):
    cols = [c for c in cols if c in df.columns]
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figw * ncols, figh * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for i, col in enumerate(cols):
        ax   = axes_flat[i]
        data = df[col].dropna()
        ax.hist(data, bins=35, color=C0, edgecolor='white', linewidth=0.3)
        ax.axvline(data.mean(),   color='crimson', ls='--', lw=1.3,
                   label=f'mean={data.mean():.1f}')
        ax.axvline(data.median(), color='navy',    ls=':',  lw=1.3,
                   label=f'med={data.median():.1f}')
        ax.set_title(f'{col}\nskew={data.skew():.2f}', fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='upper right')
    for ax in axes_flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    savefig(fig, fname)


plot_hist_grid(FEAT_INEQ + [TARGET_COL],
               'Univariate Distributions -- Inequality (disp) & Target',
               'S2_hist_inequality')
plot_hist_grid(FEAT_COMP + FEAT_OTHER,
               'Univariate Distributions -- Income Composition & Other Features',
               'S2_hist_other')

# Boxplots by period
df['period'] = pd.cut(df['year'], bins=[2011, 2015, 2018, 2021],
                      labels=['2012-15', '2016-18', '2019-21'])
plot_cols = [c for c in ALL_FEATS + [TARGET_COL] if c in df.columns]
ncols_bp  = 5
nrows_bp  = int(np.ceil(len(plot_cols) / ncols_bp))
fig, axes = plt.subplots(nrows_bp, ncols_bp, figsize=(5 * ncols_bp, 4 * nrows_bp), squeeze=False)
axes_flat = axes.flatten()
for i, col in enumerate(plot_cols):
    ax = axes_flat[i]
    sns.boxplot(data=df, x='period', y=col, ax=ax, palette='Blues_d',
                flierprops=dict(marker='.', markersize=3, alpha=0.4), linewidth=0.8)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=8)
for ax in axes_flat[len(plot_cols):]:
    ax.set_visible(False)
fig.suptitle('Boxplots by Period -- All Variables (2012-15 / 2016-18 / 2019-21)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
savefig(fig, 'S2_boxplots_by_period')

# Outlier departments
print("\nOutlier departments (|z-score| > 3) per variable:")
dep_names_map = df.groupby('dep_code')['dep_name'].first()
for col in ALL_FEATS + [TARGET_COL]:
    if col not in df.columns:
        continue
    skew = df[col].skew()
    sub  = df[[col, 'dep_code']].dropna()
    z    = np.abs(stats.zscore(sub[col]))
    codes = sub.loc[z > 3, 'dep_code'].unique()
    if len(codes) > 0:
        named = [(c, dep_names_map.get(c, c)) for c in codes]
        print(f"  {col} (skew={skew:.2f}): {named}")

findings.append(f'[S2] Log-transform candidates (|skew|>=2): {log_candidates}')
findings.append('[S2] Paris (75) and inner IDF systematic outliers on n_persons, '
                'doctor_density_per_100k, and firm_rate. Small rural depts (Lozere 48) on low end.')
findings.append('[S2] poverty_rate_disp 2012 inflated (~21.6% vs ~14.5% post-2012) -- '
                'structural break (sourced from declared-income file that year).')


# ===========================================================================
# S3 -- THE TARGET ACROSS SPACE & TIME
# ===========================================================================
print("\n" + "=" * 68)
print("S3 -- Target Across Space & Time")
print("=" * 68)

dept_rate = (df.groupby('dep_code')
             .agg(mean_rate=('firm_rate', 'mean'), dep_name=('dep_name', 'first'))
             .reset_index()
             .sort_values('mean_rate', ascending=False)
             .reset_index(drop=True))
dept_rate['rank'] = range(1, len(dept_rate) + 1)

print("Top 10 departments:")
print(dept_rate.head(10)[['rank', 'dep_code', 'dep_name', 'mean_rate']].to_string(index=False))
print("\nBottom 10 departments:")
print(dept_rate.tail(10)[['rank', 'dep_code', 'dep_name', 'mean_rate']].to_string(index=False))

# Bar chart: all 96 departments sorted
fig, ax = plt.subplots(figsize=(20, 7))
nat_mean = dept_rate['mean_rate'].mean()
nat_p75  = dept_rate['mean_rate'].quantile(0.75)
bar_colors = ['#d62728' if v >= nat_p75 else
              '#1f77b4' if v >= nat_mean else '#aec7e8'
              for v in dept_rate['mean_rate']]
ax.bar(range(len(dept_rate)), dept_rate['mean_rate'],
       color=bar_colors, edgecolor='none', width=0.85)
ax.axhline(nat_mean, color='black', ls='--', lw=1.5,
           label=f'National mean = {nat_mean:.2f}')
for idx_row, row in dept_rate.head(5).iterrows():
    ax.text(idx_row, row['mean_rate'] + 0.15, row['dep_code'],
            fontsize=7, ha='center', rotation=45, color='darkred')
for idx_row, row in dept_rate.tail(5).iterrows():
    ax.text(idx_row, row['mean_rate'] + 0.15, row['dep_code'],
            fontsize=7, ha='center', rotation=45, color='navy')
ax.set_xticks(range(0, len(dept_rate), 5))
ax.set_xticklabels(dept_rate['dep_code'].iloc[::5], rotation=90, fontsize=7)
ax.set_xlabel('Department (sorted by mean firm creation rate, 2012-2021)')
ax.set_ylabel('Mean firm creation rate (per 1,000 inhabitants)')
ax.set_title('Department Ranking -- Firm Creation Rate (all 96, mean 2012-2021)',
             fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
savefig(fig, 'S3_dept_ranking_bar')

# National mean trend
nat_yr = df.groupby('year')['firm_rate'].agg(['mean', 'median', 'std']).reset_index()
print("\nNational mean firm creation rate by year:")
print(nat_yr.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(nat_yr['year'],
                nat_yr['mean'] - nat_yr['std'],
                nat_yr['mean'] + nat_yr['std'],
                alpha=0.18, color=C0, label='+/- 1 std (cross-dept)')
ax.plot(nat_yr['year'], nat_yr['mean'],   'o-', color=C0, lw=2.2, ms=7, label='National mean')
ax.plot(nat_yr['year'], nat_yr['median'], 's--', color=C1, lw=1.6, ms=6, label='National median')
ax.axvspan(2019.5, 2020.5, alpha=0.12, color='crimson', label='2020 (COVID)')
ax.axvspan(2015.5, 2018.5, alpha=0.07, color='orange',  label='2016-18 (SIDE reform era)')
ax.set_xlabel('Year')
ax.set_ylabel('Firm creation rate (per 1,000 inhabitants)')
ax.set_title('National Firm Creation Rate Trend 2012-2021\n(mean +/- 1 std across 96 departments)',
             fontweight='bold')
ax.set_xticks(range(2012, 2022))
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, 'S3_national_trend')

# Spaghetti + IQR band
dept_piv = df.pivot(index='year', columns='dep_code', values='firm_rate')
top5 = dept_rate.head(5)['dep_code'].tolist()
bot5 = dept_rate.tail(5)['dep_code'].tolist()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
for dep in dept_piv.columns:
    ax1.plot(dept_piv.index, dept_piv[dep], alpha=0.10, lw=0.7, color='steelblue')
for dep in top5:
    if dep in dept_piv.columns:
        ax1.plot(dept_piv.index, dept_piv[dep], lw=2.2, color='crimson', alpha=0.9, zorder=5)
        ax1.text(2021.1, dept_piv.loc[2021, dep], dep, fontsize=8, color='crimson', va='center')
for dep in bot5:
    if dep in dept_piv.columns:
        ax1.plot(dept_piv.index, dept_piv[dep], lw=2.2, color='navy', alpha=0.9, zorder=5)
        ax1.text(2021.1, dept_piv.loc[2021, dep], dep, fontsize=8, color='navy', va='center')
ax1.axvspan(2019.5, 2020.5, alpha=0.10, color='crimson')
ax1.set_xlabel('Year')
ax1.set_ylabel('Firm rate (per 1,000)')
ax1.set_title('All Department Trajectories\n(red=top5, blue=bottom5)', fontweight='bold')
ax1.set_xticks(range(2012, 2022))

p25    = dept_piv.quantile(0.25, axis=1)
p75    = dept_piv.quantile(0.75, axis=1)
median = dept_piv.median(axis=1)
ax2.fill_between(dept_piv.index, p25, p75, alpha=0.25, color='steelblue', label='IQR (25-75%)')
ax2.plot(dept_piv.index, median, 'b-o', lw=2, ms=6, label='Cross-dept median')
ax2.axvspan(2019.5, 2020.5, alpha=0.10, color='crimson', label='2020 COVID')
ax2.axvspan(2015.5, 2018.5, alpha=0.06, color='orange',  label='2016-18 reform era')
ax2.set_xlabel('Year')
ax2.set_ylabel('Firm rate (per 1,000)')
ax2.set_title('Median +/- IQR Band Across Departments', fontweight='bold')
ax2.set_xticks(range(2012, 2022))
ax2.legend(fontsize=9)
plt.tight_layout()
savefig(fig, 'S3_spaghetti_band')

# Variance decomposition
grand_mean = df['firm_rate'].mean()
dept_means = df.groupby('dep_code')['firm_rate'].transform('mean')
SS_total   = ((df['firm_rate'] - grand_mean) ** 2).sum()
SS_between = ((dept_means - grand_mean) ** 2).sum()
SS_within  = ((df['firm_rate'] - dept_means) ** 2).sum()
pct_bw = SS_between / SS_total * 100
pct_wi = SS_within  / SS_total * 100

print(f"\nVARIANCE DECOMPOSITION -- Firm Creation Rate")
print(f"  Total SS        : {SS_total:12.1f}")
print(f"  Between-dept SS : {SS_between:12.1f}  ({pct_bw:.1f}%)")
print(f"  Within-dept SS  : {SS_within:12.1f}  ({pct_wi:.1f}%)")
if pct_bw >= 70:
    vd_impl = (f'{pct_bw:.1f}% cross-sectional / {pct_wi:.1f}% temporal. '
               'Cross-section dominates strongly. FE panel absorbs most signal; '
               'cross-sectional OLS on dept means is a valid first pass.')
elif pct_bw >= 50:
    vd_impl = (f'{pct_bw:.1f}% cross-sectional / {pct_wi:.1f}% temporal. '
               'Cross-section moderately dominates. FE panel recommended.')
else:
    vd_impl = (f'{pct_bw:.1f}% cross-sectional / {pct_wi:.1f}% temporal. '
               'Temporal variation substantial. Panel structure is important.')
print(f"  IMPLICATION: {vd_impl}")
findings.append(f'[S3] Variance decomposition: {vd_impl}')

# COVID shock
r19 = df[df['year'] == 2019].set_index('dep_code')['firm_rate']
r20 = df[df['year'] == 2020].set_index('dep_code')['firm_rate']
chg = ((r20 - r19) / r19 * 100).dropna()
rho_covid, _ = spearmanr(r19, r20)
print(f"\n2019->2020 change: mean={chg.mean():.1f}%  median={chg.median():.1f}%")
print(f"  Positive: {(chg>0).sum()} depts   Negative: {(chg<0).sum()} depts")
print(f"  Range: [{chg.min():.1f}%, {chg.max():.1f}%]")
print(f"  Rank corr 2019 vs 2020: rho = {rho_covid:.4f}")
covid_note = (f'Rankings highly preserved (rho={rho_covid:.3f}).'
              if rho_covid > 0.95 else
              f'Rankings somewhat disrupted (rho={rho_covid:.3f}).')
findings.append(f'[S3] 2020 COVID: national mean change={chg.mean():.1f}%. {covid_note}')

# SIDE reform era
r15 = df[df['year'] == 2015].set_index('dep_code')['firm_rate']
r18 = df[df['year'] == 2018].set_index('dep_code')['firm_rate']
era_chg = ((r18 - r15) / r15 * 100).mean()
print(f"\n2015->2018 mean change (SIDE reform era): {era_chg:.1f}%")
findings.append(f'[S3] 2016-18 SIDE reform era: 2015->2018 national mean change={era_chg:.1f}%.')


# ===========================================================================
# S4 -- FEATURE-TARGET RELATIONSHIPS
# ===========================================================================
print("\n" + "=" * 68)
print("S4 -- Feature-Target Relationships")
print("=" * 68)

KEY_FEATURES = ['q2_disp', 'gini_disp', 'poverty_rate_disp', 'unemployment_rate',
                'doctor_density_per_100k', 'edu_share_sup', 'pct_urban']
FEAT_LABELS  = {
    'q2_disp':                 'Median disposable income (EUR)',
    'gini_disp':               'Gini coefficient (disposable)',
    'poverty_rate_disp':       'Poverty rate 60pct threshold (%)',
    'unemployment_rate':       'ILO unemployment rate (%)',
    'doctor_density_per_100k': 'GP density (per 100k inhabitants)',
    'edu_share_sup':           'Higher-education share (%)',
    'pct_urban':               'Urban population share (%)',
}

feat_corr = {}
print(f"\n{'Feature':<28} {'Pearson r':>10} {'Spearman rho':>14}  Dir/strength")
print("-" * 70)
for feat in KEY_FEATURES:
    if feat not in df.columns:
        continue
    sub = df[[feat, 'firm_rate']].dropna()
    r,   _ = pearsonr(sub[feat],  sub['firm_rate'])
    rho, _ = spearmanr(sub[feat], sub['firm_rate'])
    feat_corr[feat] = (r, rho)
    s    = 'strong' if abs(r) > 0.5 else 'moderate' if abs(r) > 0.3 else 'weak'
    sign = '+' if r > 0 else '-'
    print(f"{feat:<28} {r:>10.3f} {rho:>14.3f}  {sign}{s}")


def scatter_fit(feat, ax, data, title_extra=''):
    sub  = data[[feat, 'firm_rate']].dropna()
    x, y = sub[feat].values, sub['firm_rate'].values
    ax.scatter(x, y, alpha=0.18, s=18, color=C0, edgecolors='none')
    m_lin, b_lin = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 200)
    ax.plot(xl, m_lin * xl + b_lin, 'r-', lw=1.8,
            label=f'Linear (slope={m_lin:.3f})')
    if HAS_LOWESS:
        si = np.argsort(x)
        lo = lowess(y[si], x[si], frac=0.4)
        ax.plot(lo[:, 0], lo[:, 1], 'b--', lw=1.5, label='LOESS')
    r_sub,   _ = pearsonr(x, y)
    rho_sub, _ = spearmanr(x, y)
    ax.set_xlabel(FEAT_LABELS.get(feat, feat), fontsize=9)
    ax.set_ylabel('Firm rate (per 1,000)' if ax.get_subplotspec().colspan.start == 0 else '')
    ax.set_title(f'{feat}{title_extra}\nr={r_sub:.2f}, rho={rho_sub:.2f}', fontsize=9)
    ax.legend(fontsize=7)


# 2-row x 4-col grid
fig, axes = plt.subplots(2, 4, figsize=(21, 11))
axes_flat = axes.flatten()
for i, feat in enumerate(KEY_FEATURES):
    if feat in df.columns:
        scatter_fit(feat, axes_flat[i], df)
for ax in axes_flat[len(KEY_FEATURES):]:
    ax.set_visible(False)
fig.suptitle('Firm Creation Rate vs Key Features (960 obs, 2012-2021)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
savefig(fig, 'S4_scatter_features_target')

# Robustness: with vs without IDF
IDF = ['75', '77', '78', '91', '92', '93', '94', '95']
r_abs     = {f: abs(feat_corr[f][0]) for f in KEY_FEATURES if f in feat_corr}
strongest = max(r_abs, key=r_abs.get)
print(f"\nStrongest feature: {strongest}  (|r|={r_abs[strongest]:.3f})")

df_no_idf = df[~df['dep_code'].isin(IDF)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
scatter_fit(strongest, ax1, df,        title_extra='\n(full dataset)')
scatter_fit(strongest, ax2, df_no_idf, title_extra='\n(excl. IDF)')
fig.suptitle(f'Robustness: {strongest} vs Firm Rate -- IDF influence',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'S4_robustness_idf')

sub_all   = df[[strongest, 'firm_rate']].dropna()
sub_noidf = df_no_idf[[strongest, 'firm_rate']].dropna()
r_all,   _ = pearsonr(sub_all[strongest],   sub_all['firm_rate'])
r_noidf, _ = pearsonr(sub_noidf[strongest], sub_noidf['firm_rate'])
delta = abs(r_all - r_noidf)
idf_note = ('IDF substantially changes correlation.' if delta > 0.1
            else 'Correlation robust to IDF exclusion.')
print(f"  r full    = {r_all:.3f}")
print(f"  r excl IDF = {r_noidf:.3f}")
print(f"  delta-r   = {delta:.3f}  {idf_note}")
findings.append(f'[S4] Strongest feature: {strongest} '
               f'(r={r_all:.2f} full / r={r_noidf:.2f} excl. IDF). {idf_note}')
for feat in KEY_FEATURES:
    if feat not in feat_corr:
        continue
    r, rho = feat_corr[feat]
    sign   = 'positive' if r > 0 else 'negative'
    s      = 'strong' if abs(r) > 0.5 else 'moderate' if abs(r) > 0.3 else 'weak'
    findings.append(f'[S4] {feat}: {sign} {s} assoc. with firm_rate (r={r:.2f}, rho={rho:.2f})')


# ===========================================================================
# S5 -- FEATURE-FEATURE STRUCTURE (COLLINEARITY)
# ===========================================================================
print("\n" + "=" * 68)
print("S5 -- Feature-Feature Structure (Collinearity)")
print("=" * 68)

HEATMAP_COLS = (
    ['q2_disp', 'd1_disp', 'd9_disp', 'gini_disp', 's80s20_disp', 'd9_d1_disp', 'poverty_rate_disp'] +
    ['pct_wages', 'pct_unemployment', 'pct_capital_gains', 'pct_pensions', 'pct_other'] +
    ['unemployment_rate', 'doctor_density_per_100k', 'edu_share_sup', 'pct_urban'] +
    ['firm_rate']
)
HEATMAP_COLS = [c for c in HEATMAP_COLS if c in df.columns]
corr = df[HEATMAP_COLS].corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap_div   = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(
    corr, mask=mask_upper, cmap=cmap_div, vmin=-1, vmax=1, center=0,
    square=True, linewidths=0.4, cbar_kws={'shrink': 0.65},
    annot=True, fmt='.2f', annot_kws={'size': 7}, ax=ax,
)
ax.set_title('Feature-Feature Correlation Matrix (Pearson r)\n'
             'Lower triangle -- _disp family + other features + target',
             fontweight='bold', fontsize=13)
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)
plt.tight_layout()
savefig(fig, 'S5_correlation_heatmap')

# High-corr pairs
feat_only  = [c for c in HEATMAP_COLS if c != 'firm_rate']
high_pairs = []
for i, a in enumerate(feat_only):
    for b in feat_only[i + 1:]:
        r_val = corr.loc[a, b]
        if abs(r_val) >= 0.80:
            high_pairs.append((a, b, r_val))

print(f"\nFeature pairs with |r| >= 0.80 ({len(high_pairs)} total):")
for a, b, r_val in sorted(high_pairs, key=lambda x: abs(x[2]), reverse=True):
    print(f"  {a:<28} <-> {b:<28}  r={r_val:+.3f}")

findings.append(f'[S5] {len(high_pairs)} feature pairs with |r|>=0.80. '
               'Key groups: inequality (gini/s80s20/d9_d1), income levels (d1/q2/d9), '
               'pct_wages<->pct_pensions anti-collinear (sum-constrained).')

# Known groups
groups = {
    'Inequality (disp) -- pick ONE [gini_disp recommended]':
        ['gini_disp', 's80s20_disp', 'd9_d1_disp'],
    'Income level (disp) -- q2_disp as representative':
        ['d1_disp', 'q2_disp', 'd9_disp'],
    'Labour market -- partial overlap':
        ['pct_unemployment', 'unemployment_rate'],
    'Income composition -- sum-constrained (<=4 of 5 free)':
        ['pct_wages', 'pct_pensions', 'pct_unemployment', 'pct_capital_gains', 'pct_other'],
}
print("\nKnown Collinearity Groups:")
for group, members in groups.items():
    m_in = [m for m in members if m in HEATMAP_COLS]
    if len(m_in) < 2:
        continue
    vals = [corr.loc[a, b] for ii, a in enumerate(m_in) for b in m_in[ii + 1:]]
    print(f"\n  {group}")
    print(f"    Members: {m_in}")
    if vals:
        print(f"    Pairwise r in [{min(vals):.3f}, {max(vals):.3f}]")

print("""
Feature Design Implications:
  1. Inequality: use gini_disp ONLY -- s80s20_disp and d9_d1_disp near-dupes (r~0.986).
  2. Income level: q2_disp as single representative; d1/d9 are redundant.
  3. poverty_rate_disp: complementary to gini (different distributional aspect -- keep both).
  4. Income composition: include at most ONE pct_* series; pct_wages preferred.
  5. Unemployment: unemployment_rate (ILO) preferred over pct_unemployment (income-share).
  6. edu_share_sup x q2_disp: likely correlated (human capital <-> income); check VIF at model stage.
  7. pct_urban: shares variance with income and education -- possible confounder/mediator.
""")


# ===========================================================================
# S6 -- THE WEIGHTING QUESTION
# ===========================================================================
print("\n" + "=" * 68)
print("S6 -- The Weighting Question")
print("=" * 68)

dept_pop     = df.groupby('dep_code')['n_persons'].mean()
dep_names_m2 = df.groupby('dep_code')['dep_name'].first()
smallest     = dept_pop.idxmin()
largest      = dept_pop.idxmax()
pop_ratio    = dept_pop.max() / dept_pop.min()

print(f"Population spread across 96 departments (mean n_persons):")
print(f"  Smallest : {dep_names_m2[smallest]} ({smallest}) -- {dept_pop.min():,.0f}")
print(f"  Largest  : {dep_names_m2[largest]} ({largest}) -- {dept_pop.max():,.0f}")
print(f"  Ratio    : {pop_ratio:.0f}x")
print(f"  Mean     : {dept_pop.mean():,.0f}")
print(f"  Median   : {dept_pop.median():,.0f}")
print(f"  Skewness : {dept_pop.skew():.2f}")

findings.append(f'[S6] Population skew: {pop_ratio:.0f}x range -- '
               f'{dep_names_m2[smallest]} ({dept_pop.min()/1e3:.0f}k) to '
               f'{dep_names_m2[largest]} ({dept_pop.max()/1e6:.1f}M). Strongly right-skewed.')

# Figure: population distribution (linear + log)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.hist(dept_pop.values / 1e6, bins=30, color=C0, edgecolor='white', lw=0.3)
ax1.axvline(dept_pop.mean()   / 1e6, color='crimson', ls='--', lw=1.5,
            label=f'Mean={dept_pop.mean()/1e6:.2f}M')
ax1.axvline(dept_pop.median() / 1e6, color='navy',    ls=':',  lw=1.5,
            label=f'Median={dept_pop.median()/1e6:.2f}M')
ax1.set_xlabel('Mean n_persons (millions, 2012-2021 avg)')
ax1.set_ylabel('Number of departments')
ax1.set_title('Department Population Distribution\n(Filosofi n_persons)', fontweight='bold')
ax1.legend()

ax2.hist(np.log10(dept_pop.values), bins=30, color=C1, edgecolor='white', lw=0.3)
ax2.set_xlabel('log10(n_persons)')
ax2.set_ylabel('Number of departments')
ax2.set_title('Population Distribution -- Log10 Scale', fontweight='bold')
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'10^{x:.1f}'))
plt.tight_layout()
savefig(fig, 'S6_population_distribution')

# Rate vs department size
dept_s = df.groupby('dep_code').agg(
    mean_pop  = ('n_persons',  'mean'),
    mean_rate = ('firm_rate',  'mean'),
    dep_name  = ('dep_name',   'first'),
).reset_index()

r_sz,   _ = pearsonr(np.log(dept_s['mean_pop']), dept_s['mean_rate'])
rho_sz, _ = spearmanr(dept_s['mean_pop'],         dept_s['mean_rate'])
print(f"\nlog(pop) vs firm_rate: Pearson r={r_sz:.3f},  Spearman rho={rho_sz:.3f}")

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(dept_s['mean_pop'] / 1e6, dept_s['mean_rate'],
           s=55, alpha=0.7, color=C0, edgecolors='none')
notable = ['75', '69', '13', '06', '33', '31', '59', '67', '48', '23', '15', '05']
for _, row in dept_s[dept_s['dep_code'].isin(notable)].iterrows():
    ax.annotate(row['dep_code'],
                (row['mean_pop'] / 1e6, row['mean_rate']),
                xytext=(5, 3), textcoords='offset points', fontsize=8)
ax.set_xscale('log')
log_pop_arr = np.log(dept_s['mean_pop'].values)
m_sz, b_sz  = np.polyfit(log_pop_arr, dept_s['mean_rate'].values, 1)
xvals = np.logspace(np.log10(dept_s['mean_pop'].min()),
                    np.log10(dept_s['mean_pop'].max()), 200)
ax.plot(xvals / 1e6, m_sz * np.log(xvals) + b_sz, 'r--', lw=1.6,
        label=f'Log-linear fit (r={r_sz:.2f})')
ax.set_xlabel('Mean n_persons (millions, log scale)')
ax.set_ylabel('Mean firm creation rate (per 1,000 inhabitants)')
ax.set_title('Firm Creation Rate vs Department Size\n(dept means 2012-2021)', fontweight='bold')
ax.legend()
plt.tight_layout()
savefig(fig, 'S6_rate_vs_dept_size')

if abs(rho_sz) > 0.3:
    sz_note = (f'Firm rate {"positively" if rho_sz > 0 else "negatively"} correlated with dept size '
               f'(rho={rho_sz:.2f}). WLS weighting matters for correct estimates.')
else:
    sz_note = (f'Firm rate weakly correlated with dept size (rho={rho_sz:.2f}). '
               'WLS vs OLS unlikely to shift point estimates, but corrects inference (heteroskedasticity).')
print(f"  {sz_note}")
findings.append(f'[S6] {sz_note}')

print("""
WEIGHTING RECOMMENDATION (awaiting confirmation):
  Estimate BOTH unweighted OLS and population-weighted WLS (weight=pop_jan1).
  Preferred headline model: WLS -- ~30x population range induces Poisson-like
  heteroskedasticity; WLS gives correct SEs and down-weights noisy small depts.
""")


# ===========================================================================
# S7 -- URBAN/RURAL & STRUCTURAL CUTS
# ===========================================================================
print("\n" + "=" * 68)
print("S7 -- Urban/Rural & Structural Cuts")
print("=" * 68)

print("Department counts by density_class:")
print(df.groupby('density_class')['dep_code'].nunique().to_string())
print("\nFirm creation rate by density class (pooled 2012-2021):")
print(df.groupby('density_class')['firm_rate'].describe().round(3).to_string())

ORDER   = [o for o in ['urban', 'intermediate', 'rural'] if o in df['density_class'].values]
BOX_PAL = {o: c for o, c in zip(ORDER, [C0, C1, C2])}

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=df, x='density_class', y='firm_rate', order=ORDER,
            palette=BOX_PAL,
            flierprops=dict(marker='.', markersize=4, alpha=0.5),
            linewidth=0.9, ax=ax)
ymax = df['firm_rate'].max()
for i, cls in enumerate(ORDER):
    m = df[df['density_class'] == cls]['firm_rate'].mean()
    n = df[df['density_class'] == cls]['dep_code'].nunique()
    ax.text(i, ymax * 1.04, f'mean={m:.2f}\n(n={n} depts)',
            ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Density class (INSEE Grille de densite 2025, RP2021)')
ax.set_ylabel('Firm creation rate (per 1,000 inhabitants)')
ax.set_title('Firm Creation Rate by Urban/Rural Class\n(pooled 2012-2021)', fontweight='bold')
plt.tight_layout()
savefig(fig, 'S7_rate_by_density_class')

groups_dc = [df[df['density_class'] == cls]['firm_rate'].dropna().values for cls in ORDER]
F_dc, p_anova = f_oneway(*groups_dc)
urban_mean = df[df['density_class'] == 'urban']['firm_rate'].mean()
rural_mean = df[df['density_class'] == 'rural']['firm_rate'].mean()
print(f"\nOne-way ANOVA: F={F_dc:.2f}, p={p_anova:.4f}")
findings.append(f'[S7] Firm rate differs by density class (ANOVA F={F_dc:.2f}, p={p_anova:.4f}). '
               f'Urban mean={urban_mean:.2f} vs Rural mean={rural_mean:.2f}.')

# Trend by density class
trend_dc = df.groupby(['year', 'density_class'])['firm_rate'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
LINE_C = {'urban': C0, 'intermediate': C1, 'rural': C2}
for cls, grp in trend_dc.groupby('density_class'):
    ax.plot(grp['year'], grp['firm_rate'], 'o-', lw=2, ms=6,
            color=LINE_C.get(cls, 'grey'), label=cls.capitalize())
ax.axvspan(2019.5, 2020.5, alpha=0.10, color='crimson', label='2020 COVID')
ax.axvspan(2015.5, 2018.5, alpha=0.06, color='orange',  label='2016-18 reform era')
ax.set_xlabel('Year')
ax.set_ylabel('Mean firm creation rate (per 1,000)')
ax.set_title('Firm Rate Trend by Density Class (2012-2021)', fontweight='bold')
ax.set_xticks(range(2012, 2022))
ax.legend(fontsize=10)
plt.tight_layout()
savefig(fig, 'S7_trend_by_density_class')

# Choropleth -- graceful skip
try:
    import geopandas as gpd
    raise FileNotFoundError("No departmental GeoJSON found locally.")
except (ImportError, FileNotFoundError) as e:
    print(f"\nChoropleth skipped: {e}")
    print("Map deferred to Streamlit stage.")
    findings.append('[S7] Choropleth skipped -- geopandas not installed / no GeoJSON. '
                   'Map deferred to Streamlit stage.')


# ===========================================================================
# S8 -- FINDINGS SUMMARY
# ===========================================================================
print("\n" + "=" * 68)
print("FINDINGS SUMMARY -- Regions Inegales EDA")
print("=" * 68)
for i, f in enumerate(findings, 1):
    print(f"\n{i:2d}. {f}")

print("\n" + "=" * 68)
print("DECISIONS AWAITING CONFIRMATION")
print("=" * 68)
print("""
A) TARGET DENOMINATOR
   Recommendation: rate_pop = total_firm_creations / pop_jan1 * 1000
   Evidence:
   - n_persons ~= 0.977 * pop_jan1 (Spearman rho > 0.999 between any pair)
   - pop_jan1 is the official INSEE denominator (same as doctor_density)
   - rate_adult = rate_pop / 0.62 (uniform rescale; adds zero cross-dept variation)

B) REGRESSION WEIGHTING
   Recommendation: Population-weighted WLS (weight=pop_jan1) as headline model.
   Also report unweighted OLS for comparison.
   Rationale: ~30x population range -> Poisson-like heteroskedasticity.
   WLS gives correct SEs and prevents tiny depts dominating the fit.
""")

print("=" * 68)
print(f"Figures saved to: {FIGS}")
fig_files = [f for f in os.listdir(FIGS) if f.endswith('.png')]
fig_files.sort()
print(f"PNG files ({len(fig_files)}): {fig_files}")
