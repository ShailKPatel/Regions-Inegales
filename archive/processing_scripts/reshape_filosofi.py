"""
Régions Inégales - Filosofi Reshape
Converts the year-siloed wide CSV into a clean 960×29 panel.
"""

import pathlib
import pandas as pd
import numpy as np

CLEAN = pathlib.Path(__file__).resolve().parents[2] / "filosofi_clean"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading filosofi_all_years.csv …")
df = pd.read_csv(CLEAN / "filosofi_all_years.csv", sep=";",
                 dtype={"CODGEO": str}, low_memory=False)
df["YEAR"] = df["YEAR"].astype(int)
print(f"  {len(df)} rows × {len(df.columns)} columns loaded.\n")

# ---------------------------------------------------------------------------
# Mapping: (source_pattern, target_name)
# Patterns with {YY} are expanded per year; patterns without are used as-is.
# Order matters: first match that finds a non-null column wins per target.
# ---------------------------------------------------------------------------
MAPPING = [
    # Identifiers
    ("CODGEO",              "dep_code"),
    ("YEAR",                "year"),
    ("LIBGEO",              "dep_name"),
    # Counts, 2012 bare, 2013+ _DEC suffix
    ("NBMEN{YY}",           "n_households"),
    ("NBMEN{YY}_DEC",       "n_households"),
    ("NBPERS{YY}",          "n_persons"),
    ("NBPERS{YY}_DEC",      "n_persons"),
    ("NBUC{YY}_DEC",        "n_uc"),
    # DEC income indicators
    ("Q1{YY}_DEC",          "q1_dec"),
    ("Q2{YY}_DEC",          "q2_dec"),
    ("Q3{YY}_DEC",          "q3_dec"),
    ("D1{YY}_DEC",          "d1_dec"),
    ("D9{YY}_DEC",          "d9_dec"),
    ("GI{YY}_DEC",          "gini_dec"),
    ("S80S20{YY}_DEC",      "s80s20_dec"),   # exact string replace, safe
    ("RD_DEC",              "d9_d1_dec"),    # no year suffix, same col every year
    # DISP income indicators ★
    ("Q1{YY}_DISP",         "q1_disp"),
    ("Q2{YY}_DISP",         "q2_disp"),
    ("Q3{YY}_DISP",         "q3_disp"),
    ("D1{YY}_DISP",         "d1_disp"),
    ("D9{YY}_DISP",         "d9_disp"),
    ("GI{YY}_DISP",         "gini_disp"),
    ("S80S20{YY}_DISP",     "s80s20_disp"),
    ("RD_DISP",             "d9_d1_disp"),   # no year suffix
    # Poverty rate
    # NOTE: poverty_rate_disp for 2012 uses TP60{YY} (bare/DEC-only source;
    # no DISP Pauvres file existed for 2012). Treat with caution in cross-year
    # comparisons. 2013–2021 uses the genuine DISP Pauvres file.
    ("TP60{YY}",            "poverty_rate_disp"),   # 2012 only
    ("TP60{YY}_PAU_DISP",   "poverty_rate_disp"),   # 2013–2021 ★
    ("TP60{YY}_PAU_DEC",    "poverty_rate_dec"),
    # Income composition (2012 bare, 2013+ _DEC)
    ("PTSA{YY}",            "pct_wages"),
    ("PTSA{YY}_DEC",        "pct_wages"),
    ("PCHO{YY}",            "pct_unemployment"),
    ("PCHO{YY}_DEC",        "pct_unemployment"),
    ("PBEN{YY}_DEC",        "pct_capital_gains"),
    ("PPEN{YY}_DEC",        "pct_pensions"),
    ("PAUT{YY}",            "pct_other"),
    ("PAUT{YY}_DEC",        "pct_other"),
]

TARGET_COLS = [
    "dep_code", "year", "dep_name",
    "n_households", "n_persons", "n_uc",
    "q1_dec", "q2_dec", "q3_dec", "d1_dec", "d9_dec", "gini_dec", "s80s20_dec", "d9_d1_dec",
    "q1_disp", "q2_disp", "q3_disp", "d1_disp", "d9_disp", "gini_disp", "s80s20_disp", "d9_d1_disp",
    "poverty_rate_disp", "poverty_rate_dec",
    "pct_wages", "pct_unemployment", "pct_capital_gains", "pct_pensions", "pct_other",
]

# ---------------------------------------------------------------------------
# Department name lookup (fills dep_name for 2018–2021 where LIBGEO is absent)
# ---------------------------------------------------------------------------
DEP_NAMES = {
    "01": "Ain", "02": "Aisne", "03": "Allier",
    "04": "Alpes-de-Haute-Provence", "05": "Hautes-Alpes", "06": "Alpes-Maritimes",
    "07": "Ardèche", "08": "Ardennes", "09": "Ariège", "10": "Aube",
    "11": "Aude", "12": "Aveyron", "13": "Bouches-du-Rhône", "14": "Calvados",
    "15": "Cantal", "16": "Charente", "17": "Charente-Maritime", "18": "Cher",
    "19": "Corrèze", "2A": "Corse-du-Sud", "2B": "Haute-Corse",
    "21": "Côte-d'Or", "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne",
    "25": "Doubs", "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir",
    "29": "Finistère", "30": "Gard", "31": "Haute-Garonne", "32": "Gers",
    "33": "Gironde", "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre",
    "37": "Indre-et-Loire", "38": "Isère", "39": "Jura", "40": "Landes",
    "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire",
    "44": "Loire-Atlantique", "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne",
    "48": "Lozère", "49": "Maine-et-Loire", "50": "Manche", "51": "Marne",
    "52": "Haute-Marne", "53": "Mayenne", "54": "Meurthe-et-Moselle",
    "55": "Meuse", "56": "Morbihan", "57": "Moselle", "58": "Nièvre",
    "59": "Nord", "60": "Oise", "61": "Orne", "62": "Pas-de-Calais",
    "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques", "65": "Hautes-Pyrénées",
    "66": "Pyrénées-Orientales", "67": "Bas-Rhin", "68": "Haut-Rhin",
    "69": "Rhône", "70": "Haute-Saône", "71": "Saône-et-Loire", "72": "Sarthe",
    "73": "Savoie", "74": "Haute-Savoie", "75": "Paris", "76": "Seine-Maritime",
    "77": "Seine-et-Marne", "78": "Yvelines", "79": "Deux-Sèvres", "80": "Somme",
    "81": "Tarn", "82": "Tarn-et-Garonne", "83": "Var", "84": "Vaucluse",
    "85": "Vendée", "86": "Vienne", "87": "Haute-Vienne", "88": "Vosges",
    "89": "Yonne", "90": "Territoire de Belfort", "91": "Essonne",
    "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis", "94": "Val-de-Marne",
    "95": "Val-d'Oise",
}

# ---------------------------------------------------------------------------
# Reshape: year-slice → rename → stack
# ---------------------------------------------------------------------------
print("Reshaping year slices …\n")
year_slices = []

for year in range(2012, 2022):
    yy = str(year)[2:]
    slice_df = df[df["YEAR"] == year].copy().reset_index(drop=True)

    # Build one dict per target, first-match-wins (no loc assignment issues)
    result: dict[str, object] = {}

    for pattern, target in MAPPING:
        if target in result:
            continue                               # already filled by an earlier mapping
        src_col = pattern.replace("{YY}", yy)      # plain string replace, no regex
        if src_col not in slice_df.columns:
            continue
        if target in ("dep_code", "dep_name"):
            result[target] = slice_df[src_col].values
        else:
            # Handle both period (2012–2019) and comma (2020–2021 INSEE format change)
            # as decimal separators before numeric conversion.
            cleaned = slice_df[src_col].astype(str).str.replace(",", ".", regex=False)
            result[target] = pd.to_numeric(cleaned, errors="coerce").values

    # Identifiers always come from the slice directly
    result["dep_code"] = slice_df["CODGEO"].values
    result["year"]     = year

    # Build the output DataFrame; missing targets filled with NaN
    out = pd.DataFrame(index=range(len(slice_df)))
    for tgt in TARGET_COLS:
        out[tgt] = result.get(tgt, np.nan)

    non_null = out.notna().sum().sum()
    print(f"  {year}: {len(out)} rows | {non_null} non-null cells "
          f"| missing targets: "
          f"{[c for c in TARGET_COLS if out[c].isna().all()]}")

    year_slices.append(out)

# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------
panel = pd.concat(year_slices, ignore_index=True)

# Fill dep_name from lookup for years where LIBGEO was absent
panel["dep_name"] = panel["dep_name"].fillna(
    panel["dep_code"].str.strip().str.upper().map(DEP_NAMES)
)

# Ensure dep_code is clean string
panel["dep_code"] = panel["dep_code"].str.strip().str.upper()

# Enforce target column order
panel = panel[TARGET_COLS]

# ---------------------------------------------------------------------------
# Final checks
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("FINAL CHECKS")
print("=" * 72)

print(f"\n  Shape: {panel.shape}  (expect ~960 rows × 29 cols)")

print("\n  Rows per year:")
for y, n in panel["year"].value_counts().sort_index().items():
    flag = " ← WARNING" if n != 96 else ""
    print(f"    {y}: {n} rows{flag}")

print("\n  Null count per column:")
null_counts = panel.isna().sum()
for col, n in null_counts.items():
    pct = n / len(panel) * 100
    flag = " ← WARNING high" if pct > 50 else ""
    print(f"    {col:<25s}: {n:>4} nulls ({pct:5.1f}%){flag}")

# Key variable range checks
checks = [
    ("q2_disp",          10_000, 40_000, "median disposable income (€)"),
    ("gini_disp",         0.20,   0.50,  "Gini disposable"),
    ("poverty_rate_disp", 5.0,   35.0,   "poverty rate 60% (%)"),
]
print("\n  Key variable range checks (by year):")
for col, lo, hi, label in checks:
    if col not in panel.columns or panel[col].isna().all():
        print(f"\n  {col}: WARNING all null")
        continue
    print(f"\n  {col}, {label}  [expected {lo}–{hi}]")
    yr_stats = (panel.groupby("year")[col]
                .agg(["min", "max", "mean"])
                .round(1))
    for yr, row in yr_stats.iterrows():
        flag = ""
        if row["min"] < lo or row["max"] > hi:
            flag = f"  ← WARNING OUTSIDE [{lo}–{hi}]"
        print(f"    {yr}: min={row['min']:>8}  max={row['max']:>8}  mean={row['mean']:>8}{flag}")

print("\n  First 5 rows:")
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)
print(panel.head(5)[["dep_code", "dep_name", "year",
                      "q2_disp", "gini_disp", "poverty_rate_disp",
                      "d1_disp", "d9_disp", "n_households"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = CLEAN / "filosofi_panel.csv"
panel.to_csv(out_path, sep=";", encoding="utf-8", index=False)
print(f"\n  Saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")
print("\nReshape complete.")
