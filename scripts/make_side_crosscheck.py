"""
Internal-consistency crosscheck for the SIDE firm-creation source.

Unlike unemployment (sources/_unemployment_bdm_crosscheck.csv) and doctor
density (sources/_doctor_density_pop_crosscheck.csv), the firm-creation
source -- which supplies total_firm_creations, the numerator of the model's sole
target variable firm_rate -- had no reproducible verification artifact at all.

This script cannot reach INSEE's servers from this environment, so it does not
attempt to fetch an external reference series the way the unemployment BDM check
does. What it does do, and commits as an artifact so the claim is checkable
without re-running anything: verify, per row, that total_firm_creations equals
both (a) the sum of its own legal-form breakdown and (b) the sum of its own
sector breakdown, across all 960 rows and all 10 years including the
2016-2018 auto-entrepreneur counting-rule reform window. This is a real,
reproducible check of the SIDE data's internal arithmetic -- it is NOT a
substitute for an external cross-reference against published INSEE Premiere
national annual figures, which remains an open item (see DATA_SOURCES.md
Source 2).

Writes: sources/_side_firm_creations_crosscheck.csv
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from panel_config import PANEL_START, PANEL_END

import pandas as pd

SRC_PATH = "sources/firm_creations_side.csv"
OUT_PATH = "sources/_side_firm_creations_crosscheck.csv"

df = pd.read_csv(SRC_PATH, sep=";", dtype={"dep_code": str})
df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)

legal_sum = (df["creations_individual"] + df["creations_sarl"]
             + df["creations_sas"] + df["creations_other_legal"])
sector_sum = (df["creations_sector_industry"] + df["creations_sector_construction"]
              + df["creations_sector_trade"] + df["creations_sector_services"])

out = pd.DataFrame({
    "dep_code": df["dep_code"],
    "year": df["year"],
    "total_firm_creations": df["total_firm_creations"],
    "legal_form_sum": legal_sum,
    "legal_form_diff": df["total_firm_creations"] - legal_sum,
    "sector_sum": sector_sum,
    "sector_diff": df["total_firm_creations"] - sector_sum,
})
out["status"] = out.apply(
    lambda r: "EXACT" if r["legal_form_diff"] == 0 and r["sector_diff"] == 0 else "MISMATCH",
    axis=1,
)
out.to_csv(OUT_PATH, sep=";", index=False)

n_rows = len(out)
n_exact = (out["status"] == "EXACT").sum()
print(f"Rows checked: {n_rows}")
print(f"Legal-form breakdown exact matches: {(out['legal_form_diff'] == 0).sum()}/{n_rows}, "
      f"max |diff| = {out['legal_form_diff'].abs().max()}")
print(f"Sector breakdown exact matches: {(out['sector_diff'] == 0).sum()}/{n_rows}, "
      f"max |diff| = {out['sector_diff'].abs().max()}")
print(f"Overall EXACT rows: {n_exact}/{n_rows}")
print()
print("National annual totals (sanity check, no external reference available "
      "in this environment -- compare by hand against INSEE Premiere bulletins "
      "'Demographie des entreprises et des etablissements' for each year):")
print(df.groupby("year")["total_firm_creations"].sum().to_string())
print()
print(f"Written: {OUT_PATH}")
