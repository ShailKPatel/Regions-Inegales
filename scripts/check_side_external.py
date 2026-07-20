"""
External anchor check for the SIDE firm-creation source (H-1 close-out).

Compares our metropole (96-dept) annual sums of total_firm_creations against
published INSEE national totals in sources/_side_external_anchor.csv.

Two vintages exist for INSEE's creation totals: the original INSEE Premiere
headline (published ~1-2 months after year end) and the restated total shown
in later editions. INSEE revises creation counts upward for 2-5 years after
first release as late SIRENE registrations get backdated to their true
creation date. The original headline for 2018-2021 undershoots our panel by
3-8%, inconsistent in both size and even sign (2012 is negative); the same
years' restated totals (2018-2021 independently corroborated against INSEE
Premiere n1984, Feb 2024) sit 2.6-3.6% ABOVE our metropole sum every year,
consistent with a stable France-entiere DOM/Mayotte share. This script
re-derives that classification from the anchor CSV, it does not assert it.

Classification rules (must match the anchor CSV's own `classification` column):
  MATCH:       |ours - published| / published <= 0.5%
  CONSISTENT:  published > ours, implied overseas share in [1%, 6%],
               and that share's spread across all rows in the file <= 2pp
  FAIL:        anything else

Run: python scripts/check_side_external.py
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from panel_config import PANEL_START, PANEL_END

import pandas as pd

ANCHOR_PATH = "sources/_side_external_anchor.csv"
PANEL_PATH = "sources/firm_creations_side.csv"

anchor = pd.read_csv(ANCHOR_PATH, sep=";")
panel = pd.read_csv(PANEL_PATH, sep=";", dtype={"dep_code": str})
panel = panel[(panel["year"] >= PANEL_START) & (panel["year"] <= PANEL_END)]

panel_sums = panel.groupby("year")["total_firm_creations"].sum()

shares = (anchor["published_value"] - anchor["our_sum"]) / anchor["published_value"] * 100
share_spread = shares.max() - shares.min()

results = []
for _, row in anchor.iterrows():
    year = int(row["year"])
    panel_sum = int(panel_sums.loc[year])
    assert panel_sum == row["our_sum"], (
        f"{year}: panel sum {panel_sum} != anchor file's recorded our_sum {row['our_sum']}"
    )

    pct_diff = (row["our_sum"] - row["published_value"]) / row["published_value"] * 100
    share = (row["published_value"] - row["our_sum"]) / row["published_value"] * 100

    if abs(pct_diff) <= 0.5:
        verdict = "MATCH"
    elif row["published_value"] > row["our_sum"] and 1.0 <= share <= 6.0 and share_spread <= 2.0:
        verdict = "CONSISTENT"
    else:
        verdict = "FAIL"

    results.append({
        "year": year,
        "our_sum": row["our_sum"],
        "published_value": row["published_value"],
        "pct_diff": round(pct_diff, 2),
        "overseas_share_pct": round(share, 2),
        "verdict": verdict,
        "matches_recorded": verdict == row["classification"],
    })

out = pd.DataFrame(results)
print(out.to_string(index=False))
print()
print(f"Overseas share range: {shares.min():.2f}% - {shares.max():.2f}%, spread {share_spread:.2f}pp")
print(f"Verdicts: {out['verdict'].value_counts().to_dict()}")
n_mismatch = (~out["matches_recorded"]).sum()
if n_mismatch:
    print(f"WARNING: {n_mismatch} row(s) disagree with the anchor CSV's recorded classification.")
    sys.exit(1)
else:
    print("All re-derived verdicts match the anchor CSV's recorded classification.")
