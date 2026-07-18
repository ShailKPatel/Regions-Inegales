"""
Number-provenance checker for FINDINGS.md.

Extracts every decimal number quoted in FINDINGS.md and checks whether a
matching number (within rounding tolerance) exists somewhere in the
generated model/findings_*.md files it is supposed to summarize. This is a
smoke test for transcription errors like reporting -0.185 when the source
file says -18.51 (see AUDIT_REPORT.md finding C-4) -- it is not a proof of
correctness, since some numbers in FINDINGS.md are legitimately derived
(sums, ratios, rounded restatements) and won't appear verbatim in any
source file. Anything flagged UNMATCHED needs a human (or the next Claude)
to manually confirm it traces back to a real computation before merging.

Usage: python scripts/check_findings_numbers.py
"""

import re
import glob

FINDINGS_PATH = "FINDINGS.md"
SOURCE_GLOBS = [
    "model/findings_*.md",
    "model/split_findings.md",
    "model/temporal_findings.md",
]

NUMBER_RE = re.compile(r"[+-]?\d+\.\d+(?:e[+-]?\d+)?")

# Lines/contexts that are legitimately derived (sums, ratios, ranks) rather
# than a single traceable source-file number -- exclude by line substring.
DERIVED_LINE_MARKERS = [
    "together account for",
    "make up the remaining",
    "ratio",
    "×",
    "x more important",
]

# Numbers verified by hand against a source OUTSIDE model/findings_*.md
# (e.g. DATA_SOURCES.md crosscheck CSVs, or a ratio recomputed from
# full-precision shares that a *.md table only prints pre-rounded). Each
# entry is (line-content substring, reason) -- keep this list short; if it
# grows, that's a sign numbers should be sourced programmatically instead.
KNOWN_EXCEPTIONS = [
    ("85.2% exact match", "sourced from sources/_unemployment_bdm_crosscheck.csv, not model/findings_*.md"),
    ("1.95x", "54%/27% urban SHAP shares are pre-rounded to 1dp in split_findings.md; "
              "1.95 is the ratio of the unrounded underlying shares"),
]


def extract_numbers(text: str) -> list[float]:
    return [float(m) for m in NUMBER_RE.findall(text)]


def close(a: float, b: float, rel_tol: float = 0.01, abs_tol: float = 0.0055) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def main() -> int:
    with open(FINDINGS_PATH, encoding="utf-8") as fh:
        findings_lines = fh.readlines()

    source_text = ""
    for pattern in SOURCE_GLOBS:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as fh:
                source_text += f"\n# {path}\n" + fh.read()
    source_numbers = extract_numbers(source_text)

    unmatched = []
    for lineno, line in enumerate(findings_lines, start=1):
        if any(marker in line for marker in DERIVED_LINE_MARKERS):
            continue
        if any(marker in line for marker, _ in KNOWN_EXCEPTIONS):
            continue
        for n in extract_numbers(line):
            if not any(close(n, s) for s in source_numbers):
                unmatched.append((lineno, n, line.strip()))

    print(f"Checked {sum(len(extract_numbers(l)) for l in findings_lines)} numeric "
          f"tokens across {len(findings_lines)} lines of {FINDINGS_PATH} "
          f"against {len(source_numbers)} numbers pooled from "
          f"{sum(len(glob.glob(g)) for g in SOURCE_GLOBS)} source files.\n")

    if not unmatched:
        print("PASS: every non-derived decimal number in FINDINGS.md has a "
              "matching value in the model/findings_*.md source files.")
        return 0

    print(f"UNMATCHED: {len(unmatched)} number(s) in FINDINGS.md have no "
          f"close match in any source findings file. Manually verify each:\n")
    for lineno, n, line in unmatched:
        print(f"  FINDINGS.md:{lineno}  value={n}\n    {line}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
