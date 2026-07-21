"""
Number-provenance checker for app/pages/*.py.

Sibling to check_findings_numbers.py. Extracts every decimal number quoted
in the Streamlit app's prose (markdown strings) and in hardcoded claim data
(e.g. SHAP-value tuples used to draw charts), then checks each one against
a matching value somewhere in FINDINGS.md, model/findings_*.md,
model/temporal_findings.md, or DATA_SOURCES.md.

Two extraction paths, since the app stores numbers two ways:
  1. Inside string literals (markdown prose, source-card descriptions) --
     scanned with a regex, CSS units (rem/px/em/pt) excluded since those
     are layout, not claims.
  2. As bare Python float literals (e.g. `("Median income", "Opportunity",
     1.1140)` feeding a chart) -- scanned via tokenize, skipping numbers
     that are the value of a known cosmetic/layout kwarg (opacity=,
     width=, x=, y=, ...), since those are chart styling, not data.

Not a proof of correctness (rounding, sign restatement, and legitimately
derived ratios can miss); anything UNMATCHED needs manual confirmation.

Usage: python scripts/check_app_numbers.py
"""

import ast
import glob
import re
import tokenize

APP_GLOB = "app/pages/*.py"
SOURCE_GLOBS = [
    "FINDINGS.md",
    "model/findings_*.md",
    "model/split_findings.md",
    "model/temporal_findings.md",
    "DATA_SOURCES.md",
]

NUMBER_RE = re.compile(r"[+-]?\d+\.\d+(?:e[+-]?\d+)?")
CSS_UNIT_RE = re.compile(r"[+-]?\d+\.\d+(?:e[+-]?\d+)?(?!\s*(?:rem|px|em|pt)\b)")

# Kwarg names whose numeric value is chart styling/layout, not a data claim.
STYLE_KWARGS = {
    "opacity", "size", "width", "len", "thickness", "height",
    "borderwidth", "borderpad", "ax", "ay", "arrowhead",
    "l", "r", "t", "b", "line_width", "marker_line_width",
    "font_size", "x", "y", "bargap",
}

# (file, lineno, reason) for bare literals that survive the STYLE_KWARGS
# filter but are still not data claims -- kept short and justified, same
# convention as check_findings_numbers.py's KNOWN_EXCEPTIONS.
LITERAL_EXCEPTIONS = {
    ("app/pages/map.py", 111): "choropleth colorscale gradient stop, not a data value",
}

# (file, line-content substring, reason) for string-embedded numbers that
# are legitimately not in the findings pool -- computed directly from the
# raw panel CSV as a descriptive stat, not a model finding.
KNOWN_EXCEPTIONS = [
    ("app/pages/overview.py", "Paris (75) averages",
     "descriptive stat computed directly from merged/france_panel_master.csv, not a model finding"),
]


def close(a: float, b: float, rel_tol: float = 0.01, abs_tol: float = 0.0055) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def extract_source_numbers() -> list[float]:
    text = ""
    for pattern in SOURCE_GLOBS:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as fh:
                text += f"\n# {path}\n" + fh.read()
    return [float(m) for m in NUMBER_RE.findall(text)]


def extract_app_numbers(path: str) -> list[tuple[int, float, str]]:
    """Returns (lineno, value, context) for every claim-like decimal in path."""
    found = []
    with open(path, "rb") as fh:
        tokens = list(tokenize.tokenize(fh.readline))

    prev1 = prev2 = None
    for tok in tokens:
        if tok.type == tokenize.NUMBER and "." in tok.string:
            kw = (
                prev2.string
                if prev2 and prev2.type == tokenize.NAME and prev1 and prev1.string == "="
                else None
            )
            if kw not in STYLE_KWARGS and (path, tok.start[0]) not in LITERAL_EXCEPTIONS:
                found.append((tok.start[0], float(tok.string), tok.line.strip()))

        elif tok.type == tokenize.STRING:
            try:
                decoded = ast.literal_eval(tok.string)
            except Exception:
                decoded = None
            if isinstance(decoded, str):
                for m in CSS_UNIT_RE.finditer(decoded):
                    found.append((tok.start[0], float(m.group()), decoded.strip()[:100]))

        if tok.type not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            prev2, prev1 = prev1, tok

    return found


def main() -> int:
    source_numbers = extract_source_numbers()

    unmatched = []
    total = 0
    for path in sorted(glob.glob(APP_GLOB)):
        for lineno, value, context in extract_app_numbers(path):
            total += 1
            if any(marker in context for _, marker, _ in KNOWN_EXCEPTIONS):
                continue
            if not any(close(value, s) for s in source_numbers):
                unmatched.append((path, lineno, value, context))

    print(f"Checked {total} numeric tokens across {len(glob.glob(APP_GLOB))} files in "
          f"{APP_GLOB} against {len(source_numbers)} numbers pooled from "
          f"{sum(len(glob.glob(g)) for g in SOURCE_GLOBS)} source files.\n")

    if not unmatched:
        print("PASS: every claim-like decimal number in app/pages/*.py has a "
              "matching value in FINDINGS.md / model/findings_*.md / DATA_SOURCES.md.")
        return 0

    print(f"UNMATCHED: {len(unmatched)} number(s) in app/pages/*.py have no "
          f"close match in any source file. Manually verify each:\n")
    for path, lineno, value, context in unmatched:
        print(f"  {path}:{lineno}  value={value}\n    {context}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
