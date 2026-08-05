"""
Shared style + verification helpers for the paper/figures/*.pdf scripts.

Every fig script must:
  1. build its own figure at the FINAL print size (figsize in inches ==
     final size, no post-hoc scaling), so font sizes set in points are
     the true final font sizes.
  2. call report_fig(fig, out_path) at the end, which saves the vector
     PDF and prints the measured (not assumed) dimensions and smallest
     rendered font size, so a too-small label is caught before the PDF
     ships to LaTeX.
"""
import os
import matplotlib
import matplotlib.text as mtext

IEEE_COL_WIDTH = 3.5    # inches, single IEEE column
IEEE_PAGE_WIDTH = 7.16  # inches, full two-column spread
MIN_FONT_PT = 8.0

# Colorblind-safe, print-legible palette (Okabe-Ito / Paul Tol derived).
# No red/green pairing anywhere.
COLOR_OPPORTUNITY = "#0072B2"   # blue
COLOR_NECESSITY   = "#D55E00"   # vermillion (distinguishable from blue in
                                 # both CVD and greyscale by lightness, not
                                 # just hue)
COLOR_OTHER       = "#999999"   # neutral grey

COLOR_UNWEIGHTED = "#0072B2"    # blue
COLOR_WEIGHTED   = "#E69F00"    # orange, high lightness contrast from blue

COLOR_WITHIN  = "#4477AA"       # dark blue, low lightness
COLOR_BETWEEN = "#CCBB44"       # light khaki, high lightness
                                 # (large lightness delta -> greyscale-legible)

SEQUENTIAL_CMAP = "YlGnBu"      # ColorBrewer-certified colorblind-safe


def _iter_texts(fig):
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if s and s.strip():
            yield t


def report_fig(fig, out_path, extra_note=None):
    """Save fig as vector PDF at its current figsize; print measured
    final dimensions and smallest rendered font size (pt) for a
    print-check against the 8pt-minimum requirement."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches=None)

    w_in, h_in = fig.get_size_inches()
    sizes = [t.get_fontsize() for t in _iter_texts(fig)]
    min_pt = min(sizes) if sizes else float("nan")

    print(f"[{out_path}]")
    print(f"  size: {w_in:.3f} in x {h_in:.3f} in")
    print(f"  smallest rendered font: {min_pt:.2f} pt "
          f"({'OK' if min_pt >= MIN_FONT_PT else 'FAIL: below 8pt minimum'})")
    if extra_note:
        print(f"  note: {extra_note}")
    print()
    return w_in, h_in, min_pt
