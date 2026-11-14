# -*- coding: utf-8 -*-
# Item 29: replace non-ASCII math chars in fig2/fig3 SVG labels with ASCII-safe text.
import os
import cairosvg

D = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-01-01\outputs\figures"

# char -> ascii label
MAP2 = {
    "\u2192": "->",   # →
    "\u00d7": "x",    # ×
    "\u00b2": "^2",   # ²
    "\u03a3": "Sum",  # Σ
    "\u2207": "grad", # ∇
    "\u03bb": "lambda",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b5": "eps",
    "\u0177": "y_hat",# ŷ
}
MAP3 = {
    "\u0394": "Delta",# Δ
    "\u03c4": "tau",  # τ
    "\u2265": ">=",   # ≥
}

for fname, mapping in [("fig2_architecture.svg", MAP2), ("fig3_duallayer.svg", MAP3)]:
    p = os.path.join(D, fname)
    txt = open(p, encoding="utf-8").read()
    before = sorted(set(ch for ch in txt if ord(ch) > 127))
    for ch, repl in mapping.items():
        txt = txt.replace(ch, repl)
    # verify no non-ascii remains
    after = sorted(set(ch for ch in txt if ord(ch) > 127))
    open(p, "w", encoding="utf-8").write(txt)
    print("%s: %s -> %s" % (fname, before, after))
    # regenerate PDF
    pdf = os.path.join(D, fname.replace(".svg", ".pdf"))
    cairosvg.svg2pdf(url=p, write_to=pdf)
    print("  regenerated", pdf, os.path.getsize(pdf), "bytes")
