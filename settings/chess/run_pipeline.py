import numpy as np
import matplotlib.pyplot as plt
import os
import time

from old_version.settings.chess.line_enumerator import PolyglotLineEnumerator
from old_version.settings.chess.lc0_evaluator import LC0Evaluator
from old_version.settings.chess.payoff_matrix import build_payoff_matrix, svd_analysis, nash_equilibrium

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

BOOK_PATH    = "gm2001.bin"
LC0_NODES    = 1
OUTPUT_DIR   = "results"
CACHE_FILE   = "lc0_cache.json"  # shared across all runs

CONFIGS = [
    {"depth": 6,  "min_weight": 10,  "label": "d6_mw10"},
    {"depth": 8,  "min_weight": 10,  "label": "d8_mw10"},
    {"depth": 10, "min_weight": 10, "label": "d10_mw10"},
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Chess Opening Payoff Matrix Pipeline")
print("=" * 60)

with LC0Evaluator(
    nodes=LC0_NODES,
    checkpoint_file=CACHE_FILE,
    checkpoint_every=10000,
) as evaluator:

    for cfg in CONFIGS:
        depth      = cfg["depth"]
        min_weight = cfg["min_weight"]
        label      = cfg["label"]

        print(f"\n{'='*60}")
        print(f"Config: depth={depth}, min_weight={min_weight}  [{label}]")
        print(f"{'='*60}")

        # ── Step 1: Enumerate lines ────────────────────────────────
        enumerator = PolyglotLineEnumerator(BOOK_PATH, min_weight=min_weight)
        lines = enumerator.enumerate(depth=depth)
        n = len(lines)
        print(f"Strategy space: {n} lines  ({n*n:,} pairs)")

        # ── Step 2: Build payoff matrix ────────────────────────────
        t0 = time.time()
        A, F, valid_mask = build_payoff_matrix(lines, evaluator, fill_illegal=0.5)
        elapsed = time.time() - t0
        print(f"  Build time: {elapsed/60:.1f} min")

        # Save matrices
        np.save(f"{OUTPUT_DIR}/A_{label}.npy",          A)
        np.save(f"{OUTPUT_DIR}/F_{label}.npy",          F)
        np.save(f"{OUTPUT_DIR}/valid_mask_{label}.npy", valid_mask)

        # Save lines list as text
        with open(f"{OUTPUT_DIR}/lines_{label}.txt", "w") as f:
            for line in lines:
                f.write(" ".join(line) + "\n")

        print(f"  Saved matrices to {OUTPUT_DIR}/")

print(f"\n{'='*60}")
print("All configs complete.")
print(f"{'='*60}")
