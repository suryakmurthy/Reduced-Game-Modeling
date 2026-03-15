import os
import numpy as np

from line_enumerator_go import GoLineEnumerator
from katago_evaluator import KataGoEvaluator
from payoff_matrix_go import (
    build_F_matrix,
    build_payoff_matrix,
    svd_analysis,
    plot_results,
)

os.makedirs("results_go", exist_ok=True)

CONFIGS = [
    # {"depth": 4, "top_k": 3, "label": "d4_k3"},
    # {"depth": 6, "top_k": 3, "label": "d6_k3"},
    {"depth": 6, "top_k": 4, "label": "d6_k4"},
]

evaluator = KataGoEvaluator(
    cache_file="results_go/go_cache.json",
    checkpoint_every=500
)
evaluator.start()

for cfg in CONFIGS:
    depth = cfg["depth"]
    top_k = cfg["top_k"]
    label = cfg["label"]

    print("\n" + "=" * 60)
    print(f"Config: depth={depth}, top_k={top_k}  [{label}]")
    print("=" * 60)

    # ── Enumerate lines ───────────────────────────────────────────
    print(f"Enumerating lines to depth {depth} (top_k={top_k})...")
    enumerator = GoLineEnumerator(depth=depth, top_k=top_k)
    lines = enumerator.enumerate()
    n = len(lines)
    print(f"  Found {n} lines.  (expected: {top_k**depth})")
    print(f"  Strategy space: {n} lines  ({n*n:,} pairs)")

    # Save lines
    lines_path = f"results_go/lines_{label}.txt"
    with open(lines_path, "w") as f:
        for line in lines:
            f.write(" ".join(line) + "\n")
    print(f"  Saved lines to {lines_path}")

    if n == 0:
        print("  No lines found — skipping config.")
        continue

    # ── Build F matrix ────────────────────────────────────────────
    F, valid_mask = build_F_matrix(lines, evaluator)

    np.save(f"results_go/F_{label}.npy", F)
    np.save(f"results_go/valid_mask_{label}.npy", valid_mask)

    print(f"  Valid strategy pairs (both directions legal): {valid_mask.sum():,}")
    print(f"  Payoff range:        [{F.min():.4f}, {F.max():.4f}]")
    print(f"  Mean absolute value: {np.abs(F - 0.5).mean():.4f}")

    # ── Build payoff matrix ───────────────────────────────────────
    A = build_payoff_matrix(F)
    np.save(f"results_go/A_{label}.npy", A)
    print(f"  Skew-symmetric check: {np.allclose(A, -A.T, atol=1e-4)}")

evaluator.stop()
print("\n" + "=" * 60)
print("All configs complete. Results saved to results_go/")
print("=" * 60)