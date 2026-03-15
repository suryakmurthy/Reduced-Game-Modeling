import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from solve_game_full import solve_full_lp_v_version
from solve_game_sampling import solve_reduced_lp_using_sampling
from solve_game_reduced import solve_reduced_lp_using_QU_vform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full and reduced LP solvers on a payoff matrix."
    )
    parser.add_argument(
        "input_matrix",
        type=str,
        help="Path to input .npy matrix file",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default="",
        help="Directory to save outputs/plots",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=200,
        help="Maximum k to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="highs-ipm",
        help="LP solver method",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=0,
        help="p parameter for solve_reduced_lp_using_QU_vform",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=3,
        help="q parameter for solve_reduced_lp_using_QU_vform",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save the plot to the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_matrix = args.input_matrix
    output_directory = args.output_directory
    k_max = args.k_max
    seed = args.seed
    method = args.method
    p = args.p
    q = args.q

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    F_raw = np.load(input_matrix)

    A = F_raw - F_raw.T
    F_true = -A

    print("Matrix shape:", F_true.shape)
    m, n = F_true.shape

    ts0 = time.perf_counter()
    res_full_one_side = solve_full_lp_v_version(F_true, method=method)
    t_full_one_side = time.perf_counter() - ts0

    if not res_full_one_side.success:
        raise RuntimeError(f"Full one-sided LP failed: {res_full_one_side.message}")

    y_full = res_full_one_side.x[:n].copy()
    y_full /= y_full.sum()

    col_lower_full = float(np.min(F_true @ y_full))

    print("\n=== Full solver (one side) diagnostics ===")
    print(f"col_lower_full = min(F @ y_full): {col_lower_full:.6f}")
    print(f"time: {t_full_one_side:.4f}s")

    k_vals = []

    col_lower_series = []
    times = []

    col_lower_series_sampling = []
    times_sampling = []

    for k in range(1, k_max + 1):
        print(f"Running k={k}")

        # ---- Schur baseline ----
        res, Qr, Ur, t_solve, t_setup, c_r = solve_reduced_lp_using_QU_vform(
            F_true, k, p=p, q=q, seed=seed, method=method
        )

        if not res.success:
            print(f"Schur FAIL at k={k}")
            print("status:", res.status)
            print("message:", res.message)
            continue

        # ---- Sampling baseline ----
        (
            res_sampling,
            yk_sampling,
            idx_sampling,
            t_setup_sampling,
            t_solve_sampling,
        ) = solve_reduced_lp_using_sampling(
            F_true, k, seed=seed, method=method
        )

        if (yk_sampling is None) or (not res_sampling.success):
            print(f"Sampling FAIL at k={k}: {res_sampling.message}")
            continue

        # ---- Evaluate Schur solution ----
        yk = res.x[:n].copy()
        s = yk.sum()
        if s <= 0:
            print(f"Schur returned nonpositive mass at k={k}")
            continue
        yk /= s

        col_lower = float(np.min(F_true @ yk))
        col_lower_series.append(col_lower)
        times.append(t_solve + t_setup)

        # ---- Evaluate sampled solution ----
        col_lower_sampling = float(np.min(F_true @ yk_sampling))
        col_lower_series_sampling.append(col_lower_sampling)
        times_sampling.append(t_setup_sampling + t_solve_sampling)

        k_vals.append(k)

    if len(k_vals) == 0:
        raise RuntimeError("No successful reduced runs completed.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reduced Rank LP — Solver Analysis", fontsize=13)

    ax = axes[0, 0]
    ax.plot(k_vals, col_lower_series, marker="o", markersize=3, label="Schur")
    ax.plot(
        k_vals,
        col_lower_series_sampling,
        marker="o",
        markersize=3,
        color="green",
        label="Sampling",
    )
    ax.axhline(
        col_lower_full,
        linewidth=1.5,
        linestyle="--",
        color="red",
        label="Full LP (one-side)",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Lower bound")
    ax.set_title("Exploitability Lower Bound")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(k_vals, times, marker="o", markersize=3, label="Schur solver")
    ax.plot(
        k_vals,
        times_sampling,
        marker="o",
        markersize=3,
        color="green",
        label="Sampling solver",
    )
    ax.axhline(
        t_full_one_side,
        linewidth=1.5,
        linestyle="--",
        color="red",
        label="Full solver",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Time (s)")
    ax.set_title("Solve Time vs k")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()


    if args.save_plot:
        plot_path = os.path.join(output_directory, "solver_analysis.png") \
            if output_directory else "solver_analysis.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved plot to {plot_path}")
    else:
        plt.show()

    if output_directory:
        print(f"\nSaved outputs in {output_directory}")


if __name__ == "__main__":
    main()