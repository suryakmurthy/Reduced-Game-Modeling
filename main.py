import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from old_version.solve_game_full import solve_full_lp_v_version
from old_version.solve_game_sampling import solve_reduced_lp_using_sampling
from old_version.solve_game_reduced import solve_reduced_lp_using_QU_vform as solve_reduced_lp_using_QU_vform
from old_version.gpu_ver_2 import solve_sparse_factored_lp, solve_sparse_factored_lp_saved_factors_scipy
from old_version.solve_game_reduced_svd import solve_reduced_lp_using_svd_vform
from old_version.double_oracle import solve_double_oracle


import csv

def save_to_csv(filename, k_vals, data_series, labels):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k"] + labels)
        for i, k in enumerate(k_vals):
            row = [k]
            for series in data_series:
                row.append(series[i] if i < len(series) else "")
            writer.writerow(row)
    print(f"Saved CSV to {filename}")


def save_sampling_runs_to_csv(filename, records):
    """
    Save every individual sampling trial.
    records: list of dicts with keys k, trial, seed, col_lower, time
    """
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "trial", "seed", "col_lower", "time"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved per-run CSV to {filename}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full and reduced LP solvers on a payoff matrix."
    )
    parser.add_argument("--input_matrix", default="settings/chess/F_d6_mw10.npy", type=str)
    parser.add_argument("--output-directory", default="results/chess")
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=5,
                        help="Number of sampling runs per k value")
    parser.add_argument("--method", type=int, default=1)
    parser.add_argument("--p", type=int, default=0)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--skip-double-oracle", action="store_true")
    parser.add_argument("--save-plot", action="store_true")
    parser.add_argument("--skip-schur", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--use-stored", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    input_matrix = args.input_matrix
    output_directory = args.output_directory
    skip_schur = args.skip_schur
    k_max = args.k_max
    seed = args.seed
    method = args.method
    p = args.p
    q = args.q
    num_trials = args.num_trials

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    F_raw = np.load(input_matrix)
    A = F_raw
    print(np.linalg.norm(A))
    F_true = -A

    print("Matrix shape:", F_true.shape)
    m, n = F_true.shape

    # ------------------------------------------------------------------ #
    #  Full LP                                                             #
    # ------------------------------------------------------------------ #
    if not args.skip_full:
        ts0 = time.perf_counter()
        res_full_one_side = solve_full_lp_v_version(F_true, method=method)
        t_full_one_side = time.perf_counter() - ts0

        if not res_full_one_side.success:
            raise RuntimeError(f"Full one-sided LP failed: {res_full_one_side.message}")

        y_full = res_full_one_side.x[:n].copy()
        y_full /= y_full.sum()
        col_lower_full = float(np.min(F_true @ y_full))

        print("\n=== Full solver (one side) diagnostics ===")
        print(f"col_lower_full = {col_lower_full:.6f}")
        print(f"time: {t_full_one_side:.4f}s")

    # ------------------------------------------------------------------ #
    #  Sparse LP                                                           #
    # ------------------------------------------------------------------ #
    if not args.skip_sparse:
        if args.use_stored:
            base = input_matrix[:-4]
            U = np.load(f"{base}_U.npy")
            V = np.load(f"{base}_V.npy")
            Ahat = np.load(f"{base}_Ahat.npy")
            print("Error Before Run:", np.max(np.abs(F_true - Ahat - U @ V.T)))
            res_full_one_side_sparse, U, V, Ahat, t_factor_sparse, t_solve_sparse, stats = \
                solve_sparse_factored_lp_saved_factors_scipy(F_true, U, V, Ahat, method='highs-ipm')
        else:
            res_full_one_side_sparse, U, V, Ahat, t_factor_sparse, t_solve_sparse, stats = \
                solve_sparse_factored_lp(F_true, method=method)
            input_dir = os.path.dirname(args.input_matrix)
            base_name = os.path.splitext(os.path.basename(args.input_matrix))[0]
            np.save(os.path.join(input_dir, f"{base_name}_U.npy"), U)
            np.save(os.path.join(input_dir, f"{base_name}_V.npy"), V)
            np.save(os.path.join(input_dir, f"{base_name}_Ahat.npy"), Ahat)

        if not res_full_one_side_sparse.success:
            raise RuntimeError(f"Sparse LP failed: {res_full_one_side_sparse.message}")
        print("Reconstruction error:", np.max(np.abs(F_true - Ahat - U @ V.T)))

        y_full_sparse = res_full_one_side_sparse.x[:n].copy()
        y_full_sparse /= y_full_sparse.sum()
        col_lower_full_sparse = float(np.min(F_true @ y_full_sparse))

        print("\n=== Sparse solver (one side) diagnostics ===")
        print(f"col_lower_full_sparse = {col_lower_full_sparse:.6f}")
        print(f"time: solve={t_solve_sparse:.4f}s, factor={t_factor_sparse:.4f}s")

    # ------------------------------------------------------------------ #
    #  Double Oracle                                                       #
    # ------------------------------------------------------------------ #
    if not args.skip_double_oracle:
        ts0 = time.perf_counter()
        p_do, q_do, v_do, R_bar_do, C_bar_do, n_iter_do, t_setup_do, t_solve_do, history_do = \
            solve_double_oracle(F_true, method=method, seed=seed)
        t_double_oracle = time.perf_counter() - ts0
        col_lower_do = float(np.min(F_true @ p_do))

        print("\n=== Double Oracle diagnostics ===")
        print(f"col_lower_do = {col_lower_do:.6f}")
        print(f"iterations: {n_iter_do}, time: {t_double_oracle:.4f}s")

    # ------------------------------------------------------------------ #
    #  Main loop over k                                                    #
    # ------------------------------------------------------------------ #
    k_vals = []

    # Schur series (unchanged: single run per k)
    col_lower_series = []
    epsilon_lower_series = []
    times_schur = []

    # Sampling series: per-k arrays of trial results
    sampling_col_lower_mean = []
    sampling_col_lower_std  = []
    sampling_time_mean      = []
    sampling_time_std       = []

    # Every individual trial stored here for CSV export
    sampling_run_records = []  # list of dicts

    rng = np.random.default_rng(seed)  # reproducible but distinct per-trial seeds

    for k in range(1, k_max + 1):
        print(f"\n--- k={k} ---")

        # ---- Schur baseline (single run) ----
        if not skip_schur:
            res, Qr, Ur, t_solve, t_setup, c_r = solve_reduced_lp_using_QU_vform(
                F_true, k, p=p, q=q, seed=seed, method=method
            )
            F_reconstructed = Qr @ Ur @ Qr.T
            epsilon_lower_series.append(-2 * np.max(np.abs(F_true - F_reconstructed)))

            yk = res.x[:n].copy()
            s = yk.sum()
            if s <= 0:
                print(f"  Schur: nonpositive mass at k={k}, skipping")
                continue
            yk /= s
            col_lower_series.append(float(np.min(F_true @ yk)))
            times_schur.append(t_solve + t_setup)

            if not res.success:
                print(f"  Schur FAIL: {res.message}")
                continue

        # ---- Sampling: num_trials runs per k ----
        trial_col_lowers = []
        trial_times = []
        any_success = False

        for trial in range(num_trials):
            trial_seed = int(rng.integers(0, 2**31))

            (res_s, yk_s, row_idx_s, col_idx_s, n1,
             t_setup_s, t_solve_s) = solve_reduced_lp_using_sampling(
                F_true, k, seed=trial_seed, method=method
            )

            if yk_s is None or not res_s.success:
                print(f"  Sampling trial {trial}: FAIL — {res_s.message}")
                continue

            col_lower_s = float(np.min(F_true @ yk_s))
            elapsed_s   = t_setup_s + t_solve_s

            trial_col_lowers.append(col_lower_s)
            trial_times.append(elapsed_s)
            any_success = True

            sampling_run_records.append({
                "k":         k,
                "trial":     trial,
                "seed":      trial_seed,
                "col_lower": col_lower_s,
                "time":      elapsed_s,
            })

            print(f"  Sampling trial {trial}: col_lower={col_lower_s:.6f}, "
                  f"time={elapsed_s:.4f}s")

        if not any_success:
            print(f"  All sampling trials failed at k={k}, skipping k.")
            continue

        # Aggregate across successful trials
        sampling_col_lower_mean.append(float(np.mean(trial_col_lowers)))
        sampling_col_lower_std.append(float(np.std(trial_col_lowers, ddof=0)))
        sampling_time_mean.append(float(np.mean(trial_times)))
        sampling_time_std.append(float(np.std(trial_times, ddof=0)))

        k_vals.append(k)

    if len(k_vals) == 0:
        raise RuntimeError("No successful reduced runs completed.")

    # ------------------------------------------------------------------ #
    #  Plotting                                                            #
    # ------------------------------------------------------------------ #
    sampling_col_lower_mean = np.array(sampling_col_lower_mean)
    sampling_col_lower_std  = np.array(sampling_col_lower_std)
    sampling_time_mean      = np.array(sampling_time_mean)
    sampling_time_std       = np.array(sampling_time_std)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reduced Rank LP — Solver Analysis", fontsize=13)

    # -- Lower bound plot --
    ax = axes[0, 0]
    if not skip_schur:
        ax.plot(k_vals, col_lower_series, marker="o", markersize=3, label="Schur")

    ax.plot(k_vals, sampling_col_lower_mean,
            marker="o", markersize=3, color="green", label="Sampling (mean)")
    ax.fill_between(k_vals,
                    sampling_col_lower_mean - sampling_col_lower_std,
                    sampling_col_lower_mean + sampling_col_lower_std,
                    alpha=0.25, color="green", label="Sampling ±1 std")

    if not args.skip_double_oracle:
        ax.axhline(col_lower_do, linewidth=1.5, linestyle="--",
                   color="orange", label="Double Oracle")
    if not args.skip_full:
        ax.axhline(col_lower_full, linewidth=1.5, linestyle="--",
                   color="red", label="Full LP")
    if not args.skip_sparse:
        ax.axhline(col_lower_full_sparse, linewidth=1.5, linestyle="--",
                   color="purple", label="Sparse LP")

    ax.set_xlabel("k")
    ax.set_ylabel("Lower bound")
    ax.set_title("Exploitability Lower Bound")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -- Solve time plot --
    ax = axes[1, 0]
    if not skip_schur:
        ax.plot(k_vals, times_schur, marker="o", markersize=3, label="Schur solver")

    ax.plot(k_vals, sampling_time_mean,
            marker="o", markersize=3, color="green", label="Sampling (mean)")
    ax.fill_between(k_vals,
                    sampling_time_mean - sampling_time_std,
                    sampling_time_mean + sampling_time_std,
                    alpha=0.25, color="green", label="Sampling ±1 std")

    if not args.skip_double_oracle:
        ax.axhline(t_double_oracle, linewidth=1.5, linestyle="--",
                   color="orange", label="Double Oracle")
    if not args.skip_full:
        ax.axhline(t_full_one_side, linewidth=1.5, linestyle="--",
                   color="red", label="Full solver")
    if not args.skip_sparse:
        ax.axhline(t_solve_sparse, linewidth=1.5, linestyle="--",
                   color="orange", label="Sparse LP Solve")

    ax.set_xlabel("k")
    ax.set_ylabel("Time (s)")
    ax.set_title("Solve Time vs k")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  CSV export                                                          #
    # ------------------------------------------------------------------ #
    if output_directory:
        # Summary: mean ± std per k
        lower_bound_csv = os.path.join(output_directory, "exploitability_lower_bound.csv")
        series_list, labels = [], []
        if not skip_schur:
            series_list += [col_lower_series, epsilon_lower_series]
            labels      += ["Schur", "Bound Value"]
        series_list += [list(sampling_col_lower_mean), list(sampling_col_lower_std)]
        labels      += ["Sampling_mean", "Sampling_std"]
        if not args.skip_full:
            series_list.append([col_lower_full] * len(k_vals))
            labels.append("Full LP")
        if not args.skip_sparse:
            series_list.append([col_lower_full_sparse] * len(k_vals))
            labels.append("Sparse LP")
        save_to_csv(lower_bound_csv, k_vals, series_list, labels)

        time_csv = os.path.join(output_directory, "solve_times.csv")
        series_list, labels = [], []
        if not skip_schur:
            series_list.append(times_schur)
            labels.append("Schur")
        series_list += [list(sampling_time_mean), list(sampling_time_std)]
        labels      += ["Sampling_mean", "Sampling_std"]
        if not args.skip_full:
            series_list.append([t_full_one_side] * len(k_vals))
            labels.append("Full LP")
        if not args.skip_sparse:
            series_list.append([t_solve_sparse] * len(k_vals))
            labels.append("Sparse LP")
        save_to_csv(time_csv, k_vals, series_list, labels)

        # Per-run records
        runs_csv = os.path.join(output_directory, "sampling_runs.csv")
        save_sampling_runs_to_csv(runs_csv, sampling_run_records)

    # ------------------------------------------------------------------ #
    #  Save / show plot                                                    #
    # ------------------------------------------------------------------ #
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