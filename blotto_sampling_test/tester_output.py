import numpy as np
from old_version.solve_game_full import solve_full_lp_v_version
from old_version.solve_game_sampling import sample_subgame, solve_reduced_lp_using_sampling

# ── Config ────────────────────────────────────────────────────────────────────
input_matrix = 'settings/blotto/F_1000.npy'
method = 'highs-ipm'
SUPPORT_TOL = 1e-6

# ── Load game ─────────────────────────────────────────────────────────────────
F_raw = np.load(input_matrix)
F_true = -F_raw
m, n = F_true.shape

# ── Ground truth ──────────────────────────────────────────────────────────────
res_full = solve_full_lp_v_version(F_true, method=method)
y_full   = res_full.x[:n].copy()
y_full  /= y_full.sum()

support_idx = np.where(y_full > SUPPORT_TOL)[0]
support_set = set(support_idx.tolist())
s = len(support_idx)

print(f"True support size: {s}  indices: {support_idx.tolist()}\n")

# ── True game value ───────────────────────────────────────────────────────────
v_true = (F_true @ y_full).min()
print(f"True game value (col player): {v_true:.6f}\n")

# ── Helper: optimality check ──────────────────────────────────────────────────
def check_optimality(F, y, v_true, tol=1e-6):
    payoffs = F @ y
    min_payoff = payoffs.min()
    gap = min_payoff - v_true
    violations = np.sum(payoffs < v_true - tol)

    return {
        "min_payoff": min_payoff,
        "gap": gap,
        "violations": violations,
        "std": np.std(payoffs)
    }

# ── Sweep over dimensions ─────────────────────────────────────────────────────
dim_values  = [16, 24, 32, 40, 48, 56, 64, 80, 100, 150, 200]
SEED        = 10
N_REPS      = 20

print(f"{'dim':>6} | {'mean |supp∩sample|':>18} | {'max |supp∩sample|':>18} "
      f"| {'full cover %':>11} | {'mean exploitability':>20} "
      f"| {'mean supp size':>17} | {'mean mass true supp':>22} "
      f"| {'mean payoff std':>18}")
print("-" * 140)

rng_master = np.random.default_rng(SEED)
rep_seeds  = rng_master.integers(0, 10_000, size=N_REPS)

# store some solutions for cross-comparison later
collected_solutions = []

for dim in dim_values:
    overlaps      = []
    full_covers   = 0
    exploits      = []
    supp_sizes    = []
    mass_true     = []
    payoff_stds   = []

    for seed in rep_seeds:
        # sample indices
        idx, _ = sample_subgame(F_true, dim=dim, seed=int(seed))
        sampled_set = set(idx.tolist())

        overlap = len(support_set & sampled_set)
        overlaps.append(overlap)
        if overlap == s:
            full_covers += 1

        # solve sampled game
        res_sub, y_lifted, *_ = solve_reduced_lp_using_sampling(
            F_true, k=dim//2, seed=int(seed), method=method
        )

        if y_lifted is not None:
            # exploitability
            payoffs   = F_true @ y_lifted
            v_sampled = payoffs.min()
            exploits.append(v_true - v_sampled)

            # support size
            supp_size = np.sum(y_lifted > SUPPORT_TOL)
            supp_sizes.append(supp_size)

            # mass on true support
            mass = y_lifted[list(support_set)].sum()
            mass_true.append(mass)

            # optimality diagnostics
            stats = check_optimality(F_true, y_lifted, v_true)
            payoff_stds.append(stats["std"])

            # store some solutions for later comparison
            if dim >= 150:
                collected_solutions.append(y_lifted)

        else:
            exploits.append(np.nan)
            supp_sizes.append(np.nan)
            mass_true.append(np.nan)
            payoff_stds.append(np.nan)

    # aggregate
    mean_ov  = np.mean(overlaps)
    max_ov   = np.max(overlaps)
    cover_pc = 100 * full_covers / N_REPS
    mean_ex  = np.nanmean(exploits)
    mean_supp = np.nanmean(supp_sizes)
    mean_mass = np.nanmean(mass_true)
    mean_std  = np.nanmean(payoff_stds)

    print(f"{dim:>6} | {mean_ov:>18.2f} | {max_ov:>18d} "
          f"| {cover_pc:>10.1f}% | {mean_ex:>20.6f} "
          f"| {mean_supp:>17.2f} | {mean_mass:>22.6f} "
          f"| {mean_std:>18.6e}")

# ── Compare sampled equilibria (degeneracy test) ──────────────────────────────
print("\n--- Cross-solution comparison (degeneracy test) ---")

if len(collected_solutions) >= 2:
    dists = []
    val_diffs = []

    for i in range(len(collected_solutions)):
        for j in range(i + 1, len(collected_solutions)):
            y1 = collected_solutions[i]
            y2 = collected_solutions[j]

            dists.append(np.linalg.norm(y1 - y2, 1))

            v1 = (F_true @ y1).min()
            v2 = (F_true @ y2).min()
            val_diffs.append(abs(v1 - v2))

    print(f"Mean L1 distance between solutions: {np.mean(dists):.4f}")
    print(f"Mean value difference: {np.mean(val_diffs):.6e}")

# ── True support weights ──────────────────────────────────────────────────────
weights = y_full[support_idx]
sorted_order = np.argsort(weights)[::-1]

print("\nTrue mixed strategy weights on support (sorted):")
print(f"{'Rank':>5} | {'Global idx':>10} | {'Weight':>12} | {'Cumulative':>12}")
print("-" * 48)

cumsum = 0.0
for rank, i in enumerate(sorted_order):
    cumsum += weights[i]
    print(f"{rank+1:>5} | {support_idx[i]:>10} | {weights[i]:>12.6f} | {cumsum:>12.6f}")

print(f"\nTop 6 cumulative weight: {weights[sorted_order[:6]].sum():.6f}")
print(f"Top 3 cumulative weight: {weights[sorted_order[:3]].sum():.6f}")