import numpy as np
from scipy.optimize import linprog
import time
import scipy.linalg as la
from solve_game_full import solve_full_lp_v_version

def sample_subgame(F, dim, seed=0):
    """
    Sample a principal subgame of size dim x dim.
    """
    n = F.shape[0]
    dim = min(dim, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=dim, replace=False)
    idx.sort()
    F_sub = F[np.ix_(idx, idx)]
    return idx, F_sub


def lift_strategy(y_sub, idx, n):
    """
    Lift a sampled strategy on the subgame back to the full space.
    """
    y_full = np.zeros(n, dtype=float)
    y_full[idx] = y_sub
    s = y_full.sum()
    if s <= 0:
        raise RuntimeError("Lifted sampled strategy has nonpositive total mass.")
    y_full /= s
    return y_full


def solve_reduced_lp_using_sampling(
    F, k, x_ub=1.0, seed=0, method="highs-ipm"
):
    """
    Sampling baseline:
      - sample a principal subgame of size (2k) x (2k)
      - solve the subgame LP
      - lift the solution back to the full space
    """
    t0 = time.perf_counter()
    n = F.shape[0]
    dim = min(2 * k, n)

    idx, F_sub = sample_subgame(F, dim=dim, seed=seed)
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    res_sub = solve_full_lp_v_version(F_sub, x_ub=x_ub, method=method)
    t_solve = time.perf_counter() - t1

    if not res_sub.success:
        return res_sub, None, idx, t_setup, t_solve

    y_sub = res_sub.x[:dim].copy()
    s = y_sub.sum()
    if s <= 0:
        return res_sub, None, idx, t_setup, t_solve
    y_sub /= s

    y_full = lift_strategy(y_sub, idx, n)
    return res_sub, y_full, idx, t_setup, t_solve

