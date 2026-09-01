import numpy as np
from scipy.optimize import linprog
import time
import scipy.linalg as la
import gurobipy as gp
from gurobipy import GRB

def solve_full_lp_v_version(F, method=1, x_ub=1.0):
    """
    Solve the row-player LP:
        min_{x,t} t
        s.t. F^T x <= t 1
             1^T x = 1
             x >= 0

    F is (n_rows x n_cols).
    x lives over the n_rows rows.
    """
    t0 = time.perf_counter()

    n_rows = F.shape[0]
    n_cols = F.shape[1]

    model = gp.Model("full_lp")
    model.setParam("OutputFlag", 0)
    model.setParam("Method", method)

    # Variables
    if x_ub is None:
        x = model.addVars(n_rows, lb=0.0, ub=GRB.INFINITY, name="x")
    else:
        x = model.addVars(n_rows, lb=0.0, ub=float(x_ub), name="x")

    t = model.addVar(lb=-GRB.INFINITY, name="t")

    # Objective: minimize t
    model.setObjective(t, GRB.MINIMIZE)

    # Constraint: sum x = 1
    model.addConstr(gp.quicksum(x[i] for i in range(n_rows)) == 1.0)

    # Constraints: F^T x <= t*1
    # i.e. for each column j: sum_i F[i,j] * x[i] <= t
    for j in range(n_cols):
        model.addConstr(
            gp.quicksum(F[i, j] * x[i] for i in range(n_rows)) <= t
        )

    t_setup = time.perf_counter() - t0

    ts0 = time.perf_counter()
    model.optimize()
    t_solve = time.perf_counter() - ts0

    class Result:
        pass

    res = Result()
    res.success = model.status == GRB.OPTIMAL
    res.status = model.status
    res.message = model.Status

    if res.success:
        x_vals = np.array([x[i].X for i in range(n_rows)])
        t_val = t.X
        res.x = np.concatenate([x_vals, np.array([t_val])])
        res.fun = t_val
    else:
        res.x = None
        res.fun = None

    return res


def sample_subgame(F, n_rows, n_cols, seed=0):
    """
    Sample a fat subgame of size n_rows x n_cols from F.

    Per the paper (Theorem 3.1), the row player samples m1 rows and n1 >> m1
    columns independently and uniformly at random. Row and column indices are
    sampled independently (Assumption 2.1).

    Parameters
    ----------
    F      : (M x N) full game matrix
    n_rows : m1 — number of rows to sample (P1's policy count)
    n_cols : n1 — number of columns to sample (must satisfy the paper's
                  bound relative to n_rows and delta, see
                  `compute_n_cols_bound`)
    seed   : RNG seed

    Returns
    -------
    row_idx : sampled row indices into F
    col_idx : sampled column indices into F
    F_sub   : (n_rows x n_cols) submatrix
    """
    M, N = F.shape
    n_rows = min(n_rows, M)
    n_cols = min(n_cols, N)

    rng = np.random.default_rng(seed)

    # Independent sampling of rows and columns (Assumption 2.1)
    row_idx = np.sort(rng.choice(M, size=n_rows, replace=False))
    col_idx = np.sort(rng.choice(N, size=n_cols, replace=False))

    F_sub = F[np.ix_(row_idx, col_idx)]
    return row_idx, col_idx, F_sub


def compute_n_cols_bound(n_rows, n_bar2, delta):
    """
    Compute the required number of columns n1 for the SSP algorithm
    to be (eps=0)-secure with confidence 1-delta (Theorem 3.1, eq. 4).

    n1 = ceil((m1 + 1) / delta - 1) * n_bar2

    Parameters
    ----------
    n_rows  : m1, number of rows P1 samples
    n_bar2  : upper bound on n2 (number of columns P2 samples)
    delta   : confidence parameter; security holds with prob >= 1 - delta

    Returns
    -------
    n1 : required number of columns (integer)
    """
    K = int(np.ceil((n_rows + 1) / delta - 1))
    return K * n_bar2


def lift_strategy(y_sub, row_idx, M):
    """
    Lift a strategy over sampled rows back to the full row space.

    Parameters
    ----------
    y_sub   : (n_rows,) strategy on the subgame rows
    row_idx : indices of the sampled rows in the full game
    M       : total number of rows in the full game

    Returns
    -------
    y_full : (M,) strategy with mass only on sampled rows
    """
    y_full = np.zeros(M, dtype=float)
    y_full[row_idx] = y_sub
    s = y_full.sum()
    if s <= 0:
        raise RuntimeError("Lifted sampled strategy has nonpositive total mass.")
    y_full /= s
    return y_full

# Check over a couple runs
def solve_reduced_lp_using_sampling(
    F, k, n_bar2=1, delta=0.01, x_ub=1.0, seed=0, method=1
):
    """
    SSP Algorithm (Theorem 3.1, eq. 4):
      1. Compute the required column count n1 from the paper's bound.
      2. Sample a fat m1 x n1 submatrix of F independently in rows and cols.
      3. Solve the row-player LP on the submatrix.
      4. Lift the row strategy back to the full space.

    Parameters
    ----------
    F       : (M x N) full game matrix
    m1      : number of rows for P1 to sample
    n_bar2  : upper bound on the number of columns P2 samples
    delta   : confidence parameter (security holds w.p. >= 1 - delta)
    x_ub    : upper bound on each x variable (or None)
    seed    : RNG seed
    method  : Gurobi LP method flag

    Returns
    -------
    res      : Gurobi result object from the subgame solve
    y_full   : (M,) lifted strategy over the full row space (or None on failure)
    row_idx  : sampled row indices
    col_idx  : sampled column indices
    n1       : number of columns actually sampled
    t_setup  : time spent on sampling + setup (seconds)
    t_solve  : time spent solving the subgame LP (seconds)
    """
    t0 = time.perf_counter()
    M, N = F.shape
    m1 = 2 * k
    # Paper's bound: n1 = ceil((m1+1)/delta - 1) * n_bar2
    n1 = compute_n_cols_bound(m1, n_bar2, delta)
    n1 = min(n1, N)  # can't sample more columns than exist

    row_idx, col_idx, F_sub = sample_subgame(F, n_rows=m1, n_cols=n1, seed=seed)
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    res_sub = solve_full_lp_v_version(F_sub, x_ub=x_ub, method=method)
    t_solve = time.perf_counter() - t1

    if not res_sub.success:
        return res_sub, None, row_idx, col_idx, n1, t_setup, t_solve

    # Extract and normalise the row strategy (first m1 entries of res.x)
    y_sub = res_sub.x[:m1].copy()
    s = y_sub.sum()
    if s <= 0:
        return res_sub, None, row_idx, col_idx, n1, t_setup, t_solve
    y_sub /= s

    y_full = lift_strategy(y_sub, row_idx, M)
    return res_sub, y_full, row_idx, col_idx, n1, t_setup, t_solve