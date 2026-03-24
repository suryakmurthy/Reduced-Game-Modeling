"""
Sparsified Linear Programming for Zero-Sum Equilibrium Finding
Zhang & Sandholm (ICML 2020) — corrected implementation.

Key fixes over the original draft:
  1. Implicit residual: inner alternating minimization computes
     R = A - (already-found U V^T) - u v^T on the fly, rather than
     operating on a stale pre-modified copy of A. (Section 5.2)
  2. LP signs: the objective and inequality coefficients are all negated
     correctly so we are maximising the game value t. (Section 4)
  3. Residual Ahat is computed once at the end of factorisation, not
     accumulated inside the outer loop via repeated in-place subtraction.
"""

import numpy as np
from collections import Counter
from scipy.optimize import linprog
import time


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _count_nnz(A, tol=1e-10):
    return int(np.count_nonzero(np.abs(A) > tol))


def _round_key(x, decimals=10):
    return float(np.round(x, decimals))


def _normalize_rank1(u, v, zero_tol=1e-12):
    """Rescale so ||u||_inf == 1, absorbing the scale into v."""
    max_u = np.max(np.abs(u)) if u.size else 0.0
    if max_u <= zero_tol:
        return u, v
    return u / max_u, v * max_u


# ---------------------------------------------------------------------------
# Algorithm 4 — sparse subproblem: best v given u for residual R = A - UV^T
# ---------------------------------------------------------------------------

def _best_v_given_u_l0(A, u, zero_tol=1e-12, round_decimals=10):
    """
    Approximate argmin_v ||R - u v^T||_0  where  R = A - sum_k u_k v_k^T.

    Algorithm 4 of the paper: for each column j, collect the ratio R[i,j]/u[i]
    for every nonzero row i in u, then accept the mode value as v[j] if it
    explains more entries than it would introduce as new nonzeros.
    """
    m, n = A.shape
    v = np.zeros(n, dtype=float)

    nz_rows = np.flatnonzero(np.abs(u) > zero_tol)
    ku = len(nz_rows)
    if ku == 0:
        return v

    for j in range(n):
        print("Iterating 3: ", j, n)
        vals = []
        nonzero_count = 0  # = len(q[j]) in the paper's notation

        for i in nz_rows:
            # Compute R[i, j] = A[i, j] - sum_k U_cols[k][i] * V_cols[k][j]
            rij = A[i, j]
            if abs(rij) > zero_tol:
                vals.append(_round_key(rij / u[i], round_decimals))
                nonzero_count += 1

        if not vals:
            continue

        mode_val, count = Counter(vals).most_common(1)[0]

        # Algorithm 4 line 9: accept only when count > ||u||_0 - len(q[j])
        if count > ku - nonzero_count:
            v[j] = mode_val

    return v


def _best_u_given_v_l0(A, v, zero_tol=1e-12, round_decimals=10):
    """
    Approximate argmin_u ||R - u v^T||_0.

    Equivalent to argmin_u' ||R^T - v u'^T||_0.
    R^T = A^T - sum_k v_k u_k^T, so we swap the roles of U_cols/V_cols.
    """
    return _best_v_given_u_l0(
        A.T, v,
        zero_tol=zero_tol,
        round_decimals=round_decimals,
    )


# ---------------------------------------------------------------------------
# Algorithm 3 — alternating minimisation for one rank-1 factor
# ---------------------------------------------------------------------------

def _residual_nnz(A, u, v, zero_tol=1e-10):
    R = A - np.outer(u, v)
    R[np.abs(R) < zero_tol] = 0.0
    return _count_nnz(R, tol=zero_tol)


def _altmin_rank1_l0(
    A,
    max_inner_iters=20, seed=0,
    zero_tol=1e-12, round_decimals=10,
):
    """
    Algorithm 3: alternating minimisation for argmin_{u,v} ||R - u v^T||_0
    where  R = A - sum_k u_k v_k^T  (the current residual).

    FIX vs original: U_cols and V_cols (already-accepted factors) are passed
    through so that _best_v_given_u_l0 / _best_u_given_v_l0 work on the
    true residual, not on a stale snapshot of A.

    Initialisation: basis vector e_{i0} as recommended in Section 5.1.
    """
    m, n = A.shape
    rng = np.random.default_rng(seed)

    i0 = int(rng.integers(0, m))
    u = np.zeros(m, dtype=float)
    u[i0] = 1.0

    base_nnz = _count_nnz(A, tol=zero_tol)

    best_u = np.zeros(m, dtype=float)
    best_v = np.zeros(n, dtype=float)
    best_obj = base_nnz  # tracks nnz after subtracting the candidate uv^T

    for iter_inner in range(max_inner_iters):
        print("Iterating 2: ", iter_inner, max_inner_iters)
        v = _best_v_given_u_l0(A, u,
                                zero_tol=zero_tol, round_decimals=round_decimals)
        u = _best_u_given_v_l0(A, v,
                                zero_tol=zero_tol, round_decimals=round_decimals)
        u, v = _normalize_rank1(u, v, zero_tol=zero_tol)

        obj = _residual_nnz(A, u, v, zero_tol=zero_tol)

        if obj < best_obj:
            best_obj = obj
            best_u = u.copy()
            best_v = v.copy()
        else:
            # Objective stopped improving — local optimum reached
            break

    return best_u, best_v, best_obj


# ---------------------------------------------------------------------------
# Algorithm 2 — outer greedy loop
# ---------------------------------------------------------------------------

def sparse_factorize_l0(
    A,
    max_outer_iters=100,
    max_inner_iters=10,
    seed=0,
    zero_tol=1e-10,
    min_improvement=5,
    max_rank=None,
):
    """
    Greedy sparse factorisation:  A = Ahat + U V^T

    Implements Algorithms 2-4 of Zhang & Sandholm (2020).

    Stopping rule (Algorithm 2): terminate when the number of unsuccessful
    outer iterations (those producing no useful rank-1 factor) exceeds the
    number of successful ones.

    Returns
    -------
    U       : (m, r) array
    V       : (n, r) array
    Ahat    : (m, n) residual array  —  A - U V^T
    stats   : dict with 'rank', 'orig_nnz', 'residual_nnz'
    """
    m, n = A.shape
    A = A.astype(float)

    U_cols = []
    V_cols = []
    rng = np.random.default_rng(seed)

    orig_nnz = _count_nnz(A, tol=zero_tol)
    prev_nnz = orig_nnz

    successful = 0
    unsuccessful = 0

    if max_rank is None:
        max_rank = min(m, n)

    for iter in range(min(max_outer_iters, max_rank)):
        print("Iterating: ", iter, min(max_outer_iters, max_rank))
        u, v, cand_nnz = _altmin_rank1_l0(
            A,
            max_inner_iters=max_inner_iters,
            seed=int(rng.integers(0, 10**9)),
            zero_tol=zero_tol,
        )

        ku = np.count_nonzero(np.abs(u) > zero_tol)
        kv = np.count_nonzero(np.abs(v) > zero_tol)
        improvement = prev_nnz - cand_nnz

        if ku <= 1 or kv <= 1 or improvement < min_improvement:
            unsuccessful += 1
            if successful > 0 and unsuccessful > successful:
                break
            continue

        U_cols.append(u)
        V_cols.append(v)
        A -= np.outer(u, v)
        prev_nnz = cand_nnz
        successful += 1
        


    # Build Ahat = A - U V^T once, cleanly
    Ahat = A.copy()
    Ahat[np.abs(Ahat) < zero_tol] = 0.0

    if U_cols:
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
    else:
        U = np.zeros((m, 0), dtype=float)
        V = np.zeros((n, 0), dtype=float)

    stats = {
        "rank": U.shape[1],
        "orig_nnz": orig_nnz,
        "residual_nnz": _count_nnz(Ahat, tol=zero_tol),
    }

    return U, V, Ahat, stats


# ---------------------------------------------------------------------------
# LP solvers
# ---------------------------------------------------------------------------

def solve_full_lp(F, x_ub=1.0, method="highs-ipm"):
    """
    Solve the row-player LP without factorisation (baseline).

    max_{x,t} t   s.t.  F^T x >= t·1,  sum(x)=1,  x>=0
    ↓  (negate for linprog minimisation)
    min_{x,t} -t  s.t.  -F^T x + t·1 <= 0,  sum(x)=1,  x>=0
    """
    m, n = F.shape
    # Variables: [x (m), t (1)]
    c = np.zeros(m + 1)
    c[-1] = -1.0

    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -F.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, float(x_ub))] * m + [(None, None)]

    return linprog(c, A_ub=A_ub, b_ub=b_ub,
                   A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method=method)


def solve_sparse_factored_lp(
    F,
    x_ub=1.0,
    method="highs-ipm",
    max_outer_iters=100,
    max_inner_iters=10,
    seed=0,
    zero_tol=1e-10,
    min_improvement=5,
    max_rank=None,
):
    """
    Solve the row-player LP using the factored form A = Ahat + U V^T.

    Reformulated LP (Section 4 of the paper), introducing w = U^T x:

        min_{x, w, t}  -t
        s.t.
          -Ahat^T x - V w + t·1_n  <=  0      [n inequalities]
          sum(x)                    =   1       [1 equality]
          U^T x  -  w               =   0       [r equalities]
          x >= 0,  w and t free

    Returns
    -------
    res        : linprog result (res.x[:m] is the row-player strategy,
                 -res.fun is the game value)
    U, V, Ahat : factorisation components
    t_factor   : wall-clock time for factorisation
    t_solve    : wall-clock time for LP solve
    stats      : factorisation statistics
    """
    t0 = time.perf_counter()
    U, V, Ahat, stats = sparse_factorize_l0(
        F,
        max_outer_iters=max_outer_iters,
        max_inner_iters=max_inner_iters,
        seed=seed,
        zero_tol=zero_tol,
        min_improvement=min_improvement,
        max_rank=max_rank,
    )
    t_factor = time.perf_counter() - t0

    m, n = F.shape
    r = U.shape[1]

    # If no useful factorisation found, fall back to the full LP
    if r == 0:
        t1 = time.perf_counter()
        res = solve_full_lp(F, x_ub=x_ub, method=method)
        return res, U, V, Ahat, t_factor, time.perf_counter() - t1, stats

    # Variables: [x (m), w (r), t (1)]
    nvar = m + r + 1
    c = np.zeros(nvar)
    c[-1] = 1.0

    A_ub = np.zeros((n, nvar))
    A_ub[:, :m] = Ahat.T
    A_ub[:, m:m+r] = V
    A_ub[:, -1] = -1.0
    b_ub = np.zeros(n)

    A_eq = np.zeros((1 + r, nvar))
    b_eq = np.zeros(1 + r)
    A_eq[0, :m] = 1.0
    b_eq[0] = 1.0
    A_eq[1:, :m] = U.T
    A_eq[1:, m:m+r] = -np.eye(r)

    bounds = [(0.0, float(x_ub))] * m + [(None, None)] * r + [(None, None)]

    t1 = time.perf_counter()
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
    )
    t_solve = time.perf_counter() - t1

    return res, U, V, Ahat, t_factor, t_solve, stats

def solve_sparse_factored_lp_saved_factors(
    F, U, V, Ahat,
    x_ub=1.0,
    method="highs-ipm",
    max_outer_iters=100,
    max_inner_iters=10,
    seed=0,
    zero_tol=1e-10,
    min_improvement=5,
    max_rank=None,
):
    """
    Solve the row-player LP using the factored form A = Ahat + U V^T.

    Reformulated LP (Section 4 of the paper), introducing w = U^T x:

        min_{x, w, t}  -t
        s.t.
          -Ahat^T x - V w + t·1_n  <=  0      [n inequalities]
          sum(x)                    =   1       [1 equality]
          U^T x  -  w               =   0       [r equalities]
          x >= 0,  w and t free

    Returns
    -------
    res        : linprog result (res.x[:m] is the row-player strategy,
                 -res.fun is the game value)
    U, V, Ahat : factorisation components
    t_factor   : wall-clock time for factorisation
    t_solve    : wall-clock time for LP solve
    stats      : factorisation statistics
    """


    m, n = F.shape
    r = U.shape[1]
    # If no useful factorisation found, fall back to the full LP
    if r == 0:
        t1 = time.perf_counter()
        res = solve_full_lp(F, x_ub=x_ub, method=method)
        return res, U, V, Ahat, 0, time.perf_counter() - t1, stats

    # Variables: [x (m), w (r), t (1)]
    nvar = m + r + 1
    c = np.zeros(nvar)
    c[-1] = 1.0

    A_ub = np.zeros((n, nvar))
    A_ub[:, :m] = Ahat.T
    A_ub[:, m:m+r] = V
    A_ub[:, -1] = -1.0
    b_ub = np.zeros(n)

    A_eq = np.zeros((1 + r, nvar))
    b_eq = np.zeros(1 + r)
    A_eq[0, :m] = 1.0
    b_eq[0] = 1.0
    A_eq[1:, :m] = U.T
    A_eq[1:, m:m+r] = -np.eye(r)

    bounds = [(0.0, float(x_ub))] * m + [(None, None)] * r + [(None, None)]

    t1 = time.perf_counter()
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
    )
    t_solve = time.perf_counter() - t1

    return res, U, V, Ahat, 0, t_solve, stats


# ---------------------------------------------------------------------------
# Test Script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    A = np.load('settings/blotto/F_1000.npy')
    print("Original nnz:", _count_nnz(A))
    A_orig = A.copy()
    U, V, Ahat, stats = sparse_factorize_l0(A, max_outer_iters=50, min_improvement=1)
    print(f"Factorisation rank: {stats['rank']}")
    print(f"Original nnz: {stats['orig_nnz']}  →  residual nnz: {stats['residual_nnz']}")
    print(f"Reconstruction error: {np.max(np.abs(A_orig - Ahat - U @ V.T)):.2e}")

    res_full = solve_full_lp(A_orig)
    res_fact, *_ = solve_sparse_factored_lp(A_orig, min_improvement=1)

    val_full = -res_full.fun
    val_fact = -res_fact.fun
    print(f"\nGame value (full LP):    {val_full:.6f}")
    print(f"Game value (factored LP): {val_fact:.6f}")
    print(f"Difference: {abs(val_full - val_fact):.2e}")