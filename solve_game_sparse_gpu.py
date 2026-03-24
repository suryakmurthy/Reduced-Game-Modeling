import numpy as np
import torch
from collections import Counter
from scipy.optimize import linprog
import time
import os

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _count_nnz_torch(A, tol=1e-10):
    return torch.count_nonzero(torch.abs(A) > tol).item()


def _normalize_rank1(u, v, zero_tol=1e-12):
    max_u = torch.max(torch.abs(u))
    if max_u <= zero_tol:
        return u, v
    return u / max_u, v * max_u


# ---------------------------------------------------------------------------
# Algorithm 4 (GPU core)
# ---------------------------------------------------------------------------

def _best_v_given_u_l0_gpu(A, u, zero_tol=1e-12, round_decimals=10):
    m, n = A.shape
    device = A.device

    v = torch.zeros(n, device=device, dtype=A.dtype)

    nz_mask = torch.abs(u) > zero_tol
    nz_rows = torch.where(nz_mask)[0]

    ku = nz_rows.numel()
    if ku == 0:
        return v

    u_nz = u[nz_rows]
    A_nz = A[nz_rows, :]

    ratios = A_nz / u_nz[:, None]
    nonzero_mask = torch.abs(A_nz) > zero_tol

    scale = 10 ** round_decimals
    ratios = torch.round(ratios * scale) / scale

    # unavoidable CPU step (mode)
    ratios_cpu = ratios.cpu().numpy()
    mask_cpu = nonzero_mask.cpu().numpy()

    for j in range(n):
        # print("Iterating 3", j, n)
        vals = ratios_cpu[mask_cpu[:, j], j]
        if vals.size == 0:
            continue

        unique, counts = np.unique(vals, return_counts=True)
        idx = np.argmax(counts)

        mode_val = unique[idx]
        count = counts[idx]
        nonzero_count = vals.size

        if count > ku - nonzero_count:
            v[j] = float(mode_val)

    return v


def _best_u_given_v_l0_gpu(A, v, **kwargs):
    return _best_v_given_u_l0_gpu(A.T, v, **kwargs)


# ---------------------------------------------------------------------------
# Algorithm 3 — alternating minimisation (GPU)
# ---------------------------------------------------------------------------

def _altmin_rank1_l0_gpu(
    A,
    max_inner_iters=20,
    seed=0,
    zero_tol=1e-12,
    round_decimals=10,
):
    m, n = A.shape
    device = A.device

    rng = np.random.default_rng(seed)

    i0 = int(rng.integers(0, m))
    u = torch.zeros(m, device=device, dtype=A.dtype)
    u[i0] = 1.0

    base_nnz = _count_nnz_torch(A, tol=zero_tol)

    best_u = u.clone()
    best_v = torch.zeros(n, device=device, dtype=A.dtype)
    best_obj = base_nnz

    for iter_inner in range(max_inner_iters):
        print("Iterating 2: ", iter_inner, max_inner_iters)
        v = _best_v_given_u_l0_gpu(
            A, u,
            zero_tol=zero_tol,
            round_decimals=round_decimals
        )

        u = _best_u_given_v_l0_gpu(
            A, v,
            zero_tol=zero_tol,
            round_decimals=round_decimals
        )

        u, v = _normalize_rank1(u, v, zero_tol=zero_tol)

        R = A - torch.outer(u, v)
        obj = _count_nnz_torch(R, tol=zero_tol)

        if obj < best_obj:
            best_obj = obj
            best_u = u.clone()
            best_v = v.clone()
        else:
            break

    return best_u, best_v, best_obj


# ---------------------------------------------------------------------------
# Algorithm 2 — outer loop (GPU)
# ---------------------------------------------------------------------------

def sparse_factorize_l0_gpu(
    A_np,
    max_outer_iters=100,
    max_inner_iters=10,
    seed=0,
    zero_tol=1e-10,
    min_improvement=5,
    max_rank=None,
):
    device = _get_device()

    A = torch.tensor(A_np, dtype=torch.float32, device=device)

    m, n = A.shape

    if max_rank is None:
        max_rank = min(m, n)

    U_cols = []
    V_cols = []

    orig_nnz = _count_nnz_torch(A, tol=zero_tol)
    prev_nnz = orig_nnz

    successful = 0
    unsuccessful = 0

    rng = np.random.default_rng(seed)

    for outer_iter in range(min(max_outer_iters, max_rank)):
        print("Iterating outer: ", outer_iter, min(max_outer_iters, max_rank))
        u, v, cand_nnz = _altmin_rank1_l0_gpu(
            A,
            max_inner_iters=max_inner_iters,
            seed=int(rng.integers(0, 10**9)),
            zero_tol=zero_tol,
        )

        ku = torch.count_nonzero(torch.abs(u) > zero_tol).item()
        kv = torch.count_nonzero(torch.abs(v) > zero_tol).item()

        improvement = prev_nnz - cand_nnz

        if ku <= 1 or kv <= 1 or improvement < min_improvement:
            unsuccessful += 1
            if successful > 0 and unsuccessful > successful:
                break
            continue

        U_cols.append(u.clone())
        V_cols.append(v.clone())

        A = A - torch.outer(u, v)
        prev_nnz = cand_nnz
        successful += 1

    # Build outputs
    if U_cols:
        U = torch.stack(U_cols, dim=1).cpu().numpy()
        V = torch.stack(V_cols, dim=1).cpu().numpy()
    else:
        U = np.zeros((m, 0))
        V = np.zeros((n, 0))

    Ahat = A.cpu().numpy()
    Ahat[np.abs(Ahat) < zero_tol] = 0.0

    stats = {
        "rank": U.shape[1],
        "orig_nnz": orig_nnz,
        "residual_nnz": np.count_nonzero(np.abs(Ahat) > zero_tol),
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
    U, V, Ahat, stats = sparse_factorize_l0_gpu(
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
    F, U, V,
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
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    input_matrix = 'settings/blotto/F_50000.npy'
    A = np.load(input_matrix)
    # Get directory of the input matrix
    input_dir = os.path.dirname(input_matrix)

    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_matrix))[0]

    # Build output paths
    U_path = os.path.join(input_dir, f"{base_name}_U.npy")
    V_path = os.path.join(input_dir, f"{base_name}_V.npy")
    A_path = os.path.join(input_dir, f"{base_name}_Ahat.npy")

    print("Running GPU factorization...")
    U, V, Ahat, stats = sparse_factorize_l0_gpu(
        A,
        max_outer_iters=50,
        min_improvement=1
    )
    # Save matrices
    np.save(U_path, U)
    np.save(V_path, V)
    np.save(A_path, Ahat)

    print(f"Saved U to {U_path}")
    print(f"Saved V to {V_path}")

    print(f"Rank: {stats['rank']}")
    print(f"NNZ: {stats['orig_nnz']} → {stats['residual_nnz']}")

    print("Reconstruction error:",
          np.max(np.abs(A - Ahat - U @ V.T)))