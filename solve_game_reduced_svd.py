import numpy as np
from scipy.optimize import linprog
import time
import scipy.linalg as la

def truncate_schur_by_omega(Qr, Ur, omega_abs_tol=None, omega_rel_tol=None):
    """
    Truncate a real skew-symmetric Schur decomposition by dropping 2x2 blocks
    with small omega.

    Parameters
    ----------
    Qr : (n, r) array
        Schur vectors
    Ur : (r, r) array
        Real Schur form (block diagonal with 2x2 skew blocks)
    omega_abs_tol : float or None
        Keep only blocks with omega >= omega_abs_tol
    omega_rel_tol : float or None
        Keep only blocks with omega >= omega_rel_tol * max_omega

    Returns
    -------
    Q_keep : (n, r_keep) array
    U_keep : (r_keep, r_keep) array
    omegas : (nblocks,) array
        All extracted omegas before truncation
    keep_block_mask : (nblocks,) bool array
        Which 2x2 blocks were kept
    """
    r = Ur.shape[0]
    if r == 0:
        return Qr, Ur, np.array([]), np.array([], dtype=bool)

    if r % 2 != 0:
        raise ValueError(f"Expected even Schur dimension for skew-symmetric form, got r={r}")

    omegas = []
    for i in range(0, r, 2):
        omega = abs(Ur[i, i + 1])
        omegas.append(omega)
    omegas = np.array(omegas)

    if len(omegas) == 0:
        return Qr[:, :0], Ur[:0, :0], omegas, np.array([], dtype=bool)

    thresh = 0.0
    if omega_abs_tol is not None:
        thresh = max(thresh, float(omega_abs_tol))
    if omega_rel_tol is not None:
        thresh = max(thresh, float(omega_rel_tol) * float(np.max(omegas)))

    keep_block_mask = omegas >= thresh

    keep_indices = []
    for b, keep in enumerate(keep_block_mask):
        if keep:
            i = 2 * b
            keep_indices.extend([i, i + 1])

    keep_indices = np.array(keep_indices, dtype=int)

    Q_keep = Qr[:, keep_indices]
    U_keep = Ur[np.ix_(keep_indices, keep_indices)]

    return Q_keep, U_keep, omegas, keep_block_mask


def topk_svd_from_F(F, k):
    """
    Truncated SVD of F
    Returns U_r, S_r, Vt_r
    """
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    U_r = U[:, :k]
    S_r = S[:k]
    Vt_r = Vt[:k, :]
    return U_r, S_r, Vt_r

# def topk_svd_from_F_power(F, k, p=10, q=2, seed=0):
#     """
#     Randomized SVD using subspace iteration (analogous to your Schur method)

#     Returns
#     -------
#     U_r : (n, k)
#     S_r : (k,)
#     Vt_r : (k, n)
#     """
#     import numpy as np
#     import scipy.linalg as la

#     n = F.shape[0]
#     r = min(n, k + p)

#     rng = np.random.default_rng(seed)

#     # --- random subspace ---
#     V = rng.standard_normal((n, r))
#     V, _ = la.qr(V, mode="economic")

#     # --- power iterations (same spirit as your Schur code) ---
#     for q_i in range(q):
#         print("SVD: ", q_i, q)
#         V = F @ (F.T @ V)
#         V, _ = la.qr(V, mode="economic")

#     # --- project ---
#     B = V.T @ F @ V  # (r x r)

#     # --- SVD in small space ---
#     Ub, S, Vt = np.linalg.svd(B, full_matrices=False)

#     # --- lift back ---
#     U_small = V @ Ub
#     V_small = V @ Vt

#     # truncate to k
#     return U_small, S, V_small

import numpy as np
from scipy.optimize import linprog
import time

def randomized_svd(F, k, p=10, n_iter=2, seed=0):
    m, n = F.shape
    rng = np.random.default_rng(seed)

    l = min(k + p, min(m, n))

    Omega = rng.standard_normal((n, l))
    Y = F @ Omega

    for iter in range(n_iter):
        # print("Iteration: ", iter, n_iter)
        Y = F @ (F.T @ Y)

    Q, _ = np.linalg.qr(Y, mode='reduced')

    B = Q.T @ F
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)

    U = Q @ U_tilde

    return U[:, :k], S[:k], Vt[:k, :]


def solve_reduced_lp_using_svd_vform(
    F, k_nominal, p=10, q=2, seed=0,
    x_ub=1.0, method="highs-ipm",
    use_shift=True, eps_shift=1e-8,
    sv_rel_tol=1e-10,
    verbose=False
):
    t0 = time.perf_counter()

    # --- dimensions ---
    m, n = F.shape

    # --- randomized SVD ---
    U_r, S_r, Vt_r = randomized_svd(
        F, k=k_nominal, p=p, n_iter=q, seed=seed
    )

    # --- truncate small singular values (IMPORTANT) ---
    if len(S_r) > 0:
        tol = sv_rel_tol * np.max(S_r)
        keep = S_r > tol

        U_r = U_r[:, keep]
        S_r = S_r[keep]
        Vt_r = Vt_r[keep, :]

    r = len(S_r)

    if verbose:
        print(f"[SVD] kept rank = {r}")
        if r > 0:
            print(f" sigma_max = {np.max(S_r):.3e}")
            print(f" sigma_min = {np.min(S_r):.3e}")

    # --- degenerate case ---
    if r == 0:
        x = np.ones(n) / n

        class Dummy:
            success = True
            status = 0
            message = "All singular values truncated; returned uniform x."
            x = np.concatenate([x, np.zeros(1)])

        return Dummy(), U_r, S_r, Vt_r, 0.0, 0.0, 0.0

    # --- compressed operator ---
    US = U_r * S_r  # (m, r)

    # --- shift to ensure feasibility ---
    c_r = 0.0
    if use_shift:
        Fr_approx = (U_r * S_r) @ Vt_r
        min_entry = np.min(Fr_approx)
        c_r = max(0.0, -min_entry + eps_shift)

    # --- LP variables: x (n), z (r), v (1) ---
    nvar = n + r + 1

    # objective: minimize v
    c = np.zeros(nvar)
    c[n + r] = 1.0

    # --- equality constraints ---
    A_eq = np.zeros((1 + r, nvar))
    b_eq = np.zeros(1 + r)

    # sum(x) = 1
    A_eq[0, :n] = 1.0
    b_eq[0] = 1.0

    # z = Vt_r x
    A_eq[1:, :n] = -Vt_r
    A_eq[1:, n:n + r] = np.eye(r)

    # --- inequality constraints ---
    A_ub = np.zeros((m, nvar))
    b_ub = np.full(m, -c_r)

    # -U_r S_r z - v ≤ -c_r
    A_ub[:, n:n + r] = -US
    A_ub[:, n + r] = -1.0

    # --- bounds ---
    bounds = (
        [(0.0, float(x_ub))] * n +   # x
        [(None, None)] * r +         # z
        [(0.0, None)]                # v ≥ 0  (CRITICAL FIX)
    )

    t_setup = time.perf_counter() - t0

    # --- solve LP ---
    ts0 = time.perf_counter()
    res = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method=method
    )
    t_solve = time.perf_counter() - ts0

    return res, U_r, S_r, Vt_r, t_solve, t_setup, c_r