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


def topk_schur_from_F_power(F, k, p=10, q=5, seed=0):
    """
    Approximate Schur decomposition using subspace iteration on F.

    Returns
    -------
    U : (n, r)   approximate Schur vectors
    D : (r, r)   real Schur form (skew-symmetric block diagonal)
    """
    # print("Checkpoint 2")
    n = F.shape[0]
    k = min(k, n // 2)
    r = min(n, 2 * k + p)

    rng = np.random.default_rng(seed)

    V = rng.standard_normal((n, r))
    V, _ = la.qr(V, mode="economic")

    for q_i in range(q):
        print(q_i, q)
        V = F @ (F.T @ V)
        V, _ = la.qr(V, mode="economic")

    B = V.T @ F @ V
    B = 0.5 * (B - B.T)  # enforce skew-symmetry numerically

    D, W = la.schur(B, output="real")
    U = V @ W
    # print("Checkpoint 3")
    return U, D


def solve_reduced_lp_using_QU_vform(
    F, k_nominal, p=10, q=3, seed=0,
    x_ub=1.0, method="highs-ipm",
    use_shift=True, eps_shift=1e-6,
    omega_abs_tol=None, omega_rel_tol=None,
    verbose=False
):
    t0 = time.perf_counter()
    # print("Checkpoint 1")

    Qr, Ur = topk_schur_from_F_power(F, k=k_nominal, p=p, q=q, seed=seed)
    n = F.shape[0]

    # ---- omega-based truncation of 2x2 Schur blocks ----
    Qr, Ur, omegas_all, keep_mask = truncate_schur_by_omega(
        Qr, Ur,
        omega_abs_tol=omega_abs_tol,
        omega_rel_tol=omega_rel_tol
    )
    r = Qr.shape[1]

    if verbose:
        nblocks_all = len(omegas_all)
        nblocks_kept = int(np.sum(keep_mask)) if len(keep_mask) else 0
        print(
            f"[k={k_nominal}] blocks kept {nblocks_kept}/{nblocks_all}, "
            f"r={r}, "
            f"omega_max={np.max(omegas_all) if len(omegas_all) else np.nan:.3e}, "
            f"omega_min_kept={np.min(omegas_all[keep_mask]) if nblocks_kept else np.nan:.3e}"
        )

    if r == 0:
        x = np.ones(n) / n

        class Dummy:
            success = True
            status = 0
            message = "All Schur blocks truncated; returned uniform x."
            x = np.concatenate([x, np.array([]), np.array([0.0])])

        t_setup = time.perf_counter() - t0
        return Dummy(), Qr, Ur, 0.0, t_setup, 0.0

    QU = Qr @ Ur
    # print("Checkpoint 4")

    c_r = 0.0
    if use_shift:
        Fr = Qr @ Ur @ Qr.T
        min_entry = np.min(Fr)
        c_r = max(0.0, -min_entry + eps_shift)

    nvar = n + r + 1
    c = np.zeros(nvar)
    c[n + r] = 1.0
    # print("Checkpoint 5")

    A_eq = np.zeros((1 + r, nvar))
    b_eq = np.zeros(1 + r)
    A_eq[0, :n] = 1.0
    b_eq[0] = 1.0
    A_eq[1:, :n] = -Qr.T
    A_eq[1:, n:n + r] = np.eye(r)
    # print("Checkpoint 6")

    A_ub = np.zeros((n, nvar))
    b_ub = np.zeros(n)
    A_ub[:, n:n + r] = -QU
    A_ub[:, n + r] = -1.0
    b_ub[:] = -c_r

    bounds = [(0.0, float(x_ub))] * n + [(None, None)] * r + [(None, None)]

    t_setup = time.perf_counter() - t0
    # print("Checkpoint 7")
    ts0 = time.perf_counter()
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method=method
    )
    t_solve = time.perf_counter() - ts0

    return res, Qr, Ur, t_solve, t_setup, c_r