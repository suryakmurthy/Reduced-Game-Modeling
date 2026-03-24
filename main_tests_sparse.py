"""
Sanity checks on sparse_lp.py using matrices/games with known answers.

Tests are grouped into two sections:
  1. Factorisation tests  — matrices where we know the exact sparse structure
  2. Game value tests     — zero-sum games where the Nash value is known analytically
"""

import numpy as np
from solve_game_sparse import (
    sparse_factorize_l0,
    solve_full_lp,
    solve_sparse_factored_lp,
    _count_nnz,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# ===========================================================================
# 1. FACTORISATION TESTS
# ===========================================================================

print("\n=== Factorisation tests ===\n")

# ---------------------------------------------------------------------------
# 1a. Zero matrix — nothing to factor, residual should stay zero
# ---------------------------------------------------------------------------
A = np.zeros((10, 10))
U, V, Ahat, stats = sparse_factorize_l0(A, min_improvement=1)
check("Zero matrix: rank=0",       stats["rank"] == 0)
check("Zero matrix: residual=0",   stats["residual_nnz"] == 0)
check("Zero matrix: reconstruction", np.max(np.abs(A - Ahat - U @ V.T)) < 1e-10)

# ---------------------------------------------------------------------------
# 1b. Identity matrix — already maximally sparse, no useful rank-1 factor
#     should exist (each rank-1 term would be denser than leaving it alone)
# ---------------------------------------------------------------------------
A = np.eye(8)
U, V, Ahat, stats = sparse_factorize_l0(A, min_improvement=2)
check("Identity: reconstruction",  np.max(np.abs(A - Ahat - U @ V.T)) < 1e-10)
check("Identity: not made denser",
      _count_nnz(Ahat) + _count_nnz(U) + _count_nnz(V) <= _count_nnz(A),
      f"original={_count_nnz(A)}, factored total={_count_nnz(Ahat)+_count_nnz(U)+_count_nnz(V)}")

# ---------------------------------------------------------------------------
# 1c. Exact rank-1 matrix — should be captured in a single outer iteration
#     Residual should be (near) zero after factorisation.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
u_true = rng.standard_normal(12)
v_true = rng.standard_normal(12)
A = np.outer(u_true, v_true)
U, V, Ahat, stats = sparse_factorize_l0(A, min_improvement=1)
check("Rank-1 matrix: reconstruction",  np.max(np.abs(A - Ahat - U @ V.T)) < 1e-9)
check("Rank-1 matrix: residual is tiny", stats["residual_nnz"] == 0,
      f"residual_nnz={stats['residual_nnz']}")

# ---------------------------------------------------------------------------
# 1d. Upper-triangular rank-1 (Example 1 in the paper)
#     A = triu(u v^T). Should compress significantly vs original nnz.
# ---------------------------------------------------------------------------
n = 32
u_true = np.ones(n)   # uniform so the structure is clean
v_true = np.ones(n)
A = np.triu(np.outer(u_true, v_true))
orig_nnz = _count_nnz(A)
U, V, Ahat, stats = sparse_factorize_l0(A, min_improvement=1)
total_factored_nnz = _count_nnz(U) + _count_nnz(V) + stats["residual_nnz"]
check("Upper-tri rank-1: reconstruction",
      np.max(np.abs(A - Ahat - U @ V.T)) < 1e-9)
check("Upper-tri rank-1: compression achieved",
      total_factored_nnz < orig_nnz,
      f"orig={orig_nnz}, factored total={total_factored_nnz}")

# ---------------------------------------------------------------------------
# 1e. Block-diagonal matrix — factorising the whole matrix should give the
#     same result as factorising each block independently (Section 5.3)
# ---------------------------------------------------------------------------
B1 = np.triu(np.outer(np.ones(8), np.ones(8)))
B2 = np.triu(np.outer(np.ones(8), np.ones(8))) * 2
A = np.block([[B1, np.zeros((8,8))],
              [np.zeros((8,8)), B2]])
U, V, Ahat, stats = sparse_factorize_l0(A, min_improvement=1)
check("Block-diagonal: reconstruction",
      np.max(np.abs(A - Ahat - U @ V.T)) < 1e-9)
check("Block-diagonal: compresses",
      _count_nnz(U) + _count_nnz(V) + stats["residual_nnz"] < _count_nnz(A),
      f"orig={_count_nnz(A)}, factored={_count_nnz(U)+_count_nnz(V)+stats['residual_nnz']}")


# ===========================================================================
# 2. GAME VALUE TESTS  (Nash equilibrium value known analytically)
# ===========================================================================

print("\n=== Game value tests ===\n")

TOL = 1e-6

def game_value_both(F, label):
    """Solve with both full and factored LP, print and check they agree."""
    res_full = solve_full_lp(F)
    res_fact, U, V, Ahat, tf, ts, stats = solve_sparse_factored_lp(
        F, min_improvement=1
    )
    val_full = -res_full.fun
    val_fact = -res_fact.fun
    agree = abs(val_full - val_fact) < TOL
    check(f"{label}: full LP solved",    res_full.success)
    check(f"{label}: factored LP solved", res_fact.success)
    check(f"{label}: both agree",        agree,
          f"full={val_full:.6f}, factored={val_fact:.6f}")
    return val_full, val_fact

# ---------------------------------------------------------------------------
# 2a. Matching pennies
#     F = [[1,-1],[-1,1]]  — Nash: both play 50/50, value = 0
# ---------------------------------------------------------------------------
F = np.array([[1., -1.],
              [-1., 1.]])
v1, v2 = game_value_both(F, "Matching pennies")
check("Matching pennies: value=0", abs(v1) < TOL, f"value={v1:.6f}")

# ---------------------------------------------------------------------------
# 2b. Rock-Paper-Scissors
#     Nash: uniform [1/3,1/3,1/3] for both, value = 0
# ---------------------------------------------------------------------------
F = np.array([[ 0., -1.,  1.],
              [ 1.,  0., -1.],
              [-1.,  1.,  0.]])
v1, v2 = game_value_both(F, "Rock-Paper-Scissors")
check("Rock-Paper-Scissors: value=0", abs(v1) < TOL, f"value={v1:.6f}")

# ---------------------------------------------------------------------------
# 2c. Strictly dominant strategy
#     Row 0 gives payoff 3 regardless of column. Value must be 3.
# ---------------------------------------------------------------------------
F = np.array([[3., 3., 3.],
              [1., 2., 0.],
              [2., 1., 1.]])
v1, v2 = game_value_both(F, "Dominant row")
check("Dominant row: value=3", abs(v1 - 3.0) < TOL, f"value={v1:.6f}")

# ---------------------------------------------------------------------------
# 2d. Diagonal payoff matrix (n x n identity scaled by c)
#     Nash: row player plays uniform, value = c/n
# ---------------------------------------------------------------------------
n, c = 5, 10.0
F = c * np.eye(n)
v1, v2 = game_value_both(F, f"{n}x{n} scaled identity")
check(f"Scaled identity: value=c/n={c/n}", abs(v1 - c/n) < TOL, f"value={v1:.6f}")

# ---------------------------------------------------------------------------
# 2e. All-ones matrix
#     Every entry = 1, so value = 1 regardless of strategies
# ---------------------------------------------------------------------------
F = np.ones((4, 6))
v1, v2 = game_value_both(F, "All-ones matrix")
check("All-ones: value=1", abs(v1 - 1.0) < TOL, f"value={v1:.6f}")

# ---------------------------------------------------------------------------
# 2f. Antisymmetric matrix (skew-symmetric)
#     For any skew-symmetric game F = -F^T, the value is always 0
#     by symmetry (both players have identical structure)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
B = rng.standard_normal((5, 5))
F = B - B.T   # skew-symmetric
v1, v2 = game_value_both(F, "Skew-symmetric (value=0)")
check("Skew-symmetric: value=0", abs(v1) < TOL, f"value={v1:.6f}")

print()