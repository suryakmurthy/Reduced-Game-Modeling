import numpy as np
import os
from typing import List, Tuple, Optional
from tqdm import tqdm

from lc0_evaluator import LC0Evaluator


# ------------------------------------------------------------------
# Move extraction and interleaving
# ------------------------------------------------------------------

def extract_white_moves(line: Tuple[str, ...]) -> Tuple[str, ...]:
    """White's moves are at even indices: 0, 2, 4, ..."""
    return line[::2]


def extract_black_moves(line: Tuple[str, ...]) -> Tuple[str, ...]:
    """Black's moves are at odd indices: 1, 3, 5, ..."""
    return line[1::2]


def interleave(white_moves: Tuple[str, ...], black_moves: Tuple[str, ...]) -> List[str]:
    """
    Interleave White and Black move sequences into a full game line.
    White moves first (index 0), then Black (index 1), alternating.
    """
    result = []
    for i in range(max(len(white_moves), len(black_moves))):
        if i < len(white_moves):
            result.append(white_moves[i])
        if i < len(black_moves):
            result.append(black_moves[i])
    return result


# ------------------------------------------------------------------
# Core matrix builder
# ------------------------------------------------------------------

def build_F_matrix(
    lines: List[Tuple[str, ...]],
    evaluator: LC0Evaluator,
    fill_illegal: float = 0.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the raw evaluation matrix F where:
        F[i, j] = f(alpha_i, beta_j)
                = White win prob when White follows line i's moves
                  and Black follows line j's moves.

    Illegal combinations (incompatible opening lines) are filled with
    `fill_illegal` (default 0.5 = neutral/no information).

    Parameters
    ----------
    lines        : list of complete lines (each a tuple of UCI strings)
    evaluator    : LC0Evaluator instance
    fill_illegal : value for illegal move combinations
    verbose      : show progress bar

    Returns
    -------
    F    : (n, n) matrix of White win probabilities
    mask : (n, n) boolean matrix, True where the combination was legal
    """
    n = len(lines)
    white_moves = [extract_white_moves(line) for line in lines]
    black_moves = [extract_black_moves(line) for line in lines]

    F = np.full((n, n), fill_illegal, dtype=np.float32)
    mask = np.zeros((n, n), dtype=bool)

    illegal_count = 0

    iterator = tqdm(range(n * n), desc="Building F matrix") if verbose else range(n * n)

    for idx in iterator:
        i, j = divmod(idx, n)
        combined = interleave(white_moves[i], black_moves[j])
        score = evaluator.evaluate_line(combined)

        if score is not None:
            F[i, j] = score
            mask[i, j] = True
        else:
            illegal_count += 1

    if verbose:
        print(f"\n  Total pairs:   {n*n:,}")
        print(f"  Legal pairs:   {mask.sum():,} ({100*mask.sum()/(n*n):.1f}%)")
        print(f"  Illegal pairs: {illegal_count:,} ({100*illegal_count/(n*n):.1f}%)")
        print(f"  Cache size:    {evaluator.cache_size():,} unique positions")

    return F, mask


def build_payoff_matrix(
    lines: List[Tuple[str, ...]],
    evaluator: LC0Evaluator,
    fill_illegal: float = 0.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the skew-symmetric payoff matrix A = F - F.T

    Returns
    -------
    A          : (n, n) skew-symmetric payoff matrix
    F          : (n, n) raw win probability matrix
    valid_mask : (n, n) boolean, True where BOTH F[i,j] and F[j,i] were legal
    """
    F, mask = build_F_matrix(lines, evaluator, fill_illegal=fill_illegal, verbose=verbose)

    # Skew-symmetric payoff: match score for player 1 using strategy i vs j
    A = F - F.T

    # Only fully trust entries where both directions were legal
    valid_mask = mask & mask.T

    # Sanity check
    assert np.allclose(A, -A.T, atol=1e-5), "Payoff matrix is not skew-symmetric!"

    if verbose:
        valid_pairs = valid_mask.sum() // 2
        print(f"\n  Valid strategy pairs (both directions legal): {valid_pairs:,}")
        print(f"  Payoff range:        [{A.min():.4f}, {A.max():.4f}]")
        print(f"  Mean absolute payoff: {np.abs(A[valid_mask]).mean():.4f}")

    return A, F, valid_mask


# ------------------------------------------------------------------
# Analysis utilities
# ------------------------------------------------------------------

def svd_analysis(
    A: np.ndarray,
    k: int = 20,
    label: str = "",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD of the payoff matrix and singular value decay report.

    For a skew-symmetric matrix, singular values come in pairs (s, s),
    and the effective rank is always even. Fast decay = low-rank structure.

    Returns U, s, Vt from np.linalg.svd(A).
    """
    U, s, Vt = np.linalg.svd(A)

    if verbose:
        tag = f" [{label}]" if label else ""
        print(f"\nSVD Analysis{tag}  shape={A.shape}  total_var={( s**2).sum():.2f}")
        print(f"  {'rank':<6} {'s':<12} {'cumvar%'}")
        print(f"  {'-'*30}")
        for i in range(min(k, len(s))):
            cumvar = 100 * (s[:i+1]**2).sum() / (s**2).sum()
            print(f"  {i:<6} {s[i]:<12.6f} {cumvar:.1f}%")

    return U, s, Vt


def nash_equilibrium(A: np.ndarray) -> np.ndarray:
    """
    Compute Nash equilibrium mixed strategy via linear programming.

    Solves: max v  s.t.  A^T x <= v*1,  x >= 0,  sum(x) = 1

    Returns the mixed strategy distribution over strategies (length n).
    """
    from scipy.optimize import linprog
    n = A.shape[0]

    c = np.zeros(n + 1)
    c[-1] = -1.0  # maximise v

    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return result.x[:n]


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from line_enumerator import PolyglotLineEnumerator

    enumerator = PolyglotLineEnumerator("gm2001.bin", min_weight=500)
    lines = enumerator.enumerate(depth=6)
    print(f"\nUsing {len(lines)} lines for test.")

    with LC0Evaluator(nodes=1) as ev:
        A, F, valid_mask = build_payoff_matrix(lines, ev)
        U, s, Vt = svd_analysis(A, k=10, label="depth=6 min_weight=500")
        nash = nash_equilibrium(A)

    print(f"\nNash equilibrium (top 5 strategies by weight):")
    top = np.argsort(nash)[::-1][:5]
    for rank, idx in enumerate(top):
        print(f"  {rank+1}. {' '.join(lines[idx])}  weight={nash[idx]:.4f}")
