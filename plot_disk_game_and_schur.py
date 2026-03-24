import argparse
import os

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import scipy.linalg as la

from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError


# ── helpers ───────────────────────────────────────────────────────────────────

def _origin_in_convex_hull(points: np.ndarray, tol: float = 1e-10) -> bool | None:
    """
    Return True if the origin lies inside (or on the boundary of) the convex
    hull of *points*, False if it is strictly outside, and None if the hull
    cannot be computed (e.g. fewer than 3 non-collinear points).
    """
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    return bool(np.all(hull.equations[:, -1] <= tol))


def _add_rotation_field(
    ax: plt.Axes,
    omega_signed: float,
    n_grid: int = 24,
    color: str = "dimgray",
    alpha: float = 0.30,
    density: float = 0.75,
) -> None:
    """
    Overlay a rotational streamplot on *ax* encoding the sign of omega.

    The Schur block [[0, -ω], [ω, 0]] generates the linear flow
        dx/dt = -ω · y
        dy/dt =  ω · x
    so positive ω → CCW, negative ω → CW.

    Limits are read from the axes after the scatter has been drawn and are
    restored afterwards (streamplot can silently expand them).
    """
    xl = np.array(ax.get_xlim())
    yl = np.array(ax.get_ylim())

    xs = np.linspace(xl[0], xl[1], n_grid)
    ys = np.linspace(yl[0], yl[1], n_grid)
    X, Y = np.meshgrid(xs, ys)

    U = -omega_signed * Y
    V =  omega_signed * X

    sp = ax.streamplot(
        xs, ys, U, V,
        density=density,
        color=color,
        linewidth=0.55,
        arrowsize=0.75,
    )
    sp.lines.set_alpha(alpha)
    sp.arrows.set_alpha(alpha)

    ax.set_xlim(xl)
    ax.set_ylim(yl)


# ── decomposition ─────────────────────────────────────────────────────────────

def topk_schur_from_F_power(
    F: npt.ArrayLike,
    k: int,
    p: int = 10,
    q: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate Schur decomposition of F via randomised subspace iteration.

    Returns
    -------
    U : (n, r) ndarray   Approximate Schur vectors.
    D : (r, r) ndarray   Real Schur form (skew-symmetric block diagonal).
    """
    n = F.shape[0]
    k = min(k, n // 2)
    r = min(n, 2 * k + p)

    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, r))
    V, _ = la.qr(V, mode="economic")

    for _ in range(q):
        V = F @ (F.T @ V)
        V, _ = la.qr(V, mode="economic")

    B = V.T @ F @ V
    B = 0.5 * (B - B.T)

    D, W = la.schur(B, output="real")
    U = V @ W
    return U, D


def extract_omegas(
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ω values from the real Schur form of a skew-symmetric matrix.

    Each 2×2 block has the form [[0, -ω], [ω, 0]], so ω = D[2i+1, 2i].
    The sign of ω encodes the direction of rotation (+ CCW, − CW).

    Returns
    -------
    omegas_abs    : (r//2,) ndarray  Magnitudes, sorted descending.
    omegas_signed : (r//2,) ndarray  Signed values in the same order.
    order         : (r//2,) ndarray of int  Original block indices.
    """
    num_blocks = D.shape[0] // 2
    signed = np.array([D[2 * i + 1, 2 * i] for i in range(num_blocks)])
    order  = np.argsort(np.abs(signed))[::-1]
    return np.abs(signed)[order], signed[order], order


def get_disc_games(
    F: np.ndarray,
    U: np.ndarray,
    omegas_abs: np.ndarray,
    omegas_signed: np.ndarray,
    order: np.ndarray,
    n: int,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Build 2-D disc-game projections for the top-n blocks.

    Returns
    -------
    disc_games     : list of (m, 2) ndarray
    signed_omegas  : list of float   Signed ω for each returned game.
    """
    disc_games, signed_out = [], []
    for rank in range(n):
        k = order[rank]
        U_2k = U[:, 2 * k: 2 * k + 2]
        w_abs = omegas_abs[rank] if omegas_abs[rank] >= 0.1 else 1.0
        Y = (U_2k.T @ F).T / np.sqrt(w_abs)
        disc_games.append(Y)
        signed_out.append(float(omegas_signed[rank]))
    return disc_games, signed_out


# ── reconstruction error ──────────────────────────────────────────────────────

def compute_reconstruction_errors(
    F: np.ndarray,
    U: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Compute the spectral-norm reconstruction error for each prefix of blocks.

    For k = 1 … (r // 2), the rank-2k approximation is:

        F̂_k = U[:, :2k] @ D[:2k, :2k] @ U[:, :2k].T

    and the error is  ‖F − F̂_k‖₂  (largest singular value of the residual).

    Parameters
    ----------
    F : (n, n) ndarray   Original payoff matrix.
    U : (n, r) ndarray   Schur vectors (columns ordered by descending |ω|).
    D : (r, r) ndarray   Real Schur form (2×2 block diagonal).

    Returns
    -------
    errors : (r // 2,) ndarray   Spectral-norm error at each block count.
    """
    num_blocks = D.shape[0] // 2
    errors = np.empty(num_blocks)
    F_hat = np.zeros_like(F)

    for k in range(num_blocks):
        u_k = U[:, 2 * k: 2 * k + 2]
        d_k = D[2 * k: 2 * k + 2, 2 * k: 2 * k + 2]
        F_hat = F_hat + u_k @ d_k @ u_k.T
        residual = F - F_hat
        errors[k] = np.linalg.norm(residual, ord=2)

    return errors


# ── game value (LP) ───────────────────────────────────────────────────────────

def solve_game_value(F: np.ndarray) -> float:
    """
    Solve the column-player security LP:

        min_{x, v}  v
        s.t.  F^T x  ≤  v · 1,   1^T x = 1,   x ≥ 0.

    Variables are ordered as  z = [x (n), v (1)].

    Returns the optimal value v*.
    """
    n = F.shape[0]

    # Objective: minimise v  →  c = [0, …, 0, 1]
    c = np.zeros(n + 1)
    c[-1] = 1.0

    # Inequality:  F^T x - v · 1 ≤ 0
    # [F^T | -1]  z  ≤  0
    A_ub = np.hstack([F.T, -np.ones((n, 1))])
    b_ub = np.zeros(n)

    # Equality:  1^T x = 1,  pad with 0 for v
    A_eq = np.ones((1, n + 1))
    A_eq[0, -1] = 0.0
    b_eq = np.array([1.0])

    # Bounds:  x_i ≥ 0,  v free
    bounds = [(0.0, None)] * n + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return float("nan")
    return float(res.fun)


def compute_value_differences(
    F: np.ndarray,
    U: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute the saddle-point value difference  |v* − v̂_k|  for each prefix
    of k blocks in the ordered Schur decomposition.

    For each k = 1 … (r // 2):
        F̂_k  = U[:, :2k] @ D[:2k, :2k] @ U[:, :2k].T
        v̂_k  = game value of F̂_k  (solved via LP)
        error = |v* − v̂_k|

    The theoretical guarantee from the paper is:
        |v* − v̂_k| ≤ ‖F − F̂_k‖₂ = ω_{k+1}

    Parameters
    ----------
    F : (n, n) ndarray   Original payoff matrix.
    U : (n, r) ndarray   Schur vectors ordered by descending |ω|.
    D : (r, r) ndarray   Real Schur form (2×2 block diagonal, same order).

    Returns
    -------
    value_diffs : (r // 2,) ndarray   |v* − v̂_k| at each block count.
    v_true      : float               Exact game value of F.
    """
    v_true = solve_game_value(F)

    num_blocks = D.shape[0] // 2
    value_diffs = np.empty(num_blocks)
    F_hat = np.zeros_like(F)

    for k in range(num_blocks):
        u_k = U[:, 2 * k: 2 * k + 2]
        d_k = D[2 * k: 2 * k + 2, 2 * k: 2 * k + 2]
        F_hat = F_hat + u_k @ d_k @ u_k.T
        v_hat = solve_game_value(F_hat)
        value_diffs[k] = abs(v_true - v_hat)

    return value_diffs, v_true


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_disc_games(
    F: npt.ArrayLike,
    n: int = 4,
    k_spectrum: int = 20,
    p: int = 10,
    q: int = 5,
    seed: int = 0,
    output_path: str | None = None,
    point_size: int = 20,
    alpha: float = 0.6,
    cmap: str = "viridis",
) -> None:
    """
    Plot the top-n disc games (with rotational vector-field overlays), the
    full ω spectrum, the spectral-norm reconstruction error curve, and the
    saddle-point value difference |v* − v̂_k| vs. theoretical bound ω_{k+1}.

    Parameters
    ----------
    F            : (m, m) array-like   Payoff matrix.
    n            : int                 Disc-game subplots to display.
    k_spectrum   : int                 Blocks extracted for ω spectrum (≥ n).
    p            : int                 Oversampling for randomised range finder.
    q            : int                 Power-iteration passes.
    seed         : int                 RNG seed.
    output_path  : str or None         Save path; None → interactive.
    point_size   : int                 Scatter marker size.
    alpha        : float               Scatter marker transparency.
    cmap         : str                 Matplotlib colormap.
    """
    F = np.asarray(F, dtype=float)

    k_total = max(n, k_spectrum)
    U, D = topk_schur_from_F_power(F, k=k_total, p=p, q=q, seed=seed)
    omegas_abs, omegas_signed, order = extract_omegas(D)
    disc_games, signed_omegas = get_disc_games(F, U, omegas_abs, omegas_signed, order, n=n)

    # Re-order U / D columns so block index matches descending-|ω| rank
    perm      = np.concatenate([[2 * k, 2 * k + 1] for k in order])
    U_ordered = U[:, perm]
    D_ordered = D[np.ix_(perm, perm)]

    recon_errors          = compute_reconstruction_errors(F, U_ordered, D_ordered)
    value_diffs, v_true   = compute_value_differences(F, U_ordered, D_ordered)

    ratings     = F.mean(axis=1)
    colour_norm = plt.Normalize(ratings.min(), ratings.max())
    colours     = cm.get_cmap(cmap)(colour_norm(ratings))

    # ── layout ────────────────────────────────────────────────────────────────
    cols      = min(n, 4)
    disc_rows = (n + cols - 1) // cols

    # Bottom row: three equal panels (ω spectrum | recon error | value diff)
    fig = plt.figure(figsize=(4 * max(cols, 3), 4 * disc_rows + 3.5))
    gs  = gridspec.GridSpec(
        disc_rows + 1, 3,
        figure=fig,
        hspace=0.60,
        wspace=0.45,
        height_ratios=[4] * disc_rows + [3],
    )

    # Disc-game axes span all 3 columns via a nested GridSpec
    disc_gs = gridspec.GridSpecFromSubplotSpec(
        disc_rows, cols,
        subplot_spec=gs[:disc_rows, :],
        hspace=0.55,
        wspace=0.40,
    )
    disc_axes = [
        fig.add_subplot(disc_gs[r, c])
        for r in range(disc_rows)
        for c in range(cols)
    ]

    omega_ax = fig.add_subplot(gs[disc_rows, 0])
    error_ax = fig.add_subplot(gs[disc_rows, 1])
    value_ax = fig.add_subplot(gs[disc_rows, 2])

    # ── disc-game subplots ────────────────────────────────────────────────────
    for i, (Y, w_signed) in enumerate(zip(disc_games, signed_omegas)):
        ax = disc_axes[i]

        ax.scatter(Y[:, 0], Y[:, 1], c=colours, s=point_size, alpha=alpha,
                   zorder=3)

        ax.set_aspect("equal", adjustable="datalim")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--", zorder=2)
        ax.axvline(0, color="k", linewidth=0.4, linestyle="--", zorder=2)
        ax.grid(True, alpha=0.25, zorder=1)

        _add_rotation_field(ax, w_signed)

        direction = "CCW ↺" if w_signed > 0 else "CW ↻"
        ax.set_title(
            f"Disc Game {i + 1}  (ω = {w_signed:+.4f},  {direction})",
            fontsize=9,
        )
        ax.set_xlabel("dim 1", labelpad=6)
        ax.set_ylabel("dim 2", labelpad=6)

        inside = _origin_in_convex_hull(Y)
        if inside is None:
            hull_label, hull_colour = "hull: N/A", "grey"
        elif inside:
            hull_label, hull_colour = "origin ∈ hull ✓", "green"
        else:
            hull_label, hull_colour = "origin ∉ hull ✗", "red"

        ax.text(
            0.03, 0.97, hull_label,
            transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            color=hull_colour, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7,
                      ec=hull_colour),
            zorder=4,
        )

    for j in range(i + 1, len(disc_axes)):
        disc_axes[j].set_visible(False)

    # ── ω spectrum panel ──────────────────────────────────────────────────────
    ranks    = np.arange(1, len(omegas_abs) + 1)
    selected = np.zeros(len(omegas_abs), dtype=bool)
    selected[:n] = True

    bar_colours = []
    for idx in range(len(omegas_abs)):
        if selected[idx]:
            bar_colours.append("crimson" if omegas_signed[idx] < 0 else "steelblue")
        else:
            bar_colours.append(
                "lightsalmon" if omegas_signed[idx] < 0 else "lightsteelblue"
            )

    omega_ax.bar(ranks, omegas_abs, color=bar_colours, width=0.6, zorder=2)
    omega_ax.plot(ranks, omegas_abs, linewidth=1.0, color="dimgray",
                  zorder=3, alpha=0.6)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="steelblue",      label=f"Top-{n}  CCW ↺"),
        Patch(facecolor="crimson",        label=f"Top-{n}  CW ↻"),
        Patch(facecolor="lightsteelblue", label="Other  CCW ↺"),
        Patch(facecolor="lightsalmon",    label="Other  CW ↻"),
    ]
    omega_ax.legend(handles=legend_handles, fontsize=7, framealpha=0.7, ncol=2)
    omega_ax.set_xlabel("Disc game rank $k$", labelpad=8)
    omega_ax.set_ylabel(r"$|\omega_k|$", labelpad=8)
    omega_ax.set_title(
        f"ω magnitude spectrum — {len(omegas_abs)} blocks  (q={q}, p={p})",
        fontsize=9,
    )
    omega_ax.xaxis.set_major_locator(
        ticker.MaxNLocator(integer=True, nbins=len(omegas_abs))
    )
    omega_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    omega_ax.tick_params(labelsize=8)
    omega_ax.set_xlim(0.5, len(omegas_abs) + 0.5)
    omega_ax.set_ylim(bottom=0)
    omega_ax.grid(True, alpha=0.25, linestyle="--", zorder=1)

    # ── reconstruction error panel ────────────────────────────────────────────
    error_ranks = np.arange(1, len(recon_errors) + 1)

    error_ax.plot(
        error_ranks, recon_errors,
        color="darkorange", linewidth=1.5, zorder=3, marker="o",
        markersize=3, label=r"$\|F - \hat{F}_k\|_2$",
    )
    error_ax.axvline(n, color="crimson", linewidth=1.0, linestyle="--",
                     zorder=2, label=f"Top-{n} cutoff")

    if n <= len(recon_errors):
        err_at_n = recon_errors[n - 1]
        error_ax.annotate(
            f"{err_at_n:.4f}",
            xy=(n, err_at_n),
            xytext=(n + 0.6, err_at_n),
            fontsize=7, color="crimson",
            arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8),
            va="center",
        )

    error_ax.set_xlabel("Number of blocks $k$", labelpad=8)
    error_ax.set_ylabel(r"$\|F - \hat{F}_k\|_2$", labelpad=8)
    error_ax.set_title(
        r"Spectral-norm reconstruction error  $\|F - U_k D_k U_k^\top\|_2$",
        fontsize=9,
    )
    error_ax.xaxis.set_major_locator(
        ticker.MaxNLocator(integer=True, nbins=len(recon_errors))
    )
    error_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    error_ax.tick_params(labelsize=8)
    error_ax.set_xlim(0.5, len(recon_errors) + 0.5)
    error_ax.set_ylim(bottom=0)
    error_ax.legend(fontsize=7, framealpha=0.7)
    error_ax.grid(True, alpha=0.25, linestyle="--", zorder=1)

    # ── saddle-point value difference panel ───────────────────────────────────
    # Theory (Section 6.3):  |v* − v̂_k| ≤ ‖F − F̂_k‖₂ = ω_{k+1}
    # recon_errors[k] = ‖F − F̂_{k+1}‖₂, so the bound for the k-block
    # approximation is recon_errors[k] (one step ahead in the array).
    # For the last block we fall back to the final recon_errors value.
    num_blocks = len(value_diffs)
    spectral_bound = np.empty(num_blocks)
    spectral_bound[:-1] = recon_errors[1:]
    spectral_bound[-1]  = recon_errors[-1]

    val_ranks = np.arange(1, num_blocks + 1)
    print(value_diffs[:5])
    print(spectral_bound[:5])
    value_ax.plot(
        val_ranks, value_diffs,
        color="mediumseagreen", linewidth=1.5, zorder=3, marker="o",
        markersize=3, label=r"$|v^* - \hat{v}_k|$  (actual)",
    )
    # value_ax.plot(
    #     val_ranks, spectral_bound,
    #     color="slategray", linewidth=1.2, linestyle="--", zorder=2,
    #     label=r"$\omega_{k+1}$  (bound)",
    # )
    # value_ax.fill_between(
    #     val_ranks, value_diffs, spectral_bound,
    #     where=(spectral_bound >= value_diffs),
    #     alpha=0.12, color="slategray", label="slack",
    # )
    value_ax.axvline(n, color="crimson", linewidth=1.0, linestyle=":",
                     zorder=2, label=f"Top-{n} cutoff")

    # Annotate exact game value
    value_ax.text(
        0.97, 0.97,
        f"$v^*$ = {v_true:.4f}",
        transform=value_ax.transAxes,
        fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8,
                  ec="mediumseagreen"),
    )

    value_ax.set_xlabel("Number of blocks $k$", labelpad=8)
    value_ax.set_ylabel(r"$|v^* - \hat{v}_k|$", labelpad=8)
    # value_ax.set_yscale('log')
    value_ax.set_title(
        r"Saddle-point value difference  $|v^* - \hat{v}_k|$"
        "\nvs. bound "
        r"$\omega_{k+1}$",
        fontsize=9,
    )
    value_ax.xaxis.set_major_locator(
        ticker.MaxNLocator(integer=True, nbins=num_blocks)
    )
    value_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    value_ax.tick_params(labelsize=8)
    value_ax.set_xlim(0.5, num_blocks + 0.5)
    # value_ax.set_ylim(bottom=0)
    value_ax.legend(fontsize=7, framealpha=0.7)
    value_ax.grid(True, alpha=0.25, linestyle="--", zorder=1)

    # ── shared colorbar ───────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=colour_norm)
    sm.set_array([])
    fig.colorbar(
        sm, ax=disc_axes[:i + 1],
        shrink=0.6, label="Mean payoff (row)",
    )

    fig.suptitle("Disc Games of Payoff Matrix F", fontsize=12, y=1.01)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
        plt.close()
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot top-n disc games with rotation fields and ω spectrum."
    )
    parser.add_argument("--input-matrix", required=True,
                        help="Path to .npy payoff matrix")
    parser.add_argument("--n",          type=int,   default=4,
                        help="Number of disc-game subplots (default: 4)")
    parser.add_argument("--k-spectrum", type=int,   default=20,
                        help="Blocks for ω spectrum (default: 20)")
    parser.add_argument("--p",          type=int,   default=10,
                        help="Oversampling parameter (default: 10)")
    parser.add_argument("--q",          type=int,   default=5,
                        help="Power-iteration passes (default: 5)")
    parser.add_argument("--seed",       type=int,   default=0,
                        help="RNG seed (default: 0)")
    parser.add_argument("--negate",     action="store_true",
                        help="Use F = -A")
    parser.add_argument("--output-path", type=str,  default=None,
                        help="Save path (default: show interactively)")
    parser.add_argument("--point-size", type=int,   default=20,
                        help="Scatter marker size (default: 20)")
    parser.add_argument("--alpha",      type=float, default=0.6,
                        help="Marker transparency (default: 0.6)")
    parser.add_argument("--cmap",       type=str,   default="viridis",
                        help="Colormap (default: viridis)")
    return parser.parse_args()


def main():
    args = parse_args()
    F = np.load(args.input_matrix)
    if args.negate:
        F = -F
    print(f"Loaded {args.input_matrix}  shape={F.shape}")
    plot_disc_games(
        F,
        n=args.n,
        k_spectrum=args.k_spectrum,
        p=args.p,
        q=args.q,
        seed=args.seed,
        output_path=args.output_path,
        point_size=args.point_size,
        alpha=args.alpha,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()