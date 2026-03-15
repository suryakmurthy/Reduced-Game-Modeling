import numpy as np
import time
from tqdm import tqdm
import scipy.linalg as la
import matplotlib.pyplot as plt


def extract_black_moves(line):
    return tuple(line[i] for i in range(0, len(line), 2))


def extract_white_moves(line):
    return tuple(line[i] for i in range(1, len(line), 2))

def interleave(black_moves, white_moves):
    result = []
    b_idx, w_idx = 0, 0
    turn = 0  # 0 = Black, 1 = White

    while b_idx < len(black_moves) or w_idx < len(white_moves):
        if turn == 0:
            if b_idx < len(black_moves):
                result.append(black_moves[b_idx])
                b_idx += 1
            else:
                break
        else:
            if w_idx < len(white_moves):
                result.append(white_moves[w_idx])
                w_idx += 1
            else:
                break
        turn = 1 - turn

    return tuple(result)

def build_F_matrix(lines, evaluator, neutral=0.5):
    n = len(lines)
    alphas = [extract_black_moves(l) for l in lines]
    betas  = [extract_white_moves(l) for l in lines]

    print("  Collecting unique positions...")
    unique_positions = set()
    for i in range(n):
        for j in range(n):
            seq = interleave(alphas[i], betas[j])
            unique_positions.add(seq)
    print(f"  Unique positions: {len(unique_positions):,}  (vs {n*n:,} pairs)")

    uncached = [p for p in unique_positions if p not in evaluator.cache]
    print(f"  Uncached positions to evaluate: {len(uncached):,}")

    for seq in tqdm(uncached, desc="  Evaluating positions"):
        evaluator.evaluate(seq)

    print("  Filling matrix from cache...")
    F = np.full((n, n), neutral, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            seq = interleave(alphas[i], betas[j])
            val = evaluator.cache.get(seq)
            if val is not None:
                F[i, j] = val

    return F, np.ones((n, n), dtype=bool)

def build_F_matrix_prev(lines, evaluator, neutral=0.5):
    n = len(lines)
    F = np.full((n, n), neutral, dtype=np.float32)
    valid_mask = np.zeros((n, n), dtype=bool)

    alphas = [extract_black_moves(l) for l in lines]
    betas  = [extract_white_moves(l) for l in lines]

    total = n * n
    legal = 0
    illegal = 0

    t0 = time.time()

    with tqdm(total=total, desc="Building F matrix") as pbar:
        for i in range(n):
            for j in range(n):
                seq = interleave(alphas[i], betas[j])
                val = evaluator.evaluate(seq)
                if val is not None:
                    F[i, j] = val
                    legal += 1
                else:
                    illegal += 1
                pbar.update(1)

    elapsed = (time.time() - t0) / 60

    print(f"\n  Total pairs:   {total:,}")
    print(f"  Legal pairs:   {legal:,} ({100*legal/total:.1f}%)")
    print(f"  Illegal pairs: {illegal:,} ({100*illegal/total:.1f}%)")
    print(f"  Cache size:    {len(evaluator.cache):,} unique positions")
    print(f"  Build time:    {elapsed:.1f} min")

    # Valid mask: both F[i,j] and F[j,i] were legal
    legal_matrix = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            seq_ij = interleave(alphas[i], betas[j])
            seq_ji = interleave(alphas[j], betas[i])
            legal_matrix[i, j] = (evaluator.evaluate(seq_ij) is not None and
                                   evaluator.evaluate(seq_ji) is not None)

    return F, legal_matrix


def build_payoff_matrix(F):
    A = F - F.T
    assert np.allclose(A, -A.T, atol=1e-4), "Payoff matrix is not skew-symmetric!"
    return A


def svd_analysis(A, top_k=30):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    total_var = np.sum(s ** 2)
    cumvar = np.cumsum(s ** 2) / total_var

    print(f"\nSVD Analysis  shape={A.shape}  total_var={total_var:.2f}")
    print(f"  {'rank':<6} {'s':<14} {'cumvar%'}")
    print("  " + "-" * 30)
    for i in range(min(top_k, len(s))):
        print(f"  {i:<6} {s[i]:<14.6f} {100*cumvar[i]:.1f}%")

    return U, s, Vt


def plot_results(A, s, label, save_path):
    n = A.shape[0]
    cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Go Opening Payoff Matrix [{label}]  n={n}", fontsize=12)

    # Singular values
    ax = axes[0]
    ax.plot(s[:50], marker="o", markersize=3)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular Value Decay")
    ax.grid(True, alpha=0.3)

    # Cumulative variance
    ax = axes[1]
    ax.plot(cumvar[:100] * 100, marker="o", markersize=3)
    ax.axhline(90, color="red", linestyle="--", linewidth=1, label="90%")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Cumulative Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Payoff matrix heatmap
    ax = axes[2]
    im = ax.imshow(A, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title("Payoff Matrix A")
    ax.set_xlabel("Strategy j")
    ax.set_ylabel("Strategy i")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved plot to {save_path}")
