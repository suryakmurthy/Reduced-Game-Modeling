import itertools
import os
import numpy as np
from math import comb

def generate_strategies(T, k):
    """
    Yield strategies as tuples (x1,...,xk) with sum xi = T (stars and bars).
    """
    for dividers in itertools.combinations(range(T + k - 1), k - 1):
        prev = -1
        strat = []
        for d in dividers:
            strat.append(d - prev - 1)
            prev = d
        strat.append((T + k - 1) - prev - 1)
        yield tuple(strat)

def build_payoff_matrix(T, k, n=10_000, dtype=np.int8):
    """
    Build a payoff matrix for Colonel Blotto:
      payoff(a,b) = (#fields where a>b) - (#fields where a<b)

    Only takes the first n pure strategies (deterministic).
    """
    total = comb(T + k - 1, k - 1)
    if total < n:
        raise ValueError(f"Only {total} strategies exist for T={T}, k={k}, but n={n} requested.")

    # Take first n strategies without materializing all of them
    S = np.fromiter(
        (x for strat in itertools.islice(generate_strategies(T, k), n) for x in strat),
        dtype=np.int16,
        count=n * k
    ).reshape(n, k)

    # Broadcast compare to compute wins/losses for every pair
    A = S[:, None, :]          # (n,1,k)
    B = S[None, :, :]          # (1,n,k)
    wins = (A > B).sum(axis=2)   # (n,n)
    losses = (A < B).sum(axis=2) # (n,n)
    M = (wins - losses).astype(dtype)  # values in [-k, k], fits in int8 for small k

    return M, S

if __name__ == "__main__":
    T = 20
    k = 6
    n = 50_000 

    M, S = build_payoff_matrix(T, k, n=n, dtype=np.int8)

    print("Strategies array shape:", S.shape)
    print("Payoff matrix shape:", M.shape)
    print("Payoff range:", int(M.min()), int(M.max()))

    zero_frac = float(np.mean(M == 0))
    print("Zero fraction:", zero_frac)

    out_dir = "blotto"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "F.npy"), M)