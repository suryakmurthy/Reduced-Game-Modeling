import numpy as np
from collections import Counter
from old_version.solve_game_full import solve_full_lp_v_version
from old_version.solve_game_sampling import solve_reduced_lp_using_sampling

# ── Config ────────────────────────────────────────────────────────────────────
input_matrix = 'settings/blotto/F_1000.npy'
method = 'highs-ipm'
SUPPORT_TOL = 1e-6
N_SAMPLES = 100
DIM = 150   # use a regime where exploitability ≈ 0

# ── Load game ─────────────────────────────────────────────────────────────────
F_raw = np.load(input_matrix)
F_true = -F_raw
m, n = F_true.shape

# ── Ground truth value ────────────────────────────────────────────────────────
res_full = solve_full_lp_v_version(F_true, method=method)
y_full   = res_full.x[:n].copy()
y_full  /= y_full.sum()
v_true   = (F_true @ y_full).min()

print(f"True game value: {v_true:.8f}\n")

# ── Collect equilibria via sampling ───────────────────────────────────────────
solutions = []

for seed in range(N_SAMPLES):
    _, y, *_ = solve_reduced_lp_using_sampling(
        F_true, k=DIM//2, seed=seed, method=method
    )

    if y is None:
        continue

    # keep only (near-)optimal solutions
    v_sampled = (F_true @ y).min()
    if abs(v_sampled - v_true) < 1e-8:
        solutions.append(y)

print(f"Collected {len(solutions)} near-optimal equilibria\n")

# ── Extract supports ──────────────────────────────────────────────────────────
supports = []
for y in solutions:
    supp = set(np.where(y > SUPPORT_TOL)[0])
    supports.append(supp)

# ── Intersection of supports ──────────────────────────────────────────────────
if supports:
    common_support = set.intersection(*supports)
else:
    common_support = set()

print("=== Support Intersection ===")
print(f"Size: {len(common_support)}")
print(f"Indices: {sorted(common_support)}\n")

# ── Union of supports ─────────────────────────────────────────────────────────
if supports:
    union_support = set.union(*supports)
else:
    union_support = set()

print("=== Support Union ===")
print(f"Size: {len(union_support)}\n")

# ── Frequency analysis ────────────────────────────────────────────────────────
counter = Counter()

for supp in supports:
    for i in supp:
        counter[i] += 1

freq = {i: count / len(supports) for i, count in counter.items()}

# sort by frequency
sorted_freq = sorted(freq.items(), key=lambda x: -x[1])

print("=== Top 20 Most Frequent Strategies ===")
print(f"{'Idx':>8} | {'Frequency':>10}")
print("-" * 25)
for i, f in sorted_freq[:20]:
    print(f"{i:>8} | {f:>10.3f}")

# ── Buckets ───────────────────────────────────────────────────────────────────
always = [i for i, f in freq.items() if f == 1.0]
often  = [i for i, f in freq.items() if f > 0.8]
rare   = [i for i, f in freq.items() if f < 0.2]

print("\n=== Frequency Buckets ===")
print(f"Always present (freq=1.0): {len(always)}")
print(f"Often present  (freq>0.8): {len(often)}")
print(f"Rare           (freq<0.2): {len(rare)}")

# ── Optional: compare with true support ───────────────────────────────────────
true_support = set(np.where(y_full > SUPPORT_TOL)[0])

overlap_with_true = {
    i: freq.get(i, 0.0)
    for i in true_support
}

print("\n=== True Support Frequencies ===")
print(f"{'Idx':>8} | {'Frequency':>10}")
print("-" * 25)
for i in sorted(true_support):
    print(f"{i:>8} | {overlap_with_true[i]:>10.3f}")