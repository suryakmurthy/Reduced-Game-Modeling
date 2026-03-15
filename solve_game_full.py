import numpy as np
import time
from scipy.optimize import linprog

def solve_full_lp_v_version(F, x_ub=1.0, method="highs-ipm"):
    """
    Solve the row-player LP:
        min_{x,t} t
        s.t. F^T x <= t 1
             1^T x = 1
             x >= 0

    Returns scipy.optimize.OptimizeResult.
    """
    t0 = time.perf_counter()
    n = F.shape[0]
    m = F.shape[1]

    c = np.zeros(n + 1)
    c[n] = 1.0

    A_ub = np.hstack([F.T, -np.ones((m, 1))])
    b_ub = np.zeros(m)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    if x_ub is None:
        x_bounds = [(0.0, None)] * n
    else:
        x_bounds = [(0.0, float(x_ub))] * n

    bounds = x_bounds + [(None, None)]

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
    )
    return res
