import numpy as np
import time
from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB

def solve_full_lp_v_version(F, method=1, x_ub=1.0):
    """
    Solve the row-player LP:
        min_{x,t} t
        s.t. F^T x <= t 1
             1^T x = 1
             x >= 0
    """
    t0 = time.perf_counter()

    n = F.shape[0]
    m = F.shape[1]

    model = gp.Model("full_lp")
    model.setParam("OutputFlag", 0)
    model.setParam("Method", method)


    # Variables
    if x_ub is None:
        x = model.addVars(n, lb=0.0, ub=GRB.INFINITY, name="x")
    else:
        x = model.addVars(n, lb=0.0, ub=float(x_ub), name="x")

    t = model.addVar(lb=-GRB.INFINITY, name="t")

    # Objective: minimize t
    model.setObjective(t, GRB.MINIMIZE)

    # Constraint: sum x = 1
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0)

    # Constraints: F^T x <= t
    # i.e. for each column j:
    # sum_i F[i,j] x[i] <= t
    for j in range(m):
        model.addConstr(
            gp.quicksum(F[i, j] * x[i] for i in range(n)) <= t
        )

    t_setup = time.perf_counter() - t0

    ts0 = time.perf_counter()
    model.optimize()
    t_solve = time.perf_counter() - ts0

    # Match scipy-style output
    class Result:
        pass

    res = Result()
    res.success = model.status == GRB.OPTIMAL
    res.status = model.status
    res.message = model.Status

    if res.success:
        x_vals = np.array([x[i].X for i in range(n)])
        t_val = t.X
        res.x = np.concatenate([x_vals, np.array([t_val])])
        res.fun = t_val
    else:
        res.x = None
        res.fun = None

    return res

def solve_full_lp_v_version_scipy(F, x_ub=1.0, method="highs-ipm"):
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
