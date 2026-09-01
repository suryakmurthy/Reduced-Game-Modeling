import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB


def solve_subgame_row(F_sub, method=1, x_ub=1.0):
    """
    Solve the row-player LP on a subgame matrix F_sub:
        min_{x, t}  t
        s.t.  F_sub^T x <= t * 1
              1^T x = 1
              x >= 0

    Returns the result object and row mixed strategy p over R_bar.
    """
    m, n = F_sub.shape

    model = gp.Model("subgame_row")
    model.setParam("OutputFlag", 0)
    model.setParam("Method", method)

    x = model.addVars(m, lb=0.0, ub=float(x_ub) if x_ub is not None else GRB.INFINITY, name="x")
    t = model.addVar(lb=-GRB.INFINITY, name="t")

    model.setObjective(t, GRB.MINIMIZE)

    model.addConstr(gp.quicksum(x[i] for i in range(m)) == 1.0)

    for j in range(n):
        model.addConstr(
            gp.quicksum(F_sub[i, j] * x[i] for i in range(m)) <= t
        )

    model.optimize()

    class Result:
        pass

    res = Result()
    res.success = model.status == GRB.OPTIMAL
    res.status = model.status

    if res.success:
        p = np.array([x[i].X for i in range(m)])
        p = np.maximum(p, 0.0)
        p /= p.sum()
        res.fun = t.X
        res.p = p
    else:
        res.p = None
        res.fun = None

    return res


def solve_subgame_col(F_sub, method=1):
    """
    Solve the column-player LP on a subgame matrix F_sub:
        max_{q, s}  s
        s.t.  F_sub q >= s * 1
              1^T q = 1
              q >= 0

    Solving this explicitly rather than relying on dual variables from the
    row LP avoids numerical issues with incorrect or missing duals.

    Returns the column mixed strategy q over C_bar.
    """
    m, n = F_sub.shape

    model = gp.Model("subgame_col")
    model.setParam("OutputFlag", 0)
    model.setParam("Method", method)

    q = model.addVars(n, lb=0.0, name="q")
    s = model.addVar(lb=-GRB.INFINITY, name="s")

    model.setObjective(s, GRB.MAXIMIZE)

    model.addConstr(gp.quicksum(q[j] for j in range(n)) == 1.0)

    for i in range(m):
        model.addConstr(
            gp.quicksum(F_sub[i, j] * q[j] for j in range(n)) >= s
        )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        q_vals = np.array([q[j].X for j in range(n)])
        q_vals = np.maximum(q_vals, 0.0)
        s = q_vals.sum()
        if s > 0:
            q_vals /= s
        else:
            q_vals = np.ones(n) / n
        return q_vals
    else:
        return np.ones(n) / n


def row_best_response(F, q_full):
    """
    Row player (minimizer) best response to a full column mixture q_full.
    Returns the index of the best response row and its value.

    r* = argmin_r  F[r, :] @ q_full
    v_lower = F[r*, :] @ q_full
    """
    values = F @ q_full          # shape (M,)
    r_star = int(np.argmin(values))
    return r_star, float(values[r_star])


def col_best_response(F, p_full):
    """
    Column player (maximizer) best response to a full row mixture p_full.
    Returns the index of the best response column and its value.

    c* = argmax_c  p_full @ F[:, c]
    v_upper = p_full @ F[:, c*]
    """
    values = p_full @ F          # shape (N,)
    c_star = int(np.argmax(values))
    return c_star, float(values[c_star])


def solve_double_oracle(
    F, eps=1e-6, max_iter=1000, x_ub=1.0, method=1, seed=0
):
    """
    Double Oracle algorithm for zero-sum matrix games (McMahan et al., 2003).

    Iteratively grows a restricted subgame by adding best-response rows and
    columns until a saddle point is found (i.e. neither player can improve by
    deviating to a strategy outside the current subgame).

    The algorithm maintains:
      R_bar : set of row indices considered so far
      C_bar : set of column indices considered so far

    On each iteration:
      1. Solve the restricted subgame LP on F[R_bar, :][:, C_bar] for both
         players explicitly (row LP and column LP separately).
      2. Find the row best response r* to the current column mixture q.
         v_lower = F[r*, :] @ q  (lower bound on game value).
      3. Find the column best response c* to the current row mixture p.
         v_upper = p @ F[:, c*]  (upper bound on game value).
      4. If v_upper - v_lower < eps: converged.
      5. If both r* and c* are already in R_bar and C_bar: converged
         (saddle point certificate — neither player can improve by deviating).
      6. Otherwise add any new r* and/or c* to R_bar and C_bar and repeat.

    Parameters
    ----------
    F        : (M x N) full game matrix; F[i,j] is cost to row player
    eps      : convergence tolerance on (v_upper - v_lower)
    max_iter : maximum number of iterations
    x_ub     : upper bound on each x variable in the row subgame LP
    method   : Gurobi LP method flag
    seed     : RNG seed for initial row/column selection

    Returns
    -------
    p_full   : (M,) mixed strategy for row player over full space
    q_full   : (N,) mixed strategy for col player over full space
    v        : game value (average of final bounds)
    R_bar    : list of row indices in final subgame
    C_bar    : list of column indices in final subgame
    n_iter   : number of iterations run
    t_setup  : cumulative time in best-response oracles (seconds)
    t_solve  : cumulative time in subgame LP solves (seconds)
    history  : list of (v_lower, v_upper) per iteration
    """
    M, N = F.shape
    rng = np.random.default_rng(seed)

    # Initialise R_bar and C_bar with one arbitrary row and column each
    r0 = int(rng.integers(0, M))
    c0 = int(rng.integers(0, N))
    R_bar = [r0]
    C_bar = [c0]

    t_setup = 0.0
    t_solve = 0.0
    history = []
    p_full = None
    q_full = None
    v_lower = -np.inf
    v_upper = np.inf

    for iteration in range(max_iter):

        # --- Step 1: solve both player LPs on the restricted subgame ---
        F_sub = F[np.ix_(R_bar, C_bar)]

        ts = time.perf_counter()
        res_row = solve_subgame_row(F_sub, method=method, x_ub=x_ub)
        q_sub = solve_subgame_col(F_sub, method=method)
        t_solve += time.perf_counter() - ts

        if not res_row.success:
            break

        # Lift strategies back to full space
        p_full = np.zeros(M)
        p_full[R_bar] = res_row.p

        q_full = np.zeros(N)
        q_full[C_bar] = q_sub

        # --- Steps 2 & 3: best response oracles ---
        to = time.perf_counter()
        r_star, v_lower = row_best_response(F, q_full)
        c_star, v_upper = col_best_response(F, p_full)
        t_setup += time.perf_counter() - to

        history.append((v_lower, v_upper))

        # --- Step 4: convergence check on bounds ---
        if v_upper - v_lower < eps:
            break

        # --- Step 5: saddle point certificate ---
        # If both best responses are already in the subgame, neither player
        # can improve by deviating outside it — we have a saddle point.
        r_already_in = r_star in R_bar
        c_already_in = c_star in C_bar
        if r_already_in and c_already_in:
            break

        # --- Step 6: grow the subgame ---
        if not r_already_in:
            R_bar.append(r_star)
        if not c_already_in:
            C_bar.append(c_star)

    v = 0.5 * (v_lower + v_upper)
    return p_full, q_full, v, R_bar, C_bar, iteration + 1, t_setup, t_solve, history