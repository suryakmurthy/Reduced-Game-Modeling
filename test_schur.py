import scipy.linalg as la
import numpy as np
import time
from old_version.solve_game_reduced import topk_schur_from_F_power

F_raw = np.load('settings/go/A_d6_k4.npy')

t1 = time.perf_counter()
la.schur(F_raw, output="real")
t2 = time.perf_counter()
print(t2-t1)
t1 = time.perf_counter()
topk_schur_from_F_power(F_raw, 45)
t2 = time.perf_counter()
print(t2-t1)
