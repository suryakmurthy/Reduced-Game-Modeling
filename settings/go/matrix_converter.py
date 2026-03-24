import numpy as np

F_raw = np.load('F_d6_k4.npy')
print(np.linalg.norm(F_raw + F_raw.T))
A = F_raw - F_raw.T
print(np.linalg.norm(A + A.T))
np.save("A_d6_k4.npy", A)