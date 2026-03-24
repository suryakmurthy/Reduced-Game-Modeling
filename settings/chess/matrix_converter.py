import numpy as np

F_raw = np.load('F_d8_mw10.npy')
print(np.linalg.norm(F_raw + F_raw.T))
A = F_raw - F_raw.T
print(np.linalg.norm(A + A.T))
np.save("A_d8_mw10.npy", A)