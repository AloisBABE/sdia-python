import numpy as np


def is_symetric(M):
    is_sym = True
    m, n = np.shape(M)
    for i in range(1, m):
        for j in range(i):
            is_sym = is_sym and (M[i, j] == M[j, i])
    return is_sym
