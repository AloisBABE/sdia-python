def is_symetric(M):
    is_sym = True
    m, n = M.shape()
    for i in range(1, m):
        for j in range(i):
            is_sym = is_sym and (M[i, j] == M[j, i])
