import numpy as np
import math as m


def fun_loss_relaxed(P, V, n, m, l):
    A = np.zeros(n, n)
    B = np.zeros(n, n, m)
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, m):
                A[i, j] += V[i, k] * P[j, k]
                B[i, j, k] = V[i, k] * P[j, k]

    sum1 = 0
    res = 0
    max_res = 0
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                for k in range(0, m):
                    res = A[i, j] - B[i, j, k] - A[i, i]
                    max_res = max(0, res)
                    sum1 += max_res

    sum2 = 0
    for k in range(0, m):
        for i in range(0, n):
            sum2 += P[i, k] * m.log(P[i, k] + np.spacing(1))

    sum2 = l * sum2

    return sum1 - sum2
