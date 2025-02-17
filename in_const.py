import numpy as np


def in_constraints(P, V, n, m):
    A = np.zeros((n, n))
    B = np.zeros((n, n, m))

    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, m):
                A[i, j] += V[i, k] * P[j, k]
                B[i, j, k] = V[i, k] * P[j, k]
    #print(B)
    ineq_const = np.zeros((n, n, m)).reshape(-1)
    #print(ineq_const.shape)
    count = 0
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                for k in range(0, m):
                        ineq_const[count] = A[i, i] - A[i, j] + B[i, j, k]
            count += 1
    print(ineq_const)
    return ineq_const


P = np.array([[1, 0], [0, 1]])
# print(loss(p))
n = len(P[:, 0])
m = len(P[0, :])
V = np.array([[1, 0.5], [0.5, 1]])

print(in_constraints(P, V, n, m))
in_eq = lambda P: in_constraints(P, V, n, m)
