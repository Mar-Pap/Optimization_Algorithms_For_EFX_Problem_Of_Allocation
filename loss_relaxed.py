import numpy as np
import math as m
from scipy.optimize import minimize


def f_rel(P, V, m, n, lam):
    A = np.zeros((n, n))
    B = np.zeros((n, n, m))
    P = P.reshape(n, m)
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, m):
                A[i, j] += V[i, k] * P[j, k]
                B[i, j, k] = V[i, k] * P[j, k]

    sum1 = 0.0
    sum2 = 0.0
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                for l in range(0, m):
                    sum1 += max(0, A[i, j] - B[i, j, l] - A[i, i])
    for k in range(0, m):
        for i in range(0, n):
            sum2 += P[i, k] * np.log(P[i, k] + np.spacing(1))

    return sum1 - lam * sum2


def eq_constraints(P, n, m):
    eq_con = np.zeros((m, 1)).reshape(-1)
    # count = 0
    P1 = P.reshape(n, m)
    for k in range(0, m):
        sum = 0.0
        for i in range(0, n):
            # print(P1[i, k])
            sum += P1[i, k]
        sum = sum - 1
        eq_con[k] = sum
    # count += 1
    # print(eq_con)
    return eq_con


V = np.array([[1.0, 0.5], [0.5, 1.0]])
n = len(V[:, 0])
m = len(V[0, :])
p0 = np.random.uniform(0.0, 1.0, (n, m))
sum = 0.0
for k in range(0, n):
    sum += p0[k, :]
for k in range(0, n):
    p0[k, :] = p0[k, :] / sum
print(p0)
p0 = p0.reshape(-1)
print(p0)

bounds = [(0, 1) for k in range(0, n * m)]

lam = 1.0
fun_loss_rel = lambda p: f_rel(p, V, m, n, lam)

eq_cons = {"type": "eq",
           "fun": lambda p: eq_constraints(p, n, m)}

res = minimize(fun_loss_rel, p0, method='SLSQP',
               constraints=eq_cons, options={'ftol': 1e-9, 'disp': True},
               bounds=bounds)
print(res)

p = res.x.reshape(n, m)
print(p)
