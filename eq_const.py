import numpy as np

"""def eq_constraints(P, n, m):
    eq_con = np.zeros((n, m)).reshape(-1)
    count = 0
    for k in range(0, n):
        sum = 0
        for k in range(0, m):
            for i in range(0, n):
                # print(P1[i, k])
                sum += P[i, k]
            sum = sum - 1
            eq_con[count] = sum
        count += 1

    return eq_con


P = np.array([[1, 0], [0, 1]])
# print(loss(p))
n = len(P[:, 0])
m = len(P[0, :])
V = np.array([[1, 0.5], [0.5, 1]])
eq_const = lambda P: eq_constraints(P, n, m)

print(eq_constraints(P, n, m))"""

from scipy.optimize import minimize


def loss(P, n, m):
    # m = 3
    # n = 2
    # v = np.array((n, m))

    P = P.reshape(n, m)
    loss = 0.0
    for k in range(0, m):
        for i in range(0, n):
            loss += -P[i, k] * np.log(P[i, k] + np.spacing(1))
    # print(loss)
    return loss


def eq_constraints(P, n, m):
    eq_con = np.zeros((m, 1)).reshape(-1)
    # count = 0
    P1 = P.reshape(n, m)
    """for k in range(0, n):
        sum = 0"""
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


V = np.array([[1, 0.5], [0.5, 1]])
n = len(V[:, 0])
m = len(V[0, :])
p0 = np.zeros((n, m))
for i in range(0, n):
    for j in range(0, m):
        #p0[i, j] = 1 / n
        if i==j:
            p0[i, j] = 0.8
        else:
            p0[i, j] = 0.2
p0 = p0.reshape(-1)
print(p0)

bounds = [(0, 1) for k in range(0, n * m)]

fun_loss = lambda p: loss(p, n, m)
"""in_eq = lambda p: in_constraints(p, V, n, m)
eq_const = lambda p: eq_constraints(p, n, m)"""

eq_cons = {"type": "eq",
           "fun": lambda p: eq_constraints(p, n, m)}

res = minimize(fun_loss, p0, method='SLSQP',
               constraints=eq_cons, options={'ftol': 1e-9, 'disp': True},
               bounds=bounds)
print(res)
p = res.x.reshape(n, m)
print(p)
