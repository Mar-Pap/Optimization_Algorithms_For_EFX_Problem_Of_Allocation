import numpy as np
from scipy.optimize import minimize


def loss(P, n, m):
    P = P.reshape(n, m)
    loss = 0.0
    H = []
    for k in range(0, m):
        h = 0.0
        for i in range(0, n):
            loss += -P[i, k] * np.log(P[i, k] + np.spacing(1))
            h += -P[i, k] * np.log(P[i, k] + np.spacing(1))
        H.append(h)
    # print(loss)
    return H, loss


def DKL(P, n, m, p0):
    P = P.reshape(n, m)
    dkl = 0.0
    for k in range(0, m):
        for i in range(0, n):
            dkl += P[i, k] * np.log((P[i, k] / p0[i, k]) + np.spacing(1))
    # print dkl
    return dkl


def L(P, p0, n, m, M):
    P = P.reshape(n, m)
    p0 = p0.reshape(n, m)
    l = 0.0
    for i in range(0, n):
        for k in range(0, m):
            if k not in M:
                l += -P[i, k] * np.log(P[i, k] + np.spacing(1))
            else:
                l += -1000 * P[i, k] * np.log((P[i, k] + np.spacing(1)) / (p0[i, k] + np.spacing(1)))
    return l


def in_constraints(P, V, n, m):
    A = np.zeros((n, n))
    B = np.zeros((n, n, m))
    P = P.reshape(n, m)
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, m):
                A[i, j] += V[i, k] * P[j, k]
                B[i, j, k] = V[i, k] * P[j, k]

    ineq_const = np.zeros((n, n, m)).reshape(-1)
    count = 0
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                for k in range(0, m):
                    ineq_const[count] = A[i, i] - A[i, j] + B[i, j, k]
            count += 1
    # print(ineq_const)
    return ineq_const


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


# V = np.array([[1, 0.5], [0.5, 1]])
# V = np.array([[0, 1], [0, 1]])
# V = np.array([[0, 0, 1], [2 / 3, 1 / 3, 0]])
# V = np.array([[1/3, 1/3, 1/3, 1], [0, 0, 0, 3/4]])
# V = np.array([[0, 0, 1/6, 1/6, 1/6, 3/6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]])
# V = np.array([[0, 0, 0, 2/5, 3/5], [0, 0, 1/5, 1/5, 3/5], [0, 1/5, 1/5, 1/5, 2/5]])
# V = np.identity(4)
V = np.identity(10)
n = len(V[:, 0])
m = len(V[0, :])

p0 = np.random.uniform(0.0, 1.0, (n, m))
sum = 0.0
for i in range(0, n):
    sum += p0[i, :]
for j in range(0, n):
    p0[j, :] = p0[j, :] / sum
# print("p0")
# print(p0)

H, L1 = loss(p0, n, m)
# print(H)

M = []
for i in range(0, len(H)):
    # print(H[i])
    if H[i] > 10 ** (-6):
        M.append(i)
# print("M")
# print(M)

bounds = [(0, 1) for k in range(0, n * m)]
eq_cons = {"type": "eq",
           "fun": lambda p: eq_constraints(p, n, m)}
in_eq = {"type": "ineq",
         "fun": lambda p: in_constraints(p, V, n, m)}
"""
T = 1000
for t in range(0, T):
    l = lambda p: L(p, p0, n, m, M)
    print(t)
    # print(p0)
    ps = np.zeros((n, m))

    for k in range(0, m):
        if k in M:
            ps[:, k] = np.random.uniform(0.0, 1.0, n)
        else:
            ps[:, k] = p0[:, k]
    # print(ps, 108)
    # print(ps)
    sum = 0.0
    for i in range(0, n):
        sum += ps[i, :]
    for j in range(0, n):
        ps[j, :] = ps[j, :] / sum
    # print("ps")
    # print(ps)
    # print(sum)

    ps = ps.reshape(-1)
    # print(p0)
    res = minimize(l, ps, method='SLSQP',
                   constraints=[eq_cons, in_eq], options={'ftol': 1e-9, 'disp': False},
                   bounds=bounds)
    #print(res)
    p1 = res.x.reshape(n, m)
    #print('p1')
    #print(p1)
    h, L1 = loss(p1, n, m)
    # print(L1)
    M = []
    for i in range(0, len(h)):
        # print(h[i])
        if h[i] > 10 ** (-6):
            M.append(i)
    #print("M")
    #print(M)
    if len(M) == 0:
        # print(M)
        break
    # ps = ps.reshape(n, m)
    # print(ps)
    p0 = p1
   print(p0) 
"""
P = np.array([[1, 0, 0, 0, 0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
ineq = in_constraints(P, V, n, m)
eqs = eq_constraints(P, n, m)
print(ineq)
print(eqs)
