import numpy as np
from scipy.optimize import minimize


def exp_loss(P, n, m, p0):
    T = 10
    P = P.reshape(n, m)
    p00 = p0.reshape(n, m)
    H = np.array((m, 1)).reshape(m)
    M = []
    ps = np.zeros((n, m))
    for k in range(0, m):
        for i in range(0, n):
            H[k] += -P[i, k] * np.log(P[i, k] + np.spacing(1))
        if H[k] != 0:
            # print(H[k])
            M.append(k)
    # print(loss)
    for t in range(0, T):
        L = 0.0
        L1 = 0.0
        L2 = 0.0
        for k in range(0, m):
            if k in M:
                for i in range(0, n):
                    L1 += P[i, k] * np.log((P[i, k] / p00[i, k]) + np.spacing(1))
            else:
                L2 += H[k]
        L = L1 + L2

        for k in range(0, m):
            if k in M:
                ps[:, k] = np.random.uniform(0.0, 1.0, m)
            else:
                ps[:, k] = p00[:, k]
        # print(ps)
        sum = 0.0
        for i in range(0, n):
            sum += ps[i, :]
        for j in range(0, n):
            ps[j, :] = ps[j, :] / sum
        # print(ps)
        ps = ps.reshape(-1)

        p = minL(n, m, ps, V)

        M = []
        for k in range(0, m):
            for i in range(0, n):
                H[k] += -p[i, k] * np.log(p[i, k] + np.spacing(1))
            if H[k] != 0:
                # print(H[k])
                M.append(k)
                print(k)
        if M:
            break
        return L, ps


def minL(m, n, ps, V):
    loss = lambda p: exp_loss(p, n, m, p0)
    bounds = [(0, 1) for k in range(0, n * m)]
    eq_cons = {"type": "eq",
               "fun": lambda p: eq_constraints(p, n, m)}
    in_eq = {"type": "ineq",
             "fun": lambda p: in_constraints(p, V, n, m)}

    res = minimize(loss, ps, method='SLSQP',
                   constraints=[eq_cons, in_eq], options={'ftol': 1e-9, 'disp': False},
                   bounds=bounds)
    # print(res)

    p = res.x.reshape(n, m)
    # print(p)

    inn = in_constraints(p, V, n, m).reshape(n, n, m)
    eqq = eq_constraints(p, n, m)
    # print(inn)
    # print(eqq)
    return p, res


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


V = np.array([[0, 1], [0, 1]])
n = len(V[:, 0])
m = len(V[0, :])
p0 = np.random.uniform(0.0, 1.0, (n, m))
# print(p0)
p0 = p0.reshape(-1)
lf, pf = minL(m, n, p0, V)
print(lf, pf)
