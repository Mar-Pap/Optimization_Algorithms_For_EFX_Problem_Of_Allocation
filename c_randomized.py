import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def minim(V, p0):
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

    n = len(V[:, 0])
    m = len(V[0, :])

    bounds = [(0, 1) for k in range(0, n * m)]

    fun_loss = lambda p: loss(p, n, m)
    """in_eq = lambda p: in_constraints(p, V, n, m)
    eq_const = lambda p: eq_constraints(p, n, m)"""

    eq_cons = {"type": "eq",
               "fun": lambda p: eq_constraints(p, n, m)}
    in_eq = {"type": "ineq",
             "fun": lambda p: in_constraints(p, V, n, m)}

    res = minimize(fun_loss, p0, method='SLSQP',
                   constraints=[eq_cons, in_eq], options={'ftol': 1e-9, 'disp': False},
                   bounds=bounds)
    # print(res)

    p = res.x.reshape(n, m)
    # print(p)

    inn = in_constraints(p, V, n, m).reshape(n, n, m)
    eqq = eq_constraints(p, n, m)
    # print(inn)
    # print(eqq)
    # print(p0)
    return loss(p, n, m), inn, eqq, p0.reshape(n, m), p


V = np.random.rand(10, 10)
print(V)
n = len(V[:, 0])
m = len(V[0, :])
p0 = np.random.uniform(0.0, 1.0, (n, m))
NoI = 1000
for t in range(0, NoI):
    print(t)
    sum = 0.0
    for i in range(0, n):
        sum += p0[i, :]
    for j in range(0, n):
        p0[j, :] = p0[j, :] / sum
    # print("p0")
    # print(p0)
    p0 = p0.reshape(-1)
    loss, in1, eq1, p0, p = minim(V, p0)
    print(loss)
    # print("p")
    # print(p)
    if loss < 10 ** (-6):
        print("loss")
        print(loss)
        print("ineq")
        print(in1)
        print("eq")
        print(eq1)
        print("P")
        print(p)
        break

    p0 = np.random.uniform(0.0, 1.0, (n, m))
