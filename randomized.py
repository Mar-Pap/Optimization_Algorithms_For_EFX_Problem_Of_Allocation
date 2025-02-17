import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import time

from scipy.stats import entropy
from statistics import mean, median

time_start = time.time()


def his(V, sd):
    def minim(V, p0):
        def loss(P, n, m):

            P = P.reshape(n, m)
            loss = 0.0
            for k in range(0, m):
                for i in range(0, n):
                    loss += P[i, k] * np.log(P[i, k] + np.spacing(1))
            # print(loss)
            # changes were done here
            loss = -loss
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

            # ineq_const = np.zeros((n, n, m)).reshape(-1)
            ineq_const = np.zeros((n * (n - 1), m)).reshape(-1)
            # print(ineq_const.shape)
            count = 0
            for i in range(0, n):
                for j in range(0, n):
                    if i != j:
                        for k in range(0, m):
                            ineq_const[count] = A[i, i] - A[i, j] + B[i, j, k]
                            # print(ineq_const[count])
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

        # inn = in_constraints(p, V, n, m).reshape(n, n, m)
        inn = in_constraints(p, V, n, m).reshape(n * (n - 1), m)
        eqq = eq_constraints(p, n, m)
        # print(inn)
        # print(eqq)
        # print(p0)
        return loss(p, n, m), inn, eqq, p0.reshape(n, m), p

    # np.random.seed(sd)
    n = len(V[:, 0])
    m = len(V[0, :])
    p0 = np.random.uniform(0.0, 1.0, (n, m))
    # print("po", p0)
    NoI = 1000
    # NoI = 100000
    # changes were done here
    for t in range(0, NoI):
        # print(t)
        # print("p0", p0)
        for i in range(0, m):
            # print(i)
            sum = 0.0
            for j in range(0, n):
                # print(p0[j, i])
                sum += p0[j, i]
            # print(sum)
            for k in range(0, n):
                p0[k, i] = p0[k, i] / sum
        # print("p0")
        # print(p0)
        p0 = p0.reshape(-1)
        loss, in1, eq1, p0, p = minim(V, p0)
        # print("p")
        # print(p)
        """print("loss")
        print(loss)
        print("inequality")
        print(in1)
        print("equality")
        print(eq1)"""
        if loss < 10 ** (-6):
            """print("loss")
            print(loss)
            print("ineq")
            print(in1)
            print("eq")
            print(eq1)"""
            break

        p0 = np.random.uniform(0.0, 1.0, (n, m))
    return loss, t, p, in1, eq1


# V = np.array([[1, 0.5], [0.5, 1]])
# V = np.array([[0, 1], [0, 1]])
# V = np.array([[0, 0, 1], [2/3, 1/3, 0]])
# V = np.array([[1/3, 1/3, 1/3, 1], [0, 0, 0, 3/4]])
# V = np.array([[0, 0, 1/6, 1/6, 1/6, 3/6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]])
# V = np.array([[0, 0, 0, 2/5, 3/5], [0, 0, 1/5, 1/5, 3/5], [0, 1/5, 1/5, 1/5, 2/5]])
# V = np.identity(5)
# V = np.concatenate((np.identity(2), 2*np.identity(2)), axis=1)
# print(V)
# n1 = 4
# n1 = 6
n1 = 10
m1 = n1 + 4
V = np.random.uniform(0.0, 1.0, (n1, m1))
"""V = np.array([[0.96, 0.6311, 0.5407, 0.6363, 0.403, 0.4101, 0.9574, 0.4378, 0.7335, 0.9533],
              [0.7716, 0.6521, 0.7012, 0.1188, 0.2779, 0.9135, 0.7196, 0.1189, 0.9724, 0.0027],
              [0.0286, 0.2747, 0.9317, 0.9962, 0.9533, 0.7821, 0.9085, 0.826, 0.9136, 0.6942],
              [0.79951, 0.4838, 0.1631, 0.7849, 0.8288, 0.856, 0.3713, 0.5687, 0.014, 0.6433],
              [0.6642, 0.1323, 0.8254, 0.3661, 0.0963, 0.5012, 0.5907, 0.2564, 0.2018, 0.9498],
              [0.0358, 0.3333, 0.7596, 0.9821, 0.6902, 0.6554, 0.2699, 0.6963, 0.1593, 0.6796]])"""
print(V)
"""V = np.array([[0.84960819, 0.80123707, 0.65202685, 0.34870895, 0.79023708, 0.29429903,
                0.12114043, 0.38035282],
               [0.71568933, 0.9519658, 0.17157561, 0.63181221, 0.11019083, 0.51972663,
                0.46896255, 0.94830787],
               [0.53573443, 0.07087314, 0.89397466, 0.59343052, 0.76023404, 0.15962074,
                0.10642394, 0.7524957],
               [0.07812366, 0.99216496, 0.92715074, 0.49226051, 0.11448245, 0.35245738,
                0.4170662, 0.88391351]])"""
noI = 50
# noI = 1
sol = []
lp = []
tl = []
tf = []
innI = []
eqqI = []
for i in range(0, noI):
    print(i)
    l, t, pl, inn, eqq = his(V, i)
    sol.append(l)
    innI.append(inn)
    eqqI.append(eqq)
    tl.append(t)
    lp.append(pl)
    end = time.time()
    te = end - time_start
    tf.append(te)
    for j in inn:
        for k in j:
            if k < -1:
                print(k)
                print("not ok")
    # print(pl)
    # print(inn)
    # print(time_elapsed)
"""print(sol)
print(lp)
print(innI)
print(eqqI)"""
# print(tl)

# print(lp)
print("max repeats", max(tl), "min repeats", min(tl), "mean repeats", mean(tl), "median repeats", median(tl))

"""end = time.time()
time_elapsed = end - time_start
print("time")
print(str(time_elapsed))"""
mt = mean(tf)
print("time mean", mt, "min time", min(tf), "max time", max(tf), "median time", median(tf))
print("max funct_loss", max(sol), "min loss", min(sol), "mean loss", mean(sol), "median loss", median(sol))
# print("max noi", max(tl), "min noi", min(tl), "mean noi", mean(tl), "median noi", median(tl))

plt.hist(sol, bins=50, density=False)
plt.xlabel('Loss Function Values')
plt.ylabel('Counts')
plt.title('Histogram of Function Loss')
# plt.xlim(0, 1)
plt.grid(True)
plt.show()

plt.hist(tl, bins=50, density=False)
plt.xlabel('Total Repeats(Loops)')
plt.ylabel('Counts')
plt.title('Histogram of Repeats(Loops)')
plt.grid(True)
plt.show()
