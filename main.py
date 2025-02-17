import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.stats as st
import time
from statistics import mean, median

time_start = time.time()


def minim(V, sd):
    def loss(P, n, m):
        # m = 3
        # n = 2
        # v = np.array((n, m))

        P = P.reshape(n, m)
        loss = 0.0
        for k in range(0, m):
            for i in range(0, n):
                loss += P[i, k] * np.log(P[i, k] + np.spacing(1))
        # print(loss)
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

        ineq_const = np.zeros((n*(n-1), m)).reshape(-1)
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
    """p0 = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            p0[i, j] = 1 / n
    p0 = p0.reshape(-1)"""
    np.random.seed(sd)
    p0 = np.random.uniform(0.0, 1.0, (n, m))
    sum = 0.0
    for i in range(0, n):
        sum += p0[i, :]
    for j in range(0, n):
        p0[j, :] = p0[j, :] / sum
    # print(p0)
    p0 = p0.reshape(-1)
    # print(p0)

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

    inn = in_constraints(p, V, n, m).reshape(n*(n-1), m)
    eqq = eq_constraints(p, n, m)
    # print(inn)
    # print(eqq)
    return loss(p, n, m), inn, eqq, p0.reshape(n, m), p


# noI = 1000
# noI = 100
noI = 10000
sol = []
innI = []
eqqI = []
p00 = []
pf = []
tf = []

# V = np.array([[1, 0.5], [0.5, 1]])
# V = np.array([[1/3, 1/3, 1/3, 1], [0, 0, 0, 3/4]])
# V = np.array([[0, 1], [0, 1]])
# V = np.array([[0, 0, 1/6, 1/6, 1/6, 3/6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]])
# V = np.array([[0, 0, 1], [2/3, 1/3, 0]])
# V = np.array([[0, 0, 0, 2/5, 3/5], [0, 0, 1/5, 1/5, 3/5], [0, 1/5, 1/5, 1/5, 2/5]])
# V = np.identity(5)
"""V = np.array([[0.96, 0.6311, 0.5407, 0.6363, 0.403, 0.4101, 0.9574, 0.4378, 0.7335, 0.9533],
              [0.7716, 0.6521, 0.7012, 0.1188, 0.2779, 0.9135, 0.7196, 0.1189, 0.9724, 0.0027],
              [0.0286, 0.2747, 0.9317, 0.9962, 0.9533, 0.7821, 0.9085, 0.826, 0.9136, 0.6942],
              [0.79951, 0.4838, 0.1631, 0.7849, 0.8288, 0.856, 0.3713, 0.5687, 0.014, 0.6433],
              [0.6642, 0.1323, 0.8254, 0.3661, 0.0963, 0.5012, 0.5907, 0.2564, 0.2018, 0.9498],
              [0.0358, 0.3333, 0.7596, 0.9821, 0.6902, 0.6554, 0.2699, 0.6963, 0.1593, 0.6796]])"""
V = np.array([[0.02539563, 0.23480337, 0.23623651, 0.84609698, 0.16438599, 0.29822631,
               0.52188924, 0.22874975, 0.49613698, 0.81818469, 0.43963526, 0.96693122,
               0.65798978, 0.95482666],
             [0.38050185, 0.45294776, 0.50695176, 0.2757392,  0.63973568, 0.17729309,
              0.15132491, 0.00309858, 0.11076458, 0.34749615, 0.16820167, 0.81165702,
              0.66584811, 0.52732649],
             [0.926263,   0.16774879, 0.119524, 0.03355165, 0.87812824, 0.65945228,
              0.67399461, 0.53553664, 0.65507648, 0.27739563, 0.06194025, 0.22020435,
              0.52896231, 0.51816256],
             [0.0918557,  0.19504637, 0.63473278, 0.79323012, 0.17676797, 0.81975624,
              0.77482914, 0.32616388, 0.02373151, 0.06932127, 0.70389327, 0.47523085,
              0.39806469, 0.42214993],
             [0.7830248,  0.06072257, 0.03825564, 0.68097494, 0.5645792,  0.06504697,
              0.23211841, 0.83013698, 0.2491467,  0.99433777, 0.10489823, 0.05805685,
              0.26376639, 0.69104187],
             [0.72512124, 0.12504598, 0.45723854, 0.80800392, 0.64192493, 0.57832936,
              0.31489336, 0.96909675, 0.44407866, 0.44755308, 0.88335566, 0.95550141,
              0.70647953, 0.67751787],
             [0.83271017, 0.31622198, 0.33690972, 0.32646571, 0.64396726, 0.3926369,
              0.60657796, 0.45708094, 0.47812867, 0.74582713, 0.76961036, 0.28964206,
              0.95411415, 0.21159689],
             [0.14411227, 0.05214234, 0.51415837, 0.28453341, 0.81102224, 0.00575681,
              0.63087184, 0.79592232, 0.82494976, 0.93316579, 0.33939252, 0.10570377,
              0.92547512, 0.80879677],
             [0.3797373,  0.08704269, 0.96026939, 0.49888086, 0.18705937, 0.39861422,
              0.87768921, 0.09336038, 0.71214083, 0.52284759, 0.64497298, 0.4980616,
              0.82468617, 0.63387298],
             [0.64755683, 0.35413204, 0.27810059, 0.18588338, 0.40739468, 0.9234397,
              0.04999301, 0.87429026, 0.20150569, 0.67508268, 0.61222252, 0.87258464,
              0.21920578, 0.28186533]])
ww = 0
for i in range(noI):
    sol1, innI1, eqqI1, p01, pf1 = minim(V, i)
    print("i", i)
    # print(sol1)
    sol.append(sol1)
    innI.append(innI1)
    eqqI.append(eqqI1)
    p00.append(p01)
    pf.append(pf1)
    end = time.time()
    te = end - time_start
    tf.append(te)
    # print("inn", innI1)
    for i in innI1:
        for j in i:
            if j<-1:
                print("not okay")
                ww += 1
    # print("p", pf1)

    """if sol1 > 0.1:
        print(p01, pf1, sol1)"""

mt = mean(tf)
print("mean time", mt, "median time", median(tf), "max time", max(tf), "min time", min(tf))
print("mean loss", mean(sol), "median loss", median(sol), "max loss", max(sol), "min loss", min(sol))
print("ww", ww)
plt.hist(sol, bins=50, density=False)
plt.xlabel('Loss Function Values')
plt.ylabel('Counts')
plt.title('Histogram of Function Loss')
# plt.xlim(0, 1)
plt.grid(True)
plt.show()

"""plt.hist(noI, bins=50, density=False)
plt.xlabel('Total Repeats(Loops)')
plt.ylabel('Counts')
plt.title('Histogram of Repeats(Loops)')
plt.grid(True)
plt.show()"""
