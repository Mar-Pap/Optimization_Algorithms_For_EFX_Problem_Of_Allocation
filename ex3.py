import numpy as np
from scipy.optimize import minimize


def f(x, p):
    sum = 0
    sum1 = 0
    for i in range(1, len(p)):
        sum = sum + p[i] * np.log(p[i] + np.spacing(1))
        sum1 = sum1 + x[i, 0] * p[i]
    # print(sum1-sum)
    return sum1 - sum


def constraints(p):
    sum2 = 0
    for i in range(0, len(p)):
        sum2 = sum2 + p[i]
    sum2 = sum2 - 1
    return sum2


d = int(input('Give me an int: '))
n = int(input('Give me an int: '))
p0 = np.zeros((n, d))
for i in range(0, d):
    p0[i] = 1 / d
# print(p0)
# print(f(p0))
# print(f_der(p0))
# 5f1 = f_der(p0)
b = (0, 1)
bounds = [(0, 1) for i in range(0, d)]
# print(bounds)

eq_cons = {'type': 'eq',
           'fun': constraints}

x = np.zeros((d, 1))
x[0, 0] = -1

fp = lambda p: f(x, p)

res = minimize(fp, p0, method='SLSQP',
               constraints=eq_cons, options={'ftol': 1e-9, 'disp': True},
               bounds=bounds)
print(res)