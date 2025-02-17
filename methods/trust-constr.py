import numpy as np
from scipy.optimize import minimize, Bounds


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


from scipy.optimize import Bounds

bounds = Bounds([0, -0.5], [1.0, 2.0])

from scipy.optimize import LinearConstraint

linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])


def cons_f(x):
    return [x[0] ** 2 + x[1], x[0] ** 2 - x[1]]


def cons_J(x):
    return [[2 * x[0], 1], [2 * x[0], -1]]


def cons_H(x, v):
    return v[0] * np.array([[2, 0], [0, 0]]) + v[1] * np.array([[2, 0], [0, 0]])


from scipy.optimize import NonlinearConstraint

nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H


x0 = np.array([0.5, 0])
res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
               constraints=[linear_constraint, nonlinear_constraint],
               options={'verbose': 1}, bounds=bounds)
print(res.x)