import numpy as np
import math as m


def loss(P, n, m):
    #m = 3
    #n = 2
    #v = np.array((n, m))
    loss = 0.0
    for k in range(0, m):
        for i in range(0, n):
            loss += -p[i, k] * np.log(p[i, k] + np.spacing(1))

    return loss


p = np.array([[1, 0], [0, 1]])
#print(loss(p))
n = len(p[:, 0])
m = len(p[0, :])

fun_loss = lambda p: loss(p, n, m)
#print(p.ravel())
#print(loss(p.ravel()))
#print(loss(p))
#print(len(p[0, :]))
#print(len(p[:, 0]))
print(fun_loss(p))
