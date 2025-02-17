import numpy as np
from matplotlib import pyplot as plt, animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math as ma

plt.rcParams["figure.figsize"] = [5.00, 10.0]
plt.rcParams["pcolor.shading"] = 'auto'


def create_animation(t, x1, x2, tensor0, tensor1, tensor2, filename='output.gif'):
    n = len(t)
    fig, ax = plt.subplots(3)
    ax[0].set_title('x1 motion')
    ax[1].set_title('x2 motion')
    ax[2].set_title('x3 motion')
    pcolor_ax0 = ax[0].pcolor(x1, x2, tensor0[0, :, :], vmin=tensor0.min(), vmax=tensor0.max())
    pcolor_ax1 = ax[1].pcolor(x1, x2, tensor1[0, :, :], vmin=tensor1.min(), vmax=tensor1.max())
    pcolor_ax2 = ax[2].pcolor(x1, x2, tensor2[0, :, :], vmin=tensor2.min(), vmax=tensor2.max())
    div0 = make_axes_locatable(ax[0])
    div1 = make_axes_locatable(ax[1])
    div2 = make_axes_locatable(ax[2])
    cax0 = div0.append_axes('right', size='5%', pad=0.2)
    cax1 = div1.append_axes('right', size='5%', pad=0.2)
    cax2 = div2.append_axes('right', size='5%', pad=0.2)

    def update(i):
        pcolor_ax0.set_array(tensor0[i, :, :].flatten())
        pcolor_ax1.set_array(tensor1[i, :, :].flatten())
        pcolor_ax2.set_array(tensor2[i, :, :].flatten())

    fig.colorbar(pcolor_ax0, cax=cax0)
    fig.colorbar(pcolor_ax1, cax=cax1)
    fig.colorbar(pcolor_ax2, cax=cax2)
    anim = animation.FuncAnimation(fig, update, interval=1000 * t[-1] / len(t), frames=n)
    anim.save(filename)


def g(x, r):
    return x / r


def uhat_prime_p(t, x1, x2, r):
    a = 5600  # m/s
    return (np.sin(x1) + np.sin(x2)) * np.exp(-((t - r / a) ** 2) + 5 * (t - r / a) - 4) * np.sin(4 * ((t - r / a) ** 3))  # u_bar' p wave


def uhat_prime_s(t, x1, x2, r):
    b = 3200  # m/s
    return (np.sin(x1) + np.sin(x2)) * np.exp(-((t - r / b) ** 2) + 5 * (t - r / b) - 4) * np.sin(4 * ((t - r / b) ** 3))  # u_bar' s wave


t = np.linspace(0, 10, 300)  # t in [0,12] sec
x1 = np.linspace(-1000, 1000, 64)  # x1 in [-1km, 1km]
x2 = np.linspace(-5000, 5000, 64)  # x2 in [-0.5km, 0.5km]
#x3 = 10000.0  # x3 = 10 km
x3 = 15000.0  #x3 = 15km


u1 = np.zeros((len(t), len(x1), len(x2)))  # (300, 64, 64)
u2 = np.zeros((t.shape[-1], x1.shape[-1], x2.shape[-1]))
u3 = np.zeros((t.shape[-1], x1.shape[-1], x2.shape[-1]))

m = 1000 * (3200**2)
A = 2000
a = 5600
b = 3200

for i, _t in enumerate(t):
    for j, _x1 in enumerate(x1):
        for k, _x2 in enumerate(x2):
            r = np.sqrt(_x1 ** 2 + _x2 ** 2 + x3 ** 2)
            g1, g2, g3 = g(_x1, r), g(_x2, r), g(x3, r)
            up = uhat_prime_p(_t, _x1, _x2, r)
            us = uhat_prime_s(_t, _x1, _x2, r)
            u1[i, j, k] = (-m*A/2*ma.pi*r)*(((g1**2)*g3*up/a**3) + (g3/(b**3))*us)
            u2[i, j, k] = (-m*A/2*ma.pi*r)*(g1 * g2 * g3 * up / a ** 3)
            u3[i, j, k] = (-m*A/2*ma.pi*r)*(((g3**2)*g1*up/a**3) + ((g1/b**3)*us))

create_animation(t, x1, x2, np.abs(u1), np.abs(u2), np.abs(u3))

plt.figure()
plt.plot(t, u1[:, 31, 31])
plt.show()
