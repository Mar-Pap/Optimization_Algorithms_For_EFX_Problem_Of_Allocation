import numpy as np
from matplotlib import pyplot as plt

p = np.arange(0.001, 1., 0.01)
# p = np.array([[1.0, 0.0], [0.0, 1.0]]).reshape(-1)
e = -p * sum(p * np.log2(p))

plt.ylabel("Entropy")
plt.xlabel("P")
plt.plot(p, -(p * np.log(p+np.spacing(1)) + (1 - p) * np.log((1 - p)+np.spacing(1))), label='Entropy')
plt.plot((min(e), min(e)), (0, 0.8), scaley=False, color='r', label='Constraints')
plt.legend()
plt.show()
