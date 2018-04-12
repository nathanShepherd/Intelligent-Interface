# Testing various methods to graph with matplotlib
# Developed by Nathan Shepherd

import numpy as np
import matplotlib.pyplot as plt

n = 100
y = [round(np.random.normal(scale=n/10)) for _ in range(n)]
x = [i for  i in range(-n, n)]

_y = []
for i in range(-n, n):
    _y.append(y.count(i))

plt.plot(x, _y)
plt.show()
