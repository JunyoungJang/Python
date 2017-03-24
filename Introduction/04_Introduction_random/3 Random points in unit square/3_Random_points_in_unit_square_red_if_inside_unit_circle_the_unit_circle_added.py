import numpy as np
import matplotlib.pyplot as plt

N_POINTS = 100

x = np.random.uniform(0, 1, [2,N_POINTS])
print x

plt.axis([0,1,0,1])
boundary_x = np.arange(0, 1, 0.01)
boundary_y = np.sqrt(1 - boundary_x**2)
plt.plot(boundary_x, boundary_y, 'r')
for i in range(N_POINTS):
    if x[0,i]**2 + x[1,i]**2 <1:
        plt.plot(x[0,i], x[1,i], 'ro')
    else:
        plt.plot(x[0, i], x[1, i], 'bo')
plt.show()