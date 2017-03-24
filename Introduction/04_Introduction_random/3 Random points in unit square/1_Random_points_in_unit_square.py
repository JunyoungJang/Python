import numpy as np
import matplotlib.pyplot as plt

N_POINTS = 100

x = np.random.uniform(0, 1, [2,N_POINTS])
print x

plt.plot(x[0,:], x[1,:], 'bo')
plt.axis([0,1,0,1])
plt.show()


