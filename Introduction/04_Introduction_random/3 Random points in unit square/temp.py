import numpy as np
import matplotlib.pyplot as plt

N_POINTS = 100

sample_original = np.random.uniform(-1, 1, [2,N_POINTS])
sample_inside_unit_circle_x = []
sample_inside_unit_circle_y = []
for i in range(N_POINTS):
    if sample_original[0,i]**2 + sample_original[1,i]**2 <1:
        sample_inside_unit_circle_x.append(sample_original[0, i])
        sample_inside_unit_circle_y.append(sample_original[1, i])

plt.axis([-1,1,-1,1])
boundary_x = np.arange(-1, 1, 0.01)
boundary_y = np.sqrt(1 - boundary_x**2)
plt.plot(boundary_x, boundary_y, 'r')
plt.plot(boundary_x, - boundary_y, 'r')
plt.plot(sample_inside_unit_circle_x, sample_inside_unit_circle_y, 'bo')
plt.show()