import numpy as np

v = np.array([10, 20, 30, 40])

print np.where(v < 25, v, 0)     # returns [10 20  0  0]
print np.where(v > 25, v/10, v)  # returns [10 20  3  4]
