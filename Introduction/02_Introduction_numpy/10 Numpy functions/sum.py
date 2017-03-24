import numpy as np

a = np.array([[1, 2], [3, 4]])

print np.sum(a)     # 10
print np.sum(a, 0)  # [4 6]
print np.sum(a, 1)  # [3 7]
