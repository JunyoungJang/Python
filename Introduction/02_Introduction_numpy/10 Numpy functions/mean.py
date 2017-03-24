import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1., 2.],
                 [1., 4.]])

a = np.mean(data)
b = np.mean(data, 0)
c = np.mean(data, 1)

print type(a), a.shape, a
print type(b), b.shape, b
print type(c), c.shape, c


