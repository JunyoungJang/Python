import numpy as np

a = np.array([1,2])
print a.shape # (2,)

b = np.expand_dims(a, axis=0)
print b.shape # (1, 2)

c = np.expand_dims(a, axis=1)
print c.shape # (2, 1)