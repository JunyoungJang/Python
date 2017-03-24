import numpy as np

a = np.array([[2, 2], [4, 4]])
b = np.array([[2, 2], [2, 2]])
c = a / b
d = np.divide(a, b)

print type(a), a.shape
print a
print type(b), b.shape
print b
print type(c), c.shape
print c
print type(d), d.shape
print d