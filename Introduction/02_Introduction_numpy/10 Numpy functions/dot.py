import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

a = np.dot(x, y)

print type(x), x.shape
print x
print type(y), y.shape
print y
print type(a), a.shape
print a