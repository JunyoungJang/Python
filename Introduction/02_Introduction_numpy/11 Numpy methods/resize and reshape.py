import numpy as np

# resize - New shape may have a different size as old. Padding or cutting are needed at the end.
a = np.array([[1,2], [3,4]])
a.resize(2, 3)
print a
a.resize(2, 2)
print a
a.resize(2, 1)
print a

print('============================')
b = np.array([[1,2], [3,4]], order='F')
b.resize(2, 3)
print b
b.resize(2, 2)
print b
b.resize(2, 1)
print b

# reshape - New shape must have the same size as old.
print('============================')
c = np.arange(6).reshape(3, 2)
print c
d = np.array([1,2,3,4,5,6]).reshape(2, 3)
print d
