import numpy as np

# If x is a string, len(x) counts characters in x including the space multiple times.
fruit = 'banana'
fruit_1 = 'I eat bananas'
fruit_2 = '     I eat bananas     '
print len(fruit)   # 6
print len(fruit_1) # 13
print len(fruit_2) # 23

# If x is a (column or row) vector, len(x) reports the length of vector x.
a = np.array([[1], [2], [3]])
b = np.array([1, 2, 3])
print len(a)
print len(b)

# If x is a matrix, len(x) reports the number of rows in matrix x.
c = np.array([[1, 2, 3], [1, 2, 3]])
d = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
e = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
print len(a)
print len(b)
print len(c)
