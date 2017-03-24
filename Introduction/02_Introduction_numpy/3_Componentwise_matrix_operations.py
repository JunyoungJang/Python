import numpy as np

a = np.array([[1,2], [3,4]], dtype=np.float64)
b = np.array([[5,6], [7,8]], dtype=np.float64)

# componentwise matrix operations
print np.sqrt(a)
print a + 1   # [[2 3] [4 5]]
print 2 * a   # [[2 4] [6 8]]
print a + b
print a - b
print a * b
print a / b





