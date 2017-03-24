import numpy as np

a = np.array([0, 10, 20, 30])
print a[a < 25]     # returns [ 0 10 20]

a = np.array([[1,2], [3, 4], [5, 6]])
print a[a > 2]      # Prints "[3 4 5 6]"

