import numpy as np

a = np.array([0, 10, 20, 30])
print a[[3,1,2]]    # returns [30 10 20]

a = np.array([[5, 6], [7, 8]])
print a[[1]]    # returns [[7 8]]

a = np.array([[1,2], [3, 4], [5, 6]])
print a[[0, 1, 2], [0, 1, 0]]  # Prints "[1 4 5]"
print a[[0, 0], [1, 1]]        # Prints "[2 2]"



