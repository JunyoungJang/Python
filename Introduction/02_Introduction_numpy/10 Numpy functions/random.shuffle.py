import numpy as np

a = np.arange(10)
print a

np.random.shuffle(a)
print a

A = np.array([[0, 1],
              [1, 2],
              [2, 3],
              [3, 4],
              [4, 5],
              [5, 6],
              [6, 7],
              [7, 8],
              [8, 9],
              [9, 10]])
B = A[a,:]
print B

