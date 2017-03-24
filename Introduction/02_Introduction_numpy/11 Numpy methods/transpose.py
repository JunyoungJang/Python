import numpy as np

# aliasing
A = np.zeros((2, 3))
B = A.transpose()
print A
print B
B[0, 0] = 1
print A # aliasing
print B # aliasing

# avoid aliasing
A = np.ones((2, 2))
B = A.copy()
B = B.transpose()
print A
print B
B[0, 0] = 1
print A # no aliasing
print B # no aliasing

