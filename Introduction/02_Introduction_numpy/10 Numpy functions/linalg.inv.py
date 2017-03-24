import numpy as np

A = np.array([[0., 1., 2., 4.], [1., 0., 3., 1.], [4., 5., 6., 7.], [1., 0., 1., 0.]])
B = np.linalg.inv(A)
print 'Matrix A\n', A
print 'Matrix B, inverse of A\n', B











