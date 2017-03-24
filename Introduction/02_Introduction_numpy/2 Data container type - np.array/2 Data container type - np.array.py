# http://cs231n.github.io/python-numpy-tutorial/#python-basic
#
# Arrays
# A numpy array is a grid of values, all of the same type.
# The number of dimensions is the rank of the array;
# the shape of an array is a tuple of integers giving the size of the array along each dimension.

import numpy as np

a, b = 'Python', 'life'
print [a, b] # list
print (a, b) # tuple
print {a: b} # dictionary
print set([a, b]) # set
print np.array([1, 2]) # array

# We can initialize numpy arrays from nested Python lists, and access elements using square brackets:
a = np.array([1, 2, 3])
print type(a)                  # Prints "<type 'numpy.ndarray'>"
print np.linalg.matrix_rank(a) # rank 1 matrix
print a.shape                  # Prints "(3,)"
print a[0], a[1], a[2]         # Prints "1 2 3"
a[0] = 5                       # Change an element of the array
print a                        # Prints "[5, 2, 3]"
b = np.array([[1,2,3], [4,5,6]])   # Create a rank 2 array
print np.linalg.matrix_rank(b)    # rank 2 matrix
print b.shape                     # Prints "(2, 3)"
print b[0, 0], b[0, 1], b[1, 0]   # Prints "1 2 4"

# Numpy also provides many functions to create arrays:
a = np.zeros((2, 2))          # Create an array of all zeros
b = np.ones((1, 2))           # Create an array of all ones
c = np.full((2, 2), 7.)        # Create a constant array
d = np.eye(2)                 # Create a 2x2 identity matrix
e = np.random.random((2, 2))  # Create an array filled with random values
print a
print b
print c
print d
print e

# Faltten arrays using array.rave1
import numpy as np
first = np.zeros((2,2,2))
second = first.ravel()
print second.shape       # returns (8,)