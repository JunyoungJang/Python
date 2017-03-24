# http://cs231n.github.io/python-numpy-tutorial/#python-basic
#
# Data types
# Every numpy array is a grid of elements of the same type.
# Numpy provides a large set of numeric datatypes that you can use to construct arrays.
# Numpy tries to guess a datatype when you create an array,
# but functions that construct arrays usually also include an optional argument to explicitly specify the datatype.
# Here is an example:

import numpy as np

# how to check data type using dtype method
x = np.array([1, 2])  # Let numpy choose the datatype
print x.dtype         # Prints "int64"
x = np.array([1.0, 2.0])  # Let numpy choose the datatype
print x.dtype             # Prints "float64"
x = np.array([1, 2], dtype=np.int64)  # Force a particular datatype
print x.dtype                         # Prints "int64"


# One can specify the data type when create an array from a list using np.array
a = np.array([1, 2, 3], dtype='f') # f means float32
print a # [ 1.  2.  3.]

# One can change the data type later using the astype method
# ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
# Copy of the array, cast to a specified type.
a = np.array([1, 2, 3, 4], dtype='f')
print a
b = a.astype(int)
print b
b[0] = 0
print a
print b # Copy of the array, cast to a specified type. No aliasing!