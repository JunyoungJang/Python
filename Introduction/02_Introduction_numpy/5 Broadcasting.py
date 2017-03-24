# http://cs231n.github.io/python-numpy-tutorial/#python-basic
#
# Broadcasting
# Broadcasting is a powerful mechanism
# that allows numpy to work with arrays of different shapes when performing arithmetic operations.
# Frequently we have a smaller array and a larger array,
# and we want to use the smaller array multiple times to perform some operation on the larger array.

import numpy as np

# For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print y

# This works; however when the matrix x is very large, computing an explicit loop in Python could be slow.
# Note that adding the vector v to each row of the matrix x
# is equivalent to forming a matrix vv by stacking multiple copies of v vertically,
# then performing elementwise summation of x and vv. We could implement this approach like this:

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print vv                 # Prints "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print y  # Prints "[[ 2  2  4
         #          [ 5  5  7]
         #          [ 8  8 10]
         #          [11 11 13]]"

# Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v.
# Consider this version, using broadcasting:

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print y  # Prints "[[ 2  2  4]
         #          [ 5  5  7]
         #          [ 8  8 10]
         #          [11 11 13]]"

# The line y = x + v works even though x has shape (4, 3) and v has shape (3,) due to broadcasting;
# this line works as if v actually had shape (4, 3),
# where each row was a copy of v, and the sum was performed elementwise.

# Broadcasting two arrays together follows these rules:
# 1. If the arrays do not have the same rank,
# prepend the shape of the lower rank array with 1s until both shapes have the same length.
# 2. The two arrays are said to be compatible in a dimension if they have the same size in the dimension,
# or if one of the arrays has size 1 in that dimension.
# 3. The arrays can be broadcast together if they are compatible in all dimensions.
# 4. After broadcasting, each array behaves
# as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
# 5. In any dimension where one array had size 1 and the other array had size greater than 1,
# the first array behaves as if it were copied along that dimension

# Functions that support broadcasting are known as universal functions.
# You can find the list of all universal functions in the documentation.

# Here are some applications of broadcasting:

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print np.reshape(v, (3, 1)) * w

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print x + v

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print (x.T + w).T
# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print x + np.reshape(w, (2, 1))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print x * 2

# Broadcasting typically makes your code more concise and faster, so you should strive to use it where possible.