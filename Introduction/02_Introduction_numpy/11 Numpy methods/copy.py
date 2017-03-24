import numpy as np

X = np.array([[2., 2.], [-3., 3.]])

# Assigning creates alias whereas copy method doesn't
a = np.ones((2,2))
b = a
b[0,0]=9
print 'a\n', a
print 'b\n', b # aliasing - a and b are same

a = np.ones((2,2))
b = a.copy()
b[0,0] = 9
print 'a\n', a
print 'b\n', b # no aliasing - a and b are different
