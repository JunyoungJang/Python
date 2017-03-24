import numpy as np

# np.arange or range (start_point, end_point_excluded, jump_size)
a1 = np.arange(1, 10, 4)    # both float and int are allowed
a2 = np.arange(1., 10., 4.) # both float and int are allowed

print type(a1), a1.shape
print a1
print type(a2), a2.shape
print a2