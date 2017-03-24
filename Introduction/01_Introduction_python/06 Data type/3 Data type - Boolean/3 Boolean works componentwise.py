import numpy as np

print np.array([3, 7]) < 5       # [ True  False ]
print np.array([3, 7]) != 5      # [ True  True  ]
print np.array([3, 7]) == 5      # [ False False ]
print np.array([3, 7]) >= 5      # [ False True  ]
# print 1 < np.array([3, 7]) < 5 # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
