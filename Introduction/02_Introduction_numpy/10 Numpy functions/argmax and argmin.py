import numpy as np

A = np.array([[31, 23,  4, 24, 27, 34],
              [18,  3, 25,  0,  6, 35],
              [28, 14, 33, 22, 20,  8],
              [13, 30, 21, 19,  7,  9],
              [16,  1, 26, 32,  2, 29],
              [17, 12,  5, 11, 10, 15]])

print np.argmax(A, 0) # [0 3 2 4 0 1] # Find max location by running over the 0-th index, and report in a row form
print np.argmin(A, 0) # [3 4 0 1 4 2] # Find min location by running over the 0-th index, and report in a row form
print np.argmax(A, 1) # [5 5 2 1 3 0] # Find max location by running over the 1-st index, and report in a row form
print np.argmin(A, 1) # [2 3 5 4 1 2] # Find min location by running over the 1-st index, and report in a row form
