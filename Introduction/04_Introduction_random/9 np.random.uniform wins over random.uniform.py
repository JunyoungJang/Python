import numpy as np
import random

# both works
print np.random.uniform(0, 1)
print random.uniform(0, 1)

# np.random.uniform wins
print np.random.uniform(0, 1, (2, 2))
# print random.uniform(0, 1, (2, 2)) # Error