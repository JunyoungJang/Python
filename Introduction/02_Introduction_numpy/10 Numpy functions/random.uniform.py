import numpy as np
import matplotlib.pyplot as plt

LOWER_BOUND = -1.
UPPER_BOUND = 1.
NUMBER_OF_SAMPLES = 10000

x = np.random.uniform(LOWER_BOUND, UPPER_BOUND, NUMBER_OF_SAMPLES)
print type(x), x.shape

plt.hist(x)
plt.show()