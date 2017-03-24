import numpy as np
import matplotlib.pyplot as plt

# https://plot.ly/matplotlib/histograms/

gaussian_numbers = np.random.randn(1000)

plt.hist(gaussian_numbers)

plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.show() # You must call plt.show() to make graphics appear.