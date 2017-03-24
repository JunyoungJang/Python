import numpy as np
import matplotlib.pyplot as plt

plt.axis([0,50,60,80])
for i in np.arange(1,5):
    z = 68 + 4 * np.random.randn(50) # 50 random points
    zm = np.cumsum(z) / range(1,len(z)+1)
    plt.plot(zm,'b')

n = np.arange(1,51)
su = 68 + 4 / np.sqrt(n)
sl = 68 - 4 / np.sqrt(n)

plt.plot(n,su,'r',n,sl,'r')

plt.show()