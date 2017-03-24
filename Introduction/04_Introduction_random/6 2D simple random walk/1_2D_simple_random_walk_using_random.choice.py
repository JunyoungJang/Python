import numpy as np
import random
import matplotlib.pyplot as plt

N_STEPS = 1000 # number of steps in simulation

# define NEWS steps
N = [ 0,  1]
E = [ 1,  0]
W = [-1,  0]
S = [ 0, -1]
moves = [N, E, W, S]

x = [0]
y = [0]
for step in range(N_STEPS):
    dx, dy = random.choice(moves)
    x.append(x[-1]+dx)
    y.append(y[-1]+dy)

plt.plot(x, y, 'b')
plt.axis([-1.5*np.sqrt(N_STEPS), 1.5*np.sqrt(N_STEPS), -1.5*np.sqrt(N_STEPS), 1.5*np.sqrt(N_STEPS)])
plt.show()
