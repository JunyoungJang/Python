import numpy as np
import random
import matplotlib.pyplot as plt

N_STEPS = 1000 # number of steps in simulation

# define UP and DOWN steps
UP = 1
DOWN = -1
moves = [UP, DOWN]

x = [0]
t = [0]
for step in range(N_STEPS):
    dx = random.choice(moves)
    x.append(x[-1]+dx)
    t.append(step+1)

plt.plot(t, x, 'b')
plt.axis([0, N_STEPS, -2*np.sqrt(N_STEPS), 2*np.sqrt(N_STEPS)])
plt.show()

# Exercise
# Generate a 1D simple random walk using random.randint.
