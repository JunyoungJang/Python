import numpy as np
import random
import matplotlib.pyplot as plt

N_STEPS = 100 # number of steps in simulation
PROB_UP = 0.6

# define UP and DOWN steps
UP = 1
DOWN = -1

x = [0]
t = [0]
for step in range(N_STEPS):
    if random.uniform(0,1) >  1-PROB_UP:
        dx = UP
    else:
        dx = DOWN
    x.append(x[-1]+dx)
    t.append(step+1)

plt.plot(t, x, 'b')
plt.show()
