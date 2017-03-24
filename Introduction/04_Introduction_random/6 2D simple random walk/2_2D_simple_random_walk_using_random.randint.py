import numpy as np
import random
import matplotlib.pyplot as plt

N_STEPS = 1000 # number of steps in simulation

# define NEWS steps
N = [ 0,  1]
E = [ 1,  0]
W = [-1,  0]
S = [ 0, -1]

def jump(jump_index):
    if jump_index == 1:
        return N
    elif jump_index == 2:
        return E
    elif jump_index == 3:
        return W
    else:
        return S

x = [0]
y = [0]
for step in range(N_STEPS):
    jump_idx = random.randint(1, 4)
    jump_now = jump(jump_idx)
    x.append(x[-1]+jump_now[0])
    y.append(y[-1]+jump_now[1])

plt.plot(x, y, 'b')
plt.axis([-1.5*np.sqrt(N_STEPS), 1.5*np.sqrt(N_STEPS), -1.5*np.sqrt(N_STEPS), 1.5*np.sqrt(N_STEPS)])
plt.show()

# Exercise
# Change the code in this section so that you do simulation using random.uniform.

# Exercise
# Change the code so that the random walker walks randomly as time goes.
