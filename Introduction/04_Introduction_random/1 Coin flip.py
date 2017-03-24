import random

N_STEPS = 10 # number of steps in simulation
PROB_UP = 0.6

coin_flip_results = []
for step in range(N_STEPS):
    if random.uniform(0,1) >  1-PROB_UP:
        coin_flip_results.append('H')
    else:
        coin_flip_results.append('T')

print(coin_flip_results)