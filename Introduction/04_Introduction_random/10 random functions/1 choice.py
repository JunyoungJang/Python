import random

N = [ 0.,  1.]
E = [ 1.,  0.]
W = [-1.,  0.]
S = [ 0., -1.]
moves = [N, E, W, S]

for _ in range(5):
    print random.choice(moves)

for _ in range(5):
    dx, dy = random.choice(moves)
    print dx, dy
