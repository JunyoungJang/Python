# Exercise:
# Construct a function is_prime that check whether the given integer is prime.

import numpy as np

def is_prime(m):
    for k in range(2,int(np.sqrt(m))+1):
        if m % k == 0:
            return False
    return True

for k in range(2,10):
    print "Is %d a prime number? " %k, is_prime(k)