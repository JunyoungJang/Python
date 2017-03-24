# Exercise:
# Construct a function prime_list_generator that produces the list of prime numbers less than equal to the given integer.

import numpy as np

def is_prime(m):
    for k in range(2,int(np.sqrt(m))+1):
        if m % k == 0:
            return False
    return True

def prime_list_generator(Upper_boubd_integer_inclusive):
    n = Upper_boubd_integer_inclusive
    prime_number_list = []
    for m in range(2,n+1):
        if is_prime(m):
            prime_number_list.append(m)
    return prime_number_list

prime_number_list = prime_list_generator(10)
print prime_number_list
print len(prime_number_list)