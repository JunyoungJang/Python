# Can put the functions into a list
import math

def area(r):
    return math.pi * r * r

def circumference(r):
    return 2 * math.pi * r

funcs = [area, circumference]

for f in funcs:
    print f(1.0)

# Can put the functions into a function
def call_it(func, value):
    return func(value)

print call_it(area, 1.0)
print call_it(circumference, 1.0)

