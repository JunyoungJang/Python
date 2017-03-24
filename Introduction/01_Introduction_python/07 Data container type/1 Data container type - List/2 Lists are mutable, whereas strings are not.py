# Lists are mutable, whereas strings are not
fruit = 'Banana'
# fruit[0] = 'b' # Error - strings are not mutable
x = [2, 3]
print x.append('BAR')       # [2, 3, 'BAR']
print x.append([1, 2])      # [2, 3, 'BAR', [1, 2]]
print x.insert(2, 5)        # [2, 3, 5, 'BAR', [1, 2]]
print x.insert(2, 'Wow')    # [2, 3, 'Wow', 5, 'BAR', [1, 2]]
print x.remove('Wow')       # [2, 3, 5, 'BAR', [1, 2]]
x[-1] = 'End'
print x                     # [2, 3, 5, 'BAR', 'End']
del x[1]
print x                     # [2, 5, 'BAR', 'End']
del x[1:3]
print x                     # [2, 'End']
