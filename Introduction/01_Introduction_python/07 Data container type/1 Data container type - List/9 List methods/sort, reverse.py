# -*- coding: utf8 -*-

# http://cs231n.github.io/python-numpy-tutorial/#python-basic
#
# Containers
# Python includes several built-in container types: lists, dictionaries, sets, and tuples.

import numpy as np

#'''
print 'list methods - append, insert, remove ---'
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
x = x * 3
print x                     # [2, 'End', 2, 'End', 2, 'End']
#'''
'''
'''
#'''
#'''
'''
print 'list methods - count ---'
gases = ['He', 'He', 'Ar', 'Kr']
print gases.count('He') # count 'He' in the list
'''
'''
'''
#'''
#'''
'''
print 'list methods - extend ---'
# compare it with append
a = [1, 2, 3]
a.extend([4, 5])
print a
'''
'''
'''
#'''
#'''
'''
print 'list methods - index ---'
gases = ['He', 'He', 'Ar', 'Kr']
print gases.index('Ar') # index the first 'Ar' in the list
print gases.index('He') # index the first 'He' in the list
print gases
'''
'''
'''
#'''
#'''
'''
print 'list methods - pop ---'
a = [1, 2, 3]
print a.pop(1) # Pop out the pointed element of the list. The default pointer points the last element: pop() = pop(-1).
print a # The pop out element has been removed.
'''
'''
'''
#'''
#'''
'''
print 'list methods - sort, reverse ---'
gases = ['He', 'Ne', 'Ar', 'Kr']
print gases
# gases = gases.sort() # gases = gases.sort() assigns None to gases - A common bug
gases.sort()           # sorted alphebatically and change gases accordingly since lists are mutable
print gases
# gases = gases.reverse() # gases = gases.reverse() assigns None to gases - A common bug
gases.reverse()           # sorted alphebatically in reverse order and change gases accordingly since lists are mutable
print gases
'''