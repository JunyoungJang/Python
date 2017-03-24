dict = {'c': 10, 'b': 1, 'a': 22, 'd': 10}
a = dict.items()         # [('a', 22), ('c', 10), ('b', 1), ('d', 10)]
a.sort()                 # [('a', 22), ('b', 1), ('c', 10), ('d', 10)]
a = sorted(dict.items()) # [('a', 22), ('b', 1), ('c', 10), ('d', 10)]

for key, value in  sorted(dict.items()):
    print key, value

