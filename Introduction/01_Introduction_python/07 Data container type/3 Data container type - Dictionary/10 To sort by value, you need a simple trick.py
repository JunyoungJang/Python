dict = {'c': 10, 'b': 1, 'a': 22, 'd': 10}
temp = []
for k, v in  dict.items(): # [('a', 22), ('c', 10), ('b', 1), ('d', 10)]
    temp.append( (v, k) ) # switch the order
    print temp
    temp.sort(reverse=True)
    print temp


