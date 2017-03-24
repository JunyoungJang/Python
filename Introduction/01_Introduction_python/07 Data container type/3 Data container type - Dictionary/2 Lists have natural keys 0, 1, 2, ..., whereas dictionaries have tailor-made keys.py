a = list()
a.append(21)
a.append(183)
a[0] = 23
print a # [23, 183]

b = dict()
b['age'] = 21
b['height'] = 183
b['age'] = 23
print b # {'age': 23, 'height': 183}
