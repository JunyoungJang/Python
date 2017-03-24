# List concatenation using +
x = [2, 'End'] + [2, 4]
print x                     # [2, 'End', 2, 4]
print len(x)                # 8

# List concatenation using *
x = [2, 'End'] * 3
print x                     # [2, 'End', 2, 'End', 2, 'End']
print len(x)  # 6

# List concatenation, not vector addition
print [1, 2, 3] + [4, 5, 6]
