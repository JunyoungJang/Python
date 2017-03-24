# append adds its argument as a single element to the end of a list.
# The length of the list itself will increase by one.
a = [1, 2, 3]
a.append([4, 5, 6])

# extend iterates over its argument adding each element to the list, extending the list.
# The length of the list will increase by however many elements were in the iterable argument.
print a
b = [1, 2, 3]
b.extend([4, 5, 6])
print b
