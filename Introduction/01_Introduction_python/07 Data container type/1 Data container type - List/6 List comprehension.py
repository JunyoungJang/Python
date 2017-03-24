nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print squares   #  [0, 1, 4, 9, 16]

nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print even_squares  #  [0, 4, 16]



