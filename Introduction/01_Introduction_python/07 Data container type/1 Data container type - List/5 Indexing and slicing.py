# indexing
a = []
x = [1, 2, 3, 4, ['a', 'b', ['Sun', 'Moon']]]
print x
print x[0]
print x[0] + x[2]
print x[-1]
print x[-1][-1][0]
print x, x[1], x[-1], x[-2] # index from 0, not 1
friends = ['Joseph', 'Glenn', 'Sally']
print friends[1]

# slicing
nums = range(5)
print nums
print nums[2:4]
print nums[2:]
print nums[:2]
print nums[:]
nums[2:4] = [8, 9] # Assign a new sublist to a slice
print nums         # Prints "[0, 1, 8, 9, 4]"
