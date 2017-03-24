print 'intersection, union, difference ---'
s1 = set([1, 2, 3, 4, 5, 6])
s2 = set([4, 5, 6, 7, 8, 9])
s3 = s1.intersection(s2)
s4 = s1.union(s2)
s5 = s1.difference(s2)
print s3 # set([4, 5, 6])
print s4 # set([1, 2, 3, 4, 5, 6, 7, 8, 9])
print s5 # set([1, 2, 3])

print 'add, update, remove ---'
s1 = set([1,2,3])
s1.add(4)
print s1 # set([1, 2, 3, 4])
s2 = set([1,2,3])
s2.update([4,5,6])
print s2 # set([1, 2, 3, 4, 5, 6])
s3 = set([1,2,3])
s3.remove(2)
print s3  # set([1, 3])