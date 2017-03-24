for i in [5,4,3,2,1]:
    print i

friends = ['Joseph', 'Glenn', 'Sally']
for friend in friends:
    print 'Happy New Year : ', friend

friends = ['Joseph', 'Glenn', 'Sally']
print len(friends) # 3
print range(len(friends)) # [0, 1, 2]
for i in range(len(friends)):
    friend = friends[i]
    print 'Happy New Year : ', friend

friends = ['Joseph', 'Glenn', 'Sally']
for i, friend in enumerate(friends):
    print i, 'Happy New Year : ', friend


