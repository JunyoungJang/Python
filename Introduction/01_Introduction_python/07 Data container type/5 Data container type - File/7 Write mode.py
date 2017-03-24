a = open('example.txt', 'w') #  at this moment the file has nothing in it
print a
a.close()

a = open('example.txt', 'w')
l = ['Line 1', 'Line 2', 'Line 3']
for item in l:
    a.write(item + '\n')
a.close()

