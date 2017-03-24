a = open('example.txt', 'a')
l = ['Line 4', 'Line 5', 'Line 6']
for item in l:
    a.write(item + '\n')
a.close()
