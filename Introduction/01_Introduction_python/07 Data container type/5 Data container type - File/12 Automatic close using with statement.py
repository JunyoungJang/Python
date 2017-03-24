with open('example.txt', 'a+') as sess:
    a = sess.readlines() # read line by line starting from the pointer, which is at the end
    print a              # []
    sess.seek(0)         # move pointer to the beginning
    b = sess.readlines() # read line by line starting from the pointer, which is at the beginning
    print b              # ['Line 1\n', 'Line 2\n', 'Line 3\n']
    sess.write('Line 7')
    sess.seek(0)         # move pointer to the beginning
    c = sess.readlines() # read line by line starting from the pointer, which is at the beginning
    print c

