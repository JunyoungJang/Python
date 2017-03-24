# Step 1: Put a text file in the file where the python code is running
# Step 2: Open the file
# open(filename, mode)
# mode - r (read - default), w (write), a (append), r+, w+, a+

a = open('example.txt', 'r')

b = a.readlines()
print b
b = [line.rstrip('\n') for line in b]
print b

for line in b:
    print line

a.close()
