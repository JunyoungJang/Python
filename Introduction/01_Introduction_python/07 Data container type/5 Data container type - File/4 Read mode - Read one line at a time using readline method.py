reader = open('haiku.txt', 'r')
line = reader.readline()  # read a single line
total = 0
count = 0
while line != '':     # Keep looping until no more lines in file
    count += 1
    total += len(line)
    line = reader.readline()
reader.close()
print 'average', float(total) / float(count)
