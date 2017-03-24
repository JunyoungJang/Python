
reader = open('haiku.txt', 'r')
total = 0
count = 0
for line in reader:    # Assign lines of text in file to loop variable one by one
    count += 1
    total += len(line)
reader.close()
print 'average', float(total) / float(count)

