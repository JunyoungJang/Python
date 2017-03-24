reader = open('haiku.txt', 'r')
writer = open('temp.txt', 'w')
for line in reader:
    writer.write(line)
reader.close()
writer.close()









