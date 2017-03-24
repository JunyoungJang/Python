BLOCKSIZE = 1024
reader = open('haiku.txt', 'r')
writer = open('temp.txt', 'w')
data = reader.read(BLOCKSIZE)
while len(data) > 0:
    writer.write(data)
    data = reader.read(BLOCKSIZE)
reader.close()
writer.close()








