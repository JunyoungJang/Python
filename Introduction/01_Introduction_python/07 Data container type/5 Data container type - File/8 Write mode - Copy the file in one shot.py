reader = open('haiku.txt', 'r')
data = reader.read()
reader.close()
writer = open('temp.txt', 'w')  # write all to the destination file
writer.write(data)
writer.close()








