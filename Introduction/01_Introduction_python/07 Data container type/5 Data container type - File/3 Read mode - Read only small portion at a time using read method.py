reader = open('haiku.txt', 'r')
data = reader.read(64)   # Read at most 64 bytes Or the empty string if there is no more data
while data != '':     # Keep looping as long as the last read returned some data
    print len(data)   # Do someting
    data = reader.read(64)   # Try to reload
print len(data)   # Should be 0 (or the loop would still be running) after loop is over
reader.close()    # It returns 64,64,64,64,4,0
