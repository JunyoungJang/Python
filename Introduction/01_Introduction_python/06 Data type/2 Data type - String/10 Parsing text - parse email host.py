data = 'From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008'

start_index = data.find('@')
print start_index # 21

end_index = data.find(' ',start_index) # find ' ' after start_index
print end_index # 31

host = data[start_index+1:end_index]
print host # @uct.ac.za

