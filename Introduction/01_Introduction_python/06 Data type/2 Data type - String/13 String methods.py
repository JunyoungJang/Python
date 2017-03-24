'''
print 'string methods - capitalize, upper, lower, swapcase ---'
s = "hello"
print s.capitalize()
print s.upper()
print s.lower()
print s.swapcase()
'''
'''
'''
#'''
#'''
'''
print 'string methods - count ---'
dna = 'acggtggtcac'
print dna.count('g') # 4
'''
'''
'''
#'''
#'''
'''
print 'string methods - find ---'
data = 'From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008'

start_index = data.find('@')
print start_index # 21

end_index = data.find(' ',start_index) # find ' ' after start_index
print end_index # 31

host = data[start_index+1:end_index]
print host # @uct.ac.za
'''
'''
'''
#'''
#'''
#'''
print 'string methods - replace ---'
dna = 'acggtggtcac'
print dna.replace('t', 'x') # original dna does not change - strings are immutable
print dna.replace('gt', '') # original dna does not change - strings are immutable
print dna                   # original dna does not change - strings are immutable
a = dna.replace('t', 'x')
print a                     # acggxggxcac
#'''
'''
'''
#'''
#'''
'''
print 'string methods - split ---'
a = 'Life is too short'
b = a.split()
print a
print b
'''
'''
'''
#'''
#'''
'''
print 'string methods - startswith ---'
line = 'Life is too short'
print line.startswith('Life') # True
print line.startswith('l')    # False
'''
'''
'''
#'''
#'''
'''
print 'string methods - strip, lstrip,rstrip - strip white space ---'
sp = '           hi           '
print sp.lstrip(), '.' # strip left empty spaces
print sp.rstrip(), '.' # strip right empty spaces
print sp.strip(), '.'  # strip empty spaces
'''
'''
'''
#'''
#'''
'''
print 'string methods - can be called together ---'
dna = 'acggtggtcac'
print dna.replace('gt', '').find('gc')
'''
