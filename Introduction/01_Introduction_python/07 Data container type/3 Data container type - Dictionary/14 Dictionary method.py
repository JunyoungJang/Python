'''
print 'dictionary methods - clear ---'
d = {'name': 'Johann Sebastian Bach', 'birth': '31 March 1685', 'death': '28 July 1750', 1: 65, 2: 7}
d.clear()
print d # {}
'''
'''
'''
#'''
#'''
'''
print 'dictionary methods - get ---'
# most common word
name_counts = dict()
names = ['csev', 'cwen', 'csev', 'zqian', 'cwen']
for name in names:
        name_counts[name] = name_counts.get(name, 0) + 1 # Get the value assiged by key. Give 0 if key does not exist yet
print name_counts # {'csev': 2, 'zqian': 1, 'cwen': 2}
'''
'''
'''
#'''
#'''
'''
print 'dictionary methods - keys, values, items ---'
d = {'name': 'Johann Sebastian Bach', 'birth': '31 March 1685', 'death': '28 July 1750', 1: 65, 2: 7}
print d.keys()   # [1, 2, 'death', 'name', 'birth']
print d.values() # [65, 7, '28 July 1750', 'Johann Sebastian Bach', '31 March 1685']
print d.items()  # [(1, 65), (2, 7), ('death', '28 July 1750'), ('name', 'Johann Sebastian Bach'), ...]
'''

