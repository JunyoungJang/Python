name_counts = dict()
names = ['csev', 'cwen', 'csev', 'zqian', 'cwen']
for name in names:
    if name not in name_counts:
        name_counts[name] = 1
    else:
        name_counts[name] = name_counts[name] + 1
print name_counts # {'csev': 2, 'zqian': 1, 'cwen': 2}

name_counts = dict()
names = ['csev', 'cwen', 'csev', 'zqian', 'cwen']
for name in names:
        name_counts[name] = name_counts.get(name, 0) + 1 # Get the value assiged by key. Give 0 if key does not exist yet
print name_counts # {'csev': 2, 'zqian': 1, 'cwen': 2}





