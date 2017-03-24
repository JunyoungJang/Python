# Loops: Iterating over a set has the same syntax as iterating over a list;
# however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:

animals = {'cat', 'dog', 'fish'}
print enumerate(animals)
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal) # Prints "#1: fish", "#2: dog", "#3: cat"

