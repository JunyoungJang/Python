# Looping through string - example 1 ---
# count 'a' in 'banana'
count = 0
for letter in 'banana':
    if letter == 'a':
        count = count + 1
print count

# Looping through string - example 2 ---
# multiple index
for index, letter in enumerate('banana'):
    print index, letter
