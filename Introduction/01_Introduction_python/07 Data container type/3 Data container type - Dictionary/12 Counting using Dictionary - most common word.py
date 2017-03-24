# counts = {word_1: word_1_count, word_2: word_2_count, ...}
counts = dict()
file_name = raw_input('Enter the text file name : ')
a = open(file_name)
text = a.read()
words = text.split()
for word in words:
    counts[word] = counts.get(word, 0) + 1

max_so_far = 'None'
max_so_far_key = 'None'
for key, value in counts.items():
    print key, value
    if max_so_far == 'None' or value > max_so_far:
        max_so_far = value
        max_so_far_key = key
print 'Max Count : ', max_so_far
print 'Most Common Word : ', max_so_far_key





