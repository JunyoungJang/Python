# counts = {word_1: word_1_count, word_2: word_2_count, ...}
counts = dict()
file_name = raw_input('Enter the text file name : ')
a = open(file_name)
text = a.read()
words = text.split()
for word in words:
    counts[word] = counts.get(word, 0) + 1

counts_sorted = sorted( [ (value, key) for key, value in counts.items() ], reverse=True)
for value, key in counts_sorted[:10]:
    print key, value





