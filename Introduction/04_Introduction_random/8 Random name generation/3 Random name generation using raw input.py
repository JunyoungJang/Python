import string
import random

Letter_set = string.ascii_letters
LETTER_set = string.ascii_uppercase
letter_set = string.ascii_lowercase

print dir(random)
print help(random.choice)
print random.choice(Letter_set)

def random_name_generator(length_of_name):

    n = length_of_name
    random_name = ''

    for i in range(n):
        if i == 0:
            random_name = random_name + random.choice(LETTER_set)
        else:
            random_name = random_name + random.choice(letter_set)

    return random_name

length_of_name = raw_input('What is the length of the random name to generate? ')
length_of_name = int(length_of_name)
print random_name_generator(length_of_name)


