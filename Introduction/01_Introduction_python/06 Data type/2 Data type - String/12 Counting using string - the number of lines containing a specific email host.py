# count the number of lines containing @uct.ac.za

file_name = raw_input('Enter the file name : ')

with open(file_name, 'r') as a:

    b = a.readlines()
    count = 0
    for line in b:
        if not '@uct.ac.za' in line:
            continue
        if '@uct.ac.za' in line:
            print line.rstrip()
            count = count + 1

    print 'Line Count containing \'@uct.ac.za\' : ', count


