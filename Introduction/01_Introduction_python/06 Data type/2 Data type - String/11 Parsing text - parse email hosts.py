with open('mbox-short.txt', 'r') as a:

    b = a.readlines()

    for line in b:
        line = line.rstrip()

        if not line.startswith('From '):
            continue

        words = line.split()
        email = words[1]
        pieces = email.split('@')
        print pieces
        host = pieces[1]
        print host

