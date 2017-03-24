import sys
if len(sys.argv)==1:
    count_lines(sys.stdin)
else:
    rd=open(sys.argv[1],'r')
    count_lines(rd)
    rd.close()
print count.py