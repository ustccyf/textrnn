import sys,os,random
count = 0
fin = open(sys.argv[1], "r")
for line in fin:
    count += 1
fin.close()
need_num = int(sys.argv[2])
if count < need_num:
    print >>sys.stderr,"need_num > count"
    exit(-1)
num_dict = {}
while len(num_dict) < need_num:
    num = random.randint(1,count)
    num_dict[num] = True
count = 0
fin = open(sys.argv[1], "r")
for line in fin:
    count += 1
    line = line.strip()
    if count in num_dict:
        print line
    else:
        print >>sys.stderr, line
fin.close()

