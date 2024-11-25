from random import randint
import commands

n = 100
T = 10
outF = "data/my/"
logF = 'logs/diffMy/'

if False:
    for i in range(T):
        adj = [[] for u in range(n)]
        m = 0

        for u in range(n - 1):
            du = randint(1, n - u - 1)
            m += du

            for j in range(du):
                v = randint(u + 1, n - 1)
                while v in adj[u]:
                    v = randint(u + 1, n - 1)
                adj[u].append(v)

            adj[u].sort()

        with open(outF + "{}.txt".format(i), 'w') as f:
            # f.write("{} {} {}\n".format(n1, m))

            for u in range(n):
                for v in adj[u]:
                    f.write("{} {}\n".format(u, v))
        


    for i in range(T):
        print i
        f = outF + "{}.txt".format(i)
        run = " ./build/exe/src/main.cu.exe -g "+f+" -d 0 -m kc  -d 0 -m kc -o degen -k 4 -p node  -q l8b -s 0  > " + logF + "{}a.txt".format(i)
        # print run
        commands.getstatusoutput(run)

        run = " ./build/exe/src/main.cu.exe -g "+f+" -d 0 -m kc  -d 0 -m kc -o degen -k 4 -p node  -q p8b -s 0  > " + logF + "{}b.txt".format(i)
        # print run
        commands.getstatusoutput(run)
    print

for i in range(T):
    c1 = 0
    c2 = 0

    with open(logF + "{}a.txt".format(i), "r") as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith('------------- Level'):
                p = line.index('Counter = ')
                c1 = int(line[p + len("Counter = "):].replace(',', ''))
    
    
    with open(logF + "{}b.txt".format(i), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('[0m------------- Level ='):
                p = line.index('Counter = ')
                c2 = int(line[p + len("Counter = "):].replace(',', ''))
    # print c1, c2
    if c1 != c2:
        print i, c1, c2