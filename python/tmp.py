n = 96
res = [0 for sid in range(n)]

i = 0
s = 0
with open("tmp.txt")  as f:
    for l in f:
        if not l.startswith('cTri:'):
            continue
        i += 1
        j = l.index(':')
        k = l.index(',')
        sid = int(l[j+1:k])
        s += sid
        
        k += 1
        while l[k] != ',':
            k += 1
        res[int(l[k+1:])] = sid

# res2 = [0 for sid in range(n)]
# s2 = 0
# with open("tmp3.txt")  as f:
#     for l in f:
#         if not l.startswith('cTri'):
#             continue
#         i += 1
#         j = l.index(':')
#         k = l.index(',')
#         sid = int(l[j+1:k])
#         s2 += sid
        
#         # k += 1
#         # while l[k] != ',':
#         #     k += 1
#         res2[int(l[k+1:])] = sid

print(s)
for i in range(n):
    if res[i] == 0:
        print i
# for i in range(n):
#     if res[i] != res2[i]:
#         print i, res[i], res2[i]
# for u in range(1000001, 1480451):
#     if ex[u] == 0:
#         print('noex', u)
