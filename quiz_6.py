# Randomly fills an array of size 10x10 True and False, displayed as 1 and 0,
# and outputs the number chess knights needed to jump from 1s to 1s
# and visit all 1s (they can jump back to locations previously visited).
#
# Written by *** and Eric Martin for COMP9021


from random import seed, randrange
import sys
import copy

dim = 10


def display_grid():
    for i in range(dim):
        print('     ', end = '')
        print(' '.join(grid[i][j] and '1' or '0' for j in range(dim)))
    print()

def find_join_point():
    pass


def explore_board():
    pass
# Replace pass above with your code

def find_one():
    one = []
    for i in range(0,10):
        for j in range(0,10):
            if grid[i][j] == 1:
                one.append((i,j))
    return(one)

def paths_sun(pa_1,pa_2):
    x_1, y_1 = pa_1
    x_2, y_2 = pa_2
    if not (0 <= x_1 < 10 and 0 <= y_1 < 10) or\
        not (0 <= x_2 < 10 and 0 <= y_2 < 10):
        return []
    if not grid[x_1][y_1]:
        return []
    if pa_1 == pa_2:
        return [[pa_2]]
    paths = [0] * 8
    grid[x_1][y_1] = 0
    paths[0] = paths_sun((x_1 + 1, y_1 - 2), (x_2, y_2))
    paths[1] = paths_sun((x_1 + 2, y_1 - 1), (x_2, y_2))
    paths[2] = paths_sun((x_1 + 2, y_1 + 1), (x_2, y_2))
    paths[3] = paths_sun((x_1 + 1, y_1 + 2), (x_2, y_2))
    paths[4] = paths_sun((x_1 - 1, y_1 + 2), (x_2, y_2))
    paths[5] = paths_sun((x_1 - 2, y_1 + 1), (x_2, y_2))
    paths[6] = paths_sun((x_1 - 2, y_1 - 1), (x_2, y_2))
    paths[7] = paths_sun((x_1 - 1, y_1 - 2), (x_2, y_2))
    grid[x_1][y_1] = 1
    return [[(x_1,y_1)] + path for i in range(8) for path in paths[i]]


try:
    for_seed, n = (int(i) for i in input('Enter two integers: ').split())
    if not n:
        raise ValueError
except ValueError:
    print('Incorrect input, giving up.')
    sys.exit()

seed(for_seed)
if n > 0:
    grid = [[randrange(n) > 0 for _ in range(dim)] for _ in range(dim)]
else:
    grid = [[randrange(-n) == 0 for _ in range(dim)] for _ in range(dim)]    
print('Here is the grid that has been generated:')

display_grid()
#nb_of_knights = explore_board()
all_one = []
all_one = find_one()

#print(all_one)
L = []
for i in range(0, len(all_one)):
    for j in range(i + 1, len(all_one)):
        # print(all_one[i],all_one[j])
        L.append(paths_sun(all_one[i], all_one[j]))
#print(L)

nb_of_knights = 0
while all_one != []:
    L = []
    for i in range(0,len(all_one)):
        for j in range(i+1,len(all_one)):
            #print(all_one[i],all_one[j])
            L.append(paths_sun(all_one[i],all_one[j]))
    LL = []
    for i in L:
        if i != []:
            LL.append(i)
    length = 0
    #print(LL)
    for i in LL:
        #num = int(max(len(i)))
        #print(len(i))
        for j in i:
            if len(j) >= length:
                length = len(j)

    #length= length-2
    #print(length)

    del_one = 0

    if LL == []:
        nb_of_knights += len(all_one)
        break

    for i in LL:
        #num = int(max(len(i)))
        #print(len(i))
        for j in i:
            if len(j) == length:
                del_one = j
    #print("zui chang lu jing ",del_one)
    no1 = False

    del_1 = []
    for i in all_one:
        if i not in del_one:
            del_1.append(i)
    aaa = copy.deepcopy(del_one)
    while True:
        listofother=[]
        #print("sheng xia d 1 ",del_1)
        for i in aaa:
            if (i[0]+1,i[1]-2) in del_1:
                listofother.append((i[0]+1,i[1]-2))
            if (i[0]+2,i[1]-1) in del_1:
                listofother.append((i[0]+2,i[1]-1))
            if (i[0]+2,i[1]+1) in del_1:
                listofother.append((i[0]+2,i[1]+1))
            if (i[0]+1,i[1]+2) in del_1:
                listofother.append((i[0]+1,i[1]+2))
            if (i[0]-1,i[1]+2) in del_1:
                listofother.append((i[0]-1,i[1]+2))
            if (i[0]-2,i[1]+1) in del_1:
                listofother.append((i[0]-2,i[1]+1))
            if (i[0]-2,i[1]-1) in del_1:
                listofother.append((i[0]-2,i[1]-1))
            if (i[0]-1,i[1]-2) in del_1:
                listofother.append((i[0]-1,i[1]-2))
        #print("11111",listofother)
        for j in listofother:
            del_1.remove(j)
        aaa=copy.deepcopy(listofother)
        if listofother==[]:
            break
    all_one = del_1
    nb_of_knights += 1
    #print("shan wan zou guo de 1",all_one)
    #print(nb_of_knights)
#print(all_one)


if not nb_of_knights:
    print('No chess knight has explored this board.')
elif nb_of_knights == 1:
    print(f'At least 1 chess knight has explored this board.')
else:
    print(f'At least {nb_of_knights} chess knights have explored this board')