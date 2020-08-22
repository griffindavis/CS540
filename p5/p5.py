import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# %matplotlib inline
import copy
import math
import heapq
from os.path import dirname, join
import sys
original_stdout = sys.stdout
current_dir = dirname(__file__)

def printMaze(height, width, img, solution):
    bottomLine = ''
    for i in range(height):
        succ = []
        horizontalLine = ''
        verticalLine = ''
        for j in range(width):
            s = ''
            c1, c2 = i * 16 + 8, j * 16 + 8
            if img[c1-8, c2] == 1: s += 'U' # can move up
            if img[c1+8, c2] == 1: s += 'D' # can move down
            if img[c1, c2-8] == 1: s += 'L' # can move left
            if img[c1, c2+8] == 1: s += 'R' # can move right
            cells[i][j].succ = s
            succ.append(s)

            if (i,j) in solution or (i,j) == (0,28):
                placeholder = '#'
            else:
                placeholder = ' '

            horizontalLine += '+'
            if 'U' not in s:
                horizontalLine += '--'
            else:
                horizontalLine += placeholder*2
            if 'L' not in s:
                verticalLine += '|' + placeholder*2
            else:
                verticalLine += placeholder*3
            if j == width - 1:
                horizontalLine += '+'
                if 'R' not in s:
                    verticalLine += '|'
                else:
                    verticalLine += ' '

            if i == height - 1: # height - 1?
                bottomLine += '+'
                if 'D' not in s:
                    bottomLine += '--'
                else:
                    bottomLine += placeholder*2
                if j == width - 1:
                    bottomLine += '+'
        print(horizontalLine)
        print(verticalLine)
    print(bottomLine)

def printSearched(height, width, sequence):
    for i in range(height):
        string = ""
        for j in range(width):
            if (i,j) in sequence:
                string += str(1) + ","
            else:
                string += str(0) + ","
        print(string[:-1])

'''
The below script is based on a 55 * 57 maze. 
Todo:
	1. Plot the maze and solution in the required format.
	2. Implement DFS algorithm. (I've given you the BFS below)
	3. Implement A* with Euclidean distance. (I've given you the one with Manhattan distance)

'''
height, width = 58, 57
X, Y = 14, 2

file_path = join(current_dir, '../Data/maze.png')
ori_img = mpimg.imread(file_path)
img = ori_img[:,:,0]

class Cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.succ = ''
        self.action = ''  # which action the parent takes to get this cell

# Height by width
cells = [[Cell(i,j) for j in range(width)] for i in range(height)]

file_path = join(current_dir, '../Data/p5_output_Q1_maze.txt')
with open(file_path, 'w+') as f:
    sys.stdout = f
    printMaze(height, width, img, [])
    sys.stdout = original_stdout
# 2    

cells[0][28].succ = cells[0][28].succ.replace('U', '')
cells[57][28].succ = cells[57][28].succ.replace('D', '')

for line in cells:
    string = ""
    for val in line:
        if string == "":
            string = val.succ
        else:
            string += "," + val.succ
    #print(string)

# bfs
visited = set()
s1 = {(0,28)}
s2 = set()
while (57,28) not in visited:
    for a in s1:
        visited.add(a)
        i, j = a[0], a[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in (s1 | s2 | visited): 
            s2.add((i-1,j))
            cells[i-1][j].action = 'U'
        if 'D' in succ and (i+1,j) not in (s1 | s2 | visited): 
            s2.add((i+1,j))
            cells[i+1][j].action = 'D'
        if 'L' in succ and (i,j-1) not in (s1 | s2 | visited): 
            s2.add((i,j-1))
            cells[i][j-1].action = 'L'
        if 'R' in succ and (i,j+1) not in (s1 | s2 | visited): 
            s2.add((i,j+1))
            cells[i][j+1].action = 'R'     
    s1 = s2
    s2 = set()
    
cur = (57,28)
s = ''
seq = []
while cur != (0,28):
    seq.append(cur)
    i, j = cur[0], cur[1]
    t = cells[i][j].action
    s += t
    if t == 'U': cur = (i+1, j)
    if t == 'D': cur = (i-1, j)
    if t == 'L': cur = (i, j+1)
    if t == 'R': cur = (i, j-1)
action = s[::-1]
seq = seq[::-1]
print(action)
# 3 

printSearched(height, width, visited)

file_path = join(current_dir, '../Data/p5_output_Q4_maze.txt')
with open(file_path, 'w+') as f:
    sys.stdout = f
    printMaze(height, width, img, seq)
    sys.stdout = original_stdout

'''
Do this after printMaze since printMaze modifies cells...
'''
cells[0][28].succ = cells[0][28].succ.replace('U', '')
cells[57][28].succ = cells[57][28].succ.replace('D', '')


#dfs
stack = [(0,28)]
visited = set()
while stack and (54,28) not in visited:
    node = stack.pop(0)
    if node not in visited:
        visited.add(node)
        i, j = node[0], node[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in visited:
            if (i-1,j) not in stack: stack += [(i-1,j)]
            cells[i-1][j].action = 'U'
        if 'D' in succ and (i+1,j) not in visited:
            if (i+1,j) not in stack: stack += [(i+1,j)]
            cells[i+1][j].action = 'D'
        if 'L' in succ and (i,j-1) not in visited:
            if (i,j-1) not in stack: stack += [(i,j-1)]
            cells[i][j-1].action = 'L'
        if 'R' in succ and (i,j+1) not in visited:
            if (i,j+1) not in stack: stack += [(i,j+1)]
            cells[i][j+1].action = 'R'

print()
print()
printSearched(height, width, visited)

'''
cur = (57,28)
s = ''
seq = []
while cur != (0,28):
    seq.append(cur)
    i, j = cur[0], cur[1]
    t = cells[i][j].action
    s += t
    if t == 'U': cur = (i+1, j)
    if t == 'D': cur = (i-1, j)
    if t == 'L': cur = (i, j+1)
    if t == 'R': cur = (i, j-1)
action = s[::-1]
seq = seq[::-1]
print(action)
'''
def manhattanToGoal(point):
    return abs(point[0] - 57) + abs(point[1] - 28)

print()
print()
for i in range(height):
    string = ""
    for j in range(width):
        string += str(manhattanToGoal((i,j))) + ","
    print(string[:-1])


## Part2
man = {(i,j): abs(i-57) + abs(j-28) for j in range(width) for i in range(height)}


# manhattan   use man
g = {(i,j): float('inf') for j in range(width) for i in range(height)}
g[(0,28)] = 0

queue = [(0,28)]
visited = set()

while queue and (57,28) not in visited:
    queue.sort(key=lambda x: g[x] + man[x])
    point = queue.pop(0)
    if point not in visited:
        visited.add(point)
        i, j = point[0], point[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in visited:
            if (i-1,j) not in queue: queue += [(i-1,j)]
            g[(i-1,j)] = min(g[(i-1,j)], g[(i,j)]+1)
        if 'D' in succ and (i+1,j) not in visited:
            if (i+1,j) not in queue: queue += [(i+1,j)]
            g[(i+1,j)] = min(g[(i+1,j)], g[(i,j)]+1)
        if 'L' in succ and (i,j-1) not in visited:
            if (i,j-1) not in queue: queue += [(i,j-1)]
            g[(i,j-1)] = min(g[(i,j-1)], g[(i,j)]+1)
        if 'R' in succ and (i,j+1) not in visited:
            if (i,j+1) not in queue: queue += [(i,j+1)]
            g[(i,j+1)] = min(g[(i,j+1)], g[(i,j)]+1)   

print()
print()  
printSearched(height, width, visited)

#euc
euc = {(i,j): math.sqrt((i-54)**2 + (j-28)**2 ) for j in range(width) for i in range(height)}
g = {(i,j): float('inf') for j in range(width) for i in range(height)}
g[(0,28)] = 0

queue = [(0,28)]
visited = set()

while queue and (57,28) not in visited:
    queue.sort(key=lambda x: g[x] + euc[x])
    point = queue.pop(0)
    if point not in visited:
        visited.add(point)
        i, j = point[0], point[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in visited:
            if (i-1,j) not in queue: queue += [(i-1,j)]
            g[(i-1,j)] = min(g[(i-1,j)], g[(i,j)]+1)
        if 'D' in succ and (i+1,j) not in visited:
            if (i+1,j) not in queue: queue += [(i+1,j)]
            g[(i+1,j)] = min(g[(i+1,j)], g[(i,j)]+1)
        if 'L' in succ and (i,j-1) not in visited:
            if (i,j-1) not in queue: queue += [(i,j-1)]
            g[(i,j-1)] = min(g[(i,j-1)], g[(i,j)]+1)
        if 'R' in succ and (i,j+1) not in visited:
            if (i,j+1) not in queue: queue += [(i,j+1)]
            g[(i,j+1)] = min(g[(i,j+1)], g[(i,j)]+1)   

print()
print()  
printSearched(height, width, visited)