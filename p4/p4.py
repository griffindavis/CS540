import numpy as np
from os.path import dirname, join
import math

current_dir = dirname(__file__)
file_path = join(current_dir, "../Data/time_series_covid19_deaths_global.csv")

# For 'South Korea', and "Bonaire Sint Eustatius and Saba" (line 145 and 257), I removed the ',' in name manually
with open(file_path) as f:
    data = list(f)[1:]

'''
Todo: 
1. Part 1 in P4.
2. Euclidean distance (currently are all manhattan in my code below)
3. Complete linkage distance
4. Total distortion
5. Output all required information in correct format

PS: Currently, I choose 
	n = num of all distinct countries, and
	m = 3 (latitude, longitude, total deaths until Jun27, 
		  i.e., 1st, 2nd, last number for each country as parameters).
	Also, for countries that have several rows, I average the latitude, longitude and sum up the deaths.

	You may need to change some of that based on your part 1 results.

'''
# create the dict
tmp_dict = {}
question1 = {}
for d in data:
    l = d.strip('\n').split(',')
    c = l[1]  # country
    if (c == 'US' or c == 'Canada'):
        if (c in question1):
            for i in range(4, len(l) - 1):
                question1[c][i - 4] += float(l[i])
        else:
            question1[c] = [float(l[index]) for index in range(4, len(l))]
    if c in tmp_dict:
        for i in range(4, len(l) - 1):
            tmp_dict[c][i - 4] += float(l[i])
    else:
        tmp_dict[c] = [float(l[index]) for index in range(4, len(l))]

#d_dict = {k:np.array([sum(v[0])/len(v[0]), sum(v[1])/len(v[1]), sum(v[2])]) for k,v in d_dict.items()}

'''
Double
Quadruple
Octuple
'''
def dubQuadOct(set):
    current = set[-1]
    start = len(set) - 1
    dbl, qdrpl, octpl = None, None, None
    for i in range(len(set) - 2, 0, -1):
        if ((dbl == None) and (set[i] <= current / 2)):
            dbl = start - i
        elif ((qdrpl == None) and (set[i] <= current / 4)):
            qdrpl = start - i
        elif (octpl == None and (set[i] <= current / 8)):
            octpl = start - i

    # unfortunately can't just drop bad data
    if (dbl == None):
        dbl == len(set) - 1
    if (qdrpl == None):
        qdrpl == len(set) - 1
    if (octpl == None):
        octpl == len(set) - 1
    return [dbl, qdrpl, octpl]

d_dict = {}
for key in tmp_dict:
    temp = dubQuadOct(tmp_dict[key])
    if (None in temp): # for bad reporting
        continue
    d_dict[key] = temp

countries = sorted([c for c in d_dict.keys()])

''' Question 1 '''
print("Question 1")
print(question1['US'])
print(question1['Canada'])
''' Question 2'''
for key in question1:
    temp = float(0)
    prev = float(0)
    for i in range(1, len(question1[key])):
        temp = question1[key][i]
        question1[key][i] = question1[key][i] - prev
        prev = temp
print("Question 2")
print(question1['US'][1:])
print(question1['Canada'][1:])

print("question 4")
for key in d_dict:
    print(str(d_dict[key][0]) + ",", str(d_dict[key][1]) + ",", str(d_dict[key][2]))

'''
manhattan distance
'''
def manhattan(a,b):
    return sum(abs(a[i]-b[i]) for i in range(len(a)))

'''
Euclidean Distance
'''
def euclidean(a, b):
    return (math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2))

'''
 single linkage distance
 '''
def sld(cluster1, cluster2): 
    res = float('inf')
    # c1, c2 each is a country in the corresponding cluster
    for c1 in cluster1:
        for c2 in cluster2:
            dist = euclidean(d_dict[c1], d_dict[c2])
            if dist < res:
                res = dist
    return res

'''
Complete linkage distance
'''
def cld(cluster1, cluster2):
    res = float(0)
    for c1 in cluster1:
        for c2 in cluster2:
            dist=euclidean(d_dict[c1], d_dict[c2])
            if dist > res: # here's the difference
                res = dist
    return res


k = 5

# hierarchical clustering (sld, 'euclidean')
n = len(d_dict)
clusters = [{d} for d in d_dict.keys()]
for _ in range(n-k):
    dist = float('inf')
    best_pair = (None, None)
    for i in range(len(clusters)-1):
        for j in range(i+1, len(clusters)):
            if sld(clusters[i], clusters[j]) < dist:
                dist = sld(clusters[i], clusters[j])
                best_pair = (i,j)
    new_clu = clusters[best_pair[0]] | clusters[best_pair[1]]
    clusters = [clusters[i] for i in range(len(clusters)) if i not in best_pair]
    clusters.append(new_clu)

'''
Prints the clusters in the proper format
'''
def printClusters(d_dict, clusters):
    string = ""
    for key in d_dict:
        for i in range(len(clusters)):
            if key in clusters[i]:
                if (string == ""):
                    string = str(i)
                else:
                    string += "," + str(i)
    print(string)

'''
Question 5
'''
print("Question 5")
printClusters(d_dict, clusters)

'''
Question 6
'''
clusters = None
clusters = [{d} for d in d_dict.keys()]
for _ in range(n-k):
    dist = float('inf')
    best_pair = (None, None)
    for i in range(len(clusters)-1):
        for j in range(i+1, len(clusters)):
            if cld(clusters[i], clusters[j]) < dist:
                dist = cld(clusters[i], clusters[j])
                best_pair = (i,j)
    new_clu = clusters[best_pair[0]] | clusters[best_pair[1]]
    clusters = [clusters[i] for i in range(len(clusters)) if i not in best_pair]
    clusters.append(new_clu)

print("Question 6")
printClusters(d_dict, clusters)

## k-means (euclidean)
import copy

'''
Finding the center
'''
def center(cluster):
    return np.average([d_dict[c] for c in cluster], axis=0)

'''main'''

init_num = np.random.choice(len(countries) - 1, k)
clusters = [{countries[i]} for i in init_num]
while True:
    new_clusters = [set() for _ in range(k)]
    centers = [center(cluster) for cluster in clusters]
    for c in countries:
        clu_ind = np.argmin([euclidean(d_dict[c], centers[i]) for i in range(k)])
        new_clusters[clu_ind].add(c)
    if all(new_clusters[i] == clusters[i] for i in range(k)):
        break
    else:
        clusters = copy.deepcopy(new_clusters)

'''
Question 7
'''
print("Question 7")
printClusters(d_dict, clusters)

'''
Question 8
'''
print("Question 8")
for k in centers:
    print("%.4f,%.4f,%.4f" %(k[0], k[1], k[2]))

def calcDistortion(values, centers):
    return (values[0] - centers[0])**2 + (values[1] - centers[1])**2 + (values[2] - centers[2])**2


'''
Question 9
'''
print("Question 9")
distortion = 0
for key in d_dict:
    for i in range(len(clusters)):
            if key in clusters[i]:
                center_values = centers[i]
                distortion += calcDistortion(d_dict[key], centers[i])
print(distortion)