import copy
import numpy as np
from math import log2
from os.path import dirname, join

limit = 1
feature_list = []
root = None
valditation = None

'''
Calculates the entropy of the data
'''
def entropy(data):
    count = len(data)
    p0 = sum(b[-1] == 2 for b in data) / count # percentage of benign

    if p0 == 0 or p0 == 1: return 0 # short-circuit

    p1 = 1 - p0 # remainder is malignant

    return -p0 * log2(p0) - p1 * log2(p1)


''' 
Calculates the information gain
'''
def informationGain(data, feature, threshold):  # x_fea <= threshold;  fea = 2,3,4,..., 10; threshold = 1,..., 9
    count = len(data)

    d1 = data[data[:, feature - 1] <= threshold]
    d2 = data[data[:, feature - 1] > threshold]

    if len(d1) == 0 or len(d2) == 0: 
        return 0

    return entropy(data) - (len(d1) / count * entropy(d1) + len(d2) / count * entropy(d2))

'''
Identifies the best split
'''
def find_best_split(data):
    c = len(data)

    c0 = sum(b[-1] == 2 for b in data) # number of benign

    if c0 == c: return (2, None)
    if c0 == 0: return (4, None)

    ig = [[informationGain(data, feature, threshold) for threshold in range(1, 10)] for feature in feature_list]

    ig = np.array(ig) # convert to array

    max_ig = max(max(i) for i in ig) # find the maximum

    if max_ig == 0:
        if c0 >= c - c0:
            return (2, None)
        else:
            return (4, None)

    ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    fea, threshold = feature_list[ind[0]], ind[1] + 1

    return (fea, threshold)

'''
Does the actual splitting
'''
def split(data, node):
    fea, threshold = node.fea, node.threshold
    d1 = data[data[:,fea-1] <= threshold]
    d2 = data[data[:, fea-1] > threshold]
    return (d1,d2)

'''
The node class
'''
class Node:
    def __init__(self, fea, threshold):
        self.fea = fea
        self.threshold = threshold
        self.left = None
        self.right = None

    def printTree(self, indent):
        if (isinstance(self.left, Node)):
            print(" " * indent + "if (x%d <= %d)" %(self.fea, self.threshold))
            self.left.printTree(indent + 1)
        else:
            print(" " * indent + "if (x%d <= %d) return %d" %(self.fea, self.threshold, self.left))


        if (isinstance(self.right, Node)):
            print(" " * indent +"else")
            self.right.printTree(indent + 1)
        else:
            print(" " * indent + "else return %d" %(self.right))

'''
Creates the tree
'''
def create_tree(data, node, depth):
    d1,d2 = split(data, node)

    f1, t1 = find_best_split(d1)
    f2, t2 = find_best_split(d2)

    if (depth >= limit - 1):
        tempAccuracy = calcAccuracy(root)

        majorityUnder = getMajorityUnderThreshold(d1, f1, t1)
        node.left = majorityUnder
        node.right = 4 if majorityUnder == 2 else 2

        # this is bad, don't do this
        if (calcAccuracy(root) < tempAccuracy):
            temp = node.left
            node.left = node.right
            node.right = temp
        return 

    if t1 == None: 
        node.left = f1
    else:
        node.left = Node(f1,t1)
        create_tree(d1, node.left, depth + 1)

    if t2 == None: 
        node.right = f2
    else:
        node.right = Node(f2,t2)
        create_tree(d2, node.right, depth + 1)

def getMajorityUnderThreshold(data, feature, threshold):
    if (threshold == None):
        return feature
    count2 = sum((instance[-1] == 2 and instance[feature] <= threshold) for instance in data)
    count4 = sum((instance[-1] == 4 and instance[feature] <= threshold) for instance in data)
    if (count2 >= count4):
        return 2
    else:
        return 4

'''
Gets the value given the feature values
'''
def getValue(featureList, root):
    currNode = root
    while(isinstance(currNode, Node)):
        if (featureList[currNode.fea - 1] <= currNode.threshold):
            if (isinstance(currNode.left, Node)):
                currNode = currNode.left
                continue
            else:
                return currNode.left
        else:
            if (isinstance(currNode.right, Node)):
                currNode = currNode.right
                continue
            else:
                return currNode.right

def calcAccuracy(root):
    global valditation
    return sum((getValue(instance, root) == instance[-1]) for instance in valditation) / len(valditation)

'''
Answer to questions found below
'''
def question1(set, root):
    total = len(set)
    countBenign = sum(instance[-1] == 2 for instance in set)
    countMalignant = total - countBenign

    print("Benign: ", countBenign)
    print("Malignant: ", countMalignant)

def question3(set, root):
    above_benign = 0
    below_benign = 0
    above_malignant = 0
    below_malignant = 0

    threshold = root.threshold
    feature = 2
    for instance in set:
        value = instance[-1]
        if (instance[feature - 1] <= threshold):
            if (value == 2):
                below_benign += 1
            else:
                below_malignant += 1
        else:
            if( value == 2):
                above_benign += 1
            else:
                above_malignant += 1
    print(above_benign, ",", below_benign, ",", above_malignant, ",", below_malignant)

def question5(root):
    root.printTree(0)

def question6(root):
    return findDepth(root, 0)

def findDepth(currNode, depth):
    if(isinstance(currNode, Node) != True):
        return depth
    depthLeft = findDepth(currNode.left, depth + 1) 
    depthRight = findDepth(currNode.right, depth + 1)
    if(depthLeft > depthRight):
        return depthLeft
    return depthRight

def question7(set, root):
    value = ""
    for instance in set:
        value = value + str(getValue(instance, root)) + ","
    return value[:-1]


def main():
    current_dir = dirname(__file__)
    file_path = join(current_dir, 'breast-cancer-wisconsin.data')

    with open(file_path, 'r') as f:
        a = [l.strip('\n').split(',') for l in f if '?' not in l]

    a = np.array(a).astype(int)   # training data
    global valditation
    valditation = a[::8]

    ''' this is for part 1'''
    global feature_list, root
    feature_list = [2]
    ig = [[informationGain(a, fea, t) for t in range(1,10)] for fea in feature_list]
    ig = np.array(ig)
    ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    root = Node(2, ind[1] + 1)
    create_tree(a, root, 0)
    '''
    output question answers
    '''
    print()
    question1(a, root)
    print("Question 2: %0.4f" %(entropy(a)))
    question3(a, root)
    print("Question 4: %0.4f" %(informationGain(a, 2, root.threshold)))

    ''' begin part 2'''
    ig = None
    ind = None
    feature_list = [3, 10, 5, 8, 4]
    ig = [[informationGain(a, fea, t) for t in range(1,10)] for fea in feature_list]
    ig = np.array(ig)
    ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    root = None
    global limit 
    limit = 100 # just get this large so that we can get some depth
    root = Node(feature_list[ind[0]], ind[1] + 1)
    create_tree(a, root, 0)
    print("Question 5")
    question5(root)
    print("Question 6: %d" %(question6(root)))

    file_path = join(current_dir, 'p2_test.txt')
    with open(file_path, 'r') as f:
        test_set = [l.strip('\n').split(',') for l in f if '?' not in l]
    test = np.array(test_set).astype(int)   # test data
    print(question7(test, root))

    ''' Prune '''
    limit = 5 # just get this large so that we can get some depth
    root = None
    root = Node(feature_list[ind[0]], ind[1] + 1)
    create_tree(a, root, 0)
    root.printTree(0)
    print("Max Depth: %d" %(question6(root)))
    print(question7(test,root)) # question 9


if __name__ == '__main__':
    main()