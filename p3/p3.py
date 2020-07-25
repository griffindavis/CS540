'''
import everything
'''
from collections import Counter, OrderedDict
from itertools import product
from random import choices
import numpy as np
import string
import sys
import re
import math
from os.path import dirname, join


def main():
    current_dir = dirname(__file__)
    file_path = join(current_dir, "../Data/interstellar.txt")
    with open(file_path, encoding='utf-8') as f:
        data = f.read()

    # clean up the script
    data = data.lower() # make everything lowercase
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = re.sub('[^a-z]+', ' ', data) # replace bad characters
    data = ' '.join(data.split(' ')) # remove subsequent spaces

    allCharacters = ' ' + string.ascii_lowercase # get all the characters

    unigram = Counter(data)
    unigram_prob = {ch: round((unigram[ch]) / (len(data)), 4) for ch in allCharacters}

    # Question 2
    print("Question 2")
    output = ""
    for number in unigram_prob:
        if output == "":
            output = str(unigram_prob[number])
        else:
            output += "," + str(unigram_prob[number])
    print(output)

    # Question 3
    bigram_no_smoothing = ngram(2, allCharacters, data, 0)
    bigram_prob_no_smoothing  = {c: bigram_no_smoothing[c] / unigram[c[0]] for c in bigram_no_smoothing}
    print("Question 3")
    outputProbabilityTable(bigram_prob_no_smoothing)

    bigram = ngram(2, allCharacters, data)  # c(ab)
    bigram_prob = {c: bigram[c] / unigram[c[0]] for c in bigram}  # p(b|a)
    print("Question 4")
    outputProbabilityTable(bigram_prob_no_smoothing)

    trigram = ngram(3, allCharacters, data)
    trigram_prob = {c: (trigram[c]) / (bigram[c[:2]]) for c in trigram}
    sentences = []
    temp = ""
    for c in allCharacters:
        if c == ' ':
            continue
        temp=gen_sen(c, 1000, bigram, bigram_prob_no_smoothing, trigram_prob, allCharacters)
        sentences.append(temp)
        print(temp)

    file_path = join(current_dir, "../Data/script.txt")
    with open(file_path, encoding='utf-8') as f:
        prof = f.read() 

    uni_list = [unigram_prob[character] for character in allCharacters]

    dict2 = Counter(prof)
    likelihood = [dict2[c] / len(prof) for c in allCharacters]
    # Question 7
    print("Question 7")
    output = ""
    for number in likelihood:
        if output == "":
            output = str(number)
        else:
            output += "," + str(number)
    print(output)

    # post - Pr(Document | letter)
    post_prof = [round(likelihood[i] / (likelihood[i] + uni_list[i]), 4) for i in range(27)]

    # Question 8
    print("Question 8")
    output = ""
    for number in post_prof:
        if output == "":
            output = str(number)
        else:
            output += "," + str(number)
    print(output)
    post_my = [1 - post_prof[i] for i in range(27)]

    output = ""
    for sentence in sentences:
        prob_prof=sum(post_prof[allCharacters.find(c)] for c in sentence)
        prob_my=sum(post_my[allCharacters.find(c)] for c in sentence)
        if prob_my > prob_prof:
            output += str(0) + ","
        else:
            output += str(1) + ","
    print(output)


'''
Outputs the probability table with rounding

Bigram
'''
def outputProbabilityTable(table):
    output = ""
    tempValue = 0
    counter = 0
    for key in table:
        counter += 1
        if output == "":
            tempValue = round(table[key], 4)
            if tempValue == 0:
                tempValue = 0.0001
            output = str(tempValue)
        else:
            tempValue = round(table[key], 4)
            if tempValue == 0:
                tempValue = 0.0001
            output += "," + str(tempValue)

        if counter == 27:
            print(output)
            counter = 0
            output = ""

'''
Creates the ngram dictionary
'''
def ngram(n, allCharacters, data, smoothing=1):
    # all possible n-grams
    d = dict.fromkeys([''.join(i) for i in product(allCharacters, repeat=n)],0) # creates a dictionary of 0's 
    # update counts
    d.update(Counter([''.join(j) for j in zip(*[data[i:] for i in range(n)])]))

    # laplace smoothing
    if smoothing ==1 :
        for val in d:
            d[val] += 1
    return d

'''
Generates the bigram choice
'''
def gen_bi(c, bigram_prob, allCharacters):
    w = [bigram_prob[c + i] for i in allCharacters]
    return choices(allCharacters, weights=w)[0]
    
'''
Generates the trigram choice
'''
def gen_tri(ab, trigram_prob, allCharacters):
    w_tri = [trigram_prob[ab + i] for i in allCharacters]
    return choices(allCharacters, weights=w_tri)[0]   

'''
Generates the sentence
'''
def gen_sen(c, num, bigram, bigram_prob, trigram_prob, allCharacters):
    res = c + gen_bi(c, bigram_prob, allCharacters)
    for i in range(num - 2):
        if bigram[res[-2:]] == 0:
            t = gen_bi(res[-1], bigram_prob, allCharacters)
        else:
            t = gen_tri(res[-2:], trigram_prob, allCharacters)
        res += t
    return res
   
if __name__ == '__main__':
    main()