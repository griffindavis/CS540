import numpy as np
from os.path import dirname, join
import random, math, sys

def main():
    filename = "./Data/train.csv"

    # get the input matrix and y vector
    input_data, actual_data = parse_csv(filename)
    #test_input, test_actual = parse_csv("./Data/test.csv")

    print("Data loaded!")

    labels=[0, 1] # can be used if we want to try other values

    # isin returns a boolean array
    # where returns array of indices where the condition is satisfied
    indices = np.where(np.isin(actual_data, labels))[0]
    #test_indices = np.where(np.isin(test_actual, labels))[0]

    input = input_data[indices] # new array of x values where they correspond to either 0 or 1
    actual = actual_data[indices] # new array of y values either 0 or 1

    #t_input = test_input[test_indices]
    #t_actual = test_actual[test_indices]

    # normalize the values
    actual[actual == labels[0]] = 0
    actual[actual == labels[1]] = 1

    # normalize the test values
    #t_actual[t_actual == labels[0]] = 0
    #t_actual[t_actual == labels[1]] = 1

    max_epochs = 100000
    num_features = input.shape[1] # shape gets the dimensions of the nd array
    #num_sets = input.share[1]
    alpha = 0.01

    # these could be defined as globals
    large_number = 1e8
    epsilon = 1e-6    
    threshold = 1e-4

    weights = np.random.rand(num_features) # array of weights
    bias = np.random.rand()

    cost_array = np.zeros(max_epochs) # initialize to zero

    for epoch in range(max_epochs):

        # matmul - matrix product of two arrays
        # transpose - I mean, transpose?
        sigmoid = 1 / (1 + np.exp(-(np.matmul(weights, np.transpose(input)) + bias)))

        weights -= alpha * np.matmul(sigmoid - actual, input) # update the weights based on the prediction

        bias -= alpha * (sigmoid - actual).sum() # update the bias

        cost = np.zeros(len(actual)) # initialize

        # how to avoid problems with logorithms 
        index = (actual == 0) & (sigmoid > 1 - threshold) | (actual == 1) & (sigmoid < threshold)
        cost[index] = large_number

        sigmoid[sigmoid < threshold] = threshold
        sigmoid[sigmoid > 1 - threshold] = threshold

        inverse_index = np.invert(index)
        cost[inverse_index] = -actual[inverse_index] * np.log(sigmoid[inverse_index]) - (1 - actual[inverse_index]) * np.log(1 - sigmoid[inverse_index])
        cost_array[epoch] = cost.sum()

        if epoch + 1 % 10 == 0:
            print('epoch = ', epoch + 1, 'cost = ', cost_array[epoch])

        if epoch > 0 and abs(cost_array[epoch - 1] - cost_array[epoch]) < epsilon:
            break


    # time to test
    current_dir = dirname(__file__)
    file_path = join(current_dir, "./Data/test.csv")
    test = np.genfromtxt(file_path, delimiter = ",", skip_header=0)
    input = test / 255.0
    predictions = np.zeros(200)
    for i in range(len(input)):
        predictions[i] = 1 / (1 + np.exp(-(np.dot(weights, np.transpose(input[i])) + bias)))
        #print("Line: ", i, " is ", pred.round())
    original_stdout = sys.stdout
    with open('output_part1.txt', 'w+') as f:
        sys.stdout = f
        print("Question 2")
        print(','.join(map("{:.4f}".format, weights)), ',', '%.4f' % bias)

        print("Question 3")
        print(','.join(map("{:.2f}".format, predictions)))

        print("Question 4")
        predictions[predictions < 0.5] = 0
        predictions[predictions >= 0.5] = 1
        print(','.join(map("{:.0f}".format, predictions)))

        sys.stdout = original_stdout
        print("Done!")

'''
Parses the CSV file and return the data
'''
def parse_csv(filename):
    current_dir = dirname(__file__)
    file_path = join(current_dir, filename)

    raw_data = np.genfromtxt(file_path, delimiter=',')
    input = raw_data[:, 1:] / 255.0
    actual = raw_data[:, 0]

    return (input, actual)

'''
Run the program on play
'''
if __name__ == "__main__":
    main() 