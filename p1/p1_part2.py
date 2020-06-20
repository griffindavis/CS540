import numpy as np
from os.path import dirname, join
import sys

class neuralNet:
    def __init__(self, input, actual):
        # Hyperparameters 
        self.numHiddenUnits = 392
        self.rate = 0.001
        self.totalEpochs = 20
        self.seed = 42786

        # based on parameters
        self.input = input
        self.actual = actual
        self.output = np.zeros(len(self.actual))

        self.size = actual.shape[0]
        self.layer1 = np.zeros((len(self.actual), self.numHiddenUnits + 1))

        # set up the weights
        np.random.seed(self.seed)
        print("Seed: ", self.seed)
        self.weightsToHiddenLayer = np.random.uniform(low=-1, high=1, size=(self.numHiddenUnits + 1, input.shape[1])) # 392 * 785

        self.weightsToOutputLayer = np.random.uniform(low=-1, high=1, size=(1, self.numHiddenUnits + 1)) # 1 * 393

        # things to store for questions
        self.testActivationValues = np.zeros(200)
        self.firstLayerPrediction = np.zeros(200)
        self.predicted = np.zeros(200)
        self.testSecondLayerActivation = np.zeros(200)

    '''
    Used to push input through the nnet
    '''
    def propForward(self):
        for index in range(self.size):
            row = self.input[index]
            
            self.layer1[index] = sigmoid(np.matmul(self.weightsToHiddenLayer, row))
            
            self.output[index] = sigmoid(np.matmul(self.layer1[index], self.weightsToOutputLayer[0]))

    '''
    Propegate the error back through the nnet and update the weights
    '''
    def propBack(self):
        for index in range(self.size):

            # set up some of the derivatives
            dcda = (self.output[index] - self.actual[index])
            dadz = self.output[index] * (1 - self.output[index])
            dzda1 = self.weightsToOutputLayer.T
            da1dz1 = self.layer1[index] * (1 - self.layer1[index])
            dz1dw1 = self.input[index]

            ''' update the weights '''
            self.weightsToOutputLayer[0] -= self.rate * dcda * dadz * self.layer1[index]
            self.weightsToHiddenLayer -= self.rate * dcda * dadz * dzda1 * np.expand_dims(da1dz1, axis=1) * np.expand_dims(dz1dw1.T, axis=0)

    '''
    Method to run the training
    '''
    def train(self):
        for epoch in range(self.totalEpochs):
            self.propForward()
            self.propBack()

            totalCorrect = sum((self.output > 0.5).astype(int) == self.actual)
            print("Epoch =", epoch, "Correct = {:.4%}".format(totalCorrect / len(self.actual)))

    '''
    Predicts the value of the image
    '''
    def predict(self, testSet):
        size = len(testSet)
        firstLayer = np.zeros((size, self.numHiddenUnits + 1))
        output = np.zeros(size)
        for index in range(size):
            firstLayer[index] = sigmoid(np.matmul(self.weightsToHiddenLayer, testSet[index]))
            output[index] = sigmoid(np.matmul(firstLayer[index], self.weightsToOutputLayer[0]))

        return firstLayer, output

    '''
    Runs the test file
    '''
    def test(self):
        current_dir = dirname(__file__)
        file_path = join(current_dir, "./Data/test.csv")
        rawFile = np.genfromtxt(file_path, delimiter=",", skip_header=0)
        print("Test file loaded...")
        input = rawFile / 255
        input=np.hstack((input, np.ones(input.shape[0]).reshape(-1,1)))

        firstLayer, output = self.predict(input)
        self.testActivationValues = np.copy(firstLayer)
        firstLayer[firstLayer < 0.5] = 0
        firstLayer[firstLayer >= 0.5] = 1
        self.firstLayerPrediction = np.copy(firstLayer)
        self.testSecondLayerActivation = np.copy(output)
        output[output < 0.5] = 0
        output[output >= 0.5] = 1
        self.predicted = np.copy(output)

    '''
    Outputs the answers for P1
    '''
    def outputQuestions(self):

        original_stdout = sys.stdout

        print("Writing to file...")

        with open('output.txt', 'w+') as f:
            sys.stdout = f
            '''
            Output the question answers
            '''
            '''
            # Question 1 - print feature vector of any one training image rounded to 2 decimal points
            print("Question 1")
            print(np.array2string(self.input[0], formatter={'float_kind':lambda x: "%.2f" % x}, separator=',', max_line_width=np.inf))
            print()

            # Question 2 - print logistic regression weights and biases 784 + 1 rounded to 4 decimals
            print("Question 2")
            print(np.array2string(self.weightsToHiddenLayer[0], formatter={'float_kind':lambda x: "%.4f" % x}, separator=',', max_line_width=np.inf))
            print()

            # Question 3 - print activation values on the test set (200 numbers between 0 and 1) rounded to 2 decimal places
            print("Question 3")
            print(np.array2string(self.testSecondLayerActivation, formatter={'float_kind':lambda x: "%.2f" % x}, separator=',', max_line_width=np.inf))
            print()

            # Question 4 - print predicted values on the test set (200 0 - 1)
            print("Question 4")
            print(np.array2string(self.predicted, formatter={'float_kind':lambda x: "%.0f" % x}, separator=',', max_line_width=np.inf))
            print()
            '''
            # Question 5 - print first layer weights and biaes (784 + 1 lines, each line 392 numbers to 4 decimal places)
            print("Question 5")
            for index in range(len(self.weightsToHiddenLayer.T)):
                print(','.join(map("{:.4f}".format, self.weightsToHiddenLayer.T[index])))
                #print(np.array2string(self.weightsToHiddenLayer.T[index], formatter={'float_kind':lambda x: "%.4f" % x}, separator=','))
            print()

            # Question 6 - second layer weights  (392 + 1 numbers to 4 decimal points)
            print("Question 6")
            print(np.array2string(self.weightsToOutputLayer, formatter={'float_kind':lambda x: "%.4f" % x}, separator=',', max_line_width=np.inf))
            print()

            # Question 7 - second layer activation values on test set (200 between 0 and 1, 2 decimal places)
            print("Question 7")
            print(np.array2string(self.testSecondLayerActivation, formatter={'float_kind':lambda x: "%.2f" % x}, separator=',', max_line_width=np.inf))
            print()

            # Question 8 - predicted values on the test set
            print("Question 8")
            print(np.array2string(self.predicted, formatter={'float_kind':lambda x: "%.0f" % x}, separator=',', max_line_width=np.inf))
            print()

            # Question 9 - feature vector of one test image that is labelled incorrectly (784 numbers rounded to 2 decimal places)
            print("Question 9")
            for index in range(len(self.predicted)):
                if (index < 100):
                    if self.predicted[index] == 1:
                        print(np.array2string(self.input[index], formatter={'float_kind':lambda x: "%.2f" % x}, separator=',', max_line_width=np.inf))
                else:
                    if self.predicted[index] == 0:
                        print(np.array2string(self.input[index], formatter={'float_kind':lambda x: "%.2f" % x}, separator=',', max_line_width=np.inf))
            print()
        sys.stdout = original_stdout
        print("Done!")

'''
Main function for running the program
'''
def main():
    print("Neural Net Time!")
    inputTrain, actualTrain = prep_data()
    nnet = neuralNet(inputTrain, actualTrain)
    nnet.train()

    nnet.test()

    nnet.outputQuestions()

'''
Parses the files and sets up the np arrays of data
'''
def prep_data():
    print("Loading data... Please wait... Python is slow...\n")
    xTrain, yTrain = load_data("./Data/train.csv")
    print("Loaded training data...")
    print()

    #xTest, yTest = load_data("./Data/mnist_test.csv")
    #print("Loaded testing data...")
    #print()
    
    test_labels = [0, 1]
    indices = np.where(np.isin(yTrain, test_labels))[0]
    #test_indices = np.where(np.isin(yTest, test_labels))[0]

    # gather the indeices we care about
    input_train = xTrain[indices]
    actual_train = yTrain[indices]
    #input_test = xTest[test_indices]
    #actual_test = yTest[test_indices]

    # normalize
    actual_train[actual_train == test_labels[0]] = 0
    actual_train[actual_train == test_labels[1]] = 1
    #actual_test[actual_test == test_labels[0]] = 0
    #actual_test[actual_test == test_labels[1]] = 1

    return input_train, actual_train #, input_test, actual_test

''' 
Function for activation 
'''
def sigmoid(input):
    return 1 / (1 + np.exp(-input))

'''
Helper function to load the data
'''
def load_data(filename):
    current_dir = dirname(__file__)
    file_path = join(current_dir, filename)

    raw_data = np.genfromtxt(file_path, delimiter=",", skip_header=0)
    input = raw_data[:, 1:] / 255.0
    actual = raw_data[:, 0]

    input = np.hstack((input, np.ones(input.shape[0]).reshape(-1,1)))
    
    return (input, actual)

'''
Make the program run on play
'''
if __name__ == '__main__':
    main()