# coding=utf-8
# implementation of a multi-layer neural network.
# One Input Layer - 17 features including bias,
# one hidden layer - dynamic number of hidden layer + bias,
# and one output layer - 26 different classification

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


class Neural(object):
    def __init__(self, eta=0.3, no_hidden=4, momentum=0.3):
        # Hyper-parameters
        self.eta = eta
        self.inputLayer = 16
        self.hiddenLayer = no_hidden
        self.outputLayer = 26
        self.momentum = momentum

        self.hidden_weights = []
        self.delta_hidden = {}
        self.input_weights = []
        self.delta_input = []

    def lookup(self,x):
        alpha = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}
        return alpha[x]

    def init_input_to_hidden_w(self):
        # In this example, we have (16 + 1)n weights from input layer to hidden layer. Plus one because of the bias
        input_weights = np.random.uniform(low=-0.25, high=0.25, size=(self.inputLayer + 1, self.hiddenLayer))
        return input_weights

    def init_hidden_to_output_w(self):
        # In this example we have (26n + 26) weights from hidden to output layer
        # plus one because of the bias
        hidden_weights = np.random.uniform(low=-0.25, high=0.25, size=(self.hiddenLayer + 1, self.outputLayer))
        return hidden_weights

    def sigmoid_activation(self,x):
        # formula = 1/1+e^-x
        return 1 / (1 + np.exp(-x))

    def back_prop(self,output,i_weight,h_weight,target,h_output,tr_eg):
        """
        :param output: output values. should be a list of 26, to see which output unit/alphabet fires
        :param i_weight: weights from input layer to hidden layer
        :param h_weight: weights from hidden to output layer
        :param target: the expected value of the training example, to be used to calculate error term.
        :return: new set of weights
        :h_output: activation values at the hidden level
        :tr_eg: training example. usually an arrray of 16 + 1 values, to be used for weight updates
        """

        # Error Term calculation
        output_error_terms = []
        hidden_error_terms = []
        # Error term for each output
        for i in range(len(output)):            # iterate through the array output
            if self.lookup(i) == target:        # t at our target output should be 0.9 and 0.1 everywhere else
                t = 0.9                         # target function of correct classification
            else:
                t = 0.1                         # target function for all incorrect classification
            e_term = output[i] * (1 - output[i]) * (t - output[i])     # error term for each output
            output_error_terms.append(e_term)                       # computes output error terms and stores in an array

        # Error term for each hidden unit
        for a in range(len(h_output)):                                  # loop through all hidden unit
            temp = h_output[a] * (1 - h_output[a]) * np.dot(h_weight[a],output_error_terms)
            hidden_error_terms.append(temp)

        # weight updates using gradient descent and momentum

        # First: update all weights from hidden to output layer
        # new weight = previous weight + (learningRate * outputErrorTerm * sigmoid activations) + momentum * previousDelta
        previous_deltas = self.delta_hidden
        deltas = {}     # store all deltas dynamically. 5 hidden layer(+ bias) means 6 keys. e.g 0:[a list of 26 delta values representing each weight]
        temp_arr = []
        for each in range(len(h_output)):
            for a in range(len(h_weight[each])):
                if len(previous_deltas) == 0:    # if first time training, no previous delta
                    each_delta = self.eta * output_error_terms[a] * h_output[each]
                else:
                    each_delta = (self.eta * output_error_terms[a] * h_output[each]) + (self.momentum * previous_deltas[each][a])
                h_weight[each][a] += each_delta                          # weight update
                temp_arr.append(each_delta)
            deltas[each] = temp_arr
            temp_arr = []
        self.delta_hidden = deltas
        self.hidden_weights = h_weight
        # print "All hidden to output weights updated!"

        # Second: Update all weights from input to hidden layer
        # Formulae: new weight = previous weight + (learningRate * HiddenErrorTerm * input) + momentum * previousDelta
        i_previous_deltas = self.delta_input
        i_deltas = {}
        i_temp_arr = []

        new = np.delete(hidden_error_terms, 0)          # since no connection to the bias from the input layer.
        for b in range(len(tr_eg)):                 # 17 input each with its unique noOfHidden layer weights
            for bb in range(len(i_weight[b])):      # starting from 1 because the bias at the hidden layer level is not connected downwards
                if len(i_previous_deltas) == 0:
                    i_each_delta = (self.eta * new[bb] * tr_eg[b])  # if first time training, no previous delta
                else:
                    i_each_delta = (self.eta * new[bb] * tr_eg[b]) + (self.momentum * i_previous_deltas[b][bb])
                i_weight[b][bb] += i_each_delta
                i_temp_arr.append(i_each_delta)
                i_each_delta = 0
            i_deltas[b] = i_temp_arr
            i_temp_arr = []
        self.delta_input = i_deltas
        self.input_weights = i_weight
        # print "All Weights Updated!"

    def feed_forward(self,input):			# Function responsible for forward propagation
        """
        :param x: input parameter
        :return: acc of training and test
        """
        # check if input_weight is empty. This checks for first training time
        if len(self.input_weights) == 0:
            self.input_weights = self.init_input_to_hidden_w()
        if len(self.hidden_weights) == 0:
            self.hidden_weights = self.init_hidden_to_output_w()

        # forward propagation of dot product to hidden layer
        dot1 = np.dot(input, self.input_weights)       # dot product from input to hidden-layer (17L,) * (17,n) where n = number of hidden neurons
        new_dot1 = self.sigmoid_activation(dot1)       # computes sigmoid activation on the the previous result. (nL,)
        new = np.insert(new_dot1, 0, 1)                # insert +1 to represent bias input.
        dot2 = np.dot(new, self.hidden_weights)        # final dot product from hidden to output layer (n+1L,) * (n+1L,26)
        res = self.sigmoid_activation(dot2)            # classification result. An array. (26L,)

        return res, self.input_weights, self.hidden_weights, new

    def train(self, epoch):			# Function responsible for training the neural network and calculating training and test acc after each epoch

        # Arranges training example as a 10000 by 16 + 1 as bias matrix
        training_matrix = np.ones(shape=(10000, 17), dtype=None)
        training_set = genfromtxt(r'C:\Users\Iyanu\Desktop\Python\ML\data\2\training-set.data', delimiter=',', dtype=None)
        # np.random.shuffle(training_set)
        for r in range(len(training_set)):
            training_matrix[r][1:17] = list(training_set[r])[1:17]

        # Arranges test example as a 10000 by 16 matrix
        test_matrix = np.ones(shape=(10000,17),dtype=None)
        test_set = genfromtxt(r'C:\Users\Iyanu\Desktop\Python\ML\data\2\test-data.data', delimiter=',', dtype=None)
        for rr in range(len(test_set)):
            test_matrix[rr][1:17] = list(test_set[rr])[1:17]

        training_acc = []		# training accuracy array
        test_acc = []			# Test accuracy array
        counts = 0
        while counts < epoch:
            # Training the Neural Network
            for i in range(len(training_set)):
                classification, input_weights, hidden_weights, hidden_activations = self.feed_forward(training_matrix[i])
                if training_set[i][0] != self.lookup(np.argmax(classification)):
                    self.back_prop(classification,input_weights,hidden_weights,training_set[i][0],hidden_activations,training_matrix[i])

            # Calculating Training Acc after one epoch
            counter = 0
            for ii in range(len(training_set)):
                tr_classification, tr_input_weights, tr_hidden_weights, tr_hidden_activations = self.feed_forward(training_matrix[ii])
                if training_set[ii][0] == self.lookup(np.argmax(tr_classification)):
                    counter += 1
            acc = float(counter)/len(training_set)              # training accuracy
            print "Training Accuracy: " + str(acc)

            # Calculating Test Acc after one epoch
            counter2 = 0
            for iii in range(len(test_set)):
                t_classification, t_input_weights, t_hidden_weights, t_hidden_activations = self.feed_forward(test_matrix[iii])
                if test_set[iii][0] == self.lookup(np.argmax(t_classification)):
                    counter2 += 1
            acc2 = float(counter2)/len(test_set)               # Test accuracy
            print "Test Accuracy: " + str(acc2) +"\n"

            training_acc.append(acc)
            test_acc.append(acc2)

            counts += 1										# epoch increment

        return training_acc, test_acc

if __name__ == "__main__":

    epoch = 200

    # creating class instances for
    n = Neural(eta=0.3, no_hidden=2, momentum=0.3)
    n2 = Neural(eta=0.3, no_hidden=8, momentum=0.3)
    tr,t = n.train(epoch)
    tr_2,t_2 = n2.train(epoch)

    # Plotting the graph.
    plt.ylabel('Accuracy')
    plt.xlabel('No of Epoch')
    plt.plot(tr, label='Training(n=2)')
    plt.plot(t, label='Test(n=2)')
    plt.plot(tr_2, label='Training(n=8)')
    plt.plot(t_2, label='Test(n=8)')
    plt.ylim(0,1.0)
    plt.title("Experiment 1, using eta=0.3, momentum=0.3, hidden_units=2&8")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
    plt.show()
