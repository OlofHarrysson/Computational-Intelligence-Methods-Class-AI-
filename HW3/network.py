import math
import numpy as np
import uuid
import random
import sys
from itertools import chain

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# def sigmoid(x):
#     if x >= 0:
#         z = math.exp(-x)
#         return 1 / (1 + z)
#     else:
#         z = math.exp(x)
#         return z / (1 + z)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def least_square(X, Y):
    # X = np.matrix(self.training_input)
    # Y = np.matrix(self.training_sol)
    rows = len(X)
    X = np.c_[np.ones(rows), X] # Add col of ones in the beginning

    x_sum = np.dot(X, self.variables) # x0*var0 + x1*var1...
    diff = x_sum - Y
    diff_pow_2 = np.power(diff, 2)

    sum = np.sum(diff_pow_2) / rows
    cost = math.sqrt(sum)

class Neuron_layer():
    def __init__(self, nbr_neurons, nbr_inputs):
        epsi = 0.12 # Found value in Coursea machine learning course exersise 4
        self.weights = np.random.rand(nbr_neurons, nbr_inputs) *2*epsi-epsi # Matrix, nbr_neur rows, nbr_inputs cols
        self.bias = np.random.rand(nbr_neurons, 1) *2*epsi-epsi


class Neural_network():
    def __init__(self, nbr_input, nbr_hidden, nbr_output):
        self.nbr_input = nbr_input
        self.hidden_layer = Neuron_layer(nbr_hidden, nbr_input)
        self.output_layer = Neuron_layer(nbr_output, nbr_hidden)

    def SGD(self, X, Y, learn_rate):
        # X = training_data[:,0]
        # Y = training_data[:,-1]

        # for a batch of X:
        #     for a X:
        #         find z2,a2,  z3,a3
        #         save em

        #         calc errors
        batch_size= len(X)


        z2_list = []
        a2_list = []
        a3_array = np.array([])
        # Forward propagation
        for i in range(batch_size):
            a1 = X[i]
            a1 = np.reshape(a1, (self.nbr_input, 1))
            # Hidden layer
            z2 = np.dot(self.hidden_layer.weights, a1) + self.hidden_layer.bias
            a2 = sigmoid(z2)
            a2 = list(chain(*a2)) # Flatten list

            # print(a2)

            # Output layer
            z3 = np.dot(self.output_layer.weights, a2) + self.output_layer.bias
            # print(z3)
            a3 = sigmoid(z3)
            a3 *= 1000 # Continious range in [0, 1000]
            # print(a3)
            z2_list.append(z2)
            a2_list.append(a2)
            a3_array = np.append(a3_array, a3)






        error_L_array = a3_array - Y


        # print(self.output_layer.weights)
        # print(error_L_array)
        # print(z2_list)
        # print(a2_list)
        # sys.exit(1)



        error_l2_list = []
        # Backwards propagation
        for i in range(batch_size): # TODO: calcs correct?
            temp = np.dot(self.output_layer.weights, error_L_array[-1])
            temp = temp[0]
            z2 = z2_list[-1]
            z2 = np.reshape(z2, (1, 3))[0]
            error_l = temp * sigmoid_prime(z2)
            error_l2_list.append(error_l)

        # Delta1
        error_l2_list = np.transpose(error_l2_list)
        a1 = X
        delta1 = np.dot(error_l2_list, a1) # TODO: Add extra 1x3 col vector for bias?

        # Delta2
        # print(error_L_array)
        error_L_array = np.reshape(error_L_array, (6,1))
        # print(np.shape(error_L_array))
        # print(a2_list)
        # print(np.shape(a2_list))
        delta2 = np.dot(np.transpose(error_L_array), a2_list)

        # temp = np.dot(error_L_array, self.hidden_layer.bias)
        # print(temp)

        # theta1_grad = delta1 / batch_size
        # theta2_grad = delta2 / batch_size

        delta1_sum = np.sum(delta1, axis=0)
        delta2_sum = np.sum(delta2, axis=0)

        error_l2_sum = np.sum(error_l2_list, axis=1)
        error_L_sum = np.sum(error_L_array, axis=0)

        error_l2_sum = np.reshape(error_l2_sum, (len(error_l2_sum), 1))
        error_L_sum = np.reshape(error_L_sum, (len(error_L_sum), 1))

        # print(error_l2_sum)
        # print(error_L_sum)


        self.hidden_layer.weights -= learn_rate / batch_size * delta1_sum
        self.output_layer.weights -= learn_rate / batch_size * delta2_sum

        self.hidden_layer.bias -= learn_rate / batch_size * error_l2_sum
        self.output_layer.bias -= learn_rate / batch_size * error_L_sum



    def print_y(self):
        print(self.hidden_layer.bias)




# Nbr input nodes = 12
# Output node should create area
# Number of neurons in hidden layer.
# Linear unbounded output activition function
# Multiplier in outmutlayer or skip activition function?
# Bias vector for each layer?
# Cost function
# stochastic gradient descent, choose random x number of inputs every itr to train
# init weights and bias to chapter 1 of book
# Gradient checking




