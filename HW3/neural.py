import math
import numpy as np
import uuid
import random
import sys


def read_file(filename):
    file = open(path, 'r')
    return file.read().splitlines()


def write_file(lines_output, path):
    file = open(path, 'w')
    file.write(lines_output)


def format_input_variables(lines):
    formated_lines = []
    for line in lines:
        formated_line = line.split(" ")
        formated_lines.append(list(map(float, formated_line))) # Converts from string to int
    return formated_lines


def create_random_vars(count, min, max):
    random_vars = []
    for i in range(count):
        random_vars.append(random.uniform(min, max))

    return random_vars

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

class Neuron():
    def __init__(self, nbr_inputs):
        self.nbr_inputs = nbr_inputs
        self.set_weights([random.uniform(-0.1,0.1) for x in range(0,nbr_inputs)])
        self.bias = 0.01
        self.bias_weight = random.uniform(-0.1,0.1)

    def sum(self, inputs):
        output = np.dot(inputs, self.weights)
        return output

    def set_weights(self, weights):
        self.weights = weights

    def __str__(self):
        return "My weights are cooooolio"

class Neuron_layer():
    def __init__(self, nbr_neurons, nbr_inputs):
        self.neurons = [Neuron(nbr_inputs) for x in range(0, nbr_neurons)]


class Neural_network():
    def __init__(self, nbr_input, nbr_hidden):
        # TODO: Create network function with layers attr
        self.input_layer = Neuron_layer(nbr_input, nbr_input)
        self.hidden_layer = Neuron_layer(nbr_hidden, nbr_input)
        self.output_layer = Neuron_layer(1, nbr_hidden)

    def doitr(self, X):

        h_layer_output = []
        for neur in self.hidden_layer.neurons:
            neur_sum = neur.sum(X)
            neur_sum = [ele + neur.bias * neur.bias_weight for ele in neur_sum]
            output = [sigmoid(x) for x in neur_sum]
            h_layer_output.append(output)

        h_layer_output = np.transpose(h_layer_output)
        # print(h_layer_output)

        comp_sol = None
        for neur in self.output_layer.neurons:
            comp_sol = neur.sum(h_layer_output) # TODO bias here?

        multi = 8
        comp_sol = [ele * multi for ele in output]
        # print(comp_sol)
        return comp_sol

    def compute_err(self, comp_sol, sol):
        print(comp_sol)
        print(sol)

        err = sol - comp_sol
        return err


# =*==*=*=*=*=*=*=START=*==*=*=*=*=*=*=
path = 'forestfires.txt'
lines = read_file(path)
lines.pop(0) # Removes inputfile header

training_data = format_input_variables(lines)
training_data = np.array(training_data)
X = training_data[:,0:12]
Y = training_data[:,12]

X -= np.mean(X, axis = 0)
X /= np.std(X, axis = 0)

nbr_input = 12
nbr_hidden = 3
network = Neural_network(nbr_input, nbr_hidden)

comp_sol = network.doitr(X)
err = network.compute_err(comp_sol, Y)



# Nbr input nodes = 12
# Output node should create area
# Every node connects to every node in next layer. What about first and last?
# Sigmund fucntion
# Number of neurons in hidden layer.
# Add BIAS
# Linear unbounded output activition function






measurment_list = []
print("******** Finished ********")


output_lines = ""
for measurment in measurment_list:
    output_lines += "{:s} {:s}\n".format(str(measurment[0]), str(measurment[1]))

hash = uuid.uuid4().hex
output_path = "cost_measure/neural/{:s}.dat".format(hash)
write_file(output_lines, output_path)


