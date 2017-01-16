import math
import numpy as np
import uuid
import random
import sys
import network as neur_net
import copy


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



# =*==*=*=*=*=*=*=START=*==*=*=*=*=*=*=
# Training data
path = 'forestfires.txt'
lines = read_file(path)
lines.pop(0) # Removes inputfile header

network_data = format_input_variables(lines)
network_data = np.array(network_data)

X = network_data[:,0:12]
Y = network_data[:,12]

Y_copy = copy.deepcopy(Y)

# Normalize X and Y
X -= np.mean(X, axis = 0)
X /= np.std(X, axis = 0)

max_y = max(Y) # Area is > 0
Y = Y / max_y

network_data = []
for x, y in zip(X, Y):
    network_data.append((x,y))

random.shuffle(network_data) # Shuffle data to get different test_data
test_data = network_data[:50] # Test data is 50 first entries
training_data = network_data[51:] # Training data is the rest

training_X = []
training_Y = []
for line in training_data:
    training_X.append(line[0])
    training_Y.append(line[1])

test_X = []
test_Y = []
for line in test_data:
    test_X.append(line[0])
    test_Y.append(line[1])


nbr_input = 12
nbr_hidden = 10
nbr_output = 1
network = neur_net.Neural_network(nbr_input, nbr_hidden, nbr_output)


epochs = 100
mini_batch_size = 3
learn_rate = 0.2
for i in range(epochs):

    # print(Y_copy)
    a1_list, z2_list, a2_list, a3_array, z3_list = network.feed_forward(len(test_X), test_X)
    error = a3_array - test_Y
    error *= max_y

    network.SGD(training_X, training_Y, learn_rate)
    # print("epoch done")

    diff_pow_2 = np.power(error, 2)
    rows = len(error) #TODO: Check if divide before sqrt?
    sum = np.sum(diff_pow_2) / rows
    cost = math.sqrt(sum)
    print(cost)


sys.exit()


measurment_list = []
print("******** Finished ********")


output_lines = ""
for measurment in measurment_list:
    output_lines += "{:s} {:s}\n".format(str(measurment[0]), str(measurment[1]))

hash = uuid.uuid4().hex
output_path = "cost_measure/neural/{:s}.dat".format(hash)
write_file(output_lines, output_path)


