import math
import numpy as np
import uuid
import random
import sys
import network as neur_net


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

training_data = format_input_variables(lines)
training_data = np.array(training_data)
X = training_data[:,0:12]
Y = training_data[:,12]
X_copy = X
Y_copy = Y

X -= np.mean(X, axis = 0)
X /= np.std(X, axis = 0)

max_y = max(Y)

Y -= np.mean(Y, axis = 0)
Y /= np.std(Y, axis = 0)

# training_data = []
# for x, y in zip(X,Y):
#     training_data.append([x, y])
# training_data = np.array(training_data)




nbr_input = 12
nbr_hidden = 3
nbr_output = 1
network = neur_net.Neural_network(nbr_input, nbr_hidden, nbr_output)


epochs = 10
mini_batch_size = 3
learn_rate = 3.0
# network.SGD(training_data, epochs, mini_batch_size, learn_rate)
for i in range(epochs):
    network.SGD(X, Y, learn_rate)
    # print("epoch done")



    output = network.get_err()
    output *= max_y
    diff = output - Y_copy

    diff_pow_2 = np.power(diff, 2)

    rows = len(diff)
    sum = np.sum(diff_pow_2) / rows
    cost = math.sqrt(sum)
    print(cost)



sys.exit()
output = sum(output) / float(len(output))
print(output)




measurment_list = []
print("******** Finished ********")


output_lines = ""
for measurment in measurment_list:
    output_lines += "{:s} {:s}\n".format(str(measurment[0]), str(measurment[1]))

hash = uuid.uuid4().hex
output_path = "cost_measure/neural/{:s}.dat".format(hash)
write_file(output_lines, output_path)


