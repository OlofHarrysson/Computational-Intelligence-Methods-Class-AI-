import random
import math
import operator
from numpy.random import choice
import copy
import time
import uuid

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


def create_random_vars(count):
    random_vars = []
    for i in range(count):
        random_vars.append(random.uniform(-500, 500))

    return random_vars


class Population:
    nbr_cost_eval = 0

    def __init__(self, nbr_indiv, training_data):
        self.nbr_indiv = nbr_indiv
        self.nbr_vars = len(training_data[0])
        self.individuals = self.init_individuals(nbr_indiv, training_data)
        self.next_generation = []

    def init_individuals(self, nbr_indiv, training_data):
        solutions = []
        for line in training_data:
            solutions.append(line.pop())
        training_sol = solutions

        individuals = []
        for i in range(nbr_indiv):
            individuals.append(Individual(training_data, training_sol))

        return individuals


    def sort(self):
        self.individuals.sort(key=operator.attrgetter('cost'))

    def print_best(self, number):
        self.sort() # TODO: Remove?
        for i in range(number):
            print(self.individuals[i].cost)





class Individual:
    def __init__(self, training_input, training_sol):
        self.variables = create_random_vars(13)
        self.training_input = training_input
        self.training_sol = training_sol
        self.cost = None
        self.calculate_cost()

    def calculate_cost(self):
        sum = 0 # sum of (h_x - y)^2
        for i, line in enumerate(self.training_input):
            line_sol = self.training_sol[i]

            x_sum = self.variables[0] # One more var than Xi in line
            for j, x in enumerate(line):
                x_sum += self.variables[j+1] * x

            diff = math.pow(x_sum - line_sol, 2)
            sum += diff


        self.cost = math.sqrt(1 / len(self.training_input) * sum)
        Population.nbr_cost_eval += 1



# =*==*=*=*=*=*=*=START=*==*=*=*=*=*=*=
path = 'forestfires.txt'
lines = read_file(path)
lines.pop(0) # Removes inputfile header
training_data = format_input_variables(lines)

pop_nbr = 100
popu = Population(1, training_data)

print("Best gen 0 ")
popu.print_best(1)

measurment_list = []
nbr_iterations = 85
min_cost = float("inf")
for x in range(nbr_iterations):
    for y in range(pop_nbr):
        popu.individuals[0].variables = create_random_vars(13)
        popu.individuals[0].calculate_cost()
        if popu.individuals[0].cost < min_cost:
            min_cost = popu.individuals[0].cost

    measurment_list.append([Population.nbr_cost_eval, min_cost])

print("******** Finished. Best ones are ********")
print(min_cost)


output_lines = ""
for measurment in measurment_list:
    output_lines += "{:s} {:s}\n".format(str(measurment[0]), str(measurment[1]))

hash = uuid.uuid4().hex
output_path = "cost_measure/random/{:s}.dat".format(hash)
write_file(output_lines, output_path)
