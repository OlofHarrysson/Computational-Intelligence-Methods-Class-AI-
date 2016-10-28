import random
import math
import operator
from numpy.random import choice
import copy
import time

def measure_calc_cost(popu, nbr_loops):
    for i in range(nbr_loops):
        for indiv in popu.individuals:
            indiv.calculate_cost()


def read_file(filename):
    file = open(path, 'r')
    return file.read().splitlines()


def format_input_variables(lines):
    formated_lines = []
    for line in lines:
        formated_line = line.split(" ")
        formated_lines.append(list(map(float, formated_line))) # Converts from string to int
    return formated_lines


def create_random_vars(count):
    random_vars = []
    for i in range(count):
        random_vars.append(random.uniform(-1000, 1000))

    return random_vars


class Population:
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
        self.sort()
        for i in range(number):
            print(self.individuals[i].cost)

    def do_selection(self):
        strongest = self.individuals.pop(0)

        population_cost = 0
        for indiv in self.individuals:
            population_cost += indiv.cost

        surv_inter = 0
        weights = []
        for indiv in self.individuals:
            surv_chance = population_cost / indiv.cost
            weights.append(surv_chance)

        norm_weights = [float(i)/sum(weights) for i in weights]

        self.next_generation = list(choice(self.individuals, self.nbr_indiv / 2 - 1, p=norm_weights, replace=False))
        self.next_generation.append(strongest) # Elitism: Strongest indivivual survives

        self.individuals = copy.deepcopy(self.next_generation) # Creates children


    def do_crossover(self):
        # Change two random creates genes
        # Remove those and repeat a given time
        random.shuffle(self.individuals)
        indivs = self.individuals

        for i in range(int(len(indivs) / 2)):
            start_i = random.randrange(self.nbr_vars)
            length = random.randrange(1, self.nbr_vars - 1)

            for j in range(length):
                temp_var = indivs[i].variables[start_i]
                indivs[i].variables[start_i] = indivs[i+1].variables[start_i]
                indivs[i+1].variables[start_i] = temp_var

                start_i += 1
                start_i = start_i % (self.nbr_vars)


    def do_mutation(self):
        for indiv in self.individuals:
            swap_i = random.randrange(self.nbr_vars)
            multiplier = random.uniform(-1.5, 1.5)
            indiv.variables[swap_i] = indiv.variables[swap_i] * multiplier


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

        self.cost = math.sqrt(sum / len(self.training_input))



# =*==*=*=*=*=*=*=START=*==*=*=*=*=*=*=
path = 'forestfires.txt'
lines = read_file(path)
lines.pop(0) # Removes inputfile header
training_data = format_input_variables(lines)


popu = Population(100, training_data)

popu.print_best(5)


for x in range(100):
    popu.sort()

    popu.do_selection()
    popu.do_crossover()
    popu.do_mutation()

    for indiv in popu.individuals:
            indiv.calculate_cost()

    popu.next_generation.extend(popu.individuals)

    popu.individuals = popu.next_generation


print("******** Finished. Best ones are ********")
print(popu.print_best(5))


##### TIME TEST #####

# start_time = time.clock()

# measure_calc_cost(popu, 10) # Seems to be slow
# # measure_sort(popu, 10) # Seems to be very fast

# stop_time = time.clock()
# elap_time = stop_time - start_time
# print(elap_time)


