import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys


def read_data(path):
    file = open(path, 'r')
    return file.read().splitlines()

# =*==*=*=*=*=*=*=START=*==*=*=*=*=*=*=

# Genetic
path = 'cost_measure/genetic/*.dat'
files = glob.glob(path)


measurment_length = None
xs = []
y_measurments = []
for name in files:
    file = open(name, 'r')
    measurment = file.read().split("\n")
    measurment.pop() # Remove empty last line
    measurment_length = len(measurment)

    ys = []
    for meas in measurment:
        meas = meas.split(" ")
        xs.append(float(meas[0]))
        ys.append(float(meas[1]))
    y_measurments.append(ys)

xs = xs[:measurment_length]

y_grouped_by_x = []
for i in range(len(y_measurments[0])):
    y_temp = []
    for j in range(len(y_measurments)):
        y_temp.append(y_measurments[j][i])
    y_grouped_by_x.append(y_temp)


avg_ys = []
best_ys = []
worst_ys = []

for ys in y_grouped_by_x:
    best = float("inf")
    worst = 0
    sum = 0
    for y in ys:
        sum += y
        if y < best:
            best = y
        if y > worst:
            worst = y

    avg = sum / len(ys)
    avg_ys.append(avg)
    best_ys.append(best)
    worst_ys.append(worst)


# print(avg_ys)
# print(best_ys)
# print(worst_ys)
# print(xs)


# sys.exit(1)

# Random Search
path = 'cost_measure/random/*.dat'
files = glob.glob(path)


r_measurment_length = None
r_xs = []
r_y_measurments = []
for name in files:
    file = open(name, 'r')
    measurment = file.read().split("\n")
    measurment.pop() # Remove empty last line
    r_measurment_length = len(measurment)

    ys = []
    for meas in measurment:
        meas = meas.split(" ")
        r_xs.append(float(meas[0]))
        ys.append(float(meas[1]))
    r_y_measurments.append(ys)

r_xs = r_xs[:r_measurment_length]

r_y_grouped_by_x = []
for i in range(len(r_y_measurments[0])):
    y_temp = []
    for j in range(len(r_y_measurments)):
        y_temp.append(r_y_measurments[j][i])
    r_y_grouped_by_x.append(y_temp)


r_avg_ys = []
r_best_ys = []
r_worst_ys = []

for ys in r_y_grouped_by_x:
    best = float("inf")
    worst = 0
    sum = 0
    for y in ys:
        sum += y
        if y < best:
            best = y
        if y > worst:
            worst = y

    avg = sum / len(ys)
    r_avg_ys.append(avg)
    r_best_ys.append(best)
    r_worst_ys.append(worst)














# Average
fig = plt.figure(1)
plt.ylabel('Cost')
plt.xlabel('Cost Evaluations')

plt.scatter(xs, avg_ys, color='red')
red_patch = mpatches.Patch(color='red', label='Genetic')

plt.scatter(r_xs, r_avg_ys, color='blue')
blue_patch = mpatches.Patch(color='blue', label='Random')

plt.legend(handles=[red_patch, blue_patch])

plt.show()
plt.close(fig)


# Best
fig = plt.figure(1)
plt.ylabel('Cost')
plt.xlabel('Cost Evaluations')

plt.scatter(xs, best_ys, color='red')
red_patch = mpatches.Patch(color='red', label='Genetic')

plt.scatter(r_xs, r_best_ys, color='blue')
blue_patch = mpatches.Patch(color='blue', label='Random')

plt.legend(handles=[red_patch, blue_patch])

plt.show()
plt.close(fig)



# Worst
fig = plt.figure(1)
plt.ylabel('Cost')
plt.xlabel('Cost Evaluations')

plt.scatter(xs, worst_ys, color='red')
red_patch = mpatches.Patch(color='red', label='Genetic')

plt.scatter(r_xs, r_worst_ys, color='blue')
blue_patch = mpatches.Patch(color='blue', label='Random')

plt.legend(handles=[red_patch, blue_patch])


plt.show()
plt.close(fig)
