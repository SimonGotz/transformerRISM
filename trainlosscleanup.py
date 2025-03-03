import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import random

trainlosses, vallosses, testlosses, configs = [], [], [], []

folder = "Tuning Model 3"
nHyperParams = 10

# Extract validation losses
for file in os.listdir(f"Results/Experiments(feb2025)/{folder}"):
    epochs = 0
    vallossesSingleRun, trainlossesSingleRun, config = [],[],[]
    val = False
    if file.split('.')[1] == 'pt' or file.split('.')[0][-4:] == 'Test':
        continue
    f = open(f"Results/Experiments(feb2025)/{folder}/{file}", 'r')
    lines = f.readlines()
    for line in lines[1:nHyperParams + 1]:
        #print(line)
        config.append(line)
    for line in lines[nHyperParams + 2:]:
        line = line.strip()
        if line == "Val losses":
            val = True
            continue
        if not val:
            trainlossesSingleRun.append(float(line))
            continue
        else:
            vallossesSingleRun.append(float(line))
    configs.append(config)
    vallosses.append(vallossesSingleRun)
    trainlosses.append(trainlossesSingleRun)

# Extract test losses
for file in os.listdir(f"Results/Experiments(feb2025)/{folder}"):
    if file.split('.')[1] == 'pt' or file.split('.')[0][-5:] == 'Train':
        continue
    f = open(f"Results/Experiments(feb2025)/{folder}/{file}", 'r')
    line = f.readline().split()
    testlosses.append(float(line[2]))

path = "Results/TuningComplexIncipitsRandom.txt"
f = open(path, 'w')
for i in range(len(trainlosses)):
    f.write(f"Run {i}, config: ")
    for config in configs[i]:
        f.write(f"{config.strip()} ")
    f.write('Train losses: ')
    for loss in trainlosses[i]:
        f.write(f"{loss} ")
    f.write(f"Val losses: ")
    for loss in vallosses[i]:
        f.write(f"{loss} ")
    f.write(f'Testloss: {testlosses[i]} \n')

def showExample(i):        
    print(len(trainlosses[i]))
    print(len(vallosses[i]))
    #print(len(testlosses[i]))
    print(testlosses[i])
    print(configs[i])
    x_train = np.arange(0, len(trainlosses[i]))
    y_train = np.array(trainlosses[i])
    x_val = np.arange(0, len(vallosses[i]))
    y_val = np.array(vallosses[i])
    spl_val = make_interp_spline(x_val,y_val)
    spl_train = make_interp_spline(x_train,y_train)
    x_val = np.linspace(x_val.min(), x_val.max(), 100)
    x_train = np.linspace(x_train.min(), x_train.max(), 100)
    y_val = spl_val(x_val)
    y_train = spl_train(x_train)
    plt.plot(x_train, y_train, label="Train")
    plt.plot(x_val, y_val, label='Validation')
    plt.legend()
    plt.show()

smallest = 0
secondsmallest = 0
for i in range(len(testlosses)):
    if testlosses[i] < testlosses[smallest]:
        secondsmallest = smallest
        smallest = i

showExample(smallest)