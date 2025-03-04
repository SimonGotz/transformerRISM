import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import random

trainlosses, vallosses, testlosses, configs = [], [], [], []

folder = "Model 2"
nHyperParams = 10

# Extract validation losses
for file in os.listdir(f"Results/Experiments(feb2025)/{folder}"):
    epochs = 0
    vallossesSingleRun, trainlossesSingleRun, config = [],[],[]
    val = False
    if file.split('.')[0][-5:] != 'Train':
        continue
    f = open(f"Results/Experiments(feb2025)/{folder}/{file}", 'r')
    lines = f.readlines()
    for line in lines[2:nHyperParams + 2]:
        #print(line)
        config.append(line)
    for line in lines[nHyperParams + 3:]:
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
    if file.split('.')[0][-4:] != 'Test':
        continue
    print(file)
    f = open(f"Results/Experiments(feb2025)/{folder}/{file}", 'r')
    line = f.readline().split()
    print(line[2])
    testloss = line[2]

path = "Results/Experiments(feb2025)/Model 1/IncipitSimpleRandom.txt"
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
f.write(f'Testloss: {testloss} \n')

def showExample(i):        
    print(trainlosses[0])
    print(vallosses[0])
    #print(len(testlosses[i]))
    #print(configs[i])
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
    plt.plot(len(vallosses[0]), float(testloss), label='Test loss', marker='o', color='gold')
    ax = plt.gca()
    #plt.xticks(range(1, len(vallosses)))
    ax.set_ylim([0, 0.65])
    plt.xticks(range(0, len(vallosses[0]) + 1, 2))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()

smallest = 0
secondsmallest = 0
#for i in range(len(testlosses)):
    #if testlosses[i] < testlosses[smallest]:
        #secondsmallest = smallest
        #smallest = i

showExample(smallest)