import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

trainlosses, vallosses, testlosses = [], [], []

l = os.listdir("../Results/Experiments(feb2025)")
l = sorted(l)
print('ITuneSimple_1Train.txt' in l)
while True: continue

for file in os.listdir("../Results/Experiments(feb2025)"):
    epochs = 0
    losses = []
    val = False
    #trainlosses, vallosses = [], []
    if file.split('.')[1] == 'pt' or file.split('.')[0][-4:] == 'Test':
        continue
    f = open("../Results/Experiments(feb2025)/" + file, 'r')
    for line in f.readlines()[6:]:
        line = line.strip()
        #if not val and line != "Val losses":
            #trainlosses.append(float(line))
        if line == "Val losses":
            val = True
            continue
        if not val:
            continue
        losses.append(float(line))
    vallosses.append(losses)

for file in os.listdir("../Results/Experiments(feb2025)"):
    i = 0
    losses = []
    val = False
    #trainlosses, vallosses = [], []
    if file.split('.')[1] == 'pt' or file.split('.')[0][-4:] == 'Test':
        continue
    f = open("../Results/Experiments(feb2025)/" + file, 'r')
    for line in f.readlines()[6:]:
        if val:
            continue
        line = line.strip()
        if line != "Val losses":
            losses.append(float(line))
        if line == "Val losses":
            val = True
            continue
    trainlosses.append(losses)


for file in os.listdir("../Results/Experiments(feb2025)"):
    if file.split('.')[1] == 'pt' or file.split('.')[0][-5:] == 'Train':
        continue
    f = open("../Results/Experiments(feb2025)/" + file, 'r')
    line = f.readline().split()
    testlosses.append(float(line[2]))

for i in range(len(vallosses)):
    losses = trainlosses[i]
    losses = losses[-(len(vallosses[i])):]
    trainlosses[i] = losses

path = "../Results/TuningSimpleIncipits.txt"
f = open(path, 'w')
for i in range(len(trainlosses)):
    f.write(f"Run {i}, Trainlosses: ")
    for loss in trainlosses[i]:
        f.write(f"{loss} ")
    f.write(f"Vallosses: ")
    for loss in vallosses[i]:
        f.write(f"{loss} ")
    f.write(f'Testloss: {testlosses[i]} \n')

x_train = np.arange(0, len(trainlosses[1]))
y_train = np.array(trainlosses[1])
x_val = np.arange(0, len(vallosses[1]))
y_val = np.array(vallosses[1])
spl_val = make_interp_spline(x_val,y_val)
spl_train = make_interp_spline(x_train,y_train)
x_val = np.linspace(x_val.min(), x_val.max(), 100)
x_train = np.linspace(x_train.min(), x_train.max(), 100)
y_val = spl_val(x_val)
y_train = spl_train(x_train)
plt.plot(x_train, y_train, label="Train")
plt.plot(x_val, y_val, label='Validation')
plt.legend()
#plt.show()