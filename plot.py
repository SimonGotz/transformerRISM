import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

run = 3
path = "../Results/TuningSimpleIncipits.txt"
f = open(path, 'r')
trainlosses, vallosses, testlosses = [],[],[]
val = False
for line in f.readlines():
    line = line.split()
    testloss = line[-1]
    testlosses.append(float(testloss))
    line = line[:-2]
    epochs = (len(line) - 4) // 2
    valloss, trainloss = [], []
    for i in range(epochs):
        trainloss.append(float(line[i+3]))
        valloss.append(float(line[i+4+epochs]))
    trainlosses.append(trainloss)
    vallosses.append(valloss)

def visualise(i):
    x_train = np.arange(0, len(trainlosses[i]))
    y_train = np.array(trainlosses[i])
    x_val = np.arange(0, len(vallosses[i]))
    y_val = np.array(vallosses[i])
    #print(y.max())
    spl_val = make_interp_spline(x_val,y_val)
    spl_train = make_interp_spline(x_train,y_train)
    x_val = np.linspace(x_val.min(), x_val.max(), 100)
    x_train = np.linspace(x_train.min(), x_train.max(), 100)
    y_val = spl_val(x_val)
    y_train = spl_train(x_train)
    plt.plot(x_train, y_train, label="Train")
    plt.plot(x_val, y_val, label='Validation')
    plt.plot(25, testlosses[i], 'ro', label='Test loss')
    plt.legend()
    plt.show()

min_testloss = 100
index_min = 0

for i in range(len(testlosses)):
    if testlosses[i] < min_testloss:
        min_testloss = testlosses[i]
        index_min = i
print(min_testloss)
visualise(index_min)