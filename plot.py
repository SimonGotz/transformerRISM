import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.cm as cm

run = 3
path = "Results/TuningSimpleIncipits.txt"
f = open(path, 'r')
trainlosses, vallosses, testlosses, configs = [],[],[], []
val = False
for line in f.readlines():
    line = line.split(':')
    config = {}
    config[line[1]] = float(line[2].split()[0]) #LR
    config[line[2].split()[1]] = float(line[3].split()[0]) # margin
    config[line[3].split()[1]] = int(line[4].split()[0]) # batch size
    config[line[4].split()[3]] = int(line[5].split()[0]) # n layers
    configs.append(config)
    testloss = line[-1]
    testlosses.append(float(testloss))
    #line = line[:-2]
    #epochs = (len(line) - 4) // 2
    valloss, trainloss = [], []
    trainloss = [float(x) for x in line[6].split()[:-2]]
    valloss = [float(x) for x in line[7].split()[:-2]]
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

def showParameters(name, values):
    #plt.scatter(name, values)
    #colors = cm.rainbow(np.linspace(0, 1, len(values)))
    colors = cm.rainbow(np.linspace(0, 1, len(values)))
    for i in range(len(values)):
        print(name)
        print(values[i])
        plt.plot(1, values[i], color=colors[i], marker='o')
    #plt.xticks([name])
    #plt.plot()
    plt.show()

min_testloss = 100
index_min = 0

testlosses = sorted(enumerate(testlosses), key=lambda i: i[1])
paramsIndices = [x[0] for x in testlosses]
params = {'LR' : [], 'margins': [], 'batch size': [], 'nlayers': []}
for i in range(3):
    l = params['LR']
    l.append(configs[i][' Learning rate'])
    params['LR'] = l
    l = params['margins']
    l.append(configs[i]['margin'])
    params['margins'] = l
    l = params['batch size']
    l.append(configs[i]['batch_size'])
    params['batch size'] = l
    l = params['nlayers']
    l.append(configs[i]['layers'])
    params['nlayers'] = l


for param in params:
    showParameters(param, params[param])
#visualise(index_min)