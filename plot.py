import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.cm as cm

nHyperParams = 10
run = 3
path1 = "Results/TuningSimpleIncipitsNohard.txt"
path2 = "Results/TuningSimpleIncipitsHard.txt"
path3 = "Results/TuningComplexIncipitsNohard.txt"

trainxs = []
valxs = []
testxs = []

def readPath(path):
    f = open(path, 'r')
    trainlosses, vallosses, testlosses, configs = [],[],[],[]
    val = False
    for line in f.readlines():
        line = line.split(':')
        config = {}
        config[line[1]] = float(line[2].split()[0]) #LR
        for i in range(nHyperParams - 1):
            config[line[i + 2].split()[1]] = float(line[i + 3].split()[0]) 
        configs.append(config)
        #print(line)
        testloss = line[-1]
        testlosses.append(float(testloss))
        valloss, trainloss = [], []
        trainloss = [float(x) for x in line[nHyperParams + 2].split()[:-2]]
        valloss = [float(x) for x in line[nHyperParams + 3].split()[:-2]]
        #print(valloss)
        trainlosses.append(trainloss)
        vallosses.append(valloss)
    return trainlosses, vallosses, testlosses, configs

trainx1, valx1, testx1, configsx1 = readPath(path1)
trainx2, valx2, testx2, configsx2 = readPath(path2)
trainx3, valx3, testx3, configsx3 = readPath(path3)
trainxs = [trainx1, trainx2, trainx3]
valxs = [valx1, valx2, valx3]
testxs = [testx1, testx2, testx3]

def visualise():
    for i in range(len(trainxs)):
        testlosses = testxs[i]
        smallest = testlosses.index(min(testlosses))
        #testlosses = sorted(enumerate(testlosses), key=lambda i: i[1])
        trainlosses = trainxs[i][smallest]
        x_train = np.arange(0, len(trainlosses))
        y_train = np.array(trainlosses)
        spl_train = make_interp_spline(x_train,y_train)
        x_train = np.linspace(x_train.min(), x_train.max(), 50)
        y_train = spl_train(x_train)
        plt.plot(x_train, y_train, label=f"Model {i + 1}")
        #plt.plot(25, testlosses[i], 'ro', label='Test loss')
    ax = plt.gca()
    #ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 0.40])
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.legend()
    plt.show()

    for i in range(len(valxs)):
        testlosses = testxs[i]
        smallest = testlosses.index(min(testlosses))
        vallosses = valxs[i][smallest]
        #print(vallosses)
        x_val = np.arange(0, len(vallosses))
        y_val = np.array(vallosses)
        spl_val = make_interp_spline(x_val,y_val)
        x_val = np.linspace(x_val.min(), x_val.max(), 50)
        y_val = spl_val(x_val)
        plt.plot(x_val, y_val, label=f'Model {i + 1}')
    ax = plt.gca()
    #ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 0.35])
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.legend()
    plt.show()

#testlosses = sorted(enumerate(testlosses), key=lambda i: i[1])
visualise()

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