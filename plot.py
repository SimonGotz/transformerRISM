import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.cm as cm

nHyperParams = 10

path1 = "Results/Experiments(mar2025)/Models/Model 1/IncipitSimpleRandom"
path2 = "Results/Experiments(mar2025)/Models/Model 2/IncipitSimpleHard"
path3 = "Results/Experiments(mar2025)/Models/Model 3/IncipitComplexRandom"
path4 = "Results/Experiments(mar2025)/Models/Model 4/IncipitComplexHard"

trainxs = []
valxs = []
testxs = []

def getTestloss(path):
    testloss = 0
    with open(f"{path}Test.txt", 'r') as f:
        testloss = float(f.readline().split(' ')[2])
    return testloss

def readPath(path):
    with open(f"{path}Train.txt", 'r') as f:
        trainlosses, vallosses = [],[]
        val = False
        config = {}
        lines = f.readlines()
        for i in range(2, 12):
            config[lines[i]] = float(lines[i].split(' ')[1]) #LR
        for i in range(13, len(lines)):
            if lines[i].split(' ')[0] == 'Val':
                val = True
                continue 
            if not val:
                trainlosses.append(float(lines[i]))
            else:
                vallosses.append(float(lines[i]))
    f.close()

    return trainlosses, vallosses, config

trainx1, valx1, configsx1 = readPath(path1)
trainx2, valx2, configsx2 = readPath(path2)
trainx3, valx3, configsx3 = readPath(path3)
trainx4, valx4, configsx4 = readPath(path4)
trainxs = [trainx1, trainx2, trainx3, trainx4]
valxs = [valx1, valx2, valx3, valx4]
testlosses = [getTestloss(path1), getTestloss(path2), getTestloss(path3), getTestloss(path4)]

color = ['gold', 'red', 'green', 'blue']

def visualise():
    for i in range(len(trainxs)):
        trainlosses = trainxs[i]
        x_train = np.arange(0, len(trainlosses))
        y_train = np.array(trainlosses)
        spl_train = make_interp_spline(x_train,y_train)
        x_train = np.linspace(x_train.min(), x_train.max(), 50)
        y_train = spl_train(x_train)
        plt.plot(x_train, y_train, label=f"Model {i + 1}", color=color[i])
    ax = plt.gca()
    #ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

    for i in range(len(valxs)):
        vallosses = valxs[i]
        x_val = np.arange(0, len(vallosses))
        y_val = np.array(vallosses)
        spl_val = make_interp_spline(x_val,y_val)
        x_val = np.linspace(x_val.min(), x_val.max(), 50)
        y_val = spl_val(x_val)
        plt.plot(x_val, y_val, label=f'Model {i + 1}', color=color[i])
        plt.plot(len(vallosses), testlosses[i], 'go', color=color[i])
    ax = plt.gca()
    #ax.set_xlim([xmin, xmax])
    #plt.xticks(range(1, len(vallosses)))
    ax.set_ylim([0, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.legend()
    plt.grid(True, linestyle='--')
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
#paramsIndices = [x[0] for x in testlosses]
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