import numpy as np
import main
import tune as t

def tune(features, runs, name, hard):
    for i in range(11,25):
        lr, batch_size, margin, nlayers = t.sample()
        main.main(lr, batch_size, margin, nlayers, "{}_{}".format(name,str(i)), features=features, hard_triplets=hard)

def experiment1():
    features = ["midipitch","duration","imaweight"]
    tune(features, 25, 'ITuneSimple', False)
    return

def experiment2():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 25, 'ITuneComplex', False)
    return

def experiment3():
    features = ["midipitch","duration","imaweight"]
    tune(features, 25, 'ITuneSimpleHard', True)
    return

def experiment4():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 25, 'ITuneComplexHard', True)
    return

def runExperiments():
    experiment1()
    #experiment2()
    #experiment3()
    #experiment4()

runExperiments()