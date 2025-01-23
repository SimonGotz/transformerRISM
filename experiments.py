import numpy as np
import main
import tune as t

def tune(features, runs, name):
    for i in range(runs):
        lr, d_model, nheads, nlayers, d_ff = t.sample()
        main.main(lr, d_model, nheads, nlayers, d_ff, "{}_{}".format(name,str(i)), features=features)

def experiment1():
    features = ["midipitch","duration","imaweight"]
    tune(features, 100, 'ITuneComplex')
    return

def experiment2():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 100, 'ITuneSimple')
    return

def experiment3():
    features = ["midipitch","duration","imaweight"]
    main.main(0.3, 32, 4, 4, 256, 'e5',features)
    return

def experiment4():
    features = ["scaledegree","beatfraction","beatstrength"]
    main.main(0.3, 32, 4, 4, 256, 'e6',features)
    return

def runExperiments():
    experiment1()
    experiment2()
    #experiment3()
    #experiment4()

runExperiments()