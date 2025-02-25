import numpy as np
import main
import tune as t

def tune(features, start, stop, name, hard, mode):
    for i in range(start, stop):
        params = t.sample()
        print(f"Starting run {i} of {name}")
        main.main(params, "{}_{}".format(name,str(i)), features=features, mode=mode, hard_triplets=hard)

def experiment1():
    features = ["midipitch","duration","imaweight"]
    tune(features, 0, 25, 'ITuneSimple', False, 'incipit')

def experiment2():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 0, 25, 'ITuneComplex', False, 'incipit')

def experiment3():
    features = ["midipitch","duration","imaweight"]
    tune(features, 19, 25, 'ITuneSimpleHard', True, 'incipit')

def experiment4():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 9, 25, 'ITuneComplexHard', True, 'incipit')

def experiment5():
    features = ["midipitch","duration","imaweight"]
    tune(features, 0, 25, 'MTuneSimple', False, 'whole')

def experiment6():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 0, 25, 'MTuneComplex', False, 'whole')

def experiment7():
    features = ["midipitch","duration","imaweight"]
    tune(features, 0, 25, 'MTuneSimpleHard', True, 'whole')

def experiment8():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 0, 25, 'MTuneComplexHard', True, 'whole')

def testExperiment():
    features = ["midipitch","duration","imaweight"]
    tune(features, 0, 1, 'TEST', True, 'incipit')

def runExperiments():
    #experiment1()    
    #experiment2()
    #experiment3()
    experiment4()
    #experiment5()
    #experiment6()
    #experiment7()
    #experiment8()

#runExperiments()
testExperiment()