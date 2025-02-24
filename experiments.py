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
    return

def experiment2():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 0, 25, 'ITuneComplex', False, 'incipit')
    return

def experiment3():
    features = ["midipitch","duration","imaweight"]
    tune(features, 19, 25, 'ITuneSimpleHard', True, 'incipit')
    return

def experiment4():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 0, 25, 'ITuneComplexHard', True, 'incipit')
    return

def experiment5():
    features = ["midipitch","duration","imaweight"]
    tune(features, 0, 25, 'MTuneSimple', False, 'whole')
    return

def experiment6():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 0, 25, 'MTuneComplex', False, 'whole')
    return

def experiment7():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 0, 25, 'MTuneSimpleHard', True, 'whole')
    return

def experiment8():
    features = ["scaledegree","beatfraction","beatstrength"]
    tune(features, 0, 25, 'MTuneComplexHard', True, 'whole')
    return

def runExperiments():
    #experiment1()    
    #experiment2()
    experiment3()
    experiment4()
    #experiment5()
    #experiment6()
    #experiment7()
    #experiment8()

runExperiments()