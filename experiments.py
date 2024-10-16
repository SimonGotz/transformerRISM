import numpy as np
import main

def experiment1():
    features = ["midipitch","duration","imaweight"]
    main.main(0.3, 32, 4, 4, 256, 'e5',features)
    return

def experiment2():
    features = ["scaledegree","beatfraction","beatstrength"]
    main.main(0.3, 32, 4, 4, 256, 'e6.3',features)
    return

def experiment3():
    features = ["midipitch","duration","imaweight"]
    main.main(0.3, 32, 4, 4, 256, 'e5',features,hard_triplets=True)
    return

def experiment4():
    features = ["scaledegree","beatfraction","beatstrength"]
    main.main(0.3, 32, 4, 4, 256, 'e6',features,hard_triplets=True)
    return

def runExperiments():
    #experiment1()
    experiment2()
    #experiment3()
    #experiment4()

runExperiments()