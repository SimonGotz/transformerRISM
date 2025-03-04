import numpy as np
import main
import tune as t
import query as q
import random

featuresSimple = ["midipitch","duration","imaweight"]
featuresComplex = ["scaledegree","beatfraction","beatstrength"]

def readParameters(modelNumber, tuningNumber, modelName):
    params = {}
    with open(f"Results\Experiments(feb2025)\Tuning Model {modelNumber}\{modelName}_{tuningNumber}train.txt") as f:
        lines = f.readlines()
        for line in lines[2:12]:
            line = line.split(":")
            if line[0] in ['lr', 'wd', 'dropout','margin','epsilon']:
                params[line[0]] = float(line[1])
            else:
                params[line[0]] = int(line[1])
    f.close()
    return params

def tune(features, start, stop, name, hard, mode):
    mAPs = []
    for i in range(start, stop):
        params = t.sample()
        print(f"Starting run {i} of {name}")
        if hard:
            params['batch_size'] = 2048
            if params['d_model'] == 1024:
                params['batch_size'] = 1024
        params['batch_size'] = 1024
        params['d_model'] = 32
        mAPs.append(main.main(params, "{}_{}".format(name,str(i)), features=features, mode=mode, hard_triplets=hard))
    print(mAPs)

def experiment1():
    features = ["scaledegree","beatfraction","beatstrength"]
    mAP = 0
    name = "smallTfSetTokenCheck_0"
    while mAP < 0.92:
        params = t.sample()
        params['batch_size'] = 8
        params['lr'] = 1e-3
        params['wd'] = 0.001
        params['d_model'] = 32
        params['n_heads'] = 2
        params['n_layers'] = 2
        params['d_ff'] = 512
        params['dropout'] = 0
        params['margin'] = 0.3
        params['epsilon'] = 1e-6
        mAP = main.main(params, name, features=features, mode='incipit', hard_triplets=False)
        #mAP = q.main()

def experiment2Tuning1():
    tune(featuresSimple, 0, 25, 'ITuneSimple', False, 'incipit')

def experiment2Tuning2():
    tune(featuresComplex, 0, 25, 'ITuneComplex', False, 'incipit')

def experiment2Tuning3():
    tune(featuresSimple, 23, 25, 'ITuneSimpleHard', True, 'incipit')

def experiment2Tuning4():
    tune(featuresComplex, 0, 25, 'ITuneComplexHard', True, 'incipit')

def experiment2Tuning5():
    tune(featuresSimple, 0, 25, 'MTuneSimpleRandom', False, 'whole')

def experiment2Tuning6():
    tune(featuresComplex, 0, 25, 'MTuneSimpleHard', True, 'whole')

def experiment2Tuning7():
    tune(featuresSimple, 0, 25, 'MTuneComplexRandom', False, 'whole')

def experiment2Tuning8():
    tune(featuresComplex, 0, 25, 'MTuneComplexHard', True, 'whole')

def experiment3Model1():
    params = readParameters(1, 4, "ITuneSimple") # best scoring model was tuning model 4
    mAP = main.main(params, "IncipitSimpleRandom", features=featuresSimple, mode='incipit', hard_triplets=False)

def experiment3Model2():
    params = readParameters(2, 0, "ITuneSimpleHard") 
    mAP = main.main(params, "IncipitSimpleHard", features=featuresSimple, mode='incipit', hard_triplets=True)

def experiment3Model3():
    params = readParameters(3, 10, "ITuneComplex")
    mAP = main.main(params, "IncipitComplexRandom", features=featuresComplex, mode='incipit', hard_triplets=False)

def experiment3Model4():
    params = readParameters(4, 19, "ITuneComplexHard")
    mAP = main.main(params, "IncipitComplexHard", features=featuresComplex, mode='incipit', hard_triplets=True)

def testExperiment():
    tune(featuresSimple, 0, 1, 'TEST2', False, 'incipit')

def testExperiment2():
    tune(featuresSimple, 0, 25, 'ITuneSimpleImpactTokenTest', False, 'incipit')

def runExperiments():
    experiment2Tuning5()
    #experiment2Tuning6()
    #experiment2Tuning7()
    #experiment2Tuning8()

runExperiments()
#testExperiment()

#def finishedExperiments():
    #experiment1()    
    #experiment2()
    #experiment3()
    #experiment4()
    #correctnessExperiment()
    #testExperiment()
    #experiment3Model1()
    #experiment3Model2()
    #experiment3Model3()
    #experiment3Model4()
    #testExperiment2()