import numpy as np
import main
import tune as t
import query as q
import random
import torch
import inputProcessor as ip

featuresSimple = ["midipitch","duration","imaweight"]
featuresComplex = ["scaledegree","beatfraction","beatstrength"]

def readParameters(modelNumber, tuningNumber, modelName):
    params = {}
    with open(f"Results\Experiments(mar2025)\Tuning\Model {modelNumber}\{modelName}_{tuningNumber}train.txt") as f:
        lines = f.readlines()
        for line in lines[2:12]:
            line = line.split(":")
            if line[0] in ['lr', 'wd', 'dropout','margin','epsilon']:
                params[line[0]] = float(line[1])
            else:
                params[line[0]] = int(line[1])
    f.close()
    return params

def tune(features, start, stop, name, hard, mode, model):
    mAPs = []
    for i in range(start, stop):
        params = t.sample()
        if hard:
            params['batch_size'] = 512
        print(f"Starting run {i} of {name}")
        mAPs.append(main.main(params, name="{}_{}".format(name,str(i)), modelNumber=model,  features=features, mode=mode, hard_triplets=hard, tuning=True))

    bestModel = mAPs.index(max(mAPs))        
    print(mAPs)
    print(bestModel) 
    return bestModel

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
        mAP = main.main(params, name, features=features, modelNumber=0, mode='incipit', hard_triplets=False, tuning=True)
        #mAP = q.main()

def experiment2Tuning1():
    return tune(featuresSimple, 0, 25, 'ITuneSimpleRandom', False, 'incipit', model=1)

def experiment2Tuning2():
    return tune(featuresSimple, 0, 25, 'ITuneSimpleHard', True, 'incipit', model=2)

def experiment2Tuning3():
    return tune(featuresComplex, 0, 25, 'ITuneComplexRandom', False, 'incipit', model=3)

def experiment2Tuning4():
    return tune(featuresComplex, 0, 25, 'ITuneComplexHard', True, 'incipit', model=4)

def experiment2Tuning5():
    return tune(featuresSimple, 22, 25, 'MTuneSimpleRandom', False, 'whole', model=5)

def experiment2Tuning6():
    return tune(featuresSimple, 0, 25, 'MTuneSimpleHard', True, 'whole', model=6)

def experiment2Tuning7():
    return tune(featuresComplex, 0, 25, 'MTuneComplexRandom', False, 'whole', model=7)

def experiment2Tuning8():
    return tune(featuresComplex, 6, 25, 'MTuneComplexHard', True, 'whole', model=8)

def experiment3Model1(tnumber):
    params = readParameters(modelNumber=1, tuningNumber=tnumber, modelName="ITuneSimpleRandom") # best scoring model was tuning model 4
    mAP = main.main(params, "IncipitSimpleRandom", features=featuresSimple, modelNumber=1, mode='incipit', hard_triplets=False, tuning=False)

def experiment3Model2(tnumber):
    params = readParameters(2, tnumber, "ITuneSimpleHard") 
    mAP = main.main(params, "IncipitSimpleHard", features=featuresSimple, modelNumber=2, mode='incipit', hard_triplets=True, tuning=False)

def experiment3Model3(tnumber):
    params = readParameters(3, tnumber, "ITuneComplexRandom")
    mAP = main.main(params, "IncipitComplexRandom", features=featuresComplex, modelNumber=3, mode='incipit', hard_triplets=False, tuning=False)

def experiment3Model4(tnumber):
    params = readParameters(4, tnumber, "ITuneComplexHard")
    mAP = main.main(params, "IncipitComplexHard", features=featuresComplex, modelNumber=4, mode='incipit', hard_triplets=True, tuning=False)

def experiment3Model5(tnumber):
    params = readParameters(5, tnumber, "MTuneSimpleRandom")
    mAP = main.main(params, "WholeSimpleRandom", features=featuresSimple, modelNumber=5, mode='whole', hard_triplets=False, tuning=False)

def experiment3Model6(tnumber):
    params = readParameters(6, tnumber, "MTuneSimpleHard")
    mAP = main.main(params, "WholeSimpleHard", features=featuresSimple, modelNumber=6, mode='whole', hard_triplets=True, tuning=False)

def experiment3Model7(tnumber):
    params = readParameters(7, tnumber, "MTuneComplexRandom")
    mAP = main.main(params, "WholeComplexRandom", features=featuresComplex, modelNumber=7, mode='whole', hard_triplets=False, tuning=False)

def experiment3Model8(tnumber):
    params = readParameters(8, tnumber, "MTuneComplexHard")
    mAP = main.main(params, "WholeComplexHard", features=featuresComplex, modelNumber=8, mode='whole', hard_triplets=True, tuning=False)

def testExperiment():
    return tune(featuresSimple, 0, 2, 'TEST2', False, 'incipit', 1)

def testExperiment2():
    tune(featuresSimple, 0, 25, 'ITuneSimpleImpactTokenTest', False, 'incipit')

def getBestModel(name):
    #corpus = ip.Corpus()
    #corpus.readData(featuresComplex, "../Thesis/Data/mtcfsinst2.0_incipits(V2)/mtcjson")
    maps = []
    for i in range(0, 25):
        with open(f"Results/MAP scores/Tuning/mAPResults{name}_{i}.txt", 'rb') as f:
            lines = f.readlines()
            mAP = float(lines[-1].split()[-1])
            maps.append(mAP)
            #maps.append(q.main(corpus.testMelodies, corpus.testLabels, f"{name}_{i}", model, mode='testing'))
            f.close()
    bestModel = maps.index(max(maps))        
    print(maps)
    print(bestModel) 
    return bestModel


def runExperiments():
    #experiment1()
    #bestModel1 = testExperiment()

    #bestModel1 = experiment2Tuning1()
    #experiment3Model1(1)
    #bestModel2 = experiment2Tuning2()
    #experiment3Model2(bestModel2)
    #bestModel3 = experiment2Tuning3()
    #experiment3Model3(4)
    #bestModel4 = experiment2Tuning4()
    #bestmodel4 = getBestModel('ITuneComplexHard')
    #experiment3Model4(bestmodel4)
    #bestModel5 = getBestModel('MtuneSimpleRandom')
    #experiment3Model5(bestModel5)
    #bestModel6 = experiment2Tuning6()
    #bestModel6 = getBestModel('MtuneSimpleHard')
    #experiment3Model6(bestModel6)
    #bestModel7 = experiment2Tuning7()
    #bestModel7 = getBestModel('MtuneComplexRandom')
    #experiment3Model7(bestModel7)
    bestModel8 = experiment2Tuning8()
    bestModel8 = getBestModel('MtuneComplexHard')
    experiment3Model8(bestModel8)

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