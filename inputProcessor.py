import numpy as np
import torch
import os
import json
import ioparser as parser
from random import sample, randint, shuffle

class Dictionary(object):
    def __init__(self):
        self.pad = '0 0 0'
        self.cls = 1
        self.entry2idx = {'0 0 0': 0, '<cls>': self.cls}
        self.idx2entry = ['0 0 0', '<cls>']
        self.counter = 2

    def add_entry(self, entry):
        self.idx2entry.append(entry)
        self.entry2idx[entry] = self.counter
        self.counter += 1
        return self.entry2idx[entry]

    def __len__(self):
        return len(self.idx2entry)

class Corpus(object):

    def __init__(self):
        self.seqLen = 0
        self.dictionary = None
        self.features = []
        self.n = 0
        self.trainper, self.validper, self.testper = 0.70, 0.15, 0.15
        self.trainsize, self.validsize, self.testsize = 0,0,0
        self.samefam, self.samefamTrain, self.samefamValid, self.samefamTest = {}, {}, {}, {}
        self.trainMelodies, self.trainEmbs, self.trainLabels, self.trainIds = [],[],[],[]
        self.validMelodies, self.validEmbs, self.validLabels, self.validIds = [],[],[],[]
        self.testMelodies, self.testEmbs, self.testLabels, self.testIds = [],[],[],[]
        self.trainset, self.validset, self.testset = [],[],[]
        self.goodFams = ["1703_0", "9216_0", "1212_0", "4882_0", "5861_0", "7791_0", "12171_0", "3680_0", "7116_0"]
        self.parser = parser.Parser(self.features)
        self.unseenMult, self.unseenSingle = 0.5, 0.5
        self.singletons, self.multitons = [], {}
        self.tf2label = {}

    def read(self, path):
        corpus = []
        count = 0
        for filename in os.listdir(path):
            f = open(os.path.join(path, filename), 'r')
            data = json.load(f)
            if data['tunefamily'] == '':
                continue
            corpus.append(data)
        return sorted(corpus, key=lambda x: x['tunefamily'])

    def determineSeqlen(self, data):
        count = 0
        for melody in data:
            if len(melody['features']['pitch40']) > 200:
                count += 1
                continue
            if len(melody['features']['pitch40']) > self.seqLen:
                self.seqLen = len(melody['features']['pitch40'])
        print("seqlen: {}".format(self.seqLen))

    def parseFeatures(self, data, features):
        for i in range(len(data)):
            data[i] = {k: data[i][k] for k in ['id', 'tunefamily', 'features', 'tunefamily_full']}
            data[i]['features'] = {k: data[i]['features'][k] for k in features}
            data[i]['features'] = self.parser.parse(data[i]['features'], self.seqLen) 
        return data 

    def tokenise(self, data, features):
        data = self.parseFeatures(data, features)
        for i in range(len(data)):
            tokens = [self.dictionary.entry2idx['<cls>']] #add cls toke
            melody = data[i]['features']
            
            for j in range(self.seqLen):
                flist = []
                for feature in features:
                    flist.append(melody[feature][j])
                flist = ' '.join(map(str,flist))
                if flist not in self.dictionary.entry2idx.keys():
                    self.dictionary.add_entry(flist)
                tokens.append(self.dictionary.entry2idx[flist])
            data[i]['tokens'] = tokens
        self.seqLen += 1 # adjust for adding cls token
        return data

    def sortFamilies(self, data, train=False):
        sameFam = {}
        for i in range(len(data)):
            if data[i]['tunefamily'] in sameFam:
                l = sameFam[data[i]['tunefamily']]
                l.append(data[i])
                sameFam[data[i]['tunefamily']] = l
            else:
                sameFam[data[i]['tunefamily']] = [data[i]]
        
        return sameFam

    def getMultitons(self, samefam):
        for fam in samefam:
            if len(samefam[fam]) == 1:
                self.singletons.append(samefam[fam][0])
            else:
                self.multitons[fam] = samefam[fam]        

    def makeLabelDict(self, data):
        count = 0
        for mel in data:
            if mel['tunefamily'] in self.tf2label.keys():
                continue
            else:
                self.tf2label[mel['tunefamily']] = count
                count += 1        

    def induceMelodies(self, data):
        melodies, labels, embs, ids = [], [], [], []
        for mel in data:
            melodies.append(mel['tokens'])
            embs.append(0) # fill with zeroes so the size will be the same as labels and trainmelodies
            ids.append(mel['id'])
            labels.append(self.tf2label[mel['tunefamily']])

        return torch.tensor(melodies), labels, ids, embs

    def saveTestFamilies(self):

        goodMultitons = {"1703_0": [], "9216_0": [], "1212_0": [], "4882_0": [], "5861_0": [], "7791_0": [], "12171_0": [], "3680_0": [], "7116_0": []}

        for i in range(len(self.data)):
            if self.data[i]['tunefamily'] in self.goodFams:
                goodMultitons[self.data[i]['tunefamily']].append(self.data[i])

        # Make sure every set contains some data of the large tunefamilies
        for multiton in goodMultitons.keys():
            size = len(goodMultitons[multiton])
            testSize = int(self.testper*size)
            validSize = int(self.validper*size)
            for i in range(len(goodMultitons[multiton])):
                if i < testSize:
                    self.testset.append(goodMultitons[multiton][i])
                elif testSize < i < testSize + validSize:
                    self.validset.append(goodMultitons[multiton][i])
                else:
                    self.trainset.append(goodMultitons[multiton][i])
            self.multitons.pop(multiton)        

        uniqueMult = int(len(self.multitons.keys()) * self.unseenMult * self.testper)
        uniqueSingle = int(len(self.singletons) * self.unseenSingle * self.testper)

        randomMultitons = list(self.multitons.keys())
        shuffle(randomMultitons)
        randomSingletons = sample(self.singletons, uniqueSingle)
        i = 0
        for i in range(0, uniqueMult // 2):
            fam = randomMultitons[i]
            for j in range(len(self.multitons[fam])):
                self.testset.append(self.multitons[fam][j])
            self.multitons.pop(fam)
        for i in range(uniqueMult // 2, uniqueMult):
            fam = randomMultitons[i]
            for j in range(len(self.multitons[fam])):
                self.validset.append(self.multitons[fam][j])
            self.multitons.pop(fam)
        for i in range(0, uniqueSingle // 2):
            for j in range(len(randomSingletons[i])):
                self.testset.append(randomSingletons[j])
            self.singletons.remove(randomSingletons[i])
        for i in range(uniqueSingle // 2, uniqueSingle):
            for j in range(len(randomSingletons[i])):
                self.validset.append(randomSingletons[j])
            self.singletons.remove(randomSingletons[i])
        
        print("Unique multiton test melodies: {}".format(len(self.testset) // 2))
        print("Unique multiton valid melodies: {}".format(len(self.validset) // 2))

    # A method to run through the data to determine the dimensions of K and group data with the same tunefamily
    def makeSplit(self):
        sameFam = self.sortFamilies(self.data)
        self.getMultitons(sameFam)
        self.saveTestFamilies()
        self.partition()

    def makefamDictionaries(self, trainset, validset, testset):
        self.samefamTrain = self.sortFamilies(trainset)
        self.samefamValid = self.sortFamilies(validset)
        self.samefamTest = self.sortFamilies(testset)
        self.makeLabelDict(trainset)
        self.makeLabelDict(validset)
        self.makeLabelDict(testset)
        self.trainMelodies, self.trainLabels, self.trainIds, self.trainEmbs = self.induceMelodies(trainset)
        self.validMelodies, self.validLabels, self.validIds, self.validEmbs = self.induceMelodies(validset)
        self.testMelodies, self.testLabels, self.testIds, self.testEmbs = self.induceMelodies(testset)

        counterMult, counterTest, counterValid = 0, 0, 0
        totalMultTest, totalMultValid = 0,0
        for fam in self.samefamTrain.keys():
            if len(self.samefamTrain[fam]) > 1:
                counterMult += 1
                if fam in self.samefamValid.keys():
                    totalMultValid += 1
                if fam in self.samefamValid.keys():
                    totalMultTest += 1
            
        for fam in self.samefamValid.keys():
            if len(self.samefamValid[fam]) > 1:
                counterValid += 1
        
        for fam in self.samefamTest.keys():
            if len(self.samefamTest[fam]) > 1:
                counterTest += 1

        print("Total amount of training fams: {}".format(len(self.samefamTrain.keys())))
        print("Total amount of valid fams: {}".format(len(self.samefamValid.keys())))
        print("Total amount of test fams: {}".format(len(self.samefamTest.keys())))

        print("Number of multiton fams both in training and validation: {}".format(totalMultValid))
        print("Number of multiton fams both in training and test: {}".format(totalMultTest))

        print("Total amount of training multiton fams: {}".format(counterMult))
        print("Total amount of valid multiton fams: {}".format(counterValid))
        print("Total amount of test multiton fams: {}".format(counterTest))

    def partition(self):
        # partition dataset into train valid and test in such a way that test and valid include unseen families
        data = self.singletons
        for fam in self.multitons.keys():
            for mel in self.multitons[fam]:
                data.append(mel)
        shuffle(data)

        i = 0
        while len(self.trainset) < self.trainsize:
            self.trainset.append(data[i])
            i += 1
        while len(self.validset) < self.validsize:
            self.validset.append(data[i])
            i += 1
        while len(self.testset) < self.testsize:
            self.testset.append(data[i])
            i += 1
        
        remainder = 12344 - (len(self.testset) + len(self.validset) + len(self.trainset))
        for i in range(remainder):
            self.trainset.append(data[-i])
        '''
            if data[i]['tunefamily'] in self.samefamValid:
                l = self.samefamValid[data[i]['tunefamily']]
                l.append(data[i])
                self.samefamValid[data[i]['tunefamily']] = l
            else:
                self.samefamValid[data[i]['tunefamily']] = [data[i]]
            if data[i+remainder]['tunefamily'] in self.samefamTest:
                l = self.samefamTest[data[i+remainder]['tunefamily']]
                l.append(data[i+remainder])
                self.samefamTest[data[i+remainder]['tunefamily']] = l
            else:
                self.samefamTest[data[i+remainder]['tunefamily']] = [data[i+remainder]]
        '''
        #print("Total amount of test fams: {}".format(len(self.samefamTest.keys())))
        #print("Total amount of valid fams: {}".format(len(self.samefamValid.keys())))
        #self.validset = self.data[self.trainsize:self.trainsize + self.validsize]
        #self.testset = self.data[self.trainsize + self.validsize:]
        
    def clean(self):
        self.data = []
        self.samefam, self.samefamTrain, self.samefamValid, self.samefamTest = {}, {}, {}, {}
        self.singletons, self.multitons = [], {}
        self.trainMelodies = []
        self.trainset, self.validset, self.testset = [],[],[]
        self.embs, self.labels = [], []

    def writeToJSON(self):
        with open('trainData.json', 'w') as f:
            json.dump(self.trainset, f)
        f.close()
        with open('validData.json', 'w') as f:
            json.dump(self.validset, f)
        f.close()
        with open('testData.json', 'w') as f:
            json.dump(self.testset, f)
        f.close()

    def makeDataSplit(self, path):
        '''
        INPUT
            The folder of the dataset
        OUTPUT
            xs = lists of all melodies in pitch40
        '''
        self.clean()
        self.data = self.read(path) 
        self.n = len(os.listdir(path))
        self.trainsize = int(len(self.data)*self.trainper)
        self.validsize = int(len(self.data)*self.validper)
        self.testsize = int(len(self.data)*self.testper)
        self.makeSplit()
        self.writeToJSON()
    
    def readData(self, features, mode='incipit'):
        self.features = features
        self.dictionary = Dictionary()

        with open('trainData.json', 'r') as f:
            self.trainset = json.load(f)
        f.close()
        with open('validData.json', 'r') as f:
            self.validset = json.load(f)
        f.close()
        with open('testData.json', 'r') as f:
            self.testset = json.load(f)
        f.close()
        if mode == 'incipit':
            self.seqLen = 19
        else:
            self.seqLen = 200
        
        self.trainsize = len(self.trainset)
        self.validsize = len(self.validset)
        self.testsize = len(self.testset)

        print("Tokenising data")
        self.trainset = self.tokenise(self.trainset, features)
        self.validset = self.tokenise(self.validset, features)
        self.testset = self.tokenise(self.testset, features)
        self.makefamDictionaries(self.trainset, self.validset, self.testset)
        print(len(self.trainset) + len(self.testset) + len(self.validset))
        print("vocab_size: {}".format(len(self.dictionary.entry2idx)))
