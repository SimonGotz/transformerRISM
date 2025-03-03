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
        if entry not in self.entry2idx:
            self.idx2entry.append(entry)
            self.entry2idx[entry] = len(self.idx2entry) - 1
            self.counter += 1
        return self.entry2idx[entry]

    def __len__(self):
        return len(self.idx2entry)

class Corpus(object):

    def __init__(self):
        self.seqLen = 0
        self.dictionary = Dictionary()
        self.ntokens = 322 #c8 in base 40 
        self.features = []
        self.n = 0
        self.trainper, self.validper, self.testper = 0.70, 0.15, 0.15
        self.trainsize, self.validsize, self.testsize = 0,0,0
        self.samefam, self.samefamTrain, self.samefamValid, self.samefamTest = {}, {}, {}, {}
        self.trainMelodies, self.embs, self.labels, self.ids = [],[],[],[]
        self.trainset, self.validset, self.testset = [],[],[]
        self.goodFams = ["1703_0", "9216_0", "1212_0", "4882_0", "5861_0", "7791_0", "12171_0", "3680_0", "7116_0"]
        self.parser = parser.Parser(self.features)
        self.unseenMult, self.unseenSingle = 0.5, 0.5
        self.singletons, self.multitons = [], {}
        self.tf2label = {}

    def determineSeqlen(self):
        count = 0
        for melody in self.data:
            if len(melody['features']['pitch40']) > 200:
                count += 1
                continue
            if len(melody['features']['pitch40']) > self.seqLen:
                self.seqLen = len(melody['features']['pitch40'])
        print("seqlen: {}".format(self.seqLen))

    def makeDictionary(self):
        #self.dictionary.add_entry('0 0') #make sure the padding values get id 0 for future masking
        for i in range(len(self.data)):
            tokens = [self.dictionary.entry2idx['<cls>']] #add cls token
            melody = self.data[i]['features']
            for j in range(self.seqLen):
                flist = []
                for feature in self.features:
                    flist.append(melody[feature][j])
                flist = ' '.join(map(str,flist))
                self.dictionary.add_entry(flist)
                tokens.append(self.dictionary.entry2idx[flist])
            self.data[i]['tokens'] = tokens
        self.seqLen += 1 # adjust for adding cls token
        print("vocab_size: {}".format(len(self.dictionary.entry2idx)))


    def parseFeatures(self):
        for i in range(len(self.data)):
            self.data[i] = {k: self.data[i][k] for k in ['id', 'tunefamily', 'features', 'tunefamily_full']}
            self.data[i]['features'] = {k: self.data[i]['features'][k] for k in self.features}
            self.data[i]['features'] = self.parser.parse(self.data[i]['features'], self.seqLen)  

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

    def sort(self, path):
        corpus = []
        count = 0
        for filename in os.listdir(path):
            f = open(os.path.join(path, filename), 'r')
            data = json.load(f)
            if data['tunefamily'] == '':
                continue
            corpus.append(data)
            if data['tunefamily'] in self.tf2label.keys():
                continue
            else:
                self.tf2label[data['tunefamily']] = count
                count += 1
        return sorted(corpus, key=lambda x: x['tunefamily'])

    def tfToLabel(self):
        for i in range(len(self.labels)):
            self.labels[i] = self.tf2label[self.labels[i]]

    def induceMelodies(self):
        for fam in self.samefamTrain.keys():
            for j in range(len(self.samefamTrain[fam])):
                self.trainMelodies.append(self.samefamTrain[fam][j]['tokens'])
                self.labels.append(self.samefamTrain[fam][j]['tunefamily'])
                self.embs.append(0) # fill with zeroes so the size will be the same as labels and trainmelodies
                self.ids.append(self.samefamTrain[fam][j]['id'])
        self.tfToLabel()
        self.trainMelodies = torch.tensor(self.trainMelodies)

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
            self.samefamTest[multiton] = []
            self.samefamValid[multiton] = []
            for i in range(len(goodMultitons[multiton])):
                if i < testSize:
                    self.samefamTest[multiton].append(goodMultitons[multiton][i])
                elif testSize < i < testSize + validSize:
                    self.samefamValid[multiton].append(goodMultitons[multiton][i])
                else:
                    self.trainset.append(goodMultitons[multiton][i])
            self.multitons.pop(multiton)        

        uniqueMult = int(len(self.multitons.keys()) * self.unseenMult * self.testper)
        uniqueSingle = int(len(self.singletons) * self.unseenSingle * self.testper)

        randomMultitons = list(self.multitons.keys())
        shuffle(randomMultitons)
        randomSingletons = sample(self.singletons, uniqueSingle*2)

        i = 0
        for i in range(uniqueMult):
            self.samefamTest[randomMultitons[i]] = self.multitons[randomMultitons[i]]
            self.samefamValid[randomMultitons[i+uniqueMult]] = self.multitons[randomMultitons[i+uniqueMult]]
            self.multitons.pop(randomMultitons[i])
            self.multitons.pop(randomMultitons[i+uniqueMult])
        for i in range(uniqueSingle):
            self.samefamTest[randomSingletons[i]['tunefamily']] = [randomSingletons[i]]
            self.samefamValid[randomSingletons[i+uniqueSingle]['tunefamily']] = [randomSingletons[i+uniqueSingle]]
            self.singletons.remove(randomSingletons[i])
            self.singletons.remove(randomSingletons[i+uniqueSingle])
        
        print("Unique multiton test fams: {}".format(len(self.samefamTest.keys()) // 2))
        print("Unique multiton valid fams: {}".format(len(self.samefamValid.keys()) // 2))

    # A method to run through the data to determine the dimensions of K and group data with the same tunefamily
    def fillCorpus(self):
        self.determineSeqlen()
        self.parseFeatures()
        self.makeDictionary()
        sameFam = self.sortFamilies(self.data)
        #self.smallPartition()
        self.getMultitons(sameFam)
        self.saveTestFamilies()
        self.partition()
        self.samefamTrain = self.sortFamilies(self.trainset)
        self.induceMelodies()
        print("Total amount of training fams: {}".format(len(self.samefamTrain.keys())))
        counterMult, counterTest, counterValid = 0, 0, 0
        totalMultTest, totalMultValid = 0,0
        for fam in self.samefamTrain:
            if len(self.samefamTrain[fam]) > 1:
                counterMult += 1
            if fam in self.samefamValid and len(self.samefamValid[fam]) > 1:
                counterValid += 1
            if fam in self.samefamTest and len(self.samefamTest[fam]) > 1:
                counterTest += 1

        print("Number of multiton fams both in training and test: {}".format(counterTest))
        print("Number of multiton fams both in training and validation: {}".format(counterValid))

        print("Total amount of training multiton fams: {}".format(counterMult))
        print("Total amount of valid multiton fams: {}".format(totalMultValid))
        print("Total amount of test multiton fams: {}".format(totalMultTest))

    def smallPartition(self):
        shuffle(self.data)
        self.trainset = self.data
        #self.validset = self.data[self.trainsize:self.trainsize + self.validsize]
        #self.testset = self.data[self.testsize:]

    def partition(self):
        # partition dataset into train valid and test in such a way that test and valid include unseen families

        data = self.singletons
        for fam in self.multitons:
            for mel in self.multitons[fam]:
                data.append(mel)
        shuffle(data)
        for melody in data[:self.trainsize]:
            self.trainset.append(melody)
        remainder = len(data[self.trainsize:]) // 2
        for i in range(remainder):
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

        print("Total amount of test fams: {}".format(len(self.samefamTest.keys())))
        print("Total amount of valid fams: {}".format(len(self.samefamValid.keys())))
        #self.validset = self.data[self.trainsize:self.trainsize + self.validsize]
        #self.testset = self.data[self.trainsize + self.validsize:]
        
    def clean(self):
        self.data = []
        self.samefam, self.samefamTrain, self.samefamValid, self.samefamTest = {}, {}, {}, {}
        self.singletons, self.multitons = [], {}
        self.trainMelodies = []
        self.trainset, self.validset, self.testset = [],[],[]
        self.embs, self.labels = [], []

    def readFolder(self, path, features, triplet=True, online=True):
        '''
        INPUT
            The folder of the dataset
        OUTPUT
            xs = lists of all melodies in pitch40
        '''
        self.clean()
        self.data = self.sort(path) 
        self.n = len(os.listdir(path))
        self.features = features
        self.trainsize = int(len(self.data)*self.trainper)
        self.validsize = int(len(self.data)*self.validper)
        self.testsize = int(len(self.data)*self.testper)
        self.fillCorpus()
        #self.partitionSmart()
