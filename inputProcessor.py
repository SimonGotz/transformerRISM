import numpy as np
import torch
import os
import json
import ioparser as parser
from random import sample, randint, shuffle

class Dictionary(object):
    def __init__(self):
        self.entry2idx = {}
        self.idx2entry = []
        self.counter = 0

    def add_entry(self, entry):
        if entry not in self.entry2idx:
            self.counter += 1
            self.idx2entry.append(entry)
            self.entry2idx[entry] = len(self.idx2entry) - 1
        return self.entry2idx[entry]

    def __len__(self):
        return len(self.idx2entry)

class Corpus(object):

    def __init__(self, path, features):
        self.seqLen = 0
        self.seqLens = []
        self.dictionary = Dictionary()
        self.totalsize = 12345
        self.ntokens = 322 #c8 in base 40 
        self.features = features
        self.n = len(os.listdir(path)) 
        self.trainx, self.trainy = 0,[]
        self.validx, self.validy = [],[]
        self.testx, self.testy = [],[]
        self.trainper, self.validper, self.testper = 0.80, 0.10, 0.10
        self.trainsize, self.validsize, self.testsize = 0,0,0
        self.testset = [
            '1703_0', '9216_0', '1212_0', '4882_0', '5861_0',
            '7791_0','12171_0','3680_0','7116_0'
        ]
        self.samefam, self.samefamTrain, self.samefamValid, self.samefamTest, self.testFam = [], [], [], [], []
        self.id2tag = {}
        self.tag = 0
        self.parser = parser.Parser(self.features)
        self.readFolder(path)

    def determineSeqlen(self):
        count = 0
        for melody in self.data:
            if len(melody['features']['pitch40']) > 500:
                count += 1
                continue
            self.seqLens.append(len(melody['features']['pitch40']))
            if len(melody['features']['pitch40']) > self.seqLen:
                self.seqLen = len(melody['features']['pitch40'])
        print(count)

    def makeDictionary(self):
        self.dictionary.add_entry('0 0') #make sure the padding values get id 0 for future masking
        for melody in self.data:
            melody = melody['features']
            for i in range(len(melody['pitch40'])):
                flist = []
                for feature in self.features:
                    flist.append(melody[feature][i])
                flist = ' '.join(map(str,flist))
                self.dictionary.add_entry(flist)


    def parseFeatures(self):
        for i in range(len(self.data)):
            self.data[i] = {k: self.data[i][k] for k in ['id', 'tunefamily', 'features']}
            self.data[i]['features'] = {k: self.data[i]['features'][k] for k in self.features}
            self.data[i]['features'] = self.parser.parse(self.data[i]['features'], self.seqLen)  

    def tokenise(self):
        for i in range(len(self.data)):
            melody = self.data[i] 
            ids = []
            for j in range(self.seqLen):
                flist = []
                for feature in self.features:
                    flist.append(melody['features'][feature][j])
                flist = ' '.join(map(str,flist))
                ids.append(self.dictionary.entry2idx[flist])
            self.data[i]['tokens'] = ids

    def sortFamilies(self):
        famcounter = 0
        lastfam = self.data[0]['tunefamily']
        self.samefam = [[self.data[0]]]
        for i in range(len(self.data)):
            if lastfam != self.data[i]['tunefamily'] and i > 0:
                lastfam = self.data[i]['tunefamily']
                self.samefam.append([self.data[i]])
                famcounter += 1
            elif i < len(self.data) - 1:
                self.samefam[famcounter].append(self.data[i])
            self.id2tag.update({self.data[i]['id'] : self.tag})
            self.tag += 1

    def sort(self, path):
        corpus = []
        for filename in os.listdir(path):
            f = open(os.path.join(path, filename), 'r')
            data = json.load(f)
            corpus.append(data)
        return sorted(corpus, key=lambda x: x['tunefamily'])

    # A method to run through the data to determine the dimensions of K and group data with the same tunefamily
    def fillCorpus(self):
        self.determineSeqlen()
        self.parseFeatures()
        self.makeDictionary()
        self.tokenise()
        self.sortFamilies()

    def extract(self, data):
        xs = []
        ys = []
        for d in data:
            #d = {k: d[k] for k in ['id', 'tunefamily', 'features']}
            #d['features'] = {k: d['features'][k] for k in self.features}
            fv = []
            ys.append(d['tunefamily'])
            for i in range(self.seqLen):
                token = []
                for feature in d['features']:
                    if i >= len(d['features'][feature]): d['features'][feature].append(0) # padding
                    if d['features'][feature][i] == None: d['features'][feature][i] = 0 # handling None values
                    token.append(d['features'][feature][i])
                fv.append(token)
            try:
                xs.append(fv)
            except:
                return
        xs = torch.tensor(xs)
        return xs, ys
    
    def makeAPN(self, corpus):
        heap = []
        anchors, positives, negatives = [], [], []
        i = 0
        while i + 1 < len(corpus):
            if corpus[i]['tunefamily'] == corpus[i+1]['tunefamily']:
                anchors.append(corpus[i])
                positives.append(corpus[i+1])
                # extra check om te kijken of anchor != positive
                i += 1
            else:
                heap.append(corpus[i])
            i += 1

        for i in range(len(anchors)):
            if len(heap) != 0:
                negatives.append(sample(heap, 1)[0])
            else:
                print("check")
                negatives.append(sample(corpus, 1)[0])

        return self.extract(anchors), self.extract(positives), self.extract(negatives)

    def partition(self, data, tn, vn, test=False, online=True):
        if test:
            return data[:70], data[70:85], data[85:100]
        trainxn = int(len(data)*tn)
        validxn = int(len(data)*vn)
        rest = len(data) - (trainxn + 2*validxn)
        train = data[:trainxn + rest]
        valid = data[trainxn + rest:trainxn+validxn + rest]
        test = data[-validxn:]
        return train, valid, test

    def partitionSmart(self):
        # partition dataset into train valid and test in such a way that test and valid include unseen families
        counter = 0
        testset = []
        self.samefam = sample(self.samefam, len(self.samefam))
        self.trainsize = int(len(self.samefam)*self.trainper)
        self.validsize = int(len(self.samefam)*self.validper)
        self.testsize = int(len(self.samefam)*self.testper)
        self.samefamTrain = self.samefam[:self.trainsize]
        self.samefamValid = self.samefam[self.trainsize:self.trainsize + self.validsize]
        self.samefamTest = self.samefam[self.trainsize + self.validsize:]
        '''
        for fam in self.samefamTrain:
            if fam[0]['tunefamily'] in self.testset:
                self.samefamTest.append(fam)
                self.testFam.append(fam)
                self.samefamTrain.pop(counter)
            counter += 1
        counter = 0
        for fam in self.samefamValid:
            if fam[0]['tunefamily'] in self.testset:
                self.samefamTest.append(fam)
                self.testFam.append(fam)
                self.samefamValid.pop(counter)
            counter += 1
        counter = 0
        for i in range(len(self.samefamTrain)):
            for j in range(len(self.samefamTrain[i])):
                counter += 1
        self.trainsize = counter
        counter = 0
        for i in range(len(self.samefamValid)):
            for j in range(len(self.samefamValid[i])):
                counter += 1
        self.validsize = counter
        counter = 0
        self.testsize = self.totalsize - self.trainsize - self.validsize
        '''
        
    def readFolder(self, path, triplet=True, online=True):
        '''
        INPUT
            The folder of the dataset
        OUTPUT
            xs = lists of all melodies in pitch40
        '''
        self.data = self.sort(path) 
        self.fillCorpus()
        self.partitionSmart()
        print(self.seqLen)
        return self.data

        if online:
            return corpus
        if triplet:
            anchors, positives, negatives = self.makeAPN(corpus)
        
        return anchors.view(-1), positives.view(-1), negatives.view(-1)

    # Not needed   

    def makeOfflineTriplets(self, corpus):
        heap = []
        triples = []
        i = 0
        while i + 1 < len(corpus):
            if corpus[i]['tunefamily'] == corpus[i+1]['tunefamily']:
                triples.append([corpus[i], corpus[i + 1]])
                i += 1
            else:
                heap.append(corpus[i])
            i += 1

        for i in range(len(triples)):
            if len(heap) != 0:
                triples[i].append(sample(heap, 1)[0])
            else:
                triples[i].append(sample(corpus, 1)[0])

        return self.extract(triples) 

    def readKeys(self, path):
        k = np.zeros((self.n, self.mk, self.dk), dtype=float)
        i = 0

        for filename in os.listdir(path):
            f = open(os.path.join(path, filename), 'r')
            data = json.load(f)
            keys = self.handleNone(data['features']['pitch40'])
            for j in range(len(keys)):
                k[i][j][keys[j]] = 1
            i += 1
        return k
    
    def readValues(self, path):
        v = np.zeros((self.n, self.mk, len(self.features)), dtype=float)
        i = 0
        for filename in os.listdir(path):
            f = open(os.path.join(path, filename), 'r')
            data = json.load(f)
            k = 0 # counter for iteration over features
            for f in self.features:
                values = self.handleNone(data['features'][f])
                for j in range(len(values)):
                    v[i][j][k] = values[j]
            
            i += 1
        return v
