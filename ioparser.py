import numpy as np
import json

# pad list l tot number m
def pad(l, m):
    i = len(l)
    while i < m: #-1 is to adjust for adding CLS token
        l.append(0)
        i += 1
    return l

    # Method to replace None values with -1
def handleNone(l):
    if not None in l:
        return l
    for i in range(len(l)):
        if l[i] == None:
            l[i] = -1
    return l

class Parser(object):

    def __init__(self, features):
        self.features = features
        self.flist = {}
        with open('valueMapCategorical.json', 'r') as f:
            data = json.load(f)
            for feature in data:
                self.flist[feature] = data[feature]

    def parseFraction(self, values):
        catlist = []
        for value in values:
            catlist.append(self.flist["fraction"].index(value) + 1)

    def parseNumerical(self, values):
        for i in range(len(values)):
            if type(values[i]) is int:
                break
            else:
                values[i] = round(values[i],1) #round numbers to decrease vocab size
        return values

    def parseCategorical(self, values, name):
        catlist = []
        for value in values:
            if value in self.flist[name]:
                catlist.append(self.flist[name].index(value) + 1)
            else:
                catlist.append(1)
        return catlist

    def checkType(self, feature):
        categorical = False
        for x in feature:
            if x == None:
                continue
            elif type(x) is str:
                categorical = True
                break
            else:
                break
        return categorical

    def calculateFracvalues(self, mel, f):
        if f not in self.features:
            self.features.append(f)
            self.flist[f] = []
        for frac in mel:
            if frac not in self.flist[f]:
                self.flist[f].append(frac)
        

    def parse(self, features, n):
        melody = features
        for f in features:
            categorical = self.checkType(melody[f])
            if categorical:
                if "frac" in f:
                    self.calculateFracvalues(melody[f], f)
                melody[f] = pad(self.parseCategorical(handleNone(melody[f]), f), n)
            else:
                melody[f] = pad(self.parseNumerical(handleNone(melody[f])), n)
        return melody