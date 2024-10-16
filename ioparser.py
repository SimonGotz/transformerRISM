import numpy as np
import json

# pad list l tot number m
def pad(l, m):
    i = len(l)
    while i < m:
        l.append(0)
        i += 1
    return l

    # Method to replace None values with -1
def handleNone(l):
    if not None in l:
        return l
    for i in range(len(l)):
        if l[i] == None:
            l[i] = -1000
    return l

class Parser(object):

    def __init__(self, features):
        self.features = features
        self.flist = {}
        with open('valueMap.json', 'r') as f:
            data = json.load(f)
            for feature in data:
                self.flist[feature] = data[feature]

    def parseFraction(self, values):
        catlist = []
        for value in values:
            catlist.append(self.flist["fraction"].index(value) + 1)

    def parseNumerical(self, feature):
        return feature

    def parseCategorical(self, values, name):
        catlist = []
        for value in values:
            catlist.append(self.flist[name].index(value) + 1)
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
                melody[f] = pad(handleNone(self.parseCategorical(melody[f], f)), n)
            else:
                melody[f] = pad(handleNone(self.parseNumerical(melody[f])), n)
        return melody