import torch
import model
import inputProcessor
import time
from random import randint
import numpy as np


def condensed_index(i, j, m):
    index = m * i + j - ((i + 2) * (i + 1)) // 2
    #index = index.long()
    return index

device = torch.device("cuda:0")

pathIncipits = "../Thesis/Data/mtcfsinst2.0_incipits/mtcjson"
pathWhole = "../Thesis/Data/mtcfsinst2.0/mtcjson"
features = ["scaledegree","beatfraction","beatstrength"]
pdist = torch.nn.PairwiseDistance()
dists = []
meanAP = 0



transformer = torch.load("../Results/Experiments/{}.pt".format('e6.3'))
print(transformer)
results = []
precisions = []
corpus = inputProcessor.Corpus()
transformer.eval()
transformer.to(device)
corpus.readFolder(pathWhole, features)
data = corpus.data

tunefamDict = {}
count = 1

#print(data[0])
for i in range(len(data)):
    if i+1 != len(data) and data[i]['tunefamily'] == data[i+1]['tunefamily']:
        count += 1
    else:
        tunefamDict[data[i]['tunefamily']] = count
        count = 1
    with torch.no_grad():
        data[i]['Embedding'] = transformer(torch.tensor([data[i]['tokens']]).to(device)).squeeze(0)

labels = [corpus.data[i]['tunefamily'] for i in range(len(corpus.data))]
embs = [corpus.data[i]['Embedding'] for i in range(len(corpus.data))]
embs = torch.tensor(torch.stack(embs))
dm = torch.pdist(embs)
m = len(data)
with open("mAPResults.txt",'r+') as f:
    for i in range(m):
        #i = randint(0, len(data) - 1)
        dists = []
        q = data[i]
        for j in range(m):
            dists.append([pdist(q['Embedding'], data[j]['Embedding']), j])
            #dists.append([dm[condensed_index(i,j,m)], j])
        dists.sort()
        count = 0
        indices = []
        qMax = tunefamDict[q['tunefamily']]
        for j in range(qMax):
            if data[dists[j][1]]['tunefamily'] == q['tunefamily']:
                count += 1
                indices.append(j)
        precisions.append(count / qMax)
        f.write("Tunefamily: {}, Hits: {}/{}, Precision:{} \n".format(q['tunefamily'], count, qMax, count / qMax))
        #print(data[dists[1][1]]['id'])
        print("Tunefamily: {}, Hits: {}/{}, Precision:{}".format(q['tunefamily'], count, qMax, count / qMax))
        print(indices)
f.close()