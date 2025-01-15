import torch
import model
import inputProcessor
import time
from random import randint
import numpy as np
from statistics import mean
import sklearn.metrics as metrics

def condensed_index(i, j, m):
    index = m * i + j - ((i + 2) * (i + 1)) // 2
    #index = index.long()
    return index

def mean_average_precision(embs, labels, metric):
    sim_matrix = 1 - metrics.pairwise_distances(embs, metric=metric)
    scores = []
    #labels = np.array(labels)
    for i, sims in enumerate(sim_matrix):
        mask = np.arange(sims.shape[0]) != i # filter query
        query_y = labels[i]
        target_y = (labels[mask] == query_y).astype(int)
        if target_y.sum() > 0:
            scores.append(metrics.average_precision_score(target_y, sims[mask]))
    return np.mean(scores)

device = torch.device("cuda:0")

pathIncipits = "../Thesis/Data/mtcfsinst2.0_incipits/mtcjson"
pathWhole = "../Thesis/Data/mtcfsinst2.0/mtcjson"
features = ["scaledegree","beatfraction","beatstrength"]
pdist = torch.nn.PairwiseDistance()
dists = []
meanAP = 0
transformer = torch.load("../Results/Experiments/{}.pt".format('e45'))
print(transformer)
results = []
precisions = []
corpus = inputProcessor.Corpus()
transformer.eval()
transformer.to(device)
corpus.readFolder(pathIncipits, features)
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
labels = np.array(labels)
embs = [corpus.data[i]['Embedding'] for i in range(len(corpus.data))]
embs = torch.tensor(torch.stack(embs))
dm = torch.pdist(embs)
m = len(data)
avg_precisions = []

meanAP = mean_average_precision(embs, labels, 'cosine')
print(meanAP)
meanAP = mean_average_precision(embs, labels, 'euclidean')
print(meanAP)

with open("mAPResults.txt",'r+') as f:
    for i in range(m):
        #i = randint(0, len(data) - 1)
        dists = []
        q = data[i]
        qMax = tunefamDict[q['tunefamily']]
        if qMax == 1:
            continue
        for j in range(m):
            dists.append([pdist(q['Embedding'], data[j]['Embedding']), j])
            #dists.append([dm[condensed_index(i,j,m)], j])
        dists.sort()
        count = 0
        rank = 0
        indices, precisions = [], []
        #for j in range(qMax):
        while count < qMax:
            if data[dists[rank][1]]['tunefamily'] == q['tunefamily']:
                count += 1
                indices.append(j)
                precisions.append(count / (rank + 1))
            rank += 1
        avg_precisions.append(mean(precisions))
        f.write("Query: {}, Tunefamily: {}, average precision:{} \n".format(q['id'], q['tunefamily'], avg_precisions[-1:]))
        #print(data[dists[1][1]]['id'])
        #print("Query: {}, Tunefamily: {}, average precision:{} ".format(q['id'], q['tunefamily'], avg_precisions[-1:]))
    print("MAP: {}".format(mean(avg_precisions)))
    f.write("MAP: {}".format(mean(avg_precisions)))
f.close()