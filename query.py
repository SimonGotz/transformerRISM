import torch
import model
import inputProcessor
import time
from random import randint
import numpy as np
from statistics import mean
import sklearn.metrics as metrics
import operator

def mean_average_precision(embs, labels, metric):
    sim_matrix = 1 - metrics.pairwise_distances(embs, metric=metric) #dist matrix met euclidean distances
    scores = []
    #labels = np.array(labels)
    for i, sims in enumerate(sim_matrix):
        mask = np.arange(sims.shape[0]) != i # filter query
        query_y = labels[i]
        target_y = (labels[mask] == query_y).astype(int)
        if target_y.sum() > 0:
            #score = metrics.average_precision_score(target_y, sims[mask])
            scores.append(metrics.average_precision_score(target_y, sims[mask]))
    return np.mean(scores)


device = torch.device("cuda")
pathIncipits = "../Thesis/Data/mtcfsinst2.0_incipits(V2)/mtcjson"
pathWhole = "../Thesis/Data/mtcfsinst2.0/mtcjson"
features = ["midipitch","duration","imaweight"]
meanAP = 0
transformer = torch.load("../Weights/HyperparameterTuning/TEST_0.pt", weights_only=False)
results = []
precisions = []
corpus = inputProcessor.Corpus()
transformer.eval()
transformer.to(device)
corpus.readFolder(pathIncipits, features)
data = corpus.data
tfsizeDict = {}

def update_embeddings(transformer, melodies):
    transformer.eval()
    start_time = time.time()
    with torch.no_grad():
        embs = transformer(melodies.to(device))
    elapsed = time.time() - start_time
    print("Embedding calculations: {:5.2f} s".format(elapsed))
    return embs

melodies = []
labels = []

for melody in data:
    melodies.append(melody['tokens'])
    labels.append(melody['tunefamily'])

for i in range(len(labels)):
    labels[i] = corpus.tf2label[labels[i]]
    if labels[i] in tfsizeDict:
        tfsizeDict[labels[i]] = tfsizeDict[labels[i]] + 1
    else:
        tfsizeDict[labels[i]] = 1
melodies = torch.tensor(melodies)
embs = update_embeddings(transformer, melodies)

def m_avg_precision(embs, labels, metric):
    dists = []
    aps = []
    dm = metrics.pairwise_distances(embs, metric=metric)
    for i, dists in enumerate(dm):
        mask = np.arange(dists.shape[0]) != i
        query_y = labels[i]
        y_true = [k for k in labels if labels[k] == query_y and i != k]

        if len(y_true) == 0: # if only 1 melody in tf
            continue

        indices = sorted(range(len(dists)), key=lambda k: dists[k])
        indices = [k for k in indices if i != k]
        count, scores = 0, []
        for y in y_true:
            count += 1
            scores.append(count / (indices.index(y) + 1))
        aps.append(mean(scores))

    return mean(aps)
    

#stime = time.time()
##meanAP = mean_average_precision(embs, labels, 'euclidean')
#print(f"Karsdorp time:{time.time() - stime}")
#print(meanAP)

stime = time.time()
mAP = m_avg_precision(embs, labels, 'euclidean')
print(f"Own time:{time.time() - stime}")
print(mAP)
while True: continue

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