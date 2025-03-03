import torch
import model
import inputProcessor
import time
import numpy as np
from statistics import mean
import sklearn.metrics as metrics
from tqdm import tqdm
import os

device = torch.device("cuda")
pathIncipits = "../Thesis/Data/mtcfsinst2.0_incipits(V2)/mtcjson"
pathWhole = "../Thesis/Data/mtcfsinst2.0/mtcjson"
featuresSimple = ["midipitch","duration","imaweight"]
featuresComplex = ["scaledegree","beatfraction","beatstrength"]
corpus = inputProcessor.Corpus()
corpus.readJSON(featuresComplex)
data = corpus.data
melodies, labels = [], []

def calculate_embeddings(transformer, melodies):
    start_time = time.time()
    with torch.no_grad():
        embs = transformer(melodies.to(device))
    elapsed = time.time() - start_time
    print("Embedding calculations: {:5.2f} s".format(elapsed))
    return embs

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

def extract(data):
    melodies, labels = [], []
    for melody in data:
        melodies.append(melody['tokens'])
        labels.append(corpus.tf2label[melody['tunefamily']])
    return torch.tensor(melodies), labels

def m_avg_precision(embs, labels):
    aps = []
    dm = metrics.pairwise_distances(embs, metric='euclidean')
    i = 0
    for i, dists in enumerate(tqdm(dm)):
        query_y = labels[i]
        y_true = [k for k in range(len(labels)) if labels[k] == query_y and k != i] # get indices of true labels

        if len(y_true) == 0: # if only 1 melody in tf -> go next
            continue

        indices = sorted(range(len(dists)), key=lambda k: dists[k]) # get sorted distance list
        indices = [k for k in indices if k != i]
        count, rank, scores = 0, 1, []

        while count < len(y_true):
            if indices[rank - 1] in y_true:
                count += 1
                scores.append(count / rank)
            rank += 1
        aps.append(mean(scores))

    return mean(aps)

def main(data, name, model, epoch=-1):
    melodies, labels = extract(data)
    with open(f"mAPResults{name}.txt",'a') as f:
        filename = name.split('_')
        model.eval()
        model.to(device)
        embs = calculate_embeddings(model, melodies)
        print(f"Calculing MAP score for Model {filename[1]} of {filename[0]}")
        mAP = m_avg_precision(embs, labels)
        print(f"MAP score: {mAP}")
        if epoch > 0:
            f.write(f"Model: {filename[1]} epoch: {epoch} mAP score: {mAP} \n")
        else:
            f.write(f"Model: {filename[1]} mAP score: {mAP} \n")
    f.close()
    return mAP

#main(data)

'''
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
f.close()'''