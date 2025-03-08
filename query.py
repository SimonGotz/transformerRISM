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
#pathIncipits = "../Thesis/Data/mtcfsinst2.0_incipits(V2)/mtcjson"
#pathWhole = "../Thesis/Data/mtcfsinst2.0/mtcjson"
featuresSimple = ["midipitch","duration","imaweight"]
featuresComplex = ["scaledegree","beatfraction","beatstrength"]

def calculate_embeddings(transformer, melodies, mode='training'):
    start_time = time.time()
    with torch.no_grad():
        if mode != 'training':
            embs = transformer(melodies.to(device))
        else:
            batch_size = len(melodies) // 8
            remainder = len(melodies) - (8 * batch_size)
            embs = transformer(melodies[0:batch_size].to(device))
            for i in range(1,8): # calculate in batches if using melodies
                embs = torch.cat((embs, transformer(melodies[i*batch_size:(i+1) * batch_size].to(device))), 0)
            embs = torch.cat((embs, transformer(melodies[-remainder:].to(device))), 0)
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

def extract(data, labeldict):
    melodies, labels = [], []
    for melody in data:
        melodies.append(melody['tokens'])
        labels.append(labeldict[melody['tunefamily']])
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

def pAtOne(embs, labels):
    p_at1 = []
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

        if indices[0] in y_true:
            p_at1.append(1)
        else:
            p_at1.append(0)

    return mean(p_at1)

def silhouetteCoefficient(embs, labels):
    sc = metrics.silhouette_coefficient(embs, labels)
    return sc

def writeResults(name, epoch, mAP, mode):
    with open(f"Results\MAP scores\mAPResults{name}.txt",'a') as f:
        if epoch > 0:
            f.write(f"Model: {name} epoch: {epoch} {mode} mAP score: {mAP} \n")
        else:
            f.write(f"Model: {name} mAP {mode} score: {mAP} \n")
    f.close()

def main(embs, labels, name, model, mode='Training', epoch=-1):
    #melodies, labels = extract(data)
    model.eval()
    model.to(device)

    print(f"Calculing {mode} MAP score for Model {name} at epoch {epoch}")
    mAP = m_avg_precision(embs, labels)
    print(f"{mode} mAP score: {mAP}")
    writeResults(name, epoch, mAP, mode)

    return mAP

def calculateMetrics(name):
    corpus = inputProcessor.Corpus()
    #device = torch.device("cpu")
    path = "../Thesis/Data/mtcfsinst2.0/mtcjson"
    corpus.readData(featuresComplex, path, mode='whole')
    #name = "smallTfSetTokenCheck_0"
    with open(f"../Weights/March/Models/{name}.pt", 'rb') as f:
        model = torch.load(f, weights_only=False)
        model.to(device)
        f.close()
    embs = calculate_embeddings(model, corpus.validMelodies, mode='test')
    labels = corpus.validLabels
    mAP = m_avg_precision(embs, labels)
    print(f"MAP: {mAP}")
    p_at1 = pAtOne(embs, labels)
    print(f"p_at1: {p_at1}")
    sc = metrics.silhouette_score(embs, labels)
    print(f"sc: {sc}")
    #writeResults(name, -1, mAP, )

#corpus = inputProcessor.Corpus()
#path = "../Thesis/Data/mtcfsinst2.0/mtcjson"
#corpus.readData(featuresSimple, path, mode='whole')

#for i in range(25):
    #with open(f"../Weights/March/Tuning/MTuneSimpleHard_{i}.pt", 'rb') as f:
        #model = torch.load(f, weights_only=False)
        #embs = calculate_embeddings(model, corpus.trainMelodies)
        #main(corpus.testMelodies, torch.tensor(corpus.testLabels), 'MTuneSimpleHardTuningAll', model, epoch=i, mode='Test')