import numpy as np
import inputProcessor
from random import sample, randint
import torch
import itertools
from collections import Counter

def hardest_negative(dists):
    hardest = np.argmax(dists)
    idx = None
    if dists[hardest] > 0:
        idx = hardest
    return idx


def random_hard_negative(dists):
    hard_ones = np.where(dists > 0)[0]
    idx = None
    if len(hard_ones) > 0:
        idx = np.random.choice(hard_ones)
    else:
        idx = np.random.randint(0, dists.shape[0])
    return idx


def semihard_negative(dists, margin=1):
    semi_hard_ones = np.where((dists < margin) & (dists > 0))[0]
    idx = None
    if len(semi_hard_ones) > 0:
        idx = np.random.choice(semi_hard_ones)
    return idx

def condensed_index(i, j, n):
    index = i * n + j - i * (i + 1) / 2 - i - 1
    index = index.long()
    return index

class TripletSelector:
    def __init__(self, method='semi-hard', margin=1):
        if method == 'hardest':
            self.sel_fn = hardest_negative
        elif method == 'random_hardest':
            self.sel_fn = random_hard_negative
        elif method == 'semi-hard':
            self.sel_fn = semihard_negative
        self.margin = margin
        self.dupes = 0
        self.triplets = []

    def makeOnlineTriplets(self, embs, labels, margin, sel_fn='semihard_negative'):
        if sel_fn == 'hardest_negative':
            self.sel_fn = hardest_negative
        else:
            self.sel_fn = semihard_negative
        anchors, positives, negatives, tfanchors, tfnegatives, triplets = [], [], [], [], [], []
        embs = torch.tensor(torch.stack(embs))
        dm = torch.pdist(embs)
        for label in set(labels):
            mask = np.in1d(labels, label)
            if sum(mask) < 2:
                continue #eliminate labels with only 1 positive
            label_idx = np.where(mask)[0]
            neg_idx = torch.LongTensor(np.where(np.logical_not(mask))[0])
            pos_pairs = torch.LongTensor(list(itertools.combinations(label_idx, 2)))
            pos_dists = dm[condensed_index(pos_pairs[:, 0], pos_pairs[:, 1], embs.shape[0])]
            for (i, j), dist in zip(pos_pairs, pos_dists):
                loss = dist - dm[condensed_index(i, neg_idx, embs.shape[0])] + self.margin
                loss = loss.data.cpu().numpy()
                if self.sel_fn is semihard_negative:
                    hard_idx = self.sel_fn(loss, margin)
                else:
                    hard_idx = self.sel_fn(loss)
                if hard_idx is not None:
                    triplets.append([i, j, neg_idx[hard_idx]])
        if not triplets:
            print('No triplets found... Sampling random hard ones.')
            triplets = self.get_triplets(embs, torch.LongTensor(labels), random_hard_negative)
        return triplets

    def getIndex(self, dat):
        while True:
            x = sample(list(dat.keys()), 1)[0]
            y = sample(list(dat.keys()), 1)[0]
            if len(dat[x]) == 1 or x == y:
                continue
            else:
                break    
        return x,y

    def sampleTriplets(self, dat, batch_size, hard=False):
        positives, negatives, anchors, tfanchors, tfnegatives = [],[],[],[],[]
        x,y = 0,0
        sameFam = False
        for i in range(batch_size):
            if not hard:
                x,y = self.getIndex(dat)
            else:
                x,y = 0,0
            data = sample(dat[x], 2)
            positives.append(data[0]['tokens'])
            anchors.append(data[1]['tokens'])
            tfanchors.append(data[0]['tunefamily'])
                
            negative = sample(dat[y], 1)[0]
            negatives.append(negative['tokens'])
            tfnegatives.append(negative['tunefamily'])
            
        return torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives), tfanchors, tfnegatives