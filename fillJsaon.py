import numpy as np
import json
import os

notes = ['A','B','C','D','E','F','G']
transposes = ['--','-','','#','##']
octaves = ['0','1','2','3','4','5','6','7','8']

transposeLite = ['-','','#']
values = []
corpus = []
path = "../Thesis/Data/mtcfsinst2.0_incipits/mtcjson"

for filename in os.listdir(path):
    with open(os.path.join(path, filename), 'r') as f:
        data = json.load(f)
        corpus.append(data)
    
corpus = sorted(corpus, key=lambda x: x['tunefamily'])

for melody in corpus:
    for frac in melody['features']['beatfraction']:
        if frac in values:
            continue
        else:
            values.append(frac)

print(values)