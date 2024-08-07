import numpy as np
import json

notes = ['A','B','C','D','E','F','G']
transposes = ['--','-','','#','##']
octaves = ['0','1','2','3','4','5','6','7','8']

transposeLite = ['-','','#']

with open('valueMap.json', 'r+') as f:
    data = json.load(f)
    for note in notes:
        for t in transposes:
            data['tonic'].append(note+t)
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()