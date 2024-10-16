import numpy as np

meanAP = 0
count = 0

with open("mAPResults.txt", 'r') as f:
    for line in f.readlines():
        count += 1
        line = line.strip().split(' ')
        meanAP += float(line[4].split(':')[1])
print(meanAP/count)