import math
import os

dirpath = 'C:/Users/simon/Desktop/Scriptie/Thesis/Tuning Results1/'
losses = {}
path = 'ResultsTuning.txt'
i = 0
f = open(path, 'w')

for filename in os.listdir(dirpath):
    if filename.split('.')[-1] == 'pt':
        continue
    else:
        filename = os.path.join(dirpath, filename)
        file = open(filename, 'r')
        parts = file.readline().split(':')
        if parts[0] == "Avg loss":
            losses[i] = float(parts[1])
            f.write("Loss run {}: {} \n".format(i, parts[1]))
            continue
        for j in range(5):
            line = file.readline()
            f.write(line + " ")
        i = i + 1
        f.write('\n')
losses = {k: v for k, v in sorted(losses.items(), key=lambda item: item[1])}
print(losses)