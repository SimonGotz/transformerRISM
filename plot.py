import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import scipy.stats as stats

pathTuning = "../Results/Experiments/e45Train.txt"
pathLengths = "melLengths.txt"
#f = open(pathLengths, 'r')
trainlosses = []
'''
Tuning Code
for line in f.readlines()[7:]:
    line = line.strip()
    if line == "Val losses":
        continue
    trainlosses.append(float(line))
'''
def parseLengthList():
    lengths = []
    count = 0
    with open('melLengths.txt', 'r') as f:
        line = f.readline()
        for l in line.split(','):
            if l != '':
                length = int(l)
                if length > 200:
                    count += 1
                lengths.append(int(l))
    
    print(len(lengths))
    print(count)
    print(count / len(lengths))
    return lengths

wholeMelLenghts = parseLengthList()

x = np.arange(0, len(wholeMelLenghts))
y = np.array(wholeMelLenghts)
print(y)
spl = make_interp_spline(x,y)

x = np.linspace(x.min(), x.max(), 10)
y_smooth = spl(x)

density = stats.gaussian_kde(y)

plt.hist(y, bins=100)
plt.xlabel("Amount of notes in melody")
plt.ylabel("Frequency")
plt.title("Melody length distribution")
plt.show()
