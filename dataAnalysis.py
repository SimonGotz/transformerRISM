import numpy as np
import inputProcessor as ip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from statistics import mean, mode, median
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

corpus = ip.Corpus()
path = "../Thesis/Data/mtcfsinst2.0/mtcjson"
features = ["midipitch","duration","imaweight"]
data = corpus.sort(path)

years = {}

for melody in data:
    if melody['year'] < 0:
        continue
    if melody['year'] in years.keys():
        years[melody['year']] += 1
    else:
        years[melody['year']] = 1

y = np.zeros(len(years.keys()))

#cmap = plt.get_cmap('plasma')
plt.figure(figsize=(10, 3), dpi=80)
x1, y1 = [1500, 2025], [0, 0]
plt.plot(x1, y1, c='grey', alpha=0.3)
plt.scatter(years.keys(), y, s=years.values(), c='mediumseagreen', alpha=0.7)
ax = plt.gca()
ax.set_xlim([1500, 2025])
plt.yticks([])  
plt.gca().set_xlabel('Year', fontsize=16)
plt.gca().set_title("Year of publication for songs in MTC", fontsize=16)
plt.xticks(fontsize=14)
#plt.title("Year of publication for songs in MTC")
#ax.set_ylim([ymin, ymax])
plt.grid(linestyle='dotted')
plt.show()

print(f"Mean of sequence lenghts: {mean(seqlens)}")
print(f"Mode of sequence lenghts: {mode(seqlens)}")
print(f"Median of sequence lenghts: {median(seqlens)}")
print(f"Max seqlen: {max(seqlens)}")
print(f"Min seqlen: {min(seqlens)}")

tfsizeDict = {}
for melody in data:
    if melody['tunefamily'] in tfsizeDict:
        tfsizeDict[melody['tunefamily']] = tfsizeDict[melody['tunefamily']] + 1
    else:
        tfsizeDict[melody['tunefamily']] = 1


maxSize = max(tfsizeDict, key=tfsizeDict.get)
classSizes = np.zeros(tfsizeDict[maxSize] + 1, dtype=np.int32)
#print(f"Largest class size: {len(classSizes) - 1}")
#print(f"Second largest class size:: {classSizes[-5:]}")

values = tfsizeDict.values()
histValues = []
for value in values:
    if value <= 10 and value > 1:
        histValues.append(value)

for tf in tfsizeDict:
    classSizes[tfsizeDict[tf]] += 1

#print(f"Second largest class size: {classSizes[-5:]}")

y = classSizes[2:5]
x = np.arange(2, len(classSizes))
plt.xlabel("Class size")
plt.ylabel("Frequency")
plt.title("Class size distribution")
plt.xticks([2,3,4,5,6,7,8,9,10])
plt.yticks([0,100,200,300,400,500,600,700,800,900,1000])
#plt.plot(x, y)
plt.hist(histValues, bins=9, align='left', color='skyblue', edgecolor='black')
#plt.show()