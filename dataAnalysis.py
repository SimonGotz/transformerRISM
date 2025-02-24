import numpy as np
import inputProcessor as ip
import matplotlib.pyplot as plt

corpus = ip.Corpus()
path = "../Thesis/Data/mtcfsinst2.0_incipits(V2)/mtcjson"
features = ["midipitch","duration","imaweight"]
corpus.readFolder(path, features)
tfsizeDict = {}
for melody in corpus.data:
    if melody['tunefamily'] in tfsizeDict:
        tfsizeDict[melody['tunefamily']] = tfsizeDict[melody['tunefamily']] + 1
    else:
        tfsizeDict[melody['tunefamily']] = 1

maxSize = max(tfsizeDict, key=tfsizeDict.get)
classSizes = np.zeros(tfsizeDict[maxSize] + 1, dtype=np.int32)
print(f"Largest class size: {len(classSizes) - 1}")

values = tfsizeDict.values()
histValues = []
for value in values:
    if value <= 10 and value > 1:
        histValues.append(value)

for tf in tfsizeDict:
    classSizes[tfsizeDict[tf]] += 1



y = classSizes[2:5]
x = np.arange(2, len(classSizes))
plt.xlabel("Class size")
plt.ylabel("Frequency")
plt.title("Class size distribution of the dataset")
plt.xticks([2,3,4,5,6,7,8,9,10])
plt.yticks([0,100,200,300,400,500,600,700,800,900,1000])
#plt.plot(x, y)
plt.hist(histValues, bins=9, align='left', color='skyblue', edgecolor='black')
plt.show()