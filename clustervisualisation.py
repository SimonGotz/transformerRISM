import plotly.express as px
from sklearn.manifold import TSNE
import torch
import inputProcessor
import numpy as np
from numpy.dtypes import StringDType
import pandas as pd

tsne = TSNE(n_components=2)
device = torch.device("cuda:0")

pathIncipits = "../Thesis/Data/mtcfsinst2.0_incipits/mtcjson"
pathWhole = "../Thesis/Data/mtcfsinst2.0/mtcjson"
features = ["scaledegree","beatfraction","beatstrength"]
pdist = torch.nn.PairwiseDistance()
dists = []
meanAP = 0
#transformer = torch.load("../Results/Experiments/{}.pt".format('e45'))
file = "smallTfSet"
transformer = torch.load(f"../Weights/HyperparameterTuning/{file}.pt", weights_only=False)
results = []
precisions = []
corpus = inputProcessor.Corpus()
transformer.eval()
transformer.to(device)
corpus.readFolder(pathIncipits, features)
data = corpus.data
embeddings, labels, ids = [], [], []

with torch.no_grad():
    embs = transformer(corpus.trainMelodies.to(device))
    labels = corpus.labels
    ids = corpus.ids

for data in corpus.data:
    if data['id'] == 'NLB070180_01' or data['id'] == 'NLB177544_01' or data['id'] == 'NLB125141_01':
        print(data)

while True: continue

labels = np.array(labels, dtype=StringDType)
X = tsne.fit_transform(embs)
dataDict = {"embs1": X[:, 0],
            "embs2": X[:, 1],
            #"embs3": X[:, 2],
            "labels": labels,
            "ids": ids
            }
df = pd.DataFrame(dataDict)
df.sort_values(by=['labels'])

fig = px.scatter(df, x='embs1', y='embs2', color='labels', hover_data={'ids':True})
fig.update_traces(marker={'size': 15})
fig.update_layout(
    title="2D Clustering on small dataset using T-sne",
    xaxis_title="First t-SNE",
    yaxis_title="Second t-SNE",
    font=dict(
        #family="Courier New, monospace",
        size=30,  # Set the font size here
        color="Black"
    )
)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
#fig.layout.scene.aspectratio = {'x':1, 'y':1, 'z':1}
#fig.update_zaxes(showticklabels=False)
fig.show()