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

file = "smallTfSetTokenCheck_0"
transformer = torch.load(f"../Weights/HyperparameterTuning/{file}.pt", weights_only=False)
corpus = inputProcessor.Corpus()
transformer.eval()
transformer.to(device)
corpus.readFolder(pathIncipits, features)

with torch.no_grad():
    embs = transformer(corpus.trainMelodies.to(device))
    labels = corpus.labels
    ids = corpus.ids

for data in corpus.data:
    if data['id'] == 'NLB184955_01' or data['id'] == 'NLB198065_01':
        print(data)

#while True: continue

def showCV(embs, labels, ids):
    labels = np.array(labels, dtype=StringDType)
    X = tsne.fit_transform(embs)
    dataDict = {"embs1": X[:, 0],
                "embs2": X[:, 1],

                "labels": labels,
                "ids": ids
                }
    df = pd.DataFrame(dataDict)
    df.sort_values(by=['labels'])
    outlier = df[df['labels'] == '4882_0']
    print(outlier)

    fig = px.scatter(df, x='embs1', y='embs2', color='labels', hover_data={'ids':True})
    fig.update_traces(marker={'size': 20})
    fig.update_layout(
        title="2D Clustering on small dataset using T-sne",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
        font=dict(
            size=30,
            color="Black"
        )
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    #fig.show()

showCV(embs, labels, ids)