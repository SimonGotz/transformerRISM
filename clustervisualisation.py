import plotly.express as px
from sklearn.manifold import TSNE
import torch
import inputProcessor
import numpy as np
from numpy.dtypes import StringDType
import pandas as pd

tsne = TSNE(n_components=2)
device = torch.device("cuda")

pathIncipits = "../Thesis/Data/mtcfsinst2.0_incipits(V2)/mtcjson"
pathWhole = "../Thesis/Data/mtcfsinst2.0/mtcjson"
featuresSimple = ["midipitch","duration","imaweight"]
featuresComplex = ["scaledegree","beatfraction","beatstrength"]

file = "WholeComplexRandom"
transformer = torch.load(f"../Weights/March/Models/{file}.pt", weights_only=False)
corpus = inputProcessor.Corpus()
transformer.eval()
transformer.to(device)
corpus.readDataSmallModel(featuresComplex, pathWhole, mode='whole')


color_discrete_map={"12171_0": 'red',
    "3680_0": 'dodgerblue',
    "1212_0": 'black',
    "7791_0": 'blueviolet',
    "1703_0": 'brown',
    "5861_0": 'hotpink',
    "4882_0": 'cyan',
    "7116_0": 'lime',
    "9216_0": 'orange'
    }

with torch.no_grad():
    embs = transformer(corpus.trainMelodies.to(device))
    labels = corpus.trainLabels
    ids = corpus.trainIds

def showCV(embs, labels, ids):
    labels = np.array(labels, dtype=StringDType)
    X = tsne.fit_transform(embs)
    dataDict = {"embs1": X[:, 0],
                "embs2": X[:, 1],
                "labels": labels,
                "ids": ids
                }
    df = pd.DataFrame(dataDict)

    fig = px.scatter(df, x='embs1', y='embs2', color='labels', color_discrete_map=color_discrete_map, hover_data={'ids':True})
    fig.update_traces(marker={'size': 20})
    fig.update_layout(
        title="2D Clustering on small dataset using model 7",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
        font=dict(
            size=30,
            color="Black"
        )
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

showCV(embs, labels, ids)