import plotly.express as px
from sklearn.manifold import TSNE
import torch
import inputProcessor
import numpy as np

tsne = TSNE(n_components=2)
device = torch.device("cuda:0")

pathIncipits = "../Thesis/Data/mtcfsinst2.0_incipits/mtcjson"
pathWhole = "../Thesis/Data/mtcfsinst2.0/mtcjson"
features = ["scaledegree","beatfraction","beatstrength"]
pdist = torch.nn.PairwiseDistance()
dists = []
meanAP = 0
#transformer = torch.load("../Results/Experiments/{}.pt".format('e45'))
transformer = torch.load("../Results/Experiments/{}.pt".format('e6.3'))
print(transformer)
results = []
precisions = []
corpus = inputProcessor.Corpus()
transformer.eval()
transformer.to(device)
corpus.readFolder(pathWhole, features)
data = corpus.data
embeddings, labels = [], []

for i in range(len(data)):
    with torch.no_grad():
        if data[i]['tunefamily'] in corpus.goodFams:
        #data[i]['Embedding'] = transformer(torch.tensor([data[i]['tokens']]).to(device)).squeeze(0)
            embeddings.append(transformer(torch.tensor([data[i]['tokens']]).to(device)).squeeze(0))
            labels.append(data[i]['tunefamily'])


#embeddings = [corpus.data[i]['Embedding'] for i in range(len(corpus.data))]
#embeddings = np.array(embeddings)
embs = torch.stack(embeddings)
embs = embs.numpy()
print(embs.shape)
X = tsne.fit_transform(embs)
print(X.shape)
#labels = [corpus.data[i]['tunefamily'] for i in range(len(corpus.data))]
labels = np.array(labels)
print(labels.shape)
print("calculated embeddings")

fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels)
fig.update_traces(marker={'size': 15})
fig.update_layout(
    title="t-SNE visualization of tunefamily classification",
    xaxis_title="First t-SNE",
    yaxis_title="Second t-SNE",
)
fig.show()