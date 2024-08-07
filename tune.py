import numpy as np
from scipy.stats import uniform
import random
import main

runs = 10

def loguniform(low=0, high=1, size=None):
    print(np.random.uniform(low, high, size))
    return np.exp(np.random.uniform(low, high, size))

def sample():
    lr = np.random.uniform(0.01,0.5)
    d_model = random.sample([8, 16, 32, 64, 128],1)
    nheads = random.sample([2,4,8],1)
    nlayers = np.random.randint(low=1,high=10)
    d_ff = random.sample([128,256,512,1024,2048],1)
    return lr, d_model[0], nheads[0], nlayers, d_ff[0]

for i in range(runs):
    lr, d_model, nheads, nlayers, d_ff = sample()
    #margin, lr, dropout, batch_size = 1.0, 0.25, 0.1, 50
    main.randomSearch(lr, d_model, nheads, nlayers, d_ff, str(i))