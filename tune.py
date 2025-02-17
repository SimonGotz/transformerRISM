import numpy as np
from scipy.stats import uniform, loguniform, truncnorm
import random
import main

def sample():
    #lr = np.random.uniform(1e-5,1e-1)
    lr = random.sample([1e-3, 2e-3, 1e-4, 2e-4, 1e-5, 2e-5, 1e-6, 2e-6], 1)
    #wd = [0.1, 0.01, 0.05, 0.001, 0.005]
    #d_model = random.sample([8, 16, 32, 64, 128],1)
    #nheads = random.sample([2,4,8],1)
    nlayers = np.random.randint(low=1,high=4)
    #d_ff = random.sample([128,256,512,1024,2048],1)
    batch_size = random.sample([16,32,64,128,256,512,1024], 1)
    #margin = random.uniform(0,1)
    margin = truncnorm.rvs(0.1,1)
    return lr[0], batch_size[0], round(margin,2), nlayers