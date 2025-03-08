import numpy as np
from scipy.stats import uniform, loguniform, truncnorm
import random
import main

def sample():
    #lr = np.random.uniform(1e-5,1e-1)
    params = {}
    params['lr'] = random.sample([1e-3, 2e-3, 1e-4, 2e-4, 1e-5, 2e-5, 1e-6, 2e-6], 1)[0]
    params['wd'] = random.sample([0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005],1)[0]
    params['d_model'] = random.sample([16, 32, 64, 128, 256, 512],1)[0]
    params['n_heads'] = random.sample([2,4,8,16],1)[0]
    params['n_layers'] = np.random.randint(low=1,high=5)
    params['d_ff'] = random.sample([128,256,512,1024,2048],1)[0]
    params['batch_size']= random.sample([16,32,64,128,256,512], 1)[0]
    params['dropout'] = random.uniform(0,0.5)
    params['margin'] = truncnorm.rvs(0.1,1)
    params['epsilon'] = random.sample([1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],1)[0]
    return params#lr[0], batch_size[0], round(margin,2), nlayers, wd, d_model, nheads, d_ff