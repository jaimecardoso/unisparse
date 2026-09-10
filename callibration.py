import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import norm

def k2ECE(y,yhat,conf, bandwidth = 0.1):
    dist = pairwise_distances(conf)
    N = y.shape[0]
    lacc = np.zeros((N,1))
    for i in range(N):
        votes = norm.pdf(dist[i,:], 0, bandwidth)
        lacc_dem = np.sum(votes)
        lacc_num = np.sum(votes[y[:,0]==yhat[:,0]])
        lacc[i] = lacc_num/lacc_dem      
    kece = np.sum(np.abs(lacc-conf))/N
    return kece

def k2ECE_memoryEfficient (y,yhat,conf, bandwidth = 0.1):
    N = y.shape[0]
    lacc = np.zeros((N,1))
    for i in range(N):
        dist = pairwise_distances(conf[[i],:], conf)
        votes = norm.pdf(dist[0,:], 0, bandwidth)
        lacc_dem = np.sum(votes)
        lacc_num = np.sum(votes[y[:,0]==yhat[:,0]])
        lacc[i] = lacc_num/lacc_dem      
    kece = np.sum(np.abs(lacc-conf))/N
    return kece
