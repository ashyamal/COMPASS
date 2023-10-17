# -*- coding: utf-8 -*-
"""

Calculate the correlation
"""


import numpy as np
from tqdm import tqdm

from gdd.utils.multiproc import MultiProcessUnorderedBarRun
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr


def pearson(x, y):
    correlation = pearsonr(x, y)[0]
    return correlation

def mutual_info(x, y):
    correlation = mutual_info_regression(x.reshape(-1, 1), y, random_state = 123)
    return correlation


def _yield_combinations(N):
    for i1 in range(N):
        for i2 in range(i1):
            yield (i1,i2)
            
def _calculate(i1, i2):
    x1 = data[:, i1]
    x2 = data[:, i2]
    ## dropna
    X = np.stack([x1,x2], axis=1)
    X = X[~np.isnan(X).any(axis=1)]
    x1 = X[:, 0]
    x2 = X[:, 1]
    if x1.any():
        dist = func(x1, x2)
    else:
        dist = np.nan
    return (i1, i2, dist)

def _fuc(x):
    i1, i2  = x
    return _calculate(i1, i2)


def pairwise_correlation(npydata, n_cpus=8, method='pearson'):
    """
    parameters
    ---------------
    method: {'pearson', 'mutual_info'}
    npydata: np.array or np.memmap, Note that the default we will calcuate the vector's correlation instead of sample's correlation, if you wish to calculate correlation between samples, you can pass data.T instead of data

    Usage
    --------------
    >>> import numpy as np
    >>> data = np.random.random_sample(size=(10000,10)
    >>> corr_matrix = pairwise_correlation(data)
    >>> corr_matrix.shape
    >>> (10,10)  
    """    
    global data, func

    if method=='pearson':
        func = pearson
    else:
        func = mutual_info
    data = npydata
    N = data.shape[1]
    lst = list(_yield_combinations(N))
    res = MultiProcessUnorderedBarRun(_fuc, lst, n_cpus=n_cpus)
    corr_matrix = np.zeros(shape = (N,N))
    for x,y,v in tqdm(res,ascii=True):
        corr_matrix[x,y] = v
        corr_matrix[y,x] = v

    return corr_matrix

    
    
if __name__ == '__main__':
    
    import numpy as np
    import pandas as pd
    from umap import UMAP
    import matplotlib.pyplot as plt
    
    X = np.random.random_sample(size=(1000000,40))
    corrmatrix = pairwise_correlation(X, n_cpus=6)
    distmatrix = 1. - corrmatrix
    embedding = UMAP(metric = 'precomputed',random_state=10)
    df = pd.DataFrame(embedding.fit_transform(distmatrix))
    ax = plt.plot(df[0],df[1], 'bo')