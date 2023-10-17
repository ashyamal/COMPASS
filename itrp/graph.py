import pandas as pd
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn as sns
sns.set(style = 'white', font_scale=1.5)




class Grapher:
    '''
    Convert tabular data into graph data
    '''
    
    def __init__(self, method = 'pearson', n_neighbours = 15, connect_most_unsimilar = True):

        self.method = method
        self.n_neighbours = n_neighbours
        self.connect_most_unsimilar = connect_most_unsimilar


    def fit(self, dfx):
        '''
        fit tabular data
        '''
        pass


    def transform(dfx):
        '''
        Transform the tabular data into graph data
        '''
        pass



