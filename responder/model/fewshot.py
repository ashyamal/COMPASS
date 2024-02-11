# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:34:45 2024

@author: Wanxiang Shen
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as prc_auc_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score


def score(y_true, y_prob, y_pred):
    
    select = ~y_true.isna() 
    y_prob = y_prob[select]
    y_true = y_true[select]
    y_pred = y_pred[select] #.map({'NR':0, 'R':1})    

    if len(y_true.unique()) == 1:
        roc = np.nan
        prc = np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
        _precision, _recall, _ = precision_recall_curve(y_true, y_prob)
        prc = prc_auc_score(_recall, _precision)
        
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    cf_matrix = confusion_matrix(y_true, y_pred, labels = [1,0])
    tp_and_fn = cf_matrix.sum(1)
    tp_and_fp = cf_matrix.sum(0)
    tp = cf_matrix.diagonal()
    precision = tp / tp_and_fp
    recall = tp / tp_and_fn
    precision, recall = precision[0], recall[0]

    return precision, recall, roc, prc, f1, acc



# Define the Prototypical Network without fine-tuning
class PrototypeNetNoFT:

    def __init__(self, feature_norm = True):
        self.prototype_class_map = {'PD':'NR', 'SD':'NR',
                                    'PR':'R', 'CR':'R'}
        self.feature_norm = feature_norm


    def fit(self, support_set):
        '''
        support_set: the last column is the RECIST label column
        '''
        self.recist_col = support_set.columns[-1]
        self.feature_col = support_set.columns[:-1]
        
        unique_recist_labels = support_set[self.recist_col].unique()
        out_recist = set(unique_recist_labels) - set(self.prototype_class_map.keys())
        assert len(out_recist) == 0, 'Unepxected RECIST labels: %s' % out_recist
        
        prototype_features = support_set.groupby('RECIST').mean()
        prototype_types = prototype_features.index
        prototype_representation = torch.tensor(prototype_features.values)
        
        if self.feature_norm:
            prototype_representation = F.normalize(prototype_representation, p=2, dim=1)

        self.prototype_representation = prototype_representation
        self.prototype_types = prototype_types
        
        return self

    
    def transform(self, query_set):
        '''
        query_set: the last column is the RECIST label column
        '''
        recist_col = self.recist_col
        assert recist_col == self.recist_col, '%s is missing!' % self.recist_col

        query_set_features = torch.tensor(query_set[self.feature_col].values)
        if self.feature_norm:
            query_set_features = F.normalize(query_set_features, p=2, dim=1)
        
        similarities = torch.mm(query_set_features, self.prototype_representation.T) # X*W + B
        probabilities = F.softmax(similarities, dim=1)
        probabilities = probabilities.detach().numpy()
        
        dfprob = pd.DataFrame(probabilities, index = query_set.index, columns = self.prototype_types)
        dfpred = dfprob.idxmax(axis=1).map(self.prototype_class_map)
        dftrue = query_set[self.recist_col].map(self.prototype_class_map)
        
        ## probability for R
        dfprob2 = dfprob.copy()
        dfprob2.columns = dfprob2.columns.map(self.prototype_class_map)
        dfprob2 = dfprob2.T.reset_index().groupby('RECIST').sum().T
        dfprob2_r = dfprob2['R']
        
        scores = score(dftrue.map({'R':1, 'NR':0}), dfprob2_r, dfpred.map({'R':1, 'NR':0}))
        dfres = dfprob.join(dfprob2).join(dfpred.to_frame('Pred')).join(dftrue.to_frame('True'))

        return dfres, scores
        

        
