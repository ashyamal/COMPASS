# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:05:13 2023

@author: Wanxiang Shen
"""

import os
import numpy as np
import pandas as pd
from glob import glob
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.nn.functional import normalize
from torch.distributions import Beta

        
def tolist(x):
    return x.tolist()

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class NoScaler:
    def __init__(self):
        pass
    def fit(self, dfx):
        return self
        
    def transform(self, dfx):
        return dfx

    def fit_transform(self, dfx):
        return dfx

    def inverse_transform(self, dfx):
        return dfx


class TCGAData(Dataset):
    
    def __init__(self, df_tpm,  df_task, augmentor, K = 1):
        '''
        df_tpm: TPM matrix
        df_task: task dataframe, one-hot encoding for classification task
        augmentor: MixupNomralAugmentor
        K: float
        '''

        self.feature_name = df_tpm.columns
        self.patient_name = df_tpm.index
        self.task_cols = df_task.columns
        self.task_dim = len(df_task.columns)
        
        X = torch.tensor(df_tpm.values,dtype=torch.float32).clone().detach()
        self.X = X

        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(df_task.values)
        self.y = torch.tensor(y, dtype=torch.float32).clone().detach()
        self.y_scaler = y_scaler

        self.K = K

        dist = torch.cdist(X, X)
        #print(dist.min())
        
        if K in [-1, 1, 0]:
            knn_value, knn_idx = dist.topk(len(X), largest=False)
        else:
            knn_value, knn_idx = dist.topk(int(len(X)*K), largest=False)
        self.knn_value = knn_value[:, 1:]
        self.knn_idx = knn_idx[:, 1:]
        
        self.augmentor = augmentor 



    def __len__(self):
        return len(self.X)

    
    def __getitem__(self, idx):

        ## anchor sample
        a = self.X[idx]
        topK_idx = self.knn_idx[idx]
        
        ## positive sample augmentation
        xp = self.augmentor.augment_p(a)[0]
        
        ## negative sample
        neg_idx = np.random.choice(topK_idx)
        n = self.X[neg_idx]

        ya = self.y[idx]
        yp = ya.clone()
        yn = self.y[neg_idx]

        xa = self.augmentor.augment_a(a)[0]
        xn = self.augmentor.augment_n(n)[0]

        x = [xa, xp, xn]
        y = [ya, yp, yn]
        
        return x, y



class GeneData(Dataset):
    def __init__(self, df_tpm):
        '''
        df_tpm: TPM matrix
        '''
        self.feature_name = df_tpm.columns
        self.patient_name = df_tpm.index
        
        X = torch.tensor(df_tpm.values,
                         dtype=torch.float32).clone().detach()
        self.X = X
        self.df_tpm = df_tpm

    def __len__(self):
        return len(self.X)

    
    def __getitem__(self, idx):
        ## anchor sample
        x = self.X[idx]
        return x





class ITRPData(Dataset):
    def __init__(self, df_tpm,  df_task):
        
        '''
        df_tpm: TPM matrix
        df_task: task dataframe, one-hot encoding for classification task (response label)
        '''

        self.feature_name = df_tpm.columns
        self.patient_name = df_tpm.index
        self.task_cols = df_task.columns #['NR', 'R',]
        self.task_dim = len(df_task.columns)
        
        X = torch.tensor(df_tpm.values, dtype=torch.float32).clone().detach()
        self.X = X

        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(df_task.values)
        self.y = torch.tensor(y, dtype=torch.float32).clone().detach()
        self.y_scaler = y_scaler

        self.responder_idx = torch.where(self.y[:, 1] == 1.)[0]
        self.nonresponder_idx = torch.where(self.y[:, 1] == 0.)[0]

    
    def __len__(self):
        return len(self.X)

    
    def __getitem__(self, idx):

        ya = self.y[idx]
        #print(y)
        ## case responder
        if ya[1] == 1:
            pos_idx = np.random.choice(self.responder_idx)
            neg_idx = np.random.choice(self.nonresponder_idx) 
        ## case non-responder
        else:
            pos_idx = np.random.choice(self.nonresponder_idx)
            neg_idx = np.random.choice(self.responder_idx)         


        ## anchor sample
        xa = self.X[idx]
        xp = self.X[pos_idx]
        xn = self.X[neg_idx]
        
        x = [xa, xp, xn]

        yp = self.y[pos_idx]
        yn = self.y[neg_idx]
        
        y = [ya, yp, yn]
        
        return x, y