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
from torchvision.transforms import transforms
from torch.nn.functional import normalize
from torch.distributions import Beta



def tolist(x):
    return x.tolist()


class TCGAData(Dataset):
    def __init__(self, df_tpm,  df_cancer, df_tmb, df_msi, df_cnv, augmentor):
        '''
        df_tpm: TPM matrix
        df_cancer: cancer type

        df_tmb: tumor mutation burden
        df_msi: msi burden
        df_cnv: cnv burden

        augmentor: MixupNomralAugmentor
        '''

        self.feature_name = df_tpm.columns
        self.patient_name = df_tpm.index
        
        X = torch.tensor(df_tpm.values,dtype=torch.float32).clone().detach()
        self.X = normalize(torch.log2(X + 1.), p=2.0, dim = 0)

        #self.X = torch.log2(X + 1.)
        
        df_cancer_type = df_cancer.copy()
        df_cancer_type['idx'] = list(range(len(df_cancer)))
        same_cancer_idx_map = df_cancer_type.groupby('cancer_type')['idx'].apply(tolist).to_dict()
        
        self.cancer_type = df_cancer_type['cancer_type'].values
        self.same_cancer_idx_map = same_cancer_idx_map

        ## mutation data
        tmb_scaler = MinMaxScaler()
        msi_scaler = MinMaxScaler()
        cnv_scaler = MinMaxScaler()
        
        tmb = tmb_scaler.fit_transform(df_tmb)
        msi = msi_scaler.fit_transform(df_msi)
        cnv = cnv_scaler.fit_transform(df_cnv)
        self.tmb = torch.tensor(tmb,dtype=torch.float32)   
        self.msi = torch.tensor(msi,dtype=torch.float32) #contain nan
        self.cnv = torch.tensor(cnv,dtype=torch.float32)
        

        self.tmb_scaler = tmb_scaler
        self.msi_scaler = msi_scaler
        self.cnv_scaler = cnv_scaler

        
        self.augmentor = augmentor 
        self.df_tpm = df_tpm
        self.df_cancer = df_cancer
        self.df_tmb = df_tmb
        self.df_msi = df_msi
        self.df_cnv = df_cnv
        

    def __len__(self):
        return len(self.X)

    
    def __getitem__(self, idx):

        ## anchor sample
        a = self.X[idx]
        cancer = self.cancer_type[idx]
        same_cancer_idx = self.same_cancer_idx_map[cancer]
        
        ## positive sample
        p = self.augmentor.augment(a)[0]
        
        ## negative sample
        neg_idx = np.random.choice(same_cancer_idx)
        n = self.X[neg_idx]

        x = [a, p, n]
        y = [self.tmb[idx], self.msi[idx], self.cnv[idx]]

        return x, y



class GeneData(Dataset):
    def __init__(self, df_tpm):
        '''
        df_tpm: TPM matrix
        '''
        self.feature_name = df_tpm.columns
        self.patient_name = df_tpm.index
        
        X = torch.tensor(df_tpm.values,dtype=torch.float32).clone().detach()
        self.X = normalize(torch.log2(X + 1.), p=2.0, dim = 0)
        self.df_tpm = df_tpm

    def __len__(self):
        return len(self.X)

    
    def __getitem__(self, idx):
        ## anchor sample
        x = self.X[idx]
        return x





class ITRPData(Dataset):
    pass