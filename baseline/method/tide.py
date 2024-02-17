#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tidepy.pred import TIDE

'''
http://tide.dfci.harvard.edu/download/
https://github.com/liulab-dfci/TIDEpy/tree/master

Jiang P, Gu S, Pan D, Fu J, Sahu A, Hu X, Li Z, Traugh N, Bu X, Li B, et al: Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response. Nat Med 2018, 24:1550-1558.

'''

NAME = 'TIDE'

# def get_tide(df_tpm, cancer='Melanoma',):
#     df_tpm_log2 = np.log2(df_tpm + 1)
#     control = df_tpm_log2.mean()
#     df_tpm_norm = df_tpm_log2 - control 
#     df_tpm_input = df_tpm_norm.T
#     result = TIDE(df_tpm_input, cancer=cancer,pretreat=True,vthres=0.,     
#                     ignore_norm=True,
#                     force_normalize=False,)

#     '''
#     ['No benefits', 'Responder', 'TIDE', 'IFNG', 'MSI Score', 'CD274', 'CD8',
#     'CTL.flag', 'Dysfunction', 'Exclusion', 'MDSC', 'CAF', 'TAM M2', 'CTL']
#     '''
#     return result


def get_tide(df_tpm, df_tpm_control, cancer='Melanoma',):
    #df_tpm_log2 = np.log2(df_tpm + 1)
    #df_tpm_norm = df_tpm_log2 #@ - df_tpm_control 
    df_tpm_input = df_tpm.T
    result = TIDE(df_tpm_input, cancer=cancer,pretreat=False,vthres=0.,     
                    ignore_norm=False,
                    force_normalize=True,)

    '''
    ['No benefits', 'Responder', 'TIDE', 'IFNG', 'MSI Score', 'CD274', 'CD8',
    'CTL.flag', 'Dysfunction', 'Exclusion', 'MDSC', 'CAF', 'TAM M2', 'CTL']
    '''
    return result
