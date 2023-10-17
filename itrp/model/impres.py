'''
Paper: Auslander N, Zhang G, Lee J S, et al. Robust prediction of response to immune checkpoint blockade therapy in metastatic melanoma[J]. Nature medicine, 2018, 24(10): 1545-1549.
https://github.com/JasonACarter/IMPRES_Correspondence/blob/master/Code/IMPRES_Correspondence.ipynb

Calculation:
IMPRES score based on logical comparison of quantile-normalized expression of pre-defined 15 checkpoint gene pairs

'''

import numpy as np
import pandas as pd
import qnorm

#The original IMPRES features given as gene1_gene2, defined as gene1>gene2:
MARKERS = ['CD274_VSIR','CD86_CD200','CD40_CD274','CD28_CD276', 
           'CD40_CD28','TNFRSF14_CD86','CD27_PDCD1','CD28_CD86','CD40_CD80',
           'CD40_PDCD1', 'CD80_TNFSF9','CD86_HAVCR2','CD86_TNFSF4', 
           'CTLA4_TNFSF4','PDCD1_TNFSF4']

NAME = 'IMPRES'

def get_impres(df_tpm):
    """
    Calculates the IMPRES score based on predifined 15 gene-gene pairs.
    
    Inputs:
    (1) df_tpm: A pandas dataframe containing samples as rows and individual genes as columns and
    ====================
    
    Calculates interger score between 0 and the number of input features
    
    Outputs the original df with the score for each sample included as a new row (termed "IMPRES")
    
    """
    log2tpm_df = np.log2(df_tpm + 1)
    pidx = log2tpm_df.index #pateint index
    fidx = log2tpm_df.columns #
    
    pairwise_interactions = pd.DataFrame(np.empty((log2tpm_df.shape[0],len(MARKERS)),
                                                  dtype=object),columns=MARKERS,index= pidx)
    for feature in MARKERS: #for each provided feature
        gene1,gene2=feature.split('_') #split provided gene1_gene2, where feature is gene1>gene2 by convention
        pairwise_interactions[feature] = np.array(1*(log2tpm_df[gene1].astype(float) > log2tpm_df[gene2].astype(float))) 
        #for each sample, calculate if feature is present
    
    IMPRES = pairwise_interactions.sum(axis=1) #sum present features for each sample
    IMPRES.name = NAME
    
    return IMPRES

