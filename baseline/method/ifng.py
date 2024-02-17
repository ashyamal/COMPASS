import qnorm
import pandas as pd
import numpy as np
import warnings

'''
Paper:
Ayers M, Lunceford J, Nebozhyn M, Murphy E, Loboda A, Kaufman DR, Albright A, Cheng JD, Kang SP, Shankaran V, et al: IFN-gamma-related mRNA profile predicts clinical response to PD-1 blockade. J Clin Invest 2017, 127:2930-2940.

Marker:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5531419/bin/jci-127-91190-g010.jpg

Calculation:
After performance of quantile normalization, a log10 transformation was applied, and signature scores were calculated by averaging of the included genes for the IFN-γ (6-gene) and expanded immune (18-gene) signatures.
'''

MARKERS = ['CXCL10', 'IFNG', 'CXCL9', 'IDO1', 'HLA-DRA', 'STAT1']
MARKERS_EIGS = ['CD3D','IDO1','CIITA','CD3E','CCL5','GZMK','CD2','HLA-DRA','CXCL13',
              'IL2RG','NKG7','HLA-E','CXCR6','LAG3','TAGAP','CXCL10','STAT1','GZMB', ]
NAME = 'IFNg'

def get_ifng(df_tpm, expand_markers = False, ncpus=8):
    """
    Calculates the IFN-gammar score based on predifined 6 and 18 markers.
    
    Inputs:
    (1) df_tpm: A pandas dataframe (RNA TPM values) containing samples as rows and individual genes as columns and
    ====================
    
    Calculates interger score between 0 and the number of input features
    
    Outputs the original df with the score for each sample included as a new row (termed "IMPRES")
    
    """
    if expand_markers:
        markers = MARKERS_EIGS
    else:
        markers = MARKERS

    df_tpm.columns
    markers_used = list(set(markers) & set(df_tpm.columns))
    markers_unused =  list(set(markers) - set(markers_used))
    warnings.warn('Markers of %s are missed and not used.' % markers_unused)
    
    ## quantile_normalize for each sample
    df_tpm_norm = qnorm.quantile_normalize(df_tpm, ncpus=ncpus, axis = 1)  
    df_tpm_norm_log10 = np.log10(df_tpm_norm + 1)

    ifng = df_tpm_norm_log10[markers_used].mean(axis=1)
    ifng.name = NAME

    return ifng
