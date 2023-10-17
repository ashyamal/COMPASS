import numpy as np
import pandas as pd


def fpkm2tpm(fpkm_matrix):
    '''
    fpkm_matrix: rows: genes, columns: samples/patients
    '''
    _fpkm2tpm = lambda fpkm: np.exp(np.log(fpkm) - np.log(np.sum(fpkm)) + np.log(1e6))
    tpm_matrix = fpkm_matrix.apply(_fpkm2tpm, axis=0)
    return tpm_matrix