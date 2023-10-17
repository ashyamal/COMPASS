#!/home/was966/micromamba/envs/RNA/bin/python

#sbatch --mem 64G -c 20 -t 5-12:00 -p long  ./run_Freeman.py

import pandas as pd
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from pandarallel import pandarallel

import seaborn as sns
sns.set(style = 'white', font_scale=1.5)
# pandarallel.initialize(nb_workers = 8, progress_bar=True) # initialize(36) or initialize(os.cpu_count()-1)


import sys
sys.path.insert(0, '/home/was966/Research/PSOG/')
from gdd.correlation import pairwise_correlation
from gdd.combination import pairwise_combination
from gdd.score import Xy_score
from gdd.plot import plot_batch
BEST = ['PIK3CD', 'TOLLIP']

data_path = '/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/TCGA-ITRP-MERGE/735/'

tcga_tpm = pd.read_pickle(os.path.join(data_path, 'TPM.TCGA.TABLE'))
itrp_tpm = pd.read_pickle(os.path.join(data_path, 'TPM.ITRP.TABLE'))
itrp_tpm_crt = pd.read_pickle(os.path.join(data_path, 'TPM.ITRP.TABLE.CORRECT'))

tcga_patient = pd.read_pickle(os.path.join(data_path, 'PATIENT.TCGA.TABLE'))
tcga_patient['cohort'] = tcga_patient.cancer_type
itrp_patient = pd.read_pickle(os.path.join(data_path, 'PATIENT.ITRP.TABLE'))
gene = pd.read_pickle(os.path.join(data_path, 'GENE.TABLE'))

dfp1 = itrp_tpm.join(itrp_patient.cohort)
dfp2 = tcga_tpm.join(tcga_patient.cohort)
dfp2 = dfp2[dfp2.cohort == 'TCGA-SKCM']
dfp = pd.concat([dfp1, dfp2])

df = itrp_tpm
gene_values = df.values

# Subtract each other and generate new features
combinations = [(gene1, gene2) for gene1 in df.columns for gene2 in df.columns if gene1 != gene2]
chunk_size = 5000000
gene_combinations_list = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)] 

for i, gene_combinations in enumerate(gene_combinations_list):
    combination_features = {}
    for gene1, gene2 in tqdm(gene_combinations, ascii=True):
        combination_feature = gene_values[:, df.columns.get_loc(gene1)] - gene_values[:, df.columns.get_loc(gene2)]
        combination_features[f'{gene1} - {gene2}'] = combination_feature
    result_df = pd.DataFrame(combination_features, index=df.index)
    file = os.path.join(data_path, 'itrp_tpm_combination_%s.pkl' % str(i).zfill(10))
    result_df.to_pickle(file)
    del result_df, combination_features


def get_score(dfxy, y_col):
    scores = []
    for cohort in dfxy.cohort.unique():
        dfxy_c = dfxy[dfxy.cohort == cohort]
        dfxy_c = dfxy_c[dfxy_c.columns[:-1]]
        prc = Xy_score(dfxy_c, y_col, method='prc', n_jobs=18)
        roc = Xy_score(dfxy_c, y_col, method='roc', n_jobs=18)
        score = prc.join(roc)
        score = score.sort_values('roc')
        score['cohort'] = cohort
        scores.append(score)
    return scores


from glob import glob
file_list = glob(os.path.join(data_path, 'itrp_tpm_combination_*.pkl'))
for file in file_list:
    save_file = file + '.score.csv'
    dfc = pd.read_pickle(file)
    dfc.index = itrp_tpm.index
    y_col = 'Freeman_response' #Overall_survival
    dfxy = dfc.join(itrp_patient[[y_col, 'cohort']])
    dfyc = dfxy[[y_col, 'cohort']]
    
    chunk_size = 100000
    combinations = dfc.columns.tolist()
    gene_combinations_list = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)] 
    
    scores = []
    for gene_combinations in gene_combinations_list:
        dfxy_chunk = dfxy[gene_combinations]
        dfxy_chunk = dfxy_chunk.join(dfyc)
        auc_scores = get_score(dfxy_chunk, y_col)
        df = pd.concat(auc_scores, axis=0)
        prc = df.reset_index().set_index(['gene_name', 'cohort'])['prc'].unstack().mean(axis=1).to_frame(name='prc')
        roc = df.reset_index().set_index(['gene_name', 'cohort'])['roc'].unstack().mean(axis=1).to_frame(name='roc')
        score = prc.join(roc)
        scores.append(score)

    dfs = pd.concat(scores)
    dfs['mean'] = dfs.mean(axis=1)
    dfs = dfs.sort_values('mean',ascending=False)
    dfs.to_csv(save_file)















