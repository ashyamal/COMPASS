#!/home/was966/micromamba/envs/responder/bin/python
#sbatch --mem 64G -c 4 -t 100:00:00 -p gpu_quad --gres=gpu:teslaV100s:1 ./ft_tmb.py



import os
from tqdm import tqdm
from itertools import chain
import pandas as pd
import numpy as np
import random, torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale=1.5)

import sys
sys.path.insert(0, '/home/was966/Research/mims-conceptor/')
from conceptor.utils import plot_embed_with_label
from conceptor import PreTrainer, FineTuner, Adapter, loadconceptor
from conceptor.utils import plot_embed_with_label, score
from conceptor.tokenizer import CANCER_CODE

def onehot(S):
    assert type(S) == pd.Series, 'Input type should be pd.Series'
    dfd = pd.get_dummies(S, dummy_na=True)
    nanidx = dfd[dfd[np.nan]].index
    dfd.loc[nanidx, :] = np.nan
    dfd = dfd.drop(columns=[np.nan])*1.
    cols = dfd.sum().sort_values(ascending=False).index.tolist()
    dfd = dfd[cols]
    return dfd
from conceptor import PreTrainer, FineTuner, Adapter, loadconceptor


data_path = '/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/TCGA-ITRP-GENESET-MERGE/15672//data'

df_tpm = pd.read_pickle(os.path.join(data_path,  'TCGA.TPM.TABLE'))
df_label = pd.read_pickle(os.path.join(data_path, 'TCGA.PATIENT.TABLE'))
df_label['ptmb'] = df_label['Noushin_pTMB']
df_label['tmb'] = df_label['Noushin_TMB']

mpath = '../checkpoint/latest/pretrainer.pt'
# load the pretrained model as a feature extractor
pretrainer = loadconceptor(mpath)


task = 'tmb'

df_label = df_label[~df_label[task].isna()]
dfy = df_label[[task]]
dfy = (dfy - dfy.min()) / (dfy.max() - dfy.min())

dfcx = df_label.cancer_type.apply(lambda x:x.split('-')[-1]).map(CANCER_CODE).to_frame('cancer_code').join(df_tpm)
dfcx = dfcx.loc[dfy.index]

test_idx = dfcx.groupby('cancer_code').sample(frac=0.1).index
train_idx = dfcx.index.difference(test_idx)
print(test_idx.shape, train_idx.shape)


dfcx_train = dfcx.loc[train_idx]
dfy_train = dfy.loc[train_idx]

dfcx_test = dfcx.loc[test_idx]
dfy_test = dfy.loc[test_idx]

ada = Adapter(pretrainer, adp_feature='TMB', lr = 1e-2, weight_decay =0.,
              batch_size = 64, epochs = 500, patience = 20)
ada.adapt(dfcx_train, dfy_train, dfcx_test = dfcx_test, dfy_test = dfy_test)


dfp = pd.DataFrame(ada.performace, columns=  ['epochs', 'train_loss', 'test_loss']).set_index('epochs')
dfp.to_csv('training_history_%s.csv' % task)
fig, ax = plt.subplots(figsize=(8,6))
dfp.plot(ax=ax)
fig.tight_layout()
fig.savefig('training_history_%s.jpg' % task)


dfe, dfp = ada.pretrainer.predict(dfcx, batch_size = 16)
dfeo, dfp = pretrainer.predict(dfcx, batch_size = 16)
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(dfe.TMB, dfy[task])
fig.tight_layout()
fig.savefig('true_vs_pred_%s.jpg' % task)

ada.pretrainer.save('../checkpoint/latest/pretrainer_%s.pt' % task)


