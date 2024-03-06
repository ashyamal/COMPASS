#!/home/was966/micromamba/envs/responder/bin/python
#sbatch --mem 64G -c 4 -t 100:00:00 -p gpu_quad --gres=gpu:rtx8000:1 ./loo_pft_dense16_f.py

import sys

sys.path.insert(0, '/home/was966/Research/mims-conceptor/')
from conceptor.utils import plot_embed_with_label
from conceptor import PreTrainer, FineTuner, loadconceptor #, get_minmal_epoch
from conceptor.utils import plot_embed_with_label,plot_performance, score
from conceptor.tokenizer import CANCER_CODE

import os
from tqdm import tqdm
from itertools import chain
import pandas as pd
import numpy as np
import random, torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale=1.3)
import warnings
warnings.filterwarnings("ignore")

def onehot(S):
    assert type(S) == pd.Series, 'Input type should be pd.Series'
    dfd = pd.get_dummies(S, dummy_na=True)
    nanidx = dfd[dfd[np.nan].astype(bool)].index
    dfd.loc[nanidx, :] = np.nan
    dfd = dfd.drop(columns=[np.nan])*1.
    cols = dfd.sum().sort_values(ascending=False).index.tolist()
    dfd = dfd[cols]
    return dfd


pretrainer = loadconceptor('../../checkpoint/latest/pretrainer.pt')
data_path = '../../00_data/'
df_label = pd.read_pickle(os.path.join(data_path, 'ITRP.PATIENT.TABLE'))
df_tpm = pd.read_pickle(os.path.join(data_path, 'ITRP.TPM.TABLE'))
df_tpm.shape, df_label.shape


df_task = onehot(df_label.response_label)
size = df_label.groupby('cohort').size()
size = size.index + "\n(n = " + size.astype(str) + ")"

dfcx = df_label.cancer_type.map(CANCER_CODE).to_frame('cancer_code').join(df_tpm)
dfy = df_task
cohorts = df_label.groupby('cohort').size().sort_values().index.tolist()

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()


mode = 'PFT'
seed = 42
params = {'mode': mode,
            'seed':seed,
            'lr': 1e-2,
            'device':'cuda',
            'weight_decay': 1e-4,
            'batch_size':8,
            'max_epochs': 100,
            'task_loss_weight':1,
            'load_decoder':False,
            'task_dense_layer': [16],
            'task_batch_norms':True,
            'entropy_weight': 0.0,
            'with_wandb': False,
            'save_best_model':False,
            'verbose': False}


work_dir = './LOO_%s_%s' % (mode, seed)
if not os.path.exists(work_dir):
    os.makedirs(work_dir)


res = []
for cohort in cohorts:
    cohort_idx = df_label[df_label['cohort']==cohort].index
    cohort_X = dfcx.loc[cohort_idx]
    cohort_y = dfy.loc[cohort_idx]

    if len(cohort_idx) > 100:
        params['batch_size'] = 16
    else:
        params['batch_size'] = 8

        
    test_cohort_name = cohort
    predict_res = []
    for train_idx, test_idx in loo.split(cohort_idx):
        print(len(train_idx), len(test_idx))
        train_X = cohort_X.iloc[train_idx]
        train_y = cohort_y.iloc[train_idx]
        test_X = cohort_X.iloc[test_idx]    
        test_y = cohort_y.iloc[test_idx]

        pretrainer = pretrainer.copy()
        finetuner = FineTuner(pretrainer, **params, 
                              work_dir= work_dir, 
                              task_name = 'Test_on_%s' % test_cohort_name, 
                              task_type='f')
        
        finetuner = finetuner.tune(dfcx_train = train_X,
                                   dfy_train = train_y, min_mcc=0.8)
        _, pred_testy = finetuner.predict(test_X, batch_size = 16)
        pred_testy['best_epoch'] = finetuner.best_epoch

        pred_testy['cohort'] = test_cohort_name
        pred_testy['n_trainable_params'] = finetuner.count_parameters()
        pred_testy['mode'] = mode
        pred_testy['seed'] = seed
        pred_testy['batch_size'] = params['batch_size']
        pred_testy['task_dense_layer'] = str(params['task_dense_layer'])
        predict_res.append(pred_testy)


    df_pred = pd.concat(predict_res)
    dfp = cohort_y.join(df_pred)
    dfp.to_csv(os.path.join(work_dir, '%s.csv' % test_cohort_name ))
    
    y_true, y_prob, y_pred = dfp['R'], dfp[1], dfp[[0, 1]].idxmax(axis=1)
    s2 = score(y_true, y_prob, y_pred)
    dfs = pd.DataFrame([s2], columns = ['ROC', 'PRC', 'F1', 'ACC'], index = ['Test'])
    dfs['cohort'] = test_cohort_name
    dfs['mode'] = mode
    dfs['seed'] = seed
    
    fig = plot_performance(y_true, y_prob, y_pred)
    fig.suptitle('Leave-One-Out in %s' % test_cohort_name, fontsize=16)
    fig.savefig(os.path.join(work_dir, 'leave_one_out_%s.jpg' % test_cohort_name))

    res.append(dfs)

dfres = pd.concat(res) #.reset_index(drop=True)
dfres.to_csv(os.path.join(work_dir, 'performance.tsv'), sep='\t')