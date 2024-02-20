#!/home/was966/micromamba/envs/responder/bin/python
#sbatch --mem 64G -c 4 -t 12:00:00 -p gpu_quad --gres=gpu:rtx8000:1 ./01_cohort_to_cohort_transfer.py

import sys

sys.path.insert(0, '/home/was966/Research/mims-conceptor/')
from conceptor.utils import plot_embed_with_label
from conceptor import PreTrainer, FineTuner, loadconceptor #, get_minmal_epoch
from conceptor.utils import plot_embed_with_label, score
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

def plot_performance(y_true, y_prob, y_pred):
    
    from sklearn.metrics import confusion_matrix
    roc, prc, f1, acc = score(y_true, y_prob, y_pred)
    dfp = pd.DataFrame([y_true, y_prob, y_pred]).T
    dfp.columns = ['Label', 'Pred. Prob.', 'Pred_label']
    dfp.Label = dfp.Label.map({0:'NR', 1:'R'})
    cf_matrix = confusion_matrix(y_true, y_pred, labels = [1, 0])
    
    tp_and_fn = cf_matrix.sum(1)
    tp_and_fp = cf_matrix.sum(0)
    tp = cf_matrix.diagonal()
    precision = tp / tp_and_fp
    recall = tp / tp_and_fn
    precision, recall = precision[0], recall[0]
    
    palette = sns.color_palette('rainbow', 12)
    
    colors = palette.as_hex()
    boxpalette = {'NR':colors[1], 'R':colors[-3]}
    swarmpalette = {'NR':colors[2], 'R':colors[-3]}
    
    fig, axes= plt.subplots(ncols=3, nrows=1, figsize=(8,3),
                            gridspec_kw={'width_ratios': [4, 1, 3]},        
                            sharex=False, sharey=False)
    
    ax2, ax1, ax3 =axes
    
    ###################################
    order = ['R', 'NR']
    sns.boxplot(x = 'Label', y = 'Pred. Prob.',data = dfp,  fliersize = 0., width = 0.5,
                order = order,
                ax=ax1, palette = boxpalette, saturation = 0.8, fill= False)
    sns.stripplot(dfp, x = 'Label', y = 'Pred. Prob.', ax=ax1, size=3, 
                  order = order,
                  palette = boxpalette, edgecolor = 'k', linewidth = 0.1)
    
    ax1.xaxis.tick_bottom() # x axis on top
    ax1.xaxis.set_label_position('bottom')
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', labelrotation=60)
    
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.yaxis.tick_left() # x axis on top
    ax1.spines[['right', 'top']].set_visible(False)
    
    
    ###################################
    group_names = ['True R','False NR','False R','True NR']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)] #,group_percentages
    labels = np.asarray(labels).reshape(2,2)
    
    
    cf_df = pd.DataFrame(cf_matrix, index = ['R', 'NR'], columns = ['R', 'NR'])
    sns.heatmap(cf_df, annot=labels, fmt='', cmap='Blues', ax=ax2, cbar = False)
    
    ax2.xaxis.tick_bottom() # x axis on top
    ax2.xaxis.set_label_position('bottom')
    #ax2.set_xlabel("Predicted Label")
    ax2.tick_params(axis='x', labelrotation=60)
    
    ax2.yaxis.tick_left() # x axis on top
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel("True Label")
    ax2.tick_params(axis='y', labelrotation=60)
    
    
    ###################################
    dfpp = pd.DataFrame([roc, prc, f1, precision, recall], 
                        index=['ROC', 'PRC', 'F1', 'Prec.', 'Recall'])
    
    dfpp.plot(kind = 'barh', ax=ax3, legend=False, color = 'b', alpha = 0.5)
    ax3.yaxis.tick_left() # x axis on top
    ax3.yaxis.set_label_position('left')
    ax3.set_xticks([0.0, 0.5, 1.0])
    ax3.xaxis.tick_bottom() # x axis on top
    ax3.xaxis.set_label_position('bottom')
    
    for y, x in dfpp[0].reset_index(drop=True).items():
        ax3.text(x, y-0.15, '%.2f' % x )
    ax3.spines[['right', 'top']].set_visible(False)
    
    fig.tight_layout(pad = 1.5)
    return fig



pretrainer = loadconceptor('../../checkpoint/latest/pretrainer.pt')
data_path = '../../00_data/'
itrp_df_label = pd.read_pickle(os.path.join(data_path, 'ITRP.PATIENT.TABLE'))
itrp_df_tpm = pd.read_pickle(os.path.join(data_path, 'ITRP.TPM.TABLE'))
itrp_df_tpm.shape, itrp_df_label.shape

#skcm_idx = (itrp_df_label.cancer_type=='SKCM')  & (itrp_df_label.Biopsy_site == 'Skin')

skcm_df_tpm = itrp_df_tpm #[skcm_idx]
skcm_df_label = itrp_df_label #[skcm_idx]

skcm_df_task = onehot(skcm_df_label.response_label)
size = skcm_df_label.groupby('cohort').size()
size = size.index + "\n(n = " + size.astype(str) + ")"

skcm_dfcx = skcm_df_label.cancer_type.map(CANCER_CODE).to_frame('cancer_code').join(skcm_df_tpm)
skcm_dfy = skcm_df_task





def cohort_to_cohort(cohorts):
    # Create a list of lists, each missing one element from the original list
    return [(cohorts[i], cohorts[:i] + cohorts[i+1:]) for i in range(len(cohorts))]
cohorts = skcm_df_label.cohort.unique().tolist()
train_test_cohorts = cohort_to_cohort(cohorts)


# In[ ]:


for mode in ['PFT', 'FFT', 'LFT']:
    
    params = {'mode': mode,
                'lr': 1e-2,
                'device':'cuda',
                'weight_decay': 1e-4,
                'batch_size':8,
                'max_epochs': 500,
                'task_loss_weight':1,
                'load_decoder':False,
                'task_dense_layer': [],
                'task_batch_norms':True,
                'entropy_weight': 0.0,
                'with_wandb': False,
                'save_best_model':False,
                'verbose': False}
    
    work_dir = './CTCT/%s' % mode
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    
    res = []
    for train_cohort, test_cohort in train_test_cohorts:
        test_cohort = test_cohort
        train_cohort = [train_cohort]
        test_cohort_name = '_'.join(test_cohort)
        train_cohort_name = '_'.join(train_cohort)
        
        test_idx = skcm_df_label[skcm_df_label.cohort.isin(test_cohort)].index
        train_idx = skcm_df_label[skcm_df_label.cohort.isin(train_cohort)].index
        
        train_X = skcm_dfcx.loc[train_idx]
        train_y = skcm_dfy.loc[train_idx]
        test_X = skcm_dfcx.loc[test_idx]    
        test_y = skcm_dfy.loc[test_idx]
        pretrainer = pretrainer.copy()
        finetuner = FineTuner(pretrainer, **params, 
                              work_dir= work_dir, 
                              task_name = 'Test_on_%s' % test_cohort_name, 
                              task_type='c')
        
        finetuner = finetuner.tune(dfcx_train = train_X,
                                   dfy_train = train_y, min_f1=0.8,)
        _, pred_trainy = finetuner.predict(train_X, batch_size = 16)
        _, pred_testy = finetuner.predict(test_X, batch_size = 16)
        s1 = score(train_y['R'], pred_trainy[1], pred_trainy.idxmax(axis=1))
        s2 = score(test_y['R'], pred_testy[1], pred_testy.idxmax(axis=1))

        df_pred = test_y[['R']].join(pred_testy[1].to_frame('prob.'))
        df_pred = df_pred.join(pred_testy.idxmax(axis=1).to_frame('pred_R'))
        df_pred = df_pred.join(skcm_df_label.cohort)
        df_test_res = df_pred.groupby('cohort').apply(lambda x:score(x['R'], x['prob.'], x['pred_R']))
        df_test_res = df_test_res.apply(pd.Series)
        df_test_res.columns = ['ROC', 'PRC', 'F1', 'ACC']
        
        s1 = score(train_y['R'], pred_trainy[1], pred_trainy.idxmax(axis=1))
        df_train_res = pd.DataFrame(s1, index = ['ROC', 'PRC', 'F1', 'ACC'], 
                                    columns = [train_cohort_name]).T
        
        print(df_train_res)
        
        dfs = df_test_res._append(df_train_res)
        
        dfs['Test_cohort'] = dfs.index
        dfs['Train_cohort'] = train_cohort_name
        dfs['n_trainable_params'] = finetuner.count_parameters()
        dfs['n_test'] = len(test_idx)
        dfs['n_train'] = len(train_idx)
        dfs['mode'] = mode
        res.append(dfs)
        
        # figs = df_pred.groupby('cohort').apply(lambda x:plot_performance(x['R'],x['prob.'],x['pred_R'] ))
        # for test_cohort_name, fig in figs.items():
        #     fig.suptitle('Train on %s, Test on %s' % (train_cohort_name, test_cohort_name))
        #     fig.savefig(os.path.join(work_dir, 'Train_%s_Test_%s.jpg' % (train_cohort_name, test_cohort_name)))

    dfres = pd.concat(res).reset_index(drop=True)
    dfres.to_csv(os.path.join(work_dir, 'performance.csv'))



mode = 'NFT'
work_dir = './CTCT/%s' % mode
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

from conceptor.model.fewshot import PrototypeNetNoFT

NFT = PrototypeNetNoFT()

skcm_dfcx = skcm_df_label.cancer_type.map(CANCER_CODE).to_frame('cancer_code').join(skcm_df_tpm)
skcm_dfx, _ = pretrainer.predict(skcm_dfcx, batch_size=16)
skcm_dfy = skcm_df_label.RECIST

res = []
for train_cohort, test_cohort in train_test_cohorts:
    test_cohort = test_cohort
    train_cohort = [train_cohort]
    test_cohort_name = '_'.join(test_cohort)
    train_cohort_name = '_'.join(train_cohort)

    test_idx = skcm_df_label[skcm_df_label.cohort.isin(test_cohort)].index
    train_idx = skcm_df_label[skcm_df_label.cohort.isin(train_cohort)].index
    
    train_X = skcm_dfx.loc[train_idx]
    train_y = skcm_dfy.loc[train_idx]
    test_X = skcm_dfx.loc[test_idx]    
    test_y = skcm_dfy.loc[test_idx]

    support_set = train_X.join(train_y)
    query_set = test_X.join(test_y)

    NFT = NFT.fit(support_set)
    dfp, scores = NFT.transform(query_set)
    precision, recall, roc, prc, f1, acc = scores

    df_pred = dfp.join(skcm_df_label.cohort)
    df_test_res = df_pred.groupby('cohort').apply(lambda x:score(x['True'].map({'R':1, 'NR':0}), 
                                                                 x['R'], 
                                                                 x['Pred'].map({'R':1, 'NR':0})))
                                                  
    dfs = df_test_res.apply(pd.Series)
    dfs.columns = ['ROC', 'PRC', 'F1', 'ACC']

    dfs['Test_cohort'] = dfs.index
    dfs['Train_cohort'] = train_cohort_name
    dfs['n_trainable_params'] = 0
    dfs['n_test'] = len(test_idx)
    dfs['n_train'] = len(train_idx)
    dfs['mode'] = mode
    res.append(dfs)

dfres = pd.concat(res)
dfres.to_csv(os.path.join(work_dir, 'performance.csv'))




