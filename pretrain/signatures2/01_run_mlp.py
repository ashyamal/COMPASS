#!/home/was966/micromamba/envs/RNA/bin/python

#sbatch --mem 32G -c 4 -t 6:00:00 -p gpu_quad --gres=gpu:1  ./01_run_mlp.py


import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale=1.5)

import torch
import torch.nn as nn
import numpy as np


import torch.utils.data as data

import sys
sys.path.insert(0, '/home/was966/Research/PSOG/pretrain/signatures2')
from data import TCGAData, GeneData
from aug import MixupNomralAugmentor
from data import TCGAData, GeneData
from model import TCGAPretrainModel
from train import predict, train

from loss import TripletLoss, TripletCosineLoss, CEWithNaNLabelsLoss, MSEWithNaNLabelsLoss
from saver import SaveBestModel
from plot import plot_embed_with_label



def onehot(S):
    assert type(S) == pd.Series, 'Input type should be pd.Series'
    dfd = pd.get_dummies(S, dummy_na=True)
    nanidx = dfd[dfd[np.nan]].index
    dfd.loc[nanidx, :] = np.nan
    dfd = dfd.drop(columns=[np.nan])*1.
    return dfd
    
data_path = '/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/TCGA-ITRP-MERGE/876/'
df_tpm = pd.read_pickle(os.path.join(data_path, 'TPM.TCGA.TABLE'))
df_tpm_normal = pd.read_pickle(os.path.join(data_path, 'TPM.TCGA.NORMAL.TABLE'))

tcga_label = pd.read_pickle(os.path.join(data_path, 'PATIENT.TCGA.TABLE'))
mutation = tcga_label[['tmb', 'cnv', 'msi']] #.fillna(0)
df_cancer = tcga_label[['cancer_type']]

df_tmb = mutation[['tmb']]
df_msi = mutation[['msi']]
df_cnv= mutation[['cnv']]
df_ctc = tcga_label[['cancer_type']]

df_msi = onehot(df_msi.msi)
df_ctc = onehot(df_ctc.cancer_type)
df_rps = onehot(tcga_label['treatment_outcome'])
df_ost = np.log10(tcga_label[['os_time']]+1)
df_oss = onehot(tcga_label.os_status)

df_pft = np.log10(tcga_label[['pfi_time']]+1)
df_pfs = onehot(tcga_label.pfi_status)





tasks = {'msi': df_msi, 
         'tmb': df_tmb, 
         'cnv':df_cnv, 
         'ctc':df_ctc, 
         'rps':df_rps, 
         'ost':df_ost,
         'oss':df_oss,
         'pft':df_pft,
         'pfs':df_pfs}
 
tasks_type_map = {'msi': 'c', 'tmb': 'r', 
                  'cnv':'r', 'ctc':'c', 
                  'rps':'c', 'ost':'r', 
                  'oss':'c', 'pft':'r', 'pfs':'c'}


## parameters
device='cuda'
lr = 1e-4
weight_decay = 1e-4
epochs = 39
batch_size = 128
embed_dim=32
triplet_margin=1.

task_loss_weight = 1.
task_dense_layer = [16]
task_batch_norms = False

encoder='mlp'
transformer_dim = 256
transformer_num_layers = 3

#encoder= 'mlp'
mlp_dense_layers = [1024, 512, 256, 128]

for sl_task,df_task  in tasks.items():
    task_type = tasks_type_map[sl_task]
    save_dir = './PretrainTCGA/%s_%s_%s' % (encoder, sl_task, task_batch_norms)

    ### setup ###
    augmentor = MixupNomralAugmentor(df_tpm_normal, df_tpm_normal.columns, beta=0.7)
    train_tcga = TCGAData(df_tpm, df_task, augmentor)
    train_loader = data.DataLoader(train_tcga, batch_size=batch_size, shuffle=True,
                                    drop_last=True, pin_memory=True, num_workers=4)
    
    input_dim = len(train_tcga.feature_name)
    task_dim = train_tcga.y.shape[1]
    model = TCGAPretrainModel(input_dim, task_dim, task_type, embed_dim, 
                              encoder = encoder, #
                              mlp_dense_layers = mlp_dense_layers, 
                              transformer_dim = transformer_dim,
                              transformer_num_layers = transformer_num_layers,
                              task_dense_layer = task_dense_layer, 
                              task_batch_norms = task_batch_norms) 
    model = model.to(device)
    
    triplet_loss = torch.jit.script(TripletLoss(margin=triplet_margin))
    ce_loss = torch.jit.script(CEWithNaNLabelsLoss())
    mse_loss = torch.jit.script(MSEWithNaNLabelsLoss())
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    saver = SaveBestModel(save_dir = save_dir, save_name = 'tcga_best_model.pth')
    ssl_loss = triplet_loss
    if task_type == 'c':
        tsk_loss = ce_loss
    else:
        tsk_loss = mse_loss

    ### training ###
    performace = []
    for epoch in tqdm(range(epochs), desc="Epochs", ascii=True):
        train_total_loss, train_ssl_loss, train_tsk_loss = train(train_loader, model, optimizer, 
                                                                 ssl_loss, tsk_loss, device, 
                                                                 alpha = task_loss_weight)
        saver(train_total_loss, epoch, model, optimizer)
        performace.append([epoch, train_total_loss, train_ssl_loss, train_tsk_loss])
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, train_total_loss))
    saver.save()


    ## plot loss
    df = pd.DataFrame(performace, columns = ['epochs', 'total_loss', 'ssl_loss', 'tsk_loss']).set_index('epochs')
    v = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    fig, ax = plt.subplots(figsize=(7,5))
    df.plot(ax = ax)
    fig.savefig(os.path.join(save_dir, 'tcga_train_loss.png'), bbox_inches='tight')
    df.to_pickle(os.path.join(save_dir, 'tcga_train_loss.pkl'))

    ### make prediction & plot on TCGA
    model = TCGAPretrainModel(**saver.inMemorySave['model_args']) #transformer_dim = 128, transformer_num_layers = 2
    model.load_state_dict(saver.inMemorySave['model_state_dict'])
    dfe, dfp = predict(df_tpm, model,  device='cpu')
    dfc = df_cancer.cancer_type.apply(lambda x:x.replace('TCGA-', ''))
    msi = tcga_label[['msi']]
    cnv = tcga_label.cnv.clip(-1, 0.5).to_frame(name='cnv')
    
    l1 = tcga_label[['race', 'vital_status', 'treatment_outcome']].fillna('NaN')
    l2 = np.log10(tcga_label[['pfi_time', 'os_time']]+1)
    l3 = tcga_label[['gender', 'age',]]


    dfd = dfe.join(dfc).join(df_tmb).join(cnv).join(msi).join(l1).join(l2).join(l3)
    
    label_col = ['cancer_type', 'tmb', 'cnv', 'msi','race', 'vital_status', 
                 'treatment_outcome','pfi_time', 'gender', 'age', 'os_time']
    label_type = ['c', 'r', 'r', 'c',  'c', 'c', 'c',  'r',  'c', 'r', 'r',]
    figs = plot_embed_with_label(dfd, n_neighbors=15, spread=5,
                                 label_col = label_col,  
                                 label_type = label_type, figsize=(5,5))
    
    for fig, name in zip(figs, label_col):
        fig.savefig(os.path.join(save_dir, 'tcga_%s.png' % name), bbox_inches='tight', )

    ### make prediction & plot on ITRP
    itrp_x = pd.read_pickle(os.path.join(data_path, 'TPM.ITRP.TABLE'))
    itrp_meta = pd.read_pickle(os.path.join(data_path, 'PATIENT.ITRP.TABLE'))
    itrp_meta['response_label'] = itrp_meta['Freeman_response']
    itrp_meta['response_label'][itrp_meta.cohort == 'Gide'] = itrp_meta['RECIST_Response'][itrp_meta.cohort == 'Gide']
    itrp_y = itrp_meta[['response_label']]
    itrp_c = itrp_meta[['cohort']]
    itrp_meta.groupby('cohort')['response_label'].value_counts().unstack().T
    
    itrp_dfe, itrp_dfp = predict(itrp_x, model,  device='cpu')

    label_col = ['cohort','response_label','RECIST','Alive (Y=1, No=0)', 'Overall_survival']
    label_type = ['c', 'c', 'c', 'c', 'r']
    dfd = itrp_dfe.join(itrp_meta[label_col])
    figs = plot_embed_with_label(dfd, n_neighbors=15,
                                spread=5,
                                s=20,
                                 label_col = label_col,  
                                 label_type = label_type, figsize=(5,5))
    for fig, name in zip(figs, label_col):
        fig.savefig(os.path.join(save_dir, 'itrp_%s.png' % name), bbox_inches='tight', )
        