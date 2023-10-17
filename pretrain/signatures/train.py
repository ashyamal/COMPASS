# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:09 2023

@author: Wanxiang Shen
"""
import torch
import torch.utils.data as Torchdata
from data import TCGAData, GeneData
import pandas as pd



class TCGATrainer:
    pass


def train(df_tpm):
    pass


def test(df_tpm):
    pass




@torch.no_grad()
def predict(df_tpm, model,  device, SL_tasks):

    model.eval()
    predict_tcga = GeneData(df_tpm)
    predict_loader = Torchdata.DataLoader(predict_tcga, 
                                     batch_size=512, 
                                     shuffle=False,
                                     pin_memory=True, num_workers=4)
    embds = []
    ys = []
    for anchor in predict_loader:
        anchor = anchor.to(device)
        anchor_emb, anchor_ys = model(anchor)
        anchor_ys = torch.concat(anchor_ys, axis=1)
        embds.append(anchor_emb)
        ys.append(anchor_ys)
    
    embeddings  = torch.concat(embds, axis=0).cpu().detach().numpy()
    predictions = torch.concat(ys, axis=0).cpu().detach().numpy()
    dfe = pd.DataFrame(embeddings, index = predict_tcga.patient_name)
    dfp = pd.DataFrame(predictions, index = predict_tcga.patient_name, 
                       columns = ['tmb', 'msi_h','msi_l',  'mss', 'cnv'])
    # tasks = []
    # for task in SL_tasks:
    #     if task == 'msi':
    #         task = ['msi_h', 'msi_l', 'mss']
    #         tasks.extend(task)
    #     else:
    #         tasks.append(task)
    # dfp = dfp[tasks]   

    return dfe, dfp