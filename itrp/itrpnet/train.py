# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:09 2023

@author: Wanxiang Shen
"""
import torch
import torch.utils.data as Torchdata
from data import TCGAData, GeneData
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(ascii=True)



def train(train_loader, model, optimizer, ssl_loss, tsk_loss, device, alpha = 0.):
    
    model.train()
    total_loss = []
    total_ssl_loss = []
    total_tsk_loss = []
    for data in train_loader:

        triplet, y_true = data
        anchor, positive, negative = triplet
        
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        y_true = y_true.to(device)
    
        optimizer.zero_grad()
        anchor_emb, y_pred = model(anchor)
        positive_emb, _ = model(positive)
        negative_emb, _ = model(negative)

        ## self-supervised learning loss SSL
        lss = ssl_loss(anchor_emb, positive_emb, negative_emb)
        tsk = tsk_loss(y_pred, y_true)
        loss = (1-alpha)*lss + tsk*alpha

        loss.backward()
        optimizer.step()

        total_loss.append(loss.cpu().detach().numpy())
        total_ssl_loss.append(lss.cpu().detach().numpy())
        total_tsk_loss.append(tsk.cpu().detach().numpy())

    train_total_loss =  np.nanmean(total_loss)
    train_ssl_loss =  np.nanmean(total_ssl_loss)
    train_tsk_loss =  np.nanmean(total_tsk_loss)
    
    return train_total_loss, train_ssl_loss, train_tsk_loss



@torch.no_grad()
def test(test_loader, model, ssl_loss, tsk_loss, device, alpha=1.):
    model.eval()
    _loss = []
    _ssl_loss = []
    _tsk_loss = []
    for data in test_loader:
        triplet, y_true = data
        anchor, positive, negative = triplet
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        y_true = y_true.to(device)

        anchor_emb, y_pred = model(anchor)
        positive_emb, _ = model(positive)
        negative_emb, _ = model(negative)
        
        lss = ssl_loss(anchor_emb, positive_emb, negative_emb)
        tsk = tsk_loss(y_pred, y_true)
        loss = (1.-alpha)*lss + tsk*alpha
        
        _loss.append(loss.cpu().detach().numpy())
        _ssl_loss.append(lss.cpu().detach().numpy())
        _tsk_loss.append(tsk.cpu().detach().numpy())
    test_total_loss =  np.nanmean(_loss)
    test_ssl_loss =  np.nanmean(_ssl_loss)
    test_tsk_loss =  np.nanmean(_tsk_loss)        
    
    return test_total_loss, test_ssl_loss, test_tsk_loss



@torch.no_grad()
def predict(df_tpm, model,  device = 'cpu', batch_size=512,  num_workers=4):
    model.eval()
    predict_tcga = GeneData(df_tpm)
    predict_loader = Torchdata.DataLoader(predict_tcga, 
                                     batch_size=batch_size, 
                                     shuffle=False,
                                     pin_memory=True, 
                                          num_workers=num_workers)
    embds = []
    ys = []
    for anchor in tqdm(predict_loader, ascii=True):
        anchor = anchor.to(device)
        anchor_emb, anchor_ys = model(anchor)
        embds.append(anchor_emb)
        ys.append(anchor_ys)
    
    embeddings  = torch.concat(embds, axis=0).cpu().detach().numpy()
    predictions = torch.concat(ys, axis=0).cpu().detach().numpy()
    dfe = pd.DataFrame(embeddings, index = predict_tcga.patient_name)
    dfp = pd.DataFrame(predictions, index = predict_tcga.patient_name)

    return dfe, dfp
    
