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
import wandb

import torch.utils.data as data
from itertools import chain

import sys
sys.path.insert(0, '/home/was966/Research/PSOG/itrp/itrpnet')
from data import TCGAData, GeneData
from aug import MixupNomralAugmentor
from data import TCGAData, GeneData
from model import TCGAPretrainModel
from train import train, test, predict
from loss import TripletLoss, TripletCosineLoss, CEWithNaNLabelsLoss, MSEWithNaNLabelsLoss
from saver import SaveBestModel
from plot import plot_embed_with_label


class TCGAPreTrainer:

    def __init__(self, 
                 df_tpm_normal,
                 aug_gene_list = [],
                 aug_beta = 0.7,
                 device='cuda',
                lr = 1e-5,
                weight_decay = 1e-4,
                epochs = 100,
                batch_size = 64,
                embed_dim=32,
                triplet_margin=1.,
                K = 500,
                task_loss_weight = 1.,
                task_dense_layer = [24, 16],
                task_batch_norms = False,
                encoder='transformer',
                 encoder_dropout = 0.,
                transformer_dim = 256,
                transformer_num_layers = 4,
                #encoder= 'mlp'
                mlp_dense_layers = [128, 64],                
                work_dir = './PretrainResults',
                 run_name_prefix = 'TCGA'
                ):

        ### augmentor ###
        self.df_tpm_normal = df_tpm_normal
        if len(aug_gene_list) == 0:
            aug_gene_list = df_tpm_normal.columns
        self.aug_gene_list = aug_gene_list
        self.augmentor = MixupNomralAugmentor(df_tpm_normal, aug_gene_list, beta=aug_beta)
        self.aug_beta = aug_beta
        
        self.device=device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.embed_dim=embed_dim
        self.triplet_margin=triplet_margin
        self.K = K
        
        self.task_loss_weight = task_loss_weight
        self.task_dense_layer = task_dense_layer
        self.task_batch_norms = task_batch_norms
        
        self.encoder=encoder
        self.encoder_dropout = encoder_dropout
        self.transformer_dim = transformer_dim
        self.transformer_num_layers = transformer_num_layers
        self.mlp_dense_layers = mlp_dense_layers
        self.work_dir = work_dir
        self.run_name_prefix = run_name_prefix

    
    def _setup(self, input_dim, task_dim, task_type, save_dir, run_name):

        model = TCGAPretrainModel(input_dim, task_dim, task_type, 
                                  embed_dim = self.embed_dim, 
                                  encoder = self.encoder, #
                                  encoder_dropout = self.encoder_dropout,
                                  mlp_dense_layers = self.mlp_dense_layers, 
                                  transformer_dim = self.transformer_dim,
                                  transformer_num_layers = self.transformer_num_layers,
                                  task_dense_layer = self.task_dense_layer, 
                                  task_batch_norms = self.task_batch_norms) 
        model = model.to(self.device)
        
        triplet_loss = TripletLoss(margin=self.triplet_margin)
        ce_loss = CEWithNaNLabelsLoss()
        mse_loss = MSEWithNaNLabelsLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        saver = SaveBestModel(save_dir = save_dir, save_name = 'model.pth')
        ssl_loss = triplet_loss
        if task_type == 'c':
            tsk_loss = ce_loss
        else:
            tsk_loss = mse_loss
            
        self.model = model
        self.ssl_loss = ssl_loss
        self.tsk_loss = tsk_loss
        self.optimizer = optimizer
        self.saver = saver

        # Initialize wandb
        wandb.init(project='itrp', entity='senwanxiang', name=run_name, save_code=True)

        # # Log model architecture to wandb
        wandb.watch(model)
        


    def train(self, df_tpm_train, df_task_train, task_name, task_type, 
              df_tpm_test = None, df_task_test = None):
        
        self.task_type = task_type
        self.task_name = task_name
        
        train_tcga = TCGAData(df_tpm_train, df_task_train, self.augmentor, K = self.K)
        train_loader = data.DataLoader(train_tcga, batch_size=self.batch_size, shuffle=True,
                                        drop_last=True, pin_memory=True, num_workers=4)
        
        input_dim = len(train_tcga.feature_name)
        task_dim = train_tcga.y.shape[1]
        
        if df_tpm_test is not None:
            test_tcga = TCGAData(df_tpm_test, df_task_test, self.augmentor, K = self.K)
            test_loader = data.DataLoader(test_tcga, batch_size=self.batch_size, shuffle=False,
                                          pin_memory=True, num_workers=4)
        else:
            test_loader = None

        
        run_name = '%s-%s-(%s-%s)-bt%s-lr%s-aug%s-k%s-marigin%s' % (self.run_name_prefix,
                                                                    self.encoder, 
                                                                    self.task_name, 
                                                                    self.task_loss_weight,
                                                                    self.batch_size,
                                                                    self.lr,
                                                                    self.aug_beta,
                                                                    self.K,
                                                                    self.triplet_margin
                                                            )
        self.run_name = run_name
        self.save_dir = os.path.join(self.work_dir, run_name)
        
        ## init model, operimizer, ...
        self._setup(input_dim, task_dim, task_type, self.save_dir, self.run_name)
        self.input_dim = input_dim
        self.task_dim = task_dim

        ### training ###
        performace = []
        for epoch in tqdm(range(self.epochs), desc="Epochs", ascii=True):
            train_total_loss, train_ssl_loss, train_tsk_loss = train(train_loader, self.model, self.optimizer, 
                                                                     self.ssl_loss, self.tsk_loss, self.device, 
                                                                     alpha = self.task_loss_weight)
            if test_loader is not None:
                test_total_loss, test_ssl_loss, test_tsk_loss = test(test_loader, self.model, self.ssl_loss, 
                                                                     self.tsk_loss, self.device, 
                                                                     alpha =self.task_loss_weight)
                self.saver(test_total_loss, epoch, self.model, self.optimizer)
            else:
                test_total_loss, test_ssl_loss, test_tsk_loss = np.nan, np.nan, np.nan
                self.saver(train_total_loss, epoch, self.model, self.optimizer)   
                
            performace.append([epoch, train_total_loss, train_ssl_loss, train_tsk_loss, 
                               test_total_loss, test_ssl_loss, test_tsk_loss])

            
            wandb.log({"train_loss": train_total_loss, 'train_ssl_loss':train_ssl_loss, 'train_%s_loss' % self.task_name :train_tsk_loss,
                       'test_loss': test_total_loss, 'test_ssl_loss':test_ssl_loss, 'test_%s_loss' % self.task_name :test_tsk_loss})
            
            print("Epoch: {}/{} - Train Loss: {:.4f} - Test Loss: {:.4f}".format(epoch+1, self.epochs, train_total_loss, test_total_loss))
        self.saver.save()
        self.performace = performace

        
        ## plot loss locally
        df = pd.DataFrame(self.performace, columns = ['epochs', 'total_loss', 'ssl_loss', '%s_loss' % self.task_name, 
                                                      'test_loss', 'test_ssl_loss', 'test_%s_loss' % self.task_name]).set_index('epochs')
        v = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
        fig, ax = plt.subplots(figsize=(7,5))
        df.plot(ax = ax)
        fig.savefig(os.path.join(self.save_dir, 'tcga_train_loss.png'), bbox_inches='tight')
        df.to_pickle(os.path.join(self.save_dir, 'tcga_train_loss.pkl'))

        return self


    def plot_embed(self, df_tpm, df_label, label_types, **kwargs):
        
        ### make prediction & plot on TCGA
        model = TCGAPretrainModel(**self.saver.inMemorySave['model_args']) #transformer_dim = 128, transformer_num_layers = 2
        model.load_state_dict(self.saver.inMemorySave['model_state_dict'])
        model = model.to(self.device)
        dfe, dfp = predict(df_tpm,  model,  device=self.device)
        dfd = dfe.join(df_label)
        label_cols = df_label.columns

        figs = plot_embed_with_label(dfd, 
                                     label_col = label_cols,  
                                     label_type = label_types, **kwargs)
        for fig, name in zip(figs, label_cols):
            fig.savefig(os.path.join(self.save_dir, 'tcga_%s.png' % name), bbox_inches='tight', )




