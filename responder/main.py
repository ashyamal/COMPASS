# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:09 2023

@author: Wanxiang Shen
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale=1.5)
from tqdm import tqdm
import torch
import wandb

import torch.utils.data as data
from copy import deepcopy
from collections import OrderedDict
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold

import random
import datetime


from responder.dataloader import TCGAData, GeneData, ITRPData
from responder.augmentor import MixupNomralAugmentor, RandomMaskAugmentor, FeatureJitterAugmentor, MaskJitterAugmentor
from responder.model.scaler import Datascaler
from responder.model.model import Responder
from responder.model.train import Trainer, Tester, Predictor, Evaluator, Extractor
from responder.model.loss import TripletLoss, CEWithNaNLabelsLoss, MAEWithNaNLabelsLoss
from responder.model.saver import SaveBestModel
from responder.utils import plot_embed_with_label


def fixseed(seed=42): 
    np.random.seed(seed)  
    random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixseed(seed=42)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def loadresponder(file, **kwargs):   
    responder = torch.load(file, **kwargs)
    if responder.with_wandb:
        responder.wandb._settings = ''
    return responder



class PreTrainer:

    def __init__(self, 
                device='cuda',
                lr = 1e-5,
                weight_decay = 1e-6,
                epochs = 100,
                patience = 10, 
                batch_size = 64,
                 
                embed_dim=32,
                task_loss_weight = 0.0,
                task_dense_layer = [24],
                task_batch_norms = True,
                task_class_weight = None, 
                encoder='transformer',
                encoder_dropout = 0.,
                num_cancer_types = 33,
                
                transformer_dim = 32,
                transformer_num_layers = 1,
                transformer_nhead = 2,
                transformer_pos_emb = 'learnable',
                
                batch_correction = 0.0,
                proj_level = 'cellpathway',
                proj_pid = False,
                proj_cancer_type = True,
                 
                triplet_metric = 'cosine',
                triplet_margin=1.,
                K = 1,                 
                seed = 42,

                work_dir = './results',
                verbose = True,
                with_wandb = False,
                wandb_project = 'pretrain',
                wandb_dir = '/n/data1/hms/dbmi/zitnik/lab/users/was966/wandb/',
                wandb_entity = 'senwanxiang',
                **encoder_kwargs
                ):

        '''
        transformer_pos_emb: {None, 'umap', 'pumap', 'gene2vect'}
        encoder:{'mlp', 'transformer'}
        '''

        self.device=device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.num_cancer_types = num_cancer_types
        
        self.triplet_margin=triplet_margin
        self.triplet_metric=triplet_metric
        
        self.K = K
        
        self.task_loss_weight = task_loss_weight
        self.task_dense_layer = task_dense_layer
        self.task_batch_norms = task_batch_norms
        self.task_class_weight = task_class_weight
        self.encoder = encoder
        self.encoder_dropout = encoder_dropout
        self.transformer_dim = transformer_dim
        self.transformer_nhead = transformer_nhead
        self.transformer_num_layers = transformer_num_layers
        self.transformer_pos_emb = transformer_pos_emb
        
        self.batch_correction = batch_correction
        self.proj_level = proj_level
        self.proj_pid = proj_pid
        self.proj_cancer_type = proj_cancer_type
        
        fixseed(seed=seed)
        self.seed = seed

        self.patience = patience # for early stopping
        
        self.work_dir = work_dir
        self.with_wandb = with_wandb
        self.verbose = verbose
        self.wandb_project = wandb_project
        self.wandb_dir = wandb_dir
        self.wandb_entity = wandb_entity
        self.encoder_kwargs = encoder_kwargs
        
    def _setup(self, input_dim, task_dim, task_type, save_dir, run_name):

        model = Responder(input_dim, task_dim, task_type, 
                          proj_level = self.proj_level,
                          proj_pid = self.proj_pid,
                          proj_cancer_type = self.proj_cancer_type,
                          
                          num_cancer_types = self.num_cancer_types,
                          embed_dim = self.embed_dim, 
                          encoder = self.encoder, 
                          encoder_dropout = self.encoder_dropout,
                          
                          transformer_dim = self.transformer_dim,
                          transformer_nhead = self.transformer_nhead,
                          transformer_num_layers = self.transformer_num_layers,
                          transformer_pos_emb = self.transformer_pos_emb,
                          task_dense_layer = self.task_dense_layer, 
                          task_batch_norms = self.task_batch_norms, 
                          **self.encoder_kwargs)
        
        model = model.to(self.device)

        ssl_loss = TripletLoss(margin=self.triplet_margin, 
                               metric = self.triplet_metric)

            
        ce_loss = CEWithNaNLabelsLoss(weights=self.task_class_weight)
        mae_loss = MAEWithNaNLabelsLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr, 
                                     weight_decay = self.weight_decay)
        saver = SaveBestModel(save_dir = save_dir, save_name = 'model.pth')

        if task_type == 'c':
            tsk_loss = ce_loss
        else:
            tsk_loss = mae_loss
            
        self.model = model
        self.ssl_loss = ssl_loss
        self.tsk_loss = tsk_loss
        self.optimizer = optimizer
        self.saver = saver
        
        if self.with_wandb:
            # Initialize wandb
            self.wandb = wandb.init(project = self.wandb_project, 
                                    entity = self.wandb_entity, 
                                    dir = self.wandb_dir,
                                    name=run_name, save_code=True)
    
            # # Log model architecture to wandb
            self.wandb.watch(model)



    def train(self, 
              dfcx_train, 
              dfy_train, 
              dfcx_test = None, 
              dfy_test = None, 
              task_name = 'notask', 
              task_type = 'c', 
              aug_method = 'jitter',
              scale_method = 'minmax', **augargs):


        ### scaler ####
        self.scale_method = scale_method
        self.scaler = Datascaler(scale_method = scale_method)
        self.scaler = self.scaler.fit(dfcx_train)
        self.aug_method = aug_method
        
        dfcx_train = self.scaler.transform(dfcx_train)

        if aug_method == 'mask':
            self.augmentor = RandomMaskAugmentor(**augargs)
        elif aug_method == 'jitter':
            self.augmentor = FeatureJitterAugmentor(**augargs)
        elif aug_method == 'mix':
            self.augmentor = MaskJitterAugmentor(**augargs)
        else:
            raise ValueError("Invalid method. Use 'mask', 'jitter' or 'mix'.")

        self.task_type = task_type
        self.task_name = task_name

        train_tcga = TCGAData(dfcx_train, dfy_train, self.augmentor, K = self.K)
        
        self.y_scaler = train_tcga.y_scaler
        self.feature_name = train_tcga.feature_name
        
        train_loader = data.DataLoader(train_tcga, batch_size=self.batch_size, 
                                        shuffle = True, worker_init_fn = worker_init_fn,
                                        drop_last=True, pin_memory=True, num_workers=4)
        
        input_dim = len(train_tcga.feature_name)
        task_dim = train_tcga.y.shape[1]
        
        if dfcx_test is not None:
            dfcx_test = self.scaler.transform(dfcx_test)
            test_tcga = TCGAData(dfcx_test, dfy_test, self.augmentor, K = self.K)
            test_loader = data.DataLoader(test_tcga, batch_size=self.batch_size, 
                                          shuffle=False, worker_init_fn = worker_init_fn,
                                          pin_memory=True, num_workers=4)
        else:
            test_loader = None

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        run_name = f"Pretrain_{self.task_name}_{formatted_time}"  # Example filename

        self.run_name = run_name
        self.save_dir = os.path.join(self.work_dir, run_name)
        
        ## init model, operimizer, ...
        self._setup(input_dim, task_dim, task_type, self.save_dir, self.run_name)
        self.input_dim = input_dim
        self.task_dim = task_dim

        ### training ###
        performace = []

        best_val_loss = float('inf') 
        patience_counter = 0  
        for epoch in range(self.epochs):
            train_total_loss, train_ssl_loss, train_tsk_loss = Trainer(train_loader, self.model, self.optimizer, 
                                                                        self.ssl_loss, self.tsk_loss, self.device, 
                                                                        alpha = self.task_loss_weight, 
                                                                        correction = self.batch_correction)

            if test_loader is not None:
                test_total_loss, test_ssl_loss, test_tsk_loss = Tester(test_loader, self.model, self.ssl_loss, 
                                                                        self.tsk_loss, self.device, 
                                                                        alpha = self.task_loss_weight,
                                                                        correction = self.batch_correction)
                
                self.saver(test_total_loss, epoch, self.model, self.optimizer, self.scaler)


                # Early Stopping Check
                if test_total_loss < best_val_loss:
                    best_val_loss = test_total_loss
                    patience_counter = 0  # Reset the counter if improvement is seen
                else:
                    patience_counter += 1  # Increment the counter if no improvement

            else:
                test_total_loss, test_ssl_loss, test_tsk_loss = np.nan, np.nan, np.nan
                self.saver(train_total_loss, epoch, self.model, self.optimizer, self.scaler)   


                # Early Stopping Check
                if train_total_loss < best_val_loss:
                    best_val_loss = train_total_loss
                    patience_counter = 0  # Reset the counter if improvement is seen
                else:
                    patience_counter += 1  # Increment the counter if no improvement

            
            performace.append([epoch, train_total_loss, train_ssl_loss, train_tsk_loss, 
                               test_total_loss, test_ssl_loss, test_tsk_loss])

            if self.with_wandb:
                self.wandb.log({"pretrain_loss": train_total_loss, 
                                'pretrain_ssl_loss':train_ssl_loss, 
                                'pretrain_%s_loss' % self.task_name :train_tsk_loss,
                                'pretrain_test_loss': test_total_loss, 
                                'pretrain_test_ssl_loss':test_ssl_loss, 
                                'pretrain_test_%s_loss' % self.task_name :test_tsk_loss})
            if self.verbose:
                print("Epoch: {}/{} - Train Loss: {:.4f} - Test Loss: {:.4f}".format(epoch+1, 
                                                                                     self.epochs, 
                                                                                     train_total_loss, 
                                                                                     test_total_loss))
        

            # for early stopping
            if patience_counter >= self.patience:
                print(f"Stopping early at epoch {epoch+1}. No improvement in validation loss for {self.patience} consecutive epochs.")
                break
                
        
        self.saver.save()
        self.performace = performace

        
        ## plot loss locally
        df = pd.DataFrame(self.performace, columns = ['epochs', 'total_loss', 'ssl_loss', '%s_loss' % self.task_name, 
                                                      'test_loss', 'test_ssl_loss', 'test_%s_loss' % self.task_name]).set_index('epochs')
        #v = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
        fig, ax = plt.subplots(figsize=(7,5))
        df.plot(ax = ax)
        fig.savefig(os.path.join(self.save_dir, 'tcga_train_loss.png'), bbox_inches='tight')
        df.to_pickle(os.path.join(self.save_dir, 'tcga_train_loss.pkl'))

        return self


    def plot_embed(self, df_tpm, df_label, label_types, **kwargs):
        dfe, dfp = self.predict(df_tpm, batch_size=self.batch_size,  num_workers=4)
        dfd = dfe.join(df_label)
        label_cols = df_label.columns

        figs = plot_embed_with_label(dfd, 
                                     label_col = label_cols,  
                                     label_type = label_types, **kwargs)
        for fig, name in zip(figs, label_cols):
            fig.savefig(os.path.join(self.save_dir, 'tcga_%s.png' % name), bbox_inches='tight', )


    def predict(self, df_tpm, batch_size=512,  num_workers=4):
        model = Responder(**self.saver.inMemorySave['model_args']) 
        model.load_state_dict(self.saver.inMemorySave['model_state_dict'])
        model = model.to(self.device)
        dfe, dfp = Predictor(df_tpm, model, self.scaler, 
                           device = self.device, batch_size=batch_size,  num_workers=num_workers)
        return dfe, dfp

    
    def extract(self, df_tpm, batch_size=512,  num_workers=4):
        model = Responder(**self.saver.inMemorySave['model_args']) 
        model.load_state_dict(self.saver.inMemorySave['model_state_dict'])
        model = model.to(self.device)
        dfg, dfc = Extractor(df_tpm, model, self.scaler, 
                           device = self.device, batch_size=batch_size, 
                             num_workers=num_workers)
        return dfg, dfc
        
    
    def save(self, mfile):
        if self.with_wandb:
            self.wandb._settings = ''
        torch.save(self, mfile)
        print('Saving the model to %s' % mfile)

    def load(self, mfile, **kwargs):
        self = loadresponder(mfile, **kwargs)
        if self.with_wandb:
            self.wandb._settings = ''
        return self

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        

    def close(self):
        if self.with_wandb:
            self.wandb.finish()
            self.wandb._settings = ''
        self.save(os.path.join(self.save_dir, 'pretrainer.pt'))

    def copy(self):
        return deepcopy(self)






class FineTuner:
    '''
    Contrastive Finetuner on ITRP datasets
    '''
    def __init__(self, 
                pretrainer,
                mode = 'head',
                load_decoder = False, 
                 
                device='cuda',
                lr = 1e-3,
                weight_decay = 1e-4,
                epochs = 100,
                patience = 10, 
                 
                batch_size = 32,
                 
                triplet_metric = 'cosine',
                triplet_margin=1.,
                
                task_loss_weight = 1.0,
                task_dense_layer = [24],
                task_batch_norms = True,
                task_class_weight = [1, 2], 
                batch_correction = 0.0,
                entropy_weight = 0.0,
                seed = 42,
                verbose = True,
                with_wandb = False,
                wandb_project = 'finetune',
                wandb_dir = '/n/data1/hms/dbmi/zitnik/lab/users/was966/wandb/',
                wandb_entity = 'senwanxiang',
                work_dir = './results',
                ):
        
        '''
        pretrainer: TCGAPreTrainer
        mode: tuning mode{head, partial, or full}
        '''
        
        self.pretrainer = pretrainer.copy()
        self.scaler = self.pretrainer.scaler
        
        self.mode = mode
        self.load_decoder = load_decoder
        
        self.device=device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.triplet_metric = triplet_metric
        self.triplet_margin = triplet_margin

        self.task_loss_weight = task_loss_weight
        self.task_dense_layer = task_dense_layer
        self.task_batch_norms = task_batch_norms
        self.task_class_weight = task_class_weight
        self.batch_correction = batch_correction

        self.entropy_weight = entropy_weight
        
        self.seed = seed
        fixseed(seed)
        
        self.work_dir = work_dir
        self.with_wandb = with_wandb
        self.wandb_project = wandb_project
        self.wandb_dir = wandb_dir
        self.wandb_entity = wandb_entity
        self.verbose = verbose

        self.params = {'mode': self.mode,   
                       'load_decoder':self.load_decoder,
                       'device':self.device,
                       'lr': self.lr,
                       'weight_decay':self.weight_decay,
                       'epochs':self.epochs,
                       'patience':self.patience,
                       'batch_size':self.batch_size,
                       'triplet_margin':self.triplet_margin, 
                       
                       'triplet_metric':self.triplet_metric,
                        'entropy_weight':self.entropy_weight,
                       'task_loss_weight':self.task_loss_weight,
                       'task_dense_layer':self.task_dense_layer,
                       'task_batch_norms': self.task_batch_norms,
                       'task_class_weight':self.task_class_weight,
                       'batch_correction':self.batch_correction,
                       'work_dir':self.work_dir,
                       'with_wandb':self.with_wandb,
                       'wandb_project':self.wandb_project,
                       'wandb_dir':self.wandb_dir,
                       'wandb_entity':self.wandb_entity,
                       'verbose':self.verbose, 
                      }

    
    def _setup(self, pretrainer, task_dim, task_type, save_dir, run_name):
        '''
        pretrainer: TCGAPreTrainer
        '''
        model_args = deepcopy(pretrainer.saver.inMemorySave['model_args'])
        model_weights = deepcopy(pretrainer.saver.inMemorySave['model_state_dict'])
        
        model_args['task_dim'] = task_dim
        model_args['task_type'] = task_type #'f'
        model_args['task_dense_layer'] = self.task_dense_layer  
        model_args['task_batch_norms'] = self.task_batch_norms

        ### define finetune model
        model = Responder(**model_args)
        
        encoder_state = OrderedDict()
        for k, v in model_weights.items():
            if self.load_decoder:
                if self.verbose:
                    print('Load: %s' % k)
                encoder_state[k] = v
            else:
                if not 'taskdecoder' in k:
                    if self.verbose:
                        print('Load: %s' % k)
                    encoder_state[k] = v


        ### load Pretrained model
        model.load_state_dict(encoder_state, strict=False)
        model = model.to(self.device)
        #model._projector = nn.Identity()

        if self.mode == 'head':
            for param in model.inputencoder.parameters():
                param.requires_grad = False
            for param in model.latentprojector.parameters():
                param.requires_grad = False
            plist = [{'params':model.taskdecoder.parameters()}]
            
        elif self.mode == 'partial':
            for param in model.inputencoder.parameters():
                param.requires_grad = False
            plist = [
                    {'params': model.latentprojector.parameters()}, #, 'lr': 5e-3
                    {'params': model.taskdecoder.parameters()}, #, 'lr': 1e-3
                    ]

        else:
            # with/wo layer decay
            plist = [
                    {'params': model.inputencoder.parameters()}, #, 'lr': 1e-6
                    {'params': model.latentprojector.parameters()}, #, 'lr': 1e-5
                    {'params': model.taskdecoder.parameters()}, #, 'lr': 1e-3
                    ]

        
        optimizer = torch.optim.Adam(plist, lr = self.lr, weight_decay = self.weight_decay)

        ssl_loss = TripletLoss(margin=self.triplet_margin, 
                               metric = self.triplet_metric)

        ce_loss = CEWithNaNLabelsLoss(weights=self.task_class_weight) 
        mae_loss = MAEWithNaNLabelsLoss()

        saver = SaveBestModel(save_dir = save_dir, save_name = 'ft_model.pth')

        if task_type == 'c':
            tsk_loss = ce_loss
        else:
            tsk_loss = mae_loss
            
        self.model = model
        self.ssl_loss = ssl_loss
        self.tsk_loss = tsk_loss
        self.optimizer = optimizer
        self.saver = saver
        
        if self.with_wandb:

            self.wandb = wandb.init(project = self.wandb_project, 
                                    entity = self.wandb_entity, 
                                    dir = self.wandb_dir,
                                    name=run_name, save_code=True)
            
            self.wandb.watch(model)


    
    def tune(self, 
             dfcx_train, 
             dfy_train,
             dfcx_test = None, 
             dfy_test = None,
             task_name = 'rps', 
             task_type = 'f',
            ):


        ### scaler ####
        dfcx_train = self.scaler.transform(dfcx_train)

        self.task_type = task_type
        self.task_name = task_name

        train_itrp = ITRPData(dfcx_train, dfy_train)


        #self.train_itrp = train_itrp
        self.y_scaler = train_itrp.y_scaler
        self.feature_name = train_itrp.feature_name
        
        train_loader = data.DataLoader(train_itrp, batch_size=self.batch_size, 
                                       shuffle=True,drop_last=True, 
                                       worker_init_fn = worker_init_fn,
                                       pin_memory=True, num_workers=4) #
        
        input_dim = len(train_itrp.feature_name)
        task_dim = train_itrp.y.shape[1]
        
        if dfcx_test is not None:
            dfcx_test = self.scaler.transform(dfcx_test)
            test_itrp = ITRPData(dfcx_test, dfy_test)

            
            test_loader = data.DataLoader(test_itrp, 
                                          batch_size=self.batch_size, 
                                          shuffle=False,
                                          worker_init_fn = worker_init_fn,
                                          pin_memory=True, num_workers=4)
        else:
            test_loader = None

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        run_name = f"Finetune_{self.task_name}_{formatted_time}"  # Example filename

        self.run_name = run_name
        self.save_dir = os.path.join(self.work_dir, run_name)
        
        ## init model, operimizer, ...
        self._setup(self.pretrainer, task_dim, task_type, self.save_dir, self.run_name)

        ## init the taskdecoder
        if task_type == 'f':
            ## claculate the support set features
            pretrainer = self.pretrainer
            dfcx_train_emb, _ = pretrainer.predict(dfcx_train, batch_size=128)
            if pretrainer.model.proj_level == 'cellpathway':
                a = pretrainer.model.celltype_feature_name
                k = pretrainer.model.ref_celltype_ids
            else:
                a = pretrainer.model.geneset_feature_name
                k = pretrainer.model.ref_geneset_ids
            proj_feature_names = [a[i] for i in range(len(a)) if i not in k]
            
            dfcx_train_emb = dfcx_train_emb[proj_feature_names]
            support_features = torch.tensor(dfcx_train_emb.values)
            support_labels = torch.tensor(dfy_train.values)
            self.model.taskdecoder.initialize_parameters(support_features, support_labels)

        self.input_dim = input_dim
        self.task_dim = task_dim

        ### training ###
        performace = []

        best_val_loss = float('inf')
        
        patience_counter = 0
        for epoch in tqdm(range(self.epochs), ascii=True):
            train_total_loss, train_ssl_loss, train_tsk_loss = Trainer(train_loader, self.model, self.optimizer, 
                                                                        self.ssl_loss, self.tsk_loss, self.device, 
                                                                        alpha = self.task_loss_weight, 
                                                                        correction = self.batch_correction,
                                                                        entropy_weight = self.entropy_weight,
                                                                      )
            
            #train_f1, train_mcc, train_prc, train_roc, train_acc = Evaluator(train_loader, model, device)
            if test_loader is not None:
                #test_f1, test_mcc, test_prc, test_roc, test_acc = Evaluator(test_loader, model, device)
                test_total_loss, test_ssl_loss, test_tsk_loss = Tester(test_loader, self.model, self.ssl_loss, 
                                                                        self.tsk_loss, self.device, 
                                                                        alpha =self.task_loss_weight,
                                                                        correction = 0.0,
                                                                      )
                self.saver(test_total_loss, epoch, self.model, self.optimizer, self.scaler)

                # Early Stopping Check
                if test_total_loss < best_val_loss:
                    best_val_loss = test_total_loss
                    patience_counter = 0  
                else:
                    patience_counter += 1  

        
            else:
                test_total_loss, test_ssl_loss, test_tsk_loss = np.nan, np.nan, np.nan
                self.saver(train_total_loss, epoch, self.model, self.optimizer, self.scaler)   
                # Early Stopping Check
                if train_total_loss < best_val_loss:
                    best_val_loss = train_total_loss
                    patience_counter = 0  
                else:
                    patience_counter += 1  

            performace.append([epoch, train_total_loss, train_ssl_loss, train_tsk_loss, 
                               test_total_loss, test_ssl_loss, test_tsk_loss])

            if self.with_wandb:
                self.wandb.log({"FT_train_loss": train_total_loss, 
                                  'FT_train_ssl_loss':train_ssl_loss,
                                  'FT_train_%s_loss' % self.task_name :train_tsk_loss,
                                  'FT_test_loss': test_total_loss, 
                                  'FT_test_ssl_loss':test_ssl_loss, 
                                  'FT_test_%s_loss' % self.task_name :test_tsk_loss})
                
            if self.verbose:
                print("Epoch: {}/{} - Train Loss: {:.4f} - Test Loss: {:.4f}".format(epoch+1, self.epochs, train_total_loss, test_total_loss))

            if patience_counter >= self.patience:
                print(f"Stopping early at epoch {epoch+1}. No improvement in validation loss for {self.patience} consecutive epochs.")
                break
        
        self.saver.save()
        self.performace = performace

        ## plot loss locally
        df = pd.DataFrame(self.performace, columns = ['epochs', 'total_loss', 'ssl_loss', '%s_loss' % self.task_name, 
                                                      'test_loss', 'test_ssl_loss', 'test_%s_loss' % self.task_name]).set_index('epochs')
        #v = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
        fig, ax = plt.subplots(figsize=(7,5))
        df.plot(ax = ax)
        fig.savefig(os.path.join(self.save_dir, '%s_train_loss.png' % self.task_name), bbox_inches='tight')
        plt.close()
        
        df.to_pickle(os.path.join(self.save_dir, '%s_train_loss.pkl' % self.task_name))

        
        return self



    def predict(self, df_tpm, batch_size=512, num_workers=4):
        model = Responder(**self.saver.inMemorySave['model_args']) 
        model.load_state_dict(self.saver.inMemorySave['model_state_dict'])
        model = model.to(self.device)
        dfe, dfp = Predictor(df_tpm, model, self.scaler, 
                             device = self.device, 
                             batch_size=batch_size,  num_workers=num_workers)
        return dfe, dfp

    
    def extract(self, df_tpm, batch_size=512, num_workers=4):
        model = Responder(**self.saver.inMemorySave['model_args']) 
        model.load_state_dict(self.saver.inMemorySave['model_state_dict'])
        model = model.to(self.device)
        dfg, dfc = Extractor(df_tpm, model, self.scaler, 
                           device = self.device, batch_size=batch_size, 
                             num_workers=num_workers)
        return dfg, dfc
        
    
    def plot_embed(self, df_tpm, df_label, label_types, **kwargs):
        
        ### make prediction & plot on TCGA
        dfe, dfp = self.predict(df_tpm, batch_size=self.batch_size,  num_workers=4)
        dfd = dfe.join(df_label)
        label_cols = df_label.columns

        figs = plot_embed_with_label(dfd, 
                                     label_col = label_cols,  
                                     label_type = label_types, **kwargs)
        for fig, name in zip(figs, label_cols):
            fig.savefig(os.path.join(self.save_dir, 'FT_%s.png' % name), bbox_inches='tight', )


    def save(self, mfile):
        if self.with_wandb:
            self.wandb._settings = ''
        torch.save(self, mfile)
        print('Saving the model to %s' % mfile)

    def load(self, mfile, **kwargs):
        self = loadresponder(mfile, **kwargs)
        if self.with_wandb:
            self.wandb._settings = ''
        return self

    def close(self):
        if self.with_wandb:
            self.wandb.finish()
            self.wandb._settings = ''
        
        self.save(os.path.join(self.save_dir, 'finetuner.pt'))

    def copy(self):
        return deepcopy(self)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    
    
    def get_tuner_epoch(self, train_X, train_y, 
                       max_epochs = 200, work_dir = './Paramtune',
                        with_wandb = False, verbose = False, fold = 3):
    
        param = deepcopy(self.params)
        param['with_wandb'] = with_wandb
        param['epochs'] = max_epochs
        param['work_dir'] = work_dir
        param['verbose'] = verbose
        
        skf = StratifiedKFold(n_splits=fold, 
                              shuffle=True, random_state=42)
        splits = skf.split(train_y.values, train_y.idxmax(axis=1))
        
        best_epochs = []
        for i, (train_idx, valid_idx) in enumerate(splits):
            X_train = train_X.iloc[train_idx]
            y_train = train_y.iloc[train_idx]
            X_val = train_X.iloc[valid_idx]
            y_val = train_y.iloc[valid_idx]
            print(len(X_train), len(X_val), 
                  y_val.idxmax(axis=1).value_counts().to_dict(), 
                  y_train.idxmax(axis=1).value_counts().to_dict())
        
            tuner = FineTuner(self.pretrainer, **param)
            tuner.tune(dfcx_train = X_train,
                        dfy_train = y_train,
                        task_name='fold_%s' % i,
                        task_type='c',
                        dfcx_test = X_val,
                        dfy_test = y_val)
        
            best_epoch = tuner.saver.inMemorySave['epoch']
            best_epochs.append(best_epoch)
        
        print(np.array(best_epochs))
        best_epochs = int(np.array(best_epochs).mean())
    
        return best_epochs
        