# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:31:25 2023

@author: Wanxiang Shen

"""
import torch.nn as nn
from ..encoder import TransformerEncoder, MLPEncoder
from ..decoder import ClassDecoder, RegDecoder
from ..projector import DisentangledProjector, EntangledProjector



class Responder(nn.Module):

    def __init__(self, 
                 input_dim, 
                 task_dim,
                 task_type,
                 num_cancer_types = 33,
                 embed_dim = 32,
                 disentangled_embed =  True,
                 embed_level = 'cellpathway',
                 encoder = 'transformer',
                 encoder_dropout = 0.,
                 task_dense_layer=[24],
                 task_batch_norms = True,
                 transformer_dim = 32,
                 transformer_num_layers = 1,
                 transformer_nhead = 2,
                 transformer_pos_emb = 'learnable',
                 mlp_dense_layers = [128],
                 **encoder_kwargs
                ):
        
        '''
        input_dim: input dim
        task_dim: supervised learning task dim
        task_type: {'r', 'c'}
        num_cancer_types: int, number cancer types, default 33.
        embed_dim: latent vector dim
        encoder: {'transfomer', 'flowformer', ...}
        task_dense_layer: dense layer of task
        transformer_pos_emb: {None, 'umap', 'pumap'}
        '''
        super().__init__()

        self.input_dim = input_dim
        self.task_dim = task_dim
        self.task_type = task_type 
        self.embed_level = embed_level
        self.disentangled_embed = disentangled_embed

        self.num_cancer_types = num_cancer_types
        self.encoder = encoder
        self.encoder_dropout = encoder_dropout
        self.mlp_dense_layers = mlp_dense_layers
        self.transformer_dim = transformer_dim
        self.transformer_num_layers = transformer_num_layers
        self.transformer_pos_emb = transformer_pos_emb
        self.transformer_nhead = transformer_nhead
        self.task_batch_norms = task_batch_norms
        self.task_dense_layer = task_dense_layer
        self.encoder_kwargs = encoder_kwargs

        self.inputencoder = TransformerEncoder(num_cancer_types=num_cancer_types,
                                               encoder_type = encoder,
                                               input_dim = input_dim, 
                                               d_model = transformer_dim, 
                                               num_layers = transformer_num_layers,
                                               nhead = transformer_nhead,
                                               dropout = encoder_dropout,
                                               pos_emb = transformer_pos_emb, 
                                               **encoder_kwargs)
        
        if disentangled_embed:
            self.latentprojector = DisentangledProjector(input_dim, transformer_dim)
            self.embed_dim = self.latentprojector.LEVEL[embed_level]
        else:
            self.latentprojector = EntangledProjector(transformer_dim)
            self.embed_dim = embed_dim
            
        model_args = {'input_dim':self.input_dim, 
                'task_dim':self.task_dim,
                'task_type':self.task_type, 
                'embed_level':self.embed_level,
                'disentangled_embed':self.disentangled_embed,
                'embed_dim': self.embed_dim, 
                'num_cancer_types':self.num_cancer_types,
                'encoder':self.encoder,
                'encoder_dropout':self.encoder_dropout,
                'mlp_dense_layers':self.mlp_dense_layers,
                'transformer_dim':self.transformer_dim,
                'transformer_nhead':self.transformer_nhead,
                'transformer_num_layers':self.transformer_num_layers,
                'transformer_pos_emb':self.transformer_pos_emb,
                'task_batch_norms':self.task_batch_norms,
                'task_dense_layer':self.task_dense_layer,
               }

        model_args.update(encoder_kwargs)
        
        self.model_args = model_args

            
        ## regression task
        if task_type == 'r':
            self.taskdecoder = RegDecoder(self.embed_dim, 
                                        dense_layers = task_dense_layer, 
                                        out_dim = task_dim, 
                                        batch_norms = task_batch_norms)
        
        ## classification task
        else:
            self.taskdecoder = ClassDecoder(self.embed_dim,
                                          dense_layers = task_dense_layer, 
                                          out_dim = task_dim, 
                                          batch_norms = task_batch_norms)

    def forward(self, x):
        # input : B, L+1, 1
        #output： B,L+2, (cancer:1,dataset:1, gene),C
        encoding = self.inputencoder(x) 
        gene_encoding = encoding[:,:2,:] # take the gene encoding only
        geneset_level_proj, cellpathway_level_proj = self.latentprojector(encoding)
        if self.embed_level == 'cellpathway':
            embedding = cellpathway_level_proj
        else:
            embedding = geneset_level_proj
        y = self.taskdecoder(embedding)
        return embedding, y



