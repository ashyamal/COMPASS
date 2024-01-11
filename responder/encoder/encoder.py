# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:31:25 2023

@author: Wanxiang Shen

"""

import torch.nn as nn
from torch.nn.modules.container import ModuleList
from torchvision.ops import MLP

import copy, torch
from .layer import CosformerLayer, PerformerLayer, VanillaTransformerLayer, FlowformerLayer
from ..embedder import GeneEmbedding
#from .layer import  FlashTransformerEncoderLayer
from .layer.norm import create_norm

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):

    def __init__(self, encoder_type= 'transformer', d_model = 32, dim_feedforward = 128,
                 nhead = 2, num_layers = 1, dropout = 0, norm = None, **kwargs):
        
        super(Encoder, self).__init__()
        
        self.encoder_type = encoder_type
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_head = d_model // nhead  #16
        self.dropout = dropout
        
        self.norm = norm

        if self.norm is not None:
            self._norm = create_norm('layernorm', d_model)


        if encoder_type == 'cosformer':
            encoder_layer = CosformerLayer(embed_dim=d_model, 
                                           num_heads=nhead,dropout=dropout, **kwargs)
        elif encoder_type == 'performer':
            encoder_layer = PerformerLayer(embed_dim=d_model, 
                                           num_heads=nhead,dropout=dropout, **kwargs)
        elif encoder_type == 'Vanillatransformer':
            encoder_layer = VanillaTransformerLayer(embed_dim=d_model, 
                                                    num_heads=nhead,dropout=dropout, **kwargs)
        elif encoder_type == 'flowformer':
            encoder_layer = FlowformerLayer(embed_dim=d_model, 
                                            num_heads=nhead, dropout=dropout, **kwargs)
        elif encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                       nhead=nhead,
                                                       dropout=dropout, 
                                                       dim_feedforward = dim_feedforward,
                                                       batch_first = True,
                                                      **kwargs
                                                      )

        # elif encoder_type == 'flashformer':
        #     encoder_layer = FlashTransformerEncoderLayer(d_model=d_model, 
        #                                                nhead=nhead,
        #                                                dropout=dropout, 
        #                                                dim_feedforward = dim_feedforward,
        #                                                batch_first = True,
        #                                               **kwargs
        #                                               ).half()
        
        else:
            raise NotImplementedError(f'Not implemented transformer type: {encoder_type}')

        self.layers = _get_clones(encoder_layer, num_layers)


    
    def forward(self, x, output_attentions=False):

        att_list = []
        for l in range(self.num_layers):
            if output_attentions:
                x, att = self.layers[l](x, output_attentions=output_attentions)
                att_list.append(att)
            else:
                x = self.layers[l](x)

        if self.norm is not None:
            x = self._norm(x)
            
        if output_attentions:
            return x, att_list
        else:
            return x





class TransformerEncoder(nn.Module):
    def __init__(self, encoder_type = 'transformer', input_dim = 876, nhead = 2,
                 d_model = 32, num_layers = 2, dropout = 0., dim_feedforward = 128,
                 pos_emb = 'learnable', **kwargs):
        '''
        encoder_type: {'transformer', 'reformer', 'performer'}
        pos_emb: {None, 'learnable', 'gene2vect', 'umap'}
        d_model: {16,32,64,128,512}
        '''
        super().__init__()
        self.encoder_type = encoder_type
        
        self.embedder = GeneEmbedding(input_dim, d_model, pos_emb)

        self.encoder = Encoder(encoder_type = encoder_type, 
                               d_model = d_model, 
                               dropout = dropout, 
                               dim_feedforward = dim_feedforward,
                               nhead = nhead, num_layers = num_layers, **kwargs)

    def forward(self,x):
        x = self.encoder(self.embedder(x))
        return x



class MLPEncoder(nn.Module):
    def __init__(self, input_dim, dense_layers = [512, 256, 128], dropout = 0.2):
        super().__init__()
        self.encoder = MLP(in_channels=input_dim, 
                           hidden_channels=dense_layers, 
                           norm_layer = torch.nn.BatchNorm1d, 
                           dropout  = dropout)
    def forward(self,x):
        x = self.encoder(x)
        return x
