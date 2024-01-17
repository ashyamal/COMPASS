
from .cellpathwayaggregator import CellPathwayAggregator
from .genesetaggregator import GeneSetAggregator
from .genesetscorer import GeneSetScorer


import torch, math
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np



cwd = os.path.dirname(__file__)

N = 1005 #15672
GENESET = pd.read_pickle(os.path.join(cwd, str(N), 'GENESET.DATA'))
CELLPATHWAY = pd.read_pickle(os.path.join(cwd, str(N), 'CELLTYPE.DATA'))

LEVEL = {'geneset':len(GENESET), 'cellpathway':len(CELLPATHWAY)}



class GeneSetProjector(nn.Module):
    def __init__(self, geneset_feature_dim, geneset_agg_mode = 'attention', geneset_score_mode = 'linear'):
        super(GeneSetProjector, self).__init__()

        self.geneset_feature_dim = geneset_feature_dim
        self.geneset_agg_mode = geneset_agg_mode
        self.geneset_score_mode = geneset_score_mode

        self.GENESET = GENESET
        self.genesets_indices = GENESET.tolist()
        self.genesets_names = GENESET.index 

        self.geneset_aggregator =  GeneSetAggregator(self.genesets_indices, self.geneset_agg_mode) 
        self.geneset_scorer = GeneSetScorer(self.geneset_feature_dim, self.geneset_score_mode)

        #self.norm = nn.LayerNorm(len(self.genesets_indices))
        #nn.BatchNorm1d(n)
    
    def forward(self, x):
        geneset_feats = self.geneset_aggregator(x)
        geneset_scores = self.geneset_scorer(geneset_feats)
        #geneset_scores = F.normalize(geneset_scores, p=2., dim=1)
        
        return geneset_scores



class CellPathwayProjector(nn.Module):

    def __init__(self, cellpathway_agg_mode = 'pooling'):
        super(CellPathwayProjector, self).__init__()
        self.cellpathway_agg_mode = cellpathway_agg_mode
        self.CELLPATHWAY = CELLPATHWAY
        self.cellpathway_indices = CELLPATHWAY.tolist()
        self.cellpathway_names = CELLPATHWAY.index 
        self.cellpathway_aggregator =  CellPathwayAggregator(self.cellpathway_indices, 
                                                             mode = self.cellpathway_agg_mode) 

    def forward(self, x):
        cellpathway_scores = self.cellpathway_aggregator(x)
        #cellpathway_scores = F.normalize(cellpathway_scores, p=2., dim=1)
        return cellpathway_scores
        


class DisentangledProjector(nn.Module):
    def __init__(self, 
                 gene_feature_dim,
                 geneset_agg_mode = 'attention', 
                 geneset_score_mode = 'linear', 
                 cellpathway_agg_mode = 'attention'):
        super(DisentangledProjector, self).__init__()

        self.gene_feature_dim=gene_feature_dim
        self.geneset_agg_mode = geneset_agg_mode
        self.geneset_score_mode = geneset_score_mode
        self.cellpathway_agg_mode = cellpathway_agg_mode
        self.genesetprojector = GeneSetProjector(self.gene_feature_dim, self.geneset_agg_mode, self.geneset_score_mode)
        self.cellpathwayprojector = CellPathwayProjector(self.cellpathway_agg_mode)

    def forward(self, x):
        geneset_scores = self.genesetprojector(x)
        cellpathway_scores = self.cellpathwayprojector(geneset_scores)
        return geneset_scores, cellpathway_scores # (256,111); (256, 32)





class EntangledProjector(nn.Module):
    def __init__(self, gene_feature_dim,  mode = 'mean'):
        '''
        reduce: {'mean', 'max', 'cls'}
        '''
        super(EntangledProjector, self).__init__()
        self.mode = mode
        self.gene_feature_dim = gene_feature_dim
        
    def forward(self, x):
        if self.mode == 'mean':
            x = torch.mean(x, dim=-1)
        elif self.mode == 'max':
            x = torch.max(x, dim=-1)[0]
        else:
            ValueError("Invalid pooling type. Use 'mean' or 'max'.")
        return x