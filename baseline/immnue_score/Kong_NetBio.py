import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS

class Kong_NetBio(GeneSetScorer):
    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):

        ALL_TARGETS = ['PD1', 'CTLA4', 'PD1_CTLA4', 'PDL1', 'PD1_PDL1_CTLA4']
        assert drug_target in ALL_TARGETS, 'Invalid target, select a target type in %s' % ALL_TARGETS

        idx = 'NetBio200.'+ drug_target
        gs = MARKERS.loc[idx]

        self.cancer_type = cancer_type
        self.drug_target = drug_target
        self.name = 'Kong_NetBio'
        self.reference = gs.Reference
        self.description = gs.Description
        self.gene_set = gs.Genes.split(':')
        self.gs = gs

    def __call__(self, df_tpm):
        scols = df_tpm.columns.intersection(self.gene_set)
        dfres = np.log2(df_tpm[scols] + 1)
        return dfres
