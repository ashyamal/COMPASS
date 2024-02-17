import pandas as pd
import os
from scoring import avgAbundance, ssGSEA

class Geneset:
    
    def __init__(self, gs):
        
        self.gs = gs
        self.name = gs.name
        self.reference = gs.Reference
        self.description = gs.Description
        self.gene_set = gs.Genes.split(':')
        

    def score_by_avg(self, df_tpm):
        self.abd = avgAbundance(self.gene_set, self.name)
        return self.abd.fit_transform(df_tpm)

    
    def score_by_ssgsea(self, df_tpm):
        self.ssgsea = ssGSEA(self.gene_set, self.name)
        return self.ssgsea.fit_transform(df_tpm)
    
    
    def score_by_inherent(self, df_tpm):    
        pass
    
    
    
cwd = os.path.dirname(__file__)
dfm = pd.read_csv(os.path.join(cwd, 'marker.tsv'), sep='\t', index_col=0)
SETS = []
for i in range(len(dfm)):
    gs = dfm.iloc[i]
    name = gs.name
    SETS.append({name:Geneset(gs)})
