import numpy as np
import pandas as pd
import gseapy as gp


class ssGSEA:
    
    def __init__(self, geneset, geneset_name):
        self.gene_sets = {geneset_name:geneset}
        self.geneset = geneset
        self.geneset_name = geneset_name        
        
        
    def fit(self, df_tpm):
        '''
        dfx: each column is one gene, each row is one samples
        '''
        dfx = df_tpm.T
        ssgsea_results = gp.ssgsea(data=dfx, 
                                   gene_sets=self.gene_sets)

        dfres = ssgsea_results.res2d
        scale_factor = (dfres.ES / dfres.NES).mean()
        self.scale_factor = scale_factor
        return self
    
    
    
    def transform(self, df_tpm):
        dfx = df_tpm.T
        ssgsea_results = gp.ssgsea(data=dfx, 
                                   gene_sets=self.gene_sets)

        dfres = ssgsea_results.res2d.set_index('Name')
        dfres['NES'] = dfres['ES'] / self.scale_factor
        dfres = dfres.loc[df_tpm.index]
        dfres['NES_%s' % self.geneset_name] = dfres['NES']
        
        return dfres['NES_%s' % self.geneset_name]

    
    def fit_transform(self, df_tpm):
        self.fit(df_tpm)
        return self.transform(df_tpm)
    
    
    
    
    
    
class avgAbundance:
    
    def __init__(self, geneset, geneset_name):
        self.gene_sets = {geneset_name:geneset}
        self.geneset = geneset
        self.geneset_name = geneset_name
        
        
    def fit(self, df_tpm):
        '''
        dfx: each column is one gene, each row is one samples
        '''
        dfx = np.log2(df_tpm + 1)
        self.used_cols = list(set(dfx.columns) & set(self.geneset))
        dfx_used = dfx[self.used_cols]        
        score = dfx_used.mean(axis=1)
        self.scale_factor = score.mean()
        return self
    
    def transform(self, df_tpm):
        dfx = np.log2(df_tpm + 1)
        dfx_used = dfx[self.used_cols]  
        score = dfx_used.mean(axis=1)
        score = score / self.scale_factor
        score.name = 'NAG_%s' % self.geneset_name
        return score

    
    def fit_transform(self, df_tpm):
        self.fit(df_tpm)
        return self.transform(df_tpm)
    
    
    
    
    
    
    