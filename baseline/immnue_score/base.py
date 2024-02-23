from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA


class GeneSetScorer:

    def __init__(self):
        self.gene_set = ['PDCD1', 'CTLA4']
        self.name = 'Base'
        
    def get_avg(self, df_tpm):
        score = avgAbundance(self.gene_set, self.name)
        return score.fit_transform(df_tpm)

    def get_pca(self, df_tpm):
        score = pcaAbundance(self.gene_set, self.name)
        return score.fit_transform(df_tpm)

    def get_org(self, df_tpm):
        score = origAbundance(self.gene_set, self.name)
        return score.fit_transform(df_tpm)
        
    def get_ssgsea(self, df_tpm):
        score = ssGSEA(self.gene_set, self.name)
        return score.fit_transform(df_tpm)
