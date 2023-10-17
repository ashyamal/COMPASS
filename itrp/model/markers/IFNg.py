'''
Paper:
Ayers M, Lunceford J, Nebozhyn M, Murphy E, Loboda A, Kaufman DR, Albright A, Cheng JD, Kang SP, Shankaran V, et al: IFN-gamma-related mRNA profile predicts clinical response to PD-1 blockade. J Clin Invest 2017, 127:2930-2940.

Marker:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5531419/bin/jci-127-91190-g010.jpg

Calculation:
After performance of quantile normalization, a log10 transformation was applied, and signature scores were calculated by averaging of the included genes for the IFN-γ (6-gene) and expanded immune (18-gene) signatures.
'''

IFNg_MAKERS = {'marker':{'IFNg6':'CXCL10:IFNG:CXCL9:IDO1:HLA-DRA:STAT1', 
                         'IFNg18':'CD3D:IDO1:CIITA:CD3E:CCL5:GZMK:CD2:HLA-DRA:CXCL13:IL2RG:NKG7:HLA-E:CXCR6:LAG3:TAGAP:CXCL10:STAT1:GZMB'},
              'marker_score_method':'average', 
               'marker_ip':'Ayers',
              'marker_type':'set',
              'paper': ['PMID:28650338']}

