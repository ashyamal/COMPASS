'''
Davoli cytotoxic immune signatures (CIS), 

Paper: 
=============
Davoli et al., Science 2017

Description:
=============
Gene markers of cytotoxic immune cell infiltrates (cytotoxic CD8+ T cells and NK cells).

Caclculation: 
=============
First, rank normalization is applied across samples. 
Second, the average of the expression of Davoli immune signature is calculated for each sample [log2(TPM+1)].	

Markers: 
=============
CD247, CD2, CD3E, GZMH, NKG7, PRF1, GZMK
'''


CIS_MAKERS = {'marker':{'CIS': 'CD247:CD2:CD3E:GZMH:NKG7:PRF1:GZMK'},
              'marker_score_method': 'average', 
              'marker_type':'set',
              'marker_ip':'Davoli',
              'paper': ['PMID:28104840']}