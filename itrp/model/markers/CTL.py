'''
CTL: cytotoxic T lymphocytes markers

Rooney MS, Shukla SA, Wu CJ, Getz G & Hacohen N Molecular and genetic properties of tumors associated with local immune cytolytic activity. Cell 160, 48–61 (2015). 

Jiang P, Gu S, Pan D, et al. Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response[J]. Nature medicine, 2018, 24(10): 1550-1558.


Kong J H, Ha D, Lee J, et al. Network-based machine learning approach to predict immunotherapy response in cancer patients[J]. Nature communications, 2022, 13(1): 3703.

'''

CTL_MAKERS = {'marker':{'CTL': 'CD8A:CD8B:GZMA:GZMB:PRF1'},
              'marker_score_method': 'average', 
              'marker_type':'set',
              'marker_ip':'Rooney',
              'paper': ['PMID:25594174', 'PMID:30127393', 'PMID:35764641']}



