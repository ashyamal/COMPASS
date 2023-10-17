'''
Immunotherapy target marker: PD1_PD-L1_CTLA4	PMID:30127393	
Kong J H, Ha D, Lee J, et al. Network-based machine learning approach to predict immunotherapy response in cancer patients[J]. Nature communications, 2022, 13(1): 3703.

Jiang P, Gu S, Pan D, et al. Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response[J]. Nature medicine, 2018, 24(10): 1550-1558.

'''


ITM_MAKERS = {'marker':{'PD1': 'PDCD1', 'PD-L1': 'CD274', 'CTLA4': 'CTLA4'},
              'marker_score_method': 'value', 
              'marker_type':'single',
              'marker_ip': 'Target',
              'paper': ['PMID:30127393', 'PMID:35764641']}
