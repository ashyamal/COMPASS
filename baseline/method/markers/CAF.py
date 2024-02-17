'''
Cancer-associated fibroblasts (CAFs)

Nurmik M, Ullmann P, Rodriguez F, et al. In search of definitions: Cancer‐associated fibroblasts and their markers[J]. International journal of cancer, 2020, 146(4): 895-905.

Jiang P, Gu S, Pan D, et al. Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response[J]. Nature medicine, 2018, 24(10): 1550-1558.

Kong J H, Ha D, Lee J, et al. Network-based machine learning approach to predict immunotherapy response in cancer patients[J]. Nature communications, 2022, 13(1): 3703.
'''

CAF_MAKERS = {'marker':{'CAF': 'FAP:ACTA2:MFAP5:COL11A1:TN-C'},
              'marker_score_method': 'average', 
              'marker_type':'set',
              'marker_ip': 'Nurmik',
              'paper': ['PMID:30734283', 'PMID:30127393', 'PMID:35764641']}
