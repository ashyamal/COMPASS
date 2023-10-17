'''
tumor-associated macrophages (TAMs)

Jiang P, Gu S, Pan D, et al. Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response[J]. Nature medicine, 2018, 24(10): 1550-1558.

Kong J H, Ha D, Lee J, et al. Network-based machine learning approach to predict immunotherapy response in cancer patients[J]. Nature communications, 2022, 13(1): 3703.
'''

TAM_MAKERS = {'marker':{'TAM': 'F13A1:FCER1A:CCL17:FOXQ1:ESPNL:CD1A:GATM:CCL13:PALLD:GALNT18:MAOA:RAMP1:STAB1:CCL26:CCL23:PARM1:CD1E:ITM2C:CALCRL:CRH:RGS18:MS4A6A:DHRS2:PON2:ALOX15:RAB33A:MOCOS:CCL18:IL17RB:FABP4:CMTM8:QPRT:CDR2L:DUOXA1:ABCC4:SYT17:PPP1R14A:PDGFC:GPT:FAM189A2:RASAL1:IPCEF1:ZNF366:MAP4K1:RAB30:PCED1B:TMIGD3:SH3BP4:RRS1:RNASE1'},
                 'marker_score_method': 'average', 
                 'marker_type':'set',
              'marker_ip':'Jiang',
                 'paper': ['PMID:30127393', 'PMID:35764641']}

