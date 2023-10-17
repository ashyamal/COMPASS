'''
Paper: Auslander N, Zhang G, Lee J S, et al. Robust prediction of response to immune checkpoint blockade therapy in metastatic melanoma[J]. Nature medicine, 2018, 24(10): 1545-1549.
https://github.com/JasonACarter/IMPRES_Correspondence/blob/master/Code/IMPRES_Correspondence.ipynb
'''

#The original IMPRES features given as gene1_gene2, defined as gene1>gene2:

IMPRES_MAKERS = {'marker':{'IMPRES': ['CD274:VSIR','CD86:CD200','CD40:CD274','CD28:CD276', 
                                      'CD40:CD28','TNFRSF14:CD86','CD27:PDCD1','CD28:CD86',
                                      'CD40:CD80','CD40:PDCD1', 'CD80:TNFSF9','CD86:HAVCR2',
                                      'CD86:TNFSF4', 'CTLA4:TNFSF4','PDCD1:TNFSF4']},
                 'marker_score_method': 'diff_bool', 
                 'marker_type':'pair',
                 'marker_ip':'Auslander',
                 'paper': ['PMID: 30127394']}