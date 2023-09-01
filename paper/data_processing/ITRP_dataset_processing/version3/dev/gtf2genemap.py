# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:53:00 2023

@author: Wanxiang Shen

The script is to generate the gene-id/name/length.map files
"""

import pandas as pd
def parse_attr(attr):
    items = attr.split('; ')
    res={}
    for it in items:
        its = it.split(' ')
        its = [it.replace('"','').replace(';','') for it in its]
        res[its[0]] = its[1]
    return res
    
# READ GTF FILES
gtf_file = '/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GENCODE/release_44/HS.gencode.v44.annotation.gtf'
gtf_data = pd.read_csv(gtf_file,sep='\t',header=None,comment='#',
                       names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'])

# EXTRACT GENES' INFO
genes = gtf_data[gtf_data['feature'] == 'gene']
genes_attr = genes.attribute.apply(parse_attr).apply(pd.Series)
genes['length'] = genes['end'] - genes['start'] + 1
dfg = genes[['seqname', 'source', 'score', 'strand', 'frame', 'length']].join(genes_attr[['gene_id', 'gene_type', 'gene_name']])
dfg = dfg.set_index('gene_id')
dfg.to_pickle('/home/was966/Research/PSOG/config/gene.map')