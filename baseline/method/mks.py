## SINGLE and Dboule marker
import numpy as np
import pandas as pd


SUBS_MARKERS = ['SLAMF6:ATG12',
                'SLAMF7:IL10',
                'PIK3CD:ATG12',
                'DEFB110:SFTPA2',
                'IKZF3:APAF1',
                'IKZF3:CD200R1',
                'IL6R:RER1',
                'IL6R:CTNNA1',
                'MAP4K1:TBX3',
                'MAP4K1:AGER']

SINGLE_MARKER = {'PDCD1':'PD1', 'CD274':'PD-L1', 'CTLA4':'CTLA4'}

AVG_MARKERS = {'CD8A:CD8B':'CD8'}


def get_mks(df_tpm):
    df = np.log2(df_tpm + 1)
    _scores = []
    for m in SUBS_MARKERS:
        p, q = m.split(':')
        s = df[p] - df[q]
        s.name = m
        _scores.append(s)

    for m, n in SINGLE_MARKER.items():
        s = df[m]
        s.name = n
        _scores.append(s)

    for m, n in AVG_MARKERS.items():
        p, q = m.split(':')
        s = df[[p, q]].mean(axis=1)
        s.name = n
        _scores.append(s)  

    dfs = pd.concat(_scores, axis=1)
    return dfs


    